/// `llmb setup` — download and install the right llama.cpp binaries automatically.
///
/// 1. Skips download if `llama-cli` and `llama-server` are already on PATH (unless `--force`).
/// 2. Queries the GitHub releases API for the latest llama.cpp release.
/// 3. Picks the correct release asset based on OS + detected GPU (.zip or .tar.gz).
/// 4. Downloads and extracts into the llama-bin cache directory.

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use crate::assets::{self, llama_bin_dir};

const GITHUB_API: &str =
    "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest";

const EXECUTABLES: &[&str] = &["llama-cli", "llama-server", "llama-cli.exe", "llama-server.exe"];
const LIB_EXTENSIONS: &[&str] = &["dll", "so", "dylib"];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    CudaCu12,
    Vulkan,
    CpuOnly,
    MacosArm64,
    MacosX64,
    LinuxCuda,
    LinuxCpu,
}

impl Backend {
    fn asset_keyword(&self) -> &'static str {
        match self {
            Backend::CudaCu12 => "bin-win-cuda-cu12",
            Backend::Vulkan => "bin-win-vulkan",
            Backend::CpuOnly => "bin-win-noavx2",
            Backend::MacosArm64 => "bin-macos-arm64",
            Backend::MacosX64 => "bin-macos-x64",
            Backend::LinuxCuda => "bin-ubuntu-x64",
            Backend::LinuxCpu => "bin-ubuntu-x64",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Backend::CudaCu12 => "Windows / NVIDIA CUDA 12",
            Backend::Vulkan => "Windows / Vulkan (AMD / Intel GPU)",
            Backend::CpuOnly => "Windows / CPU only (no GPU)",
            Backend::MacosArm64 => "macOS / Apple Silicon",
            Backend::MacosX64 => "macOS / Intel",
            Backend::LinuxCuda => "Linux / NVIDIA CUDA",
            Backend::LinuxCpu => "Linux / CPU only",
        }
    }

    fn cudart_keyword(&self) -> Option<&'static str> {
        match self {
            Backend::CudaCu12 | Backend::LinuxCuda => Some("cudart-llama-bin-win-cuda"),
            _ => None,
        }
    }
}

/// Detect the best backend for the current machine.
pub fn detect_backend() -> Backend {
    let os = std::env::consts::OS;

    match os {
        "macos" => {
            if std::env::consts::ARCH == "aarch64" {
                Backend::MacosArm64
            } else {
                Backend::MacosX64
            }
        }
        "linux" => {
            if has_nvidia_smi() {
                Backend::LinuxCuda
            } else {
                Backend::LinuxCpu
            }
        }
        _ /* windows */ => {
            if has_nvidia_smi() {
                Backend::CudaCu12
            } else if has_vulkan() {
                Backend::Vulkan
            } else {
                Backend::CpuOnly
            }
        }
    }
}

fn has_nvidia_smi() -> bool {
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn has_vulkan() -> bool {
    std::process::Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Download and install the backend for the current machine.
pub fn run(force: bool) -> anyhow::Result<()> {
    let bin_dir = llama_bin_dir()?;

    let cached = {
        let cli = bin_dir.join(assets::llama_cli_exe());
        let server = bin_dir.join(assets::llama_server_exe());
        cli.exists() && server.exists()
    };

    if cached && !force {
        println!(
            "{} llama.cpp binaries already installed in {}",
            "Skipping:".yellow(),
            bin_dir.display()
        );
        println!("Run with --force to reinstall.");
        return Ok(());
    }

    if !force && binaries_on_path() {
        let cli = assets::find_llama_cli()?;
        let server = assets::find_llama_server()?;
        println!(
            "{} llama-cli and llama-server already on PATH",
            "Skipping:".yellow()
        );
        println!("  llama-cli:    {}", cli.display());
        println!("  llama-server: {}", server.display());
        println!("Run with --force to download copies into the cache instead.");
        return Ok(());
    }

    let backend = detect_backend();
    println!("Detected platform: {}", backend.label().bold());
    println!("Querying GitHub for the latest llama.cpp release...");

    let release = fetch_latest_release()?;
    println!("Latest release: {}", release.tag_name.bold());

    let keyword = backend.asset_keyword();
    let main_asset = find_asset(&release.assets, keyword).ok_or_else(|| {
        anyhow::anyhow!(
            "No asset matching '{}' found in release {}.\n\
             Browse manually: https://github.com/ggerganov/llama.cpp/releases",
            keyword,
            release.tag_name
        )
    })?;

    println!(
        "Downloading {} ({:.1} MB)...",
        main_asset.name.bold(),
        main_asset.size as f64 / 1_048_576.0
    );
    let archive = download_to_memory(&main_asset.browser_download_url, main_asset.size)?;
    extract_archive(&archive, &bin_dir, &main_asset.name)?;

    if let Some(cudart_kw) = backend.cudart_keyword() {
        if let Some(cudart_asset) = find_asset(&release.assets, cudart_kw) {
            println!(
                "Downloading CUDA runtime {} ({:.1} MB)...",
                cudart_asset.name.bold(),
                cudart_asset.size as f64 / 1_048_576.0
            );
            let cudart_archive =
                download_to_memory(&cudart_asset.browser_download_url, cudart_asset.size)?;
            extract_archive(&cudart_archive, &bin_dir, &cudart_asset.name)?;
        }
    }

    println!(
        "\n{} llama.cpp binaries installed to {}",
        "Done.".green().bold(),
        bin_dir.display()
    );
    println!("You can now run:  llmb models fetch && llmb bench");
    Ok(())
}

fn binaries_on_path() -> bool {
    assets::find_llama_cli().is_ok() && assets::find_llama_server().is_ok()
}

/// Pick the best release asset for a platform keyword (.zip or .tar.gz).
fn find_asset<'a>(assets: &'a [Asset], keyword: &str) -> Option<&'a Asset> {
    let mut matches: Vec<&Asset> = assets
        .iter()
        .filter(|a| matches_asset_name(&a.name, keyword))
        .collect();
    if matches.is_empty() {
        return None;
    }
    matches.sort_by_key(|a| asset_rank(&a.name));
    matches.into_iter().next()
}

fn matches_asset_name(name: &str, keyword: &str) -> bool {
    name.contains(keyword) && (name.ends_with(".zip") || name.ends_with(".tar.gz"))
}

/// Lower rank = preferred. Prefer plain builds over optional accelerators (e.g. kleidiai).
fn asset_rank(name: &str) -> (u8, usize) {
    let variant_penalty = if name.contains("kleidiai") {
        1
    } else {
        0
    };
    (variant_penalty, name.len())
}

// ── GitHub API ────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(serde::Deserialize)]
struct Asset {
    name: String,
    size: u64,
    browser_download_url: String,
}

fn fetch_latest_release() -> anyhow::Result<Release> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("llmb-benchmark/0.1")
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let release: Release = client.get(GITHUB_API).send()?.json()?;
    Ok(release)
}

// ── HTTP download ─────────────────────────────────────────────────────────────

fn download_to_memory(url: &str, expected_size: u64) -> anyhow::Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()?;

    let response = client.get(url).send()?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP {} downloading {}", response.status(), url);
    }

    let pb = ProgressBar::new(expected_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] \
             {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let bytes = response.bytes()?;
    pb.inc(bytes.len() as u64);
    pb.finish_and_clear();

    Ok(bytes.to_vec())
}

// ── Archive extraction ────────────────────────────────────────────────────────

fn extract_archive(data: &[u8], dest: &Path, archive_name: &str) -> anyhow::Result<()> {
    let extracted = if archive_name.ends_with(".tar.gz") {
        extract_tar_gz(data, dest)?
    } else if archive_name.ends_with(".zip") {
        extract_zip(data, dest)?
    } else {
        anyhow::bail!("Unsupported archive format: {}", archive_name);
    };

    println!(
        "  {} Extracted {} files from {}",
        "✓".green(),
        extracted,
        archive_name
    );
    Ok(())
}

fn extract_zip(data: &[u8], dest: &Path) -> anyhow::Result<usize> {
    let cursor = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let mut extracted = 0usize;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();
        let Some(basename) = path_basename(&name) else {
            continue;
        };
        if !should_extract_file(basename) {
            continue;
        }
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        write_extracted_file(dest, basename, &buf)?;
        extracted += 1;
    }
    Ok(extracted)
}

fn extract_tar_gz(data: &[u8], dest: &Path) -> anyhow::Result<usize> {
    let cursor = Cursor::new(data);
    let decoder = flate2::read::GzDecoder::new(cursor);
    let mut archive = tar::Archive::new(decoder);

    let mut extracted = 0usize;
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.into_owned();
        let Some(basename) = path_basename(path.to_str().unwrap_or("")) else {
            continue;
        };
        if !should_extract_file(basename) {
            continue;
        }
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        write_extracted_file(dest, basename, &buf)?;
        extracted += 1;
    }
    Ok(extracted)
}

fn path_basename(name: &str) -> Option<&str> {
    name.split('/').next_back().filter(|s| !s.is_empty())
}

fn should_extract_file(basename: &str) -> bool {
    let lowercase = basename.to_lowercase();
    if EXECUTABLES.contains(&basename) {
        return true;
    }
    // Match .so, .so.1, .so.1.2.3, .dll, .dylib
    LIB_EXTENSIONS
        .iter()
        .any(|ext| {
            lowercase.ends_with(ext) || 
            lowercase.contains(&format!("{}.", ext))
        })
}

fn write_extracted_file(dest: &Path, basename: &str, data: &[u8]) -> anyhow::Result<()> {
    let out_path: PathBuf = dest.join(basename);
    let mut out_file = std::fs::File::create(&out_path)?;
    out_file.write_all(data)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if EXECUTABLES.contains(&basename) {
            std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(0o755))?;
        } else {
            // Ensure libraries are also readable/executable if needed (some distros want +x on .so)
            std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(0o644))?;
        }

        // Create symlinks for versioned libraries (e.g., libllama.so.0.0.9189 -> libllama.so.0 -> libllama.so)
        if basename.contains(".so") {
            create_library_symlinks(dest, basename)?;
        }
    }

    Ok(())
}

#[cfg(unix)]
fn create_library_symlinks(dest: &Path, basename: &str) -> anyhow::Result<()> {
    // If we have libllama.so.0.0.9189
    // We want to create:
    //   libllama.so.0
    //   libllama.so
    let parts: Vec<&str> = basename.split('.').collect();
    if parts.len() > 2 && parts[1] == "so" {
        let mut current_name = parts[0].to_string() + ".so";
        // Create the base .so if it doesn't exist
        let base_path = dest.join(&current_name);
        if !base_path.exists() {
            let _ = std::os::unix::fs::symlink(basename, base_path);
        }

        // Create intermediate versioned symlinks (e.g., .so.0)
        for i in 2..parts.len() {
            current_name.push('.');
            current_name.push_str(parts[i]);
            if current_name == basename {
                break;
            }
            let link_path = dest.join(&current_name);
            if !link_path.exists() {
                let _ = std::os::unix::fs::symlink(basename, link_path);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> Asset {
        Asset {
            name: name.to_string(),
            size: 0,
            browser_download_url: String::new(),
        }
    }

    #[test]
    fn find_asset_prefers_plain_macos_arm64_tarball() {
        let assets = [
            asset("llama-b9189-bin-macos-arm64-kleidiai.tar.gz"),
            asset("llama-b9189-bin-macos-arm64.tar.gz"),
            asset("llama-b9189-bin-macos-x64.tar.gz"),
        ];
        let picked = find_asset(&assets, "bin-macos-arm64").unwrap();
        assert_eq!(picked.name, "llama-b9189-bin-macos-arm64.tar.gz");
    }

    #[test]
    fn find_asset_accepts_zip_and_tar_gz() {
        let assets = [asset("llama-b1000-bin-macos-arm64.zip")];
        assert!(find_asset(&assets, "bin-macos-arm64").is_some());

        let assets = [asset("llama-b1000-bin-macos-arm64.tar.gz")];
        assert!(find_asset(&assets, "bin-macos-arm64").is_some());
    }

    #[test]
    fn find_asset_rejects_unrelated_archives() {
        let assets = [asset("llama-b9189-bin-macos-x64.tar.gz")];
        assert!(find_asset(&assets, "bin-macos-arm64").is_none());
    }
}
