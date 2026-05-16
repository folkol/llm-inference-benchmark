/// `llmb setup` — download and install the right llama.cpp binaries automatically.
///
/// 1. Queries the GitHub releases API for the latest llama.cpp release.
/// 2. Picks the correct asset ZIP based on OS + detected GPU.
/// 3. Downloads and extracts into the llama-bin cache directory.

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Write;

use crate::assets::llama_bin_dir;

const GITHUB_API: &str =
    "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest";

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
            Backend::CudaCu12    => "bin-win-cuda-cu12",
            Backend::Vulkan      => "bin-win-vulkan",
            Backend::CpuOnly     => "bin-win-noavx2",
            Backend::MacosArm64  => "bin-macos-arm64",
            Backend::MacosX64    => "bin-macos-x64",
            Backend::LinuxCuda   => "bin-ubuntu-x64",
            Backend::LinuxCpu    => "bin-ubuntu-x64",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Backend::CudaCu12    => "Windows / NVIDIA CUDA 12",
            Backend::Vulkan      => "Windows / Vulkan (AMD / Intel GPU)",
            Backend::CpuOnly     => "Windows / CPU only (no GPU)",
            Backend::MacosArm64  => "macOS / Apple Silicon",
            Backend::MacosX64    => "macOS / Intel",
            Backend::LinuxCuda   => "Linux / NVIDIA CUDA",
            Backend::LinuxCpu    => "Linux / CPU only",
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
    // vulkaninfo is typically present when Vulkan drivers are installed.
    std::process::Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Download and install the backend for the given `Backend` variant.
pub fn run(force: bool) -> anyhow::Result<()> {
    let bin_dir = llama_bin_dir()?;

    // Check if already installed.
    let already_installed = {
        let cli = bin_dir.join(crate::assets::llama_cli_exe());
        cli.exists()
    };

    if already_installed && !force {
        println!(
            "{} llama-cli already installed in {}",
            "Skipping:".yellow(),
            bin_dir.display()
        );
        println!("Run with --force to reinstall.");
        return Ok(());
    }

    let backend = detect_backend();
    println!("Detected platform: {}", backend.label().bold());
    println!("Querying GitHub for the latest llama.cpp release...");

    let release = fetch_latest_release()?;
    println!("Latest release: {}", release.tag_name.bold());

    // Find the main binary asset.
    let keyword = backend.asset_keyword();
    let main_asset = release
        .assets
        .iter()
        .find(|a| a.name.contains(keyword) && a.name.ends_with(".zip"))
        .ok_or_else(|| {
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
    let main_zip = download_to_memory(&main_asset.browser_download_url, main_asset.size)?;
    extract_zip(&main_zip, &bin_dir, &main_asset.name)?;

    // On Windows with CUDA, also download the cudart DLL bundle.
    if let Some(cudart_kw) = backend.cudart_keyword() {
        if let Some(cudart_asset) = release
            .assets
            .iter()
            .find(|a| a.name.contains(cudart_kw) && a.name.ends_with(".zip"))
        {
            println!(
                "Downloading CUDA runtime {} ({:.1} MB)...",
                cudart_asset.name.bold(),
                cudart_asset.size as f64 / 1_048_576.0
            );
            let cudart_zip =
                download_to_memory(&cudart_asset.browser_download_url, cudart_asset.size)?;
            extract_zip(&cudart_zip, &bin_dir, &cudart_asset.name)?;
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

// ── ZIP extraction ────────────────────────────────────────────────────────────

fn extract_zip(data: &[u8], dest: &std::path::Path, zip_name: &str) -> anyhow::Result<()> {
    use std::io::Read;
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let executables = ["llama-cli", "llama-server", "llama-cli.exe", "llama-server.exe"];
    let dll_extensions = ["dll", "so", "dylib"];

    let mut extracted = 0usize;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();

        // Extract executables and shared libraries; skip everything else.
        let basename = name.split('/').last().unwrap_or(&name);
        let ext = basename.rsplit('.').next().unwrap_or("").to_lowercase();
        let is_exe = executables.contains(&basename);
        let is_lib = dll_extensions.contains(&ext.as_str());

        if !is_exe && !is_lib {
            continue;
        }

        let out_path = dest.join(basename);
        let mut out_file = std::fs::File::create(&out_path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        out_file.write_all(&buf)?;

        // Make executable on Unix.
        #[cfg(unix)]
        if is_exe {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(0o755))?;
        }

        extracted += 1;
    }

    println!(
        "  {} Extracted {} files from {}",
        "✓".green(),
        extracted,
        zip_name
    );
    Ok(())
}
