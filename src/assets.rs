use crate::config::ModelConfig;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Returns the platform-appropriate cache directory.
pub fn cache_dir() -> anyhow::Result<PathBuf> {
    let base = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?;
    let dir = base.join("llm-benchmark");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Returns the path to the llama.cpp binary directory in cache.
pub fn llama_bin_dir() -> anyhow::Result<PathBuf> {
    let dir = cache_dir()?.join("llama-bin");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Returns the platform executable name for llama-cli (cold-start) and llama-server (batch).
pub fn llama_cli_exe() -> &'static str {
    if cfg!(target_os = "windows") {
        "llama-cli.exe"
    } else {
        "llama-cli"
    }
}

pub fn llama_server_exe() -> &'static str {
    if cfg!(target_os = "windows") {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// Resolves the full path to llama-cli, searching PATH first then cache.
pub fn find_llama_cli() -> anyhow::Result<PathBuf> {
    find_exe(llama_cli_exe())
}

pub fn find_llama_server() -> anyhow::Result<PathBuf> {
    find_exe(llama_server_exe())
}

fn find_exe(name: &str) -> anyhow::Result<PathBuf> {
    // 1. Check PATH
    if let Ok(p) = which_exe(name) {
        return Ok(p);
    }
    // 2. Check cache bin dir
    let cached = llama_bin_dir()?.join(name);
    if cached.exists() {
        return Ok(cached);
    }
    anyhow::bail!(
        "Cannot find '{}'. Install llama.cpp and ensure it is on PATH, \
         or download prebuilt binaries to {} .\n\
         See: https://github.com/ggerganov/llama.cpp/releases",
        name,
        llama_bin_dir()?.display()
    )
}

fn which_exe(name: &str) -> anyhow::Result<PathBuf> {
    let out = std::process::Command::new(if cfg!(target_os = "windows") {
        "where"
    } else {
        "which"
    })
    .arg(name)
    .output()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout);
        let first = s.lines().next().unwrap_or("").trim().to_string();
        if !first.is_empty() {
            return Ok(PathBuf::from(first));
        }
    }
    anyhow::bail!("not found in PATH")
}

/// Download all configured models that are not already cached.
pub fn fetch_models(models: &[ModelConfig]) -> anyhow::Result<()> {
    let cache = cache_dir()?;
    for model in models {
        let dest = cache.join(&model.filename);
        if dest.exists() {
            if let Some(expected) = &model.sha256 {
                print!("Verifying {} ... ", model.name);
                match verify_sha256(&dest, expected) {
                    Ok(true) => println!("{}", "ok".green()),
                    Ok(false) => {
                        println!("{} – re-downloading", "checksum mismatch".red());
                        std::fs::remove_file(&dest)?;
                        download_file(&model.url, &dest, &model.name)?;
                        verify_or_bail(&dest, expected, &model.name)?;
                    }
                    Err(e) => println!("warn: {}", e),
                }
            } else {
                println!("{} {} (no checksum configured)", "Cached".green(), model.name);
            }
            continue;
        }
        download_file(&model.url, &dest, &model.name)?;
        if let Some(expected) = &model.sha256 {
            verify_or_bail(&dest, expected, &model.name)?;
        }
    }
    Ok(())
}

fn download_file(url: &str, dest: &PathBuf, label: &str) -> anyhow::Result<()> {
    use std::io::Write;

    println!("Downloading {} ...", label.bold());

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()?;

    let response = client.get(url).send()?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP {} for {}", response.status(), url);
    }

    let total = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let tmp = dest.with_extension("tmp");
    let mut file = std::fs::File::create(&tmp)?;
    let bytes = response.bytes()?;
    pb.inc(bytes.len() as u64);
    file.write_all(&bytes)?;
    pb.finish_with_message("done");

    std::fs::rename(&tmp, dest)?;
    Ok(())
}

pub fn verify_sha256(path: &PathBuf, expected: &str) -> anyhow::Result<bool> {
    let data = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let result = hex::encode(hasher.finalize());
    Ok(result.eq_ignore_ascii_case(expected))
}

fn verify_or_bail(path: &PathBuf, expected: &str, name: &str) -> anyhow::Result<()> {
    match verify_sha256(path, expected)? {
        true => {
            println!("{} checksum verified for {}", "✓".green(), name);
            Ok(())
        }
        false => anyhow::bail!("SHA256 mismatch for {} after download", name),
    }
}
