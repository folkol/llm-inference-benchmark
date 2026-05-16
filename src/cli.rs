use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::{Path, PathBuf};

use crate::{
    assets,
    config,
    hardware,
    report,
    runner,
};

#[derive(Parser)]
#[command(
    name = "llmb",
    version,
    about = "Cross-platform LLM inference benchmark tool",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Write a starter benchmark config to the current directory
    Init {
        #[arg(default_value = "bench.toml")]
        config: PathBuf,
    },
    /// Manage GGUF models in the local cache
    Models {
        #[command(subcommand)]
        command: ModelsCommands,
    },
    /// Run the benchmark matrix
    Bench {
        /// Path to benchmark config file
        #[arg(short, long, default_value = "bench.toml")]
        config: PathBuf,
        /// Output directory for this run
        #[arg(short, long, default_value = "reports/latest")]
        out: PathBuf,
        /// Comma-separated device overrides (cpu,gpu,auto); empty = use config
        #[arg(long, default_value = "")]
        devices: String,
        /// Comma-separated model name filters; empty = run all configured models
        #[arg(long, default_value = "")]
        models: String,
        /// Number of warm repetitions per scenario
        #[arg(long, default_value_t = 3)]
        runs: u32,
    },
    /// Open the HTML report for the most-recent run
    Report {
        #[arg(default_value = "reports/latest")]
        dir: PathBuf,
    },
}

#[derive(Subcommand)]
pub enum ModelsCommands {
    /// List configured models and their cache status
    List {
        #[arg(short, long, default_value = "bench.toml")]
        config: PathBuf,
    },
    /// Download missing models and verify checksums
    Fetch {
        #[arg(short, long, default_value = "bench.toml")]
        config: PathBuf,
    },
}

pub fn cmd_init(config_path: &Path) -> anyhow::Result<()> {
    if config_path.exists() {
        println!(
            "{} already exists – skipping. Delete it first to regenerate.",
            config_path.display()
        );
        return Ok(());
    }
    let default_toml = include_str!("../configs/default.toml");
    std::fs::write(config_path, default_toml)?;
    println!(
        "{} {}",
        "Created".green().bold(),
        config_path.display()
    );
    println!("Edit it to adjust models, devices, and workloads, then run:");
    println!("  llmb models fetch");
    println!("  llmb bench run");
    Ok(())
}

pub fn cmd_models_list(config_path: &Path) -> anyhow::Result<()> {
    let cfg = config::load(config_path)?;
    let cache_dir = assets::cache_dir()?;
    println!("{:<40} {:<12} {:<10} {}", "Name", "Params", "Quant", "Status");
    println!("{}", "-".repeat(80));
    for m in &cfg.models {
        let path = cache_dir.join(&m.filename);
        let status = if path.exists() {
            "cached".green().to_string()
        } else {
            "missing".yellow().to_string()
        };
        println!(
            "{:<40} {:<12} {:<10} {}",
            m.name,
            m.params.as_deref().unwrap_or("-"),
            m.quantization.as_deref().unwrap_or("-"),
            status
        );
    }
    Ok(())
}

pub fn cmd_models_fetch(config_path: &Path) -> anyhow::Result<()> {
    let cfg = config::load(config_path)?;
    assets::fetch_models(&cfg.models)?;
    Ok(())
}

pub fn cmd_bench_run(
    config_path: &Path,
    out_dir: &Path,
    devices_filter: &str,
    models_filter: &str,
    warm_runs: u32,
) -> anyhow::Result<()> {
    let mut cfg = config::load(config_path)?;

    if !devices_filter.is_empty() {
        cfg.devices = devices_filter
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
    }

    if !models_filter.is_empty() {
        let names: Vec<&str> = models_filter.split(',').map(|s| s.trim()).collect();
        cfg.models.retain(|m| names.iter().any(|n| m.name.contains(n)));
    }

    cfg.warm_runs = warm_runs;

    let hw = hardware::detect();
    println!("{}", hw.summary());

    std::fs::create_dir_all(out_dir)?;

    let results = runner::run_matrix(&cfg, &hw, out_dir)?;
    report::write_all(&results, &hw, &cfg, out_dir)?;

    let html = out_dir.join("report.html");
    println!(
        "\n{} Report written to {}",
        "Done.".green().bold(),
        html.display()
    );
    Ok(())
}

pub fn cmd_report_open(dir: &Path) -> anyhow::Result<()> {
    let html = dir.join("report.html");
    if !html.exists() {
        anyhow::bail!("No report found at {}", html.display());
    }
    open_browser(&html)?;
    Ok(())
}

fn open_browser(path: &Path) -> anyhow::Result<()> {
    let url = format!("file://{}", path.display());
    #[cfg(target_os = "windows")]
    std::process::Command::new("cmd")
        .args(["/c", "start", "", &url])
        .spawn()?;
    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg(&url).spawn()?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open").arg(&url).spawn()?;
    Ok(())
}
