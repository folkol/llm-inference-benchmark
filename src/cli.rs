use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::{Path, PathBuf};

use crate::{
    assets,
    compare,
    config,
    doctor,
    hardware,
    interrupt,
    report,
    runner,
    setup,
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
        /// Output directory; omit to write under `./reports/<os-arch>__CPU-…__GPU-…__<run-id>/`.
        #[arg(short, long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Comma-separated device overrides (cpu,gpu,auto); empty = use config
        #[arg(long, default_value = "")]
        devices: String,
        /// Comma-separated model name filters; empty = run all configured models
        #[arg(long, default_value = "")]
        models: String,
        /// Number of warm repetitions per scenario
        #[arg(long, default_value_t = 1)]
        runs: u32,
    },
    /// Directory that contains `report.html`; omit to open the newest under `./reports/`.
    Report {
        dir: Option<PathBuf>,
    },
    /// Download and install the right llama.cpp binaries for this machine
    Setup {
        /// Re-download even if already installed
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Check the environment and report what needs to be fixed before benchmarking
    Doctor {
        /// Path to bench config (used to check model cache status)
        #[arg(short, long, default_value = "bench.toml")]
        config: PathBuf,
    },
    /// Combine results.json files from multiple machines into one comparison report
    Compare {
        /// Two or more results.json files (one per machine)
        #[arg(required = true, num_args = 2..)]
        results: Vec<PathBuf>,
        /// Output HTML file
        #[arg(short, long, default_value = "comparison.html")]
        out: PathBuf,
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
    out: Option<PathBuf>,
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

    interrupt::install();

    let hw = hardware::detect();
    println!("{}", hw.summary());

    let run_id = runner::fresh_run_id();
    let out_dir = match out {
        Some(p) => p,
        None => PathBuf::from("reports").join(format!("{}__{}", hw.report_path_label(), run_id)),
    };

    println!(
        "{} {}",
        "Output dir:".cyan().bold(),
        out_dir.display()
    );

    std::fs::create_dir_all(&out_dir)?;

    let results = runner::run_matrix(&cfg, &hw, &out_dir, run_id)?;
    report::write_all(&results, &hw, &cfg, &out_dir)?;

    let html = out_dir.join("report.html");
    println!(
        "\n{} Report written to {}",
        "Done.".green().bold(),
        html.display()
    );
    Ok(())
}

pub fn cmd_setup(force: bool) -> anyhow::Result<()> {
    setup::run(force)
}

pub fn cmd_doctor(config_path: &Path) -> anyhow::Result<()> {
    let cfg = config::load(config_path).ok();
    let result = doctor::run(cfg.as_ref());
    if result.failed {
        std::process::exit(1);
    }
    Ok(())
}

pub fn cmd_compare(results: &[PathBuf], out: &Path) -> anyhow::Result<()> {
    compare::generate(results, out)?;
    println!("Comparison report written to {}", out.display());
    open_browser(out)?;
    Ok(())
}

pub fn cmd_report_open(dir: Option<PathBuf>) -> anyhow::Result<()> {
    let dir = match dir {
        Some(p) => p,
        None => newest_report_dir(Path::new("reports"))?,
    };

    let html = dir.join("report.html");
    if !html.exists() {
        anyhow::bail!("No report.html in {}", dir.display());
    }
    open_browser(&html)?;
    Ok(())
}

fn newest_report_dir(base: &Path) -> anyhow::Result<PathBuf> {
    if !base.is_dir() {
        anyhow::bail!(
            "No '{}' directory here. Run `{}` first, or pass the report folder explicitly.",
            base.display(),
            "llmb bench".yellow()
        );
    }

    /// Default runs land in `./reports/<slug>/`; curated runs live in `./reports/samples/<slug>/`.
    const MAX_SCAN_DEPTH: usize = 8;

    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let mut stack: Vec<(PathBuf, usize)> = vec![(base.to_path_buf(), 0)];

    while let Some((path, depth)) = stack.pop() {
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|name| name.starts_with('.'))
        {
            continue;
        }

        let report_html = path.join("report.html");
        if report_html.is_file() {
            let mtime = std::fs::metadata(&report_html)?.modified()?;
            if best
                .as_ref()
                .map_or(true, |(t, _)| mtime > *t)
            {
                best = Some((mtime, path));
            }
            continue;
        }

        if depth >= MAX_SCAN_DEPTH || !path.is_dir() {
            continue;
        }

        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            let child = entry.path();
            stack.push((child, depth + 1));
        }
    }

    match best {
        Some((_, p)) => Ok(p),
        None => anyhow::bail!(
            "No report.html found under '{}' (searched nested folders).\n\
             Try `{}` after a benchmark run.",
            base.display(),
            "llmb bench".yellow()
        ),
    }
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
