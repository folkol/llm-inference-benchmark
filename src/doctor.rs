/// `llmb doctor` — environment health check.
///
/// Each check prints a status line:
///   [ok]   everything fine
///   [warn] works but suboptimal
///   [fail] blocking problem with a concrete fix suggestion
///
/// Exit code is 0 if no [fail] checks, 1 if any fail.

use colored::Colorize;
use std::path::Path;

use crate::{assets, config::BenchConfig};

pub struct DoctorResult {
    pub failed: bool,
}

struct Check {
    label: String,
    status: Status,
    detail: String,
    fix: Option<String>,
}

enum Status {
    Ok,
    Warn,
    Fail,
}

impl Check {
    fn ok(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { label: label.into(), status: Status::Ok, detail: detail.into(), fix: None }
    }
    fn warn(label: impl Into<String>, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self { label: label.into(), status: Status::Warn, detail: detail.into(), fix: Some(fix.into()) }
    }
    fn fail(label: impl Into<String>, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self { label: label.into(), status: Status::Fail, detail: detail.into(), fix: Some(fix.into()) }
    }

    fn print(&self) {
        let tag = match self.status {
            Status::Ok   => "[ok]  ".green().bold(),
            Status::Warn => "[warn]".yellow().bold(),
            Status::Fail => "[fail]".red().bold(),
        };
        println!("{} {:<35} {}", tag, self.label, self.detail);
        if let Some(fix) = &self.fix {
            for line in fix.lines() {
                println!("       {}", line.dimmed());
            }
        }
    }

    fn is_fail(&self) -> bool {
        matches!(self.status, Status::Fail)
    }
}

pub fn run(cfg: Option<&BenchConfig>) -> DoctorResult {
    let mut checks: Vec<Check> = Vec::new();

    check_llama_cli(&mut checks);
    check_llama_server(&mut checks);
    check_llama_version(&mut checks);
    check_cuda_runtime(&mut checks);
    check_cache_dir(&mut checks);
    check_disk_space(&mut checks);
    check_ram(&mut checks, cfg);
    check_gpu(&mut checks, cfg);
    if let Some(cfg) = cfg {
        check_models(&mut checks, cfg);
    }

    println!("\n{}\n", "llmb doctor".bold().underline());

    let mut failed = false;
    for check in &checks {
        check.print();
        if check.is_fail() {
            failed = true;
        }
    }

    println!();
    if failed {
        println!("{}", "Some checks failed. Fix the issues above before running llmb bench.".red());
    } else {
        println!("{}", "All checks passed. You are ready to run: llmb bench".green().bold());
    }

    DoctorResult { failed }
}

// ── individual checks ─────────────────────────────────────────────────────────

fn check_llama_cli(out: &mut Vec<Check>) {
    match assets::find_llama_cli() {
        Ok(path) => out.push(Check::ok(
            "llama-cli",
            format!("found at {}", path.display()),
        )),
        Err(_) => out.push(Check::fail(
            "llama-cli",
            "not found on PATH or in cache bin dir",
            "Run:  llmb setup\n\
             This will auto-detect your GPU and download the correct llama.cpp binary.\n\
             \n\
             Or install manually from:\n\
             https://github.com/ggerganov/llama.cpp/releases",
        )),
    }
}

fn check_llama_server(out: &mut Vec<Check>) {
    match assets::find_llama_server() {
        Ok(path) => out.push(Check::ok(
            "llama-server",
            format!("found at {}", path.display()),
        )),
        Err(_) => out.push(Check::warn(
            "llama-server",
            "not found — batch benchmarks will be skipped",
            "Run:  llmb setup\n\
             llama-server is included in the same ZIP as llama-cli and will be installed automatically.",
        )),
    }
}

fn check_llama_version(out: &mut Vec<Check>) {
    let Ok(cli) = assets::find_llama_cli() else { return };

    match std::process::Command::new(&cli).arg("--version").output() {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_string();
            // llama.cpp b9174+ supports --system-prompt; earlier builds do not.
            let build_num: Option<u32> = text
                .split_whitespace()
                .find_map(|w| w.parse().ok());
            match build_num {
                Some(n) if n >= 3000 => out.push(Check::ok("llama-cli version", &text)),
                Some(_) => out.push(Check::warn(
                    "llama-cli version",
                    format!("{} — may be too old", text),
                    "Build b3000+ recommended for --system-prompt support.\n\
                     Download a newer release from:\n\
                     https://github.com/ggerganov/llama.cpp/releases",
                )),
                None => out.push(Check::ok("llama-cli version", &text)),
            }
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            out.push(Check::fail(
                "llama-cli version",
                "llama-cli exited with an error",
                format!(
                    "Running `llama-cli --version` failed:\n{}\n\
                     The binary may be corrupt or missing required DLLs.",
                    stderr.lines().take(4).collect::<Vec<_>>().join("\n")
                ),
            ));
        }
        Err(e) => out.push(Check::fail(
            "llama-cli version",
            format!("could not execute: {}", e),
            "Ensure llama-cli is executable and all required DLLs are present.",
        )),
    }
}

fn check_cuda_runtime(out: &mut Vec<Check>) {
    // Only relevant on Windows; on Linux/macOS the runtime is handled differently.
    if std::env::consts::OS != "windows" {
        return;
    }

    // Check for nvidia-smi to confirm an NVIDIA GPU is present.
    let has_gpu = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !has_gpu {
        return; // No NVIDIA GPU — CUDA runtime not needed.
    }

    // Check that cudart64_*.dll is findable (it ships in the cudart ZIP from llama.cpp releases).
    let bin_dir = match assets::llama_bin_dir() {
        Ok(d) => d,
        Err(_) => return,
    };

    let cudart_found = std::fs::read_dir(&bin_dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .to_lowercase()
                        .starts_with("cudart64_")
                })
        })
        .unwrap_or(false);

    if cudart_found {
        out.push(Check::ok("CUDA runtime DLLs", "cudart64_*.dll found in llama bin dir"));
    } else {
        // Also check if it's in PATH / system dirs.
        let in_path = dll_in_path("cudart64_");
        if in_path {
            out.push(Check::ok("CUDA runtime DLLs", "cudart64_*.dll found on PATH / system"));
        } else {
        out.push(Check::fail(
            "CUDA runtime DLLs",
            "cudart64_*.dll not found — llama-cli will crash at runtime",
            "Run:  llmb setup\n\
             This downloads and extracts the CUDA runtime DLLs automatically.",
        ));
        }
    }
}

fn dll_in_path(prefix: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(';') {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let name = entry.file_name().to_string_lossy().to_lowercase();
                    if name.starts_with(prefix) && name.ends_with(".dll") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn check_cache_dir(out: &mut Vec<Check>) {
    match assets::cache_dir() {
        Ok(dir) => out.push(Check::ok("cache directory", dir.display().to_string())),
        Err(e) => out.push(Check::fail(
            "cache directory",
            format!("cannot create: {}", e),
            "Check that your home directory is writable.",
        )),
    }
}

fn check_disk_space(out: &mut Vec<Check>) {
    let Ok(cache) = assets::cache_dir() else { return };

    // Walk cache dir and sum file sizes.
    let used_gb: f64 = dir_size_gb(&cache);

    // Estimate remaining space on the volume the cache sits on.
    let free_gb = free_space_gb(&cache);

    let label = "disk space";
    let detail = format!(
        "{:.1} GB used in cache, {:.1} GB free",
        used_gb, free_gb
    );

    // Warn if less than 5 GB free (32B model is ~19.8 GB but they may have it cached already).
    if free_gb < 5.0 {
        out.push(Check::fail(
            label,
            detail,
            "Less than 5 GB free. Free up disk space before running models fetch.",
        ));
    } else if free_gb < 25.0 {
        out.push(Check::warn(
            label,
            detail,
            "Less than 25 GB free. The 32B model needs ~19.8 GB. \
             Ensure you have enough space before models fetch.",
        ));
    } else {
        out.push(Check::ok(label, detail));
    }
}

fn dir_size_gb(dir: &Path) -> f64 {
    let bytes: u64 = walkdir_size(dir);
    bytes as f64 / 1_073_741_824.0
}

fn walkdir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .filter_map(|e| e.ok())
        .map(|e| {
            let meta = e.metadata().ok();
            if let Some(m) = meta {
                if m.is_dir() {
                    walkdir_size(&e.path())
                } else {
                    m.len()
                }
            } else {
                0
            }
        })
        .sum()
}

fn free_space_gb(path: &Path) -> f64 {
    // Use platform-specific calls to get available disk space.
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let mut free_bytes: u64 = 0;
        let mut total_bytes: u64 = 0;
        let mut total_free: u64 = 0;
        unsafe {
            windows_get_disk_free(wide.as_ptr(), &mut free_bytes, &mut total_bytes, &mut total_free);
        }
        free_bytes as f64 / 1_073_741_824.0
    }
    #[cfg(not(target_os = "windows"))]
    {
        // Use `df` as a portable fallback on Linux/macOS.
        let out = std::process::Command::new("df")
            .arg("-k")
            .arg(path)
            .output()
            .ok();
        if let Some(o) = out {
            let text = String::from_utf8_lossy(&o.stdout);
            // df output: Filesystem 1K-blocks Used Available Use% Mounted
            if let Some(line) = text.lines().nth(1) {
                if let Some(avail) = line.split_whitespace().nth(3) {
                    if let Ok(kb) = avail.parse::<u64>() {
                        return kb as f64 / 1_048_576.0; // KB → GB
                    }
                }
            }
        }
        999.0 // unknown
    }
}

#[cfg(target_os = "windows")]
unsafe fn windows_get_disk_free(
    path: *const u16,
    free_bytes_available: &mut u64,
    total_bytes: &mut u64,
    total_free_bytes: &mut u64,
) {
    // GetDiskFreeSpaceExW
    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            lpDirectoryName: *const u16,
            lpFreeBytesAvailableToCaller: *mut u64,
            lpTotalNumberOfBytes: *mut u64,
            lpTotalNumberOfFreeBytes: *mut u64,
        ) -> i32;
    }
    GetDiskFreeSpaceExW(path, free_bytes_available, total_bytes, total_free_bytes);
}

fn check_ram(out: &mut Vec<Check>, cfg: Option<&BenchConfig>) {
    use sysinfo::{System, SystemExt};
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_gb = sys.total_memory() as f64 / 1_073_741_824.0;
    let avail_gb = sys.available_memory() as f64 / 1_073_741_824.0;

    let detail = format!("{:.1} GB total, {:.1} GB available", total_gb, avail_gb);

    // Estimate the largest model the user has configured.
    let largest_gb = cfg
        .map(|c| largest_model_estimate_gb(c))
        .unwrap_or(0.0);

    if largest_gb > 0.0 && avail_gb < largest_gb * 1.1 {
        out.push(Check::warn(
            "RAM",
            detail,
            format!(
                "Largest configured model is ~{:.0} GB. \
                 You may need to close other applications before running.",
                largest_gb
            ),
        ));
    } else {
        out.push(Check::ok("RAM", detail));
    }
}

fn largest_model_estimate_gb(cfg: &BenchConfig) -> f64 {
    // Rough Q4_K_M size estimate: ~0.6 GB per billion params.
    cfg.models
        .iter()
        .filter_map(|m| m.params.as_deref())
        .filter_map(|p| p.trim_end_matches('B').parse::<f64>().ok())
        .map(|b| b * 0.6)
        .fold(0.0_f64, f64::max)
}

fn check_gpu(out: &mut Vec<Check>, cfg: Option<&BenchConfig>) {
    // Only check if the config actually requests GPU benchmarking.
    let wants_gpu = cfg
        .map(|c| c.devices.iter().any(|d| d == "gpu" || d == "mixed"))
        .unwrap_or(true);

    if !wants_gpu {
        return;
    }

    // Try nvidia-smi.
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if o.status.success() {
            let text = String::from_utf8_lossy(&o.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.splitn(2, ',').collect();
                if parts.len() == 2 {
                    let name = parts[0].trim();
                    let vram_mb: u64 = parts[1].trim().parse().unwrap_or(0);
                    let vram_gb = vram_mb as f64 / 1024.0;
                    let largest_gb = cfg
                        .map(|c| largest_model_estimate_gb(c))
                        .unwrap_or(0.0);
                    let detail = format!("{} — {:.0} GB VRAM", name, vram_gb);
                    if largest_gb > vram_gb {
                        out.push(Check::warn(
                            "GPU",
                            detail,
                            format!(
                                "Largest model (~{:.0} GB) exceeds VRAM ({:.0} GB).\n\
                                 llama.cpp will use mixed CPU/GPU for those models,\n\
                                 which is expected — this is the VRAM cliff you want to measure.",
                                largest_gb, vram_gb
                            ),
                        ));
                    } else {
                        out.push(Check::ok("GPU", detail));
                    }
                }
            }
            return;
        }
    }

    // Try Metal (macOS).
    #[cfg(target_os = "macos")]
    {
        if std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            out.push(Check::ok("GPU", "Apple Silicon GPU (Metal) detected"));
            return;
        }
    }

    out.push(Check::warn(
        "GPU",
        "no GPU detected (nvidia-smi not found)",
        "If you have a GPU, ensure the driver is installed and nvidia-smi is on PATH.\n\
         Benchmarks will run on CPU only.",
    ));
}

fn check_models(out: &mut Vec<Check>, cfg: &BenchConfig) {
    let Ok(cache) = assets::cache_dir() else { return };

    let mut cached = 0usize;
    let mut missing = Vec::new();

    for m in &cfg.models {
        let path = cache.join(&m.filename);
        if path.exists() {
            cached += 1;
        } else {
            missing.push(m.name.clone());
        }
    }

    if missing.is_empty() {
        out.push(Check::ok(
            "models",
            format!("all {} models cached", cached),
        ));
    } else {
        out.push(Check::fail(
            "models",
            format!("{} cached, {} missing: {}", cached, missing.len(), missing.join(", ")),
            "Run: llmb models fetch",
        ));
    }
}
