use std::path::Path;
use std::time::Instant;

use chrono::Utc;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use crate::{
    assets,
    config::{BenchConfig, ModelConfig, WorkloadConfig},
    hardware::HardwareInfo,
    interrupt,
    metrics::{AggregatedMetrics, BatchMetrics, RawSample},
};

/// Top-level output of a complete benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResults {
    /// Increment when `results.json` shape changes materially.
    pub schema_version: u8,
    pub run_id: String,
    pub timestamp: String,
    pub hardware: HardwareInfo,
    pub scenarios: Vec<ScenarioResult>,
}

/// Result for one (model × workload × device) combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub model_name: String,
    pub workload_id: String,
    pub workload_label: String,
    pub device: String,
    pub cold_samples: Vec<RawSample>,
    pub warm_samples: Vec<RawSample>,
    pub cold_metrics: AggregatedMetrics,
    pub warm_metrics: AggregatedMetrics,
    pub batch_results: Vec<BatchMetrics>,
}

pub fn fresh_run_id() -> String {
    format!("{}", Utc::now().format("%Y%m%dT%H%M%S"))
}

pub fn run_matrix(
    cfg: &BenchConfig,
    hw: &HardwareInfo,
    out_dir: &Path,
    run_id: String,
) -> anyhow::Result<RunResults> {
    let server_exe = assets::find_llama_server()?;
    let model_cache = assets::cache_dir()?;

    warn_if_low_memory(cfg);

    let mut results = RunResults {
        schema_version: 2,
        run_id,
        timestamp: Utc::now().to_rfc3339(),
        hardware: hw.clone(),
        scenarios: Vec::new(),
    };

    let total = cfg.models.len() * cfg.workloads.len() * cfg.devices.len();
    let mut done = 0usize;

    for model in &cfg.models {
        if interrupt::requested() {
            println!("\n{}", "Benchmark stopped by user.".yellow());
            break;
        }

        let model_path = model_cache.join(&model.filename);
        if !model_path.exists() {
            eprintln!(
                "{} model file not found: {} – run `llmb models fetch` first",
                "Skip".yellow(),
                model_path.display()
            );
            continue;
        }

        for device in &cfg.devices {
            if interrupt::requested() { break; }
            for workload in &cfg.workloads {
                if interrupt::requested() { break; }

                done += 1;
                let scenario_timeout = estimate_timeout_secs(model, device);

                            println!(
                    "\n[{}/{}] {} | {} | device={} (timeout={}s)",
                    done,
                    total,
                    model.name.bold(),
                    workload.label,
                    device.cyan(),
                    scenario_timeout,
                );

                let scenario = run_scenario(
                    &server_exe,
                    &model_path,
                    model,
                    workload,
                    device,
                    cfg,
                    out_dir,
                )?;

                let line = format!(
                    "  warm_tps={:.1}  cold_tps={:.1}  load={:.0}ms  ttft={:.0}ms",
                    scenario.warm_metrics.tokens_per_sec_mean,
                    scenario.cold_metrics.tokens_per_sec_mean,
                    scenario.cold_metrics.load_time_ms_mean,
                    scenario.cold_metrics.ttft_ms_mean,
                );
                println!("{}", line.dimmed());

                results.scenarios.push(scenario);
            }
        }
    }

    Ok(results)
}

fn run_scenario(
    server_exe: &Path,
    model_path: &Path,
    model: &ModelConfig,
    workload: &WorkloadConfig,
    device: &str,
    cfg: &BenchConfig,
    _out_dir: &Path,
) -> anyhow::Result<ScenarioResult> {
    // ── RAM safety check ─────────────────────────────────────────────────────
    // Refuse to spawn llama-cli if there is not enough free RAM.
    // A shortage causes the OS to thrash: other applications freeze, and even
    // killing llama-cli may take minutes.  Better to record a skipped scenario
    // than to destabilize the machine.
    //
    // Heuristic: model weights (Q4_K_M ≈ 0.6 GB per billion params) +
    //            KV-cache headroom (≈ 50 % of weight size).
    // For GPU runs the model fits in VRAM so we still check system RAM for the
    // KV-cache buffer but give a smaller threshold.
    if let Some(reason) = check_ram_available(model, device) {
        println!("  {}", reason.yellow());
        return Ok(skipped_scenario(model, workload, device));
    }

    let timeout_secs = estimate_timeout_secs(model, device);

    let port = 18765u16;
    let parallel = 1;
    let server_start = Instant::now();
    let mut server = spawn_server(server_exe, model_path, model, device, cfg, port, parallel)?;
    let client = reqwest::blocking::Client::new();
    let base_url = format!("http://127.0.0.1:{}", port);
    let url = format!("{}/completion", base_url);

    let scenario_result = (|| -> anyhow::Result<ScenarioResult> {
        print!("  server load ... ");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let wait_timeout = std::time::Duration::from_secs(timeout_secs);
        if !wait_for_server_ready(&mut server, &client, &base_url, wait_timeout) {
            anyhow::bail!("llama-server failed to start or load model within {}s", timeout_secs);
        }
        let load_ms = server_start.elapsed().as_millis() as f64;
        println!("ready in {:.0}ms", load_ms);

        // 1. Cold-ish sample: server startup/model load + first completion.
        let cold_sample = run_server_completion(
            &client,
            &url,
            workload,
            42,
            load_ms,
            timeout_secs,
            "cold request",
        )?;
    let cold_samples = vec![cold_sample];
    let cold_metrics = AggregatedMetrics::from_samples(&cold_samples);

    // 2. Warm runs reuse the loaded server.
    let mut warm_samples = Vec::new();
    let warm_n = cfg.warm_runs;
    for i in 0..warm_n {
        let label = format!("warm {}/{}", i + 1, warm_n);
        let sample = run_server_completion(
            &client,
            &url,
            workload,
            42 + i as u64,
            0.0,
            timeout_secs,
            &label,
        )?;
        warm_samples.push(sample);
    }
    let warm_metrics = AggregatedMetrics::from_samples(&warm_samples);

    let batch_results = Vec::new();

    Ok(ScenarioResult {
        model_name: model.name.clone(),
        workload_id: workload.id.clone(),
        workload_label: workload.label.clone(),
        device: device.to_string(),
        cold_samples,
        warm_samples,
        cold_metrics,
        warm_metrics,
        batch_results,
    })
    })();

    let _ = server.kill();
    let _ = server.wait();

    scenario_result
}

fn spawn_server(
    server_exe: &Path,
    model_path: &Path,
    model: &ModelConfig,
    device: &str,
    cfg: &BenchConfig,
    port: u16,
    parallel: u32,
) -> anyhow::Result<std::process::Child> {
    let mut server_cmd = std::process::Command::new(server_exe);

    // On Linux, we might need to point LD_LIBRARY_PATH to the directory containing
    // the server executable if it has bundled shared libraries (like libllama.so).
    #[cfg(target_os = "linux")]
    if let Some(parent) = server_exe.parent() {
        if let Ok(abs_parent) = parent.canonicalize() {
            let mut ld_path = abs_parent.to_string_lossy().into_owned();
            if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
                ld_path = format!("{}:{}", ld_path, existing);
            }
            server_cmd.env("LD_LIBRARY_PATH", ld_path);
        }
    }

    let total_context = model.context_length.saturating_mul(parallel);
    server_cmd.arg("--model").arg(model_path);
    server_cmd.arg("--port").arg(port.to_string());
    // llama-server divides the total context across parallel slots. Keep the
    // configured per-request context available for each slot.
    server_cmd.arg("--ctx-size").arg(total_context.to_string());
    server_cmd.arg("--parallel").arg(parallel.to_string());
    server_cmd.arg("--threads").arg(num_cpus().to_string());

    match device {
        "gpu" => {
            server_cmd.arg("--n-gpu-layers").arg(cfg.gpu_layers.to_string());
        }
        "cpu" => {
            // Hide CUDA devices at process start, and tell llama.cpp not to use
            // any device backend. This makes CPU runs truly CPU-only.
            server_cmd.env("CUDA_VISIBLE_DEVICES", "-1");
            server_cmd.arg("--device").arg("none");
            server_cmd.arg("--n-gpu-layers").arg("0");
        }
        "mixed" => {
            server_cmd
                .arg("--n-gpu-layers")
                .arg(cfg.mixed_gpu_layers.to_string());
        }
        _ => {}
    }

    server_cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    Ok(server_cmd.spawn()?)
}

fn run_server_completion(
    client: &reqwest::blocking::Client,
    url: &str,
    workload: &WorkloadConfig,
    seed: u64,
    load_time_ms: f64,
    timeout_secs: u64,
    label: &str,
) -> anyhow::Result<RawSample> {
    let body = completion_body(workload, seed);
    let start = Instant::now();
    let resp = client
        .post(url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .send()?;
    let request_ms = start.elapsed().as_millis() as f64;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        anyhow::bail!("server HTTP {}: {}", status, text);
    }

    let json: serde_json::Value = resp.json()?;
    let sample = sample_from_server_json(&json, workload.max_tokens, load_time_ms, request_ms);
    println!(
        "  {} load={:.0}ms  wall={:.0}ms  tok={}  tps={:.1}",
        label,
        sample.load_time_ms,
        sample.wall_time_ms,
        sample.gen_tokens,
        sample.tokens_per_sec
    );
    Ok(sample)
}

fn completion_body(workload: &WorkloadConfig, seed: u64) -> serde_json::Value {
    serde_json::json!({
        "prompt": workload.prompt,
        "n_predict": workload.max_tokens,
        "seed": seed,
        "cache_prompt": true,
    })
}

fn sample_from_server_json(
    json: &serde_json::Value,
    max_tokens: u32,
    load_time_ms: f64,
    request_ms: f64,
) -> RawSample {
    let gen_tokens = server_token_count(json).unwrap_or(max_tokens as u64);
    let prompt_tokens = json_u64(json, &["tokens_evaluated"])
        .or_else(|| json_u64(json, &["timings", "prompt_n"]))
        .unwrap_or(0);
    let prompt_ms = json_f64(json, &["timings", "prompt_ms"])
        .or_else(|| json_f64(json, &["prompt_ms"]))
        .unwrap_or(0.0);
    let gen_ms = json_f64(json, &["timings", "predicted_ms"])
        .or_else(|| json_f64(json, &["predicted_ms"]))
        .unwrap_or(request_ms.max(1.0));
    let tps = json_f64(json, &["timings", "predicted_per_second"])
        .or_else(|| json_f64(json, &["tokens_per_second"]))
        .unwrap_or_else(|| gen_tokens as f64 / (gen_ms / 1000.0).max(0.001));

    RawSample {
        completion: server_completion_text(json),
        wall_time_ms: load_time_ms + request_ms,
        load_time_ms,
        prompt_tokens,
        prompt_eval_ms: prompt_ms,
        gen_tokens,
        gen_eval_ms: gen_ms,
        tokens_per_sec: tps,
        ttft_ms: load_time_ms + prompt_ms,
        success: true,
    }
}

fn server_completion_text(json: &serde_json::Value) -> Option<String> {
    json.get("content")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            json.get("text")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .or_else(|| {
            json.get("generation")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
}

fn server_token_count(json: &serde_json::Value) -> Option<u64> {
    json_u64(json, &["tokens_predicted"])
        .or_else(|| json_u64(json, &["timings", "predicted_n"]))
        .or_else(|| json_u64(json, &["completion_tokens"]))
}

fn json_u64(json: &serde_json::Value, path: &[&str]) -> Option<u64> {
    let mut value = json;
    for key in path {
        value = value.get(*key)?;
    }
    value.as_u64()
}

fn json_f64(json: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut value = json;
    for key in path {
        value = value.get(*key)?;
    }
    value.as_f64()
}

fn wait_for_server_ready(
    child: &mut std::process::Child,
    client: &reqwest::blocking::Client,
    base_url: &str,
    timeout: std::time::Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    let health_url = format!("{}/health", base_url);

    while Instant::now() < deadline {
        // Check if the process has already exited (crashed)
        if let Ok(Some(status)) = child.try_wait() {
            eprintln!("\n  {} llama-server exited early with status: {}", "Error:".red(), status);
            return false;
        }

        if let Ok(resp) = client
            .get(&health_url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
        {
            if resp.status().is_success() {
                return true;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    false
}

/// Returns `Some(reason)` if there is not enough free RAM to safely run this
/// model, or `None` if it looks safe to proceed.
fn check_ram_available(model: &ModelConfig, device: &str) -> Option<String> {
    use sysinfo::{System, SystemExt};
    let mut sys = System::new_all();
    sys.refresh_memory();
    let avail_gb = sys.available_memory() as f64 / 1_073_741_824.0;

    let params_b: f64 = model
        .params
        .as_deref()
        .and_then(|p| p.trim_end_matches('B').parse().ok())
        .unwrap_or(4.0);

    // Weight size estimate + 50 % for KV-cache and runtime overhead.
    let weight_gb = params_b * 0.6;
    let required_gb = match device {
        "gpu" => weight_gb * 0.15, // most weight is in VRAM; only KV-cache in RAM
        _ => weight_gb * 1.5,
    };

    // Keep a 1.5 GB safety margin for the OS and other processes to breathe.
    let safe_threshold = required_gb + 1.5;

    if avail_gb < safe_threshold {
        Some(format!(
            "SKIP: only {:.1} GB RAM available, need ~{:.1} GB for {} ({}) — \
             close other apps and retry.",
            avail_gb, safe_threshold, model.name, device
        ))
    } else {
        None
    }
}

/// Build a zeroed-out scenario result that records a skipped run.
fn skipped_scenario(
    model: &ModelConfig,
    workload: &WorkloadConfig,
    device: &str,
) -> ScenarioResult {
    let empty = AggregatedMetrics::default();
    ScenarioResult {
        model_name: model.name.clone(),
        workload_id: workload.id.clone(),
        workload_label: workload.label.clone(),
        device: device.to_string(),
        cold_samples: vec![],
        warm_samples: vec![],
        cold_metrics: empty.clone(),
        warm_metrics: empty,
        batch_results: vec![],
    }
}

fn warn_if_low_memory(cfg: &BenchConfig) {
    use sysinfo::{System, SystemExt};
    let mut sys = System::new_all();
    sys.refresh_memory();
    let avail_gb = sys.available_memory() as f64 / 1_073_741_824.0;

    let largest_gb: f64 = cfg
        .models
        .iter()
        .filter_map(|m| m.params.as_deref())
        .filter_map(|p| p.trim_end_matches('B').parse::<f64>().ok())
        .map(|b| b * 0.6)
        .fold(0.0_f64, f64::max);

    if avail_gb < 2.0 {
        println!(
            "{} Only {:.1} GB RAM available — close other applications before benchmarking.",
            "WARNING:".yellow().bold(),
            avail_gb
        );
    } else if largest_gb > 0.0 && avail_gb < largest_gb * 1.5 + 1.5 {
        println!(
            "{} {:.1} GB RAM available; the largest model needs ~{:.1} GB. \
             Scenarios that don't fit will be skipped automatically.",
            "Note:".yellow().bold(),
            avail_gb,
            largest_gb * 1.5 + 1.5,
        );
    }
}

/// Estimate a generous per-inference timeout.
///
/// The CUDA-enabled llama-cli binary initialises the GPU runtime at startup
/// even for CPU-only runs (--n-gpu-layers 0).  This can take 30–60 s on a
/// cold system.  We add a fixed 90 s base to cover that overhead, then scale
/// by model size and device.  The cap is 20 min so a truly stuck process
/// doesn't block the entire run indefinitely.
fn estimate_timeout_secs(model: &crate::config::ModelConfig, device: &str) -> u64 {
    let params_b: f64 = model
        .params
        .as_deref()
        .and_then(|p| p.trim_end_matches('B').parse().ok())
        .unwrap_or(8.0);

    // Per-param scaling: how many seconds per billion params.
    let secs_per_b = match device {
        "gpu"   => 10.0,   // GPU is fast
        "mixed" => 30.0,
        _       => 60.0,   // CPU: ~60 s per billion params on a mid-range desktop
    };

    // Fixed overhead for process startup + model load.
    // CPU runs use --device none so the CUDA runtime is never initialised;
    // GPU/mixed runs still need time for the driver + CUDA JIT warm-up.
    let startup_overhead = match device {
        "cpu" => 30.0_f64,
        _     => 90.0_f64,
    };

    let total = startup_overhead + params_b * secs_per_b * 4.0;
    (total as u64).clamp(120, 1200) // 2 min minimum, 20 min cap
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

