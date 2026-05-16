use std::path::Path;
use std::time::Instant;

use chrono::Utc;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use crate::{
    assets,
    config::{BenchConfig, ModelConfig, WorkloadConfig},
    hardware::HardwareInfo,
    llama::{self, LlamaArgs},
    metrics::{AggregatedMetrics, BatchMetrics, RawSample},
    scoring::{self, ScoreBreakdown},
};

/// Top-level output of a complete benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResults {
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
    pub score: f64,
    pub score_breakdown: ScoreBreakdownSer,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoreBreakdownSer {
    pub tokens_per_sec_score: f64,
    pub load_time_score: f64,
    pub ttft_score: f64,
    pub batch_throughput_score: f64,
    pub aggregate: f64,
}

impl From<ScoreBreakdown> for ScoreBreakdownSer {
    fn from(s: ScoreBreakdown) -> Self {
        Self {
            tokens_per_sec_score: s.tokens_per_sec_score,
            load_time_score: s.load_time_score,
            ttft_score: s.ttft_score,
            batch_throughput_score: s.batch_throughput_score,
            aggregate: s.aggregate,
        }
    }
}

pub fn run_matrix(
    cfg: &BenchConfig,
    hw: &HardwareInfo,
    out_dir: &Path,
) -> anyhow::Result<RunResults> {
    let cli_exe = assets::find_llama_cli()?;
    let model_cache = assets::cache_dir()?;

    let run_id = format!("{}", Utc::now().format("%Y%m%dT%H%M%S"));
    let mut results = RunResults {
        schema_version: 1,
        run_id: run_id.clone(),
        timestamp: Utc::now().to_rfc3339(),
        hardware: hw.clone(),
        scenarios: Vec::new(),
    };

    let total = cfg.models.len() * cfg.workloads.len() * cfg.devices.len();
    let mut done = 0usize;

    for model in &cfg.models {
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
            for workload in &cfg.workloads {
                done += 1;
                println!(
                    "\n[{}/{}] {} | {} | device={}",
                    done,
                    total,
                    model.name.bold(),
                    workload.label,
                    device.cyan()
                );

                let scenario = run_scenario(
                    &cli_exe,
                    &model_path,
                    model,
                    workload,
                    device,
                    cfg,
                    out_dir,
                )?;

                let line = format!(
                    "  tps={:.1}  load={:.0}ms  ttft={:.0}ms  score={:.1}",
                    scenario.warm_metrics.tokens_per_sec_mean,
                    scenario.cold_metrics.load_time_ms_mean,
                    scenario.cold_metrics.ttft_ms_mean,
                    scenario.score
                );
                println!("{}", line.dimmed());

                results.scenarios.push(scenario);
            }
        }
    }

    Ok(results)
}

fn run_scenario(
    cli_exe: &Path,
    model_path: &Path,
    model: &ModelConfig,
    workload: &WorkloadConfig,
    device: &str,
    cfg: &BenchConfig,
    _out_dir: &Path,
) -> anyhow::Result<ScenarioResult> {
    let base_args = LlamaArgs {
        exe: cli_exe,
        model: model_path,
        model_cfg: model,
        workload,
        cpu_threads: cfg.cpu_threads,
        gpu_layers: cfg.gpu_layers,
        device,
        seed: 42,
    };

    // 1. Cold start (1 run)
    print!("  cold start ... ");
    let cold_sample = llama::run_cold(&base_args)?;
    println!(
        "load={:.0}ms  ttft={:.0}ms  tps={:.1}",
        cold_sample.load_time_ms,
        cold_sample.ttft_ms,
        cold_sample.tokens_per_sec
    );
    let cold_samples = vec![cold_sample];
    let cold_metrics = AggregatedMetrics::from_samples(&cold_samples);

    // 2. Warm runs (cfg.warm_runs repetitions)
    let mut warm_samples = Vec::new();
    print!("  warm runs ({}x) ... ", cfg.warm_runs);
    let t = Instant::now();
    for i in 0..cfg.warm_runs {
        let sample = llama::run_cold(&LlamaArgs {
            seed: 42 + i as u64,
            ..base_args
        })?;
        warm_samples.push(sample);
    }
    let warm_metrics = AggregatedMetrics::from_samples(&warm_samples);
    println!(
        "done in {:.1}s – mean tps={:.1}",
        t.elapsed().as_secs_f64(),
        warm_metrics.tokens_per_sec_mean
    );

    // 3. Batch runs using llama-server (best-effort; skip gracefully)
    let batch_results = run_batch_scenarios(
        model_path,
        model,
        workload,
        device,
        cfg,
    );

    // 4. Compute score
    let batch_tps = batch_results
        .iter()
        .map(|b| b.throughput_tokens_per_sec)
        .fold(0.0_f64, f64::max);

    let breakdown = scoring::compute_score(&warm_metrics, &cold_metrics, batch_tps, &cfg.scoring);
    let score = breakdown.aggregate;

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
        score,
        score_breakdown: breakdown.into(),
    })
}

/// Attempt batch benchmarks via llama-server + HTTP. Gracefully skipped if server is unavailable.
fn run_batch_scenarios(
    model_path: &Path,
    model: &ModelConfig,
    workload: &WorkloadConfig,
    device: &str,
    cfg: &BenchConfig,
) -> Vec<BatchMetrics> {
    let server_exe = match assets::find_llama_server() {
        Ok(e) => e,
        Err(_) => {
            println!("  batch: llama-server not found – skipping batch benchmarks");
            return Vec::new();
        }
    };
    let server_exe = server_exe.as_path();

    let port = 18765u16;
    let mut results = Vec::new();

    for &batch_size in &cfg.batch_sizes {
        match run_single_batch(
            server_exe,
            model_path,
            model,
            workload,
            device,
            cfg,
            port,
            batch_size,
        ) {
            Ok(bm) => {
                println!(
                    "  batch={} → {:.1} req/s, {:.1} tok/s",
                    batch_size, bm.throughput_req_per_sec, bm.throughput_tokens_per_sec
                );
                results.push(bm);
            }
            Err(e) => {
                println!("  batch={}: {}", batch_size, e);
            }
        }
    }

    results
}

fn run_single_batch(
    server_exe: &Path,
    model_path: &Path,
    model: &ModelConfig,
    workload: &WorkloadConfig,
    device: &str,
    cfg: &BenchConfig,
    port: u16,
    batch_size: u32,
) -> anyhow::Result<BatchMetrics> {

    // Build server command
    let mut server_cmd = std::process::Command::new(server_exe);
    server_cmd.arg("--model").arg(model_path);
    server_cmd.arg("--port").arg(port.to_string());
    server_cmd.arg("--ctx-size").arg(model.context_length.to_string());
    server_cmd.arg("--parallel").arg(batch_size.to_string());
    server_cmd.arg("--threads").arg(num_cpus().to_string());
    match device {
        "gpu" => { server_cmd.arg("--n-gpu-layers").arg(cfg.gpu_layers.to_string()); }
        "cpu" => { server_cmd.arg("--n-gpu-layers").arg("0"); }
        _ => {}
    }

    // Suppress server output
    server_cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    let mut server = server_cmd.spawn()?;

    // Wait for server to be ready
    let ready = wait_for_server(port, std::time::Duration::from_secs(120));
    if !ready {
        let _ = server.kill();
        anyhow::bail!("llama-server did not start within 120 s");
    }

    let prompt = llama::build_prompt(workload);
    let url = format!("http://127.0.0.1:{}/completion", port);
    let max_tokens = workload.max_tokens;
    let body = serde_json::json!({
        "prompt": prompt,
        "n_predict": max_tokens,
        "seed": 42
    });

    let client = reqwest::blocking::Client::new();
    let wall_start = Instant::now();

    // Send batch_size concurrent requests using threads
    let handles: Vec<_> = (0..batch_size)
        .map(|_| {
            let client = client.clone();
            let url = url.clone();
            let body = body.clone();
            std::thread::spawn(move || -> anyhow::Result<(f64, u64)> {
                let t = Instant::now();
                let resp = client
                    .post(&url)
                    .json(&body)
                    .timeout(std::time::Duration::from_secs(600))
                    .send()?;
                let elapsed = t.elapsed().as_millis() as f64;
                if !resp.status().is_success() {
                    anyhow::bail!("server HTTP {}", resp.status());
                }
                let json: serde_json::Value = resp.json()?;
                let tokens = json["tokens_predicted"]
                    .as_u64()
                    .unwrap_or(max_tokens as u64);
                Ok((elapsed, tokens))
            })
        })
        .collect();

    let mut latencies = Vec::new();
    let mut total_tokens = 0u64;
    let mut successes = 0u32;

    for h in handles {
        match h.join() {
            Ok(Ok((lat, tok))) => {
                latencies.push(lat);
                total_tokens += tok;
                successes += 1;
            }
            _ => {}
        }
    }

    let total_wall = wall_start.elapsed().as_millis() as f64;

    let _ = server.kill();

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&latencies, 50.0);
    let p95 = percentile(&latencies, 95.0);

    Ok(BatchMetrics {
        batch_size,
        total_requests: batch_size,
        successful_requests: successes,
        total_wall_ms: total_wall,
        throughput_req_per_sec: successes as f64 / (total_wall / 1000.0).max(0.001),
        throughput_tokens_per_sec: total_tokens as f64 / (total_wall / 1000.0).max(0.001),
        latency_p50_ms: p50,
        latency_p95_ms: p95,
        total_tokens_generated: total_tokens,
    })
}

fn wait_for_server(port: u16, timeout: std::time::Duration) -> bool {
    use std::net::TcpStream;
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect(format!("127.0.0.1:{}", port)).is_ok() {
            // Give server a moment to register routes after TCP is up
            std::thread::sleep(std::time::Duration::from_millis(500));
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    false
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}
