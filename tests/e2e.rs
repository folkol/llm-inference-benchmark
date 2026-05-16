/// End-to-end tests that spawn a real llama-cli process against a real model.
///
/// These tests REQUIRE:
///   1. llama-cli (or llama-cli.exe) on PATH or in the llmb cache bin directory
///   2. The 0.5B Qwen model in the llmb model cache, OR set LLMB_TEST_MODEL
///      to an absolute path to any GGUF file.
///
/// To run:
///   cargo test --test e2e
///
/// To use a specific model:
///   LLMB_TEST_MODEL=/path/to/model.gguf cargo test --test e2e

use llmb::{
    assets,
    config::{ModelConfig, WorkloadConfig},
    llama::{LlamaArgs, run_cold},
    metrics::AggregatedMetrics,
};
use std::path::PathBuf;

// ── helpers ──────────────────────────────────────────────────────────────────

fn llama_cli() -> PathBuf {
    assets::find_llama_cli()
        .expect("llama-cli not found. Install llama.cpp and add it to PATH.\nDownload: https://github.com/ggerganov/llama.cpp/releases")
}

fn test_model() -> PathBuf {
    if let Ok(p) = std::env::var("LLMB_TEST_MODEL") {
        let path = PathBuf::from(&p);
        assert!(
            path.exists(),
            "LLMB_TEST_MODEL={} does not exist",
            p
        );
        return path;
    }

    let path = assets::cache_dir()
        .expect("cannot determine cache directory")
        .join("qwen2.5-0.5b-instruct-q4_k_m.gguf");

    assert!(
        path.exists(),
        "test model not found at {}.\nRun `llmb models fetch` or set LLMB_TEST_MODEL=/path/to/model.gguf",
        path.display()
    );

    path
}

fn tiny_model_cfg() -> ModelConfig {
    ModelConfig {
        name: "e2e-test-model".to_string(),
        filename: "test.gguf".to_string(),
        url: String::new(),
        sha256: None,
        params: Some("0.5B".to_string()),
        quantization: Some("Q4_K_M".to_string()),
        context_length: 512,
    }
}

fn tiny_workload() -> WorkloadConfig {
    WorkloadConfig {
        id: "e2e-minimal".to_string(),
        label: "E2E Minimal".to_string(),
        system: Some("You are a concise assistant.".to_string()),
        prompt: "Reply with exactly one word: hello".to_string(),
        max_tokens: 16,
    }
}

fn cpu_args<'a>(
    cli: &'a std::path::Path,
    model_path: &'a std::path::Path,
    model_cfg: &'a ModelConfig,
    workload: &'a WorkloadConfig,
    seed: u64,
) -> LlamaArgs<'a> {
    LlamaArgs {
        exe: cli,
        model: model_path,
        model_cfg,
        workload,
        cpu_threads: 2,
        gpu_layers: 0,
        mixed_gpu_layers: 16,
        device: "cpu",
        seed,
        timeout_secs: 180,
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

/// Smoke test: llama-cli runs, exits 0, and produces a parseable timing sample.
#[test]
fn test_e2e_cold_inference_succeeds() {
    let cli = llama_cli();
    let model_path = test_model();
    let model_cfg = tiny_model_cfg();
    let workload = tiny_workload();

    let sample = run_cold(&cpu_args(&cli, &model_path, &model_cfg, &workload, 42), "e2e")
        .expect("run_cold must not return an error");

    assert!(
        sample.success,
        "llama-cli exited with an error; check that the model file is a valid GGUF"
    );
    assert!(sample.wall_time_ms > 0.0, "wall time must be positive");
    assert!(sample.load_time_ms >= 0.0, "load time must be non-negative");
}

/// Determinism: the same seed produces the same token count across two runs.
#[test]
fn test_e2e_same_seed_same_token_count() {
    let cli = llama_cli();
    let model_path = test_model();
    let model_cfg = tiny_model_cfg();
    let workload = tiny_workload();

    let a = run_cold(&cpu_args(&cli, &model_path, &model_cfg, &workload, 42), "e2e-a")
        .expect("first run failed");
    let b = run_cold(&cpu_args(&cli, &model_path, &model_cfg, &workload, 42), "e2e-b")
        .expect("second run failed");

    assert!(a.success && b.success, "both runs must succeed");
    assert_eq!(
        a.gen_tokens, b.gen_tokens,
        "same seed must produce same token count ({} vs {})",
        a.gen_tokens, b.gen_tokens
    );
}

/// Different seeds both complete without error.
#[test]
fn test_e2e_different_seeds_both_complete() {
    let cli = llama_cli();
    let model_path = test_model();
    let model_cfg = tiny_model_cfg();
    let workload = tiny_workload();

    let a = run_cold(&cpu_args(&cli, &model_path, &model_cfg, &workload, 42), "e2e-42")
        .expect("seed 42 run failed");
    let b = run_cold(&cpu_args(&cli, &model_path, &model_cfg, &workload, 99), "e2e-99")
        .expect("seed 99 run failed");

    assert!(a.success, "seed 42 run must succeed");
    assert!(b.success, "seed 99 run must succeed");
}

/// Metrics aggregation over real samples: p50 ≤ p95, mean tok/s > 0.
#[test]
fn test_e2e_aggregated_metrics_are_sane() {
    let cli = llama_cli();
    let model_path = test_model();
    let model_cfg = tiny_model_cfg();
    let workload = tiny_workload();

    let samples: Vec<_> = (0..3u64)
        .map(|i| {
            run_cold(
                &cpu_args(&cli, &model_path, &model_cfg, &workload, 42 + i),
                &format!("e2e-{i}"),
            )
                .expect("run failed")
        })
        .collect();

    let agg = AggregatedMetrics::from_samples(&samples);

    assert_eq!(agg.n, 3);
    assert!(agg.success_rate > 0.99, "all 3 runs should succeed");
    assert!(agg.tokens_per_sec_mean > 0.0, "mean tok/s must be positive");
    assert!(
        agg.tokens_per_sec_p50 <= agg.tokens_per_sec_p95 * 1.01,
        "p50 must not exceed p95 (p50={}, p95={})",
        agg.tokens_per_sec_p50,
        agg.tokens_per_sec_p95
    );

    println!(
        "tps_mean={:.1}  tps_p50={:.1}  tps_p95={:.1}  load={:.0}ms  ttft={:.0}ms",
        agg.tokens_per_sec_mean,
        agg.tokens_per_sec_p50,
        agg.tokens_per_sec_p95,
        agg.load_time_ms_mean,
        agg.ttft_ms_mean,
    );
}

/// tok/s must be in a plausible range — catches silent parser breakage if
/// llama.cpp changes its log format.
#[test]
fn test_e2e_tokens_per_sec_plausible() {
    let cli = llama_cli();
    let model_path = test_model();
    let model_cfg = tiny_model_cfg();
    let workload = tiny_workload();

    let sample = run_cold(&cpu_args(&cli, &model_path, &model_cfg, &workload, 42), "e2e-tps")
        .expect("run failed");

    assert!(sample.success, "run must succeed");
    assert!(
        sample.tokens_per_sec >= 0.1,
        "tok/s suspiciously low ({}) — timing parser may be broken",
        sample.tokens_per_sec
    );
    assert!(
        sample.tokens_per_sec <= 10_000.0,
        "tok/s suspiciously high ({}) — timing parser may be broken",
        sample.tokens_per_sec
    );
}
