use llmb::{
    config::{BenchConfig, WorkloadConfig},
    metrics::{AggregatedMetrics, RawSample},
    workloads::default_workloads,
};

// ─── Config parsing ───────────────────────────────────────────────────────────

#[test]
fn test_default_config_parses() {
    let toml = include_str!("../configs/default.toml");
    let cfg: BenchConfig = toml::from_str(toml).expect("default config must parse");
    assert!(!cfg.models.is_empty(), "default config must have models");
    assert!(cfg.warm_runs > 0);
}

#[test]
fn test_minimal_config_parses() {
    let toml = r#"
[[models]]
name = "test"
filename = "test.gguf"
url = "https://example.com/test.gguf"
"#;
    let cfg: BenchConfig = toml::from_str(toml).expect("minimal config must parse");
    assert_eq!(cfg.models.len(), 1);
    assert_eq!(cfg.models[0].name, "test");
}

// ─── Workloads ────────────────────────────────────────────────────────────────

#[test]
fn test_default_workloads_are_nonempty() {
    let wl = default_workloads();
    assert!(!wl.is_empty());
    for w in &wl {
        assert!(!w.id.is_empty());
        assert!(!w.prompt.is_empty());
        assert!(w.max_tokens > 0);
    }
}

#[test]
fn test_workload_ids_are_unique() {
    let wl = default_workloads();
    let mut ids: Vec<&str> = wl.iter().map(|w| w.id.as_str()).collect();
    ids.dedup();
    assert_eq!(ids.len(), wl.len(), "workload IDs must be unique");
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

fn make_sample(tps: f64, load_ms: f64, ttft_ms: f64) -> RawSample {
    RawSample {
        completion: None,
        wall_time_ms: ttft_ms + 100.0,
        load_time_ms: load_ms,
        prompt_tokens: 128,
        prompt_eval_ms: 200.0,
        gen_tokens: 100,
        gen_eval_ms: 100.0 / tps * 1000.0,
        tokens_per_sec: tps,
        ttft_ms,
        success: true,
    }
}

#[test]
fn test_aggregated_metrics_from_samples() {
    let samples = vec![
        make_sample(20.0, 1000.0, 1200.0),
        make_sample(40.0, 900.0, 1100.0),
        make_sample(30.0, 950.0, 1050.0),
    ];
    let agg = AggregatedMetrics::from_samples(&samples);
    assert_eq!(agg.n, 3);
    assert!(agg.success_rate > 0.99);
    assert!((agg.tokens_per_sec_mean - 30.0).abs() < 1.0);
}

#[test]
fn test_aggregated_metrics_empty() {
    let agg = AggregatedMetrics::from_samples(&[]);
    assert_eq!(agg.n, 0);
    assert_eq!(agg.tokens_per_sec_mean, 0.0);
}

#[test]
fn test_aggregated_metrics_with_failures() {
    let mut samples = vec![make_sample(50.0, 500.0, 600.0)];
    samples.push(RawSample {
        success: false,
        ..make_sample(0.0, 0.0, 0.0)
    });
    let agg = AggregatedMetrics::from_samples(&samples);
    assert!((agg.success_rate - 0.5).abs() < 1e-6);
    assert!((agg.tokens_per_sec_mean - 50.0).abs() < 1.0);
}

// ─── llama timing parser ──────────────────────────────────────────────────────

#[test]
fn test_llama_prompt_building() {
    // build_prompt returns only the user content; system prompt is passed
    // separately to llama-cli via --system-prompt so the model's embedded
    // chat template (tokenizer.chat_template in the GGUF) is applied.
    let workload = WorkloadConfig {
        id: "test".to_string(),
        label: "Test".to_string(),
        system: Some("You are a robot.".to_string()),
        prompt: "Hello".to_string(),
        max_tokens: 100,
    };
    let prompt = llmb::llama::build_prompt(&workload);
    // The user prompt is returned verbatim; system is NOT embedded here.
    assert_eq!(prompt, "Hello");
    // The system prompt lives on the workload struct, accessible separately.
    assert_eq!(workload.system.as_deref(), Some("You are a robot."));
}

#[test]
fn test_llama_prompt_no_system() {
    let workload = WorkloadConfig {
        id: "test".to_string(),
        label: "Test".to_string(),
        system: None,
        prompt: "Just a prompt".to_string(),
        max_tokens: 100,
    };
    let prompt = llmb::llama::build_prompt(&workload);
    assert_eq!(prompt, "Just a prompt");
}
