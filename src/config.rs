use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    /// Devices to benchmark: "cpu", "gpu", "auto"
    #[serde(default = "default_devices")]
    pub devices: Vec<String>,

    /// Number of warm repetitions per scenario (CLI may override)
    #[serde(default = "default_warm_runs")]
    pub warm_runs: u32,

    /// Batch sizes to test in batch mode
    #[serde(default = "default_batch_sizes")]
    pub batch_sizes: Vec<u32>,

    /// Maximum tokens to generate per completion
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Number of CPU threads to use (0 = auto)
    #[serde(default)]
    pub cpu_threads: u32,

    /// GPU layers to offload (-1 = all)
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,

    /// Models to benchmark
    pub models: Vec<ModelConfig>,

    /// Workloads to run
    #[serde(default)]
    pub workloads: Vec<WorkloadConfig>,

    /// Scoring weights (values sum to 1.0)
    #[serde(default)]
    pub scoring: ScoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Short human-readable name, e.g. "Qwen2.5-0.5B-Q4"
    pub name: String,

    /// GGUF filename on disk (in the cache)
    pub filename: String,

    /// Download URL (Hugging Face or direct)
    pub url: String,

    /// Expected SHA-256 hex digest (optional)
    pub sha256: Option<String>,

    /// Parameter class for display, e.g. "0.5B"
    pub params: Option<String>,

    /// Quantization label, e.g. "Q4_K_M"
    pub quantization: Option<String>,

    /// Context length to use in prompts
    #[serde(default = "default_context")]
    pub context_length: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    /// Identifier used in output
    pub id: String,
    /// Human-readable label
    pub label: String,
    /// System prompt (if any)
    pub system: Option<String>,
    /// User prompt
    pub prompt: String,
    /// Target generation tokens (hint for --predict flag)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoringConfig {
    /// Weight for tokens/sec metric
    #[serde(default = "default_weight_tps")]
    pub weight_tokens_per_sec: f64,
    /// Weight for cold-start load time (lower is better)
    #[serde(default = "default_weight_load")]
    pub weight_load_time: f64,
    /// Weight for TTFT
    #[serde(default = "default_weight_ttft")]
    pub weight_ttft: f64,
    /// Weight for batch throughput
    #[serde(default = "default_weight_batch")]
    pub weight_batch_throughput: f64,
}

fn default_devices() -> Vec<String> {
    vec!["cpu".to_string()]
}
fn default_warm_runs() -> u32 {
    3
}
fn default_batch_sizes() -> Vec<u32> {
    vec![1, 4, 8]
}
fn default_max_tokens() -> u32 {
    256
}
fn default_gpu_layers() -> i32 {
    -1
}
fn default_context() -> u32 {
    2048
}
fn default_weight_tps() -> f64 {
    0.4
}
fn default_weight_load() -> f64 {
    0.2
}
fn default_weight_ttft() -> f64 {
    0.2
}
fn default_weight_batch() -> f64 {
    0.2
}

pub fn load(path: &Path) -> anyhow::Result<BenchConfig> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot read config {}: {}", path.display(), e))?;
    let mut cfg: BenchConfig = toml::from_str(&text)
        .map_err(|e| anyhow::anyhow!("Config parse error in {}: {}", path.display(), e))?;
    if cfg.workloads.is_empty() {
        cfg.workloads = crate::workloads::default_workloads();
    }
    Ok(cfg)
}
