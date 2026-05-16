use serde::{Deserialize, Serialize};

/// One raw timing sample from a single llama.cpp invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSample {
    /// Total wall-clock time for the entire process (ms)
    pub wall_time_ms: f64,
    /// Time from process start to model fully loaded (ms)
    pub load_time_ms: f64,
    /// Number of prompt tokens processed
    pub prompt_tokens: u64,
    /// Time to evaluate (prefill) the prompt (ms)
    pub prompt_eval_ms: f64,
    /// Number of generated tokens
    pub gen_tokens: u64,
    /// Time to generate all tokens after prefill (ms)
    pub gen_eval_ms: f64,
    /// Tokens per second during generation phase
    pub tokens_per_sec: f64,
    /// Time to first token = load + prompt eval (ms)
    pub ttft_ms: f64,
    /// Whether the llama-cli process exited with code 0
    pub success: bool,
}

/// Aggregated statistics across multiple RawSamples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub n: usize,
    pub tokens_per_sec_mean: f64,
    pub tokens_per_sec_p50: f64,
    pub tokens_per_sec_p95: f64,
    pub load_time_ms_mean: f64,
    pub ttft_ms_mean: f64,
    pub ttft_ms_p95: f64,
    pub wall_time_ms_mean: f64,
    pub gen_tokens_mean: f64,
    pub success_rate: f64,
}

impl AggregatedMetrics {
    pub fn from_samples(samples: &[RawSample]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self::zeros();
        }

        let successful: Vec<&RawSample> = samples.iter().filter(|s| s.success).collect();
        let success_rate = successful.len() as f64 / n as f64;

        let mut tps: Vec<f64> = successful.iter().map(|s| s.tokens_per_sec).collect();
        let mut ttft: Vec<f64> = successful.iter().map(|s| s.ttft_ms).collect();

        tps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ttft.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
        let percentile = |v: &[f64], p: f64| {
            if v.is_empty() {
                return 0.0;
            }
            let idx = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
            v[idx.min(v.len() - 1)]
        };

        Self {
            n,
            tokens_per_sec_mean: mean(&tps),
            tokens_per_sec_p50: percentile(&tps, 50.0),
            tokens_per_sec_p95: percentile(&tps, 95.0),
            load_time_ms_mean: mean(
                &successful.iter().map(|s| s.load_time_ms).collect::<Vec<_>>(),
            ),
            ttft_ms_mean: mean(&ttft),
            ttft_ms_p95: percentile(&ttft, 95.0),
            wall_time_ms_mean: mean(
                &successful.iter().map(|s| s.wall_time_ms).collect::<Vec<_>>(),
            ),
            gen_tokens_mean: mean(
                &successful.iter().map(|s| s.gen_tokens as f64).collect::<Vec<_>>(),
            ),
            success_rate,
        }
    }

    fn zeros() -> Self {
        Self {
            n: 0,
            tokens_per_sec_mean: 0.0,
            tokens_per_sec_p50: 0.0,
            tokens_per_sec_p95: 0.0,
            load_time_ms_mean: 0.0,
            ttft_ms_mean: 0.0,
            ttft_ms_p95: 0.0,
            wall_time_ms_mean: 0.0,
            gen_tokens_mean: 0.0,
            success_rate: 0.0,
        }
    }
}

/// Aggregated metrics for a batch run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMetrics {
    pub batch_size: u32,
    pub total_requests: u32,
    pub successful_requests: u32,
    pub total_wall_ms: f64,
    pub throughput_req_per_sec: f64,
    pub throughput_tokens_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub total_tokens_generated: u64,
}
