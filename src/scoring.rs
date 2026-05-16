use crate::config::ScoringConfig;
use crate::metrics::AggregatedMetrics;

/// A single normalized score in [0, 100].
#[derive(Debug, Clone)]
pub struct ScoreBreakdown {
    pub tokens_per_sec_score: f64,
    pub load_time_score: f64,
    pub ttft_score: f64,
    pub batch_throughput_score: f64,
    pub aggregate: f64,
}

/// Baseline references for normalization.
/// These are deliberately conservative so modest hardware still scores >0.
/// Values are tunable; they represent a "baseline pass" level.
pub struct Baselines {
    /// Tokens/sec considered "reference" performance → 100 points
    pub tokens_per_sec_ref: f64,
    /// Load time (ms) considered good → 100 points (lower is better)
    pub load_time_ms_ref: f64,
    /// TTFT (ms) considered good → 100 points
    pub ttft_ms_ref: f64,
    /// Batch tokens/sec considered good
    pub batch_tps_ref: f64,
}

impl Default for Baselines {
    fn default() -> Self {
        Self {
            tokens_per_sec_ref: 30.0,  // 30 t/s → score 100
            load_time_ms_ref: 3000.0,  // 3 s load → score 100
            ttft_ms_ref: 4000.0,       // 4 s TTFT → score 100
            batch_tps_ref: 100.0,      // 100 batch t/s → score 100
        }
    }
}

pub fn compute_score(
    warm: &AggregatedMetrics,
    cold: &AggregatedMetrics,
    batch_tps: f64,
    cfg: &ScoringConfig,
) -> ScoreBreakdown {
    let b = Baselines::default();

    // All scores are clamped to [0, 100]
    // Higher is better in all cases; for "lower is better" metrics we invert.

    let tps_score = clamp100(warm.tokens_per_sec_mean / b.tokens_per_sec_ref * 100.0);

    // Load time: score 100 if ≤ ref, linearly worse above
    let load_score = clamp100(b.load_time_ms_ref / cold.load_time_ms_mean.max(1.0) * 100.0);

    let ttft_score = clamp100(b.ttft_ms_ref / cold.ttft_ms_mean.max(1.0) * 100.0);

    let batch_score = clamp100(batch_tps / b.batch_tps_ref * 100.0);

    let w = cfg;
    let aggregate = tps_score * w.weight_tokens_per_sec
        + load_score * w.weight_load_time
        + ttft_score * w.weight_ttft
        + batch_score * w.weight_batch_throughput;

    ScoreBreakdown {
        tokens_per_sec_score: tps_score,
        load_time_score: load_score,
        ttft_score,
        batch_throughput_score: batch_score,
        aggregate,
    }
}

fn clamp100(v: f64) -> f64 {
    v.clamp(0.0, 100.0)
}
