use std::path::Path;
use std::time::{Duration, Instant};

use crate::config::{ModelConfig, WorkloadConfig};
use crate::metrics::RawSample;

/// Arguments for a single llama-cli completion call.
pub struct LlamaArgs<'a> {
    pub exe: &'a Path,
    pub model: &'a Path,
    pub model_cfg: &'a ModelConfig,
    pub workload: &'a WorkloadConfig,
    pub cpu_threads: u32,
    pub gpu_layers: i32,
    pub device: &'a str,
    pub seed: u64,
}

/// Run a single cold-start inference and return the raw sample.
/// A cold start spawns a fresh llama-cli process for every call, measuring
/// full binary startup + model load + prefill + generation time.
pub fn run_cold(args: &LlamaArgs<'_>) -> anyhow::Result<RawSample> {
    let prompt = build_prompt(args.workload);
    let mut cmd = build_cli_command(args, &prompt);

    let start = Instant::now();
    let output = cmd.output()?;
    let wall_time = start.elapsed();

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    parse_sample(&stdout, &stderr, wall_time, output.status.success())
}

/// Build the prompt string from a workload definition.
pub fn build_prompt(workload: &WorkloadConfig) -> String {
    if let Some(sys) = &workload.system {
        format!(
            "<|system|>\n{sys}\n<|user|>\n{}\n<|assistant|>",
            workload.prompt
        )
    } else {
        workload.prompt.clone()
    }
}

fn build_cli_command(args: &LlamaArgs<'_>, prompt: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(args.exe);
    cmd.arg("--model").arg(args.model);
    cmd.arg("--prompt").arg(prompt);
    cmd.arg("--n-predict")
        .arg(args.workload.max_tokens.to_string());
    cmd.arg("--ctx-size")
        .arg(args.model_cfg.context_length.to_string());

    let threads = if args.cpu_threads == 0 {
        num_cpus()
    } else {
        args.cpu_threads
    };
    cmd.arg("--threads").arg(threads.to_string());

    match args.device {
        "gpu" => {
            cmd.arg("--n-gpu-layers").arg(args.gpu_layers.to_string());
        }
        "cpu" => {
            cmd.arg("--n-gpu-layers").arg("0");
        }
        _ => {
            // auto: let llama.cpp decide
        }
    }

    // Always emit timing statistics to stderr
    cmd.arg("--log-disable"); // cleaner stdout
    cmd.arg("--seed").arg(args.seed.to_string());
    cmd.arg("--no-mmap"); // cleaner cold-start measurement

    cmd
}

/// Parse llama.cpp stderr timing output into a RawSample.
/// llama.cpp prints lines like:
///   llama_print_timings:        load time =   1234.56 ms
///   llama_print_timings:  prompt eval time =    567.89 ms /   128 tokens
///   llama_print_timings:        eval time =   2345.67 ms /   200 runs  (   11.73 ms per token,    85.26 tokens per second)
fn parse_sample(
    _stdout: &str,
    stderr: &str,
    wall_time: Duration,
    success: bool,
) -> anyhow::Result<RawSample> {
    let mut load_time_ms = None::<f64>;
    let mut prompt_tokens = None::<u64>;
    let mut prompt_eval_ms = None::<f64>;
    let mut gen_tokens = None::<u64>;
    let mut gen_eval_ms = None::<f64>;
    let mut tokens_per_sec = None::<f64>;

    for line in stderr.lines() {
        let line = line.trim();

        if let Some(rest) = line.strip_prefix("llama_print_timings:") {
            let rest = rest.trim();

            if rest.starts_with("load time") {
                load_time_ms = parse_ms_value(rest);
            } else if rest.starts_with("prompt eval time") {
                prompt_eval_ms = parse_ms_value(rest);
                prompt_tokens = parse_token_count(rest);
            } else if rest.starts_with("eval time") {
                gen_eval_ms = parse_ms_value(rest);
                gen_tokens = parse_run_count(rest);
                tokens_per_sec = parse_tps(rest);
            }
        }
    }

    // Fallback: if llama.cpp didn't emit timing (older builds), estimate from wall time
    let load_ms = load_time_ms.unwrap_or(0.0);
    let gen_ms = gen_eval_ms.unwrap_or_else(|| wall_time.as_millis() as f64 - load_ms);
    let tps = tokens_per_sec.or_else(|| {
        if let (Some(t), ms) = (gen_tokens, gen_ms) {
            if ms > 0.0 {
                Some(t as f64 / (ms / 1000.0))
            } else {
                None
            }
        } else {
            None
        }
    });

    let ttft_ms = prompt_eval_ms.map(|p| load_ms + p);

    Ok(RawSample {
        wall_time_ms: wall_time.as_millis() as f64,
        load_time_ms: load_time_ms.unwrap_or(0.0),
        prompt_tokens: prompt_tokens.unwrap_or(0),
        prompt_eval_ms: prompt_eval_ms.unwrap_or(0.0),
        gen_tokens: gen_tokens.unwrap_or(0),
        gen_eval_ms: gen_ms,
        tokens_per_sec: tps.unwrap_or(0.0),
        ttft_ms: ttft_ms.unwrap_or(wall_time.as_millis() as f64),
        success,
    })
}

// ── tiny parsers for llama.cpp timing lines ───────────────────────────────────

fn parse_ms_value(s: &str) -> Option<f64> {
    // Look for pattern "= <number> ms"
    let eq = s.find('=')?;
    let after = &s[eq + 1..];
    let ms_pos = after.find(" ms")?;
    after[..ms_pos].trim().parse().ok()
}

fn parse_token_count(s: &str) -> Option<u64> {
    // "123 tokens"
    let pos = s.find(" tokens")?;
    let before = s[..pos].rsplit_once(' ')?.1;
    before.trim().parse().ok()
}

fn parse_run_count(s: &str) -> Option<u64> {
    // "200 runs"
    let pos = s.find(" runs")?;
    let before = s[..pos].rsplit_once(' ')?.1;
    before.trim().parse().ok()
}

fn parse_tps(s: &str) -> Option<f64> {
    // "85.26 tokens per second"
    let pos = s.find(" tokens per second")?;
    let before = s[..pos].rsplit_once(' ')?.1;
    before.trim().parse().ok()
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}
