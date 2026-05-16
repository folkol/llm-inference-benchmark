use std::io::Write as _;
use std::path::Path;
use std::sync::mpsc;
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
    /// Layers to offload for device="gpu" (-1 = all)
    pub gpu_layers: i32,
    /// Layers to offload for device="mixed" (explicit partial count)
    pub mixed_gpu_layers: u32,
    pub device: &'a str,
    pub seed: u64,
    /// Kill llama-cli and record a failed sample after this many seconds.
    /// Prevents indefinite hangs when a model is too large for available RAM.
    pub timeout_secs: u64,
}

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const LINE_WIDTH: usize = 80;

/// Run a single inference and return the raw sample.
///
/// `label` is printed as the line prefix, e.g. "cold start" or "warm 2/3".
/// This function owns the terminal line: it shows a live spinner while the
/// child is running, then overwrites the line with the final metrics (or a
/// timeout/error message) before returning.
///
/// Stdout and stderr are drained on background threads to prevent pipe-buffer
/// deadlock (llama-cli streams generated tokens to stdout continuously).
pub fn run_cold(args: &LlamaArgs<'_>, label: &str) -> anyhow::Result<RawSample> {
    use std::process::Stdio;

    enum StreamEvent {
        Stdout(String),
        Stderr(String),
    }

    let prompt = build_prompt(args.workload);
    let mut cmd = build_cli_command(args, &prompt);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    // On Windows put the child in its own process group so that any
    // CTRL_C_EVENT it generates does not propagate to llmb.exe.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }

    let start = Instant::now();
    let mut child = cmd.spawn()?;

    // Assign the child to a Windows Job Object so it is automatically
    // terminated if llmb.exe exits for any reason (crash, Ctrl+C, task-kill).
    // Without this, CREATE_NEW_PROCESS_GROUP children become orphans that keep
    // consuming CPU/RAM after the parent is gone.
    #[cfg(windows)]
    assign_to_job(&child);

    let child_stdout = child.stdout.take().expect("stdout piped");
    let child_stderr = child.stderr.take().expect("stderr piped");

    let (event_tx, event_rx) = mpsc::channel::<StreamEvent>();

    let stdout_tx = event_tx.clone();
    let stdout_thread = std::thread::spawn(move || {
        use std::io::Read;
        let mut reader = std::io::BufReader::new(child_stdout);
        let mut buf = [0_u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
                    let _ = stdout_tx.send(StreamEvent::Stdout(chunk));
                }
            }
        }
    });
    let stderr_tx = event_tx.clone();
    let stderr_thread = std::thread::spawn(move || {
        use std::io::Read;
        let mut reader = std::io::BufReader::new(child_stderr);
        let mut buf = [0_u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
                    let _ = stderr_tx.send(StreamEvent::Stderr(chunk));
                }
            }
        }
    });
    drop(event_tx);

    let timeout = Duration::from_secs(args.timeout_secs);
    let max_tokens = args.workload.max_tokens;
    let mut spin_idx = 0usize;
    let mut last_spin = Instant::now();
    let mut stdout = String::new();
    let mut stderr = String::new();
    let mut phase = "starting";
    let mut first_stdout_ms = None::<f64>;

    // exit_status is Some(success) once the process exits naturally.
    let mut exit_ok: Option<bool> = None;

    let timed_out = loop {
        match child.try_wait()? {
            Some(status) => {
                exit_ok = Some(status.success());
                break false;
            }
            None if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                break true;
            }
            None => {
                while let Ok(event) = event_rx.try_recv() {
                    match event {
                        StreamEvent::Stdout(chunk) => {
                            if first_stdout_ms.is_none() && !chunk.trim().is_empty() {
                                first_stdout_ms = Some(start.elapsed().as_millis() as f64);
                                phase = "generating";
                            }
                            stdout.push_str(&chunk);
                        }
                        StreamEvent::Stderr(chunk) => {
                            if let Some(p) = describe_llama_phase(&chunk) {
                                phase = p;
                            }
                            stderr.push_str(&chunk);
                        }
                    }
                }
                if last_spin.elapsed() >= Duration::from_millis(250) {
                    let elapsed = start.elapsed().as_secs();
                    let pct = (elapsed * 100 / timeout.as_secs().max(1)).min(99);
                    let bar_done = (pct * 20 / 100) as usize;
                    let bar: String = format!(
                        "[{}{}]",
                        "█".repeat(bar_done),
                        "░".repeat(20 - bar_done)
                    );
                    print!(
                        "\r  {} {} {} {}s/{}s  {} ({} tok max)  ",
                        label,
                        SPINNER[spin_idx % SPINNER.len()],
                        bar,
                        elapsed,
                        timeout.as_secs(),
                        phase,
                        max_tokens,
                    );
                    std::io::stdout().flush().ok();
                    spin_idx += 1;
                    last_spin = Instant::now();
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    };

    let wall_time = start.elapsed();

    // Collect any remaining piped output after process exit.
    let _ = stdout_thread.join();
    let _ = stderr_thread.join();
    while let Ok(event) = event_rx.try_recv() {
        match event {
            StreamEvent::Stdout(chunk) => stdout.push_str(&chunk),
            StreamEvent::Stderr(chunk) => stderr.push_str(&chunk),
        }
    }

    if timed_out {
        println!(
            "\r  {} [TIMEOUT {}s while {} — too slow or model too large for RAM]{}",
            label,
            args.timeout_secs,
            phase,
            " ".repeat(LINE_WIDTH),
        );
        return Ok(RawSample {
            completion: None,
            wall_time_ms: wall_time.as_millis() as f64,
            load_time_ms: 0.0,
            prompt_tokens: 0,
            prompt_eval_ms: 0.0,
            gen_tokens: 0,
            gen_eval_ms: 0.0,
            tokens_per_sec: 0.0,
            ttft_ms: wall_time.as_millis() as f64,
            success: false,
        });
    }

    let succeeded = exit_ok.unwrap_or(false);
    if !succeeded {
        let tail: Vec<&str> = stderr
            .lines()
            .filter(|l| !l.trim().is_empty())
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        eprintln!("\r  {} [llama-cli error]", label);
        for l in &tail { eprintln!("    {}", l); }
    }

    let sample = parse_sample(&stdout, &stderr, wall_time, succeeded)?;

    // Overwrite spinner with final metrics on the same line.
    println!(
        "\r  {} load={:.0}ms  ttft={:.0}ms  first_out={:.0}ms  tps={:.1}{}",
        label,
        sample.load_time_ms,
        sample.ttft_ms,
        first_stdout_ms.unwrap_or(0.0),
        sample.tokens_per_sec,
        " ".repeat(LINE_WIDTH),
    );

    Ok(sample)
}

/// Returns the raw user prompt (no envelope).
/// System prompt is passed separately via --system-prompt so llama.cpp can
/// apply the chat template that is embedded in the GGUF model file itself.
pub fn build_prompt(workload: &WorkloadConfig) -> String {
    workload.prompt.clone()
}

fn build_cli_command(args: &LlamaArgs<'_>, prompt: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(args.exe);
    cmd.arg("--model").arg(args.model);

    // Pass system and user parts separately so llama.cpp applies the chat
    // template that is stored inside the GGUF file (tokenizer.chat_template).
    // This works correctly for Llama-3, Qwen, Mistral, Gemma, etc. without
    // any per-model formatting logic in our code.
    if let Some(sys) = &args.workload.system {
        cmd.arg("--system-prompt").arg(sys);
    }
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
            // CUDA_VISIBLE_DEVICES=-1 is the most reliable way to prevent the
            // CUDA runtime from initialising at all for CPU-only runs.  Without
            // this, llama-cli (CUDA build) probes all GPUs and JIT-compiles
            // kernels at startup even when --n-gpu-layers 0, causing a 60-120 s
            // penalty on Windows before any inference begins.  Setting this env
            // var hides all GPUs from the CUDA runtime before the process starts.
            cmd.env("CUDA_VISIBLE_DEVICES", "-1");
            cmd.arg("--device").arg("none");
            cmd.arg("--n-gpu-layers").arg("0");
        }
        "mixed" => {
            // Explicit partial offload: mixed_gpu_layers layers on GPU, rest on CPU.
            cmd.arg("--n-gpu-layers").arg(args.mixed_gpu_layers.to_string());
        }
        _ => {
            // "auto": let llama.cpp decide based on available VRAM
        }
    }

    // --simple-io: subprocess-friendly mode (no readline / ANSI / console writes).
    cmd.arg("--simple-io");
    // Newer llama.cpp builds auto-enable conversation mode for chat-template
    // models. Without these flags llama-cli answers the prompt, then waits at
    // an interactive prompt forever, so the benchmark hits its timeout even
    // though inference already finished.
    cmd.arg("--no-conversation");
    cmd.arg("--single-turn");
    cmd.arg("--seed").arg(args.seed.to_string());
    // We intentionally do NOT pass --no-mmap.
    // --no-mmap forces the entire model into anonymous RAM immediately, which
    // causes OOM on large models (e.g. 32B needs ~20 GB allocated at once even
    // on a 32 GB machine that already has 10 GB in use).  With mmap, Windows/
    // Linux can evict model pages under pressure.  The load-time measurement
    // still captures "how long the user waits from process start to first token"
    // — which is the realistic cold-start experience.

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
        completion: None,
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

fn describe_llama_phase(chunk: &str) -> Option<&'static str> {
    let lower = chunk.to_ascii_lowercase();
    if lower.contains("llama_model_loader")
        || lower.contains("loading model")
        || lower.contains("load_tensors")
    {
        Some("loading model")
    } else if lower.contains("kv cache")
        || lower.contains("llama_kv_cache")
        || lower.contains("context")
    {
        Some("creating context")
    } else if lower.contains("sampler")
        || lower.contains("generate")
    {
        Some("starting generation")
    } else if lower.contains("prompt eval") {
        Some("prompt eval")
    } else {
        None
    }
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

/// Add the child process to a Windows Job Object configured with
/// JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.  When llmb.exe exits for any reason
/// (normal exit, panic, Ctrl+C, task-kill), the OS closes the job handle and
/// automatically terminates every process in the job.  This prevents
/// llama-cli from becoming an orphan that keeps consuming CPU and RAM.
#[cfg(windows)]
fn assign_to_job(child: &std::process::Child) {
    use std::os::windows::io::AsRawHandle;

    // Declare only what we need from the Win32 API directly — no extra crate.
    type HANDLE = *mut std::ffi::c_void;
    type BOOL   = i32;

    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE:       u32 = 0x0000_2000;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION:     i32 = 9;

    #[repr(C)]
    struct BasicLimitInfo {
        per_process_user_time_limit: i64,
        per_job_user_time_limit:     i64,
        limit_flags:                 u32,
        minimum_working_set_size:    usize,
        maximum_working_set_size:    usize,
        active_process_limit:        u32,
        affinity:                    usize,
        priority_class:              u32,
        scheduling_class:            u32,
    }
    #[repr(C)]
    struct ExtendedLimitInfo {
        basic:                    BasicLimitInfo,
        io_info:                  [u64; 6],
        process_memory_limit:     usize,
        job_memory_limit:         usize,
        peak_process_memory_used: usize,
        peak_job_memory_used:     usize,
    }

    extern "system" {
        fn CreateJobObjectW(attrs: *const u8, name: *const u16) -> HANDLE;
        fn SetInformationJobObject(
            job:   HANDLE,
            class: i32,
            info:  *const std::ffi::c_void,
            len:   u32,
        ) -> BOOL;
        fn AssignProcessToJobObject(job: HANDLE, process: HANDLE) -> BOOL;
    }

    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() { return; }

        let mut info: ExtendedLimitInfo = std::mem::zeroed();
        info.basic.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        let ok = SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            &info as *const _ as *const _,
            std::mem::size_of::<ExtendedLimitInfo>() as u32,
        );
        if ok == 0 { return; }

        AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE);
        // Leak the handle intentionally: the job (and kill-on-close guarantee)
        // lives as long as llmb.exe does.  The OS reclaims it on process exit.
        let _ = std::mem::ManuallyDrop::new(job);
    }
}
