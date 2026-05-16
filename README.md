# llmb – Cross-Platform LLM Inference Benchmark

A single compiled CLI that measures LLM inference performance across machines and produces comparable reports with scores and graphs.

## What it measures

| Dimension | Details |
|-----------|---------|
| **Models** | Any GGUF model; defaults include Qwen2.5 (0.5B–3B) and Llama-3.2 (1B/3B) |
| **Weight sizes** | Configurable; start with Q4_K_M quantized models that run on CPU |
| **Workloads** | Summarization, code generation, generic assistant (fixed prompts for reproducibility) |
| **Devices** | `cpu`, `gpu`, `auto` via llama.cpp flags |
| **Run modes** | Cold start (fresh process per call), warm (repeated calls), batch (concurrent via llama-server) |

## Metrics collected

- **Tokens/second** (generation phase, mean / p50 / p95)
- **Load time** (cold start: process launch → model ready)
- **Time to first token** (TTFT, cold)
- **Batch throughput** (req/s and tok/s at batch sizes 1, 4, 8)
- **Memory** (reported where available from llama.cpp output)

## Scoring

Each scenario receives a score in **[0–100]** built from four weighted components:

| Component | Default weight |
|-----------|---------------|
| Tokens/sec (warm) | 40% |
| Cold-start load time | 20% |
| TTFT cold | 20% |
| Batch throughput | 20% |

Both the raw metrics and score breakdown are included in every report so you can reweight offline.

## Prerequisites

1. **llama.cpp** — `llama-cli` must be on `PATH` (and `llama-server` for batch benchmarks).
   - Windows: download prebuilt ZIP from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) and add the folder to `PATH`.
   - macOS/Linux: `brew install llama.cpp` or build from source.
2. **Disk space** — models range from ~400 MB (0.5B Q4) to ~5 GB (8B Q4). The default set is ~7 GB total.
3. **RAM** — 0.5–1.5B models need ~1 GB; 3B needs ~2.5 GB; 7–8B needs ~5–6 GB.

## Quick start

```sh
# 1. Download or build the llmb binary (or cargo build --release)
cargo build --release
cp target/release/llmb ~/.local/bin/   # Linux/macOS
# On Windows add target\release to PATH

# 2. Create a config in your working directory
llmb init

# 3. Download models (large download, ~7 GB for defaults)
llmb models fetch

# 4. Run the benchmark (CPU only by default)
llmb bench

# 5. Open the report
llmb report
```

The default output folder is `./reports/<os-arch>__CPU-…__GPU-…__<run-id>/` (detected hardware plus a UTC `run_id`). Use `--out DIR` for a stable path (`llmb bench --out reports/latest` restores the previous layout).

Without arguments, `llmb report` opens the newest `report.html` anywhere under `./reports/` (recursive, so it includes `./reports/samples/...`).

### Multiple machines (Windows, Linux, other PCs)

Every run defaults to `./reports/<os-arch>__CPU-…__GPU-…__<run-id>/`, so hosts usually get **distinct directory names**: e.g. `linux-x86_64__…` vs `windows-x86_64__…`, and different GPUs or CPUs show up in the slug. Repeated runs on the same box differ by `run-id` (UTC timestamp).

Runs under `./reports/` are **local-only** unless you publish them on purpose. To keep curated reports in git (benchmarks from this machine under Linux, or tarballs copied from another computer), drop each finished run folder under **`reports/samples/`** — one subdirectory per run. That tree is tracked; anything else directly under `./reports/` stays ignored.

## GPU benchmarking

Edit `bench.toml` and change `devices`:

```toml
devices = ["cpu", "gpu"]
```

For NVIDIA (CUDA), use a CUDA-enabled llama.cpp binary. For Apple Silicon, use the Metal-enabled build. The tool passes `--n-gpu-layers -1` for GPU (all layers) and `--n-gpu-layers 0` for CPU.

## Commands

```
llmb init                          # write bench.toml to current directory
llmb models list                   # show models and cache status
llmb models fetch                  # download missing models
llmb bench                         # run with bench.toml defaults
llmb bench --devices cpu,gpu       # override devices
llmb bench --models Qwen2.5-0.5B   # run only matching models
llmb bench --runs 5                # 5 warm repetitions instead of 3
llmb bench --out reports/my-run    # fixed directory (overwrite same path each run)
llmb report                        # newest report under ./reports/
llmb report reports/my-run         # open one run explicitly
```

## Output files

Each run writes three files to the output directory:

| File | Contents |
|------|---------|
| `results.json` | Canonical output: hardware info, software versions, all raw samples, computed scores |
| `results.csv` | Flat rows — easy to import into Excel / Google Sheets for cross-machine comparison |
| `report.html` | Self-contained HTML with Chart.js graphs (score, tok/s, load time, TTFT) |

## Comparing machines

Run `llmb bench` on each machine and collect the `results.csv` files. Because the workloads are fixed and the scoring formula is deterministic, the aggregate scores are directly comparable. You can also open multiple `results.json` files in any tool that reads JSON.

## Config reference

```toml
# bench.toml
devices      = ["cpu"]         # "cpu", "gpu", or "auto"
warm_runs    = 3               # warm repetitions per scenario
batch_sizes  = [1, 4, 8]      # batch concurrency levels
max_tokens   = 256             # default generation budget
cpu_threads  = 0               # 0 = auto-detect
gpu_layers   = -1              # -1 = offload all layers

[scoring]
weight_tokens_per_sec   = 0.40
weight_load_time        = 0.20
weight_ttft             = 0.20
weight_batch_throughput = 0.20

[[models]]
name         = "Qwen2.5-0.5B-Q4_K_M"
filename     = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
url          = "https://huggingface.co/..."
params       = "0.5B"
quantization = "Q4_K_M"
context_length = 2048
sha256       = "optional hex digest for verification"
```

## Adding custom models

Any GGUF model from Hugging Face (or a local path as a `file://` URL) can be added to the `[[models]]` section. The benchmark will skip models whose files are not present in the cache and remind you to run `llmb models fetch`.

## Model cache location

| OS | Cache path |
|----|-----------|
| Windows | `%LOCALAPPDATA%\llm-benchmark` |
| macOS | `~/Library/Caches/llm-benchmark` |
| Linux | `~/.cache/llm-benchmark` |

## Building from source on each platform

```sh
# All platforms (requires Rust 1.80+)
git clone <repo>
cd llm-inference-benchmark
cargo build --release

# The binary is at:
#   target/release/llmb          (Linux/macOS)
#   target\release\llmb.exe      (Windows)
```

On Windows without Visual Studio, use the GNU toolchain:

```powershell
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
# Add MinGW64 bin directory to PATH, then:
cargo build --release
```

## Reproducibility notes

- All workload prompts are hardcoded in the binary (not read from disk at runtime) so they cannot drift between runs.
- The random seed is fixed to 42 for all llama.cpp calls.
- Hardware information (CPU, RAM, GPU, OS version) is recorded in every `results.json`.
- llama.cpp version is included when detectable.
