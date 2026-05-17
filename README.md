# llmb – Cross-Platform LLM Inference Benchmark

A single compiled CLI that measures LLM inference performance across machines and produces comparable **tokens/second** reports (plus load time and TTFT).

## What it measures

| Dimension | Details |
|-----------|---------|
| **Models** | Any GGUF model; defaults include Qwen2.5 (0.5B–3B) and Llama-3.2 (1B/3B) |
| **Weight sizes** | Configurable; start with Q4_K_M quantized models that run on CPU |
| **Workloads** | Summarization, code generation, generic assistant (fixed prompts for reproducibility) |
| **Devices** | `cpu`, `gpu`, `auto` via llama.cpp flags |
| **Run modes** | First request after server load (“cold”), then **warm_runs** sequential requests on the same loaded `llama-server` (no composite score—raw metrics only) |

## Metrics collected

- **Tokens/second** — mean / p50 / p95 from the generation phase (**warm** repeats and **cold** first completion)
- **Load time** — model ready (from server start through health / first use, as recorded per scenario)
- **Time to first token (TTFT)** on the cold path
- **Memory** — where available from llama.cpp output

Concurrent batch benchmarking is optional via `batch_sizes` in config; default configs leave it empty.

1. **llama.cpp** — `llama-server` must be on `PATH` or installed via `llmb setup` (default benchmarks drive the HTTP API).
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

# 4. Run the benchmark matrix from bench.toml (default config uses cpu + gpu when available)
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
| `results.json` | Canonical output: hardware info, `schema_version`, every scenario’s cold/warm metrics and raw samples |
| `results.csv` | Flat rows keyed by model × workload × device — primarily **tokens/sec**, load, TTFT |
| `report.html` | Chart.js bar charts: warm tok/s, cold tok/s, load time, TTFT |

## Comparing machines

Run `llmb bench` on each machine and collect `results.csv` (or feed several `results.json` files into **`llmb compare`**). Workloads and token budgets are fixed in the binary, so **tokens/second** columns are directly comparable given the same model file and similar llama.cpp builds.

## Sample runs (four machines)

These are checked-in reports from the **same workload matrix** (model **Qwen3-0.6B-Q4_K_M**, multiple scenarios on **CPU and GPU**). Numbers are **unweighted averages** across all **12** scenarios in each `results.json`, plus averages over the **6** GPU-only scenarios. Different **llama.cpp** builds and drivers apply; `llama_cpp_version` was often not recorded. Curated HTML under `reports/samples/` may still show an older layout until you re-run **`llmb bench`** with this version.

| Machine (OS · CPU · accelerator) | Avg warm tok/s (12) | Avg cold tok/s (12) | Avg warm tok/s (GPU ×6) | Avg cold tok/s (GPU ×6) |
|----------------------------------|--------------------:|--------------------:|------------------------:|------------------------:|
| [Windows · Intel Core Ultra 9 285K · RTX 4080 SUPER](reports/samples/windows-x86_64__CPU-Intel_Core_Ultra_9_285K__GPU-NVIDIA_GeForce_RTX_4080_SUPER__20260516T225708/report.html) | 228 | 216 | 366 | 343 |
| [Windows · AMD Ryzen 9 9950X3D · RTX 5090](reports/samples/windows-x86_64__CPU-AMD_Ryzen_9_9950X3D_16_Core_Processor__GPU-NVIDIA_GeForce_RTX_5090__20260517T122142/report.html) | 290 | 262 | 515 | 459 |
| [Linux · Intel Core Ultra 9 285K · RTX 4080 SUPER](reports/linux-x86_64__CPU-Intel_Core_Ultra_9_285K__GPU-NVIDIA_GeForce_RTX_4080_SUPER__20260517T004235/report.html) | 234 | 225 | 413 | 396 |
| [macOS · Apple M4 · Apple Silicon GPU](reports/macos-aarch64__CPU-Apple_M4__GPU-Apple_Silicon_GPU__20260516T232836/report.html) | 75 | 70 | 83 | 74 |

**How to read this:** higher **tok/s** is faster generation. **Cold** = first completion after the model is loaded on the server; **warm** = subsequent repeats in the same scenario. Open each **`report.html`** for full per-scenario bars and latency columns.

## Config reference

```toml
# bench.toml
devices      = ["cpu", "gpu"]
warm_runs    = 1
batch_sizes  = []            # optional: e.g. [1, 4, 8] for batch experiments
max_tokens   = 256
cpu_threads  = 0
gpu_layers   = -1
mixed_gpu_layers = 16

[[models]]
name         = "Qwen3-0.6B-Q4_K_M"
filename     = "Qwen_Qwen3-0.6B-Q4_K_M.gguf"
url          = "https://huggingface.co/..."
params       = "0.6B"
quantization = "Q4_K_M"
context_length = 1024
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
