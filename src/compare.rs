/// Cross-machine comparison report.
///
/// Reads two or more results.json files (one per machine / run) and produces
/// a single self-contained HTML page that places every machine side-by-side so
/// the VRAM-overflow cliff is immediately visible.
use std::path::Path;

use crate::runner::RunResults;

pub fn generate(result_files: &[std::path::PathBuf], out: &Path) -> anyhow::Result<()> {
    let mut runs: Vec<RunResults> = Vec::new();
    for path in result_files {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Cannot read {}: {}", path.display(), e))?;
        let r: RunResults = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("Cannot parse {}: {}", path.display(), e))?;
        runs.push(r);
    }
    let html = render(&runs)?;
    std::fs::write(out, html)?;
    Ok(())
}

/// A short label used on charts to identify a machine.
fn machine_label(r: &RunResults) -> String {
    format!(
        "{} / {} / {:.0}GB RAM",
        r.hardware.os,
        r.hardware.cpu_brand
            .split_whitespace()
            .take(3)
            .collect::<Vec<_>>()
            .join(" "),
        r.hardware.ram_total_gb
    )
}

fn render(runs: &[RunResults]) -> anyhow::Result<String> {
    // Collect all (model, workload, device) combinations across all runs.
    // We build separate chart datasets per combination.

    let machine_labels: Vec<String> = runs.iter().map(machine_label).collect();
    let machine_labels_json = serde_json::to_string(&machine_labels)?;

    // ── Chart 1: tok/s vs model size, per machine (one line per machine) ──────
    // For each machine, for each model (sorted by param class), collect mean tok/s
    // averaged over all workloads (so we get one number per model×machine).
    let all_models = unique_models(runs);
    let all_models_json = serde_json::to_string(&all_models)?;

    let tps_by_machine: Vec<Vec<Option<f64>>> = runs
        .iter()
        .map(|r| {
            all_models
                .iter()
                .map(|m| mean_tps_for_model(r, m))
                .collect()
        })
        .collect();
    let tps_by_machine_json = serde_json::to_string(&tps_by_machine)?;

    // ── Chart 2: load time vs model size, per machine ─────────────────────────
    let load_by_machine: Vec<Vec<Option<f64>>> = runs
        .iter()
        .map(|r| {
            all_models
                .iter()
                .map(|m| mean_load_for_model(r, m))
                .collect()
        })
        .collect();
    let load_by_machine_json = serde_json::to_string(&load_by_machine)?;

    // ── Chart 3: TTFT cold vs model, per machine ──────────────────────────────
    let ttft_by_machine: Vec<Vec<Option<f64>>> = runs
        .iter()
        .map(|r| {
            all_models
                .iter()
                .map(|m| mean_ttft_for_model(r, m))
                .collect()
        })
        .collect();
    let ttft_by_machine_json = serde_json::to_string(&ttft_by_machine)?;

    // ── Chart 4: cold tok/s vs model, per machine ───────────────────────────────
    let cold_tps_by_machine: Vec<Vec<Option<f64>>> = runs
        .iter()
        .map(|r| {
            all_models
                .iter()
                .map(|m| mean_cold_tps_for_model(r, m))
                .collect()
        })
        .collect();
    let cold_tps_by_machine_json = serde_json::to_string(&cold_tps_by_machine)?;

    // ── Table rows: every scenario from every machine ─────────────────────────
    let table_rows = table_rows(runs, &machine_labels);

    // ── Hardware table ────────────────────────────────────────────────────────
    let hw_rows = hw_rows(runs, &machine_labels);

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>LLM Benchmark — Cross-Machine Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d2e; --accent: #7c6af7;
    --text: #e2e8f0; --muted: #94a3b8; --border: #2d3748;
    --green: #22c55e; --yellow: #eab308; --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; color: var(--accent); margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.2rem; color: var(--muted); margin: 2rem 0 1rem;
       border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 1.5rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }}
  .card h3 {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 1rem;
              text-transform: uppercase; letter-spacing: .05em; }}
  canvas {{ max-height: 320px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: var(--border); color: var(--muted); text-align: left;
        padding: 0.5rem 0.75rem; font-weight: 600; }}
  td {{ padding: 0.45rem 0.75rem; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: rgba(124,106,247,.07); }}
  td.num {{ text-align: right; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; font-size: 0.82rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .cliff-note {{
    background: rgba(239,68,68,.12); border: 1px solid var(--red);
    border-radius: 8px; padding: 0.75rem 1rem; color: #fca5a5;
    font-size: 0.85rem; margin-bottom: 1.5rem;
  }}
</style>
</head>
<body>
<h1>LLM Benchmark — Cross-Machine Comparison</h1>
<p class="meta">Generated from {n_runs} result files across {n_models} models</p>

<div class="cliff-note">
  <strong>Reading the VRAM cliff:</strong> On a machine where a model no longer fits in VRAM,
  llama.cpp must shuffle layers between GPU and CPU each forward pass. This appears as a sharp
  drop in tokens/sec and a steep rise in load time as model size crosses the VRAM limit.
  Look for the inflection point in the charts below.
</div>

<h2>Hardware</h2>
<div class="card" style="overflow-x:auto">
  <table>
    <thead><tr><th>Machine</th><th>OS</th><th>CPU</th><th>RAM</th><th>GPU</th></tr></thead>
    <tbody>{hw_rows}</tbody>
  </table>
</div>

<h2>Performance vs Model Size</h2>
<div id="legend" class="legend"></div>
<div class="grid">
  <div class="card">
    <h3>Tokens / second — warm (higher = better)</h3>
    <canvas id="chartTps"></canvas>
  </div>
  <div class="card">
    <h3>Tokens / second — cold (first completion after load)</h3>
    <canvas id="chartColdTps"></canvas>
  </div>
  <div class="card">
    <h3>Cold-start load time ms (lower = better)</h3>
    <canvas id="chartLoad"></canvas>
  </div>
  <div class="card">
    <h3>Time to first token — cold ms (lower = better)</h3>
    <canvas id="chartTtft"></canvas>
  </div>
</div>

<h2>All Scenarios</h2>
<div class="card" style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>Machine</th><th>Model</th><th>Workload</th><th>Device</th>
      <th>Warm tok/s</th><th>Cold tok/s</th><th>Load (ms)</th><th>TTFT cold (ms)</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>
</div>

<script>
const MACHINE_LABELS = {machine_labels_json};
const MODELS         = {all_models_json};
const TPS            = {tps_by_machine_json};
const LOAD           = {load_by_machine_json};
const TTFT           = {ttft_by_machine_json};
const COLD_TPS       = {cold_tps_by_machine_json};

const PALETTE = [
  '#7c6af7','#22c55e','#f59e0b','#38bdf8','#f472b6',
  '#a78bfa','#34d399','#fb923c','#60a5fa','#e879f9'
];

function color(i) {{ return PALETTE[i % PALETTE.length]; }}

// Build legend
const legendEl = document.getElementById('legend');
MACHINE_LABELS.forEach((lbl, i) => {{
  legendEl.innerHTML +=
    `<span class="legend-item">
       <span class="legend-dot" style="background:${{color(i)}}"></span>
       ${{lbl}}
     </span>`;
}});

function lineChart(id, datasets, yLabel) {{
  new Chart(document.getElementById(id).getContext('2d'), {{
    type: 'line',
    data: {{
      labels: MODELS,
      datasets
    }},
    options: {{
      responsive: true,
      spanGaps: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '#2d3748' }}, ticks: {{ color: '#94a3b8' }} }},
        y: {{ grid: {{ color: '#2d3748' }}, ticks: {{ color: '#94a3b8' }},
              title: {{ display: true, text: yLabel, color: '#94a3b8' }} }}
      }}
    }}
  }});
}}

function makeDatasets(data) {{
  return MACHINE_LABELS.map((lbl, i) => ({{
    label: lbl,
    data: data[i],
    borderColor: color(i),
    backgroundColor: color(i) + '33',
    borderWidth: 2,
    pointRadius: 4,
    tension: 0.2,
  }}));
}}

lineChart('chartTps',   makeDatasets(TPS),   'tok/s');
lineChart('chartColdTps', makeDatasets(COLD_TPS), 'tok/s');
lineChart('chartLoad',  makeDatasets(LOAD),  'ms');
lineChart('chartTtft',  makeDatasets(TTFT),  'ms');
</script>
</body>
</html>"#,
        n_runs = runs.len(),
        n_models = all_models.len(),
        hw_rows = hw_rows,
        table_rows = table_rows,
        machine_labels_json = machine_labels_json,
        all_models_json = all_models_json,
        tps_by_machine_json = tps_by_machine_json,
        load_by_machine_json = load_by_machine_json,
        ttft_by_machine_json = ttft_by_machine_json,
        cold_tps_by_machine_json = cold_tps_by_machine_json,
    );

    Ok(html)
}

/// Sorted list of all model names across all runs, ordered by param class where
/// possible so the x-axis reads small→large.
fn unique_models(runs: &[RunResults]) -> Vec<String> {
    let mut names: Vec<String> = runs
        .iter()
        .flat_map(|r| r.scenarios.iter().map(|s| s.model_name.clone()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    // Sort by embedded gigabyte count if present (e.g. "0.5B", "7B", "32B"),
    // otherwise alphabetically.
    names.sort_by(|a, b| param_key(a).partial_cmp(&param_key(b)).unwrap_or(std::cmp::Ordering::Equal));
    names
}

fn param_key(name: &str) -> f64 {
    // Look for patterns like "0.5B", "1B", "3B", "7B", "14B", "32B", "70B"
    for part in name.split('-') {
        let part = part.trim_end_matches('B');
        if let Ok(v) = part.parse::<f64>() {
            return v;
        }
    }
    f64::MAX
}

fn mean_tps_for_model(r: &RunResults, model: &str) -> Option<f64> {
    let vals: Vec<f64> = r
        .scenarios
        .iter()
        .filter(|s| s.model_name == model && s.warm_metrics.success_rate > 0.0)
        .map(|s| s.warm_metrics.tokens_per_sec_mean)
        .collect();
    mean(&vals)
}

fn mean_load_for_model(r: &RunResults, model: &str) -> Option<f64> {
    let vals: Vec<f64> = r
        .scenarios
        .iter()
        .filter(|s| s.model_name == model && s.cold_metrics.success_rate > 0.0)
        .map(|s| s.cold_metrics.load_time_ms_mean)
        .collect();
    mean(&vals)
}

fn mean_ttft_for_model(r: &RunResults, model: &str) -> Option<f64> {
    let vals: Vec<f64> = r
        .scenarios
        .iter()
        .filter(|s| s.model_name == model && s.cold_metrics.success_rate > 0.0)
        .map(|s| s.cold_metrics.ttft_ms_mean)
        .collect();
    mean(&vals)
}

fn mean_cold_tps_for_model(r: &RunResults, model: &str) -> Option<f64> {
    let vals: Vec<f64> = r
        .scenarios
        .iter()
        .filter(|s| s.model_name == model && s.cold_metrics.success_rate > 0.0)
        .map(|s| s.cold_metrics.tokens_per_sec_mean)
        .collect();
    mean(&vals)
}

fn mean(vals: &[f64]) -> Option<f64> {
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

fn hw_rows(runs: &[RunResults], labels: &[String]) -> String {
    runs.iter()
        .zip(labels.iter())
        .map(|(r, lbl)| {
            let hw = &r.hardware;
            let gpus = if hw.gpu_info.is_empty() {
                "none".to_string()
            } else {
                hw.gpu_info
                    .iter()
                    .map(|g| {
                        let v = g.vram_mb.map(|m| format!(" {}MB", m)).unwrap_or_default();
                        format!("{}{}", g.name, v)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!(
                "<tr><td>{lbl}</td><td>{os} {osv}</td><td>{cpu}</td><td>{ram:.0} GB</td><td>{gpus}</td></tr>",
                lbl = html_escape(lbl),
                os = hw.os,
                osv = hw.os_version,
                cpu = html_escape(&hw.cpu_brand),
                ram = hw.ram_total_gb,
                gpus = html_escape(&gpus),
            )
        })
        .collect()
}

fn table_rows(runs: &[RunResults], labels: &[String]) -> String {
    let mut rows = String::new();
    for (r, lbl) in runs.iter().zip(labels.iter()) {
        for s in &r.scenarios {
            rows.push_str(&format!(
                r#"<tr>
  <td>{lbl}</td>
  <td>{model}</td><td>{wl}</td><td>{dev}</td>
  <td class="num">{wtps:.1}</td><td class="num">{ctps:.1}</td><td class="num">{load:.0}</td><td class="num">{ttft:.0}</td>
</tr>"#,
                lbl = html_escape(lbl),
                model = html_escape(&s.model_name),
                wl = html_escape(&s.workload_label),
                dev = html_escape(&s.device),
                wtps = s.warm_metrics.tokens_per_sec_mean,
                ctps = s.cold_metrics.tokens_per_sec_mean,
                load = s.cold_metrics.load_time_ms_mean,
                ttft = s.cold_metrics.ttft_ms_mean,
            ));
        }
    }
    rows
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
