use std::path::Path;

use crate::{
    config::BenchConfig,
    hardware::HardwareInfo,
    runner::{RunResults, ScenarioResult},
};

pub fn write_all(
    results: &RunResults,
    _hw: &HardwareInfo,
    _cfg: &BenchConfig,
    out_dir: &Path,
) -> anyhow::Result<()> {
    write_json(results, out_dir)?;
    write_csv(results, out_dir)?;
    write_html(results, out_dir)?;
    Ok(())
}

// ─── JSON ────────────────────────────────────────────────────────────────────

fn write_json(results: &RunResults, out_dir: &Path) -> anyhow::Result<()> {
    let path = out_dir.join("results.json");
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(&path, &json)?;
    println!("JSON → {}", path.display());
    Ok(())
}

// ─── CSV ─────────────────────────────────────────────────────────────────────

fn write_csv(results: &RunResults, out_dir: &Path) -> anyhow::Result<()> {
    let path = out_dir.join("results.csv");
    let mut w = csv::Writer::from_path(&path)?;

    w.write_record([
        "run_id",
        "timestamp",
        "model",
        "workload",
        "device",
        "tps_mean",
        "tps_p50",
        "tps_p95",
        "load_time_ms",
        "ttft_ms_mean",
        "ttft_ms_p95",
        "wall_ms_mean",
        "cold_tps",
        "cold_ttft_ms",
        "score",
        "score_tps",
        "score_load",
        "score_ttft",
        "score_batch",
    ])?;

    for s in &results.scenarios {
        w.write_record([
            &results.run_id,
            &results.timestamp,
            &s.model_name,
            &s.workload_id,
            &s.device,
            &fmt(s.warm_metrics.tokens_per_sec_mean),
            &fmt(s.warm_metrics.tokens_per_sec_p50),
            &fmt(s.warm_metrics.tokens_per_sec_p95),
            &fmt(s.warm_metrics.load_time_ms_mean),
            &fmt(s.warm_metrics.ttft_ms_mean),
            &fmt(s.warm_metrics.ttft_ms_p95),
            &fmt(s.warm_metrics.wall_time_ms_mean),
            &fmt(s.cold_metrics.tokens_per_sec_mean),
            &fmt(s.cold_metrics.ttft_ms_mean),
            &fmt(s.score),
            &fmt(s.score_breakdown.tokens_per_sec_score),
            &fmt(s.score_breakdown.load_time_score),
            &fmt(s.score_breakdown.ttft_score),
            &fmt(s.score_breakdown.batch_throughput_score),
        ])?;
    }

    w.flush()?;
    println!("CSV  → {}", path.display());
    Ok(())
}

fn fmt(v: f64) -> String {
    format!("{:.3}", v)
}

// ─── HTML ────────────────────────────────────────────────────────────────────

fn write_html(results: &RunResults, out_dir: &Path) -> anyhow::Result<()> {
    let path = out_dir.join("report.html");
    let html = render_html(results)?;
    std::fs::write(&path, html)?;
    println!("HTML → {}", path.display());
    Ok(())
}

fn render_html(results: &RunResults) -> anyhow::Result<String> {
    let hw = &results.hardware;
    let scenarios = &results.scenarios;

    // Build JSON data blobs for Chart.js
    let labels_json = serde_json::to_string(
        &scenarios
            .iter()
            .map(|s| format!("{} / {} / {}", s.model_name, s.workload_id, s.device))
            .collect::<Vec<_>>(),
    )?;

    let tps_json = serde_json::to_string(
        &scenarios
            .iter()
            .map(|s| round2(s.warm_metrics.tokens_per_sec_mean))
            .collect::<Vec<_>>(),
    )?;

    let score_json = serde_json::to_string(
        &scenarios
            .iter()
            .map(|s| round2(s.score))
            .collect::<Vec<_>>(),
    )?;

    let load_json = serde_json::to_string(
        &scenarios
            .iter()
            .map(|s| round2(s.cold_metrics.load_time_ms_mean))
            .collect::<Vec<_>>(),
    )?;

    let ttft_json = serde_json::to_string(
        &scenarios
            .iter()
            .map(|s| round2(s.cold_metrics.ttft_ms_mean))
            .collect::<Vec<_>>(),
    )?;

    let scenario_rows = scenarios_table_rows(scenarios);
    let hw_summary = hw_table(hw);

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>LLM Benchmark Report – {run_id}</title>
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
  h2 {{ font-size: 1.2rem; color: var(--muted); margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.5rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }}
  .card h3 {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: .05em; }}
  canvas {{ max-height: 300px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: var(--border); color: var(--muted); text-align: left; padding: 0.5rem 0.75rem; font-weight: 600; }}
  td {{ padding: 0.45rem 0.75rem; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: rgba(124,106,247,.07); }}
  .score {{ font-weight: 700; }}
  .score-hi {{ color: var(--green); }}
  .score-mid {{ color: var(--yellow); }}
  .score-lo {{ color: var(--red); }}
  .hw-table td:first-child {{ color: var(--muted); width: 160px; }}
</style>
</head>
<body>
<h1>LLM Benchmark Report</h1>
<p class="meta">Run ID: {run_id} &nbsp;|&nbsp; {timestamp}</p>

<h2>Hardware</h2>
<div class="card">
  <table class="hw-table">{hw_summary}</table>
</div>

<h2>Charts</h2>
<div class="grid">
  <div class="card"><h3>Aggregate Score (higher = better)</h3><canvas id="chartScore"></canvas></div>
  <div class="card"><h3>Tokens / second – warm (higher = better)</h3><canvas id="chartTps"></canvas></div>
  <div class="card"><h3>Cold-start load time (ms, lower = better)</h3><canvas id="chartLoad"></canvas></div>
  <div class="card"><h3>Time to first token – cold (ms, lower = better)</h3><canvas id="chartTtft"></canvas></div>
</div>

<h2>Scenario Results</h2>
<div class="card" style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>Model</th><th>Workload</th><th>Device</th>
      <th>TPS (mean)</th><th>TPS (p50)</th>
      <th>Load (ms)</th><th>TTFT cold (ms)</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>{scenario_rows}</tbody>
</table>
</div>

<script>
const LABELS = {labels_json};
const TPS    = {tps_json};
const SCORE  = {score_json};
const LOAD   = {load_json};
const TTFT   = {ttft_json};

const PALETTE = ['#7c6af7','#22c55e','#f59e0b','#38bdf8','#f472b6','#a78bfa','#34d399'];

function color(i) {{ return PALETTE[i % PALETTE.length]; }}

function barChart(id, data, label, reversed) {{
  const ctx = document.getElementById(id).getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: LABELS,
      datasets: [{{ label, data, backgroundColor: LABELS.map((_, i) => color(i)), borderRadius: 4 }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '#2d3748' }}, ticks: {{ color: '#94a3b8' }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: '#e2e8f0', font: {{ size: 10 }} }} }}
      }}
    }}
  }});
}}

barChart('chartScore', SCORE, 'Score');
barChart('chartTps',   TPS,   'tok/s');
barChart('chartLoad',  LOAD,  'ms', true);
barChart('chartTtft',  TTFT,  'ms', true);
</script>
</body>
</html>"#,
        run_id = results.run_id,
        timestamp = results.timestamp,
        hw_summary = hw_summary,
        scenario_rows = scenario_rows,
        labels_json = labels_json,
        tps_json = tps_json,
        score_json = score_json,
        load_json = load_json,
        ttft_json = ttft_json,
    );

    Ok(html)
}

fn hw_table(hw: &HardwareInfo) -> String {
    let gpus = if hw.gpu_info.is_empty() {
        "none detected".to_string()
    } else {
        hw.gpu_info
            .iter()
            .map(|g| {
                let vram = g
                    .vram_mb
                    .map(|v| format!(" ({} MB VRAM)", v))
                    .unwrap_or_default();
                format!("{} [{}]{}", g.name, g.vendor, vram)
            })
            .collect::<Vec<_>>()
            .join("<br>")
    };

    let llama_ver = hw.llama_cpp_version.as_deref().unwrap_or("unknown");

    format!(
        r#"<tr><td>OS</td><td>{} {}</td></tr>
<tr><td>CPU</td><td>{} ({} physical / {} logical)</td></tr>
<tr><td>RAM</td><td>{:.1} GB</td></tr>
<tr><td>GPU</td><td>{}</td></tr>
<tr><td>llama.cpp</td><td>{}</td></tr>"#,
        hw.os,
        hw.os_version,
        hw.cpu_brand,
        hw.cpu_cores_physical,
        hw.cpu_cores_logical,
        hw.ram_total_gb,
        gpus,
        llama_ver
    )
}

fn scenarios_table_rows(scenarios: &[ScenarioResult]) -> String {
    scenarios
        .iter()
        .map(|s| {
            let sc = s.score;
            let cls = if sc >= 70.0 {
                "score-hi"
            } else if sc >= 40.0 {
                "score-mid"
            } else {
                "score-lo"
            };
            format!(
                r#"<tr>
  <td>{model}</td><td>{wl}</td><td>{dev}</td>
  <td>{tps:.1}</td><td>{tps50:.1}</td>
  <td>{load:.0}</td><td>{ttft:.0}</td>
  <td class="score {cls}">{sc:.1}</td>
</tr>"#,
                model = html_escape(&s.model_name),
                wl = html_escape(&s.workload_label),
                dev = html_escape(&s.device),
                tps = s.warm_metrics.tokens_per_sec_mean,
                tps50 = s.warm_metrics.tokens_per_sec_p50,
                load = s.cold_metrics.load_time_ms_mean,
                ttft = s.cold_metrics.ttft_ms_mean,
                sc = sc,
                cls = cls,
            )
        })
        .collect()
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
