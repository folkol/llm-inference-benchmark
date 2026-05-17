#!/usr/bin/env python3
"""
Regenerate README throughput tables + reports/samples/cross-machine-matrix.html
from checked-in sample results.json files.

Run from repo root:
    python scripts/gen_cross_machine_matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical sample runs (same matrix: two Qwen3 models × 3 workloads × cpu/gpu).
# Column order: Windows/Linux on the same physical Ultra 9 285K + RTX 4080 box side-by-side,
# then other hosts.
RUNS: list[dict[str, str]] = [
    {
        "id": "win-4080",
        "label": "Windows · Ultra 9 285K · RTX 4080 SUPER",
        "json": "reports/samples/windows-x86_64__CPU-Intel_Core_Ultra_9_285K__GPU-NVIDIA_GeForce_RTX_4080_SUPER__20260516T225708/results.json",
        "report": "reports/samples/windows-x86_64__CPU-Intel_Core_Ultra_9_285K__GPU-NVIDIA_GeForce_RTX_4080_SUPER__20260516T225708/report.html",
    },
    {
        "id": "linux-4080",
        "label": "Linux · Ultra 9 285K · RTX 4080 SUPER",
        "json": "reports/linux-x86_64__CPU-Intel_Core_Ultra_9_285K__GPU-NVIDIA_GeForce_RTX_4080_SUPER__20260517T004235/results.json",
        "report": "reports/linux-x86_64__CPU-Intel_Core_Ultra_9_285K__GPU-NVIDIA_GeForce_RTX_4080_SUPER__20260517T004235/report.html",
    },
    {
        "id": "win-5090",
        "label": "Windows · Ryzen 9950X3D · RTX 5090",
        "json": "reports/samples/windows-x86_64__CPU-AMD_Ryzen_9_9950X3D_16_Core_Processor__GPU-NVIDIA_GeForce_RTX_5090__20260517T122142/results.json",
        "report": "reports/samples/windows-x86_64__CPU-AMD_Ryzen_9_9950X3D_16_Core_Processor__GPU-NVIDIA_GeForce_RTX_5090__20260517T122142/report.html",
    },
    {
        "id": "mac-m4",
        "label": "macOS · Apple M4 · Apple Silicon GPU",
        "json": "reports/macos-aarch64__CPU-Apple_M4__GPU-Apple_Silicon_GPU__20260516T232836/results.json",
        "report": "reports/macos-aarch64__CPU-Apple_M4__GPU-Apple_Silicon_GPU__20260516T232836/report.html",
    },
]


def load_scenarios(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["scenarios"]


def scenario_sort_key(s: dict) -> tuple[int, str, int]:
    """All CPU scenarios first, then GPU; within device by model then workload."""
    dev_rank = 0 if s["device"] == "cpu" else 1
    model = s["model_name"]
    order = {"summarize": 0, "code": 1, "assistant": 2}
    wid = s["workload_id"]
    return (dev_rank, model, order.get(wid, 99))


def scenario_row_markdown(s: dict) -> str:
    """Row label when CPU/GPU is implied by table section."""
    model = s["model_name"].replace("-Q4_K_M", "")
    return f"{model} · {s['workload_label']}"


def tok_mean(metrics: dict, key: str = "tokens_per_sec_mean") -> float | None:
    if metrics.get("success_rate", 1.0) == 0:
        return None
    v = metrics.get(key)
    return float(v) if v is not None else None


def fmt_cell(v: float | None) -> str:
    if v is None:
        return "—"
    return str(round(v))


def markdown_tables(
    matrix_warm: dict,
    matrix_cold: dict,
    rows_plain: list[str],
    scenarios_ref: list[dict],
) -> str:
    lines: list[str] = []

    def emit_device_table(section: str, mat: dict[str, dict[str, float | None]], device: str) -> None:
        lines.append(f"**{section}**")
        lines.append("")
        header = "| Scenario | " + " | ".join(
            f"[{r['label']}]({r['report']})" for r in RUNS
        )
        header += " |"
        sep = "|" + "---|" * (1 + len(RUNS))
        lines.append(header)
        lines.append(sep)
        for plain, scen in zip(rows_plain, scenarios_ref):
            if scen["device"] != device:
                continue
            md = scenario_row_markdown(scen)
            cells = [fmt_cell(mat[plain][r["id"]]) for r in RUNS]
            lines.append("| " + md + " | " + " | ".join(cells) + " |")
        lines.append("")

    def emit_metric_pair(title: str, mat: dict[str, dict[str, float | None]]) -> None:
        lines.append(f"**{title}**")
        lines.append("")
        emit_device_table("CPU", mat, "cpu")
        emit_device_table("GPU", mat, "gpu")

    emit_metric_pair("Warm throughput (mean tok/s, generation phase)", matrix_warm)
    emit_metric_pair("Cold throughput (mean tok/s, first completion after load)", matrix_cold)
    return "\n".join(lines).rstrip()


def write_html(
    path: Path,
    ordered_labels_plain: list[str],
    matrix_warm: dict[str, dict[str, float | None]],
    matrix_cold: dict[str, dict[str, float | None]],
) -> None:
    run_meta = [{"id": r["id"], "label": r["label"]} for r in RUNS]
    payload = {
        "runs": run_meta,
        "rows": ordered_labels_plain,
        "warm": matrix_warm,
        "cold": matrix_cold,
    }
    matrix_json = json.dumps(payload, separators=(",", ":"))

    doc = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>llmb — cross-machine throughput (samples)</title>
  <style>
    :root {{
      font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --bg: #0f1419;
      --fg: #e6edf3;
      --muted: #8b949e;
      --border: #30363d;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      line-height: 1.45;
      padding: 1.25rem 1.5rem 2rem;
    }}
    h1 {{ font-size: 1.15rem; font-weight: 650; margin: 0 0 0.35rem; }}
    p.lead {{ color: var(--muted); margin: 0 0 1rem; font-size: 0.92rem; max-width: 52rem; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem 1.25rem;
      align-items: center;
      margin-bottom: 1rem;
    }}
    .toolbar label {{ font-size: 0.82rem; color: var(--muted); }}
    button {{
      background: #21262d;
      border: 1px solid var(--border);
      color: var(--fg);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.82rem;
    }}
    button.active {{ background: #388bfd33; border-color: #388bfd; }}
    table {{
      border-collapse: collapse;
      font-size: 0.78rem;
      width: max-content;
      max-width: 100%;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.35rem 0.5rem;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      min-width: 14rem;
      max-width: 22rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    th {{ background: #161b22; font-weight: 600; }}
    .cell {{
      font-variant-numeric: tabular-nums;
      font-weight: 600;
      color: #0f1419;
    }}
    tr.section td {{
      background: #161b22;
      color: #8b949e;
      font-weight: 600;
      text-align: left;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-top: 1px solid #30363d;
    }}
    footer {{ margin-top: 1.25rem; font-size: 0.78rem; color: var(--muted); }}
  </style>
</head>
<body>
  <h1>Cross-machine throughput — sample runs</h1>
  <p class="lead">
    Checked-in <code>results.json</code> samples (same workload matrix per machine).
    Columns group <strong>Windows and Linux on the same physical Ultra 9 285K + RTX 4080 box</strong>, then the Ryzen + 5090 Windows PC and the M4 Mac.
    Rows list all <strong>CPU</strong> scenarios first, then <strong>GPU</strong> (section headers in the table). Cells are rounded mean tokens/s for the generation phase; use Warm vs Cold to compare steady-state vs first completion after load.
  </p>
  <div class="toolbar">
    <span><label>Metric</label><br/>
      <button type="button" id="btn-warm" class="active">Warm tok/s</button>
      <button type="button" id="btn-cold">Cold tok/s</button>
    </span>
  </div>
  <div id="wrap"></div>
  <footer>Generated by <code>scripts/gen_cross_machine_matrix.py</code>. Re-run after updating sample JSON.</footer>
  <script type="application/json" id="matrix-data">__MATRIX_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('matrix-data').textContent);

    function numericValues(mat) {{
      const out = [];
      for (const row of DATA.rows) {{
        for (const run of DATA.runs) {{
          const v = mat[row][run.id];
          if (typeof v === 'number' && !Number.isNaN(v)) out.push(v);
        }}
      }}
      return out;
    }}

    function heatColor(val, vmin, vmax) {{
      if (val == null || vmax <= vmin) return '#2d3748';
      let t = (val - vmin) / (vmax - vmin);
      t = Math.max(0, Math.min(1, t));
      const lo = [45, 35, 52], hi = [43, 96, 56];
      const rgb = lo.map((a, i) => Math.round((a / 100 + (hi[i]/100 - a/100) * t) * 255));
      return '#' + rgb.map(x => x.toString(16).padStart(2, '0')).join('');
    }}

    let mode = 'warm';

    function render() {{
      const mat = mode === 'warm' ? DATA.warm : DATA.cold;
      const vals = numericValues(mat);
      if (!vals.length) {{
        document.getElementById('wrap').innerHTML = '<p>No numeric data.</p>';
        return;
      }}
      const vmin = Math.min(...vals);
      const vmax = Math.max(...vals);

      let h = '<table><thead><tr><th>Scenario</th>';
      for (const run of DATA.runs) {{
        h += '<th>' + escapeHtml(run.label) + '</th>';
      }}
      h += '</tr></thead><tbody>';
      let prevSection = null;
      for (const row of DATA.rows) {{
        const section = row.endsWith(' · CPU') ? 'CPU' : row.endsWith(' · GPU') ? 'GPU' : '';
        if (section && section !== prevSection) {{
          const colspan = DATA.runs.length + 1;
          h += '<tr class="section"><td colspan="' + colspan + '">' + section + '</td></tr>';
          prevSection = section;
        }}
        const displayRow = (() => {{
          const p = row.split(' · ');
          if (p.length >= 2 && (p[p.length - 1] === 'CPU' || p[p.length - 1] === 'GPU')) {{
            p.pop();
            return p.join(' · ');
          }}
          return row;
        }})();
        h += '<tr><td>' + escapeHtml(displayRow) + '</td>';
        for (const run of DATA.runs) {{
          const v = mat[row][run.id];
          const txt = v == null ? '—' : String(Math.round(v));
          const bg = v == null ? '#21262d' : heatColor(v, vmin, vmax);
          const fg = v == null ? '#8b949e' : '#0f1419';
          h += '<td class="cell" style="background:' + bg + ';color:' + fg + '">' + txt + '</td>';
        }}
        h += '</tr>';
      }}
      h += '</tbody></table>';
      document.getElementById('wrap').innerHTML = h;
    }}

    function escapeHtml(s) {{
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}

    document.getElementById('btn-warm').onclick = () => {{
      mode = 'warm';
      document.getElementById('btn-warm').classList.add('active');
      document.getElementById('btn-cold').classList.remove('active');
      render();
    }};
    document.getElementById('btn-cold').onclick = () => {{
      mode = 'cold';
      document.getElementById('btn-cold').classList.add('active');
      document.getElementById('btn-warm').classList.remove('active');
      render();
    }};
    render();
  </script>
</body>
</html>
"""
    doc = doc.replace("{{", "{").replace("}}", "}")
    path.write_text(doc.replace("__MATRIX_JSON__", matrix_json), encoding="utf-8")


def main() -> None:
    paths = [REPO_ROOT / r["json"] for r in RUNS]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"missing sample JSON: {p.relative_to(REPO_ROOT)}")

    scenarios_ref = sorted(load_scenarios(paths[0]), key=scenario_sort_key)

    ordered_plain: list[str] = []
    for s in scenarios_ref:
        ordered_plain.append(
            f"{s['model_name'].replace('-Q4_K_M', '')} · {s['workload_label']} · {s['device'].upper()}"
        )

    matrix_warm: dict[str, dict[str, float | None]] = {}
    matrix_cold: dict[str, dict[str, float | None]] = {}
    for plain in ordered_plain:
        matrix_warm[plain] = {}
        matrix_cold[plain] = {}

    for run in RUNS:
        scen = load_scenarios(REPO_ROOT / run["json"])
        by_key = {(s["model_name"], s["workload_id"], s["device"]): s for s in scen}
        for s in scenarios_ref:
            k = (s["model_name"], s["workload_id"], s["device"])
            plain = f"{s['model_name'].replace('-Q4_K_M', '')} · {s['workload_label']} · {s['device'].upper()}"
            src = by_key.get(k)
            if not src:
                matrix_warm[plain][run["id"]] = None
                matrix_cold[plain][run["id"]] = None
                continue
            matrix_warm[plain][run["id"]] = tok_mean(src["warm_metrics"])
            matrix_cold[plain][run["id"]] = tok_mean(src["cold_metrics"])

    md = markdown_tables(matrix_warm, matrix_cold, ordered_plain, scenarios_ref)
    snippet_path = REPO_ROOT / "scripts" / "README-cross-machine-snippet.generated.md"
    snippet_path.write_text(md + "\n", encoding="utf-8")

    print(f"wrote {snippet_path.relative_to(REPO_ROOT)}")
    out_html = REPO_ROOT / "reports/samples/cross-machine-matrix.html"
    write_html(out_html, ordered_plain, matrix_warm, matrix_cold)
    print(f"wrote {out_html.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
