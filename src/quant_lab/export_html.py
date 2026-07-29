"""Export experiment index to static HTML dashboard."""

from __future__ import annotations

import html
import json
from pathlib import Path

from quant_lab.store import ExperimentStore


def _esc(text: str) -> str:
    return html.escape(str(text))


def render_dashboard(store: ExperimentStore) -> str:
    rows = store.list_runs()
    cards = []
    for row in rows[:200]:
        metrics = json.loads(row.metrics_json) if row.metrics_json else {}
        metric_bits = []
        for key in ("total_return", "max_drawdown", "calmar", "sharpe", "strategy"):
            if key in metrics:
                metric_bits.append(f"<li><b>{_esc(key)}</b>: {_esc(metrics[key])}</li>")
        cards.append(
            f"""
            <article class="card">
              <h3>{_esc(row.project)} / {_esc(row.run_id)}</h3>
              <p class="meta">{_esc(row.run_type)} · {_esc(row.scanned_at)}</p>
              <p class="path">{_esc(row.run_path)}</p>
              <ul>{''.join(metric_bits) or '<li>No metrics</li>'}</ul>
            </article>
            """
        )

    body = "\n".join(cards) or "<p>No runs indexed yet. Run <code>quant-lab scan</code> first.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>quant-lab dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0b1020; color: #e8ecf4; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .sub {{ color: #9aa7bd; margin-bottom: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }}
    .card {{ background: #151b2e; border: 1px solid #24304a; border-radius: 12px; padding: 1rem; }}
    .meta, .path {{ color: #9aa7bd; font-size: 0.85rem; word-break: break-all; }}
    ul {{ margin: 0.5rem 0 0; padding-left: 1.2rem; }}
  </style>
</head>
<body>
  <h1>quant-lab dashboard</h1>
  <p class="sub">{len(rows)} indexed run(s)</p>
  <div class="grid">
    {body}
  </div>
</body>
</html>
"""


def export_html(db_path: Path, out_path: Path) -> Path:
    store = ExperimentStore(db_path)
    content = render_dashboard(store)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path
