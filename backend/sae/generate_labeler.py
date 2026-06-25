"""
Generate the SAE feature labeler HTML for the SAE2 / qwen22_knn variant.

Adapted from Jiahao-Jerry/Style_SAE index.html:
  - Same sidebar + lift-bar UI
  - Examples changed from original/rewrite pairs to top activating posts
    (no synthetic pairs yet; those come from pairs.py later)
  - Axes updated to the 9 SAE2 axes
  - Thresholds updated to SAE2_CONFIRM / SAE2_PARTIAL

Output: data/sae2/labeler.html  (open directly in a browser, no server needed)

Run: python backend/sae/generate_labeler.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))

from config.axes import ALL_AXIS_NAMES
from config.settings import (
    SAE2_VARIANTS_DIR, SAE2_DATASET_FILE, SAE2_LABELS_FILE,
    SAE2_CONFIRM, SAE2_PARTIAL, SAE2_DEAD_DENSITY,
)

VARIANT      = "qwen22_knn"
TOP_K        = 10
OUT_FILE     = APP_ROOT / "data/sae2/labeler.html"

AXIS_COLORS = {
    "reading_level":    "#6c8ebf",
    "concreteness":     "#d6b656",
    "narrativity":      "#d79b00",
    "hedging":          "#ae4132",
    "tone":             "#82b366",
    "warmth":           "#e8734a",
    "self_disclosure":  "#9673a6",
    "casualness":       "#23748a",
    "humor":            "#5cb85c",
}


def build_features() -> list[dict]:
    variant_dir = APP_ROOT / SAE2_VARIANTS_DIR / VARIANT

    activations = np.load(variant_dir / "feature_activations.npy")  # (9500, 128)
    corr_records = json.loads((variant_dir / "correlations.json").read_text())

    dataset = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labels  = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)

    # Build post_id → axis scores map (only for labeled posts)
    axis_scores: dict[str, dict] = {}
    for row in labels.itertuples():
        scores = {ax: round(float(getattr(row, ax)), 3)
                  for ax in ALL_AXIS_NAMES if hasattr(row, ax)}
        axis_scores[str(row.post_id)] = scores

    corr_by_feat = {r["feature"]: r for r in corr_records}
    features = []

    for f_idx in range(activations.shape[1]):
        col  = activations[:, f_idx]
        corr = corr_by_feat.get(f_idx, {})

        # Top-K posts by activation
        top_idx = np.argsort(-col)[:TOP_K]
        examples = []
        for i in top_idx:
            if col[i] <= 0:
                break
            row = dataset.iloc[int(i)]
            pid = str(row["post_id"])
            examples.append({
                "pid":        pid,
                "topic":      str(row.get("topic_name", "")),
                "activation": round(float(col[int(i)]), 4),
                "text":       str(row.get("text", ""))[:600],
                "axes":       axis_scores.get(pid, {}),
            })

        lifts = corr.get("lifts", {})
        rs    = corr.get("correlations", {})
        for ax in ALL_AXIS_NAMES:
            lifts.setdefault(ax, 0.0)
            rs.setdefault(ax, 0.0)

        features.append({
            "f":         f_idx,
            "density":   corr.get("density", 0.0),
            "category":  corr.get("category", "novel_candidate"),
            "lifts":     {ax: round(lifts.get(ax, 0.0), 4) for ax in ALL_AXIS_NAMES},
            "rs":        {ax: round(rs.get(ax, 0.0), 4)    for ax in ALL_AXIS_NAMES},
            "best_axis": corr.get("best_axis"),
            "best_r":    round(corr.get("best_r", 0.0), 4),
            "best_lift": round(corr.get("best_lift", 0.0), 4),
            "examples":  examples,
        })

    return features


def render_html(features: list[dict]) -> str:
    features_json = json.dumps(features, ensure_ascii=False)
    axis_colors_json = json.dumps(AXIS_COLORS)
    axis_names_json  = json.dumps(ALL_AXIS_NAMES)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAE Feature Labeler — qwen22_knn</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f1117; color: #e8eaf0; font-family: 'Segoe UI', sans-serif; display: flex; height: 100vh; overflow: hidden; }}

  #sidebar {{ width: 280px; min-width: 280px; background: #1e2130; border-right: 1px solid #2a2d3e; display: flex; flex-direction: column; }}
  #sidebar-header {{ padding: 14px 16px; border-bottom: 1px solid #2a2d3e; font-size: 13px; color: #aaa; }}
  #sidebar-header strong {{ color: #e8eaf0; font-size: 15px; display: block; margin-bottom: 4px; }}
  #feature-list {{ overflow-y: auto; flex: 1; }}
  .feat-item {{ padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #23263a; transition: background 0.15s; }}
  .feat-item:hover {{ background: #252840; }}
  .feat-item.active {{ background: #2e3355; border-left: 3px solid #6c8ebf; }}
  .feat-item .feat-title {{ font-size: 13px; font-weight: 600; }}
  .feat-item .feat-sub {{ font-size: 11px; color: #aaa; margin-top: 2px; }}
  .cat-badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; margin-left: 6px; }}

  #detail {{ flex: 1; overflow-y: auto; padding: 24px 32px; }}
  #detail h2 {{ font-size: 20px; margin-bottom: 6px; }}
  #detail .meta {{ font-size: 13px; color: #aaa; margin-bottom: 20px; }}
  .section-title {{ font-size: 13px; font-weight: 700; color: #aaa; text-transform: uppercase; letter-spacing: 0.08em; margin: 24px 0 10px; }}

  .axis-row {{ margin-bottom: 10px; }}
  .axis-row-label {{ font-size: 12px; color: #ccc; margin-bottom: 3px; }}
  .axis-row-label.best {{ color: #fff; font-weight: 700; }}
  .metric-row {{ display: flex; align-items: center; margin-bottom: 3px; }}
  .metric-tag {{ width: 36px; font-size: 10px; color: #777; text-align: right; padding-right: 8px; flex-shrink: 0; }}
  .bar-bg {{ flex: 1; height: 14px; background: #2a2d3e; border-radius: 3px; }}
  .bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
  .bar-val {{ width: 58px; text-align: right; font-size: 11px; color: #ccc; padding-left: 8px; }}

  .example-card {{ background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px; }}
  .example-meta {{ font-size: 11px; color: #888; margin-bottom: 8px; }}
  .activation-badge {{ background: #2a2d3e; padding: 1px 6px; border-radius: 3px; font-family: monospace; color: #e8eaf0; }}
  .text-block {{ font-size: 13px; line-height: 1.6; color: #d0d4e8; white-space: pre-wrap; }}
  .axes-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .axis-chip {{ font-size: 10px; padding: 2px 7px; border-radius: 10px; background: #2a2d3e; color: #aaa; }}
  .axis-chip.high {{ color: #5cb85c; }}
  .axis-chip.low  {{ color: #e6a817; }}

  #filter-bar {{ padding: 8px 12px; border-bottom: 1px solid #2a2d3e; }}
  #filter-bar select {{ background: #2a2d3e; color: #e8eaf0; border: none; padding: 4px 8px; border-radius: 4px; font-size: 12px; width: 100%; }}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">
    <strong>SAE Feature Labeler</strong>
    qwen22_knn · 128 features
  </div>
  <div id="filter-bar">
    <select id="cat-filter" onchange="applyFilter()">
      <option value="all">All categories</option>
      <option value="confirms_axis">✓ confirms_axis</option>
      <option value="partial_overlap">~ partial_overlap</option>
      <option value="novel_candidate">? novel_candidate</option>
      <option value="dead">✗ dead</option>
    </select>
  </div>
  <div id="feature-list"></div>
</div>
<div id="detail"><p style="color:#555;margin-top:40px;text-align:center">Select a feature →</p></div>

<script>
const FEATURES     = {features_json};
const AXIS_COLORS  = {axis_colors_json};
const CAT_COLORS   = {{"confirms_axis":"#5cb85c","partial_overlap":"#e6a817","novel_candidate":"#6c8ebf","dead":"#555"}};
const AXIS_NAMES   = {axis_names_json};
const CONFIRM_LIFT = {SAE2_CONFIRM};
const PARTIAL_LIFT = {SAE2_PARTIAL};

function buildSidebar(items) {{
  const list = document.getElementById('feature-list');
  list.innerHTML = '';
  items.forEach((feat, i) => {{
    const item = document.createElement('div');
    item.className = 'feat-item';
    item.dataset.fidx = feat.f;
    const catColor = CAT_COLORS[feat.category] || '#555';
    const axLabel  = feat.best_axis ? feat.best_axis.replace(/_/g,' ') : '—';
    const sym      = feat.category === 'confirms_axis' ? '✓' : feat.category === 'partial_overlap' ? '~' : feat.category === 'dead' ? '✗' : '?';
    item.innerHTML = `
      <div class="feat-title">
        F${{feat.f}}
        <span class="cat-badge" style="background:${{catColor}}22;color:${{catColor}}">${{sym}}</span>
      </div>
      <div class="feat-sub">${{axLabel}} · lift=${{feat.best_lift.toFixed(3)}} · ρ=${{feat.density.toFixed(3)}}</div>`;
    item.addEventListener('click', () => showFeature(feat.f));
    list.appendChild(item);
  }});
}}

function applyFilter() {{
  const cat = document.getElementById('cat-filter').value;
  const filtered = cat === 'all' ? FEATURES : FEATURES.filter(f => f.category === cat);
  buildSidebar(filtered);
}}

function showFeature(fIdx) {{
  document.querySelectorAll('.feat-item').forEach(el => el.classList.remove('active'));
  const active = document.querySelector(`.feat-item[data-fidx="${{fIdx}}"]`);
  if (active) active.classList.add('active');

  const feat = FEATURES.find(f => f.f === fIdx);
  if (!feat) return;
  const catColor = CAT_COLORS[feat.category] || '#555';

  const maxLift = Math.max(...Object.values(feat.lifts).map(Math.abs), 0.01);
  const maxR    = Math.max(...Object.values(feat.rs).map(Math.abs), 0.01);
  const absMax  = Math.max(maxLift, maxR);

  let liftHtml = '';
  AXIS_NAMES.forEach(ax => {{
    const lift   = feat.lifts[ax] || 0;
    const r      = feat.rs[ax]    || 0;
    const color  = AXIS_COLORS[ax] || '#6c8ebf';
    const isBest = ax === feat.best_axis;
    const liftPct = Math.abs(lift) / absMax * 100;
    const rPct    = Math.abs(r)    / absMax * 100;
    liftHtml += `
      <div class="axis-row">
        <div class="axis-row-label ${{isBest ? 'best' : ''}}">${{ax.replace(/_/g,' ')}}</div>
        <div class="metric-row">
          <div class="metric-tag">lift</div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:${{liftPct}}%;background:${{color}};opacity:${{isBest ? 1 : 0.55}}"></div>
          </div>
          <div class="bar-val">${{lift >= 0 ? '+' : ''}}${{lift.toFixed(3)}}</div>
        </div>
        <div class="metric-row">
          <div class="metric-tag">r</div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:${{rPct}}%;background:${{color}};opacity:${{isBest ? 0.65 : 0.3}}"></div>
          </div>
          <div class="bar-val">${{r >= 0 ? '+' : ''}}${{r.toFixed(3)}}</div>
        </div>
      </div>`;
  }});

  let exHtml = '';
  if (!feat.examples || feat.examples.length === 0) {{
    exHtml = '<p style="color:#555">Feature never fires on this corpus.</p>';
  }} else {{
    feat.examples.forEach(ex => {{
      const axesHtml = Object.entries(ex.axes || {{}}).map(([ax, v]) => {{
        const cls = v >= 0.6 ? 'high' : v <= 0.3 ? 'low' : '';
        return `<span class="axis-chip ${{cls}}">${{ax.replace(/_/g,' ')}} ${{v.toFixed(2)}}</span>`;
      }}).join('');
      exHtml += `
        <div class="example-card">
          <div class="example-meta">
            post ${{ex.pid}} &nbsp;·&nbsp; ${{ex.topic}} &nbsp;·&nbsp;
            activation: <span class="activation-badge">${{ex.activation.toFixed(4)}}</span>
          </div>
          <div class="text-block">${{escHtml(ex.text)}}</div>
          ${{axesHtml ? `<div class="axes-row">${{axesHtml}}</div>` : ''}}
        </div>`;
    }});
  }}

  document.getElementById('detail').innerHTML = `
    <h2>Feature F${{feat.f}}</h2>
    <div class="meta">
      <span class="cat-badge" style="background:${{catColor}}22;color:${{catColor}};font-size:12px">${{feat.category}}</span>
      &nbsp; best axis: <strong>${{feat.best_axis || '—'}}</strong>
      &nbsp; lift: <strong>${{feat.best_lift.toFixed(4)}}</strong>
      &nbsp; r: <strong>${{feat.best_r.toFixed(4)}}</strong>
      &nbsp; density: <strong>${{feat.density.toFixed(4)}}</strong>
    </div>
    <div class="section-title">Lift across 9 axes</div>
    ${{liftHtml}}
    <div class="section-title">Top ${{feat.examples.length}} activating posts</div>
    ${{exHtml}}`;
}}

function escHtml(s) {{
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

buildSidebar(FEATURES);
if (FEATURES.length > 0) showFeature(FEATURES[0].f);
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("Building feature data…")
    features = build_features()
    confirmed = sum(1 for f in features if f["category"] == "confirms_axis")
    partial   = sum(1 for f in features if f["category"] == "partial_overlap")
    novel     = sum(1 for f in features if f["category"] == "novel_candidate")
    dead      = sum(1 for f in features if f["category"] == "dead")
    print(f"  {len(features)} features: {confirmed} confirmed, {partial} partial, {novel} novel, {dead} dead")

    print("Rendering HTML…")
    html = render_html(features)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Saved → {OUT_FILE}")
    print(f"Open in browser: file://{OUT_FILE}")
