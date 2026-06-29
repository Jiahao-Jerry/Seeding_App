"""
Generate the SAE feature labeler HTML for all 7 qwen_knn variants.

Outputs docs/index.html (GitHub Pages) with a layer-switcher dropdown so
all variants can be explored in one page without a server.

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
    SAE2_CONFIRM, SAE2_PARTIAL, SAE2_DEAD_DENSITY, SAE2_QWEN_LAYERS,
)

VARIANTS = ["qwen24_raw", "qwen24_knn", "qwen24_knn_l0004", "qwen24_knn_k25_l0004", "bge_raw", "bge_knn"]
TOP_K    = 10
OUT_FILE = APP_ROOT / "docs" / "index.html"

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


def build_features(variant: str, dataset: pd.DataFrame, labels: pd.DataFrame) -> list[dict]:
    variant_dir = APP_ROOT / SAE2_VARIANTS_DIR / variant

    activations = np.load(variant_dir / "feature_activations.npy")
    corr_records = json.loads((variant_dir / "correlations.json").read_text())

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


def build_variant_meta(features: list[dict], variant: str) -> dict:
    """Collect config + stats for one variant from meta.json + feature list."""
    vdir = APP_ROOT / SAE2_VARIANTS_DIR / variant
    meta = json.loads((vdir / "meta.json").read_text())
    fl   = meta["final_loss"]
    hp   = meta.get("hparams", {})
    conf  = sum(1 for f in features if f["category"] == "confirms_axis")
    part  = sum(1 for f in features if f["category"] == "partial_overlap")
    nov   = sum(1 for f in features if f["category"] == "novel_candidate")
    dead  = sum(1 for f in features if f["category"] == "dead")
    n     = len(features)
    return {
        "config": {
            "space":   meta.get("space", "?"),
            "layer":   meta.get("layer", "-"),
            "l1":      hp.get("l1_coef", meta.get("hparams", {}).get("l1_coef", "?")),
            "k":       meta.get("knn_k", 20),
            "removal": meta.get("removal", "?"),
        },
        "score":    None,   # filled in JS from correlations
        "dead_pct": round(100 * fl.get("dead_features", 0) / n) if n else 0,
        "density":  round(fl.get("mean_density_sample", 0.0), 3),
        "recon":    round(fl.get("recon", 0.0), 4),
        "sparsity": round(fl.get("sparsity", 0.0), 4),
        "total":    round(fl.get("total", 0.0), 4),
        "confirmed": conf,
        "partial":   part,
        "novel":     nov,
        "dead":      dead,
    }


def render_html(all_features: dict[str, list[dict]],
                variant_meta: dict[str, dict]) -> str:
    all_features_json  = json.dumps(all_features, ensure_ascii=False)
    variant_meta_json  = json.dumps(variant_meta, ensure_ascii=False)
    axis_colors_json   = json.dumps(AXIS_COLORS)
    axis_names_json    = json.dumps(ALL_AXIS_NAMES)
    variants_json      = json.dumps(VARIANTS)
    default_variant    = "qwen24_knn"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAE Feature Labeler</title>
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

  #filter-bar {{ padding: 8px 12px; border-bottom: 1px solid #2a2d3e; display: flex; flex-direction: column; gap: 6px; }}
  #filter-bar select {{ background: #2a2d3e; color: #e8eaf0; border: none; padding: 4px 8px; border-radius: 4px; font-size: 12px; width: 100%; }}
  #layer-select {{ font-weight: 700; color: #6c8ebf !important; }}

  #variant-stats {{ padding: 10px 12px; border-bottom: 1px solid #2a2d3e; font-size: 11px; }}
  #variant-stats .stats-title {{ font-size: 10px; font-weight: 700; color: #6c8ebf; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 7px; }}
  #variant-stats .cfg-row {{ color: #aaa; margin-bottom: 4px; display: flex; flex-wrap: wrap; gap: 4px; }}
  #variant-stats .cfg-chip {{ background: #2a2d3e; border-radius: 3px; padding: 1px 6px; color: #c8cadd; font-family: monospace; }}
  #variant-stats .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; margin-top: 6px; }}
  #variant-stats .stat-cell {{ background: #16192a; border-radius: 4px; padding: 5px 8px; }}
  #variant-stats .stat-label {{ font-size: 9px; color: #666; text-transform: uppercase; letter-spacing: 0.06em; }}
  #variant-stats .stat-val {{ font-size: 13px; font-weight: 700; color: #e8eaf0; margin-top: 1px; }}
  #variant-stats .cat-counts {{ display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap; }}
  #variant-stats .cat-pill {{ font-size: 10px; padding: 2px 7px; border-radius: 10px; font-weight: 600; }}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">
    <strong>SAE Feature Labeler</strong>
    <span id="variant-label">qwen24_knn · 128 features</span>
  </div>
  <div id="filter-bar">
    <select id="layer-select" onchange="switchLayer(this.value)">
    </select>
    <select id="cat-filter" onchange="applyFilter()">
      <option value="all">All categories</option>
      <option value="confirms_axis">✓ confirms_axis</option>
      <option value="partial_overlap">~ partial_overlap</option>
      <option value="novel_candidate">? novel_candidate</option>
      <option value="dead">✗ dead</option>
    </select>
  </div>
  <div id="variant-stats"></div>
  <div id="feature-list"></div>
</div>
<div id="detail"><p style="color:#555;margin-top:40px;text-align:center">Select a feature →</p></div>

<script>
const ALL_FEATURES  = {all_features_json};
const VARIANT_META  = {variant_meta_json};
const AXIS_COLORS   = {axis_colors_json};
const CAT_COLORS    = {{"confirms_axis":"#5cb85c","partial_overlap":"#e6a817","novel_candidate":"#6c8ebf","dead":"#555"}};
const AXIS_NAMES    = {axis_names_json};
const VARIANTS      = {variants_json};
const CONFIRM_LIFT  = {SAE2_CONFIRM};
const PARTIAL_LIFT  = {SAE2_PARTIAL};
const CAT_ORDER     = {{"confirms_axis": 0, "partial_overlap": 1, "novel_candidate": 2, "dead": 3}};

let FEATURES = [];
let currentVariant = '{default_variant}';

// Populate layer dropdown
const layerSel = document.getElementById('layer-select');
VARIANTS.forEach(v => {{
  const opt = document.createElement('option');
  opt.value = v;
  opt.textContent = v;
  if (v === currentVariant) opt.selected = true;
  layerSel.appendChild(opt);
}});

function updateStats(variant) {{
  const m = VARIANT_META[variant];
  if (!m) {{ document.getElementById('variant-stats').innerHTML = ''; return; }}
  const c = m.config;
  const spaceLabel = c.space === 'qwen' ? 'Qwen2.5-7B' : 'BGE-M3';
  const layerLabel = c.layer !== null && c.layer !== '-' ? 'L' + c.layer : '—';
  const removalLabel = c.removal === 'knn' ? 'kNN (k=' + c.k + ')' : 'raw';
  document.getElementById('variant-stats').innerHTML = `
    <div class="stats-title">Variant Config &amp; Stats</div>
    <div class="cfg-row">
      <span class="cfg-chip">${{spaceLabel}}</span>
      <span class="cfg-chip">layer ${{layerLabel}}</span>
      <span class="cfg-chip">${{removalLabel}}</span>
      <span class="cfg-chip">L1=${{c.l1}}</span>
    </div>
    <div class="stat-grid">
      <div class="stat-cell"><div class="stat-label">Dead %</div><div class="stat-val">${{m.dead_pct}}%</div></div>
      <div class="stat-cell"><div class="stat-label">Density</div><div class="stat-val">${{m.density}}</div></div>
      <div class="stat-cell"><div class="stat-label">Recon</div><div class="stat-val">${{m.recon}}</div></div>
      <div class="stat-cell"><div class="stat-label">Sparsity</div><div class="stat-val">${{m.sparsity}}</div></div>
      <div class="stat-cell"><div class="stat-label">Total Loss</div><div class="stat-val">${{m.total}}</div></div>
    </div>
    <div class="cat-counts">
      <span class="cat-pill" style="background:#5cb85c22;color:#5cb85c">✓ ${{m.confirmed}} conf</span>
      <span class="cat-pill" style="background:#e6a81722;color:#e6a817">~ ${{m.partial}} partial</span>
      <span class="cat-pill" style="background:#6c8ebf22;color:#6c8ebf">? ${{m.novel}} novel</span>
      <span class="cat-pill" style="background:#55555522;color:#888">✗ ${{m.dead}} dead</span>
    </div>`;
}}

function switchLayer(variant) {{
  currentVariant = variant;
  FEATURES = ALL_FEATURES[variant] || [];
  const n = FEATURES.length;
  document.getElementById('variant-label').textContent = variant + ' · ' + n + ' features';
  document.getElementById('cat-filter').value = 'all';
  document.getElementById('detail').innerHTML = '<p style="color:#555;margin-top:40px;text-align:center">Select a feature →</p>';
  updateStats(variant);
  applyFilter();
}}

function buildSidebar(items) {{
  const sorted = [...items].sort((a, b) => {{
    const catDiff = (CAT_ORDER[a.category] ?? 9) - (CAT_ORDER[b.category] ?? 9);
    if (catDiff !== 0) return catDiff;
    return Math.abs(b.best_lift) - Math.abs(a.best_lift);
  }});
  const list = document.getElementById('feature-list');
  list.innerHTML = '';
  sorted.forEach(feat => {{
    const item = document.createElement('div');
    item.className = 'feat-item';
    item.dataset.fidx = feat.f;
    const catColor = CAT_COLORS[feat.category] || '#555';
    const axLabel  = feat.best_axis ? feat.best_axis.replace(/_/g,' ') : '—';
    const sym      = feat.category === 'confirms_axis' ? '✓' : feat.category === 'partial_overlap' ? '~' : feat.category === 'dead' ? '✗' : '?';
    item.innerHTML = `
      <div class="feat-title">F${{feat.f}}<span class="cat-badge" style="background:${{catColor}}22;color:${{catColor}}">${{sym}}</span></div>
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
  const maxLift  = Math.max(...Object.values(feat.lifts).map(Math.abs), 0.01);
  const maxR     = Math.max(...Object.values(feat.rs).map(Math.abs), 0.01);
  const absMax   = Math.max(maxLift, maxR);

  let liftHtml = '';
  AXIS_NAMES.forEach(ax => {{
    const lift   = feat.lifts[ax] || 0;
    const r      = feat.rs[ax]    || 0;
    const color  = AXIS_COLORS[ax] || '#6c8ebf';
    const isBest = ax === feat.best_axis;
    liftHtml += `
      <div class="axis-row">
        <div class="axis-row-label ${{isBest ? 'best' : ''}}">${{ax.replace(/_/g,' ')}}</div>
        <div class="metric-row">
          <div class="metric-tag">lift</div>
          <div class="bar-bg"><div class="bar-fill" style="width:${{Math.abs(lift)/absMax*100}}%;background:${{color}};opacity:${{isBest?1:0.55}}"></div></div>
          <div class="bar-val">${{lift>=0?'+':''}}${{lift.toFixed(3)}}</div>
        </div>
        <div class="metric-row">
          <div class="metric-tag">r</div>
          <div class="bar-bg"><div class="bar-fill" style="width:${{Math.abs(r)/absMax*100}}%;background:${{color}};opacity:${{isBest?0.65:0.3}}"></div></div>
          <div class="bar-val">${{r>=0?'+':''}}${{r.toFixed(3)}}</div>
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
          <div class="example-meta">post ${{ex.pid}} &nbsp;·&nbsp; ${{ex.topic}} &nbsp;·&nbsp; activation: <span class="activation-badge">${{ex.activation.toFixed(4)}}</span></div>
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
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

switchLayer(currentVariant);
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("Loading dataset and labels…")
    dataset = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labels  = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)
    print(f"  {len(dataset)} posts, {len(labels)} labeled")

    all_features: dict[str, list[dict]] = {}
    variant_meta: dict[str, dict] = {}
    for variant in VARIANTS:
        print(f"Building {variant}…")
        feats = build_features(variant, dataset, labels)
        confirmed = sum(1 for f in feats if f["category"] == "confirms_axis")
        partial   = sum(1 for f in feats if f["category"] == "partial_overlap")
        novel     = sum(1 for f in feats if f["category"] == "novel_candidate")
        dead      = sum(1 for f in feats if f["category"] == "dead")
        print(f"  {len(feats)} features: {confirmed} confirmed, {partial} partial, {novel} novel, {dead} dead")
        all_features[variant] = feats
        variant_meta[variant] = build_variant_meta(feats, variant)

    print("Rendering HTML…")
    html = render_html(all_features, variant_meta)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Saved → {OUT_FILE}")
