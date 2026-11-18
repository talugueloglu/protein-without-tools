#!/usr/bin/env python3
"""Generate an interactive standalone HTML 3Dmol.js viewer for Stage 1 folds.

Embeds crystal Cα structures alongside reconstructed / ESMFold predicted 3D traces
so you can rotate, zoom, inspect superposition overlays directly in the browser,
embed the widget in your blog post, or record a clean screen-capture GIF for Twitter/X.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Fable-Protein-Bench — Stage 1 Interactive 3D Superposition Explorer</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; }
    header { padding: 18px 28px; border-bottom: 1px solid #1e293b; display: flex; justify-content: space-between; align-items: center; }
    h1 { font-size: 18px; margin: 0; font-weight: 600; }
    .controls { display: flex; gap: 12px; align-items: center; }
    select { background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 7px 12px; font-size: 13px; }
    .container { display: grid; grid-template-columns: 1fr 340px; height: calc(100vh - 65px); }
    #viewer { width: 100%; height: 100%; position: relative; }
    aside { padding: 20px; border-left: 1px solid #1e293b; background: #0b1120; overflow-y: auto; font-size: 13px; line-height: 1.55; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; background: #1e293b; color: #38bdf8; }
    .metric-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1e293b; }
    .legend { margin-top: 16px; padding: 10px; background: #1e293b; border-radius: 6px; font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>Fable-Protein-Bench: Stage 1 Structure Superposition (Crystal Grey vs. Model Rainbow)</h1>
    <div class="controls">
      <label>Target:</label>
      <select id="targetSelect"></select>
      <label>Model:</label>
      <select id="modelSelect"></select>
    </div>
  </header>
  <div class="container">
    <div id="viewer"></div>
    <aside id="metaPanel"></aside>
  </div>
  <script>
    const PAYLOAD = __PAYLOAD_JSON__;
    const viewer = $3Dmol.createViewer("viewer", { backgroundColor: "#0f172a" });
    const tSel = document.getElementById("targetSelect");
    const mSel = document.getElementById("modelSelect");
    const panel = document.getElementById("metaPanel");

    PAYLOAD.targets.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t.target_id;
      opt.textContent = `${t.target_id} (${t.pdb_id}:${t.chain_id}, L=${t.length}, ${t.temporal_split})`;
      tSel.appendChild(opt);
    });

    PAYLOAD.models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      mSel.appendChild(opt);
    });

    function renderPair() {
      const tid = tSel.value;
      const mid = mSel.value;
      const tMeta = PAYLOAD.targets.find(x => x.target_id === tid);
      const gtPdb = PAYLOAD.gt_pdbs[tid];
      const predKey = `${mid}__${tid}`;
      const predPdb = PAYLOAD.pred_pdbs[predKey];
      const metrics = (PAYLOAD.metrics[predKey] || {});

      viewer.removeAllModels();
      if (gtPdb) {
        const m0 = viewer.addModel(gtPdb, "pdb");
        viewer.setStyle({ model: m0 }, { trace: { color: "#94a3b8", radius: 0.32, opacity: 0.75 } });
      }
      if (predPdb) {
        const m1 = viewer.addModel(predPdb, "pdb");
        viewer.setStyle({ model: m1 }, { trace: { colorscheme: "roygb", radius: 0.45 } });
      }
      viewer.zoomTo();
      viewer.render();

      panel.innerHTML = `
        <div><span class="badge">${tMeta.temporal_split}</span> <span class="badge">${tMeta.fold_class}</span></div>
        <h3 style="margin:10px 0 4px 0;">${tMeta.pdb_id} (Chain ${tMeta.chain_id})</h3>
        <div style="color:#94a3b8;margin-bottom:14px;">${tMeta.note || tMeta.description}</div>
        <div class="metric-row"><span>Length (AA)</span><strong>${tMeta.length}</strong></div>
        <div class="metric-row"><span>Release Date</span><strong>${tMeta.release_date || "pre-2025"}</strong></div>
        <div class="metric-row"><span>TM-score (SO(3))</span><strong>${metrics.tm_score !== undefined ? metrics.tm_score.toFixed(3) : "N/A"}</strong></div>
        <div class="metric-row"><span>Global Cα RMSD</span><strong>${metrics.rmsd_ca !== undefined ? metrics.rmsd_ca.toFixed(2) + " Å" : "N/A"}</strong></div>
        <div class="metric-row"><span>Cα-lDDT</span><strong>${metrics.lddt_ca !== undefined ? metrics.lddt_ca.toFixed(3) : "N/A"}</strong></div>
        <div class="metric-row"><span>Long-Range P@L (|i-j|≥24)</span><strong>${metrics.p_at_l_sep24 !== undefined ? (100 * metrics.p_at_l_sep24).toFixed(1) + "%" : "N/A"}</strong></div>
        <div class="metric-row"><span>Secondary Struct Q3</span><strong>${metrics.q3_accuracy !== undefined ? (100 * metrics.q3_accuracy).toFixed(1) + "%" : "N/A"}</strong></div>
        <div class="legend">
          <strong>Color Key:</strong><br/>
          • <strong>Slate Grey tube:</strong> Experimental Crystal Cα trace<br/>
          • <strong>Rainbow (N=Blue → C=Red):</strong> Reconstructed Model Cα trace
        </div>
      `;
    }

    tSel.addEventListener("change", renderPair);
    mSel.addEventListener("change", renderPair);
    if (PAYLOAD.targets.length && PAYLOAD.models.length) renderPair();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive 3Dmol.js Stage 1 HTML explorer.")
    parser.add_argument("--data-dir", default="data/stage1_benchmark")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--out-html", default="outputs/figures/stage1_interactive_3d.html")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    res_dir = Path(args.results_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((res_dir / "stage1_evaluation_report.json").read_text(encoding="utf-8"))

    gt_pdbs = {}
    for t in manifest["targets"]:
        tid = t["target_id"]
        p = data_dir / "ground_truth" / f"{tid}_ca.pdb"
        if p.exists():
            gt_pdbs[tid] = p.read_text(encoding="utf-8")

    pred_pdbs = {}
    metrics = {}
    models = []
    recon_dir = res_dir / "reconstructed_pdbs"
    for entry in report.get("stage1a_forward_folding", []):
        m = entry["model"]
        models.append(m)
        for pt in entry.get("per_target", []):
            key = f"{m}__{pt['target_id']}"
            metrics[key] = pt
            pf = recon_dir / f"{key}.pdb"
            if pf.exists():
                pred_pdbs[key] = pf.read_text(encoding="utf-8")

    payload = {
        "targets": manifest["targets"],
        "models": models,
        "gt_pdbs": gt_pdbs,
        "pred_pdbs": pred_pdbs,
        "metrics": metrics,
    }
    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", json.dumps(payload))
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved interactive 3Dmol.js Stage 1 explorer -> {out_path}")


if __name__ == "__main__":
    main()
