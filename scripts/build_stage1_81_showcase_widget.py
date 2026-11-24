#!/usr/bin/env python3
"""Build a compact (<480px) interactive HTML showcase widget for all N=81 Stage 1 targets.

Combines:
  - Interactive `PDB SeqID %` vs `AlphaFold DB SeqID %` scatter plot (click/hover any of 81 targets)
  - Stratum filters & quick-jump spotlight buttons (`6MRR`, `37NT`, `47WY`, `23WS`, `36BS`, natural 2026)
  - Live 3D rotatable Cα viewer superposing the Kabsch-aligned Chiral L-BFGS reconstructed fold
    (`Oracle_TrueContacts`) directly onto the experimental crystal backbone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from protein_without_tools.tmscore import evaluate_3d_superposition, kabsch_superpose


def parse_ca_from_pdb(pdb_path: Path) -> np.ndarray:
    coords = []
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/stage1_benchmark")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--out-html", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    res_dir = Path(args.results_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))

    # Map oracle TM/RMSD if report exists
    oracle_metrics: Dict[str, Dict[str, float]] = {}
    report_path = res_dir / "stage1_evaluation_report.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        for fs in rep.get("stage1a_forward_folding", []):
            if fs["model"] == "Oracle_TrueContacts":
                for pt in fs.get("per_target", []):
                    oracle_metrics[pt["target_id"]] = {
                        "tm": round(float(pt["tm_score"]), 3),
                        "rmsd": round(float(pt["rmsd_ca"]), 2),
                    }

    items: List[Dict[str, Any]] = []
    for rec in manifest["targets"]:
        tid = rec["target_id"]
        npz = np.load(data_dir / "ground_truth" / f"{tid}.npz", allow_pickle=False)
        ca_true = npz["coords_ca"].astype(np.float64)
        q3 = str(npz["q3"])
        center = ca_true.mean(axis=0)
        ca_centered = ca_true - center

        # Superpose Oracle reconstructed Cα onto centered crystal Cα if available
        recon_path = res_dir / "reconstructed_pdbs" / f"Oracle_TrueContacts__{tid}.pdb"
        recon_aligned_list = None
        tm_val = oracle_metrics.get(tid, {}).get("tm", 0.0)
        rmsd_val = oracle_metrics.get(tid, {}).get("rmsd", 0.0)
        if recon_path.exists():
            ca_rec = parse_ca_from_pdb(recon_path)
            if ca_rec.shape == ca_true.shape:
                sup = evaluate_3d_superposition(ca_rec, ca_centered)
                if tm_val == 0.0:
                    tm_val = round(float(sup.tm_score), 3)
                    rmsd_val = round(float(sup.rmsd_global), 2)
                recon_aligned_list = np.round(sup.aligned_pred_ca, 2).tolist()

        items.append(
            {
                "id": tid,
                "pdb": rec["pdb_id"],
                "chain": rec["chain_id"],
                "len": rec["length"],
                "stratum": rec["stratum"],
                "split": rec["temporal_split"],
                "fold": rec["fold_class"],
                "date": rec["release_date"],
                "desc": (rec.get("description") or rec.get("note") or "")[:54],
                "pdb_seqid": round(float(rec.get("pdb_precutoff_seqid_max", 0.0)), 1),
                "pdb_fs": round(float(rec.get("pdb100_precutoff_foldseek_prob", 0.0)), 2),
                "afdb_seqid": round(float(rec.get("afdb_max_seqid", 0.0)), 1),
                "afdb_fs": round(float(rec.get("afdb_foldseek_prob", 0.0)), 2),
                "afdb_hit": rec.get("afdb_closest_predicted_struct") or "None",
                "tm": tm_val,
                "rmsd": rmsd_val,
                "q3": q3,
                "ca": np.round(ca_centered, 2).tolist(),
                "recon": recon_aligned_list,
            }
        )

    payload_json = json.dumps(items, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-3 select-none">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-3.5 shadow-sm">
    <!-- Top Header & Filter Bar -->
    <div class="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-[var(--border)]">
      <div>
        <div class="flex items-center gap-2">
          <span class="font-semibold text-sm">Stage 1 Interactive Dataset &amp; 3D Reconstructor Explorer</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-[var(--secondary)] text-[var(--secondary-foreground)] font-medium">N = 81 Crystal Targets</span>
        </div>
      </div>
      <div class="flex flex-wrap items-center gap-1" id="filterBar">
        <button data-filter="all" class="flt-btn text-xs px-2 py-1 rounded-md border border-[var(--border)] font-medium">All (81)</button>
        <button data-filter="post_cutoff_de_novo" class="flt-btn text-xs px-2 py-1 rounded-md border border-[var(--border)]">2026 De Novo (30)</button>
        <button data-filter="post_cutoff_pdb_novel" class="flt-btn text-xs px-2 py-1 rounded-md border border-[var(--border)]">PDB-Novel / AFDB (21)</button>
        <button data-filter="post_cutoff_homolog_gradient" class="flt-btn text-xs px-2 py-1 rounded-md border border-[var(--border)]">Homologs (15)</button>
        <button data-filter="pre_cutoff_control" class="flt-btn text-xs px-2 py-1 rounded-md border border-[var(--border)]">Pre-Cutoff (15)</button>
      </div>
    </div>

    <!-- Main 2-Column Interactive Grid -->
    <div class="grid grid-cols-12 gap-3 pt-2.5 items-stretch">
      <!-- Left: PDB vs AFDB Scatter Plot -->
      <div class="col-span-5 flex flex-col justify-between border border-[var(--border)] rounded-lg p-2.5 bg-[var(--background)]">
        <div class="flex items-center justify-between mb-1">
          <span class="text-xs font-semibold">Pre-Cutoff PDB vs. AlphaFold DB SeqID</span>
          <span class="text-xs text-[var(--muted-foreground)]">Click any dot</span>
        </div>
        <div class="relative flex-1 flex items-center justify-center">
          <canvas id="scatterCanvas" width="290" height="235" class="cursor-pointer w-full h-[235px]"></canvas>
        </div>
        <!-- Quick-jump Spotlight Pills -->
        <div class="pt-1.5 border-t border-[var(--border)] flex flex-wrap items-center gap-1">
          <span class="text-xs text-[var(--muted-foreground)] mr-0.5">Spotlights:</span>
          <button data-pdb="47WY" class="spot-btn text-xs px-1.5 py-0.5 rounded border border-[var(--border)] hover:opacity-80">47WY (De Novo)</button>
          <button data-pdb="23WS" class="spot-btn text-xs px-1.5 py-0.5 rounded border border-[var(--border)] hover:opacity-80">23WS (Binder)</button>
          <button data-pdb="6MRR" class="spot-btn text-xs px-1.5 py-0.5 rounded border border-[var(--border)] hover:opacity-80">6MRR (@sokrypton)</button>
          <button data-pdb="37NT" class="spot-btn text-xs px-1.5 py-0.5 rounded border border-[var(--border)] hover:opacity-80">37NT (@WUFang)</button>
        </div>
      </div>

      <!-- Right: 3D Structure + Reconstructor Overlay -->
      <div class="col-span-7 flex flex-col justify-between border border-[var(--border)] rounded-lg p-2.5 bg-[var(--background)]">
        <div class="flex flex-wrap items-start justify-between gap-2 pb-1.5 border-b border-[var(--border)]">
          <div>
            <div class="flex items-center gap-1.5">
              <span id="selTitle" class="font-semibold text-sm">47WY</span>
              <span id="selBadge" class="text-xs px-1.5 py-0.5 rounded font-medium"></span>
              <span id="selMeta" class="text-xs text-[var(--muted-foreground)]"></span>
            </div>
            <div id="selDesc" class="text-xs text-[var(--muted-foreground)] truncate max-w-[360px]"></div>
          </div>
          <label class="flex items-center gap-1.5 text-xs cursor-pointer px-2 py-1 rounded border border-[var(--border)] bg-[var(--card)]">
            <input type="checkbox" id="toggleRecon" checked class="accent-cyan-500" />
            <span>Overlay L-BFGS Fold</span>
          </label>
        </div>

        <!-- Live 3D Canvas -->
        <div class="relative flex items-center justify-center my-1">
          <canvas id="molCanvas" width="380" height="205" class="cursor-grab active:cursor-grabbing w-full h-[205px]"></canvas>
          <div class="absolute bottom-1 left-1.5 flex items-center gap-2 text-xs text-[var(--muted-foreground)] pointer-events-none">
            <span><span style="color:#ef4444">●</span> α-Helix</span>
            <span><span style="color:#14b8a6">●</span> β-Strand</span>
            <span id="reconLegend"><span style="color:#38bdf8">╌</span> L-BFGS Recon</span>
          </div>
          <div class="absolute bottom-1 right-1.5 text-xs text-[var(--muted-foreground)] pointer-events-none">Drag to rotate · Scroll zoom</div>
        </div>

        <!-- Bottom Quantitative Similarity & Calibration Metrics -->
        <div class="grid grid-cols-4 gap-1.5 pt-1.5 border-t border-[var(--border)] text-center">
          <div class="rounded bg-[var(--card)] px-1.5 py-1 border border-[var(--border)]">
            <div class="text-xs text-[var(--muted-foreground)]">PDB SeqID / FS</div>
            <div id="mPdb" class="text-xs font-semibold">0.0% / 0.12</div>
          </div>
          <div class="rounded bg-[var(--card)] px-1.5 py-1 border border-[var(--border)]">
            <div class="text-xs text-[var(--muted-foreground)]">AFDB SeqID / FS</div>
            <div id="mAfdb" class="text-xs font-semibold">0.0% / 0.45</div>
          </div>
          <div class="rounded bg-[var(--card)] px-1.5 py-1 border border-[var(--border)]">
            <div class="text-xs text-[var(--muted-foreground)]">Oracle TM-score</div>
            <div id="mTm" class="text-xs font-semibold">0.812</div>
          </div>
          <div class="rounded bg-[var(--card)] px-1.5 py-1 border border-[var(--border)]">
            <div class="text-xs text-[var(--muted-foreground)]">Recon Cα RMSD</div>
            <div id="mRmsd" class="text-xs font-semibold">2.14 Å</div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const TARGETS = {payload_json};

const STRATA_META = {{
  post_cutoff_de_novo: {{ label: "2026 De Novo", color: "#ec4899" }},
  post_cutoff_pdb_novel: {{ label: "PDB-Novel / AFDB", color: "#10b981" }},
  post_cutoff_homolog_gradient: {{ label: "2026 Homolog", color: "#f59e0b" }},
  pre_cutoff_control: {{ label: "Pre-Cutoff", color: "#3b82f6" }}
}};

let activeFilter = "all";
let selectedIdx = TARGETS.findIndex(t => t.pdb === "47WY");
if (selectedIdx < 0) selectedIdx = 0;

let rotX = -0.35, rotY = 0.65, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0, autoSpin = true;

const scatterCanvas = document.getElementById("scatterCanvas");
const sCtx = scatterCanvas.getContext("2d");
const molCanvas = document.getElementById("molCanvas");
const mCtx = molCanvas.getContext("2d");

function isLightMode() {{
  return document.documentElement.classList.contains("light");
}}

function syncFilterButtons() {{
  document.querySelectorAll(".flt-btn").forEach(btn => {{
    const on = btn.dataset.filter === activeFilter;
    btn.style.background = on ? "var(--primary)" : "var(--card)";
    btn.style.color = on ? "var(--primary-foreground)" : "var(--foreground)";
  }});
}}

function drawScatter() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = scatterCanvas.getBoundingClientRect();
  const W = rect.width || 290, H = rect.height || 235;
  scatterCanvas.width = W * dpr;
  scatterCanvas.height = H * dpr;
  sCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const light = isLightMode();
  const axisCol = light ? "#64748b" : "#94a3b8";
  const gridCol = light ? "rgba(100,116,139,0.18)" : "rgba(148,163,184,0.18)";
  const pad = {{ l: 34, r: 12, t: 10, b: 28 }};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  sCtx.clearRect(0, 0, W, H);

  const xOf = v => pad.l + (Math.max(0, Math.min(100, v)) / 100) * pw;
  const yOf = v => pad.t + ph - (Math.max(0, Math.min(100, v)) / 100) * ph;

  // Highlight <30% zero-prior zone
  sCtx.fillStyle = light ? "rgba(236,72,153,0.08)" : "rgba(236,72,153,0.13)";
  sCtx.fillRect(xOf(0), yOf(30), xOf(30) - xOf(0), yOf(0) - yOf(30));

  // Grid & 30% homology thresholds
  sCtx.strokeStyle = gridCol;
  sCtx.lineWidth = 1;
  [0, 30, 60, 100].forEach(tick => {{
    sCtx.beginPath();
    sCtx.moveTo(xOf(tick), pad.t); sCtx.lineTo(xOf(tick), pad.t + ph);
    sCtx.moveTo(pad.l, yOf(tick)); sCtx.lineTo(pad.l + pw, yOf(tick));
    sCtx.stroke();
  }});

  sCtx.setLineDash([4, 3]);
  sCtx.strokeStyle = light ? "rgba(236,72,153,0.55)" : "rgba(244,114,182,0.55)";
  sCtx.beginPath();
  sCtx.moveTo(xOf(30), pad.t); sCtx.lineTo(xOf(30), pad.t + ph);
  sCtx.moveTo(pad.l, yOf(30)); sCtx.lineTo(pad.l + pw, yOf(30));
  sCtx.stroke();
  sCtx.setLineDash([]);

  // Axes labels
  sCtx.fillStyle = axisCol;
  sCtx.font = "12px sans-serif";
  sCtx.textAlign = "center";
  [0, 30, 60, 100].forEach(tick => {{
    sCtx.fillText(tick + "%", xOf(tick), H - 14);
  }});
  sCtx.fillText("Pre-Cutoff PDB SeqID", pad.l + pw / 2, H - 2);

  sCtx.textAlign = "right";
  [0, 30, 60, 100].forEach(tick => {{
    sCtx.fillText(tick + "", pad.l - 4, yOf(tick) + 4);
  }});

  // Plot points (deterministic tiny jitter on exact 0,0 overlaps so all 81 are clickable)
  TARGETS.forEach((t, idx) => {{
    const pass = activeFilter === "all" || t.stratum === activeFilter;
    const jx = ((idx * 17) % 7 - 3) * (t.pdb_seqid === 0 ? 0.55 : 0.2);
    const jy = ((idx * 29) % 7 - 3) * (t.afdb_seqid === 0 ? 0.55 : 0.2);
    const cx = xOf(t.pdb_seqid) + jx;
    const cy = yOf(t.afdb_seqid) + jy;
    t._px = cx; t._py = cy;

    const col = (STRATA_META[t.stratum] || {{ color: "#888" }}).color;
    sCtx.globalAlpha = pass ? (idx === selectedIdx ? 1.0 : 0.82) : 0.14;
    sCtx.beginPath();
    sCtx.arc(cx, cy, idx === selectedIdx ? 6.2 : 4.2, 0, Math.PI * 2);
    sCtx.fillStyle = col;
    sCtx.fill();
    if (idx === selectedIdx) {{
      sCtx.lineWidth = 2;
      sCtx.strokeStyle = light ? "#0f172a" : "#ffffff";
      sCtx.stroke();
    }}
  }});
  sCtx.globalAlpha = 1.0;
}}

scatterCanvas.addEventListener("click", e => {{
  const rect = scatterCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = -1, bestD = 1e9;
  TARGETS.forEach((t, idx) => {{
    if (activeFilter !== "all" && t.stratum !== activeFilter) return;
    const d = Math.hypot((t._px || 0) - mx, (t._py || 0) - my);
    if (d < bestD && d < 18) {{ bestD = d; best = idx; }}
  }});
  if (best >= 0) {{
    selectedIdx = best;
    updateSelectionUI();
    drawScatter();
  }}
}});

function rotatePt(p, rx, ry) {{
  const x = p[0], y = p[1], z = p[2];
  const x1 = x * Math.cos(ry) + z * Math.sin(ry);
  const z1 = -x * Math.sin(ry) + z * Math.cos(ry);
  const y2 = y * Math.cos(rx) - z1 * Math.sin(rx);
  const z2 = y * Math.sin(rx) + z1 * Math.cos(rx);
  return [x1, y2, z2];
}}

function drawMolecule() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = molCanvas.getBoundingClientRect();
  const W = rect.width || 380, H = rect.height || 205;
  molCanvas.width = W * dpr;
  molCanvas.height = H * dpr;
  mCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  mCtx.clearRect(0, 0, W, H);

  const t = TARGETS[selectedIdx];
  if (!t) return;
  const light = isLightMode();
  const ca = t.ca;
  const maxR = Math.max(12, ...ca.map(p => Math.hypot(p[0], p[1], p[2])));
  const scale = (Math.min(W, H) * 0.41 * zoom) / maxR;

  const proj = ca.map(p => {{
    const r = rotatePt(p, rotX, rotY);
    const persp = 1 + r[2] / (maxR * 3.5);
    return [W / 2 + r[0] * scale * persp, H / 2 - r[1] * scale * persp, r[2]];
  }});

  // Draw crystal Cα segments depth-sorted
  const segs = [];
  for (let i = 0; i < proj.length - 1; i++) {{
    segs.push({{ i, z: 0.5 * (proj[i][2] + proj[i + 1][2]), type: "cryst" }});
  }}

  const showRecon = document.getElementById("toggleRecon").checked && t.recon && t.recon.length === ca.length;
  let rProj = null;
  if (showRecon) {{
    rProj = t.recon.map(p => {{
      const r = rotatePt(p, rotX, rotY);
      const persp = 1 + r[2] / (maxR * 3.5);
      return [W / 2 + r[0] * scale * persp, H / 2 - r[1] * scale * persp, r[2]];
    }});
    for (let i = 0; i < rProj.length - 1; i++) {{
      segs.push({{ i, z: 0.5 * (rProj[i][2] + rProj[i + 1][2]) - 0.1, type: "recon" }});
    }}
  }}

  segs.sort((a, b) => a.z - b.z);

  for (const s of segs) {{
    const i = s.i;
    if (s.type === "cryst") {{
      const ss = t.q3[i] || "C";
      let col = light ? "#64748b" : "#94a3b8";
      let lw = ss === "C" ? 2.4 : 4.2;
      if (ss === "H") col = "#ef4444";
      else if (ss === "E") col = "#14b8a6";
      mCtx.strokeStyle = col;
      mCtx.lineWidth = lw;
      mCtx.lineCap = "round";
      mCtx.setLineDash([]);
      mCtx.beginPath();
      mCtx.moveTo(proj[i][0], proj[i][1]);
      mCtx.lineTo(proj[i + 1][0], proj[i + 1][1]);
      mCtx.stroke();
    }} else if (rProj) {{
      mCtx.strokeStyle = light ? "rgba(2,132,199,0.82)" : "rgba(56,189,248,0.85)";
      mCtx.lineWidth = 1.9;
      mCtx.setLineDash([3, 2.5]);
      mCtx.beginPath();
      mCtx.moveTo(rProj[i][0], rProj[i][1]);
      mCtx.lineTo(rProj[i + 1][0], rProj[i + 1][1]);
      mCtx.stroke();
    }}
  }}
  mCtx.setLineDash([]);
}}

function updateSelectionUI() {{
  const t = TARGETS[selectedIdx];
  const sm = STRATA_META[t.stratum] || {{ label: t.stratum, color: "#64748b" }};
  document.getElementById("selTitle").textContent = `${{t.pdb}}:${{t.chain}} (${{t.id}})`;
  const badge = document.getElementById("selBadge");
  badge.textContent = sm.label;
  badge.style.background = sm.color + "22";
  badge.style.color = sm.color;
  document.getElementById("selMeta").textContent = `${{t.len}} AA · ${{t.fold}} · ${{t.date}}`;
  document.getElementById("selDesc").textContent = t.desc || "Experimental crystal monomer";
  document.getElementById("mPdb").textContent = `${{t.pdb_seqid}}% / ${{t.pdb_fs}}`;
  document.getElementById("mAfdb").textContent = `${{t.afdb_seqid}}% / ${{t.afdb_fs}}`;
  document.getElementById("mTm").textContent = t.tm ? t.tm.toFixed(3) : "—";
  document.getElementById("mRmsd").textContent = t.rmsd ? `${{t.rmsd.toFixed(2)}} Å` : "—";
  document.getElementById("reconLegend").style.opacity = document.getElementById("toggleRecon").checked ? "1" : "0.3";
}}

document.querySelectorAll(".flt-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    activeFilter = btn.dataset.filter;
    if (activeFilter !== "all" && TARGETS[selectedIdx].stratum !== activeFilter) {{
      const next = TARGETS.findIndex(t => t.stratum === activeFilter);
      if (next >= 0) selectedIdx = next;
    }}
    syncFilterButtons();
    updateSelectionUI();
    drawScatter();
  }});
}});

document.querySelectorAll(".spot-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    const pdb = btn.dataset.pdb;
    const idx = TARGETS.findIndex(t => t.pdb === pdb);
    if (idx >= 0) {{
      activeFilter = "all";
      selectedIdx = idx;
      syncFilterButtons();
      updateSelectionUI();
      drawScatter();
    }}
  }});
}});

document.getElementById("toggleRecon").addEventListener("change", () => {{
  updateSelectionUI();
  drawMolecule();
}});

molCanvas.addEventListener("mousedown", e => {{ dragging = true; autoSpin = false; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener("mouseup", () => {{ dragging = false; }});
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  rotY += (e.clientX - lastX) * 0.012;
  rotX += (e.clientY - lastY) * 0.012;
  lastX = e.clientX; lastY = e.clientY;
  drawMolecule();
}});
molCanvas.addEventListener("wheel", e => {{
  e.preventDefault();
  zoom = Math.max(0.55, Math.min(2.2, zoom * (e.deltaY < 0 ? 1.08 : 0.92)));
  drawMolecule();
}}, {{ passive: false }});

function tick() {{
  if (autoSpin && !dragging) {{
    rotY += 0.008;
    drawMolecule();
  }}
  requestAnimationFrame(tick);
}}

window.addEventListener("resize", () => {{ drawScatter(); drawMolecule(); }});
syncFilterButtons();
updateSelectionUI();
drawScatter();
drawMolecule();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
    Path(args.out_html).write_text(html, encoding="utf-8")
    print(f"Wrote interactive showcase HTML ({len(html) / 1024:.1f} KB) -> {args.out_html}")


if __name__ == "__main__":
    main()
