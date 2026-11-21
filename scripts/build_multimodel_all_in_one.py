#!/usr/bin/env python3
"""Compile multi-model per-question records & render single-view 6-panel comparison (no tab switching).

Shows `GPT-6.1 Sol`, `GPT-6.2 Sol`, and `GPT-6.2 Astra` side-by-side inside every difficulty tier
across all 5 tests (`1, 3, 4, 5, 6`), plus Panel 6 overall multi-model scorecard.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.stage2a_homology import parse_remote_homology_response
from fable_bench.stage2b_cafa import (
    CAFATarget,
    compute_cafa_fmax,
    compute_term_information_content,
    parse_cafa_response,
    parse_go_obo,
)

MODEL_ORDER = [
    ("gpt-6.1-sol_think4096", "GPT-6.1 Sol", "#0ea5e9"),
    ("gpt-6.2-sol_think4096", "GPT-6.2 Sol", "#6366f1"),
    ("gpt-6.2-astra_think4096", "GPT-6.2 Astra", "#e11d48"),
]
RE_REAS = re.compile(r"openai_reasoning_tokens=(\d+)")


def _load_pred_meta(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    if not jsonl_path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for ln in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        tid = str(obj.get("target_id") or obj.get("pair_id") or "")
        m = RE_REAS.search(str(obj.get("thinking_text") or ""))
        obj["reasoning_tokens"] = int(m.group(1)) if m else 0
        out[tid] = obj
    return out


def compile_multimodel_records() -> List[Dict[str, Any]]:
    rdir = REPO / "outputs" / "results"
    pdir = REPO / "outputs" / "predictions"
    out_rows: List[Dict[str, Any]] = []

    # Preload shared manifests & GO DAG once
    m1_meta = {
        t["target_id"]: t
        for t in json.loads((REPO / "data/stage1_benchmark/manifest.json").read_text(encoding="utf-8"))["targets"]
    }
    s1_by_m = {
        x["model"]: x
        for x in json.loads((rdir / "stage1_evaluation_report.json").read_text(encoding="utf-8"))["stage1a_forward_folding"]
    }
    m3_pairs = json.loads((REPO / "data/stage2a_homology/manifest.json").read_text(encoding="utf-8"))["pairs"]

    s2b_dir = REPO / "data/stage2b_cafa"
    dag = parse_go_obo(str(s2b_dir / "go_subdag.obo"))
    m4 = json.loads((s2b_dir / "manifest.json").read_text(encoding="utf-8"))["targets"]
    ref_bank = json.loads((s2b_dir / "reference_swissprot_go.json").read_text(encoding="utf-8"))
    ref_prop = [dag.propagate_set(r["go_terms"]) for r in ref_bank]
    ic_map = compute_term_information_content(dag, ref_prop)
    ref_kmers = [{r["sequence"][i : i + 3] for i in range(max(0, len(r["sequence"]) - 2))} for r in ref_bank]
    t4_sims: Dict[str, float] = {}
    for t in m4:
        qs = t["sequence"]
        qks = {qs[i : i + 3] for i in range(max(0, len(qs) - 2))}
        qd = math.sqrt(max(1, len(qks)))
        t4_sims[t["target_id"]] = max(len(qks & rk) / (qd * math.sqrt(max(1, len(rk)))) for rk in ref_kmers)
    med_sim4 = float(np.median(list(t4_sims.values())))

    m5_meta = {
        t["target_id"]: t
        for t in json.loads((REPO / "data/alanine_scan_benchmark/manifest.json").read_text(encoding="utf-8"))["targets"]
    }
    s5_by_m = {
        x["model"]: x
        for x in json.loads((rdir / "stage3a_alanine_report.json").read_text(encoding="utf-8"))["stage3a_alanine_scan"]
    }

    m6_meta = {
        t["target_id"]: t
        for t in json.loads((REPO / "data/binding_alanine_benchmark/manifest.json").read_text(encoding="utf-8"))["targets"]
    }
    s6_by_m = {
        x["model"]: x
        for x in json.loads((rdir / "live_stage6_binding_report.json").read_text(encoding="utf-8"))["test6_binding_alanine_ranking"]
    }

    t1_tier_map = {
        "pre_cutoff_control": (1, "Known"),
        "post_cutoff_de_novo": (2, "DeNovo"),
        "post_cutoff_homolog_gradient": (3, "Homol"),
        "post_cutoff_pdb_novel": (4, "PDBNov"),
        "post_cutoff_struct_novel": (5, "TrueNov"),
    }
    t3_tier_map = {
        "1_seq_easy": (1, "R1 Easy"),
        "2_foldseek_3di": (2, "R2 3Di"),
        "3_deep_tea_zone": (3, "R3 TEA"),
        "4_hard_negative_decoy": (4, "Decoys"),
    }

    for mtag, mshort, _mcol in MODEL_ORDER:
        # Test 1
        if mtag in s1_by_m:
            pred1 = _load_pred_meta(pdir / f"{mtag}_stage1a.jsonl")
            for g in s1_by_m[mtag]["per_target"]:
                tid = g["target_id"]
                mt = m1_meta[tid]
                do, dt = t1_tier_map.get(mt.get("stratum", ""), (4, "PDBNov"))
                sp = "Pre-Cutoff" if mt["temporal_split"] == "pre_cutoff" else "Post-Cutoff"
                out_rows.append(
                    {
                        "m": mshort,
                        "t": 1,
                        "id": tid,
                        "nm": f"PDB {mt['pdb_id']}:{mt['chain_id']} ({mt.get('description','')[:26]})",
                        "sp": sp,
                        "do": do,
                        "dt": dt,
                        "sc": round(float(g["tm_score"]), 3),
                        "rt": pred1.get(tid, {}).get("reasoning_tokens", 0),
                        "ab": 0,
                    }
                )

        # Test 3
        pred3 = _load_pred_meta(pdir / f"{mtag}_stage2a.jsonl")
        if pred3:
            for p in m3_pairs:
                pid = p["pair_id"]
                if pid not in pred3:
                    continue
                pr = parse_remote_homology_response(pred3[pid].get("text", ""))
                do, dt = t3_tier_map[p["rung"]]
                sp = "Homolog" if p["label_homolog"] == 1 else "Decoy"
                out_rows.append(
                    {
                        "m": mshort,
                        "t": 3,
                        "id": pid,
                        "nm": f"{p['domain_a']} × {p['domain_b']}",
                        "sp": sp,
                        "do": do,
                        "dt": dt,
                        "sc": round(float(pr.prob_homolog), 3),
                        "rt": pred3[pid].get("reasoning_tokens", 0),
                        "ab": 0,
                    }
                )

        # Test 4
        pred4 = _load_pred_meta(pdir / f"{mtag}_stage2b.jsonl")
        if pred4:
            for t in m4:
                tid = t["target_id"]
                if tid not in pred4:
                    continue
                t_obj = CAFATarget(
                    target_id=tid,
                    sequence=t["sequence"],
                    length=t["length"],
                    true_terms_mf=t["true_terms_mf"],
                    true_terms_bp=t["true_terms_bp"],
                    true_terms_cc=t["true_terms_cc"],
                    knowledge_type=t.get("knowledge_type", "NK"),
                )
                pmap, _ = parse_cafa_response(pred4[tid].get("text", ""))
                asp_sc = []
                for asp, terms in (("MF", t["true_terms_mf"]), ("BP", t["true_terms_bp"]), ("CC", t["true_terms_cc"])):
                    if terms:
                        asp_sc.append(float(compute_cafa_fmax(dag, [t_obj], {tid: pmap}, aspect=asp, ic_map=ic_map)["wfmax"]))
                wf_i = float(np.mean(asp_sc)) if asp_sc else 0.0
                is_post = t.get("temporal_tier") == "post_may_2026"
                is_hard = t4_sims[tid] < med_sim4
                if not is_hard and not is_post:
                    do, dt = 1, "Hom+Early"
                elif not is_hard and is_post:
                    do, dt = 2, "Hom+Post"
                elif is_hard and not is_post:
                    do, dt = 3, "Low+Early"
                else:
                    do, dt = 4, "Low+Post"
                out_rows.append(
                    {
                        "m": mshort,
                        "t": 4,
                        "id": tid,
                        "nm": f"UniProt {t['uniprot_accession']}",
                        "sp": "Post-May26" if is_post else "Early-2026",
                        "do": do,
                        "dt": dt,
                        "sc": round(wf_i, 3),
                        "rt": pred4[tid].get("reasoning_tokens", 0),
                        "ab": 0,
                    }
                )

        # Test 5
        if mtag in s5_by_m:
            pred5 = _load_pred_meta(pdir / f"{mtag}_stage3a.jsonl")
            for g in s5_by_m[mtag]["per_target"]:
                tid = g["target_id"]
                mt = m5_meta[tid]
                cat = mt["category"]
                if cat == "de_novo_ddg":
                    do, dt, sp = 1, "DeNovo", "Pre-Cutoff"
                elif cat == "natural_ddg":
                    do, dt, sp = 2, "Natural", "Pre-Cutoff"
                elif cat == "dual_stability_function":
                    do, dt, sp = 3, "DualDMS", "Post-May26"
                else:
                    do, dt, sp = 4, "PostMay26", "Post-May26"
                out_rows.append(
                    {
                        "m": mshort,
                        "t": 5,
                        "id": tid,
                        "nm": mt.get("source_name", tid)[:34],
                        "sp": sp,
                        "do": do,
                        "dt": dt,
                        "sc": round(float(g["ndcg_at_5"]), 3),
                        "rt": pred5.get(tid, {}).get("reasoning_tokens", 0),
                        "ab": 0,
                    }
                )

        # Test 6
        if mtag in s6_by_m:
            pred6 = _load_pred_meta(pdir / f"{mtag}_stage6_binding.jsonl")
            for g in s6_by_m[mtag]["per_target"]:
                tid = g["target_id"]
                mt = m6_meta[tid]
                mem = mt.get("memorization_stratum", "known_pdb_pre2026")
                sname = mt.get("source_name", tid)
                if mem == "known_pdb_pre2026" and mt["category"] == "skempi_biophysical_ddg":
                    do, dt, sp = 1, "SKEMPI", "Known PDB"
                elif mem == "known_pdb_pre2026":
                    do, dt, sp = 2, "DispPDB", "Known PDB"
                elif any(k in sname for k in ("KRAS", "GRB2", "RAF1", "MLH1")):
                    do, dt, sp = 3, "PostPPI", "Unseen/Post26"
                else:
                    do, dt, sp = 4, "Unseen", "Unseen/Post26"
                out_rows.append(
                    {
                        "m": mshort,
                        "t": 6,
                        "id": tid,
                        "nm": sname[:36],
                        "sp": sp,
                        "do": do,
                        "dt": dt,
                        "sc": round(float(g["ndcg_at_5"]), 3),
                        "rt": pred6.get(tid, {}).get("reasoning_tokens", 0),
                        "ab": 0 if g["parse_ok"] else 1,
                    }
                )

    return out_rows


HTML_MULTI_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-3 shadow-sm space-y-2">
    <div class="flex flex-wrap items-center justify-between gap-2">
      <div class="text-sm font-bold leading-tight">Frontier Comparison — All Per-Question Scores &amp; Tiers in One View</div>
      <div class="flex flex-wrap items-center gap-3 text-xs">
        <span class="inline-flex items-center gap-1 font-semibold"><span class="w-2.5 h-2.5 rounded-full inline-block" style="background:#0ea5e9"></span>GPT-6.1 Sol</span>
        <span class="inline-flex items-center gap-1 font-semibold"><span class="w-2.5 h-2.5 rounded-full inline-block" style="background:#6366f1"></span>GPT-6.2 Sol</span>
        <span class="inline-flex items-center gap-1 font-semibold"><span class="w-2.5 h-2.5 rounded-full inline-block" style="background:#e11d48"></span>GPT-6.2 Astra</span>
        <span class="text-[var(--muted-foreground)]">Bar = Tier Mean • Dashed = Baseline</span>
      </div>
    </div>

    <canvas id="multi-canvas" width="920" height="368" class="w-full h-[368px] rounded-lg bg-[var(--background)] border border-[var(--border)] cursor-crosshair"></canvas>

    <div id="status-bar" class="text-xs px-2.5 py-1 rounded bg-[var(--secondary)]/60 border border-[var(--border)] flex items-center justify-between gap-2 truncate">
      <span class="text-[var(--muted-foreground)]">Hover or click any dot across the panels above to inspect model, question ID, difficulty tier, split, and score.</span>
    </div>
  </div>

  <script>
    const RECS = __RECS_JSON__;
    const SUMMARY = __SUMMARY_JSON__;

    const MODELS = [
      { name: 'GPT-6.1 Sol', col: '#0ea5e9' },
      { name: 'GPT-6.2 Sol', col: '#6366f1' },
      { name: 'GPT-6.2 Astra', col: '#e11d48' }
    ];

    const PANELS = [
      { t: 1, title: '1. Forward Folding (N=100 • TM)', maxY: 0.78, refY: 0.50, ticks: ['Known', 'DeNovo', 'Homol', 'PDBNov', 'TrueNov'] },
      { t: 3, title: '3. Remote Homology (N=120 • Prob)', maxY: 1.06, refY: 0.62, ticks: ['R1 Easy', 'R2 3Di', 'R3 TEA', 'Decoys'] },
      { t: 4, title: '4. Function CAFA (N=60 • wFmax)', maxY: 1.06, refY: 0.251, ticks: ['Hom+Early', 'Hom+Post', 'Low+Early', 'Low+Post'] },
      { t: 5, title: '5. Monomer Ala (N=52 • NDCG@5)', maxY: 1.06, refY: 0.615, ticks: ['DeNovo', 'Natural', 'DualDMS', 'PostMay26'] },
      { t: 6, title: '6. Binding Ala (N=38 • NDCG@5)', maxY: 1.06, refY: 0.526, ticks: ['SKEMPI', 'DispPDB', 'PostPPI', 'Unseen'] }
    ];

    function jitter(s) {
      let h = 2166136261;
      for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
      return ((h >>> 0) / 4294967295) - 0.5;
    }

    let hitPts = [];
    let pinned = -1;

    function draw() {
      const c = document.getElementById('multi-canvas');
      const dpr = window.devicePixelRatio || 1;
      const W = c.clientWidth || 920;
      const H = 368;
      c.width = W * dpr;
      c.height = H * dpr;
      const ctx = c.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, W, H);

      const fg = getComputedStyle(document.body).color || '#0f172a';
      const grid = 'rgba(148, 163, 184, 0.22)';
      const border = 'rgba(148, 163, 184, 0.35)';

      const cols = 3, rows = 2;
      const cellW = W / cols, cellH = H / rows;
      hitPts = [];

      PANELS.forEach((p, pIdx) => {
        const col = pIdx % cols, row = Math.floor(pIdx / cols);
        const x0 = col * cellW, y0 = row * cellH;
        const pL = x0 + 34, pR = x0 + cellW - 10, pT = y0 + 24, pB = y0 + cellH - 24;
        const pw = pR - pL, ph = pB - pT;

        if (col > 0) { ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0 + 6); ctx.lineTo(x0, y0 + cellH - 6); ctx.stroke(); }
        if (row > 0) { ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 + 6, y0); ctx.lineTo(x0 + cellW - 6, y0); ctx.stroke(); }

        ctx.fillStyle = fg; ctx.font = 'bold 12px sans-serif';
        ctx.fillText(p.title, x0 + 10, y0 + 15);

        const yPx = v => pT + ph * (1 - Math.max(0, Math.min(p.maxY, v)) / p.maxY);
        const yGrid = p.maxY < 0.9 ? [0, 0.25, 0.5, 0.75] : [0, 0.5, 1.0];
        ctx.font = '12px monospace';
        yGrid.forEach(gv => {
          const gy = yPx(gv);
          ctx.strokeStyle = grid; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(pL, gy); ctx.lineTo(pR, gy); ctx.stroke();
          ctx.globalAlpha = 0.65;
          ctx.fillText(gv === 0 ? '0' : gv.toFixed(gv === 0.25 || gv === 0.75 ? 2 : 1), x0 + 4, gy + 4);
          ctx.globalAlpha = 1.0;
        });

        const ry = yPx(p.refY);
        ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = '#64748b'; ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.moveTo(pL, ry); ctx.lineTo(pR, ry); ctx.stroke(); ctx.restore();

        const sub = RECS.filter(r => r.t === p.t);
        const presentModels = MODELS.filter(m => sub.some(r => r.m === m.name));
        const nT = p.ticks.length;
        const bucketW = pw / nT;

        p.ticks.forEach((tk, ti) => {
          const cx = pL + bucketW * (ti + 0.5);
          ctx.fillStyle = fg; ctx.font = '12px sans-serif'; ctx.globalAlpha = 0.78;
          ctx.fillText(tk, cx - ctx.measureText(tk).width / 2, y0 + cellH - 7);
          ctx.globalAlpha = 1.0;

          presentModels.forEach((mod, mi) => {
            const mOff = (mi - (presentModels.length - 1) / 2) * (bucketW * 0.28);
            const mcx = cx + mOff;
            const tsub = sub.filter(r => r.do === ti + 1 && r.m === mod.name);
            if (tsub.length) {
              const mean = tsub.reduce((a, b) => a + b.sc, 0) / tsub.length;
              const my = yPx(mean);
              ctx.strokeStyle = mod.col; ctx.lineWidth = 2.4;
              ctx.beginPath(); ctx.moveTo(mcx - 7, my); ctx.lineTo(mcx + 7, my); ctx.stroke();
            }
            tsub.forEach(r => {
              const px = mcx + jitter(r.id + mod.name) * Math.min(9, bucketW * 0.16);
              const py = yPx(r.sc);
              hitPts.push({ x: px, y: py, col: mod.col, r });
            });
          });
        });
      });

      // Panel 6: Overall Scorecard Grouped Bars across Models + Baselines
      const x0 = 2 * cellW, y0 = cellH;
      ctx.strokeStyle = border; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y0 + 6); ctx.lineTo(x0, y0 + cellH - 6); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0 + 6, y0); ctx.lineTo(x0 + cellW - 6, y0); ctx.stroke();

      ctx.fillStyle = fg; ctx.font = 'bold 12px sans-serif';
      ctx.fillText('Overall Scorecard (Models vs 1D Gray)', x0 + 10, y0 + 15);

      const pL = x0 + 30, pR = x0 + cellW - 10, pT = y0 + 26, pB = y0 + cellH - 24;
      const pw = pR - pL, ph = pB - pT;
      [0, 0.5, 1.0].forEach(gv => {
        const gy = pT + ph * (1 - gv);
        ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(pL, gy); ctx.lineTo(pR, gy); ctx.stroke();
        ctx.fillStyle = fg; ctx.font = '12px monospace'; ctx.globalAlpha = 0.65;
        ctx.fillText(gv === 0 ? '0' : gv.toFixed(1), x0 + 4, gy + 4);
        ctx.globalAlpha = 1.0;
      });

      SUMMARY.forEach((s, si) => {
        const gx = pL + pw * ((si + 0.5) / SUMMARY.length);
        const bars = s.bars;
        const bw = Math.min(9.5, (pw / SUMMARY.length) * (0.76 / bars.length));
        bars.forEach((b, bi) => {
          const bx = gx + (bi - (bars.length - 1) / 2) * (bw + 1.2) - bw / 2;
          const bh = ph * b.v;
          ctx.fillStyle = b.c;
          ctx.fillRect(bx, pB - bh, bw, bh);
        });
        ctx.fillStyle = fg; ctx.font = '12px sans-serif'; ctx.globalAlpha = 0.8;
        ctx.fillText(s.lbl, gx - ctx.measureText(s.lbl).width / 2, y0 + cellH - 7);
        ctx.globalAlpha = 1.0;
      });

      hitPts.forEach((pt, idx) => {
        const sel = idx === pinned;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, sel ? 5.5 : 2.9, 0, Math.PI * 2);
        if (pt.r.ab) {
          ctx.strokeStyle = pt.col; ctx.lineWidth = 1.5; ctx.stroke();
        } else {
          ctx.fillStyle = pt.col; ctx.globalAlpha = sel ? 1.0 : 0.72; ctx.fill(); ctx.globalAlpha = 1.0;
          if (sel) { ctx.strokeStyle = fg; ctx.lineWidth = 1.8; ctx.stroke(); }
        }
      });
    }

    function updateBar(r) {
      const ab = r.ab ? ' [ABSTAINED]' : '';
      document.getElementById('status-bar').innerHTML =
        '<span><strong>[' + r.m + ']</strong> <strong class="font-mono">T' + r.t + ' • ' + r.id + '</strong> — ' + r.nm + ab +
        ' <span class="text-[var(--muted-foreground)]">(' + r.dt + ' • ' + r.sp + ' • ' + r.rt.toLocaleString() + ' tok)</span></span>' +
        '<span class="font-mono font-bold shrink-0">Score: ' + r.sc.toFixed(3) + '</span>';
    }

    const cv = document.getElementById('multi-canvas');
    cv.addEventListener('mousemove', e => {
      if (pinned >= 0) return;
      const rc = cv.getBoundingClientRect();
      const mx = e.clientX - rc.left, my = e.clientY - rc.top;
      let best = -1, bd = 13;
      hitPts.forEach((p, i) => { const d = Math.hypot(p.x - mx, p.y - my); if (d < bd) { bd = d; best = i; } });
      if (best >= 0) updateBar(hitPts[best].r);
    });
    cv.addEventListener('click', e => {
      const rc = cv.getBoundingClientRect();
      const mx = e.clientX - rc.left, my = e.clientY - rc.top;
      let best = -1, bd = 13;
      hitPts.forEach((p, i) => { const d = Math.hypot(p.x - mx, p.y - my); if (d < bd) { bd = d; best = i; } });
      pinned = best === pinned ? -1 : best;
      if (pinned >= 0) updateBar(hitPts[pinned].r);
      draw();
    });

    window.addEventListener('resize', draw);
    draw();
  </script>
</body>
</html>
"""


def build_summary_bars(unified: Dict[str, Any]) -> List[Dict[str, Any]]:
    t1 = {m["model"]: m["overall"]["tm_score"] for m in unified["test1_forward_folding"]}
    t3 = {k: v["overall_auroc"] for k, v in unified["test3_remote_homology"].items()}
    t4 = {k: v["macro_wfmax"] for k, v in unified["test4_cafa_function"].items()}
    t5 = {m["model"]: m["overall"]["ndcg_at_5"] for m in unified["test5_monomer_alanine_ranking"]}
    t6 = {m["model"]: m["overall"]["ndcg_at_5"] for m in unified["test6_binding_alanine_ranking"]}

    specs = [
        ("T1 Fold", t1, 0.145),
        ("T3 Hom", t3, 0.754),
        ("T4 CAFA", t4, 0.251),
        ("T5 Mono", t5, 0.615),
        ("T6 Bind", t6, 0.526),
    ]
    out = []
    for lbl, dmap, base1d in specs:
        bars = [{"v": round(base1d, 3), "c": "#94a3b8"}]
        for mtag, _mshort, mcol in MODEL_ORDER:
            if mtag in dmap:
                bars.append({"v": round(float(dmap[mtag]), 3), "c": mcol})
        out.append({"lbl": lbl, "bars": bars})
    return out


def main() -> None:
    recs = compile_multimodel_records()
    (REPO / "outputs/results/local_13456_multimodel_per_question.json").write_text(
        json.dumps(recs, indent=2), encoding="utf-8"
    )
    unified = json.loads((REPO / "outputs/results/local_13456_unified_summary.json").read_text(encoding="utf-8"))
    sum_bars = build_summary_bars(unified)

    html = (
        HTML_MULTI_TEMPLATE.replace("__RECS_JSON__", json.dumps(recs, separators=(",", ":")))
        .replace("__SUMMARY_JSON__", json.dumps(sum_bars, separators=(",", ":")))
    )
    out_html = REPO / "outputs/figures/frontier_all_in_one.html"
    out_html.write_text(html, encoding="utf-8")

    js = re.findall(r"<script>(.*?)</script>", html, flags=re.S)[0]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
        tf.write(js)
        p = tf.name
    subprocess.run(["node", "--check", p], check=True)
    print(f"Compiled {len(recs)} multi-model per-question records -> {out_html} ({out_html.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
