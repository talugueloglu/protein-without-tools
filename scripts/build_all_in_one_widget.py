#!/usr/bin/env python3
"""Build a single-view 6-panel interactive canvas showing all 370 questions at once (no tab switching)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-3 shadow-sm space-y-2">
    <div class="flex flex-wrap items-center justify-between gap-2">
      <div class="text-sm font-bold leading-tight">GPT-6.2 Sol (High) — All 370 Questions Across All 5 Tests (Zero Tab Switching)</div>
      <div class="flex flex-wrap items-center gap-3 text-xs text-[var(--muted-foreground)]">
        <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full inline-block" style="background:#3b82f6"></span>Known / Pre-Cutoff / Homolog</span>
        <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full inline-block" style="background:#f97316"></span>Post-Cutoff / Unseen</span>
        <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full inline-block" style="background:#ef4444"></span>Decoy</span>
        <span>Bar = Tier Mean • Dashed = Baseline</span>
      </div>
    </div>

    <canvas id="all-canvas" width="900" height="368" class="w-full h-[368px] rounded-lg bg-[var(--background)] border border-[var(--border)] cursor-crosshair"></canvas>

    <div id="status-bar" class="text-xs px-2.5 py-1 rounded bg-[var(--secondary)]/60 border border-[var(--border)] flex items-center justify-between gap-2 truncate">
      <span class="text-[var(--muted-foreground)]">Hover or click any of the 370 dots across the 5 panels above to inspect that individual question.</span>
    </div>
  </div>

  <script>
    const RECS = __COMPACT_JSON__;

    const PANELS = [
      { t: 1, title: '1. Forward Folding (N=100 • TM)', maxY: 0.78, refY: 0.50, ticks: ['Known', 'DeNovo', 'Homol', 'PDBNov', 'TrueNov'] },
      { t: 3, title: '3. Remote Homology (N=120 • Prob)', maxY: 1.06, refY: 0.62, ticks: ['R1 Easy', 'R2 3Di', 'R3 TEA', 'Decoys'] },
      { t: 4, title: '4. Function CAFA (N=60 • wFmax)', maxY: 1.06, refY: 0.251, ticks: ['Hom+Early', 'Hom+Post', 'Low+Early', 'Low+Post'] },
      { t: 5, title: '5. Monomer Ala (N=52 • NDCG@5)', maxY: 1.06, refY: 0.615, ticks: ['DeNovo', 'Natural', 'DualDMS', 'PostMay26'] },
      { t: 6, title: '6. Binding Ala (N=38 • NDCG@5)', maxY: 1.06, refY: 0.526, ticks: ['SKEMPI', 'DispPDB', 'PostPPI', 'Unseen'] }
    ];

    const SUMMARY = [
      { lbl: 'T1 Fold', vals: [0.284, 0.264, 0.145] },
      { lbl: 'T3 Hom', vals: [0.863, 0.811, 0.754] },
      { lbl: 'T4 CAFA', vals: [0.377, 0.241, 0.251] },
      { lbl: 'T5 Mono', vals: [0.693, 0.609, 0.615] },
      { lbl: 'T6 Bind', vals: [0.599, 0.547, 0.526] }
    ];

    function ptColor(r) {
      if (r.sp.includes('Decoy')) return '#ef4444';
      if (r.sp.includes('Post-') || r.sp.includes('Unseen')) return '#f97316';
      return '#3b82f6';
    }

    function jitter(s) {
      let h = 2166136261;
      for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
      return ((h >>> 0) / 4294967295) - 0.5;
    }

    let hitPts = [];
    let pinned = -1;

    function drawAll() {
      const c = document.getElementById('all-canvas');
      const dpr = window.devicePixelRatio || 1;
      const W = c.clientWidth || 900;
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
      const cellW = W / cols;
      const cellH = H / rows;
      hitPts = [];

      PANELS.forEach((p, pIdx) => {
        const col = pIdx % cols, row = Math.floor(pIdx / cols);
        const x0 = col * cellW, y0 = row * cellH;
        const pL = x0 + 34, pR = x0 + cellW - 10, pT = y0 + 24, pB = y0 + cellH - 24;
        const pw = pR - pL, ph = pB - pT;

        if (col > 0) { ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0 + 6); ctx.lineTo(x0, y0 + cellH - 6); ctx.stroke(); }
        if (row > 0) { ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 + 6, y0); ctx.lineTo(x0 + cellW - 6, y0); ctx.stroke(); }

        ctx.fillStyle = fg;
        ctx.font = 'bold 12px sans-serif';
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
        const nT = p.ticks.length;
        p.ticks.forEach((tk, ti) => {
          const tsub = sub.filter(r => r.do === ti + 1);
          const cx = pL + pw * ((ti + 0.5) / nT);

          if (tsub.length) {
            const mean = tsub.reduce((a, b) => a + b.sc, 0) / tsub.length;
            const my = yPx(mean);
            ctx.strokeStyle = fg; ctx.lineWidth = 2.0;
            ctx.beginPath(); ctx.moveTo(cx - 12, my); ctx.lineTo(cx + 12, my); ctx.stroke();
          }

          ctx.fillStyle = fg; ctx.font = '12px sans-serif'; ctx.globalAlpha = 0.78;
          const tw = ctx.measureText(tk).width;
          ctx.fillText(tk, cx - tw / 2, y0 + cellH - 7);
          ctx.globalAlpha = 1.0;

          tsub.forEach(r => {
            const px = cx + jitter(r.id) * Math.min(22, (pw / nT) * 0.55);
            const py = yPx(r.sc);
            hitPts.push({ x: px, y: py, r });
          });
        });
      });

      // Panel 6: Overall Scorecard Grouped Bars
      const x0 = 2 * cellW, y0 = cellH;
      ctx.strokeStyle = border; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y0 + 6); ctx.lineTo(x0, y0 + cellH - 6); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0 + 6, y0); ctx.lineTo(x0 + cellW - 6, y0); ctx.stroke();

      ctx.fillStyle = fg; ctx.font = 'bold 12px sans-serif';
      ctx.fillText('Summary: Sol (Indigo) vs 3D (Grn) & 1D', x0 + 10, y0 + 15);

      const pL = x0 + 30, pR = x0 + cellW - 10, pT = y0 + 26, pB = y0 + cellH - 24;
      const pw = pR - pL, ph = pB - pT;
      [0, 0.5, 1.0].forEach(gv => {
        const gy = pT + ph * (1 - gv);
        ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(pL, gy); ctx.lineTo(pR, gy); ctx.stroke();
        ctx.fillStyle = fg; ctx.font = '12px monospace'; ctx.globalAlpha = 0.65;
        ctx.fillText(gv === 0 ? '0' : gv.toFixed(1), x0 + 4, gy + 4);
        ctx.globalAlpha = 1.0;
      });

      const cols3 = ['#6366f1', '#10b981', '#94a3b8'];
      SUMMARY.forEach((s, si) => {
        const gx = pL + pw * ((si + 0.5) / SUMMARY.length);
        const bw = Math.min(11, (pw / SUMMARY.length) * 0.24);
        s.vals.forEach((v, vi) => {
          const bx = gx + (vi - 1) * (bw + 1.5) - bw / 2;
          const bh = ph * v;
          ctx.fillStyle = cols3[vi];
          ctx.fillRect(bx, pB - bh, bw, bh);
        });
        ctx.fillStyle = fg; ctx.font = '12px sans-serif'; ctx.globalAlpha = 0.8;
        ctx.fillText(s.lbl, gx - ctx.measureText(s.lbl).width / 2, y0 + cellH - 7);
        ctx.globalAlpha = 1.0;
      });

      hitPts.forEach((pt, idx) => {
        const col = ptColor(pt.r);
        const sel = idx === pinned;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, sel ? 5.8 : 3.6, 0, Math.PI * 2);
        if (pt.r.ab) {
          ctx.strokeStyle = col; ctx.lineWidth = 1.7; ctx.stroke();
        } else {
          ctx.fillStyle = col; ctx.globalAlpha = sel ? 1.0 : 0.82; ctx.fill(); ctx.globalAlpha = 1.0;
          if (sel) { ctx.strokeStyle = fg; ctx.lineWidth = 1.8; ctx.stroke(); }
        }
      });
    }

    function updateBar(r) {
      const ab = r.ab ? ' [ABSTAINED]' : '';
      document.getElementById('status-bar').innerHTML =
        '<span><strong class="font-mono">T' + r.t + ' • ' + r.id + '</strong> — ' + r.nm + ab +
        ' <span class="text-[var(--muted-foreground)]">(' + r.dt + ' • ' + r.sp + ' • ' + r.rt.toLocaleString() + ' tok)</span></span>' +
        '<span class="font-mono font-bold shrink-0">Score: ' + r.sc.toFixed(3) + '</span>';
    }

    const cv = document.getElementById('all-canvas');
    cv.addEventListener('mousemove', e => {
      if (pinned >= 0) return;
      const rc = cv.getBoundingClientRect();
      const mx = e.clientX - rc.left, my = e.clientY - rc.top;
      let best = -1, bd = 14;
      hitPts.forEach((p, i) => { const d = Math.hypot(p.x - mx, p.y - my); if (d < bd) { bd = d; best = i; } });
      if (best >= 0) updateBar(hitPts[best].r);
    });
    cv.addEventListener('click', e => {
      const rc = cv.getBoundingClientRect();
      const mx = e.clientX - rc.left, my = e.clientY - rc.top;
      let best = -1, bd = 14;
      hitPts.forEach((p, i) => { const d = Math.hypot(p.x - mx, p.y - my); if (d < bd) { bd = d; best = i; } });
      pinned = best === pinned ? -1 : best;
      if (pinned >= 0) updateBar(hitPts[pinned].r);
      drawAll();
    });

    window.addEventListener('resize', drawAll);
    drawAll();
  </script>
</body>
</html>
"""


def main() -> None:
    recs = json.loads((REPO / "outputs/results/local_13456_per_question_records.json").read_text(encoding="utf-8"))
    compact = [
        {
            "t": r["test"],
            "id": r["id"],
            "nm": r["name"][:38],
            "sp": r["split"],
            "do": r["diff_order"],
            "dt": r["diff_tier"].split(": ", 1)[-1],
            "sc": round(r["score"], 3),
            "s2": round(r["secondary_score"], 2),
            "rt": r["reasoning_tokens"],
            "ab": 1 if r.get("abstained") else 0,
        }
        for r in recs
    ]
    out_html = HTML_TEMPLATE.replace("__COMPACT_JSON__", json.dumps(compact, separators=(",", ":")))
    out_repo = REPO / "outputs/figures/gpt62_sol_all_in_one.html"
    out_repo.write_text(out_html, encoding="utf-8")

    js = re.findall(r"<script>(.*?)</script>", out_html, flags=re.S)[0]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
        tf.write(js)
        p = tf.name
    subprocess.run(["node", "--check", p], check=True)
    print(f"Built {out_repo} ({out_repo.stat().st_size / 1024:.1f} KB) — JS verified OK.")


if __name__ == "__main__":
    main()
