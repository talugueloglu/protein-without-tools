#!/usr/bin/env python3
"""Inject the 8 established baselines (1 clean domain baseline per task) into proteinbench.html,
updating DS (all 522 datapoints), FIGDATA, FIGS (f1..f9), RESULT_SPEC, and DETAIL_SPEC."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]


NEW_JS_TAIL = r"""/* ---------------------------------------------------------------- *
 * Results table, aggregated in the browser from the embedded
 * per-datapoint scores so it always matches the data.
 * ---------------------------------------------------------------- */
var MODELS = [
  { k: 'gpt-6.1-sol_think4096',   n: 'gpt-6.1-sol' },
  { k: 'gpt-6.2-sol_think4096',   n: 'gpt-6.2-sol' },
  { k: 'gpt-6.2-astra_think4096', n: 'gpt-6.2-astra' }
];

var TASK_BASELINE = {
  1: { k: 'baseline_esmfold2',                 n: 'ESMFold-2' },
  2: { k: 'baseline_proteinmpnn',              n: 'ProteinMPNN' },
  3: { k: 'baseline_t3',                       n: '__T3_BASE_NAME__' },
  4: { k: 'baseline_t4',                       n: '__T4_BASE_NAME__' },
  5: { k: 'baseline_esm2_650m',                n: 'ESM-2 (650M)' },
  6: { k: 'baseline_esm2_650m',                n: 'ESM-2 (650M)' },
  7: { k: 'baseline_rfd3_mpnn',                n: 'RFdiffusion3 + ProteinMPNN' },
  8: { k: 'baseline_boltzgen_closed_loop_af2', n: 'BoltzGen', trajKey: 'baseline_boltzgen' }
};

var SCORE_LABEL = {
  'baseline_esmfold2':                 'ESMFold-2 (baseline)',
  'baseline_proteinmpnn':              'ProteinMPNN (baseline)',
  'baseline_t3':                       '__T3_BASE_NAME__ (baseline)',
  'baseline_t4':                       '__T4_BASE_NAME__ (baseline)',
  'baseline_esm2_650m':                'ESM-2 650M (baseline)',
  'baseline_rfd3_mpnn':                'RFdiffusion3 + ProteinMPNN (baseline)',
  'baseline_boltzgen_closed_loop_af2': 'BoltzGen (baseline)',
  'gpt-6.1-sol_think4096':             'GPT-6.1 Sol',
  'gpt-6.2-sol_think4096':             'GPT-6.2 Sol',
  'gpt-6.2-astra_think4096':           'GPT-6.2 Astra',
  'gpt-6.1-sol_think4096_closed_loop_af2':   'GPT-6.1 Sol',
  'gpt-6.2-sol_think4096_closed_loop_af2':   'GPT-6.2 Sol',
  'gpt-6.2-astra_think4096_closed_loop_af2': 'GPT-6.2 Astra'
};

function isNum(x) { return typeof x === 'number' && isFinite(x); }
function rMean(a) { a = a.filter(isNum); return a.length ? a.reduce(function(x,y){return x+y;},0)/a.length : null; }
function rPct(a, test) {
  a = a.filter(function(x){ return x !== undefined && x !== null; });
  return a.length ? 100 * a.filter(test).length / a.length : null;
}
function fmt(v, d, unit) {
  if (v === null || v === undefined) return '—';
  return v.toFixed(d) + (unit || '');
}

function scoresFor(t, modelKey) {
  var out = [];
  DS.forEach(function(e){
    if (e.t !== t) return;
    var s = e.sc && e.sc[modelKey];
    if (s) out.push(s);
  });
  return out;
}

/* Remote homology ships raw model JSON, so it is graded here against
   the stored label rather than read off a precomputed metric. */
function homologyStats(modelKey) {
  var ok = 0, n = 0, pos = [], neg = [];
  DS.forEach(function(e){
    if (e.t !== 3 || e.hl === undefined) return;
    var s = e.sc && e.sc[modelKey];
    if (!s || !s.raw) return;
    var p;
    try { p = JSON.parse(s.raw); } catch (err) { return; }
    var truth = !!e.hl, pred = !!p.same_superfamily;
    var prob = isNum(p.probability) ? p.probability : 0.5;
    (truth ? pos : neg).push(prob);
    n++; if (truth === pred) ok++;
  });
  var auroc = null;
  if (pos.length && neg.length) {
    var w = 0;
    for (var i = 0; i < pos.length; i++)
      for (var j = 0; j < neg.length; j++)
        w += (pos[i] > neg[j]) ? 1 : (pos[i] === neg[j] ? 0.5 : 0);
    auroc = w / (pos.length * neg.length);
  }
  return { acc: n ? 100*ok/n : null, auroc: auroc, n: n };
}

function cafaAspect(modelKey, idx) {
  var s = (FIGDATA.t4 && FIGDATA.t4.series) || [];
  for (var i = 0; i < s.length; i++) {
    if (s[i].k === modelKey || s[i].n === modelKey) return idx === 'macro' ? s[i].macro : s[i].v[idx];
  }
  return null;
}

var RESULT_SPEC = [
  { t:1, name:'1. Forward folding', note:'TM-score against the native fold (baseline: single-sequence ESMFold-2).',
    cols:[['TM-score',     function(S){ return fmt(rMean(S.map(function(s){return s.tm_score;})),3); }],
          ['P@L (|i−j|≥24)',function(S){ return fmt(rMean(S.map(function(s){return s.p_at_l_sep24;})),3); }],
          ['Q3',           function(S){ return fmt(rMean(S.map(function(s){return s.q3_accuracy;})),3); }]] },
  { t:2, name:'2. Inverse folding', note:'Designed sequence refolded with ESMFold-2 vs input backbone (baseline: ProteinMPNN).',
    cols:[['scTM',         function(S){ return fmt(rMean(S.map(function(s){return s.sc_tm_score;})),3); }],
          ['scRMSD < 2 Å', function(S){ return fmt(rPct(S.map(function(s){return s.sc_rmsd_ca;}), function(x){return x<2;}),1,'%'); }],
          ['NSR',          function(S){ return fmt(rMean(S.map(function(s){return s.nsr;})),3); }]] },
  { t:3, name:'3. Remote homology', note:'Graded against stored SCOPe superfamily label (baseline: __T3_BASE_NAME__).',
    cols:[['Accuracy',     function(S,m){ return fmt(homologyStats(m).acc,1,'%'); }],
          ['AUROC',        function(S,m){ return fmt(homologyStats(m).auroc,3); }]] },
  { t:4, name:'4. Function prediction', note:'IC-weighted CAFA wFₘₐₓ on 2026 experimental GO holdouts (baseline: __T4_BASE_NAME__).',
    cols:[['Macro wFₘₐₓ',  function(S,m){ return fmt(cafaAspect(m,'macro'),3); }],
          ['MF',           function(S,m){ return fmt(cafaAspect(m,0),3); }],
          ['BP',           function(S,m){ return fmt(cafaAspect(m,1),3); }],
          ['CC',           function(S,m){ return fmt(cafaAspect(m,2),3); }]] },
  { t:5, name:'5. Monomer alanine ranking', note:'Same-AA decoys hold a 1D lookup table at 50% (baseline: ESM-2 650M).',
    cols:[['NDCG@5',       function(S){ return fmt(rMean(S.map(function(s){return s.ndcg_at_5;})),3); }],
          ['Same-AA win',  function(S){ return fmt(rMean(S.map(function(s){return s.same_aa_win_rate;})),3); }],
          ['Spearman ρ',   function(S){ return fmt(rMean(S.map(function(s){return s.spearman_rho;})),3); }]] },
  { t:6, name:'6. Binding alanine ranking', note:'Monomer-destabilising mutations excluded (baseline: ESM-2 650M).',
    cols:[['NDCG@5',       function(S){ return fmt(rMean(S.map(function(s){return s.ndcg_at_5;})),3); }],
          ['Same-AA win',  function(S){ return fmt(rMean(S.map(function(s){return s.same_aa_win_rate;})),3); }],
          ['Spearman ρ',   function(S){ return fmt(rMean(S.map(function(s){return s.spearman_rho;})),3); }]] },
  { t:7, name:'7. Monomer & motif design', note:'Graded by ESMFold-2 + OpenFold3 gates (baseline: RFdiffusion3 + ProteinMPNN).',
    cols:[['4-gate pass',  function(S){ return fmt(rPct(S.map(function(s){return s.esmfold2_4gate_pass;}), function(x){return !!x;}),1,'%'); }],
          ['pLDDT',        function(S){ return fmt(rMean(S.map(function(s){return s.plddt_mean;})),1); }],
          ['Cross-oracle ≤2Å', function(S){ return fmt(rPct(S.map(function(s){return s.cross_oracle_sc_rmsd;}), function(x){return isNum(x)&&x<=2.0;}),1,'%'); }]] },
  { t:8, name:'8. Closed-loop epitope binders', note:'16 AlphaFold2 shots per epitope (baseline: BoltzGen).', suffix:'_closed_loop_af2',
    cols:[['Hit@16',       function(S){ return fmt(rPct(S.map(function(s){return s.hit_achieved;}), function(x){return !!x;}),1,'%'); }],
          ['Best AF2 score', function(S){ return fmt(rMean(S.map(function(s){return s.best_composite_score;})),3); }],
          ['Mean shots',   function(S){ return fmt(rMean(S.map(function(s){return s.shots_to_first_hit;})),1); }]] }
];

var GRAY1 = '#9aa0a6', GRAY2 = '#c6cad0';
var SERIES = [
  { k:'__baseline__',            n:'Established baseline', c:GRAY1, base:true },
  { k:'gpt-6.1-sol_think4096',   n:'gpt-6.1-sol',          c:'#2a78d6' },
  { k:'gpt-6.2-sol_think4096',   n:'gpt-6.2-sol',          c:'#eb6834' },
  { k:'gpt-6.2-astra_think4096', n:'gpt-6.2-astra',        c:'#1baf7a' }
];

function roundedTopBar(x, y, w, h, r) {
  if (h <= 0.5) return '';
  r = Math.min(r, w/2, h);
  return 'M' + x + ' ' + (y+h) +
         'L' + x + ' ' + (y+r) +
         'Q' + x + ' ' + y + ' ' + (x+r) + ' ' + y +
         'L' + (x+w-r) + ' ' + y +
         'Q' + (x+w) + ' ' + y + ' ' + (x+w) + ' ' + (y+r) +
         'L' + (x+w) + ' ' + (y+h) + 'Z';
}

/* ---------------------------------------------------------------- *
 * Detailed metrics as small multiples: one panel per metric, each
 * with its own scale.
 * ---------------------------------------------------------------- */
var DETAIL_SPEC = [
  { t:1, name:'1. Forward folding (baseline: ESMFold-2)', metrics:[
      { label:'TM-score', max:1,   get:function(S){ return rMean(S.map(function(s){return s.tm_score;})); } },
      { label:'P@L (|i−j|≥24)', max:1, get:function(S){ return rMean(S.map(function(s){return s.p_at_l_sep24;})); } },
      { label:'Q3', max:1,         get:function(S){ return rMean(S.map(function(s){return s.q3_accuracy;})); } } ] },
  { t:2, name:'2. Inverse folding (baseline: ProteinMPNN)', metrics:[
      { label:'scTM', max:1,       get:function(S){ return rMean(S.map(function(s){return s.sc_tm_score;})); } },
      { label:'scRMSD < 2 Å', max:100, pct:true, get:function(S){ return rPct(S.map(function(s){return s.sc_rmsd_ca;}), function(x){return x<2;}); } },
      { label:'NSR', max:1,        get:function(S){ return rMean(S.map(function(s){return s.nsr;})); } } ] },
  { t:3, name:'3. Remote homology (baseline: __T3_BASE_NAME__)', metrics:[
      { label:'Accuracy', max:100, pct:true, get:function(S,m){ return homologyStats(m).acc; } },
      { label:'AUROC', max:1,      get:function(S,m){ return homologyStats(m).auroc; } } ] },
  { t:4, name:'4. Function prediction (baseline: __T4_BASE_NAME__)', metrics:[
      { label:'Macro wFₘₐₓ', max:1, get:function(S,m){ return cafaAspect(m,'macro'); } },
      { label:'MF wFₘₐₓ', max:1,    get:function(S,m){ return cafaAspect(m,0); } },
      { label:'BP wFₘₐₓ', max:1,    get:function(S,m){ return cafaAspect(m,1); } },
      { label:'CC wFₘₐₓ', max:1,    get:function(S,m){ return cafaAspect(m,2); } } ] },
  { t:5, name:'5. Monomer alanine ranking (baseline: ESM-2 650M)', metrics:[
      { label:'NDCG@5', max:1,     get:function(S){ return rMean(S.map(function(s){return s.ndcg_at_5;})); } },
      { label:'Same-AA win', max:1, floor:0.5, get:function(S){ return rMean(S.map(function(s){return s.same_aa_win_rate;})); } },
      { label:'Spearman ρ', max:1, get:function(S){ return rMean(S.map(function(s){return s.spearman_rho;})); } } ] },
  { t:6, name:'6. Binding alanine ranking (baseline: ESM-2 650M)', metrics:[
      { label:'NDCG@5', max:1,     get:function(S){ return rMean(S.map(function(s){return s.ndcg_at_5;})); } },
      { label:'Same-AA win', max:1, floor:0.5, get:function(S){ return rMean(S.map(function(s){return s.same_aa_win_rate;})); } },
      { label:'Spearman ρ', max:1, get:function(S){ return rMean(S.map(function(s){return s.spearman_rho;})); } } ] },
  { t:7, name:'7. Monomer & motif design (baseline: RFdiffusion3 + ProteinMPNN)', metrics:[
      { label:'4-gate pass', max:100, pct:true, get:function(S){ return rPct(S.map(function(s){return s.esmfold2_4gate_pass;}), function(x){return !!x;}); } },
      { label:'pLDDT', max:100,    get:function(S){ return rMean(S.map(function(s){return s.plddt_mean;})); } },
      { label:'Cross-oracle ≤ 2 Å', max:100, pct:true, get:function(S){ return rPct(S.map(function(s){return s.cross_oracle_sc_rmsd;}), function(x){return isNum(x)&&x<=2.0;}); } } ] },
  { t:8, name:'8. Closed-loop epitope binders (baseline: BoltzGen)', suffix:'_closed_loop_af2', metrics:[
      { label:'Hit@16', max:100, pct:true, get:function(S){ return rPct(S.map(function(s){return s.hit_achieved;}), function(x){return !!x;}); } },
      { label:'Best AF2 score', max:1, floor:0.75, get:function(S){ return rMean(S.map(function(s){return s.best_composite_score;})); } },
      { label:'Shots to first hit', max:16, lower:true, get:function(S){ return rMean(S.map(function(s){return s.shots_to_first_hit;})); } } ] }
];

function meterRow(metric, values, names) {
  var W = 300, H = 22, cy = 11, r = 4.6;
  var xOf = function(v){ return 2 + (Math.max(0, Math.min(v, metric.max)) / metric.max) * (W - 4); };
  var fmtv = function(v){
    if (v === null || v === undefined || isNaN(v)) return null;
    return metric.pct ? v.toFixed(0) + '%' : (metric.max > 16 ? v.toFixed(0) : v.toFixed(2));
  };
  var any = values.some(function(v){ return fmtv(v) !== null; });

  var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" preserveAspectRatio="none" class="mt-svg" role="img" aria-label="' + esc(metric.label) + '">';
  s += '<line x1="2" y1="' + cy + '" x2="' + (W-2) + '" y2="' + cy + '" class="mt-track-line"/>';
  if (metric.floor !== undefined) {
    s += '<line x1="' + xOf(metric.floor) + '" y1="3" x2="' + xOf(metric.floor) + '" y2="' + (H-3) + '" class="mt-floor"/>';
  }
  if (any) {
    SERIES.forEach(function(se, i){
      var v = values[i], nm = (names && names[i]) || se.n;
      if (fmtv(v) === null) return;
      s += '<circle cx="' + xOf(v) + '" cy="' + cy + '" r="' + r + '" fill="' + se.c + '" stroke="#fff" stroke-width="2" class="cl-bar" ' +
           'data-tip="' + esc(nm) + ' · ' + esc(metric.label) + ' ' + fmtv(v) + '"><title>' + esc(nm) + ' — ' + esc(metric.label) + ': ' + fmtv(v) + '</title></circle>';
    });
  }
  s += '</svg>';

  var figs = any
    ? SERIES.map(function(se, i){
        var f = fmtv(values[i]);
        return '<span class="mt-fig"' + (f===null?' data-na="1"':'') + '><i style="background:' + se.c + '"></i>' + (f===null ? '—' : f) + '</span>';
      }).join('')
    : '<span class="mt-none">not reached by any model</span>';

  return '<div class="mt-row">' +
           '<div class="mt-name">' + esc(metric.label) +
             (metric.lower ? '<span class="mt-dir">lower is better</span>' : '') + '</div>' +
           '<div class="mt-track">' + s + '<span class="mt-max">' +
             (metric.pct ? metric.max + '%' : (metric.max === 1 ? '1.0' : metric.max)) + '</span></div>' +
           '<div class="mt-vals">' + figs + '</div>' +
         '</div>';
}

/* ================================================================= *
 * Figures. Each is its own chart with its own axis; one established
 * baseline per task is drawn in de-emphasis gray alongside the three
 * frontier reasoning models across every holdout stratum.
 * ================================================================= */
var MCOL = { 'gpt-6.1-sol_think4096':'#2a78d6', 'gpt-6.2-sol_think4096':'#eb6834', 'gpt-6.2-astra_think4096':'#1baf7a' };
var MNAME = { 'gpt-6.1-sol_think4096':'GPT-6.1 Sol', 'gpt-6.2-sol_think4096':'GPT-6.2 Sol', 'gpt-6.2-astra_think4096':'GPT-6.2 Astra' };
var MODEL_KEYS = ['gpt-6.1-sol_think4096','gpt-6.2-sol_think4096','gpt-6.2-astra_think4096'];

var STRATA = [
  { k:'pre_cutoff_control',           l:'Known\n(pre-2026)' },
  { k:'post_cutoff_de_novo',          l:'De novo' },
  { k:'post_cutoff_homolog_gradient', l:'Homolog\ngradient' },
  { k:'post_cutoff_pdb_novel',        l:'PDB-novel' },
  { k:'post_cutoff_struct_novel',     l:'Truly novel' }
];

var T5_STRATA = [
  { k:'de_novo_ddg',             l:'De novo\nminiproteins' },
  { k:'natural_ddg',             l:'Natural\nprotein ΔΔG' },
  { k:'dual_stability_function', l:'Stability &\nfunction' },
  { k:'post_cutoff_mavedb',      l:'Post-cutoff\nMaveDB' }
];

function byStratumList(strataList, t, mkey, field) {
  var acc = {};
  DS.forEach(function(e){
    if (e.t !== t) return;
    var s = e.sc && e.sc[mkey]; if (!s) return;
    var v = s[field]; if (typeof v !== 'number' || !isFinite(v)) return;
    (acc[e.st] = acc[e.st] || []).push(v);
  });
  return strataList.map(function(st){
    var a = acc[st.k];
    return a && a.length ? a.reduce(function(x,y){return x+y;},0)/a.length : null;
  });
}
function byStratum(t, mkey, field) { return byStratumList(STRATA, t, mkey, field); }
function stratumNList(strataList, t) {
  var c = {};
  DS.forEach(function(e){ if (e.t === t) c[e.st] = (c[e.st]||0)+1; });
  return strataList.map(function(st){ return c[st.k]||0; });
}
function stratumN(t) { return stratumNList(STRATA, t); }

function svgTextW(str, fs) {
  var s = String(str || ''), w = 0;
  for (var i = 0; i < s.length; i++) {
    var ch = s[i];
    if ('MW@%'.indexOf(ch) !== -1) w += 0.94;
    else if ('→–—'.indexOf(ch) !== -1) w += 0.84;
    else if ('iljI1.,:;|()[]! '.indexOf(ch) !== -1) w += 0.36;
    else if (ch >= 'A' && ch <= 'Z') w += 0.68;
    else if (ch >= '0' && ch <= '9') w += 0.60;
    else w += 0.56;
  }
  return w * (fs || 13);
}

function legendRows(series, mL, right, y0) {
  var lx = mL, row = 0, rowGap = 20, out = '';
  series.forEach(function(se){
    var raw = String(se.n || ''), label = esc(raw), tw = svgTextW(raw, 13);
    if (se.line) {
      var w0 = 20 + tw + 24;
      if (lx > mL && lx + w0 - 24 > right) { lx = mL; row++; }
      var yy = y0 + row * rowGap;
      out += '<line x1="'+lx+'" y1="'+(yy-4)+'" x2="'+(lx+14)+'" y2="'+(yy-4)+'" class="cl-floor"/>';
      out += '<text x="'+(lx+20)+'" y="'+yy+'" class="cl-leg">'+label+'</text>';
      lx += w0; return;
    }
    var wNeeded = 16 + tw + 24;
    if (lx > mL && lx + wNeeded - 24 > right) { lx = mL; row++; }
    var y = y0 + row * rowGap;
    out += '<rect x="' + lx + '" y="' + (y - 9) + '" width="10" height="10" rx="2" fill="' + se.c + '"/>';
    out += '<text x="' + (lx + 16) + '" y="' + y + '" class="cl-leg">' + label + '</text>';
    lx += wNeeded;
  });
  return { svg: out, rows: row + 1, rowGap: rowGap };
}

function niceMax(v) {
  if (!isFinite(v) || v <= 0) return 1;
  var steps = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1,2,4,5,8,10,16,20,25,50,100];
  for (var i=0;i<steps.length;i++) if (steps[i] >= v) return steps[i];
  return Math.ceil(v);
}
function niceTicks(max) {
  var steps=[0.02,0.05,0.1,0.2,0.25,0.5,1,2,2.5,5,10,20,25,50,100];
  for (var i=0;i<steps.length;i++){
    var k=max/steps[i];
    if (k<=6.0001 && Math.abs(k-Math.round(k))<1e-6) return Math.round(k);
  }
  return 5;
}
function fitMax(series, refs, floor) {
  var m = 0, r = 0;
  (series||[]).forEach(function(se){ (se.v||[]).forEach(function(v){ if (typeof v==='number'&&isFinite(v)&&v>m) m=v; }); });
  (refs||[]).forEach(function(x){ if (x.v>r) r=x.v; });
  var fitted = niceMax(Math.max(m*1.12, r*1.04));
  if (m <= 1 && r <= 1 && fitted > 1) fitted = 1;
  return Math.max(floor||0, fitted);
}
function tickLabel(v, max, ticks) {
  var step = max / ticks;
  if (step >= 1 - 1e-9 && Math.abs(step - Math.round(step)) < 1e-9) return v.toFixed(0);
  var d = Math.abs(step*10 - Math.round(step*10)) < 1e-6 ? 1 : 2;
  return v.toFixed(d);
}

function groupedBars(o) {
  var maxLines = 1;
  (o.cats || []).forEach(function(c){
    var n = String(c).split('\n').length;
    if (n > maxLines) maxLines = n;
  });
  var lineStep = 15;
  var W=o.w||880, mL=66, mR=o.mR||16, mT=o.mT||44;
  var mB = o.mB || (18 + maxLines * lineStep + (o.subCats ? 18 : 8));
  var lgSeries=o.series.concat((o.refs||[]).map(function(r){ return {n:r.l,line:true}; }));
  var lg=legendRows(lgSeries, mL, W-mR, 20);
  if (lg.rows>1) mT += (lg.rows-1) * lg.rowGap;
  var H = o.h || Math.max(304, mT + 198 + mB);
  var max=o.maxFixed||fitMax(o.series,o.refs,o.floorMax);
  var pw=W-mL-mR, ph=H-mT-mB, y0=mT+ph;
  var nS=o.series.length, band=pw/o.cats.length;
  var BAR=Math.min(22, Math.max(8, (band-18)/nS - 2)), GAP=2;
  var inner=nS*BAR+(nS-1)*GAP;
  var yOf=function(v){ return y0-(Math.max(0,Math.min(v,max))/max)*ph; };
  var fmt=o.fmt||function(v){ return v.toFixed(2); };
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="'+esc(o.aria||'chart')+'">';
  s+=lg.svg;

  var ticks=o.ticks||niceTicks(max);
  for (var i=0;i<=ticks;i++){
    var v=max*i/ticks, y=yOf(v);
    s+='<line x1="'+mL+'" y1="'+y+'" x2="'+(W-mR)+'" y2="'+y+'" class="cl-grid"/>';
    s+='<text x="'+(mL-9)+'" y="'+(y+4)+'" class="cl-tick" text-anchor="end">'+(o.tickFmt?o.tickFmt(v):tickLabel(v,max,ticks))+'</text>';
  }
  (o.refs||[]).forEach(function(rf){
    var y=yOf(rf.v);
    s+='<line x1="'+mL+'" y1="'+y+'" x2="'+(W-mR)+'" y2="'+y+'" class="cl-floor"/>';
  });

  o.cats.forEach(function(cat,gi){
    var gx=mL+gi*band, bx=gx+(band-inner)/2;
    o.series.forEach(function(se,si){
      var v=se.v[gi], x=bx+si*(BAR+GAP);
      if (v===null||v===undefined||isNaN(v)) return;
      var h=Math.max(v===0?2.5:0, y0-yOf(v));
      s+='<path d="'+roundedTopBar(x,y0-h,BAR,h,3.5)+'" fill="'+se.c+'" class="cl-bar" data-tip="'+esc(se.n)+' · '+esc(String(cat).replace(/\n/g,' '))+' '+fmt(v)+'"><title>'+esc(se.n)+' — '+fmt(v)+'</title></path>';
    });
    String(cat).split('\n').forEach(function(line,li){
      s+='<text x="'+(gx+band/2)+'" y="'+(y0+18+li*lineStep)+'" class="cl-cat" text-anchor="middle">'+esc(line)+'</text>';
    });
    if (o.subCats) s+='<text x="'+(gx+band/2)+'" y="'+(y0+18+maxLines*lineStep+2)+'" class="cl-metric" text-anchor="middle">'+esc(o.subCats[gi])+'</text>';
  });
  s+='<line x1="'+mL+'" y1="'+y0+'" x2="'+(W-mR)+'" y2="'+y0+'" class="cl-axisline"/>';
  s+='<text class="cl-axis" transform="translate(15,'+(mT+ph/2)+') rotate(-90)" text-anchor="middle">'+esc(o.ylabel||'')+'</text>';
  return s+'</svg>';
}

function lineChart(o) {
  var W=880, mL=66, mR=22, mT=44, mB=50;
  var lg2=legendRows(o.series.concat((o.refs||[]).map(function(r){ return {n:r.l,line:true}; })), mL, W-mR, 20);
  if (lg2.rows>1) mT += (lg2.rows-1) * lg2.rowGap;
  var H = o.h || Math.max(304, mT + 200 + mB);
  var max=o.maxFixed||fitMax(o.series,o.refs,o.floorMax);
  var pw=W-mL-mR, ph=H-mT-mB, y0=mT+ph, n=o.x.length;
  var xOf=function(i){ return mL+(n===1?pw/2:(i/(n-1))*pw); };
  var yOf=function(v){ return y0-(Math.max(0,Math.min(v,max))/max)*ph; };
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="'+esc(o.aria||'trajectory')+'">';
  s+=lg2.svg;
  var lt=niceTicks(max);
  for (var i=0;i<=lt;i++){
    var v=max*i/lt, y=yOf(v);
    s+='<line x1="'+mL+'" y1="'+y+'" x2="'+(W-mR)+'" y2="'+y+'" class="cl-grid"/>';
    s+='<text x="'+(mL-9)+'" y="'+(y+4)+'" class="cl-tick" text-anchor="end">'+tickLabel(v,max,lt)+'</text>';
  }
  (o.refs||[]).forEach(function(rf){
    var y=yOf(rf.v);
    s+='<line x1="'+mL+'" y1="'+y+'" x2="'+(W-mR)+'" y2="'+y+'" class="cl-floor"/>';
  });
  o.series.forEach(function(se){
    if (se.band && se.band.length===se.v.length) {
      var up='',dn='';
      se.v.forEach(function(v,i){ up+=(i?'L':'M')+xOf(i)+' '+yOf(v+se.band[i]); });
      for (var j=se.v.length-1;j>=0;j--) dn+='L'+xOf(j)+' '+yOf(se.v[j]-se.band[j]);
      s+='<path d="'+up+dn+'Z" fill="'+se.c+'" opacity="0.10"/>';
    }
    var d='';
    se.v.forEach(function(v,i){ d+=(i?'L':'M')+xOf(i)+' '+yOf(v); });
    var dash = se.dash ? ' stroke-dasharray="5 3"' : '';
    s+='<path d="'+d+'" fill="none" stroke="'+se.c+'" stroke-width="2"'+dash+' stroke-linejoin="round" stroke-linecap="round"/>';
    se.v.forEach(function(v,i){
      if (i%3===0||i===se.v.length-1)
        s+='<circle cx="'+xOf(i)+'" cy="'+yOf(v)+'" r="4" fill="'+se.c+'" stroke="#fff" stroke-width="2" class="cl-bar" data-tip="'+esc(se.n)+' · turn '+o.x[i]+' · '+v.toFixed(3)+'"><title>'+esc(se.n)+' turn '+o.x[i]+': '+v.toFixed(3)+'</title></circle>';
    });
  });
  o.x.forEach(function(xv,i){
    if (i%3===0||i===n-1) s+='<text x="'+xOf(i)+'" y="'+(y0+18)+'" class="cl-cat" text-anchor="middle">'+xv+'</text>';
  });
  s+='<line x1="'+mL+'" y1="'+y0+'" x2="'+(W-mR)+'" y2="'+y0+'" class="cl-axisline"/>';
  s+='<text class="cl-axis" transform="translate(15,'+(mT+ph/2)+') rotate(-90)" text-anchor="middle">'+esc(o.ylabel||'')+'</text>';
  s+='<text x="'+(mL+pw/2)+'" y="'+(y0+40)+'" class="cl-axis" text-anchor="middle">'+esc(o.xlabel||'')+'</text>';
  return s+'</svg>';
}

function modelSeriesWithBase(baseName, baseKey, valsFor) {
  return [{ n:baseName, c:GRAY1, v:valsFor(baseKey) }].concat(
    MODEL_KEYS.map(function(m){ return { n:MNAME[m], c:MCOL[m], v:valsFor(m) }; })
  );
}

var T7_GATE_LABELS = [
  'Gate 1\nHygiene',
  'Gate 5\n<30% identity',
  'Gate 3\nscRMSD ≤ 2 Å',
  'Gate 4\nFold / motif',
  'Gate 2\npLDDT ≥ 80'
];

var FIGS = [
 { id:'f1', t:'1. Forward folding', star:1,
   p:['Models predict contacts and secondary structure from one anonymised sequence, reconstructed to 3D and scored by TM-score across the five novelty strata; single-sequence ESMFold-2 (grey) is the baseline. ESMFold-2 reaches 0.88–0.91 TM on four strata (0.82 on truly novel), while reasoning models remain below the 0.50 fold-match line.'],
   build:function(){
     var ns=stratumN(1);
     return groupedBars({ ylabel:'TM-score vs native',
       cats:STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}),
       series:modelSeriesWithBase('ESMFold-2', 'baseline_esmfold2', function(m){ return byStratum(1,m,'tm_score'); }),
       refs:[{v:0.5,l:'fold match 0.50'}],
       fmt:function(v){return v.toFixed(3);}, aria:'TM-score by novelty stratum with ESMFold-2 baseline' });
   } },
 { id:'f2', t:'2. Inverse folding', star:2,
   p:['Each method designs a sequence for a stripped Cα-only backbone (all residues UNK), refolded with single-sequence ESMFold-2 and scored by scTM. ProteinMPNN (grey) reaches 0.88–0.94 scTM across all strata (above the 0.885 native refolding ceiling); reasoning models average 0.43–0.46 scTM.'],
   build:function(){
     var ns=stratumN(2), ref=(FIGDATA.t2_base||[])[0];
     return groupedBars({ ylabel:'ESMFold-2 refolding scTM',
       cats:STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}),
       series:modelSeriesWithBase('ProteinMPNN', 'baseline_proteinmpnn', function(m){ return byStratum(2,m,'sc_tm_score'); }),
       refs:[{v:0.5,l:'fold match 0.50'}].concat(ref?[{v:ref.v,l:ref.n+' '+ref.v}]:[]),
       fmt:function(v){return v.toFixed(3);}, aria:'Refolding scTM by novelty stratum with ProteinMPNN baseline' });
   } },
 { id:'f3', t:'3. Remote homology', star:3,
   p:['__T3_PROSE__'],
   build:function(){
     var S=FIGDATA.t3.series.map(function(se){
       return { n:(MNAME[se.n]||se.n)+' (AUROC '+se.auroc+')', c:se.base?GRAY1:MCOL[se.n], v:se.v };
     });
     return groupedBars({ ylabel:'Recall @ 10% FPR', cats:FIGDATA.t3.rungs,
       series:S, fmt:function(v){return v.toFixed(2);}, aria:'Recall at 10% FPR by rung' });
   } },
 { id:'f4', t:'4. Function prediction', star:4,
   p:['__T4_PROSE__'],
   build:function(){
     var S=FIGDATA.t4.series.map(function(se){
       return { n:(MNAME[se.n]||se.n)+' ('+se.macro+')', c:se.base?GRAY1:MCOL[se.n], v:se.v };
     });
     return groupedBars({ ylabel:'IC-weighted wFₘₐₓ',
       cats:['Molecular\nfunction','Biological\nprocess','Cellular\ncomponent'],
       series:S, fmt:function(v){return v.toFixed(3);}, aria:'IC-weighted wFmax by GO aspect' });
   } },
 { id:'f5', t:'5. Monomer alanine ranking', star:5,
   p:['NDCG@5 on 18-mutation X→Ala panels with same-AA decoys (0.50 floor) across four experimental regimes. ESM-2 (650M) masked marginals (grey) score 0.707 on natural folds but drop to 0.548 on zero-homolog de novo miniproteins, where reasoning models pull ahead.'],
   build:function(){
     var ns=stratumNList(T5_STRATA, 5);
     return groupedBars({ ylabel:'NDCG@5 across strata',
       cats:T5_STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}),
       series:modelSeriesWithBase('ESM-2 (650M)', 'baseline_esm2_650m', function(m){ return byStratumList(T5_STRATA,5,m,'ndcg_at_5'); }),
       refs:[{v:0.5,l:'decoy floor 0.50'}], fmt:function(v){return v.toFixed(3);},
       aria:'Monomer alanine ranking NDCG@5 by protein type with ESM-2 650M baseline' });
   } },
 { id:'f6', t:'6. Interface binding ranking', star:6,
   p:['Spearman ρ for ranking 18 interface X→Ala mutations (monomer-destabilising sites excluded), split by pre-2026 PDB presence (n=31 vs n=7). ESM-2 (650M) stays flat at ρ ≈ 0.16 on both splits, while reasoning models drop from 0.44–0.52 on known complexes to 0.10–0.16 on unseen/post-2026 ones.'],
   build:function(){
     var S=FIGDATA.t6.series.map(function(se){
       var memo=se.memo||{};
       return { n:(MNAME[se.n]||se.n), c:se.base?GRAY1:MCOL[se.n],
                v:[memo.known_pdb_pre2026, memo.unseen_or_post2026] };
     });
     return groupedBars({ ylabel:'Spearman ρ',
       cats:['Known complex\n(pre-2026 PDB)','Unseen /\npost-2026'],
       subCats:['n=31','n=7'],
       series:S, fmt:function(v){return v===null?'—':v.toFixed(3);},
       aria:'Spearman rho by memorisation stratum with ESM-2 650M baseline' });
   } },
 { id:'f7', t:'7. De novo and motif design', star:7,
   p:['Single-shot pass rates across five gates on 40 specifications (24 fold-conditioned, 16 motif scaffolds). With one unscreened sample per task, RFdiffusion3 + ProteinMPNN (grey) reaches 48% fold/motif adherence (mean motif RMSD 1.45 Å) and 5% pLDDT ≥ 80; text models remain at 0% on pLDDT ≥ 80.'],
   build:function(){
     return groupedBars({ ylabel:'pass fraction (n=40)',
       cats:T7_GATE_LABELS,
       series:FIGDATA.t7.series.map(function(se){ return {n:MNAME[se.n]||se.n,c:se.base?GRAY1:MCOL[se.n],v:se.v}; }),
       fmt:function(v){return (v*100).toFixed(0)+'%';}, aria:'Design gate pass fractions with RFdiffusion3 + ProteinMPNN baseline' });
   } },
 { id:'f8', t:'8. Closed-loop binder design', star:8,
   p:['__T8_PROSE__'],
   build:function(){
     var tr=FIGDATA.t8.traj;
     var xs=[]; for (var i=1;i<=16;i++) xs.push(i);
     var order=['baseline_boltzgen'].concat(MODEL_KEYS).filter(function(m){return tr[m];});
     return lineChart({ ylabel:'running-best interface score', xlabel:'proposal / closed-loop turn (1 … 16)', x:xs,
       series:order.map(function(m){
         var d=tr[m], isB=(m==='baseline_boltzgen');
         var nm=isB?'BoltzGen':MNAME[m];
         return { n:nm+' ('+d.mean[0].toFixed(2)+' → '+d.mean[d.mean.length-1].toFixed(2)+')',
                  c:isB?GRAY1:MCOL[m], v:d.mean, band:d.sem, dash:isB };
       }),
       refs:[{v:0.75,l:'hit threshold 0.75'}], aria:'Closed-loop optimisation trajectory with BoltzGen baseline' });
   } },
 { id:'f9', t:'8b. First shot versus best of sixteen', star:8,
   p:['Mean AlphaFold2 composite score on shot 1 (blind) versus the best of 16 proposals. Sixteen shots roughly double scores across all methods, though averages remain below the 0.75 on-epitope hit threshold.'],
   build:function(){
     var tr=FIGDATA.t8.traj;
     var ks=['baseline_boltzgen'].concat(MODEL_KEYS).filter(function(m){return tr[m];});
     return groupedBars({ ylabel:'AF2 interface score',
       cats:ks.map(function(m){ return m==='baseline_boltzgen'?'BoltzGen':MNAME[m]; }),
       series:[{n:'first shot (blind)',c:GRAY1,v:ks.map(function(m){return tr[m].mean[0];})},
               {n:'best of 16',c:'#2a78d6',v:ks.map(function(m){return tr[m].best;})}],
       refs:[{v:0.75,l:'hit threshold 0.75'}], fmt:function(v){return v.toFixed(3);},
       aria:'First shot versus best of sixteen with BoltzGen baseline' });
   } }
];

function renderFigures() {
  var host=document.getElementById('results-figures');
  if (!host || typeof DS==='undefined' || typeof FIGDATA==='undefined') return;
  host.innerHTML = FIGS.map(function(f){
    var svg='';
    try { svg=f.build(); } catch(e){ svg='<p class="muted small">This figure could not be drawn.</p>'; }
    return '<figure class="fig" id="'+f.id+'"><h3>'+esc(f.t)+(f.star ? star(f.star) : '')+'</h3>'+
           f.p.map(function(x){ return '<p class="fig-p">'+x+'</p>'; }).join('')+
           '<div class="fig-plot">'+svg+'</div></figure>';
  }).join('');
  var tip=document.getElementById('cl-tooltip');
  if (!tip){ tip=document.createElement('div'); tip.id='cl-tooltip'; tip.className='cl-tooltip'; document.body.appendChild(tip); }
  host.querySelectorAll('.cl-bar').forEach(function(b){
    b.addEventListener('mouseenter',function(ev){
      tip.textContent=b.getAttribute('data-tip'); tip.style.opacity='1';
      var r=ev.target.getBoundingClientRect();
      tip.style.left=(r.left+r.width/2+window.scrollX)+'px';
      tip.style.top=(r.top+window.scrollY-34)+'px';
    });
    b.addEventListener('mouseleave',function(){ tip.style.opacity='0'; });
  });
}

function renderDetailCharts() {
  var host = document.getElementById('results-detail');
  if (!host || typeof DS === 'undefined') return;

  var leg = '<div class="sm-legend">' + SERIES.map(function(se){
    return '<span><i style="background:' + se.c + '"></i>' + esc(se.n) + '</span>';
  }).join('') + '<span class="sm-hint">grey dot = task specialist baseline</span></div>';

  host.innerHTML = leg + DETAIL_SPEC.map(function(spec){
    var tb = TASK_BASELINE[spec.t];
    var rows = spec.metrics.map(function(metric){
      var names = [];
      var vals = SERIES.map(function(se){
        var key = se.base ? (tb ? tb.k : '') : (se.k + (spec.suffix || ''));
        names.push(se.base ? (tb ? tb.n : se.n) : se.n);
        if (!key) return null;
        var S = scoresFor(spec.t, key);
        if (!S.length && spec.t !== 4) return null;
        var v = metric.get(S, key);
        return (v === null || v === undefined || isNaN(v)) ? null : v;
      });
      return meterRow(metric, vals, names);
    }).join('');
    return '<div class="mt-task"><h4>' + esc(spec.name) + '</h4>' + rows + '</div>';
  }).join('');

  var tip = document.getElementById('cl-tooltip');
  if (!tip) { tip = document.createElement('div'); tip.id = 'cl-tooltip'; tip.className = 'cl-tooltip'; document.body.appendChild(tip); }
  host.querySelectorAll('.cl-bar').forEach(function(b){
    b.addEventListener('mouseenter', function(ev){
      tip.textContent = b.getAttribute('data-tip'); tip.style.opacity = '1';
      var rc = ev.target.getBoundingClientRect();
      tip.style.left = (rc.left + rc.width/2 + window.scrollX) + 'px';
      tip.style.top  = (rc.top + window.scrollY - 34) + 'px';
    });
    b.addEventListener('mouseleave', function(){ tip.style.opacity = '0'; });
  });
}

function renderResults() {
  var host = document.getElementById('results-tables');
  if (!host || typeof DS === 'undefined') return;
  host.innerHTML = RESULT_SPEC.map(function(spec){
    var head = '<tr><th>Model / Baseline</th>' + spec.cols.map(function(c){ return '<th class="r">' + c[0] + '</th>'; }).join('') + '</tr>';
    var tb = TASK_BASELINE[spec.t];
    var rowSpecs = (tb ? [{ k: tb.k, n: tb.n + ' (baseline)', isBase: true }] : []).concat(
      MODELS.map(function(m){ return { k: m.k + (spec.suffix || ''), n: m.n, isBase: false }; })
    );
    var body = rowSpecs.map(function(r){
      var S = scoresFor(spec.t, r.k);
      if (!S.length && spec.t !== 4) return '';
      var style = r.isBase ? ' style="color:#5f6773"' : '';
      return '<tr' + style + '><td>' + esc(r.n) + '</td>' +
        spec.cols.map(function(c){ return '<td class="r">' + c[1](S, r.k) + '</td>'; }).join('') + '</tr>';
    }).join('');
    return '<div class="restab"><h3>' + spec.name + '</h3>' +
           '<p class="rnote">' + spec.note + '</p>' +
           '<table class="plain"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>';
  }).join('');
}

function initDataset() {
  if (typeof DS === 'undefined' || !DS.length) return;
  dsRenderCounts();
  dsRenderControls();
  dsRenderTable();
  dsRenderDetail();
  var q = document.getElementById('ds-q'), qt;
  q.addEventListener('input', function(){
    clearTimeout(qt);
    qt = setTimeout(function(){ dsQuery = q.value; dsSel = null; dsRenderTable(); }, 140);
  });
  document.querySelectorAll('.ds-table thead th').forEach(function(th){
    th.addEventListener('click', function(ev){
      if (ev && ev.target && ev.target.closest && ev.target.closest('a.star')) return;
      var k = th.getAttribute('data-sort');
      if (!k) return;
      if (dsSort === k) dsAsc = !dsAsc; else { dsSort = k; dsAsc = true; }
      dsRenderTable();
    });
  });
}

function boot(label, fn) {
  try { fn(); } catch (e) {
    try { console.error('ProteinBench: ' + label + ' failed —', e && (e.stack || e.message)); } catch (_) {}
  }
}
boot('figures', renderFigures);
boot('detail charts', renderDetailCharts);
boot('results', renderResults);
boot('explorer', renderExplorer);
boot('datapoints', initDataset);
loadMolstar(function(ok){
  MOLSTAR_OK = ok;
  if (!ok) return;
  boot('explorer (Mol*)', renderExplorer);
  boot('datapoint structure', dsRenderDetail);
});
let rzT;
window.addEventListener('resize', () => {
  clearTimeout(rzT);
  rzT = setTimeout(renderExplorer, 180);
});


</script>
</body>
</html>"""


def _apply_redesign_polish(raw_html: str) -> str:
    """Apply the exact CSS, per-task block wrappers, Task 8 PDB rows, and star-link scroll/flash polish."""
    import re

    # 1. Card summary radius flattening when open + block wrapper CSS
    old_disc_css = """  details.disc>summary{list-style:none;cursor:pointer;position:relative;padding:15px 46px 15px 18px;}
  details.disc>summary::-webkit-details-marker{display:none;}
  details.disc>summary::after{content:'';position:absolute;right:20px;top:50%;width:7px;height:7px;
    border-right:1.6px solid var(--muted);border-bottom:1.6px solid var(--muted);
    transform:translateY(-75%) rotate(45deg);transition:transform .16s ease;}
  details.disc[open]>summary::after{transform:translateY(-25%) rotate(-135deg);}
  details.disc>summary:hover{background:var(--soft);border-radius:9px;}
  details.disc[open]>summary:hover{border-radius:9px 9px 0 0;}
  .disc-t{display:block;font-family:var(--serif);font-size:18.5px;color:var(--ink);}
  .disc-s{display:block;font-size:13.5px;color:var(--muted);margin-top:3px;}
  .disc-b{padding:2px 18px 20px;border-top:1px solid var(--rule2);}
  .disc-h{font-family:var(--sans);font-size:12px;letter-spacing:.09em;text-transform:uppercase;
    color:var(--muted);font-weight:600;margin:30px 0 10px;scroll-margin-top:90px;
    padding-bottom:7px;border-bottom:1px solid var(--rule2);}
  .disc-b>.disc-h:first-of-type{margin-top:20px;}
  .disc-intro{margin-top:16px;}
  .disc-b table.plain{font-size:14px;}"""
    new_disc_css = """  details.disc>summary{list-style:none;cursor:pointer;position:relative;padding:15px 46px 15px 18px;border-radius:9px;}
  details.disc>summary::-webkit-details-marker{display:none;}
  details.disc>summary::after{content:'';position:absolute;right:20px;top:50%;width:7px;height:7px;
    border-right:1.6px solid var(--muted);border-bottom:1.6px solid var(--muted);
    transform:translateY(-75%) rotate(45deg);transition:transform .16s ease;}
  details.disc[open]>summary::after{transform:translateY(-25%) rotate(-135deg);}
  details.disc[open]>summary{border-radius:9px 9px 0 0;}
  details.disc>summary:hover{background:var(--soft);border-radius:9px;}
  details.disc[open]>summary:hover{border-radius:9px 9px 0 0;}
  .disc-t{display:block;font-family:var(--serif);font-size:18.5px;color:var(--ink);}
  .disc-s{display:block;font-size:13.5px;color:var(--muted);margin-top:3px;}
  .disc-b{padding:2px 18px 20px;border-top:1px solid var(--rule2);}
  .disc-blk{scroll-margin-top:86px;border-radius:6px;padding:4px 10px 6px;margin:0 -10px;}
  .disc-blk p:last-child{margin-bottom:2px;}
  .disc-h{font-family:var(--sans);font-size:12px;letter-spacing:.09em;text-transform:uppercase;
    color:var(--muted);font-weight:600;margin:24px 0 10px;scroll-margin-top:86px;
    padding-bottom:7px;border-bottom:1px solid var(--rule2);}
  .disc-blk:first-of-type .disc-h,.disc-b>.disc-h:first-of-type{margin-top:16px;}
  .disc-intro{margin-top:16px;}"""
    assert old_disc_css in raw_html, "Expected details.disc CSS block not found"
    raw_html = raw_html.replace(old_disc_css, new_disc_css, 1)

    # 2. Page tables 15px, panel tables 14px
    old_tbl_css = "  table.plain{width:100%;border-collapse:collapse;font-size:14.5px;}"
    new_tbl_css = "  table.plain{width:100%;border-collapse:collapse;font-size:15px;}\n  .disc-b table.plain{font-size:14px;}"
    assert old_tbl_css in raw_html, "Expected table.plain CSS line not found"
    raw_html = raw_html.replace(old_tbl_css, new_tbl_css, 1)

    # 3. Inline mono-grey .ds-key in prose vs spaced .ds-field .ds-key in browser
    old_key_css = "  .ds-key{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:10px;}"
    new_key_css = "  .ds-key{font-family:var(--mono);font-size:11.5px;color:var(--muted);}\n  .ds-field .ds-key{font-size:11px;margin-left:10px;}"
    assert old_key_css in raw_html, "Expected .ds-key CSS line not found"
    raw_html = raw_html.replace(old_key_css, new_key_css, 1)

    # 4. Replace verbose #prep body with concise, accurate descriptions (keeping all 8 blocks, tables, Task 8 PDB codes, and mono-grey stratum keys)
    concise_prep_inner = """    <p class="disc-intro">Every prompt strips PDB IDs, gene symbols, organism names, and alignments. All 522 targets (702 calls per track) are filterable by stratum in the datapoint browser.</p>

    <div class="disc-blk" id="prep-t1">
    <h3 class="disc-h">Task 1 &middot; Forward folding &mdash; 100 targets</h3>
    <p>Each candidate was structurally searched against the pre-May-2026 PDB; its highest TM-score to any prior chain sets its structural novelty (TM &gt; 0.5 shares a fold; &lt; 0.3 is random). Designed proteins are grouped separately by provenance.</p>
    <div style="overflow-x:auto">
    <table class="plain">
      <thead><tr><th>Stratum</th><th class="r">n</th><th>Deposited</th><th>Best pre-cutoff match</th><th>Definition</th></tr></thead>
      <tbody>
        <tr><td><b>Known</b> (pre-2026)</td><td class="r">15</td><td>1987 &ndash; 2020</td><td>TM 0.87 &ndash; 1.00</td><td>Pre-cutoff PDB positive control.</td></tr>
        <tr><td><b>Homolog gradient</b></td><td class="r">12</td><td>2026</td><td>TM 0.65 &ndash; 0.94</td><td>New entries with a close pre-cutoff structural homolog.</td></tr>
        <tr><td><b>PDB-novel</b></td><td class="r">27</td><td>2026</td><td>TM 0.53 &ndash; 0.81</td><td>Natural 2026 proteins with only partial structural precedent.</td></tr>
        <tr><td><b><em>De novo</em></b></td><td class="r">28</td><td>2026</td><td>TM 0.45 &ndash; 0.91</td><td>Computationally designed folds with no evolutionary homologs.</td></tr>
        <tr><td><b>Truly novel</b></td><td class="r">18</td><td>2026</td><td>TM 0.33 &ndash; 0.53</td><td>No pre-cutoff PDB chain reaches the 0.50 same-fold line.</td></tr>
      </tbody>
    </table>
    </div>
    </div>

    <div class="disc-blk" id="prep-t2">
    <h3 class="disc-h">Task 2 &middot; Inverse folding &mdash; 100 targets</h3>
    <p>Same 100 proteins and five strata as Task 1. Inputs keep only C&alpha; coordinates (every residue relabelled UNK), backbone dihedrals, and C&alpha; burial counts. Wild-type sequences were refolded through the same ESMFold-2 pipeline to set the 0.885 ceiling.</p>
    </div>

    <div class="disc-blk" id="prep-t3">
    <h3 class="disc-h">Task 3 &middot; Remote homology &mdash; 120 pairs</h3>
    <p>SCOPe domain pairs split by which search method fails (30 pairs per rung; rungs 2&ndash;4 matched near 11% identity). Released files carry SCOPe families, sequence identity, and Smith&ndash;Waterman scores; the Foldseek E-values separating rungs 2 and 3 are not stored in the file.</p>
    <div style="overflow-x:auto">
    <table class="plain">
      <thead><tr><th>Rung</th><th class="r">n</th><th>Sequence identity</th><th>Selection rule</th></tr></thead>
      <tbody>
        <tr><td><b>1 &middot; Sequence-easy</b></td><td class="r">30</td><td>18 &ndash; 38% (med. 23%)</td><td>Same SCOPe superfamily; recovered by Smith&ndash;Waterman.</td></tr>
        <tr><td><b>2 &middot; Structure-caught</b></td><td class="r">30</td><td>9 &ndash; 14% (med. 11%)</td><td>Same superfamily; sequence alignment fails, Foldseek 3Di hits.</td></tr>
        <tr><td><b>3 &middot; Deep twilight</b></td><td class="r">30</td><td>8 &ndash; 16% (med. 11%)</td><td>Same superfamily; both sequence and Foldseek search fail.</td></tr>
        <tr><td><b>4 &middot; Matched decoy</b></td><td class="r">30</td><td>9 &ndash; 14% (med. 11%)</td><td>Different superfamilies matched on structural class and identity.</td></tr>
      </tbody>
    </table>
    </div>
    </div>

    <div class="disc-blk" id="prep-t4">
    <h3 class="disc-h">Task 4 &middot; Function prediction &mdash; 60 proteins</h3>
    <p>QuickGO targets graded strictly on experimentally evidenced Gene Ontology terms. Date caveat: <span class="ds-key">post_may_2026</span> (n=11) has 7 entries dated 2026-07-25 and 4 pre-cutoff entries (2025-11-19 / 2026-03-31); <span class="ds-key">early_2026_uniprot</span> (n=49) spans 2022-02-03 to 2026-01-08 with 47 sharing the bulk date 2025-10-27. The true post-cutoff holdout is <b>7 proteins</b>.</p>
    </div>

    <div class="disc-blk" id="prep-t5">
    <h3 class="disc-h">Task 5 &middot; Monomer alanine ranking &mdash; 52 panels</h3>
    <p>Each 18-candidate panel pairs 6 hotspots, 6 moderate sites, and 6 same-AA decoys (holding residue-letter lookup to 50%), drawn across four experimental readouts.</p>
    <div style="overflow-x:auto">
    <table class="plain">
      <thead><tr><th>Source</th><th class="r">n</th><th>Measured quantity</th></tr></thead>
      <tbody>
        <tr><td><b><em>De novo</em> miniproteins</b></td><td class="r">24</td><td>Folding &Delta;&Delta;G (MegaScale); designed folds with no natural homologs.</td></tr>
        <tr><td><b>Natural proteins</b></td><td class="r">14</td><td>Folding &Delta;&Delta;G on natural folds.</td></tr>
        <tr><td><b>MaveDB deep scans</b></td><td class="r">12</td><td>Post-cutoff deep mutational scanning activity.</td></tr>
        <tr><td><b>Stability and function</b></td><td class="r">2</td><td>Paired stability and activity on the same protein.</td></tr>
      </tbody>
    </table>
    </div>
    </div>

    <div class="disc-blk" id="prep-t6">
    <h3 class="disc-h">Task 6 &middot; Interface binding ranking &mdash; 38 complexes</h3>
    <p>Interface X&rarr;Ala panels graded on measured binding &Delta;&Delta;G after removing monomer-destabilising mutations. Split by pre-2026 PDB presence: 31 in <span class="ds-key">known_pdb_pre2026</span> and 7 in <span class="ds-key">unseen_or_post2026</span>.</p>
    </div>

    <div class="disc-blk" id="prep-t7">
    <h3 class="disc-h">Task 7 &middot; De novo and motif design &mdash; 40 specifications</h3>
    <p>Forty written specs graded on five single-shot gates: sequence hygiene, &lt;30% identity to natural parents, ESMFold-2/OpenFold3 agreement (scRMSD &le; 2 &Aring;), pLDDT &ge; 80, and the structural gate below.</p>
    <div style="overflow-x:auto">
    <table class="plain">
      <thead><tr><th>Kind</th><th class="r">n</th><th>Length</th><th>Structural gate</th></tr></thead>
      <tbody>
        <tr><td><b>Fold-conditioned</b></td><td class="r">24</td><td>64 &ndash; 136 AA</td><td>Refolded design reaches TM &ge; 0.50 against the target fold.</td></tr>
        <tr><td><b>Motif scaffolding</b></td><td class="r">16</td><td>70 &ndash; 128 AA</td><td>Held crystal motif stays within C&alpha; RMSD &le; 1.5 &Aring;.</td></tr>
      </tbody>
    </table>
    </div>
    </div>

    <div class="disc-blk" id="prep-t8">
    <h3 class="disc-h">Task 8 &middot; Closed-loop binder design &mdash; 12 epitopes</h3>
    <p>Twelve target chains with five hotspot residues each; methods get 16 binder proposals (55&ndash;85 AA) scored by AlphaFold2 and ESMFold-2. Only the third group is a post-cutoff crystal holdout.</p>
    <div style="overflow-x:auto">
    <table class="plain">
      <thead><tr><th>Group</th><th class="r">n</th><th>Targets &amp; PDB codes</th><th>Role</th></tr></thead>
      <tbody>
        <tr><td><b>ProtDBench classic</b></td><td class="r">4</td><td><span style="white-space:nowrap">PD-L1, TrkA, IL-7R&alpha;, PDGFR&beta;</span><br><span class="muted">PDB 5O45, 1WWW, 3DI3, 3MJG</span></td><td>Standard prior-art binder targets (2001&ndash;2017).</td></tr>
        <tr><td><b>BoltzProt1 novel</b></td><td class="r">5</td><td><span style="white-space:nowrap">AMBP, GM2A, HNMT, MZB1, RFK</span><br><span class="muted">PDB 3QKG, 1G13, 1JQD, 7AAH, 1NB0</span></td><td>Solved folds repurposed as non-standard binder targets (2001&ndash;2021).</td></tr>
        <tr><td><b>Post-May-2026 crystal</b></td><td class="r">3</td><td><span style="white-space:nowrap">LEDGF PWWP, Aurora A, MCL1</span><br><span class="muted">PDB 9S83, 9RVK, 9TLV</span></td><td>Post-cutoff 2026 crystal complexes with held-out binders.</td></tr>
      </tbody>
    </table>
    </div>
    </div>"""
    disc_b_pattern = re.compile(
        r'(<div class="disc-b">\n)(.*?)(\n    </div>\n    </details>)',
        re.DOTALL,
    )
    m = disc_b_pattern.search(raw_html)
    assert m is not None, "Expected .disc-b body not found"
    raw_html = raw_html[: m.start()] + m.group(1) + concise_prep_inner + m.group(3) + raw_html[m.end() :]

    # 5. Trim TASK_BLURB in Task Explorer to concise, accurate task descriptions
    concise_task_blurb = """var TASK_BLURB = {
  '1':'Input: one anonymised sequence (no IDs or alignments). Models predict residue contacts and secondary structure, reconstructed to 3D and superposed on the native fold.',
  '2':'Input: a stripped Cα backbone (all residues UNK) with dihedrals and burial counts. Models design a sequence, which ESMFold-2 refolds and superposes back onto the input.',
  '3':'Input: two sequences, often below 15% identity. Models predict whether they share a SCOPe superfamily, graded against class- and identity-matched decoys.',
  '4':'Input: one anonymised sequence. Models predict Gene Ontology terms across MF, BP, and CC, graded by IC-weighted wFmax against experimentally evidenced annotations.',
  '5':'Input: one sequence and 18 X→Ala mutations (6 hotspots, 6 moderate, 6 same-AA decoys). Models rank candidates by measured fold destabilisation (ΔΔG or DMS activity).',
  '6':'Input: two interacting chains and 18 interface X→Ala mutations (monomer-destabilising sites excluded). Models rank candidates by measured binding loss.',
  '7':'Input: a fold topology spec or a crystal motif to scaffold. Each single-shot sequence is graded on five gates: hygiene, <30% identity, cross-oracle agreement, geometry, and pLDDT ≥ 80.',
  '8':'Input: a target chain and five epitope hotspots. Over 16 rounds, models propose 55–85 AA binders and receive AlphaFold2 interface scores after each attempt.'
};"""
    blurb_pattern = re.compile(r"var TASK_BLURB = \{.*?\};", re.DOTALL)
    assert blurb_pattern.search(raw_html) is not None, "Expected TASK_BLURB block not found"
    raw_html = blurb_pattern.sub(lambda _: concise_task_blurb, raw_html, count=1)

    # 6. Prevent meter-row value pills from clashing/overflowing with 4 series on narrower screens
    old_mt_css = "  .mt-vals{display:flex;gap:12px;}"
    new_mt_css = "  .mt-vals{display:flex;flex-wrap:wrap;gap:4px 10px;justify-content:flex-end;}"
    assert old_mt_css in raw_html, "Expected .mt-vals CSS line not found"
    raw_html = raw_html.replace(old_mt_css, new_mt_css, 1)

    # 7. Flash #prep-d cleanly when clicking #prep (browser column header *) and #prep-tK when clicking task *
    old_open_prep = "    var t = document.getElementById(id) || document.getElementById('prep');"
    new_open_prep = "    var t = (id === 'prep' ? d : document.getElementById(id)) || d;"
    assert old_open_prep in raw_html, "Expected openPrep target lookup not found"
    raw_html = raw_html.replace(old_open_prep, new_open_prep, 1)

    return raw_html


def main() -> None:
    backup_html = REPO / "website" / "proteinbench_redesign_backup.html"
    src_html = REPO / "website" / "proteinbench_redesign.html"
    if not backup_html.exists():
        shutil.copyfile(src_html, backup_html)

    raw_html = _apply_redesign_polish(backup_html.read_text(encoding="utf-8"))
    lines = raw_html.splitlines()

    # Locate DS and FIGDATA lines
    ds_line_idx = next(i for i, ln in enumerate(lines) if ln.startswith("const DS = "))
    fd_line_idx = next(i for i, ln in enumerate(lines) if ln.startswith("const FIGDATA = "))
    tail_start_idx = next(i for i, ln in enumerate(lines) if ln.startswith("var MODELS = [")) - 4

    ds: List[Dict[str, Any]] = json.loads(lines[ds_line_idx][len("const DS = ") :].rstrip(";"))
    fd: Dict[str, Any] = json.loads(lines[fd_line_idx][len("const FIGDATA = ") :].rstrip(";"))

    # --- Load all 8 baseline results ---
    # 1. Task 1: ESMFold-2
    t1_res = json.loads((REPO / "outputs" / "results" / "baseline_t1_esmfold2.json").read_text(encoding="utf-8"))
    t1_per = t1_res["per_target"]

    # 2. Task 2: ProteinMPNN
    t2_rows = {
        json.loads(ln)["target_id"]: json.loads(ln)
        for ln in (REPO / "outputs" / "predictions" / "baseline_proteinmpnn_stage1b.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    }

    # 3 & 4. Tasks 3 & 4: ESM-2 (650M) & Smith-Waterman homology
    t34_res = json.loads((REPO / "outputs" / "results" / "baseline_t3_t4_esmc_and_sw.json").read_text(encoding="utf-8"))
    t3_esm2 = t34_res["t3_esm2_650m"]
    t3_sw = t34_res["t3_sw"]
    # Pick whichever established sequence-only baseline (ESM-2 650M vs BLOSUM62 SW) is stronger/cleaner,
    # preferring ESM-2 (650M) since BLOSUM62 was used to define Rung 1 vs Rung 2!
    if t3_esm2["overall_auroc"] >= 0.70:
        t3_base_name = "ESM-2 (650M)"
        t3_per_raw = t34_res["t3_esm2_rows"]
        t3_chosen = t3_esm2
        t3_prose = (
            f"Recall at 10% FPR across three SCOPe superfamily difficulty rungs paired with matched decoys. "
            f"All methods saturate on Rung 1; ESM-2 (650M) cosine similarity (grey, AUROC {t3_chosen['overall_auroc']:.3f}) "
            f"provides the sequence embedding baseline, while reasoning models separate on Rungs 2–3."
        )
    else:
        t3_base_name = "BLOSUM62 SW"
        t3_per_raw = t34_res["t3_sw_rows"]
        t3_chosen = t3_sw
        t3_prose = (
            f"Recall at 10% FPR across three SCOPe superfamily difficulty rungs paired with matched decoys. "
            f"BLOSUM62 Smith–Waterman (grey, AUROC {t3_chosen['overall_auroc']:.3f}) provides the local-alignment baseline."
        )
    t3_base_series = {
        "n": t3_base_name,
        "base": True,
        "auroc": round(float(t3_chosen["overall_auroc"]), 3),
        "v": [
            round(float(t3_chosen["by_rung"]["1_seq_easy"]["recall_at_fpr10"]), 3),
            round(float(t3_chosen["by_rung"]["2_foldseek_3di"]["recall_at_fpr10"]), 3),
            round(float(t3_chosen["by_rung"]["3_deep_tea_zone"]["recall_at_fpr10"]), 3),
        ],
    }

    # Task 4: Official CAFA Smith-Waterman sequence-homology transfer baseline
    t4_sw = t34_res["t4_models"]["baseline_sw_homology"]
    t4_esm2 = t34_res["t4_models"].get("baseline_esm2_650m_knn")
    if t4_esm2 and t4_esm2["macro_wfmax"] >= t4_sw["macro_wfmax"]:
        t4_base_name = "ESM-2 (650M) k-NN"
        t4_chosen = t4_esm2
        t4_prose = (
            f"IC-weighted CAFA wFₘₐₓ across MF, BP, and CC on experimentally evidenced GO annotations. "
            f"ESM-2 (650M) k-NN transfer against pre-2025 Swiss-Prot (grey, macro {t4_chosen['macro_wfmax']:.3f}) is the baseline."
        )
    else:
        t4_base_name = "Smith–Waterman"
        t4_chosen = t4_sw
        t4_prose = (
            f"IC-weighted CAFA wFₘₐₓ across MF, BP, and CC on experimentally evidenced GO annotations. "
            f"Smith–Waterman homology transfer against pre-2025 Swiss-Prot (grey, macro {t4_chosen['macro_wfmax']:.3f}) "
            f"is the baseline; all three reasoning models score 0.337–0.392 macro wFₘₐₓ."
        )

    # 5. Task 5: ESM-2 (650M) Monomer Alanine Marginals
    t5_res = json.loads((REPO / "outputs" / "results" / "baseline_t5_esm2_650m.json").read_text(encoding="utf-8"))
    t5_per = {g["target_id"]: g for g in t5_res["per_target"]}

    # 6. Task 6: ESM-2 (650M) Interface Binding Marginals
    t6_res = json.loads((REPO / "outputs" / "results" / "baseline_t6_esm2_650m.json").read_text(encoding="utf-8"))
    t6_per = {g["target_id"]: g for g in t6_res["per_target"]}

    # 7. Task 7: RFdiffusion3 + ProteinMPNN
    t7_rep = json.loads((REPO / "outputs" / "results" / "test7_esmfold2_evaluation_report.json").read_text(encoding="utf-8"))
    t7_b = t7_rep["models"]["baseline_rfd3_mpnn"]
    t7_per = {g["task_id"]: g for g in t7_b["per_task"]}

    # 8. Task 8: BoltzGen
    t8_b = json.loads(
        (REPO / "outputs" / "predictions" / "stage3b_binder_campaign_specialist_boltzgen_af2.json").read_text(encoding="utf-8")
    )
    t8_sum = t8_b["summary"]
    t8_per = {c["target_id"]: c for c in t8_sum["per_target"]}

    # --- Inject into DS (all 522 datapoints) ---
    for d in ds:
        t = d["t"]
        tid = d["id"]
        sc = d.setdefault("sc", {})
        if t == 1 and tid in t1_per:
            r = t1_per[tid]
            sc["baseline_esmfold2"] = {
                "tm_score": round(float(r["tm_score"]), 3),
                "p_at_l_sep24": round(float(r["p_at_l_sep24"]), 3),
                "q3_accuracy": round(float(r["q3_accuracy"]), 3),
            }
        elif t == 2 and tid in t2_rows:
            r = t2_rows[tid]
            sc["baseline_proteinmpnn"] = {
                "sc_tm_score": round(float(r["sc_tm_score"]), 3),
                "sc_rmsd_ca": round(float(r["sc_rmsd_ca"]), 2),
                "plddt": round(float(r["esmfold_plddt"]), 1),
                "nsr": round(float(r["nsr_overall"]), 3),
            }
        elif t == 3 and tid in t3_per_raw:
            sc["baseline_t3"] = {"raw": t3_per_raw[tid]}
        elif t == 4:
            if tid in t4_chosen["per_target"]:
                sc["baseline_t4"] = t4_chosen["per_target"][tid]
            for mk in ("gpt-6.1-sol_think4096", "gpt-6.2-sol_think4096", "gpt-6.2-astra_think4096"):
                if mk in sc and tid in t34_res["t4_models"][mk]["per_target"]:
                    sc[mk].update(t34_res["t4_models"][mk]["per_target"][tid])
        elif t == 5 and tid in t5_per:
            r = t5_per[tid]
            sc["baseline_esm2_650m"] = {
                "ndcg_at_5": round(float(r["ndcg_at_5"]), 3),
                "same_aa_win_rate": round(float(r["same_aa_win_rate"]), 3),
                "spearman_rho": round(float(r["spearman_rho"]), 3),
            }
        elif t == 6 and tid in t6_per:
            r = t6_per[tid]
            sc["baseline_esm2_650m"] = {
                "ndcg_at_5": round(float(r["ndcg_at_5"]), 3),
                "same_aa_win_rate": round(float(r["same_aa_win_rate"]), 3),
                "spearman_rho": round(float(r["spearman_rho"]), 3),
            }
        elif t == 7 and tid in t7_per:
            r = t7_per[tid]
            sc["baseline_rfd3_mpnn"] = {
                "plddt_mean": round(float(r["plddt_mean"] or 0.0), 2),
                "tm_to_target_fold": round(float(r["tm_to_target_fold"]), 4) if r.get("tm_to_target_fold") is not None else None,
                "motif_ca_rmsd": r.get("motif_ca_rmsd"),
                "cross_oracle_sc_rmsd": r.get("cross_oracle_sc_rmsd"),
                "esmfold2_4gate_pass": bool(r["esmfold2_4gate_pass"]),
            }
        elif t == 8 and tid in t8_per:
            r = t8_per[tid]
            sc["baseline_boltzgen_closed_loop_af2"] = {
                "hit_achieved": int(r["hit_achieved"]),
                "shots_to_first_hit": r["shots_to_first_hit"],
                "best_iptm_on_epitope": round(float(r["best_iptm_on_epitope"]), 4),
                "best_ipae_on_epitope": round(float(r["best_ipae_on_epitope"]), 2),
                "best_composite_score": round(float(r["running_best_composite"][-1]), 4),
            }

    # --- Update FIGDATA cleanly (1 established baseline per task) ---
    fd["t1_base"] = [{"n": "fold match", "v": 0.5}]
    fd["t2_base"] = [{"n": "wild-type ceiling", "v": 0.885}]

    llm_t3 = [s for s in fd["t3"]["series"] if not s.get("base")]
    fd["t3"]["series"] = [t3_base_series] + llm_t3

    llm_t4 = []
    for s in fd["t4"]["series"]:
        if not s.get("base"):
            s_copy = dict(s)
            s_copy["k"] = s["n"]
            llm_t4.append(s_copy)
    fd["t4"]["series"] = [
        {
            "k": "baseline_t4",
            "n": t4_base_name,
            "base": True,
            "macro": round(float(t4_chosen["macro_wfmax"]), 3),
            "v": [
                round(float(t4_chosen["aspects"]["MF"]), 3),
                round(float(t4_chosen["aspects"]["BP"]), 3),
                round(float(t4_chosen["aspects"]["CC"]), 3),
            ],
        }
    ] + llm_t4

    fd["t5_base"] = [
        {
            "n": "ESM-2 (650M)",
            "ndcg": round(float(t5_res["overall"]["ndcg_at_5"]), 3),
            "win": round(float(t5_res["overall"]["same_aa_win_rate"]), 3),
        }
    ]

    llm_t6 = [s for s in fd["t6"]["series"] if not s.get("base")]
    fd["t6"]["series"] = [
        {
            "n": "ESM-2 (650M)",
            "base": True,
            "overall": round(float(t6_res["overall"]["spearman_rho"]), 3),
            "memo": {
                "known_pdb_pre2026": round(float(t6_res["by_memorization_stratum"]["known_pdb_pre2026"]["spearman_rho"]), 3),
                "unseen_or_post2026": round(float(t6_res["by_memorization_stratum"]["unseen_or_post2026"]["spearman_rho"]), 3),
            },
        }
    ] + llm_t6

    # Task 7 gate order in FIGDATA.t7.gates:
    # [Gate 1 hygiene, Gate 5 <30% identity, Gate 3 cross-oracle scRMSD<=2A, Gate 4 fold/motif adherence, Gate 2 pLDDT>=80]
    llm_t7 = [s for s in fd["t7"]["series"] if not s.get("base")]
    fd["t7"]["series"] = [
        {
            "n": "RFdiffusion3 + ProteinMPNN",
            "base": True,
            "v": [
                round(float(t7_b["gate1_hygiene_rate"]), 3),
                round(float(t7_b["gate5_novelty_rate"]), 3),
                round(float(t7_b["gate3_cross_oracle_sc_rate"]), 3),
                round(float(t7_b["gate4_adherence_rate"]), 3),
                round(float(t7_b["gate2_foldable_rate"]), 3),
            ],
        }
    ] + llm_t7

    # Task 8: Add BoltzGen trajectory and ensure `.best` is `mean[-1]` (best-of-16 composite AF2 score) across all series!
    for mk, tr_obj in fd["t8"]["traj"].items():
        tr_obj["best"] = round(float(tr_obj["mean"][-1]), 4)
    fd["t8"]["traj"]["baseline_boltzgen"] = {
        "mean": t8_sum["trajectory_mean"],
        "sem": t8_sum["trajectory_sem"],
        "best": round(float(t8_sum["trajectory_mean"][-1]), 4),
        "gain": round(float(t8_sum["trajectory_mean"][-1] - t8_sum["trajectory_mean"][0]), 4),
        "hit": round(float(t8_sum["hit_rate"]), 4),
    }

    bgen_hit_pct = t8_sum["hit_rate"] * 100.0
    bgen_s0 = t8_sum["trajectory_mean"][0]
    bgen_s16 = t8_sum["trajectory_mean"][-1]
    t8_prose = (
        f"Running-best AlphaFold2 composite score (± SEM) over 16 proposals across 12 epitopes. "
        f"Reasoning models iterate on prior-round oracle feedback, while BoltzGen (grey) generates 16 all-atom designs "
        f"per epitope (rising from {bgen_s0:.2f} to {bgen_s16:.2f}; Hit@16 = {bgen_hit_pct:.1f}%)."
    )

    # Update dsRenderDetail label display so score keys render with clean display names
    head_lines = lines[:tail_start_idx]
    for i, ln in enumerate(head_lines):
        if "return '<div class=\"ds-score\"><span class=\"m\">' + esc(m) + '</span>" in ln:
            head_lines[i] = ln.replace("esc(m)", "esc((typeof SCORE_LABEL!=='undefined'&&SCORE_LABEL[m])||m)")

    head_lines[ds_line_idx] = "const DS = " + json.dumps(ds, separators=(",", ":")) + ";"
    head_lines[fd_line_idx] = "const FIGDATA = " + json.dumps(fd, separators=(",", ":")) + ";"

    js_tail = (
        NEW_JS_TAIL.replace("__T3_BASE_NAME__", t3_base_name)
        .replace("__T4_BASE_NAME__", t4_base_name)
        .replace("__T3_PROSE__", t3_prose)
        .replace("__T4_PROSE__", t4_prose)
        .replace("__T8_PROSE__", t8_prose)
    )

    full_html = "\n".join(head_lines) + "\n" + js_tail + "\n"

    # Validate the entire embedded <script> block with `node --check`!
    s0 = full_html.index("<script>") + len("<script>")
    s1 = full_html.rindex("</script>")
    js_code = full_html[s0:s1]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tf:
        tf.write(js_code)
        tmp_js = Path(tf.name)
    try:
        subprocess.run(["node", "--check", str(tmp_js)], check=True)
        print("[OK] Embedded JavaScript passed `node --check`!", flush=True)
    finally:
        tmp_js.unlink(missing_ok=True)

    # Write updated HTML to workspace targets
    for dst in (
        REPO / "website" / "index.html",
        REPO / "website" / "proteinbench_artifact.html",
    ):
        dst.write_text(full_html, encoding="utf-8")
        print(f"Wrote {dst} ({len(full_html):,} bytes)", flush=True)

    print("\n=== SUMMARY OF ESTABLISHED BASELINES INJECTED ===", flush=True)
    print(f"  Task 1 (ESMFold-2)                 : TM={t1_res['overall']['tm_score']:.3f} | P@L={t1_res['overall']['p_at_l_sep24']:.3f} | Q3={t1_res['overall']['q3_accuracy']:.3f}", flush=True)
    sc_tms = [r["sc_tm_score"] for r in t2_rows.values()]
    sc_2a = [1.0 if r["sc_rmsd_ca"] < 2.0 else 0.0 for r in t2_rows.values()]
    print(f"  Task 2 (ProteinMPNN)               : scTM={np.mean(sc_tms):.3f} | scRMSD<2A={np.mean(sc_2a)*100:.1f}% | NSR={np.mean([r['nsr_overall'] for r in t2_rows.values()]):.3f}", flush=True)
    print(f"  Task 3 ({t3_base_name:<25s}) : AUROC={t3_base_series['auroc']:.3f} | Rungs={t3_base_series['v']}", flush=True)
    print(f"  Task 4 ({t4_base_name:<25s}) : Macro wFmax={t4_chosen['macro_wfmax']:.3f} | MF/BP/CC={t4_chosen['aspects']}", flush=True)
    print(f"  Task 5 (ESM-2 650M)                : NDCG@5={t5_res['overall']['ndcg_at_5']:.3f} | SameAA={t5_res['overall']['same_aa_win_rate']:.3f} | Rho={t5_res['overall']['spearman_rho']:.3f}", flush=True)
    print(f"  Task 6 (ESM-2 650M)                : Rho={t6_res['overall']['spearman_rho']:.3f} | Known={t6_res['by_memorization_stratum']['known_pdb_pre2026']['spearman_rho']:.3f} | Unseen={t6_res['by_memorization_stratum']['unseen_or_post2026']['spearman_rho']:.3f}", flush=True)
    print(f"  Task 7 (RFdiffusion3 + ProteinMPNN): 4-Gate={t7_b['esmfold2_4gate_success_rate']*100:.1f}% | 5-Gate={t7_b['design_success_rate_5gate']*100:.1f}% | pLDDT>=80={t7_b['gate2_foldable_rate']*100:.1f}% | pLDDT={t7_b['mean_plddt']:.1f}", flush=True)
    print(f"  Task 8 (BoltzGen)                  : Hit@16={t8_sum['hit_rate']*100:.1f}% | AF2 Composite={bgen_s0:.3f} -> {bgen_s16:.3f} | On-Epitope ipTM={t8_sum['mean_best_iptm']:.3f}", flush=True)


if __name__ == "__main__":
    main()
