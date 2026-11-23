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

function legendRows(series, mL, right, y0) {
  var lx = mL, row = 0, out = '';
  series.forEach(function(se){
    if (se.line) {
      var label0 = esc(se.n), w0 = 18 + label0.length*7.3 + 22;
      if (lx > mL && lx + w0 > right) { lx = mL; row++; }
      var yy = y0 + row*17;
      out += '<line x1="'+lx+'" y1="'+(yy-4)+'" x2="'+(lx+13)+'" y2="'+(yy-4)+'" class="cl-floor"/>';
      out += '<text x="'+(lx+18)+'" y="'+yy+'" class="cl-leg">'+label0+'</text>';
      lx += w0; return;
    }
    var label = esc(se.n);
    var wNeeded = 15 + label.length * 7.3 + 22;
    if (lx > mL && lx + wNeeded > right) { lx = mL; row++; }
    var y = y0 + row * 17;
    out += '<rect x="' + lx + '" y="' + (y - 9) + '" width="10" height="10" rx="2" fill="' + se.c + '"/>';
    out += '<text x="' + (lx + 15) + '" y="' + y + '" class="cl-leg">' + label + '</text>';
    lx += wNeeded;
  });
  return { svg: out, rows: row + 1 };
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
  var W=o.w||880, H=o.h||300, mL=60, mR=o.mR||14, mT=o.mT||46, mB=o.mB||(o.twoLine?64:48);
  var max=o.maxFixed||fitMax(o.series,o.refs,o.floorMax);
  var pw=W-mL-mR, ph=H-mT-mB, y0=mT+ph;
  var nS=o.series.length, band=pw/o.cats.length;
  var BAR=Math.min(22, Math.max(8, (band-18)/nS - 2)), GAP=2;
  var inner=nS*BAR+(nS-1)*GAP;
  var yOf=function(v){ return y0-(Math.max(0,Math.min(v,max))/max)*ph; };
  var fmt=o.fmt||function(v){ return v.toFixed(2); };
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="'+esc(o.aria||'chart')+'">';

  var lgSeries=o.series.concat((o.refs||[]).map(function(r){ return {n:r.l,line:true}; }));
  var lg=legendRows(lgSeries, mL, W-mR, 23);
  if (lg.rows>1){ var extra=(lg.rows-1)*17; mT+=extra; ph=H-mT-mB+0; }
  ph=H-mT-mB; y0=mT+ph; yOf=function(v){ return y0-(Math.max(0,Math.min(v,max))/max)*ph; };
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
      s+='<text x="'+(gx+band/2)+'" y="'+(y0+18+li*13)+'" class="cl-cat" text-anchor="middle">'+esc(line)+'</text>';
    });
    if (o.subCats) s+='<text x="'+(gx+band/2)+'" y="'+(y0+(o.twoLine?46:33))+'" class="cl-metric" text-anchor="middle">'+esc(o.subCats[gi])+'</text>';
  });
  s+='<line x1="'+mL+'" y1="'+y0+'" x2="'+(W-mR)+'" y2="'+y0+'" class="cl-axisline"/>';
  s+='<text class="cl-axis" transform="translate(13,'+(mT+ph/2)+') rotate(-90)" text-anchor="middle">'+esc(o.ylabel||'')+'</text>';
  return s+'</svg>';
}

function lineChart(o) {
  var W=880,H=300,mL=60,mR=14,mT=46,mB=46;
  var max=o.maxFixed||fitMax(o.series,o.refs,o.floorMax);
  var pw=W-mL-mR, ph=H-mT-mB, y0=mT+ph, n=o.x.length;
  var xOf=function(i){ return mL+(n===1?pw/2:(i/(n-1))*pw); };
  var yOf=function(v){ return y0-(Math.max(0,Math.min(v,max))/max)*ph; };
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="'+esc(o.aria||'trajectory')+'">';
  var lg2=legendRows(o.series.concat((o.refs||[]).map(function(r){ return {n:r.l,line:true}; })), mL, W-mR, 23);
  if (lg2.rows>1){ mT+=(lg2.rows-1)*17; ph=H-mT-mB; y0=mT+ph;
    yOf=function(v){ return y0-(Math.max(0,Math.min(v,max))/max)*ph; }; }
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
  s+='<text class="cl-axis" transform="translate(13,'+(mT+ph/2)+') rotate(-90)" text-anchor="middle">'+esc(o.ylabel||'')+'</text>';
  s+='<text x="'+(mL+pw/2)+'" y="'+(y0+38)+'" class="cl-axis" text-anchor="middle">'+esc(o.xlabel||'')+'</text>';
  return s+'</svg>';
}

function modelSeriesWithBase(baseName, baseKey, valsFor) {
  return [{ n:baseName, c:GRAY1, v:valsFor(baseKey) }].concat(
    MODEL_KEYS.map(function(m){ return { n:MNAME[m], c:MCOL[m], v:valsFor(m) }; })
  );
}

var FIGS = [
 { id:'f1', t:'1. Forward folding', star:1,
   p:['One anonymised sequence goes in. Reasoning models predict residue contacts and secondary structure that are reconstructed to 3D coordinates and superposed on the native fold; ESMFold-2 (drawn in grey) folds the identical single sequence directly with no multiple-sequence alignment. The five clusters are the novelty strata set out under How the data is prepared.',
      'ESMFold-2 clears 0.88 TM-score on four of the five strata and dips modestly to 0.82 on truly novel post-cutoff folds. Among the language models, the known pre-2026 column does not stand clear of the post-cutoff strata, and no language-model bar reaches the 0.50 fold-match threshold.'],
   build:function(){
     var ns=stratumN(1);
     return groupedBars({ ylabel:'TM-score vs native',
       cats:STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}), twoLine:true,
       series:modelSeriesWithBase('ESMFold-2', 'baseline_esmfold2', function(m){ return byStratum(1,m,'tm_score'); }),
       refs:[{v:0.5,l:'fold match 0.50'}],
       fmt:function(v){return v.toFixed(3);}, aria:'TM-score by novelty stratum with ESMFold-2 baseline' });
   } },
 { id:'f2', t:'2. Inverse folding', star:2,
   p:['Here the structure is given and the sequence withheld. Every side chain is stripped from the backbone and every residue relabelled UNK, and each method writes a sequence to fold back to that shape; ESMFold-2 refolds every design through the same single-sequence pipeline and superposes the result on the input backbone.',
      'ProteinMPNN (drawn in grey) clears 0.88–0.94 scTM across all five novelty strata — slightly exceeding the 0.885 native wild-type refolding ceiling because it idealises core packing without evolutionary active-site trade-offs. Frontier reasoning models reach 0.43–0.46 mean scTM overall, crossing 0.50 chiefly on idealised de novo backbones.'],
   build:function(){
     var ns=stratumN(2), ref=(FIGDATA.t2_base||[])[0];
     return groupedBars({ ylabel:'ESMFold-2 refolding scTM',
       cats:STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}), twoLine:true,
       series:modelSeriesWithBase('ProteinMPNN', 'baseline_proteinmpnn', function(m){ return byStratum(2,m,'sc_tm_score'); }),
       refs:[{v:0.5,l:'fold match 0.50'}].concat(ref?[{v:ref.v,l:ref.n+' '+ref.v}]:[]),
       fmt:function(v){return v.toFixed(3);}, aria:'Refolding scTM by novelty stratum with ProteinMPNN baseline' });
   } },
 { id:'f3', t:'3. Remote homology', star:3,
   p:['Two sequences, sometimes below 15% identity: do they descend from the same fold? Difficulty runs along the x-axis and every rung is paired against decoys matched for length and structural class, with recall measured at a fixed 10% false-positive rate.',
      '__T3_PROSE__'],
   build:function(){
     var S=FIGDATA.t3.series.map(function(se){
       return { n:(MNAME[se.n]||se.n)+' (AUROC '+se.auroc+')', c:se.base?GRAY1:MCOL[se.n], v:se.v };
     });
     return groupedBars({ ylabel:'Recall @ 10% FPR', cats:FIGDATA.t3.rungs,
       series:S, fmt:function(v){return v.toFixed(2);}, aria:'Recall at 10% FPR by rung' });
   } },
 { id:'f4', t:'4. Function prediction', star:4,
   p:['One sequence with the organism and gene name removed; each method predicts Gene Ontology terms, graded against experimental annotations released during 2026 across Molecular Function, Biological Process, and Cellular Component using information-content-weighted wFₘₐₓ.',
      '__T4_PROSE__'],
   build:function(){
     var S=FIGDATA.t4.series.map(function(se){
       return { n:(MNAME[se.n]||se.n)+' ('+se.macro+')', c:se.base?GRAY1:MCOL[se.n], v:se.v };
     });
     return groupedBars({ ylabel:'IC-weighted wFₘₐₓ',
       cats:['Molecular\nfunction','Biological\nprocess','Cellular\ncomponent'],
       series:S, fmt:function(v){return v.toFixed(3);}, aria:'IC-weighted wFmax by GO aspect' });
   } },
 { id:'f5', t:'5. Monomer alanine ranking', star:5,
   p:['Eighteen alanine mutations per protein, ranked by how much each destabilises the fold across four experimental regimes: synthetic de novo miniproteins, natural proteins, dual stability-and-function scans, and post-cutoff MaveDB deep mutational scans. Half of each panel consists of same-amino-acid decoys that pin any 1D residue lookup table to 0.50.',
      'ESM-2 (650M) masked marginals, drawn in grey from the identical single sequence, reveal a sharp split by origin: strong on natural evolved folds (NDCG@5 0.707) where homologous families populate UniRef, but dropping to 0.548 on synthetic de novo miniproteins with zero evolutionary history — the one regime where reasoning models pull ahead of the sequence language model.'],
   build:function(){
     var ns=stratumNList(T5_STRATA, 5);
     return groupedBars({ ylabel:'NDCG@5 across strata',
       cats:T5_STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}), twoLine:true,
       series:modelSeriesWithBase('ESM-2 (650M)', 'baseline_esm2_650m', function(m){ return byStratumList(T5_STRATA,5,m,'ndcg_at_5'); }),
       refs:[{v:0.5,l:'decoy floor 0.50'}], fmt:function(v){return v.toFixed(3);},
       aria:'Monomer alanine ranking NDCG@5 by protein type with ESM-2 650M baseline' });
   } },
 { id:'f6', t:'6. Interface binding ranking', star:6,
   p:['The same ranking exercise across a protein–protein interface, scored against measured changes in binding free energy with monomer-destabilising mutations filtered out beforehand. Spearman ρ compares the predicted ordering of the 18-mutation panel against experiment.',
      'ESM-2 (650M) complex marginals, drawn in grey, stay completely flat between pre-2026 PDB complexes (ρ = 0.162, n=31) and unseen or post-2026 complexes (ρ = 0.159, n=7). By contrast, all three reasoning models score 0.44–0.52 on pre-2026 PDB interfaces and drop back to the ESM-2 baseline (0.10–0.16) when no prior structure exists.'],
   build:function(){
     var S=FIGDATA.t6.series.map(function(se){
       var memo=se.memo||{};
       return { n:(MNAME[se.n]||se.n), c:se.base?GRAY1:MCOL[se.n],
                v:[memo.known_pdb_pre2026, memo.unseen_or_post2026] };
     });
     return groupedBars({ ylabel:'Spearman ρ',
       cats:['Known complex\n(pre-2026 PDB)','Unseen /\npost-2026'],
       subCats:['n=31','n=7'], twoLine:true,
       series:S, fmt:function(v){return v===null?'—':v.toFixed(3);},
       aria:'Spearman rho by memorisation stratum with ESM-2 650M baseline' });
   } },
 { id:'f7', t:'7. De novo and motif design', star:7,
   p:['Forty design specifications — 24 fold-conditioned monomers and 16 functional motif scaffolds — each put through five independent gates under a strict single-shot budget (one unscreened sequence per task): sequence hygiene, <30% sequence identity to natural parents, cross-oracle agreement between ESMFold-2 and OpenFold3 (scRMSD ≤ 2 Å), fold or motif geometric adherence, and refolding confidence (pLDDT ≥ 80).',
      'Even restricted to a single unscreened sample rather than standard multi-sample filtering, RFdiffusion3 + ProteinMPNN (drawn in grey) hits 48% on fold and sub-1.5 Å motif adherence (mean motif RMSD 1.45 Å vs 12.5% for text models) and reaches 5% single-shot pLDDT ≥ 80 pass rate (mean pLDDT 62.6), whereas all three text reasoning models remain at 0% on refolding confidence.'],
   build:function(){
     return groupedBars({ ylabel:'pass fraction (n=40)',
       cats:FIGDATA.t7.gates.map(function(g){ var p=g.split(' · '); return p[0]+'\n'+(p[1]||''); }),
       series:FIGDATA.t7.series.map(function(se){ return {n:MNAME[se.n]||se.n,c:se.base?GRAY1:MCOL[se.n],v:se.v}; }),
       fmt:function(v){return (v*100).toFixed(0)+'%';}, aria:'Design gate pass fractions with RFdiffusion3 + ProteinMPNN baseline' });
   } },
 { id:'f8', t:'8. Closed-loop binder design', star:8,
   p:['Twelve epitopes, sixteen proposals each, scored by the identical AlphaFold2 initial-guess and ESMFold-2 monomer oracle. Reasoning models propose iteratively using prior-round oracle feedback; BoltzGen (drawn in grey) generates 16 all-atom binder designs against the same target sequence and epitope hotspots, plotted as its unbiased k=1…16 accumulation curve.',
      '__T8_PROSE__'],
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
   p:['The same sixteen-proposal campaigns read as a before-and-after on the AlphaFold2 composite interface score: a single unfiltered proposal (shot 1) against the best design found across all sixteen shots.',
      'Across all three language models and the BoltzGen 3D diffusion baseline, sixteen shots substantially lift interface quality over a single blind draw, while illustrating how stringent the multi-gate on-epitope AlphaFold2 hit threshold (0.75) remains on hard targets.'],
   build:function(){
     var tr=FIGDATA.t8.traj;
     var ks=['baseline_boltzgen'].concat(MODEL_KEYS).filter(function(m){return tr[m];});
     return groupedBars({ ylabel:'AF2 interface score',
       cats:ks.map(function(m){ return (m==='baseline_boltzgen'?'BoltzGen':MNAME[m]).replace(/ /g,'\n'); }),
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

    # 4. Task 8 table: target names on one single line with PDB codes in muted grey beneath
    old_t8_rows = """        <tr><td><b>ProtDBench classic</b></td><td class="r">4</td><td>PD-L1, TrkA, IL-7R&alpha;, PDGFR&beta;<br><span class="muted">PDB 2001 &ndash; 2017</span></td><td>Standard targets from an existing protein-design benchmark. Long solved and much worked on, so this group is the one where prior art is most likely to help.</td></tr>
        <tr><td><b>BoltzProt1 novel</b></td><td class="r">5</td><td>AMBP, GM2A, HNMT, MZB1, RFK<br><span class="muted">PDB 2001 &ndash; 2021</span></td><td>Long-solved structures selected as novel binder targets. The fold is familiar; the design problem is not a standard one.</td></tr>
        <tr><td><b>Post-May-2026 crystal</b></td><td class="r">3</td><td>LEDGF PWWP, Aurora A, MCL1<br><span class="muted">PDB 9S83, 9RVK, 9TLV, all 2026</span></td><td>The only true holdout of the three. Deposited after the cutoff, so neither the complex nor its binder could have been read.</td></tr>"""
    new_t8_rows = """        <tr><td><b>ProtDBench classic</b></td><td class="r">4</td><td style="white-space:nowrap">PD-L1, TrkA, IL-7R&alpha;, PDGFR&beta;<br><span class="muted">PDB 5O45, 1WWW, 3DI3, 3MJG &middot; 2001&ndash;2017</span></td><td>Standard targets from an existing protein-design benchmark. Long solved and much worked on, so this group is the one where prior art is most likely to help.</td></tr>
        <tr><td><b>BoltzProt1 novel</b></td><td class="r">5</td><td style="white-space:nowrap">AMBP, GM2A, HNMT, MZB1, RFK<br><span class="muted">PDB 3QKG, 1G13, 1JQD, 7AAH, 1NB0 &middot; 2001&ndash;2021</span></td><td>Long-solved structures selected as novel binder targets. The fold is familiar; the design problem is not a standard one.</td></tr>
        <tr><td><b>Post-May-2026 crystal</b></td><td class="r">3</td><td style="white-space:nowrap">LEDGF PWWP, Aurora A, MCL1<br><span class="muted">PDB 9S83, 9RVK, 9TLV &middot; all 2026</span></td><td>The only true holdout of the three. Deposited after the cutoff, so neither the complex nor its binder could have been read.</td></tr>"""
    assert old_t8_rows in raw_html, "Expected Task 8 table rows not found"
    raw_html = raw_html.replace(old_t8_rows, new_t8_rows, 1)

    # 5. Wrap each task block (#prep-t1 .. #prep-t8) inside .disc-b in <div class="disc-blk" id="prep-tK">...</div>
    # so clicking a star link washes the entire matching task block (or #prep-d for #prep).
    disc_b_pattern = re.compile(
        r'(<p class="disc-intro">.*?</p>\n\n)(.*?)(\n    </div>\n    </details>)',
        re.DOTALL,
    )
    m = disc_b_pattern.search(raw_html)
    assert m is not None, "Expected .disc-b body not found"
    body = m.group(2)
    parts = re.split(r'(?=    <h3 class="disc-h" id="prep-t\d+">)', body)
    wrapped_parts: List[str] = []
    for part in parts:
        if not part.strip():
            continue
        pm = re.match(r'    <h3 class="disc-h" id="(prep-t\d+)">', part)
        assert pm is not None, f"Unexpected block start: {part[:60]}"
        blk_id = pm.group(1)
        part_clean = part.replace(f' id="{blk_id}"', "", 1).rstrip()
        wrapped_parts.append(f'    <div class="disc-blk" id="{blk_id}">\n{part_clean}\n    </div>')
    assert len(wrapped_parts) == 8, f"Expected 8 task blocks in #prep, got {len(wrapped_parts)}"
    raw_html = (
        raw_html[: m.start()]
        + m.group(1)
        + "\n\n".join(wrapped_parts)
        + m.group(3)
        + raw_html[m.end() :]
    )

    # 6. Flash #prep-d cleanly when clicking #prep (browser column header *) and #prep-tK when clicking task *
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
            f"Rung 1 is solvable from sequence similarity and all methods approach saturation there. "
            f"ESM-2 (650M) embedding cosine similarity (drawn in grey, AUROC {t3_chosen['overall_auroc']:.3f}) "
            f"provides a modern protein language model sequence baseline; the methods separate most sharply "
            f"at Rung 2 and Rung 3, the deep twilight zone where standard sequence and structure search fail."
        )
    else:
        t3_base_name = "BLOSUM62 Smith–Waterman"
        t3_per_raw = t34_res["t3_sw_rows"]
        t3_chosen = t3_sw
        t3_prose = (
            f"Rung 1 is solvable from local alignment and all methods saturate near 1.0. "
            f"BLOSUM62 Smith–Waterman (drawn in grey, AUROC {t3_chosen['overall_auroc']:.3f}) "
            f"provides the canonical sequence-alignment baseline; the methods separate at Rung 2 and Rung 3 "
            f"in the deep twilight zone."
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
            f"ESM-2 (650M) embedding k-NN transfer against the pre-2025 Swiss-Prot reference bank "
            f"(drawn in grey, macro wFₘₐₓ {t4_chosen['macro_wfmax']:.3f}) is the established protein language model "
            f"function baseline; all three frontier reasoning models exceed it on Molecular Function and Cellular Component."
        )
    else:
        t4_base_name = "Smith–Waterman homology"
        t4_chosen = t4_sw
        t4_prose = (
            f"Smith–Waterman sequence-homology transfer against the pre-2025 Swiss-Prot reference bank "
            f"(drawn in grey, macro wFₘₐₓ {t4_chosen['macro_wfmax']:.3f}) is the canonical CAFA BLAST-equivalent "
            f"baseline; all three frontier reasoning models outperform it across Molecular Function, Biological Process, and Cellular Component."
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
        f"All three language-model curves climb across the sixteen feedback rounds, showing genuine in-context "
        f"refinement from negative oracle evidence, while BoltzGen rises from {bgen_s0:.2f} on a single draw to "
        f"{bgen_s16:.2f} across sixteen samples (Hit@16 = {bgen_hit_pct:.1f}%)."
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
