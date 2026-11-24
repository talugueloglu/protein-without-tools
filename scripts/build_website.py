#!/usr/bin/env python3
"""Inject the 8 established baselines (1 clean domain baseline per task) into proteinbench.html,
updating DS (all 582 datapoints), FIGDATA, FIGS (f1..f8 + f4b), RESULT_SPEC, and DETAIL_SPEC."""

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
  5: { k: 'baseline_saprot_650m',              n: 'SaProt (650M)' },
  6: { k: 'baseline_stabddg',                  n: 'Boltz-2 + StaB-ddG' },
  7: { k: 'baseline_rfd3_mpnn',                n: 'RFdiffusion3 + ProteinMPNN' },
  8: { k: 'baseline_boltzgen_closed_loop_af2', n: 'BoltzGen', trajKey: 'baseline_boltzgen' }
};

var SCORE_LABEL = {
  'baseline_esmfold2':                 'ESMFold-2 (baseline)',
  'baseline_proteinmpnn':              'ProteinMPNN (baseline)',
  'baseline_t3':                       '__T3_BASE_NAME__ (baseline)',
  'baseline_t4':                       '__T4_BASE_NAME__ (baseline)',
  'baseline_saprot_650m':              'ESMFold-2 → SaProt 650M (baseline)',
  'baseline_stabddg':                  'Boltz-2 → StaB-ddG (baseline)',
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
  { t:4, name:'4. Function prediction', note:'CAFA-6 IA-weighted wFₘₐₓ on strict Post-cutoff NK holdouts (baseline: __T4_BASE_NAME__).',
    cols:[['Macro wFₘₐₓ',  function(S,m){ return fmt(cafaAspect(m,'macro'),3); }],
          ['MF',           function(S,m){ return fmt(cafaAspect(m,0),3); }],
          ['BP',           function(S,m){ return fmt(cafaAspect(m,1),3); }],
          ['CC',           function(S,m){ return fmt(cafaAspect(m,2),3); }]] },
  { t:5, name:'5. Monomer alanine ranking', note:'Same-AA decoys hold a 1D lookup table at 50% (baseline: ESMFold-2 → SaProt 650M).',
    cols:[['NDCG@5',       function(S){ return fmt(rMean(S.map(function(s){return s.ndcg_at_5;})),3); }],
          ['Same-AA win',  function(S){ return fmt(rMean(S.map(function(s){return s.same_aa_win_rate;})),3); }],
          ['Spearman ρ',   function(S){ return fmt(rMean(S.map(function(s){return s.spearman_rho;})),3); }]] },
  { t:6, name:'6. Binding alanine ranking', note:'Monomer-destabilising mutations excluded (baseline: Boltz-2 → StaB-ddG, zero SKEMPI training).',
    cols:[['NDCG@5',       function(S){ return fmt(rMean(S.map(function(s){return s.ndcg_at_5;})),3); }],
          ['Same-AA win',  function(S){ return fmt(rMean(S.map(function(s){return s.same_aa_win_rate;})),3); }],
          ['Spearman ρ',   function(S){ return fmt(rMean(S.map(function(s){return s.spearman_rho;})),3); }]] },
  { t:7, name:'7. Monomer & motif design', note:'Proteina-Complexa structural criteria on ESMFold-2 / OpenFold3 refolds (baseline: RFdiffusion3 + ProteinMPNN).',
    cols:[['Designable pass',  function(S){ return fmt(rPct(S.map(function(s){return s.designable_pass;}), function(x){return !!x;}),1,'%'); }],
          ['pLDDT',        function(S){ return fmt(rMean(S.map(function(s){return s.plddt_mean;})),1); }],
          ['scRMSD ≤ 2 Å', function(S){ return fmt(rPct(S.map(function(s){return s.cross_oracle_sc_rmsd;}), function(x){return isNum(x)&&x<=2.0;}),1,'%'); }]] },
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
  { t:5, name:'5. Monomer alanine ranking (baseline: ESMFold-2 → SaProt 650M)', metrics:[
      { label:'NDCG@5', max:1,     get:function(S){ return rMean(S.map(function(s){return s.ndcg_at_5;})); } },
      { label:'Same-AA win', max:1, floor:0.5, get:function(S){ return rMean(S.map(function(s){return s.same_aa_win_rate;})); } },
      { label:'Spearman ρ', max:1, get:function(S){ return rMean(S.map(function(s){return s.spearman_rho;})); } } ] },
  { t:6, name:'6. Binding alanine ranking (baseline: Boltz-2 → StaB-ddG)', metrics:[
      { label:'NDCG@5', max:1,     get:function(S){ return rMean(S.map(function(s){return s.ndcg_at_5;})); } },
      { label:'Same-AA win', max:1, floor:0.5, get:function(S){ return rMean(S.map(function(s){return s.same_aa_win_rate;})); } },
      { label:'Spearman ρ', max:1, get:function(S){ return rMean(S.map(function(s){return s.spearman_rho;})); } } ] },
  { t:7, name:'7. Monomer & motif design (baseline: RFdiffusion3 + ProteinMPNN)', metrics:[
      { label:'Designable pass', max:100, pct:true, get:function(S){ return rPct(S.map(function(s){return s.designable_pass;}), function(x){return !!x;}); } },
      { label:'pLDDT', max:100,    get:function(S){ return rMean(S.map(function(s){return s.plddt_mean;})); } },
      { label:'scRMSD ≤ 2 Å', max:100, pct:true, get:function(S){ return rPct(S.map(function(s){return s.cross_oracle_sc_rmsd;}), function(x){return isNum(x)&&x<=2.0;}); } } ] },
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

function spreadYs(items, minGap, yMin, yMax) {
  var arr = items.slice().sort(function(a, b){ return a.y - b.y; });
  for (var iter = 0; iter < 8; iter++) {
    for (var i = 1; i < arr.length; i++) {
      var diff = arr[i].y - arr[i - 1].y;
      if (diff < minGap) {
        var push = (minGap - diff) / 2;
        arr[i - 1].y -= push;
        arr[i].y += push;
      }
    }
    for (var j = 0; j < arr.length; j++) {
      if (arr[j].y < yMin) arr[j].y = yMin;
      if (arr[j].y > yMax) arr[j].y = yMax;
    }
  }
  return arr;
}

function slopePanels(o) {
  var W = 880, mL = 66, mR = 24, mT = 68, mB = 46;
  var lg = legendRows(o.series, mL, W - mR, 20);
  if (lg.rows > 1) mT += (lg.rows - 1) * lg.rowGap;
  var H = o.h || Math.max(338, mT + 206 + mB);
  var allVals = [];
  o.series.forEach(function(se){
    (se.pairs || []).forEach(function(pr){ allVals.push(pr[0], pr[1]); });
  });
  var max = o.maxFixed || fitMax([{ v: allVals }], [], o.floorMax || 0.7);
  var pw = W - mL - mR, ph = H - mT - mB, y0 = mT + ph;
  var yOf = function(v){ return y0 - (Math.max(0, Math.min(v, max)) / max) * ph; };
  var fmt = o.fmt || function(v){ return v.toFixed(3); };

  var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="' + esc(o.aria || 'slope comparison') + '">';
  s += lg.svg;

  var ticks = niceTicks(max);
  for (var i = 0; i <= ticks; i++) {
    var v = max * i / ticks, y = yOf(v);
    s += '<line x1="' + mL + '" y1="' + y + '" x2="' + (W - mR) + '" y2="' + y + '" class="cl-grid"/>';
    s += '<text x="' + (mL - 9) + '" y="' + (y + 4) + '" class="cl-tick" text-anchor="end">' + tickLabel(v, max, ticks) + '</text>';
  }

  var nP = o.panels.length, gap = 36, pW = (pw - (nP - 1) * gap) / nP, padX = 50;
  o.panels.forEach(function(pan, pi){
    var px0 = mL + pi * (pW + gap), px1 = px0 + pW, pcx = (px0 + px1) / 2;
    var xPre = px0 + padX, xPost = px1 - padX;
    // Panel title & subtitle above plot area
    s += '<text x="' + pcx + '" y="' + (mT - 24) + '" class="cl-val" text-anchor="middle">' + esc(pan.t) + '</text>';
    if (pan.s) s += '<text x="' + pcx + '" y="' + (mT - 9) + '" class="cl-metric" text-anchor="middle">' + esc(pan.s) + '</text>';
    // Vertical guides & baseline
    s += '<line x1="' + xPre + '" y1="' + mT + '" x2="' + xPre + '" y2="' + y0 + '" class="cl-axisline" stroke-dasharray="2 3"/>';
    s += '<line x1="' + xPost + '" y1="' + mT + '" x2="' + xPost + '" y2="' + y0 + '" class="cl-axisline" stroke-dasharray="2 3"/>';
    s += '<line x1="' + px0 + '" y1="' + y0 + '" x2="' + px1 + '" y2="' + y0 + '" class="cl-axisline"/>';
    s += '<text x="' + xPre + '" y="' + (y0 + 18) + '" class="cl-cat" text-anchor="middle">' + esc(pan.l0 || 'Pre 2025') + '</text>';
    s += '<text x="' + xPost + '" y="' + (y0 + 18) + '" class="cl-cat" text-anchor="middle">' + esc(pan.l1 || 'Post 2026') + '</text>';

    var leftLabels = [], rightLabels = [];
    o.series.forEach(function(se){
      var pr = se.pairs[pi];
      if (!pr || pr[0] === null || pr[1] === null) return;
      var y1 = yOf(pr[0]), y2 = yOf(pr[1]);
      var dash = se.base ? ' stroke-dasharray="5 3"' : '';
      var delta = pr[1] - pr[0];
      var dStr = (delta >= 0 ? '+' : '−') + Math.abs(delta).toFixed(3);
      var tip = se.n + ' · ' + pan.t + ': ' + fmt(pr[0]) + ' → ' + fmt(pr[1]) + ' (Δ ' + dStr + ')';
      s += '<line x1="' + xPre + '" y1="' + y1 + '" x2="' + xPost + '" y2="' + y2 + '" stroke="' + se.c + '" stroke-width="' + (se.base ? '2.2' : '2.3') + '"' + dash + ' stroke-linecap="round"/>';
      s += '<circle cx="' + xPre + '" cy="' + y1 + '" r="4.3" fill="' + se.c + '" stroke="#fff" stroke-width="1.8" class="cl-bar" data-tip="' + esc(tip) + '"><title>' + esc(tip) + '</title></circle>';
      s += '<circle cx="' + xPost + '" cy="' + y2 + '" r="4.3" fill="' + se.c + '" stroke="#fff" stroke-width="1.8" class="cl-bar" data-tip="' + esc(tip) + '"><title>' + esc(tip) + '</title></circle>';
      leftLabels.push({ y: y1, txt: pr[0].toFixed(2), c: se.c, base: se.base });
      rightLabels.push({ y: y2, txt: pr[1].toFixed(2), c: se.c, base: se.base });
    });

    spreadYs(leftLabels, 13, mT + 8, y0 - 4).forEach(function(lb){
      s += '<text x="' + (xPre - 8) + '" y="' + (lb.y + 3.5) + '" class="cl-metric" text-anchor="end" style="fill:' + (lb.base ? '#5f6773' : lb.c) + ';font-weight:600">' + lb.txt + '</text>';
    });
    spreadYs(rightLabels, 13, mT + 8, y0 - 4).forEach(function(lb){
      s += '<text x="' + (xPost + 8) + '" y="' + (lb.y + 3.5) + '" class="cl-metric" text-anchor="start" style="fill:' + (lb.base ? '#5f6773' : lb.c) + ';font-weight:600">' + lb.txt + '</text>';
    });
  });

  s += '<text class="cl-axis" transform="translate(15,' + (mT + ph / 2) + ') rotate(-90)" text-anchor="middle">' + esc(o.ylabel || '') + '</text>';
  return s + '</svg>';
}

function modelSeriesWithBase(baseName, baseKey, valsFor) {
  return [{ n:baseName, c:GRAY1, v:valsFor(baseKey) }].concat(
    MODEL_KEYS.map(function(m){ return { n:MNAME[m], c:MCOL[m], v:valsFor(m) }; })
  );
}

var T7_GATE_LABELS = [
  'Fold fidelity\nTM ≥ 0.50',
  'Motif fidelity\nRMSD ≤ 1.0 Å',
  'Confidence\npLDDT ≥ 70',
  'Self-consistency\nscRMSD ≤ 2 Å',
  'Designable\nall criteria'
];

var PREP_CARDS = {
  1: {
    sub: '100 RCSB PDB monomers · Foldseek structural novelty strata',
    body: '<p>We pull 100 monomeric structures from the <a href="https://www.rcsb.org/" target="_blank" rel="noopener">RCSB PDB</a> and search each one against the pre-May-2026 PDB with <a href="https://search.foldseek.com/" target="_blank" rel="noopener">Foldseek</a>. A target\'s highest TM-score to any older chain sets its novelty bin (above 0.5 shares a fold; below 0.3 is random), with computationally designed proteins separated first.</p>' +
      '<div style="overflow-x:auto"><table class="plain">' +
      '<thead><tr><th>Stratum</th><th class="r">n</th><th>Deposited</th><th>Best pre-cutoff match</th><th>Definition</th></tr></thead>' +
      '<tbody>' +
      '<tr><td><b>Known</b> (pre-2026)</td><td class="r">15</td><td>1987 &ndash; 2020</td><td>TM 0.87 &ndash; 1.00</td><td>Pre-cutoff PDB positive control.</td></tr>' +
      '<tr><td><b>Homolog gradient</b></td><td class="r">12</td><td>2026</td><td>TM 0.65 &ndash; 0.94</td><td>New entries with a close pre-cutoff structural homolog.</td></tr>' +
      '<tr><td><b>PDB-novel</b></td><td class="r">27</td><td>2026</td><td>TM 0.53 &ndash; 0.81</td><td>Natural 2026 proteins with only partial structural precedent.</td></tr>' +
      '<tr><td><b><em>De novo</em></b></td><td class="r">28</td><td>2026</td><td>TM 0.45 &ndash; 0.91</td><td>Computationally designed folds with no evolutionary homologs.</td></tr>' +
      '<tr><td><b>Truly novel</b></td><td class="r">18</td><td>2026</td><td>TM 0.33 &ndash; 0.53</td><td>No pre-cutoff PDB chain reaches the 0.50 same-fold line.</td></tr>' +
      '</tbody></table></div>'
  },
  2: {
    sub: 'Same 100 RCSB PDB folds · stripped Cα backbones (all UNK)',
    body: '<p>We reuse the exact 100 <a href="https://www.rcsb.org/" target="_blank" rel="noopener">RCSB PDB</a> structures and five strata from Task 1, deleting every side chain (all residues relabelled UNK) so only C&alpha; coordinates, backbone angles, and burial counts remain. Refolding the native wild-type sequences through ESMFold-2 sets the 0.885 ceiling.</p>'
  },
  3: {
    sub: '120 SCOPe domain pairs · TEA & Foldseek difficulty rungs',
    body: '<p>We draw 120 <a href="https://scop.berkeley.edu/" target="_blank" rel="noopener">SCOPe</a> domain pairs via <a href="https://github.com/PickyBinders/tea" target="_blank" rel="noopener">TEA</a> and <a href="https://search.foldseek.com/" target="_blank" rel="noopener">Foldseek</a>, grouped by which search method fails (30 pairs per rung; Rungs 2&ndash;4 sit near 11% sequence identity). Rung 4 provides the 30 negative decoys used to set the 10% FPR cutoff and AUROC.</p>' +
      '<div style="overflow-x:auto"><table class="plain">' +
      '<thead><tr><th>Rung</th><th class="r">n</th><th>Sequence identity</th><th>Selection rule</th></tr></thead>' +
      '<tbody>' +
      '<tr><td><b>1 &middot; Sequence-easy</b></td><td class="r">30</td><td>18 &ndash; 38% (med. 23%)</td><td>Same SCOPe superfamily; recovered by Smith&ndash;Waterman.</td></tr>' +
      '<tr><td><b>2 &middot; Structure-caught</b></td><td class="r">30</td><td>9 &ndash; 14% (med. 11%)</td><td>Same superfamily; sequence alignment fails, Foldseek 3Di hits.</td></tr>' +
      '<tr><td><b>3 &middot; Deep twilight</b></td><td class="r">30</td><td>8 &ndash; 16% (med. 11%)</td><td>Same superfamily; both sequence and Foldseek search fail.</td></tr>' +
      '<tr><td><b>4 &middot; Matched decoy</b></td><td class="r">30</td><td>9 &ndash; 14% (med. 11%)</td><td>Different superfamilies matched on structural class and identity (negative class).</td></tr>' +
      '</tbody></table></div>'
  },
  4: {
    sub: '120 UniProtKB proteins · EBI GOA 2×3 CAFA-6 split (Pre vs. Post × NK/LK/PK)',
    body: '<p>We pull 120 non-redundant <a href="https://www.uniprot.org/" target="_blank" rel="noopener">UniProtKB</a> proteins (3-mer Jaccard &le; 0.28) with curated non-IntAct experimental <a href="https://geneontology.org/" target="_blank" rel="noopener">Gene Ontology</a> annotations from <a href="https://www.ebi.ac.uk/GOA" target="_blank" rel="noopener">EBI GOA</a> (<a href="https://www.ebi.ac.uk/QuickGO/" target="_blank" rel="noopener">QuickGO</a> double-verified for Post-NK, excluding trivial <span class="ds-key">GO:0005515</span>-only). We grade every model with CAFA-6 conditional Information Accretion (IA) in <code>full</code> mode across matched Pre-cutoff (May&ndash;Oct 2025) and Post-cutoff (May&ndash;Oct 2026) windows.</p>' +
      '<div style="overflow-x:auto"><table class="plain">' +
      '<thead><tr><th>Cell</th><th class="r">n</th><th>Window</th><th>CAFA-6 track &amp; grading rule</th></tr></thead>' +
      '<tbody>' +
      '<tr><td><b>Pre-cutoff NK</b> (<span class="ds-key">pre_nk</span>)</td><td class="r">20</td><td>May &ndash; Oct 2025</td><td>Zero experimental GO before 2025-05-01; graded on all gained aspects (pre-training control).</td></tr>' +
      '<tr><td><b>Pre-cutoff LK</b> (<span class="ds-key">pre_lk</span>)</td><td class="r">20</td><td>May &ndash; Oct 2025</td><td>Had 1&ndash;2 aspects before 2025-05-01; graded only on the newly added aspect.</td></tr>' +
      '<tr><td><b>Pre-cutoff PK</b> (<span class="ds-key">pre_pk</span>)</td><td class="r">20</td><td>May &ndash; Oct 2025</td><td>Pre-2025-05-01 terms masked out of both truth and predictions (<code>-known</code>).</td></tr>' +
      '<tr><td><b>Post-cutoff NK</b> (<span class="ds-key">post_nk</span>)</td><td class="r">20</td><td>May &ndash; Oct 2026</td><td>Zero experimental GO before 2026-05-01 and post-May-2026 papers; clean blind test.</td></tr>' +
      '<tr><td><b>Post-cutoff LK</b> (<span class="ds-key">post_lk</span>)</td><td class="r">20</td><td>May &ndash; Oct 2026</td><td>Had 1&ndash;2 aspects before 2026-05-01; graded only on the newly added aspect.</td></tr>' +
      '<tr><td><b>Post-cutoff PK</b> (<span class="ds-key">post_pk</span>)</td><td class="r">20</td><td>May &ndash; Oct 2026</td><td>Pre-2026-05-01 terms masked out of both truth and predictions (<code>-known</code>).</td></tr>' +
      '</tbody></table></div>'
  },
  5: {
    sub: '52 panels (18 mutations each) · MegaScale & MaveDB',
    body: '<p>Each panel picks 18 X&rarr;Ala mutations from <a href="https://huggingface.co/datasets/RosettaCommons/MegaScale" target="_blank" rel="noopener">MegaScale</a> (<a href="https://zenodo.org/records/7992926" target="_blank" rel="noopener">Zenodo</a>) or <a href="https://www.mavedb.org/" target="_blank" rel="noopener">MaveDB</a>: 6 strong hotspots, 6 moderate sites, and 6 neutral decoys sharing the exact same wild-type amino acids so residue-letter lookup stays at 50%.</p>' +
      '<div style="overflow-x:auto"><table class="plain">' +
      '<thead><tr><th>Source dataset</th><th class="r">n</th><th>Measured quantity</th></tr></thead>' +
      '<tbody>' +
      '<tr><td><b><em>De novo</em> miniproteins</b> (<a href="https://huggingface.co/datasets/RosettaCommons/MegaScale" target="_blank" rel="noopener">MegaScale</a>)</td><td class="r">24</td><td>Folding &Delta;&Delta;G (cDNA proteolysis); designed folds with no natural homologs.</td></tr>' +
      '<tr><td><b>Natural proteins</b> (<a href="https://huggingface.co/datasets/RosettaCommons/MegaScale" target="_blank" rel="noopener">MegaScale</a>)</td><td class="r">14</td><td>Folding &Delta;&Delta;G on natural folds.</td></tr>' +
      '<tr><td><b>Deep mutational scans</b> (<a href="https://www.mavedb.org/" target="_blank" rel="noopener">MaveDB</a>)</td><td class="r">12</td><td>Post-cutoff deep mutational scanning activity.</td></tr>' +
      '<tr><td><b>Stability and function</b> (<a href="https://www.mavedb.org/" target="_blank" rel="noopener">MaveDB</a>)</td><td class="r">2</td><td>Paired abundance/stability and activity on the same protein.</td></tr>' +
      '</tbody></table></div>'
  },
  6: {
    sub: '38 complexes · SKEMPI v2.0, Open-AlphaSeq & MaveDB PPI DMS',
    body: '<p>We remove monomer-destabilising mutations first so every panel isolates binding affinity, combining three datasets split into 31 complexes already in the pre-2026 PDB (<span class="ds-key">known_pdb_pre2026</span>) and 7 unseen or post-2026 complexes (<span class="ds-key">unseen_or_post2026</span>).</p>' +
      '<div style="overflow-x:auto"><table class="plain">' +
      '<thead><tr><th>Source dataset</th><th class="r">n</th><th>Readout &amp; holdout</th></tr></thead>' +
      '<tbody>' +
      '<tr><td><b><a href="https://life.bsc.es/pid/skempi2/" target="_blank" rel="noopener">SKEMPI v2.0</a></b></td><td class="r">28</td><td>Purified SPR/ITC binding &Delta;&Delta;G across pre-2026 PDB complexes.</td></tr>' +
      '<tr><td><b><a href="https://huggingface.co/datasets/aalphabio/open-alphaseq" target="_blank" rel="noopener">Open-AlphaSeq</a></b></td><td class="r">3</td><td>Oncology checkpoint scFv&ndash;target binding <em>K</em><sub>d</sub> (CDR/surface-filtered; zero-PDB holdouts).</td></tr>' +
      '<tr><td><b><a href="https://www.mavedb.org/" target="_blank" rel="noopener">MaveDB</a> PPI binding DMS</b></td><td class="r">7</td><td>Abundance-controlled PPI binding DMS (3 pre-cutoff + 4 post-May-2026 releases).</td></tr>' +
      '</tbody></table></div>'
  },
  7: {
    sub: '40 specifications · RCSB PDB fold templates & crystal motifs',
    body: '<p>Each prompt specifies a fold template or crystal motif from the <a href="https://www.rcsb.org/" target="_blank" rel="noopener">RCSB PDB</a> and asks for a single 1D sequence. We fold every sequence with ESMFold-2 and OpenFold3 and require fold/motif geometry below, ESMFold-2 pLDDT &ge; 70, cross-oracle scRMSD &le; 2.0 &Aring;, clean sequence hygiene, and &lt;30% identity to natural parents.</p>' +
      '<div style="overflow-x:auto"><table class="plain">' +
      '<thead><tr><th>Kind</th><th class="r">n</th><th>Length</th><th>Structural fidelity</th></tr></thead>' +
      '<tbody>' +
      '<tr><td><b>Fold-conditioned</b></td><td class="r">24</td><td>64 &ndash; 136 AA</td><td>Refolded design reaches TM &ge; 0.50 against the target fold.</td></tr>' +
      '<tr><td><b>Motif scaffolding</b></td><td class="r">16</td><td>70 &ndash; 128 AA</td><td>Held crystal motif stays within C&alpha; RMSD &le; 1.0 &Aring;.</td></tr>' +
      '</tbody></table></div>'
  },
  8: {
    sub: '12 RCSB PDB epitopes · ProtDBench, BoltzProt1 & post-May-2026 crystals',
    body: '<p>We pick 12 <a href="https://www.rcsb.org/" target="_blank" rel="noopener">RCSB PDB</a> target chains with five hotspot residues each and give every method 16 binder proposals (55&ndash;85 AA) scored by AlphaFold2 and ESMFold-2. Only the third group is a post-cutoff crystal holdout.</p>' +
      '<div style="overflow-x:auto"><table class="plain">' +
      '<thead><tr><th>Group</th><th class="r">n</th><th>Targets &amp; PDB codes</th><th>Role</th></tr></thead>' +
      '<tbody>' +
      '<tr><td><b>ProtDBench classic</b></td><td class="r">4</td><td><span style="white-space:nowrap">PD-L1, TrkA, IL-7R&alpha;, PDGFR&beta;</span><br><span class="muted">PDB <a href="https://www.rcsb.org/structure/5O45" target="_blank" rel="noopener">5O45</a>, <a href="https://www.rcsb.org/structure/1WWW" target="_blank" rel="noopener">1WWW</a>, <a href="https://www.rcsb.org/structure/3DI3" target="_blank" rel="noopener">3DI3</a>, <a href="https://www.rcsb.org/structure/3MJG" target="_blank" rel="noopener">3MJG</a></span></td><td>Standard prior-art binder targets (2001&ndash;2017).</td></tr>' +
      '<tr><td><b>BoltzProt1 novel</b></td><td class="r">5</td><td><span style="white-space:nowrap">AMBP, GM2A, HNMT, MZB1, RFK</span><br><span class="muted">PDB <a href="https://www.rcsb.org/structure/3QKG" target="_blank" rel="noopener">3QKG</a>, <a href="https://www.rcsb.org/structure/1G13" target="_blank" rel="noopener">1G13</a>, <a href="https://www.rcsb.org/structure/1JQD" target="_blank" rel="noopener">1JQD</a>, <a href="https://www.rcsb.org/structure/7AAH" target="_blank" rel="noopener">7AAH</a>, <a href="https://www.rcsb.org/structure/1NB0" target="_blank" rel="noopener">1NB0</a></span></td><td>Solved folds repurposed as non-standard binder targets (2001&ndash;2021).</td></tr>' +
      '<tr><td><b>Post-May-2026 crystal</b></td><td class="r">3</td><td><span style="white-space:nowrap">LEDGF PWWP, Aurora A, MCL1</span><br><span class="muted">PDB <a href="https://www.rcsb.org/structure/9S83" target="_blank" rel="noopener">9S83</a>, <a href="https://www.rcsb.org/structure/9RVK" target="_blank" rel="noopener">9RVK</a>, <a href="https://www.rcsb.org/structure/9TLV" target="_blank" rel="noopener">9TLV</a></span></td><td>Post-cutoff 2026 crystal complexes with held-out binders.</td></tr>' +
      '</tbody></table></div>'
  }
};

var FIGS = [
 { id:'f1', t:'1. Forward folding', star:1, prep:1,
   p:['We give each model one protein sequence with no IDs or alignments and ask it to predict residue contacts and secondary structure, then fold those predictions into 3D and score them by TM-score against the native structure. Single-sequence ESMFold-2 (grey) solves almost every target (0.88–0.91 TM, and 0.82 on truly novel folds), while all three reasoning models stay below the 0.50 fold-match line.'],
   build:function(){
     var ns=stratumN(1);
     return groupedBars({ ylabel:'TM-score vs native',
       cats:STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}),
       series:modelSeriesWithBase('ESMFold-2', 'baseline_esmfold2', function(m){ return byStratum(1,m,'tm_score'); }),
       refs:[{v:0.5,l:'fold match 0.50'}],
       fmt:function(v){return v.toFixed(3);}, aria:'TM-score by novelty stratum with ESMFold-2 baseline' });
   } },
 { id:'f2', t:'2. Inverse folding', star:2, prep:2,
   p:['We strip every side chain off the same 100 backbones (leaving only Cα coordinates), ask each model to write a sequence that folds into that shape, and refold the output with ESMFold-2. ProteinMPNN (grey) recovers the fold across all five strata (0.88–0.94 scTM, above the 0.885 wild-type refolding ceiling), while reasoning models average 0.43–0.46 scTM.'],
   build:function(){
     var ns=stratumN(2), ref=(FIGDATA.t2_base||[])[0];
     return groupedBars({ ylabel:'ESMFold-2 refolding scTM',
       cats:STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}),
       series:modelSeriesWithBase('ProteinMPNN', 'baseline_proteinmpnn', function(m){ return byStratum(2,m,'sc_tm_score'); }),
       refs:[{v:0.5,l:'fold match 0.50'}].concat(ref?[{v:ref.v,l:ref.n+' '+ref.v}]:[]),
       fmt:function(v){return v.toFixed(3);}, aria:'Refolding scTM by novelty stratum with ProteinMPNN baseline' });
   } },
 { id:'f3', t:'3. Remote homology', star:3, prep:3,
   p:['__T3_PROSE__'],
   build:function(){
     var S=FIGDATA.t3.series.map(function(se){
       return { n:(MNAME[se.n]||se.n)+' (AUROC '+se.auroc+')', c:se.base?GRAY1:MCOL[se.n], v:se.v };
     });
     return groupedBars({ ylabel:'Recall @ 10% FPR', cats:FIGDATA.t3.rungs,
       series:S, fmt:function(v){return v.toFixed(2);}, aria:'Recall at 10% FPR by rung' });
   } },
 { id:'f4', t:'4. Function prediction (Post-cutoff No-Knowledge)', star:4,
   p:['__T4_PROSE__'],
   build:function(){
     var S=FIGDATA.t4.series.map(function(se){
       return { n:(MNAME[se.n]||se.n)+' ('+se.macro+')', c:se.base?GRAY1:MCOL[se.n], v:se.v };
     });
     return groupedBars({ ylabel:'CAFA-6 IA-weighted wFₘₐₓ',
       cats:['Molecular\nfunction','Biological\nprocess','Cellular\ncomponent'],
       subCats:FIGDATA.t4.aspect_ns || ['n=13','n=14','n=14'],
       series:S, fmt:function(v){return v.toFixed(3);}, aria:'Post-cutoff No-Knowledge CAFA-6 IA-weighted wFmax by GO aspect' });
   } },
 { id:'f4b', t:'4b. Pre- vs. Post-cutoff across CAFA-6 knowledge tracks (NK, LK, PK)', star:4, prep:4,
   p:['__T4B_PROSE__'],
   build:function(){
     var S=FIGDATA.t4.series.map(function(se){
       var cv = se.cells_v || [0,0,0,0,0,0];
       return { n:(MNAME[se.n]||se.n), c:se.base?GRAY1:MCOL[se.n], base:!!se.base,
                pairs:[[cv[0],cv[3]],[cv[1],cv[4]],[cv[2],cv[5]]] };
     });
     return slopePanels({ ylabel:'CAFA-6 IA-weighted macro wFₘₐₓ',
       panels:[
         { t:'No-Knowledge (NK)', s:'zero prior GOA · n=20 / 20', l0:'Pre 2025', l1:'Post 2026' },
         { t:'Limited-Knowledge (LK)', s:'new aspect only · n=20 / 20', l0:'Pre 2025', l1:'Post 2026' },
         { t:'Partial-Knowledge (PK)', s:'-known masked · n=20 / 20', l0:'Pre 2025', l1:'Post 2026' }
       ],
       series:S, floorMax:0.7, aria:'Pre-cutoff vs Post-cutoff CAFA-6 wFmax across NK, LK, and PK tracks' });
   } },
 { id:'f5', t:'5. Monomer alanine ranking', star:5, prep:5,
   p:['We give each model a protein sequence and 18 single-site X→Ala mutations—mixing true destabilising hotspots with neutral decoys of the exact same amino acid so guessing by residue letter scores 0.50—and ask it to rank them from most to least damaging. ESMFold-2 + SaProt (grey, zero MegaScale training) wins on natural proteins (0.830) and paired stability/function scans (0.995), while reasoning models lead on synthetic <em>de novo</em> miniproteins with no natural homologs (0.74–0.76 vs 0.632).'],
   build:function(){
     var ns=stratumNList(T5_STRATA, 5);
     return groupedBars({ ylabel:'NDCG@5 across strata',
       cats:T5_STRATA.map(function(s){return s.l;}),
       subCats:ns.map(function(n){return 'n='+n;}),
       series:modelSeriesWithBase('SaProt (650M)', 'baseline_saprot_650m', function(m){ return byStratumList(T5_STRATA,5,m,'ndcg_at_5'); }),
       refs:[{v:0.5,l:'decoy floor 0.50'}], fmt:function(v){return v.toFixed(3);},
       aria:'Monomer alanine ranking NDCG@5 by protein type with ESMFold-2 SaProt 650M baseline' });
   } },
 { id:'f6', t:'6. Interface binding ranking', star:6, prep:6,
   p:['We give each model two interacting protein chains and 18 X→Ala mutations (with monomer-unfolding sites already removed) and ask it to rank which mutations weaken binding most. On complexes already in the pre-2026 PDB (n=31), Boltz-2 + StaB-ddG (grey, zero SKEMPI training) and the reasoning models score similarly (ρ = 0.43 vs 0.44–0.52); both collapse on complexes missing from the pre-2026 PDB (n=7; 0.045 vs 0.10–0.16).'],
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
       aria:'Spearman rho by memorisation stratum with Boltz-2 StaB-ddG baseline' });
   } },
 { id:'f7', t:'7. De novo and motif design', star:7, prep:7,
   p:['We ask each model to write a new sequence in one shot that either adopts a requested fold (n=24) or scaffolds a fixed crystal motif (n=16), then fold it with ESMFold-2 and OpenFold3 and check five criteria (fold TM ≥ 0.50, motif RMSD ≤ 1.0 Å, pLDDT ≥ 70, cross-oracle scRMSD ≤ 2 Å, and <30% identity to natural proteins). RFdiffusion3 + ProteinMPNN (grey) hits 100% fold fidelity, 63% motif fidelity, and 5% overall pass; reasoning models hit 0% on fold topology and 0% overall.'],
   build:function(){
     return groupedBars({ ylabel:'pass fraction',
       cats:T7_GATE_LABELS,
       subCats:['n=24','n=16','n=40','n=40','n=40'],
       series:FIGDATA.t7.series.map(function(se){ return {n:MNAME[se.n]||se.n,c:se.base?GRAY1:MCOL[se.n],v:se.v}; }),
       fmt:function(v){return (v*100).toFixed(0)+'%';}, aria:'Proteina-Complexa structural design metrics with RFdiffusion3 + ProteinMPNN baseline' });
   } },
 { id:'f8', t:'8. Closed-loop binder design', star:8, prep:8,
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
   } }
];

function renderFigures() {
  var host=document.getElementById('results-figures');
  if (!host || typeof DS==='undefined' || typeof FIGDATA==='undefined') return;
  host.innerHTML = FIGS.map(function(f){
    var svg='';
    try { svg=f.build(); } catch(e){ svg='<p class="muted small">This figure could not be drawn.</p>'; }
    var pc = f.prep && PREP_CARDS[f.prep];
    var prepHtml = pc
      ? '<details class="disc fig-prep" id="prep-t' + f.prep + '">' +
        '<summary><span class="disc-t">How the data is prepared</span><span class="disc-s">' + pc.sub + '</span></summary>' +
        '<div class="disc-b">' + pc.body + '</div></details>'
      : '';
    return '<figure class="fig" id="'+f.id+'"><h3>'+esc(f.t)+(f.star ? star(f.star) : '')+'</h3>'+
           f.p.map(function(x){ return '<p class="fig-p">'+x+'</p>'; }).join('')+
           '<div class="fig-plot">'+svg+'</div>'+prepHtml+'</figure>';
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
    try { console.error('Proteins Without Tools: ' + label + ' failed —', e && (e.stack || e.message)); } catch (_) {}
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
    """Apply Works-in-Progress editorial typography (Lora + Spectral), subtle cool-blue paper palette,
    per-figure data-prep cards, Task 8 PDB rows, and star-link scroll/flash polish."""
    import re

    # 0. Editorial typography (Lora headings + Spectral body prose + Inter UI) & subtle cool-blue paper background
    old_gfonts = '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
    new_gfonts = '<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
    assert old_gfonts in raw_html, "Expected Google Fonts link not found"
    raw_html = raw_html.replace(old_gfonts, new_gfonts, 1)

    old_root_body = """  :root{
    --ink:#1a1f2b;
    --body:rgba(26,31,43,.84);
    --muted:rgba(26,31,43,.55);
    --rule:#e4e7eb;
    --rule2:#f0f2f4;
    --navy:#1e293b;
    --link:#1b4b8f;
    --soft:#f7f8fa;
    --serif:"Source Serif 4",Charter,Georgia,Cambria,"Times New Roman",serif;
    --sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --rose:#b4324a; --amber:#a8750f; --sky:#2d7d9a; --emerald:#2f7a5b; --gold:#b8860b;
  }
  *,*::before,*::after{box-sizing:border-box;}
  html{-webkit-text-size-adjust:100%;scroll-behavior:smooth;}
  body{margin:0;background:#fff;color:var(--body);font-family:var(--sans);
    font-size:17px;line-height:1.72;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}"""
    new_root_body = """  :root{
    --bg:#f4f7fb;
    --surface:#ffffff;
    --ink:#0f172a;
    --body:rgba(15,23,42,.86);
    --muted:rgba(15,23,42,.56);
    --rule:#d9e2ee;
    --rule2:#e6edf6;
    --navy:#0f172a;
    --link:#1b4b8f;
    --soft:#eaf0f8;
    --serif:"Lora",Georgia,Cambria,"Times New Roman",serif;
    --body-serif:"Spectral",Georgia,Cambria,"Times New Roman",serif;
    --sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --rose:#b4324a; --amber:#a8750f; --sky:#2d7d9a; --emerald:#2f7a5b; --gold:#b8860b;
  }
  *,*::before,*::after{box-sizing:border-box;}
  html{-webkit-text-size-adjust:100%;scroll-behavior:smooth;}
  body{margin:0;background:var(--bg);color:var(--body);font-family:var(--body-serif);
    font-size:18px;line-height:1.72;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}
  nav.top,button,.btn,.btn-ghost,.tabs,.tab,.actions,.xp-meta,.xp-link,.xp-foot,.legend,
  table.plain,.ds-counts,.ds-controls,.ds-table,.ds-detail,.sm-legend,.mt-row,.disc-s,
  details.tblview summary,.rnote,.figcap,.cl-tooltip,footer{font-family:var(--sans);}"""
    assert old_root_body in raw_html, "Expected :root/body CSS block not found"
    raw_html = raw_html.replace(old_root_body, new_root_body, 1)

    raw_html = raw_html.replace(
        "nav.top{position:sticky;top:0;z-index:50;background:rgba(255,255,255,.93);",
        "nav.top{position:sticky;top:0;z-index:50;background:rgba(244,247,251,.92);",
        1,
    )
    raw_html = raw_html.replace(
        "details.disc{border:1px solid var(--rule);border-radius:10px;margin:0;}",
        "details.disc{border:1px solid var(--rule);border-radius:10px;margin:0;background:rgba(255,255,255,.82);}",
        1,
    )
    raw_html = raw_html.replace(
        "  .cl-grid{stroke:#eceef1;stroke-width:1;}\n  .cl-axisline{stroke:#d7dbe0;stroke-width:1;}",
        "  .cl-grid{stroke:#dde5f0;stroke-width:1;}\n  .cl-axisline{stroke:#c6d2e2;stroke-width:1;}",
        1,
    )
    raw_html = raw_html.replace(
        "  .ds-tablewrap{border:1px solid var(--rule);border-radius:4px;max-height:560px;overflow:auto;}",
        "  .ds-tablewrap{border:1px solid var(--rule);border-radius:6px;max-height:560px;overflow:auto;background:#fff;}",
        1,
    )
    raw_html = raw_html.replace(
        "  .fig-p{font-size:15.5px;line-height:1.72;color:var(--body);margin:0 0 13px;}",
        "  .fig-p{font-size:17px;line-height:1.72;color:var(--body);margin:0 0 13px;}",
        1,
    )
    raw_html = raw_html.replace(
        "  .xp-blurb{font-size:15.5px;line-height:1.72;color:var(--body);margin:0 0 18px;}",
        "  .xp-blurb{font-size:17px;line-height:1.72;color:var(--body);margin:0 0 18px;}",
        1,
    )
    raw_html = raw_html.replace(
        "  header.hero .blurb{max-width:62ch;margin:14px auto 18px;color:var(--body);font-size:17.5px;line-height:1.6;}",
        "  header.hero .blurb{max-width:62ch;margin:14px auto 18px;color:var(--body);font-size:19px;line-height:1.6;}",
        1,
    )
    raw_html = raw_html.replace(
        "  .lead{font-size:16px;color:var(--muted);margin:-4px 0 18px;}",
        "  .lead{font-size:17px;color:var(--muted);margin:-4px 0 18px;}",
        1,
    )

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
  details.disc.fig-prep{margin:16px auto 0;max-width:840px;scroll-margin-top:86px;}
  details.disc.fig-prep>summary{padding:11px 42px 11px 16px;}
  details.disc.fig-prep>summary::after{right:18px;}
  details.disc.fig-prep .disc-t{font-size:15.5px;font-weight:500;}
  details.disc.fig-prep .disc-s{font-size:13px;margin-top:1px;}
  details.disc.fig-prep .disc-b{padding:14px 16px 16px;}
  details.disc.fig-prep .disc-b p{font-size:15.5px;line-height:1.68;margin:0 0 12px;}
  details.disc.fig-prep .disc-b p:last-child{margin-bottom:0;}
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

    # 3b. Rename title/wordmark/hero/BibTeX from ProteinBench -> "Proteins Without Tools" & update 522 -> 582 datapoints
    raw_html = raw_html.replace("522 targets, held-out experimental data", "582 targets, held-out experimental data", 1)
    raw_html = raw_html.replace("All 522 datapoints", "All 582 datapoints")
    raw_html = raw_html.replace("All 522 prompts audited", "All 582 prompts audited", 1)
    raw_html = raw_html.replace("label:'All 522'", "label:'All 582'", 1)
    raw_html = raw_html.replace("Full 522-datapoint browser", "Full 582-datapoint browser", 1)
    raw_html = raw_html.replace("badge: 'N = 60 2026 QuickGO Holdouts'", "badge: 'N = 120 CAFA-6 Pre/Post × NK/LK/PK'", 1)
    raw_html = raw_html.replace(
        "  post_may_2026:'Post-May-2026 annotation',\n  early_2026_uniprot:'Early-2026 UniProt',",
        "  pre_nk:'Pre-cutoff NK',\n  pre_lk:'Pre-cutoff LK',\n  pre_pk:'Pre-cutoff PK',\n  post_nk:'Post-cutoff NK',\n  post_lk:'Post-cutoff LK',\n  post_pk:'Post-cutoff PK',",
        1,
    )
    raw_html = raw_html.replace(
        "<title>ProteinBench: Folding and Binding Physics in Frontier Reasoning Models</title>\n"
        '<meta name="description" content="An eight-task structural biology benchmark: 582 targets, held-out experimental data, per-datapoint structures and results." />',
        "<title>Proteins Without Tools: Folding and Binding Physics in Frontier Reasoning Models</title>\n"
        '<meta name="description" content="Can frontier reasoning models solve real structural biology problems from weights and reasoning alone? Eight tasks, 582 targets, and specialist baselines." />\n'
        '<meta name="author" content="Talu Güloğlu" />\n'
        '<meta property="og:type" content="article" />\n'
        '<meta property="og:title" content="Proteins Without Tools" />\n'
        '<meta property="og:description" content="An 8-task, 582-target structural biology benchmark testing frontier reasoning models without external tools against specialist baselines." />\n'
        '<meta property="og:url" content="https://talugueloglu.github.io/protein-without-tools/" />\n'
        '<meta property="og:image" content="https://talugueloglu.github.io/protein-without-tools/social-preview.png" />\n'
        '<meta property="og:image:width" content="1280" />\n'
        '<meta property="og:image:height" content="640" />\n'
        '<meta name="twitter:card" content="summary_large_image" />\n'
        '<meta name="twitter:site" content="@gueloglut" />\n'
        '<meta name="twitter:image" content="https://talugueloglu.github.io/protein-without-tools/social-preview.png" />\n'
        '<link rel="icon" href="favicon.png" type="image/png" />',
        1,
    )
    raw_html = raw_html.replace(
        '<p class="authors">Talu Gueloglu</p>',
        '<p class="authors"><a href="https://talugueloglu.github.io/homepage/" style="color:inherit;text-decoration:none;border-bottom:1px solid var(--hair);">Talu Güloğlu</a> &middot; <a href="https://talugueloglu.github.io/homepage/blogs/" style="color:var(--ink-3);text-decoration:none;">Writing</a></p>',
        1,
    )
    raw_html = raw_html.replace(
        "    </div>\n\n  </header>",
        "    </div>\n\n"
        '    <figure class="hero-video" style="max-width:740px;margin:28px auto 0;background:#fff;border:1px solid var(--hair);border-radius:8px;overflow:hidden;box-shadow:0 10px 28px -18px rgba(15,23,42,.24);">\n'
        '      <video controls playsinline muted preload="metadata" style="display:block;width:100%;height:auto;background:#0f172a;">\n'
        '        <source src="demo.mp4" type="video/mp4" />\n'
        "      </video>\n"
        "    </figure>\n\n"
        "  </header>",
        1,
    )
    raw_html = raw_html.replace(
        ".wordmark{font-family:var(--serif);font-weight:600;font-size:18px;color:var(--ink);\n    letter-spacing:.01em;border-bottom:2px solid var(--navy);padding-bottom:1px;}",
        ".wordmark{font-family:var(--serif);font-weight:600;font-size:17px;color:var(--ink);\n    letter-spacing:.01em;border-bottom:2px solid var(--navy);padding-bottom:1px;white-space:nowrap;}",
        1,
    )
    raw_html = raw_html.replace("margin:0 auto 14px;max-width:17ch;}", "margin:0 auto 14px;max-width:26ch;}", 1)
    raw_html = raw_html.replace('<span class="wordmark">ProteinBench</span>', '<span class="wordmark">Proteins Without Tools</span>', 1)
    raw_html = raw_html.replace("<h1>ProteinBench</h1>", "<h1>Proteins Without Tools</h1>", 1)
    raw_html = raw_html.replace(
        "@article{proteinbench2026,\n  title   = {ProteinBench: From Sequence Memorization to Folding and Binding Physics in Frontier Reasoning Models},\n  author  = {Gueloglu, Talu},\n  journal = {arXiv preprint},\n  year    = {2026}\n}",
        "@misc{gueloglu2026proteinswithouttools,\n  title  = {Proteins Without Tools: Folding and Binding Physics in Frontier Reasoning Models},\n  author = {Gueloglu, Talu},\n  year   = {2026},\n  url    = {https://github.com/talugueloglu/protein-without-tools}\n}",
        1,
    )
    raw_html = raw_html.replace(
        '<a class="btn primary" href="#abstract">\n        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>\n        Paper\n      </a>',
        '<a class="btn primary" href="#abstract">\n        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>\n        Overview\n      </a>',
        1,
    )

    # 3c. Crisp, direct abstract describing what we test and what happens
    concise_abstract = """  <section id="abstract">
    <h2>Abstract</h2>
    <p>Frontier language models read protein sequences alongside the structural biology literature during pre-training, so a right answer can come from reciting a known entry instead of reasoning about physics. We test three frontier reasoning models on eight protein tasks with every identifier stripped and no external tools &mdash; no alignments, no databases, and no structure predictors &mdash; covering forward folding, inverse folding, remote homology, function prediction, monomer and interface mutational ranking, single-shot design, and 16-round closed-loop binder design.</p>
    <p>To separate recall from generalisation, we group folding targets by structural distance (TM-score) to the pre-May-2026 PDB, pair every mutational hotspot with a same-amino-acid neutral decoy (holding residue-letter guessing to 50%), and test function prediction across matched pre- and post-cutoff CAFA-6 windows. Every task runs alongside a specialist baseline (ESMFold-2, ProteinMPNN, ESM-2 650M, NetGO 4.0, ESMFold-2 + SaProt 650M, Boltz-2 + StaB-ddG, RFdiffusion3 + ProteinMPNN, and BoltzGen).</p>
    <p>Without tools, reasoning models beat sequence embeddings on twilight-zone remote homology, beat structure-aware SaProt on synthetic <em>de novo</em> monomer scans, and climb steadily across 16 rounds of binder feedback. They still fall far behind specialists on 3D forward folding, inverse folding, unseen post-cutoff function prediction, and single-shot design, and collapse on binding interfaces missing from the pre-2026 PDB.</p>
  </section>"""
    abs_pattern = re.compile(r'  <section id="abstract">.*?</section>', re.DOTALL)
    assert abs_pattern.search(raw_html) is not None, "Expected #abstract section not found"
    raw_html = abs_pattern.sub(lambda _: concise_abstract, raw_html, count=1)

    # 3d. Wire GitHub repo link into hero Code button + update #code shell block with fresh-clone setup & actual flags
    raw_html = raw_html.replace(
        '<a class="btn" href="#code">\n        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>\n        Code\n      </a>',
        '<a class="btn" href="https://github.com/talugueloglu/protein-without-tools" target="_blank" rel="noopener">\n        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>\n        GitHub\n      </a>',
        1,
    )
    new_code_block = """  <section id="code">
    <h2>Running the benchmark</h2>
    <p class="lead">Repository: <a href="https://github.com/talugueloglu/protein-without-tools" target="_blank" rel="noopener">github.com/talugueloglu/protein-without-tools</a></p>
    <pre class="shell"><span class="c"># clone &amp; install (Python 3.11+)</span>
git clone https://github.com/talugueloglu/protein-without-tools.git &amp;&amp; cd protein-without-tools
uv venv &amp;&amp; uv pip install -e ".[dev,cloud]"
cp .env.example .env

<span class="c"># unit, anti-cheat, CAFA-6 masking, and leakage tests</span>
.venv/bin/pytest -v

<span class="c"># single-shot tasks 1&ndash;7 (gpt-6.1-sol, gpt-6.2-sol, gpt-6.2-astra)</span>
.venv/bin/python scripts/run_benchmark_llms.py --provider openai --model gpt-6.2-astra --stages 1,2,3,4,5,6,7 --thinking-budget 4096

<span class="c"># task 8: 16-round closed-loop AlphaFold2 binder campaigns</span>
.venv/bin/python scripts/run_stage3b_closed_loop_llms.py --provider openai --model gpt-6.2-astra --mode closed_loop --oracle modal_af2 --shots 16 --thinking-budget 4096

<span class="c"># score Task 4 (CAFA-6 2x3 split) and rebuild website/index.html</span>
.venv/bin/python scripts/eval_t4_cafa6_6cell.py
.venv/bin/python scripts/build_website.py</pre>
  </section>"""
    code_section_pattern = re.compile(r'  <section id="code">.*?</section>', re.DOTALL)
    assert code_section_pattern.search(raw_html), "Expected #code section not found"
    raw_html = code_section_pattern.sub(lambda _: new_code_block, raw_html, count=1)

    # 4. Remove standalone #prep section & nav link (each task's data-prep card now renders directly beneath its figure in #results)
    old_nav_data = '      <a href="#prep">Data</a>\n'
    assert old_nav_data in raw_html, "Expected Data nav link not found"
    raw_html = raw_html.replace(old_nav_data, "", 1)

    prep_section_pattern = re.compile(r'\n  <section id="prep">.*?</section>\n', re.DOTALL)
    assert prep_section_pattern.search(raw_html) is not None, "Expected #prep section not found"
    raw_html = prep_section_pattern.sub("\n", raw_html, count=1)

    old_results_head = '  <section id="results">\n    <h2>Results</h2>'
    new_results_head = (
        '  <section id="results">\n'
        '    <h2>Results</h2>\n'
        '    <p class="lead">Every prompt strips PDB IDs, gene names, organism labels, and sequence alignments. '
        'Open the card under any figure for datasets, target selection, and holdout rules.</p>'
    )
    assert old_results_head in raw_html, "Expected #results heading not found"
    raw_html = raw_html.replace(old_results_head, new_results_head, 1)

    # 5. Trim TASK_BLURB in Task Explorer to active, plain-English descriptions
    concise_task_blurb = """var TASK_BLURB = {
  '1':'We give the model one sequence with no IDs or alignments. It predicts residue contacts and secondary structure, which we fold into 3D and align to the native structure.',
  '2':'We strip a backbone to bare Cα coordinates (every residue marked UNK), plus backbone angles and burial counts. The model writes a sequence, and ESMFold-2 refolds it to check if it recovers the shape.',
  '3':'We give the model two sequences — often below 15% identity — and ask whether they share a SCOPe superfamily, graded against class- and identity-matched decoys.',
  '4':'We give the model one sequence and ask for experimental Gene Ontology terms across MF, BP, and CC, graded by CAFA-6 IA-weighted wFmax across pre- and post-cutoff NK, LK, and PK splits.',
  '5':'We give the model one sequence and 18 X→Ala mutations (6 hotspots, 6 moderate, 6 same-AA decoys) and ask it to rank them by measured folding stability loss (ΔΔG or DMS activity).',
  '6':'We give the model two interacting chains and 18 interface X→Ala mutations (with monomer-unfolding sites removed) and ask it to rank which mutations weaken binding most.',
  '7':'We give the model a target fold or a fixed crystal motif and ask for a single 1D sequence. We fold it with ESMFold-2 and OpenFold3 and grade it on fold/motif accuracy, pLDDT ≥ 70, and cross-oracle scRMSD ≤ 2 Å.',
  '8':'We give the model a target chain and five hotspot residues. Across 16 rounds, it proposes 55–85 AA binders and receives AlphaFold2 interface scores after each attempt.'
};"""
    blurb_pattern = re.compile(r"var TASK_BLURB = \{.*?\};", re.DOTALL)
    assert blurb_pattern.search(raw_html) is not None, "Expected TASK_BLURB block not found"
    raw_html = blurb_pattern.sub(lambda _: concise_task_blurb, raw_html, count=1)

    # 6. Auto-link upstream dataset/record identifiers in Datapoint Browser Source field (PDB, UniProt, MaveDB, SCOPe, SKEMPI, Open-AlphaSeq)
    old_src_line = "'<div class=\"ds-field\"><b>Source</b><span>' + esc(d.src || '—') + (d.desc ? ' · ' + esc(d.desc) : '') + '</span></div>' +"
    new_src_line = "'<div class=\"ds-field\"><b>Source</b><span>' + linkifySource(d.src || '—', d) + (d.desc ? ' · ' + esc(d.desc) : '') + '</span></div>' +"
    assert old_src_line in raw_html, "Expected dsRenderDetail Source line not found"
    raw_html = raw_html.replace(old_src_line, new_src_line, 1)

    linkify_fn = """function linkifySource(raw, d) {
  var s = esc(raw || '—');
  s = s.replace(/\\b(urn:mavedb:[0-9a-zA-Z-]+)\\b/g, '<a href="https://www.mavedb.org/score-sets/$1" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/\\bUniProt\\s+([A-Z0-9]{6,10})\\b/g, 'UniProt <a href="https://www.uniprot.org/uniprotkb/$1" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/\\bPDB\\s+([0-9][A-Za-z0-9]{3})\\b/g, 'PDB <a href="https://www.rcsb.org/structure/$1" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/\\(([0-9][A-Z0-9]{3})\\)/g, '(<a href="https://www.rcsb.org/structure/$1" target="_blank" rel="noopener">$1</a>)');
  s = s.replace(/\\b(d[0-9][a-z0-9]{3}[a-z0-9_\\.]{2})\\b/g, '<a href="https://scop.berkeley.edu/sid=$1" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/Open-AlphaSeq/g, '<a href="https://huggingface.co/datasets/aalphabio/open-alphaseq" target="_blank" rel="noopener">Open-AlphaSeq</a>');
  s = s.replace(/SKEMPI(?:\\s+v2\\.0)?/g, '<a href="https://life.bsc.es/pid/skempi2/" target="_blank" rel="noopener">$&</a>');
  if (d && d.t === 5 && (d.st === 'de_novo_ddg' || d.st === 'natural_ddg')) {
    s += ' · <a href="https://huggingface.co/datasets/RosettaCommons/MegaScale" target="_blank" rel="noopener">MegaScale</a>';
  }
  if (d && d.t === 6 && d.st === 'known_pdb_pre2026' && s.indexOf('skempi2') === -1 && s.indexOf('mavedb') === -1) {
    s += ' · <a href="https://life.bsc.es/pid/skempi2/" target="_blank" rel="noopener">SKEMPI v2.0</a>';
  }
  return s;
}
function dsRenderDetail() {"""
    assert "function dsRenderDetail() {" in raw_html, "Expected dsRenderDetail function not found"
    raw_html = raw_html.replace("function dsRenderDetail() {", linkify_fn, 1)

    # 7. Prevent meter-row value pills from clashing/overflowing with 4 series on narrower screens
    old_mt_css = "  .mt-vals{display:flex;gap:12px;}"
    new_mt_css = "  .mt-vals{display:flex;flex-wrap:wrap;gap:4px 10px;justify-content:flex-end;}"
    assert old_mt_css in raw_html, "Expected .mt-vals CSS line not found"
    raw_html = raw_html.replace(old_mt_css, new_mt_css, 1)

    # 8. Update openPrep() to expand, scroll to, and flash the targeted per-figure <details id="prep-tK"> card
    old_open_prep_fn = """function openPrep(ev, href) {
  if (ev) ev.preventDefault();
  var d = document.getElementById('prep-d');
  if (!d) return;
  d.open = true;
  var id = (href || '#prep').replace('#', '');
  var go = function(){
    var t = document.getElementById(id) || document.getElementById('prep');
    if (!t) return;
    window.scrollTo({ top: t.getBoundingClientRect().top + window.scrollY - 86, behavior: 'smooth' });
    t.classList.remove('prep-flash'); void t.offsetWidth; t.classList.add('prep-flash');
  };
  if (window.requestAnimationFrame) window.requestAnimationFrame(go); else go();
}"""
    new_open_prep_fn = """function openPrep(ev, href) {
  if (ev) ev.preventDefault();
  var id = (href || '#prep').replace('#', '');
  if (id === 'prep') {
    id = (typeof dsTask === 'number' && dsTask >= 1 && dsTask <= 8) ? ('prep-t' + dsTask) : 'prep-t1';
  }
  var t = document.getElementById(id) || document.getElementById('prep-t1');
  if (!t) return;
  if (t.tagName === 'DETAILS') t.open = true;
  var go = function(){
    window.scrollTo({ top: t.getBoundingClientRect().top + window.scrollY - 86, behavior: 'smooth' });
    t.classList.remove('prep-flash'); void t.offsetWidth; t.classList.add('prep-flash');
  };
  if (window.requestAnimationFrame) window.requestAnimationFrame(go); else go();
}"""
    assert old_open_prep_fn in raw_html, "Expected openPrep function not found"
    raw_html = raw_html.replace(old_open_prep_fn, new_open_prep_fn, 1)

    return raw_html


def main() -> None:
    template_html = REPO / "website" / "template.html"
    if not template_html.exists():
        raise FileNotFoundError(f"Website template not found: {template_html}")

    raw_html = _apply_redesign_polish(template_html.read_text(encoding="utf-8"))
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
            f"We give each model two protein sequences and ask whether they belong to the same SCOPe superfamily, "
            f"scoring recall at 10% false-positive rate against 30 class- and identity-matched decoys. "
            f"Everyone solves the easy pairs, but below 15% sequence identity (Rungs 2 and 3) GPT-6.2 Sol and "
            f"GPT-6.2 Astra pull ahead of ESM-2 (650M) (grey, AUROC {t3_chosen['overall_auroc']:.3f})."
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

    # Task 4: CAFA-6 2x3 (Pre/Post x NK/LK/PK, N=120) — Official NetGO 4.0 + 3 reasoning models
    t4_cafa6 = json.loads((REPO / "outputs" / "results" / "t4_cafa6_6cell_results.json").read_text(encoding="utf-8"))
    t4_base_name = "NetGO 4.0"
    t4_chosen = t4_cafa6["models"]["baseline_t4"]
    t4_astra = t4_cafa6["models"]["gpt-6.2-astra_think4096"]
    t4_sol2 = t4_cafa6["models"]["gpt-6.2-sol_think4096"]
    t4_sol1 = t4_cafa6["models"]["gpt-6.1-sol_think4096"]
    pnk_netgo = t4_chosen["by_cell"]["post_nk"]
    pnk_llms = [m["by_cell"]["post_nk"]["macro_wfmax"] for m in (t4_sol1, t4_sol2, t4_astra)]
    t4_prose = (
        f"We give each model one sequence and ask it to predict Gene Ontology terms for 20 proteins that had zero "
        f"experimental annotations before May 2026 and whose papers came out between May and October 2026. "
        f"NetGO 4.0 (grey, macro {pnk_netgo['macro_wfmax']:.3f}) beats all three reasoning models "
        f"({min(pnk_llms):.3f}–{max(pnk_llms):.3f}) across Molecular Function, Biological Process, and Cellular Component."
    )
    t4b_prose = (
        f"To separate recall from blind prediction, we compare matched 6-month windows before the cutoff (May–Oct 2025) "
        f"and after it (May–Oct 2026) across No-Knowledge (NK), Limited-Knowledge (LK, graded only on the newly added "
        f"ontology), and Partial-Knowledge (PK, pre-cutoff terms masked out of both truth and predictions). "
        f"On NK, GPT-6.2 Astra beats NetGO 4.0 before the cutoff ({t4_astra['by_cell']['pre_nk']['macro_wfmax']:.3f} vs "
        f"{t4_chosen['by_cell']['pre_nk']['macro_wfmax']:.3f}) but drops −"
        f"{(t4_astra['by_cell']['pre_nk']['macro_wfmax'] - t4_astra['by_cell']['post_nk']['macro_wfmax']):.3f} afterward "
        f"({t4_astra['by_cell']['post_nk']['macro_wfmax']:.3f} vs {pnk_netgo['macro_wfmax']:.3f}). "
        f"<b>Caveat on LK and PK:</b> most targets there are well-known 1990–2007 human, mouse, and rat genes "
        f"(such as LMAN1, GTPBP6, and Prion) whose biology was published decades ago and only received a new GOA stamp "
        f"in 2026, so Post-cutoff NK above is the only clean blind test."
    )

    # 5. Task 5: ESMFold-2 -> SaProt (650M, AF2)
    t5_res = json.loads((REPO / "outputs" / "results" / "baseline_t5_saprot_650m.json").read_text(encoding="utf-8"))
    t5_per = {g["target_id"]: g for g in t5_res["per_target"]}

    # 6. Task 6: Boltz-2 -> StaB-ddG (stability_finetuned.pt, ZERO SKEMPI fine-tuning)
    t6_res = json.loads((REPO / "outputs" / "results" / "baseline_t6_stabddg.json").read_text(encoding="utf-8"))
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

    # --- Rebuild Task 4 rows in DS from the 120-target CAFA-6 2x3 manifest + predictions ---
    t4_manifest = json.loads((REPO / "data" / "stage2b_cafa" / "manifest.json").read_text(encoding="utf-8"))["targets"]
    t4_prompts = {
        json.loads(ln)["target_id"]: json.loads(ln)["prompt"]
        for ln in (REPO / "outputs" / "prompts" / "stage2b_cafa_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    t4_raw_llms: Dict[str, Dict[str, str]] = {}
    for mk in ("gpt-6.1-sol_think4096", "gpt-6.2-sol_think4096", "gpt-6.2-astra_think4096"):
        p_f = REPO / "outputs" / "predictions" / f"{mk}_stage2b.jsonl"
        t4_raw_llms[mk] = {
            json.loads(ln)["target_id"]: str(json.loads(ln).get("text") or "")
            for ln in p_f.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

    new_t4_ds: List[Dict[str, Any]] = []
    for row in t4_manifest:
        tid = row["target_id"]
        parts_gt = []
        for asp, k in (("MF", "delta_terms_mf"), ("BP", "delta_terms_bp"), ("CC", "delta_terms_cc")):
            if asp in row.get("eval_aspects", []):
                terms = row.get(k) or row.get(f"true_terms_{asp.lower()}", [])
                if terms:
                    parts_gt.append(f"{asp} ({len(terms)}): " + ", ".join(terms[:5]))
        sc_t4: Dict[str, Any] = {}
        if tid in t4_chosen["per_target"]:
            sc_t4["baseline_t4"] = t4_chosen["per_target"][tid]
        for mk in ("gpt-6.1-sol_think4096", "gpt-6.2-sol_think4096", "gpt-6.2-astra_think4096"):
            rec_m = dict(t4_cafa6["models"][mk]["per_target"].get(tid, {}))
            if tid in t4_raw_llms[mk]:
                rec_m["raw"] = t4_raw_llms[mk][tid]
            sc_t4[mk] = rec_m
        new_t4_ds.append(
            {
                "t": 4,
                "id": tid,
                "src": f"UniProt {row['uniprot_accession']}",
                "desc": f"{row['organism']} ({row['knowledge_type']}, {', '.join(row.get('eval_aspects', []))})",
                "st": row["temporal_tier"],
                "len": int(row["length"]),
                "seq": row["sequence"],
                "gt": " | ".join(parts_gt),
                "pr": t4_prompts.get(tid, ""),
                "sc": sc_t4,
            }
        )

    # Replace old Task 4 entries in DS in-place (preserving task ordering 1..8)
    first_t4_idx = next(i for i, d in enumerate(ds) if d["t"] == 4)
    ds = [d for d in ds[:first_t4_idx] if d["t"] != 4] + new_t4_ds + [d for d in ds[first_t4_idx:] if d["t"] != 4]

    # --- Inject into DS (all 582 datapoints) ---
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
            pass
        elif t == 5 and tid in t5_per:
            r = t5_per[tid]
            sc["baseline_saprot_650m"] = {
                "ndcg_at_5": round(float(r["ndcg_at_5"]), 3),
                "same_aa_win_rate": round(float(r["same_aa_win_rate"]), 3),
                "spearman_rho": round(float(r["spearman_rho"]), 3),
            }
        elif t == 6 and tid in t6_per:
            r = t6_per[tid]
            sc["baseline_stabddg"] = {
                "ndcg_at_5": round(float(r["ndcg_at_5"]), 3),
                "same_aa_win_rate": round(float(r["same_aa_win_rate"]), 3),
                "spearman_rho": round(float(r["spearman_rho"]), 3),
            }
        elif t == 7:
            def _t7_pc_pass(rec: Dict[str, Any]) -> bool:
                sc_ok = rec.get("cross_oracle_sc_rmsd") is not None and float(rec["cross_oracle_sc_rmsd"]) <= 2.0
                pl_ok = rec.get("plddt_mean") is not None and float(rec["plddt_mean"]) >= 70.0
                hyg_ok = bool(rec.get("gate1_hygiene")) and bool(rec.get("gate5_novelty_lt30"))
                if rec.get("category") == "fold_conditioned":
                    geom_ok = rec.get("tm_to_target_fold") is not None and float(rec["tm_to_target_fold"]) >= 0.50
                else:
                    geom_ok = rec.get("motif_ca_rmsd") is not None and float(rec["motif_ca_rmsd"]) <= 1.0
                return bool(hyg_ok and sc_ok and pl_ok and geom_ok)

            if tid in t7_per:
                r = t7_per[tid]
                sc["baseline_rfd3_mpnn"] = {
                    "plddt_mean": round(float(r["plddt_mean"] or 0.0), 2),
                    "tm_to_target_fold": round(float(r["tm_to_target_fold"]), 4) if r.get("tm_to_target_fold") is not None else None,
                    "motif_ca_rmsd": r.get("motif_ca_rmsd"),
                    "cross_oracle_sc_rmsd": r.get("cross_oracle_sc_rmsd"),
                    "esmfold2_4gate_pass": bool(r["esmfold2_4gate_pass"]),
                    "designable_pass": _t7_pc_pass(r),
                }
            for mk in ("gpt-6.1-sol_think4096", "gpt-6.2-sol_think4096", "gpt-6.2-astra_think4096"):
                m_pt = {g["task_id"]: g for g in t7_rep["models"][mk]["per_task"]}
                if mk in sc and tid in m_pt:
                    sc[mk]["designable_pass"] = _t7_pc_pass(m_pt[tid])
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

    cell_keys = ["pre_nk", "pre_lk", "pre_pk", "post_nk", "post_lk", "post_pk"]
    pnk_n_by_asp = t4_chosen["by_cell"]["post_nk"]["n_by_aspect"]
    fd["t4"] = {
        "aspect_ns": [f"n={pnk_n_by_asp['MF']}", f"n={pnk_n_by_asp['BP']}", f"n={pnk_n_by_asp['CC']}"],
        "cells": [
            "Pre-cutoff\nNK",
            "Pre-cutoff\nLK",
            "Pre-cutoff\nPK",
            "Post-cutoff\nNK",
            "Post-cutoff\nLK",
            "Post-cutoff\nPK",
        ],
        "ns": ["n=20"] * 6,
        "series": [
            {
                "k": mk,
                "n": t4_base_name if mk == "baseline_t4" else mk,
                "base": (mk == "baseline_t4"),
                # Headline macro & MF/BP/CC reflect strict Post-cutoff NK (uncontaminated blind test)
                "macro": round(float(t4_cafa6["models"][mk]["by_cell"]["post_nk"]["macro_wfmax"]), 3),
                "v": [
                    round(float(t4_cafa6["models"][mk]["by_cell"]["post_nk"]["aspects"]["MF"]), 3),
                    round(float(t4_cafa6["models"][mk]["by_cell"]["post_nk"]["aspects"]["BP"]), 3),
                    round(float(t4_cafa6["models"][mk]["by_cell"]["post_nk"]["aspects"]["CC"]), 3),
                ],
                "cells_v": [
                    round(float(t4_cafa6["models"][mk]["by_cell"][ck]["macro_wfmax"]), 3)
                    for ck in cell_keys
                ],
            }
            for mk in ("baseline_t4", "gpt-6.1-sol_think4096", "gpt-6.2-sol_think4096", "gpt-6.2-astra_think4096")
        ],
    }

    fd["t5_base"] = [
        {
            "n": "SaProt (650M)",
            "ndcg": round(float(t5_res["overall"]["ndcg_at_5"]), 3),
            "win": round(float(t5_res["overall"]["same_aa_win_rate"]), 3),
        }
    ]

    llm_t6 = [s for s in fd["t6"]["series"] if not s.get("base")]
    fd["t6"]["series"] = [
        {
            "n": "Boltz-2 + StaB-ddG",
            "base": True,
            "overall": round(float(t6_res["overall"]["spearman_rho"]), 3),
            "memo": {
                "known_pdb_pre2026": round(float(t6_res["by_memorization_stratum"]["known_pdb_pre2026"]["spearman_rho"]), 3),
                "unseen_or_post2026": round(float(t6_res["by_memorization_stratum"]["unseen_or_post2026"]["spearman_rho"]), 3),
            },
        }
    ] + llm_t6

    # Task 7 Proteina-Complexa structural metrics:
    # [Fold TM >= 0.50 (n=24), Motif RMSD <= 1.0A (n=16), pLDDT >= 70 (n=40), scRMSD <= 2.0A (n=40), Designable pass (n=40)]
    def _t7_pc_vec(mkey: str) -> List[float]:
        pt = t7_rep["models"][mkey]["per_task"]
        folds = [x for x in pt if x["category"] == "fold_conditioned"]
        motifs = [x for x in pt if x["category"] == "motif_scaffolding"]
        fold_ok = sum(1 for x in folds if (x.get("tm_to_target_fold") or 0.0) >= 0.50) / float(len(folds))
        motif_ok = sum(1 for x in motifs if x.get("motif_ca_rmsd") is not None and float(x["motif_ca_rmsd"]) <= 1.0) / float(len(motifs))
        pl70_ok = sum(1 for x in pt if (x.get("plddt_mean") or 0.0) >= 70.0) / float(len(pt))
        sc2_ok = sum(1 for x in pt if x.get("cross_oracle_sc_rmsd") is not None and float(x["cross_oracle_sc_rmsd"]) <= 2.0) / float(len(pt))
        all_ok = sum(
            1
            for x in pt
            if bool(x.get("gate1_hygiene"))
            and bool(x.get("gate5_novelty_lt30"))
            and (x.get("plddt_mean") or 0.0) >= 70.0
            and x.get("cross_oracle_sc_rmsd") is not None
            and float(x["cross_oracle_sc_rmsd"]) <= 2.0
            and (
                ((x.get("tm_to_target_fold") or 0.0) >= 0.50)
                if x["category"] == "fold_conditioned"
                else (x.get("motif_ca_rmsd") is not None and float(x["motif_ca_rmsd"]) <= 1.0)
            )
        ) / float(len(pt))
        return [round(v, 3) for v in (fold_ok, motif_ok, pl70_ok, sc2_ok, all_ok)]

    fd["t7"]["gates"] = [
        "Fold fidelity · TM ≥ 0.50",
        "Motif fidelity · RMSD ≤ 1.0 Å",
        "Confidence · pLDDT ≥ 70",
        "Self-consistency · scRMSD ≤ 2 Å",
        "Designable · all criteria",
    ]
    fd["t7"]["series"] = [
        {"n": "RFdiffusion3 + ProteinMPNN", "base": True, "v": _t7_pc_vec("baseline_rfd3_mpnn")},
        {"n": "gpt-6.1-sol_think4096", "base": False, "v": _t7_pc_vec("gpt-6.1-sol_think4096")},
        {"n": "gpt-6.2-sol_think4096", "base": False, "v": _t7_pc_vec("gpt-6.2-sol_think4096")},
        {"n": "gpt-6.2-astra_think4096", "base": False, "v": _t7_pc_vec("gpt-6.2-astra_think4096")},
    ]

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
        f"We give each model a target protein and five surface hotspots, let it propose a 55–85 AA binder, and feed back "
        f"AlphaFold2 interface scores over 16 rounds across 12 epitopes (while BoltzGen in grey generates 16 independent "
        f"all-atom designs). Reasoning models use the feedback to climb steadily, though BoltzGen reaches a higher mean "
        f"score ({bgen_s0:.2f} → {bgen_s16:.2f}; Hit@16 = {bgen_hit_pct:.1f}%)."
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
        .replace("__T4B_PROSE__", t4b_prose)
        .replace("__T8_PROSE__", t8_prose)
    )

    full_html = "\n".join(head_lines) + "\n" + js_tail + "\n"

    # Validate the entire embedded <script> block with `node --check` when Node.js is available
    if shutil.which("node"):
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
    else:
        print("[INFO] `node` not found on PATH; skipping JS syntax check.", flush=True)

    # Write standalone website and synced 582-datapoint JSONL export
    index_dst = REPO / "website" / "index.html"
    index_dst.write_text(full_html, encoding="utf-8")
    print(f"Wrote {index_dst} ({len(full_html):,} bytes)", flush=True)

    data_dir = REPO / "website" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dst = data_dir / "datapoints_v1.jsonl"
    jsonl_dst.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ds) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {jsonl_dst} ({len(ds)} datapoints, {jsonl_dst.stat().st_size:,} bytes)", flush=True)

    print("\n=== SUMMARY OF ESTABLISHED BASELINES INJECTED ===", flush=True)
    print(f"  Task 1 (ESMFold-2)                 : TM={t1_res['overall']['tm_score']:.3f} | P@L={t1_res['overall']['p_at_l_sep24']:.3f} | Q3={t1_res['overall']['q3_accuracy']:.3f}", flush=True)
    sc_tms = [r["sc_tm_score"] for r in t2_rows.values()]
    sc_2a = [1.0 if r["sc_rmsd_ca"] < 2.0 else 0.0 for r in t2_rows.values()]
    print(f"  Task 2 (ProteinMPNN)               : scTM={np.mean(sc_tms):.3f} | scRMSD<2A={np.mean(sc_2a)*100:.1f}% | NSR={np.mean([r['nsr_overall'] for r in t2_rows.values()]):.3f}", flush=True)
    print(f"  Task 3 ({t3_base_name:<25s}) : AUROC={t3_base_series['auroc']:.3f} | Rungs={t3_base_series['v']}", flush=True)
    print(f"  Task 4 ({t4_base_name:<25s}) : Macro wFmax={t4_chosen['macro_wfmax']:.3f} | MF/BP/CC={t4_chosen['aspects']}", flush=True)
    print(f"  Task 5 (ESMFold-2 -> SaProt 650M)  : NDCG@5={t5_res['overall']['ndcg_at_5']:.3f} | SameAA={t5_res['overall']['same_aa_win_rate']:.3f} | Rho={t5_res['overall']['spearman_rho']:.3f}", flush=True)
    print(f"  Task 6 (Boltz-2 -> StaB-ddG)       : Rho={t6_res['overall']['spearman_rho']:.3f} | Known={t6_res['by_memorization_stratum']['known_pdb_pre2026']['spearman_rho']:.3f} | Unseen={t6_res['by_memorization_stratum']['unseen_or_post2026']['spearman_rho']:.3f}", flush=True)
    print(f"  Task 7 (RFdiffusion3 + ProteinMPNN): 4-Gate={t7_b['esmfold2_4gate_success_rate']*100:.1f}% | 5-Gate={t7_b['design_success_rate_5gate']*100:.1f}% | pLDDT>=80={t7_b['gate2_foldable_rate']*100:.1f}% | pLDDT={t7_b['mean_plddt']:.1f}", flush=True)
    print(f"  Task 8 (BoltzGen)                  : Hit@16={t8_sum['hit_rate']*100:.1f}% | AF2 Composite={bgen_s0:.3f} -> {bgen_s16:.3f} | On-Epitope ipTM={t8_sum['mean_best_iptm']:.3f}", flush=True)


if __name__ == "__main__":
    main()
