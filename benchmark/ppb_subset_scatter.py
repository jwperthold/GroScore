#!/usr/bin/env python3
"""
Scatter plot: predicted pKd (from GroScore) vs experimental pKd.

Layout: 3×2 grid — one subplot per PPB subset (5 panels) plus one combined
panel with all subsets coloured and overall correlation statistics.

Conversion: pKd = -0.0176 × GroScore + 3.4513  (AMBER19SB linear fit)

Usage:
    python3 ppb_subset_scatter.py [-o output.png]

    default output: benchmark/ppb_subset_scatter.png
"""
import os, sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

SCORES_GS     = '/home/jwperthold/GroScore/bm_ppb_amber/scores_avg.gs'
BENCHMARK_CSV = '/home/jwperthold/GroScore/bm_ppb_amber/benchmark.csv'
OUT = os.path.join(os.path.dirname(__file__), 'ppb_subset_scatter.png')

for i, a in enumerate(sys.argv[1:], 1):
    if a == '-o' and i < len(sys.argv) - 1:
        OUT = sys.argv[i + 1]

SLOPE     = -0.0176
INTERCEPT =  3.4513

# ── load ──────────────────────────────────────────────────────────────────────
scores = {}
with open(SCORES_GS) as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.strip().split()
        if len(p) >= 2:
            try:
                v = float(p[1])
                if not np.isnan(v):
                    scores[p[0].upper()] = v
            except ValueError:
                pass

rows = []
with open(BENCHMARK_CSV) as f:
    for row in csv.DictReader(f):
        pdb = row['pdb_id'].strip().upper()
        if pdb not in scores: continue
        try:
            pkd = float(row['pkd'])
        except (ValueError, KeyError):
            continue
        pred_pkd = SLOPE * scores[pdb] + INTERCEPT
        rows.append({'pdb': pdb, 'source': row.get('source', '').strip(),
                     'pkd': pkd, 'pred_pkd': pred_pkd})

print(f'Total matched: {len(rows)}')

# ── assign subsets ────────────────────────────────────────────────────────────
SUBSETS = [
    ('Affinity Benchmark v5.5', 'AffinityBench', '#1565C0'),
    ('SKEMPI v2.0',             'SKEMPI',        '#2E7D32'),
    ('PDBbind v2020',           'PDBbind',       '#E65100'),
    ('SAbDab',                  'SAbDab',        '#6A1B9A'),
    ('ATLAS',                   'ATLAS',         '#B71C1C'),
]

bucketed = {key: [] for _, key, _ in SUBSETS}
for r in rows:
    for name, key, _ in SUBSETS:
        if name in r['source']:
            bucketed[key].append(r)
            break

# ── shared axis range ─────────────────────────────────────────────────────────
all_exp  = np.array([r['pkd']      for r in rows])
all_pred = np.array([r['pred_pkd'] for r in rows])
pad      = 0.5
lim_lo   = all_exp.min() - pad
lim_hi   = all_exp.max() + pad

def draw_reference(ax):
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            'k--', linewidth=1, alpha=0.7)
    ax.plot([lim_lo, lim_hi], [lim_lo - 1, lim_hi - 1],
            'k:', linewidth=1, alpha=0.2)
    ax.plot([lim_lo, lim_hi], [lim_lo + 1, lim_hi + 1],
            'k:', linewidth=1, alpha=0.2)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.grid(True, alpha=0.3)

def stats_annotation(x, y):
    r, _  = stats.pearsonr(x, y)
    rho,_ = stats.spearmanr(x, y)
    rmse  = np.sqrt(np.mean((x - y) ** 2))
    return f'r = {r:.2f}, ρ = {rho:.2f}\nRMSE = {rmse:.2f}  n = {len(x)}'

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes_flat = axes.flatten()

for idx, (name, key, color) in enumerate(SUBSETS):
    ax  = axes_flat[idx]
    pts = bucketed[key]

    draw_reference(ax)

    if pts:
        x = np.array([r['pkd']      for r in pts])
        y = np.array([r['pred_pkd'] for r in pts])
        ax.scatter(x, y, color=color, s=14, alpha=0.6, linewidths=0)
        if len(pts) >= 3:
            sl, ic, _, _, _ = stats.linregress(x, y)
            xfit = np.linspace(lim_lo, lim_hi, 100)
            ax.plot(xfit, sl * xfit + ic, color=color, linewidth=2, alpha=0.9)
        ax.text(0.04, 0.96, stats_annotation(x, y),
                transform=ax.transAxes, fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#ccc'))

    ax.set_title(f'{key}', fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('Experimental pKd', fontsize=10)
    ax.set_ylabel('Predicted pKd', fontsize=10)

# ── combined panel (bottom right) ─────────────────────────────────────────────
ax = axes_flat[5]
draw_reference(ax)

for name, key, color in SUBSETS:
    pts = bucketed[key]
    if not pts: continue
    x = np.array([r['pkd']      for r in pts])
    y = np.array([r['pred_pkd'] for r in pts])
    ax.scatter(x, y, color=color, s=12, alpha=0.55, linewidths=0,
               label=f'{key} (n={len(pts)})')
    if len(pts) >= 3:
        sl, ic, _, _, _ = stats.linregress(x, y)
        xfit = np.linspace(lim_lo, lim_hi, 100)
        ax.plot(xfit, sl * xfit + ic, color=color, linewidth=1.5, alpha=0.85)

# Overall stats annotation
pr, _  = stats.pearsonr(all_exp, all_pred)
sr, _  = stats.spearmanr(all_exp, all_pred)
rmse   = np.sqrt(np.mean((all_exp - all_pred) ** 2))
ax.text(0.04, 0.96,
        f'All  r = {pr:.2f}, ρ = {sr:.2f}\nRMSE = {rmse:.2f}  n = {len(rows)}',
        transform=ax.transAxes, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#ccc'))

ax.set_title('All subsets combined', fontsize=12, fontweight='bold')
ax.set_xlabel('Experimental pKd', fontsize=10)
ax.set_ylabel('Predicted pKd', fontsize=10)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor='#ccc')

fig.suptitle(
    'GroScore Predictions vs Experimental Binding Affinity — PPB Subsets',
    fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Saved → {OUT}')
