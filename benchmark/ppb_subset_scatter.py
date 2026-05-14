#!/usr/bin/env python3
"""
Scatter plot: predicted pKd (from GroScore) vs experimental pKd, coloured by PPB subset.

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
    ('SAbDab',                  'SAbDab',         '#6A1B9A'),
    ('ATLAS',                   'ATLAS',          '#B71C1C'),
]

bucketed = {key: [] for _, key, _ in SUBSETS}
unbucketed = []
for r in rows:
    matched = False
    for name, key, _ in SUBSETS:
        if name in r['source']:
            bucketed[key].append(r)
            matched = True
            break
    if not matched:
        unbucketed.append(r)

# ── plot ──────────────────────────────────────────────────────────────────────
all_exp  = np.array([r['pkd']      for r in rows])
all_pred = np.array([r['pred_pkd'] for r in rows])

# Axis range: follow experimental pKd extent (clips extreme-score outliers)
pad     = 0.5
min_pkd = all_exp.min() - pad
max_pkd = all_exp.max() + pad

pearson_r,  _ = stats.pearsonr(all_exp, all_pred)
spearman_r, _ = stats.spearmanr(all_exp, all_pred)
rmse = np.sqrt(np.mean((all_exp - all_pred) ** 2))

plt.figure(figsize=(10, 8))
ax = plt.gca()

for name, key, color in SUBSETS:
    pts = bucketed[key]
    if not pts: continue
    x = np.array([r['pkd']      for r in pts])
    y = np.array([r['pred_pkd'] for r in pts])
    ax.scatter(x, y, color=color, s=12, alpha=0.55,
               linewidths=0, label=f'{key} (n={len(pts)})')
    if len(pts) >= 3:
        sl, ic, _, _, _ = stats.linregress(x, y)
        xfit = np.array([max(x.min(), min_pkd), min(x.max(), max_pkd)])
        ax.plot(xfit, sl * xfit + ic, color=color, linewidth=1.5, alpha=0.85)

if unbucketed:
    x = [r['pkd']      for r in unbucketed]
    y = [r['pred_pkd'] for r in unbucketed]
    ax.scatter(x, y, color='#888', s=12, alpha=0.4,
               linewidths=0, label=f'Other (n={len(unbucketed)})')

# Identity and ±1 pKd bands
ax.plot([min_pkd, max_pkd], [min_pkd, max_pkd],
        'k--', linewidth=1, alpha=0.7, label='Identity')
ax.plot([min_pkd, max_pkd], [min_pkd - 1, max_pkd - 1],
        'k:', linewidth=1, alpha=0.2, label='Identity ± 1')
ax.plot([min_pkd, max_pkd], [min_pkd + 1, max_pkd + 1],
        'k:', linewidth=1, alpha=0.2)

ax.set_xlabel('Experimental pKd', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted pKd (from GroScore)', fontsize=12, fontweight='bold')
ax.set_title(
    'GroScore Predictions vs Experimental Binding Affinity — PPB Subsets\n'
    f'Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}, RMSE = {rmse:.2f}',
    fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right', framealpha=0.9, edgecolor='#ccc')
ax.grid(True, alpha=0.3)
ax.set_xlim(min_pkd, max_pkd)
ax.set_ylim(min_pkd, max_pkd)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Saved → {OUT}')
