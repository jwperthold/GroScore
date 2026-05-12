#!/usr/bin/env python3
import numpy as np
from scipy import stats
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load benchmark pKd values (deduplicated, from run directory)
pkd_map = {}
with open('/home/jwperthold/GroScore/bm_ppb_amber/benchmark.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pdb_id = row['pdb_id'].strip().upper()
        try:
            pkd_map[pdb_id] = float(row['pkd'])
        except (ValueError, KeyError):
            pass

def load_scores(path):
    """Load scores file -> dict pdb_id: (score, ci95)."""
    scores = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) < 3: continue
            pdb = parts[0].upper()
            try:
                score = float(parts[1])
                ci95 = float(parts[2])
                if not (np.isnan(score) or np.isnan(ci95)):
                    scores[pdb] = (score, ci95)
            except ValueError:
                pass
    return scores

# Load all cycle files n=2..5
cycle_files = {
    n: f'/home/jwperthold/GroScore/bm_ppb_amber/scores_avg_c{n}.gs'
    for n in range(2, 6)
}
cycle_scores = {n: load_scores(p) for n, p in cycle_files.items()}

# Error thresholds: 15, 20, ... up to ~200 kJ/mol
thresholds = np.arange(15, 205, 5)

colors = {2: '#d62728', 3: '#ff7f0e', 4: '#2ca02c', 5: '#1f77b4'}
labels = {2: 'n ≤ 2 cycles', 3: 'n ≤ 3 cycles', 4: 'n ≤ 4 cycles', 5: 'n ≤ 5 cycles'}

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()

for n in [2, 3, 4, 5]:
    sc = cycle_scores[n]
    spearman_vals = []
    n_vals = []

    for t in thresholds:
        pairs = [
            (sc[pdb][0], pkd_map[pdb])
            for pdb in sc
            if pdb in pkd_map and sc[pdb][1] <= t
        ]
        if len(pairs) < 10:
            spearman_vals.append(np.nan)
            n_vals.append(len(pairs))
            continue
        scores_arr = np.array([p[0] for p in pairs])
        pkd_arr = np.array([p[1] for p in pairs])
        sr, _ = stats.spearmanr(scores_arr, pkd_arr)
        spearman_vals.append(-sr)  # negate: larger |score| = stronger binding
        n_vals.append(len(pairs))

    spearman_vals = np.array(spearman_vals)
    n_vals = np.array(n_vals)
    valid = ~np.isnan(spearman_vals)

    ax1.plot(thresholds[valid], spearman_vals[valid],
             color=colors[n], lw=2, marker='o', ms=4, label=labels[n])
    # N as faint dashed line on right axis (only for n=5 to avoid clutter)
    if n == 5:
        ax2.plot(thresholds[valid], n_vals[valid],
                 color='gray', lw=1, ls='--', alpha=0.5)

ax1.set_xlabel('Maximum CI₉₅ threshold (kJ/mol)', fontsize=12)
ax1.set_ylabel('Spearman ρ', fontsize=12)
ax1.set_xlim(thresholds[0] - 2, thresholds[-1] + 2)
ax1.set_ylim(0, 0.45)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax1.grid(True, alpha=0.25)
ax1.legend(fontsize=10, loc='lower right')
ax1.set_title('Spearman ρ vs. score error threshold (PPB-Affinity benchmark)',
              fontsize=12, fontweight='bold')

ax2.set_ylabel('N structures included (n=5, dashed)', fontsize=10, color='gray')
ax2.tick_params(axis='y', colors='gray')
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.1)

plt.tight_layout()
out = '/home/jwperthold/GroScore/benchmark/ppb_convergence_ci.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('Saved:', out)

# Also print a summary table
print('\n%6s  %5s  ' % ('CI<=', 'n=2') + '  '.join('%5s' % ('n=%d' % n) for n in [3,4,5]) + '  (N at n=5)')
print('-' * 55)
for i, t in enumerate(thresholds):
    row = '%5.0f  ' % t
    n5_n = None
    for n in [2, 3, 4, 5]:
        sc = cycle_scores[n]
        pairs = [(sc[pdb][0], pkd_map[pdb]) for pdb in sc
                 if pdb in pkd_map and sc[pdb][1] <= t]
        if n == 5:
            n5_n = len(pairs)
        if len(pairs) < 10:
            row += '   ---  '
        else:
            scores_arr = np.array([p[0] for p in pairs])
            pkd_arr = np.array([p[1] for p in pairs])
            sr, _ = stats.spearmanr(scores_arr, pkd_arr)
            row += '%+6.3f  ' % (-sr)
    row += '  (N=%d)' % (n5_n or 0)
    print(row)
