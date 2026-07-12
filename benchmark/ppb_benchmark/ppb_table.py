#!/usr/bin/env python3
import numpy as np
from scipy import stats
from scipy.stats import rankdata
import csv

# Load scores (filter out nan)
scores = {}
with open('/home/jwperthold/GroScore/bm_ppb_amber/scores_avg.gs') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                v = float(parts[1])
                if not np.isnan(v):
                    scores[parts[0].upper()] = v
            except ValueError:
                pass

# Load deduplicated benchmark.csv from the run directory
rows = []
with open('/home/jwperthold/GroScore/bm_ppb_amber/benchmark.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pdb_id = row['pdb_id'].strip().upper()
        if pdb_id not in scores:
            continue
        try:
            pkd = float(row['pkd'])
        except (ValueError, KeyError):
            continue
        source = row.get('source', '').strip()
        rows.append({'pdb': pdb_id, 'source': source, 'pkd': pkd, 'score': scores[pdb_id]})

print('Total matched (non-nan): %d' % len(rows))

def bootstrap_ci(x, y, n=50000, seed=42):
    np.random.seed(seed)
    n_pts = len(x)
    idx = np.random.randint(0, n_pts, (n, n_pts))
    xb, yb = x[idx], y[idx]
    xv = np.var(xb, axis=1); yv = np.var(yb, axis=1)
    valid = (xv > 0) & (yv > 0)
    xb, yb = xb[valid], yb[valid]
    n_v = xb.shape[0]
    if n_v == 0:
        return (0.0, 0.0, 0.0, 0.0)
    xm = xb.mean(1, keepdims=True); ym = yb.mean(1, keepdims=True)
    xc = xb - xm; yc = yb - ym
    pearson_b = (xc*yc).sum(1) / np.sqrt((xc**2).sum(1) * (yc**2).sum(1))
    rx = np.vstack([rankdata(xb[i]) for i in range(n_v)]).astype(float)
    ry = np.vstack([rankdata(yb[i]) for i in range(n_v)]).astype(float)
    rxm = rx.mean(1, keepdims=True); rym = ry.mean(1, keepdims=True)
    rxc = rx - rxm; ryc = ry - rym
    denom = np.sqrt((rxc**2).sum(1) * (ryc**2).sum(1))
    with np.errstate(divide='ignore', invalid='ignore'):
        spearman_b = np.where(denom > 0, (rxc*ryc).sum(1) / denom, 0.0)
    return (np.percentile(pearson_b, 2.5), np.percentile(pearson_b, 97.5),
            np.percentile(spearman_b, 2.5), np.percentile(spearman_b, 97.5))

subset_names = ['Affinity Benchmark v5.5', 'SKEMPI v2.0', 'PDBbind v2020', 'SAbDab', 'ATLAS']
subsets = {name: [] for name in subset_names}
subsets['All combined'] = rows

for r in rows:
    for name in subset_names:
        if name in r['source']:
            subsets[name].append(r)
            break

hdr = '%-28s %5s  %-24s  %-24s  %8s  %10s' % (
    'Source', 'N', 'Pearson r', 'Spearman rho', 'sigma', 'mean(pKd)')
print('\n' + hdr)
print('-' * 108)

for name in subset_names + ['All combined']:
    data = subsets[name]
    if len(data) < 3:
        print('%-28s %5d  (too few)' % (name, len(data)))
        continue
    x = np.array([d['score'] for d in data])
    y = np.array([d['pkd'] for d in data])
    pr, _ = stats.pearsonr(x, y)
    sr, _ = stats.spearmanr(x, y)
    sigma = np.std(y, ddof=1)
    mean_pkd = np.mean(y)
    pl, pu, sl, su = bootstrap_ci(x, y)
    pearson_str = '%+.2f [%+.2f, %+.2f]' % (pr, pl, pu)
    spearman_str = '%+.2f [%+.2f, %+.2f]' % (sr, sl, su)
    print('%-28s %5d  %-24s  %-24s  %8.2f  %10.2f' % (
        name, len(data), pearson_str, spearman_str, sigma, mean_pkd))
