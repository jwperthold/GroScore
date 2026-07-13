#!/usr/bin/env python3
"""
Collect rmsd_rebound.gs from all benchmark dirs and plot histograms
comparing cutout vs no-cutout for each force field.

Usage:
    python3 plot_rmsd_comparison.py [base_dir] [-o output.png]

    base_dir: parent of bm_charmm/, bm_amber/, bm_gromos/, ...
              default: /home/jwperthold/GroScore
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────
BASE = '/home/jwperthold/GroScore'
OUT  = os.path.join(os.path.dirname(__file__), 'rmsd_comparison.png')
for i, a in enumerate(sys.argv[1:], 1):
    if a == '-o' and i < len(sys.argv) - 1:
        OUT = sys.argv[i + 1]
    elif not a.startswith('-'):
        BASE = a

FORCEFIELDS = ['amber_opc', 'amber', 'charmm', 'gromos']
FF_LABEL    = {'amber_opc': 'AMBER19SB/OPC', 'charmm': 'CHARMM36m/TIP3P', 'amber': 'AMBER19SB/OPC3', 'gromos': 'GROMOS54A8/SPC'}

COLORS = {
    'cutout':   '#1565C0',   # dark blue
    'nocutout': '#B71C1C',   # dark red
}
ALPHA       = 0.55
BIN_W       = 0.5    # Å
GOOD_THR    = 5.0    # Å — threshold for "successful rebinding"

# ── helpers ───────────────────────────────────────────────────────────────────
def read_gs(path):
    """Return (struct_ids, cycles, rmsds) arrays from rmsd_rebound.gs."""
    structs, cycles, vals = [], [], []
    if not os.path.isfile(path):
        return structs, cycles, vals
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 3:
                try:
                    structs.append(p[0])
                    cycles.append(int(p[1]))
                    vals.append(float(p[2]))
                except ValueError:
                    pass
    return structs, cycles, vals


def per_struct_means(structs, vals):
    """Return per-structure mean RMSD dict."""
    d = {}
    for s, v in zip(structs, vals):
        d.setdefault(s, []).append(v)
    return {s: np.mean(vs) for s, vs in d.items()}


def stats_str(v):
    if len(v) == 0:
        return 'n=0'
    pct = 100 * np.sum(np.array(v) <= GOOD_THR) / len(v)
    return (f'n={len(v)}, med={np.median(v):.1f}, '
            f'mean={np.mean(v):.1f} Å, ≤{GOOD_THR:.0f}Å: {pct:.0f}%')


# ── load data ─────────────────────────────────────────────────────────────────
data = {}
partial = False
for ff in FORCEFIELDS:
    cut_path   = os.path.join(BASE, f'bm_{ff}',          'rmsd_rebound.gs')
    nocut_path = os.path.join(BASE, f'bm_{ff}_nocutout', 'rmsd_rebound.gs')
    s_c, cy_c, v_c   = read_gs(cut_path)
    s_n, cy_n, v_n   = read_gs(nocut_path)
    data[ff] = {
        'cutout':   {'structs': s_c, 'cycles': cy_c, 'vals': np.array(v_c, dtype=float)},
        'nocutout': {'structs': s_n, 'cycles': cy_n, 'vals': np.array(v_n, dtype=float)},
    }
    n_c = len(v_c); n_n = len(v_n)
    print(f'{ff:8s}  cutout n={n_c:3d}  no-cutout n={n_n:3d}', end='')
    if n_c == 0 or n_n == 0:
        print('  [MISSING]'); partial = True
    elif n_c < 200 or n_n < 200:
        print('  [PARTIAL — still running?]'); partial = True
    else:
        print()

if partial:
    print('  Note: some files are partial — re-run when benchmarks finish.\n')

# ── global x range ────────────────────────────────────────────────────────────
all_vals = np.concatenate([
    data[ff][key]['vals']
    for ff in FORCEFIELDS
    for key in ('cutout', 'nocutout')
    if len(data[ff][key]['vals']) > 0
])
X_MAX = np.ceil(all_vals.max() / BIN_W) * BIN_W if len(all_vals) else 15.0
BINS  = np.arange(0, X_MAX + BIN_W, BIN_W)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
title = 'Backbone RMSD: Rebinding Quality — Cutout vs No-Cutout'
if partial:
    title += '  [PARTIAL DATA]'
fig.suptitle(title, fontsize=12, y=1.01)

for ax, ff in zip(axes.flat, FORCEFIELDS):
    for key, label, color in [
        ('cutout',   'Cutout',    COLORS['cutout']),
        ('nocutout', 'No-Cutout', COLORS['nocutout']),
    ]:
        v = data[ff][key]['vals']
        if len(v) == 0:
            continue

        ax.hist(v, bins=BINS, alpha=ALPHA, color=color,
                edgecolor='white', linewidth=0.4, label=label)

    ax.set_xlim(0, X_MAX)
    ax.set_title(FF_LABEL[ff], fontsize=12, fontweight='bold', pad=6)
    ax.set_xlabel('Backbone RMSD (Å)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.tick_params(axis='y', labelsize=9)

    # Stats annotation
    lines = []
    for key, lbl in [('cutout', 'Cutout'), ('nocutout', 'No-cutout')]:
        v = data[ff][key]['vals']
        lines.append(f'{lbl}: {stats_str(v)}')
    ax.text(0.98, 0.97, '\n'.join(lines),
            transform=ax.transAxes, fontsize=7.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.88, ec='#bbb'))

    ax.legend(fontsize=9, loc='upper left',
              framealpha=0.85, edgecolor='#bbb')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'\nSaved → {OUT}')
