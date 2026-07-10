#!/usr/bin/env python3
"""GroScore vs I-RMSD density scatter for the CAPRI benchmark (thesis Figure 3-3).

Per target: gray scatter of GroScore [kJ/mol] against interface-RMSD [nm], with
2-D kernel-density contours (green-blue) and 1-D marginal densities (blue), all
computed by KDE — matching the thesis / paper figures. Near-native (favourable,
strongly negative GroScore, low I-RMSD) poses cluster at the bottom-left.

Failed / un-scored poses are drawn at GroScore = 0 (paper convention).

Usage:
  python3 plot_capri_scatter.py                 # grid of all scored targets
  python3 plot_capri_scatter.py --target T47    # single-target figure
  python3 plot_capri_scatter.py --metric lrms   # use L-RMSD instead of I-RMSD
  python3 plot_capri_scatter.py -o ../results/capri_scatter.png
"""
import os
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import capri_common as cc

GRAY = "0.55"
BLUE = "#4C72B0"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--target", default=None, help="plot a single target (e.g. T47)")
ap.add_argument("-s", "--score-file", default="scores_avg.gs")
ap.add_argument("--metric", choices=["irms", "lrms"], default="irms",
                help="RMSD on the x-axis (default: irms)")
ap.add_argument("--targets-root", default=cc.REPO_ROOT)
ap.add_argument("--db", default=cc.DB_DIR)
ap.add_argument("--ncols", type=int, default=4, help="columns in the grid (default: 4)")
ap.add_argument("-o", "--out", default=None)
args = ap.parse_args()

XKEY = "irms_nm" if args.metric == "irms" else "lrms_nm"
XLABEL = "I-RMSD [nm]" if args.metric == "irms" else "L-RMSD [nm]"


def target_xy(t):
    """Return (x_rmsd_nm, y_groscore) arrays for a scored target, else None."""
    rows, n_numeric, scored = cc.join_target(t, args.targets_root, args.db, args.score_file)
    if not scored or n_numeric == 0:
        return None
    x = np.array([r[XKEY] for r in rows])
    y = np.array([r["score"] for r in rows])
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _marginal(ax, data, orient):
    data = data[np.isfinite(data)]
    if len(data) < 3 or np.std(data) == 0:
        return
    try:
        kde = gaussian_kde(data)
    except Exception:
        return
    grid = np.linspace(data.min(), data.max(), 200)
    d = kde(grid)
    if orient == "x":
        ax.fill_between(grid, d, color=BLUE, alpha=0.4, linewidth=0)
        ax.plot(grid, d, color=BLUE, lw=1)
    else:
        ax.fill_betweenx(grid, d, color=BLUE, alpha=0.4, linewidth=0)
        ax.plot(d, grid, color=BLUE, lw=1)


def draw_joint(fig, cell, x, y, title):
    inner = cell.subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                             wspace=0.04, hspace=0.04)
    ax_main = fig.add_subplot(inner[1, 0])
    ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)

    ax_main.scatter(x, y, s=6, color=GRAY, alpha=0.35, linewidths=0)

    # 2-D KDE contours (green-blue)
    if len(x) > 5 and np.std(x) > 0 and np.std(y) > 0:
        try:
            kde = gaussian_kde(np.vstack([x, y]))
            xg = np.linspace(x.min(), x.max(), 100)
            yg = np.linspace(y.min(), y.max(), 100)
            XX, YY = np.meshgrid(xg, yg)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax_main.contour(XX, YY, ZZ, levels=7, cmap="GnBu", linewidths=1.0, alpha=0.9)
        except Exception:
            pass

    _marginal(ax_top, x, "x")
    _marginal(ax_right, y, "y")

    ax_top.axis("off")
    ax_right.axis("off")
    ax_top.set_title("%s  (n=%d)" % (title, len(x)), fontsize=11, fontweight="bold")
    ax_main.set_xlabel(XLABEL, fontsize=9)
    ax_main.set_ylabel("GroScore [kJ/mol]", fontsize=9)
    ax_main.tick_params(labelsize=8)
    ax_main.margins(x=0.02)
    return ax_main


# ── select targets ────────────────────────────────────────────────────────────
if args.target:
    targets = [args.target]
else:
    targets = [t for t in cc.TARGETS if target_xy(t) is not None]

if not targets:
    raise SystemExit("No scored targets found (looked for %s in each target dir)." % args.score_file)

data = {t: target_xy(t) for t in targets}
missing = [t for t in targets if data[t] is None]
if missing:
    raise SystemExit("Target(s) not scored: %s" % ", ".join(missing))

# ── figure ────────────────────────────────────────────────────────────────────
n = len(targets)
ncols = 1 if n == 1 else min(args.ncols, n)
nrows = math.ceil(n / ncols)
fig = plt.figure(figsize=(4.6 * ncols, 4.2 * nrows))
outer = fig.add_gridspec(nrows, ncols, wspace=0.28, hspace=0.28)

for i, t in enumerate(targets):
    x, y = data[t]
    draw_joint(fig, outer[i // ncols, i % ncols], x, y, t)

fig.suptitle("GroScore vs %s — CAPRI Score_set" % XLABEL,
             fontsize=14, fontweight="bold", y=1.0)

out = args.out or os.path.join(
    cc.REPO_ROOT, "benchmark", "results",
    "capri_scatter_%s.png" % (args.target if args.target else "all"))
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print("Saved %s  (%d target%s)" % (out, n, "" if n == 1 else "s"))
