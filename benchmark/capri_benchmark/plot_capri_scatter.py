#!/usr/bin/env python3
"""GroScore vs I-RMSD density scatter for the CAPRI benchmark (thesis Figure 3-3).

Per target: gray scatter of GroScore [kJ/mol] against interface-RMSD [nm], with
2-D kernel-density contours (green-blue) and 1-D marginal densities (blue), all
computed by KDE — matching the thesis / paper figures. Near-native (favourable,
strongly negative GroScore, low I-RMSD) poses cluster at the bottom-left.

Failed / un-scored poses carry no score and are excluded from the scatter as well as
from the ROC-AUC and the plotted native percentile (both computed over the scored
poses only). The console native-rank report below still lists all poses.

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
ap.add_argument("--positive", choices=["acceptable", "medium", "high"], default="acceptable",
                help="quality threshold counted as near-native for the ROC-AUC (default: acceptable)")
ap.add_argument("--with-native", action=argparse.BooleanOptionalAction, default=True,
                help="include the native/experimental structure as a positive in the ROC-AUC (default: yes)")
ap.add_argument("-o", "--out", default=None)
args = ap.parse_args()

XKEY = "irms_nm" if args.metric == "irms" else "lrms_nm"
XLABEL = "I-RMSD [nm]" if args.metric == "irms" else "L-RMSD [nm]"
PMIN = {"acceptable": 1, "medium": 2, "high": 3}[args.positive]

# native / experimental reference scores (natives/<target>/ scored -> natives/scores_*.gs),
# keyed by target name; plotted as a star at I-RMSD = 0.
native_scores, native_numeric = cc.load_scores(
    os.path.join(cc.REPO_ROOT, "natives", args.score_file))


def target_data(t):
    """Return (x_rmsd_nm, y_groscore, roc_auc, n_total) for a scored target, else None.

    Only actually-scored poses are plotted; failed / un-scored poses are excluded from
    the scatter and from the ROC-AUC / plotted native percentile (all over the scored
    poses only, matching analyze_capri.py). n_total (all poses, failed included) is
    returned only for the console native-rank report.
    """
    rows, n_numeric, scored = cc.join_target(t, args.targets_root, args.db, args.score_file)
    if not scored or n_numeric == 0:
        return None
    nat = native_scores.get(t) if (args.with_native and t in native_numeric) else None
    roc = cc.roc_auc(rows, PMIN, native_score=nat)
    scored_rows = [r for r in rows if r.get("scored")]
    x = np.array([r[XKEY] for r in scored_rows])
    y = np.array([r["score"] for r in scored_rows])
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m], roc, len(rows)


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


def draw_joint(fig, cell, x, y, title, roc, native=None):
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
    ax_top.set_title("%s  (N=%d)" % (title, len(x)), fontsize=11, fontweight="bold")
    ax_main.set_xlabel(XLABEL, fontsize=9)
    ax_main.set_ylabel("GroScore [kJ/mol]", fontsize=9)
    ax_main.tick_params(labelsize=8)
    ax_main.margins(x=0.02)
    # native / experimental reference structure: gold star at I-RMSD = 0
    if native is not None and np.isfinite(native):
        ax_main.plot(0.0, native, marker="*", markersize=8, mfc="gold",
                     mec="black", mew=0.6, linestyle="None", zorder=6,)
        x0, x1 = ax_main.get_xlim()
        ax_main.set_xlim(left=min(x0, -0.03 * (x1 - x0)))
        #ax_main.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85,
        #               handletextpad=0.2, borderpad=0.3)
    lbl = ("ROC-AUC = %.2f" % roc) if np.isfinite(roc) else "ROC-AUC = n/a"
    if native is not None and np.isfinite(native) and len(y):
        lbl += "\nNative: Top %.1f%%" % (100.0 * np.sum(np.asarray(y) < native) / len(y))
    ax_main.text(0.96, 0.96, lbl, transform=ax_main.transAxes, fontsize=9,
                 va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.7"))
    return ax_main


# ── select targets ────────────────────────────────────────────────────────────
cand = [args.target] if args.target else list(cc.TARGETS)
data = {t: target_data(t) for t in cand}
targets = [t for t in cand if data[t] is not None]

if not targets:
    if args.target:
        raise SystemExit("Target %s is not scored (looked for %s)." % (args.target, args.score_file))
    raise SystemExit("No scored targets found (looked for %s in each target dir)." % args.score_file)

# ── native-structure percentile vs the scored poses (console only) ────────────
print("\nNative structure rank vs all poses incl. failed (GroScore, more negative = better):")
_printed = False
for t in targets:
    if t not in native_numeric:
        continue
    _printed = True
    _, ys, _, n = data[t]                   # n = all poses (incl. failed) for the percentile
    nat = native_scores[t]
    n_better = int(np.sum(np.asarray(ys) < nat))   # poses more favourable than the native
    print("  %-5s native = %8.1f kJ/mol   beats %5.1f%% of %d poses  ->  top %4.1f%%"
          % (t, nat, 100.0 * (n - n_better) / n, n, 100.0 * n_better / n))
if not _printed:
    print("  (no target yet has both a scored native and scored poses)")

# ── figure ────────────────────────────────────────────────────────────────────
n = len(targets)
ncols = 1 if n == 1 else min(args.ncols, n)
nrows = math.ceil(n / ncols)
fig = plt.figure(figsize=(4.6 * ncols, 4.2 * nrows))
outer = fig.add_gridspec(nrows, ncols, wspace=0.28, hspace=0.28)

for i, t in enumerate(targets):
    x, y, roc, _ = data[t]
    native = native_scores.get(t) if t in native_numeric else None
    draw_joint(fig, outer[i // ncols, i % ncols], x, y, t, roc, native)

fig.suptitle("GroScore vs %s — CAPRI Score_set" % XLABEL,
             fontsize=14, fontweight="bold", y=1.0)

out = args.out or os.path.join(
    cc.REPO_ROOT, "benchmark", "results",
    "capri_scatter_%s.png" % (args.target if args.target else "all"))
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print("Saved %s  (%d target%s)" % (out, n, "" if n == 1 else "s"))
