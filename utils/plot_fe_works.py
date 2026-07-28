#!/usr/bin/env python3
#
# plot_fe_works.py - Work distributions for the GroScore-FE legs (Crooks diagnostic).
#
# One figure, one row per structure, two panels per row:
#
#   left  "Bound-state restraints"  forward = W_intro   (restraints switched on)
#                                   reverse = W_remove  (switched off)
#   right "Unbinding / rebinding"   forward = pull work + dhdl work (unbinding)
#                                   reverse = pull work + dhdl work (rebinding)
#
# The reverse distribution is plotted sign-aligned (-W_reverse), which is the
# standard Crooks presentation: the forward and reverse histograms should overlap
# and cross at dG. Wide separation = the leg is driven too fast (hysteresis), and
# the estimate is dominated by dissipated work rather than the free energy.
# Vertical rules mark the average and CGI estimates.
#
# Usage:
#   python3 utils/plot_fe_works.py                     # run from the project dir
#   python3 utils/plot_fe_works.py -o works.png --bins 20
#

import os, re, sys, math, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sign conventions -- must match groscore_fe.py.
SIGN_PULL_FWD = -1.0
SIGN_PULL_REV = +1.0

# Palette: categorical slots 1-2 (validated, CVD dE 24.7), plus chrome/ink tokens.
FWD, REV = "#2a78d6", "#eb6834"
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

ap = argparse.ArgumentParser(description="Plot GroScore-FE leg work distributions.")
ap.add_argument("-s", "--structparams", default="sp.gs", help="structure list (default: sp.gs)")
ap.add_argument("-r", "--results", default="results_fe.gs", help="legacy results file")
ap.add_argument("-d", "--resultsdir", default="results_fe.d", help="per-cycle results dir")
ap.add_argument("-o", "--out", default="fe_works.png", help="output image (default: fe_works.png)")
ap.add_argument("--bins", type=int, default=0, help="histogram bins (0 = automatic)")
args = ap.parse_args()


def read_works(filepath, workdir):
    """{struct_id: {cycle: (W_intro, Wu_pull, Wu_dhdl, Wr_pull, Wr_dhdl, W_remove)}}"""
    works = {}

    def take(line):
        if line.strip().startswith("#"):
            return
        t = line.split()
        if len(t) < 8:
            return
        try:
            cyc = int(t[1]); vals = [float(x) for x in t[2:8]]
        except ValueError:
            return
        if any(math.isnan(v) for v in vals):
            return
        works.setdefault(t[0], {})[cyc] = vals      # dedup by cycle, last wins

    if os.path.isfile(filepath):
        for line in open(filepath):
            take(line)
    for p in sorted(glob.glob(os.path.join(workdir, "*.gs"))):
        try:
            for line in open(p):
                take(line)
        except OSError:
            pass
    return works


def cgi_point(fwd, rev_aligned):
    """Crooks-Gaussian-Intersection of two work distributions."""
    if len(fwd) < 3 or len(rev_aligned) < 3:
        return float("nan")
    ap_, vp = np.mean(fwd), np.var(fwd)
    aq, vq = np.mean(rev_aligned), np.var(rev_aligned)
    if vp <= 0 or vq <= 0 or vp == vq:
        return float("nan")
    dinv = 1.0 / vp - 1.0 / vq
    t1 = ap_ / vp - aq / vq
    inner = (ap_ - aq) ** 2 / (vp * vq) + 2.0 * dinv * math.log(vq / vp)
    if inner < 0:
        return float("nan")
    t2 = math.sqrt(inner)
    s1, s2 = (t1 + t2) / dinv, (t1 - t2) / dinv
    mid = (ap_ + aq) / 2.0
    return s2 if abs(mid - s1) > abs(mid - s2) else s1


def panel(ax, fwd, rev, title, nbins):
    """Forward vs sign-aligned reverse histogram with estimate rules."""
    rev_al = [-w for w in rev]
    lo = min(min(fwd), min(rev_al))
    hi = max(max(fwd), max(rev_al))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    bins = np.linspace(lo - pad, hi + pad, nbins + 1)

    binw = bins[1] - bins[0]
    series = []
    for data, color, label in ((fwd, FWD, "forward"), (rev_al, REV, "reverse (−W)")):
        d = np.asarray(data, float)
        ax.hist(d, bins=bins, color=color, alpha=0.45, edgecolor=color,
                linewidth=1.5,
                label="%s  n=%d, μ=%.1f, σ=%.1f" % (label, len(d), d.mean(), d.std()))
        series.append((d, color))

    avg = (np.mean(fwd) + np.mean(rev_al)) / 2.0
    cgi = cgi_point(fwd, rev_al)
    diss = (np.mean(fwd) - np.mean(rev_al)) / 2.0     # per-direction dissipation

    ax.axvline(avg, color=INK, lw=2.0, label="avg  %.1f" % avg)
    if np.isfinite(cgi):
        ax.axvline(cgi, color=INK, lw=2.0, ls="--", label="CGI  %.1f" % cgi)

    # Gaussian fits -- these are exactly what CGI intersects, so drawing them
    # shows whether the reported crossing sits inside the sampled region or is
    # extrapolated into a gap where neither distribution has data.
    x0, x1 = ax.get_xlim()
    grid = np.linspace(x0, x1, 400)
    for d, color in series:
        sd = d.std()
        if sd <= 0 or len(d) < 2:
            continue
        pdf = np.exp(-0.5 * ((grid - d.mean()) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        ax.plot(grid, pdf * len(d) * binw, color=color, lw=2.0)
    ax.set_xlim(x0, x1)

    ax.set_title(title, fontsize=10, color=INK, fontweight="bold", loc="left")
    ax.set_xlabel("work [kJ/mol]", fontsize=9, color=SECONDARY)
    ax.set_ylabel("cycles", fontsize=9, color=SECONDARY)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.text(0.98, 0.97, "dissipation %.0f kJ/mol" % diss, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=SECONDARY)
    # Opaque-ish legend surface: the avg/CGI rules would otherwise strike through
    # the label text wherever they cross the legend box.
    leg = ax.legend(fontsize=7.5, frameon=True, labelcolor=SECONDARY, loc="upper left",
                    facecolor=SURFACE, edgecolor="none", framealpha=0.92)
    leg.set_zorder(5)


# ---- collect ----------------------------------------------------------------
works = read_works(args.results, args.resultsdir)

order = []
if os.path.isfile(args.structparams):
    for line in open(args.structparams):
        if not line.strip().startswith("#") and line.split():
            order.append(line.split()[0])
order = [s for s in order if s in works] or sorted(works)
if not order:
    sys.exit("No work data found (looked in %s and %s/)." % (args.results, args.resultsdir))

# ---- figure -----------------------------------------------------------------
n = len(order)
fig, axes = plt.subplots(n, 2, figsize=(11, 3.4 * n), squeeze=False,
                         facecolor=SURFACE)
for row, sid in enumerate(order):
    cyc = works[sid]
    keys = sorted(cyc)
    W_intro  = [cyc[c][0] for c in keys]
    Wu_pull  = [cyc[c][1] for c in keys]
    Wu_dhdl  = [cyc[c][2] for c in keys]
    Wr_pull  = [cyc[c][3] for c in keys]
    Wr_dhdl  = [cyc[c][4] for c in keys]
    W_remove = [cyc[c][5] for c in keys]

    # unbinding / rebinding totals = pull work + dhdl work
    Wtot_f = [SIGN_PULL_FWD * p + d for p, d in zip(Wu_pull, Wu_dhdl)]
    Wtot_r = [SIGN_PULL_REV * p + d for p, d in zip(Wr_pull, Wr_dhdl)]

    nb = args.bins if args.bins > 0 else max(6, min(25, len(keys) // 3 + 4))
    for ax in axes[row]:
        ax.set_facecolor(SURFACE)
    panel(axes[row][0], W_intro, W_remove,
          "%s — bound-state restraints (dhdl)" % sid, nb)
    panel(axes[row][1], Wtot_f, Wtot_r,
          "%s — unbinding / rebinding (pull + dhdl)" % sid, nb)

fig.suptitle("GroScore-FE leg work distributions — forward vs sign-aligned reverse",
             fontsize=13, fontweight="bold", color=INK, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(args.out, dpi=180, facecolor=SURFACE)
print("Wrote %s  (%d structure%s: %s)"
      % (args.out, n, "" if n == 1 else "s", ", ".join(order)))
for sid in order:
    print("  %-8s %d cycles" % (sid, len(works[sid])))
