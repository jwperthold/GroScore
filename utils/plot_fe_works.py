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
ap.add_argument("--temp", type=float, default=310.0,
                help="temperature in K for the sigma^2/2RT check (default: 310, "
                     "must match groscore_fe.py --temp)")
args = ap.parse_args()

RT = 0.00831446261815324 * args.temp  # kJ/mol, same constant as groscore_fe.py


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
    # Discriminant of Goette & Grubmueller eq. (12). Their log is of the SIGMA
    # ratio with a factor 2; ln of the VARIANCE ratio already absorbs it, so no
    # extra 2 here. See groscore.py calculate_scores and tests/test_cgi.py.
    inner = (ap_ - aq) ** 2 / (vp * vq) + dinv * math.log(vq / vp)
    if inner < 0:
        return float("nan")
    t2 = math.sqrt(inner)
    s1, s2 = (t1 + t2) / dinv, (t1 - t2) / dinv
    mid = (ap_ + aq) / 2.0
    return s2 if abs(mid - s1) > abs(mid - s2) else s1


def panel(ax, fwd, rev, title, nbins):
    """Forward vs sign-aligned reverse histogram with estimate rules.

    Returns the leg statistics that gaussian_check() turns into a verdict.
    """
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

    return {"n": len(fwd), "avg": avg, "cgi": cgi, "diss": diss,
            "sd_f": float(np.std(fwd)), "sd_r": float(np.std(rev_al))}


def gaussian_check(st, rt):
    """Near-equilibrium consistency of one leg.

    Linear response (Gaussian work distributions) gives W_diss = sigma^2 / 2RT
    per direction. The plotted dissipation averages the two directions, so the
    prediction it should be compared against is (sf^2 + sr^2) / 4RT. Three ways
    the leg can fail, all of which undermine the CGI estimate:

      FD   measured dissipation does not match the width of the distributions,
           so the works are not Gaussian and CGI's two-Gaussian model is wrong
      VAR  the two directions have very different widths; the equal split behind
           the reported dissipation is then unjustified, and CGI and the average
           estimator will disagree
      SEP  the histograms sit more than ~2 pooled sigma apart, i.e. barely
           overlap, so the CGI crossing is extrapolated into a gap where neither
           distribution has samples
    """
    sd_f, sd_r, diss = st["sd_f"], st["sd_r"], st["diss"]
    pred = (sd_f ** 2 + sd_r ** 2) / (4.0 * rt)
    pooled = math.sqrt((sd_f ** 2 + sd_r ** 2) / 2.0)
    ratio = diss / pred if pred > 0 else float("nan")
    widths = sd_f / sd_r if sd_r > 0 else float("nan")
    sep = 2.0 * diss / pooled if pooled > 0 else float("nan")   # gap = 2*diss

    flags = []
    if not (np.isfinite(ratio) and 0.5 <= ratio <= 2.0):
        flags.append("FD")
    if not (np.isfinite(widths) and 0.5 <= widths <= 2.0):
        flags.append("VAR")
    if np.isfinite(sep) and sep > 2.0:
        flags.append("SEP")
    st.update(pred=pred, ratio=ratio, widths=widths, sep=sep,
              verdict="+".join(flags) if flags else "OK")
    return st


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
stats = []          # (struct_id, leg, stats dict) in plot order
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
    stats.append((sid, "restraints",
                  gaussian_check(panel(axes[row][0], W_intro, W_remove,
                                       "%s — bound-state restraints (dhdl)" % sid, nb), RT)))
    stats.append((sid, "unbind/rebind",
                  gaussian_check(panel(axes[row][1], Wtot_f, Wtot_r,
                                       "%s — unbinding / rebinding (pull + dhdl)" % sid, nb), RT)))

fig.suptitle("GroScore-FE leg work distributions — forward vs sign-aligned reverse",
             fontsize=13, fontweight="bold", color=INK, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(args.out, dpi=180, facecolor=SURFACE)
print("Wrote %s  (%d structure%s: %s)"
      % (args.out, n, "" if n == 1 else "s", ", ".join(order)))

# ---- Gaussian / near-equilibrium consistency --------------------------------
#
# CGI models both work distributions as Gaussians and reports where they cross.
# That is only meaningful if the works really are Gaussian, if the two directions
# have comparable widths, and if the crossing lies inside the sampled region.
# This table tests all three; it is diagnostic only and changes no result.

def cell(x, fmt="%8.2f"):
    return "     nan" if not np.isfinite(x) else fmt % x

# Long-form explanation of each flag, printed only for the flags that actually
# fired. The maths is short enough to state in full, and the failure modes are
# distinct enough that "which one fired" changes what you should do next.
FLAG_HELP = {
"FD": [
 "FD -- the work distributions are not Gaussian",
 "",
 "  Crooks' fluctuation theorem relates the two directions of the same switching",
 "  process,",
 "",
 "        P_f(W) / P_r(-W)  =  exp( (W - dG) / RT )",
 "",
 "  Impose that on two Gaussians and the model becomes very rigid: the exponent of",
 "  a Gaussian is quadratic in W, so matching both sides term by term forces the",
 "  two distributions to share one width, and ties the dissipation to that width",
 "  alone,",
 "",
 "        W_diss  =  <W> -/+ dG  =  sigma^2 / (2 RT)      in each direction",
 "",
 "  That is the fluctuation-dissipation relation: once the works are Gaussian, the",
 "  spread of a leg fully determines how much it dissipates. Averaging over the two",
 "  directions gives diss_pred = (sf^2 + sr^2) / 4RT, so ratio = diss / diss_pred",
 "  must sit near 1 wherever the Gaussian picture holds.",
 "",
 "  ratio >> 1  the leg dissipates far more than its own spread permits. The real",
 "              distribution is skewed, with a long low-work tail that N cycles",
 "              never reach; the fitted Gaussian is too narrow and sits too far",
 "              out. This is the signature of switching too fast, and it biases",
 "              the CGI crossing away from dG.",
 "  ratio << 1  the spread is too large for the observed dissipation. That usually",
 "              means outlier cycles or cycles that are not independent, i.e. a",
 "              sampling problem rather than a protocol problem.",
],
"VAR": [
 "VAR -- the two directions have very different widths",
 "",
 "  The same Crooks-plus-Gaussian argument demands sf = sr exactly. Strongly unequal",
 "  widths mean the forward and reverse legs are not exploring the same process, so",
 "  splitting the hysteresis evenly between them -- which is precisely what the",
 "  single 'diss' number does -- has no justification, and CGI and the average",
 "  estimator will not agree.",
 "",
 "  The opposite limit deserves a warning of its own. CGI locates the crossing via",
 "",
 "        dG  =  [ <W_f>/sf^2 - <-W_r>/sr^2  -/+ sqrt(...) ] / ( 1/sf^2 - 1/sr^2 )",
 "",
 "  whose denominator vanishes as sf -> sr. In that limit two equally wide Gaussians",
 "  cross exactly at their midpoint, which is dG_avg -- so CGI carries no extra",
 "  information there, and computing it as a ratio of two vanishing numbers only",
 "  amplifies noise. When sf/sr is close to 1, prefer dG_avg and read CGI as",
 "  confirmation, not as an independent estimate.",
],
"SEP": [
 "SEP -- the histograms barely overlap",
 "",
 "  The distance between the plotted means is",
 "",
 "        <W_f> - <-W_r>  =  <W_f> + <W_r>  =  2 * diss",
 "",
 "  and sep divides that gap by the pooled width sqrt((sf^2 + sr^2)/2), expressing",
 "  the hysteresis in units of the distributions' own spread. Above about 2 the two",
 "  curves meet only in their tails, where neither has samples: the reported crossing",
 "  is then produced by extrapolating the Gaussian fit into empty space, and it moves",
 "  as soon as one more cycle lands anywhere near it.",
 "",
 "  This is not a defect of CGI in particular. Every bidirectional estimator (CGI,",
 "  BAR, Crooks) needs the forward and reverse work ensembles to overlap, because dG",
 "  is extracted from the region where both are populated. The natural scale is RT:",
 "  a leg dissipating a few RT converges comfortably, one dissipating tens of RT",
 "  needs exponentially many cycles to sample the tail that carries the answer.",
],
}

print("")
print("Gaussian / near-equilibrium consistency  (RT = %.3f kJ/mol at %.1f K)"
      % (RT, args.temp))
print("-" * 78)
print("CGI fits a Gaussian to each work distribution and reports where the two curves")
print("cross. That crossing is dG only if the works really are Gaussian, comparably")
print("wide, and overlapping. The three flags test exactly those preconditions.")
print("")
print("  W_f, W_r    forward / reverse work of the leg, one value per cycle")
print("  diss        ( <W_f> + <W_r> ) / 2        dissipated work; dG cancels from the sum")
print("  dG_avg      ( <W_f> - <W_r> ) / 2        the antisymmetric partner of diss")
print("  sf, sr      std. dev. of the forward / sign-aligned reverse works")
print("  diss_pred   ( sf^2 + sr^2 ) / 4RT        linear-response prediction for diss")
print("  ratio       diss / diss_pred             1.0 if the works are Gaussian")
print("  sep         2 * diss / sqrt( (sf^2 + sr^2) / 2 )   mean gap, in pooled sigma")
print("")
print("  %-10s %-13s %4s %9s %9s %7s %7s %6s %8s   %s"
      % ("structure", "leg", "n", "diss", "diss_pred", "ratio", "sf/sr", "sep",
         "diss/RT", "verdict"))
for sid, leg, st in stats:
    print("  %-10s %-13s %4d %9s %9s %7s %7s %6s %8s   %s"
          % (sid, leg, st["n"], cell(st["diss"]), cell(st["pred"]),
             cell(st["ratio"], "%7.2f"), cell(st["widths"], "%7.2f"),
             cell(st["sep"], "%6.1f"), cell(st["diss"] / RT, "%8.1f"), st["verdict"]))

bad = [(sid, leg, st) for sid, leg, st in stats if st["verdict"] != "OK"]
print("")
if not bad:
    print("All %d legs are consistent with the Gaussian assumption: the dissipation" % len(stats))
    print("matches the width of the distributions, the two directions are comparably")
    print("wide, and the histograms overlap. The CGI crossings are measured, not")
    print("extrapolated, and can be read at face value.")
else:
    print("%d of %d legs failed at least one check." % (len(bad), len(stats)))

    # One block per flag that fired anywhere, each listing the legs it applies to
    # with the number that triggered it.
    for flag in ("FD", "VAR", "SEP"):
        hits = [(sid, leg, st) for sid, leg, st in bad if flag in st["verdict"].split("+")]
        if not hits:
            continue
        print("")
        print("-" * 78)
        for line in FLAG_HELP[flag]:
            print(line)
        print("")
        print("  affected legs:")
        for sid, leg, st in hits:
            if flag == "FD":
                detail = "ratio %.2f (diss %.1f vs %.1f predicted)" % (
                    st["ratio"], st["diss"], st["pred"])
            elif flag == "VAR":
                detail = "sf/sr %.2f (sf %.1f, sr %.1f)" % (st["widths"], st["sd_f"], st["sd_r"])
            else:
                detail = "sep %.1f sigma (diss %.1f = %.0f RT)" % (
                    st["sep"], st["diss"], st["diss"] / RT)
            print("    %-10s %-13s  %s" % (sid, leg, detail))

    print("")
    print("-" * 78)
    print("What to do about it")
    print("")
    print("  Prefer dG_avg over CGI on the flagged legs: the average estimator makes no")
    print("  Gaussian assumption and degrades gracefully, whereas a CGI crossing drawn")
    print("  from unsampled tails can land anywhere.")
    print("")
    print("  To fix the physics rather than the readout, dissipate less by switching")
    print("  more SLOWLY -- longer legs at a proportionally lower pull rate. Dissipated")
    print("  work falls roughly linearly with the switching time in the near-equilibrium")
    print("  regime, and narrower, less separated distributions follow. Leg length and")
    print("  rate are coupled by rate x time = 1.0 nm, so nsteps in the leg mdp and")
    print("  --pull-rate in make_boresch.py must always be changed together.")
    print("")
    print("  Running more cycles will NOT clear these flags. More cycles shrink the")
    print("  confidence interval around whatever the estimator converges to; they do")
    print("  not reduce the hysteresis that separates the two distributions, which is")
    print("  set by the switching rate alone.")
