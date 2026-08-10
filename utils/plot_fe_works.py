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
from statistics import NormalDist
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
    pick = s2 if abs(mid - s1) > abs(mid - s2) else s1
    # Degenerate-crossing fallback (Goette & Grubmueller p. 449): neither root
    # between the means means the Gaussians are too close to intersect
    # meaningfully, and the mean of both is the better estimate. The drawn CGI
    # rule then coincides with the avg rule, which is the honest picture.
    if not (min(ap_, aq) <= pick <= max(ap_, aq)):
        return mid
    return pick


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


# Two equally wide Gaussians cross halfway between their means, i.e. sep/2 sigma
# out in either tail, so "is the crossing sampled?" is really "does a run of n
# cycles reach sep/2 sigma?". The most extreme of n standard normals sits near
# z_n = Phi^-1(1 - 1/n), so the crossing only leaves the sampled region once
# sep > 2 * z_n. That limit tightens when there are few cycles and relaxes when
# there are many, which a flat threshold cannot do -- the old fixed 2.0 flagged
# legs whose tails comfortably covered the gap. The cap stops a long run from
# licensing an arbitrarily wide gap: past 4 sigma two Gaussians share about 5%
# of their area no matter how many samples each has.
SEP_CAP = 4.0


def sep_limit(n):
    """Largest mean gap, in pooled sigma, that n cycles per direction can span."""
    if n < N_MIN_SEP:
        return float("nan")          # too few cycles to speak of a tail at all
    return min(SEP_CAP, 2.0 * NormalDist().inv_cdf(1.0 - 1.0 / n))


# A metric computed from too few cycles is not evidence either way, and a flag
# raised on one is noise dressed up as a finding. Each check therefore reports
# one of three states -- ok, flagged, or n/a -- instead of collapsing "no data"
# into "failed": a leg with a single cycle used to come back FD+VAR purely
# because 0/0 is not a number.
N_MIN_WIDTH = 3     # below this a std. dev. is not a width
N_MIN_SEP = 4       # sep_limit's extreme-value argument needs a tail to reach
CHECKS = ("FD", "VAR", "SEP")

# FD and VAR both compare a measured ratio against 1 and both allow a factor of
# TOL either way: a leg has to dissipate twice what its own widths permit, or
# run twice as wide in one direction as in the other, before the two-Gaussian
# picture is called broken. That tolerance says how much mismatch MATTERS and
# does not depend on n. What does depend on n is how much mismatch noise alone
# produces -- a std. dev. from 6 cycles is worth about +-45%, so an apparent
# factor of two there is unremarkable. Each band is therefore the wider of the
# two: the material tolerance, or a Z_NOISE-sigma sampling band. At n = 16 the
# FD sampling band comes out at 0.50-2.00, so the tolerance is what a leg of
# that length would have been judged against anyway; the widening only bites on
# short runs. It never tightens BELOW the tolerance either, so a very long run
# cannot report a 5% mismatch as a defect just because it is resolvable.
TOL = 2.0           # factor-two material tolerance, both directions
Z_NOISE = 2.0       # sampling bands quoted at two sigma


def tol_band(sigma_log):
    """Multiplicative band around 1: factor TOL, widened to Z_NOISE sigma."""
    hw = max(math.log(TOL), Z_NOISE * sigma_log)
    return math.exp(-hw), math.exp(hw)


def band(value, lo, hi, testable=True):
    """ok / flag / n/a for a metric that must sit inside [lo, hi].

    A nan bound means the band itself could not be formed, which is as much a
    reason to abstain as a nan value.
    """
    if not (testable and np.isfinite(value) and not (math.isnan(lo) or math.isnan(hi))):
        return "n/a"
    return "ok" if lo <= value <= hi else "flag"


def fd_check(sd_f, sd_r, diss, ratio, n):
    """FD state, plus the band it was judged against.

    diss_pred inherits the sampling noise of the two variances and diss that of
    the two means; for a normal sample those two are independent, so their
    relative variances add. Near equilibrium diss is not resolved from zero at
    all and the ratio is 0/0 -- a leg that does not measurably dissipate cannot
    be tested against its own widths, which is n/a rather than a failure.
    """
    nan = float("nan")
    v2 = sd_f ** 2 + sd_r ** 2
    if n < N_MIN_WIDTH or v2 <= 0:
        return "n/a", nan, nan
    sd_diss = math.sqrt(v2 / (4.0 * n))
    if abs(diss) <= Z_NOISE * sd_diss:
        return "n/a", nan, nan
    if diss < 0:
        # Resolved NEGATIVE dissipation: the two directions are not the same
        # process (or the works are mis-signed). No band can rescue that.
        return "flag", nan, nan
    rel_pred = 2.0 * (sd_f ** 4 + sd_r ** 4) / ((n - 1) * v2 ** 2)
    lo, hi = tol_band(math.sqrt(rel_pred + (sd_diss / diss) ** 2))
    return band(ratio, lo, hi), lo, hi


def var_check(sd_f, sd_r, widths, n):
    """VAR state, plus the band it was judged against.

    Var(ln s) = 1/(2(n-1)) for a normal sample, and the two directions are
    independent, so ln(sf/sr) carries a sampling sigma of 1/sqrt(n-1).
    """
    nan = float("nan")
    if n < N_MIN_WIDTH or sd_f <= 0 or sd_r <= 0:
        return "n/a", nan, nan
    lo, hi = tol_band(1.0 / math.sqrt(n - 1))
    return band(widths, lo, hi), lo, hi


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
      SEP  the histograms sit further apart than the cycles can span -- more
           than 2 * z_n pooled sigma, z_n = Phi^-1(1 - 1/n) -- so the CGI
           crossing is extrapolated into a gap where neither has samples

    Every threshold is referred to the sampling noise of n cycles, so none of
    the three can fire on a mismatch that n cycles would produce by chance. A
    check whose input is undefined at that n reports n/a and takes part in no
    verdict.
    """
    sd_f, sd_r, diss, n = st["sd_f"], st["sd_r"], st["diss"], st["n"]
    pred = (sd_f ** 2 + sd_r ** 2) / (4.0 * rt)
    pooled = math.sqrt((sd_f ** 2 + sd_r ** 2) / 2.0)
    ratio = diss / pred if pred > 0 else float("nan")
    widths = sd_f / sd_r if sd_r > 0 else float("nan")
    sep = 2.0 * diss / pooled if pooled > 0 else float("nan")   # gap = 2*diss
    lim = sep_limit(n)

    fd_state, fd_lo, fd_hi = fd_check(sd_f, sd_r, diss, ratio, n)
    var_state, var_lo, var_hi = var_check(sd_f, sd_r, widths, n)
    state = {"FD": fd_state, "VAR": var_state,
             # One-sided: a negative sep means the two means have crossed over,
             # which puts the crossing between them -- sampled, not extrapolated.
             "SEP": band(sep, -math.inf, lim)}
    flags = [c for c in CHECKS if state[c] == "flag"]
    skipped = [c for c in CHECKS if state[c] == "n/a"]

    if len(skipped) == len(CHECKS):
        verdict = "n/a (n=%d)" % n
    else:
        verdict = ("+".join(flags) if flags else "OK")
        if skipped:
            verdict += " (%s n/a)" % ",".join(skipped)
    st.update(pred=pred, ratio=ratio, widths=widths, sep=sep, sep_lim=lim,
              fd_band=(fd_lo, fd_hi), var_band=(var_lo, var_hi),
              state=state, flags=flags, skipped=skipped, verdict=verdict)
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
    """Format a metric, or right-align 'n/a' in the same width if it has none."""
    if np.isfinite(x):
        return fmt % x
    return "n/a".rjust(int(re.match(r"%(\d+)", fmt).group(1)))


def why_na(check, st):
    """Why a check abstained on this leg, in a few words."""
    n = st["n"]
    if check == "SEP":
        return "n < %d" % N_MIN_SEP
    if n < N_MIN_WIDTH:
        return "n < %d" % N_MIN_WIDTH
    if st["sd_f"] <= 0 or st["sd_r"] <= 0:
        return "a width is zero"
    return "diss %.1f not resolved from zero (+-%.1f)" % (
        st["diss"], Z_NOISE * math.sqrt((st["sd_f"] ** 2 + st["sd_r"] ** 2) / (4.0 * n)))

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
 "  'Near' means a factor of two, or the 2-sigma sampling band of n cycles where",
 "  that is wider. Both ends of the ratio carry noise -- diss_pred from the two",
 "  sample variances, diss from the two sample means -- and at n = 16 they combine",
 "  to a band of 0.50-2.00, so a factor of two is simply what a run of that length",
 "  can resolve. At n = 6 the same noise spans roughly 0.4-2.4 and the check backs",
 "  off accordingly, rather than reporting the shortness of the run as a defect.",
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
 "  'Strongly' is again a factor of two, or the 2-sigma sampling band where that is",
 "  wider. A sample std. dev. carries Var(ln s) = 1/(2(n-1)), so ln(sf/sr) has a",
 "  sampling sigma of 1/sqrt(n-1) -- 0.26 at n = 16, a 2-sigma band of 0.60-1.68",
 "  that sits comfortably inside the factor of two. Only below about n = 10 does",
 "  the noise band overtake the tolerance and become what the check enforces.",
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
 "  the hysteresis in units of the distributions' own spread. Two equally wide",
 "  Gaussians cross halfway between their means, so the crossing lies sep/2 sigma",
 "  into either tail -- measured only if the cycles actually reach that far out.",
 "  The most extreme of n samples sits near",
 "",
 "        z_n  =  Phi^-1( 1 - 1/n )        1.53 at n = 16, 1.64 at n = 20",
 "",
 "  so the leg is flagged once sep exceeds 2 * z_n (capped at 4.0). Past that the",
 "  reported crossing is produced by extrapolating the Gaussian fit into empty",
 "  space, and it moves as soon as one more cycle lands anywhere near it. The",
 "  limit tightens with few cycles and relaxes with many, because what decides",
 "  the question is not the size of the gap but whether the samples span it.",
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
print("wide, and overlapping. The three flags test exactly those preconditions, and")
print("report n/a for whichever of them the cycle count cannot answer.")
print("")
print("  W_f, W_r    forward / reverse work of the leg, one value per cycle")
print("  diss        ( <W_f> + <W_r> ) / 2        dissipated work; dG cancels from the sum")
print("  dG_avg      ( <W_f> - <W_r> ) / 2        the antisymmetric partner of diss")
print("  sf, sr      std. dev. of the forward / sign-aligned reverse works")
print("  diss_pred   ( sf^2 + sr^2 ) / 4RT        linear-response prediction for diss")
print("  ratio       diss / diss_pred             1.0 if the works are Gaussian")
print("  sep         2 * diss / sqrt( (sf^2 + sr^2) / 2 )   mean gap, in pooled sigma")
print("  sep_max     2 * Phi^-1( 1 - 1/n ), capped at 4.0   how far n cycles reach")
print("")
print("ratio and sf/sr are flagged outside a factor of %.1f, widened to a %.0f-sigma"
      % (TOL, Z_NOISE))
print("sampling band wherever n is small enough for noise alone to reach that far.")
print("")
print("  %-10s %-13s %4s %9s %9s %7s %7s %6s %8s %8s   %s"
      % ("structure", "leg", "n", "diss", "diss_pred", "ratio", "sf/sr", "sep",
         "sep_max", "diss/RT", "verdict"))
for sid, leg, st in stats:
    print("  %-10s %-13s %4d %9s %9s %7s %7s %6s %8s %8s   %s"
          % (sid, leg, st["n"], cell(st["diss"]), cell(st["pred"]),
             cell(st["ratio"], "%7.2f"), cell(st["widths"], "%7.2f"),
             cell(st["sep"], "%6.1f"), cell(st["sep_lim"], "%8.1f"),
             cell(st["diss"] / RT, "%8.1f"), st["verdict"]))

bad = [(sid, leg, st) for sid, leg, st in stats if st["flags"]]
thin = [(sid, leg, st) for sid, leg, st in stats if st["skipped"]]
print("")
if not bad:
    print("No leg failed a check it had the cycles to answer: where testable, the")
    print("dissipation matches the width of the distributions, the two directions are")
    print("comparably wide, and the histograms overlap. Those CGI crossings are")
    print("measured, not extrapolated, and can be read at face value.")
else:
    print("%d of %d legs failed at least one check." % (len(bad), len(stats)))

    # One block per flag that fired anywhere, each listing the legs it applies to
    # with the number that triggered it.
    for flag in CHECKS:
        hits = [(sid, leg, st) for sid, leg, st in bad if flag in st["flags"]]
        if not hits:
            continue
        print("")
        print("-" * 78)
        for line in FLAG_HELP[flag]:
            print(line)
        print("")
        print("  affected legs:")
        for sid, leg, st in hits:
            if flag == "FD" and not np.isfinite(st["fd_band"][1]):
                detail = "dissipation %.1f kJ/mol (%.0f RT) is resolved and NEGATIVE" % (
                    st["diss"], st["diss"] / RT)
            elif flag == "FD":
                detail = "ratio %.2f, outside %.2f-%.2f (diss %.1f vs %.1f predicted)" % (
                    st["ratio"], st["fd_band"][0], st["fd_band"][1],
                    st["diss"], st["pred"])
            elif flag == "VAR":
                detail = "sf/sr %.2f, outside %.2f-%.2f (sf %.1f, sr %.1f)" % (
                    st["widths"], st["var_band"][0], st["var_band"][1],
                    st["sd_f"], st["sd_r"])
            else:
                detail = "sep %.1f sigma vs %.1f reachable at n=%d (diss %.1f = %.0f RT)" % (
                    st["sep"], st["sep_lim"], st["n"], st["diss"], st["diss"] / RT)
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

# Skipped checks are reported separately from failed ones: a leg that has not
# run enough cycles has not been judged, and saying so is the only honest
# summary. This block also fires when nothing was flagged.
if thin:
    print("")
    print("-" * 78)
    print("n/a -- the check has no answer on this leg")
    print("")
    print("  FD and VAR are read off the widths of the two work distributions, and SEP")
    print("  asks how far those widths let the sampled tails reach. A std. dev. from")
    print("  fewer than %d cycles is not a width, so FD and VAR abstain below that; SEP" % N_MIN_WIDTH)
    print("  needs %d, because 2 * Phi^-1(1 - 1/n) is not a tail position until there is" % N_MIN_SEP)
    print("  a tail. FD abstains for one further reason: a leg whose dissipation is not")
    print("  resolved from zero has no ratio to test, diss / diss_pred being 0/0 there.")
    print("")
    print("  An abstaining check is neither a pass nor a failure -- the leg is simply")
    print("  untested on that point, and the CGI crossing carries no evidence for or")
    print("  against it. More cycles turn these into real verdicts; a leg that abstains")
    print("  only because it barely dissipates is in no trouble to begin with.")
    print("")
    print("  untested legs:")
    for sid, leg, st in thin:
        grouped = {}
        for check in st["skipped"]:
            grouped.setdefault(why_na(check, st), []).append(check)
        for reason, checks in grouped.items():
            print("    %-10s %-13s  %-11s  %s" % (sid, leg, ",".join(checks), reason))
