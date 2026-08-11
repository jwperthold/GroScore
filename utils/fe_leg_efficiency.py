#!/usr/bin/env python3
#
# fe_leg_efficiency.py - where a GroScore-FE leg's hysteresis comes from, and
# what lengthening the leg vs running more cycles would actually buy.
#
# The Crooks table printed by groscore_fe.py says WHETHER a leg has converged.
# This says WHY it has not, and which lever to pull. It reads only files a
# finished cycle already leaves behind and runs no simulation.
#
# The unbinding leg drives ONE protocol parameter s = tau/t: the pull reference
# moves 0 -> L while lambda ramps 0 -> 1, locked together by rate * t = L. Its
# conjugate force is therefore
#
#     X(s) = L * F_pull(s) + dH/dlambda(s)          [kJ/mol]
#
# and the switching work is W = int_0^1 X ds. Everything here is derived from
# the cumulative W(s) of each cycle, rebuilt from the small per-cycle reductions
# job_fe.run already writes -- NOT from the pullf .xvg files, which carry one
# column per perturbed restraint and run to tens of MB each:
#
#     <leg>_<n>_DG.dat           cumulative pull work vs time (integrate.py)
#     <leg>_<n>_dGdt.dat         summed pull force vs time
#     <leg>_<n>_dhdl_Wdhdl.dat   cumulative dH/dlambda work vs lambda
#
# The reconstruction is checked against results_fe.d on every run, so a silent
# convention drift in integrate.py shows up immediately rather than quietly
# rescaling every number below.
#
# Usage:
#   python3 ../utils/fe_leg_efficiency.py -s 2KTF          # from the project dir
#   python3 ../utils/fe_leg_efficiency.py -s 2KTF --leg bound
#

import argparse, glob, math, os, re, sys
import numpy as np
from statistics import NormalDist

TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz"))
NORM = NormalDist()

# Sign conventions -- must match groscore_fe.py.
SIGN_PULL_FWD = -1.0
SIGN_PULL_REV = +1.0

# Which mdp/prefix pair each leg is made of: (forward tag, reverse tag, mdp).
LEGS = {
    "unbind": ("bindfwd", "bindrev", "bind_fe.mdp"),
    "bound": ("boundfwd", "boundrev", "boundfwd.mdp"),
}

ap = argparse.ArgumentParser(
    description="Diagnose a GroScore-FE leg and cost out longer legs vs more cycles.")
ap.add_argument("-s", "--struct", required=True, help="Structure ID, e.g. 2KTF.")
ap.add_argument("-d", "--dir", default=None,
                help="Structure directory (default: the structure ID).")
ap.add_argument("-r", "--resultsdir", default="results_fe.d",
                help="Per-cycle results dir (default: results_fe.d).")
ap.add_argument("--leg", default="unbind", choices=sorted(LEGS),
                help="Which leg pair to analyse (default: unbind).")
ap.add_argument("--temp", type=float, default=310.0,
                help="Temperature in K (default: 310, must match the run).")
ap.add_argument("--equil", type=float, default=1.1,
                help="Per-cycle equilibration outside the switching legs, ns "
                     "(default: 1.1 = nvt_1-5 + npt_fe).")
ap.add_argument("--max-friction-cycles", type=int, default=12, metavar="N",
                help="Cycles used for the friction estimate (default: 12); it is "
                     "an ensemble average and converges quickly.")
ap.add_argument("--windows", type=int, default=20, metavar="N",
                help="Windows along the ramp for the friction estimate (default: 20).")
args = ap.parse_args()

RT = 0.00831446261815324 * args.temp
SDIR = args.dir or args.struct
FWD_TAG, REV_TAG, LEG_MDP = LEGS[args.leg]


# ---- small readers -----------------------------------------------------------

def mdp_time_ns(path):
    """Simulated time of an mdp, in ns, or nan."""
    try:
        text = open(path).read()
    except OSError:
        return float("nan")
    def get(key):
        m = re.search(r"^\s*%s\s*=\s*([0-9.eE+-]+)" % key, text, re.M)
        return float(m.group(1)) if m else float("nan")
    return get("nsteps") * get("dt") / 1000.0


def boresch_value(key):
    p = os.path.join(SDIR, "boresch_analytical.gs")
    try:
        for line in open(p):
            f = line.split()
            if len(f) >= 2 and f[0] == key:
                return float(f[1])
    except (OSError, ValueError):
        pass
    return float("nan")


def two_cols(path):
    a = np.loadtxt(path)
    return a[:, 0], a[:, 1]


def read_works():
    """{cycle: [W_intro, Wu_pull, Wu_dhdl, Wr_pull, Wr_dhdl, W_remove]}, NaN rows
    dropped exactly as groscore_fe.read_works drops them."""
    out = {}
    pat = os.path.join(args.resultsdir, "%s_c*.gs" % args.struct)
    for path in sorted(glob.glob(pat)) + ["results_fe.gs"]:
        if not os.path.isfile(path):
            continue
        for line in open(path):
            f = line.split()
            if len(f) < 8 or f[0] != args.struct:
                continue
            try:
                vals = [float(x) for x in f[2:8]]
            except ValueError:
                continue
            if not any(math.isnan(v) for v in vals):
                out[int(f[1])] = vals
    return out


def leg_pair(works):
    """(forward, sign-aligned reverse) works of the selected leg, per cycle."""
    ks = sorted(works)
    if args.leg == "bound":
        f = np.array([works[c][0] for c in ks])
        r_raw = np.array([works[c][5] for c in ks])
    else:
        f = np.array([SIGN_PULL_FWD * works[c][1] + works[c][2] for c in ks])
        r_raw = np.array([SIGN_PULL_REV * works[c][3] + works[c][4] for c in ks])
    return ks, f, r_raw, -r_raw


def cycles_with_traces(tag):
    out = []
    for p in glob.glob(os.path.join(SDIR, "%s_*_dGdt.dat" % tag)):
        c = int(re.search(r"_(\d+)_dGdt", p).group(1))
        if os.path.isfile(os.path.join(SDIR, "%s_%d_dhdl_Wdhdl.dat" % (tag, c))):
            out.append(c)
    return sorted(out)


def cumulative_work(tag, cyc, npts=2001):
    """Cumulative total work of one leg run on a common s grid in [0, 1].

    integrate.py prints -DG*rate as the pull work and groscore_fe.py then applies
    SIGN_PULL_*, so the same two operations are applied here; the dhdl channel is
    already cumulative in lambda."""
    t, dg = two_cols(os.path.join(SDIR, "%s_%d_DG.dat" % (tag, cyc)))
    lam, wd = two_cols(os.path.join(SDIR, "%s_%d_dhdl_Wdhdl.dat" % (tag, cyc)))
    sign = SIGN_PULL_FWD if tag == FWD_TAG else SIGN_PULL_REV
    w_pull = sign * (-dg)
    s_pull = (t - t[0]) / (t[-1] - t[0])
    s_dhdl = np.abs(lam - lam[0]) / abs(lam[-1] - lam[0])
    g = np.linspace(0.0, 1.0, npts)
    return g, np.interp(g, s_pull, w_pull) + np.interp(g, s_dhdl, wd)


def tau_int(x, dt, max_lag=2000):
    """Integrated autocorrelation time, Sokal automatic windowing."""
    x = x - x.mean()
    n = len(x)
    v = float(np.dot(x, x)) / n
    if v <= 0:
        return 0.0
    acc = 0.5
    for k in range(1, min(max_lag, n - 1)):
        acc += float(np.dot(x[:-k], x[k:])) / (n - k) / v
        if k >= 6 * acc:
            break
    return max(acc, 0.0) * dt


# ---- gather ------------------------------------------------------------------

works = read_works()
if not works:
    sys.exit("No complete cycles for %s in %s/ (or results_fe.gs)."
             % (args.struct, args.resultsdir))

ks, Wf, Wr_raw, Wr = leg_pair(works)
n = len(ks)
sd_f, sd_r = Wf.std(ddof=1), Wr.std(ddof=1)
diss = (Wf.mean() - Wr.mean()) / 2.0
pooled = math.sqrt((sd_f ** 2 + sd_r ** 2) / 2.0)
sep = 2.0 * diss / pooled if pooled > 0 else float("nan")
sep_max = 2.0 * NORM.inv_cdf(1.0 - 1.0 / n) if n >= 4 else float("nan")

t_leg = mdp_time_ns(os.path.join(SDIR, LEG_MDP))
L = boresch_value("pull_dist_nm")
if not np.isfinite(L):
    L = 1.0

print("")
print("GroScore-FE leg efficiency -- %s, %s leg" % (args.struct, args.leg))
print("=" * 78)
print("  %d complete cycles   leg %.1f ns   pull %.2f nm   RT %.3f kJ/mol at %.1f K"
      % (n, t_leg, L, RT, args.temp))

# ---- 1. overlap --------------------------------------------------------------
#
# Every bidirectional estimator -- average, CGI, BAR -- extracts dG from the
# region where BOTH distributions are populated. Whether that region contains any
# samples at all is a stronger and blunter question than sep, and it is the first
# thing to answer.

gap = max(0.0, max(Wf.min(), Wr.min()) - min(Wf.max(), Wr.max()))
inside = int(((Wf >= Wr.min()) & (Wf <= Wr.max())).sum()
             + ((Wr >= Wf.min()) & (Wr <= Wf.max())).sum())
print("")
print("-- overlap " + "-" * 67)
print("  forward            [%9.1f , %9.1f]  mean %8.1f  sd %6.1f"
      % (Wf.min(), Wf.max(), Wf.mean(), sd_f))
print("  reverse (-W)       [%9.1f , %9.1f]  mean %8.1f  sd %6.1f"
      % (Wr.min(), Wr.max(), Wr.mean(), sd_r))
print("  dissipation        %8.1f kJ/mol = %.1f RT" % (diss, diss / RT))
print("  sep %.2f vs sep_max %.2f       works inside the other's range: %d of %d"
      % (sep, sep_max, inside, 2 * n))
if inside == 0:
    print("  EMPTY GAP of %.1f kJ/mol (%.1f RT): no estimator has data to work with"
          % (gap, gap / RT))
    print("  here, so avg/CGI/BAR are all reporting the fitted model, not the run.")

# Linear response ties the width to the dissipation; a measured width far above
# that means variance is arriving without hysteresis to pay for it.
lr_sigma = math.sqrt(2 * RT * diss) if diss > 0 else float("nan")
if np.isfinite(lr_sigma) and lr_sigma > 0:
    print("  linear response at this dissipation implies sigma %.1f; measured pooled"
          % lr_sigma)
    print("  %.1f (%.1fx) -- %s" % (pooled, pooled / lr_sigma,
          "extra variance at no extra hysteresis" if pooled > 1.3 * lr_sigma
          else "consistent"))

# ---- 2. trajectory-level diagnosis ------------------------------------------

cf, cr = cycles_with_traces(FWD_TAG), cycles_with_traces(REV_TAG)
have_traces = len(cf) >= 3 and len(cr) >= 3
if not have_traces:
    print("")
    print("-- no per-cycle reductions in %s/, skipping the trajectory diagnosis." % SDIR)
    if args.leg == "bound":
        print("   The bound legs switch restraints without pulling, so they have no")
        print("   pull channel and never write _DG.dat/_dGdt.dat. Expected.")
    else:
        print("   An archived structure keeps them inside its tarball.")
else:
    g, _ = cumulative_work(FWD_TAG, cf[0])
    Cf = np.array([cumulative_work(FWD_TAG, c)[1] for c in cf])
    Cr = np.array([cumulative_work(REV_TAG, c)[1] for c in cr])

    # Reconstruction check: the endpoint of every rebuilt curve must reproduce the
    # work in results_fe.d, or a convention has drifted and nothing below holds.
    err = max((abs(Cf[i, -1] - Wf[ks.index(c)]) for i, c in enumerate(cf)
               if c in ks), default=float("nan"))
    print("")
    print("-- reconstruction check: max |W(1) - results_fe.d| = %.3f kJ/mol" % err)

    # Dissipation density along the path. diss = int h(u) du with
    # h(u) = [f(u) + r(1-u)]/2, f and r the mean force densities of the two
    # directions; the reverse is traversed backwards, hence 1-u.
    h = (np.gradient(Cf.mean(axis=0), g) + np.gradient(Cr.mean(axis=0), g)[::-1]) / 2.0
    tot = TRAPZ(h, g)
    print("")
    print("-- where the hysteresis is generated (integrates to %.1f vs %.1f measured)"
          % (tot, diss))
    for a in np.arange(0.0, 1.0, 0.1):
        m = (g >= a) & (g <= a + 0.1 + 1e-9)
        part = TRAPZ(h[m], g[m])
        share = 100 * part / tot if tot else float("nan")
        bar = "#" * int(round(max(share, 0) / 2))
        print("   %.1f-%.1f  %8.1f kJ/mol  %5.1f %%  %s" % (a, a + 0.1, part, share, bar))

    # Sivak & Crooks: dissipation is minimised at speed ~ 1/sqrt(zeta), giving
    # (int sqrt(zeta) du)^2 / t against int zeta du / t for the constant-rate ramp
    # in use. Assumes local linear response, so read it as an upper bound on the
    # gain -- but it says unambiguously where the time is worth spending.
    pos = np.clip(h, 1e-9, None)
    gain = TRAPZ(pos, g) / TRAPZ(np.sqrt(pos), g) ** 2
    print("")
    print("   a non-uniform ramp of the SAME total length would dissipate %.2fx less"
          % gain)
    print("   (local rate ~ 1/sqrt(density); worth %.2fx more uniform time, free)" % gain)

    # Is the spread generated along the ramp, or chosen at the start? Independent
    # increments would give var(W(s))/var(W(1)) = s and corr(W(s), W(1)) = sqrt(s).
    print("")
    print("-- how the forward spread builds (iid increments would track column 4)")
    print("      s     sd W(s)   var ratio   iid      corr(W(s), W(1))")
    fin = Cf[:, -1]
    for u in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        col = Cf[:, int(u * (len(g) - 1))]
        vr = (col.std(ddof=1) / fin.std(ddof=1)) ** 2 if fin.std(ddof=1) > 0 else np.nan
        cc = np.corrcoef(col, fin)[0, 1] if col.std() > 0 else np.nan
        print("   %5.2f   %8.1f   %9.2f   %6.2f   %14.2f" % (u, col.std(ddof=1), vr, u, cc))

    # Friction. Near equilibrium W_diss = (1/t) int zeta ds with
    # zeta = <dX dX> tau_c / kT, X = dW/ds the conjugate force. Comparing that
    # against the measured dissipation tests whether the 1/t law applies at all:
    # if the fluctuations account for only a fraction of the hysteresis, the rest
    # is carried by modes slower than the window and a slower ramp will not
    # deliver 1/t.
    print("")
    print("-- linear-response check (does dissipation scale as 1/t?)")
    for label, tag, cyl in (("forward", FWD_TAG, cf), ("reverse", REV_TAG, cr)):
        zs = []
        for c in cyl[:args.max_friction_cycles]:
            gg, w = cumulative_work(tag, c, npts=10000)
            X = np.gradient(w, gg)
            dt = t_leg * 1000.0 / (len(X) - 1)
            edges = np.linspace(0, len(X), args.windows + 1).astype(int)
            z = []
            for a, b in zip(edges[:-1], edges[1:]):
                seg = X[a:b]
                if len(seg) < 50:
                    continue
                k = np.arange(len(seg))
                seg = seg - np.polyval(np.polyfit(k, seg, 2), k)   # strip the drift
                z.append(seg.var(ddof=1) * tau_int(seg, dt))
            if z:
                zs.append(z)
        if zs:
            pred = np.array(zs).mean(axis=0).mean() / RT / (t_leg * 1000.0)
            print("   %-8s friction predicts W_diss = %7.2f kJ/mol (%.1f RT) at %.0f ns"
                  % (label, pred, pred / RT, t_leg))
    print("   %-8s measured                = %7.2f kJ/mol (%.1f RT)"
          % ("", diss, diss / RT))
    print("   A prediction far BELOW the measurement means the hysteresis is carried")
    print("   by modes slower than the averaging window, and a uniformly slower ramp")
    print("   relieves it much more weakly than 1/t.")

# ---- 3. is the spread inherited from the starting structure? ----------------
#
# The bound-state leg and the unbinding leg are separate simulations started from
# the same per-cycle structure. If cycles differed mainly by their starting point
# the two would correlate; if they do not, the heterogeneity is generated during
# the switch and no amount of equilibration up front will remove it.

W_intro = np.array([works[c][0] for c in ks])
W_unb = np.array([SIGN_PULL_FWD * works[c][1] + works[c][2] for c in ks])
if n >= 4:
    print("")
    print("-- independent legs of the same cycle")
    print("   corr(W_intro, W_unbind) = %+.2f   (near zero: the spread is made during"
          % np.corrcoef(W_intro, W_unb)[0, 1])
    print("   the switch, not inherited from the starting structure)")

# ---- 4. what each lever costs ------------------------------------------------

t_bound = mdp_time_ns(os.path.join(SDIR, "boundfwd.mdp"))
t_hold = mdp_time_ns(os.path.join(SDIR, "nptrev_fe.mdp"))
fixed = args.equil + (0.0 if math.isnan(t_bound) else 2 * t_bound) \
        + (0.0 if math.isnan(t_hold) else t_hold)
t_unb = mdp_time_ns(os.path.join(SDIR, "bind_fe.mdp"))
per_cycle = 2 * t_unb + fixed

print("")
print("-- cost model " + "-" * 64)
print("   per cycle: 2 x %.1f ns switching + %.1f ns fixed = %.1f ns  (%.0f%% switching)"
      % (t_unb, fixed, per_cycle, 100 * 2 * t_unb / per_cycle))
print("   %d cycles = %.0f ns of MD for this structure" % (n, n * per_cycle))

print("")
print("-- what each lever would buy " + "-" * 49)
if not (np.isfinite(sep) and np.isfinite(sep_max) and sep > sep_max):
    print("   This leg already satisfies sep <= sep_max, so neither lever is needed")
    print("   for overlap; more cycles still shrink the CI as 1/sqrt(n).")
else:
    print("   Adding cycles does NOT move sep; it raises sep_max and shrinks the CI as")
    print("   1/sqrt(n). Lengthening the leg lowers sep but costs ~2t per cycle.")
    need = next((m for m in range(4, 10 ** 7)
                 if 2 * NORM.inv_cdf(1.0 - 1.0 / m) >= sep), None)
    if need:
        print("   cycles only, leg unchanged: n = %d to reach sep_max %.2f -> %.0f ns (%.1fx)"
              % (need, sep, need * per_cycle, need * per_cycle / (n * per_cycle)))
    if inside == 0:
        print("   ...but with an empty gap that only makes sep_max pass on paper: the")
        print("   forward works would have to reach %.1f sigma below their mean to put"
              % ((Wf.mean() - Wr.max()) / sd_f if sd_f > 0 else float("nan")))
        print("   a single sample in the overlap region, which no cycle count reaches.")
    print("   time only, n unchanged, for sep <= %.2f under diss ~ t^-p:" % sep_max)
    for p in (1.0, 0.5, 0.25):
        tn = t_unb * (sep / sep_max) ** (2.0 / p)
        cost = n * (2 * tn + fixed)
        print("      p = %.2f -> %8.1f ns legs -> %9.0f ns (%.1fx)"
              % (p, tn, cost, cost / (n * per_cycle)))
print("")
print("   At a FIXED budget the average estimator has SE ~ sqrt(RT*d/n) with")
print("   n ~ 1/t, so SE ~ t^((1-p)/2). p = 1 is the ceiling, because 1/t is the")
print("   near-equilibrium limit of the dissipation. Doubling the leg at equal cost:")
for p in (1.0, 0.5, 0.25):
    print("      p = %.2f -> SE x %.2f   %s"
          % (p, 2 ** ((1 - p) / 2),
             "break-even" if p >= 1.0 else "worse than spending it on cycles"))
print("   That is the VARIANCE side only. Lower dissipation also removes BIAS,")
print("   which no number of cycles can. Measure p with a short run at 2x the leg")
print("   length before committing to either.")
print("")
