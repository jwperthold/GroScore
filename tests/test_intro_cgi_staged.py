#!/usr/bin/env python3
"""dG_intro_cgi is the staged sum over the bound sub-legs; the bound-leg dissipation
check fires beyond the two-sided 95% Student t quantile at n - 1 degrees of freedom,
from 10 cycles on, and the report states the flags chance alone predicts; a
LIGAND_FAIL structure is reported and never scored.

Three properties of groscore_fe.py, each pinned because its failure is silent:

  1. THE INTRO CGI POINT ESTIMATE IS THE STAGED SUM. Its CI (the joint bootstrap)
     and dG_unbind_cgi were already sums over sub-legs and stages, but the point
     estimate was the one-shot CGI of the summed bound switch, so the column and
     its interval described two different estimators. On pass 2 they differed by
     up to 8.04 kJ/mol (1E6E; bm_fe_amber_2/_ana/digest/corr.txt F2). The works
     below are built so the two disagree by 1.77 kJ/mol, and both target values
     are fixed numbers, checked here against an independent bisection for the
     crossing of the two fitted Gaussians.

  2. THE v1.5 ACCEPTANCE CHECK. A bound sub-leg whose two directions start from
     equilibrium cannot have negative mean dissipation (<W_f> + <W_r>)/2; pass 2
     had it negative in 39 of 43 structures and nothing flagged it. The first cut,
     2 standard errors from 3 cycles on, flagged an exactly reversible sub-leg
     9.5% of the time at n = 3 and 3.9% at n = 10, because se comes from the same
     few cycles (review QC-3). The cut is now the t quantile, so the critical
     values are FIXED numbers from the t table: 2.2621571628540993 at n = 10
     (df 9), 2.2281388519649385 at n = 11, 2.093024054408263 at n = 20. The
     per-cycle values are m + (-3)*5 + (+3)*5, whose standard error is exactly 1,
     and every m used is a dyadic fraction, so z = m exactly. The cases separate
     the t quantile from 2 se (z -2.1), from a normal 1.96 and a one-sided t 1.833
     (z -2.1), and df n - 1 from df n (z -2.25 sits between 2.228 and 2.262); the
     strict inequality is checked at exactly t_crit through neg_diss_flag. Nine
     cycles give no verdict, ten do. The report counts the expected chance flags:
     2.5% of the tested sub-legs.

  3. A STAGE-0 STATUS WINS OVER RESULT ROWS. v1.5's job_fe.run writes LIGAND_FAIL
     when an extracted ligand does not reach topol.top. Such a structure must get a
     nan row whose Note names the status, and must be named in the report, even if
     an earlier setup left complete cycles behind.

Run:  python3 tests/test_intro_cgi_staged.py
"""

import math, os, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "utils"))
import fe_protocol as P

failures = []


def check(name, ok, detail=""):
    print("  %-62s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


# Four cycles of bound works, opposite width ratios in the two sub-legs. Reverse
# works are RAW, as job_fe.run stores them.
F1 = [10.0, 11.0, 12.0, 11.0]
V1 = [10.0, 0.0, -10.0, 0.0]
F2 = [10.0, 30.0, 50.0, 30.0]
V2 = [-9.0, -10.0, -11.0, -10.0]
STAGED = 21.1761442780      # CGI(sub-leg 1) 9.2242058592 + CGI(sub-leg 2) 11.9519384188
ONE_SHOT = 22.9480982261    # CGI of the summed switch: what the point used to be
AVG = 25.5                  # (mean f - mean(-v)) / 2, identical either way

# Student t, upper 0.975 quantile, from the table (df = n - 1).
T10, T11, T20 = 2.2621571628540993, 2.2281388519649385, 2.093024054408263


def crossing(fwd, rev):
    """Independent CGI: the root of log N(mu_f, s_f) - log N(-mu_r, s_r) between
    the two means, by bisection. Population sd, as the scorer uses."""
    f, v = np.asarray(fwd, float), np.asarray(rev, float)
    m1, s1, m2, s2 = f.mean(), f.std(), (-v).mean(), v.std()
    g = lambda x: (-math.log(s1) - 0.5 * ((x - m1) / s1) ** 2) - \
                  (-math.log(s2) - 0.5 * ((x - m2) / s2) ** 2)
    lo, hi = min(m1, m2), max(m1, m2)
    if g(lo) * g(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


print("\n0. the fixed numbers are Gaussian intersections, independently of groscore_fe")
c_st = crossing(F1, V1) + crossing(F2, V2)
c_1s = crossing(np.add(F1, F2), np.add(V1, V2))
check("bisection, staged sum = %.6f" % STAGED, abs(c_st - STAGED) < 1e-6, "%.10f" % c_st)
check("bisection, one-shot = %.6f" % ONE_SHOT, abs(c_1s - ONE_SHOT) < 1e-6, "%.10f" % c_1s)
check("and they differ by more than any file rounding", abs(STAGED - ONE_SHOT) > 1.7)

# groscore_fe runs its argument parsing and run_config.gs bookkeeping at import,
# so it is imported from a scratch directory, never from the repository.
work = tempfile.mkdtemp(prefix="intro_cgi_")
here = os.getcwd()
os.chdir(work)
sys.argv = [sys.argv[0]]
sys.path.insert(0, REPO)
G = None
try:
    import groscore_fe as G
except Exception as e:                                   # pragma: no cover
    check("groscore_fe imports", False, "%s: %s" % (type(e).__name__, e))
os.chdir(here)

print("\n1. score_structure: the point estimate is the staged sum")
if G is not None:
    try:
        # Filler stages with unequal widths, so their CGI is defined.
        stages = [(L, [5.0, 6.0, 7.0, 6.0], [-3.0, -4.0, -6.0, -3.0])
                  for L in P.stage_letters()[:2]]
        Wf = np.sum([s[1] for s in stages], axis=0)
        Wv = np.sum([s[2] for s in stages], axis=0)
        r = G.score_structure(np.add(F1, F2), np.add(V1, V2), Wf, Wv, -30.0,
                              stages=stages, bound=[("1", F1, V1), ("2", F2, V2)],
                              n_boot=2000, n_boot_bar=200)
        check("intro_cgi = %.6f, the staged sum" % STAGED,
              abs(r["intro_cgi"] - STAGED) < 1e-6, "%.10f" % r["intro_cgi"])
        check("and not the one-shot %.6f" % ONE_SHOT,
              abs(r["intro_cgi"] - ONE_SHOT) > 1.7, "%.10f" % r["intro_cgi"])
        check("intro_avg is untouched (%.1f)" % AVG, abs(r["intro_avg"] - AVG) < 1e-9,
              "%.10f" % r["intro_avg"])
        check("dG_bind_cgi is built on the staged value",
              abs(r["bind_cgi"] + (STAGED + r["unb_cgi"] - 30.0)) < 1e-9,
              "%.6f" % r["bind_cgi"])
        # A run from before the bound leg was split keeps its one-shot value: with
        # one sub-leg the two are the same estimator.
        r1 = G.score_structure(np.add(F1, F2), np.add(V1, V2), Wf, Wv, -30.0,
                               stages=stages, bound=None, n_boot=2000, n_boot_bar=200)
        check("an unsplit bound leg still reads its own CGI",
              abs(r1["intro_cgi"] - ONE_SHOT) < 1e-6, "%.10f" % r1["intro_cgi"])
    except Exception as e:
        check("score_structure runs", False, "%s: %s" % (type(e).__name__, e))

print("\n2. bound_dissipation: NEG_DISS beyond the t quantile, from 10 cycles")
PAT = np.array([-3.0] * 5 + [3.0] * 5)    # mean 0, sum of squares 90: se exactly 1
Q = np.array([0.5, -1.5, 2.5, -1.5, 0.5, 1.0, -1.0, 0.0, 2.0, -2.5])


def subleg(m, n=10):
    """Forward and raw reverse works whose per-cycle (f + v)/2 is m + PAT. With n
    other than 10 the pattern is cut or repeated; only the verdict is read then."""
    p = np.resize(PAT, n)
    q = np.resize(Q, n)
    return list(30.0 + 2.0 * (m + p) + 2.0 * q), list(-30.0 - 2.0 * q)


# Dyadic, so z = m exactly. T10 = 2.26216: just inside, just beyond, and between
# df n (2.22814) and df n - 1.
INSIDE, BEYOND, BETWEEN = -2.2578125, -2.265625, -2.25

if G is not None:
    try:
        check("t_crit at n = 10 is the df 9 quantile %.6f" % T10,
              abs(G.neg_diss_crit(10) - T10) < 1e-12, repr(G.neg_diss_crit(10)))
        check("t_crit at n = 11 is the df 10 quantile %.6f" % T11,
              abs(G.neg_diss_crit(11) - T11) < 1e-12, repr(G.neg_diss_crit(11)))
        check("t_crit at n = 20 is the df 19 quantile %.6f" % T20,
              abs(G.neg_diss_crit(20) - T20) < 1e-12, repr(G.neg_diss_crit(20)))
        check("no t_crit at n = 9", not np.isfinite(G.neg_diss_crit(9)),
              repr(G.neg_diss_crit(9)))
        check("exactly at t_crit: not flagged (strict)",
              G.neg_diss_flag(-T10 * 1.0, 1.0, T10) is False)
        check("one ulp beyond t_crit: flagged",
              G.neg_diss_flag(np.nextafter(-T10, -np.inf), 1.0, T10) is True)
        check("one ulp inside t_crit: not flagged",
              G.neg_diss_flag(np.nextafter(-T10, 0.0), 1.0, T10) is False)
        check("no verdict without a t_crit",
              G.neg_diss_flag(-50.0, 1.0, float("nan")) is None)

        rows = G.bound_dissipation([
            ("in", *subleg(INSIDE)), ("out", *subleg(BEYOND)),
            ("dfn", *subleg(BETWEEN)), ("z21", *subleg(-2.125)),
            ("pos", *subleg(0.5)), ("n9", *subleg(-5.0, 9)),
            ("n10", *subleg(-5.0, 10)), ("n11", *subleg(-5.0, 11))])
        by = {row[0]: row for row in rows}
        _, n, diss, se, crit, flag = by["in"]
        check("the construction is exact: diss %r, se 1.0, n 10" % INSIDE,
              diss == INSIDE and se == 1.0 and n == 10, "%r %r %r" % (diss, se, n))
        check("z %.4f, inside t_crit: not flagged" % INSIDE, flag is False, str(by["in"]))
        check("z %.4f, beyond t_crit: flagged" % BEYOND, by["out"][5] is True,
              str(by["out"]))
        check("z -2.25: not flagged (df n would flag it)", by["dfn"][5] is False,
              str(by["dfn"]))
        check("z -2.125: not flagged (2 se, 1.96 or one-sided 1.833 would)",
              by["z21"][5] is False, str(by["z21"]))
        check("positive dissipation is never flagged", by["pos"][5] is False, str(by["pos"]))
        check("9 cycles: no verdict even at z -5", by["n9"][5] is None, str(by["n9"]))
        check("10 cycles: a verdict, flagged at z -5", by["n10"][5] is True, str(by["n10"]))
        check("11 cycles: a verdict, flagged at z -5", by["n11"][5] is True, str(by["n11"]))
        check("the row carries its own t_crit",
              abs(by["n11"][4] - T11) < 1e-12 and not np.isfinite(by["n9"][4]),
              "%r %r" % (by["n11"][4], by["n9"][4]))
        check("the interval is 95% two-sided and the minimum 10 cycles",
              G.NEG_DISS_CONF == 0.95 and G.N_MIN_DISS == 10)
    except Exception as e:
        check("bound_dissipation runs", False, "%s: %s" % (type(e).__name__, e))

print("\n3. end to end through groscore_fe.py")
# One result row per cycle in the protocol's own layout, so the test follows any
# reshaping of the ramp. Pull channels are zero and the dhdl channels carry the
# works, which makes the stored value the physical work in both directions.
NAMES = [w[0] for w in P.works()]
BOUND = [b[0] for b in P.BOUND]
FILL_F = [5.0, 6.0, 7.0, 6.0]
FILL_V = [-3.0, -4.0, -6.0, -3.0]


def rows_for(sid, bound_works, n=4):
    """bound_works: {sub-leg name: (fwd list, rev list)}. Ramp stages are filler."""
    out = []
    for c in range(n):
        vals = {}
        for L in P.stage_letters():
            vals["Wu%s_pull" % L], vals["Wu%s_dhdl" % L] = 0.0, FILL_F[c % 4]
            vals["Wr%s_pull" % L], vals["Wr%s_dhdl" % L] = 0.0, FILL_V[c % 4]
        for name in BOUND:
            f, v = bound_works.get(name, subleg(0.5, n))
            vals["W_intro%s" % name], vals["W_remove%s" % name] = f[c], v[c]
        out.append("%s %d %s 1.50" % (sid, c + 1, " ".join("%.6f" % vals[k] for k in NAMES)))
    return out


run = os.path.join(work, "run")
os.makedirs(os.path.join(run, "results_fe.d"))
struct = {
    "CGI":   ({BOUND[0]: (F1, V1), BOUND[-1]: (F2, V2)}, 4),
    "NEGD":  ({BOUND[0]: subleg(BETWEEN), BOUND[-1]: subleg(BEYOND)}, 10),
    "ATTH":  ({BOUND[0]: subleg(INSIDE), BOUND[-1]: subleg(0.5)}, 10),
    "SHORT": ({BOUND[0]: subleg(-5.0, 9), BOUND[-1]: subleg(-5.0, 9)}, 9),
    "LIGF":  ({BOUND[0]: (F1, V1), BOUND[-1]: (F2, V2)}, 4),
}
for sid, (bw, n) in struct.items():
    for c, line in enumerate(rows_for(sid, bw, n)):
        with open(os.path.join(run, "results_fe.d", "%s_c%d.gs" % (sid, c + 1)), "w") as f:
            f.write(line + "\n")
with open(os.path.join(run, "sp.gs"), "w") as f:
    f.write("# Structure_ID  Chains_for_Protein_B\n")
    for sid in struct:
        f.write("%s B\n" % sid)
with open(os.path.join(run, "results_analytical.gs"), "w") as f:
    for sid in struct:
        f.write("%s -30.0\n" % sid)
# LIGF set up once, then failed its ligand check on a later setup: the LAST line
# for a structure is its state.
with open(os.path.join(run, "results_0.gs"), "w") as f:
    f.write("# Stage-0 setup status\n%sLIGF LIGAND_FAIL\n"
            % "".join("%s OK\n" % s for s in struct))
with open(os.path.join(run, "run_config.gs"), "w") as f:
    f.write("numruns 10\n")
# struct_map.gs present means "already submitted": this run only scores.
open(os.path.join(run, "struct_map.gs"), "w").write("# test\n")
proc = subprocess.run([sys.executable, os.path.join(REPO, "groscore_fe.py")], cwd=run,
                      capture_output=True, text=True, env=dict(os.environ, MPLBACKEND="Agg"))
out = proc.stdout
scores = os.path.join(run, "scores_fe.gs")
ran = proc.returncode == 0 and os.path.isfile(scores)
check("groscore_fe.py ran and wrote scores_fe.gs", ran,
      "" if ran else (proc.stderr or "")[-300:])
if os.path.isfile(scores):
    L = [l.rstrip("\n") for l in open(scores)]
    hdr = [l for l in L if l.startswith("# Structure_ID")][0].lstrip("#").split()
    body = {l.split("\t")[0]: dict(zip(hdr, l.split("\t")))
            for l in L if not l.startswith("#") and l.strip()}
    cgi = body.get("CGI", {})
    check("CGI: dG_intro_cgi in the file is the staged %.2f" % STAGED,
          cgi.get("dG_intro_cgi") == "%.2f" % STAGED, str(cgi.get("dG_intro_cgi")))
    # The acceptance check belongs to the report, not to the table: every column
    # of scores_fe.gs is read positionally by the benchmark scripts.
    check("scores_fe.gs gains no column for it",
          not any("diss" in c.lower() for c in hdr) and "NEG_DISS" not in open(scores).read())
    lig = body.get("LIGF", {})
    numeric = [c for c in hdr if c not in ("Structure_ID", "Ncycles", "Note")]
    check("LIGF: every numeric column is nan despite 4 complete cycles",
          bool(lig) and all(lig[c] == "nan" for c in numeric))
    check("LIGF: Ncycles 0 and Note LIGAND_FAIL",
          lig.get("Ncycles") == "0" and lig.get("Note") == "LIGAND_FAIL",
          "%s / %s" % (lig.get("Ncycles"), lig.get("Note")))
check("the report names it: 'Not scored: LIGAND_FAIL LIGF.'",
      "Not scored: LIGAND_FAIL LIGF." in out,
      [l for l in out.splitlines() if l.startswith("Not scored")].__repr__())


def diss_line(sid, name):
    for line in out.splitlines():
        t = line.split()
        if len(t) >= 8 and t[0] == sid and t[1] == name:
            return t
    return None


ln = diss_line("NEGD", BOUND[-1])
check("NEGD sub-leg %s (z %.4f) is printed NEG_DISS" % (BOUND[-1], BEYOND),
      ln is not None and ln[-1] == "NEG_DISS", str(ln))
ln = diss_line("NEGD", BOUND[0])
check("NEGD sub-leg %s (z -2.25) is printed '-'" % BOUND[0],
      ln is not None and ln[-1] == "-", str(ln))
ln = diss_line("ATTH", BOUND[0])
check("ATTH sub-leg %s (z %.4f) is printed '-' with se 1.00, t_crit 2.262"
      % (BOUND[0], INSIDE),
      ln is not None and ln[-1] == "-" and ln[3] == "-2.26" and ln[4] == "1.00"
      and ln[6] == "2.262", str(ln))
ln = diss_line("SHORT", BOUND[0])
check("SHORT (9 cycles) is printed n/a, not NEG_DISS",
      ln is not None and ln[-1] == "n/a" and ln[6] == "n/a", str(ln))
check("the summary names exactly the flagged structure, out of the tested ones",
      "NEG_DISS in 1 of 2 tested structures (2 more with fewer than 10 cycles): NEGD."
      in out, [l for l in out.splitlines() if "NEG_DISS in" in l].__repr__())
check("and the flags chance predicts: 2.5% of the 4 tested sub-legs, 0.10",
      "sub-legs, flags 0.10 of them (2.5% each); observed 1." in out,
      [l for l in out.splitlines() if "flags 0." in l].__repr__())
check("LIGF, never scored, is not in the dissipation table either",
      diss_line("LIGF", BOUND[0]) is None)

shutil.rmtree(work, ignore_errors=True)
print("")
if failures:
    print("FAILED (%d): %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("All staged-intro-CGI and dissipation checks passed.")
