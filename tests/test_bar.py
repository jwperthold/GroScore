#!/usr/bin/env python3
"""Regression test for the Bennett Acceptance Ratio estimator.

utils/estimators.py solves the Bennett root with a bracketed bisection
vectorised over rows, rather than calling pymbar, because pymbar's bar() cannot
be vectorised over bootstrap rows and at 730 us per call the existing 50000-row
bootstrap would cost about 73 s per structure. That buys a 500x speedup and owns
about 40 lines of numerics, so the numerics are pinned here, five ways:

  1. against the analytic Crooks-Gaussian answer, dF = mu_F - sigma^2/2,
  2. against an independent scalar bisection written longhand below, which pins
     the vectorisation without sharing its code,
  3. against a deliberate sign flip, which is the failure this module is most
     likely to suffer and least likely to show: a flipped w_R still returns a
     confident finite number, just the wrong one,
  4. one case per degeneracy token, asserting NaN AND, for the worst case, that
     the raw solver would have returned something plausible, so the guard cannot
     be quietly relaxed later,
  5. against pymbar itself when it is importable, skipped otherwise.

Background: the whole point of the guard is that a degenerate BAR is not
detectable from the solver's output. Identical works with no overlap return
25.0; a single cycle returns a finite number; 24 cycles at the dissipation the
classic engine actually runs at return the wrong sign. The solver brackets
cleanly in all of them. Only overlap_count == 0 catches them.

Run:  python3 tests/test_bar.py
"""

import math, os, sys, warnings
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "utils"))
import estimators as est

failures = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- references

def crooks_pair(n, dF, sigma, seed=1234):
    """Forward and reverse works satisfying Crooks, in kT, reverse in own sign.

    For two Gaussians of equal width obeying P_f(W)/P_r(-W) = exp(W - dF), the
    forward mean is dF + sigma^2/2 and the sign-aligned reverse mean is
    dF - sigma^2/2. Returning the reverse in ITS OWN sign means negating that.
    """
    rng = np.random.default_rng(seed)
    wF = rng.normal(dF + 0.5 * sigma ** 2, sigma, n)
    wR = -rng.normal(dF - 0.5 * sigma ** 2, sigma, n)
    return wF, wR


def bar_scalar(wF, wR, tol=1e-13):
    """Independent scalar bisection on Bennett's implicit equation.

    Deliberately written longhand, with plain math.log rather than logsumexp and
    a while loop rather than a fixed count, so that agreeing with the vectorised
    path is evidence and not tautology.
    """
    nF, nR = len(wF), len(wR)
    M = math.log(nF / nR)

    def fz(d):
        a = sum(-math.log1p(math.exp(min(M + w - d, 700.0))) for w in wF)
        b = sum(-math.log1p(math.exp(min(-(M - w - d), 700.0))) for w in wR)
        # logsumexp of identical-weight terms reduces to a mean shift; compare
        # the two sums directly in the same way _fzero does.
        mx_a = max(-math.log1p(math.exp(min(M + w - d, 700.0))) for w in wF)
        mx_b = max(-math.log1p(math.exp(min(-(M - w - d), 700.0))) for w in wR)
        la = mx_a + math.log(sum(math.exp(-math.log1p(math.exp(min(M + w - d, 700.0))) - mx_a) for w in wF))
        lb = mx_b + math.log(sum(math.exp(-math.log1p(math.exp(min(-(M - w - d), 700.0))) - mx_b) for w in wR))
        return la - lb

    lo = min(min(wF), min(-np.asarray(wR))) - 200.0
    hi = max(max(wF), max(-np.asarray(wR))) + 200.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if fz(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- 1. analytic

print("\n1. analytic Crooks-Gaussian reference")
for n, tol in ((200000, 0.05), (2000000, 0.01)):
    wF, wR = crooks_pair(n, dF=3.0, sigma=2.0)
    v, why = est.bar(wF, wR, rt=1.0)
    check("n=%-8d recovers dF = 3.0 kT" % n, why == "" and abs(v - 3.0) < tol,
          "got %.4f (%s)" % (v, why or "ok"))

# rt is a pure scale factor: the same data in kJ/mol must give RT times the kT answer.
RT = 0.00831446261815324 * 310.0
wF, wR = crooks_pair(200000, dF=3.0, sigma=2.0)
v_kj, _ = est.bar(wF * RT, wR * RT, rt=RT)
v_kt, _ = est.bar(wF, wR, rt=1.0)
check("rt scaling is exact", abs(v_kj - v_kt * RT) < 1e-9 * abs(v_kj),
      "%.6f vs %.6f kJ/mol" % (v_kj, v_kt * RT))


# ------------------------------------------------- 2. independent scalar root

print("\n2. vectorised path against an independent scalar bisection")
wF, wR = crooks_pair(400, dF=2.0, sigma=1.5, seed=77)
v_vec = est._bar_rows_kt(wF[None, :], wR[None, :])[0]
v_sca = bar_scalar(list(wF), list(wR))
check("vectorised == scalar", abs(v_vec - v_sca) < 1e-9,
      "%.12f vs %.12f" % (v_vec, v_sca))

# Every row of a multi-row solve must equal its own scalar root.
rows_f = np.stack([crooks_pair(300, 1.0 + i, 1.2, seed=100 + i)[0] for i in range(4)])
rows_r = np.stack([crooks_pair(300, 1.0 + i, 1.2, seed=100 + i)[1] for i in range(4)])
v_rows = est._bar_rows_kt(rows_f, rows_r)
ok = all(abs(v_rows[i] - bar_scalar(list(rows_f[i]), list(rows_r[i]))) < 1e-9
         for i in range(4))
check("all rows of a batched solve match", ok,
      "max dev %.2e" % max(abs(v_rows[i] - bar_scalar(list(rows_f[i]), list(rows_r[i])))
                           for i in range(4)))


# --------------------------------------------------------- 3. sign convention

print("\n3. sign convention")
wF, wR = crooks_pair(200000, dF=3.0, sigma=2.0, seed=5)
v_ok, _ = est.bar(wF, wR, rt=1.0)
v_flip, _ = est.bar(wF, -wR, rt=1.0)
check("w_R passed in its OWN sign, not negated",
      abs(v_ok - 3.0) < 0.05 and abs(v_flip - 3.0) > 0.5,
      "correct %.4f, flipped %.4f" % (v_ok, v_flip))

# groscore.py stores sign-ALIGNED pushes (job.run calls integrate.py without -r),
# so its call site must negate. Build pushes the way the classic engine does and
# confirm the negation recovers the same answer.
pushes_aligned = -wR                      # what results_<even>.gs holds
v_classic, _ = est.bar(wF, -pushes_aligned, rt=1.0)
check("classic sign-aligned pushes recover the same value",
      abs(v_classic - v_ok) < 1e-12, "%.6f vs %.6f" % (v_classic, v_ok))


# ------------------------------------------------------------- 4. degeneracy

print("\n4. degeneracy guard")

# The case that matters most: cleanly separated, identical works. The raw solver
# is confident and wrong; the guard must catch it.
f_id = np.full(6, 100.0)
r_id = np.full(6, -50.0)                  # sign-aligned +50, no overlap with 100
v, why = est.bar(f_id, r_id, rt=1.0)
raw = est._bar_rows_kt(f_id[None, :], r_id[None, :])[0]
check("no overlap -> NaN, not the solver's confident answer",
      math.isnan(v) and why == "BAR_NO_OVERLAP" and np.isfinite(raw),
      "guard=%s, raw solver would return %.3f" % (why, raw))

# Real-scale separation: 24 cycles at the dissipation the classic engine runs at.
rng = np.random.default_rng(9)
f_sep = rng.normal(+300.0, 30.0, 24)
r_sep = -rng.normal(-300.0, 30.0, 24)
v, why = est.bar(f_sep, r_sep, rt=RT)
raw = est._bar_rows_kt(f_sep[None, :] / RT, r_sep[None, :] / RT)[0] * RT
check("132 RT separation -> NaN, raw solver still finite",
      math.isnan(v) and why == "BAR_NO_OVERLAP" and np.isfinite(raw),
      "raw would return %.2f kJ/mol" % raw)

for n in (1, 2):
    wf, wr = crooks_pair(n, dF=3.0, sigma=2.0, seed=n)
    v, why = est.bar(wf, wr, rt=1.0)
    check("n=%d -> BAR_N_LT_3" % n, math.isnan(v) and why == "BAR_N_LT_3", why)

wf, wr = crooks_pair(20, dF=3.0, sigma=2.0, seed=3)
wf[4] = np.nan
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    v, why = est.bar(wf, wr, rt=1.0)
check("a NaN work -> BAR_NONFINITE, no RuntimeWarning escapes",
      math.isnan(v) and why == "BAR_NONFINITE"
      and not any(issubclass(w.category, RuntimeWarning) for w in caught),
      "%s, %d warning(s)" % (why, len(caught)))

wf, wr = crooks_pair(10, dF=3.0, sigma=2.0, seed=4)
v, why = est.bar(wf, wr[:5], rt=1.0)
check("mismatched lengths -> BAR_UNPAIRED", math.isnan(v) and why == "BAR_UNPAIRED", why)

# The guard must not over-fire: identical works that DO overlap are degenerate
# data but a well-posed problem, and BAR should answer.
v, why = est.bar(np.full(6, 10.0), np.full(6, -10.0), rt=1.0)
check("identical works WITH overlap still return a value",
      why == "" and np.isfinite(v), "%.4f" % v)


# ------------------------------------------------------------ overlap_count

print("\n5. overlap_count")
check("separated -> 0", est.overlap_count([10.0, 11.0], [-1.0, -2.0]) == 0)
check("nested -> 2n", est.overlap_count([0.0, 5.0], [0.0, -5.0]) == 4,
      str(est.overlap_count([0.0, 5.0], [0.0, -5.0])))
check("empty -> 0", est.overlap_count([], []) == 0)
check("non-finite -> 0", est.overlap_count([1.0, np.nan], [-1.0, -2.0]) == 0)


# ---------------------------------------------------------------- bootstrap

print("\n6. bootstrap")
rng = np.random.default_rng(12345)
idx = rng.integers(0, 24, size=(5000, 24))

# Comfortable overlap: 0.5 kT dissipation. Essentially every row should solve.
wF, wR = crooks_pair(24, dF=3.0, sigma=1.0, seed=11)
b = est.bar_bootstrap(wF, wR, 1.0, idx)
check("returns one value per row", b.shape == (5000,), str(b.shape))
check("well-overlapping data -> nearly all rows finite",
      np.isfinite(b).mean() > 0.98, "%.3f finite" % np.isfinite(b).mean())
check("bootstrap mean near the point estimate",
      abs(np.nanmean(b) - est.bar(wF, wR, 1.0)[0]) < 0.2,
      "%.4f vs %.4f" % (np.nanmean(b), est.bar(wF, wR, 1.0)[0]))

# Marginal overlap: 2 kT dissipation, which the FULL sample clears but which
# individual resamples do not always. Losing a fraction of rows is the intended
# behaviour, not a defect: a resample whose range collapses has no crossing
# region to read dG from, so NaN and let np.nanstd drop it. Pinned so that a
# future change which silently starts trusting those rows is caught.
wF2, wR2 = crooks_pair(24, dF=3.0, sigma=2.0, seed=11)
b2 = est.bar_bootstrap(wF2, wR2, 1.0, idx)
frac = np.isfinite(b2).mean()
check("marginal overlap -> some rows dropped, most kept",
      0.5 < frac < 1.0, "%.3f finite (full sample overlaps, resamples vary)" % frac)
check("dropping them does not shift the mean",
      abs(np.nanmean(b2) - est.bar(wF2, wR2, 1.0)[0]) < 0.3,
      "%.4f vs %.4f" % (np.nanmean(b2), est.bar(wF2, wR2, 1.0)[0]))

idx_sep = rng.integers(0, 24, size=(500, 24))
b_sep = est.bar_bootstrap(f_sep, r_sep, RT, idx_sep)
check("separated data -> all rows NaN", np.isnan(b_sep).all(),
      "%d finite" % np.isfinite(b_sep).sum())


# ------------------------------------------------------- 7. pymbar crosscheck

print("\n7. pymbar cross-check (optional)")
os.environ["PYMBAR_DISABLE_JAX"] = "1"
try:
    import logging
    logging.getLogger("pymbar").setLevel(logging.ERROR)
    from pymbar.other_estimators import bar as pymbar_bar
except Exception as e:
    print("  %-58s SKIP  (%s)" % ("pymbar not importable", type(e).__name__))
else:
    wF, wR = crooks_pair(50000, dF=3.0, sigma=2.0, seed=21)
    ours, _ = est.bar(wF, wR, rt=1.0)
    theirs = pymbar_bar(wF, wR, compute_uncertainty=False)["Delta_f"]
    check("agrees with pymbar to 1e-6", abs(ours - theirs) < 1e-6,
          "%.10f vs %.10f" % (ours, theirs))


# ---------------------------------------------------------------------- exit

print("")
if failures:
    print("FAILED (%d): %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("All BAR tests passed.")
