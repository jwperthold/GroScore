#!/usr/bin/env python3
"""Regression test for the Crooks Gaussian Intersection estimator.

The CGI crossing is implemented in three places (groscore.py twice, once scalar
and once vectorised for the bootstrap; groscore_fe.py; utils/plot_fe_works.py).
They cannot share a helper without restructuring the entry points, so this test
pins the maths down instead, three ways:

  1. against eq. (12) of Goette & Grubmueller, J Comput Chem 30, 447-456 (2009),
     transcribed literally in the paper's own standard-deviation form,
  2. against a numerical root of P_f(W) - P_r(-W) found with Brent's method,
  3. end to end, by running groscore.py on synthetic works and reading the CGI
     value back out of scores_cgi.gs.

Background: until 2026-08-10 the shipped discriminant read

    (Wf - Wr)^2/(vf*vr) + 2*(1/vf - 1/vr)*ln(vr/vf)

i.e. the paper's factor 2 kept while the logarithm was taken of the VARIANCE
ratio rather than the SIGMA ratio. Since 2*ln(sr/sf) = ln(vr/vf) the factor was
applied twice, displacing the reported crossing whenever the two work
distributions differed in width.

Run:  python3 tests/test_cgi.py
"""

import math, os, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-6
failures = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- references

def cgi_paper(wf, sf, wr_aligned, sr):
    """Eq. (12) verbatim, in the paper's sigma form.

    The paper writes means as Wf and -Wr; `wr_aligned` is its -Wr, which is what
    GroScore stores. Hence (Wf + Wr) in the paper is (wf - wr_aligned) here.
    """
    v1, v2 = sf ** 2, sr ** 2
    dinv = 1.0 / v1 - 1.0 / v2
    lead = wf / v1 - wr_aligned / v2
    inner = (wf - wr_aligned) ** 2 / (v1 * v2) + 2.0 * dinv * math.log(sr / sf)
    if inner < 0:
        return float("nan")
    root = math.sqrt(inner)
    s1, s2 = (lead + root) / dinv, (lead - root) / dinv
    mid = (wf + wr_aligned) / 2.0                 # paper: (Wf - Wr)/2
    return s2 if abs(mid - s1) > abs(mid - s2) else s1


def cgi_shipped(wf, sf, wr_aligned, sr, fallback=True):
    """The expression as it appears in groscore.py / groscore_fe.py today.

    fallback=False returns the raw crossing, so the tests can check the
    intersection property itself without the degenerate-case substitution.
    """
    vp, vq = sf ** 2, sr ** 2
    dinv = 1.0 / vp - 1.0 / vq
    t1 = wf / vp - wr_aligned / vq
    inner = (wf - wr_aligned) ** 2 / (vp * vq) + dinv * math.log(vq / vp)
    if inner < 0:
        return float("nan")
    t2 = math.sqrt(inner)
    s1, s2 = (t1 + t2) / dinv, (t1 - t2) / dinv
    mid = (wf + wr_aligned) / 2.0
    pick = s2 if abs(mid - s1) > abs(mid - s2) else s1
    if fallback and not (min(wf, wr_aligned) <= pick <= max(wf, wr_aligned)):
        return mid                      # Goette & Grubmueller p. 449
    return pick


def cross_numeric(wf, sf, wr_aligned, sr):
    """Root of P_f(x) - P_r(-x) between the two means, by bisection."""
    pdf = lambda x, m, s: math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    g = lambda x: pdf(x, wf, sf) - pdf(x, wr_aligned, sr)
    lo, hi = min(wf, wr_aligned), max(wf, wr_aligned)
    if g(lo) * g(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- the tests

print("1. shipped formula == Goette & Grubmueller eq. (12), sigma form")
def logpdf(x, m, s):
    return -math.log(s * math.sqrt(2 * math.pi)) - 0.5 * ((x - m) / s) ** 2


rng = np.random.default_rng(20260810)
worst_paper = worst_cross = 0.0
n_cmp = n_between = n_fb_bad = 0
for _ in range(20000):
    wf = rng.uniform(-200, 200)
    wr = wf + rng.uniform(-150, 150)
    sf, sr = rng.uniform(1, 60), rng.uniform(1, 60)
    if abs(sf - sr) < 1e-3:
        continue                                   # denominator -> 0, guarded in callers
    a = cgi_shipped(wf, sf, wr, sr, fallback=False)
    b = cgi_paper(wf, sf, wr, sr)
    if not (np.isfinite(a) and np.isfinite(b)):
        continue
    n_cmp += 1
    worst_paper = max(worst_paper, abs(a - b))
    # The defining property: the answer is a point where the two densities are
    # equal. Compared in log space, which stays accurate far into the tails.
    worst_cross = max(worst_cross, abs(logpdf(a, wf, sf) - logpdf(a, wr, sr)))
    # Fallback bookkeeping: does the raw crossing lie between the two means?
    inside = min(wf, wr) <= a <= max(wf, wr)
    n_between += int(inside)
    got = cgi_shipped(wf, sf, wr, sr, fallback=True)
    want = a if inside else (wf + wr) / 2.0
    if abs(got - want) > 1e-9:
        n_fb_bad += 1
check("shipped vs paper eq. (12) over %d random pairs" % n_cmp,
      worst_paper < TOL, "max |diff| = %.3e" % worst_paper)

print("")
print("2. the returned value really is a Gaussian intersection")
# Two Gaussians of unequal width cross twice. The paper takes the root closest to
# (Wf - Wr)/2, which is usually the one between the means but in extreme cases is
# the tail root -- so the invariant to test is P_f = P_r at the answer, not that
# the answer sits between the means.
check("log P_f == log P_r at the raw crossing",
      worst_cross < 1e-6, "max |diff| = %.3e" % worst_cross)

print("")
print("3. degenerate-crossing fallback (Goette & Grubmueller p. 449)")
check("fallback fires exactly when no root lies between the means",
      n_fb_bad == 0, "%d mismatches" % n_fb_bad)
print("       (%d of %d crossings lie between the means, %d fall back to the mean)"
      % (n_between, n_cmp, n_cmp - n_between))

# A concrete degenerate pair: nearly coincident means, very different widths.
wf_d, sf_d, wr_d, sr_d = 10.0, 4.0, 10.4, 22.0
raw = cgi_shipped(wf_d, sf_d, wr_d, sr_d, fallback=False)
got = cgi_shipped(wf_d, sf_d, wr_d, sr_d, fallback=True)
check("means 10.0/10.4, sigma 4/22: raw crossing %.1f -> mean %.2f" % (raw, (wf_d + wr_d) / 2),
      abs(got - (wf_d + wr_d) / 2.0) < 1e-9, "got %.2f" % got)

# A well-behaved pair must be untouched by the fallback.
raw_ok = cgi_shipped(188.6, 40.4, 117.3, 11.6, fallback=False)
got_ok = cgi_shipped(188.6, 40.4, 117.3, 11.6, fallback=True)
check("well-separated pair is not altered by the fallback",
      abs(raw_ok - got_ok) < 1e-12, "%.3f" % got_ok)

# The panel that exposed the bug: 2KTF bound-state restraints.
wf, sf, wr, sr = 188.6, 40.4, 117.3, 11.6
got, want = cgi_shipped(wf, sf, wr, sr), cross_numeric(wf, sf, wr, sr)
check("2KTF panel (mu 188.6/117.3, sigma 40.4/11.6) -> %.3f" % want,
      abs(got - want) < 1e-4, "got %.3f" % got)

# The pre-fix expression, kept explicit so the bug cannot silently return.
def cgi_buggy(wf, sf, wr, sr):
    vp, vq = sf ** 2, sr ** 2
    dinv, t1 = 1.0 / vp - 1.0 / vq, wf / vp - wr / vq
    t2 = math.sqrt((wf - wr) ** 2 / (vp * vq) + 2.0 * dinv * math.log(vq / vp))
    s1, s2 = (t1 + t2) / dinv, (t1 - t2) / dinv
    mid = (wf + wr) / 2.0
    return s2 if abs(mid - s1) > abs(mid - s2) else s1

check("the pre-2026-08-10 expression is genuinely different",
      abs(cgi_buggy(wf, sf, wr, sr) - want) > 1.0,
      "buggy %.3f vs true %.3f" % (cgi_buggy(wf, sf, wr, sr), want))

print("")
print("4. end to end: groscore.py -> scores_cgi.gs")
# 20 cycles of pull/push works with deliberately unequal spread, where the old
# and new formulas disagree by several kJ/mol.
NP = 24
pulls = rng.normal(120.0, 30.0, NP)
pushes = rng.normal(-60.0, 9.0, NP)
with tempfile.TemporaryDirectory(dir=REPO) as td:
    os.mkdir(os.path.join(td, "X1"))
    with open(os.path.join(td, "sp.gs"), "w") as f:
        f.write("X1\tB\n")
    with open(os.path.join(td, "results_0.gs"), "w") as f:
        f.write("# stage 0\nX1 OK\n")
    for i in range(NP):
        with open(os.path.join(td, "results_%d.gs" % (2 * i + 1)), "w") as f:
            f.write("X1 %.6f\n" % pulls[i])
        with open(os.path.join(td, "results_%d.gs" % (2 * i + 2)), "w") as f:
            f.write("X1 %.6f 1.0\n" % pushes[i])
    r = subprocess.run([sys.executable, os.path.join(REPO, "groscore.py"), "-n", str(NP)],
                       cwd=td, capture_output=True, text=True)
    path = os.path.join(td, "scores_cgi.gs")
    if r.returncode != 0 or not os.path.isfile(path):
        check("groscore.py produced scores_cgi.gs", False, r.stderr.strip()[-200:])
    else:
        got = float("nan")
        for line in open(path):
            if line.startswith("X1"):
                got = float(line.split()[1])
        want = cross_numeric(float(np.mean(pulls)), float(np.std(pulls)),
                             float(np.mean(pushes)), float(np.std(pushes)))
        check("scores_cgi.gs CGI matches the true crossing (%.1f)" % want,
              abs(got - want) < 0.05, "got %.1f" % got)

print("")
if failures:
    print("FAILED: %s" % ", ".join(failures))
    sys.exit(1)
print("All CGI checks passed.")
