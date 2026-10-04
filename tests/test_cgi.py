#!/usr/bin/env python3
"""Regression test for the Crooks Gaussian Intersection estimator.

The CGI crossing is implemented twice: groscore.py's cgi_crossing, vectorised and
used for both the point estimate and the bootstrap, and groscore_fe.py's, which
its work-distribution diagnostic reuses. They cannot share a helper without
restructuring the entry points, so this test pins the maths of the classic one
down instead, taking the function out of groscore.py itself (ast), four ways:

  1. against eq. (12) of Goette & Grubmueller, J Comput Chem 30, 447-456 (2009),
     transcribed literally in the paper's own standard-deviation form,
  2. against the defining property, P_f = P_r at the returned point, and a
     numerical root of P_f(W) - P_r(-W) found by bisection,
  3. the degenerate cases: the paper's p. 449 fallback to the mean, equal widths
     (the exact answer is the midpoint, not 0/0), widths equal to within rounding
     (the textbook closed form gave -256.0 for a true -210.0) and a zero width (no
     crossing exists: nan, never the mean),
  4. end to end, by running groscore.py on synthetic works and reading the CGI
     value and its Score_note back out of scores_cgi.gs: the overlap gate
     (separated works are nan with CGI_NO_OVERLAP; touching works pass), the
     20-cycle minimum (19 nan, 20 a value), a zero width (CGI_DEGENERATE), the
     per-cycle and the legacy layouts, and a seeded, finite CI95.

Background: until 2026-08-10 the shipped discriminant read

    (Wf - Wr)^2/(vf*vr) + 2*(1/vf - 1/vr)*ln(vr/vf)

i.e. the paper's factor 2 kept while the logarithm was taken of the VARIANCE
ratio rather than the SIGMA ratio. Since 2*ln(sr/sf) = ln(vr/vf) the factor was
applied twice, displacing the reported crossing whenever the two work
distributions differed in width. Until v1.5 there was no overlap gate: on works
132 RT apart, where BAR says BAR_NO_OVERLAP, the CGI moved by the full half gap
with the width ratio alone (v15_work digest_audit SCORER-2), and a nan crossing
was silently replaced by the mean.

Run:  python3 tests/test_cgi.py
"""

import ast, math, os, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROSCORE = os.path.join(REPO, "groscore.py")
TOL = 1e-6
failures = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- the function under test

def from_groscore(name):
    """A top-level function of groscore.py, compiled on its own (the script runs
    argparse at import, so it cannot simply be imported)."""
    tree = ast.parse(open(GROSCORE).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np, "math": math}
            exec(compile(ast.Module(body=[node], type_ignores=[]), GROSCORE, "exec"), ns)
            return ns[name]
    return None


cgi_fn = from_groscore("cgi_crossing")
check("groscore.py defines cgi_crossing", cgi_fn is not None)


def cgi_shipped(wf, sf, wr_aligned, sr, fallback=True):
    """groscore.py's crossing for one pair of means and standard deviations."""
    if cgi_fn is None:
        return float("nan")
    return float(cgi_fn(wf, sf ** 2, wr_aligned, sr ** 2, fallback=fallback))


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

print("1. groscore.py's crossing == Goette & Grubmueller eq. (12), sigma form")
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
        continue                                   # the paper's form divides by ~0 here
    a = cgi_shipped(wf, sf, wr, sr, fallback=False)
    b = cgi_paper(wf, sf, wr, sr)
    if not np.isfinite(b):
        continue
    n_cmp += 1
    if not np.isfinite(a):
        worst_paper = worst_cross = float("inf")
        continue
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
      n_cmp > 19000 and worst_paper < TOL, "max |diff| = %.3e" % worst_paper)

print("")
print("2. the returned value really is a Gaussian intersection")
# Two Gaussians of unequal width cross twice. The paper takes the root closest to
# (Wf - Wr)/2, which is usually the one between the means but in extreme cases is
# the tail root, so the invariant to test is P_f = P_r at the answer, not that
# the answer sits between the means.
check("log P_f == log P_r at the raw crossing",
      worst_cross < 1e-6, "max |diff| = %.3e" % worst_cross)

print("")
print("3. degenerate crossings")
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

# Equal widths cross exactly once, at the midpoint of the means.
got = cgi_shipped(-380.0, 30.0, -40.0, 30.0, fallback=False)
check("equal sigma 30/30, means -380/-40 -> the midpoint -210.0",
      abs(got - (-210.0)) < 1e-9, "got %r" % got)
got = cgi_shipped(-380.0, 30.0, -40.0, 30.0 * (1 + 1e-12), fallback=False)
check("sigma equal to within 1e-12 -> still -210.0 (closed form: -256)",
      abs(got - (-210.0)) < 1e-3, "got %r" % got)
got = cgi_shipped(-380.0, 30.0, -40.0, 30.0 * (1 + 1e-6), fallback=False)
check("sigma 1e-6 apart -> the crossing near the midpoint (-210.0)",
      abs(got - (-210.0)) < 1e-2, "got %r" % got)
for what, args in (("pull width 0", (-200.0, 0.0, -190.0, 5.0)),
                   ("push width 0", (-200.0, 5.0, -190.0, 0.0)),
                   ("both widths 0, equal means", (-200.0, 0.0, -200.0, 0.0))):
    got = cgi_shipped(*args)
    check("%s -> nan, never the mean" % what, np.isnan(got), "got %r" % got)
if cgi_fn is not None:
    v = cgi_fn(np.array([-380.0, 10.0, -200.0]), np.array([900.0, 16.0, 0.0]),
               np.array([-40.0, 10.4, -190.0]), np.array([900.0, 484.0, 25.0]))
    check("vectorised rows are independent: -210.0, 10.2, nan",
          v.shape == (3,) and abs(v[0] + 210.0) < 1e-9 and abs(v[1] - 10.2) < 1e-9
          and np.isnan(v[2]), str(v))
else:
    check("vectorised rows are independent: -210.0, 10.2, nan", False, "no cgi_crossing")

print("")
print("4. end to end: groscore.py -> scores_cgi.gs")


def score_row(pulls, pushes, layout="percycle", n=None):
    """Run groscore.py on one structure X1 and return its scores_cgi.gs fields."""
    td = tempfile.mkdtemp(prefix="cgi_")
    try:
        with open(os.path.join(td, "sp.gs"), "w") as f:
            f.write("X1\tB\n")
        with open(os.path.join(td, "results_0.gs"), "w") as f:
            f.write("# stage 0\nX1 OK\n")
        if layout == "legacy":
            for i in range(len(pulls)):
                with open(os.path.join(td, "results_%d.gs" % (2 * i + 1)), "w") as f:
                    f.write("X1 %.6f\n" % pulls[i])
                with open(os.path.join(td, "results_%d.gs" % (2 * i + 2)), "w") as f:
                    f.write("X1 %.6f 1.0\n" % pushes[i])
        else:
            os.mkdir(os.path.join(td, "results.d"))
            for i in range(len(pulls)):
                with open(os.path.join(td, "results.d", "X1_c%d.gs" % (i + 1)), "w") as f:
                    f.write("X1 %d %.6f %.6f 1.0\n" % (i + 1, pulls[i], pushes[i]))
        r = subprocess.run([sys.executable, GROSCORE, "-n", str(n or len(pulls))],
                           cwd=td, capture_output=True, text=True)
        path = os.path.join(td, "scores_cgi.gs")
        if not os.path.isfile(path):
            return None, r.stderr.strip()[-200:]
        for line in open(path):
            if line.startswith("X1"):
                return line.split(), ""
        return None, "no X1 row"
    finally:
        shutil.rmtree(td, ignore_errors=True)


def note(fields):
    return fields[10] if fields and len(fields) > 10 else None


# Overlapping works of unequal spread, where the old and new discriminant
# disagree by several kJ/mol, as the original end-to-end case had them.
NP = 24
pulls = rng.normal(-110.0, 30.0, NP)
pushes = rng.normal(-90.0, 9.0, NP)
want = cross_numeric(float(np.mean(pulls)), float(np.std(pulls)),
                     float(np.mean(pushes)), float(np.std(pushes)))
for layout in ("percycle", "legacy"):
    fields, err = score_row(pulls, pushes, layout)
    got = float(fields[1]) if fields else float("nan")
    check("%s rows: CGI matches the true crossing (%.1f)" % (layout, want),
          fields is not None and abs(got - want) < 0.05 and note(fields) == "nan",
          "got %s, note %s %s" % (fields[1] if fields else None, note(fields), err))
ci = float(fields[2]) if fields else float("nan")
check("  with a finite CI95", np.isfinite(ci) and ci > 0, "CI95 %r" % ci)
again, _ = score_row(pulls, pushes)
check("  that is seeded: a second run gives the same row", again == score_row(pulls, pushes)[0]
      and again is not None, str(again))

# Separated works: every pull below every push, as the classic engine produces
# them (132 RT apart). Only the avg is reported; the CGI is not a measurement.
sep_p = rng.normal(-300.0, 10.0, NP)
sep_q = rng.normal(-100.0, 30.0, NP)
fields, err = score_row(sep_p, sep_q)
check("separated works -> CGI nan with CGI_NO_OVERLAP",
      fields is not None and fields[1] == "nan" and note(fields) == "CGI_NO_OVERLAP",
      "got %s %s" % (fields, err))

# The gate's edge: the largest pull equal to the smallest push still overlaps.
tp = [-200.0 - k for k in range(20)]
tq = [-200.0 + 2.0 * k for k in range(20)]
want_t = cross_numeric(float(np.mean(tp)), float(np.std(tp)), float(np.mean(tq)), float(np.std(tq)))
fields, err = score_row(tp, tq)
check("touching works (max pull == min push) -> a CGI (%.1f)" % want_t,
      fields is not None and fields[1] != "nan" and abs(float(fields[1]) - want_t) < 0.05,
      "got %s %s" % (fields, err))
fields, err = score_row(tp, [x + 0.001 for x in tq])
check("  0.001 apart -> CGI_NO_OVERLAP",
      fields is not None and fields[1] == "nan" and note(fields) == "CGI_NO_OVERLAP",
      "got %s %s" % (fields, err))

# The minimum cycle count, from below and at it.
fields, err = score_row(pulls[:19], pushes[:19])
check("19 cycles -> nan with CGI_N_LT_20",
      fields is not None and fields[1] == "nan" and note(fields) == "CGI_N_LT_20",
      "got %s %s" % (fields, err))
fields, err = score_row(pulls[:20], pushes[:20])
check("20 cycles -> a CGI", fields is not None and fields[1] != "nan" and note(fields) == "nan",
      "got %s %s" % (fields, err))

# A zero width: no Gaussian, so no crossing. Through v1.4 twenty -0.0 works gave
# 0.0 and a constant push crashed math.log.
fields, err = score_row([-200.0] * 20, [-200.0] * 20)
check("20 identical works -> nan with CGI_DEGENERATE (v1.4: the mean)",
      fields is not None and fields[1] == "nan" and note(fields) == "CGI_DEGENERATE",
      "got %s %s" % (fields, err))
fields, err = score_row(list(pulls[:20]), [-100.0] * 20)
check("a constant push -> nan with CGI_DEGENERATE (v1.4: crash)",
      fields is not None and fields[1] == "nan" and note(fields) == "CGI_DEGENERATE",
      "got %s %s" % (fields, err))

print("")
if failures:
    print("FAILED: %s" % ", ".join(failures))
    sys.exit(1)
print("All CGI checks passed.")
