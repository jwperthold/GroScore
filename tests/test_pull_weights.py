#!/usr/bin/env python3
"""The per-coordinate pull-path correction, and the weighted integration it forces.

Interface references used to grow by a common scalar u while the partner actually
translated u along the pull axis. Fixing that gives every coordinate its own rate,
which breaks the identity the pull integrator was built on:

    W = sum_i rate_i * integral(F_i dt)   ==   rate * integral(sum_i F_i dt)

The right-hand side is what integrate.py computes from a single -r, and it is only
equal to the work while every rate is the same. So integrate.py grew -R, and the
thing that MUST hold is that -R with uniform weights reproduces the old number
exactly -- otherwise every previously scored cycle silently changes meaning.

Standalone, no pytest: python3 tests/test_pull_weights.py
"""
import os, sys, subprocess, tempfile, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEGRATE = os.path.join(ROOT, "utils", "integrate.py")

failures = []


def check(name, ok, detail=""):
    print("  %-64s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def run(args):
    r = subprocess.run([sys.executable, INTEGRATE] + args,
                       capture_output=True, text=True, cwd=tempfile.gettempdir())
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def write_pullf(path, nco, nrow=200, seed=0):
    """A pull-force file shaped like the real thing: time, then one column per
    coordinate, forces drifting and noisy."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 40, nco)
    with open(path, "w") as f:
        f.write("# fake\n@ title \"pull force\"\n")
        for k in range(nrow):
            t = 2.0 * k
            row = base * (1.0 + 0.004 * k) + rng.normal(0, 8, nco)
            f.write("%.4f " % t + " ".join("%.6f" % v for v in row) + "\n")
    return path


def write_weights(path, w):
    with open(path, "w") as f:
        f.write("# test weights\n")
        for x in w:
            f.write("%.10g\n" % x)
    return path


tmp = tempfile.mkdtemp(prefix="pullw_")
NCO = 25
pf = write_pullf(os.path.join(tmp, "leg_1_pullf.xvg"), NCO)

print("\n[1] uniform weights must reproduce the unweighted arithmetic exactly")
rc0, plain, _ = run(["-f", pf, "-nr", str(NCO), "-r", "1.765e-05"])
wf1 = write_weights(os.path.join(tmp, "ones.gs"), [1.0] * NCO)
rc1, weighted, err1 = run(["-f", pf, "-nr", str(NCO), "-r", "1.765e-05", "-R", wf1])
check("unweighted run succeeds", rc0 == 0, plain)
check("weighted run succeeds", rc1 == 0, err1)
check("outputs are byte-identical", plain == weighted,
      "%r vs %r" % (plain, weighted))

print("\n[2] the weighted sum is the work, not a rescaling of the plain sum")
# Independent reimplementation straight from the definition.
rows = [l.split() for l in open(pf) if l[0] not in "#@"]
t = np.array([float(r[0]) for r in rows])
F = np.array([[float(x) for x in r[1:1 + NCO]] for r in rows])
rng = np.random.default_rng(7)
w = rng.uniform(-0.4, 1.0, NCO)          # includes a coordinate moving BACKWARDS
rate = 1.765e-05
wf2 = write_weights(os.path.join(tmp, "mixed.gs"), w)
rc2, got, err2 = run(["-f", pf, "-nr", str(NCO), "-r", "%.10g" % rate, "-R", wf2])
# integrate.py trapezoids the summed force in time then scales by the rate, and
# prints the negative (groscore_fe.py carries the direction).
fw = F @ w
want = -rate * float(np.trapezoid(fw, t) if hasattr(np, "trapezoid")
                     else np.trapz(fw, t))
check("weighted result matches an independent trapezoid", rc2 == 0 and
      abs(float(got) - want) < 1e-6 * max(1.0, abs(want)),
      "got %s want %.10g (%s)" % (got, want, err2))

print("\n[3] a weights file that disagrees with -nr is fatal, not padded")
wf3 = write_weights(os.path.join(tmp, "short.gs"), [1.0] * (NCO - 3))
rc3, out3, err3 = run(["-f", pf, "-nr", str(NCO), "-r", "1e-5", "-R", wf3])
check("wrong length exits nonzero", rc3 != 0, "rc=%d out=%r" % (rc3, out3))
check("and says so", "weights" in err3 and str(NCO) in err3, err3)
rc4, _, err4 = run(["-f", pf, "-nr", str(NCO), "-r", "1e-5", "-R",
                    os.path.join(tmp, "nope.gs")])
check("a missing weights file exits nonzero", rc4 != 0, err4)

print("\n[4] the reference path: |r + u n| beats |r| + u on real contact geometry")
# Reproduces what make_boresch now writes, from the shipped 2KTF setup if it is
# here; otherwise from a synthetic interface with the same statistics.
d = os.path.join(ROOT, "test6", "2KTF")
vecs, src = None, "synthetic"
try:
    cf = os.path.join(d, "interface_contacts.gs")
    gro = os.path.join(d, "npt_probe_cluster.gro")
    if os.path.isfile(cf) and os.path.isfile(gro):
        xyz = {}
        with open(gro) as fh:
            fh.readline(); nat = int(fh.readline())
            for i in range(nat):
                L = fh.readline()
                xyz[i + 1] = (float(L[20:28]), float(L[28:36]), float(L[36:44]))
        pr = [l.split() for l in open(cf) if not l.startswith("#") and l.strip()]
        vecs = np.array([np.array(xyz[int(p[1])]) - np.array(xyz[int(p[0])])
                         for p in pr])
        src = "test6/2KTF, %d pairs" % len(vecs)
except (OSError, ValueError, KeyError, IndexError):
    vecs = None
if vecs is None or not len(vecs):
    rng = np.random.default_rng(3)
    v = rng.normal(size=(200, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    vecs = v * rng.uniform(0.3, 0.6, 200)[:, None]

nhat = np.array([0.0, 0.0, 1.0])
if src != "synthetic":
    nhat = np.array([0.37, -0.51, 0.77]); nhat /= np.linalg.norm(nhat)
d0 = np.linalg.norm(vecs, axis=1)
RAMP = [(0.0, 0.3), (0.3, 0.5), (0.5, 1.0)]


def truth(u):
    return np.linalg.norm(vecs + u * nhat, axis=1)


def rms(x):
    return float(np.sqrt((x ** 2).mean())) * 10.0


worst_old = worst_new = 0.0
for u in np.linspace(0.0, 1.0, 51):
    worst_old = max(worst_old, rms((d0 + u) - truth(u)))
    for a, b in RAMP:                       # one chord per stage, as shipped
        if a - 1e-9 <= u <= b + 1e-9:
            f = (u - a) / (b - a)
            chord = truth(a) + f * (truth(b) - truth(a))
            worst_new = max(worst_new, rms(chord - truth(u)))
            break
print("      source: %s" % src)
print("      worst mismatch: common scalar %.2f A -> per-stage chord %.2f A"
      % (worst_old, worst_new))
check("the per-stage chord is at least 5x closer to the true path",
      worst_new * 5.0 < worst_old, "%.3f vs %.3f" % (worst_new, worst_old))
check("and it is exact at every stage boundary",
      all(rms(truth(b) - truth(b)) < 1e-12 for _, b in RAMP))

print("\n[5] weights are direction-independent, so one file serves fwd and rev")
for a, b in RAMP:
    fwd = (truth(b) - truth(a)) / ((b - a) * 1.0)
    rev = (truth(a) - truth(b)) / ((a - b) * 1.0)
    check("stage %.1f-%.1f: forward and reverse weights agree" % (a, b),
          np.allclose(fwd, rev, atol=1e-12))

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all pull-weight checks passed")
