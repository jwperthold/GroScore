#!/usr/bin/env python3
"""A triad the measured anchor search accepts must fit the box budget whichever r the
setup then writes.

THE FAILURE (review R1). The anchor search admits a triad when the ENSEMBLE mean r
plus pull_dist fits R_LIMIT_FRAC of the pull limit. The pose gate replaces the
ensemble mean by the SNAPSHOT for all six coordinates when any one of them is more
than 15 deg / 0.12 nm from -f, and the final guard then checks the snapshot r
against the same budget. A triad whose snapshot r is the larger one passed the
search and stopped the setup with BORESCH_R_TOO_LARGE after the five probes: 2KTF
in a 7.15 nm box with the gate at 2 deg (accepted r 1.091, snapshot 1.112, budget
2.102 nm). On the real 1IJK probe the first v1.5 round accepted r 2.201 nm whose
snapshot r was 2.263 against a 3.257 nm budget, and passed only because the gate did
not fire there.

THE HARNESS. The real make_boresch.py, end to end, with gmx replaced: a wrapper
serves gmx trjconv from a synthetic probe ensemble (40 frames of a complex of two
rigid sheets of ALA, one continuous chain each, tumbling by up to 20 deg and
jiggled by 0.012 nm), and every other gmx call fails, so masses fall back to element
masses and the readback is off. -f is the same complex with the ligand lifted
0.4 nm, so the pose gate fires on r for any triad.

WHAT IS PINNED:

  [1] in a box with room (9.0 nm, budget 2.646 nm) the measured search accepts, the
      gate fires on r, and the snapshot r is written.
  [2] in a 6.47 nm box (budget 1.902 nm) the setup completes on measured anchors;
      the accepted triad's ensemble AND snapshot r fit the budget, and so does the
      r0_release written (0.488 / 0.883 nm here). Budgeting the ensemble r alone
      accepts r 0.559 nm whose snapshot r is 0.920, and the first v1.5 round,
      which also keyed the groups by seed, accepts r 0.663 nm with a snapshot r of
      0.998: both then stop with BORESCH_R_TOO_LARGE (1.920 and 1.998 nm).

Fails on the first v1.5 round (BORESCH_R_TOO_LARGE in [2]) and passes now.

Standalone, no pytest: python3 tests/test_boresch_r_budget.py
"""
import math, os, re, shutil, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(ROOT, "utils")
TPL = os.path.join(ROOT, "settings", "amber19sb_opc3")
sys.path.insert(0, UTILS)
import fe_protocol as P

failures = []


def check(name, ok, detail=""):
    print("  %-76s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


# ------------------------------------------------------------------ the complex
def strand(nres, origin, axis):
    axis = np.asarray(axis, float)
    up = np.array([0.0, 0.0, 1.0])
    out, p = [], np.asarray(origin, float)
    for _ in range(nres):
        n = p
        ca = n + 0.110 * axis + 0.096 * up
        c = ca + 0.120 * axis - 0.093 * up
        out.append([n, ca, c])
        p = c + 0.133 * axis
    return out


def meander(nres, nstrands, origin):
    """Residues [N, CA, C] of ONE continuous chain folded into antiparallel strands
    0.564 nm apart, a turn residue along +y between them."""
    out, p = [], np.asarray(origin, float)
    for s in range(nstrands):
        seg = strand(nres, p, (1, 0, 0) if s % 2 == 0 else (-1, 0, 0))
        out += seg
        if s < nstrands - 1:
            n = seg[-1][2] + np.array([0.0, 0.133, 0.0])
            ca = n + np.array([0.0, 0.146, 0.0])
            c = ca + np.array([0.0, 0.152, 0.0])
            out.append([n, ca, c])
            p = c + np.array([0.0, 0.133, 0.0])
    return out


def atoms_of(residues, res0, zdir):
    """(resnum, name, xyz): O off C in the plane, CB off CA toward the partner."""
    out = []
    for k, (n, ca, c) in enumerate(residues):
        for nm, x in (("N", n), ("CA", ca), ("C", c), ("O", c + np.array([0.0, -0.12, 0.0])),
                      ("CB", ca + np.array([0.0, 0.0, 0.15 * zdir]))):
            out.append((res0 + k, nm, x))
    return out


REC = atoms_of(meander(8, 5, (0.0, 0.0, 0.0)), 1, +1)
NREC = REC[-1][0]
LIG = atoms_of(meander(6, 4, (0.4, 0.3, 0.5)), NREC + 1, -1)
ATOMS = REC + LIG
X0 = np.array([a[2] for a in ATOMS]) + 3.0
IS_LIG = np.array([a[0] > NREC for a in ATOMS])
LIFT = 0.4                       # nm the ligand sits higher in -f than in the ensemble


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


rng = np.random.default_rng(3)
FRAMES = np.array([(X0 - X0.mean(0)) @ rot(rng.normal(size=3), rng.uniform(0, math.radians(20))).T
                   + X0.mean(0) + rng.normal(0, 0.3, 3) + rng.normal(0, 0.012, X0.shape)
                   for _ in range(40)])

# gmx trjconv served from FRAMES; every other gmx call fails.
WRAP = r'''
import runpy, subprocess, sys, types
import numpy as np
def fake(cmd, *a, **k):
    if list(cmd[:2]) == ["gmx", "trjconv"]:
        o = dict(zip(cmd[2::2], cmd[3::2]))
        z = np.load(o["-f"] + ".npz")
        sel = [int(t) for ln in open(o["-n"]) if not ln.startswith("[") for t in ln.split()]
        X = z["X"][:, np.array(sel) - 1, :]
        with open(o["-o"], "w") as f:
            for fr in range(len(X)):
                f.write("t= %d\n%5d\n" % (fr, len(sel)))
                for i, a in enumerate(sel):
                    f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % ((1, "X", "X", a % 100000) + tuple(X[fr, i])))
                b = z["M"][fr]
                f.write("%10.5f%10.5f%10.5f\n" % (b[0, 0], b[1, 1], b[2, 2]))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return types.SimpleNamespace(returncode=1, stdout="", stderr="no gmx in this test")
subprocess.run = fake
utils = sys.argv[1]
sys.argv = [utils + "/make_boresch.py"] + sys.argv[2:]
runpy.run_path(utils + "/make_boresch.py", run_name="__main__")
'''


def run(box_l):
    """(stderr, exit code, boresch_analytical.gs as a dict) of a setup whose legs run
    in a cubic box of edge box_l."""
    d = tempfile.mkdtemp(prefix="rbudget_")
    try:
        snap = X0.copy()
        snap[IS_LIG] += np.array([0.0, 0.0, LIFT])
        with open(os.path.join(d, "f.gro"), "w") as fh:
            fh.write("synthetic\n%5d\n" % len(ATOMS))
            for k, ((r, nm, _), x) in enumerate(zip(ATOMS, snap), 1):
                fh.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (r, "ALA", nm, k, x[0], x[1], x[2]))
            fh.write("%10.5f%10.5f%10.5f\n" % (12.0, 12.0, 12.0))
        with open(os.path.join(d, "box.gro"), "w") as fh:
            fh.write("box\n    1\n    1SOL     OW    1   1.000   1.000   1.000\n"
                     "%10.5f%10.5f%10.5f\n" % (box_l, box_l, box_l))
        open(os.path.join(d, "chain_map.gs"), "w").write(
            "".join("%d\n" % r for r in range(NREC + 1, ATOMS[-1][0] + 1)))
        np.savez(os.path.join(d, "rep1.xtc.npz"), X=FRAMES,
                 M=np.repeat(np.diag([12.0] * 3)[None], len(FRAMES), 0))
        for f in ("rep1.xtc", "rep1.tpr"):
            open(os.path.join(d, f), "w").close()
        for m in sorted({l["mdp"] for l in P.legs()}) + ["npt_c_fe.mdp"]:
            shutil.copy(os.path.join(TPL, m), d)
        open(os.path.join(d, "wrap.py"), "w").write(WRAP)
        r = subprocess.run([sys.executable, "wrap.py", UTILS, "-f", "f.gro", "-m", "chain_map.gs",
                            "--pull-dist", "%g" % P.PULL_DIST, "--traj", "rep1.xtc",
                            "--tpr", "rep1.tpr", "--index", "index.ndx", "--no-readback",
                            "--box-from", "box.gro"], cwd=d, capture_output=True, text=True,
                           timeout=900)
        an = {}
        p = os.path.join(d, "boresch_analytical.gs")
        if os.path.isfile(p):
            for ln in open(p):
                f = ln.split()
                if len(f) >= 2 and not f[0].startswith("#"):
                    an[f[0]] = f[1]
        return r.stderr, r.returncode, an
    finally:
        shutil.rmtree(d, ignore_errors=True)


def budget(box_l):
    return 0.6 * 0.49 * box_l


def accepted(err):
    """(ensemble r, snapshot r) of the triad the search accepted, from its log."""
    m = re.search(r"the accepted set sits at ([0-9.]+) \(snapshot ([0-9.]+)\)", err)
    return (float(m.group(1)), float(m.group(2))) if m else None


print("\n[1] a box with room: the gate fires on r and the snapshot r is written")
err, rc, an = run(9.0)
check("exit %d, measured anchors accepted" % rc,
      rc == 0 and "accepted eps_max" in err and "burial heuristic" not in err, err[-400:])
check("the pose gate fired on r (-f has the ligand 0.4 nm higher)", "POSE GATE - r:" in err)
check("and the snapshot r was written (%s)" % an.get("reference_source", "?"),
      an.get("reference_source", "").startswith("single_reference_frame,_the_ensemble_mean"))

print("\n[2] a 6.47 nm box: the accepted triad fits the budget with either r")
B = budget(6.47)
err, rc, an = run(6.47)
acc = accepted(err)
print("      budget %.3f nm; accepted (ensemble r, snapshot r) %s" % (B, acc))
check("the setup completes (exit %d)%s" % (rc, "" if rc == 0 else ": " + (
      re.findall(r"ERROR - (.*)", err) or ["?"])[-1][:90]), rc == 0, err[-300:])
check("on measured anchors, not the burial heuristic",
      "accepted eps_max" in err and "burial heuristic" not in err)
check("the search reports the combinations it turned away and the accepted triad's "
      "two r", acc is not None, err[-300:])
if acc:
    check("ensemble r %.3f + 1.0 and snapshot r %.3f + 1.0 both within %.3f nm"
          % (acc[0], acc[1], B), max(acc) + P.PULL_DIST <= B)
r0 = float(an.get("r0_release_nm", "nan"))
check("r0_release %.3f nm within the budget, and the gate fired" % r0,
      r0 <= B and "POSE GATE - r:" in err)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all box-budget checks passed")
