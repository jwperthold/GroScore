#!/usr/bin/env python3
"""measure_box.py sizes the box from the physical diameter of the protein complex.

Pass 2 (bm_fe_amber_2/_ana/digest: setup.txt S5, crash.txt box-measure-artifact,
perf.txt Q3) found its running-max D_max larger than the probe box it was measured
in for 18 of 46 structures (2AJF 20.07 nm in a 9.75 nm box). Two causes: cutout
fragments that mdrun left in different periodic images, which -pbc cluster cannot
rejoin because a cutout partner is one moleculetype of unbonded fragments; and
additives in Protein_Struct (sulfate as neutral H2SO4, glycols, detached sugars)
drifting through the solvent. 11 structures then fell back to a 1.8 nm pad, 7 kept
boxes up to 3.6 times too large. Pinned here, through the command line job_fe.run
uses:

  1. A synthetic two-replica probe in which a fragmented complex rotates and
     drifts by more than half a box, every frame wrapped atom by atom as mdrun
     writes it, a sulfate drifts off into the solvent and a zinc stays bound,
     measures the complex's own diameter. With --compression 1 and the -c
     structure being the complex itself, the returned editconf -d is then exactly
     the requested pad, 1.500 nm, whatever the diameter is.
  2. The box is sized from the 99th percentile, the widest frame left once
     floor(n/100) are set aside. 250 frames set aside 2: one or two frames with a
     terminus flung 1.4 nm out leave the result at 1.500 (just below and at the
     limit), a third one moves it (just above), by exactly half of what that frame
     adds to the diameter. The percentile is over the POOLED frames: two outliers
     in ONE replica of 125 still leave 1.500, where a percentile per replica (1 of
     125 set aside, then the widest replica) would give 2.144 (review QC-2; on the
     real 2KTF probe that statistic moves D from 5.164 to 5.241 nm).
  3. The frame count arithmetic, at fixed n.
  4. Which residues count as protein: every cap GROMACS lists as Protein, CT3 and
     NH2 included (review QC-9), but no ligand or ion, even one with atoms named N,
     C or CA.

Needs gmx (grompp for the fixture's run input, trjconv inside measure_box.py).

Run:  python3 tests/test_measure_box_v15.py
"""

import math, os, re, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB = os.path.join(REPO, "utils", "measure_box.py")
failures = []


def check(name, ok, detail=""):
    print("  %-66s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


def dodecahedron(d):
    return np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [d / 2, d / 2, d * math.sqrt(2) / 2]])


def wrap_atoms(x, box):
    """Every atom into the brick, as mdrun's put_atoms_in_box does: along c, b, a."""
    x = np.array(x, float)
    for m in (2, 1, 0):
        x -= np.floor(x[:, m] / box[m, m])[:, None] * box[m]
    return x


def rotation(axis, deg):
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def gro_frame(atoms, xyz, box, t):
    out = ["synthetic probe t= %.3f step= %d" % (t, int(t)), "%d" % len(atoms)]
    for i, ((rn, res, an), (x, y, z)) in enumerate(zip(atoms, xyz)):
        out.append("%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (rn, res, an, i + 1, x, y, z))
    out.append("".join("%10.5f" % v for v in (box[0, 0], box[1, 1], box[2, 2], box[0, 1],
                                                 box[0, 2], box[1, 0], box[1, 2],
                                                 box[2, 0], box[2, 1])))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- the complex
# Eight 8-residue strands, 0.5 nm apart, as unbonded fragments: chain A at strand
# positions 0, 7, 3 (sequence-consecutive fragments 3.5 nm apart), chain B between.
FRAG_A, FRAG_B = [0, 7, 3], [1, 2, 4, 5, 6]
atoms, X, frag = [], [], []
rn = 0
for pos in (FRAG_A, FRAG_B):
    for k in pos:
        for r in range(8):
            rn += 1
            ca = np.array([0.5 * k, 0.38 * r, 0.0])
            for an, o in (("N", (0.04, -0.13, 0.03)), ("CA", (0.0, 0.0, 0.0)),
                          ("C", (-0.04, 0.13, -0.03))):
                atoms.append((rn, "ALA", an))
                X.append(ca + np.array(o))
                frag.append(len(frag) and (frag[-1] + (1 if (r == 0 and an == "N") else 0)))
X = np.array(X)
NP = len(atoms)                                    # 192 protein atoms
NA = 3 * 8 * len(FRAG_A)
CENTER = X.mean(axis=0)
# A sulfate on the sheet that will drift off, and a zinc that stays bound.
SO4 = [("S", (0.0, 0.0, 0.50)), ("O1", (0.12, 0.08, 0.55)), ("O2", (-0.12, 0.08, 0.55)),
       ("O3", (0.0, -0.14, 0.55)), ("O4", (0.0, 0.0, 0.35))]
atoms += [(rn + 1, "SO4", n) for n, _ in SO4] + [(rn + 2, "ZN", "ZN")]
X_SO4 = np.array([CENTER + np.array(o) for _, o in SO4])
X_ZN = CENTER + np.array([0.0, 0.0, -0.30])
TRUE_D = max(np.linalg.norm(a - b) for a in X for b in X)
# The outlier: chain A's first N atom, at the sheet's corner, flung 1.4 nm outward.
OUT_D = max(np.linalg.norm(a - b) for a in np.vstack([X[1:], X[:1] + [-1.0, -1.0, 0.0]])
            for b in np.vstack([X[1:], X[:1] + [-1.0, -1.0, 0.0]]))

BOX = dodecahedron(7.0)
AXES = ([0.3, 0.5, 0.81], [-0.6, 0.2, 0.77])


def replica(path, rep, nframes, outliers):
    """One probe replica: rotating, drifting complex, drifting sulfate, wrapped."""
    with open(path, "w") as f:
        for t in range(nframes):
            R = rotation(AXES[rep], 15.0 * rep + 0.7 * t)
            T = np.array([1.0 + 2.0 * rep, 2.0, 1.5]) + t * np.array([0.031, -0.023, 0.017])
            body = X.copy()
            if t in outliers:
                body[0] += np.array([-1.0, -1.0, 0.0])
            prot = (body - CENTER) @ R.T + CENTER + T
            so4 = (X_SO4 - CENTER) @ R.T + CENTER + T + t * np.array([0.03, 0.025, 0.035])
            zn = (X_ZN - CENTER) @ R.T + CENTER + T
            xyz = wrap_atoms(np.vstack([prot, so4, zn[None]]), BOX)
            f.write(gro_frame(atoms, xyz, BOX, 10.0 * t))


def itp_chain(name, first, last):
    lines = ["[ moleculetype ]", "%s 3" % name, "", "[ atoms ]"]
    for i in range(first, last):
        r, res, an = atoms[i]
        t, m = ("NX", 14.007) if an == "N" else ("CX", 12.011)
        lines.append("%6d %4s %5d %5s %5s %5d 0.0 %8.3f" % (i - first + 1, t, r, res, an,
                                                          i - first + 1, m))
    lines += ["", "[ bonds ]"]
    for i in range(first, last - 1):
        if frag[i] == frag[i + 1]:
            lines.append("%6d %6d 1 0.14 100000" % (i - first + 1, i - first + 2))
    return "\n".join(lines) + "\n"


def fixture(d):
    """The probe directory measure_box sees: tpr, index with Protein_Struct, -c structure."""
    open(os.path.join(d, "a.itp"), "w").write(itp_chain("Protein_chain_A", 0, NA))
    open(os.path.join(d, "b.itp"), "w").write(itp_chain("Protein_chain_B", NA, NP))
    open(os.path.join(d, "topol.top"), "w").write(
        "[ defaults ]\n1 2 no 1.0 1.0\n\n[ atomtypes ]\n"
        "NX 7 14.007 0.0 A 0.30 0.10\nCX 6 12.011 0.0 A 0.30 0.10\n"
        "SX 16 32.060 0.0 A 0.30 0.10\nOX 8 15.999 0.0 A 0.30 0.10\n"
        "ZX 30 65.380 0.0 A 0.20 0.10\n\n"
        '#include "a.itp"\n#include "b.itp"\n\n'
        "[ moleculetype ]\nSO4 3\n\n[ atoms ]\n"
        "1 SX 1 SO4 S 1 0.0 32.06\n2 OX 1 SO4 O1 2 0.0 16.0\n3 OX 1 SO4 O2 3 0.0 16.0\n"
        "4 OX 1 SO4 O3 4 0.0 16.0\n5 OX 1 SO4 O4 5 0.0 16.0\n\n[ bonds ]\n"
        "1 2 1 0.15 100000\n1 3 1 0.15 100000\n1 4 1 0.15 100000\n1 5 1 0.15 100000\n\n"
        "[ moleculetype ]\nZN 1\n\n[ atoms ]\n1 ZX 1 ZN ZN 1 0.0 65.38\n\n"
        "[ system ]\nsynthetic probe\n\n[ molecules ]\n"
        "Protein_chain_A 1\nProtein_chain_B 1\nSO4 1\nZN 1\n")
    open(os.path.join(d, "fix.mdp"), "w").write(
        "integrator = md\nnsteps = 0\ncutoff-scheme = Verlet\npbc = xyz\n"
        "coulombtype = Cut-off\nrcoulomb = 0.9\nrvdw = 0.9\n")
    struct = np.vstack([X, X_SO4, X_ZN[None]]) + np.array([1.0, 2.0, 1.5])
    open(os.path.join(d, "struct.gro"), "w").write(gro_frame(atoms, struct, BOX, 0.0))
    n = len(atoms)
    with open(os.path.join(d, "index.ndx"), "w") as f:
        for g in ("System", "Protein_Struct"):
            f.write("[ %s ]\n" % g)
            f.write("\n".join(" ".join("%d" % (i + 1) for i in range(s, min(s + 15, n)))
                              for s in range(0, n, 15)) + "\n")
    r = subprocess.run(["gmx", "grompp", "-f", "fix.mdp", "-c", "struct.gro", "-p",
                        "topol.top", "-o", "probe.tpr", "-maxwarn", "20"], cwd=d,
                       capture_output=True, text=True, env=dict(os.environ, GMX_MAXBACKUP="-1"))
    return r.returncode == 0 and os.path.isfile(os.path.join(d, "probe.tpr")), r.stderr


def measure(d, out1, out2, tag):
    """measure_box on two replicas of 125 frames, the job_fe.run command line plus
    --compression 1. Returns (pad or nan, stderr)."""
    replica(os.path.join(d, "rep1_%s.gro" % tag), 0, 125, out1)
    replica(os.path.join(d, "rep2_%s.gro" % tag), 1, 125, out2)
    r = subprocess.run([sys.executable, MB, "-f", "rep1_%s.gro" % tag, "rep2_%s.gro" % tag,
                        "-s", "probe.tpr", "probe.tpr", "-n", "index.ndx",
                        "-g", "Protein_Struct", "-c", "struct.gro", "--pad", "1.5",
                        "--compression", "1.0"], cwd=d, capture_output=True, text=True,
                       env=dict(os.environ, GMX_MAXBACKUP="-1"))
    try:
        return float(r.stdout.split()[-1]), r.stderr
    except (ValueError, IndexError):
        return float("nan"), r.stderr


work = tempfile.mkdtemp(prefix="measure_box_")
ok_fix, err = fixture(work)

print("\n1. split fragments, more than half a box of drift, a drifting additive")
check("fixture: grompp builds the probe run input", ok_fix, "" if ok_fix else err[-300:])
print("   true protein diameter %.4f nm; one outlier frame makes it %.4f" % (TRUE_D, OUT_D))
if ok_fix:
    pad0, e0 = measure(work, set(), set(), "clean")
    check("the editconf -d is the requested 1.500, i.e. D is the true diameter",
          abs(pad0 - 1.5) <= 0.003, "%.3f" % pad0)
    m = re.search(r"D = p\S+ = ([0-9.]+) nm", e0)
    D = float(m.group(1)) if m else float("nan")
    check("and the D it reports is %.3f nm" % TRUE_D, abs(D - TRUE_D) <= 0.003, "%.4f" % D)
    check("the sulfate and the zinc are named as left out",
          "1 x SO4" in e0 and "1 x ZN" in e0,
          ([l for l in e0.splitlines() if "left out" in l] or [e0[-200:]])[0])
    check("250 pooled frames, 2 of them set aside",
          "250 frames pooled" in e0 and "once the 2 widest are set aside" in e0)

print("\n2. a frame-level outlier does not set the box; 1% of the frames does")
if ok_fix:
    pad1, _ = measure(work, {40}, set(), "o1")
    pad2, _ = measure(work, {40}, {90}, "o2")
    pad3, e3 = measure(work, {40}, {10, 90}, "o3")
    check("1 outlier frame of 250 (below the 2 set aside): 1.500", abs(pad1 - 1.5) <= 0.003,
          "%.3f" % pad1)
    check("2 outlier frames of 250 (exactly the 2 set aside): 1.500", abs(pad2 - 1.5) <= 0.003,
          "%.3f" % pad2)
    want3 = 1.5 + (OUT_D - TRUE_D) / 2.0
    check("3 outlier frames of 250 (one more than set aside): %.3f" % want3,
          abs(pad3 - want3) <= 0.004, "%.3f" % pad3)
    pad4, e4 = measure(work, {40, 90}, set(), "o4")
    check("2 outliers, both in replica 1 of 125 frames: 1.500 (pooled)",
          abs(pad4 - 1.5) <= 0.003, "%.3f; per-replica p99 would give %.3f" % (pad4, want3))

print("\n3. the number of frames set aside, at fixed n")
code = ("import sys; sys.path.insert(0, %r); import measure_box as mb; "
        "print(' '.join(str(mb.frames_set_aside(n, p)) for n, p in "
        "((99, 99), (100, 99), (199, 99), (200, 99), (250, 99), (10005, 99), (10005, 99.9))))"
        % os.path.join(REPO, "utils"))
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=work)
check("n 99/100/199/200/250/10005 at 99, 10005 at 99.9 -> 0 1 1 2 2 100 10",
      r.stdout.split() == ["0", "1", "1", "2", "2", "100", "10"],
      (r.stdout.strip() or r.stderr.strip()[-160:]))

print("\n4. which residues the diameter counts")
# One residue per line: name and atom names. The amino acid and the six caps are
# polymer (atoms 0-15); the ligand carries atoms named N and C but no CA, and a
# calcium ion's lone atom is named CA, so neither is.
RES = [("ALA", ["N", "CA", "C", "O", "CB"]), ("ACE", ["CH3", "C", "O"]), ("NME", ["N", "CH3"]),
       ("NHE", ["N"]), ("NAC", ["N", "CA"]), ("CT3", ["N", "CH3"]), ("NH2", ["N"]),
       ("LIG", ["N", "C", "O"]), ("CA", ["CA"])]
heads = ["%5d%-5s%5s%5d" % (k + 1, res, an, 0)
         for k, (res, names) in enumerate(RES) for an in names]
code = ("import sys; sys.path.insert(0, %r); import pbc_tools as p; "
        "print(' '.join(str(i) for i in p.polymer_atoms(%r)))"
        % (os.path.join(REPO, "utils"), heads))
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=work)
check("protein and all six caps (ACE NME NHE NAC CT3 NH2): atoms 0-15",
      r.stdout.split() == [str(i) for i in range(16)],
      (r.stdout.strip() or r.stderr.strip()[-160:]))

shutil.rmtree(work, ignore_errors=True)
print("")
if failures:
    print("FAILED (%d): %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("All measure_box checks passed.")
