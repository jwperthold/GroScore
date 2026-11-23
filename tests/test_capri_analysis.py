#!/usr/bin/env python3
"""capri/analyze_round.py: i-RMSD, FCC and the clustering on synthetic inputs.

  1. kabsch_rmsd is 0 for a rotated and translated copy, and moving half the
     atoms by 1 A gives less than the unfitted 0.71 A (the fit absorbs part).
  2. fcc_matrix is |C_i & C_j| / |C_i|, asymmetric, and 0 for a model without
     contacts.
  3. cluster_fcc: with neighbours needing FCC(i,j) >= cutoff and FCC(j,i) >=
     cutoff*strictness, the model with most free neighbours becomes a centre;
     a group smaller than min_size stays unclustered.
  4. end to end on a run directory of 6 two-chain models (4 poses within 0.3 A
     of each other, 2 far away; 5 scored): pKd = -0.0176 * score + 3.4513, the
     best-scoring model is the reference with i-RMSD 0, the 4 close poses form
     cluster 1 with i-RMSD < 0.5 A, the far ones have i-RMSD > 5 A, a BROKEN
     status gives no pKd, and both plots are written.

Run with the groscore environment:  python tests/test_capri_analysis.py
"""
import importlib.util, math, os, subprocess, sys, tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "capri", "analyze_round.py")
spec = importlib.util.spec_from_file_location("analyze_round", SCRIPT)
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)
failures = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# 1. Kabsch
rng = np.random.default_rng(1)
p = rng.normal(size=(30, 3)) * 5
th = 0.7
rot = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
check(ar.kabsch_rmsd(p, p @ rot.T + [3, -2, 8]) < 1e-9, "1: rotated + translated copy has RMSD 0")
q = p.copy()
q[:15] += [1.0, 0, 0]
check(0.3 < ar.kabsch_rmsd(p, q) < 0.5, "1: half the atoms moved 1 A gives RMSD below the unfitted 0.71")

# 2. FCC
C = [{1, 2, 3, 4}, {1, 2}, set()]
f = ar.fcc_matrix(C)
check(abs(f[0, 1] - 0.5) < 1e-12 and abs(f[1, 0] - 1.0) < 1e-12, "2: FCC is asymmetric, |Ci&Cj|/|Ci|")
check(f[2].sum() == 0 and f[:, 2].sum() == 0, "2: a model without contacts shares nothing")

# 3. clustering: models 0-4 share a core, 5-6 another pair, 7 alone
core = set(range(10))
C = [core, core | {10}, core | {11}, core | {12}, core - {0}, {50, 51, 52}, {50, 51, 52, 53}, {90}]
cl = ar.cluster_fcc(ar.fcc_matrix(C), 0.6, 0.75, 4)
check(len(cl) == 1 and sorted(cl[0]) == [0, 1, 2, 3, 4], "3: the 5-model group is one cluster, the pair stays out")
cl = ar.cluster_fcc(ar.fcc_matrix(C), 0.6, 0.75, 2)
check(len(cl) == 2 and sorted(cl[1]) == [5, 6], "3: with min size 2 the pair forms cluster 2")


# 4. end to end
def chain_atoms(chain, centre, n):
    out = []
    for r in range(n):
        for k, name in enumerate(("N", "CA", "C", "O", "CB")):
            out.append((chain, r + 1, name, centre + np.array([r * 3.8, k * 0.9, (r % 2) * 1.5])))
    return out


def write_model(path, shift):
    atoms = chain_atoms("A", np.zeros(3), 8) + chain_atoms("B", np.array([0.0, 5.5, 0.0]) + shift, 8)
    with open(path, "w") as o:
        for i, (ch, r, name, x) in enumerate(atoms, 1):
            o.write("ATOM  %5d  %-3s ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s\n"
                    % (i, name, ch, r, x[0], x[1], x[2], name[0]))
        o.write("END\n")


with tempfile.TemporaryDirectory() as tmp:
    run = os.path.join(tmp, "T999")
    shifts = [(0, 0, 0), (0.2, 0, 0), (0, 0.2, 0), (0, 0, 0.3), (14, 25, 0), (-20, 20, 5)]
    scores = ["-200.0", "-150.0", "-100.0", "nan", "-50.0", "BROKEN"]
    os.makedirs(run)
    with open(os.path.join(run, "sp.gs"), "w") as sp, open(os.path.join(run, "scores_avg.gs"), "w") as sc:
        sc.write("# Structure_ID  Score  CI95  Cycles_Used  RMSD_mean_A  RMSD_max_A  RMSD_flag\n")
        for i, (sh, s) in enumerate(zip(shifts, scores)):
            sid = "T999_%04d" % i
            os.makedirs(os.path.join(run, sid))
            write_model(os.path.join(run, sid, "input.pdb"), np.array(sh, dtype=float))
            sp.write("%s  B\n" % sid)
            sc.write("%s\t%s\tnan\t1\t2.0\t2.0\tOK\n" % (sid, s))
    p = subprocess.run([sys.executable, SCRIPT, run, "--min-scored", "2", "--min-size", "4"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    check(p.returncode == 0, "4: runs (exit %d)%s" % (p.returncode, "" if p.returncode == 0 else "\n" + p.stdout))
    rows = {}
    if p.returncode == 0:
        for line in open(os.path.join(run, "analysis", "models.gs")):
            if not line.startswith("#"):
                a = line.split()
                rows[a[0]] = a
    if rows:
        check(abs(float(rows["T999_0000"][6]) - (-0.0176 * -200 + 3.4513)) < 0.006, "4: pKd from the calibration")
        check(float(rows["T999_0000"][7]) < 1e-6, "4: best-scoring model is the reference (i-RMSD 0)")
        check(all(rows["T999_%04d" % i][9] == "1" and float(rows["T999_%04d" % i][7]) < 0.5 for i in range(4)),
              "4: the 4 close poses form cluster 1, i-RMSD < 0.5 A")
        check(all(float(rows["T999_%04d" % i][7]) > 5 for i in (4, 5)), "4: far poses have i-RMSD > 5 A")
        check(rows["T999_0005"][6] == "nan" and rows["T999_0005"][3] == "BROKEN", "4: BROKEN has no pKd")
        check(all(os.path.isfile(os.path.join(run, "analysis", f)) for f in ("irmsd_vs_pkd.png", "clusters_pkd.png")),
              "4: both plots written")

print("\n%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
