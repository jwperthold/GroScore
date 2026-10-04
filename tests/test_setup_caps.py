#!/usr/bin/env python3
"""cap_termini.py: only the cap atoms move, and the same input gives the same
output every time.

Audit SH4 (digest_audit/audit_shared.txt): the v1.4 'cap-only' minimisation held
the protein with a 1000 kJ/mol/nm^2 restraint in vacuum and copied every
minimised position back, and it started from random cap positions on whatever
platform and thread count OpenMM picked. Three runs on the identical 2FD6
fixed.pdb moved 663-892 of 5163 non-cap heavy atoms by more than 0.1 A (up to
2.8 A) relative to each other, and the cutouts built from them had 348, 348 and
355 residues. Pinned here on 13 residues in two chains cut from the 1GCQ
fixed.pdb (103 heavy atoms, no hydrogens):

  1. Two runs in separate processes, with different PYTHONHASHSEED and
     OPENMM_CPU_THREADS, write byte-identical capped PDBs and chain maps, under
     AMBER caps (ACE/NME) and under GROMOS caps (--ace-only --rename-ace-carbon).
  2. Every one of the 103 input atoms keeps its exact coordinate text. v1.4 moved
     every one of them.
  3. The structure around it: AMBER 17 residues and 113 atoms (two ACE of 3 heavy
     atoms, two NME of 2), protein B renumbered to residues 10-17; GROMOS 15
     residues and 111 atoms (two ACE whose methyl is named CA, an OXT added to each
     C-terminal residue), protein B 9-15. Each cap is bonded to its residue: the
     ACE C - N and C - NME N distances lie in 1.2-1.5 A.

Run:  python3 tests/test_setup_caps.py
"""

import os, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "utils", "cap_termini.py")
failures = []

INPUT = """\
ATOM      1  N   THR B   1      19.072   4.457  42.625  1.00  0.00           N
ATOM      2  CA  THR B   1      18.993   5.712  41.815  1.00  0.00           C
ATOM      3  C   THR B   1      17.630   6.359  42.014  1.00  0.00           C
ATOM      4  O   THR B   1      16.589   5.735  41.770  1.00  0.00           O
ATOM      5  CB  THR B   1      19.148   5.426  40.309  1.00  0.00           C
ATOM      6  OG1 THR B   1      20.114   4.379  40.118  1.00  0.00           O
ATOM      7  CG2 THR B   1      19.572   6.699  39.561  1.00  0.00           C
ATOM      8  N   TYR B   2      17.648   7.595  42.480  1.00  0.00           N
ATOM      9  CA  TYR B   2      16.428   8.331  42.708  1.00  0.00           C
ATOM     10  C   TYR B   2      16.474   9.636  41.937  1.00  0.00           C
ATOM     11  O   TYR B   2      17.538  10.230  41.760  1.00  0.00           O
ATOM     12  CB  TYR B   2      16.252   8.620  44.205  1.00  0.00           C
ATOM     13  CG  TYR B   2      15.799   7.419  44.999  1.00  0.00           C
ATOM     14  CD1 TYR B   2      16.696   6.410  45.340  1.00  0.00           C
ATOM     15  CD2 TYR B   2      14.460   7.265  45.364  1.00  0.00           C
ATOM     16  CE1 TYR B   2      16.267   5.267  46.016  1.00  0.00           C
ATOM     17  CE2 TYR B   2      14.019   6.128  46.045  1.00  0.00           C
ATOM     18  CZ  TYR B   2      14.927   5.127  46.363  1.00  0.00           C
ATOM     19  OH  TYR B   2      14.502   3.957  46.974  1.00  0.00           O
ATOM     20  N   VAL B   3      15.322  10.045  41.433  1.00  0.00           N
ATOM     21  CA  VAL B   3      15.206  11.297  40.727  1.00  0.00           C
ATOM     22  C   VAL B   3      14.105  12.048  41.452  1.00  0.00           C
ATOM     23  O   VAL B   3      13.311  11.439  42.176  1.00  0.00           O
ATOM     24  CB  VAL B   3      14.845  11.095  39.229  1.00  0.00           C
ATOM     25  CG1 VAL B   3      16.009  10.450  38.506  1.00  0.00           C
ATOM     26  CG2 VAL B   3      13.602  10.239  39.075  1.00  0.00           C
ATOM     27  N   GLN B   4      14.131  13.365  41.363  1.00  0.00           N
ATOM     28  CA  GLN B   4      13.114  14.167  41.998  1.00  0.00           C
ATOM     29  C   GLN B   4      12.245  14.803  40.918  1.00  0.00           C
ATOM     30  O   GLN B   4      12.756  15.345  39.935  1.00  0.00           O
ATOM     31  CB  GLN B   4      13.735  15.262  42.855  1.00  0.00           C
ATOM     32  CG  GLN B   4      12.695  16.100  43.559  1.00  0.00           C
ATOM     33  CD  GLN B   4      13.270  17.302  44.252  1.00  0.00           C
ATOM     34  OE1 GLN B   4      14.046  18.072  43.670  1.00  0.00           O
ATOM     35  NE2 GLN B   4      12.853  17.514  45.497  1.00  0.00           N
ATOM     36  N   ALA B   5      10.935  14.703  41.080  1.00  0.00           N
ATOM     37  CA  ALA B   5      10.010  15.313  40.130  1.00  0.00           C
ATOM     38  C   ALA B   5      10.069  16.835  40.240  1.00  0.00           C
ATOM     39  O   ALA B   5       9.974  17.399  41.344  1.00  0.00           O
ATOM     40  CB  ALA B   5       8.575  14.817  40.390  1.00  0.00           C
ATOM     41  N   LEU B   6      10.302  17.504  39.112  1.00  0.00           N
ATOM     42  CA  LEU B   6      10.350  18.962  39.071  1.00  0.00           C
ATOM     43  C   LEU B   6       8.967  19.535  38.752  1.00  0.00           C
ATOM     44  O   LEU B   6       8.728  20.731  38.899  1.00  0.00           O
ATOM     45  CB  LEU B   6      11.369  19.441  38.023  1.00  0.00           C
ATOM     46  CG  LEU B   6      12.845  19.104  38.299  1.00  0.00           C
ATOM     47  CD1 LEU B   6      13.720  19.717  37.202  1.00  0.00           C
ATOM     48  CD2 LEU B   6      13.254  19.668  39.664  1.00  0.00           C
ATOM     49  N   PHE B   7       8.064  18.661  38.315  1.00  0.00           N
ATOM     50  CA  PHE B   7       6.700  19.039  37.955  1.00  0.00           C
ATOM     51  C   PHE B   7       5.750  17.901  38.281  1.00  0.00           C
ATOM     52  O   PHE B   7       6.177  16.752  38.424  1.00  0.00           O
ATOM     53  CB  PHE B   7       6.590  19.293  36.435  1.00  0.00           C
ATOM     54  CG  PHE B   7       7.413  20.457  35.938  1.00  0.00           C
ATOM     55  CD1 PHE B   7       6.860  21.728  35.838  1.00  0.00           C
ATOM     56  CD2 PHE B   7       8.746  20.279  35.568  1.00  0.00           C
ATOM     57  CE1 PHE B   7       7.615  22.803  35.385  1.00  0.00           C
ATOM     58  CE2 PHE B   7       9.509  21.351  35.113  1.00  0.00           C
ATOM     59  CZ  PHE B   7       8.944  22.610  35.022  1.00  0.00           C
TER
ATOM     60  N   GLY C   8      27.114  19.683  28.266  1.00  0.00           N
ATOM     61  CA  GLY C   8      25.886  18.885  28.568  1.00  0.00           C
ATOM     62  C   GLY C   8      26.193  17.617  29.346  1.00  0.00           C
ATOM     63  O   GLY C   8      27.364  17.247  29.481  1.00  0.00           O
ATOM     64  N   SER C   9      25.145  16.952  29.841  1.00  0.00           N
ATOM     65  CA  SER C   9      25.260  15.707  30.605  1.00  0.00           C
ATOM     66  C   SER C   9      23.877  15.039  30.671  1.00  0.00           C
ATOM     67  O   SER C   9      22.958  15.427  29.944  1.00  0.00           O
ATOM     68  CB  SER C   9      25.790  16.003  32.008  1.00  0.00           C
ATOM     69  OG  SER C   9      24.996  16.991  32.633  1.00  0.00           O
ATOM     70  N   HIS C  10      23.702  14.057  31.545  1.00  0.00           N
ATOM     71  CA  HIS C  10      22.408  13.378  31.638  1.00  0.00           C
ATOM     72  C   HIS C  10      21.440  14.110  32.568  1.00  0.00           C
ATOM     73  O   HIS C  10      21.078  13.611  33.645  1.00  0.00           O
ATOM     74  CB  HIS C  10      22.614  11.932  32.077  1.00  0.00           C
ATOM     75  CG  HIS C  10      23.706  11.239  31.316  1.00  0.00           C
ATOM     76  ND1 HIS C  10      23.825  11.323  29.945  1.00  0.00           N
ATOM     77  CD2 HIS C  10      24.752  10.487  31.739  1.00  0.00           C
ATOM     78  CE1 HIS C  10      24.895  10.653  29.553  1.00  0.00           C
ATOM     79  NE2 HIS C  10      25.475  10.138  30.619  1.00  0.00           N
ATOM     80  N   MET C  11      21.022  15.294  32.126  1.00  0.00           N
ATOM     81  CA  MET C  11      20.118  16.160  32.875  1.00  0.00           C
ATOM     82  C   MET C  11      18.849  16.436  32.071  1.00  0.00           C
ATOM     83  O   MET C  11      18.825  16.252  30.849  1.00  0.00           O
ATOM     84  CB  MET C  11      20.828  17.480  33.163  1.00  0.00           C
ATOM     85  CG  MET C  11      21.922  17.373  34.224  1.00  0.00           C
ATOM     86  SD  MET C  11      21.203  17.124  35.856  1.00  0.00           S
ATOM     87  CE  MET C  11      20.656  18.760  36.256  1.00  0.00           C
ATOM     88  N   PRO C  12      17.769  16.862  32.742  1.00  0.00           N
ATOM     89  CA  PRO C  12      16.517  17.154  32.040  1.00  0.00           C
ATOM     90  C   PRO C  12      16.572  18.431  31.193  1.00  0.00           C
ATOM     91  O   PRO C  12      17.300  19.381  31.505  1.00  0.00           O
ATOM     92  CB  PRO C  12      15.523  17.309  33.188  1.00  0.00           C
ATOM     93  CG  PRO C  12      16.377  17.884  34.258  1.00  0.00           C
ATOM     94  CD  PRO C  12      17.580  16.997  34.195  1.00  0.00           C
ATOM     95  N   LYS C  13      15.803  18.427  30.105  1.00  0.00           N
ATOM     96  CA  LYS C  13      15.696  19.572  29.217  1.00  0.00           C
ATOM     97  C   LYS C  13      14.302  20.183  29.370  1.00  0.00           C
ATOM     98  O   LYS C  13      13.306  19.456  29.467  1.00  0.00           O
ATOM     99  CB  LYS C  13      15.899  19.144  27.773  1.00  0.00           C
ATOM    100  CG  LYS C  13      17.291  18.662  27.481  1.00  0.00           C
ATOM    101  CD  LYS C  13      17.409  18.338  26.011  1.00  0.00           C
ATOM    102  CE  LYS C  13      18.689  17.581  25.710  1.00  0.00           C
ATOM    103  NZ  LYS C  13      19.353  18.216  24.529  1.00  0.00           N
TER
END
"""


def check(name, ok, detail=""):
    detail = " | ".join(l.strip() for l in str(detail).splitlines() if l.strip())
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


def run(flags, env_extra):
    """One cap_termini.py run in a fresh directory: (rc, capped text, chain_map text, log)."""
    d = tempfile.mkdtemp(prefix="caps_")
    try:
        open(os.path.join(d, "input.pdb"), "w").write(INPUT)
        open(os.path.join(d, "chain_map.gs"), "w").write(
            "# Sequential residue numbers belonging to protein B\n# Generated from input.pdb for chain(s): C\n"
            + "".join("%d\n" % r for r in range(8, 14)))
        env = dict(os.environ, **env_extra)
        p = subprocess.run([sys.executable, SCRIPT, "-f", "input.pdb", "-o", "capped.pdb", "-m", "chain_map.gs"] + flags,
                           cwd=d, capture_output=True, text=True, env=env)
        out = open(os.path.join(d, "capped.pdb")).read() if os.path.isfile(os.path.join(d, "capped.pdb")) else ""
        return p.returncode, out, open(os.path.join(d, "chain_map.gs")).read(), p.stdout + p.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)


def residues(text):
    """[(resname, resnum, [(atom, 'x y z text')])] in file order."""
    out, key = [], None
    for l in text.splitlines():
        if l.startswith("ATOM"):
            k = (l[17:20], l[22:26])
            if k != key:
                key = k
                out.append((l[17:20].strip(), int(l[22:26]), []))
            out[-1][2].append((l[12:16].strip(), l[30:54]))
    return out


def xyz(s):
    return np.array([float(s[0:8]), float(s[8:16]), float(s[16:24])])


def main():
    if not os.path.isfile(SCRIPT):
        print("missing %s" % SCRIPT)
        sys.exit(1)
    inp = [r for r in residues(INPUT)]
    for tag, flags, nres, natoms, bres, methyl in (
            ("amber", [], 17, 113, list(range(10, 18)), "CH3"),
            ("gromos", ["--ace-only", "--rename-ace-carbon"], 15, 111, list(range(9, 16)), "CA")):
        print("[%s caps]" % tag)
        rc1, out1, cm1, log1 = run(flags, {"PYTHONHASHSEED": "1", "OPENMM_CPU_THREADS": "8"})
        rc2, out2, cm2, log2 = run(flags, {"PYTHONHASHSEED": "2"})
        check("[1] %s: both runs rc 0" % tag, rc1 == 0 and rc2 == 0, (log1 + log2)[-300:] if (rc1 or rc2) else "")
        check("[1] %s: byte-identical capped PDB (hash seed 1 / 2, threads 8 / all)" % tag,
              out1 == out2 and out1 != "")
        check("[1] %s: identical chain_map.gs" % tag, cm1 == cm2)
        res = residues(out1)
        names = [r[0] for r in res]
        check("[3] %s: %d residues, %d atoms" % (tag, nres, natoms),
              len(res) == nres and sum(len(r[2]) for r in res) == natoms,
              "%d residues, %d atoms" % (len(res), sum(len(r[2]) for r in res)))
        bmap = [int(l) for l in cm1.splitlines() if l.strip() and not l.startswith("#")]
        check("[3] %s: protein B residues %d-%d" % (tag, bres[0], bres[-1]), bmap == bres, str(bmap))
        # [2] the 103 input atoms, residue by residue (caps skipped), exact text
        body = [r for r in res if r[0] not in ("ACE", "NME")]
        same = moved = missing = 0
        worst = 0.0
        for (rn_in, _, a_in), (rn_out, _, a_out) in zip(inp, body):
            dout = dict(a_out)
            for name, s in a_in:
                if name not in dout:
                    missing += 1
                elif dout[name] == s:
                    same += 1
                else:
                    moved += 1
                    worst = max(worst, float(np.linalg.norm(xyz(dout[name]) - xyz(s))))
        check("[2] %s: all 103 input atoms keep their coordinate text" % tag,
              same == 103 and moved == 0 and missing == 0 and len(body) == 13,
              "same %d, moved %d (max %.3f A), missing %d" % (same, moved, worst, missing))
        # [3] cap identity and bonds
        aces = [r for r in res if r[0] == "ACE"]
        check("[3] %s: ACE methyl named %s" % (tag, methyl), len(aces) == 2 and all(methyl in dict(r[2]) for r in aces))
        bonds = []
        for i, r in enumerate(res):
            if r[0] == "ACE" and i + 1 < len(res):
                bonds.append(np.linalg.norm(xyz(dict(r[2])["C"]) - xyz(dict(res[i + 1][2])["N"])))
            if r[0] == "NME" and i > 0:
                bonds.append(np.linalg.norm(xyz(dict(r[2])["N"]) - xyz(dict(res[i - 1][2])["C"])))
        want_b = 4 if tag == "amber" else 2
        check("[3] %s: %d cap bonds, all 1.2-1.5 A" % (tag, want_b),
              len(bonds) == want_b and all(1.2 <= b <= 1.5 for b in bonds), " ".join("%.3f" % b for b in bonds))
        if tag == "gromos":
            oxt = [r[1] for r in res if "OXT" in dict(r[2])]
            check("[3] gromos: OXT added to the two C-terminal residues (8, 15)", oxt == [8, 15], str(oxt))

    print()
    if failures:
        print("%d check(s) FAILED:" % len(failures))
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
