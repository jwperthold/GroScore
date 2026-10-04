#!/usr/bin/env python3
"""merge_crystal_waters.py writes each crystal water with as many atoms as the SOL
topology has, keeps the waters near any solute atom, and numbers them after the
last residue.

Audit (digest_audit/audit_restraints.txt, verifier extra finding): GROMOS 54A8's
spc.itp carries its [ atoms ] block twice, in #ifndef HEAVY_H ... #else ... #endif.
The reader ignored the preprocessor, took six site names, wrote every crystal water
with 6 atoms against 3 in the topology, and grompp stopped every GROMOS 54A8 setup
with crystal waters at stage 0 (1A2K 4626 vs 4374 atoms, 7CEI 2930 vs 2567; 40 of
46 benchmark cutouts keep crystal waters). Pinned here, with two waters each:

  1. The repository's gromos54a8.ff/spc.itp (included by absolute path, as pdb2gmx
     writes a GMXLIB force field): 6 new atoms OW HW1 HW2 OW HW1 HW2, SOL 2. v1.4: 12.
  2. Nested #ifdef / #ifndef / #else / #endif with a #define inside the file keep
     exactly the default branch: OW HW1 HW2.
  3. A bare include 'testff.ff/opc4.itp' (as pdb2gmx writes a force field from the
     GROMACS library) is found through GMXLIB: 4 sites, the M site 0.01594 nm from
     O, as OPC's geometry gives. v1.4 looked in the topology directory only and fell
     back to 3 sites, the wrong count for any 4-site water.
  4. --proximity, 0.5 nm, measured from every solute atom read by column: a water
     0.30 nm from the S of a ligand written with a right-justified residue name
     (merge_ligand.py's old format, which the split parse read with x = atom
     number) is kept (v1.4 dropped it); waters 0.499 and exactly 0.500 nm from the
     nearest solute atom are kept, 0.501 nm is removed.
  5. The waters are numbered after the last residue, 368 named 1PE: 369, 370.
     The regex read '  3681PE' as residue 3681 and started at 3682.

Run:  python3 tests/test_setup_waters.py
"""

import os, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "utils", "merge_crystal_waters.py")
SPC = os.path.join(REPO, "forcefield", "gromos54a8.ff", "spc.itp")
failures = []

NESTED_ITP = """\
[ moleculetype ]
SOL 2
[ atoms ]
#define USE_THREE
#ifdef USE_THREE
     1     OW   1    SOL     OW      1      -0.82   15.99940
#ifdef NOT_DEFINED
     2      H   1    SOL     HX      1       0.41    1.00800
#else
     2      H   1    SOL    HW1      1       0.41    1.00800
#endif
#else
     1     OW   1    SOL     OX      1      -0.82   15.99940
#endif
#ifndef USE_THREE
     3      H   1    SOL     HY      1       0.41    1.00800
#else
     3      H   1    SOL    HW2      1       0.41    1.00800
#endif
#ifndef FLEXIBLE
[ settles ]
1 1 0.1 0.16330
#endif
"""

OPC4_ITP = """\
[ moleculetype ]
SOL 2
[ atoms ]
  1   OW_opc      1       SOL       OW       1       0        16.00000
  2   HW_opc      1       SOL       HW1      1       0.679142  1.00800
  3   HW_opc      1       SOL       HW2      1       0.679142  1.00800
  4   MW          1       SOL       MW       1      -1.358284  0.00000
#ifndef FLEXIBLE
[ settles ]
1	1	0.08724331	0.13712051
#endif
[ virtual_sites3 ]
4       1       2       3       1       0.14772234      0.14772234
"""


def check(name, ok, detail=""):
    detail = " | ".join(l.strip() for l in str(detail).splitlines() if l.strip())
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


def gro_line(resnum, resname, atom, num, x, y, z, right=False):
    rn = ("%5s" % resname) if right else ("%-5s" % resname)
    return "%5d%s%5s%5d%8.3f%8.3f%8.3f\n" % (resnum, rn, atom, num, x, y, z)


def setup(d, include, waters_A, atoms, env=None, proximity=False):
    """Write conf.gro (atoms: (resnum, resname, atom, x, y, z, right_justified)),
    topol.top (water itp #included as given), crystal_waters.pdb (oxygens in A)
    and run the script. Returns (rc, log, new atom lines, SOL line of topol.top)."""
    os.makedirs(d)
    with open(os.path.join(d, "conf.gro"), "w") as f:
        f.write("test\n%5d\n" % len(atoms))
        for i, (rn, res, an, x, y, z, right) in enumerate(atoms):
            f.write(gro_line(rn, res, an, i + 1, x, y, z, right))
        f.write("   5.00000   5.00000   5.00000\n")
    with open(os.path.join(d, "topol.top"), "w") as f:
        f.write("; test topology\n\n; Include water topology\n#include \"%s\"\n\n[ system ]\ntest\n\n"
                "[ molecules ]\nProtein_chain_A     1\n" % include)
    with open(os.path.join(d, "crystal_waters.pdb"), "w") as f:
        for i, (x, y, z) in enumerate(waters_A):
            f.write("HETATM%5d  O   HOH W%4d    %8.3f%8.3f%8.3f  1.00 20.00           O\n" % (i + 1, i + 1, x, y, z))
        f.write("END\n")
    args = [sys.executable, SCRIPT, "-t", "topol.top", "-c", "conf.gro"] + (["--proximity"] if proximity else [])
    p = subprocess.run(args, cwd=d, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    lines = open(os.path.join(d, "conf.gro")).read().splitlines()
    new = lines[2 + len(atoms):-1]
    sol = [l for l in open(os.path.join(d, "topol.top")) if l.startswith("SOL")]
    return p.returncode, p.stdout + p.stderr, new, (sol[0].split() if sol else [])


PROTEIN = [(1, "ALA", "N", 1.000, 1.000, 1.000, False), (1, "ALA", "CA", 1.100, 1.000, 1.000, False)]


def main():
    for f in (SCRIPT, SPC):
        if not os.path.isfile(f):
            print("missing %s" % f)
            sys.exit(1)
    tmp = tempfile.mkdtemp(prefix="waters_")
    try:
        two = [(15.0, 30.0, 10.0), (15.0, 10.0, 30.0)]

        print("[1] GROMOS 54A8 spc.itp, two crystal waters")
        rc, log, new, sol = setup(os.path.join(tmp, "gromos"), SPC, two, PROTEIN)
        names = [l[10:15].strip() for l in new]
        check("[1] rc 0, 6 new atoms OW HW1 HW2 OW HW1 HW2",
              rc == 0 and names == ["OW", "HW1", "HW2"] * 2, "rc %d %s" % (rc, names))
        check("[1] topol.top: SOL 2", sol == ["SOL", "2"], str(sol))

        print("[2] nested #ifdef / #ifndef / #else / #endif with #define")
        nested = os.path.join(tmp, "nested.itp")
        open(nested, "w").write(NESTED_ITP)
        rc, log, new, sol = setup(os.path.join(tmp, "nested"), nested, two, PROTEIN)
        names = [l[10:15].strip() for l in new]
        check("[2] the default branch only: OW HW1 HW2 per water",
              rc == 0 and names == ["OW", "HW1", "HW2"] * 2, "rc %d %s" % (rc, names))

        print("[3] bare include resolved through GMXLIB, 4-site water")
        lib = os.path.join(tmp, "gmxlib")
        os.makedirs(os.path.join(lib, "testff.ff"))
        open(os.path.join(lib, "testff.ff", "opc4.itp"), "w").write(OPC4_ITP)
        rc, log, new, sol = setup(os.path.join(tmp, "opc4"), "testff.ff/opc4.itp", two, PROTEIN,
                                  env={"GMXLIB": lib, "GMXDATA": os.path.join(tmp, "nowhere")})
        names = [l[10:15].strip() for l in new]
        check("[3] 8 new atoms OW HW1 HW2 MW x 2", rc == 0 and names == ["OW", "HW1", "HW2", "MW"] * 2,
              "rc %d %s %s" % (rc, names, log[-160:]))
        if len(new) >= 4:
            o = [float(new[0][20 + 8 * k:28 + 8 * k]) for k in range(3)]
            m = [float(new[3][20 + 8 * k:28 + 8 * k]) for k in range(3)]
            dom = sum((a - b) ** 2 for a, b in zip(o, m)) ** 0.5
            check("[3] M site 0.016 nm from O (OPC: 0.01594, written %8.3f)", abs(dom - 0.016) < 1e-9, "%.4f" % dom)

        print("[4] --proximity 0.5 nm, by column, from every solute atom")
        # The sulfate is atoms 3 and 4 at x 2.0 / 2.1 nm: a split parse of its
        # right-justified lines reads x = 3 and 4, a nanometre off.
        solute = PROTEIN + [(2, "SO4", "S", 2.000, 3.000, 3.000, True), (2, "SO4", "O1", 2.100, 3.100, 3.100, True)]
        waters = [(20.0, 30.0, 27.0),     # 0.30 nm below the sulfate S, 0.42 from O1
                  (10.0, 10.0, 5.01),     # 0.499 nm below N (1, 1, 1); CA is farther
                  (10.0, 10.0, 5.00),     # 0.500 nm
                  (10.0, 10.0, 4.99)]     # 0.501 nm
        rc, log, new, sol = setup(os.path.join(tmp, "prox"), SPC, waters, solute, proximity=True)
        kept_z = [round(float(l[36:44]), 3) for l in new if l[10:15].strip() == "OW"]
        check("[4] water 0.30 nm from the right-justified SO4 kept", 2.7 in kept_z, "OW z %s" % kept_z)
        check("[4] waters 0.499 and 0.500 nm from N kept, 0.501 removed",
              sorted(z for z in kept_z if z < 1.0) == [0.5, 0.501], "OW z %s" % kept_z)
        check("[4] topol.top: SOL 3", sol == ["SOL", "3"], str(sol))

        print("[5] water residue numbers after residue 368 named 1PE")
        solute = PROTEIN + [(368, "1PE", "C1", 3.000, 3.000, 3.000, False)]
        rc, log, new, sol = setup(os.path.join(tmp, "digit"), SPC, two, solute)
        nums = sorted({int(l[0:5]) for l in new})
        check("[5] waters numbered 369, 370", nums == [369, 370], str(nums))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("%d check(s) FAILED:" % len(failures))
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
