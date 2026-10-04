#!/usr/bin/env python3
"""fix_ion_protonation.py knows every force field the runners pass it.

The runners call it with --ff $FF_BASE: amber19sb_opc3, amber19sb_opc, charmm36 or
gromos54a8 (gromos54a8_rf arrives with its _rf removed). It stripped '_opc' before
'_opc3', so amber19sb_opc3 became 'amber19sb3', and every AMBER19SB/OPC3 setup
logged "unknown force field 'amber19sb3', using AMBER naming" twice before falling
back to the right names (v15_work review E2). Pinned on a cysteine whose SG sits
0.23 nm from a Zn2+: no warning, and the thiolate name of each force field.

Run:  python3 tests/test_fix_ion_protonation_ff.py
"""
import os, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "utils", "fix_ion_protonation.py")
failures = []


def check(name, ok, detail=""):
    print("  %-66s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:300]))
    if not ok:
        failures.append(name)


PDB = ("ATOM      1  N   CYS A   1      10.000  10.000  10.000  1.00  0.00           N\n"
       "ATOM      2  CA  CYS A   1      11.450  10.000  10.000  1.00  0.00           C\n"
       "ATOM      3  SG  CYS A   1      12.000  11.500  10.000  1.00  0.00           S\n"
       "HETATM    4 ZN    ZN A   2      12.000  13.800  10.000  1.00  0.00          ZN\n"
       "END\n")
d = tempfile.mkdtemp(prefix="ionprot_")
try:
    for ff, want in (("amber19sb_opc3", "CYM"), ("amber19sb_opc", "CYM"), ("charmm36", "CYM"),
                     ("gromos54a8", "CYS1")):
        open(os.path.join(d, "x.pdb"), "w").write(PDB)
        open(os.path.join(d, "ion_residues.gs"), "w").write("2\n")
        p = subprocess.run([sys.executable, SCRIPT, "-f", "x.pdb", "-m", "ion_residues.gs", "--ff", ff],
                           cwd=d, capture_output=True, text=True)
        got = {l[17:21].strip() for l in open(os.path.join(d, "x.pdb")) if l.startswith("ATOM")}
        check("--ff %s: no 'unknown force field', cysteine renamed %s" % (ff, want),
              p.returncode == 0 and "unknown force field" not in p.stdout + p.stderr and got == {want},
              (p.returncode, (p.stdout + p.stderr)[-200:], got))
finally:
    shutil.rmtree(d, ignore_errors=True)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all force-field naming checks passed")
