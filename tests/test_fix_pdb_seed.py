#!/usr/bin/env python3
"""fix_pdb.py gives the same bytes on every run of one structure.

PDBFixer's addMissingAtoms() places the atoms it adds from templates and minimises
them; where an added atom still sits closer than 0.13 nm to another one it runs
Langevin dynamics, and its integrator took a random seed unless it was given one.
fix_pdb.py gave none, so two setups of one structure differed wherever side-chain
atoms were missing: on 1E6E's renumbered.pdb in 14 atom lines (the side chains of
GLU A 1 and ASP B 458, 492 and 562, the OXT of SER B 570; integrator report of the
glycan round), and with them the caps, the cutout and every result after them. A
seed alone was not enough: OpenMM ran PDBFixer on CUDA where present, and two of the
46 HADDOCK inputs (2HLE, 1Z0K) still differed between seeded runs; three seeded runs
of 2HLE gave three outputs. With the seed, the CPU platform and one thread, as
cap_termini.py runs, every run gives the same bytes. Pinned on an extract of the 1E6E
file, on which four unseeded runs gave four different outputs, and on the whole 2HLE
file (tests/data/fix_pdb, PDB data CC0):

  [1] two runs of fix_pdb.py, each in a directory of its own, write byte-identical
      fixed.pdb files, for 1E6E and for 2HLE;
  [2] and the atoms were really added: GLU A 1 of 1E6E has CB, CG, CD, OE1 and OE2,
      ASP B 458 its CG, OD1 and OD2, SER B 570 an OXT.

Run:  python3 tests/test_fix_pdb_seed.py   (about 40 s)
"""
import gzip, os, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "utils", "fix_pdb.py")
DATA = os.path.join(REPO, "tests", "data", "fix_pdb")
failures = []


def check(name, ok, detail=""):
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:300]))
    if not ok:
        failures.append(name)


def atoms_of(text, chain, num):
    return {l[12:16].strip() for l in text.splitlines()
            if l.startswith("ATOM") and l[21] == chain and l[22:26].strip() == str(num)}


def twice(tag, text):
    """fix_pdb.py run twice on `text`, each in a directory of its own: [bytes or None]."""
    outs = []
    for run in ("a", "b"):
        d = os.path.join(tmp, tag + run)
        os.makedirs(d)
        open(os.path.join(d, "renumbered.pdb"), "w").write(text)
        p = subprocess.run([sys.executable, FIX, "-f", "renumbered.pdb", "-o", "fixed.pdb"], cwd=d,
                           capture_output=True, text=True, timeout=900)
        path = os.path.join(d, "fixed.pdb")
        outs.append(open(path, "rb").read() if p.returncode == 0 and os.path.isfile(path) else None)
        check("%s run %s: fix_pdb.py exits 0 and writes fixed.pdb" % (tag, run), outs[-1] is not None,
              (p.returncode, p.stderr[-300:]))
    return outs


def same(tag, outs):
    differ = []
    if None not in outs:
        a, b = outs[0].decode().splitlines(), outs[1].decode().splitlines()
        differ = [x for x, y in zip(a, b) if x != y] + (["(lengths %d, %d)" % (len(a), len(b))] if len(a) != len(b) else [])
    check("%s: two runs give byte-identical fixed.pdb files" % tag, None not in outs and outs[0] == outs[1],
          "%d differing lines, first %s" % (len(differ), differ[:1]))


tmp = tempfile.mkdtemp(prefix="fix_pdb_seed_")
try:
    e6e = twice("1E6E", open(os.path.join(DATA, "1E6E_missing_atoms.pdb")).read())
    hle = twice("2HLE", gzip.open(os.path.join(DATA, "2HLE_renumbered.pdb.gz"), "rt").read())
    print("\n[1] the same bytes on every run")
    same("1E6E extract (the seed)", e6e)
    same("2HLE (the seed, the CPU platform, one thread)", hle)
    print("\n[2] the missing atoms are there")
    text = e6e[0].decode() if e6e[0] else ""
    check("GLU A 1: CB CG CD OE1 OE2", {"CB", "CG", "CD", "OE1", "OE2"} <= atoms_of(text, "A", 1),
          sorted(atoms_of(text, "A", 1)))
    check("ASP B 458: CG OD1 OD2", {"CG", "OD1", "OD2"} <= atoms_of(text, "B", 458), sorted(atoms_of(text, "B", 458)))
    check("SER B 570: OXT", "OXT" in atoms_of(text, "B", 570), sorted(atoms_of(text, "B", 570)))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all fix_pdb.py seed checks passed")
