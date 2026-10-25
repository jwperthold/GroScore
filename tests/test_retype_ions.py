#!/usr/bin/env python3
"""The ions pdb2gmx writes get the atom types of the water model (utils/retype_ions.py).

GROMACS 2026's amber19sb.ff defines one Li/Merz ion set per water model, but its
aminoacids.rtp hard-codes the OPC-fitted types (MG Mg2+_opc, ...), so under
amber19sb_opc3 every crystal ion was typed for OPC water in OPC3 water (21 of the 43
structures of bm_fe_amber_3a that passed stage 0). Pinned, on a small fake force
field and topology:

  [1] under OPC3 every [ atoms ] type ending in _opc becomes its _opc3 twin, in
      topol.top and in every topol_*.itp it includes from its own directory, the
      B type of a perturbed atom too; nothing else in the files changes, comments
      and other sections included, and the log names each change
  [2] under OPC water, and for a topology without _opc types, nothing changes
      (and no force field needs to be found)
  [3] a twin the force field lacks: exit 1 and no file changed, the others either
  [4] the force field is found as grompp finds it: the topology's directory (a
      local amber19sb.ff copy) first, then GMXLIB, then $GMXDATA/top; not found
      with something to retype: exit 1, nothing changed
  [5] the installed GROMACS amber19sb.ff, when there is one: every _opc atom type
      has its _opc3 twin, and aminoacids.rtp still hard-codes _opc ions (why this
      exists); skipped without an installation
  [6] both runners carry the same retype_ions() and call it right after both of
      their pdb2gmx calls

Standalone, no pytest: python3 tests/test_retype_ions.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RI = os.path.join(ROOT, "utils", "retype_ions.py")
fails = 0


def check(name, ok, detail=""):
    global fails
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:300]))
    if not ok:
        fails += 1


FFNONB = """[ atomtypes ]
; name  at.num  mass  charge  ptype  sigma  epsilon
  CT        6     12.01   0.0000   A   0.339967   0.457730
  Mg2+_opc  12    24.305  0.0000   A   0.236979   0.029996
  Mg2+_opc3 12    24.305  0.0000   A   0.242859   0.044157
  Cl-_opc   17    35.45   0.0000   A   0.420465   0.418400
  Cl-_opc3  17    35.45   0.0000   A   0.410840   0.418400
  Zn2+_opc  30    65.4    0.0000   A   0.217000   0.010000
"""
FFITP = '[ defaults ]\n1 2 yes 0.5 0.8333\n\n#include "ffnonbonded.itp"\n#include "ffbonded.itp"\n'

TOP = """; topology as pdb2gmx writes it
#include "fake.ff/forcefield.itp"

#include "topol_Protein_chain_A.itp"
#include "topol_Ion_chain_A2.itp"
#include "topol_Other_chain_B.itp"
#include "fake.ff/opc3.itp"
#include "fake.ff/ions_opc3.itp"

[ system ]
test

[ molecules ]
Protein_chain_A 1
Ion_chain_A2 1
Other_chain_B 1
"""
PROT = """[ moleculetype ]
Protein_chain_A 3

[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass
     1         CT      1    ALA     CA      1     0.0337     12.01   ; qtot 0.0337
"""
ION = """; Include file for Ion_chain_A2
[ moleculetype ]
; Name            nrexcl
Ion_chain_A2        3

[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB
; residue 609 MG rtp MG q +2.0
     1   Mg2+_opc    609     MG     MG      1          2     24.305   ; qtot 2
"""
OTHER = """[ moleculetype ]
Other_chain_B 3

[ atoms ]
     1   Cl-_opc       1     CL     CL      1         -1      35.45   Cl-_opc   -1   35.45
     2   Mg2+_opc      2     MG     MG      2          2     24.305
[ bonds ]
; Mg2+_opc in a comment and in another section stays as it is
"""


def tree(d, ff_where="local", ffnonb=FFNONB, files=None):
    """A structure directory; the fake force field in d ('local'), or in d/../lib."""
    os.makedirs(d)
    ffdir = os.path.join(d if ff_where == "local" else os.path.join(d, "..", "lib"), "fake.ff")
    os.makedirs(ffdir, exist_ok=True)
    open(os.path.join(ffdir, "forcefield.itp"), "w").write(FFITP)
    open(os.path.join(ffdir, "ffnonbonded.itp"), "w").write(ffnonb)
    open(os.path.join(ffdir, "ffbonded.itp"), "w").write("[ bondtypes ]\n")
    for name, text in (files or {"topol.top": TOP, "topol_Protein_chain_A.itp": PROT,
                                 "topol_Ion_chain_A2.itp": ION, "topol_Other_chain_B.itp": OTHER}).items():
        open(os.path.join(d, name), "w").write(text)
    return os.path.normpath(os.path.join(ffdir, ".."))


def run(d, water="opc3", **env):
    e = {k: v for k, v in os.environ.items() if k not in ("GMXLIB", "GMXDATA")}
    e["PATH"] = "/usr/bin:/bin"          # no gmx: the fallback beside gmx is not taken here
    e.update(env)
    return subprocess.run([sys.executable, RI, "-t", "topol.top", "--water", water], cwd=d,
                          capture_output=True, text=True, env=e, timeout=60)


def read(d, name):
    return open(os.path.join(d, name)).read()


tmp = tempfile.mkdtemp(prefix="test_retype_ions_")

print("\n[1] OPC3: every _opc type becomes its _opc3 twin, nothing else changes")
d = os.path.join(tmp, "opc3"); tree(d)
r = run(d)
check("exit 0", r.returncode == 0, r.stderr[-300:])
ion = read(d, "topol_Ion_chain_A2.itp")
check("Ion_chain_A2: Mg2+_opc -> Mg2+_opc3, the rest of the line kept",
      ion == ION.replace("     1   Mg2+_opc    609", "     1   Mg2+_opc3    609"), ion)
oth = read(d, "topol_Other_chain_B.itp")
want = OTHER.replace("     1   Cl-_opc       1", "     1   Cl-_opc3       1").replace(
    "Cl-_opc   -1   35.45", "Cl-_opc3   -1   35.45").replace("     2   Mg2+_opc      2", "     2   Mg2+_opc3      2")
check("Other_chain_B: type and B type of Cl, the Mg; the comment in [ bonds ] untouched", oth == want, oth)
check("topol.top and the protein itp byte-identical", read(d, "topol.top") == TOP
      and read(d, "topol_Protein_chain_A.itp") == PROT)
check("the log names each change",
      "retype_ions: topol_Ion_chain_A2.itp: Mg2+_opc -> Mg2+_opc3 (1 atom)" in r.stdout
      and "retype_ions: topol_Other_chain_B.itp: Cl-_opc -> Cl-_opc3 (2 atoms), Mg2+_opc -> Mg2+_opc3 (1 atom)"
      in r.stdout, r.stdout)
r = run(d)
check("a second run changes nothing (no _opc type is left)", r.returncode == 0 and not r.stdout.strip()
      and read(d, "topol_Ion_chain_A2.itp") == ion, r.stdout)
d = os.path.join(tmp, "single"); tree(d, files={"topol.top": TOP.split("#include \"topol_")[0] + ION
                                                + "\n[ system ]\nx\n\n[ molecules ]\nIon_chain_A2 1\n"})
r = run(d)
check("a topology pdb2gmx wrote as one file: its own [ atoms ] retyped",
      r.returncode == 0 and "Mg2+_opc3    609" in read(d, "topol.top") and "Mg2+_opc " not in read(d, "topol.top"),
      read(d, "topol.top")[-300:])

print("\n[2] OPC water, or no _opc type: nothing changes")
d = os.path.join(tmp, "opc"); tree(d)
r = run(d, water="opc")
check("--water opc: exit 0, every file byte-identical", r.returncode == 0 and read(d, "topol_Ion_chain_A2.itp") == ION
      and read(d, "topol_Other_chain_B.itp") == OTHER, r.stderr[-200:])
d = os.path.join(tmp, "noion"); os.makedirs(d)
open(os.path.join(d, "topol.top"), "w").write('#include "nowhere.ff/forcefield.itp"\n#include "topol_Protein_chain_A.itp"\n')
open(os.path.join(d, "topol_Protein_chain_A.itp"), "w").write(PROT)
r = run(d)
check("no _opc type: exit 0 although the force field is nowhere to be found",
      r.returncode == 0 and read(d, "topol_Protein_chain_A.itp") == PROT, r.stderr[-200:])
d = os.path.join(tmp, "tip3p"); tree(d)
r = run(d, water="tip3p")
check("another water model gets its own twins: Mg2+_tip3p is not defined -> refused",
      r.returncode == 1 and "Mg2+_tip3p" in r.stderr and read(d, "topol_Ion_chain_A2.itp") == ION, r.stderr[-200:])

print("\n[3] a twin the force field lacks: nothing changed at all")
d = os.path.join(tmp, "notwin"); tree(d, files={"topol.top": TOP, "topol_Protein_chain_A.itp": PROT,
                                                 "topol_Ion_chain_A2.itp": ION,
                                                 "topol_Other_chain_B.itp": OTHER.replace("Cl-_opc", "Zn2+_opc")})
r = run(d)
check("exit 1, naming Zn2+_opc3", r.returncode == 1 and "Zn2+_opc3" in r.stderr, r.stderr[-300:])
check("  and the Ion itp, which could have been retyped, is unchanged too",
      read(d, "topol_Ion_chain_A2.itp") == ION, read(d, "topol_Ion_chain_A2.itp"))

print("\n[4] the force field is found as grompp finds it")
d = os.path.join(tmp, "gmxlib", "s"); lib = tree(d, ff_where="lib")
r = run(d)
check("not beside the topology: not found, exit 1, nothing changed",
      r.returncode == 1 and "not found" in r.stderr and read(d, "topol_Ion_chain_A2.itp") == ION, r.stderr[-200:])
r = run(d, GMXLIB="/nonexistent:" + lib)
check("in a GMXLIB entry: found and retyped", r.returncode == 0 and "Mg2+_opc3" in read(d, "topol_Ion_chain_A2.itp"),
      r.stderr[-200:])
d = os.path.join(tmp, "gmxdata", "s"); lib = tree(d, ff_where="lib")
data = os.path.join(tmp, "gmxdata", "data")
os.makedirs(os.path.join(data, "top"))
shutil.move(os.path.join(lib, "fake.ff"), os.path.join(data, "top", "fake.ff"))
r = run(d, GMXDATA=data)
check("in $GMXDATA/top: found and retyped", r.returncode == 0 and "Mg2+_opc3" in read(d, "topol_Ion_chain_A2.itp"),
      r.stderr[-200:])
# a local copy beside the topology (made for NCAAs or FeS) wins over GMXLIB: here the
# local copy lacks the Mg twin, the GMXLIB one has it, so the refusal shows which was read
d = os.path.join(tmp, "localwins", "s"); tree(d, ffnonb=FFNONB.replace("  Mg2+_opc3 12    24.305  0.0000   A"
                                                                          "   0.242859   0.044157\n", ""))
other = os.path.join(tmp, "localwins", "lib"); os.makedirs(os.path.join(other, "fake.ff"))
for n, t in (("forcefield.itp", FFITP), ("ffnonbonded.itp", FFNONB), ("ffbonded.itp", "")):
    open(os.path.join(other, "fake.ff", n), "w").write(t)
r = run(d, GMXLIB=other)
check("the force field beside the topology wins over GMXLIB (as in grompp)",
      r.returncode == 1 and "Mg2+_opc3" in r.stderr and os.path.join("localwins", "s", "fake.ff") in r.stderr,
      r.stderr[-300:])

print("\n[5] the installed GROMACS amber19sb.ff")
top = None
cands = []
if os.environ.get("GMXDATA"):
    cands.append(os.path.join(os.environ["GMXDATA"], "top"))
gmx = shutil.which("gmx")
if gmx:
    cands.append(os.path.join(os.path.dirname(os.path.realpath(gmx)), "..", "share", "gromacs", "top"))
for c in cands:
    if os.path.isfile(os.path.join(c, "amber19sb.ff", "ffnonbonded.itp")):
        top = os.path.join(c, "amber19sb.ff")
        break
if top is None:
    print("  (no GROMACS amber19sb.ff found: GMXDATA unset and no gmx on PATH; skipped)")
else:
    names = set()
    sec = None
    for line in open(os.path.join(top, "ffnonbonded.itp")):
        s = line.split(";")[0].strip()
        if s.startswith("["):
            sec = s.strip("[] ")
        elif sec == "atomtypes" and s:
            names.add(s.split()[0])
    opc = sorted(n for n in names if n.endswith("_opc"))
    check("every _opc atom type of amber19sb.ff has its _opc3 twin (%d of them)" % len(opc),
          opc and all(n + "3" in names for n in opc), [n for n in opc if n + "3" not in names])
    rtp = open(os.path.join(top, "aminoacids.rtp")).read()
    check("aminoacids.rtp still types MG as Mg2+_opc (the reason for retype_ions.py)",
          re.search(r"^\s*MG\s+Mg2\+_opc\s", rtp, re.M) is not None)

print("\n[6] both runners call it after every pdb2gmx")
texts = {}
for runner in ("job.run", "job_fe.run"):
    src = open(os.path.join(ROOT, runner)).read()
    m = re.search(r"^retype_ions\(\) \{\n.*?^\}\n", src, re.S | re.M)
    texts[runner] = m.group(0) if m else None
    lines = src.splitlines()
    calls = [i for i, l in enumerate(lines) if "gmx pdb2gmx" in l]
    after = []
    for i in calls:
        j = i + 1
        while j < len(lines) and (lines[j].strip().startswith("#") or "|| stage0_fail FAILED \"pdb2gmx produced" in lines[j]):
            j += 1
        after.append(lines[j].strip() if j < len(lines) else "")
    check("%s: two pdb2gmx calls, each followed by retype_ions" % runner,
          len(calls) == 2 and after == ["retype_ions", "retype_ions"], after)
check("the same retype_ions() in both runners", texts["job.run"] and texts["job.run"] == texts["job_fe.run"])
check("  passing -t topol.top --water \"$WATER_MODEL\" and stopping the structure as FAILED",
      texts["job.run"] and '--water "$WATER_MODEL"' in texts["job.run"] and "stage0_fail FAILED" in texts["job.run"])

shutil.rmtree(tmp, ignore_errors=True)
print("\n%s" % ("ALL PASS" if fails == 0 else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
