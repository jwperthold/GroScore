#!/usr/bin/env python3
"""No ligand is dropped or mangled silently, and the clear-cut chemistry is right.

On the 46-structure HADDOCK FE pass of 2026-09 the ligand stage lost six ligand
instances in five structures without a word that reached any status file (1KXP
ATP and 1HE8 GNP to sqm, the AlF3 of 1WQ1 and 1GRN to a missing Al-F bond term,
the two iodides of 1J2J built as H-I), built ten sulfates as neutral H2SO4 and the
2MTA phosphate as HP(OH)2, perceived two glycol additives of 2FD6 as enol ethers,
and gave 2FD6's chain-H PGE to protein B. The AlF3 alone cost 1GRN +5.0 and 1WQ1
+4.8 pKD. This pins, with fixed coordinates taken from those entries:

  [1] sulfate is SO4 2- (5 atoms, charges summing to -2), not H2SO4
  [2] inorganic phosphate is HPO4 2-, also when the entry holds only P and two O
  [3] a ligand that cannot be built leaves 'RESNAME CHAIN RESNUM reason' in
      ligand_failed.gs, also when the process is killed mid-run
  [4] merge_ligand.py writes every extracted ligand that misses topol.top into
      ligand_failed.gs, takes each ligand's chain from the ligand itself, flags a
      ligand covalently bonded to the protein, and follows --ligand-param (contract
      F4): off-user lists the ligands and writes nothing, off-ff and a missing option
      without a manifest write the off-ff line (fail closed); a bonded sugar that
      fuse_glycans.py left as a ligand gets its reason from glycan_unfused.gs
      ('glycan not fused: ...', also when that reason holds a ';', and only once on
      a repeated merge), and so does such a sugar in the region without a bond
      (the rest of a glycan linked beyond the covalent rule) or without parameters,
      after parametrize_ligand.py's reason (fusion review F1, F6), while a line
      makes no failure of a ligand outside the region; a ligand bonded to a sugar
      of a fused glycan, an ATOM record in fused.pdb, is bonded to the protein
  [5] the composition, order and position checks catch a wrong molecule, and the
      2FD6 ETX that OpenBabel builds as an enol ether is built from its CCD entry in
      the repository cache (no network), or flagged when that entry is unavailable
  [6] a lone iodide is the force field's I- ion: no '#include "-"', its .gro at the
      PDB coordinates / 10, and a topology without the ion fails
  [7] the pH 7.4 acid rules on the classes the first round fixed
  [8] every bond-order threshold just below and just above (and at it where the
      number is exact): C=C 1.50, C=N 1.44, C=O 1.36 A (for a carboxylate the shorter
      C-O), sp3-sp3 C-C 1.43 and C-N 1.38 A, sp3 out-of-plane 0.30 A, sp2 0.25 A,
      hydrogen only on C, N, O, S, Se; oxyanion completion only for sulfate/phosphate
      residue names, P-O 1.35-1.75 A, never from three coplanar oxygens; merge's
      proximity 0.50 nm (0.61 nm unparametrised) and covalent C-N 1.82 A
The AM1-BCC retry, the CCD path, the further chemistry and the merge details have
their own files: test_ligand_am1bcc.py, test_ligand_ccd.py, test_ligand_chemistry.py,
test_ligand_merge.py.

[1], [2], [3], [5] and [6] run parametrize_ligand.py (OpenFF and AM1-BCC on
molecules of at most 16 atoms; CCD entries from forcefield/ccd_cache); the whole
file takes about 15-20 s.

Standalone, no pytest: python3 tests/test_ligand_failures.py
"""
import contextlib
import importlib.util
import io
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL = os.path.join(ROOT, "utils", "parametrize_ligand.py")
ML = os.path.join(ROOT, "utils", "merge_ligand.py")
fails = 0


def check(name, ok, detail=""):
    global fails
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % detail))
    if not ok:
        fails += 1


def hetatm(serial, name, resname, chain, resnum, xyz, el):
    nm = (" " + name) if len(name) < 4 else name
    return "HETATM%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00 20.00          %2s" % (
        serial, nm, resname, chain, resnum, xyz[0], xyz[1], xyz[2], el)


def pdb(resname, chain, resnum, atoms):
    return "\n".join(hetatm(i + 1, n, resname, chain, resnum, xyz, el) for i, (n, el, xyz) in enumerate(atoms)) + "\nEND\n"


# Deposited coordinates (A), copied from the benchmark inputs.
SO4_1S1Q = [("S", "S", (103.766, 32.676, 62.630)), ("O1", "O", (104.417, 31.475, 62.123)),
            ("O2", "O", (102.792, 33.195, 61.670)), ("O3", "O", (103.039, 32.358, 63.869)),
            ("O4", "O", (104.802, 33.686, 62.874))]
PO4_2MTA = [("P", "P", (31.448, 0.000, 0.000)), ("O2", "O", (32.320, -1.016, 0.718)),
            ("O3", "O", (30.568, 0.718, 1.016))]
# the same phosphate with its two crystallographic two-fold mates (x, -y, -z) written out
PO4_FULL = PO4_2MTA + [("O1", "O", (32.320, 1.016, -0.718)), ("O4", "O", (30.568, -0.718, -1.016))]
AF3_1WQ1 = [("AL", "AL", (-18.354, 18.161, 18.004)), ("F1", "F", (-17.560, 17.298, 19.172)),
            ("F2", "F", (-17.467, 19.109, 16.915)), ("F3", "F", (-20.003, 18.082, 17.909))]
ETX_2FD6_H = [("C2", "C", (7.515, -24.658, 15.421)), ("O2", "O", (7.060, -25.934, 15.837)),
              ("C3", "C", (5.826, -26.588, 15.600)), ("C4", "C", (6.000, -28.153, 15.380)),
              ("O1", "O", (8.397, -23.205, 13.458)), ("C1", "C", (8.150, -24.560, 13.986))]
IOD_1J2J = [("I", "I", (16.778, 12.078, 24.003))]


def run_param(d, fname, resname, chain, inst=0, extra=()):
    r = subprocess.run([sys.executable, PL, "-f", fname, "-r", resname, "-c", chain, "-i", str(inst)] + list(extra),
                       cwd=d, capture_output=True, text=True, timeout=600)
    return r


def itp_atoms(path):
    """[(name, charge)] of the [ atoms ] block, and the '; Total charge' header value."""
    atoms, total, sec = [], None, None
    for line in open(path):
        m = re.match(r";\s*Total charge:\s*(-?\d+)", line)
        if m:
            total = int(m.group(1))
        s = line.split(";")[0].strip()
        if s.startswith("["):
            sec = s.strip("[] ")
            continue
        if sec == "atoms" and s:
            p = s.split()
            atoms.append((p[4], float(p[6])))
    return atoms, total


def failed_lines(d):
    p = os.path.join(d, "ligand_failed.gs")
    return [l.rstrip("\n") for l in open(p) if l.strip()] if os.path.isfile(p) else []


def load_param_module():
    """utils/parametrize_ligand.py as a module, or None if it cannot be imported
    without running (the pre-v1.5 script parsed its arguments at import)."""
    spec = importlib.util.spec_from_file_location("parametrize_ligand_under_test", PL)
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [PL]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        return None
    finally:
        sys.argv = argv
    return mod


tmp = tempfile.mkdtemp(prefix="test_ligand_failures_")

print("\n[1] sulfate is the dianion")
d = os.path.join(tmp, "so4"); os.makedirs(d)
open(os.path.join(d, "ligand_SO4_A_503.pdb"), "w").write(pdb("SO4", "A", 503, SO4_1S1Q))
r = run_param(d, "ligand_SO4_A_503.pdb", "SO4", "A")
check("parametrize_ligand.py succeeds on the 1S1Q sulfate", r.returncode == 0, r.stderr[-300:])
itp = os.path.join(d, "ligand_SO4.itp")
atoms, total = itp_atoms(itp) if os.path.isfile(itp) else ([], None)
check("the itp says total charge -2", total == -2, str(total))
check("five atoms, S and four O, no hydrogen", sorted(a[0][0] for a in atoms) == ["O", "O", "O", "O", "S"],
      str([a[0] for a in atoms]))
check("partial charges sum to -2.000", atoms and abs(sum(a[1] for a in atoms) + 2.0) < 1e-3,
      "%.4f" % sum(a[1] for a in atoms) if atoms else "no atoms")
check("and the instance is not in ligand_failed.gs", not failed_lines(d), str(failed_lines(d)))

print("\n[2] inorganic phosphate is HPO4 2-")
for label, atoms_in, fname in (("complete PO4", PO4_FULL, "ligand_PO4_H.pdb"),
                               ("2MTA PO4, P and two O deposited", PO4_2MTA, "ligand_PO4_H.pdb")):
    d = os.path.join(tmp, "po4_%d" % len(atoms_in)); os.makedirs(d)
    open(os.path.join(d, fname), "w").write(pdb("PO4", "H", 0, atoms_in))
    r = run_param(d, fname, "PO4", "H")
    itp = os.path.join(d, "ligand_PO4.itp")
    atoms, total = itp_atoms(itp) if os.path.isfile(itp) else ([], None)
    els = sorted(a[0][0] for a in atoms)
    check("%s: built, total charge -2" % label, r.returncode == 0 and total == -2, "rc %d, %s" % (r.returncode, total))
    check("%s: P, four O and one H (HPO4)" % label, els == ["H", "O", "O", "O", "O", "P"], str(els))
    check("%s: charges sum to -2.000" % label, atoms and abs(sum(a[1] for a in atoms) + 2.0) < 1e-3)
    if len(atoms_in) == 3:
        gro = os.path.join(d, "ligand_PO4_H.gro")
        xyz = [tuple(float(l[20 + 8 * k:28 + 8 * k]) * 10 for k in range(3))
               for l in open(gro).readlines()[2:7]] if os.path.isfile(gro) else []
        mates = [(32.320, 1.016, -0.718), (30.568, -0.718, -1.016)]
        hit = all(any(sum((p - q) ** 2 for p, q in zip(m, x)) ** 0.5 < 0.05 for x in xyz) for m in mates)
        check("2MTA: the two added O sit on the two-fold mates of the deposited ones", hit, str(xyz))

print("\n[3] a ligand that cannot be built is recorded, never just dropped")
d = os.path.join(tmp, "af3"); os.makedirs(d)
open(os.path.join(d, "ligand_AF3_R.pdb"), "w").write(pdb("AF3", "R", 169, AF3_1WQ1))
r = run_param(d, "ligand_AF3_R.pdb", "AF3", "R")
lines = failed_lines(d)
check("AlF3 (no Al-F bond term in OpenFF) exits non-zero", r.returncode != 0, str(r.returncode))
check("ligand_failed.gs holds exactly one line for it", len(lines) == 1, str(lines))
f = lines[0].split() if lines else []
check("the line is 'AF3 R 169 reason ...'", f[:3] == ["AF3", "R", "169"] and len(f) >= 4, str(lines))
check("the reason names the missing bond parameters", bool(lines) and "Bond" in lines[0], str(lines))
check("no manifest line and no itp for it",
      not os.path.isfile(os.path.join(d, "ligand_manifest.gs")) and not os.path.isfile(os.path.join(d, "ligand_AF3.itp")))
# a run killed mid-way (timeout, OOM) leaves the line it wrote first
d = os.path.join(tmp, "killed"); os.makedirs(d)
open(os.path.join(d, "ligand_SO4_A_503.pdb"), "w").write(pdb("SO4", "A", 503, SO4_1S1Q))
p = subprocess.Popen([sys.executable, PL, "-f", "ligand_SO4_A_503.pdb", "-r", "SO4", "-c", "A"], cwd=d,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
while time.time() - t0 < 20 and not failed_lines(d) and p.poll() is None:
    time.sleep(0.02)
p.send_signal(signal.SIGKILL)
p.wait()
lines = failed_lines(d)
check("a SIGKILLed run leaves 'SO4 A 503 ... did not finish'",
      len(lines) == 1 and lines[0].startswith("SO4 A 503 ") and "did not finish" in lines[0], str(lines))

print("\n[4] merge_ligand.py accounts for every extracted ligand")
TOP = ('; test topology\n#include "amber19sb.ff/forcefield.itp"\n\n#include "topol_Protein_chain_A.itp"\n%s\n'
       '[ system ]\ntest\n\n[ molecules ]\n; Compound        #mols\nProtein_chain_A     1\n')
# a five-residue stretch; residue 3 is an Asn whose ND2 sits at (1.000, 1.000, 1.145) nm
PROT = [(1, "ALA", "N", (0.50, 0.50, 0.50)), (1, "ALA", "CA", (0.60, 0.50, 0.50)),
        (2, "ALA", "N", (0.70, 0.50, 0.50)), (3, "ASN", "CB", (1.00, 1.00, 1.30)),
        (3, "ASN", "CG", (1.00, 1.00, 1.22)), (3, "ASN", "ND2", (1.00, 1.00, 1.145)),
        (4, "ALA", "N", (0.70, 0.70, 0.50)), (5, "ALA", "C", (0.80, 0.80, 0.60))]


def setup_merge(d, ions_itp=None, conf=None):
    """A structure directory as merge_ligand.py finds it. input.pdb is the deposited
    protein; conf.gro the pipeline's, by default at the same coordinates. The force
    field include is resolved inside the directory (an empty amber19sb.ff)."""
    os.makedirs(os.path.join(d, "amber19sb.ff"))
    w = lambda n, s: open(os.path.join(d, n), "w").write(s)
    w("amber19sb.ff/forcefield.itp", "; empty force field for the test\n")
    w("topol_Protein_chain_A.itp", "; dummy\n")
    if ions_itp is not None:
        w("ions_test.itp", ions_itp)
    w("topol.top", TOP % ('#include "ions_test.itp"\n' if ions_itp is not None else ""))
    w("input.pdb", "".join("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00 20.00           %s\n"
                           % (i + 1, " " + nm, res, rn, 10 * x, 10 * y, 10 * z, nm[0])
                           for i, (rn, res, nm, (x, y, z)) in enumerate(PROT)) + "END\n")
    w("conf.gro", "test\n%5d\n" % len(conf or PROT) + "".join(
        "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (rn, res, nm, i + 1, x, y, z)
        for i, (rn, res, nm, (x, y, z)) in enumerate(conf or PROT)) + "   5.00000   5.00000   5.00000\n")
    w("chain_map.gs", "# Sequential residue numbers belonging to protein B\n# Generated from input.pdb for chain(s): B\n5\n")
    return w


def lig_files(w, name, resname, chain, resnum, atoms_nm, inst):
    """an extracted PDB, and a parametrised .gro/.itp as parametrize_ligand.py leaves them."""
    w("ligand_%s.pdb" % name, pdb(resname, chain, resnum, [(n, el, tuple(10 * c for c in xyz)) for n, el, xyz in atoms_nm]))
    gro = "ligand_%s_%s.gro" % (inst, chain)
    w(gro, "x\n%5d\n" % len(atoms_nm) + "".join("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (1, resname, n, k + 1, *xyz)
                                              for k, (n, el, xyz) in enumerate(atoms_nm)) + "   5.00000   5.00000   5.00000\n")
    w("ligand_%s.itp" % inst, "[ moleculetype ]\n%s 3\n" % inst)
    w("ligand_%s_atomtypes.itp" % inst, "[ atomtypes ]\n")
    return "\t".join([inst, "ligand_%s.itp" % inst, "ligand_%s_atomtypes.itp" % inst, gro, str(len(atoms_nm)),
                      chain, str(resnum), "ligand_%s.pdb" % name])


def run_merge(d, *extra):
    return subprocess.run([sys.executable, ML, "-t", "topol.top", "-c", "conf.gro", "-m", "chain_map.gs"] + list(extra),
                          cwd=d, capture_output=True, text=True, timeout=120)


# nearest protein atom 0.34-0.40 nm (in range, not bonded); GOL 0.25-0.32 nm; EDO 3 nm away
SO4_NM = [("S", "S", (1.30, 1.25, 1.10)), ("O1", "O", (1.35, 1.30, 1.18)), ("O2", "O", (1.25, 1.20, 1.02)),
          ("O3", "O", (1.40, 1.17, 1.08)), ("O4", "O", (1.22, 1.33, 1.12))]
GOL_NM = [("C1", "C", (0.50, 0.80, 0.60)), ("O1", "O", (0.45, 0.85, 0.70)), ("C2", "C", (0.40, 0.75, 0.55))]
EDO_FAR = [("C1", "C", (3.50, 3.50, 3.50)), ("O1", "O", (3.55, 3.60, 3.50))]

d = os.path.join(tmp, "merge"); w = setup_merge(d)
line = lig_files(w, "SO4_B", "SO4", "B", 158, SO4_NM, "SO4")
w("ligand_GOL_A.pdb", pdb("GOL", "A", 7, [(n, el, tuple(10 * c for c in xyz)) for n, el, xyz in GOL_NM]))
w("ligand_EDO_A.pdb", pdb("EDO", "A", 9, [(n, el, tuple(10 * c for c in xyz)) for n, el, xyz in EDO_FAR]))
w("ligand_manifest.gs", line + "\n")
# renumber_pdb.py writes ligand_info.gs in PDB order: GOL before SO4
w("ligand_info.gs", "# Ligand residue name, chain, number of heavy atoms\nGOL\tA\t3\nSO4\tB\t5\nEDO\tA\t2\n")
r = run_merge(d, "--ligand-param", "on")
lines = failed_lines(d)
check("merge_ligand.py --ligand-param on exits 0", r.returncode == 0, r.stderr[-300:])
check("the unparametrised GOL A 7 next to the protein is in ligand_failed.gs",
      any(l.startswith("GOL A 7 ") for l in lines), str(lines))
check("the unparametrised EDO 2 nm from the protein is not (outside the region)",
      not any(l.startswith("EDO ") for l in lines), str(lines))
check("the parametrised SO4 reached topol.top", '#include "ligand_SO4.itp"' in open(os.path.join(d, "topol.top")).read())
cm = [l.strip() for l in open(os.path.join(d, "chain_map.gs")) if not l.startswith("#")]
check("SO4 of chain B is assigned to protein B (chain from the ligand, not the line number)", "6" in cm, str(cm))

d = os.path.join(tmp, "covalent"); w = setup_merge(d)
# a sugar-like ligand whose C1 is 1.45 A from ASN 3 ND2, as an N-glycan's is; its
# other atoms are 2.2 A or more from every protein atom
NAG_NM = [("C1", "C", (1.00, 1.00, 1.000)), ("O5", "O", (1.10, 0.95, 0.95)), ("C2", "C", (0.92, 0.90, 0.94))]
w("ligand_manifest.gs", lig_files(w, "NAG_A", "NAG", "A", 501, NAG_NM, "NAG") + "\n")
r = run_merge(d, "--ligand-param", "on")
lines = failed_lines(d)
check("a ligand 1.45 A from ASN ND2 in the deposited model is flagged as covalently bonded",
      len(lines) == 1 and lines[0].startswith("NAG A 501 ") and "covalent" in lines[0], str(lines))
check("and is still merged as before (the classic engine keeps it)",
      '#include "ligand_NAG.itp"' in open(os.path.join(d, "topol.top")).read())
r = run_merge(d, "--ligand-param", "on")   # a second merge must not take its own line for a parametrisation verdict
lines2 = failed_lines(d)
check("a repeated merge gives the same single line, not a doubled reason",
      lines2 == lines and lines2 and lines2[0].count("covalent") == 1, str(lines2))
# fuse_glycans.py says in glycan_unfused.gs why it left a glycan's sugars as ligands;
# the covalent line of such a sugar carries that reason (a ';' in it becomes ',', so a
# repeated merge does not take the rest for parametrize_ligand.py's verdict).
d = os.path.join(tmp, "covalent_glycan"); w = setup_merge(d)
w("ligand_manifest.gs", lig_files(w, "NAG_A", "NAG", "A", 501, NAG_NM, "NAG") + "\n")
w("glycan_unfused.gs", "NAG A 501 9 sugars, more than the cap of 3; not all saccharides: SO4 A 545 (NON-POLYMER)\n"
                       "GOL A 7 an unbonded ligand named here by mistake\n")
w("ligand_GOL_A.pdb", pdb("GOL", "A", 7, [(n, el, tuple(10 * c for c in xyz)) for n, el, xyz in EDO_FAR]))
r = run_merge(d, "--ligand-param", "on")
lines = failed_lines(d)
check("a sugar with a glycan_unfused.gs line: one line with 'covalently bonded' and 'glycan not fused:'",
      len(lines) == 1 and lines[0].startswith("NAG A 501 merge: covalently bonded in the deposited model, NAG C1 to protein "
                                              "ASN A 3 ND2 (1.45 A)")
      and lines[0].endswith("; merge: glycan not fused: 9 sugars, more than the cap of 3, not all saccharides: SO4 A 545 "
                            "(NON-POLYMER)"), str(lines))
r = run_merge(d, "--ligand-param", "on")
check("  a repeated merge gives the same line, the glycan reason once", failed_lines(d) == lines
      and lines[0].count("glycan not fused") == 1, str(failed_lines(d)))
check("  the line for a ligand that is not bonded (GOL 3 nm away) makes no failure",
      not any(l.startswith("GOL ") for l in failed_lines(d)), str(failed_lines(d)))
# The sugars of a glycan fuse_glycans.py left as ligands fail in the region with its
# reason, whatever else their line says (fusion review F1, F6): the rest of a glycan
# whose link is longer than the covalent rule and was not joined (an ambiguous
# stretched link, here 1FZC NAG K 2 with a deposited O1) has no bond here and was
# merged as a free molecule, and a sugar whose parametrisation failed got that failure
# alone.
d = os.path.join(tmp, "glycan_reasons"); w = setup_merge(d)
NAG_NEAR_NM = [("C1", "C", (1.00, 1.00, 0.85)), ("O5", "O", (1.10, 0.95, 0.80)), ("C2", "C", (0.92, 0.90, 0.79))]
w("ligand_manifest.gs", lig_files(w, "NAG_A_502", "NAG", "A", 502, NAG_NEAR_NM, "NAG") + "\n")
w("ligand_NAG_A_503.pdb", pdb("NAG", "A", 503, [(n, el, (10 * x, 10 * y, 10 * z - 3.0))
                                                 for n, el, (x, y, z) in NAG_NEAR_NM]))
w("ligand_failed.gs", "NAG A 503 AM1-BCC charges failed (antechamber/sqm): the deposited conformer failed\n")
STRETCHED = ("NAG A 502 C1 lies 1.85 A from NAG A 501 O4, a glycosidic link longer than the covalent rule that is "
             "not joined (its leaving atom O1 is deposited, 1.40 A from C1)")
w("glycan_unfused.gs", "NAG A 502 %s\nNAG A 503 4 sugars, more than the cap of 3\n" % STRETCHED)
r = run_merge(d, "--ligand-param", "on")
lines = failed_lines(d)
check("a parametrised sugar 3 A from the protein, no bond, with a glycan_unfused.gs line: a failure with the reason",
      "NAG A 502 merge: glycan not fused: " + STRETCHED in lines, str(lines))
check("  an unparametrised one: parametrize_ligand.py's reason and then the glycan reason",
      "NAG A 503 AM1-BCC charges failed (antechamber/sqm): the deposited conformer failed; merge: glycan not fused: "
      "4 sugars, more than the cap of 3" in lines, str(lines))
r = run_merge(d, "--ligand-param", "on")
check("  a repeated merge gives the same lines", failed_lines(d) == lines, str(failed_lines(d)))
# A fused glycan is protein: its sugars are ATOM records of the fused residue in
# fused.pdb but HETATM records in input.pdb, so a ligand bonded to one was merged as
# a free molecule (fusion review F1). The deposited protein is read from fused.pdb.
d = os.path.join(tmp, "fused_protein"); w = setup_merge(d)
O4A = (1.30, 1.00, 1.145)
w("input.pdb", open(os.path.join(d, "input.pdb")).read().replace(
    "END\n", "HETATM  901  O4  NAG A 600    %8.3f%8.3f%8.3f  1.00 20.00           O\nEND\n" % tuple(10 * c for c in O4A)))
w("fused.pdb", open(os.path.join(d, "input.pdb")).read().replace(
    "HETATM  901  O4  NAG A 600", "ATOM    901  O4A NG1 A   3"))
GAL_NM = [("C1", "C", (1.30, 1.00, 1.002)), ("O5", "O", (1.40, 0.95, 0.96)), ("C2", "C", (1.22, 0.90, 0.95))]
w("ligand_manifest.gs", lig_files(w, "GAL_A", "GAL", "A", 601, GAL_NM, "GAL") + "\n")
r = run_merge(d, "--ligand-param", "on")
lines = failed_lines(d)
check("a ligand 1.43 A from a sugar atom of a fused residue (ATOM in fused.pdb): covalently bonded to it",
      len(lines) == 1 and lines[0].startswith("GAL A 601 merge: covalently bonded in the deposited model, GAL C1 to "
                                              "protein NG1 A 3 O4A (1.43 A)"), str(lines))
# cap_termini.py's vacuum minimisation can push a side chain into an empty ligand site
# (1Z0K: Thr OG1 1.62 A from a GTP oxygen in conf.gro, 2.46 A in the deposit). Only the
# deposited model decides whether a ligand is bonded.
d = os.path.join(tmp, "clash"); conf = [p for p in PROT]
conf[5] = (3, "ASN", "ND2", (1.00, 1.00, 1.145))
w = setup_merge(d, conf=conf)
PROT_DEPOSITED_FAR = [(rn, res, nm, (x, y, z + (0.12 if nm == "ND2" else 0.0))) for rn, res, nm, (x, y, z) in PROT]
w("input.pdb", "".join("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00 20.00           %s\n"
                       % (i + 1, " " + nm, res, rn, 10 * x, 10 * y, 10 * z, nm[0])
                       for i, (rn, res, nm, (x, y, z)) in enumerate(PROT_DEPOSITED_FAR)) + "END\n")
w("ligand_manifest.gs", lig_files(w, "NAG_A", "NAG", "A", 501, NAG_NM, "NAG") + "\n")
r = run_merge(d, "--ligand-param", "on")
check("a contact that is 1.45 A only in conf.gro (2.65 A deposited) is not a covalent bond",
      not failed_lines(d), str(failed_lines(d)))

# --ligand-param (contract F4)
OFF_FF = ("GOL A 7 not parametrised: force field amber19sb has no ligand parametrisation; "
          "pass --no-ligand-param to run without ligands")
d = os.path.join(tmp, "offuser"); w = setup_merge(d)
w("ligand_GOL_A.pdb", pdb("GOL", "A", 7, [(n, el, tuple(10 * c for c in xyz)) for n, el, xyz in GOL_NM]))
w("ligand_info.gs", "# Ligand residue name, chain, number of heavy atoms\nGOL\tA\t3\n")
w("ligand_failed.gs", "GOL A 7 stale line of an earlier run\n")
r = run_merge(d, "--ligand-param", "off-user")
check("off-user: the left-out ligand is named on stdout", "NOT in the topology" in r.stdout and "GOL A 7" in r.stdout,
      r.stdout[-300:])
check("off-user: no ligand_failed.gs (a stale one is removed)", not failed_lines(d), str(failed_lines(d)))
d = os.path.join(tmp, "offuser_stale"); w = setup_merge(d)
w("ligand_manifest.gs", lig_files(w, "SO4_B", "SO4", "B", 158, SO4_NM, "SO4") + "\n")
r = run_merge(d, "--ligand-param", "off-user")
check("off-user: a manifest left from an earlier run is not merged",
      "ligand_SO4.itp" not in open(os.path.join(d, "topol.top")).read() and not failed_lines(d),
      open(os.path.join(d, "topol.top")).read()[-200:])
for label, extra in (("off-ff", ("--ligand-param", "off-ff")), ("no option and no manifest (fail closed)", ())):
    d = os.path.join(tmp, "offff_%d" % len(extra)); w = setup_merge(d)
    w("ligand_GOL_A.pdb", pdb("GOL", "A", 7, [(n, el, tuple(10 * c for c in xyz)) for n, el, xyz in GOL_NM]))
    w("ligand_EDO_A.pdb", pdb("EDO", "A", 9, [(n, el, tuple(10 * c for c in xyz)) for n, el, xyz in EDO_FAR]))
    r = run_merge(d, *extra)
    check("%s: exactly the contract line for the in-region GOL" % label, failed_lines(d) == [OFF_FF],
          str(failed_lines(d)))
d = os.path.join(tmp, "default_on"); w = setup_merge(d)
w("ligand_manifest.gs", lig_files(w, "SO4_B", "SO4", "B", 158, SO4_NM, "SO4") + "\n")
r = run_merge(d)
check("no option but a manifest: 'on' (the SO4 is merged, no line)",
      '#include "ligand_SO4.itp"' in open(os.path.join(d, "topol.top")).read() and not failed_lines(d),
      r.stdout[-200:])

print("\n[5] the checks catch a wrong molecule")
mod = load_param_module()
check("utils/parametrize_ligand.py imports as a module (main() behind __name__)", mod is not None)
if mod is not None and hasattr(mod, "chemistry_problems"):
    m = Chem.AddHs(Chem.MolFromSmiles("OCCOCC"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    X = m.GetConformer().GetPositions()
    ref = [("A%d" % a.GetIdx(), a.GetSymbol().upper()) + tuple(X[a.GetIdx()]) for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    check("the right molecule passes", mod.chemistry_problems(m, ref) == [], str(mod.chemistry_problems(m, ref)))
    p = mod.chemistry_problems(m, ref + [("O9", "O", 0.0, 0.0, 0.0)])
    check("one deposited heavy atom more than built: caught",
          p == ["heavy atoms differ from the deposited ligand: built C4O2, deposited C4O3"], str(p))
    p = mod.chemistry_problems(m, ref[:-1] + [(ref[-1][0], "N") + ref[-1][2:]])
    check("a different element: caught with both formulas",
          p == ["heavy atoms differ from the deposited ligand: built C4O2, deposited C3N1O2"], str(p))
    # same composition, but the two oxygens' positions swapped in the deposited list
    swapped = list(ref)
    swapped[0], swapped[3] = (swapped[0][0],) + ref[3][1:], (swapped[3][0],) + ref[0][1:]
    p = mod.chemistry_problems(m, swapped)
    check("same atoms in another order or place: caught", any("order and position" in s for s in p), str(p))
else:
    check("chemistry_problems() exists", False)
# black box: the 2FD6 ETX that OpenBabel builds as an enol ether (C1=C2 at 1.57 A)
d = os.path.join(tmp, "etx"); os.makedirs(d)
open(os.path.join(d, "ligand_ETX_H.pdb"), "w").write(pdb("ETX", "H", 403, ETX_2FD6_H))
r = run_param(d, "ligand_ETX_H.pdb", "ETX", "H")
sdf = os.path.join(d, "ligand_ETX.sdf")
built = Chem.MolFromMolFile(sdf, removeHs=False) if os.path.isfile(sdf) else None
itp = os.path.join(d, "ligand_ETX.itp")
head = open(itp).read().split("[ moleculetype ]")[0] if os.path.isfile(itp) else ""
check("2FD6 ETX: built (16 atoms, no C=C) and not in ligand_failed.gs",
      built is not None and built.GetNumAtoms() == 16 and not built.HasSubstructMatch(Chem.MolFromSmarts("C=C"))
      and not failed_lines(d), "%s %s" % (built.GetNumAtoms() if built else None, failed_lines(d)))
check("its itp names the CCD entry ETX in the repository cache",
      "CCD entry ETX (local cache forcefield/ccd_cache/ETX.cif" in head, head)
if mod is not None and hasattr(mod, "build_molecule") and hasattr(mod, "ccd_cache"):
    real_fetch = mod.ccd_cache.fetch
    mod.ccd_cache.fetch = lambda cid, network=True: (None, "%s unavailable in this test" % cid)
    try:
        etx = pdb("ETX", "H", 403, ETX_2FD6_H)
        with contextlib.redirect_stdout(io.StringIO()):
            mol, notes, probs = mod.build_molecule(etx, mod.pdb_atoms(etx), "ETX", 7.4)
        check("without the CCD entry the OpenBabel enol ether is flagged (C1-C2 1.57 A double)",
              any("C1-C2" in s and "double bond" in s for s in probs) or any("C2-C1" in s and "double bond" in s
                                                                              for s in probs), str(probs))
    finally:
        mod.ccd_cache.fetch = real_fetch
else:
    check("build_molecule() and the CCD cache exist", False)

print("\n[6] a lone iodide is the force field's ion")
d = os.path.join(tmp, "iod"); os.makedirs(d)
open(os.path.join(d, "ligand_IOD_A.pdb"), "w").write(pdb("IOD", "A", 1, IOD_1J2J))
r = run_param(d, "ligand_IOD_A.pdb", "IOD", "A")
man = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "ligand_manifest.gs"))] \
    if os.path.isfile(os.path.join(d, "ligand_manifest.gs")) else []
check("1J2J iodide: one manifest line, moleculetype I, no itp of its own",
      len(man) == 1 and man[0][0] == "I" and man[0][1] == "-", str(man))
check("and no ligand_failed.gs line", r.returncode == 0 and not failed_lines(d), str(failed_lines(d)))
gro = os.path.join(d, man[0][3]) if man else ""
g = open(gro).readlines() if gro and os.path.isfile(gro) else []
xyz = tuple(float(g[2][20 + 8 * k:28 + 8 * k]) for k in range(3)) if len(g) > 2 else None
check("its .gro holds the deposited position in nm (16.778, 12.078, 24.003 A / 10)",
      xyz is not None and all(abs(a - b) < 6e-4 for a, b in zip(xyz, (1.6778, 1.2078, 2.4003))), str(xyz))
check("and is a GROMACS line: residue name left-justified, atom name right-justified",
      len(g) > 2 and g[2][5:10] == "I    " and g[2][10:15] == "    I", repr(g[2] if len(g) > 2 else None))
IONS_I = "[ moleculetype ]\n; molname nrexcl\nI 1\n\n[ atoms ]\n1 I-_opc3 1 I I 1 -1.000000\n"
for label, ions in (("defines", IONS_I), ("does not define", "[ moleculetype ]\nBR 1\n")):
    dm = os.path.join(tmp, "iod_merge_" + label.split()[0]); w = setup_merge(dm, ions_itp=ions)
    w("ligand_IOD_A.pdb", pdb("IOD", "A", 1, [("I", "I", (12.0, 11.0, 10.0))]))
    w("ligand_IOD_A.gro", "x\n    1\n    1I        I    1   1.200   1.100   1.000\n   5.00000   5.00000   5.00000\n")
    w("ligand_manifest.gs", "\t".join(["I", "-", "-", "ligand_IOD_A.gro", "1", "A", "1", "ligand_IOD_A.pdb"]) + "\n")
    r = run_merge(dm, "--ligand-param", "on")
    top = open(os.path.join(dm, "topol.top")).read()
    if label == "defines":
        check("topology that defines I: 'I 1' in [ molecules ], no failure",
              re.search(r"^I\s+1\s*$", top, re.M) is not None and not failed_lines(dm), str(failed_lines(dm)))
        check("and no '#include \"-\"' (the ion comes with the force field)", '#include "-"' not in top, top)
    else:
        check("topology without I: ligand_failed.gs says so",
              any(l.startswith("IOD A 1 ") and "moleculetype I" in l for l in failed_lines(dm)), str(failed_lines(dm)))

print("\n[7] pH 7.4 acid rules")
if mod is not None and hasattr(mod, "ionize_acids"):
    for smi, q in (("OS(=O)(=O)O", -2), ("OP(=O)(O)O", -2), ("CC(=O)O", -1), ("COP(=O)(O)OC", -1),
                   ("COP(=O)(O)O", -2), ("COP(=O)(O)OP(=O)(O)OP(=O)(O)O", -4), ("CS(=O)(=O)O", -1),
                   ("OP(=O)([O-])[O-]", -2), ("[O-]P(=O)([O-])[O-]", -2), ("CCO", 0)):
        m, _ = mod.ionize_acids(Chem.AddHs(Chem.MolFromSmiles(smi)))
        check("%-32s -> %+d" % (smi, q), Chem.GetFormalCharge(m) == q, str(Chem.GetFormalCharge(m)))
else:
    check("ionize_acids() exists", False)

print("\n[8] every threshold from both sides (fixed numbers, not read from the code)")
if mod is not None and hasattr(mod, "chemistry_problems"):
    from rdkit.Geometry import Point3D

    def placed(smi, xyz):
        """explicit-H molecule whose heavy atoms sit at xyz (A), and the matching deposited list"""
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=11)
        c = m.GetConformer()
        for i, p in enumerate(xyz):
            c.SetAtomPosition(i, Point3D(*p))
        return m, [("A%d" % i, m.GetAtomWithIdx(i).GetSymbol().upper()) + tuple(p) for i, p in enumerate(xyz)]

    def probs(smi, xyz):
        m, ref = placed(smi, xyz)
        return mod.chemistry_problems(m, ref)

    # double bonds may be at most C=C 1.50 A (2FD6 enol ethers: 1.545-1.583 A), C=N 1.44, C=O 1.36
    for smi, lim in (("C=C", 1.50), ("C=N", 1.44), ("C=O", 1.36)):
        for dd, bad in ((lim - 0.01, False), (lim, False), (lim + 0.01, True)):
            if dd == lim and smi != "C=C":
                continue          # 1.44 and 1.36 are not exact in binary: below and above only
            p = probs(smi, [(0.0, 0.0, 0.0), (dd, 0.0, 0.0)])
            check("%s at %.2f A %s" % (smi, dd, "flagged" if bad else "accepted"), bool(p) == bad, str(p))
    # a carboxylate is judged by its shorter C-O, whichever O carries the double bond
    # (3MC0 ACT: C=O 1.444, C-OXT 1.302 A)
    for d1, d2, bad in ((1.444, 1.302, False), (1.40, 1.37, True), (1.40, 1.35, False)):
        p = probs("CC(=O)[O-]", [(-1.0, -1.2, 0.0), (0.0, 0.0, 0.0), (d1, 0.0, 0.0), (-0.5 * d2, 0.866 * d2, 0.0)])
        check("carboxylate C=O %.3f / C-O %.3f A: %s" % (d1, d2, "flagged" if bad else "accepted"), bool(p) == bad, str(p))
    # single bonds between two sp3 atoms: C-C at least 1.43 A (1RLB: 1.340-1.419), C-N 1.38 (1IBR 1.293)
    for smi, lim in (("CC", 1.43), ("CN", 1.38)):
        for dd, bad in ((lim - 0.01, True), (lim + 0.01, False)):
            p = probs(smi, [(0.0, 0.0, 0.0), (dd, 0.0, 0.0)])
            check("sp3 %s at %.2f A %s" % (smi, dd, "flagged" if bad else "accepted"), bool(p) == bad, str(p))
    # an sp3 carbon with three heavy neighbours at least 0.30 A out of their plane; sp2 at most 0.25
    tri = [(1.5, 0.0, 0.0), (-0.75, 1.299038, 0.0), (-0.75, -1.299038, 0.0)]
    for h, bad in ((0.29, True), (0.31, False)):
        p = probs("CC(C)C", [tri[0], (0.0, 0.0, h), tri[1], tri[2]])
        check("sp3 centre %.2f A out of plane %s" % (h, "flagged" if bad else "accepted"), bool(p) == bad, str(p))
    tri2 = [(1.4, 0.0, 0.0), (-0.7, 1.212436, 0.0), (-0.7, -1.212436, 0.0)]
    for h, bad in ((0.24, False), (0.26, True)):
        p = probs("CC(C)=C", [tri2[0], (0.0, 0.0, h), tri2[1], tri2[2]])
        p = [s for s in p if "sp2" in s]
        check("sp2 centre %.2f A out of plane %s" % (h, "flagged" if bad else "accepted"), bool(p) == bad, str(p))
    # hydrogen belongs on C, N, O, S and Se only
    for smi, el, bad in (("P", "P", True), ("B", "B", True), ("Cl", "Cl", True), ("CS", "S", False),
                         ("C[SeH]", "Se", False), ("CO", "O", False)):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=5)
        X = m.GetConformer().GetPositions()
        ref = [("A%d" % a.GetIdx(), a.GetSymbol().upper()) + tuple(X[a.GetIdx()]) for a in m.GetAtoms() if a.GetAtomicNum() > 1]
        p = [s for s in mod.chemistry_problems(m, ref) if s.startswith("hydrogen on")]
        check("H on the %s of %s: %s" % (el, smi, "flagged" if bad else "accepted"), bool(p) == bad, str(p))
    # oxyanion completion: a phosphate residue with P-O 1.35-1.75 A
    for blen, ok in ((1.34, False), (1.36, True), (1.74, True), (1.76, False)):
        u = (0.0, 0.8164966, 0.5773503)    # two O 109.47 deg apart
        txt = pdb("PO4", "H", 0, [("P", "P", (0.0, 0.0, 0.0)), ("O1", "O", (blen * u[1], 0.0, blen * u[2])),
                                  ("O2", "O", (-blen * u[1], 0.0, blen * u[2]))])
        try:
            out, note = mod.complete_oxyanion(txt, "PO4")
            got = note is not None and len(mod.pdb_atoms(out)) == 5
        except mod.LigandFailure:
            got = False
        except TypeError:
            got = None
        check("phosphate with P-O %.2f A: %s" % (blen, "completed" if ok else "refused"), got == ok, str(got))
else:
    check("chemistry_problems() and complete_oxyanion() exist", False)


def edge_merge(name, prot, lig_pdb_atoms, lig_gro_atoms):
    """one ligand next to a two-atom protein; lig_gro_atoms None = not parametrised"""
    d = os.path.join(tmp, name); os.makedirs(d)
    w = lambda n, s: open(os.path.join(d, n), "w").write(s)
    w("forcefield.itp", "; empty\n"); w("topol_Protein_chain_A.itp", "; dummy\n")
    w("topol.top", TOP.replace("amber19sb.ff/forcefield.itp", "forcefield.itp") % "")
    w("input.pdb", "".join("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00 20.00           %s\n"
                           % (i + 1, " " + nm, res, i + 1, 10 * x, 10 * y, 10 * z, nm[0])
                           for i, (res, nm, (x, y, z)) in enumerate(prot)) + "END\n")
    w("conf.gro", "t\n%5d\n" % len(prot) + "".join("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (i + 1, res, nm, i + 1, x, y, z)
                                                for i, (res, nm, (x, y, z)) in enumerate(prot)) + "   9.00000   9.00000   9.00000\n")
    w("chain_map.gs", "# Sequential residue numbers belonging to protein B\n# Generated from input.pdb for chain(s): B\n")
    w("ligand_GOL_A.pdb", pdb("GOL", "A", 7, [(n, el, tuple(10 * c for c in xyz)) for n, el, xyz in lig_pdb_atoms]))
    if lig_gro_atoms is not None:
        w("ligand_GOL_A.gro", "x\n%5d\n" % len(lig_gro_atoms) + "".join(
            "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (1, "GOL", n, k + 1, *xyz) for k, (n, el, xyz) in enumerate(lig_gro_atoms))
          + "   9.00000   9.00000   9.00000\n")
        w("ligand_GOL.itp", "[ moleculetype ]\nGOL 3\n"); w("ligand_GOL_atomtypes.itp", "[ atomtypes ]\n")
        w("ligand_manifest.gs", "\t".join(["GOL", "ligand_GOL.itp", "ligand_GOL_atomtypes.itp", "ligand_GOL_A.gro",
                                           str(len(lig_gro_atoms)), "A", "7", "ligand_GOL_A.pdb"]) + "\n")
    else:
        w("ligand_SO4_A.pdb", pdb("SO4", "A", 1, [("S", "S", (60.0, 60.0, 60.0))]))   # a parametrised one far away,
        w("ligand_SO4_A.gro", "x\n    1\n    1SO4      S    1   6.000   6.000   6.000\n   9.00000   9.00000   9.00000\n")
        w("ligand_SO4.itp", "[ moleculetype ]\nSO4 3\n")                             # so parametrisation did run
        w("ligand_manifest.gs", "\t".join(["SO4", "ligand_SO4.itp", "-", "ligand_SO4_A.gro", "1", "A", "1",
                                           "ligand_SO4_A.pdb"]) + "\n")
    r = run_merge(d, "--ligand-param", "on")
    return d, '#include "ligand_GOL.itp"' in open(os.path.join(d, "topol.top")).read(), failed_lines(d), r


TWO = [("ALA", "N", (1.0, 1.0, 1.0)), ("ALA", "CA", (4.0, 4.0, 4.0))]
# a parametrised ligand is kept when any of its atoms is within 0.5 nm of the protein
for dist, inside in ((0.490, True), (0.500, True), (0.510, False)):
    _, merged, lines, r = edge_merge("prox_%d" % int(dist * 1000), TWO, [("C1", "C", (1.0 + dist, 1.0, 1.0))],
                                     [("C1", "C", (1.0 + dist, 1.0, 1.0))])
    check("parametrised ligand %.3f nm from the protein: %s" % (dist, "merged" if inside else "left out"),
          merged == inside and not lines, "merged=%s lines=%s" % (merged, lines))
# one without parameters gets 0.11 nm more room (its hydrogens are missing): 0.61 nm
for dist, failure in ((0.600, True), (0.620, False)):
    _, merged, lines, r = edge_merge("unpar_%d" % int(dist * 1000), TWO, [("C1", "C", (1.0 + dist, 1.0, 1.0))], None)
    check("unparametrised ligand %.3f nm from the protein: %s" % (dist, "a failure" if failure else "outside, no failure"),
          any(l.startswith("GOL A 7 ") for l in lines) == failure, str(lines))
# covalent C-N: bonded below r_C + r_N + 0.35 = 1.82 A
for dist, bonded in ((0.180, True), (0.184, False)):
    nd2 = [("ASN", "ND2", (1.0, 1.0, 1.0)), ("ASN", "CB", (4.0, 4.0, 4.0))]
    _, merged, lines, r = edge_merge("cov_%d" % int(dist * 1000), nd2, [("C1", "C", (1.0 + dist, 1.0, 1.0))],
                                     [("C1", "C", (1.0 + dist, 1.0, 1.0))])
    check("ligand C %.2f A from ASN ND2: %s" % (dist * 10, "covalently bonded" if bonded else "not bonded"),
          any("covalent" in l for l in lines) == bonded, str(lines))

shutil.rmtree(tmp, ignore_errors=True)
print("\n%s" % ("ALL PASS" if fails == 0 else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
