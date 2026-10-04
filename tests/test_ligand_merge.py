#!/usr/bin/env python3
"""merge_ligand.py and make_ion_restraints.py: what the ligand merge leaves behind.

  [gro]    merged ligand atoms are written the way GROMACS writes them,
           '%5d%-5s%5s%5d': a three-letter name (every instance 0: SO4, GDP, NAG) is
           left-justified, so split() gives '7SO4' as GROMACS-written frames do. The
           right-justified name made split-based readers take the atom number for x
           (merge_crystal_waters.py --proximity lost a 1J2J water; make_disres_en.py
           on conf_cutout.gro saw none of the 67 sulfate restraints of 1DFJ).
  [col9]   the rewritten manifest carries each merged ligand's residue number in
           column 9, merged ligands first in GRO order.
  [ions]   make_ion_restraints.py finds each ligand residue's chain by that residue
           number (or, in a manifest without column 9, by the molecule name in the
           coordinates), never by line position: a chain-A IMD next to a chain-A Mg
           keeps its Mg-N3 coordination restraint when the manifest lists the ligands
           in another order than ligand_residues.gs (a chain-B I- first; or SO4_9
           listed before IMD_10, while the moleculetype sort puts IMD_10 first).
  [L9]     a contact shorter than any bond the element pair forms is a clash in the
           deposited model, printed and merged, not a covalent link: O-O 1.39 A clash,
           1.41 bonded; N-O 1.34 / 1.36; C-N 1.17 / 1.19 (0.80 of the radius sum 1.47).
           5HYS: sulfate O 1.16-1.30 A from Ser OG, Gly N or Tyr OH, no LINK record.
  [L7]     an altloc ligand is judged on its kept conformer: the dropped altloc's
           atom 1.45 A from an Asn ND2 is not a covalent bond.
  [rule]   the covalent rule is one definition, utils/covalent.py, which
           merge_ligand.py and fuse_glycans.py both import (neither has radii of its
           own), so a glycan fuse_glycans.py leaves unfused is one merge_ligand.py
           calls bonded: C-N bonded below 1.82 A (1.81 bond, 1.83 none) and a clash
           below 0.80 of the radius sum (1.17 clash, 1.19 bond), O-O 1.39 / 1.41,
           N-O 1.34 / 1.36, S-O bonded below 2.06 A (2.05 / 2.07).

Runs merge_ligand.py and make_ion_restraints.py in temporary directories; a few
seconds.

Standalone, no pytest: python3 tests/test_ligand_merge.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "utils", "merge_ligand.py")
MI = os.path.join(ROOT, "utils", "make_ion_restraints.py")
fails = 0


def check(name, ok, detail=""):
    global fails
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % detail))
    if not ok:
        fails += 1


def hetatm(serial, name, resname, chain, resnum, xyz, el, alt=" ", occ=1.00):
    nm = (" " + name) if len(name) < 4 else name
    return "HETATM%5d %-4s%1s%3s %1s%4d    %8.3f%8.3f%8.3f%6.2f 20.00          %2s" % (
        serial, nm, alt, resname, chain, resnum, xyz[0], xyz[1], xyz[2], occ, el)


TOP = ('; test\n#include "amber19sb.ff/forcefield.itp"\n\n#include "topol_Protein_chain_A.itp"\n%s\n'
       '[ system ]\ntest\n\n[ molecules ]\n; Compound        #mols\nProtein_chain_A     1\n')


def structure(d, prot, ions_itp=None, b_res=()):
    """prot: [(resnum, resname, atom, (x, y, z) nm)], deposited = pipeline coordinates."""
    os.makedirs(os.path.join(d, "amber19sb.ff"))
    w = lambda n, s: open(os.path.join(d, n), "w").write(s)
    w("amber19sb.ff/forcefield.itp", "; empty\n")
    w("topol_Protein_chain_A.itp", "; dummy\n")
    if ions_itp:
        w("ions_test.itp", ions_itp)
    w("topol.top", TOP % ('#include "ions_test.itp"\n' if ions_itp else ""))
    w("input.pdb", "".join("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00 20.00          %2s\n"
                           % (i + 1, " " + nm if len(nm) < 4 else nm, res, rn, 10 * x, 10 * y, 10 * z, nm[0])
                           for i, (rn, res, nm, (x, y, z)) in enumerate(prot) if res not in ("MG",)) + "END\n")
    w("conf.gro", "t\n%5d\n" % len(prot) + "".join("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (rn, res, nm, i + 1, x, y, z)
                                                for i, (rn, res, nm, (x, y, z)) in enumerate(prot))
      + "   6.00000   6.00000   6.00000\n")
    w("chain_map.gs", "# Sequential residue numbers belonging to protein B\n# Generated from input.pdb for chain(s): B\n"
      + "".join("%d\n" % r for r in b_res))
    return w


def ligand(w, fname, resname, chain, resnum, mol_name, atoms_nm, alt_rows=None):
    """extracted PDB (alt_rows: [(name, el, alt, occ, xyz nm)] instead of atoms_nm) and its parameters."""
    if alt_rows:
        text = "\n".join(hetatm(i + 1, n, resname, chain, resnum, tuple(10 * c for c in xyz), el, alt, occ)
                         for i, (n, el, alt, occ, xyz) in enumerate(alt_rows)) + "\nEND\n"
        atoms_nm = [(n, el, xyz) for n, el, alt, occ, xyz in alt_rows if alt in (" ", "A")]
    else:
        text = "\n".join(hetatm(i + 1, n, resname, chain, resnum, tuple(10 * c for c in xyz), el)
                         for i, (n, el, xyz) in enumerate(atoms_nm)) + "\nEND\n"
    w(fname, text)
    gro = "ligand_%s_%s.gro" % (mol_name, chain)
    w(gro, "x\n%5d\n" % len(atoms_nm) + "".join("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (1, resname, n, k + 1, *xyz)
                                              for k, (n, el, xyz) in enumerate(atoms_nm)) + "   5.00000   5.00000   5.00000\n")
    if resname == "IOD":
        return "\t".join([mol_name, "-", "-", gro, "1", chain, str(resnum), fname])
    w("ligand_%s.itp" % mol_name, "[ moleculetype ]\n%s 3\n" % mol_name)
    w("ligand_%s_atomtypes.itp" % mol_name, "[ atomtypes ]\n")
    return "\t".join([mol_name, "ligand_%s.itp" % mol_name, "ligand_%s_atomtypes.itp" % mol_name, gro,
                      str(len(atoms_nm)), chain, str(resnum), fname])


def merge(d):
    return subprocess.run([sys.executable, ML, "-t", "topol.top", "-c", "conf.gro", "-m", "chain_map.gs",
                           "--ligand-param", "on"], cwd=d, capture_output=True, text=True, timeout=120)


def failed_lines(d):
    p = os.path.join(d, "ligand_failed.gs")
    return [l.strip() for l in open(p) if l.strip()] if os.path.isfile(p) else []


tmp = tempfile.mkdtemp(prefix="test_ligand_merge_")
PROT = [(1, "ALA", "N", (1.00, 1.00, 1.00)), (1, "ALA", "CA", (1.10, 1.00, 1.00)), (2, "GLY", "N", (1.20, 1.10, 1.00)),
        (3, "SER", "OG", (1.50, 1.50, 1.50)), (4, "ASN", "ND2", (2.00, 2.00, 2.00)), (5, "ALA", "C", (2.40, 2.40, 2.40))]

print("\n[gro] and [col9]")
d = os.path.join(tmp, "gro"); w = structure(d, PROT)
SO4 = [("S", "S", (1.30, 1.25, 1.10)), ("O1", "O", (1.35, 1.30, 1.18)), ("O2", "O", (1.25, 1.20, 1.02)),
       ("O3", "O", (1.40, 1.17, 1.08)), ("O4", "O", (1.22, 1.33, 1.12))]
GOL = [("C1", "C", (1.05, 1.30, 1.00)), ("O1", "O", (1.00, 1.40, 1.00)), ("C2", "C", (1.15, 1.35, 1.00))]
man = [ligand(w, "ligand_SO4_A.pdb", "SO4", "A", 301, "SO4", SO4), ligand(w, "ligand_GOL_A.pdb", "GOL", "A", 302, "GOL_1", GOL)]
w("ligand_manifest.gs", "\n".join(man) + "\n")
r = merge(d)
conf = open(os.path.join(d, "conf.gro")).readlines()
lig = [l for l in conf[2:-1] if l[5:10].strip() in ("SO4", "GOL_1")]
check("merge exits 0", r.returncode == 0, r.stderr[-300:])
check("SO4 instance 0: residue name left-justified ('SO4  ')", any(l[5:10] == "SO4  " for l in lig), repr(lig[:1]))
so4 = [l for l in lig if l[5:10] == "SO4  "]
check("split() gives 'NSO4' then the atom name, as for a GROMACS-written frame",
      bool(so4) and re.fullmatch(r"\d+SO4", so4[0].split()[0]) is not None and so4[0].split()[1] == "S", repr(so4[:1]))
check("GOL_1 fills the 5 columns exactly", any(l[5:10] == "GOL_1" for l in lig), repr(lig[-1:]))
res_gs = [int(l) for l in open(os.path.join(d, "ligand_residues.gs")) if l.strip() and not l.startswith("#")]
mlines = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "ligand_manifest.gs")) if l.strip()]
check("manifest column 9 holds the merged residue numbers, in GRO order",
      [int(m[8]) for m in mlines if len(m) > 8] == res_gs and len(res_gs) == 2 and [m[0] for m in mlines] == ["GOL_1", "SO4"],
      "%s %s" % (mlines, res_gs))
gro_res = {int(l[0:5]): l[5:10].strip() for l in conf[2:-1]}
check("and each number is that ligand's residue in conf.gro",
      all(gro_res.get(int(m[8])) == m[0][:5] for m in mlines if len(m) > 8), str(gro_res))

print("\n[ions] make_ion_restraints.py pairs ligand residues by number or name")
MG_PROT = [(1, "ALA", "N", (1.00, 1.00, 1.00)), (1, "ALA", "CA", (1.10, 1.00, 1.00)), (2, "ASP", "OD1", (3.00, 3.00, 3.00)),
           (3, "ALA", "C", (3.40, 3.40, 3.40)), (4, "MG", "MG", (2.00, 2.00, 2.00))]
IMD = [("N1", "N", (2.25, 2.10, 2.00)), ("C2", "C", (2.30, 2.20, 2.00)), ("N3", "N", (2.21, 2.00, 2.00))]   # N3 0.21 nm from Mg


def ion_dir(name, manifest_rows, residues, gro_ligs, b_res):
    """conf_cutout.gro with the protein, the Mg (residue 4) and the ligand residues as merged."""
    d = os.path.join(tmp, name); os.makedirs(d)
    w = lambda n, s: open(os.path.join(d, n), "w").write(s)
    atoms = [(rn, res, nm, xyz) for rn, res, nm, xyz in MG_PROT]
    for resnum, resname, rows in gro_ligs:
        atoms += [(resnum, resname[:5], n, xyz) for n, el, xyz in rows]   # as merge_ligand.py writes it
    w("conf_cutout.gro", "t\n%5d\n" % len(atoms) + "".join("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (rn, res, nm, i + 1, *xyz)
                                                        for i, (rn, res, nm, xyz) in enumerate(atoms)) + "   6.0   6.0   6.0\n")
    w("chain_map.gs", "# B\n# Generated from input.pdb for chain(s): B\n" + "".join("%d\n" % r for r in b_res))
    w("ion_residues.gs", "# ions\n4\n")
    w("ligand_residues.gs", "# Residue numbers of parametrized ligands\n" + "".join("%d\n" % r for r in residues))
    w("ligand_manifest.gs", "\n".join("\t".join(r) for r in manifest_rows) + "\n")
    w("topol.top", "; test\n")
    r = subprocess.run([sys.executable, MI, "-f", "conf_cutout.gro", "-m", "chain_map.gs"], cwd=d,
                       capture_output=True, text=True, timeout=120)
    return d, open(os.path.join(d, "topol.top")).read(), r


# moleculetype order I (chain B, residue 5) before IMD (chain A, residue 6); the manifest in
# parametrisation order lists IMD first. Column 9 names each line's residue.
rows = [["IMD", "ligand_IMD.itp", "ligand_IMD_atomtypes.itp", "ligand_IMD_A.gro", "6", "A", "301", "ligand_IMD_A.pdb", "6"],
        ["I", "-", "-", "ligand_IOD_1_B.gro", "1", "B", "302", "ligand_IOD_B.pdb", "5"]]
d, top, r = ion_dir("ions_col9", rows, [5, 6], [(5, "I", [("I", "I", (4.5, 4.5, 4.5))]), (6, "IMD", IMD)], b_res=[5])
check("column 9: the chain-A IMD keeps its Mg-N3 restraint", re.search(r"MG-N3\(6IMD\)", top) is not None, top[-400:])
# a manifest without column 9 (before v1.5): by molecule name; SO4_9 (chain B) is listed before IMD_10
# (chain A), while the moleculetype sort put IMD_10 (GRO name IMD_1) first
rows = [["SO4_9", "ligand_SO4_9.itp", "ligand_SO4_9_atomtypes.itp", "ligand_SO4_9_B.gro", "5", "B", "401", "ligand_SO4_B.pdb"],
        ["IMD_10", "ligand_IMD_10.itp", "ligand_IMD_10_atomtypes.itp", "ligand_IMD_10_A.gro", "6", "A", "301", "ligand_IMD_A.pdb"]]
d, top, r = ion_dir("ions_name", rows, [5, 6], [(5, "IMD_10", IMD), (6, "SO4_9", [("S", "S", (4.5, 4.5, 4.5))])], b_res=[6])
check("by name (IMD_10 vs SO4_9): the IMD keeps its Mg-N3 restraint", re.search(r"MG-N3\(5IMD_1\)", top) is not None, top[-400:])
# a residue whose name contradicts its column 9 line is left to chain_map.gs, with a warning
rows = [["IMD", "ligand_IMD.itp", "ligand_IMD_atomtypes.itp", "ligand_IMD_B.gro", "6", "B", "301", "ligand_IMD_B.pdb", "5"]]
d, top, r = ion_dir("ions_mismatch", rows, [5], [(5, "GOL", IMD)], b_res=[])
check("a contradicting line is not used (warning, residue stays on chain_map's side)",
      "Warning: ligand residue 5 is GOL" in r.stdout and re.search(r"MG-N3", top) is not None, r.stdout[-300:])
# end to end: merge (moleculetype sort I < IMD) and make_ion_restraints
d = os.path.join(tmp, "ions_e2e")
w = structure(d, MG_PROT, ions_itp="[ moleculetype ]\nI 1\n")
w("ion_residues.gs", "# ions\n4\n")
man = [ligand(w, "ligand_IMD_A.pdb", "IMD", "A", 301, "IMD", IMD),
       ligand(w, "ligand_IOD_B.pdb", "IOD", "B", 302, "I", [("I", "I", (2.6, 2.6, 2.6))])]
w("ligand_manifest.gs", "\n".join(man) + "\n")
r = merge(d)
shutil.copy(os.path.join(d, "conf.gro"), os.path.join(d, "conf_cutout.gro"))
r2 = subprocess.run([sys.executable, MI, "-f", "conf_cutout.gro", "-m", "chain_map.gs"], cwd=d, capture_output=True,
                    text=True, timeout=120)
top = open(os.path.join(d, "topol.top")).read()
check("merge then make_ion_restraints: the IMD's Mg-N3 restraint is there", re.search(r"MG-N3\(\d+IMD\)", top) is not None,
      r.stdout[-200:] + r2.stdout[-300:])

print("\n[L9] a clash in the deposited model is not a covalent bond")


def contact(name, prot_atom, lig_el, dist_a):
    """one parametrised ligand atom dist_a A from one protein atom (along x)"""
    d = os.path.join(tmp, name)
    pa = [(1, prot_atom[0], prot_atom[1], (1.0, 1.0, 1.0)), (2, "ALA", "CA", (4.0, 4.0, 4.0))]
    w = structure(d, pa)
    w("ligand_manifest.gs", ligand(w, "ligand_LIG_A.pdb", "LIG", "A", 501, "LIG",
                                   [("X1", lig_el, (1.0 + dist_a / 10.0, 1.0, 1.0))]) + "\n")
    r = merge(d)
    merged = '#include "ligand_LIG.itp"' in open(os.path.join(d, "topol.top")).read()
    return failed_lines(d), merged, r.stdout


for prot_atom, el, dist, kind in ((("SER", "OG"), "O", 1.39, "clash"), (("SER", "OG"), "O", 1.41, "bond"),
                                  (("GLY", "N"), "O", 1.34, "clash"), (("GLY", "N"), "O", 1.36, "bond"),
                                  (("ASN", "ND2"), "C", 1.17, "clash"), (("ASN", "ND2"), "C", 1.19, "bond"),
                                  (("SER", "OG"), "O", 1.20, "clash")):
    lines, merged, out = contact("%s_%s_%d" % (prot_atom[1], el, int(dist * 100)), prot_atom, el, dist)
    if kind == "clash":
        check("%s-%s %.2f A: clash, printed, merged, no ligand_failed.gs line" % (el, prot_atom[1], dist),
              not lines and merged and "clash in the deposited model" in out, "%s %s" % (lines, out[-200:]))
    else:
        check("%s-%s %.2f A: covalently bonded (ligand_failed.gs line)" % (el, prot_atom[1], dist),
              len(lines) == 1 and "covalently bonded" in lines[0] and merged, str(lines))

print("\n[L7] an altloc ligand is judged on the conformer it keeps")
d = os.path.join(tmp, "altloc"); w = structure(d, PROT)
ND2 = (2.00, 2.00, 2.00)
alt_rows = [("C1", "C", "A", 0.60, (2.60, 2.60, 2.60)), ("O1", "O", "A", 0.60, (2.70, 2.60, 2.60)),
            ("C1", "C", "B", 0.40, (2.00, 2.00, 2.145)), ("O1", "O", "B", 0.40, (2.00, 2.10, 2.25))]
w("ligand_manifest.gs", ligand(w, "ligand_EDO_A.pdb", "EDO", "A", 601, "EDO", None, alt_rows=alt_rows) + "\n")
r = merge(d)
check("the dropped altloc B (C1 1.45 A from ND2) does not make a covalent bond", not failed_lines(d), str(failed_lines(d)))

print("\n[rule] one covalent rule, utils/covalent.py")
CV = os.path.join(ROOT, "utils", "covalent.py")
if os.path.isfile(CV):
    import importlib.util
    spec = importlib.util.spec_from_file_location("covalent_under_test", CV)
    cov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cov)
    for e1, e2, dist, kind in (("C", "N", 1.81, "bond"), ("N", "C", 1.83, None), ("C", "N", 1.17, "clash"),
                               ("C", "N", 1.19, "bond"), ("O", "O", 1.39, "clash"), ("O", "O", 1.41, "bond"),
                               ("O", "N", 1.34, "clash"), ("N", "O", 1.36, "bond"), ("S", "O", 2.05, "bond"),
                               ("O", "S", 2.07, None)):
        check("%s-%s %.2f A: %s" % (e1, e2, dist, kind or "no contact"), cov.contact_kind(e1, e2, dist) == kind,
              str(cov.contact_kind(e1, e2, dist)))
else:
    check("utils/covalent.py exists", False, CV)
for path in ("merge_ligand.py", "fuse_glycans.py"):
    src = open(os.path.join(ROOT, "utils", path)).read() if os.path.isfile(os.path.join(ROOT, "utils", path)) else ""
    check("%s imports the rule from covalent.py and has no radii of its own" % path,
          re.search(r"^from covalent import .*contact_kind", src, re.M) is not None
          and re.search(r"^\s*COVALENT_RADII\s*=", src, re.M) is None
          and re.search(r"^def contact_kind", src, re.M) is None, path)

shutil.rmtree(tmp, ignore_errors=True)
print("\n%s" % ("ALL PASS" if fails == 0 else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
