#!/usr/bin/env python3
"""Bond orders from the CCD entry: the cache, the exact match, the guanine tautomer.

OpenBabel guesses bond orders from the deposited geometry. For 8 of the 9 GDP, GTP
and GNP of the HADDOCK benchmark it put the guanine ring H on N3 and none on N1
(the 3H tautomer; the canonical 9-substituted guanine has N1-H), and nothing
noticed (review L3). Its geometric checks, calibrated on 44 HADDOCK instances,
flagged 12 correct molecules among 1400 PDB ligand instances (review L1). And the
template repairs needed files.rcsb.org, which compute nodes may not reach. v1.5
takes bond orders, charges and hydrogens from the CCD entry of the residue name,
from the repository cache first, whenever the entry describes exactly the deposited
heavy atoms. This pins, with the 1A2K GDP C 220 as deposited:

  [cache]  utils/ccd_cache.py reads the cached GDP entry (28 heavy atoms, 15 H, 45
           bonds, quoted names such as O5' and H5'', PB=O1B double), serves cached
           entries without touching the network, skips a cached file that is not
           the entry (an error page), reports a miss with the reason, and its
           prefetch fetches every HETATM residue name of a set of PDB files except
           water and ions, keeps what is cached, and refuses non-entries.
  [exact]  the GDP is built from the cached entry, matched by atom name, with the
           source, the entry's modification date and the match in the itp note:
           40 atoms, -3, H on N1 and none on N3; with renamed atoms it is matched by
           bond graph to the same molecule; with one oxygen missing it is not exact.
           The CCD build goes through OpenBabel's pH model like OpenBabel's own
           molecules: the neutral CCD allylamine (5E6J AYE) becomes C=CC[NH3+].
  [L3]     without the entry OpenBabel's 3H tautomer is flagged ('guanine built as
           the 3H tautomer').
  [L1]     a CCD build skips the geometric checks: the GDP with C5' moved to 1.38 A
           from C4' (below the sp3-sp3 C-C limit of 1.43) passes as a CCD build,
           while the full checks on the same molecule report the bond.
  [L-1]    a match by atom name must fit the deposited geometry: with N1/N3, PA/PB
           or C2/C8 swapped the GDP is matched by bond graph to the deposited
           molecule (v15_work review L-1: built by name with bonds of 2.4-5.6 A).
  [L-2]    a cached entry cut before or inside its bond loop is not built (no
           radicals), the reason names the file, files.rcsb.org is asked for the
           complete entry, and with network that entry is used.
  [L-5]    no prefix search of search.rcsb.org after a CCD miss; [L-8] the
           molecule's problems lead, the CCD miss comes last, labelled.
  [L-6]    the CCD identifier is the residue name of the records, not -r.
  [V1]     the entry itself decides 'incomplete' (an atom without a bond), not
           radical electrons: the real nitric oxide entry (NO) is complete, and as
           a radical it stops the ligand (LigandFailure naming the unpaired
           electron, a ligand_failed.gs line, exit 1) from files.rcsb.org, and from
           the cache when files.rcsb.org confirms it or cannot be asked (then the
           reason says the cached file could not be checked), where OpenBabel had
           built HNO without a flag; an entry that files.rcsb.org serves incomplete
           in the same way as the cache stops it too; with no network the OpenBabel
           note names the incomplete cached file, not only the failed request; and
           a build from files.rcsb.org's copy names the damaged cached file.
  [N1]     a cached GDP.cif cut just before its last bond row (N3-C4) keeps a bond
           on every atom and builds as a radical: with files.rcsb.org serving the
           entry the GDP is built from that copy (40 atoms, -3, the damaged file
           named), with no network the LigandFailure names the cached file as not
           checked; it was 'the CCD entry GDP is a radical' (verifier N1).
  [N2]     files.rcsb.org serving AYE with its N1-HN1A row cut and no cached copy:
           'the CCD entry AYE at files.rcsb.org is incomplete ...' after one
           request, also when the cached copy is cut elsewhere.
  [V4]     prefetched files get the mode open() gives (0644 under umask 022, 0640
           under 027; mkstemp's was 0600), and an unreadable cached file is a
           reason naming it, not a PermissionError.
  [N3]     prefetch over an unreadable cached file fetches it again instead of
           ending in a PermissionError traceback, and a directory it cannot write
           is a failure with the reason.
  [parents] the prefetch adds ASN, SER and THR, the amino acids fuse_glycans.py
           fuses a glycan with, whenever an input has a HETATM residue other than
           water and ions.

In-process, no network (urlopen is replaced by a function that fails the test if
it is called), no AM1-BCC; a few seconds.

Standalone, no pytest: python3 tests/test_ligand_ccd.py
"""
import contextlib
import importlib.util
import io
import os
import shutil
import stat
import sys
import tempfile

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL = os.path.join(ROOT, "utils", "parametrize_ligand.py")
fails = 0


def check(name, ok, detail=""):
    global fails
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % detail))
    if not ok:
        fails += 1


def load():
    spec = importlib.util.spec_from_file_location("parametrize_ligand_under_test", PL)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [PL]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        return None
    finally:
        sys.argv = argv
    return mod


# 1A2K GDP C 220, deposited (A)
GDP_1A2K = [("PB", "P", 53.384, -6.066, 83.175), ("O1B", "O", 53.625, -4.573, 83.255),
            ("O2B", "O", 51.910, -6.195, 83.185), ("O3B", "O", 53.972, -6.622, 81.994),
            ("O3A", "O", 54.117, -6.462, 84.578), ("PA", "P", 55.395, -7.243, 85.008),
            ("O1A", "O", 56.667, -6.523, 84.770), ("O2A", "O", 55.109, -7.391, 86.482),
            ("O5'", "O", 55.223, -8.574, 84.230), ("C5'", "C", 54.467, -9.634, 84.775),
            ("C4'", "C", 55.063, -10.949, 85.156), ("O4'", "O", 55.915, -11.483, 84.122),
            ("C3'", "C", 55.835, -11.045, 86.461), ("O3'", "O", 55.499, -12.184, 87.250),
            ("C2'", "C", 57.285, -11.096, 86.008), ("O2'", "O", 58.087, -11.644, 87.034),
            ("C1'", "C", 57.168, -11.895, 84.665), ("N9", "N", 58.332, -11.566, 83.810),
            ("C8", "C", 58.627, -10.321, 83.285), ("N7", "N", 59.741, -10.253, 82.598),
            ("C5", "C", 60.241, -11.553, 82.713), ("C6", "C", 61.428, -12.070, 82.140),
            ("O6", "O", 62.239, -11.506, 81.412), ("N1", "N", 61.553, -13.417, 82.450),
            ("C2", "C", 60.640, -14.156, 83.194), ("N2", "N", 61.021, -15.421, 83.359),
            ("N3", "N", 59.513, -13.680, 83.682), ("C4", "C", 59.379, -12.357, 83.424)]


def gdp_pdb(rows, names=None):
    out = []
    for i, (n, el, x, y, z) in enumerate(rows):
        n = names[i] if names else n
        nm = (" " + n) if len(n) < 4 else n
        out.append("HETATM%5d %-4s GDP C 220    %8.3f%8.3f%8.3f  1.00 30.00          %2s" % (3564 + i, nm, x, y, z, el))
    return "\n".join(out) + "\nEND\n"


mod = load()
if mod is None or not hasattr(mod, "ccd_cache"):
    print("parametrize_ligand.py has no CCD cache (utils/ccd_cache.py)")
    sys.exit(1)
C = mod.ccd_cache
net_calls = []


def no_network(*a, **k):
    net_calls.append(a[0] if a else k)
    raise OSError("network disabled in this test")


C.urllib.request.urlopen = no_network
tmp = tempfile.mkdtemp(prefix="test_ligand_ccd_")
os.environ.pop("GROSCORE_CCD_CACHE", None)


def build(text, resname):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.build_molecule(text, mod.pdb_atoms(text), resname, 7.4)


def nh(mol, i):
    return mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)


print("\n[cache] utils/ccd_cache.py")
text, src = C.fetch("GDP")
check("GDP comes from the repository cache", text is not None and src.endswith(os.path.join("forcefield", "ccd_cache", "GDP.cif")),
      str(src))
comp = C.component(text) if text else {"atoms": [], "bonds": []}
heavy = [a for a in comp["atoms"] if a["element"] != "H"]
names = {a["name"] for a in comp["atoms"]}
check("28 heavy atoms, 15 hydrogens, 45 bonds", (len(heavy), len(comp["atoms"]) - len(heavy), len(comp["bonds"])) == (28, 15, 45),
      "%d %d %d" % (len(heavy), len(comp["atoms"]) - len(heavy), len(comp["bonds"])))
check("quoted names read whole: O5', H5'', HO3' (a leaving atom)",
      {"O5'", "H5''", "HO3'"} <= names and any(a["name"] == "HO3'" and a["leaving"] for a in comp["atoms"]), str(sorted(names)[:8]))
check("PB=O1B double, PB-O2B single", ("PB", "O1B", "DOUB", "N") in comp["bonds"] and ("PB", "O2B", "SING", "N") in comp["bonds"])
check("no network call for a cached entry", not net_calls, str(net_calls))
text, why = C.fetch("ZZZ")
check("an uncached entry with no network: None and the reason",
      text is None and "not in the CCD cache" in why and "did not answer" in why, str(why))
extra = os.path.join(tmp, "extra"); os.makedirs(extra)
open(os.path.join(extra, "ZZZ.cif"), "w").write("<html>404 Not Found</html>\n")
os.environ["GROSCORE_CCD_CACHE"] = extra
text, why = C.fetch("ZZZ")
check("a cached file that is not the entry (an error page) is ignored", text is None, str(why))
mini = ("data_ZZZ\n_chem_comp.id ZZZ\n_chem_comp.pdbx_modified_date 2020-01-01\nloop_\n_chem_comp_atom.comp_id\n"
        "_chem_comp_atom.atom_id\n_chem_comp_atom.type_symbol\n_chem_comp_atom.charge\nZZZ O1 O 0\nZZZ H1 H 0\nZZZ H2 H 0\n"
        "loop_\n_chem_comp_bond.comp_id\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n_chem_comp_bond.value_order\n"
        "ZZZ O1 H1 SING\nZZZ O1 H2 SING\n")
open(os.path.join(extra, "ZZZ.cif"), "w").write(mini)
text, src = C.fetch("ZZZ")
check("an entry in $GROSCORE_CCD_CACHE is found", text is not None and src == os.path.join(extra, "ZZZ.cif"), str(src))
os.environ.pop("GROSCORE_CCD_CACHE")
# prefetch
inp = os.path.join(tmp, "input.pdb")
open(inp, "w").write("".join("HETATM%5d %-4s %3s A%4d       0.000   0.000   0.000  1.00 20.00          %2s\n"
                             % (i + 1, n, r, i + 1, e) for i, (n, r, e) in enumerate(
                                 [(" O  ", "HOH", "O"), ("MG  ", " MG", "MG"), (" PB ", "GDP", "P"), (" S  ", "SO4", "S"),
                                  ("CL  ", " CL", "CL"), (" I  ", "IOD", "I"), (" C1 ", "1PE", "C")])))
check("ligand_ids: every HETATM residue name but water and ions",
      C.ligand_ids([inp]) == ["1PE", "GDP", "IOD", "SO4"], str(C.ligand_ids([inp])))
served = []


class Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body.encode()


def fake_urlopen(url, timeout=None):
    cid = url.rsplit("/", 1)[1].split(".")[0]
    served.append(cid)
    if cid == "IOD":
        return Resp("<html>not found</html>")
    return Resp("data_%s\n_chem_comp.id %s\nloop_\n_chem_comp_atom.comp_id\n_chem_comp_atom.atom_id\n%s X\n" % (cid, cid, cid))


C.urllib.request.urlopen = fake_urlopen
out = os.path.join(tmp, "cache")
os.makedirs(out)
open(os.path.join(out, "GDP.cif"), "w").write("data_GDP\n_chem_comp_atom.atom_id X\n")
fetched, kept, failed = C.prefetch(C.ligand_ids([inp]), out)
check("prefetch: fetched 1PE and SO4, kept the cached GDP, refused the non-entry IOD",
      fetched == ["1PE", "SO4"] and kept == ["GDP"] and [f[0] for f in failed] == ["IOD"] and "GDP" not in served,
      "%s %s %s served %s" % (fetched, kept, failed, served))
check("and wrote them as <ID>.cif, nothing for IOD",
      sorted(os.listdir(out)) == ["1PE.cif", "GDP.cif", "SO4.cif"], str(sorted(os.listdir(out))))
fetched, kept, failed = C.prefetch(["GDP"], out, refresh=True)
check("--refresh fetches a cached entry again", fetched == ["GDP"], str(fetched))
C.urllib.request.urlopen = no_network
net_calls.clear()

print("\n[exact] the 1A2K GDP from its CCD entry")
mol, notes, probs = build(gdp_pdb(GDP_1A2K), "GDP")
names_dep = [r[0] for r in GDP_1A2K]
i1, i3 = names_dep.index("N1"), names_dep.index("N3")
check("40 atoms, total charge -3, no problem", mol.GetNumAtoms() == 40 and Chem.GetFormalCharge(mol) == -3 and not probs,
      "%d %d %s" % (mol.GetNumAtoms(), Chem.GetFormalCharge(mol), probs))
check("guanine N1 carries one H and N3 none (the canonical tautomer)", nh(mol, i1) == 1 and nh(mol, i3) == 0,
      "N1 %d N3 %d" % (nh(mol, i1), nh(mol, i3)))
check("the itp note names entry, cache, match and modification date",
      notes and notes[0] == "bond orders, charges and hydrogens from the CCD entry GDP (local cache "
      "forcefield/ccd_cache/GDP.cif, matched by atom name, entry modified 2024-09-27)", str(notes[:1]))
check("no network call", not net_calls, str(net_calls))
smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
renamed = ["X%d" % (k + 1) for k in range(len(GDP_1A2K))]
mol2, notes2, probs2 = build(gdp_pdb(GDP_1A2K, renamed), "GDP")
check("renamed atoms: matched by bond graph, the same molecule",
      notes2 and "matched by bond graph" in notes2[0] and Chem.MolToSmiles(Chem.RemoveHs(mol2)) == smi and not probs2,
      "%s %s" % (notes2[:1], probs2))
# the CCD build is protonated by OpenBabel's pH model, as OpenBabel's own perception is:
# allylamine (5E6J AYE B 76, CCD entry neutral) becomes C=CC[NH3+] at pH 7.4 (pKa 9.5)
AYE_CIF = ("data_AYE\n_chem_comp.id AYE\n_chem_comp.pdbx_modified_date 2024-10-08\nloop_\n"
           "_chem_comp_atom.comp_id\n_chem_comp_atom.atom_id\n_chem_comp_atom.type_symbol\n_chem_comp_atom.charge\n"
           + "".join("AYE %s %s 0\n" % (n, e) for n, e in (("C2", "C"), ("C3", "C"), ("C1", "C"), ("N1", "N"), ("H2", "H"),
                                                       ("H3", "H"), ("H3A", "H"), ("H1", "H"), ("H1A", "H"), ("HN1", "H"),
                                                       ("HN1A", "H")))
           + "loop_\n_chem_comp_bond.comp_id\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n"
           "_chem_comp_bond.value_order\n"
           + "".join("AYE %s %s %s\n" % b for b in (("C2", "C3", "DOUB"), ("C2", "C1", "SING"), ("C1", "N1", "SING"),
                                                    ("C2", "H2", "SING"), ("C3", "H3", "SING"), ("C3", "H3A", "SING"),
                                                    ("C1", "H1", "SING"), ("C1", "H1A", "SING"), ("N1", "HN1", "SING"),
                                                    ("N1", "HN1A", "SING"))))
aye_cache = os.path.join(tmp, "aye"); os.makedirs(aye_cache)
open(os.path.join(aye_cache, "AYE.cif"), "w").write(AYE_CIF)
os.environ["GROSCORE_CCD_CACHE"] = aye_cache
AYE_5E6J = "".join("HETATM%5d  %-3s AYE B  76    %8.3f%8.3f%8.3f  1.00 65.00           %s\n" % (3082 + k, n, x, y, z, n[0])
                   for k, (n, x, y, z) in enumerate((("C2", 110.800, 11.996, 283.069), ("C3", 109.908, 12.660, 282.308),
                                                     ("C1", 110.371, 11.407, 284.394), ("N1", 111.513, 10.915, 285.089))))
try:
    mol6, notes6, probs6 = build(AYE_5E6J + "END\n", "AYE")
    check("5E6J AYE from its CCD entry, pH 7.4: C=CC[NH3+], +1, 12 atoms",
          notes6 and "CCD entry AYE" in notes6[0] and Chem.MolToSmiles(Chem.RemoveHs(mol6)) == "C=CC[NH3+]"
          and Chem.GetFormalCharge(mol6) == 1 and mol6.GetNumAtoms() == 12 and not probs6,
          "%s %s %d %s" % (notes6[:1], Chem.MolToSmiles(Chem.RemoveHs(mol6)), Chem.GetFormalCharge(mol6), probs6))
finally:
    os.environ.pop("GROSCORE_CCD_CACHE", None)
short = [r for r in GDP_1A2K if r[0] != "O2B"]
mol3, notes3, probs3 = build(gdp_pdb(short), "GDP")
check("one oxygen missing: not the exact entry, built by OpenBabel and said why",
      notes3 and notes3[0].startswith("bond orders from OpenBabel") and "28 heavy atoms, 27 are deposited" in " ".join(notes3 + probs3),
      "%s %s" % (notes3[:1], probs3[:2]))

print("\n[L3] without the entry, OpenBabel's guanine tautomer is flagged")
real_fetch = C.fetch
C.fetch = lambda cid, network=True: (None, "%s unavailable in this test" % cid)
net_calls.clear()
try:
    mol4, notes4, probs4 = build(gdp_pdb(GDP_1A2K), "GDP")
    check("the 1A2K GDP by OpenBabel: 'guanine built as the 3H tautomer: H on N3, none on N1'",
          any(p == "guanine built as the 3H tautomer: H on N3, none on N1" for p in probs4), str(probs4))
    check("(OpenBabel indeed put the H on N3)", nh(mol4, i3) == 1 and nh(mol4, i1) == 0,
          "N1 %d N3 %d" % (nh(mol4, i1), nh(mol4, i3)))
    # L-8: the molecule's problems lead the LIGAND_FAIL reason; why the CCD entry was
    # not used comes last, labelled, instead of reading as a defect of the molecule.
    check("L-8: the problems come first, the CCD miss last as 'no exact CCD entry: ...'",
          probs4[:1] == ["guanine built as the 3H tautomer: H on N3, none on N1"]
          and probs4[-1] == "no exact CCD entry: GDP unavailable in this test", str(probs4))
    # L-5: no search of search.rcsb.org for IDs starting with the residue name (the
    # live API rejects the query, so it only ever cost a failing request)
    check("L-5: no network request after the CCD miss (no prefix search)", not net_calls, str(net_calls))
finally:
    C.fetch = real_fetch

print("\n[L-1] a by-name match must fit the deposited geometry")
# Swapped names: the entry's bonds would join atoms 2.4-5.6 A apart. The bond graph
# finds the deposited molecule instead; v1.4 and the first v1.5 round built the
# connectivity on the wrong atoms 'by atom name' with no problem reported.
for a, b in (("N1", "N3"), ("PA", "PB"), ("C2", "C8")):
    names = [r[0] for r in GDP_1A2K]
    ia, ib = names.index(a), names.index(b)
    names[ia], names[ib] = b, a
    mol7, notes7, probs7 = build(gdp_pdb(GDP_1A2K, names), "GDP")
    X = mol7.GetConformer().GetPositions()
    longest = max(float(np.linalg.norm(X[x.GetBeginAtomIdx()] - X[x.GetEndAtomIdx()])) for x in mol7.GetBonds()
                  if x.GetBeginAtom().GetAtomicNum() > 1 and x.GetEndAtom().GetAtomicNum() > 1)
    check("%s/%s swapped: matched by bond graph, the deposited molecule, longest bond %.2f A"
          % (a, b, longest),
          notes7 and "matched by bond graph; its atom names would bond" in notes7[0]
          and Chem.MolToSmiles(Chem.RemoveHs(mol7)) == smi and longest < 1.9 and not probs7,
          "%s %s %.2f" % (notes7[:1], probs7, longest))

print("\n[L-2] an incomplete cached entry is not built")
# A cache file cut short: before the bond loop (every atom its own radical fragment)
# or halfway through it. Before: built 'by atom name' at +5 or +2 with radicals and no
# problem reported, then an OpenFF radical error that named no file.
gdp_text = open(os.path.join(ROOT, "forcefield", "ccd_cache", "GDP.cif")).read()
lines_ = gdp_text.splitlines(True)
i_bond = next(i for i, l in enumerate(lines_) if l.startswith("_chem_comp_bond."))
loop_bond = max(i for i in range(i_bond) if lines_[i].startswith("loop_"))
bond_rows = [i for i in range(i_bond, len(lines_)) if not lines_[i].startswith(("_", "#", "loop_"))
             and len(lines_[i].split()) >= 4]
cut_cache = os.path.join(tmp, "cutcache")
os.makedirs(cut_cache)
real_default = C.DEFAULT_CACHE
C.DEFAULT_CACHE = cut_cache          # this cache only, so the shipped GDP.cif is not found first
try:
    for label, text in (("cut before the bond loop", "".join(lines_[:loop_bond])),
                        ("cut halfway through the bond loop", "".join(lines_[:bond_rows[len(bond_rows) // 2]]))):
        open(os.path.join(cut_cache, "GDP.cif"), "w").write(text)
        net_calls.clear()
        mol8, notes8, probs8 = build(gdp_pdb(GDP_1A2K), "GDP")
        rad = sum(x.GetNumRadicalElectrons() for x in mol8.GetAtoms())
        said = " ".join(notes8 + probs8)
        check("%s: not built from it (no radicals), and the reason names the file" % label,
              rad == 0 and "CCD entry GDP (" not in notes8[0] and os.path.join(cut_cache, "GDP.cif") in said
              and ("not a complete entry" in said or "is incomplete" in said), "%d radicals; %s" % (rad, said[:300]))
        check("  files.rcsb.org was asked for the complete entry", len(net_calls) == 1, str(net_calls))

    def serve_gdp(url, timeout=None):
        net_calls.append(url)
        return Resp(gdp_text)

    C.urllib.request.urlopen = serve_gdp
    open(os.path.join(cut_cache, "GDP.cif"), "w").write("".join(lines_[:bond_rows[len(bond_rows) // 2]]))
    mol9, notes9, probs9 = build(gdp_pdb(GDP_1A2K), "GDP")
    check("with network: the complete entry from files.rcsb.org, 40 atoms, -3, no problem",
          notes9 and "CCD entry GDP (files.rcsb.org, matched by atom name" in notes9[0] and mol9.GetNumAtoms() == 40
          and Chem.GetFormalCharge(mol9) == -3 and not probs9, "%s %s" % (notes9[:1], probs9))
finally:
    C.DEFAULT_CACHE = real_default
    C.urllib.request.urlopen = no_network

print("\n[L-6] the CCD identifier is the records' residue name, not -r")
# The runners take -r from the file name, which for a chain outside A-Z reads
# 'GDP_a'; the CCD lookup and the oxyanion completion then skipped the exact entry.
class Seen(Exception):
    pass


seen = {}
real_build, real_oxy = mod.build_molecule, mod.complete_oxyanion


def spy_oxy(text, resname):
    seen["oxy"] = resname
    return real_oxy(text, resname)


def spy_build(text, atoms, resname, ph):
    seen["build"] = resname
    raise Seen()


mod.complete_oxyanion, mod.build_molecule = spy_oxy, spy_build
text_a = gdp_pdb(GDP_1A2K).replace(" GDP C 220", " GDP a 220")
args_a = type("A", (), {"file": "ligand_GDP_a.pdb", "resname": "GDP_a", "chain": "a", "instance": 0,
                        "ph": 7.4, "ff": "openff-2.2.1.offxml"})()
try:
    mod.parametrize(args_a, text_a, mod.instance_of(text_a, "GDP_a", "a"))
except Seen:
    pass
finally:
    mod.complete_oxyanion, mod.build_molecule = real_oxy, real_build
check("chain 'a', -r GDP_a: the CCD lookup and the oxyanion rule are given GDP",
      seen == {"oxy": "GDP", "build": "GDP"}, str(seen))

print("\n[L1] a CCD build skips the geometric checks")
moved = [list(r) for r in GDP_1A2K]
c4, c5 = np.array(moved[10][2:]), np.array(moved[9][2:])
new5 = c4 + (c5 - c4) / np.linalg.norm(c5 - c4) * 1.38
moved[9][2:] = list(new5)
moved = [tuple(r) for r in moved]
mol5, notes5, probs5 = build(gdp_pdb(moved), "GDP")
check("C5'-C4' at 1.38 A: the CCD build passes", notes5 and "CCD entry GDP" in notes5[0] and not probs5, str(probs5))
full = mod.chemistry_problems(mol5, mod.pdb_atoms(gdp_pdb(moved)))
check("while the full checks on the same molecule report it",
      any("1.38 A short but joins two sp3 atoms" in p for p in full), str(full))

print("\n[V1] a radical entry stops the ligand; the entry itself decides 'incomplete'")
# A complete entry that built with radical electrons was taken for a cut cache file,
# files.rcsb.org's copy got the same verdict, and OpenBabel closed the shell: nitric
# oxide became [H]N=O, no problem reported, AM1-BCC charges, merged (v15_work review
# V1). The real NO entry from files.rcsb.org (2026-10-04) up to its bond record, with
# the trailing blanks stripped; the records after it are not read.
NO_CIF = (
    'data_NO\n'
    '#\n'
    '_chem_comp.id                                    NO\n'
    '_chem_comp.name                                  "NITRIC OXIDE"\n'
    '_chem_comp.type                                  NON-POLYMER\n'
    '_chem_comp.pdbx_type                             HETAIN\n'
    '_chem_comp.formula                               "N O"\n'
    '_chem_comp.mon_nstd_parent_comp_id               ?\n'
    '_chem_comp.pdbx_synonyms                         "Nitrogen monoxide"\n'
    '_chem_comp.pdbx_formal_charge                    0\n'
    '_chem_comp.pdbx_initial_date                     1999-07-08\n'
    '_chem_comp.pdbx_modified_date                    2024-09-27\n'
    '_chem_comp.pdbx_ambiguous_flag                   Y\n'
    '_chem_comp.pdbx_release_status                   REL\n'
    '_chem_comp.pdbx_replaced_by                      ?\n'
    '_chem_comp.pdbx_replaces                         NMO\n'
    '_chem_comp.formula_weight                        30.006\n'
    '_chem_comp.one_letter_code                       ?\n'
    '_chem_comp.three_letter_code                     NO\n'
    '_chem_comp.pdbx_model_coordinates_details        ?\n'
    '_chem_comp.pdbx_model_coordinates_missing_flag   N\n'
    '_chem_comp.pdbx_ideal_coordinates_details        Corina\n'
    '_chem_comp.pdbx_ideal_coordinates_missing_flag   N\n'
    '_chem_comp.pdbx_model_coordinates_db_code        1AOQ\n'
    '_chem_comp.pdbx_subcomponent_list                ?\n'
    '_chem_comp.pdbx_processing_site                  RCSB\n'
    '_chem_comp.pdbx_pcm                              N\n'
    '#\n'
    'loop_\n'
    '_chem_comp_atom.comp_id\n'
    '_chem_comp_atom.atom_id\n'
    '_chem_comp_atom.alt_atom_id\n'
    '_chem_comp_atom.type_symbol\n'
    '_chem_comp_atom.charge\n'
    '_chem_comp_atom.pdbx_align\n'
    '_chem_comp_atom.pdbx_aromatic_flag\n'
    '_chem_comp_atom.pdbx_leaving_atom_flag\n'
    '_chem_comp_atom.pdbx_stereo_config\n'
    '_chem_comp_atom.pdbx_backbone_atom_flag\n'
    '_chem_comp_atom.pdbx_n_terminal_atom_flag\n'
    '_chem_comp_atom.pdbx_c_terminal_atom_flag\n'
    '_chem_comp_atom.model_Cartn_x\n'
    '_chem_comp_atom.model_Cartn_y\n'
    '_chem_comp_atom.model_Cartn_z\n'
    '_chem_comp_atom.pdbx_model_Cartn_x_ideal\n'
    '_chem_comp_atom.pdbx_model_Cartn_y_ideal\n'
    '_chem_comp_atom.pdbx_model_Cartn_z_ideal\n'
    '_chem_comp_atom.pdbx_component_atom_id\n'
    '_chem_comp_atom.pdbx_component_comp_id\n'
    '_chem_comp_atom.pdbx_ordinal\n'
    'NO N N N 0 1 N N N N N N -1.718 17.207 81.431 -0.075 0.000 0.608  N NO 1\n'
    'NO O O O 0 1 N N N N N N -1.412 18.062 80.408 0.020  0.000 -0.605 O NO 2\n'
    '#\n'
    '_chem_comp_bond.comp_id              NO\n'
    '_chem_comp_bond.atom_id_1            N\n'
    '_chem_comp_bond.atom_id_2            O\n'
    '_chem_comp_bond.value_order          DOUB\n'
    '_chem_comp_bond.pdbx_aromatic_flag   N\n'
    '_chem_comp_bond.pdbx_stereo_config   N\n'
    '_chem_comp_bond.pdbx_ordinal         1\n'
    '#\n'
)
NO_PDB = ("HETATM    1  N   NO  A 401      10.000  10.000  10.000  1.00 20.00           N\n"
          "HETATM    2  O   NO  A 401      11.150  10.000  10.000  1.00 20.00           O\n"
          "END\n")


def outcome(text, resname):
    """'LigandFailure: <reason>', or 'built <SMILES> | <first note>'."""
    try:
        m, n, p = build(text, resname)
        return "built %s | %s" % (Chem.MolToSmiles(Chem.RemoveHs(m)), n[0] if n else "")
    except mod.LigandFailure as e:
        return "LigandFailure: %s" % e


def serve(body):
    def opener(url, timeout=None):
        net_calls.append(url)
        return Resp(body)
    return opener


v1 = os.path.join(tmp, "v1")
os.makedirs(v1)
open(os.path.join(v1, "NO.cif"), "w").write(NO_CIF)
shipped = os.path.join(ROOT, "forcefield", "ccd_cache")
judged = {}
for f in sorted(os.listdir(shipped)):
    if f.endswith(".cif"):
        judged[f[:-4]] = (mod.incomplete_entry(C.component(open(os.path.join(shipped, f)).read()))
                          if hasattr(mod, "incomplete_entry") else "no incomplete_entry()")
judged["NO"] = mod.incomplete_entry(C.component(NO_CIF)) if hasattr(mod, "incomplete_entry") else "no incomplete_entry()"
check("all %d shipped entries and NO are complete by the entry's own test" % (len(judged) - 1),
      len(judged) == 25 and all(v is None for v in judged.values()),
      str({k: v for k, v in judged.items() if v is not None}))
RADICAL_END = "OpenFF cannot parametrise a radical, and OpenBabel would build another molecule from the deposited atoms"
os.environ["GROSCORE_CCD_CACHE"] = v1
net_calls.clear()
try:
    got = outcome(NO_PDB, "NO")
finally:
    os.environ.pop("GROSCORE_CCD_CACHE", None)
check("NO from a cache directory, no network: LigandFailure naming the file as not checked",
      got == "LigandFailure: the CCD entry NO from local cache %s builds as a radical, 1 unpaired electron on N, and "
             "files.rcsb.org did not answer for NO (network disabled in this test), so the cached file could not be "
             "checked against the entry (where files.rcsb.org answers, remove %s and run 'python3 utils/ccd_cache.py "
             "-o %s <input PDB>', which fetches only what is missing); %s"
             % (os.path.join(v1, "NO.cif"), os.path.join(v1, "NO.cif"), v1, RADICAL_END), got)
check("  not handed to OpenBabel; files.rcsb.org asked once", "[H]N=O" not in got and len(net_calls) == 1, str(net_calls))
os.environ["GROSCORE_CCD_CACHE"] = v1
C.urllib.request.urlopen = serve(NO_CIF)
net_calls.clear()
try:
    got = outcome(NO_PDB, "NO")
finally:
    os.environ.pop("GROSCORE_CCD_CACHE", None)
    C.urllib.request.urlopen = no_network
check("NO cached and served by files.rcsb.org: 'a radical in local cache ... and at files.rcsb.org alike'",
      got == "LigandFailure: the CCD entry NO is a radical in local cache %s and at files.rcsb.org alike, 1 unpaired "
             "electron on N; %s" % (os.path.join(v1, "NO.cif"), RADICAL_END) and len(net_calls) == 1,
      "%s %s" % (got, net_calls))
C.urllib.request.urlopen = serve(NO_CIF)
net_calls.clear()
try:
    got = outcome(NO_PDB, "NO")
finally:
    C.urllib.request.urlopen = no_network
check("NO from files.rcsb.org: the same LigandFailure",
      got.startswith("LigandFailure: the CCD entry NO (files.rcsb.org) is a radical, 1 unpaired electron on N; ")
      and len(net_calls) == 1, "%s %s" % (got, net_calls))
# End to end: what the runners see is a ligand_failed.gs line and exit 1 (LIGAND_FAIL).
e2e = os.path.join(tmp, "v1_e2e")
os.makedirs(e2e)
open(os.path.join(e2e, "ligand_NO_A.pdb"), "w").write(NO_PDB)
os.environ["GROSCORE_CCD_CACHE"] = v1
cwd, argv, rc = os.getcwd(), sys.argv, None
os.chdir(e2e)
sys.argv = [PL, "-f", "ligand_NO_A.pdb", "-r", "NO", "-c", "A"]
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mod.main()
    rc = 0
except SystemExit as e:
    rc = e.code
finally:
    os.chdir(cwd)
    sys.argv = argv
    os.environ.pop("GROSCORE_CCD_CACHE", None)
failed_gs = os.path.join(e2e, "ligand_failed.gs")
line = open(failed_gs).read().strip() if os.path.isfile(failed_gs) else ""
check("parametrize_ligand.py: exit 1, 'NO A 401 the CCD entry NO ... builds as a radical' in ligand_failed.gs",
      rc == 1 and line.startswith("NO A 401 the CCD entry NO from local cache ")
      and "builds as a radical, 1 unpaired electron on N" in line and "could not be checked" in line,
      "rc %s, %r" % (rc, line[:200]))
check("  and no manifest line, so merge_ligand.py has nothing to merge",
      not os.path.isfile(os.path.join(e2e, "ligand_manifest.gs")), str(sorted(os.listdir(e2e))))
# The entry decides 'incomplete'. AYE with the N1-HN1A bond cut: HN1A has no bond.
AYE_CUT = AYE_CIF.replace("AYE N1 HN1A SING\n", "")
aye_cut = os.path.join(tmp, "aye_cut")
os.makedirs(aye_cut)
open(os.path.join(aye_cut, "AYE.cif"), "w").write(AYE_CUT)
cut_said = "the CCD entry AYE from local cache %s is incomplete (1 of its 11 atoms has no bond: HN1A)" % (
    os.path.join(aye_cut, "AYE.cif"))
os.environ["GROSCORE_CCD_CACHE"] = aye_cut
try:
    C.urllib.request.urlopen = serve(AYE_CUT)
    net_calls.clear()
    got = outcome(AYE_5E6J + "END\n", "AYE")
    check("cut in the cache and at files.rcsb.org alike: the entry itself, LigandFailure",
          got == "LigandFailure: " + cut_said + "; files.rcsb.org serves it incomplete in the same way, so that is "
          "the entry itself; not built, as OpenBabel would build another molecule from the deposited atoms"
          and len(net_calls) == 1, "%s %s" % (got, net_calls))
    C.urllib.request.urlopen = serve(AYE_CIF)
    got = outcome(AYE_5E6J + "END\n", "AYE")
    check("cut in the cache only: the complete entry from files.rcsb.org, C=CC[NH3+]",
          got.startswith("built C=CC[NH3+] | bond orders, charges and hydrogens from the CCD entry AYE (files.rcsb.org,"), got)
    check("  and the itp note names the damaged cached file",
          got.endswith("; it replaced the damaged local cache %s, which is incomplete (1 of its 11 atoms has no bond: "
                       "HN1A)" % os.path.join(aye_cut, "AYE.cif")), got)
    C.urllib.request.urlopen = no_network
    got = outcome(AYE_5E6J + "END\n", "AYE")
    check("cut in the cache, no network: OpenBabel, and its note names the file and the request",
          got.startswith("built C=CC[NH3+] | bond orders from OpenBabel (" + cut_said + "; files.rcsb.org did not answer for AYE"),
          got)
finally:
    C.urllib.request.urlopen = no_network
    os.environ.pop("GROSCORE_CCD_CACHE", None)

print("\n[N2] files.rcsb.org serves an incomplete entry, and the cache holds none or another cut")
# No AYE anywhere in the cache: only files.rcsb.org's copy, with HN1A left without a
# bond. Any incomplete copy there stops the ligand; letting it pass to OpenBabel built
# C=CC[NH3+] with no flag and the whole [V1] section still passed (sabotage V1f).
INCOMPLETE_END = "; not built, as OpenBabel would build another molecule from the deposited atoms"
C.urllib.request.urlopen = serve(AYE_CUT)
net_calls.clear()
try:
    got = outcome(AYE_5E6J + "END\n", "AYE")
finally:
    C.urllib.request.urlopen = no_network
check("no cached copy: 'the CCD entry AYE at files.rcsb.org is incomplete (... HN1A)', one request",
      got == "LigandFailure: the CCD entry AYE at files.rcsb.org is incomplete (1 of its 11 atoms has no bond: HN1A)"
      + INCOMPLETE_END and len(net_calls) == 1, "%s %s" % (got, net_calls))
aye_cut2 = os.path.join(tmp, "aye_cut2")
os.makedirs(aye_cut2)
open(os.path.join(aye_cut2, "AYE.cif"), "w").write(AYE_CIF.replace("AYE C1 H1A SING\n", ""))
os.environ["GROSCORE_CCD_CACHE"] = aye_cut2
C.urllib.request.urlopen = serve(AYE_CUT)
net_calls.clear()
try:
    got = outcome(AYE_5E6J + "END\n", "AYE")
finally:
    C.urllib.request.urlopen = no_network
    os.environ.pop("GROSCORE_CCD_CACHE", None)
check("cached copy cut at C1-H1A, files.rcsb.org's at N1-HN1A: both named, LigandFailure",
      got == "LigandFailure: the CCD entry AYE from local cache %s is incomplete (1 of its 11 atoms has no bond: H1A); "
             "the CCD entry AYE at files.rcsb.org is incomplete (1 of its 11 atoms has no bond: HN1A)%s"
      % (os.path.join(aye_cut2, "AYE.cif"), INCOMPLETE_END) and len(net_calls) == 1, "%s %s" % (got, net_calls))

print("\n[N1] a cached entry that builds as a radical is checked against files.rcsb.org")
# GDP.cif cut just before its last bond row: every atom keeps a bond, so the entry is
# complete by its own test, and the GDP builds with radicals on N3 and C4. It was
# reported as 'the CCD entry GDP ... is a radical' and the ligand stopped, although
# files.rcsb.org serves the whole entry (verifier N1; 12 of 4,582 cuts behave so).
i_row = i_bond
while lines_[i_row].startswith("_chem_comp_bond."):
    i_row += 1
while not lines_[i_row + 1].startswith("#"):
    i_row += 1                       # the last row of the bond loop
last_cut = "".join(lines_[:i_row])
check("the cut: valid, complete by the entry's test, last bond row 'GDP N3 C4 SING N N 45' gone",
      C.valid_entry(last_cut, "GDP") and mod.incomplete_entry(C.component(last_cut)) is None
      and lines_[i_row].split() == ["GDP", "N3", "C4", "SING", "N", "N", "45"]
      and len(C.component(last_cut)["bonds"]) == 44, lines_[i_row])
n1 = os.path.join(tmp, "n1")
os.makedirs(n1)
open(os.path.join(n1, "GDP.cif"), "w").write(last_cut)
real_default = C.DEFAULT_CACHE
C.DEFAULT_CACHE = n1                 # this cache only, so the shipped GDP.cif is not found first
try:
    C.urllib.request.urlopen = serve(gdp_text)
    net_calls.clear()
    try:
        m10, notes10, probs10 = build(gdp_pdb(GDP_1A2K), "GDP")
    except mod.LigandFailure as e:
        m10, notes10, probs10 = None, ["LigandFailure: %s" % e], ["raised"]
    check("with files.rcsb.org serving GDP: built from its copy, 40 atoms, -3, no problem, one request",
          m10 is not None and notes10[0].startswith("bond orders, charges and hydrogens from the CCD entry GDP "
                                                    "(files.rcsb.org, matched by atom name") and m10.GetNumAtoms() == 40
          and Chem.GetFormalCharge(m10) == -3 and not probs10 and len(net_calls) == 1,
          "%s %s %s" % (notes10[:1], probs10, net_calls))
    check("  the itp note names the damaged cached file and its radicals",
          notes10 and notes10[0].endswith("; it replaced the damaged local cache %s, which builds as a radical (2 unpaired "
                                          "electrons on N3, C4)" % os.path.join(n1, "GDP.cif")), notes10[:1])
    C.urllib.request.urlopen = no_network
    net_calls.clear()
    got = outcome(gdp_pdb(GDP_1A2K), "GDP")
    # The prefetch keeps a cached file it can read as an entry, so the failure says how
    # to fetch it again (fusion review F9): remove that file and prefetch without
    # --refresh, which would overwrite every valid entry of the input (verifier V2).
    check("with no network: LigandFailure naming the cached file as not checked, and how to fetch it again",
          got == "LigandFailure: the CCD entry GDP from local cache %s builds as a radical, 2 unpaired electrons on N3, C4, "
                 "and files.rcsb.org did not answer for GDP (network disabled in this test), so the cached file could not "
                 "be checked against the entry (where files.rcsb.org answers, remove %s and run 'python3 "
                 "utils/ccd_cache.py -o %s <input PDB>', which fetches only what is missing); %s"
                 % (os.path.join(n1, "GDP.cif"), os.path.join(n1, "GDP.cif"), n1, RADICAL_END)
          and len(net_calls) == 1, got)
finally:
    C.DEFAULT_CACHE = real_default
    C.urllib.request.urlopen = no_network

print("\n[V4] cache files: readable by others; an unreadable one is a reason, not a crash")
# prefetch wrote through mkstemp, whose files are 0600, so in a cache shared with other
# users they could not read the entries, and fetch() died of the PermissionError: the
# ligand ended as LIGAND_FAIL and parametrize_ncaa.py in a traceback (review V4).
C.urllib.request.urlopen = fake_urlopen
modes = []
try:
    for um in (0o022, 0o027):
        out4 = os.path.join(tmp, "mode%03o" % um)
        os.makedirs(out4)
        old = os.umask(um)
        try:
            C.prefetch(["1PE"], out4)
        finally:
            os.umask(old)
        modes.append("%04o" % stat.S_IMODE(os.stat(os.path.join(out4, "1PE.cif")).st_mode))
finally:
    C.urllib.request.urlopen = no_network
check("prefetch: 0644 under umask 022, 0640 under umask 027", modes == ["0644", "0640"], str(modes))
locked = os.path.join(tmp, "locked")
os.makedirs(locked)
p4 = os.path.join(locked, "GDP.cif")
shutil.copy(os.path.join(shipped, "GDP.cif"), p4)
os.chmod(p4, 0)
real_default = C.DEFAULT_CACHE
C.DEFAULT_CACHE = locked
net_calls.clear()
try:
    try:
        text, why = C.fetch("GDP")
        got = "%s | %s" % ("found" if text else "none", why)
    except Exception as e:
        got = "%s: %s" % (type(e).__name__, e)
    check("an unreadable cached GDP.cif: no exception, the reason names it, files.rcsb.org asked",
          got.startswith("none | GDP is not in the CCD cache and files.rcsb.org did not answer")
          and got.endswith("; the cached %s could not be read (Permission denied)" % p4) and len(net_calls) == 1, got)
    os.environ["GROSCORE_CCD_CACHE"] = shipped
    try:
        text, why = C.fetch("GDP")
        got = "%s | %s" % ("found" if text else "none", why)
    except Exception as e:
        got = "%s: %s" % (type(e).__name__, e)
    check("  and a readable copy further down the cache path is used", got == "found | " + os.path.join(shipped, "GDP.cif"),
          got)
finally:
    C.DEFAULT_CACHE = real_default
    os.environ.pop("GROSCORE_CCD_CACHE", None)
    os.chmod(p4, 0o644)

print("\n[N3] prefetch over a cached file it cannot read")
# Its keep/refresh read had no guard: a cached file of mode 0600 owned by another user
# (the L-10 code wrote them so) ended the prefetch CLI in a PermissionError traceback.
n3 = os.path.join(tmp, "n3")
os.makedirs(n3)
p3 = os.path.join(n3, "GDP.cif")
shutil.copy(os.path.join(shipped, "GDP.cif"), p3)
os.chmod(p3, 0)
C.urllib.request.urlopen = serve(gdp_text)
net_calls.clear()
try:
    try:
        fetched, kept, failed = C.prefetch(["GDP"], n3)
        got = "fetched %s kept %s failed %s" % (fetched, kept, failed)
    except Exception as e:
        got = "%s: %s" % (type(e).__name__, e)
    check("an unreadable cached GDP.cif: fetched again, no exception", got == "fetched ['GDP'] kept [] failed []"
          and len(net_calls) == 1, got)
    check("  and the file is the readable entry now", os.access(p3, os.R_OK) and open(p3).read() == gdp_text, "")
    os.chmod(p3, 0)
    os.chmod(n3, 0o555)
    try:
        fetched, kept, failed = C.prefetch(["GDP"], n3)
        got = "fetched %s kept %s failed %s" % (fetched, kept, failed)
    except Exception as e:
        got = "%s: %s" % (type(e).__name__, e)
    check("  in a directory it cannot write: a failure naming both, no exception",
          got == "fetched [] kept [] failed [('GDP', 'could not write %s (Permission denied); the cached %s could not "
                 "be read (Permission denied)')]" % (p3, p3) and not [f for f in os.listdir(n3) if f.endswith(".tmp")], got)
finally:
    os.chmod(n3, 0o755)
    os.chmod(p3, 0o644)
    C.urllib.request.urlopen = no_network

print("\n[parents] the prefetch fetches ASN, SER and THR with the HETATM residues")
only_water = os.path.join(tmp, "water.pdb")
open(only_water, "w").write("HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00 20.00           O\n"
                            "HETATM    2 MG    MG A   2       2.000   0.000   0.000  1.00 20.00          MG\n")
ids = C.prefetch_ids([inp]) if hasattr(C, "prefetch_ids") else "no prefetch_ids()"
check("prefetch_ids: the HETATM names and ASN SER THR", ids == ["1PE", "ASN", "GDP", "IOD", "SER", "SO4", "THR"], str(ids))
ids = C.prefetch_ids([only_water]) if hasattr(C, "prefetch_ids") else "no prefetch_ids()"
check("  none for an input with water and ions only", ids == [], str(ids))
check("  and forcefield/ccd_cache ships all three", all(
    C.fetch(x, network=False)[1] == os.path.join(shipped, x + ".cif") for x in ("ASN", "SER", "THR")), "")

shutil.rmtree(tmp, ignore_errors=True)
print("\n%s" % ("ALL PASS" if fails == 0 else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
