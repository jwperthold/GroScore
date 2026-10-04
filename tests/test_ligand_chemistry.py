#!/usr/bin/env python3
"""Ligand chemistry the v1.5 review found wrong after the first round (review L2, L5,
L6, L7, L8, L10, L12), pinned with fixed molecules and numbers:

  [L2]  OpenBabel perceives sulfonates and sulfate monoesters as S(OH)3-X (2XQR HEPES
        came out at +2, 4AQE MES at +1). oxo_sulfur() gives such a sulfur two S=O and
        the sulfur rule deprotonates the rest: CS(O)(O)O -> -1, COS(O)(O)O -> -1,
        S(OH)4 -> -2; the 4AQE MES deposit through OpenBabel ends with a sulfonate
        (two S=O, one O-, no S-OH). A leftover S(OH)3 is a check failure.
  [L5]  only sulfate and phosphate residue names are completed: SO3 (sulfite), SO2 and
        H2S are left as deposited; three tetrahedral oxygens are completed, three
        coplanar ones refused (|u1+u2+u3| 0.49 refused, 0.51 completed), a lone P
        refused with its own message.
  [L6]  P-X-P chains with no bridge leaving them keep one proton: PPi -3, PPPi -4,
        etidronate -3, methylenediphosphonate -3, imidodiphosphate -3; a phosphonate
        is -1 (methylphosphonic acid pKa2 7.7, the AMP-PCP gamma 8.4), so an AMP-PCP-
        like triphosphate ester is -3; unchanged: ATP-like -4, ADP-like -3, the GNP-like
        imido triphosphate -4, a phosphate monoester -2.
  [L8]  bicarbonate -1 (CO3 2- is protonated back), formate -1, a thiophosphate's
        terminal S counts like an O (GTPgammaS gamma -2, the whole -4), and an S-H on a
        three-coordinate sulfur is flagged (a sulfonium carries none).
  [L7]  select_altloc() keeps one conformer: the higher mean occupancy (5SWK SO4 B 201:
        B 0.54 over A 0.46), the first on a tie, every record without an altloc, and
        blanks column 17; a ligand without altlocs is returned unchanged.
  [L10] element_of() with blank element columns follows the PDB name alignment: CL1
        and BR1 in column 13 are Cl and Br, FE is Fe, ' CA ' is a carbon, 'CA  '
        calcium, HG21 a hydrogen, '1HG2' a hydrogen, CL12 chlorine, C12A carbon.
  [L12] outside the pH window the acid states are neither applied nor checked: the
        1S1Q sulfate through OpenBabel stays H2SO4 at pH 6.0 and 8.5 (the first round
        reported '0 of 2 O- expected at pH 7.4' and swapped in the -2 template).

In-process (parametrize_ligand.py imported; OpenBabel for the MES and pH cases, no
AM1-BCC, no network: the CCD lookup is switched off where OpenBabel is meant to run);
takes a few seconds.

Standalone, no pytest: python3 tests/test_ligand_chemistry.py
"""
import contextlib
import importlib.util
import io
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")   # UFF typer chatter about S(OH)3 during embedding
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


def hetatm(serial, name, resname, chain, resnum, xyz, el, alt=" ", occ=1.00):
    nm = (" " + name) if len(name) < 4 else name
    return "HETATM%5d %-4s%1s%3s %1s%4d    %8.3f%8.3f%8.3f%6.2f 20.00          %2s" % (
        serial, nm, alt, resname, chain, resnum, xyz[0], xyz[1], xyz[2], occ, el)


def pdb(resname, chain, resnum, atoms):
    return "\n".join(hetatm(i + 1, n, resname, chain, resnum, xyz, el) for i, (n, el, xyz) in enumerate(atoms)) + "\nEND\n"


def q_of(mod, smi, embed=True):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    if embed:
        AllChem.EmbedMolecule(m, randomSeed=3)
    m, _ = mod.oxo_sulfur(m) if hasattr(mod, "oxo_sulfur") else (m, [])
    m, _ = mod.ionize_acids(m)
    return m


mod = load()
if mod is None:
    print("parametrize_ligand.py cannot be imported")
    sys.exit(1)


def offline():
    """switch off the CCD lookup, so OpenBabel runs"""
    saved = {}
    if hasattr(mod, "ccd_cache"):
        saved["fetch"] = mod.ccd_cache.fetch
        mod.ccd_cache.fetch = lambda cid, network=True: (None, "%s unavailable in this test" % cid)
    return saved


def online(saved):
    if "fetch" in saved:
        mod.ccd_cache.fetch = saved.pop("fetch")


def build(text, resname, ph=7.4):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = mod.build_molecule(text, mod.pdb_atoms(text), resname, ph)
    return res, buf.getvalue()


print("\n[L2] sulfonates and sulfate esters perceived as S(OH)3")
for smi, q in (("CS(O)(O)O", -1), ("COS(O)(O)O", -1), ("OS(O)(O)O", -2), ("CS(=O)(=O)O", -1)):
    try:
        m = q_of(mod, smi)
        s = [a for a in m.GetAtoms() if a.GetAtomicNum() == 16][0]
        dbl = sum(1 for b in s.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        sh = sum(1 for n in s.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetTotalNumHs(includeNeighbors=True))
        check("%-14s -> %+d, two S=O, no S-O-H" % (smi, q), Chem.GetFormalCharge(m) == q and dbl == 2 and sh == 0,
              "q %d, S=O %d, S-OH %d" % (Chem.GetFormalCharge(m), dbl, sh))
    except Exception as e:
        check("%-14s -> %+d" % (smi, q), False, "%s: %s" % (type(e).__name__, e))
m = Chem.AddHs(Chem.MolFromSmiles("CS(O)(O)O"))
AllChem.EmbedMolecule(m, randomSeed=3)
X = m.GetConformer().GetPositions()
ref = [("A%d" % a.GetIdx(), a.GetSymbol().upper()) + tuple(X[a.GetIdx()]) for a in m.GetAtoms() if a.GetAtomicNum() > 1]
p = mod.chemistry_problems(m, ref, geometry=False) if "geometry" in mod.chemistry_problems.__code__.co_varnames \
    else mod.chemistry_problems(m, ref)
check("a sulfur left as S(OH)3 is a check failure", any("no S=O" in s for s in p), str(p))
# 4AQE MES A 1211, deposited
MES_4AQE = [("O1", "O", (31.366, -17.952, 42.150)), ("C2", "C", (32.212, -17.845, 43.298)),
            ("C3", "C", (33.160, -16.655, 43.196)), ("N4", "N", (33.922, -16.703, 41.931)),
            ("C5", "C", (32.945, -16.610, 40.825)), ("C6", "C", (32.037, -17.838, 40.885)),
            ("C7", "C", (34.967, -15.649, 41.961)), ("C8", "C", (35.247, -15.046, 40.599)),
            ("S", "S", (36.837, -14.708, 40.349)), ("O1S", "O", (36.932, -13.603, 39.402)),
            ("O2S", "O", (37.498, -15.878, 39.781)), ("O3S", "O", (37.602, -14.313, 41.550))]
saved = offline()
try:
    (mol, notes, probs), out = build(pdb("MES", "A", 1211, MES_4AQE), "MES")
    s = mol.GetAtomWithIdx(8)
    dbl = sum(1 for b in s.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    anion = sum(1 for n in s.GetNeighbors() if n.GetFormalCharge() == -1)
    soh = sum(1 for n in s.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetTotalNumHs(includeNeighbors=True))
    check("4AQE MES through OpenBabel: a sulfonate (two S=O, one O-, no S-OH), no problem",
          dbl == 2 and anion == 1 and soh == 0 and not probs, "S=O %d O- %d S-OH %d %s" % (dbl, anion, soh, probs))
except Exception as e:
    check("4AQE MES through OpenBabel", False, "%s: %s" % (type(e).__name__, e))
finally:
    online(saved)

print("\n[L5] oxyanion completion follows the residue name")
SO3 = [("S", "S", (0.0, 0.0, 0.0)), ("O1", "O", (1.48, 0.0, 0.0)), ("O2", "O", (-0.74, 1.28, 0.0)),
       ("O3", "O", (-0.74, -1.28, 0.0))]


def complete(text, resname):
    try:
        out, note = mod.complete_oxyanion(text, resname)
        return len([a for a in mod.pdb_atoms(out)]), note
    except mod.LigandFailure as e:
        return None, str(e)
    except TypeError as e:
        return None, "TypeError %s" % e


for resname, atoms, want in (("SO3", SO3, 4), ("SO2", SO3[:3], 3), ("H2S", SO3[:1], 1)):
    n, note = complete(pdb(resname, "A", 1, atoms), resname)
    check("%s (%d atoms) is left as deposited" % (resname, len(atoms)), n == want and note is None, "%s %s" % (n, note))
# three tetrahedral oxygens: the fourth is placed
th = [(0.0, 0.0, 0.0)] + [tuple(1.5 * v for v in u) for u in ((0.0, 0.9428, -0.3333), (0.8165, -0.4714, -0.3333),
                                                             (-0.8165, -0.4714, -0.3333))]
n, note = complete(pdb("SO4", "A", 1, [("S", "S", th[0])] + [("O%d" % (k + 1), "O", th[k + 1]) for k in range(3)]), "SO4")
check("sulfate with three tetrahedral O: completed to five atoms", n == 5 and note and "completed sulfate" in note,
      "%s %s" % (n, note))
# three oxygens at polar angle theta: |u1 + u2 + u3| = 3 |cos theta|
for norm, ok in ((0.49, False), (0.51, True)):
    c = norm / 3.0
    s = (1.0 - c * c) ** 0.5
    oxs = [(1.5 * s * cs, 1.5 * s * sn, -1.5 * c) for cs, sn in ((1.0, 0.0), (-0.5, 0.8660254), (-0.5, -0.8660254))]
    n, note = complete(pdb("PO4", "A", 1, [("P", "P", (0.0, 0.0, 0.0))] + [("O%d" % (k + 1), "O", o) for k, o in enumerate(oxs)]), "PO4")
    check("phosphate with three O, |u1+u2+u3| = %.2f: %s" % (norm, "completed" if ok else "refused"),
          (n == 5) if ok else (n is None and "coplanar" in (note or "")), "%s %s" % (n, note))
n, note = complete(pdb("PO4", "A", 1, [("P", "P", (0.0, 0.0, 0.0))]), "PO4")
check("a lone P named PO4 is refused with '0 of 4 oxygens'", n is None and "0 of 4" in (note or ""), "%s %s" % (n, note))

print("\n[L6] polyphosphates, bisphosphonates, phosphonates")
for smi, q, what in (("OP(=O)(O)OP(=O)(O)O", -3, "pyrophosphate"),
                     ("OP(=O)(O)OP(=O)(O)OP(=O)(O)O", -4, "triphosphate"),
                     ("CC(O)(P(=O)(O)O)P(=O)(O)O", -3, "etidronate"),
                     ("OP(=O)(O)CP(=O)(O)O", -3, "methylenediphosphonate"),
                     ("OP(=O)(O)NP(=O)(O)O", -3, "imidodiphosphate"),
                     ("CP(=O)(O)O", -1, "methylphosphonic acid"),
                     ("COP(=O)(O)OP(=O)(O)CP(=O)(O)O", -3, "AMP-PCP-like ester"),
                     ("COP(=O)(O)OP(=O)(O)OP(=O)(O)O", -4, "ATP-like ester"),
                     ("COP(=O)(O)OP(=O)(O)O", -3, "ADP-like ester"),
                     ("COP(=O)(O)OP(=O)(O)NP(=O)(O)O", -4, "GNP-like imido ester (kept at -4)"),
                     ("CCOP(=O)(O)O", -2, "phosphate monoester")):
    try:
        m = q_of(mod, smi, embed=False)
        check("%-24s %-32s -> %+d" % (what, smi, q), Chem.GetFormalCharge(m) == q, str(Chem.GetFormalCharge(m)))
    except Exception as e:
        check("%-24s -> %+d" % (what, q), False, "%s: %s" % (type(e).__name__, e))

print("\n[L8] carbonate, formate, thiophosphate, sulfonium")
for smi, q, what in (("OC(=O)O", -1, "carbonic acid -> HCO3-"), ("[O-]C(=O)[O-]", -1, "carbonate -> HCO3-"),
                     ("OC=O", -1, "formic acid -> formate"),
                     ("COP(=O)(O)OP(=O)(O)OP(=S)(O)O", -4, "GTPgammaS-like, P=S"),
                     ("COP(=O)(O)OP(=O)(O)OP(=O)(O)S", -4, "GTPgammaS-like, P-SH")):
    try:
        m = q_of(mod, smi, embed=False)
        check("%-30s -> %+d" % (what, q), Chem.GetFormalCharge(m) == q, str(Chem.GetFormalCharge(m)))
    except Exception as e:
        check("%-30s -> %+d" % (what, q), False, "%s: %s" % (type(e).__name__, e))
for smi, bad in (("C[SH](C)C", True), ("C[S+](C)C", False), ("CSC", False)):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=3)
    X = m.GetConformer().GetPositions()
    ref = [("A%d" % a.GetIdx(), a.GetSymbol().upper()) + tuple(X[a.GetIdx()]) for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    p = [s for s in mod.chemistry_problems(m, ref) if "three-coordinate S" in s]
    check("%-10s: %s" % (smi, "S-H on a three-coordinate S flagged" if bad else "accepted"), bool(p) == bad, str(p))

print("\n[L7] one conformer of an altloc ligand")
SO4_5SWK = [("S", "S", "A", 0.46, (56.145, 9.888, 28.158)), ("S", "S", "B", 0.54, (55.240, 11.989, 26.199)),
            ("O1", "O", "A", 0.46, (56.165, 10.540, 26.847)), ("O1", "O", "B", 0.54, (55.292, 11.398, 27.533)),
            ("O2", "O", "A", 0.46, (56.351, 8.458, 28.012)), ("O2", "O", "B", 0.54, (53.951, 11.688, 25.575)),
            ("O3", "O", "A", 0.46, (57.220, 10.432, 28.988)), ("O3", "O", "B", 0.54, (56.310, 11.426, 25.385)),
            ("O4", "O", "A", 0.46, (54.863, 10.125, 28.819)), ("O4", "O", "B", 0.54, (55.410, 13.435, 26.308))]


def alt_pdb(rows):
    return "\n".join(hetatm(i + 1, n, "SO4", "B", 201, xyz, el, alt, occ)
                     for i, (n, el, alt, occ, xyz) in enumerate(rows)) + "\nEND\n"


if hasattr(mod, "select_altloc"):
    text, note = mod.select_altloc(alt_pdb(SO4_5SWK))
    kept = mod.pdb_atoms(text)
    recs = [l for l in text.splitlines() if l.startswith("HETATM")]
    check("5SWK SO4: altloc B (0.54) kept, 5 atoms, column 17 blank",
          len(kept) == 5 and abs(kept[0][2] - 55.240) < 1e-6 and all(l[16] == " " for l in recs), str(kept[:1]))
    check("and the note says so", note is not None and "B kept" in note and "A dropped" in note, str(note))
    tie = [(n, el, alt, 0.50, xyz) for n, el, alt, occ, xyz in SO4_5SWK]
    text, note = mod.select_altloc(alt_pdb(tie))
    check("a tie keeps the first altloc in the file (A)", abs(mod.pdb_atoms(text)[0][2] - 56.145) < 1e-6, str(note))
    mixed = [("S", "S", " ", 1.0, (56.145, 9.888, 28.158))] + [r for r in SO4_5SWK if r[0] != "S"]
    text, note = mod.select_altloc(alt_pdb(mixed))
    check("records without an altloc stay (S plus the four O of B)", len(mod.pdb_atoms(text)) == 5, str(len(mod.pdb_atoms(text))))
    plain = pdb("SO4", "B", 201, [(n, el, xyz) for n, el, alt, occ, xyz in SO4_5SWK if alt == "A"])
    text, note = mod.select_altloc(plain)
    check("a ligand without altlocs comes back unchanged, no note", text == plain and note is None)
else:
    check("select_altloc() exists", False)

print("\n[L10] the element when columns 77-78 are blank")


def rec(field):
    return "HETATM    1 %-4s EF6 A   1       0.000   0.000   0.000  1.00 20.00" % field


if hasattr(mod, "element_of"):
    for field, el in (("CL1 ", "CL"), ("BR1 ", "BR"), ("FE  ", "FE"), (" CA ", "C"), ("CA  ", "CA"), ("HG21", "H"),
                      ("1HG2", "H"), ("CL12", "CL"), ("C12A", "C"), (" N1 ", "N"), (" O5'", "O")):
        check("%-6r -> %s" % (field, el), mod.element_of(rec(field)) == el, mod.element_of(rec(field)))
    full = rec("CL1 ") + "          CL"
    check("columns 77-78 win when present (CL1 with element Cl)", mod.element_of(full) == "CL", mod.element_of(full))
check("pdb_atoms reads CL1 without element columns as chlorine",
      mod.pdb_atoms(rec("CL1 ") + "\n")[0][1] == "CL", str(mod.pdb_atoms(rec("CL1 ") + "\n")))

print("\n[L12] outside the pH window the acid states are neither applied nor checked")
SO4_1S1Q = [("S", "S", (103.766, 32.676, 62.630)), ("O1", "O", (104.417, 31.475, 62.123)),
            ("O2", "O", (102.792, 33.195, 61.670)), ("O3", "O", (103.039, 32.358, 63.869)),
            ("O4", "O", (104.802, 33.686, 62.874))]
saved = offline()
try:
    for ph in (6.0, 8.5):
        (mol, notes, probs), out = build(pdb("SO4", "A", 503, SO4_1S1Q), "SO4", ph)
        check("SO4 through OpenBabel at pH %.1f: no acid-state problem" % ph,
              not any("expected at pH" in s for s in probs), str(probs))
        check("and it keeps OpenBabel's neutral H2SO4 (no template swapped in)",
              Chem.GetFormalCharge(mol) == 0 and notes and notes[0].startswith("bond orders from OpenBabel"),
              "%d %s" % (Chem.GetFormalCharge(mol), notes[:1]))
        check("and the warning says the rules are neither applied nor checked",
              "neither applied nor checked" in out, out[:200])
    (mol, notes, probs), out = build(pdb("SO4", "A", 503, SO4_1S1Q), "SO4", 7.4)
    check("at pH 7.4 the same sulfate is SO4 2-", Chem.GetFormalCharge(mol) == -2 and not probs,
          "%d %s" % (Chem.GetFormalCharge(mol), probs))
except Exception as e:
    check("pH window cases", False, "%s: %s" % (type(e).__name__, e))
finally:
    online(saved)

print("\n%s" % ("ALL PASS" if fails == 0 else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
