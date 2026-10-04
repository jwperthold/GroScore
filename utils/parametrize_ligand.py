#!/usr/bin/env python3
#
# parametrize_ligand.py - Parametrize a small molecule with OpenFF for GROMACS
#
# Reads a ligand from PDB HETATM records, takes its bond orders from the wwPDB
# Chemical Component Dictionary (CCD) entry of its residue name when that entry
# describes exactly the deposited heavy atoms (OpenBabel's perception otherwise),
# assigns protonation at physiological pH, parametrizes with OpenFF, and exports
# GROMACS-compatible topology (.itp) and coordinate (.gro) files.
#
# Requires: openbabel, rdkit, openff-toolkit, openff-interchange (groscore conda env)
#
# Usage: python parametrize_ligand.py -f ligand_EF6_A.pdb -r EF6
#
# NO LIGAND IS DROPPED OR MANGLED SILENTLY. Every extracted instance this script
# cannot build, or builds knowing that the molecule is not the deposited one, gets
# one line in ligand_failed.gs:
#
#     RESNAME CHAIN RESNUM reason text
#
# (CHAIN "-" when the PDB record has none). job_fe.run and job.run stop the structure
# with LIGAND_FAIL on any line, and merge_ligand.py reconciles the file against what
# actually reached topol.top. The line is written FIRST, as "did not finish", and
# only replaced or removed once the outcome is known, so a run killed by a timeout
# or by the OOM killer still leaves its mark.
#
# Why this exists: on the 46-structure HADDOCK FE pass of 2026-09 this script
# dropped six ligand instances in five structures without a word that reached any
# status file (1KXP ATP and 1HE8 GNP to an sqm failure, the AlF3 of 1WQ1 and 1GRN
# to a missing Al-F bond term, the two iodides of 1J2J built as H-I), and built ten
# sulfates as neutral H2SO4, the FAD of 1E6E at charge 0, the GNP of 1IBR as a
# 50-atom sp3 guanine, the retinoid of 1RLB as a saturated C20 alcohol, the
# phosphate of 2MTA as HP(OH)2 and two glycol additives of 2FD6 as enol ethers.
# The AlF3 errors alone cost 1GRN +5.0 and 1WQ1 +4.8 pKD.
#
# WHERE THE CHEMISTRY COMES FROM, in this order (build_molecule()):
#   1. The CCD entry named by the residue name, from the local cache first
#      (forcefield/ccd_cache, see utils/ccd_cache.py) and files.rcsb.org second,
#      when its heavy atoms are exactly the deposited ones: by atom name, or, for
#      renamed atoms, by an identical bond graph. Its bond orders, formal charges and
#      hydrogens are the depositor's own definition of the molecule. OpenBabel instead
#      guesses them from the geometry: it put the guanine H of every GDP, GTP and GNP
#      it built for the HADDOCK benchmark on N3 instead of N1 (review L3), built HEPES
#      and MES sulfonates as S(OH)3 (L2) and formate as methanediol (L8). The geometric
#      bond-order checks are not applied to a CCD build: on 1400 ligand instances of 172
#      PDB entries they flagged 12 correct molecules (L1). An entry that describes the
#      deposited atoms but is a radical (nitric oxide), or is incomplete at
#      files.rcsb.org as well, stops the ligand: OpenFF cannot parametrise it, and
#      OpenBabel would build another molecule from the same atoms (review V1). A cached
#      copy that is incomplete or builds as a radical is first checked against
#      files.rcsb.org's, which is used when it builds (review N1).
#   2. OpenBabel's perception from the deposited coordinates (v1.4's strategy), which
#      must then pass every check in chemistry_problems().
# On every path the hydrogens are OpenBabel's pH model applied to the chosen bond
# orders, exactly as v1.4 protonated OpenBabel's own perception, followed by the
# explicit pH 7.4 acid rules below. The itp header names the source, the CCD entry's
# modification date and where it was read from.
#

import os
import sys
import argparse
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccd_cache  # noqa: E402  (standard library only)

LIGAND_FAILED = "ligand_failed.gs"
MANIFEST = "ligand_manifest.gs"
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class LigandFailure(Exception):
    """A ligand this script refuses to build; the message is the ligand_failed.gs reason."""


# ---------------------------------------------------------------------------
# The failure record and the PDB records. Standard library only: this has to work
# before the heavy imports, which are themselves among the things that can fail,
# and merge_ligand.py imports it.
# ---------------------------------------------------------------------------

# Two-letter element symbols. A PDB atom name puts a one-letter element in column 14
# (" CA " is a carbon) and a two-letter one in column 13 ("CA  " is calcium, "CL1 " a
# chlorine); when columns 77-78 are blank that alignment is all that tells them
# apart. The old first-letter fallback read CL1 as carbon and FE as fluorine, so 6UD7
# EF6 without element columns failed the composition check as C14Cl1N3O4S2 against
# C15N3O4S2 (review L10). Four-character names start in column 13 whatever their
# element, so there only CL and BR are read as two letters (HG21 is a hydrogen, not
# mercury).
TWO_LETTER_ELEMENTS = {
    "AG", "AL", "AS", "AU", "BA", "BE", "BI", "BR", "CA", "CD", "CE", "CL", "CO", "CR", "CS", "CU",
    "DY", "ER", "EU", "FE", "GA", "GD", "GE", "HF", "HG", "HO", "IN", "IR", "KR", "LA", "LI", "LU",
    "MG", "MN", "MO", "NA", "NI", "OS", "PB", "PD", "PR", "PT", "RB", "RE", "RH", "RU", "SB", "SC",
    "SE", "SI", "SM", "SN", "SR", "TA", "TB", "TE", "TI", "TL", "TM", "XE", "YB", "ZN", "ZR"}


def element_of(line):
    """Element of an ATOM/HETATM record: columns 77-78, else the PDB name alignment."""
    el = line[76:78].strip().upper() if len(line) >= 78 else ""
    if not el:
        field = (line[12:16] + "    ")[:4]
        name = field.strip()
        if not name:
            return ""
        if field[0] == " " or field[0].isdigit():      # one-letter element, or '1HG2'-style H
            el = "".join(c for c in name if c.isalpha())[:1].upper()
        elif name[:2].upper() in TWO_LETTER_ELEMENTS and (len(name) < 4 or name[:2].upper() in ("CL", "BR")):
            el = name[:2].upper()
        else:
            el = name[0].upper()
    return "H" if el == "D" else el


def pdb_atoms(pdb_text):
    """[(name, ELEMENT, x, y, z)] of every ATOM/HETATM record; coordinates in A."""
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append((line[12:16].strip(), element_of(line),
                      float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return atoms


def select_altloc(pdb_text):
    """(pdb_text, note): one conformer of a ligand deposited with alternate locations.

    renumber_pdb.py writes every altloc record into ligand_*.pdb, and OpenBabel then
    bonded the copies into one molecule that the composition check, comparing against
    the same doubled atom list, accepted: 5SWK SO4 B 201 became two S(OH)3 joined by an
    O-O bond, 4Y8D EDO C 202 an 18-atom 'C=C(O)CO.COO' (review L7). Kept: the records
    without an altloc and those of the altloc with the highest mean occupancy, the first
    in the file on a tie, with column 17 blanked. note is None when there is no altloc."""
    lines = pdb_text.splitlines()
    occ, order = {}, []
    for l in lines:
        if l.startswith(("ATOM  ", "HETATM")) and len(l) > 16 and l[16] != " ":
            if l[16] not in occ:
                occ[l[16]] = []
                order.append(l[16])
            try:
                occ[l[16]].append(float(l[54:60]))
            except ValueError:
                occ[l[16]].append(0.0)
    if not order:
        return pdb_text, None
    mean = {a: sum(v) / len(v) for a, v in occ.items()}
    best = order[0]
    for a in order[1:]:
        if mean[a] > mean[best] + 1e-6:
            best = a
    out = []
    for l in lines:
        if l.startswith(("ATOM  ", "HETATM")) and len(l) > 16 and l[16] != " ":
            if l[16] != best:
                continue
            l = l[:16] + " " + l[17:]
        out.append(l)
    dropped = [a for a in order if a != best]
    note = "alternate location %s kept (occupancy %.2f)%s" % (
        best, mean[best], "; %s dropped" % "/".join(dropped) if dropped else "")
    return "\n".join(out) + "\n", note


def instance_of(pdb_text, resname="", chain=""):
    """(RESNAME, CHAIN, RESNUM) of the ligand instance, read from its first record.

    These three fields are exactly what renumber_pdb.py keys an instance by, so
    they name it uniquely. A blank chain is written as '-' to keep the
    ligand_failed.gs line at four or more whitespace-separated fields."""
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            return (line[17:20].strip() or resname or "UNK",
                    line[21:22].strip() or chain.strip() or "-",
                    line[22:26].strip() or "0")
    return (resname or "UNK", chain.strip() or "-", "0")


def record_failure(path, inst, reason):
    """Set this instance's line in `path` to `reason`; a reason of None removes it.

    Lines of other instances are kept. An empty result removes the file, since
    "absent or empty" is what tells the runners that every ligand is fine."""
    key = "%s %s %s " % inst
    lines = []
    if os.path.isfile(path):
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip() and not l.startswith(key)]
    if reason:
        lines.append(key + " ".join(str(reason).split()))
    if lines:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
    elif os.path.isfile(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Inorganic oxyanions with oxygens missing from the deposited model
# ---------------------------------------------------------------------------

# Sulfate and phosphate are tetrahedral. The 2MTA phosphate sits on a
# crystallographic two-fold axis, so the entry holds P and two oxygens and the other
# two are symmetry mates: OpenBabel then built HP(OH)2, a molecule that is not in the
# crystal. The missing oxygens follow from the present ones (on an ideal tetrahedron
# c + d = -(a + b) and c - d lies along a x b), and for 2MTA they land on the two-fold
# images of the deposited ones. Only residues whose CCD entry IS that ion are
# completed: the first version completed any lone S or P with two or three oxygens,
# and rebuilt sulfite (SO3) and SO2 as sulfate (review L5).
OXYANION_IDS = {"SO4": "sulfate", "SUL": "sulfate", "PO4": "phosphate", "PI": "phosphate",
                "2HP": "phosphate", "IPS": "phosphate"}
OXYANION_BOND = (1.35, 1.75)   # A, accepted S-O / P-O range before completing
# Three oxygens fix the fourth only when they are not coplanar with the centre: on the
# ideal tetrahedron |u1 + u2 + u3| = 1, for a planar PO3 it is 0 and its direction is
# noise (the fourth O landed exactly on top of O1).
OXYANION_TRIAD_MIN = 0.5


def complete_oxyanion(pdb_text, resname):
    """(pdb_text, note): a sulfate or phosphate residue with 2 or 3 of its 4 oxygens is
    completed on the ideal tetrahedron; anything else is returned unchanged."""
    kind = OXYANION_IDS.get(resname.strip().upper())
    if kind is None:
        return pdb_text, None
    import numpy as np
    recs = [l for l in pdb_text.splitlines() if l.startswith(("ATOM  ", "HETATM"))]
    atoms = pdb_atoms(pdb_text)
    heavy = [a for a in atoms if a[1] != "H"]
    cel = "S" if kind == "sulfate" else "P"
    centres = [a for a in heavy if a[1] == cel]
    if len(centres) != 1 or any(a[1] not in (cel, "O") for a in heavy):
        return pdb_text, None       # not the ion its name says; the checks judge what it is
    nox = sum(1 for a in heavy if a[1] == "O")
    if nox == 4:
        return pdb_text, None
    if nox not in (2, 3):
        raise LigandFailure("incomplete %s: %d of 4 oxygens deposited, too few to place the rest" % (kind, nox))
    c = np.array(centres[0][2:])
    ox = [np.array(a[2:]) for a in heavy if a[1] == "O"]
    lens = [np.linalg.norm(o - c) for o in ox]
    if not all(OXYANION_BOND[0] <= d <= OXYANION_BOND[1] for d in lens):
        raise LigandFailure("incomplete %s: %d of 4 oxygens deposited and they are not bonded to %s "
                            "(%s A)" % (kind, nox, cel, ", ".join("%.2f" % d for d in lens)))
    u = [(o - c) / np.linalg.norm(o - c) for o in ox]
    if nox == 3:
        s = u[0] + u[1] + u[2]
        if np.linalg.norm(s) < OXYANION_TRIAD_MIN:
            raise LigandFailure("incomplete %s: the three deposited oxygens are coplanar with %s "
                                "(|u1+u2+u3| = %.2f), so the fourth has no place" % (kind, cel, np.linalg.norm(s)))
        new = [-s / np.linalg.norm(s)]
    else:
        s = -(u[0] + u[1]) / 2.0
        n = np.cross(u[0], u[1])
        if np.linalg.norm(n) < 1e-3 or np.dot(s, s) >= 1.0:
            raise LigandFailure("incomplete %s: the two deposited oxygens are collinear with %s" % (kind, cel))
        n /= np.linalg.norm(n)
        t = np.sqrt(1.0 - np.dot(s, s))
        new = [s + t * n, s - t * n]
    blen = float(np.mean(lens))
    taken = {a[0] for a in atoms}
    names = [nm for nm in ("O1", "O2", "O3", "O4", "O5", "O6") if nm not in taken]
    serials = [int(l[6:11]) for l in recs if l[6:11].strip().isdigit()]
    tmpl = recs[0]
    out = []
    for k, v in enumerate(new):
        xyz = c + blen * v
        nm = names[k]
        line = (tmpl[:6] + "%5d" % (max(serials or [0]) + k + 1) + tmpl[11:12]
                + ("%-4s" % (" " + nm if len(nm) < 4 else nm)) + tmpl[16:30]
                + "%8.3f%8.3f%8.3f" % tuple(xyz) + tmpl[54:76].ljust(22) + " O")
        out.append(line)
    body = [l for l in pdb_text.splitlines() if not l.startswith("END")]
    note = ("completed %s: %d of 4 oxygens deposited, %s placed on the ideal tetrahedron at %.2f A"
            % (kind, nox, " and ".join(names[:len(new)]), blen))
    return "\n".join(body + out + ["END"]) + "\n", note


# ---------------------------------------------------------------------------
# Acid groups at pH 7.4
# ---------------------------------------------------------------------------

# OpenBabel's pH model leaves sulfuric acid neutral (ten pass-2 sulfates were built
# as H2SO4), and the CCD entries are mostly the neutral acids (GDP with three P-OH),
# so every path goes through the same explicit rules, which encode textbook pKa
# values and nothing else:
#   carboxylic acid R-COOH, pKa 4-5                                  -> R-COO-
#   carbonic acid, pKa1 6.35, pKa2 10.3                              -> HCO3-
#     (bicarbonate came out as CO3 2- from OpenBabel, review L8)
#   phosphorus(V) with one P=O (or P=S), by its bridges to the rest of the molecule
#   (O-R, N, C or S-R; a terminal S counts like a terminal O, so GTPgammaS loses
#   its S-H, review L8):
#     0  orthophosphate, pKa2 7.2                                    -> HPO4 2-
#     1  O or N: monoester, terminal phosphate of an NDP or NTP,
#        pKa2 6.0-7.0                                                -> -2
#     1  C: phosphonate; methylphosphonic acid pKa2 7.7, the
#        gamma-phosphonate of AMP-PCP and GMP-PCP 8.4                -> -1
#     2  diester, anhydride bridge, pKa ~1                           -> -1
#     3  triester                                                    ->  0
#   A chain of phosphorus centres joined P-X-P (X = O, C or N) none of whose bridges
#   leaves the chain (pyrophosphate, triphosphate, a geminal bisphosphonate) keeps one
#   proton, since its last pKa is 9-11 (pyrophosphate 9.3, etidronate 11.3): PPi is
#   -3, PPPi -4 and zoledronate -3, where the per-centre rules gave -4, -5 and -4
#   (review L6). The first terminal centre in atom order carries the two O-.
#   sulfur(VI) with two S=O: every other terminal O, pKa < 2         -> O-
# Not modelled: the imido bridge of GNP is treated like an anhydride O, which keeps
# the -4 of v1.4, although free AMP-PNP has a gamma pKa of 7.7 (Yount et al. 1971),
# so in water at pH 7.4 two thirds of free GNP carry -3 (every GNP of the benchmark
# sits on a Mg2+, which favours -4); and the coupling of many phosphates on one ring
# (IP6 is built at -12, about -8 to -9 at pH 7.4). GDP, GTP and ATP satisfy the rules
# as OpenBabel and the CCD path build them (-3, -4, -4).
ACID_PH_WINDOW = (6.9, 7.9)


def _terminal(atom, elements=(8,)):
    """Neighbours of `atom` with an atomic number in `elements` that are bonded to no
    other heavy atom."""
    return [n for n in atom.GetNeighbors()
            if n.GetAtomicNum() in elements and sum(1 for m in n.GetNeighbors() if m.GetAtomicNum() > 1) == 1]


def acid_plan(mol):
    """[(centre index, label, [single-bonded terminal O/S indices], wanted anion count)]."""
    from rdkit import Chem
    plan, phos = [], {}
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z not in (6, 15, 16):
            continue
        if any(n.GetAtomicNum() == 1 for n in a.GetNeighbors()) and z != 6:
            continue          # P-H / S-H: not an oxyacid centre (flagged by the H check instead)
        term = _terminal(a, (8, 16) if z == 15 else (8,))
        btype = {t.GetIdx(): mol.GetBondBetweenAtoms(a.GetIdx(), t.GetIdx()).GetBondType() for t in term}
        dbl = [t for t in term if btype[t.GetIdx()] == Chem.BondType.DOUBLE]
        sgl = [t.GetIdx() for t in term if btype[t.GetIdx()] == Chem.BondType.SINGLE]
        if z == 6:
            if a.GetDegree() == 3 and len(dbl) == 1 and len(sgl) == 1:
                plan.append((a.GetIdx(), "carboxylic acid", sgl, 1))
            elif a.GetDegree() == 3 and len(dbl) == 1 and len(sgl) == 2:
                plan.append((a.GetIdx(), "carbonate", sgl, 1))
        elif z == 15:
            if a.GetDegree() == 4 and len(dbl) == 1:
                tset = {t.GetIdx() for t in term}
                bridges = [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in tset]
                phos[a.GetIdx()] = (bridges, sgl)
        elif z == 16:
            if a.GetDegree() == 4 and len(dbl) == 2:
                label = "sulfate" if len(term) == 4 else "sulfate/sulfonate (%d bridging)" % (4 - len(term))
                plan.append((a.GetIdx(), label, sgl, len(sgl)))
    want, label = {}, {}
    for p, (bridges, sgl) in phos.items():
        nb = len(bridges)
        if nb == 1 and mol.GetAtomWithIdx(bridges[0]).GetAtomicNum() == 6:
            want[p], label[p] = 1, "phosphonate"
        else:
            want[p] = {0: 2, 1: 2, 2: 1}.get(nb, 0)
            label[p] = "phosphate" if nb == 0 else "phosphate (%d bridging)" % nb
    # P-X-P chains: centres sharing a bridge atom
    partners = {p: set() for p in phos}
    for p, (bridges, _) in phos.items():
        for q, (bq, _) in phos.items():
            if q != p and set(bridges) & set(bq):
                partners[p].add(q)
    seen = set()
    for p in sorted(phos):
        if p in seen or not partners[p]:
            continue
        chain, todo = set(), [p]
        while todo:
            q = todo.pop()
            if q not in chain:
                chain.add(q)
                todo += list(partners[q])
        seen |= chain
        free = all(any(x in phos[r][0] for r in chain if r != q) for q in chain for x in phos[q][0])
        if not free:
            continue
        ends = sorted(q for q in chain if len(phos[q][0]) == 1)
        for k, q in enumerate(ends):
            want[q] = 2 if k == 0 else 1
            label[q] = "polyphosphate/bisphosphonate end"
    for p in sorted(phos):
        sgl = phos[p][1]
        plan.append((p, label[p], sgl, min(want[p], len(sgl))))
    return plan


def _is_anion(mol, idx):
    o = mol.GetAtomWithIdx(idx)
    return o.GetFormalCharge() == -1 and o.GetTotalNumHs(includeNeighbors=True) == 0


def heavy_names(mol, ref_atoms):
    """{atom index: deposited atom name} for the heavy atoms, which come first and in
    deposited order on every perception path (chemistry_problems() verifies it)."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return dict(zip(heavy, [r[0] for r in ref_atoms if r[1] != "H"]))


def oxo_sulfur(mol, names=None):
    """(mol, notes): S(=O)2 for a four-coordinate sulfur that was built with only single
    bonds and two or more terminal oxygens. OpenBabel perceives sulfonates and sulfate
    monoesters as S(OH)3-X (HEPES, MES, sulfotyrosine, PAPS), which no acid rule
    recognises: 2XQR HEPES came out at +2 and 4AQE MES at +1 (review L2). The two
    shortest S-O bonds become double and lose their hydrogen; the sulfur rule then
    deprotonates the rest. Heavy-atom indices stay valid (hydrogens come last)."""
    import numpy as np
    from rdkit import Chem
    names = names or {}
    notes = []
    for _ in range(mol.GetNumAtoms()):
        hit = None
        for a in mol.GetAtoms():
            if a.GetAtomicNum() != 16 or a.GetFormalCharge() != 0:
                continue
            nbrs = list(a.GetNeighbors())
            if len(nbrs) != 4 or any(n.GetAtomicNum() == 1 for n in nbrs):
                continue
            if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
                continue
            term = _terminal(a, (8,))
            if len(term) >= 2:
                hit = (a, term)
                break
        if hit is None:
            return mol, notes
        a, term = hit
        X = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
        if X is not None:
            term.sort(key=lambda o: float(np.linalg.norm(X[a.GetIdx()] - X[o.GetIdx()])))
        rw = Chem.RWMol(mol)
        hs = []
        for o in term[:2]:
            rw.GetBondBetweenAtoms(a.GetIdx(), o.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
            ro = rw.GetAtomWithIdx(o.GetIdx())
            ro.SetFormalCharge(0)
            ro.SetNoImplicit(True)
            ro.SetNumExplicitHs(0)
            hs += [n.GetIdx() for n in o.GetNeighbors() if n.GetAtomicNum() == 1]
        for h in sorted(hs, reverse=True):
            rw.RemoveAtom(h)
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        notes.append("S(=O)2 at %s: built with %d single S-O bonds to terminal oxygens"
                     % (names.get(a.GetIdx(), "S%d" % (a.GetIdx() + 1)), len(term)))
        mol = m
    return mol, notes


def ionize_acids(mol, names=None):
    """(mol, notes): put every carboxylic acid, carbonate, phosphate and sulfate group of
    an explicit-hydrogen RDKit molecule into its pH 7.4 state (see the table above).
    Only hydrogens are removed or added, so heavy-atom indices stay valid."""
    from rdkit import Chem
    names = names or {}
    notes = []
    for _ in range(4 * mol.GetNumAtoms() + 1):   # one centre per pass: removing H renumbers atoms
        plan = acid_plan(mol)
        change = None
        for centre, label, sgl, want in plan:
            have = [i for i in sgl if _is_anion(mol, i)]
            if len(have) < want:
                cand = [i for i in sgl if i not in have]
                change = ("deprotonate", centre, label, cand[:want - len(have)])
                break
            if len(have) > want:
                change = ("protonate", centre, label, have[:len(have) - want])
                break
        if change is None:
            return mol, notes
        what, centre, label, idxs = change
        rw = Chem.RWMol(mol)
        where = names.get(centre, "%s%d" % (mol.GetAtomWithIdx(centre).GetSymbol(), centre + 1))
        if what == "deprotonate":
            hs = []
            for i in idxs:
                o = rw.GetAtomWithIdx(i)
                h = [n.GetIdx() for n in o.GetNeighbors() if n.GetAtomicNum() == 1]
                hs += h[:1]
                o.SetFormalCharge(-1)
                o.SetNoImplicit(True)
                o.SetNumExplicitHs(0)
            for h in sorted(hs, reverse=True):
                rw.RemoveAtom(h)
            m = rw.GetMol()
            Chem.SanitizeMol(m)
            notes.append("%s at %s: %d deprotonated for pH 7.4" % (label, where, len(idxs)))
        else:
            for i in idxs:
                o = rw.GetAtomWithIdx(i)
                o.SetFormalCharge(0)
                o.SetNoImplicit(False)
            m = rw.GetMol()
            Chem.SanitizeMol(m)
            m = Chem.AddHs(m, addCoords=True)
            notes.append("%s at %s: %d protonated for pH 7.4" % (label, where, len(idxs)))
        mol = m
    return mol, notes


# ---------------------------------------------------------------------------
# Is the built molecule the deposited one?
# ---------------------------------------------------------------------------

# Bond orders against the deposited geometry, for molecules NOT built from their
# exact CCD entry (OpenBabel's perception). The thresholds
# were calibrated on all 44 ligand instances of the 46 HADDOCK structures as
# OpenBabel perceives them:
#  - a C=C, C=N or C=O double bond longer than the limit: the four enol-ether bonds
#    of 2FD6 ETX and PGE are 1.545-1.583 A, a real C=C is 1.34 and conjugated ones
#    stay below 1.45. For a carbon with two terminal oxygens (carboxylate,
#    carbonate) the shorter C-O is tested, whichever O carries the double bond: 3MC0
#    ACT has C-O 1.444 and C-OXT 1.302 A with the double bond on O (review L1).
#  - a single bond between two sp3 atoms shorter than the limit: the saturated 1RLB
#    retinoid has C-C "single" bonds of 1.340-1.419 A (the C=C of its polyene), the
#    1IBR guanine an sp3-sp3 C-N of 1.293 A.
#  - a carbon with three heavy neighbours that was built sp3 but lies in their plane:
#    1IBR GNP 0.000 A, 1RLB REA 0.133, 0.210, 0.290 (a tetrahedral centre with 1.53 A
#    bonds: 0.51). Built-sp2 carbons stay below 0.08 A.
# On the HADDOCK set every correct ligand stays inside these limits, but not on PDB
# ligands at large: in 1332 correct molecules of 172 PPB-Affinity and glue entries, 5
# sp3-sp3 C-C bonds are below 1.43 A and sp3 heights of 0.13-0.305 A occur (review
# L1). That is why a CCD build skips these checks; they remain the guard for what
# OpenBabel guesses.
# Aromatic bonds are not length-checked: OpenBabel assigns aromaticity from ring
# planarity, and the FAD and ATP rings carry formally single C-C bonds of 1.48 A.
# There is no minimum for an sp3 C-O single bond (review L8 proposed about 1.33 A, to
# catch formate built as methanediol and the PLP aldehyde built as CH2OH): among the
# CCD builds of the review's 1400-instance survey, correct sp3 C-O single bonds go
# down to 1.225 A (3QEL ifenprodil), 1.246 (3SDA GCY) and 1.304 A (5UWS GNP). Those
# ligands now come from their CCD entries instead.
DOUBLE_MAX = {("C", "C"): 1.50, ("C", "N"): 1.44, ("C", "O"): 1.36}
SP3_SINGLE_MIN = {("C", "C"): 1.43, ("C", "N"): 1.38}
SP3_PLANE_MIN = 0.30   # A
SP2_PLANE_MAX = 0.25   # A
H_CARRIERS = {"C", "N", "O", "S", "SE"}
# A 9-substituted guanine with its ring H on N3 and none on N1 (the 3H-keto
# tautomer): what OpenBabel built for 8 of the 9 benchmark guanine nucleotides (L3).
GUANINE_3H = "[#8]~[#6]1~[#7;H0;X2]~[#6](~[#7;H2])~[#7;H1]~[#6]2~[#7]~[#6]~[#7]~[#6]~1~2"


def _formula(els):
    from collections import Counter
    c = Counter(els)
    return "".join("%s%d" % (e.capitalize(), c[e]) for e in sorted(c))


def chemistry_problems(mol, ref_atoms, geometry=True, chemistry=True, acids=True):
    """Reasons why `mol` (RDKit, explicit H, heavy atoms first in deposited order and
    at the deposited coordinates) is not the deposited ligand; [] when it is.

    Always: the heavy-atom composition, order and positions. chemistry: hydrogen
    carriers, three-coordinate S-H, sulfur left as S(OH)3, the guanine 3H tautomer.
    acids: the pH 7.4 acid states (off when the rules are, review L12). geometry: the
    bond-order checks above. A CCD build is checked for composition and acids only."""
    import numpy as np
    from rdkit import Chem
    probs = []
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    ref = [r for r in ref_atoms if r[1] != "H"]
    built_els = [a.GetSymbol().upper() for a in heavy]
    ref_els = [r[1] for r in ref]
    if sorted(built_els) != sorted(ref_els):
        return ["heavy atoms differ from the deposited ligand: built %s, deposited %s"
                % (_formula(built_els), _formula(ref_els))]
    X = mol.GetConformer().GetPositions() if mol.GetNumConformers() else None
    if X is None or any(e != r[1] or np.linalg.norm(X[a.GetIdx()] - np.array(r[2:])) > 0.05
                        for a, e, r in zip(heavy, built_els, ref)):
        return ["heavy atoms are not the deposited ones in order and position"]
    name = {a.GetIdx(): r[0] for a, r in zip(heavy, ref)}

    if chemistry:
        for a in heavy:
            nh = a.GetTotalNumHs(includeNeighbors=True)
            if a.GetSymbol().upper() not in H_CARRIERS and nh > 0:
                probs.append("hydrogen on %s %s" % (a.GetSymbol(), name[a.GetIdx()]))
            if a.GetAtomicNum() == 16 and nh > 0 and sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() > 1) >= 3:
                probs.append("hydrogen on the three-coordinate S %s (a sulfonium carries none)" % name[a.GetIdx()])
            if (a.GetAtomicNum() == 16 and len(_terminal(a, (8,))) >= 3
                    and not any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())):
                probs.append("S %s has no S=O but %d terminal oxygens" % (name[a.GetIdx()], len(_terminal(a, (8,)))))
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(GUANINE_3H)):
            probs.append("guanine built as the 3H tautomer: H on %s, none on %s"
                         % (name.get(match[5], "N3"), name.get(match[2], "N1")))

    if acids:
        for centre, label, sgl, want in acid_plan(mol):
            have = sum(1 for i in sgl if _is_anion(mol, i))
            if have != want:
                probs.append("%s %s has %d of %d O- expected at pH 7.4" % (label, name[centre], have, want))

    if not geometry:
        return probs
    for b in mol.GetBonds():
        i, j = b.GetBeginAtom(), b.GetEndAtom()
        if i.GetAtomicNum() == 1 or j.GetAtomicNum() == 1:
            continue
        pair = tuple(sorted((i.GetSymbol().upper(), j.GetSymbol().upper())))
        d = float(np.linalg.norm(X[i.GetIdx()] - X[j.GetIdx()]))
        lbl = "%s-%s" % (name[i.GetIdx()], name[j.GetIdx()])
        if b.GetBondType() == Chem.BondType.DOUBLE and pair in DOUBLE_MAX:
            if pair == ("C", "O"):
                c = i if i.GetAtomicNum() == 6 else j
                d = min([d] + [float(np.linalg.norm(X[c.GetIdx()] - X[o.GetIdx()])) for o in _terminal(c, (8,))])
            if d > DOUBLE_MAX[pair]:
                probs.append("%s is %.2f A long but was given a double bond" % (lbl, d))
        if (b.GetBondType() == Chem.BondType.SINGLE and pair in SP3_SINGLE_MIN and d < SP3_SINGLE_MIN[pair]
                and i.GetHybridization() == Chem.HybridizationType.SP3
                and j.GetHybridization() == Chem.HybridizationType.SP3):
            probs.append("%s is %.2f A short but joins two sp3 atoms" % (lbl, d))

    for a in heavy:
        if a.GetAtomicNum() != 6:
            continue
        hn = [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(hn) != 3:
            continue
        p = X[hn]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(nrm) < 1e-6:
            continue
        h = abs(float(np.dot(X[a.GetIdx()] - p[0], nrm / np.linalg.norm(nrm))))
        sp3 = a.GetHybridization() == Chem.HybridizationType.SP3
        if sp3 and h < SP3_PLANE_MIN:
            probs.append("%s was built sp3 but lies %.2f A from the plane of its neighbours" % (name[a.GetIdx()], h))
        if not sp3 and h > SP2_PLANE_MAX:
            probs.append("%s was built sp2 but sits %.2f A out of the plane of its neighbours" % (name[a.GetIdx()], h))
    return probs


# ---------------------------------------------------------------------------
# Monatomic halides: the force field's own ions
# ---------------------------------------------------------------------------

# A lone halogen in a crystal is a halide ion. OpenBabel completed the 1J2J iodides
# to H-I, which OpenFF has no bond term for, and both were lost. The AMBER19SB tree
# GROMACS ships defines these anions in every ions_<water>.itp (I-_opc3 and I-_opc
# among them), and pdb2gmx already includes that file in topol.top, so the ion is
# simulated with the force field's own Li/Merz parameters and not with invented
# ones. merge_ligand.py confirms that topol.top really defines the moleculetype.
FF_HALIDES = {"F": "F", "BR": "BR", "I": "I"}   # element -> moleculetype (= atom and residue name)


# ---------------------------------------------------------------------------
# Bond orders: the CCD entry, OpenBabel
# ---------------------------------------------------------------------------

CCD_ORDER = {"SING": 1, "DOUB": 2, "TRIP": 3}


def ob_protonate(mol, ph):
    """Explicit-hydrogen RDKit molecule from a heavy-atom one (bond orders, formal
    charges and every atom's hydrogen count set; deposited coordinates). The hydrogen
    counts enter OpenBabel as implicit hydrogens, and AddHydrogens(correctForPH)
    applies OpenBabel's pH model and places them, which is how v1.4 protonated
    OpenBabel's own perception. Heavy atoms keep their order; hydrogens follow."""
    from rdkit import Chem
    from openbabel import openbabel as ob
    k = Chem.Mol(mol)
    Chem.Kekulize(k, clearAromaticFlags=True)
    conf = k.GetConformer()
    obm = ob.OBMol()
    obm.BeginModify()
    for a in k.GetAtoms():
        oa = obm.NewAtom()
        oa.SetAtomicNum(a.GetAtomicNum())
        p = conf.GetAtomPosition(a.GetIdx())
        oa.SetVector(p.x, p.y, p.z)
        oa.SetFormalCharge(a.GetFormalCharge())
    for b in k.GetBonds():
        order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}[b.GetBondType()]
        obm.AddBond(b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1, order)
    obm.EndModify()
    for a in k.GetAtoms():
        obm.GetAtom(a.GetIdx() + 1).SetImplicitHCount(a.GetTotalNumHs())
    obm.AddHydrogens(False, True, ph)
    conv = ob.OBConversion()
    conv.SetOutFormat("sdf")
    return Chem.MolFromMolBlock(conv.WriteString(obm), removeHs=False, sanitize=True)


def describe_source(src):
    """'files.rcsb.org', or a cache file relative to the repository when it is in it."""
    if src == "files.rcsb.org":
        return src
    rel = os.path.relpath(src, REPO)
    return "local cache " + (src if rel.startswith("..") else rel)


# A match by atom name must also fit the deposited geometry: every heavy-atom bond
# of the entry at most BOND_SPAN_MAX times the covalent radius sum of its two atoms
# in the deposited coordinates. The names alone were trusted, and the geometric
# checks are off for a CCD build, so a deposit whose names do not fit its atoms got
# the entry's connectivity on the wrong atoms without a flag: the 1A2K GDP with
# N1/N3, PA/PB or C2/C8 swapped was built 'by atom name' with heavy-atom bonds of
# 2.4 to 5.6 A, and AM1-BCC converged on it (review L-1). Such a match goes to the
# bond graph below, which finds the deposited molecule. On wwPDB-annotated deposits
# the longest entry bond of 757 by-name matches is 1.14 times the radius sum (5UWS
# GNP O4'-C1', 1.61 A).
BOND_SPAN_MAX = 1.3


def name_misfit(comp, mapping, heavy_ref):
    """(atom 1, atom 2, length in A) of the first heavy-atom bond of the entry that the
    mapping {deposited heavy index: CCD atom name} stretches beyond BOND_SPAN_MAX
    covalent radius sums, or None when every bond fits."""
    import numpy as np
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    pos = {n: i for i, n in mapping.items()}
    el = {a["name"]: a["element"] for a in comp["atoms"]}
    for n1, n2, order, arom in comp["bonds"]:
        if n1 in pos and n2 in pos:
            d = float(np.linalg.norm(np.array(heavy_ref[pos[n1]][2:]) - np.array(heavy_ref[pos[n2]][2:])))
            rsum = pt.GetRcovalent(el[n1].capitalize()) + pt.GetRcovalent(el[n2].capitalize())
            if d > BOND_SPAN_MAX * rsum:
                return n1, n2, d
    return None


def ccd_mapping(comp, heavy_ref, pdb_content):
    """({deposited heavy index: CCD atom name}, how) when the CCD entry's heavy atoms are
    exactly the deposited ones, else (None, why not). By atom name first (CCD atom_id,
    then alt_atom_id) when the named atoms also fit the entry's bonds (name_misfit),
    then, for renamed atoms, by an identical bond graph: the same number of heavy atoms
    and bonds, the deposited bonds from RDKit's proximity rule."""
    from rdkit import Chem
    cheavy = [a for a in comp["atoms"] if a["element"] not in ("H", "D")]
    dep_names = [r[0] for r in heavy_ref]
    misfit = None
    for key in ("name", "alt"):
        by = {a[key]: a for a in cheavy}
        if (len(by) == len(cheavy) and sorted(dep_names) == sorted(by)
                and all(by[r[0]]["element"] == r[1] for r in heavy_ref)):
            mapping = {i: by[r[0]]["name"] for i, r in enumerate(heavy_ref)}
            misfit = name_misfit(comp, mapping, heavy_ref)
            if misfit is None:
                return mapping, "by atom name"
    # The names do not fit the geometry: said in the match note, or in the reason.
    named = "" if misfit is None else (
        "; its atom names would bond %s-%s at %.2f A" % misfit)
    if len(cheavy) != len(heavy_ref):
        return None, "the CCD entry %s has %d heavy atoms, %d are deposited" % (comp["id"], len(cheavy), len(heavy_ref))
    lines = [l for l in pdb_content.splitlines() if l.startswith(("ATOM  ", "HETATM")) and element_of(l) != "H"]
    dep = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", removeHs=False, sanitize=False, proximityBonding=True)
    if dep is None or dep.GetNumAtoms() != len(heavy_ref):
        return None, "the deposited atoms of %s could not be read as one molecule%s" % (comp["id"], named)
    names = {a["name"] for a in cheavy}
    tm = Chem.RWMol()
    for a in cheavy:
        at = Chem.Atom(a["element"][0] + a["element"][1:].lower())
        tm.AddAtom(at)
    idx = {a["name"]: k for k, a in enumerate(cheavy)}
    nb = 0
    for n1, n2, order, arom in comp["bonds"]:
        if n1 in names and n2 in names:
            tm.AddBond(idx[n1], idx[n2], Chem.BondType.SINGLE)
            nb += 1
    for b in dep.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
    if nb != dep.GetNumBonds():
        return None, "the CCD entry %s has %d heavy-atom bonds, the deposited %s geometry %d%s" % (
            comp["id"], nb, comp["id"], dep.GetNumBonds(), named)
    t = tm.GetMol()
    for m in (t, dep):
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
    match = dep.GetSubstructMatch(t)
    if len(match) != len(cheavy):
        return None, "the deposited bonds of %s are not those of its CCD entry%s" % (comp["id"], named)
    return {match[k]: cheavy[k]["name"] for k in range(len(cheavy))}, "by bond graph" + named


def ccd_build(comp, mapping, heavy_ref):
    """Heavy-atom RDKit molecule at the deposited coordinates with the CCD entry's bond
    orders, formal charges and hydrogen counts (from the H atoms the entry lists)."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    by = {a["name"]: a for a in comp["atoms"]}
    hcount = {n: 0 for n in mapping.values()}
    for n1, n2, order, arom in comp["bonds"]:
        e1, e2 = by[n1]["element"], by[n2]["element"]
        if e1 in ("H", "D") and n2 in hcount:
            hcount[n2] += 1
        elif e2 in ("H", "D") and n1 in hcount:
            hcount[n1] += 1
    rw = Chem.RWMol()
    for i in range(len(heavy_ref)):
        ca = by[mapping[i]]
        a = Chem.Atom(ca["element"][0] + ca["element"][1:].lower())
        a.SetFormalCharge(ca["charge"])
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(hcount[ca["name"]])
        rw.AddAtom(a)
    pos = {n: i for i, n in mapping.items()}
    for n1, n2, order, arom in comp["bonds"]:
        if n1 in pos and n2 in pos:
            if order not in CCD_ORDER:
                raise ValueError("bond order %s between %s and %s" % (order, n1, n2))
            rw.AddBond(pos[n1], pos[n2], {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                                          3: Chem.BondType.TRIPLE}[CCD_ORDER[order]])
    m = rw.GetMol()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, r in enumerate(heavy_ref):
        conf.SetAtomPosition(i, Point3D(*r[2:]))
    m.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(m)
    return m


def ccd_entry(ccd_id, info, cache=True):
    """(component, source) of a CCD entry, or (None, None) with the reason in info.
    cache=False reads files.rcsb.org only."""
    text, src = ccd_cache.fetch(ccd_id) if cache else ccd_cache.fetch(ccd_id, cache=False)
    if text is None:
        info.append(src)
        return None, None
    try:
        return ccd_cache.component(text), src
    except Exception as e:
        info.append("the CCD entry %s could not be read (%s)" % (ccd_id, e))
        return None, None


def incomplete_entry(comp):
    """Why the CCD entry `comp` is not complete, or None, judged from the entry itself:
    in an entry of two or more atoms every atom, every hydrogen too, has a bond. A
    cache file cut short in its bond loop passed as the entry: the GDP built from one
    was +5 with 28 fragments and 83 radical electrons, reported no problem, and
    stopped as an OpenFF radical error that named no file (review L-2). Radical
    electrons in the build were the test after that, and they took a genuine radical
    for a cut file: the nitric oxide entry (NO) went to OpenBabel, which built HNO
    without a flag (review V1). In the CCD of 2026-09-26 (51,346 entries) no released
    entry has an atom without a bond, while 127 released entries build as radicals
    from their own bonds and hydrogens (NO, nitrogen dioxide 2NO, nitroxide spin
    labels, Fe-S clusters). The formula is no test either: 4 released entries (5ZV,
    JQZ, LFF, SFB) list fewer hydrogens than their _chem_comp.formula and build a
    closed-shell molecule, and a cut file loses bonds, not atoms (the atom loop comes
    first)."""
    if len(comp["atoms"]) < 2:
        return None
    bonded = {n for b in comp["bonds"] for n in b[:2]}
    loose = [a["name"] for a in comp["atoms"] if a["name"] not in bonded]
    if loose:
        return "%d of its %d atoms %s no bond: %s%s" % (len(loose), len(comp["atoms"]),
                                                       "has" if len(loose) == 1 else "have",
                                                       ", ".join(loose[:4]), ", ..." if len(loose) > 4 else "")
    return None


RADICAL_END = ("OpenFF cannot parametrise a radical, and OpenBabel would build another molecule from the "
               "deposited atoms")
INCOMPLETE_END = "; not built, as OpenBabel would build another molecule from the deposited atoms"


def _heavy_build(comp, mapping, heavy_ref, info):
    """ccd_build(), or None with the reason in info."""
    try:
        return ccd_build(comp, mapping, heavy_ref)
    except Exception as e:
        info.append("the CCD entry %s could not be built on the deposited atoms (%s: %s)"
                    % (comp["id"], type(e).__name__, " ".join(str(e).split())[:120]))
        return None


def _unpaired(heavy, mapping):
    """'2 unpaired electrons on N3, C4' for a heavy-atom build with radical electrons,
    None for a closed shell."""
    rad = [(mapping[a.GetIdx()], a.GetNumRadicalElectrons()) for a in heavy.GetAtoms() if a.GetNumRadicalElectrons()]
    if not rad:
        return None
    n = sum(k for _, k in rad)
    return "%d unpaired electron%s on %s" % (n, "" if n == 1 else "s", ", ".join(name for name, _ in rad[:6])
                                             + (", ..." if len(rad) > 6 else ""))


def perceive_ccd(pdb_content, heavy_ref, ccd_id, ph, info):
    """(explicit-H molecule, header note) built from the CCD entry ccd_id when it
    describes exactly the deposited heavy atoms; (None, None) otherwise, the reason in
    info. A cached entry that is incomplete, or that builds as a radical, is replaced
    by the one at files.rcsb.org, and the note names the damaged file.

    Raises LigandFailure when the entry that describes the deposited atoms is a
    radical, or is incomplete at files.rcsb.org as well: that is the molecule itself,
    OpenFF cannot parametrise it, and OpenBabel would build another molecule from the
    same atoms (it closed nitric oxide to HNO; review V1). A cached copy that builds as
    a radical while files.rcsb.org does not answer stops the ligand too, naming the
    file that could not be checked."""
    comp, src = ccd_entry(ccd_id, info)
    if comp is None:
        return None, None
    broken, cached, damaged = incomplete_entry(comp), None, None
    if broken is not None and src != "files.rcsb.org":
        info.append("the CCD entry %s from %s is incomplete (%s)" % (comp["id"], describe_source(src), broken))
        cached, damaged = broken, "%s, which is incomplete (%s)" % (describe_source(src), broken)
        comp, src = ccd_entry(ccd_id, info, cache=False)
        if comp is None:
            return None, None
        broken = incomplete_entry(comp)
    mapping, how = ccd_mapping(comp, heavy_ref, pdb_content)
    if mapping is None:
        info.append(how)
        return None, None
    if broken is not None:
        info.append("files.rcsb.org serves it incomplete in the same way, so that is the entry itself"
                    if broken == cached else "the CCD entry %s at files.rcsb.org is incomplete (%s)" % (comp["id"], broken))
        raise LigandFailure("; ".join(info) + INCOMPLETE_END)
    heavy = _heavy_build(comp, mapping, heavy_ref, info)
    if heavy is None:
        return None, None
    rad = _unpaired(heavy, mapping)
    if rad and src != "files.rcsb.org":
        # A cache file cut so that every atom keeps a bond, for example just before a
        # ring-closure bond row, builds as a radical as well, and only files.rcsb.org's
        # copy tells it from a genuine radical entry: the cached GDP.cif cut before its
        # last bond row (N3-C4) was reported as 'the CCD entry GDP is a radical, 2
        # unpaired electrons on N3, C4', where files.rcsb.org's entry builds the GDP
        # (review N1). The ligand stops only when that copy is a radical too (NO), or
        # when files.rcsb.org cannot be asked; the prefetch keeps a cached file that
        # reads as an entry, so the reason says how to fetch it again (fusion review F9):
        # remove that one file and prefetch, which fetches only what is missing. The
        # prefetch with --refresh would fetch every entry of the input again and
        # overwrite valid ones, in the default directory files the repository ships
        # (verifier V2).
        said = "the CCD entry %s from %s builds as a radical, %s" % (comp["id"], describe_source(src), rad)
        asked = []
        comp2, src2 = ccd_entry(ccd_id, asked, cache=False)
        if comp2 is None:
            raise LigandFailure("%s, and %s, so the cached file could not be checked against the entry (where "
                                "files.rcsb.org answers, remove %s and run 'python3 utils/ccd_cache.py -o %s "
                                "<input PDB>', which fetches only what is missing); %s"
                                % (said, "; ".join(asked) or "files.rcsb.org did not answer", src,
                                   os.path.dirname(src), RADICAL_END))
        info.append(said)
        mapping2, how2 = ccd_mapping(comp2, heavy_ref, pdb_content)
        if mapping2 is None:
            info.append(how2)
            return None, None
        broken = incomplete_entry(comp2)
        if broken is not None:
            info.append("the CCD entry %s at files.rcsb.org is incomplete (%s)" % (comp2["id"], broken))
            raise LigandFailure("; ".join(info) + INCOMPLETE_END)
        heavy2 = _heavy_build(comp2, mapping2, heavy_ref, info)
        if heavy2 is None:
            return None, None
        rad2 = _unpaired(heavy2, mapping2)
        if rad2:
            raise LigandFailure("the CCD entry %s is a radical in %s and at files.rcsb.org alike, %s; %s"
                                % (comp2["id"], describe_source(src), rad2, RADICAL_END))
        damaged = "%s, which builds as a radical (%s)" % (describe_source(src), rad)
        comp, src, mapping, how, heavy = comp2, src2, mapping2, how2, heavy2
    elif rad:
        raise LigandFailure("the CCD entry %s (%s) is a radical, %s; %s" % (comp["id"], describe_source(src), rad,
                                                                           RADICAL_END))
    try:
        mol = ob_protonate(heavy, ph)
    except Exception as e:
        info.append("the CCD entry %s could not be built on the deposited atoms (%s: %s)"
                    % (comp["id"], type(e).__name__, " ".join(str(e).split())[:120]))
        return None, None
    if mol is None:
        info.append("the CCD entry %s could not be protonated" % comp["id"])
        return None, None
    note = "bond orders, charges and hydrogens from the CCD entry %s (%s, matched %s, entry modified %s)" % (
        comp["id"], describe_source(src), how, comp["modified"] or "?")
    if damaged:
        note += "; it replaced the damaged " + damaged
    print("CCD %s: %s, %d atoms, SMILES %s" % (comp["id"], how, mol.GetNumAtoms(), _smiles(mol)))
    return mol, note


def _smiles(mol):
    from rdkit import Chem
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return "?"


def perceive_openbabel(pdb_content, ph):
    """(rdkit mol or None, kekulization_ok): OpenBabel bond orders and pH hydrogens."""
    from rdkit import Chem
    from openbabel import openbabel as ob
    print("OpenBabel: perceiving bond orders from 3D coordinates...")

    # Capture OpenBabel warnings to detect kekulization failures. The log is
    # process-wide and keeps its messages, so it is cleared first: otherwise one
    # ligand's kekulization warning would condemn every later one in the process.
    ob.obErrorLog.SetOutputLevel(ob.obWarning)
    ob.obErrorLog.ClearLog()
    ob.obErrorLog.StartLogging()

    conv = ob.OBConversion()
    conv.SetInFormat("pdb")
    mol_ob = ob.OBMol()
    conv.ReadString(mol_ob, pdb_content)
    mol_ob.PerceiveBondOrders()

    ob_warnings = ob.obErrorLog.GetMessagesOfLevel(ob.obWarning)
    kekulization_ok = not any("kekulize" in w.lower() for w in ob_warnings)
    ob.obErrorLog.StopLogging()

    mol_ob.AddHydrogens(False, True, ph)  # polarOnly=False, correctForPH=True
    print(f"OpenBabel: {mol_ob.NumAtoms()} atoms (with H), charge={mol_ob.GetTotalCharge()}")
    if not kekulization_ok:
        print("Warning: OpenBabel failed to kekulize aromatic bonds")
    conv.SetOutFormat("sdf")
    return Chem.MolFromMolBlock(conv.WriteString(mol_ob), removeHs=False, sanitize=True), kekulization_ok


def clean_copy(mol):
    """Strip PDB residue metadata (OpenFF requires all-or-none residue info) by a
    round trip through SDF; returns (mol, sdf block)."""
    from rdkit import Chem
    sdf_block = Chem.MolToMolBlock(mol)
    return Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=True), sdf_block


def build_molecule(pdb_content, ref_atoms, resname, ph):
    """(mol, notes, problems): the molecule the two sources give (see the header).

    resname is the CCD identifier, the residue name of the ligand's records. The
    exact CCD entry wins when it describes the deposited heavy atoms; it is checked
    for composition and acid states only. Otherwise OpenBabel's molecule must pass
    every check in chemistry_problems(). When it does not, it is returned with its
    problems, and the caller builds it and flags it."""
    acid_rules = ACID_PH_WINDOW[0] <= ph <= ACID_PH_WINDOW[1]
    if not acid_rules:
        print(f"Warning: pH {ph} is outside {ACID_PH_WINDOW}: the pH 7.4 acid rules are neither applied nor "
              f"checked; acid groups keep the state OpenBabel's pH model gives them")
    heavy_ref = [r for r in ref_atoms if r[1] != "H"]
    candidates, info = [], []

    def finish(mol, source, full_checks=True):
        mol, _ = clean_copy(mol)
        if mol is None:
            return None
        names = heavy_names(mol, ref_atoms)
        mol, notes = oxo_sulfur(mol, names)
        if acid_rules:
            mol, more = ionize_acids(mol, names)
            notes += more
        for n in notes:
            print(f"{source}: {n}")
        probs = chemistry_problems(mol, ref_atoms, geometry=full_checks, chemistry=full_checks, acids=acid_rules)
        for p in probs:
            print(f"Check failed ({source}): {p}")
        return mol, notes, probs

    # 1. the CCD entry of this residue name
    mol_c, head = perceive_ccd(pdb_content, heavy_ref, resname, ph, info)
    if mol_c is not None:
        got = finish(mol_c, "CCD %s" % resname, full_checks=False)
        if got is not None:
            return got[0], [head] + got[1], got[2]
        info.append("RDKit could not read the CCD build of %s" % resname)
    # Every reason, not the last one: after an incomplete cached entry the last was
    # 'files.rcsb.org did not answer', and the itp note hid the file (review V1).
    why_not_ccd = "; ".join(info) if info else "no CCD entry"
    print(f"No exact CCD build ({why_not_ccd}); OpenBabel perception")

    # 2. OpenBabel
    why = "OpenBabel failed to kekulize"
    mol_ob, kek_ok = perceive_openbabel(pdb_content, ph)
    if mol_ob is not None and kek_ok:
        got = finish(mol_ob, "OpenBabel")
        if got is not None:
            candidates.append(got + ("OpenBabel",))
            if not got[2]:
                return got[0], ["bond orders from OpenBabel (%s)" % why_not_ccd] + got[1], []
            print("OpenBabel bond orders do not fit the deposited geometry")
            why = "OpenBabel's did not fit the deposited geometry: " + got[2][0]
    elif mol_ob is None and kek_ok:
        why = "RDKit could not read OpenBabel's molecule"

    # No third source. A search of search.rcsb.org for CCD IDs starting with the
    # residue name used to follow here; the live API rejects that query (HTTP 400,
    # no 'starts_with' operator), so it never yielded a candidate and only cost one
    # failing request per flagged ligand (review L-5).
    if not candidates:
        print("Error: could not determine bond orders.", file=sys.stderr)
        raise LigandFailure("could not determine bond orders (%s; no CCD entry matching the deposited atoms)%s"
                            % (why, "".join("; " + s for s in info)))
    mol, notes, probs, source = candidates[0]
    # The problems of the molecule first, then, labelled, why the CCD entry was not
    # used: led by that reason, the LIGAND_FAIL line read as a defect of the built
    # molecule and spent one of its three slots on it (review L-8).
    return mol, ["bond orders from %s" % source] + notes, probs + (
        ["no exact CCD entry: " + "; ".join(info)] if info else [])


# ---------------------------------------------------------------------------
# AM1-BCC charges: the deposited conformer first, then seeded retries, capped
# ---------------------------------------------------------------------------

# v1.4 ran AM1-BCC once, on the one conformer the OpenFF AmberTools wrapper embeds
# with RDKit (randomSeed=1). sqm failed there for 1KXP ATP and 1HE8 GNP (both lost),
# and for 2BTF ATP on one machine but not on another: which conformer converges
# depends on the machine and, through the embedding, on the atom order, and the
# converged charges differ by up to 0.375 e on one atom between conformers of 1KXP
# ATP (review L11). Now:
#   1. the deposited conformer (heavy atoms as deposited, hydrogens placed by
#      OpenBabel), which needs no embedding and is the same on every machine; it
#      converged for 1KXP ATP in 46 s, where seed 1 fails (1H1V ATP is the other way
#      round: its deposited conformer fails after 274 s, and a seeded retry is used);
#   2. then RDKit conformers embedded with each seed in AM1BCC_RETRY_SEEDS, in that
#      order (seed 1 is how v1.4 embedded its one conformer). Retries run while their
#      total wall time stays below AM1BCC_RETRY_WALL_S; the running sqm is killed at
#      the cap.
# One sqm run takes 10-120 s for an ATP-sized ligand (up to 274 s for a failing one)
# and 9-20 min for the 84-atom FAD of 1E6E, so the cap lets a nucleotide try all five
# seeds and stops a FAD after about three: without it six FAD attempts took about
# 2 h. The first attempt is never cut and the first retry gets the whole hour, so a
# ligand is lost to the cap only when its deposited conformer fails and its seed-1
# run takes longer than an hour. The order is fixed, but sqm is not reproducible on a
# marginal conformer: on identical input 1H1V ATP's seed 1 converged in one of five
# runs, and the run kept used seed 2, with charges up to 0.52 e from seed 1's; two
# converged runs of the 1E6E FAD differ by up to 0.068 e on one atom. So the
# conformer used, and the charges, can differ from run to run; the itp header
# records which conformer was used.
AM1BCC_RETRY_SEEDS = (1, 2, 3, 4, 5)
AM1BCC_RETRY_WALL_S = 3600.0


def is_charge_failure(exc):
    text = exc if isinstance(exc, str) else "%s: %s" % (type(exc).__name__, exc)
    return any(k in text for k in ("assign_partial_charges", "ChargeCalculationError",
                                   "Cannot properly run", "sqm", "antechamber"))


def charge_failure_text(name, text):
    """A short reason for a failed AM1-BCC run, for the log and the itp note. OpenFF wraps
    the antechamber failure in a 'No registered toolkits can provide the capability'
    message of several kB that repeats the molecule and its coordinates; the AmberTools
    entry in it is the part that says what failed (antechamber's own 'Cannot properly
    run sqm' is already in the job log)."""
    import re
    text = " ".join(str(text).split())
    m = re.search(r"ToolkitWrapper around AmberTools version \S+ <class '[^']+'> : (.*?)(?= ToolkitWrapper around|$)",
                  text)
    if m:
        st = re.search(r"returned non-zero exit status (-?\d+)", m.group(1))
        return "assign_partial_charges via AmberTools: " + (
            "antechamber/sqm exit status %s" % st.group(1) if st else m.group(1)[:160])
    return "%s: %s" % (name, text[:400])


def seeded_conformer(offmol, seed):
    """One RDKit conformer embedded with `seed`, as the OpenFF wrapper embeds its own."""
    from rdkit.Chem import AllChem
    from openff.units import unit
    rd = offmol.to_rdkit()
    rd.RemoveAllConformers()
    cids = list(AllChem.EmbedMultipleConfs(rd, numConfs=1, randomSeed=seed))
    if not cids:
        cids = list(AllChem.EmbedMultipleConfs(rd, numConfs=1, randomSeed=seed, useRandomCoords=True))
    if not cids:
        return None
    return unit.Quantity(rd.GetConformer(cids[0]).GetPositions(), unit.angstrom)


def am1bcc_attempt(offmol, conformer, timeout=None):
    """AM1-BCC partial charges ([e] per atom) from one antechamber/sqm run started on
    `conformer`. It runs in a child process in its own process group, so that at
    `timeout` (s; None waits) it is killed together with antechamber and sqm; then it
    raises TimeoutError. A failed run raises RuntimeError with the failure text.
    SIGTERM or SIGHUP of this process, like SIGINT, kills the group and ends it
    (exit 143 or 129); the group also dies with this process when it is SIGKILLed."""
    import multiprocessing as mp
    import shutil
    import signal
    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    work = tempfile.mkdtemp(prefix="am1bcc_", dir=os.getcwd())
    parent = os.getpid()

    def child():
        os.setpgid(0, 0)
        # The attempt dies with the parametrisation. Its own process group (python,
        # antechamber, sqm) outlived every kill of the parent but SIGINT, and its
        # am1bcc_ directory stayed (review L-3): GNU timeout and SLURM signal the
        # parent, never this group. On the parent's death the kernel now sends SIGTERM
        # here (PR_SET_PDEATHSIG), and SIGTERM kills the whole group.
        try:
            signal.signal(signal.SIGTERM, lambda s, f: os.killpg(0, signal.SIGKILL))
            import ctypes
            ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGTERM))   # 1 = PR_SET_PDEATHSIG
        except Exception:
            pass
        if os.getppid() != parent:          # the parent died before prctl took effect
            os.killpg(0, signal.SIGKILL)
        tempfile.tempdir = work
        try:
            from openff.toolkit import Molecule
            from openff.units import unit
            trial = Molecule(offmol)
            trial.assign_partial_charges("am1bcc", use_conformers=[conformer])
            send.send(("ok", [float(x) for x in trial.partial_charges.m_as(unit.elementary_charge)]))
        except BaseException as e:
            send.send(("error", charge_failure_text(type(e).__name__, e)))
        finally:
            send.close()
            os._exit(0)

    def kill():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        p.join()

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    p = ctx.Process(target=child)
    p.start()
    send.close()
    # SIGTERM and SIGHUP end the wait as SIGINT does, through the except path below,
    # which kills the attempt's group and removes its directory. By default they ended
    # this process alone.
    old = {}
    try:
        for s in (signal.SIGTERM, signal.SIGHUP):
            old[s] = signal.signal(s, stop)
    except ValueError:                      # not the main thread: no handlers here
        pass
    try:
        if not recv.poll(timeout):
            kill()
            raise TimeoutError("AM1-BCC did not finish within %.0f s" % timeout)
        try:
            status, payload = recv.recv()
        except EOFError:
            status, payload = "error", "the AM1-BCC process died (exit code %s)" % p.exitcode
        p.join()
    except BaseException:
        # an interrupt while waiting: the child's own process group would outlive us
        if p.is_alive():
            kill()
        raise
    finally:
        for s, h in old.items():
            signal.signal(s, h if h is not None else signal.SIG_DFL)
        recv.close()
        shutil.rmtree(work, ignore_errors=True)
    if status != "ok":
        raise RuntimeError(payload)
    return payload


def am1bcc_charges(offmol, clock=time.monotonic):
    """(charges, itp note) from the first attempt that converges (see the comment
    above); LigandFailure when none does. Only charge failures are retried."""
    deposited = offmol.conformers[0] if offmol.conformers else None
    attempts = ([("the deposited conformer", deposited, None)] if deposited is not None else []) + \
        [("an RDKit conformer embedded with seed %d" % s, None, s) for s in AM1BCC_RETRY_SEEDS]
    failed, retry_t0 = [], None
    for k, (label, conf, seed) in enumerate(attempts):
        timeout = None
        if seed is not None:
            if retry_t0 is None:
                retry_t0 = clock()
            left = AM1BCC_RETRY_WALL_S - (clock() - retry_t0)
            if left <= 0:
                failed.append("%s not tried (retry cap of %.0f s spent)" % (label, AM1BCC_RETRY_WALL_S))
                break
            conf = seeded_conformer(offmol, seed)
            if conf is None:
                failed.append("%s could not be embedded" % label)
                print(f"AM1-BCC: {failed[-1]}")
                continue
            timeout = left
        t0 = clock()
        try:
            charges = am1bcc_attempt(offmol, conf, timeout)
        except TimeoutError:
            failed.append("%s stopped at the retry cap of %.0f s" % (label, AM1BCC_RETRY_WALL_S))
            print(f"AM1-BCC: {failed[-1]}")
            break
        except Exception as e:
            text = str(e)
            if not is_charge_failure(text):
                raise
            failed.append("%s failed (%s)" % (label, text.splitlines()[0][:100] if text else type(e).__name__))
            print(f"AM1-BCC on {label} failed after {clock() - t0:.0f} s: {text[:160]}")
            continue
        print(f"AM1-BCC on {label} succeeded after {clock() - t0:.0f} s")
        note = "AM1-BCC charges from %s (attempt %d of %d)" % (label, k + 1, len(attempts))
        if failed:
            note += "; before it: " + "; ".join(failed)
        return charges, note
    raise LigandFailure("AM1-BCC charges failed (antechamber/sqm): " + "; ".join(failed))


def make_interchange(ff, offmol):
    """(interchange, charge note): OpenFF parameters with AM1-BCC charges.

    The valence parameters are assigned first, with the formal charges as placeholder
    partial charges (Interchange requires them to sum to the net charge), so that a
    ligand OpenFF cannot type (the AlF3 of 1WQ1 and 1GRN: no Al-F bond term) fails
    with that reason before any sqm run is spent on it."""
    import numpy as np
    from openff.toolkit import Molecule, Topology
    from openff.interchange import Interchange
    from openff.units import unit
    topology = Topology.from_molecules([offmol])
    topology.box_vectors = np.eye(3) * 5.0 * unit.nanometer  # dummy box for export
    probe = Molecule(offmol)
    probe.partial_charges = unit.Quantity(
        np.array([a.formal_charge.m_as(unit.elementary_charge) for a in offmol.atoms], dtype=float),
        unit.elementary_charge)
    Interchange.from_smirnoff(ff, topology, charge_from_molecules=[probe])
    charges, note = am1bcc_charges(offmol)
    charged = Molecule(offmol)
    charged.partial_charges = unit.Quantity(np.array(charges), unit.elementary_charge)
    return Interchange.from_smirnoff(ff, topology, charge_from_molecules=[charged]), note


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def drop_manifest_line(source):
    """Remove this instance's manifest line; returns the remaining lines."""
    lines = []
    if os.path.isfile(MANIFEST):
        with open(MANIFEST) as f:
            lines = [l for l in f.read().splitlines()
                     if l.strip() and not (len(l.split("\t")) >= 8 and l.split("\t")[7] == source)]
        if lines:
            with open(MANIFEST, "w") as f:
                f.write("\n".join(lines) + "\n")
        else:
            os.remove(MANIFEST)
    return lines


def append_manifest(mol_name, itp_path, atomtypes_path, gro_path, natoms, inst, source):
    """One manifest line per instance. Columns 1-5 are the original format
    (make_ion_restraints.py reads column 4); 6-8 name the instance (chain, residue
    number, extracted PDB) so merge_ligand.py can tell which extracted ligand each
    line belongs to without relying on file order. A rerun replaces its own line.
    merge_ligand.py adds column 9, the residue number the ligand got in the merged
    coordinates."""
    lines = drop_manifest_line(source)
    lines.append("\t".join([mol_name, itp_path, atomtypes_path, gro_path, str(natoms), inst[1], inst[2], source]))
    with open(MANIFEST, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_ff_ion(args, inst, atoms, source):
    """Monatomic halide: a one-atom .gro named after the force field's moleculetype."""
    el = atoms[0][1]
    molname = FF_HALIDES[el]
    instance_suffix = f"_{args.instance}" if args.instance > 0 else ""
    chain_suffix = f"_{args.chain}" if args.chain else ""
    gro_path = f"ligand_{args.resname}{instance_suffix}{chain_suffix}.gro"
    x, y, z = (atoms[0][2] / 10.0, atoms[0][3] / 10.0, atoms[0][4] / 10.0)
    with open(gro_path, "w") as f:
        f.write(f"{args.resname} ligand: force-field ion {molname}\n")
        f.write(f"{1:5d}\n")
        f.write(f"{1:5d}{molname:<5s}{molname:>5s}{1:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"{5.0:10.5f}{5.0:10.5f}{5.0:10.5f}\n")
    print(f"{args.resname}: monatomic {el.capitalize()}, simulated as the force field's ion "
          f"(moleculetype {molname}, charge -1); wrote {gro_path}")
    append_manifest(molname, "-", "-", gro_path, 1, inst, source)


def parametrize(args, pdb_content, inst):
    """Build and write one ligand instance. Returns the flag reason ('' when the
    ligand is right); raises LigandFailure (or anything else) when it cannot."""
    source = os.path.basename(args.file)
    pdb_content, altloc = select_altloc(pdb_content)
    if altloc:
        print(f"{args.resname}: {altloc}")
    atoms = pdb_atoms(pdb_content)
    heavy = [a for a in atoms if a[1] != "H"]
    if not heavy:
        raise LigandFailure("no heavy atoms in %s" % source)

    if len(heavy) == 1 and heavy[0][1] in FF_HALIDES and len(atoms) == 1:
        write_ff_ion(args, inst, heavy, source)
        return ""

    # The CCD identifier is the residue name of the records (inst[0]), not -r: the
    # runners take -r from the file name, which for a chain outside A-Z reads 'GDP_a'
    # or 'SO4_ ', and the exact CCD entry and the oxyanion completion were then skipped
    # (review L-6). -r still names the files and the molecule.
    pdb_content, completion = complete_oxyanion(pdb_content, inst[0])
    if completion:
        print(f"{args.resname}: {completion}")
        atoms = pdb_atoms(pdb_content)

    from rdkit import Chem
    mol_rdkit, notes, problems = build_molecule(pdb_content, atoms, inst[0], args.ph)
    notes = [n for n in (altloc, completion) if n] + notes

    # Strip PDB residue metadata (OpenFF requires all-or-none residue info)
    mol_rdkit, sdf_block = clean_copy(mol_rdkit)
    sdf_path = f"ligand_{args.resname}.sdf"
    with open(sdf_path, "w") as f:
        f.write(sdf_block)
    print(f"Wrote {sdf_path}")

    total_charge = Chem.GetFormalCharge(mol_rdkit)
    print(f"RDKit: {mol_rdkit.GetNumAtoms()} atoms, SMILES: {Chem.MolToSmiles(mol_rdkit)}")
    flag = ""
    if problems:
        flag = "built, but not the deposited molecule: " + "; ".join(problems[:3])
        if len(problems) > 3:
            flag += "; and %d more" % (len(problems) - 3)
        print(f"WARNING: {args.resname} {inst[1]} {inst[2]} is not the deposited molecule ({flag}); "
              f"recorded in {LIGAND_FAILED}", file=sys.stderr)

    from openff.toolkit import Molecule, ForceField
    offmol = Molecule.from_rdkit(mol_rdkit, allow_undefined_stereo=True)
    print(f"OpenFF: {offmol.n_atoms} atoms")
    ff = ForceField(args.ff)
    interchange, charge_note = make_interchange(ff, offmol)
    notes.append(charge_note)
    print(f"Parametrized with {args.ff}")

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "ligand")
        interchange.to_gromacs(prefix)

        # Parse the monolithic .top to extract [ atomtypes ] and [ moleculetype ]
        with open(f"{prefix}.top") as f:
            top_lines = f.readlines()
        sections = {}
        for i, line in enumerate(top_lines):
            if line.strip().startswith("["):
                section_name = line.strip().strip("[]").strip()
                if section_name not in sections:
                    sections[section_name] = {"start": i, "end": len(top_lines)}
                for s in sections:
                    if s != section_name and sections[s]["end"] == len(top_lines) and sections[s]["start"] < i:
                        sections[s]["end"] = i
        atomtypes_lines = []
        if "atomtypes" in sections:
            s = sections["atomtypes"]
            atomtypes_lines = top_lines[s["start"]:s["end"]]

        # Everything from [ moleculetype ] to just before [ system ]. [ settles ] is
        # skipped: OpenFF emits an empty one for non-water ligands, which makes GROMACS
        # treat the ligand as a SETTLE moleculetype and fail next to SOL.
        moltype_lines = []
        in_moltype = in_settles = False
        for line in top_lines:
            section = line.strip().strip("[]").strip() if line.strip().startswith("[") else None
            if section == "moleculetype":
                in_moltype, in_settles = True, False
            elif section == "settles":
                in_settles = True
                continue
            elif section in ("system", "molecules"):
                in_moltype = in_settles = False
                continue
            elif section is not None:
                in_settles = False
            if in_moltype and not in_settles:
                moltype_lines.append(line)

        # One molecule type per instance: copies of one ligand can differ in atom count
        instance_suffix = f"_{args.instance}" if args.instance > 0 else ""
        mol_name = f"{args.resname}{instance_suffix}"
        atomtypes_lines = [l.replace("MOL0", mol_name) for l in atomtypes_lines]
        moltype_lines = [l.replace("MOL0", mol_name) for l in moltype_lines]

        atomtypes_path = f"ligand_{mol_name}_atomtypes.itp"
        with open(atomtypes_path, "w") as f:
            f.write(f"; Atom types for ligand {args.resname} (instance {args.instance})\n")
            f.write(f"; Parametrized with OpenFF {args.ff}\n\n")
            for line in atomtypes_lines:
                f.write(line)

        # The header says where the chemistry and the charges came from.
        itp_path = f"ligand_{mol_name}.itp"
        with open(itp_path, "w") as f:
            f.write(f"; Molecule type for ligand {args.resname} (instance {args.instance})\n")
            f.write(f"; Parametrized with OpenFF {args.ff}\n")
            for n in notes:
                f.write(f"; {n}\n")
            if flag:
                f.write(f"; NOT THE DEPOSITED MOLECULE ({LIGAND_FAILED}): {flag}\n")
            f.write(f"; Total charge: {total_charge}\n\n")
            for line in moltype_lines:
                f.write(line)
        print(f"Wrote {itp_path} ({len(atomtypes_lines)} atomtype lines, {len(moltype_lines)} moleculetype lines)")

        # ligand.gro: the residue name left-justified in its 5 columns, as GROMACS
        # writes '%5d%-5s%5s%5d'
        chain_suffix = f"_{args.chain}" if args.chain else ""
        gro_path = f"ligand_{mol_name}{chain_suffix}.gro"
        with open(f"{prefix}.gro") as f:
            gro_lines = f.readlines()
        with open(gro_path, "w") as f:
            f.write(f"{args.resname} ligand\n")
            n_atoms_gro = int(gro_lines[1].strip())
            f.write(f"{n_atoms_gro:5d}\n")
            for line in gro_lines[2:-1]:
                f.write(line.replace("MOL0", f"{args.resname:<4s}"[:4]))
            f.write(gro_lines[-1])
        print(f"Wrote {gro_path} ({n_atoms_gro} atoms)")

        append_manifest(mol_name, itp_path, atomtypes_path, gro_path, n_atoms_gro, inst, source)
    return flag


def main():
    parser = argparse.ArgumentParser(description="Parametrize a small molecule with OpenFF for GROMACS.")
    parser.add_argument('-f', '--file', type=str, required=True, help="Input PDB file with ligand HETATM records")
    parser.add_argument('-r', '--resname', type=str, required=True, help="Residue name of the ligand")
    parser.add_argument('-c', '--chain', type=str, default="", help="Chain ID (for per-instance output filenames)")
    parser.add_argument('-i', '--instance', type=int, default=0, help="Instance number (for unique molecule names across same resname)")
    parser.add_argument('--ph', type=float, default=7.4, help="pH for protonation state assignment (default: 7.4)")
    parser.add_argument('--ff', type=str, default="openff-2.2.1.offxml", help="OpenFF force field (default: openff-2.2.1.offxml)")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)   # stdout and stderr share the job log
    except Exception:
        pass

    if not os.path.isfile(args.file):
        print(f"Error: Input file '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.file) as f:
        pdb_content = f.read()
    inst = instance_of(pdb_content, args.resname, args.chain)

    # Fail closed: the line is there until this instance is known to be fine, and a
    # manifest line left by an earlier run of this instance goes until it is rebuilt.
    record_failure(LIGAND_FAILED, inst, "parametrisation started but did not finish (crashed or killed)")
    drop_manifest_line(os.path.basename(args.file))
    try:
        flag = parametrize(args, pdb_content, inst)
    except LigandFailure as e:
        print(f"Error: {inst[0]} {inst[1]} {inst[2]}: {e}", file=sys.stderr)
        record_failure(LIGAND_FAILED, inst, str(e))
        sys.exit(1)
    except Exception as e:
        # OpenFF, sqm, RDKit or network errors: keep the traceback the log always had
        sys.stdout.flush()
        traceback.print_exc()
        text = " ".join(str(e).split())
        record_failure(LIGAND_FAILED, inst, "%s: %s" % (type(e).__name__, text[:240]))
        sys.exit(1)
    record_failure(LIGAND_FAILED, inst, flag or None)


if __name__ == "__main__":
    main()
