#!/usr/bin/env python3
#
# fuse_glycans.py - Fuse each covalently attached glycan with its amino acid into one residue
#
# Usage (job.run and job_fe.run, before renumber_pdb.py, when ligands are parametrised,
# i.e. AMBER19SB with OpenFF):
#   python3 fuse_glycans.py -f input.pdb -o fused.pdb -d glycans.json
#       [--max-sugars N] [--exclude-file glycan_failed.gs] [--unfused glycan_unfused.gs]
#
# WHY. A glycan is covalently bonded to its amino acid: the C1 of an N-glycan's first
# GlcNAc sits 1.45 A from Asn ND2. As ligands its sugars would be simulated detached,
# so merge_ligand.py stopped 1MAH (NAG on ASN A 350) and 2AJF (NAGs on ASN A 53, A 322,
# A 546 and E 330) as LIGAND_FAIL. Here the glycan and its amino acid become one
# non-canonical residue, which parametrize_ncaa.py parametrises like any NCAA from the
# chemistry this script writes into glycans.json: heavy atoms, formal charges,
# hydrogen counts and bond orders. That chemistry is never perceived from the geometry
# and never looked up by the residue name: NG1 is itself a CCD entry (GalNAc
# 1-phosphate, parent A2G), and the unmodified NCAA path took that entry's parent and
# its 19-heavy-atom template for the 22-atom fused 1MAH residue.
#
# WHAT IS FUSED. Covalent contacts come first, under covalent.py's rule (the one
# merge_ligand.py applies) on the deposited heavy atoms. Residues renumber_pdb.py
# keeps as protein (ATOM records, its standard names and caps, HETATM residues it
# finds peptide-bonded) are the protein side; the other HETATM residues, water and
# ions left out, form components through their bonds. So do ATOM-record residues
# without a backbone N, CA or C that are no standard residue, water or ion: modelling
# and MD tools write glycans as ATOM records, and as protein such a sugar stopped the
# setup at pdb2gmx without a glycan reason (review F7). A deposited glycosidic link
# longer than covalent.py's rule but shorter than LONG_LINK_A is joined as a link
# (stretched_links(): the anomeric carbon of a saccharide and the closest hydroxyl O
# of another saccharide, or ASN ND2, SER OG or THR OG1), and the log says 'joined
# stretched link NAG K 1 O4-NAG K 2 C1 1.85 A'. That kept 1FZC NAG K 2 (1.85 A from
# NAG K 1 O4) and the 2FD6 site link ASN U 172 ND2-NAG C 1 C1 (1.99 A) from splitting
# or stopping their glycans; the fused residue gets the normal bond parameters, and
# the minimisation pulls the link to bond length. CCD entries are read for the
# members of a component bonded to the protein, and for a residue with a carbon
# within LONG_LINK_A of an oxygen of another residue or a protein acceptor, the
# stretched-link check. A component that contains a saccharide (by the
# _chem_comp.type of its CCD entry) is fused when
#   - every member is a saccharide;
#   - it hangs by exactly one bond on ASN ND2, SER OG or THR OG1 of a residue with
#     peptide neighbours on both sides (C-N < 2.0 A). A contact of a CCD leaving atom
#     (a deposited O1) is no bond: that atom leaves with the link (review F3: a 1YPZ
#     O1 1.51 A from ND2 counted as a second bond). A sugar on the OD1 of an Asn
#     whose ND2 carries none is on a flipped amide, which X-ray density cannot tell
#     from the right one, and N-glycans link through the amide N: OD1 and ND2 swap
#     coordinates and the glycan is fused (review F4, 2PO6 ASN E 42);
#   - its sugars form a tree in which each sugar bonds its parent through its
#     anomeric carbon, the carbon that carries a CCD leaving heavy atom (O1, or O2 of
#     a sialic acid), to an oxygen of the parent (review F8: a C1 at C-C bond
#     distance from a C6 passed as a C-glycoside);
#   - the deposited heavy atoms of each sugar are its CCD heavy atoms without that
#     leaving atom (a deposited one is dropped and logged), and every bond of the
#     definition is at most BOND_SPAN_MAX covalent radius sums long on the deposited
#     coordinates, so the atom names fit the CCD bonds (review F5);
#   - no other saccharide's anomeric carbon, bonded to nothing and not joined, lies
#     within LONG_LINK_A of its sugars: a stretched link left unjoined because it is
#     ambiguous (its leaving O deposited as an atom of its own, two acceptors about
#     equally close, or one acceptor closest to two anomeric carbons). Fusing the
#     part on the protein would leave the rest as a free ligand, so both parts stay
#     ligands, each member with the reason (review F1);
#   - when --max-sugars or $GROSCORE_GLYCAN_MAX_SUGARS sets a cap, it has at most
#     that many sugars (no cap by default);
#   - its residue name is not in --exclude-file (its parametrisation failed), and,
#     in that fallback run, its chemistry is one the first pass fused (review R2:
#     parametrize_ncaa.py made no parameters for any other).
# Anything else stays HETATM (an ATOM-record glycan is written as HETATM records):
# renumber_pdb.py extracts it as ligands, and merge_ligand.py stops it as LIGAND_FAIL
# when it lies in the simulated region, with this script's reason from
# glycan_unfused.gs ('RESNAME CHAIN RESNUM reason', one line per member). A component
# without a saccharide gets no line: a covalently bonded inhibitor stays
# merge_ligand.py's business alone (the first version reported every MSE of 1S1Q,
# the HIC and caps of 2BTF and the TRQ of 2MTA as 'not fused', review F9).
#
# THE FUSED RESIDUE. Named NG<k> on Asn, SG<k> on Ser and TG<k> on Thr, k = 1-9 then
# A-Z, one name per distinct chemistry, in order of first appearance in the file.
# Skipped: names in the input, residue names of the bundled force fields and of
# PDBFixer's and OpenMM's tables, and substrings of amber19sb aminoacids.hdb (NGL).
# An existing glycans.json keeps its names for chemistries that did not change, so
# the fallback re-run after a failed parametrisation renames nothing. The amino acid
# keeps its atom names (N, CA, C, O, CB, CG, OD1, ND2); a sugar heavy atom is its CCD
# name plus a tag letter, A for the sugar on the amino acid, then breadth first with
# children ordered by acceptor atom name, CCD code, chain, number and insertion code
# (C1A, N2A, O7A; C10A and O1AA of a sialic acid), so the same glycan gets the same
# names at any site. Hydrogens are 'H' plus the parent name, without its C for a
# carbon (H1A, H6A1/H6A2, HN2A, HO3A), or HX0-HXZ where that is longer than 4
# characters or taken. Hydrogen counts: the amino acid keeps its in-chain hydrogens
# (CCD hydrogens that are not leaving atoms) less one on the acceptor atom; a sugar
# atom keeps the hydrogens of its CCD entry less one per link it accepts.
#
# fused.pdb is the input with each fused amino acid's records renamed (as ATOM
# records) and its sugars' heavy-atom records moved right behind them as ATOM records
# of the residue: tagged names, the amino acid's chain, number and insertion code,
# element columns. Sugar hydrogens and leaving atoms are dropped. An altloc unit keeps
# one conformer, the altloc OpenMM keeps for the amino acid (its first in the file;
# its sugars' first when it has none), with column 17 blanked; the other altloc
# records of the amino acid and its sugars are dropped and logged, as a sugar record
# left HETATM would be extracted as a ligand and stopped by the covalent rule (review
# F2). The link must be a bond in that conformer. A flipped amide's OD1 and ND2
# records exchange their coordinates. The records of a glycan written as ATOM records
# that is not fused become HETATM records. When nothing is fused and no glycan
# arrived as ATOM records, fused.pdb is a byte copy of the input.
#
# SIZE. Every glycan is fused, however many sugars it has, so that no glycan is
# simulated detached. AM1-BCC time grows about cubically with the residue: on the
# deposited conformer Ac-Asn(GlcNAc)-NHMe took 54-102 s, Asn + GlcNAc + Fuc 161 s and
# Asn + GlcNAc2 + Man 532 s, and a 9-sugar Fc glycan would take about 2 h of the
# setup. $GROSCORE_GLYCAN_MAX_SUGARS (or --max-sugars) sets an optional cap: above it
# a glycan stays a ligand and stops the structure only when it lies in the simulated
# region. The 26 tag letters bound a fused glycan at 26 sugars.
#
# STEREO. Every fused sugar gets a logged comparison with its CCD code: the sign of
# the signed volume at each CCD stereocentre, deposited against the CCD ideal
# coordinates, the leaving atom standing for the aglycone atom. A mismatch (a wrong
# anomer or ring centre in the deposit) is reported, not refused (review F11).
#
# Exit 0 unless the input cannot be read. Standard library only.
#

import os
import re
import sys
import json
import math
import glob
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import ccd_cache  # noqa: E402  (standard library only)
from covalent import COVALENT_RADII, COVALENT_SLACK, contact_kind  # noqa: E402
from parametrize_ligand import element_of, describe_source  # noqa: E402  (standard library at import)

MAX_SUGARS = None          # no cap unless $GROSCORE_GLYCAN_MAX_SUGARS or --max-sugars sets one
MAX_SUGARS_ENV = "GROSCORE_GLYCAN_MAX_SUGARS"
# amino acid: (acceptor atom, residue-name family); ccd_cache.GLYCAN_PARENTS ships them
SITES = {"ASN": ("ND2", "NG"), "SER": ("OG", "SG"), "THR": ("OG1", "TG")}
NAME_SUFFIXES = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TAGS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HX_BASES = ["HX" + c for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
CCD_ORDER = {"SING": 1, "DOUB": 2, "TRIP": 3}
PEPTIDE_BOND_MAX = 2.0     # A, renumber_pdb.py's peptide-bond rule
# renumber_pdb.py's residue classes: these HETATM residues are protein or ions there,
# never ligands, so they are protein (or neither) here too.
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIE", "HID", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
    "HISA", "HISB", "HISH", "HISD", "HISE", "HISP",
    "CYSH", "CYS1", "CYS2", "ASPH", "GLUH", "LYSH", "ARGN",
    "ACE", "NME", "NHE", "NH2",
}
METAL_CLUSTERS = {"FES", "SF4"}
NOT_LIGANDS = ccd_cache.WATERS | ccd_cache.ION_RESIDUES | METAL_CLUSTERS
# Every element pair the rule knows is bonded below r1 + r2 + slack, at most 2.75 A
# (Se-Se), so the 27 cells of that size around an atom hold all its bond partners.
CELL = 2 * max(COVALENT_RADII.values()) + COVALENT_SLACK
# The taken names a scan of 2026-10-04 found (GROMACS 2026.0 share/gromacs/top,
# forcefield/, PDBFixer, OpenMM): SGB is a CHARMM36 residue, and NGL lies inside NGLN,
# NGLU and NGLY of the amber19sb aminoacids.hdb, which parametrize_ncaa.py searched by
# substring until merge_hdb() compared whole entry names; skipping it costs nothing.
# Kept when the scan cannot find the GROMACS installation.
TAKEN_MEASURED = {"NGL", "SGB"}
# An sp3 centre with three heavy neighbours at 1.4-1.5 A spans a signed volume of
# 2.2-2.6 A^3 (C1-C5 of the deposited 1MAH GlcNAc: 2.45-2.62 in magnitude); below
# FLAT_A3 it is close to planar and its handedness is not read.
FLAT_A3 = 0.5
# A deposited glycosidic link can be longer than the covalent rule's C-O limit (1.77
# A): 4 of about 2229 sugar-sugar LINKs of a 400-entry X-ray scan were 1.85-1.98 A
# (1FZC, 1LWU, 1MWA, 2AEP), and site links can be stretched too (2FD6 ASN U 172
# ND2-NAG C 1 C1, 1.99 A). The free anomeric carbon of a saccharide within LONG_LINK_A
# of an acceptor is such a link and is joined (stretched_links()); a non-bonded C...O
# or C...N contact is never that short (review F1).
LONG_LINK_A = 2.5
# The names of a deposited sugar must fit its CCD bonds: every bond of the definition
# at most BOND_SPAN_MAX covalent radius sums (covalent.py) long on the deposited
# coordinates. Over the 1440 fused instances of the scan the longest was 1.35 sums
# (1YY9 C6-O6, 1.92 A, a stretched bond). Names swapped on the 1MAH NAG stretch a
# bond to 1.9-2.6 sums (O4/O6: C4-O4 3.15 A; C2/C4: C2-N2 3.81 A); such a sugar was
# fused with only a stereo warning, where the ligand path's name_misfit() refuses the
# match (review F5).
BOND_SPAN_MAX = 1.5
# A deposited O1 drawn on top of the acceptor (2FD6 NAG C 1 O1, 0.79 A from ASN U 172
# ND2, with C1 1.99 A from ND2) is the acceptor drawn twice, not an atom between the
# two: closer than OVERLAP_A to the acceptor it does not stop a stretched link.
OVERLAP_A = 1.0
# Two acceptors whose distances to one anomeric carbon differ by less than TIE_A leave
# no closest one to pick: that link stays unjoined.
TIE_A = 0.2
SITE_ACCEPTORS = {(res, acc) for res, (acc, _) in SITES.items()}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def read_records(lines):
    """Every ATOM/HETATM record with coordinates, in file order."""
    recs = []
    for i, l in enumerate(lines):
        if not l.startswith(("ATOM  ", "HETATM")) or len(l) < 54:
            continue
        try:
            xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
        except ValueError:
            continue
        recs.append({"i": i, "het": l.startswith("HETATM"), "name": l[12:16].strip(), "alt": l[16],
                     "res": l[17:20].strip(), "chain": l[21], "num": l[22:26].strip(),
                     "icode": l[26] if len(l) > 26 else " ", "el": element_of(l), "xyz": xyz})
    return recs


def rkey(r):
    return (r["res"], r["chain"], r["num"], r["icode"])


def label(key):
    """'NAG A 544', with the insertion code appended and a blank chain written '-'."""
    return "%s %s %s%s" % (key[0], key[1].strip() or "-", key[2], key[3].strip())


def modified_residues(lines):
    """renumber_pdb.py's pre-scan, the same rule: HETATM residue names with a backbone
    N and CA one of whose instances is peptide-bonded (C-N < 2.0 A) to an ATOM residue
    up to four numbers away. renumber_pdb.py keeps every residue of such a name as
    protein (MSE, TRQ, HIC), so they are protein atoms here."""
    het, backbone = {}, {}
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            resname, atom, chain = line[17:20].strip(), line[12:16].strip(), line[21]
            try:
                resnum = int(line[22:26].strip())
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except (ValueError, IndexError):
                continue
            if line.startswith("HETATM") and resname not in ccd_cache.ION_RESIDUES and resname != "HOH":
                if atom in ("N", "CA", "C", "O"):
                    het.setdefault(resname, {}).setdefault((chain, resnum), {})[atom] = xyz
            elif line.startswith("ATOM") and atom in ("N", "C"):
                backbone.setdefault((chain, resnum), {})[atom] = xyz
    found = set()
    for resname, insts in het.items():
        for (chain, resnum), at in insts.items():
            if "N" not in at or "CA" not in at:
                continue
            linked = any("C" in backbone.get((chain, resnum - k), {})
                         and math.dist(backbone[(chain, resnum - k)]["C"], at["N"]) < PEPTIDE_BOND_MAX
                         for k in range(1, 5))
            if not linked and "C" in at:
                linked = any("N" in backbone.get((chain, resnum + k), {})
                             and math.dist(at["C"], backbone[(chain, resnum + k)]["N"]) < PEPTIDE_BOND_MAX
                             for k in range(1, 5))
            if linked:
                found.add(resname)
                break
    return found


class Grid:
    """Heavy atoms binned in CELL-sized cubes, for the bond search."""

    def __init__(self, atoms):
        self.cells = {}
        for a in atoms:
            self.cells.setdefault(self.cell(a["xyz"]), []).append(a)

    @staticmethod
    def cell(xyz):
        return tuple(int(math.floor(c / CELL)) for c in xyz)

    def near(self, xyz):
        cx, cy, cz = self.cell(xyz)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield from self.cells.get((cx + dx, cy + dy, cz + dz), ())


def compatible(a, b):
    """Can the two records coexist: one has no altloc, or both have the same."""
    return a["alt"] == " " or b["alt"] == " " or a["alt"] == b["alt"]


def bonded(a, b):
    """The distance in A when a and b are bonded under covalent.py's rule, else None
    (also for an element the rule does not know)."""
    if a["el"] not in COVALENT_RADII or b["el"] not in COVALENT_RADII:
        return None
    d = math.dist(a["xyz"], b["xyz"])
    return d if contact_kind(a["el"], b["el"], d) == "bond" else None


def link_d(a, b, stretched):
    """bonded(), or the distance of a stretched glycosidic link find_glycans() joined
    between these two atoms (`stretched`: {frozenset of two (residue, atom name)})."""
    d = bonded(a, b)
    if d is None and frozenset(((rkey(a), a["name"]), (rkey(b), b["name"]))) in stretched:
        d = math.dist(a["xyz"], b["xyz"])
        if d >= LONG_LINK_A:
            d = None
    return d


# ---------------------------------------------------------------------------
# CCD entries
# ---------------------------------------------------------------------------

class Entries:
    """CCD entries by residue name, each read once through ccd_cache.fetch (cache first)."""

    def __init__(self):
        self.got = {}

    def __call__(self, name):
        """(entry, None) or (None, why not); entry is component() plus 'type', 'cif' and 'source'."""
        if name not in self.got:
            text, src = ccd_cache.fetch(name)
            if text is None:
                self.got[name] = (None, src)
            else:
                try:
                    e = ccd_cache.component(text)
                    e["cif"] = ccd_cache.parse_cif(text)
                    e["type"] = e["cif"].get("_chem_comp.type", ["?"])[0]
                    e["source"] = describe_source(src)
                    self.got[name] = (e, None)
                except Exception as ex:
                    self.got[name] = (None, "the CCD entry %s could not be read (%s)" % (name, ex))
        return self.got[name]


def is_saccharide(entry):
    return entry is not None and "saccharide" in entry["type"].lower()


def neighbours(entry, name):
    """[(neighbour, its atom dict)] of atom `name` in the entry's bonds."""
    by = {a["name"]: a for a in entry["atoms"]}
    return [(y, by[y]) for n1, n2, o, ar in entry["bonds"] for x, y in ((n1, n2), (n2, n1))
            if x == name and y in by]


def h_count(entry, name, leaving_too):
    """Hydrogens bonded to `name`; leaving_too False counts only those that are not
    leaving atoms (an amino acid's in-chain hydrogens: ASN N keeps H, loses H2)."""
    return sum(1 for y, a in neighbours(entry, name) if a["element"] in ("H", "D") and (leaving_too or not a["leaving"]))


def leaving_heavy(entry, name):
    """CCD leaving heavy atoms bonded to `name`: the anomeric O of a sugar."""
    return [y for y, a in neighbours(entry, name) if a["leaving"] and a["element"] not in ("H", "D")]


def leaving_names(entry):
    """Every CCD leaving heavy atom of the entry (a sugar's O1, an amino acid's OXT);
    none without an entry."""
    return {a["name"] for a in entry["atoms"] if a["leaving"] and a["element"] not in ("H", "D")} if entry else set()


def anomeric_carbons(entry):
    """The carbons of the entry that carry a CCD leaving heavy atom: a sugar's
    anomeric carbon, the one it links through."""
    return {a["name"] for a in entry["atoms"] if a["element"] == "C" and not a["leaving"]
            and leaving_heavy(entry, a["name"])}


def ccd_coords(cif):
    """{atom: (x, y, z)} from the CCD ideal coordinates, the model ones where missing."""
    ids = cif.get("_chem_comp_atom.atom_id", [])
    out = {}
    for key in ("_chem_comp_atom.pdbx_model_Cartn_%s_ideal", "_chem_comp_atom.model_Cartn_%s"):
        cols = [cif.get(key % c, []) for c in "xyz"]
        if all(len(c) == len(ids) for c in cols):
            for n, x, y, z in zip(ids, *cols):
                if n not in out:
                    try:
                        out[n] = (float(x), float(y), float(z))
                    except ValueError:
                        pass
    return out


def signed_volume(c, a, b, d):
    u, v, w = ([p[k] - c[k] for k in range(3)] for p in (a, b, d))
    return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


def stereo_note(entry, dep_xyz, leave, aglycone_xyz):
    """('C1 C2 C3 C4 C5 as in NAG', mismatch?) with inverted, flat and unchecked
    centres named. dep_xyz: deposited coordinates by CCD atom name; a leaving atom
    stands for the aglycone atom (the acceptor the sugar is bonded to)."""
    cif, code = entry["cif"], entry["id"] or "?"
    ideal = ccd_coords(cif)
    same, inverted, flat, unchecked = [], [], [], []
    for name, conf in zip(cif.get("_chem_comp_atom.atom_id", []), cif.get("_chem_comp_atom.pdbx_stereo_config", [])):
        if conf.upper() not in ("R", "S"):
            continue
        nbrs = sorted({y for y, a in neighbours(entry, name) if a["element"] not in ("H", "D")})[:3]
        dep = [aglycone_xyz if n in leave else dep_xyz.get(n) for n in nbrs]
        ref = [ideal.get(n) for n in nbrs]
        if len(nbrs) < 3 or name not in dep_xyz or name not in ideal or None in dep or None in ref:
            unchecked.append(name)
            continue
        v_dep = signed_volume(dep_xyz[name], *dep)
        if abs(v_dep) < FLAT_A3:
            flat.append(name)
        elif (v_dep > 0) == (signed_volume(ideal[name], *ref) > 0):
            same.append(name)
        else:
            inverted.append(name)
    parts = []
    if inverted:
        parts.append("%s inverted against %s" % (" ".join(inverted), code))
    if flat:
        parts.append("%s flat (|V| < %.1f A^3), not read" % (" ".join(flat), FLAT_A3))
    if same:
        parts.append("%s as in %s" % (" ".join(same), code))
    if unchecked:
        parts.append("%s not checked" % " ".join(unchecked))
    return "; ".join(parts) or "no CCD stereocentre", bool(inverted or flat)


# ---------------------------------------------------------------------------
# Residue names
# ---------------------------------------------------------------------------

def gromacs_top_dirs():
    """The bundled force fields: GROMACS's share/gromacs/top and the repository's
    forcefield/. Neither $GMXLIB nor the working directory: the amber19sb.ff copy there
    holds the fused residues of an earlier pass, and the names would drift."""
    dirs = []
    if os.environ.get("GMXDATA"):
        dirs.append(os.path.join(os.environ["GMXDATA"], "top"))
    gmx = shutil.which("gmx")
    if gmx:
        dirs.append(os.path.join(os.path.dirname(os.path.realpath(gmx)), "..", "share", "gromacs", "top"))
    dirs.append(os.path.join(REPO, "forcefield"))
    out = []
    for d in dirs:
        if os.path.isdir(d) and os.path.realpath(d) not in [os.path.realpath(x) for x in out]:
            out.append(d)
    return out


def tool_residue_names():
    """Residue names PDBFixer substitutes (its 'substitutions' table) or OpenMM's PDB
    reader knows or renames (app/data/pdbNames.xml), read from the installed packages
    without importing them (23 ms)."""
    import ast
    import importlib.util
    names = set()
    try:
        spec = importlib.util.find_spec("pdbfixer")
        with open(os.path.join(list(spec.submodule_search_locations)[0], "pdbfixer.py")) as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "substitutions"
                                                    for t in node.targets):
                names |= set(ast.literal_eval(node.value))
    except Exception:
        pass
    try:
        spec = importlib.util.find_spec("openmm")
        with open(os.path.join(list(spec.submodule_search_locations)[0], "app", "data", "pdbNames.xml")) as f:
            text = f.read()
        for m in re.finditer(r"<Residue\s+([^>]*)>", text):
            names |= set(re.findall(r'(?:name|alt\d+)="([^"]+)"', m.group(1)))
    except Exception:
        pass
    return names


def taken_names():
    """The candidate names NG1..NGZ, SG1..SGZ and TG1..TGZ that a force field or tool
    already uses: rtp entries and residuetypes.dat of the bundled force fields, the
    PDBFixer and OpenMM tables, and substrings of the amber19sb aminoacids.hdb."""
    candidates = [f + c for _, f in SITES.values() for c in NAME_SUFFIXES]
    names, texts = tool_residue_names(), []
    for d in gromacs_top_dirs():
        rt = os.path.join(d, "residuetypes.dat")
        if os.path.isfile(rt):
            with open(rt, errors="replace") as f:
                names |= {p.split()[0] for p in f if p.split()}
        for rtp in glob.glob(os.path.join(d, "*.ff", "*.rtp")):
            with open(rtp, errors="replace") as f:
                names |= set(re.findall(r"^\s*\[\s*(\S+)\s*\]", f.read(), re.M))
        hdb = os.path.join(d, "amber19sb.ff", "aminoacids.hdb")
        if os.path.isfile(hdb):
            with open(hdb, errors="replace") as f:
                texts.append(f.read())
    return TAKEN_MEASURED | {c for c in candidates if c in names or any(c in t for t in texts)}


def next_name(family, taken):
    """The first of family + 1..9, A..Z that is not taken, or None."""
    for c in NAME_SUFFIXES:
        if family + c not in taken:
            return family + c
    return None


def signature(d):
    """The chemistry of a definition: parent, link, sugar codes in tag order, atoms
    (name, element, charge, h) and bonds. One residue name per signature."""
    return json.dumps([d["parent"], d["link"], [s["ccd"] for s in d["sugars"]],
                       [[a["name"], a["element"], a["charge"], a["h"]] for a in d["atoms"]],
                       sorted(sorted(b[:2]) + [b[2]] for b in d["bonds"])])


# ---------------------------------------------------------------------------
# One glycan
# ---------------------------------------------------------------------------

def amino_acid_h_names(atoms):
    """Hydrogen names the amino-acid part may get downstream (parametrize_ncaa.py's
    assign_h_names() rule and the AMBER names, a superset); the sugar hydrogens keep
    clear of them."""
    names = {"H", "HA"}
    for a in atoms:
        n, k = a["name"], a["h"]
        if k:
            names.update(["H" + n, "H" + n[1:]] + ["H%s%d" % (n[1:], j) for j in range(1, k + 2)])
    return names


def hydrogen_names(atoms, sugar_atoms, reserved):
    """{sugar heavy atom: [H names]}: 'H' + the name without its C for a carbon, 'H' +
    the name otherwise, digits 1..n for a group; HX0..HXZ when that is longer than 4
    characters or taken (the SIA C11 methyl: H11A1 has 5). Raises ValueError when the
    36 HX bases run out."""
    used = set(reserved) | {a["name"] for a in atoms}
    bases = iter(HX_BASES)
    out = {}
    for a in atoms:
        if a["name"] not in sugar_atoms or a["h"] == 0:
            continue
        n = a["h"]
        group = (lambda base: [base] if n == 1 else ["%s%d" % (base, j + 1) for j in range(n)])
        names = group("H" + (a["name"][1:] if a["element"] == "C" else a["name"]))
        while any(len(x) > 4 or x in used for x in names):
            base = next(bases, None)
            if base is None:
                raise ValueError("no free hydrogen name left in HX0-HXZ for %s" % a["name"])
            names = group(base)
        used.update(names)
        out[a["name"]] = names
    return out


def conformer_of(unit, records):
    """The altloc letter of a fusion unit [amino acid, sugars...]: the first altloc of
    the amino acid's records in file order, the one OpenMM's reader keeps, else the
    first of its sugars'; None without altlocs."""
    for k in unit:
        for r in records[k]:
            if r["alt"] != " ":
                return r["alt"]
    return None


def heavy(rs, alt):
    """The heavy-atom records of one residue in the chosen conformer."""
    return [r for r in rs if r["el"] not in ("H", "") and r["alt"] in (" ", alt or " ")]


def atom_check(what, deposited, need, extra_said):
    """'' or 'NAG A 544: missing O6, ...': deposited heavy atoms against the needed
    ones; extra_said introduces deposited atoms that are not needed."""
    names = [r["name"] for r in deposited]
    dep = {r["name"]: r for r in deposited}
    want = {a["name"] for a in need}
    missing = [a["name"] for a in need if a["name"] not in dep]
    extra = [n for n in dep if n not in want]
    dup = sorted({n for n in names if names.count(n) > 1})
    wrong = ["%s is %s, %s in the CCD entry" % (a["name"], dep[a["name"]]["el"], a["element"]) for a in need
             if a["name"] in dep and dep[a["name"]]["el"] != a["element"]]
    parts = [("missing " + " ".join(missing)) if missing else "",
             ("%s: %s" % (extra_said, " ".join(extra))) if extra else "",
             ("duplicate atoms " + " ".join(dup)) if dup else ""] + wrong
    parts = [p for p in parts if p]
    return "%s: %s" % (what, ", ".join(parts)) if parts else ""


def build_unit(site, members, records, entries, flip=False, stretched=frozenset()):
    """((definition without a name, instance, report entry, what to write), None) for
    the glycan `members` on the amino acid `site`, or (None, reason). flip: the glycan
    is bonded to the OD1 of the Asn `site`, whose OD1 and ND2 exchange coordinates.
    stretched: the links find_glycans() joined beyond the covalent rule, which count as
    bonds here (link_d())."""
    res = site[0]
    acc_name, family = SITES[res]
    alt = conformer_of([site] + members, records)
    if alt is not None:
        lacking = [label(k) for k in members if any(r["alt"] != " " for r in records[k])
                   and not any(r["alt"] == alt for r in records[k])]
        if lacking:
            return None, "%s %s no altloc %s, the conformer OpenMM keeps for %s" % (
                ", ".join(lacking), "has" if len(lacking) == 1 else "have", alt, label(site))
    aa, aa_why = entries(res)
    if aa is None:
        return None, "no CCD entry for %s (%s)" % (res, aa_why)
    in_chain = [a for a in aa["atoms"] if a["element"] not in ("H", "D") and not a["leaving"]]
    aa_dep = heavy(records[site], alt)
    if flip:
        by = {r["name"]: r for r in aa_dep}
        if "OD1" in by and "ND2" in by:
            aa_dep = [dict(r, xyz=by[{"OD1": "ND2", "ND2": "OD1"}[r["name"]]]["xyz"]) if r["name"] in ("OD1", "ND2")
                      else r for r in aa_dep]
    why = [x for x in [atom_check(label(site), aa_dep, in_chain, "atoms that are not in-chain %s atoms" % res)] if x]
    acc = next((r for r in aa_dep if r["name"] == acc_name), None)
    # the sugars in that conformer, their bonds to each other and to the acceptor. A CCD
    # leaving atom (a deposited O1) makes no bond: it leaves with the link (review F3).
    sug = {k: heavy(records[k], alt) for k in members}
    linkable = {k: [r for r in sug[k] if r["name"] not in leaving_names(entries(k[0])[0])] for k in members}
    pairs = {}                       # {frozenset(k1, k2): [(k1, atom of k1, k2, atom of k2, d)]}
    for x, k1 in enumerate(members):
        for k2 in members[x + 1:]:
            for a in linkable[k1]:
                for b in linkable[k2]:
                    d = link_d(a, b, stretched)
                    if d is not None:
                        pairs.setdefault(frozenset((k1, k2)), []).append((k1, a, k2, b, d))
    root = [(k, b, d) for k in members for b in linkable[k] for d in [link_d(acc, b, stretched) if acc else None]
            if d is not None]
    if len(root) != 1:
        where = " in altloc %s, the conformer OpenMM keeps" % alt if alt else ""
        why.append("the link to %s %s is not a bond%s" % (label(site), acc_name, where) if not root else
                   "%s %s bonds %d sugar atoms%s" % (label(site), acc_name, len(root), where))
        return None, "; ".join(why)
    twice = ["%s and %s" % (label(ls[0][0]), label(ls[0][2])) for ls in pairs.values() if len(ls) > 1]
    if twice:
        why.append("bonded to each other more than once: " + "; ".join(twice))
    # the tree, breadth first from the sugar on the amino acid; donor[k] = (k's atom
    # bonded to its parent, the parent's acceptor atom, d)
    rk, rb, rd = root[0]
    order, parent, donor = [rk], {rk: None}, {rk: (rb, acc, rd)}
    head = 0
    while head < len(order):
        k = order[head]
        head += 1
        kids = []
        for p, ls in pairs.items():
            other = next(iter(p - {k})) if k in p else None
            if other is None or other in parent:
                continue
            k1, a, k2, b, d = ls[0]
            mine, theirs = (a, b) if k1 == k else (b, a)
            kids.append(((mine["name"],) + other, other, theirs, mine, d))
        for _, other, theirs, mine, d in sorted(kids, key=lambda t: t[0]):
            order.append(other)
            parent[other] = k
            donor[other] = (theirs, mine, d)
    if len(order) != len(members) or len(pairs) != len(members) - 1:
        why.append("the sugars do not form a tree hanging from %s %s (%d sugars, %d bonded pairs%s)" % (
            label(site), acc_name, len(members), len(pairs), ", altloc %s" % alt if alt else ""))
    if len(order) > len(TAGS):
        why.append("%d sugars, more than the %d tag letters" % (len(order), len(TAGS)))
    if why:
        return None, "; ".join(why)
    tag = {k: TAGS[n] for n, k in enumerate(order)}
    accepted = {}
    for k in order[1:]:
        key = (parent[k], donor[k][1]["name"])
        accepted[key] = accepted.get(key, 0) + 1
        # A glycosidic link joins the anomeric carbon to an oxygen of the parent; a C1
        # at C-C bond distance from a C6 is no such link (review F8).
        if donor[k][1]["el"] != "O":
            why.append("%s is linked to %s %s, a %s, not to a hydroxyl O" % (
                label(k), label(parent[k]), donor[k][1]["name"], donor[k][1]["el"]))
    # the atoms: the amino acid's in CCD order, then each sugar's in CCD order
    in_names = {a["name"] for a in in_chain}
    atoms = [{"name": a["name"], "element": a["element"], "charge": a["charge"],
              "h": h_count(aa, a["name"], leaving_too=False) - (1 if a["name"] == acc_name else 0),
              "src": "%s %s" % (label(site), a["name"])} for a in in_chain]
    bonds = []
    for n1, n2, o, ar in aa["bonds"]:
        if n1 in in_names and n2 in in_names:
            if o not in CCD_ORDER:
                why.append("bond order %s %s-%s in the CCD entry %s" % (o, n1, n2, res))
            bonds.append([n1, n2, CCD_ORDER.get(o, 1)])
    seen, keep, stereo, sugars, dropped = set(in_names), {}, [], [], []
    for k in order:
        e, e_why = entries(k[0])
        if e is None:
            why.append("no CCD entry for %s (%s)" % (k[0], e_why))
            continue
        anomeric = donor[k][0]["name"]
        leave = leaving_heavy(e, anomeric)
        carriers = [a["name"] for a in e["atoms"] if a["element"] == "C" and not a["leaving"]
                    and leaving_heavy(e, a["name"])]
        if anomeric not in carriers or len(leave) != 1:
            why.append("%s is bonded to %s through %s, not through its anomeric carbon (%s)" % (
                label(k), label(parent[k]) if parent[k] else "%s %s" % (label(site), acc_name), anomeric,
                "the carbon carrying a CCD leaving O: " + " ".join(carriers) if carriers
                else "its CCD entry has no leaving heavy atom on a carbon"))
            continue
        need = [a for a in e["atoms"] if a["element"] not in ("H", "D") and a["name"] not in leave]
        bad = atom_check(label(k), [r for r in sug[k] if r["name"] not in leave], need,
                         "atoms not in the CCD entry %s" % k[0])
        if bad:
            why.append(bad)
            continue
        dep = {r["name"]: r for r in sug[k]}
        dropped += ["%s %s" % (label(k), n) for n in leave if n in dep]
        keep[k] = {}
        for a in need:
            new = a["name"] + tag[k]
            if len(new) > 4 or new in seen or not new.upper().startswith(a["element"]):
                why.append("%s atom %s cannot be tagged %s (at most 4 characters, unique, starting with its "
                           "element)" % (label(k), a["name"], new))
                continue
            seen.add(new)
            h = h_count(e, a["name"], leaving_too=True) - accepted.get((k, a["name"]), 0)
            if h < 0:
                why.append("%s %s accepts %d link(s) but carries %d hydrogen(s) in the CCD entry" % (
                    label(k), a["name"], accepted[(k, a["name"])], h + accepted[(k, a["name"])]))
            keep[k][a["name"]] = new
            atoms.append({"name": new, "element": a["element"], "charge": a["charge"], "h": h,
                          "src": "%s %s" % (label(k), a["name"])})
        for n1, n2, o, ar in e["bonds"]:
            if n1 in keep[k] and n2 in keep[k]:
                if o not in CCD_ORDER:
                    why.append("bond order %s %s-%s in the CCD entry %s" % (o, n1, n2, k[0]))
                bonds.append([keep[k][n1], keep[k][n2], CCD_ORDER.get(o, 1)])
        note, odd = stereo_note(e, {n: r["xyz"] for n, r in dep.items()}, leave, donor[k][1]["xyz"])
        stereo.append(("%s (%s): %s" % (label(k), tag[k], note), odd))
        sugars.append({"tag": tag[k], "ccd": k[0], "type": e["type"], "source": e["source"],
                       "modified": e["modified"] or "?"})
    # The atom names must fit the CCD bonds on the deposited coordinates (BOND_SPAN_MAX).
    pos = {r["name"]: r["xyz"] for r in aa_dep}
    for k, names in keep.items():
        pos.update((names[r["name"]], r["xyz"]) for r in sug[k] if r["name"] in names)
    el = {a["name"]: a["element"] for a in atoms}
    src = {a["name"]: a["src"].rsplit(" ", 1) for a in atoms}          # (residue label, CCD atom name)
    stretched = {}
    for n1, n2, o in bonds:
        if n1 in pos and n2 in pos and el[n1] in COVALENT_RADII and el[n2] in COVALENT_RADII:
            d = math.dist(pos[n1], pos[n2])
            if d > BOND_SPAN_MAX * (COVALENT_RADII[el[n1]] + COVALENT_RADII[el[n2]]):
                stretched.setdefault(src[n1][0], []).append("%s-%s %.2f A" % (src[n1][1], src[n2][1], d))
    for what, ls in stretched.items():
        why.append("%s: the atom names do not fit the CCD entry's bonds, %s (more than %.1f covalent radius sums)"
                   % (what, ", ".join(ls), BOND_SPAN_MAX))
    if why:
        return None, "; ".join(why)
    link = [acc_name, keep[rk][rb["name"]]]
    bonds.append(link + [1])
    for k in order[1:]:
        bonds.append([keep[parent[k]][donor[k][1]["name"]], keep[k][donor[k][0]["name"]], 1])
    try:
        hn = hydrogen_names(atoms, {n for k in order for n in keep[k].values()},
                            amino_acid_h_names(atoms[:len(in_chain)]))
    except ValueError as ex:
        return None, str(ex)
    definition = {"resname": None, "family": family, "parent": res, "link": link, "atoms": atoms,
                  "bonds": bonds, "hnames": hn, "sugars": sugars, "instances": []}
    instance = {"chain": site[1], "resnum": site[2], "icode": site[3].strip(),
                "sugars": [label(k) for k in order], "link_A": round(rd, 3), "stereo": [s for s, _ in stereo]}
    # The joined stretched links, for the record; pdb2gmx gives them the normal bond
    # parameters, and the minimisation pulls them to bond length.
    long_ones = (["%s %s-%s %s %.2f A" % (label(site), acc_name, label(rk), rb["name"], rd)] if bonded(acc, rb) is None
                 else []) + ["%s %s-%s %s %.2f A" % (label(parent[k]), donor[k][1]["name"], label(k), donor[k][0]["name"],
                                                    donor[k][2]) for k in order[1:] if bonded(donor[k][0], donor[k][1]) is None]
    if long_ones:
        instance["stretched_links"] = long_ones
    report = {"site": label(site), "sugars": [label(k) for k in order],
              "link": "%s-%s %.2f A" % (acc_name, rb["name"], rd), "stereo": stereo, "dropped": dropped, "alt": alt,
              "flip": flip}
    write = {"site": site, "alt": alt, "order": order, "keep": keep, "flip": flip}
    return (definition, instance, report, write), None


# ---------------------------------------------------------------------------
# The input
# ---------------------------------------------------------------------------

def stretched_links(cand, pgrid, cgrid, records, entries, plinks, linked):
    """({(residue, carbon): why not joined}, [(carbon record, acceptor record, d,
    acceptor on the protein?)]): the glycosidic links deposited longer than the
    covalent rule, up to LONG_LINK_A, that count as links.

    A carbon qualifies when it is the anomeric carbon of a saccharide (it carries the
    CCD leaving O) and has no bond to another residue. Its acceptor is the closest
    atom within LONG_LINK_A that could take the link: a hydroxyl O (one with a
    hydrogen in its CCD entry, not a leaving atom) of another saccharide, or ASN ND2,
    SER OG or THR OG1 of the protein, in either case not yet bonded to another
    residue (bonds through a leaving atom do not count, as they leave with the link).
    Not joined, and so still refused with the reason, when it is ambiguous:
      - its leaving O is deposited as an atom of its own (farther than OVERLAP_A from
        the acceptor): the carbon is a reducing end next to something, not a link;
      - the two closest acceptors lie within TIE_A of each other;
      - another anomeric carbon picked the same acceptor.
    CCD entries are read only for a residue with such a contact."""
    taken = {}

    def site_free(p):
        """No residue is bonded to this protein acceptor except through a leaving atom."""
        if (rkey(p), p["name"]) not in taken:
            taken[(rkey(p), p["name"])] = not any(
                (l[2], l[3]) == (rkey(p), p["name"]) and l[1] not in leaving_names(entries(l[0][0])[0])
                for l in plinks)
        return taken[(rkey(p), p["name"])]
    unjoined, claims = {}, {}
    for c in cand:
        if c["el"] != "C" or (rkey(c), c["name"]) in linked:
            continue
        near = [(math.dist(c["xyz"], q["xyz"]), q, False) for q in cgrid.near(c["xyz"])
                if q["el"] == "O" and rkey(q) != rkey(c) and compatible(c, q) and (rkey(q), q["name"]) not in linked]
        near += [(math.dist(c["xyz"], p["xyz"]), p, True) for p in pgrid.near(c["xyz"])
                 if (p["res"], p["name"]) in SITE_ACCEPTORS and compatible(c, p)]
        near = sorted((t for t in near if t[0] < LONG_LINK_A), key=lambda t: t[0])
        if not near:
            continue
        e = entries(c["res"])[0]
        if not is_saccharide(e) or c["name"] not in anomeric_carbons(e):
            continue
        ok = []
        for d, q, site in near:
            if site:
                if not site_free(q):
                    continue
            else:
                eq = entries(q["res"])[0]
                if not is_saccharide(eq) or q["name"] in leaving_names(eq) or h_count(eq, q["name"], True) == 0:
                    continue
            ok.append((d, q, site))
        if not ok:
            continue
        d, q, site = ok[0]
        key = (rkey(c), c["name"])
        own = [r for r in records[rkey(c)] if r["name"] in leaving_heavy(e, c["name"]) and compatible(r, c)
               and math.dist(r["xyz"], q["xyz"]) >= OVERLAP_A]
        if own:
            unjoined[key] = "its leaving atom %s is deposited, %.2f A from %s" % (
                own[0]["name"], math.dist(own[0]["xyz"], c["xyz"]), c["name"])
        elif len(ok) > 1 and ok[1][0] - d < TIE_A:
            unjoined[key] = "%s %s at %.2f A and %s %s at %.2f A are about equally close" % (
                label(rkey(q)), q["name"], d, label(rkey(ok[1][1])), ok[1][1]["name"], ok[1][0])
        else:
            claims.setdefault((rkey(q), q["name"]), []).append((c, q, d, site))
    joined = []
    for acc, cl in claims.items():
        if len(cl) == 1:
            joined.append(cl[0])
            continue
        for c, q, d, site in cl:
            unjoined[(rkey(c), c["name"])] = "%s %s is the closest acceptor of %d anomeric carbons" % (
                label(acc[0]), acc[1], len(cl))
    return unjoined, sorted(joined, key=lambda t: t[0]["i"])


def find_glycans(lines, recs, entries, cap):
    """([(members, site key or None, reason or None, flipped amide?)], notes, records
    by residue, stretched links joined): every component bonded to the protein that contains a saccharide, in
    order of first appearance, with the reason it cannot be fused when there is one at
    this level, and the components linked to one of them beyond the covalent rule, with
    that reason; notes name the bonded components without a saccharide."""
    modified = modified_residues(lines)
    backbone = {rkey(r) for r in recs if r["name"] in ("N", "CA", "C")}
    records, prot, cand = {}, [], []
    for r in recs:
        records.setdefault(rkey(r), []).append(r)
        if r["el"] not in COVALENT_RADII:
            continue                      # hydrogens, metals: no part of a covalent link here
        if (r["res"] in STANDARD_RESIDUES or r["res"] in modified
                or (not r["het"] and (rkey(r) in backbone or r["res"] in NOT_LIGANDS))):
            if r["res"] not in ccd_cache.ION_RESIDUES:
                prot.append(r)
        elif r["res"] not in NOT_LIGANDS:
            cand.append(r)                # HETATM, or an ATOM-record residue without a backbone
    pgrid, cgrid = Grid(prot), Grid(cand)
    plinks, up, linked = set(), {}, set()

    def top(k):
        while up.get(k, k) != k:
            k = up[k]
        return k
    for c in cand:
        for p in pgrid.near(c["xyz"]):
            if compatible(c, p) and bonded(c, p) is not None:
                plinks.add((rkey(c), c["name"], rkey(p), p["name"]))
                linked.add((rkey(c), c["name"]))
        for q in cgrid.near(c["xyz"]):
            if q["i"] > c["i"] and rkey(q) != rkey(c) and compatible(c, q) and bonded(c, q) is not None:
                linked.update(((rkey(c), c["name"]), (rkey(q), q["name"])))
                a, b = top(rkey(c)), top(rkey(q))
                if a != b:
                    up[a] = b
    unjoined, joined = stretched_links(cand, pgrid, cgrid, records, entries, plinks, linked)
    stretched = set()
    for c, q, d, site in joined:
        stretched.add(frozenset(((rkey(c), c["name"]), (rkey(q), q["name"]))))
        linked.add((rkey(c), c["name"]))
        if site:
            plinks.add((rkey(c), c["name"], rkey(q), q["name"]))
        else:
            linked.add((rkey(q), q["name"]))
            a, b = top(rkey(c)), top(rkey(q))
            if a != b:
                up[a] = b
    comps = {}
    for r in cand:
        members = comps.setdefault(top(rkey(r)), [])
        if rkey(r) not in members:
            members.append(rkey(r))

    def long_links(members):
        """{residue: (its anomeric carbon, member, member atom, d)}: the closest contact of
        each saccharide outside `members` whose anomeric carbon, bonded to nothing, lies
        within LONG_LINK_A of a member's heavy atom."""
        found = {}
        for k in members:
            for m in records[k]:
                if m["el"] not in COVALENT_RADII:
                    continue
                for q in cgrid.near(m["xyz"]):
                    o = rkey(q)
                    if o in members or q["el"] != "C" or (o, q["name"]) in linked or not compatible(m, q):
                        continue
                    d = math.dist(m["xyz"], q["xyz"])
                    if d >= LONG_LINK_A or (o in found and found[o][3] <= d):
                        continue
                    e = entries(o[0])[0]
                    if is_saccharide(e) and q["name"] in anomeric_carbons(e):
                        found[o] = (q["name"], k, m["name"], d)
        return found
    out, notes, beyond = {}, [], {}
    for root, members in comps.items():
        mine = sorted((l for l in plinks if l[0] in members), key=lambda l: (members.index(l[0]), l[1], l[2], l[3]))
        if not mine:
            continue                      # not bonded to the protein: an ordinary ligand
        got = {k: entries(k[0]) for k in members}
        if not any(is_saccharide(e) for e, _ in got.values()):
            unknown = ["%s (%s)" % (k[0], w) for k, (e, w) in got.items() if e is None]
            notes.append("bonded to the protein, not a glycan (no saccharide): %s%s" % (
                ", ".join(label(k) for k in members), "; no CCD entry for " + ", ".join(unknown) if unknown else ""))
            continue
        why, site, flip = [], None, False
        said = lambda ls: ", ".join("%s %s-%s %s" % (label(l[0]), l[1], label(l[2]), l[3]) for l in ls)
        links = [l for l in mine if l[1] not in leaving_names(got[l[0]][0])]
        if not links:
            gone = sorted({l[1] for l in mine})
            why.append("bonded to the protein only through the CCD leaving atom%s %s (%s), which the link replaces"
                       % ("s" if len(gone) > 1 else "", " ".join(gone), said(mine)))
        elif len(links) != 1:
            why.append("%d bonds to the protein (%s)" % (len(links), said(links)))
        else:
            s_key, s_atom = links[0][2], links[0][3]
            flip = s_key[0] == "ASN" and s_atom == "OD1"
            if SITES.get(s_key[0], (None,))[0] != s_atom and not flip:
                why.append("attached to %s %s, not to ASN ND2, SER OG or THR OG1" % (label(s_key), s_atom))
            else:
                site = s_key
                for atom, other, side in (("N", "C", "before"), ("C", "N", "after")):
                    if not any(p["name"] == other and rkey(p) != site and math.dist(p["xyz"], a["xyz"]) < PEPTIDE_BOND_MAX
                               for a in records[site] if a["name"] == atom for p in pgrid.near(a["xyz"])):
                        why.append("%s has no peptide neighbour %s it" % (label(site), side))
        noentry = sorted({"%s (%s)" % (k[0], w) for k, (e, w) in got.items() if e is None})
        if noentry:
            why.append("no CCD entry for " + ", ".join(noentry))
        nonsug = ["%s (%s)" % (label(k), e["type"]) for k, (e, w) in got.items() if e is not None and not is_saccharide(e)]
        if nonsug:
            why.append("not all saccharides: " + ", ".join(nonsug))
        if cap is not None and len(members) > cap:
            why.append("%d sugars, more than the cap of %d" % (len(members), cap))
        for o, (qa, k, ma, d) in sorted(long_links(members).items(), key=lambda t: (t[1][3], t[0])):
            text = "%s %s lies %.2f A from %s %s, a glycosidic link longer than the covalent rule that is not " \
                   "joined (%s)" % (label(o), qa, d, label(k), ma, unjoined.get(
                       (o, qa), "no hydroxyl O of a sugar, ASN ND2, SER OG or THR OG1 within %.1f A of it" % LONG_LINK_A))
            why.append(text)
            beyond.setdefault(top(o), []).append(text)
        out[root] = (members, site, why, flip)
    found = []
    for r in comps:
        if r in out:
            members, site, why, flip = out[r]
            why = why + [t for t in beyond.get(r, []) if t not in why]
            found.append((members, site, "; ".join(why) or None, flip))
    # A component such a link joins to a glycan stays ligands with the same reason, also
    # when it is not bonded to the protein itself (1FZC NAG K 2).
    found += [(comps[r], None, "; ".join(beyond[r]), False) for r in comps if r in beyond and r not in out]
    notes = ["joined stretched link %s %s-%s %s %.2f A" % (label(rkey(q)), q["name"], label(rkey(c)), c["name"], d)
             for c, q, d, site in joined] + notes
    return found, notes, records, frozenset(stretched)


def read_exclusions(path):
    """{residue name: reason} from glycan_failed.gs ('RESNAME reason')."""
    out = {}
    if path and os.path.isfile(path):
        with open(path) as f:
            for line in f:
                p = line.split(None, 1)
                if p:
                    out[p[0]] = " ".join(p[1].split()) if len(p) > 1 else "no reason recorded"
    return out


def read_old_names(path):
    """{signature: residue name} of an existing glycans.json, fused and excluded."""
    try:
        with open(path) as f:
            old = json.load(f)
        return {signature(d): d["resname"] for d in old.get("residues", []) + old.get("excluded", [])}
    except Exception:
        return {}


def write_fused(lines, units, records, out_path, to_het=()):
    """The input with each unit's amino acid renamed (ATOM records, one conformer,
    column 17 blank, a flipped amide's OD1 and ND2 coordinates exchanged) and its sugar
    heavy atoms moved behind it as ATOM records, and the records of the line numbers
    in `to_het` written as HETATM; returns the notes on dropped altloc records."""
    drop, rename, insert, notes, swap = set(), {}, {}, [], {}
    for u in units:
        site, alt, dropped = u["site"], u["alt"], {}
        for r in records[site]:
            if r["alt"] in (" ", alt or " "):
                rename[r["i"]] = u["resname"]
            else:
                drop.add(r["i"])
                dropped[label(site)] = dropped.get(label(site), 0) + 1
        if u.get("flip"):
            at = {r["name"]: r["i"] for r in records[site] if r["alt"] in (" ", alt or " ")}
            swap.update({at["OD1"]: at["ND2"], at["ND2"]: at["OD1"]})
        resnum_icode = lines[records[site][0]["i"]][22:27]
        new = []
        for k in u["order"]:
            for r in records[k]:
                drop.add(r["i"])
                if r["alt"] not in (" ", alt or " "):
                    dropped[label(k)] = dropped.get(label(k), 0) + 1
                    continue
                nm = u["keep"][k].get(r["name"])
                if nm is None:
                    continue                      # a hydrogen, or the leaving atom
                el = u["elements"][nm]
                l = lines[r["i"]].ljust(80)
                field = nm.ljust(4) if len(nm) == 4 or len(el) == 2 else " " + nm.ljust(3)
                new.append(("ATOM  " + l[6:12] + field + " " + "%3s" % u["resname"] + l[20] + site[1] + resnum_icode
                            + l[27:76] + "%2s" % el + l[78:80]).rstrip())
        insert[max(r["i"] for r in records[site])] = new
        if dropped:
            notes.append("  altloc %s kept for %s and its sugars (the conformer OpenMM keeps); dropped %s" % (
                alt, label(site), ", ".join("%d record(s) of %s" % (n, k) for k, n in dropped.items())))
    out, to_het = [], set(to_het)
    for i, l in enumerate(lines):
        if i not in drop:
            if i in rename:
                l = ("ATOM  " + l[6:] if l.startswith("HETATM") else l)
                l = l[:16] + " " + "%3s" % rename[i] + l[20:]
                if i in swap:
                    l = l[:30] + lines[swap[i]][30:54] + l[54:]
            elif i in to_het:
                l = "HETATM" + l[6:]
            out.append(l)
        out.extend(insert.get(i, ()))
    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")
    return notes


def positive_int(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def default_cap():
    """(cap, where from): $GROSCORE_GLYCAN_MAX_SUGARS when it is a whole number of at
    least 1, else MAX_SUGARS, None for no cap (a bad value is said, not silently
    replaced)."""
    v = os.environ.get(MAX_SUGARS_ENV, "").strip()
    if v:
        try:
            if int(v) >= 1:
                return int(v), "$" + MAX_SUGARS_ENV
        except ValueError:
            pass
        print("Warning: $%s=%r is not a whole number of at least 1; %s" % (
            MAX_SUGARS_ENV, v, "no cap" if MAX_SUGARS is None else "the cap stays %d" % MAX_SUGARS))
    return MAX_SUGARS, "default"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fuse each covalently attached glycan with its Asn, Ser or Thr into one "
                                             "residue for the NCAA path.")
    ap.add_argument("-f", required=True, help="input PDB (input.pdb)")
    ap.add_argument("-o", required=True, help="output PDB (fused.pdb); a byte copy of the input when nothing is fused")
    ap.add_argument("-d", required=True, help="definitions for parametrize_ncaa.py (glycans.json); an existing one "
                                              "lends its residue names to the chemistries that did not change")
    ap.add_argument("--max-sugars", type=positive_int, default=None,
                    help="fuse glycans of at most this many sugars (default $%s, else no cap)" % MAX_SUGARS_ENV)
    ap.add_argument("--exclude-file", default=None,
                    help="'RESNAME reason' lines (glycan_failed.gs): fused residues that failed to parametrise; their "
                         "glycans stay ligands and their sugars get the reason in --unfused. This is the fallback "
                         "after parametrize_ncaa.py, so only chemistries the existing -d file defines are fused: it "
                         "made parameters for no other")
    ap.add_argument("--unfused", default="glycan_unfused.gs",
                    help="'RESNAME CHAIN RESNUM reason' for every member of a glycan left as ligands (default "
                         "%(default)s; removed when there is none)")
    args = ap.parse_args(argv)
    try:
        with open(args.f) as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        print("fuse_glycans.py: cannot read %s (%s)" % (args.f, getattr(e, "strerror", None) or e), file=sys.stderr)
        return 1
    cap, cap_from = (args.max_sugars, "--max-sugars") if args.max_sugars is not None else default_cap()
    lines = text.splitlines()
    recs = read_records(lines)
    entries = Entries()
    found, notes, records, stretched = find_glycans(lines, recs, entries, cap)
    for n in notes:
        print(n)
    exclude = read_exclusions(args.exclude_file)
    old = read_old_names(args.d) if os.path.isfile(args.d) else {}
    print("fuse_glycans: %d glycan(s) bonded to the protein; %s%s" % (
        len(found), "no cap on the number of sugars" if cap is None else "cap %d sugar(s) (%s)" % (cap, cap_from),
        "; excluded: " + ", ".join(sorted(exclude)) if exclude else ""))
    input_names = {r["res"] for r in recs}
    taken = None
    defs, excluded, by_sig, units, sites = [], [], {}, [], set()
    unfused, report, to_het = [], {"fused": [], "not_fused": []}, []
    for members, site, why, flip in found:
        got = None
        if why is None and site in sites:
            why = "%s already carries a fused glycan" % label(site)
        if why is None:
            try:
                got, why = build_unit(site, members, records, entries, flip, stretched)
            except Exception as ex:          # an entry this code did not foresee: one glycan, not the run
                why = "could not be built (%s: %s)" % (type(ex).__name__, " ".join(str(ex).split())[:120])
        if got is not None:
            definition, instance, rep, write = got
            sig = signature(definition)
            # The fallback removes fusions and adds none: a chemistry the first pass did
            # not fuse has no parameters (review R2: a CCD entry that answered only now
            # fused a glycan, and pdb2gmx stopped on its residue without an rtp entry).
            if args.exclude_file and sig not in old:
                got = None
                why = "the first pass did not fuse this chemistry, so parametrize_ncaa.py made no parameters for it"
        if got is not None:
            if sig not in by_sig:
                names_now = {d["resname"] for d in defs + excluded}
                name = old.get(sig)
                if name is None or name in input_names or name in names_now:
                    if taken is None:
                        taken = taken_names()
                    name = next_name(definition["family"], taken | input_names | set(old.values()) | names_now)
                if name is None:
                    why = "no free residue name left in %s1-%sZ" % (definition["family"], definition["family"])
                else:
                    definition["resname"] = name
                    if name in exclude:
                        definition["reason"] = exclude[name]
                        excluded.append(definition)
                    else:
                        defs.append(definition)
                    by_sig[sig] = definition
        if got is not None and why is None:
            definition = by_sig[sig]
            if "reason" in definition:
                why = "%s could not be parametrised: %s" % (definition["resname"], definition["reason"])
            else:
                definition["instances"].append(instance)
                sites.add(site)
                write.update(resname=definition["resname"],
                             elements={a["name"]: a["element"] for a in definition["atoms"]})
                units.append(write)
                report["fused"].append({"resname": definition["resname"], "site": rep["site"], "sugars": rep["sugars"],
                                        "link": rep["link"], "stereo": [s for s, _ in rep["stereo"]],
                                        "altloc": rep["alt"], "amide_flipped": rep["flip"]})
                print("fused %s: %s + %s (%s)" % (definition["resname"], rep["site"], ", ".join(rep["sugars"]),
                                                  rep["link"]))
                if rep["flip"]:
                    print("  %s: OD1 and ND2 swapped, as the deposit bonds the sugar to OD1: an N-glycan links "
                          "through the amide N, and X-ray density does not tell O from N" % rep["site"])
                for s, odd in rep["stereo"]:
                    print("  %sstereo %s" % ("WARNING " if odd else "", s))
                for x in rep["dropped"]:
                    print("  dropped the deposited leaving atom %s: the link takes its place" % x)
                continue
        report["not_fused"].append({"residues": [label(k) for k in members], "why": why})
        print("NOT fused %s: %s" % (", ".join(label(k) for k in members), why))
        unfused += ["%s %s %s %s" % (k[0], k[1].strip() or "-", k[2], " ".join(why.split())) for k in members]
        # ATOM records would keep the sugars in the protein, where pdb2gmx stops on them
        # without a word about the glycan; as HETATM renumber_pdb.py extracts them.
        atom = [r["i"] for k in members for r in records[k] if not r["het"]]
        if atom:
            to_het += atom
            print("  written as HETATM records, so that they are ligands: %s" % ", ".join(
                label(k) for k in members if any(not r["het"] for r in records[k])))
    if units or to_het:
        for n in write_fused(lines, units, records, args.o, to_het):
            print(n)
    else:
        shutil.copyfile(args.f, args.o)
    with open(args.d, "w") as f:
        json.dump({"residues": defs, "excluded": excluded, "max_sugars": cap, "report": report}, f, indent=1)
    if unfused:
        with open(args.unfused, "w") as f:
            f.write("\n".join(unfused) + "\n")
    elif os.path.isfile(args.unfused):
        os.remove(args.unfused)
    print("fuse_glycans: %d glycan(s) fused%s, %d left as ligands; wrote %s and %s" % (
        len(units), (" into " + ", ".join("%s (%d)" % (d["resname"], len(d["instances"])) for d in defs)) if defs else "",
        len(report["not_fused"]), args.o, args.d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
