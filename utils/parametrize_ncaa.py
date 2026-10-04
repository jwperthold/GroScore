#!/usr/bin/env python3
#
# parametrize_ncaa.py - Parametrize non-standard amino acids with OpenFF for GROMACS
#
# Generates custom force field entries (RTP, HDB, atom types, bonded parameters)
# that combine AMBER19SB backbone parameters with OpenFF Sage sidechain parameters.
# Creates a local force field directory with injected NCAA definitions.
#
# Requires: openbabel, rdkit, openff-toolkit, openff-interchange (groscore conda env)
#
# Usage: python parametrize_ncaa.py -f fixed.pdb [--gmx-ff amber19sb]
#
# FUSED GLYCANS. utils/fuse_glycans.py makes a glycan covalently bonded to ASN ND2,
# SER OG or THR OG1 one residue with that amino acid (NG<k>, SG<k>, TG<k>) and writes
# its chemistry to glycans.json in the structure directory: heavy atoms with element,
# formal charge and hydrogen count, bonds with orders, hydrogen names, all taken from
# the CCD entries of the amino acid and the sugars. A residue named there is built
# from that definition (parametrize_fused()), its parent is the definition's, and its
# name is never looked up in the CCD. One that fails goes to glycan_failed.gs
# ('RESNAME reason'), never to ncaa_failed.gs: the runners then fuse again without it,
# and its sugars become ligands that merge_ligand.py judges by region.
#

import os
import sys
import json
import argparse
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccd_cache  # noqa: E402  (standard library only)

parser = argparse.ArgumentParser(description="Parametrize NCAAs with OpenFF for GROMACS.")
parser.add_argument('-f', '--file', type=str, required=True, help="Input PDB file (fixed.pdb)")
parser.add_argument('--ff', type=str, default='openff-2.2.1.offxml', help="OpenFF force field")
parser.add_argument('--gmx-ff', type=str, default='amber19sb', help="GROMACS base force field name")
parser.add_argument('--ph', type=float, default=7.4, help="pH for protonation (default: 7.4)")
args = parser.parse_args()

# ---- Constants ----
STD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYX', 'GLN', 'GLU', 'GLY',
    'HID', 'HIE', 'HIP', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'ACE', 'NME', 'NHE',
    'ZN', 'CA', 'MG', 'CU', 'CU1', 'NA', 'CL',
}

# AMBER19SB backbone: atom_name -> (type, charge)
# N, H, C, O have fixed charges across all residues
# CA, HA charges vary per residue — we'll use OpenFF charges for those
BACKBONE_FIXED_CHARGES = {'N': -0.4157, 'H': 0.2719, 'C': 0.5973, 'O': -0.5679}
BACKBONE_TYPES = {'N': 'N', 'H': 'H', 'CA': 'XC', 'HA': 'H1', 'C': 'C', 'O': 'O'}
BACKBONE_NAMES = set(BACKBONE_TYPES.keys())
CB_TYPE = 'CT'
CB_H_TYPE = 'HC'
# amber19sb types a hydrogen on an aliphatic carbon by that carbon's electron-
# withdrawing neighbours (N, O, S, halogens): HC with none, H1 with one (the CB
# hydrogens of SER and THR next to OG, of CYS next to SG; MET HG next to SD, PRO and
# ARG HD next to N), H2 with two, H3 with three. The CB hydrogens of an NCAA are typed
# the same way: a glycosylated Ser or Thr (SG<k>, TG<k>), phospho-Ser and phospho-Thr
# get H1 (critic F12c), and so does a Cys-derived NCAA such as the CME of 1S9D, which
# kept HC while only an oxygen counted (ncaa review F3). HC, H1, H2 and H3 differ only
# in sigma, 0.2650, 0.2471, 0.2293 and 0.2115 nm (epsilon 0.0657 kJ/mol all).
# An amine N is the exception: amber19sb types the hydrogens next to NZ HP in both LYS
# (NH3+) and LYN (NH2), and the H1 the count gave was GAFF's convention, not the force
# field's (verifier V4). A CB bonded to an sp3 amine N (DPP, 2,3-diaminopropionic
# acid) gets HP; an amide or guanidinium N (PRO, ARG HD) stays H1.
CB_H_TYPES = ('HC', 'H1', 'H2', 'H3')           # by the number of such neighbours
CB_AMINE_H_TYPE = 'HP'
# amber19sb has no XC-CT-HP angle (an HP on the CB next to the CMAP CA type XC never
# occurs in its residues), so grompp would stop on such a CB. XC takes CT's bonded
# terms, and this is its copy of CT-CT-HP, written to ncaa_bonded.itp when HP is used.
HP_CB_ANGLE = {('HP', 'CT', 'XC'): (1, 109.5, 418.4)}
ELECTRON_WITHDRAWING = {7, 8, 9, 16, 17, 35, 53}  # N, O, F, S, Cl, Br, I


def is_amine_n(atom):
    """An sp3 amine nitrogen of an RDKit molecule, charged or not: no multiple bond,
    not aromatic, and no neighbour with a multiple bond (that is an amide, an aniline
    or a sulfonamide N)."""
    from rdkit import Chem
    single = Chem.BondType.SINGLE
    return (atom.GetAtomicNum() == 7 and not atom.GetIsAromatic()
            and all(b.GetBondType() == single for b in atom.GetBonds())
            and not any(n.GetIsAromatic() or any(b.GetBondType() != single for b in n.GetBonds())
                        for n in atom.GetNeighbors()))

# ---- Fused glycan residues (utils/fuse_glycans.py) ----
# Their names are authoritative and never read as CCD codes: nearly every 3-character
# code is a CCD entry (11 of 220 probed were free), NG1 among them (GalNAc
# 1-phosphate, parent A2G). Before this the script took that entry's parent and, after
# OpenBabel built the 22-atom fused Asn-GlcNAc of 1MAH at charge +1, its 19-atom RCSB
# template, which failed; NG1 then went to ncaa_failed.gs, the parent-replacement path
# (design run 1MAH_orig).
GLYCANS_JSON = 'glycans.json'
GLYCAN_FAILED = 'glycan_failed.gs'
NCAA_FAILED = 'ncaa_failed.gs'
GLYCAN_DEFS = {}      # resname -> definition from glycans.json (main section)
CHARGE_NOTES = {}     # resname -> how a fused residue's charges were made, for ncaa.rtp
# Hydrogen-name bases fuse_glycans.py takes for a name that would exceed 4 characters
# (the SIA C11 methyl: H11B1) or clash; a parent that gains hydrogens at the pH takes an
# unused one (fused_h_names()).
HX_POOL = ['HX' + c for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']
# The whole AM1-BCC time of a fused residue is bounded, first attempt included
# (fused_interchange()).
GLYCAN_WALL_ENV = 'GROSCORE_GLYCAN_AM1BCC_WALL_S'


def load_glycan_defs(path):
    """{resname: definition} from fuse_glycans.py's glycans.json; {} without the file."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {r['resname']: r for r in json.load(f)['residues']}


def ccd_parent(resname, glycans=None):
    """(parent residue, note) of an NCAA from its CCD entry, read from the local CCD
    cache first and files.rcsb.org second (utils/ccd_cache.py): (parent, None) when the
    entry names one; ('ALA', why) when it names none, as for D-amino acids (DAL, DPN)
    and beta-amino acids (BAL, ACB), which keep ALA's CMAP as in v1.4 (review V2);
    (None, why not) when the entry cannot be read. A fused glycan residue, a name in
    `glycans` (the glycans.json definitions), takes the definition's parent, and its
    CCD entry is never read: the real NG1 would give A2G."""
    if glycans and resname in glycans:
        parent = str(glycans[resname].get('parent') or '').strip().upper()
        if not parent:
            return None, f"glycans.json names no parent residue for {resname}"
        return parent, None
    text, src = ccd_cache.fetch(resname)
    if text is None:
        return None, src
    parent = ccd_cache.parse_cif(text).get('_chem_comp.mon_nstd_parent_comp_id', ['?'])[0]
    parent = parent.split(',')[0].strip().upper()
    if parent in ('', '?', '.'):
        return 'ALA', f"the CCD entry {resname} names no parent residue, so it gets ALA's CMAP"
    return parent, None


def ccd_template(ccd_id):
    """(molecule, source): the CCD entry ccd_id as an RDKit molecule with its hydrogens,
    bond orders, formal charges and ideal coordinates (the model ones where the entry
    has none), its stereo from those coordinates, read from the local CCD cache first
    and files.rcsb.org second (utils/ccd_cache.py). Strategy 2's template. It used to be
    files.rcsb.org's <ID>_ideal.sdf and nothing else, so on a node without network
    every NCAA Strategy 2 builds went to its parent residue although the cache holds
    its entry (2MTA TRQ, 2BTF HIC; ncaa review P2). For 56 NCAAs (HIC, MSE, TRQ, SEP,
    TPO, PTR, D- and beta-amino acids among them) this molecule is atom for atom the
    one that SDF gave: elements, charges, hydrogens, bonds, coordinates and stereo.
    Raises RuntimeError with the reason when the entry cannot be read or built."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    text, src = ccd_cache.fetch(ccd_id)
    if text is None:
        raise RuntimeError(src)
    cif, comp = ccd_cache.parse_cif(text), ccd_cache.component(text)
    ids = cif.get('_chem_comp_atom.atom_id', [])
    xyz = {}
    for key in ('_chem_comp_atom.pdbx_model_Cartn_%s_ideal', '_chem_comp_atom.model_Cartn_%s'):
        cols = [cif.get(key % c, []) for c in 'xyz']
        if all(len(c) == len(ids) for c in cols):
            for n, x, y, z in zip(ids, *cols):
                if n not in xyz:
                    try:
                        xyz[n] = (float(x), float(y), float(z))
                    except ValueError:
                        pass
    order = {'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE, 'TRIP': Chem.BondType.TRIPLE}
    rw, idx = Chem.RWMol(), {}
    for a in comp['atoms']:
        if a['name'] not in xyz:
            raise RuntimeError(f"the CCD entry {ccd_id} has no coordinates for {a['name']}")
        atom = Chem.Atom(a['element'].capitalize())
        atom.SetFormalCharge(a['charge'])
        idx[a['name']] = rw.AddAtom(atom)
    for n1, n2, o, arom in comp['bonds']:
        if o not in order:
            raise RuntimeError(f"bond order {o} between {n1} and {n2} in the CCD entry {ccd_id}")
        rw.AddBond(idx[n1], idx[n2], order[o])
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for a in comp['atoms']:
        conf.SetAtomPosition(idx[a['name']], Point3D(*xyz[a['name']]))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol, src


# ---- PDB parsing ----

def parse_pdb(path):
    """Parse PDB file, return list of atom dicts."""
    atoms = []
    with open(path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')) or len(line) < 54:
                continue
            atoms.append({
                'name': line[12:16].strip(),
                'resname': line[17:20].strip(),
                'chain': line[21],
                'resnum': int(line[22:26]),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
            })
    return atoms


def find_ncaas(atoms):
    """Find non-standard amino acid residues (have backbone but not in standard set).
    Returns dict: resname -> [(chain, resnum)]."""
    residue_atoms = {}
    residue_names = {}
    for a in atoms:
        key = (a['chain'], a['resnum'])
        residue_atoms.setdefault(key, set()).add(a['name'])
        residue_names[key] = a['resname']

    ncaas = {}
    for key, atom_names in residue_atoms.items():
        rn = residue_names[key]
        if rn in STD_RESIDUES:
            continue
        if {'N', 'CA', 'C', 'O'} <= atom_names:
            ncaas.setdefault(rn, []).append(key)
    return ncaas


# ---- Capped tripeptide building ----

def build_capped_pdb(atoms, chain, resnum):
    """Build ACE-NCAA-NME capped tripeptide from PDB coordinates.

    Returns: (pdb_text, ncaa_heavy_coords, ncaa_atom_names)
    ncaa_heavy_coords: dict atom_name -> np.array([x,y,z]) in Angstrom
    """
    ncaa_atoms = [a for a in atoms if a['chain'] == chain and a['resnum'] == resnum]
    chain_resnums = sorted(set(a['resnum'] for a in atoms if a['chain'] == chain))
    idx = chain_resnums.index(resnum)
    prev_rn = chain_resnums[idx - 1] if idx > 0 else None
    next_rn = chain_resnums[idx + 1] if idx < len(chain_resnums) - 1 else None

    prev_atoms = [a for a in atoms if a['chain'] == chain and a['resnum'] == prev_rn] if prev_rn else []
    next_atoms = [a for a in atoms if a['chain'] == chain and a['resnum'] == next_rn] if next_rn else []

    prev_C = next((a for a in prev_atoms if a['name'] == 'C'), None)
    prev_O = next((a for a in prev_atoms if a['name'] == 'O'), None)
    next_N = next((a for a in next_atoms if a['name'] == 'N'), None)

    ncaa_N = next((a for a in ncaa_atoms if a['name'] == 'N'), None)
    ncaa_C = next((a for a in ncaa_atoms if a['name'] == 'C'), None)

    lines = []
    serial = 1

    def coord(a):
        return np.array([a['x'], a['y'], a['z']])

    def pdb_line(s, name, resn, ch, rn, xyz, elem):
        nonlocal serial
        name_fmt = f" {name:<3s}" if len(name) < 4 else f"{name:<4s}"
        # PDB element column must be at cols 77-78 (10 spaces of padding after tempFactor)
        line = (f"ATOM  {serial:5d} {name_fmt} {resn:>3s} {ch}{rn:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2s}")
        serial += 1
        return line

    # ACE cap: CH3-C(=O) from previous residue backbone
    if prev_C and prev_O and ncaa_N:
        c_xyz = coord(prev_C)
        o_xyz = coord(prev_O)
        n_xyz = coord(ncaa_N)
        cn_dir = c_xyz - n_xyz
        cn_dir /= np.linalg.norm(cn_dir)
        ch3_xyz = c_xyz + cn_dir * 1.52
        for name, xyz, elem in [('CH3', ch3_xyz, 'C'), ('C', c_xyz, 'C'), ('O', o_xyz, 'O')]:
            lines.append(pdb_line(serial, name, 'ACE', chain, resnum - 1, xyz, elem))

    # NCAA atoms from PDB (heavy atoms tracked separately for matching)
    ncaa_heavy_coords = {}
    ncaa_atom_names = []
    for a in ncaa_atoms:
        elem = a['name'][0]
        lines.append(pdb_line(serial, a['name'], a['resname'], chain, resnum,
                              coord(a), elem))
        # Only track heavy atoms for coordinate matching (skip H)
        if not a['name'].strip().startswith('H'):
            ncaa_heavy_coords[a['name']] = coord(a)
            ncaa_atom_names.append(a['name'])

    # NME cap: N(H)-CH3 from next residue backbone
    if ncaa_C and next_N:
        c_xyz = coord(ncaa_C)
        n_xyz = coord(next_N)
        nc_dir = n_xyz - c_xyz
        nc_dir /= np.linalg.norm(nc_dir)
        ch3_xyz = n_xyz + nc_dir * 1.47
        for name, xyz, elem in [('N', n_xyz, 'N'), ('CH3', ch3_xyz, 'C')]:
            lines.append(pdb_line(serial, name, 'NME', chain, resnum + 1, xyz, elem))

    lines.append("END")
    return '\n'.join(lines) + '\n', ncaa_heavy_coords, ncaa_atom_names


# ---- Fused glycan residues: the chemistry from glycans.json ----

def fused_h_names(d, h_parent):
    """({hydrogen index: name}, notes) for the hydrogens of a fused residue as they
    are after protonation at the pH; h_parent = {hydrogen index: parent heavy-atom
    name}, residue atoms only.

    The definition's names (hnames) come from the CCD hydrogen counts, but OpenBabel's
    pH model and the acid rules remove the proton of every carboxylic acid, sulfate and
    phosphate: SIA's O1B loses its H. Naming those CCD names regardless stopped every
    sialylated glycan after its full AM1-BCC run, as 'unassigned HO6A' on an acid
    variant of the 1MAH NG1 after 133 s (critic F5). So the hydrogens that exist are
    named: a parent takes the definition's names in order for as many as it keeps, and
    the trailing ones are dropped (and logged); a parent that gained hydrogens takes an
    unused base from HX0..HXZ for all of them, since pdb2gmx names a group from one
    base (base1, base2, ...). The pool also serves a sugar atom the definition gives
    no hydrogen and no names: the generic NCAA rule would call the hydrogen of a
    sialic acid's N5A H5A, the name of C5A's. Parents the definition gives no names
    that keep their count, the amino acid's, are named as for any NCAA
    (assign_h_names()). The names must be unique, distinct from the heavy-atom names
    and at most 4 characters."""
    groups = {}
    for h in sorted(h_parent):
        groups.setdefault(h_parent[h], []).append(h)
    hnames = d.get('hnames') or {}
    defined_h = {a['name']: a['h'] for a in d['atoms']}
    heavy = set(defined_h)
    taken = heavy | {n for v in hnames.values() for n in v}
    pool = [b for b in HX_POOL if not any(n == b or (n[:-1] == b and n[-1].isdigit()) for n in taken)]
    names, notes = {}, []
    for parent, hs in groups.items():
        n = len(hs)
        listed = hnames.get(parent)
        if listed is None and n <= defined_h.get(parent, 0):
            got = [assign_h_names({h: parent for h in hs})[h] for h in hs]
        elif listed is not None and n <= len(listed):
            got = list(listed[:n])
            if n < len(listed):
                notes.append(f"{parent} keeps {n} of {len(listed)} hydrogens at the pH: {' '.join(listed[n:])} dropped")
        else:
            if not pool:
                raise RuntimeError(f"no free hydrogen name left for the {n} hydrogens of {parent}")
            base = pool.pop(0)
            got = [base] if n == 1 else [f"{base}{k}" for k in range(1, n + 1)]
            had = len(listed) if listed is not None else defined_h.get(parent, 0)
            notes.append(f"{parent} gained hydrogens at the pH ({had} -> {n}): named {' '.join(got)}")
        names.update(zip(hs, got))
    for parent, want in hnames.items():
        if want and parent not in groups:
            notes.append(f"{parent} keeps 0 of {len(want)} hydrogens at the pH: {' '.join(want)} dropped")
    values = list(names.values())
    dup = sorted({n for n in values if values.count(n) > 1})
    clash = sorted(set(values) & heavy)
    long_names = sorted(n for n in values if len(n) > 4)
    if dup or clash or long_names:
        raise RuntimeError(f"hydrogen names: duplicates {dup}, equal to a heavy atom {clash}, "
                           f"longer than 4 characters {long_names}")
    return names, notes


def fused_hdb_problems(resname, mol, names, n_res, h_parent, h_names):
    """Why the hdb entry generate_hdb() would write for this molecule could not make
    pdb2gmx rebuild exactly these hydrogens ([] when it can). Run on the protonated
    RDKit molecule before any sqm time is spent (critic F5): pdb2gmx -ignh drops the
    hydrogens and adds them from the hdb, and under -missing an hdb line it cannot
    follow leaves the residue short of atoms without a word. A line exists for every
    parent with 1 to 3 hydrogens, with 2 reference atoms (3 for a single tetrahedral H),
    and its names, base1..baseN for a group, are the rtp names."""
    adj = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    zs = sorted({a.GetAtomicNum() for a in mol.GetAtoms()})
    top = {'atomtypes': [{'name': f'Z{z}', 'at_num': z} for z in zs],
           'atoms': [{'type': f'Z{a.GetAtomicNum()}'} for a in mol.GetAtoms()]}
    entry = [line for line in generate_hdb(resname, names[:n_res], h_parent, h_names,
                                           {i: names[i] for i in range(n_res)}, adj, top).split('\n')
             if line.strip()]
    want = {}
    for h, p in h_parent.items():
        want.setdefault(p, []).append(h_names[h])
    probs, seen = [], set()
    head = entry[0].split()
    if len(head) != 2 or int(head[1]) != len(entry) - 1:
        probs.append(f"header {entry[0]!r} for {len(entry) - 1} lines")
    for line in entry[1:]:
        f = line.split('\t')
        n, kind, base, parent, refs = int(f[0]), int(f[1]), f[2], f[3], [r for r in f[4:] if r]
        seen.add(parent)
        made = [base] if n == 1 else [f"{base}{k}" for k in range(1, n + 1)]
        if sorted(made) != sorted(want.get(parent, [])):
            probs.append(f"{parent}: the hdb line makes {' '.join(made)}, the rtp has "
                         f"{' '.join(sorted(want.get(parent, [])))}")
        if len(refs) != (3 if kind == 5 else 2) or len(set(refs)) != len(refs) or parent in refs:
            probs.append(f"{parent}: reference atoms {' '.join(refs) or '-'} for an hdb line of type {kind}")
    for parent in sorted(set(want) - seen):
        probs.append(f"{parent}: {len(want[parent])} hydrogens and no hdb line")
    return probs


def fused_interchange(ff, offmol, clock=None):
    """parametrize_ligand.make_interchange() (valence typing first, then AM1-BCC on the
    deposited conformer and seeded retries) with the AM1-BCC time bounded as a whole,
    first attempt included.

    The ligand path never cuts its first attempt. A fused residue's AM1-BCC grows
    roughly with the cube of its size: 54-102 s with one sugar, 161 s with two and
    532 s with three, an estimated 25 min with five and 2 h for a 9-sugar Fc glycan.
    Glycans have no size cap by default, so neither has this time: the ligand path's
    AM1BCC_RETRY_WALL_S of 3600 s would send every large glycan back to the ligands,
    where it stops the structure, after an hour of sqm. Unlimited is the default; the
    job's own time limit is the bound. GROSCORE_GLYCAN_AM1BCC_WALL_S (s) sets a wall
    for fused residues, after which the glycan goes back to the ligands, so that a run
    can bound the setup or force the fallback (critic F4). Each attempt gets what the
    attempts before it left of the wall: the retries used to get a wall of their own
    after the first attempt's, nearly twice the value in all (runner review R3).
    clock: time.monotonic, or a fake one for a test."""
    import time
    import parametrize_ligand as PL
    clock = clock or time.monotonic
    wall = float('inf')
    if os.environ.get(GLYCAN_WALL_ENV):
        try:
            wall = float(os.environ[GLYCAN_WALL_ENV])
            print(f"  {GLYCAN_WALL_ENV}={wall:g} s bounds the AM1-BCC of the fused residue")
        except ValueError:
            print(f"  WARNING: {GLYCAN_WALL_ENV}={os.environ[GLYCAN_WALL_ENV]!r} is not a number of "
                  f"seconds; the AM1-BCC of the fused residue stays unlimited")
    attempt, retry_wall = PL.am1bcc_attempt, PL.AM1BCC_RETRY_WALL_S
    start = []

    def bounded(mol, conformer, timeout=None):
        if not start:
            start.append(clock())
        left = wall - (clock() - start[0])
        if left <= 0:
            raise TimeoutError("AM1-BCC did not finish within %.0f s" % wall)
        left = left if timeout is None else min(timeout, left)
        # am1bcc_attempt() waits without limit on None; an infinite timeout would
        # overflow the poll.
        return attempt(mol, conformer, None if left == float('inf') else left)

    PL.am1bcc_attempt, PL.AM1BCC_RETRY_WALL_S = bounded, wall
    try:
        return PL.make_interchange(ff, offmol)
    finally:
        PL.am1bcc_attempt, PL.AM1BCC_RETRY_WALL_S = attempt, retry_wall


def parametrize_fused(pdb_text, ncaa_heavy_coords, ncaa_resname, ff_name, ph):
    """parametrize_capped() for a fused glycan residue: Ace-X-NMe from its glycans.json
    definition on the canonical instance's deposited coordinates.

    Heavy atoms, elements, formal charges, hydrogen counts and bond orders are the
    definition's (the CCD entries of the amino acid and its sugars, as
    fuse_glycans.py joined them); the caps are build_capped_pdb()'s. Hydrogens and
    charges follow the ligand path: OpenBabel's pH model on those hydrogen counts
    (parametrize_ligand.ob_protonate), the pH 7.4 acid rules (ionize_acids: a sialic
    acid carboxylate becomes COO-), then OpenFF valence typing and AM1-BCC on the
    deposited conformer with seeded retries (fused_interchange()). On 1MAH the
    deposited conformer's AM1-BCC took 102 s against 126 s for OpenFF's own conformer,
    and the charges differ by up to 0.057 e per atom (0.012 e on average). No OpenBabel
    bond perception, no CCD template and no MCS name recovery: each reads the
    residue name as a CCD code.

    Returns parametrize_capped()'s tuple; the names by index cover the residue's
    hydrogens too (fused_h_names()), the cap atoms stay None."""
    import time
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Geometry import Point3D
    from openff.toolkit import Molecule, ForceField
    import parametrize_ligand as PL

    d = GLYCAN_DEFS[ncaa_resname]
    defined = [a['name'] for a in d['atoms']]
    missing = [n for n in defined if n not in ncaa_heavy_coords]
    extra = sorted(set(ncaa_heavy_coords) - set(defined))
    if missing or extra:
        raise RuntimeError(f"the deposited residue is not its glycans.json definition "
                           f"(missing {','.join(missing) or '-'}, extra {','.join(extra) or '-'})")
    if not {'N', 'CA', 'C'} <= set(defined):
        raise RuntimeError("the glycans.json definition has no backbone N, CA and C")
    caps = {}
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM') and line[17:20].strip() in ('ACE', 'NME'):
            caps[(line[17:20].strip(), line[12:16].strip())] = tuple(float(line[c:c + 8]) for c in (30, 38, 46))
    cap_atoms = [('ACE', 'CH3', 'C', 3), ('ACE', 'C', 'C', 0), ('ACE', 'O', 'O', 0),
                 ('NME', 'N', 'N', 1), ('NME', 'CH3', 'C', 3)]
    if any((r, n) not in caps for r, n, _, _ in cap_atoms):
        raise RuntimeError("no peptide neighbour on both sides to cap the residue with ACE and NME")

    rw = Chem.RWMol()
    idx, names, xyz = {}, [], []

    def add(name, element, charge, h, pos):
        a = Chem.Atom(str(element).capitalize())
        a.SetFormalCharge(int(charge))
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(int(h))
        idx[name] = rw.AddAtom(a)
        names.append(name)
        xyz.append(pos)

    # The residue's atoms first, in definition order, so that index i < n_res is
    # definition atom i; the caps after them.
    for a in d['atoms']:
        add(a['name'], a['element'], a.get('charge', 0), a['h'], tuple(ncaa_heavy_coords[a['name']]))
    for r, n, el, h in cap_atoms:
        add(f'{r}:{n}', el, 0, h, caps[(r, n)])
    n_res = len(defined)
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE,
             'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE, 'TRIP': Chem.BondType.TRIPLE}
    for a1, a2, o in d['bonds']:
        rw.AddBond(idx[a1], idx[a2], order[o])
    for a1, a2, o in (('ACE:CH3', 'ACE:C', 1), ('ACE:C', 'ACE:O', 2), ('ACE:C', 'N', 1),
                      ('C', 'NME:N', 1), ('NME:N', 'NME:CH3', 1)):
        rw.AddBond(idx[a1], idx[a2], order[o])
    m = rw.GetMol()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*p))
    m.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(m)
    rad = [names[a.GetIdx()] for a in m.GetAtoms() if a.GetNumRadicalElectrons()]
    if rad:
        raise RuntimeError("radical atoms in the definition: " + ','.join(rad))

    mol = PL.ob_protonate(m, ph)
    if mol is None:
        raise RuntimeError("RDKit could not read OpenBabel's protonated molecule")
    if [a.GetAtomicNum() for a in mol.GetAtoms()][:m.GetNumAtoms()] != [a.GetAtomicNum() for a in m.GetAtoms()]:
        raise RuntimeError("OpenBabel's protonated molecule does not keep the heavy atoms in order")
    print(f"  fused chemistry: {rdMolDescriptors.CalcMolFormula(Chem.AddHs(m))} (CCD hydrogens), "
          f"{rdMolDescriptors.CalcMolFormula(mol)} (OpenBabel, pH {ph:g})")
    if PL.ACID_PH_WINDOW[0] <= ph <= PL.ACID_PH_WINDOW[1]:
        mol, notes = PL.ionize_acids(mol, dict(enumerate(names)))
        for n in notes:
            print(f"  acid rule: {n}")
    else:
        print(f"  Warning: pH {ph} is outside {PL.ACID_PH_WINDOW}: the pH 7.4 acid rules are not applied")
    formal_charge = Chem.GetFormalCharge(mol)

    # Hydrogen names and the hdb entry, before any sqm time is spent (critic F5)
    h_parent = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            nb = [x.GetIdx() for x in a.GetNeighbors()]
            if len(nb) == 1 and nb[0] < n_res:
                h_parent[a.GetIdx()] = names[nb[0]]
    h_names, notes = fused_h_names(d, h_parent)
    for n in notes:
        print(f"  hydrogen names: {n}")
    probs = fused_hdb_problems(ncaa_resname, mol, names, n_res, h_parent, h_names)
    if probs:
        raise RuntimeError("no hdb entry can rebuild its hydrogens: " + "; ".join(probs))

    mol, _ = PL.clean_copy(mol)
    if mol is None:
        raise RuntimeError("the SDF round trip of the protonated molecule failed")
    Chem.AssignStereochemistryFrom3D(mol)
    print(f"  capped molecule: {mol.GetNumAtoms()} atoms, formal charge {formal_charge:+d}, "
          f"SMILES {Chem.MolToSmiles(Chem.RemoveHs(mol))}")
    offmol = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    t0 = time.monotonic()
    interchange, note = fused_interchange(ForceField(ff_name), offmol)
    print(f"  {note}; OpenFF parameters and charges took {time.monotonic() - t0:.0f} s")
    CHARGE_NOTES[ncaa_resname] = note
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "ncaa")
        interchange.to_gromacs(prefix)
        with open(f"{prefix}.top") as f:
            top_content = f.read()
        with open(f"{prefix}.gro") as f:
            gro_content = f.read()
    # Names go by index, so the topology must keep the molecule's atom order.
    top = parse_top(top_content)
    at_num = {at['name']: at['at_num'] for at in top['atomtypes']}
    if [at_num.get(a['type']) for a in top['atoms']] != [a.GetAtomicNum() for a in mol.GetAtoms()]:
        raise RuntimeError("the OpenFF topology does not keep the molecule's atom order")
    names_by_idx = [None] * mol.GetNumAtoms()
    for i in range(n_res):
        names_by_idx[i] = names[i]
    for h, name in h_names.items():
        names_by_idx[h] = name
    return top_content, gro_content, mol, formal_charge, names_by_idx


def fused_rtp_comment(resname, formal_charge):
    """The ncaa.rtp comment of a fused residue: what it is made of, the CCD entries
    with source and modification date as glycans.json records them, and how its
    charges were made."""
    d = GLYCAN_DEFS[resname]
    sugars = d.get('sugars') or []
    lines = [f"; {resname}: {d.get('parent', '?')}"
             + ''.join(f" + {s.get('ccd', '?')} (tag {s.get('tag', '?')})" for s in sugars)
             + f", fused by utils/fuse_glycans.py ({GLYCANS_JSON}), link {'-'.join(d.get('link') or ['?'])}"]
    for s in sugars:
        lines.append(f";   CCD {s.get('ccd', '?')}: {s.get('source') or 'source not recorded'}, "
                     f"modified {s.get('modified') or '?'}")
    lines.append(f";   charges: {CHARGE_NOTES.get(resname, '?')}; residue charge {formal_charge:+d}, "
                 f"the residual on CA")
    return '\n'.join(lines) + '\n'


# ---- OpenBabel / RDKit / OpenFF parametrization ----

def parametrize_capped(pdb_text, ncaa_heavy_coords, ncaa_resname, ff_name, ph):
    """Parametrize capped tripeptide with OpenFF. Returns parsed topology data.

    Strategy 1: OpenBabel bond perception (works for simple NCAAs)
    Strategy 2: the CCD entry as template (ccd_template()) + manual ACE/NME capping
    A fused glycan residue goes to parametrize_fused() instead.
    """
    if ncaa_resname in GLYCAN_DEFS:
        return parametrize_fused(pdb_text, ncaa_heavy_coords, ncaa_resname, ff_name, ph)
    from openbabel import openbabel as ob
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from openff.toolkit import Molecule, ForceField, Topology
    from openff.interchange import Interchange
    from openff.units import unit
    import parametrize_ligand as PL

    mol_rdkit = None
    formal_charge = 0

    # Strategy 1: OpenBabel bond perception on full capped tripeptide
    print("  Strategy 1: OpenBabel bond perception...")
    ob.obErrorLog.SetOutputLevel(ob.obWarning)
    ob.obErrorLog.StartLogging()

    conv = ob.OBConversion()
    conv.SetInFormat("pdb")
    mol_ob = ob.OBMol()
    conv.ReadString(mol_ob, pdb_text)
    mol_ob.PerceiveBondOrders()

    ob_warnings = ob.obErrorLog.GetMessagesOfLevel(ob.obWarning)
    kekulize_ok = not any("kekulize" in w.lower() for w in ob_warnings)
    ob.obErrorLog.StopLogging()

    if kekulize_ok:
        mol_ob.AddHydrogens(False, True, ph)
        formal_charge = mol_ob.GetTotalCharge()
        print(f"  OpenBabel: {mol_ob.NumAtoms()} atoms, charge={formal_charge}")

        conv.SetOutFormat("sdf")
        sdf_block = conv.WriteString(mol_ob)
        mol_rdkit = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=True)

    # Strategy 2: the CCD entry as template for the NCAA + manual ACE/NME capping
    if mol_rdkit is None:
        print("  Strategy 2: CCD template + manual capping...")
        try:
            template_h, template_src = ccd_template(ncaa_resname)
            print(f"  template from the CCD entry {ncaa_resname} ({PL.describe_source(template_src)})")
            template = Chem.RemoveHs(template_h) if template_h else None

            if template is None:
                raise RuntimeError(f"CCD template for {ncaa_resname} failed")

            # Remove OXT from template (not present mid-chain)
            rwt = Chem.RWMol(template)
            # Find OXT: single-bonded O on a C that also has a double-bonded O
            oxt_indices = []
            for atom in rwt.GetAtoms():
                if atom.GetSymbol() != 'O':
                    continue
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) != 1 or neighbors[0].GetSymbol() != 'C':
                    continue
                c_atom = neighbors[0]
                bond = rwt.GetBondBetweenAtoms(atom.GetIdx(), c_atom.GetIdx())
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    continue
                # Check if the C has another O with double bond
                for nb in c_atom.GetNeighbors():
                    if nb.GetIdx() == atom.GetIdx():
                        continue
                    if nb.GetSymbol() == 'O':
                        b2 = rwt.GetBondBetweenAtoms(nb.GetIdx(), c_atom.GetIdx())
                        if b2.GetBondType() == Chem.BondType.DOUBLE:
                            oxt_indices.append(atom.GetIdx())
                            break
            for idx in sorted(oxt_indices, reverse=True):
                rwt.RemoveAtom(idx)
            template_clean = rwt.GetMol()

            print(f"  Template: {template_clean.GetNumAtoms()} heavy atoms "
                  f"(PDB has {len(ncaa_heavy_coords)})")

            # Read NCAA atoms from PDB
            ncaa_pdb_lines = []
            for line in pdb_text.split('\n'):
                if line.startswith('ATOM') and len(line) > 20:
                    rn = line[17:20].strip()
                    if rn == ncaa_resname:
                        ncaa_pdb_lines.append(line)
            ncaa_pdb_lines.append("END")
            ncaa_pdb_mol = Chem.MolFromPDBBlock('\n'.join(ncaa_pdb_lines) + '\n',
                                                 removeHs=True, sanitize=False)

            if ncaa_pdb_mol is None:
                raise RuntimeError("RDKit could not parse NCAA PDB block")

            # Assign bond orders from template
            try:
                mol_assigned = AllChem.AssignBondOrdersFromTemplate(template_clean, ncaa_pdb_mol)
                print(f"  Bond orders assigned from the CCD template")
            except Exception as e:
                print(f"  Template matching failed ({e})")
                # Fallback: use template directly with ideal coords
                # Assign PDB atom names from the CCD CIF (the template carries none)
                mol_assigned = template_clean
                try:
                    # The CCD entry from the local cache first (utils/ccd_cache.py),
                    # files.rcsb.org second, as for the ligands.
                    cif_data, cif_src = ccd_cache.fetch(ncaa_resname)
                    if cif_data is None:
                        raise RuntimeError(cif_src)
                    ccd_names = []
                    in_atom_block = False
                    for cif_line in cif_data.split('\n'):
                        if '_chem_comp_atom.atom_id' in cif_line:
                            in_atom_block = True
                            continue
                        if in_atom_block and cif_line.startswith('_chem_comp'):
                            continue
                        if in_atom_block and (cif_line.startswith('#') or cif_line.startswith('_') or cif_line.startswith('loop')):
                            in_atom_block = False
                            continue
                        if in_atom_block and cif_line.strip():
                            parts = cif_line.split()
                            if len(parts) >= 2 and parts[0] == ncaa_resname:
                                ccd_names.append(parts[1])
                    # Filter to heavy atoms only (matching template_clean which has removeHs)
                    heavy_names = [n for n in ccd_names if not n.startswith('H') and n != 'OXT']
                    if len(heavy_names) == mol_assigned.GetNumAtoms():
                        for i, name in enumerate(heavy_names):
                            info = Chem.AtomPDBResidueInfo()
                            info.SetName(f" {name:<3s}" if len(name) < 4 else name)
                            info.SetResidueName(ncaa_resname)
                            info.SetResidueNumber(1)
                            info.SetIsHeteroAtom(False)
                            mol_assigned.GetAtomWithIdx(i).SetPDBResidueInfo(info)
                        print(f"  Assigned {len(heavy_names)} atom names from CCD CIF")
                    else:
                        print(f"  Warning: CCD has {len(heavy_names)} heavy atoms vs {mol_assigned.GetNumAtoms()} in template")
                except Exception as cif_err:
                    print(f"  Warning: could not get CCD atom names ({cif_err})")
                print("  Using template with ideal coordinates")

            # Build capped molecule: ACE-NCAA-NME
            rwmol = Chem.RWMol(mol_assigned)

            # Find backbone N and C by PDB info or by topology
            n_idx = c_idx = ca_idx = None
            for atom in rwmol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info:
                    name = info.GetName().strip()
                    if name == 'N':
                        n_idx = atom.GetIdx()
                    elif name == 'C':
                        c_idx = atom.GetIdx()
                    elif name == 'CA':
                        ca_idx = atom.GetIdx()

            # Fallback: find by topology if no PDB info
            if n_idx is None or c_idx is None:
                for atom in rwmol.GetAtoms():
                    if atom.GetSymbol() == 'N' and not atom.IsInRing():
                        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                        if 'C' in neighbors:
                            n_idx = atom.GetIdx()
                    if atom.GetSymbol() == 'C' and not atom.IsInRing():
                        o_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'O')
                        if o_count >= 1:
                            c_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                            if c_neighbors:
                                c_idx = atom.GetIdx()

            if n_idx is None or c_idx is None:
                raise RuntimeError(f"backbone N={n_idx} C={c_idx} not found")

            # Add ACE cap: CH3-C(=O)-N
            ace_ch3 = rwmol.AddAtom(Chem.Atom(6))
            ace_c = rwmol.AddAtom(Chem.Atom(6))
            ace_o = rwmol.AddAtom(Chem.Atom(8))
            rwmol.AddBond(ace_ch3, ace_c, Chem.BondType.SINGLE)
            rwmol.AddBond(ace_c, ace_o, Chem.BondType.DOUBLE)
            rwmol.AddBond(ace_c, n_idx, Chem.BondType.SINGLE)

            # Add NME cap: C-N(H)-CH3
            nme_n = rwmol.AddAtom(Chem.Atom(7))
            nme_ch3 = rwmol.AddAtom(Chem.Atom(6))
            rwmol.AddBond(c_idx, nme_n, Chem.BondType.SINGLE)
            rwmol.AddBond(nme_n, nme_ch3, Chem.BondType.SINGLE)

            # Set 3D coordinates
            conf = rwmol.GetConformer()

            # Update NCAA coords from PDB
            for atom in rwmol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info:
                    name = info.GetName().strip()
                    if name in ncaa_heavy_coords:
                        c = ncaa_heavy_coords[name]
                        conf.SetAtomPosition(atom.GetIdx(), (c[0], c[1], c[2]))

            # Set cap coords from the capped PDB text
            for line in pdb_text.split('\n'):
                if not line.startswith('ATOM') or len(line) < 54:
                    continue
                rn = line[17:20].strip()
                name = line[12:16].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                if rn == 'ACE':
                    if name == 'CH3':
                        conf.SetAtomPosition(ace_ch3, (x, y, z))
                    elif name == 'C':
                        conf.SetAtomPosition(ace_c, (x, y, z))
                    elif name == 'O':
                        conf.SetAtomPosition(ace_o, (x, y, z))
                elif rn == 'NME':
                    if name == 'N':
                        conf.SetAtomPosition(nme_n, (x, y, z))
                    elif name == 'CH3':
                        conf.SetAtomPosition(nme_ch3, (x, y, z))

            mol_capped = rwmol.GetMol()
            Chem.SanitizeMol(mol_capped)
            # The CCD entry is the neutral molecule (SEP, TPO and PTR as phosphoric
            # acids, ORN, DAB and DPP with NH2), and RDKit's AddHs kept it neutral at
            # any pH (ncaa review P1). The hydrogens now come from OpenBabel's pH model
            # on the CCD hydrogen counts (parametrize_ligand.ob_protonate, an amine
            # NH3+), and the pH 7.4 acid rules follow below, as for the ligands and
            # the fused glycan residues.
            mol_rdkit = PL.ob_protonate(mol_capped, ph)
            if mol_rdkit is None:
                raise RuntimeError("RDKit could not read OpenBabel's protonated molecule")
            if ([a.GetAtomicNum() for a in mol_rdkit.GetAtoms()][:mol_capped.GetNumAtoms()]
                    != [a.GetAtomicNum() for a in mol_capped.GetAtoms()]):
                raise RuntimeError("OpenBabel's protonated molecule does not keep the heavy atoms in order")
            formal_charge = Chem.GetFormalCharge(mol_rdkit)
            print(f"  Capped molecule: {mol_rdkit.GetNumAtoms()} atoms, charge={formal_charge} "
                  f"(OpenBabel, pH {ph:g})")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"CCD template approach failed: {e}")

    # The pH 7.4 acid rules of the ligand path, on either strategy's molecule: a
    # phosphate monoester (SEP, TPO, PTR) -2, a carboxylate -1. OpenBabel's pH
    # model is not the ligand path's last word on acids either. Only hydrogens change,
    # so the heavy-atom order holds; the integer residue charge then goes to CA as for
    # every NCAA.
    if PL.ACID_PH_WINDOW[0] <= ph <= PL.ACID_PH_WINDOW[1]:
        mol_rdkit, notes = PL.ionize_acids(mol_rdkit)
        for n in notes:
            print(f"  acid rule: {n}")
    else:
        print(f"  Warning: pH {ph} is outside {PL.ACID_PH_WINDOW}: the pH 7.4 acid rules are not applied")
    formal_charge = Chem.GetFormalCharge(mol_rdkit)
    print(f"  residue charge at pH {ph:g}: {formal_charge:+d} (capped molecule)")

    # Strip PDB metadata (OpenFF requires clean mol without residue info)
    sdf_clean = Chem.MolToMolBlock(mol_rdkit)
    mol_rdkit = Chem.MolFromMolBlock(sdf_clean, removeHs=False, sanitize=True)
    if mol_rdkit is None:
        raise RuntimeError("final SDF round-trip failed")
    print(f"  RDKit: {mol_rdkit.GetNumAtoms()} atoms, SMILES: {Chem.MolToSmiles(mol_rdkit)}")

    # OpenFF
    offmol = Molecule.from_rdkit(mol_rdkit, allow_undefined_stereo=True)
    ff = ForceField(ff_name)
    topology = Topology.from_molecules([offmol])
    topology.box_vectors = np.eye(3) * 5.0 * unit.nanometer

    interchange = Interchange.from_smirnoff(ff, topology)
    print(f"  Parametrized with {ff_name}")

    # Export to GROMACS in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "ncaa")
        interchange.to_gromacs(prefix)

        with open(f"{prefix}.top") as f:
            top_content = f.read()
        with open(f"{prefix}.gro") as f:
            gro_content = f.read()

    # Recover PDB atom names via MCS match against the input-PDB NCAA fragment.
    # OpenFF's GRO emits generic element labels ("C", "N", "O") and the MolBlock
    # round-trip above drops any PDBResidueInfo set in Strategy 2's CCD-CIF path,
    # so neither coord nor name matching can recover full atom-name assignment for
    # NCAAs whose template coords differ from the input PDB (e.g. \u03b2-amino acids
    # B3E, HMR where coord matching only found 3/7 and 3/12 respectively).
    atom_names_by_idx = [None] * mol_rdkit.GetNumAtoms()
    ref_lines = [l for l in pdb_text.split('\n')
                 if l.startswith('ATOM') and len(l) > 20 and l[17:20].strip() == ncaa_resname]
    ref_lines.append('END')
    ref_mol = Chem.MolFromPDBBlock('\n'.join(ref_lines) + '\n', removeHs=True, sanitize=False)
    if ref_mol is not None and ref_mol.GetNumAtoms() > 0:
        try:
            from rdkit.Chem import rdFMCS
            mol_heavy = Chem.RemoveHs(mol_rdkit)
            heavy_to_rdkit = [a.GetIdx() for a in mol_rdkit.GetAtoms() if a.GetAtomicNum() > 1]
            mcs = rdFMCS.FindMCS(
                [mol_heavy, ref_mol],
                bondCompare=rdFMCS.BondCompare.CompareAny,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                ringMatchesRingOnly=False,
                completeRingsOnly=False,
                timeout=30,
            )
            if mcs.numAtoms > 0:
                q = Chem.MolFromSmarts(mcs.smartsString)
                m_big = mol_heavy.GetSubstructMatch(q)
                m_ref = ref_mol.GetSubstructMatch(q)
                for i in range(min(len(m_big), len(m_ref))):
                    info = ref_mol.GetAtomWithIdx(m_ref[i]).GetPDBResidueInfo()
                    if info is None:
                        continue
                    atom_names_by_idx[heavy_to_rdkit[m_big[i]]] = info.GetName().strip()
                n_named = sum(1 for n in atom_names_by_idx if n is not None)
                print(f"  MCS mapped {n_named}/{ref_mol.GetNumAtoms()} NCAA atoms to PDB names")
        except Exception as e:
            print(f"  MCS atom-name recovery failed ({e}); will rely on coord/name match")

    return top_content, gro_content, mol_rdkit, formal_charge, atom_names_by_idx


# ---- Topology parsing ----

def parse_top(content):
    """Parse GROMACS topology from OpenFF Interchange output."""
    result = {'atomtypes': [], 'atoms': [], 'bonds': [], 'angles': [],
              'dihedrals': [], 'impropers': []}
    section = None
    for line in content.split('\n'):
        s = line.strip()
        if s.startswith('['):
            section = s.strip('[] ').lower()
            continue
        if not s or s.startswith(';') or s.startswith('#'):
            continue
        p = s.split()
        try:
            if section == 'atomtypes' and len(p) >= 7:
                result['atomtypes'].append({
                    'name': p[0], 'at_num': int(p[1]), 'mass': float(p[2]),
                    'sigma': float(p[5]), 'epsilon': float(p[6]),
                })
            elif section == 'atoms' and len(p) >= 7:
                result['atoms'].append({
                    'idx': int(p[0]) - 1, 'type': p[1],
                    'charge': float(p[6]),
                    'mass': float(p[7]) if len(p) > 7 else 0.0,
                })
            elif section == 'bonds' and len(p) >= 5:
                result['bonds'].append((int(p[0])-1, int(p[1])-1,
                                        int(p[2]), float(p[3]), float(p[4])))
            elif section == 'angles' and len(p) >= 6:
                result['angles'].append((int(p[0])-1, int(p[1])-1, int(p[2])-1,
                                         int(p[3]), float(p[4]), float(p[5])))
            elif section == 'dihedrals' and len(p) >= 8:
                func = int(p[4])
                entry = (int(p[0])-1, int(p[1])-1, int(p[2])-1, int(p[3])-1,
                         func, float(p[5]), float(p[6]), int(float(p[7])))
                if func == 4:
                    result['impropers'].append(entry)
                else:
                    result['dihedrals'].append(entry)
        except (ValueError, IndexError):
            continue
    return result


def parse_gro_coords(content):
    """Parse GRO file, return list of (x,y,z) arrays in nm."""
    lines = content.strip().split('\n')
    n = int(lines[1].strip())
    coords = []
    for line in lines[2:2+n]:
        coords.append(np.array([float(line[20:28]), float(line[28:36]), float(line[36:44])]))
    return coords


# ---- Atom matching and type/charge assignment ----

def match_ncaa_atoms(gro_coords, ncaa_heavy_coords, gro_content=None):
    """Match GRO atom indices to NCAA PDB atom names.
    First tries coordinate proximity (PDB Å, GRO nm), then fills remaining unmatched
    GRO atoms by atom name from the GRO file. Previously the name-based fallback only
    fired when coordinate matching matched nothing, which silently dropped atoms for
    \u03b2-amino acids and other NCAAs where the PDB template coords drift from GRO
    coords enough that only some (not all) atoms match within 0.05 \u00c5 (e.g. B3E: 3/7)."""
    mapping = {}
    for gro_idx, gc in enumerate(gro_coords):
        for name, pc in ncaa_heavy_coords.items():
            if name in mapping.values():
                continue
            if np.linalg.norm(gc - pc / 10.0) < 0.005:  # 0.05 Angstrom
                mapping[gro_idx] = name
                break

    # Fill unmatched GRO atoms by atom name from the GRO file.
    if gro_content is not None:
        gro_lines = gro_content.strip().split('\n')
        n = int(gro_lines[1].strip())
        ncaa_names_set = set(ncaa_heavy_coords.keys())
        for i, line in enumerate(gro_lines[2:2+n]):
            if i in mapping or len(line) < 15:
                continue
            atom_name = line[10:15].strip()
            if atom_name in ncaa_names_set and atom_name not in mapping.values():
                mapping[i] = atom_name
    return mapping


def build_adjacency(top):
    """Build adjacency dict from bond list."""
    adj = {}
    for ai, aj, *_ in top['bonds']:
        adj.setdefault(ai, set()).add(aj)
        adj.setdefault(aj, set()).add(ai)
    return adj


def find_ncaa_h_atoms(adj, ncaa_heavy_indices, top):
    """Find H atoms bonded to NCAA heavy atoms. Returns dict: h_idx -> parent_name."""
    # Build element lookup from atomtypes
    type_elem = {}
    for at in top['atomtypes']:
        type_elem[at['name']] = at['at_num']

    h_atoms = {}
    for h_idx in range(len(top['atoms'])):
        if h_idx in ncaa_heavy_indices:
            continue
        atom_type = top['atoms'][h_idx]['type']
        if type_elem.get(atom_type, 0) != 1:
            continue
        for neighbor in adj.get(h_idx, set()):
            if neighbor in ncaa_heavy_indices:
                h_atoms[h_idx] = ncaa_heavy_indices[neighbor]
                break
    return h_atoms


def assign_h_names(h_atoms):
    """Assign proper PDB hydrogen names based on parent heavy atom."""
    h_counts = {}
    h_names = {}
    # Count H per parent
    parent_h_total = {}
    for parent in h_atoms.values():
        parent_h_total[parent] = parent_h_total.get(parent, 0) + 1

    for h_idx in sorted(h_atoms.keys()):
        parent = h_atoms[h_idx]
        h_counts.setdefault(parent, 0)
        h_counts[parent] += 1
        count = h_counts[parent]
        total = parent_h_total[parent]

        if parent == 'N':
            h_names[h_idx] = 'H'
        elif parent == 'CA':
            h_names[h_idx] = 'HA'
        elif parent == 'CB':
            h_names[h_idx] = f'HB{count}'
        elif parent.startswith('O'):
            h_names[h_idx] = f'H{parent}'
        else:
            suffix = parent[1:]  # CD1→D1, NE1→E1, CZ2→Z2
            if total == 1:
                h_names[h_idx] = f'H{suffix}'
            else:
                h_names[h_idx] = f'H{suffix}{count}'
    return h_names


def assign_types_and_charges(ncaa_atom_map, h_atoms, h_names, top, resname, formal_charge, mol=None):
    """Assign atom types and charges for NCAA atoms.

    Backbone: AMBER19SB types and fixed charges (N,H,C,O)
    CA, HA: AMBER19SB types, OpenFF charges
    CB: AMBER19SB type CT, OpenFF charge; its hydrogens HC, H1, H2 or H3 by the number
    of CB's electron-withdrawing neighbours (CB_H_TYPES), HP next to an amine N
    (CB_AMINE_H_TYPE; needs `mol`, the RDKit molecule in the topology's atom order)
    Sidechain: OpenFF types (prefixed), OpenFF charges
    """
    prefix = resname + '_'
    all_ncaa = set(ncaa_atom_map.keys()) | set(h_atoms.keys())
    type_map = {}
    charge_map = {}
    cb_h_type = CB_H_TYPE
    cb_idx = next((i for i, n in ncaa_atom_map.items() if n == 'CB'), None)
    if cb_idx is not None:
        at_num = {at['name']: at['at_num'] for at in top['atomtypes']}
        cb_nbrs = build_adjacency(top).get(cb_idx, ())
        n_ew = sum(1 for j in cb_nbrs if at_num.get(top['atoms'][j]['type']) in ELECTRON_WITHDRAWING)
        cb_h_type = CB_H_TYPES[min(n_ew, len(CB_H_TYPES) - 1)]
        if mol is not None and any(is_amine_n(mol.GetAtomWithIdx(j)) for j in cb_nbrs):
            cb_h_type = CB_AMINE_H_TYPE

    for idx in all_ncaa:
        if idx in ncaa_atom_map:
            name = ncaa_atom_map[idx]
            if name in BACKBONE_TYPES:
                type_map[idx] = BACKBONE_TYPES[name]
                if name in BACKBONE_FIXED_CHARGES:
                    charge_map[idx] = BACKBONE_FIXED_CHARGES[name]
                else:
                    charge_map[idx] = top['atoms'][idx]['charge']
            elif name == 'CB':
                type_map[idx] = CB_TYPE
                charge_map[idx] = top['atoms'][idx]['charge']
            else:
                type_map[idx] = prefix + top['atoms'][idx]['type']
                charge_map[idx] = top['atoms'][idx]['charge']
        else:
            # H atom
            parent = h_atoms[idx]
            if parent in ('N',):
                type_map[idx] = 'H'
                charge_map[idx] = BACKBONE_FIXED_CHARGES.get('H', top['atoms'][idx]['charge'])
            elif parent in ('CA',):
                type_map[idx] = 'H1'
                charge_map[idx] = top['atoms'][idx]['charge']
            elif parent == 'CB':
                type_map[idx] = cb_h_type
                charge_map[idx] = top['atoms'][idx]['charge']
            else:
                type_map[idx] = prefix + top['atoms'][idx]['type']
                charge_map[idx] = top['atoms'][idx]['charge']

    # Adjust CA charge for total charge balance
    total = sum(charge_map.values())
    correction = formal_charge - total
    ca_idx = next((i for i, n in ncaa_atom_map.items() if n == 'CA'), None)
    if ca_idx is not None:
        charge_map[ca_idx] += correction
        print(f"  Charge correction on CA: {correction:+.6f} (total: {formal_charge})")
        # The rtp holds 6 decimals. Rounded one by one, the written charges of the
        # 1MAH NG1 summed to -1.4e-5 e and those of the 5XJE residues to +1.3e-5 and
        # +1.7e-5 e (critic F7), so pdb2gmx and grompp saw a residue that is not
        # integer. The charges are rounded to whole micro-e here, and the residual of
        # the rounded sum goes to CA as well: the written charges sum exactly to the
        # formal charge.
        micro = {i: int(round(q * 1e6)) for i, q in charge_map.items()}
        micro[ca_idx] += int(round(formal_charge * 1e6)) - sum(micro.values())
        charge_map = {i: m / 1e6 for i, m in micro.items()}

    return type_map, charge_map, prefix


# ---- Collect new atom types and bonded parameters ----

def collect_new_params(top, all_ncaa, type_map, prefix):
    """Collect new atom types and bonded params involving NCAA sidechain types."""
    new_atomtypes = {}
    new_bonds = {}
    new_angles = {}
    new_dihedrals = []
    new_impropers = []

    # Atom types: any prefixed type is new
    for idx, final_type in type_map.items():
        if final_type.startswith(prefix):
            orig_type = top['atoms'][idx]['type']
            at = next((a for a in top['atomtypes'] if a['name'] == orig_type), None)
            if at and final_type not in new_atomtypes:
                new_atomtypes[final_type] = at

    def get_type(i):
        return type_map.get(i, '??')

    def has_new_type(indices):
        return any(get_type(i).startswith(prefix) for i in indices)

    # Bonds involving ≥1 new type
    for ai, aj, func, b0, kb in top['bonds']:
        if ai in all_ncaa and aj in all_ncaa and has_new_type((ai, aj)):
            key = tuple(sorted([get_type(ai), get_type(aj)]))
            if key not in new_bonds:
                new_bonds[key] = (func, b0, kb)

    # Angles
    for ai, aj, ak, func, theta, k in top['angles']:
        if all(i in all_ncaa for i in (ai, aj, ak)) and has_new_type((ai, aj, ak)):
            types = (get_type(ai), get_type(aj), get_type(ak))
            key = types if types[0] <= types[2] else tuple(reversed(types))
            if key not in new_angles:
                new_angles[key] = (func, theta, k)

    # Proper dihedrals
    for ai, aj, ak, al, func, phase, kd, pn in top['dihedrals']:
        if all(i in all_ncaa for i in (ai, aj, ak, al)) and has_new_type((ai, aj, ak, al)):
            types = (get_type(ai), get_type(aj), get_type(ak), get_type(al))
            new_dihedrals.append(types + (func, phase, kd, pn))

    # Improper dihedrals
    for ai, aj, ak, al, func, phase, kd, pn in top['impropers']:
        if all(i in all_ncaa for i in (ai, aj, ak, al)) and has_new_type((ai, aj, ak, al)):
            types = (get_type(ai), get_type(aj), get_type(ak), get_type(al))
            new_impropers.append(types + (func, phase, kd, pn))

    # Also collect cross-boundary bonded params (backbone AMBER + sidechain new types)
    # These involve NCAA atoms where some have AMBER types and some have prefixed types
    for ai, aj, func, b0, kb in top['bonds']:
        if ai in all_ncaa and aj in all_ncaa:
            types = (get_type(ai), get_type(aj))
            if has_new_type((ai, aj)):
                key = tuple(sorted(types))
                if key not in new_bonds:
                    new_bonds[key] = (func, b0, kb)

    for ai, aj, ak, func, theta, k in top['angles']:
        if all(i in all_ncaa for i in (ai, aj, ak)):
            if has_new_type((ai, aj, ak)):
                types = (get_type(ai), get_type(aj), get_type(ak))
                key = types if types[0] <= types[2] else tuple(reversed(types))
                if key not in new_angles:
                    new_angles[key] = (func, theta, k)

    for ai, aj, ak, al, func, phase, kd, pn in top['dihedrals']:
        if all(i in all_ncaa for i in (ai, aj, ak, al)):
            if has_new_type((ai, aj, ak, al)):
                types = (get_type(ai), get_type(aj), get_type(ak), get_type(al))
                new_dihedrals.append(types + (func, phase, kd, pn))

    for ai, aj, ak, al, func, phase, kd, pn in top['impropers']:
        if all(i in all_ncaa for i in (ai, aj, ak, al)):
            if has_new_type((ai, aj, ak, al)):
                types = (get_type(ai), get_type(aj), get_type(ak), get_type(al))
                new_impropers.append(types + (func, phase, kd, pn))

    # Deduplicate dihedrals (same type quartet can have multiple terms with different pn)
    seen_dih = set()
    unique_dih = []
    for entry in new_dihedrals:
        key = entry[:4] + (entry[4],) + (entry[7],)  # types + func + pn
        if key not in seen_dih:
            seen_dih.add(key)
            unique_dih.append(entry)
    new_dihedrals = unique_dih

    seen_imp = set()
    unique_imp = []
    for entry in new_impropers:
        key = entry[:5] + (entry[7],)
        if key not in seen_imp:
            seen_imp.add(key)
            unique_imp.append(entry)
    new_impropers = unique_imp

    return new_atomtypes, new_bonds, new_angles, new_dihedrals, new_impropers


# ---- Generate RTP entry ----

def generate_rtp(resname, ncaa_atom_map, h_atoms, h_names, ncaa_atom_names,
                 type_map, charge_map, top, all_ncaa, adj):
    """Generate RTP entry text for the NCAA residue."""
    lines = [f'[ {resname} ]', ' [ atoms ]']

    # Order: N, H, CA, HA, CB, HB*, sidechain atoms, C, O
    atom_order = []

    # Backbone start: N (+ its H), CA (+ its HA)
    for parent_name in ['N', 'CA']:
        idx = next((i for i, n in ncaa_atom_map.items() if n == parent_name), None)
        if idx is not None:
            atom_order.append(idx)
            for h_idx in sorted(h_atoms.keys()):
                if h_atoms[h_idx] == parent_name:
                    atom_order.append(h_idx)

    # CB + HB
    cb_idx = next((i for i, n in ncaa_atom_map.items() if n == 'CB'), None)
    if cb_idx is not None:
        atom_order.append(cb_idx)
        for h_idx in sorted(h_atoms.keys()):
            if h_atoms[h_idx] == 'CB':
                atom_order.append(h_idx)

    # Sidechain heavy atoms + their H
    for name in ncaa_atom_names:
        if name in BACKBONE_NAMES or name == 'CB':
            continue
        idx = next((i for i, n in ncaa_atom_map.items() if n == name), None)
        if idx is not None:
            atom_order.append(idx)
            for h_idx in sorted(h_atoms.keys()):
                if h_atoms[h_idx] == name:
                    atom_order.append(h_idx)

    # Backbone end
    for name in ['C', 'O']:
        idx = next((i for i, n in ncaa_atom_map.items() if n == name), None)
        if idx is not None:
            atom_order.append(idx)

    # Write [ atoms ]
    for cgnr, idx in enumerate(atom_order, 1):
        name = ncaa_atom_map.get(idx, h_names.get(idx, '??'))
        atype = type_map[idx]
        charge = charge_map[idx]
        lines.append(f'  {name:<6s} {atype:<12s} {charge:12.6f} {cgnr:5d}')

    # Write [ bonds ] — intra-residue bonds
    lines.append(' [ bonds ]')
    idx_set = set(atom_order)
    for ai, aj, *_ in top['bonds']:
        if ai in idx_set and aj in idx_set:
            name_i = ncaa_atom_map.get(ai, h_names.get(ai))
            name_j = ncaa_atom_map.get(aj, h_names.get(aj))
            if name_i and name_j:
                lines.append(f'  {name_i:<6s} {name_j}')
    # Inter-residue peptide bond
    lines.append('  -C     N')

    # Write [ impropers ]
    lines.append(' [ impropers ]')
    # Standard peptide bond impropers
    lines.append(' -C    CA    N     H')
    lines.append(' CA    +N    C     O')
    # Sidechain impropers from OpenFF, sorted: OpenFF lists them in an order that
    # changes from run to run, so two runs on one input wrote ncaa.rtp files that
    # differed in that order only (pdb2gmx sorts them; ncaa review F4).
    side = []
    for entry in top['impropers']:
        ai, aj, ak, al = entry[:4]
        if all(i in idx_set for i in (ai, aj, ak, al)):
            names = [ncaa_atom_map.get(i, h_names.get(i)) for i in (ai, aj, ak, al)]
            if all(names):
                # Skip pure backbone impropers (already listed)
                if not all(n in BACKBONE_NAMES for n in names):
                    side.append(f' {names[0]:<5s} {names[1]:<5s} {names[2]:<5s} {names[3]}')
    lines += sorted(side)

    # CMAP
    lines.append(' [ cmap ]')
    lines.append(' -C    N     CA    C     +N')

    return '\n'.join(lines)


# ---- Generate HDB entry ----

def generate_hdb(resname, ncaa_atom_names, h_atoms, h_names, ncaa_atom_map, adj, top):
    """Generate HDB entry for hydrogen addition."""
    # Build element lookup
    type_elem = {}
    for at in top['atomtypes']:
        type_elem[at['name']] = at['at_num']

    # Group H by parent
    parent_h_count = {}
    for parent in h_atoms.values():
        parent_h_count[parent] = parent_h_count.get(parent, 0) + 1

    hdb_lines = []

    # Process parents in NCAA atom order
    def get_refs(parent_name, heavy_neighbors, n_needed):
        """Get n_needed reference atoms; traverse grandparents if not enough direct neighbors."""
        refs = heavy_neighbors[:n_needed]
        while len(refs) < n_needed:
            # Go one level deeper: find neighbors of the last ref atom
            last_ref = refs[-1] if refs else None
            if last_ref is None:
                break
            last_idx = next((i for i, n in ncaa_atom_map.items() if n == last_ref), None)
            if last_idx is None:
                break
            found = False
            for gp_nb in sorted(adj.get(last_idx, set())):
                gp_name = ncaa_atom_map.get(gp_nb)
                if gp_name and gp_name != parent_name and gp_name not in refs:
                    refs.append(gp_name)
                    found = True
                    break
            if not found:
                break
        return refs

    ordered_parents = ['N', 'CA', 'CB'] + [n for n in ncaa_atom_names
                                            if n not in BACKBONE_NAMES and n != 'CB']
    for parent_name in ordered_parents:
        if parent_name not in parent_h_count:
            continue
        n_h = parent_h_count[parent_name]
        parent_idx = next((i for i, n in ncaa_atom_map.items() if n == parent_name), None)
        if parent_idx is None:
            continue

        # Get heavy-atom neighbors of parent
        heavy_neighbors = []
        for nb in sorted(adj.get(parent_idx, set())):
            nb_name = ncaa_atom_map.get(nb)
            if nb_name:
                heavy_neighbors.append(nb_name)

        # Get H names for this parent
        parent_h_names = sorted(h_names[i] for i, p in h_atoms.items() if p == parent_name)
        # Base name for HDB: strip only the last digit (instance number)
        # e.g., HG21 → HG2 (not HG), HB1 → HB, HO2P → HO2P (single H, no stripping)
        if parent_h_names:
            if n_h > 1:
                # Multi-H group: strip last char (instance digit) from first H name
                base_name = parent_h_names[0][:-1] if parent_h_names[0][-1].isdigit() else parent_h_names[0]
            else:
                base_name = parent_h_names[0]
        else:
            base_name = 'H'

        # Determine parent element
        parent_elem = type_elem.get(top['atoms'][parent_idx]['type'], 6)

        if parent_name == 'N':
            hdb_lines.append(f"1\t1\tH\tN\t-C\tCA")
        elif parent_name == 'CA':
            refs = [n for n in heavy_neighbors if n not in ('H', 'HA')]
            refs = get_refs(parent_name, refs, 3)
            hdb_lines.append(f"1\t5\tHA\tCA\t" + "\t".join(refs))
        elif n_h == 3:
            refs = get_refs(parent_name, heavy_neighbors, 2)
            hdb_lines.append(f"3\t4\t{base_name}\t{parent_name}\t" + "\t".join(refs))
        elif n_h == 2:
            refs = get_refs(parent_name, heavy_neighbors, 2)
            hdb_lines.append(f"2\t6\t{base_name}\t{parent_name}\t" + "\t".join(refs))
        elif n_h == 1:
            h_name = parent_h_names[0]
            if parent_elem == 8 or parent_elem == 16:  # oxygen/sulfur → hydroxyl/thiol
                refs = get_refs(parent_name, heavy_neighbors, 2)
                hdb_lines.append(f"1\t2\t{h_name}\t{parent_name}\t" + "\t".join(refs))
            elif parent_elem == 7:  # nitrogen
                refs = get_refs(parent_name, heavy_neighbors, 2)
                hdb_lines.append(f"1\t1\t{h_name}\t{parent_name}\t" + "\t".join(refs))
            else:  # carbon
                n_total = len(adj.get(parent_idx, set()))
                if n_total <= 3:  # sp2
                    refs = heavy_neighbors[:2]
                    hdb_lines.append(f"1\t1\t{h_name}\t{parent_name}\t" + "\t".join(refs))
                else:  # sp3
                    refs = get_refs(parent_name, heavy_neighbors, 3)
                    hdb_lines.append(f"1\t5\t{h_name}\t{parent_name}\t" + "\t".join(refs))

    return f"{resname}\t{len(hdb_lines)}\n" + '\n'.join(hdb_lines)


# ---- Write force field files ----

def merge_hdb(text, entries):
    """(text, replaced names): aminoacids.hdb `text` with every generate_hdb() entry in
    `entries` appended after any entry of the same residue name has been dropped (its
    header and its count lines; blank and comment lines do not count, as for pdb2gmx).

    Names compare as whole words, the first token of an entry header. The substring
    test this replaces appended no entry for a residue named NGL, which is part of NGLN,
    NGLU and NGLY in amber19sb's aminoacids.hdb, and pdb2gmx then built it without
    hydrogens (design 5(e)). It also kept the old entry of a name a local force field
    already held, e.g. after a setup restart that did not remove amber19sb.ff, though
    the name may now stand for another residue (critic F12a)."""
    names = {e.split(None, 1)[0] for e in entries}
    lines = text.split('\n')
    out, replaced, i = [], [], 0
    while i < len(lines):
        tok = lines[i].split()
        if len(tok) == 2 and tok[0] in names and tok[1].isdigit():
            left, i = int(tok[1]), i + 1
            while left and i < len(lines):
                if lines[i].split(';')[0].strip():
                    left -= 1
                i += 1
            replaced.append(tok[0])
            continue
        out.append(lines[i])
        i += 1
    merged = '\n'.join(out).rstrip('\n') + '\n'
    for e in entries:
        merged += '\n' + e + '\n'
    return merged, replaced


def write_ncaa_atomtypes(path, atomtypes):
    """Write ncaa_atomtypes.itp with new sidechain atom types."""
    with open(path, 'w') as f:
        f.write("; NCAA sidechain atom types (generated by parametrize_ncaa.py)\n\n")
        f.write("[ atomtypes ]\n")
        f.write("; name    at.num    mass    charge   ptype   sigma         epsilon\n")
        for name, at in sorted(atomtypes.items()):
            f.write(f" {name:<12s} {at['at_num']:3d}  {at['mass']:10.4f}  0.0000  A  "
                    f"{at['sigma']:.10f}  {at['epsilon']:.10f}\n")


def write_ncaa_bonded(path, bonds, angles, dihedrals, impropers):
    """Write ncaa_bonded.itp with new bonded parameters."""
    with open(path, 'w') as f:
        f.write("; NCAA bonded parameters (generated by parametrize_ncaa.py)\n\n")

        if bonds:
            f.write("[ bondtypes ]\n")
            f.write("; i    j  func       b0          kb\n")
            for (t1, t2), (func, b0, kb) in sorted(bonds.items()):
                f.write(f" {t1:<12s} {t2:<12s} {func:4d}  {b0:.6f}  {kb:.2f}\n")
            f.write("\n")

        if angles:
            f.write("[ angletypes ]\n")
            f.write("; i    j    k  func       th0          cth\n")
            for (t1, t2, t3), (func, theta, k) in sorted(angles.items()):
                f.write(f" {t1:<12s} {t2:<12s} {t3:<12s} {func:4d}  {theta:.4f}  {k:.4f}\n")
            f.write("\n")

        if dihedrals:
            f.write("[ dihedraltypes ]\n")
            f.write("; i    j    k    l  func      phase      kd    pn\n")
            # Group by type quartet so multiple terms are on consecutive lines
            from collections import defaultdict
            dih_groups = defaultdict(list)
            for entry in dihedrals:
                key = entry[:4]
                dih_groups[key].append(entry)
            for key in sorted(dih_groups.keys()):
                for entry in sorted(dih_groups[key], key=lambda x: x[7]):  # sort by pn
                    t1, t2, t3, t4, func, phase, kd, pn = entry
                    # Use function type 9 (multi-term periodic) instead of 1
                    # Type 9 allows multiple terms per quartet on consecutive lines
                    f.write(f" {t1:<12s} {t2:<12s} {t3:<12s} {t4:<12s}    9  "
                            f"{phase:10.4f}  {kd:12.6f}  {pn:4d}\n")
            f.write("\n")

        if impropers:
            f.write("; improper dihedrals\n")
            f.write("[ dihedraltypes ]\n")
            f.write("; i    j    k    l  func      phase      kd    pn\n")
            for entry in sorted(impropers, key=lambda x: x[:4]):
                t1, t2, t3, t4, func, phase, kd, pn = entry
                f.write(f" {t1:<12s} {t2:<12s} {t3:<12s} {t4:<12s} {func:4d}  "
                        f"{phase:10.4f}  {kd:12.6f}  {pn:4d}\n")


# ==== MAIN ====

if not os.path.isfile(args.file):
    print(f"Error: {args.file} not found.", file=sys.stderr)
    sys.exit(1)

# The failure lists describe this run only. They used to be written and never removed,
# so one left by a setup that died before conf.gro sent a residue that parametrises
# now down the fallback of the next attempt (critic F3; the runners remove them too).
for _stale in (NCAA_FAILED, GLYCAN_FAILED):
    if os.path.exists(_stale):
        os.remove(_stale)

pdb_atoms = parse_pdb(args.file)
ncaa_types = find_ncaas(pdb_atoms)

if not ncaa_types:
    print("No NCAA residues found in the input PDB.")
    sys.exit(0)

print(f"Found NCAA residues: {', '.join(f'{rn} ({len(inst)} instance(s))' for rn, inst in ncaa_types.items())}")

# ---- GROMOS mode: classify NCAAs using FF's RTP; skip OpenFF entirely ----
# Must run before the OpenFF loop below. Searches GMXLIB for custom FFs.
if args.gmx_ff.startswith('gromos'):
    import subprocess as _sp_g
    _g_search_dirs = ['/usr/local/gromacs/share/gromacs/top', '/usr/share/gromacs/top']
    try:
        _g_ver = _sp_g.run(['gmx', '--version'], capture_output=True, text=True, timeout=10)
        for _g_line in _g_ver.stdout.split('\n'):
            if 'Data prefix' in _g_line:
                _g_search_dirs.insert(0, _g_line.split()[-1] + '/share/gromacs/top')
                break
    except Exception:
        pass
    for _g_gmxlib in os.environ.get('GMXLIB', '').split(':'):
        if _g_gmxlib:
            _g_search_dirs.append(_g_gmxlib)
    _g_ff_src = None
    for _g_dir in _g_search_dirs:
        _g_cand = os.path.join(_g_dir, f'{args.gmx_ff}.ff')
        if os.path.isdir(_g_cand):
            _g_ff_src = _g_cand
            break
    if _g_ff_src is None:
        print(f"Error: Could not find {args.gmx_ff}.ff in GROMACS installation or GMXLIB", file=sys.stderr)
        sys.exit(1)

    def _load_ff_residues(rtp_path):
        _section_kw = {'bondedtypes', 'atoms', 'bonds', 'impropers', 'dihedrals',
                       'exclusions', 'settles', 'constraints', 'pairs', 'angles'}
        names = set()
        with open(rtp_path) as _f:
            for _line in _f:
                _s = _line.strip()
                if _s.startswith('[') and _s.endswith(']'):
                    _n = _s[1:-1].strip()
                    if _n not in _section_kw:
                        names.add(_n)
        return names

    _g_residues = _load_ff_residues(os.path.join(_g_ff_src, 'aminoacids.rtp'))
    _g_ions_rtp = os.path.join(_g_ff_src, 'ions.rtp')
    if os.path.isfile(_g_ions_rtp):
        _g_residues |= _load_ff_residues(_g_ions_rtp)

    # PDB CCD codes → (GROMOS resname, {pdb_atom: gromos_atom})
    # Covers PTMs whose RCSB names differ from GROMOS 54A8 internal names.
    # Atom renames listed for heavy atoms only; pdb2gmx -ignh adds H from HDB.
    _PDB_TO_GROMOS = {
        # Phosphorylated residues (T1P/S1P/Y1P = -1 charge; T2P/S2P/Y2P = -2)
        # Use -1 by default (one phosphate OH); pdb2gmx adds HE3/HI3 from HDB.
        'SEP': ('S1P', {'P': 'PD', 'O1P': 'OE1', 'O2P': 'OE2', 'O3P': 'OE3'}),
        'TPO': ('T1P', {'P': 'PD', 'O1P': 'OE1', 'O2P': 'OE2', 'O3P': 'OE3'}),
        'PTR': ('Y1P', {'P': 'PT',  'O1P': 'OI1', 'O2P': 'OI2', 'O3P': 'OI3'}),
        # Sulfotyrosine
        'TYS': ('YSU', {'S': 'ST',  'O1S': 'OI1', 'O2S': 'OI2', 'O3S': 'OI3'}),
        # D-amino acids with different PDB CCD name
        'DAL': ('DALA', {}),
        # Norleucine (straight-chain — same heavy atom names as GROMOS LNO)
        'NLE': ('LNO', {}),
        # Cysteic acid (GROMOS CSE = PDB OCS: SG, OD1, OD2, OD3 same names)
        'OCS': ('CSE', {}),
        # Cysteine sulfinic acid (GROMOS CSA = PDB CSO: SG, OD1, OD2 same names)
        'CSO': ('CSA', {}),
    }

    gromos_known, gromos_unknown, gromos_renames = [], [], {}
    for resname in ncaa_types:
        if resname in _g_residues:
            gromos_known.append(resname)
            print(f"  {resname}: in {args.gmx_ff}.ff — pdb2gmx will handle directly")
        elif resname in _PDB_TO_GROMOS:
            gros_name, atom_map = _PDB_TO_GROMOS[resname]
            gromos_known.append(gros_name)
            gromos_renames[resname] = (gros_name, atom_map)
            print(f"  {resname}: maps to {args.gmx_ff}.ff residue {gros_name} — will rename in PDB")
        else:
            gromos_unknown.append(resname)
            print(f"  {resname}: NOT in {args.gmx_ff}.ff — will be replaced with parent residue")

    if gromos_known:
        with open('ncaa_residues.gs', 'w') as _f:
            for _rn in gromos_known:
                _f.write(f"{_rn}\n")
    if gromos_unknown:
        with open('ncaa_failed.gs', 'w') as _f:
            for _rn in gromos_unknown:
                _f.write(f"{_rn}\n")
    if gromos_renames:
        # ncaa_rename.gs: PDB_resname<TAB>GROMOS_resname<TAB>atom=atom,... (one per line)
        with open('ncaa_rename.gs', 'w') as _f:
            for _pdb_rn, (_gros_rn, _atom_map) in gromos_renames.items():
                _atoms_str = ','.join(f"{k}={v}" for k, v in _atom_map.items())
                _f.write(f"{_pdb_rn}\t{_gros_rn}\t{_atoms_str}\n")
        print(f"  Renames to apply to PDB: {list(gromos_renames.keys())}")

    print(f"\nGROMOS NCAA summary: {len(gromos_known)} FF-native ({len(gromos_renames)} need PDB rename), "
          f"{len(gromos_unknown)} unknown (→ parent replacement)")
    sys.exit(0)

# Accumulate across all NCAA types
all_atomtypes = {}
all_bonds = {}
all_angles = {}
all_dihedrals = []
all_impropers = []
all_rtp = []
all_hdb = []

failed_ncaas = []  # NCAAs that couldn't be parametrized → fall back to PDBFixer replacement
failed_glycans = {}  # fused glycan residue -> why it could not be parametrised (glycan_failed.gs)
parents = {}       # NCAA -> parent residue, whose AMBER19SB CMAP it gets

# The fused glycan residues and their chemistry. An unreadable glycans.json ends the
# run with exit 1, which the runners count as a failure of every fused residue: read
# as an ordinary NCAA, a fused name would be looked up in the CCD.
try:
    GLYCAN_DEFS = load_glycan_defs(GLYCANS_JSON)
except (OSError, ValueError, KeyError, TypeError) as e:
    print(f"Error: cannot read {GLYCANS_JSON} ({type(e).__name__}: {e})", file=sys.stderr)
    sys.exit(1)
if GLYCAN_DEFS:
    print(f"Fused glycan residues defined in {GLYCANS_JSON}: {', '.join(GLYCAN_DEFS)}")


def fail(resname, reason):
    """Send a residue that could not be parametrised down its fallback: a fused glycan
    residue to glycan_failed.gs (its sugars go back to the ligands), never to
    ncaa_failed.gs, whose parent replacement would look its name up in the CCD
    (fix_pdb.py); any other NCAA to ncaa_failed.gs (its parent residue)."""
    if resname in GLYCAN_DEFS:
        print(f"  {resname} is a fused glycan residue: {GLYCAN_FAILED}, its sugars go back to the ligands")
        failed_glycans[resname] = ' '.join(str(reason).split())[:300] or 'parametrisation failed'
    else:
        print(f"  Will fall back to PDBFixer parent-residue replacement for {resname}")
        failed_ncaas.append(resname)


for resname, instances in ncaa_types.items():
    # THE PARENT FIRST. It used to be read from files.rcsb.org only, after the
    # parametrisation, with ALA as the silent fallback: on a node without network the
    # TRQ of 2MTA and the HIC of 2BTF got ALA's CMAP although the CCD cache holds their
    # entries (review L-4). That fixed the parent alone: Strategy 2 still downloaded its
    # template, so offline both residues became their parents anyway, until its template
    # came from the cache as well (ccd_template(), ncaa review P2). Now the cache
    # answers first for both, and an NCAA whose CCD entry
    # cannot be read fails like one that cannot be parametrized (ncaa_failed.gs:
    # fix_pdb.py puts in the parent residue when PDBFixer knows the NCAA or
    # files.rcsb.org names its parent, and otherwise pdb2gmx stops the structure). An
    # entry that names no parent (D- and beta-amino acids) keeps ALA's CMAP, with a
    # note: failing those too sent every D-amino acid to PDBFixer, which renames DAL
    # to ALA, and left BAL and ACB for pdb2gmx to stop on (review V2). A fused glycan
    # residue's parent is its definition's.
    if args.gmx_ff.startswith('amber19sb'):
        parent, why = ccd_parent(resname, GLYCAN_DEFS)
        if parent is None:
            print(f"\n  WARNING: {resname}: no parent residue for its CMAP ({why})")
            fail(resname, f"no parent residue for its CMAP ({why})")
            continue
        if why:
            print(f"\n  NOTE: {resname}: {why}")
        parents[resname] = parent
    # Choose the instance with the most heavy atoms as canonical: crystal structures
    # sometimes have truncated copies of the same NCAA (missing sidechain atoms),
    # and parametrizing the shortest version produces an rtp that pdb2gmx rejects
    # for the complete copies (e.g. 4DZV's B3E residue 196 has 7 atoms while residue
    # 203 has 10 — rtp must include the full sidechain to cover both).
    def _heavy_count(ch_rn):
        ch, rn = ch_rn
        return sum(1 for a in pdb_atoms
                   if a['chain'] == ch and a['resnum'] == rn and not a['name'].startswith('H'))
    chain, resnum = max(instances, key=_heavy_count)
    print(f"\n=== Parametrizing {resname} (chain {chain}, residue {resnum}) ===")

    try:
        # Build capped tripeptide
        pdb_text, ncaa_heavy_coords, ncaa_atom_names = build_capped_pdb(pdb_atoms, chain, resnum)

        # Parametrize with OpenFF
        top_content, gro_content, mol_rdkit, formal_charge, atom_names_by_idx = parametrize_capped(
            pdb_text, ncaa_heavy_coords, resname, args.ff, args.ph)

        # Parse topology and coordinates
        top = parse_top(top_content)
        gro_coords = parse_gro_coords(gro_content)

        # Primary mapping: atom_names_by_idx from MCS match (robust across coord/naming
        # drift). Augment with coord/name matching from match_ncaa_atoms as a fallback
        # for atoms MCS could not resolve.
        ncaa_atom_map = {}
        for i, name in enumerate(atom_names_by_idx):
            if name is not None and name in ncaa_heavy_coords and name not in ncaa_atom_map.values():
                ncaa_atom_map[i] = name
        coord_map = match_ncaa_atoms(gro_coords, ncaa_heavy_coords, gro_content)
        for gro_idx, name in coord_map.items():
            if gro_idx not in ncaa_atom_map and name not in ncaa_atom_map.values():
                ncaa_atom_map[gro_idx] = name
        print(f"  Matched {len(ncaa_atom_map)}/{len(ncaa_heavy_coords)} NCAA heavy atoms")
        for name in ncaa_heavy_coords:
            if name not in ncaa_atom_map.values():
                print(f"    WARNING: unmatched atom {name}")

        # Build adjacency and find H atoms
        adj = build_adjacency(top)
        h_atoms = find_ncaa_h_atoms(adj, ncaa_atom_map, top)
        h_names = assign_h_names(h_atoms)
        if resname in GLYCAN_DEFS:
            # Named on the protonated molecule (fused_h_names()); assign_h_names()
            # gives C2A and N2A of a GlcNAc the same H2A.
            h_names = {h: atom_names_by_idx[h] for h in h_atoms}
            named = {i for i, n in enumerate(atom_names_by_idx) if n is not None and i not in ncaa_atom_map}
            if None in h_names.values() or set(h_atoms) != named:
                raise RuntimeError("the hydrogens of the topology are not the ones named on the molecule")
        all_ncaa = set(ncaa_atom_map.keys()) | set(h_atoms.keys())

        print(f"  NCAA atoms: {len(ncaa_atom_map)} heavy + {len(h_atoms)} H = {len(all_ncaa)} total")

        # Detect cyclic NCAAs: sidechain atom bonded to backbone N that actually
        # closes a ring back to N through other atoms (e.g. PCA/pyroglutamic acid —
        # lactam ring N-CA-CB-CG-CD-N). The hybrid AMBER backbone + OpenFF sidechain
        # approach can't handle ring closures between backbone and sidechain atoms.
        # A \u03b2-amino acid (N-CB-CA-C backbone) also has N bonded to a non-standard
        # atom (CB), but there is no ring closure — so we verify by checking whether
        # the candidate atom has an alternate path back to N not using the direct bond.
        backbone_n_idx = None
        for idx, name in ncaa_atom_map.items():
            if name == 'N':
                backbone_n_idx = idx
                break
        if backbone_n_idx is not None:
            sidechain_atoms = {idx for idx in ncaa_atom_map if ncaa_atom_map[idx] not in BACKBONE_NAMES}
            n_neighbors = set(adj.get(backbone_n_idx, []))
            candidate_bonds = n_neighbors & sidechain_atoms
            cyclic_bond = set()
            for nb in candidate_bonds:
                visited = {nb}
                stack = [x for x in adj.get(nb, []) if x != backbone_n_idx]
                for x in stack:
                    visited.add(x)
                reaches_n = False
                while stack:
                    cur = stack.pop()
                    if cur == backbone_n_idx:
                        reaches_n = True
                        break
                    for nxt in adj.get(cur, []):
                        if nxt not in visited:
                            visited.add(nxt)
                            stack.append(nxt)
                if reaches_n:
                    cyclic_bond.add(nb)
            if cyclic_bond:
                sc_name = ncaa_atom_map.get(list(cyclic_bond)[0], '?')
                print(f"  Cyclic NCAA detected: backbone N bonded to sidechain {sc_name}")
                fail(resname, f"cyclic: backbone N bonded to side-chain atom {sc_name}")
                continue

        # Assign types and charges
        type_map, charge_map, prefix = assign_types_and_charges(
            ncaa_atom_map, h_atoms, h_names, top, resname, formal_charge, mol_rdkit)

        # Collect new parameters
        atomtypes, bonds, angles, dihedrals, impropers = collect_new_params(
            top, all_ncaa, type_map, prefix)
        all_atomtypes.update(atomtypes)
        all_bonds.update(bonds)
        all_angles.update(angles)
        if CB_AMINE_H_TYPE in type_map.values():
            all_angles.update(HP_CB_ANGLE)
        all_dihedrals.extend(dihedrals)
        all_impropers.extend(impropers)

        print(f"  New atom types: {len(atomtypes)}")
        print(f"  New bonded params: {len(bonds)} bonds, {len(angles)} angles, "
              f"{len(dihedrals)} dihedrals, {len(impropers)} impropers")

        # Generate RTP and HDB
        rtp = generate_rtp(resname, ncaa_atom_map, h_atoms, h_names, ncaa_atom_names,
                           type_map, charge_map, top, all_ncaa, adj)
        hdb = generate_hdb(resname, ncaa_atom_names, h_atoms, h_names, ncaa_atom_map, adj, top)
        if resname in GLYCAN_DEFS:
            rtp = fused_rtp_comment(resname, formal_charge) + rtp
        all_rtp.append(rtp)
        all_hdb.append(hdb)

    except Exception as e:
        print(f"\n  WARNING: Failed to parametrize {resname}: {e}")
        fail(resname, e)
        continue

if failed_ncaas:
    print(f"\nNCAAs that will be replaced by parent residue: {', '.join(failed_ncaas)}")
    # Write failed NCAAs to file so job.run can re-run fix_pdb.py without keeping them
    # (must write even if some other NCAAs succeeded, otherwise the failed ones remain
    #  in the PDB and pdb2gmx fails)
    with open(NCAA_FAILED, 'w') as f:
        for rn in failed_ncaas:
            f.write(f"{rn}\n")
if failed_glycans:
    print(f"\nFused glycan residues that go back to the ligands: {', '.join(failed_glycans)}")
    with open(GLYCAN_FAILED, 'w') as f:
        for rn, why in failed_glycans.items():
            f.write(f"{rn} {why}\n")

if not all_rtp:
    print("No NCAA residues were successfully parametrized.")
    sys.exit(0)

# ---- Create local force field copy ----
gmx_ff_src = None
import subprocess as _sp
_gmx_search_dirs = ['/usr/local/gromacs/share/gromacs/top', '/usr/share/gromacs/top']
try:
    _gmx_ver = _sp.run(['gmx', '--version'], capture_output=True, text=True, timeout=10)
    for _line in _gmx_ver.stdout.split('\n'):
        if 'Data prefix' in _line:
            _gmx_search_dirs.insert(0, _line.split()[-1] + '/share/gromacs/top')
            break
except Exception:
    pass
# Also search GMXLIB (contains custom FFs like gromos54a8.ff from GroScore)
for _gmxlib_dir in os.environ.get('GMXLIB', '').split(':'):
    if _gmxlib_dir:
        _gmx_search_dirs.append(_gmxlib_dir)
for search_dir in _gmx_search_dirs:
    candidate = os.path.join(search_dir, f'{args.gmx_ff}.ff')
    if os.path.isdir(candidate):
        gmx_ff_src = candidate
        break

if gmx_ff_src is None:
    print(f"Error: Could not find {args.gmx_ff}.ff in GROMACS installation or GMXLIB", file=sys.stderr)
    sys.exit(1)

local_ff = f'{args.gmx_ff}.ff'
if not os.path.exists(local_ff):
    shutil.copytree(gmx_ff_src, local_ff)
    print(f"\nCopied {gmx_ff_src} -> {local_ff}")
else:
    print(f"\nUsing existing local FF: {local_ff}")

# Write NCAA atom types
at_path = os.path.join(local_ff, 'ncaa_atomtypes.itp')
write_ncaa_atomtypes(at_path, all_atomtypes)
print(f"Wrote {at_path} ({len(all_atomtypes)} atom types)")

# Write NCAA bonded parameters
bp_path = os.path.join(local_ff, 'ncaa_bonded.itp')
write_ncaa_bonded(bp_path, all_bonds, all_angles, all_dihedrals, all_impropers)
print(f"Wrote {bp_path}")

# Modify forcefield.itp to include new files
ff_itp_path = os.path.join(local_ff, 'forcefield.itp')
with open(ff_itp_path) as f:
    ff_content = f.read()

if 'ncaa_atomtypes.itp' not in ff_content:
    # Insert after ffnonbonded.itp include
    ff_content = ff_content.replace(
        '#include "ffnonbonded.itp"',
        '#include "ffnonbonded.itp"\n#include "ncaa_atomtypes.itp"'
    )
    # Insert after ffbonded.itp include
    ff_content = ff_content.replace(
        '#include "ffbonded.itp"',
        '#include "ffbonded.itp"\n#include "ncaa_bonded.itp"'
    )
    with open(ff_itp_path, 'w') as f:
        f.write(ff_content)
    print(f"Updated {ff_itp_path} with NCAA includes")

# Append new atom types to atomtypes.atp (pdb2gmx reads this file for type validation)
atp_path = os.path.join(local_ff, 'atomtypes.atp')
with open(atp_path) as f:
    atp_content = f.read()
with open(atp_path, 'a') as f:
    for type_name, at in sorted(all_atomtypes.items()):
        if type_name not in set(line.split()[0] for line in atp_content.split('\n') if line.strip() and not line.startswith(';')):
            f.write(f"{type_name:<14s} {at['mass']:.2f}\n")
print(f"Appended {len(all_atomtypes)} types to {atp_path}")

# Write NCAA RTP entries
rtp_path = os.path.join(local_ff, 'ncaa.rtp')
with open(rtp_path, 'w') as f:
    f.write("; NCAA residue topology entries (generated by parametrize_ncaa.py)\n\n")
    f.write("[ bondedtypes ]\n")
    f.write("; bonds  angles  dihedrals  impropers all_dihedrals nrexcl HH14 RemoveDih\n")
    f.write("     1       1          9          4        1         3      1     0\n\n")
    for rtp in all_rtp:
        f.write(rtp + '\n\n')
print(f"Wrote {rtp_path}")

# Append HDB entries to aminoacids.hdb, each in place of an entry of the same name
hdb_path = os.path.join(local_ff, 'aminoacids.hdb')
with open(hdb_path) as f:
    hdb_content = f.read()
hdb_content, hdb_replaced = merge_hdb(hdb_content, all_hdb)
with open(hdb_path, 'w') as f:
    f.write(hdb_content)
for hdb in all_hdb:
    resname = hdb.split(None, 1)[0]
    print(f"{'Replaced' if resname in hdb_replaced else 'Appended'} {resname} HDB entry in {hdb_path}")

# Write output info
with open('ncaa_residues.gs', 'w') as f:
    f.write("# Non-standard amino acid residues parametrized with OpenFF\n")
    for resname in ncaa_types:
        f.write(f"{resname}\n")

# Create/update residuetypes.dat in current directory (pdb2gmx needs NCAA listed as Protein)
# GROMACS searches GMXLIB for this file
restypes_src = os.path.join(os.path.dirname(gmx_ff_src), 'residuetypes.dat')
restypes_local = 'residuetypes.dat'
if os.path.isfile(restypes_src) and not os.path.isfile(restypes_local):
    shutil.copy2(restypes_src, restypes_local)
if os.path.isfile(restypes_local):
    with open(restypes_local) as f:
        rt_content = f.read()
    existing = set(line.split()[0] for line in rt_content.split('\n') if line.strip() and not line.startswith(';'))
    with open(restypes_local, 'a') as f:
        for resname in ncaa_types:
            if resname not in existing:
                f.write(f"{resname}\tProtein\n")
    print(f"Updated {restypes_local} with NCAA residues as Protein type")

# Add CMAP entries for NCAA residues (copy from parent amino acid)
# AMBER19SB uses residue-specific CMAP types like N-TRP, XC-TRP, C-TRP
cmap_path = os.path.join(local_ff, 'cmap.itp')
if os.path.isfile(cmap_path):
    with open(cmap_path) as f:
        cmap_content = f.read()

    cmap_additions = []
    for resname in ncaa_types:
        if f'N-{resname}' in cmap_content:
            continue  # already has CMAP
        if resname not in parents:
            continue  # failed above (ncaa_failed.gs)

        # The parent residue from its CCD entry (read before the parametrisation)
        parent = parents[resname]
        print(f"NCAA {resname}: parent residue = {parent}")

        # Copy parent's CMAP entry with NCAA residue name
        # AMBER19SB uses protonation-specific names that differ from RCSB CCD parent:
        # HIS → HID/HIE/HIP, CYS → CYX/CYM, ASP → ASH, GLU → GLH, LYS → LYN
        # Try exact parent first, then AMBER-specific fallback names
        AMBER_PARENT_FALLBACKS = {
            'HIS': ['HID', 'HIE', 'HIP'],
            'CYS': ['CYX', 'CYM'],
            'ASP': ['ASH'],
            'GLU': ['GLH'],
            'LYS': ['LYN'],
        }
        cmap_parent = parent
        parent_marker = f'N-{cmap_parent} XC-{cmap_parent} C-{cmap_parent}'
        if parent_marker not in cmap_content and parent in AMBER_PARENT_FALLBACKS:
            for fallback in AMBER_PARENT_FALLBACKS[parent]:
                candidate = f'N-{fallback} XC-{fallback} C-{fallback}'
                if candidate in cmap_content:
                    cmap_parent = fallback
                    parent_marker = candidate
                    break
        if parent_marker in cmap_content:
            # Extract the full CMAP block for parent
            idx = cmap_content.index(parent_marker)
            # Find the line start
            line_start = cmap_content.rfind('\n', 0, idx) + 1
            # Find the end (next cmaptypes entry or end of section)
            rest = cmap_content[line_start:]
            # The CMAP data block continues with backslash-terminated lines
            block_lines = []
            for line in rest.split('\n'):
                block_lines.append(line)
                if line.strip() and not line.rstrip().endswith('\\'):
                    break
            parent_block = '\n'.join(block_lines)
            # Replace parent name with NCAA name
            ncaa_block = parent_block.replace(f'N-{cmap_parent}', f'N-{resname}')
            ncaa_block = ncaa_block.replace(f'XC-{cmap_parent}', f'XC-{resname}')
            ncaa_block = ncaa_block.replace(f'C-{cmap_parent}', f'C-{resname}')
            cmap_additions.append(ncaa_block)
            src = f"{cmap_parent} (AMBER name for {parent})" if cmap_parent != parent else parent
            print(f"Added CMAP for {resname} (copied from {src})")

    if cmap_additions:
        with open(cmap_path, 'a') as f:
            f.write('\n\n; NCAA CMAP entries\n')
            for block in cmap_additions:
                f.write(block + '\n')

print(f"\nWrote ncaa_residues.gs")
print(f"Local force field ready: {local_ff}")
print("Set pdb2gmx -ff flag to use local FF directory")
