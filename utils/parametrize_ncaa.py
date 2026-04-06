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

import os
import sys
import argparse
import tempfile
import shutil
import numpy as np

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
        line = (f"ATOM  {serial:5d} {name_fmt} {resn:>3s} {ch}{rn:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {elem:>2s}")
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


# ---- OpenBabel / RDKit / OpenFF parametrization ----

def parametrize_capped(pdb_text, ncaa_heavy_coords, ncaa_resname, ff_name, ph):
    """Parametrize capped tripeptide with OpenFF. Returns parsed topology data.

    Strategy 1: OpenBabel bond perception (works for simple NCAAs)
    Strategy 2: RCSB template for NCAA + manual ACE/NME capping
    """
    from openbabel import openbabel as ob
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from openff.toolkit import Molecule, ForceField, Topology
    from openff.interchange import Interchange
    from openff.units import unit
    import urllib.request

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

    # Strategy 2: RCSB template for NCAA + manual ACE/NME capping
    if mol_rdkit is None:
        print("  Strategy 2: RCSB template + manual capping...")
        try:
            # Download NCAA ideal SDF
            url = f"https://files.rcsb.org/ligands/download/{ncaa_resname}_ideal.sdf"
            sdf_data = urllib.request.urlopen(url, timeout=15).read().decode()
            template_h = Chem.MolFromMolBlock(sdf_data, removeHs=False, sanitize=True)
            template = Chem.RemoveHs(template_h) if template_h else None

            if template is None:
                print(f"  Error: RCSB template for {ncaa_resname} failed", file=sys.stderr)
                sys.exit(1)

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
                print("  Error: RDKit could not parse NCAA PDB block", file=sys.stderr)
                sys.exit(1)

            # Assign bond orders from template
            try:
                mol_assigned = AllChem.AssignBondOrdersFromTemplate(template_clean, ncaa_pdb_mol)
                print(f"  Bond orders assigned from RCSB template")
            except Exception as e:
                print(f"  Template matching failed ({e})")
                # Fallback: use template directly with ideal coords
                # Assign PDB atom names from CCD CIF (SDF doesn't preserve them)
                mol_assigned = template_clean
                try:
                    cif_url = f"https://files.rcsb.org/ligands/download/{ncaa_resname}.cif"
                    cif_data = urllib.request.urlopen(cif_url, timeout=10).read().decode()
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
                print(f"  Error: backbone N={n_idx} C={c_idx} not found", file=sys.stderr)
                sys.exit(1)

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
            mol_rdkit = Chem.AddHs(mol_capped, addCoords=True)
            formal_charge = Chem.GetFormalCharge(mol_rdkit)
            print(f"  Capped molecule: {mol_rdkit.GetNumAtoms()} atoms, charge={formal_charge}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Error: RCSB template approach failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Strip PDB metadata (OpenFF requires clean mol without residue info)
    sdf_clean = Chem.MolToMolBlock(mol_rdkit)
    mol_rdkit = Chem.MolFromMolBlock(sdf_clean, removeHs=False, sanitize=True)
    if mol_rdkit is None:
        print("Error: final SDF round-trip failed", file=sys.stderr)
        sys.exit(1)
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

    return top_content, gro_content, mol_rdkit, formal_charge


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
    """Match GRO atom indices to NCAA PDB atom names by coordinate proximity.
    PDB coords are in Angstrom, GRO in nm.
    Falls back to atom-name matching from GRO residue names if coordinate matching fails
    (e.g., when ideal template coordinates were used instead of PDB coordinates)."""
    mapping = {}
    for gro_idx, gc in enumerate(gro_coords):
        for name, pc in ncaa_heavy_coords.items():
            if name in mapping.values():
                continue
            if np.linalg.norm(gc - pc / 10.0) < 0.005:  # 0.05 Angstrom
                mapping[gro_idx] = name
                break

    # Fallback: match by atom name from GRO file if coordinate matching failed
    if len(mapping) == 0 and gro_content is not None:
        gro_lines = gro_content.strip().split('\n')
        n = int(gro_lines[1].strip())
        ncaa_names_set = set(ncaa_heavy_coords.keys())
        for i, line in enumerate(gro_lines[2:2+n]):
            if len(line) < 15:
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


def assign_types_and_charges(ncaa_atom_map, h_atoms, h_names, top, resname, formal_charge):
    """Assign atom types and charges for NCAA atoms.

    Backbone: AMBER19SB types and fixed charges (N,H,C,O)
    CA, HA: AMBER19SB types, OpenFF charges
    CB: AMBER19SB type CT, OpenFF charge
    Sidechain: OpenFF types (prefixed), OpenFF charges
    """
    prefix = resname + '_'
    all_ncaa = set(ncaa_atom_map.keys()) | set(h_atoms.keys())
    type_map = {}
    charge_map = {}

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
                type_map[idx] = CB_H_TYPE
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
    # Sidechain impropers from OpenFF
    for entry in top['impropers']:
        ai, aj, ak, al = entry[:4]
        if all(i in idx_set for i in (ai, aj, ak, al)):
            names = [ncaa_atom_map.get(i, h_names.get(i)) for i in (ai, aj, ak, al)]
            if all(names):
                # Skip pure backbone impropers (already listed)
                if not all(n in BACKBONE_NAMES for n in names):
                    lines.append(f' {names[0]:<5s} {names[1]:<5s} {names[2]:<5s} {names[3]}')

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

pdb_atoms = parse_pdb(args.file)
ncaa_types = find_ncaas(pdb_atoms)

if not ncaa_types:
    print("No NCAA residues found in the input PDB.")
    sys.exit(0)

print(f"Found NCAA residues: {', '.join(f'{rn} ({len(inst)} instance(s))' for rn, inst in ncaa_types.items())}")

# Accumulate across all NCAA types
all_atomtypes = {}
all_bonds = {}
all_angles = {}
all_dihedrals = []
all_impropers = []
all_rtp = []
all_hdb = []

failed_ncaas = []  # NCAAs that couldn't be parametrized → fall back to PDBFixer replacement

for resname, instances in ncaa_types.items():
    chain, resnum = instances[0]
    print(f"\n=== Parametrizing {resname} (chain {chain}, residue {resnum}) ===")

    try:
        # Build capped tripeptide
        pdb_text, ncaa_heavy_coords, ncaa_atom_names = build_capped_pdb(pdb_atoms, chain, resnum)

        # Parametrize with OpenFF
        top_content, gro_content, mol_rdkit, formal_charge = parametrize_capped(
            pdb_text, ncaa_heavy_coords, resname, args.ff, args.ph)

        # Parse topology and coordinates
        top = parse_top(top_content)
        gro_coords = parse_gro_coords(gro_content)

        # Match NCAA atoms by coordinates
        ncaa_atom_map = match_ncaa_atoms(gro_coords, ncaa_heavy_coords, gro_content)
        print(f"  Matched {len(ncaa_atom_map)}/{len(ncaa_heavy_coords)} NCAA heavy atoms")
        for name in ncaa_heavy_coords:
            if name not in ncaa_atom_map.values():
                print(f"    WARNING: unmatched atom {name}")

        # Build adjacency and find H atoms
        adj = build_adjacency(top)
        h_atoms = find_ncaa_h_atoms(adj, ncaa_atom_map, top)
        h_names = assign_h_names(h_atoms)
        all_ncaa = set(ncaa_atom_map.keys()) | set(h_atoms.keys())

        print(f"  NCAA atoms: {len(ncaa_atom_map)} heavy + {len(h_atoms)} H = {len(all_ncaa)} total")

        # Detect cyclic NCAAs: sidechain atom bonded to backbone N
        # (e.g. PCA/pyroglutamic acid — lactam ring connecting CD to N)
        # The hybrid AMBER backbone + OpenFF sidechain approach can't handle
        # ring closures between backbone and sidechain atoms.
        backbone_n_idx = None
        for idx, name in ncaa_atom_map.items():
            if name == 'N':
                backbone_n_idx = idx
                break
        if backbone_n_idx is not None:
            sidechain_atoms = {idx for idx in ncaa_atom_map if ncaa_atom_map[idx] not in BACKBONE_NAMES}
            n_neighbors = set(adj.get(backbone_n_idx, []))
            cyclic_bond = n_neighbors & sidechain_atoms
            if cyclic_bond:
                sc_name = ncaa_atom_map.get(list(cyclic_bond)[0], '?')
                print(f"  Cyclic NCAA detected: backbone N bonded to sidechain {sc_name}")
                print(f"  Will fall back to PDBFixer parent-residue replacement for {resname}")
                failed_ncaas.append(resname)
                continue

        # Assign types and charges
        type_map, charge_map, prefix = assign_types_and_charges(
            ncaa_atom_map, h_atoms, h_names, top, resname, formal_charge)

        # Collect new parameters
        atomtypes, bonds, angles, dihedrals, impropers = collect_new_params(
            top, all_ncaa, type_map, prefix)
        all_atomtypes.update(atomtypes)
        all_bonds.update(bonds)
        all_angles.update(angles)
        all_dihedrals.extend(dihedrals)
        all_impropers.extend(impropers)

        print(f"  New atom types: {len(atomtypes)}")
        print(f"  New bonded params: {len(bonds)} bonds, {len(angles)} angles, "
              f"{len(dihedrals)} dihedrals, {len(impropers)} impropers")

        # Generate RTP and HDB
        rtp = generate_rtp(resname, ncaa_atom_map, h_atoms, h_names, ncaa_atom_names,
                           type_map, charge_map, top, all_ncaa, adj)
        hdb = generate_hdb(resname, ncaa_atom_names, h_atoms, h_names, ncaa_atom_map, adj, top)
        all_rtp.append(rtp)
        all_hdb.append(hdb)

    except Exception as e:
        print(f"\n  WARNING: Failed to parametrize {resname}: {e}")
        print(f"  Will fall back to PDBFixer parent-residue replacement for {resname}")
        failed_ncaas.append(resname)
        continue

if failed_ncaas:
    print(f"\nNCAAs that will be replaced by parent residue: {', '.join(failed_ncaas)}")

if not all_rtp:
    print("No NCAA residues were successfully parametrized.")
    # Write failed NCAAs to file so fix_pdb.py can handle them
    if failed_ncaas:
        with open('ncaa_failed.gs', 'w') as f:
            for rn in failed_ncaas:
                f.write(f"{rn}\n")
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
for search_dir in _gmx_search_dirs:
    candidate = os.path.join(search_dir, f'{args.gmx_ff}.ff')
    if os.path.isdir(candidate):
        gmx_ff_src = candidate
        break

if gmx_ff_src is None:
    print(f"Error: Could not find {args.gmx_ff}.ff in GROMACS installation", file=sys.stderr)
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

# Append HDB entries to aminoacids.hdb
hdb_path = os.path.join(local_ff, 'aminoacids.hdb')
with open(hdb_path) as f:
    hdb_content = f.read()

for hdb in all_hdb:
    resname = hdb.split('\t')[0]
    if resname not in hdb_content:
        with open(hdb_path, 'a') as f:
            f.write('\n' + hdb + '\n')
        print(f"Appended {resname} HDB entry to {hdb_path}")

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
    import urllib.request as _urlreq
    with open(cmap_path) as f:
        cmap_content = f.read()

    cmap_additions = []
    for resname in ncaa_types:
        if f'N-{resname}' in cmap_content:
            continue  # already has CMAP

        # Determine parent residue from RCSB CCD
        parent = 'ALA'
        try:
            url = f"https://files.rcsb.org/ligands/download/{resname}.cif"
            cif_data = _urlreq.urlopen(url, timeout=15).read().decode()
            for line in cif_data.split('\n'):
                if '_chem_comp.mon_nstd_parent_comp_id' in line:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1] != '?':
                        parent = parts[1]
                        break
        except Exception:
            pass
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
