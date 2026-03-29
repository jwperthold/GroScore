#!/usr/bin/env python3
#
# fix_ion_protonation.py - Set protonation states for ion-coordinating residues
#
# Metal-coordinating residues need specific protonation:
#   CYS → deprotonated thiolate (lone pair on S coordinates the metal)
#   HIS → the coordinating N must be deprotonated (lone pair available)
#         ND1 coordinates → Nε protonated
#         NE2 coordinates → Nδ protonated
#
# Usage: python fix_ion_protonation.py -f fixed_capped.pdb -m ion_residues.gs --ff amber19sb
#

import argparse
import math

COORD_CUTOFF = 3.0  # Angstrom — typical metal-ligand coordination distance

# Force-field-specific residue names for protonation states
FF_NAMES = {
    'amber19sb': {
        'cys_deprot': 'CYM',    # deprotonated cysteine (thiolate)
        'his_nd_prot': 'HID',   # Nδ protonated (NE2 free for coordination)
        'his_ne_prot': 'HIE',   # Nε protonated (ND1 free for coordination)
    },
    'charmm36': {
        'cys_deprot': 'CYM',
        'his_nd_prot': 'HSD',   # Nδ protonated
        'his_ne_prot': 'HSE',   # Nε protonated
    },
    'gromos54a8': {
        'cys_deprot': 'CYS',    # GROMOS: CYS = deprotonated, CYSH = protonated
        'his_nd_prot': 'HISA',  # Nδ protonated
        'his_ne_prot': 'HISB',  # Nε protonated
    },
}

# All HIS variant names across force fields
ALL_HIS = {'HIS', 'HIE', 'HID', 'HIP', 'HSD', 'HSE', 'HSP', 'HISA', 'HISB', 'HISH', 'HISD', 'HISE', 'HISP'}
ALL_CYS = {'CYS', 'CYSH', 'CYM', 'CYX'}

parser = argparse.ArgumentParser(description="Fix protonation states for ion-coordinating residues.")
parser.add_argument('-f', '--file', type=str, required=True, help="Input PDB file (modified in-place)")
parser.add_argument('-m', '--ionres', type=str, default='ion_residues.gs', help="Ion residues file")
parser.add_argument('--ff', type=str, default='amber19sb', help="Force field name")
args = parser.parse_args()

# Determine FF naming
ff_base = args.ff.replace('_opc', '').replace('_opc3', '')
if ff_base not in FF_NAMES:
    print(f"Warning: unknown force field '{ff_base}', using AMBER naming")
    ff_base = 'amber19sb'
names = FF_NAMES[ff_base]

# Read ion residue numbers
import os
ion_resnums = set()
if os.path.isfile(args.ionres):
    with open(args.ionres) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                try:
                    ion_resnums.add(int(s))
                except ValueError:
                    pass

if not ion_resnums:
    # No structural ions — nothing to fix
    exit(0)

# Parse PDB atoms
atoms = []
with open(args.file) as f:
    for line in f:
        if line.startswith(('ATOM', 'HETATM')) and len(line) >= 54:
            atoms.append({
                'name': line[12:16].strip(),
                'resname': line[17:20].strip(),
                'resnum': int(line[22:26]),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
            })

# Collect ion coordinates
ion_coords = [(a['x'], a['y'], a['z']) for a in atoms if a['resnum'] in ion_resnums]

if not ion_coords:
    exit(0)

def min_ion_dist(x, y, z):
    return min(math.sqrt((x-ix)**2 + (y-iy)**2 + (z-iz)**2) for ix, iy, iz in ion_coords)

# Detect coordinating CYS and HIS
rename_map = {}  # resnum -> new_resname

for a in atoms:
    rn = a['resname']
    resnum = a['resnum']

    # CYS: check SG distance to nearest ion
    if rn in ALL_CYS and a['name'] == 'SG' and resnum not in rename_map:
        dist = min_ion_dist(a['x'], a['y'], a['z'])
        if dist < COORD_CUTOFF:
            rename_map[resnum] = names['cys_deprot']
            print(f"{rn} {resnum} → {names['cys_deprot']} (SG-ion: {dist:.2f} Å)")

    # HIS: check ND1 and NE2 distance to nearest ion
    elif rn in ALL_HIS and a['name'] in ('ND1', 'NE2') and resnum not in rename_map:
        dist = min_ion_dist(a['x'], a['y'], a['z'])
        if dist < COORD_CUTOFF:
            if a['name'] == 'ND1':
                # ND1 coordinates metal → ND1 must be deprotonated → protonate Nε
                rename_map[resnum] = names['his_ne_prot']
                print(f"{rn} {resnum} → {names['his_ne_prot']} (ND1-ion: {dist:.2f} Å)")
            else:
                # NE2 coordinates metal → NE2 must be deprotonated → protonate Nδ
                rename_map[resnum] = names['his_nd_prot']
                print(f"{rn} {resnum} → {names['his_nd_prot']} (NE2-ion: {dist:.2f} Å)")

if not rename_map:
    print("No ion-coordinating CYS/HIS residues found.")
    exit(0)

# Rewrite PDB with renamed residues
with open(args.file) as f:
    lines = f.readlines()

with open(args.file, 'w') as f:
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')) and len(line) >= 26:
            try:
                resnum = int(line[22:26])
                if resnum in rename_map:
                    new_name = rename_map[resnum]
                    # PDB columns 18-20 (0-indexed 17-19) = 3-char residue name
                    # Some FFs use 4-char names (HISA, HISB, CYSH) which need column 17 too
                    if len(new_name) <= 3:
                        line = line[:17] + f'{new_name:>3s}' + line[20:]
                    else:
                        line = line[:17] + f'{new_name:>4s}' + line[21:]
            except (ValueError, IndexError):
                pass
        f.write(line)

print(f"Updated {len(rename_map)} residue protonation state(s) in {args.file}")
