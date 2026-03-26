#!/usr/bin/env python3
#
# make_ion_restraints.py - Generate topology-level coordination restraints for structural ions
#
# Detects ion-protein coordination pairs from the initial structure and appends
# [ intermolecular_interactions ] harmonic bond restraints to topol.top.
# These restraints are active from the first energy minimization onwards,
# ensuring coordination geometry is maintained throughout the simulation.
#
# Usage: python make_ion_restraints.py -f conf_cutout.gro -m chain_map.gs
#

import os
import sys
import re
import argparse
import numpy as np
from scipy.spatial.distance import cdist

# Ion residue names supported by all GroScore force fields
ION_RESIDUES = {"ZN", "CA", "MG", "CU", "CU1", "NA", "CL"}

# Coordination detection cutoff (nm)
# Captures bonds at 0.20-0.24 nm with margin; excludes non-coordinating
# HIS nitrogen at ~0.31-0.33 nm from Zn
COORD_CUTOFF = 0.3

# Harmonic restraint force constant (kJ/mol/nm²)
# Strong enough to maintain coordination, weaker than covalent bonds (~250000)
COORD_K = 10000.0

parser = argparse.ArgumentParser(description="Generate ion coordination restraints for topology.")
parser.add_argument('-f', '--input', type=str, required=True, help="Input coordinate file (GRO format)")
parser.add_argument('-m', '--chainmap', type=str, required=True, help="Chain map file (chain_map.gs)")
args = parser.parse_args()

# Read ion residue numbers
ion_residues = set()
ion_map_path = os.path.join(os.path.dirname(args.chainmap), "ion_residues.gs")
if os.path.isfile(ion_map_path):
    with open(ion_map_path) as f:
        for line in f:
            if not line.strip().startswith("#"):
                try:
                    ion_residues.add(int(line.strip()))
                except (ValueError, IndexError):
                    pass

if not ion_residues:
    print("No structural ions found, skipping coordination restraints.")
    sys.exit(0)

# Parse GRO file - collect ion atoms and protein heavy atoms
ion_atoms = []   # [(atomnum, resname_field, atomname, x, y, z), ...]
prot_atoms = []  # [(atomnum, resname_field, atomname, x, y, z), ...]

if not os.path.isfile(args.input):
    print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
    sys.exit(1)

with open(args.input) as f:
    for line in f:
        if not line.strip().startswith("#"):
            left = line[:15]
            right = line[15:]
            tmp = left.split() + right.split()
            try:
                s = re.search(r"\d+(\.\d+)?", tmp[0])
                resnum = int(s.group(0))
                atomname = tmp[1]
                atomnum = int(tmp[2])
                x, y, z = float(tmp[3]), float(tmp[4]), float(tmp[5])

                # Extract residue name (strip digits from GRO resname field)
                res3 = re.sub(r'\d+', '', tmp[0])

                # Skip solvent
                if res3 == "SOL":
                    continue

                # Collect ion atoms and coordinating protein atoms (S, N, O only)
                if resnum in ion_residues:
                    ion_atoms.append((atomnum, tmp[0], atomname, x, y, z))
                elif atomname[0] in ("S", "N", "O"):
                    prot_atoms.append((atomnum, tmp[0], atomname, x, y, z))
            except (ValueError, IndexError, AttributeError):
                pass

if not ion_atoms:
    print("No ion atoms found in coordinate file, skipping coordination restraints.")
    sys.exit(0)

if not prot_atoms:
    print("No protein atoms found in coordinate file, skipping coordination restraints.")
    sys.exit(0)

# Calculate distances between all ion atoms and protein heavy atoms
ion_coords = np.array([(a[3], a[4], a[5]) for a in ion_atoms], dtype=np.float64)
prot_coords = np.array([(a[3], a[4], a[5]) for a in prot_atoms], dtype=np.float64)

distances = cdist(ion_coords, prot_coords)

# Find coordination pairs within cutoff
coord_pairs = []  # [(ion_atomnum, prot_atomnum, distance, ion_name, prot_resname, prot_atomname), ...]

for i in range(len(ion_atoms)):
    for j in range(len(prot_atoms)):
        dist = distances[i, j]
        if dist <= COORD_CUTOFF:
            ion_atomnum = ion_atoms[i][0]
            ion_resname = ion_atoms[i][1]
            ion_name = ion_atoms[i][2]
            prot_atomnum = prot_atoms[j][0]
            prot_resname = prot_atoms[j][1]
            prot_atomname = prot_atoms[j][2]
            coord_pairs.append((ion_atomnum, prot_atomnum, dist,
                                ion_name, prot_resname, prot_atomname))

if not coord_pairs:
    print("No ion coordination pairs detected within cutoff, skipping restraints.")
    sys.exit(0)

# Append [ intermolecular_interactions ] to topol.top
topol_path = "topol.top"
if not os.path.isfile(topol_path):
    print(f"Error: {topol_path} not found.", file=sys.stderr)
    sys.exit(1)

with open(topol_path, "a") as f:
    f.write("\n; Ion coordination restraints (auto-detected, cutoff={:.2f} nm, k={:.0f} kJ/mol/nm^2)\n".format(
        COORD_CUTOFF, COORD_K))
    f.write("[ intermolecular_interactions ]\n")
    f.write("[ bonds ]\n")
    f.write("; ai      aj  funct      b0(nm)      kb(kJ/mol/nm^2)  ; description\n")
    for ion_num, prot_num, dist, ion_name, prot_resname, prot_atomname in coord_pairs:
        f.write(f"  {ion_num:6d}  {prot_num:6d}      6  {dist:12.6f}  {COORD_K:14.2f}  "
                f"; {ion_name}-{prot_atomname}({prot_resname})\n")

print(f"Generated {len(coord_pairs)} ion coordination restraint(s):")
for ion_num, prot_num, dist, ion_name, prot_resname, prot_atomname in coord_pairs:
    print(f"  {ion_name} (atom {ion_num}) - {prot_atomname} of {prot_resname} (atom {prot_num}): {dist:.3f} nm")
