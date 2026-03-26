#!/usr/bin/env python3
#
# merge_crystal_waters.py - Merge crystal waters into GROMACS topology/coordinates
#
# Crystal waters from the PDB are added as SOL molecules, compatible with all
# force fields' water models. They are placed after protein+ligand+ion atoms
# in the GRO file and added to [ molecules ] in topol.top.
#
# Usage: python merge_crystal_waters.py -t topol.top -c conf.gro
#

import os
import sys
import re
import argparse

parser = argparse.ArgumentParser(description="Merge crystal waters into GROMACS system.")
parser.add_argument('-t', '--topol', type=str, default="topol.top", help="Topology file")
parser.add_argument('-c', '--conf', type=str, required=True, help="Coordinate file (GRO)")
args = parser.parse_args()

water_pdb = "crystal_waters.pdb"
if not os.path.isfile(water_pdb):
    print("No crystal_waters.pdb found, skipping crystal water merge.")
    sys.exit(0)

# Read crystal water coordinates from PDB
# Each HOH has 1 oxygen (OW) — hydrogens will be added by GROMACS
water_coords = []  # [(x_nm, y_nm, z_nm), ...]
with open(water_pdb) as f:
    for line in f:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atomname = line[12:16].strip()
            # Only take the oxygen atom (skip any H if present)
            if atomname in ("O", "OW"):
                x = float(line[30:38]) / 10.0  # Å to nm
                y = float(line[38:46]) / 10.0
                z = float(line[46:54]) / 10.0
                water_coords.append((x, y, z))

if not water_coords:
    print("No water oxygen atoms found in crystal_waters.pdb, skipping.")
    sys.exit(0)

n_waters = len(water_coords)
print(f"Found {n_waters} crystal water(s)")

# ---- Merge coordinates ----
with open(args.conf) as f:
    gro_lines = f.readlines()

title = gro_lines[0]
n_existing = int(gro_lines[1].strip())
existing_atoms = gro_lines[2:2 + n_existing]
box_line = gro_lines[2 + n_existing]

# Find max residue number and atom number
max_resnum = 0
max_atomnum = n_existing
for line in existing_atoms:
    m = re.match(r"\s*(\d+)", line)
    if m:
        r = int(m.group(1))
        if r > max_resnum:
            max_resnum = r

# Write water atoms in GRO format: OW + HW1 + HW2 (3 atoms per SOL)
# Generate H positions at ideal geometry from O position
# OPC3: O-H = 0.08724 nm, H-O-H = 103.6 deg
import math
oh_dist = 0.08724  # nm
hoh_angle = 103.6 * math.pi / 180.0
# Place H1 and H2 relative to O (arbitrary orientation, will relax during emin)
dz = oh_dist * math.cos(hoh_angle / 2.0)
dy = oh_dist * math.sin(hoh_angle / 2.0)

water_atom_lines = []
next_resnum = max_resnum + 1
next_atomnum = max_atomnum + 1

for x, y, z in water_coords:
    # OW
    water_atom_lines.append(f"{next_resnum:5d}{'SOL':>5s}{'  OW':5s}{next_atomnum:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
    next_atomnum += 1
    # HW1
    water_atom_lines.append(f"{next_resnum:5d}{'SOL':>5s}{' HW1':5s}{next_atomnum:5d}{x:8.3f}{y + dy:8.3f}{z + dz:8.3f}\n")
    next_atomnum += 1
    # HW2
    water_atom_lines.append(f"{next_resnum:5d}{'SOL':>5s}{' HW2':5s}{next_atomnum:5d}{x:8.3f}{y - dy:8.3f}{z + dz:8.3f}\n")
    next_atomnum += 1
    next_resnum += 1

total_atoms = n_existing + len(water_atom_lines)
with open(args.conf, "w") as f:
    f.write(title)
    f.write(f"{total_atoms:5d}\n")
    for line in existing_atoms:
        f.write(line)
    for line in water_atom_lines:
        f.write(line)
    f.write(box_line)

print(f"Added {n_waters} crystal water(s) to {args.conf} ({total_atoms} total atoms)")

# ---- Add SOL to topology [ molecules ] ----
with open(args.topol) as f:
    topol_lines = f.readlines()

# Find the end of [ molecules ] section (before any [ intermolecular_interactions ])
# or end of file
insert_idx = len(topol_lines)
for i, line in enumerate(topol_lines):
    if "intermolecular_interactions" in line:
        # Insert before this section
        # Go back to skip comment lines
        insert_idx = i
        while insert_idx > 0 and topol_lines[insert_idx - 1].strip().startswith(";"):
            insert_idx -= 1
        break

# Check if SOL already exists in [ molecules ]
sol_exists = False
for line in topol_lines:
    if line.strip().startswith("SOL"):
        sol_exists = True
        break

if sol_exists:
    # Update existing SOL count
    new_lines = []
    for line in topol_lines:
        if line.strip().startswith("SOL"):
            parts = line.split()
            old_count = int(parts[1])
            new_count = old_count + n_waters
            new_lines.append(f"{'SOL':20s} {new_count}\n")
        else:
            new_lines.append(line)
    with open(args.topol, "w") as f:
        f.writelines(new_lines)
    print(f"Updated SOL count in {args.topol}")
else:
    # Insert new SOL entry
    topol_lines.insert(insert_idx, f"{'SOL':20s} {n_waters}\n")
    with open(args.topol, "w") as f:
        f.writelines(topol_lines)
    print(f"Added SOL entry ({n_waters}) to {args.topol}")
