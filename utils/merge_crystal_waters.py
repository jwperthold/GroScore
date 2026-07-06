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
import math
import argparse

parser = argparse.ArgumentParser(description="Merge crystal waters into GROMACS system.")
parser.add_argument('-t', '--topol', type=str, default="topol.top", help="Topology file")
parser.add_argument('-c', '--conf', type=str, required=True, help="Coordinate file (GRO)")
parser.add_argument('--proximity', action='store_true',
                    help="Only include waters within 0.5 nm of protein atoms (for cutout systems)")
args = parser.parse_args()


def water_model_sites(topol_path):
    """Read the SOL water model layout from the water .itp #included in the topology.

    Returns (atom_names, m_dist):
      atom_names : ordered list of SOL atom names, e.g. ['OW','HW1','HW2'] or
                   ['OW','HW1','HW2','MW'] for a 4-point model.
      m_dist     : O-M distance (nm) of the 4-point M virtual site, or None for a
                   3-point model.

    4-point models (OPC/TIP4P) carry an M virtual site that MUST be present in the
    coordinate file, otherwise grompp aborts (topology has N*4 SOL atoms, the
    coordinates only N*3). m_dist is derived from the model's own geometry (the
    SETTLE O-H / H-H distances and the [ virtual_sites3 ] weight):

        d_OM = 2*a*sqrt(doH^2 - (dHH/2)^2)    ( = 0.01594 nm = 0.1594 A for OPC )

    and the site is placed on the H-O-H bisector at that distance from the oxygen.
    Falls back to a 3-site model if the water .itp cannot be resolved.
    """
    default = (['OW', 'HW1', 'HW2'], None)
    if not os.path.isfile(topol_path):
        return default
    topdir = os.path.dirname(os.path.abspath(topol_path))
    lines = open(topol_path).read().splitlines()
    itp_rel = None
    for i, l in enumerate(lines):
        if "water topology" in l.lower():
            for l2 in lines[i + 1:]:
                m = re.search(r'#include\s+"([^"]+)"', l2)
                if m:
                    itp_rel = m.group(1)
                    break
                if l2.strip().startswith('['):
                    break
            break
    if not itp_rel:
        return default
    itp_path = os.path.normpath(os.path.join(topdir, itp_rel))
    if not os.path.isfile(itp_path):
        return default
    names, section = [], None
    doh = dhh = vsite_a = None
    for line in open(itp_path):
        s = line.split(';')[0].strip()
        if not s:
            continue
        if s.startswith('['):
            section = s.strip('[]').strip().lower()
            continue
        p = s.split()
        if section == 'atoms' and len(p) >= 5:
            names.append(p[4])
        elif section == 'settles' and len(p) >= 4:
            doh, dhh = float(p[2]), float(p[3])
        elif section == 'virtual_sites3' and len(p) >= 7 and p[4] == '1':
            vsite_a = float(p[5])
    if not names:
        return default
    m_dist = None
    if len(names) >= 4 and None not in (vsite_a, doh, dhh):
        m_dist = 2.0 * vsite_a * math.sqrt(doh ** 2 - (dhh / 2.0) ** 2)
    return names, m_dist


PROXIMITY_CUTOFF = 0.5  # nm

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

print(f"Found {len(water_coords)} crystal water(s)")

# ---- Read existing GRO ----
with open(args.conf) as f:
    gro_lines = f.readlines()

title = gro_lines[0]
n_existing = int(gro_lines[1].strip())
existing_atoms = gro_lines[2:2 + n_existing]
box_line = gro_lines[2 + n_existing]

# ---- Proximity filter ----
if args.proximity:
    import numpy as np
    from scipy.spatial.distance import cdist

    prot_coords = []
    for line in existing_atoms:
        try:
            left = line[:15]
            right = line[15:]
            tmp = left.split() + right.split()
            x, y, z = float(tmp[3]), float(tmp[4]), float(tmp[5])
            prot_coords.append((x, y, z))
        except (ValueError, IndexError):
            pass

    if prot_coords:
        dists = cdist(np.array(water_coords), np.array(prot_coords))
        min_dists = np.min(dists, axis=1)
        kept = [c for i, c in enumerate(water_coords) if min_dists[i] <= PROXIMITY_CUTOFF]
        print(f"Proximity filter ({PROXIMITY_CUTOFF} nm): kept {len(kept)}, removed {len(water_coords) - len(kept)} distant water(s)")
        water_coords = kept

if not water_coords:
    print("No crystal waters to merge, skipping.")
    sys.exit(0)

n_waters = len(water_coords)

# Find max residue number and atom number
max_resnum = 0
max_atomnum = n_existing
for line in existing_atoms:
    m = re.match(r"\s*(\d+)", line)
    if m:
        r = int(m.group(1))
        if r > max_resnum:
            max_resnum = r

# Determine the SOL water model layout from the topology's included water .itp.
# 3-point models (SPC/TIP3P/OPC3) have 3 atoms; 4-point models (OPC/TIP4P) add an
# M virtual site that must also be written to the coordinate file.
atom_names, m_dist = water_model_sites(args.topol)
n_sites = len(atom_names)

# Placeholder H geometry (OPC O-H = 0.08724 nm, H-O-H = 103.6 deg). The rigid-water
# SETTLE constraints reset the exact geometry during grompp/emin, so only the
# relative arrangement matters here. H1/H2 are symmetric about the +z bisector.
oh_dist = 0.08724  # nm
hoh_angle = 103.6 * math.pi / 180.0
dz = oh_dist * math.cos(hoh_angle / 2.0)
dy = oh_dist * math.sin(hoh_angle / 2.0)

water_atom_lines = []
next_resnum = max_resnum + 1
next_atomnum = max_atomnum + 1

for x, y, z in water_coords:
    coords = [(x, y, z), (x, y + dy, z + dz), (x, y - dy, z + dz)]  # OW, HW1, HW2
    # 4-point M virtual site: on the H-O-H bisector (+z here), d_OM from O toward
    # the hydrogens (0.1594 A for OPC). Its presence makes the .gro atom count
    # match the 4-site SOL topology; grompp reconstructs its exact position.
    if m_dist is not None and n_sites >= 4:
        coords.append((x, y, z + m_dist))
    while len(coords) < n_sites:      # any further sites (5-point, unused) -> at O
        coords.append((x, y, z))
    for si in range(n_sites):
        px, py, pz = coords[si]
        water_atom_lines.append(
            f"{next_resnum:5d}{'SOL':>5s}{atom_names[si]:>5s}{next_atomnum:5d}{px:8.3f}{py:8.3f}{pz:8.3f}\n")
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
