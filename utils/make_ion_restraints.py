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
ION_RESIDUES = {"BA", "CA", "CD", "CL", "CO", "CS", "CU", "CU1", "FE", "FE2", "HG", "K", "LI", "MG", "MN", "NA", "NI", "PB", "SD", "SR", "ZN"}

# Coordination detection cutoff (nm)
# Captures bonds at 0.20-0.24 nm with margin; excludes non-coordinating
# HIS nitrogen at ~0.31-0.33 nm from Zn
COORD_CUTOFF = 0.3

# Harmonic restraint force constant (kJ/mol/nm²)
# Strong enough to maintain coordination, weaker than covalent bonds (~250000)
COORD_K = 10000.0

# Optimal coordination distances (nm) per (ion, coordinating element) pair.
# Sources:
#   Zn-S: CHARMM36 SS-ZN bond parameter (Stote & Karplus, 1995)
#   Zn-N: CHARMM36 NR2-ZN bond parameter (Stote & Karplus, 1995)
#   Zn-O: Dudev & Lim, JACS 2000; Koca et al., J Comput Chem 2003
#   Ca-O: Marchand & Bhatt, J Phys Chem B 2020; Dudev & Lim, Chem Rev 2003
#   Ca-N: Dudev & Lim, Chem Rev 2003
#   Mg-O: Dudev & Lim, JACS 1999; Zheng et al., J Mol Biol 2008
#   Mg-N: Zheng et al., J Mol Biol 2008
#   Cu-S: Comba & Remenyi, Coord Chem Rev 2003
#   Cu-N: Comba & Remenyi, Coord Chem Rev 2003
#   Cu-O: Comba & Remenyi, Coord Chem Rev 2003
OPTIMAL_DISTANCES = {
    # Zinc (tetrahedral coordination)
    ("ZN", "S"): 0.232,   # Zn-S(Cys): CHARMM36 parameter
    ("ZN", "N"): 0.207,   # Zn-N(His): CHARMM36 parameter
    ("ZN", "O"): 0.210,   # Zn-O(Asp/Glu/water)
    # Calcium (octahedral/7-8 coordination)
    ("CA", "O"): 0.236,   # Ca-O(Asp/Glu/backbone carbonyl)
    ("CA", "N"): 0.245,   # Ca-N(backbone amide) - rare
    # Magnesium (octahedral coordination)
    ("MG", "O"): 0.210,   # Mg-O(Asp/Glu/water)
    ("MG", "N"): 0.220,   # Mg-N(His) - rare
    # Copper(II) (square planar / distorted octahedral)
    ("CU", "S"): 0.226,   # Cu(II)-S(Cys/Met)
    ("CU", "N"): 0.200,   # Cu(II)-N(His)
    ("CU", "O"): 0.197,   # Cu(II)-O(Asp/Glu/water)
    # Copper(I) (tetrahedral/linear)
    ("CU1", "S"): 0.221,  # Cu(I)-S(Cys/Met)
    ("CU1", "N"): 0.202,  # Cu(I)-N(His)
    # Iron (Fe2+/Fe3+ in [2Fe-2S] / [4Fe-4S] clusters)
    ("FE", "S"): 0.230,   # Fe-S(Cys/bridging sulfide): ~2.3 Å
    ("FE", "N"): 0.220,   # Fe-N(His)
    ("FE", "O"): 0.210,   # Fe-O(Asp/Glu)
    ("FE2", "S"): 0.230,  # Fe2+-S
    ("FE2", "N"): 0.220,  # Fe2+-N
    ("FE2", "O"): 0.210,  # Fe2+-O
    # Sulfide ions in FeS clusters (restrain to Fe)
    ("SD", "FE"): 0.226,  # S(bridge)-Fe: ~2.26 Å in [2Fe-2S]
    # Manganese (similar coordination geometry to Zn/Fe)
    ("MN", "S"): 0.240,   # Mn-S(Cys)
    ("MN", "N"): 0.220,   # Mn-N(His)
    ("MN", "O"): 0.215,   # Mn-O(Asp/Glu/water)
    # Cobalt (similar to Zn)
    ("CO", "S"): 0.230,   # Co-S(Cys)
    ("CO", "N"): 0.210,   # Co-N(His)
    ("CO", "O"): 0.210,   # Co-O(Asp/Glu)
    # Nickel (similar to Mg/Zn)
    ("NI", "S"): 0.225,   # Ni-S(Cys)
    ("NI", "N"): 0.210,   # Ni-N(His)
    ("NI", "O"): 0.205,   # Ni-O(Asp/Glu)
    # Cadmium (similar to Ca, slightly larger)
    ("CD", "S"): 0.255,   # Cd-S(Cys)
    ("CD", "N"): 0.230,   # Cd-N(His)
    ("CD", "O"): 0.230,   # Cd-O(Asp/Glu)
    # Strontium (similar to Ca)
    ("SR", "O"): 0.250,   # Sr-O
    # Barium (larger than Ca)
    ("BA", "O"): 0.270,   # Ba-O
    # Mercury
    ("HG", "S"): 0.240,   # Hg-S(Cys)
    ("HG", "N"): 0.220,   # Hg-N(His)
    # Lead
    ("PB", "S"): 0.260,   # Pb-S(Cys)
    ("PB", "O"): 0.250,   # Pb-O
}
# Fallback distance if a specific pair is not parametrized
DEFAULT_DISTANCE = 0.215

parser = argparse.ArgumentParser(description="Generate ion coordination restraints for topology.")
parser.add_argument('-f', '--input', type=str, required=True, help="Input coordinate file (GRO format)")
parser.add_argument('-m', '--chainmap', type=str, required=True, help="Chain map file (chain_map.gs)")
args = parser.parse_args()

# Read protein B residue numbers from chain_map.gs
residues_b = set()
with open(args.chainmap) as f:
    for line in f:
        s = line.strip()
        if s and not s.startswith("#"):
            try:
                residues_b.add(int(s))
            except (ValueError, IndexError):
                pass

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

                # Collect ion atoms (by residue name) and coordinating protein atoms (S, N, O only)
                # Also track residue number for same-protein filtering
                if res3 in ION_RESIDUES:
                    ion_atoms.append((atomnum, tmp[0], atomname, x, y, z, resnum))
                elif atomname[0] in ("S", "N", "O"):
                    prot_atoms.append((atomnum, tmp[0], atomname, x, y, z, resnum))
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
coord_pairs = []  # [(ion_atomnum, prot_atomnum, measured_dist, optimal_dist, ion_resname, ion_name, prot_resname, prot_atomname), ...]

# Only metal ions coordinate protein atoms (lone pair donation from S, N, O)
METAL_IONS = {"ZN", "CA", "MG", "CU", "CU1", "FE", "FE2",
              "MN", "CO", "NI", "CD", "SR", "BA", "HG", "PB"}

for i in range(len(ion_atoms)):
    ion_resname_field = ion_atoms[i][1]
    ion_res3 = re.sub(r'\d+', '', ion_resname_field)
    if ion_res3 not in METAL_IONS:
        continue  # non-metal ions (SD, CL, NA) don't coordinate protein atoms
    ion_resnum = ion_atoms[i][6]
    ion_is_b = ion_resnum in residues_b
    for j in range(len(prot_atoms)):
        dist = distances[i, j]
        if dist <= COORD_CUTOFF:
            prot_resnum = prot_atoms[j][6]
            prot_is_b = prot_resnum in residues_b
            # Only restrain ion to atoms on the SAME protein side
            # Cross-protein restraints would resist pulling separation
            if ion_is_b != prot_is_b:
                continue
            ion_atomnum = ion_atoms[i][0]
            ion_name = ion_atoms[i][2]
            prot_atomnum = prot_atoms[j][0]
            prot_resname = prot_atoms[j][1]
            prot_atomname = prot_atoms[j][2]
            # Look up optimal distance for this ion-element pair
            coord_element = prot_atomname[0]  # S, N, or O
            optimal_dist = OPTIMAL_DISTANCES.get((ion_res3, coord_element), DEFAULT_DISTANCE)
            coord_pairs.append((ion_atomnum, prot_atomnum, dist, optimal_dist,
                                ion_res3, ion_name, prot_resname, prot_atomname))

# Also detect intra-cluster ion-ion coordination (e.g., Fe-S in [2Fe-2S])
if len(ion_atoms) > 1:
    ion_ion_dists = cdist(ion_coords, ion_coords)
    for i in range(len(ion_atoms)):
        ion_i_res3 = re.sub(r'\d+', '', ion_atoms[i][1])
        for j in range(i + 1, len(ion_atoms)):
            dist = ion_ion_dists[i, j]
            if dist <= COORD_CUTOFF:
                # Only restrain ions on the same protein side
                if (ion_atoms[i][6] in residues_b) != (ion_atoms[j][6] in residues_b):
                    continue
                ion_j_res3 = re.sub(r'\d+', '', ion_atoms[j][1])
                # Look up optimal distance (try both orderings)
                opt = OPTIMAL_DISTANCES.get((ion_i_res3, ion_j_res3),
                      OPTIMAL_DISTANCES.get((ion_j_res3, ion_i_res3), DEFAULT_DISTANCE))
                coord_pairs.append((ion_atoms[i][0], ion_atoms[j][0], dist, opt,
                                    ion_i_res3, ion_atoms[i][2],
                                    ion_j_res3, ion_atoms[j][2]))

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
    f.write("; Restraint distances use optimal values from force field parameters and literature\n")
    f.write("[ intermolecular_interactions ]\n")
    f.write("[ bonds ]\n")
    f.write("; ai      aj  funct      b0(nm)      kb(kJ/mol/nm^2)  ; description\n")
    for ion_num, prot_num, meas_dist, opt_dist, ion_res, ion_name, prot_resname, prot_atomname in coord_pairs:
        f.write(f"  {ion_num:6d}  {prot_num:6d}      6  {opt_dist:12.6f}  {COORD_K:14.2f}  "
                f"; {ion_name}-{prot_atomname}({prot_resname}) measured={meas_dist:.3f}\n")

print(f"Generated {len(coord_pairs)} ion coordination restraint(s):")
for ion_num, prot_num, meas_dist, opt_dist, ion_res, ion_name, prot_resname, prot_atomname in coord_pairs:
    print(f"  {ion_name} (atom {ion_num}) - {prot_atomname} of {prot_resname} (atom {prot_num}): "
          f"measured={meas_dist:.3f} nm, restraint={opt_dist:.3f} nm")
