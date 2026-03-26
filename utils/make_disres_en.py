#!/usr/bin/env python3
#

import os, sys, re, argparse
import numpy as np
from scipy.spatial.distance import cdist


#------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate distance restraints for pulling and elastic network.")
parser.add_argument('-f','--input', type=str, default="npt_cluster.gro", help="Input coordinate file (default: npt_cluster.gro)")
parser.add_argument('-m','--chainmap', type=str, required=True, help="Chain map file containing residue numbers for protein B.")
args=parser.parse_args()

#------------------------------------------------------

def read_chain_map(filepath):
  """Read chain_map.gs file and return set of residue numbers belonging to protein B."""
  residues_b = set()
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          try:
            residues_b.add(int(line.strip()))
          except (ValueError, IndexError):
            pass
  return residues_b

residues_b = read_chain_map(args.chainmap)

# Read ion residue numbers to determine max structural residue number
ion_residues = set()
ion_map_path = os.path.join(os.path.dirname(args.chainmap), "ion_residues.gs")
if os.path.isfile(ion_map_path):
  for line in open(ion_map_path):
    if not line.strip().startswith("#"):
      try:
        ion_residues.add(int(line.strip()))
      except (ValueError, IndexError):
        pass

# Max residue number for protein + structural ions (everything above is counterion/solvent)
all_structural = residues_b | ion_residues
# Protein A residues: 1 to max(residues_b) excluding residues_b gives the complement
# Use the max of all known structural residue numbers as threshold
max_structural_resnum = max(all_structural) if all_structural else 0

#------------------------------------------------------

# cutoff and elastic network parameters
interfacecutoff = 0.6
en_min = 0.4
en_max = 0.9
enk = 250

# Read coordinate file
# Store: resname, atomname, atomnum, coords
prot1_data = []  # [(resname, atomname, atomnum, x, y, z), ...]
prot2_data = []

if os.path.isfile(args.input):
  with open(args.input, "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        left = line[:15]
        right = line[15:]
        tmp = left.split() + right.split()
        try:
          s = re.search(r"\d+(\.\d+)?", tmp[0])
          resnum = int(s.group(0))
          atomname = tmp[1]
          atomnum = tmp[2]
          x, y, z = float(tmp[3]), float(tmp[4]), float(tmp[5])
          # Skip solvent and counterions - extract residue name from GRO field
          res3 = re.sub(r'\d+', '', tmp[0])
          if res3 == "SOL" or resnum > max_structural_resnum:
            continue
          if resnum not in residues_b:
            prot1_data.append((tmp[0], atomname, atomnum, x, y, z))
          else:
            prot2_data.append((tmp[0], atomname, atomnum, x, y, z))
        except (ValueError, IndexError, AttributeError):
          pass

len1 = len(prot1_data)
len2 = len(prot2_data)

# Extract coordinates as numpy arrays for vectorized distance calculations
prot1_coords = np.array([(d[3], d[4], d[5]) for d in prot1_data], dtype=np.float64) if len1 > 0 else np.empty((0, 3))
prot2_coords = np.array([(d[3], d[4], d[5]) for d in prot2_data], dtype=np.float64) if len2 > 0 else np.empty((0, 3))

# Find indices of atoms that are NOT H* or MN* (for interface calculation)
prot1_valid = np.array([i for i in range(len1) if prot1_data[i][1][0] != "H" and prot1_data[i][1][:2] != "MN"])
prot2_valid = np.array([i for i in range(len2) if prot2_data[i][1][0] != "H" and prot2_data[i][1][:2] != "MN"])

# Calculate inter-protein distance restraints using cdist
# Only calculate for valid atoms
interdis = []
if len(prot1_valid) > 0 and len(prot2_valid) > 0:
  valid_coords1 = prot1_coords[prot1_valid]
  valid_coords2 = prot2_coords[prot2_valid]

  # Calculate all pairwise distances at once
  all_distances = cdist(valid_coords1, valid_coords2)

  # Find pairs within cutoff - iterate in same order as original (i then j)
  for i_idx, i in enumerate(prot1_valid):
    for j_idx, j in enumerate(prot2_valid):
      dist = all_distances[i_idx, j_idx]
      if dist <= interfacecutoff:
        interdis.append((i, j, dist))

numinterdis = len(interdis)

# Function to find anchor residues and build elastic network
def build_elastic_network(prot_data, prot_coords):
  """Build elastic network for a protein.

  Args:
    prot_data: list of (resname, atomname, atomnum, x, y, z)
    prot_coords: numpy array of coordinates

  Returns:
    en_pairs: list of (i, j, distance) tuples for elastic network
    protkeep: list of indices into prot_data for kept CA atoms
  """
  prot_len = len(prot_data)

  # Find anchor residues (fragment termini)
  # Method 1: GROMOS-style with OT or H2 atoms
  # Method 2: ACE/NME-capped (CHARMM36/AMBER19SB) - find neighbors of caps
  anchor_resnames = set()

  # Collect all unique resnames and their types
  resname_to_type = {}
  for i in range(prot_len):
    resname = prot_data[i][0]
    res3 = resname[-3:]
    resname_to_type[resname] = res3

  # Method 1: Find residues with OT or H2 atoms (GROMOS)
  for i in range(prot_len):
    atomname = prot_data[i][1]
    if atomname == "OT" or atomname == "H2":
      anchor_resnames.add(prot_data[i][0])

  # Method 2: Find residues neighboring ACE or NME (CHARMM36/AMBER19SB)
  # Build a map of resnum -> resname
  resnum_to_resname = {}
  for resname in resname_to_type:
    s = re.search(r"\d+", resname)
    if s:
      resnum = int(s.group(0))
      resnum_to_resname[resnum] = resname

  # Find ACE and NME residues and their neighbors
  for resname, res3 in resname_to_type.items():
    if res3 == "ACE":
      # ACE caps the N-terminus, so the next residue is the anchor
      s = re.search(r"\d+", resname)
      if s:
        ace_num = int(s.group(0))
        next_num = ace_num + 1
        if next_num in resnum_to_resname:
          next_resname = resnum_to_resname[next_num]
          # Don't add if the next residue is also a cap
          if resname_to_type.get(next_resname) not in ("ACE", "NME"):
            anchor_resnames.add(next_resname)
    elif res3 == "NME":
      # NME caps the C-terminus, so the previous residue is the anchor
      s = re.search(r"\d+", resname)
      if s:
        nme_num = int(s.group(0))
        prev_num = nme_num - 1
        if prev_num in resnum_to_resname:
          prev_resname = resnum_to_resname[prev_num]
          # Don't add if the previous residue is also a cap
          if resname_to_type.get(prev_resname) not in ("ACE", "NME"):
            anchor_resnames.add(prev_resname)

  # Find CA atoms for anchor residues
  anchor_indices = []
  for i in range(prot_len):
    if prot_data[i][1] == "CA" and prot_data[i][0] in anchor_resnames:
      anchor_indices.append(i)

  # Skip first and last anchor (as in original: protanchor[1:lena-1])
  if len(anchor_indices) >= 2:
    anchor_indices = anchor_indices[1:-1]
  else:
    anchor_indices = []

  if len(anchor_indices) == 0:
    return [], []

  anchor_coords = prot_coords[anchor_indices]

  # Find all CA atoms
  ca_indices = [i for i in range(prot_len) if prot_data[i][1] == "CA"]

  if len(ca_indices) == 0:
    return [], []

  ca_coords = prot_coords[ca_indices]

  # Find CA atoms within 0.9 of any anchor
  dist_to_anchors = cdist(ca_coords, anchor_coords)
  keep_mask = np.any(dist_to_anchors <= 0.9, axis=1)

  # Get indices of kept CAs (preserving order)
  protkeep = [ca_indices[i] for i in range(len(ca_indices)) if keep_mask[i]]

  if len(protkeep) < 2:
    return [], protkeep

  # Build elastic network - pairs of kept CAs within 0.4-0.9
  keep_coords = prot_coords[protkeep]
  keep_distances = cdist(keep_coords, keep_coords)

  en_pairs = []
  for i in range(len(protkeep)):
    for j in range(i + 1, len(protkeep)):
      dist = keep_distances[i, j]
      if en_min <= dist <= en_max:
        en_pairs.append((i, j, dist))

  return en_pairs, protkeep

# Build elastic networks
en1dis, protkeep1 = build_elastic_network(prot1_data, prot1_coords)
en2dis, protkeep2 = build_elastic_network(prot2_data, prot2_coords)

numen1dis = len(en1dis)
numen2dis = len(en2dis)

print(str(numinterdis))

# Write index file
with open("index.ndx", "a") as index:
  # Inter-protein distance restraint groups
  for i, j, dist in interdis:
    atomnum1 = prot1_data[i][2]
    atomnum2 = prot2_data[j][2]
    index.write(f"[ a_{atomnum1} ]\n")
    index.write(f"{atomnum1}\n")
    index.write(f"[ a_{atomnum2} ]\n")
    index.write(f"{atomnum2}\n")

  # Elastic network groups for prot1
  for i, j, dist in en1dis:
    atomnum1 = prot1_data[protkeep1[i]][2]
    atomnum2 = prot1_data[protkeep1[j]][2]
    index.write(f"[ a_{atomnum1} ]\n")
    index.write(f"{atomnum1}\n")
    index.write(f"[ a_{atomnum2} ]\n")
    index.write(f"{atomnum2}\n")

  # Elastic network groups for prot2
  for i, j, dist in en2dis:
    atomnum1 = prot2_data[protkeep2[i]][2]
    atomnum2 = prot2_data[protkeep2[j]][2]
    index.write(f"[ a_{atomnum1} ]\n")
    index.write(f"{atomnum1}\n")
    index.write(f"[ a_{atomnum2} ]\n")
    index.write(f"{atomnum2}\n")

# Helper function to write MDP pull configurations
def write_mdp_config(filename, interdis, en1dis, en2dis, protkeep1, protkeep2,
                     prot1_data, prot2_data, k, enk, rate_inter, init_offset_inter):
  """Write pull configuration to MDP file."""
  with open(filename, "a") as f:
    total_groups = (numinterdis + numen1dis + numen2dis) * 2
    total_coords = numinterdis + numen1dis + numen2dis

    f.write(f"pull-ngroups            = {total_groups:.0f}\n")
    f.write(f"pull-ncoords            = {total_coords:.0f}\n")
    f.write("\n")

    groupcount = 1
    coordcount = 1

    # Inter-protein restraints
    for idx, (i, j, dist) in enumerate(interdis):
      atomnum1 = prot1_data[i][2]
      atomnum2 = prot2_data[j][2]

      f.write(f"pull-group{groupcount:.0f}-name        = a_{float(atomnum1):.0f}\n")
      g1 = groupcount
      groupcount += 1
      f.write(f"pull-group{groupcount:.0f}-name        = a_{float(atomnum2):.0f}\n")
      g2 = groupcount
      groupcount += 1

      f.write(f"pull-coord{coordcount:.0f}-type        = umbrella\n")
      f.write(f"pull-coord{coordcount:.0f}-geometry    = distance\n")
      f.write(f"pull-coord{coordcount:.0f}-dim         = Y Y Y\n")
      f.write(f"pull-coord{coordcount:.0f}-start       = no\n")
      f.write(f"pull-coord{coordcount:.0f}-rate        = {rate_inter}\n")
      f.write(f"pull-coord{coordcount:.0f}-groups      = {g1:.0f} {g2:.0f}\n")
      f.write(f"pull-coord{coordcount:.0f}-init        = {dist + init_offset_inter:.8f}\n")
      f.write(f"pull-coord{coordcount:.0f}-k           = {k:.8f}\n")
      f.write("\n")
      coordcount += 1

    # Elastic network for prot1
    for i, j, dist in en1dis:
      atomnum1 = prot1_data[protkeep1[i]][2]
      atomnum2 = prot1_data[protkeep1[j]][2]

      f.write(f"pull-group{groupcount:.0f}-name        = a_{float(atomnum1):.0f}\n")
      g1 = groupcount
      groupcount += 1
      f.write(f"pull-group{groupcount:.0f}-name        = a_{float(atomnum2):.0f}\n")
      g2 = groupcount
      groupcount += 1

      f.write(f"pull-coord{coordcount:.0f}-type        = umbrella\n")
      f.write(f"pull-coord{coordcount:.0f}-geometry    = distance\n")
      f.write(f"pull-coord{coordcount:.0f}-dim         = Y Y Y\n")
      f.write(f"pull-coord{coordcount:.0f}-start       = no\n")
      f.write(f"pull-coord{coordcount:.0f}-rate        = 0\n")
      f.write(f"pull-coord{coordcount:.0f}-groups      = {g1:.0f} {g2:.0f}\n")
      f.write(f"pull-coord{coordcount:.0f}-init        = {dist:.8f}\n")
      f.write(f"pull-coord{coordcount:.0f}-k           = {enk:.8f}\n")
      f.write("\n")
      coordcount += 1

    # Elastic network for prot2
    for i, j, dist in en2dis:
      atomnum1 = prot2_data[protkeep2[i]][2]
      atomnum2 = prot2_data[protkeep2[j]][2]

      f.write(f"pull-group{groupcount:.0f}-name        = a_{float(atomnum1):.0f}\n")
      g1 = groupcount
      groupcount += 1
      f.write(f"pull-group{groupcount:.0f}-name        = a_{float(atomnum2):.0f}\n")
      g2 = groupcount
      groupcount += 1

      f.write(f"pull-coord{coordcount:.0f}-type        = umbrella\n")
      f.write(f"pull-coord{coordcount:.0f}-geometry    = distance\n")
      f.write(f"pull-coord{coordcount:.0f}-dim         = Y Y Y\n")
      f.write(f"pull-coord{coordcount:.0f}-start       = no\n")
      f.write(f"pull-coord{coordcount:.0f}-rate        = 0\n")
      f.write(f"pull-coord{coordcount:.0f}-groups      = {g1:.0f} {g2:.0f}\n")
      f.write(f"pull-coord{coordcount:.0f}-init        = {dist:.8f}\n")
      f.write(f"pull-coord{coordcount:.0f}-k           = {enk:.8f}\n")
      f.write("\n")
      coordcount += 1

# Calculate k value
k = 25000.0 / numinterdis if numinterdis > 0 else 0

# Write all MDP files
write_mdp_config("bind.mdp", interdis, en1dis, en2dis, protkeep1, protkeep2,
                 prot1_data, prot2_data, k, enk, rate_inter=0.0002, init_offset_inter=0)

write_mdp_config("nptrev.mdp", interdis, en1dis, en2dis, protkeep1, protkeep2,
                 prot1_data, prot2_data, k, enk, rate_inter=0, init_offset_inter=1.0)

write_mdp_config("bindrev.mdp", interdis, en1dis, en2dis, protkeep1, protkeep2,
                 prot1_data, prot2_data, k, enk, rate_inter=-0.0002, init_offset_inter=1.0)
