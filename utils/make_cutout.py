#!/usr/bin/env python3
#
# PLEASE REPORT BUGS, QUESTIONS AND COMMENTS TO jan@ackergarten.at
#

import os, sys, re, argparse
import numpy as np
from scipy.spatial.distance import cdist

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Cut-out protein around the interface.")
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

#------------------------------------------------------

# cutoff parameters
interfacecutoff = 0.6
keepcutoff = 2.0
en_min = 0.4
en_max = 0.9

# read coordinate file - separate string and numeric data for efficiency
# Lists for string data (residue name, atom name)
prot1_resname = []
prot1_atomname = []
prot2_resname = []
prot2_atomname = []

# Lists for coordinates (will convert to numpy array)
prot1_coords = []
prot2_coords = []

if os.path.isfile("conf.gro"):
  with open("conf.gro", "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        left = line[:15]
        right = line[15:]
        tmp = left.split() + right.split()
        try:
          s = re.search(r"\d+(\.\d+)?", tmp[0])
          resnum = int(s.group(0))
          atomname = tmp[1]
          # Skip CL, H*, MN*
          if atomname == "CL" or atomname[0] == "H" or atomname[:2] == "MN":
            continue
          coords = [float(tmp[3]), float(tmp[4]), float(tmp[5])]
          if resnum not in residues_b:
            prot1_resname.append(tmp[0])
            prot1_atomname.append(atomname)
            prot1_coords.append(coords)
          else:
            prot2_resname.append(tmp[0])
            prot2_atomname.append(atomname)
            prot2_coords.append(coords)
        except (ValueError, IndexError, AttributeError):
          pass

# Convert to numpy arrays
prot1_coords = np.array(prot1_coords, dtype=np.float64)
prot2_coords = np.array(prot2_coords, dtype=np.float64)
len1 = len(prot1_resname)
len2 = len(prot2_resname)

if len1 == 0 or len2 == 0:
  # No atoms in one or both proteins - write empty file
  open("cutout.pdb", "w").close()
  sys.exit(0)

# Calculate all inter-protein distances at once using cdist
# This replaces the O(n*m) nested loop with a single vectorized call
all_distances = cdist(prot1_coords, prot2_coords)

# Find interface atom pairs (distance <= interfacecutoff)
interface_mask = all_distances <= interfacecutoff
interface_pairs = np.argwhere(interface_mask)

if len(interface_pairs) == 0:
  # No interface found
  open("cutout.pdb", "w").close()
  sys.exit(0)

# Get unique interface atom indices for each protein
interface_atoms1 = np.unique(interface_pairs[:, 0])
interface_atoms2 = np.unique(interface_pairs[:, 1])

# Find atoms within keepcutoff of interface atoms
# For prot1: calculate distances from all prot1 atoms to prot1 interface atoms
dist_to_interface1 = cdist(prot1_coords, prot1_coords[interface_atoms1])
keep1_mask = np.any(dist_to_interface1 <= keepcutoff, axis=1)
keep1_indices = np.where(keep1_mask)[0]
keep1_resnames = set(prot1_resname[i] for i in keep1_indices)

# For prot2: calculate distances from all prot2 atoms to prot2 interface atoms
dist_to_interface2 = cdist(prot2_coords, prot2_coords[interface_atoms2])
keep2_mask = np.any(dist_to_interface2 <= keepcutoff, axis=1)
keep2_indices = np.where(keep2_mask)[0]
keep2_resnames = set(prot2_resname[i] for i in keep2_indices)

# Get CA atoms from kept residues for elastic network check
protkeep1_indices = [i for i in range(len1) if prot1_resname[i] in keep1_resnames and prot1_atomname[i] == "CA"]
protkeep2_indices = [i for i in range(len2) if prot2_resname[i] in keep2_resnames and prot2_atomname[i] == "CA"]

# Check elastic network distances for CA atoms
laterkeep1_resnames = set()
if len(protkeep1_indices) > 1:
  ca_coords1 = prot1_coords[protkeep1_indices]
  ca_distances1 = cdist(ca_coords1, ca_coords1)
  # For each CA, check if any other CA is within EN distance range
  for i in range(len(protkeep1_indices)):
    # Check distances to all other CAs (excluding self)
    dists = ca_distances1[i]
    mask = (dists >= en_min) & (dists <= en_max)
    mask[i] = False  # Exclude self
    if np.any(mask):
      laterkeep1_resnames.add(prot1_resname[protkeep1_indices[i]])

laterkeep2_resnames = set()
if len(protkeep2_indices) > 1:
  ca_coords2 = prot2_coords[protkeep2_indices]
  ca_distances2 = cdist(ca_coords2, ca_coords2)
  for i in range(len(protkeep2_indices)):
    dists = ca_distances2[i]
    mask = (dists >= en_min) & (dists <= en_max)
    mask[i] = False
    if np.any(mask):
      laterkeep2_resnames.add(prot2_resname[protkeep2_indices[i]])

# Helper function to extract residue number from resname (e.g., "123ALA" -> 123, "123CLYS" -> 123)
def get_resnum(resname):
  # Extract leading digits to handle both 3-letter (ALA) and 4-letter (CLYS, NLYS) residue codes
  digits = ''
  for c in resname:
    if c.isdigit():
      digits += c
    else:
      break
  return int(digits)

# Minimum fragment length (residues)
MIN_FRAGMENT_LEN = 3

def extend_small_fragments(kept_resnames, all_resnames):
  """Extend fragments smaller than MIN_FRAGMENT_LEN by adding neighboring residues."""
  if len(kept_resnames) == 0:
    return kept_resnames

  # Get unique residue numbers from kept resnames
  kept_resnums = sorted(set(get_resnum(rn) for rn in kept_resnames))

  # Get all available residue numbers from original protein
  all_resnums = sorted(set(get_resnum(rn) for rn in all_resnames))
  all_resnums_set = set(all_resnums)

  # Build mapping from resnum to resname (for adding back)
  resnum_to_resname = {}
  for rn in all_resnames:
    resnum = get_resnum(rn)
    if resnum not in resnum_to_resname:
      resnum_to_resname[resnum] = rn

  # Identify fragments (contiguous residue groups)
  fragments = []
  current_frag = [kept_resnums[0]]
  for i in range(1, len(kept_resnums)):
    if kept_resnums[i] == kept_resnums[i-1] + 1:
      current_frag.append(kept_resnums[i])
    else:
      fragments.append(current_frag)
      current_frag = [kept_resnums[i]]
  fragments.append(current_frag)

  # Extend small fragments
  extended_resnums = set(kept_resnums)
  for frag in fragments:
    if len(frag) < MIN_FRAGMENT_LEN:
      needed = MIN_FRAGMENT_LEN - len(frag)
      frag_start = frag[0]
      frag_end = frag[-1]

      # Try to extend in both directions alternately
      added = 0
      extend_before = 1
      extend_after = 1
      while added < needed:
        # Try extending before
        candidate = frag_start - extend_before
        if candidate in all_resnums_set and candidate not in extended_resnums:
          extended_resnums.add(candidate)
          added += 1
          extend_before += 1
          if added >= needed:
            break
        else:
          extend_before += 1

        # Try extending after
        candidate = frag_end + extend_after
        if candidate in all_resnums_set and candidate not in extended_resnums:
          extended_resnums.add(candidate)
          added += 1
          extend_after += 1
          if added >= needed:
            break
        else:
          extend_after += 1

        # Prevent infinite loop if no more residues available
        if extend_before > 100 and extend_after > 100:
          break

  # Convert back to resnames
  extended_resnames = set()
  for rn in all_resnames:
    if get_resnum(rn) in extended_resnums:
      extended_resnames.add(rn)

  return extended_resnames

# Extend small fragments by adding neighboring residues
laterkeep1_resnames = extend_small_fragments(laterkeep1_resnames, prot1_resname)
laterkeep2_resnames = extend_small_fragments(laterkeep2_resnames, prot2_resname)

# Collect final atoms to keep
protlaterkeep1 = [(prot1_resname[i], prot1_atomname[i], prot1_coords[i])
                  for i in range(len1) if prot1_resname[i] in laterkeep1_resnames]
protlaterkeep2 = [(prot2_resname[i], prot2_atomname[i], prot2_coords[i])
                  for i in range(len2) if prot2_resname[i] in laterkeep2_resnames]

lenlk1 = len(protlaterkeep1)
lenlk2 = len(protlaterkeep2)

# Write cutout PDB file
with open("cutout.pdb", "w") as pdbfile:
  atom_num = 0
  prev_resnum = None

  # Write protein A atoms
  for i in range(lenlk1):
    resname, atomname, coords = protlaterkeep1[i]
    resnum = get_resnum(resname)
    res3 = resname[-3:]

    # Write TER record if there's a gap in residue numbering
    if prev_resnum is not None and resnum > prev_resnum + 1:
      atom_num += 1
      pdbfile.write(f"TER  {atom_num:>6}       {' ':>2} A{prev_resnum:>4}\n")

    atom_num += 1
    pdbfile.write(f"ATOM {atom_num:>6}  {atomname:<3} {res3:>2} A{resnum:>4}    "
                  f"{coords[0]*10:8.3f}{coords[1]*10:8.3f}{coords[2]*10:8.3f}"
                  f"  1.00  0.00          {atomname[0]:>2}\n")
    prev_resnum = resnum

  # Write TER between protein A and protein B
  if lenlk1 > 0 and lenlk2 > 0:
    atom_num += 1
    pdbfile.write(f"TER  {atom_num:>6}       {' ':>2} A{prev_resnum:>4}\n")

  prev_resnum = None

  # Write protein B atoms
  for i in range(lenlk2):
    resname, atomname, coords = protlaterkeep2[i]
    resnum = get_resnum(resname)
    res3 = resname[-3:]

    # Write TER record if there's a gap in residue numbering
    if prev_resnum is not None and resnum > prev_resnum + 1:
      atom_num += 1
      pdbfile.write(f"TER  {atom_num:>6}       {' ':>2} B{prev_resnum:>4}\n")

    atom_num += 1
    pdbfile.write(f"ATOM {atom_num:>6}  {atomname:<3} {res3:>2} B{resnum:>4}    "
                  f"{coords[0]*10:8.3f}{coords[1]*10:8.3f}{coords[2]*10:8.3f}"
                  f"  1.00  0.00          {atomname[0]:>2}\n")
    prev_resnum = resnum
