#!/usr/bin/env python3
#

import os, sys, re, argparse
import numpy as np
from scipy.spatial.distance import cdist

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Check for broken loops at protein-protein interface.")
parser.add_argument('-f','--confile', type=str, default="conf.gro", required=True, help="GROMACS coordinate file.")
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

# Read coordinate file in a single pass, collecting both CA and all atoms
ca1_resnums = []
ca1_coords = []
ca2_resnums = []
ca2_coords = []
all1_resnums = []
all1_coords = []
all2_resnums = []
all2_coords = []

linecount = 0
if os.path.isfile(args.confile):
  with open(args.confile, "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        tmp = line.split()
        try:
          s = re.search(r"\d+(\.\d+)?", tmp[0])
          resnum = int(s.group(0))
          x, y, z = float(tmp[3]), float(tmp[4]), float(tmp[5])
          atomname = tmp[1]

          if resnum not in residues_b:
            if atomname == "CA":
              ca1_resnums.append(resnum)
              ca1_coords.append([x, y, z])
            if linecount > 1:
              all1_resnums.append(resnum)
              all1_coords.append([x, y, z])
          else:
            if atomname == "CA":
              ca2_resnums.append(resnum)
              ca2_coords.append([x, y, z])
            if linecount > 1:
              all2_resnums.append(resnum)
              all2_coords.append([x, y, z])
        except (ValueError, IndexError, AttributeError):
          pass
        linecount += 1

# Convert to numpy arrays
ca1_coords = np.array(ca1_coords) if ca1_coords else np.empty((0, 3))
ca2_coords = np.array(ca2_coords) if ca2_coords else np.empty((0, 3))
all1_coords = np.array(all1_coords) if all1_coords else np.empty((0, 3))
all2_coords = np.array(all2_coords) if all2_coords else np.empty((0, 3))

# Find broken loops (consecutive CAs > 0.4 nm apart)
def find_broken_loops(resnums, coords):
  """Find pairs of residue numbers where consecutive CAs are > 0.4 nm apart."""
  broken = []
  if len(coords) < 2:
    return broken

  # Calculate distances between consecutive CAs
  diffs = coords[1:] - coords[:-1]
  distances = np.sqrt(np.sum(diffs**2, axis=1))

  # Find where distance > 0.41 (normal CA-CA is 0.38-0.40 nm; united-atom FFs can reach 0.401)
  broken_indices = np.where(distances > 0.41)[0]
  for idx in broken_indices:
    broken.append((resnums[idx], resnums[idx + 1]))

  return broken

foundbrokenloops1 = find_broken_loops(ca1_resnums, ca1_coords)
foundbrokenloops2 = find_broken_loops(ca2_resnums, ca2_coords)

# Check if both ends of a broken loop are near atoms of the other protein
def check_broken_loop_near_other(broken_loops, own_resnums, own_coords, other_coords, cutoff=0.4):
  """Check if both ends of any broken loop are within cutoff of the other protein."""
  if len(broken_loops) == 0 or len(other_coords) == 0:
    return False

  # Build index: resnum -> list of atom indices
  resnum_to_indices = {}
  for i, resnum in enumerate(own_resnums):
    if resnum not in resnum_to_indices:
      resnum_to_indices[resnum] = []
    resnum_to_indices[resnum].append(i)

  for res1, res2 in broken_loops:
    # Check if any atom of res1 is near any atom of other protein
    res1_near = False
    if res1 in resnum_to_indices:
      res1_indices = resnum_to_indices[res1]
      res1_coords = own_coords[res1_indices]
      # Calculate min distance from res1 atoms to all other protein atoms
      distances = cdist(res1_coords, other_coords)
      if np.any(distances < cutoff):
        res1_near = True

    if not res1_near:
      continue

    # Check if any atom of res2 is near any atom of other protein
    if res2 in resnum_to_indices:
      res2_indices = resnum_to_indices[res2]
      res2_coords = own_coords[res2_indices]
      distances = cdist(res2_coords, other_coords)
      if np.any(distances < cutoff):
        return True  # Both ends are near

  return False

# Check broken loops in protein 1 against protein 2
bothnear = check_broken_loop_near_other(foundbrokenloops1, all1_resnums, all1_coords, all2_coords)

# If not found, check broken loops in protein 2 against protein 1
if not bothnear:
  bothnear = check_broken_loop_near_other(foundbrokenloops2, all2_resnums, all2_coords, all1_coords)

if bothnear:
  print("1")
else:
  print("0")
