#!/usr/bin/env python3
#

import os, sys, re, argparse
import numpy as np
from scipy.spatial.distance import cdist

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Check for entangled loops between proteins.")
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

# Read coordinate file - only CAs
ca1_coords = []
ca2_coords = []

if os.path.isfile(args.confile):
  with open(args.confile, "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        tmp = line.split()
        try:
          s = re.search(r"\d+(\.\d+)?", tmp[0])
          resnum = int(s.group(0))
          if tmp[1] == "CA":
            coords = [float(tmp[3]), float(tmp[4]), float(tmp[5])]
            if resnum not in residues_b:
              ca1_coords.append(coords)
            else:
              ca2_coords.append(coords)
        except (ValueError, IndexError, AttributeError):
          pass

# Convert to numpy arrays
coords1 = np.array(ca1_coords, dtype=np.float64)
coords2 = np.array(ca2_coords, dtype=np.float64)

len1 = len(coords1)
len2 = len(coords2)

# Pre-compute segment vectors (CA[i+1] - CA[i]) for both proteins
if len1 > 1:
  segments1 = coords1[1:] - coords1[:-1]  # Shape: (len1-1, 3)
else:
  segments1 = np.empty((0, 3))

if len2 > 1:
  segments2 = coords2[1:] - coords2[:-1]  # Shape: (len2-1, 3)
else:
  segments2 = np.empty((0, 3))

# Pre-compute midpoints of segments
if len1 > 1:
  midpoints1 = 0.5 * (coords1[1:] + coords1[:-1])
else:
  midpoints1 = np.empty((0, 3))

if len2 > 1:
  midpoints2 = 0.5 * (coords2[1:] + coords2[:-1])
else:
  midpoints2 = np.empty((0, 3))

# Pre-compute distances between all CA pairs for thread endpoint checks
# and distances between thread centers
if len1 > 0 and len2 > 0:
  ca_distances_within1 = cdist(coords1, coords1)
  ca_distances_within2 = cdist(coords2, coords2)
  ca_distances_cross = cdist(coords1, coords2)
else:
  # No entanglement possible with empty protein group
  print("after equilibration no entangled loops were found.")
  sys.exit(0)

outeroutermax = 0.0

# Thread lengths from 5 to 20
for thread_len1 in range(5, 21):
  outermax = 0.0

  # All starting positions for threads in protein 1
  for j in range(len1 - thread_len1):
    # Check if thread ends are within 1 nm
    if ca_distances_within1[j, j + thread_len1 - 1] >= 1.0:
      continue

    outer = 0.0
    center1_idx = j + thread_len1 // 2 - 1  # Center CA of thread 1

    # For each segment k in this thread
    for k in range(j, j + thread_len1):
      if k >= len1 - 1:
        break

      innermax = 0.0

      # Thread lengths for protein 2
      for thread_len2 in range(5, 21):
        # All starting positions for threads in protein 2
        for m in range(len2 - thread_len2):
          # Check if thread ends are within 1 nm AND centers are within 1 nm
          center2_idx = m + thread_len2 // 2 - 1

          if ca_distances_within2[m, m + thread_len2 - 1] >= 1.0:
            continue
          if ca_distances_cross[center1_idx, center2_idx] >= 1.0:
            continue

          inner = 0.0

          # Compute linking number contribution for all segment pairs
          # between segment k of thread 1 and all segments in thread 2
          for n in range(m, m + thread_len2):
            if n >= len2 - 1:
              break

            # Difference vector between midpoints
            dif = midpoints1[k] - midpoints2[n]
            norm = np.linalg.norm(dif)

            if norm > 0:
              # Cross product of segment vectors
              cross = np.cross(segments1[k], segments2[n])
              # Gauss linking integral contribution
              inner += np.dot(dif / (norm ** 3), cross)

          if abs(inner) > innermax:
            innermax = abs(inner)

      outer += innermax

    if abs(outer) > outermax:
      outermax = abs(outer)

  if outermax > outeroutermax:
    outeroutermax = outermax

# Check if linking number exceeds threshold
linking_number = (1.0 / (4.0 * 3.14159265359)) * outeroutermax

if linking_number > 1.0:
  print("1")
else:
  print("0")
