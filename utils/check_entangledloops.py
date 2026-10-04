#!/usr/bin/env python3
#
# check_entangledloops.py - stage-0 gate of job.run and job_fe.run on emin_vac.gro.
# Prints 1 when a CA thread of one protein links with a CA thread of the other
# (Gauss linking number above 1), else 0.
#
# Prints 2 and exits 1 when the check cannot be made: a missing or unreadable
# frame, a side without CA atoms, any internal error. A missing frame used to give
# empty CA lists and the verdict 0, i.e. a pass.

import os, sys, argparse
import numpy as np
from scipy.spatial.distance import cdist

import pbc_tools as pbc

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Check for entangled loops between proteins.")
parser.add_argument('-f','--confile', type=str, default="conf.gro", required=True, help="GROMACS coordinate file.")
parser.add_argument('-m','--chainmap', type=str, required=True, help="Chain map file containing residue numbers for protein B.")
args=parser.parse_args()

def fail(msg):
  print("2")
  sys.stderr.write("check_entangledloops: ERROR - %s\n" % msg)
  sys.exit(1)

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

# Read coordinate file - only CAs of protein residues, fixed .gro columns by
# position (pbc_tools). The str.split() parse lost every atom numbered 10000 or
# more and read Ca2+ ions (residue CA, atom CA) as chain CAs (audit SH6).
if not os.path.isfile(args.confile):
  fail("coordinate file %s does not exist" % args.confile)
try:
  frame = pbc.read_gro(args.confile)
  ca1_coords = []
  ca2_coords = []
  for s, e in pbc.residue_blocks(frame.head):
    names = [h[10:15].strip() for h in frame.head[s:e]]
    if not {"N", "CA", "C"} <= set(names):
      continue
    resnum = int(frame.head[s][0:5])
    i = s + names.index("CA")
    (ca2_coords if resnum in residues_b else ca1_coords).append(frame.xyz[i])
except (OSError, ValueError) as e:
  fail("cannot read %s: %s" % (args.confile, e))

# Convert to numpy arrays
coords1 = np.array(ca1_coords, dtype=np.float64).reshape(-1, 3)
coords2 = np.array(ca2_coords, dtype=np.float64).reshape(-1, 3)

len1 = len(coords1)
len2 = len(coords2)
if len1 == 0 or len2 == 0:
  fail("%s has %d receptor and %d ligand CA atoms (check chain_map.gs)" % (args.confile, len1, len2))

def linking_verdict():
  """The Gauss linking test itself, unchanged from v1.4."""
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
  # and distances between thread centers (both sides are non-empty, see above)
  ca_distances_within1 = cdist(coords1, coords1)
  ca_distances_within2 = cdist(coords2, coords2)
  ca_distances_cross = cdist(coords1, coords2)

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

  return "1" if linking_number > 1.0 else "0"

try:
  verdict = linking_verdict()
except Exception as e:        # an internal error must never read as a verdict
  fail("%s: %s" % (type(e).__name__, e))
print(verdict)
