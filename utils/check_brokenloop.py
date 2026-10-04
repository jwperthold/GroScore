#!/usr/bin/env python3
#
# check_brokenloop.py - stage-0 gate of job.run and job_fe.run on conf.gro.
# Prints 1 when a chain break (consecutive CA atoms more than 0.42 nm apart) has
# both ends within 0.4 nm of the other protein, else 0.
#
# Prints 2 and exits 1 when the check cannot be made: a missing or unreadable
# frame, a side without CA atoms, any internal error. The runners test
# '[[ $bl -eq 0 ]]', which an empty output passes, so a crash used to read as
# 'no broken loop'; '2' fails that test (the structure stops) and is neither verdict.

import os, sys, argparse
import numpy as np
from scipy.spatial.distance import cdist

import pbc_tools as pbc

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Check for broken loops at protein-protein interface.")
parser.add_argument('-f','--confile', type=str, default="conf.gro", required=True, help="GROMACS coordinate file.")
parser.add_argument('-m','--chainmap', type=str, required=True, help="Chain map file containing residue numbers for protein B.")
args=parser.parse_args()

def fail(msg):
  print("2")
  sys.stderr.write("check_brokenloop: ERROR - %s\n" % msg)
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

#------------------------------------------------------

def read_sides(path, residues_b):
  """CA atoms (resnum, xyz) and all atoms (resnum, xyz) of each side, file order.

  Fixed .gro columns by position (pbc_tools). The str.split() parse this replaces
  merged the atom name and number from atom 10000 on ('CA10000') and dropped every
  later atom: on conf.gro it saw 9999 of 14958 atoms of 1HE8 and none of protein
  A's 166 CA, and 30 of 174 CA of 2AJF's protein B (audit SH6). CA atoms are taken
  from protein residues only (N, CA and C present), not from Ca2+ ions (residue CA,
  atom CA), whose distance from the chain read as a break."""
  frame = pbc.read_gro(path)
  head, xyz = frame.head, frame.xyz
  ca = {1: ([], []), 2: ([], [])}
  allat = {1: ([], []), 2: ([], [])}
  for s, e in pbc.residue_blocks(head):
    resnum = int(head[s][0:5])
    names = [h[10:15].strip() for h in head[s:e]]
    side = 2 if resnum in residues_b else 1
    protein = {"N", "CA", "C"} <= set(names)
    for i, n in zip(range(s, e), names):
      allat[side][0].append(resnum)
      allat[side][1].append(xyz[i])
      if protein and n == "CA":
        ca[side][0].append(resnum)
        ca[side][1].append(xyz[i])
  as_arr = lambda c: np.array(c, dtype=float) if c else np.empty((0, 3))
  return ({k: (v[0], as_arr(v[1])) for k, v in ca.items()},
          {k: (v[0], as_arr(v[1])) for k, v in allat.items()})

# Find broken loops (consecutive CAs > 0.42 nm apart)
def find_broken_loops(resnums, coords):
  """Find pairs of residue numbers where consecutive CAs are > 0.42 nm apart."""
  broken = []
  if len(coords) < 2:
    return broken

  # Calculate distances between consecutive CAs
  diffs = coords[1:] - coords[:-1]
  distances = np.sqrt(np.sum(diffs**2, axis=1))

  # Find where distance > 0.42 (normal CA-CA is 0.38-0.40 nm; united-atom FFs can reach 0.401)
  broken_indices = np.where(distances > 0.42)[0]
  for idx in broken_indices:
    broken.append((resnums[idx], resnums[idx + 1]))

  return broken

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

def main():
  if not os.path.isfile(args.confile):
    fail("coordinate file %s does not exist" % args.confile)
  residues_b = read_chain_map(args.chainmap)
  try:
    ca, allat = read_sides(args.confile, residues_b)
  except (OSError, ValueError) as e:
    fail("cannot read %s: %s" % (args.confile, e))
  for side, label in ((1, "receptor"), (2, "ligand (chain_map.gs)")):
    if len(ca[side][0]) == 0:
      fail("no protein CA atom on the %s side of %s" % (label, args.confile))

  foundbrokenloops1 = find_broken_loops(*ca[1])
  foundbrokenloops2 = find_broken_loops(*ca[2])

  # Check broken loops in protein 1 against protein 2
  bothnear = check_broken_loop_near_other(foundbrokenloops1, allat[1][0], allat[1][1], allat[2][1])

  # If not found, check broken loops in protein 2 against protein 1
  if not bothnear:
    bothnear = check_broken_loop_near_other(foundbrokenloops2, allat[2][0], allat[2][1], allat[1][1])

  print("1" if bothnear else "0")

if __name__ == "__main__":
  try:
    main()
  except SystemExit:
    raise
  except Exception as e:      # an internal error must never read as a verdict
    fail("%s: %s" % (type(e).__name__, e))
