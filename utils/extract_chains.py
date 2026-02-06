#!/usr/bin/env python3
#

import os
import sys
import argparse

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Extract chain information from PDB file and generate residue map for protein B.")
parser.add_argument('-f','--pdbfile', type=str, default="input.pdb", required=True, help="Input PDB file.")
parser.add_argument('-c','--chains', type=str, required=True, help="Comma-separated chain IDs for protein B (e.g., 'B' or 'B,C').")
args = parser.parse_args()

#------------------------------------------------------

# Parse chain IDs for protein B
chains_b = set(c.strip() for c in args.chains.split(','))

# Read PDB file and track residues in order
# We need to determine the sequential residue numbering that pdb2gmx will produce
# pdb2gmx renumbers residues sequentially across all chains

if not os.path.isfile(args.pdbfile):
  print(f"Error: PDB file '{args.pdbfile}' not found.", file=sys.stderr)
  sys.exit(1)

# First pass: collect all unique (chain, resnum) pairs in order of appearance
seen_residues = []  # list of (chain_id, original_resnum)
seen_set = set()

with open(args.pdbfile, "r") as f:
  for line in f:
    if line.startswith("ATOM") or line.startswith("HETATM"):
      chain_id = line[21]
      try:
        resnum = int(line[22:26].strip())
        key = (chain_id, resnum)
        if key not in seen_set:
          seen_set.add(key)
          seen_residues.append(key)
      except (ValueError, IndexError):
        pass

if not seen_residues:
  print(f"Error: No residues found in {args.pdbfile}.", file=sys.stderr)
  sys.exit(1)

# Now map to sequential residue numbers (1, 2, 3, ...) as pdb2gmx would assign
# and identify which sequential numbers belong to protein B chains
residues_b = set()
seq_num = 1

for chain_id, orig_resnum in seen_residues:
  if chain_id in chains_b:
    residues_b.add(seq_num)
  seq_num += 1

if not residues_b:
  print(f"Error: No residues found for chain(s) {args.chains} in {args.pdbfile}.", file=sys.stderr)
  sys.exit(1)

# Write chain_map.gs file with sequential residue numbers belonging to protein B
with open("chain_map.gs", "w") as f:
  f.write("# Sequential residue numbers belonging to protein B (after pdb2gmx renumbering)\n")
  f.write(f"# Generated from {args.pdbfile} for chain(s): {args.chains}\n")
  f.write(f"# Original chains: {','.join(sorted(chains_b))}\n")
  for resnum in sorted(residues_b):
    f.write(f"{resnum}\n")

print(f"Generated chain_map.gs with {len(residues_b)} sequential residues for chain(s) {args.chains}")
