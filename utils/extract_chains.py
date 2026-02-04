#!/usr/bin/env python3
#
# PLEASE REPORT BUGS, QUESTIONS AND COMMENTS TO JAN.PERTHOLD@BOKU.AC.AT
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

# Parse chain IDs
chains_b = set(c.strip() for c in args.chains.split(','))

# Read PDB file and extract residue numbers for specified chains
residues_b = set()

if os.path.isfile(args.pdbfile):
  with open(args.pdbfile, "r") as f:
    for line in f:
      if line.startswith("ATOM") or line.startswith("HETATM"):
        # PDB format: columns 22 is chain ID, columns 23-26 are residue number
        chain_id = line[21].strip()
        try:
          resnum = int(line[22:26].strip())
          if chain_id in chains_b:
            residues_b.add(resnum)
        except (ValueError, IndexError):
          pass
else:
  print(f"Error: PDB file '{args.pdbfile}' not found.", file=sys.stderr)
  sys.exit(1)

if not residues_b:
  print(f"Error: No residues found for chain(s) {args.chains} in {args.pdbfile}.", file=sys.stderr)
  sys.exit(1)

# Write chain_map.gs file with residue numbers belonging to protein B
with open("chain_map.gs", "w") as f:
  f.write("# Residue numbers belonging to protein B\n")
  f.write(f"# Generated from {args.pdbfile} for chain(s): {args.chains}\n")
  for resnum in sorted(residues_b):
    f.write(f"{resnum}\n")

print(f"Generated chain_map.gs with {len(residues_b)} residues from chain(s) {args.chains}")
