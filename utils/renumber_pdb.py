#!/usr/bin/env python3
#
# Renumber PDB residues sequentially across all chains
# This ensures unique residue numbers in the output GRO file
#

import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Renumber PDB residues sequentially and output chain mapping.")
parser.add_argument('-f','--pdbfile', type=str, required=True, help="Input PDB file.")
parser.add_argument('-o','--output', type=str, default="renumbered.pdb", help="Output PDB file.")
parser.add_argument('-c','--chains', type=str, required=True, help="Comma-separated chain IDs for protein B.")
args = parser.parse_args()

chains_b = set(c.strip() for c in args.chains.split(','))

if not os.path.isfile(args.pdbfile):
    print(f"Error: PDB file '{args.pdbfile}' not found.", file=sys.stderr)
    sys.exit(1)

# First pass: collect all unique (chain, resnum) pairs and assign sequential numbers
seen_residues = {}  # (chain, resnum) -> new_resnum
current_resnum = 0
last_key = None

# Also track which new residue numbers belong to protein B
residues_b = set()

with open(args.pdbfile, "r") as f:
    for line in f:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            chain_id = line[21]
            try:
                orig_resnum = int(line[22:26].strip())
                key = (chain_id, orig_resnum)
                if key != last_key:
                    current_resnum += 1
                    seen_residues[key] = current_resnum
                    if chain_id in chains_b:
                        residues_b.add(current_resnum)
                    last_key = key
            except (ValueError, IndexError):
                pass

# Second pass: write renumbered PDB
with open(args.pdbfile, "r") as fin, open(args.output, "w") as fout:
    for line in fin:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            chain_id = line[21]
            try:
                orig_resnum = int(line[22:26].strip())
                key = (chain_id, orig_resnum)
                new_resnum = seen_residues.get(key, orig_resnum)
                # Rebuild line with new residue number (columns 23-26, right-justified)
                new_line = line[:22] + f"{new_resnum:4d}" + line[26:]
                fout.write(new_line)
            except (ValueError, IndexError):
                fout.write(line)
        else:
            fout.write(line)

# Write chain_map.gs with the new sequential residue numbers for protein B
with open("chain_map.gs", "w") as f:
    f.write("# Sequential residue numbers belonging to protein B\n")
    f.write(f"# Generated from {args.pdbfile} for chain(s): {args.chains}\n")
    for resnum in sorted(residues_b):
        f.write(f"{resnum}\n")

print(f"Renumbered {current_resnum} residues, wrote {args.output}")
print(f"Generated chain_map.gs with {len(residues_b)} residues for chain(s) {args.chains}")
