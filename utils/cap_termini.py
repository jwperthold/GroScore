#!/usr/bin/env python3
#
# cap_termini.py - Add ACE/NME capping residues to all fragment termini
#
# Usage: python cap_termini.py -f input.pdb -o output.pdb [-m chain_map.gs]
#

import os
import sys
import argparse
from pdbfixer import PDBFixer
from openmm import unit

parser = argparse.ArgumentParser(description="Add ACE/NME caps to fragment termini using PDBFixer")
parser.add_argument('-f', '--file', type=str, required=True, help="Input PDB file")
parser.add_argument('-o', '--output', type=str, required=True, help="Output PDB file")
parser.add_argument('-m', '--chainmap', type=str, default=None, help="Chain map file to update (chain_map.gs)")
parser.add_argument('--rename-nme-carbon', action='store_true', help="Rename NME carbon from C to CH3 (for CHARMM36)")
parser.add_argument('--ace-only', action='store_true', help="Only add ACE caps (skip NME), for use with COOH C-term patches")
parser.add_argument('--rename-ace-carbon', action='store_true', help="Rename ACE methyl carbon from CH3 to CA (for GROMOS)")
args = parser.parse_args()

if not os.path.isfile(args.file):
    print(f"Error: PDB file '{args.file}' not found.", file=sys.stderr)
    sys.exit(1)

# Read original chain_map.gs before modifying anything
original_b_resnums = set()
if args.chainmap and os.path.isfile(args.chainmap):
    with open(args.chainmap, "r") as f:
        for line in f:
            if not line.strip().startswith("#"):
                try:
                    original_b_resnums.add(int(line.strip()))
                except (ValueError, IndexError):
                    pass

# Read original PDB to map residue numbers to chain IDs
# We need this to determine protein A/B membership for ACE/NME caps
orig_resnum_to_chain = {}
with open(args.file, "r") as f:
    for line in f:
        if line.startswith("ATOM"):
            chain_id = line[21]
            try:
                resnum = int(line[22:26].strip())
                if resnum not in orig_resnum_to_chain:
                    orig_resnum_to_chain[resnum] = chain_id
            except (ValueError, IndexError):
                pass

# Load PDB with PDBFixer
fixer = PDBFixer(filename=args.file)
fixer.findMissingResidues()
fixer.missingResidues = {}  # Clear auto-detected missing residues

# Add ACE/NME caps to each chain's termini
for chain in fixer.topology.chains():
    residues = list(chain.residues())
    if len(residues) == 0:
        continue
    # Skip if already capped (e.g. from a previous capping step)
    if residues[0].name != 'ACE':
        fixer.missingResidues[(chain.index, 0)] = ['ACE']
    if not args.ace_only:
        if residues[-1].name != 'NME':
            fixer.missingResidues[(chain.index, len(residues))] = ['NME']

fixer.findMissingAtoms()
fixer.addMissingAtoms()

# Build mapping: for each PDBFixer chain, determine if it belongs to protein B
# All original residues in a PDBFixer chain share the same protein membership
# (TER records create separate chains, and chains don't mix proteins)
chain_is_b = {}
for chain in fixer.topology.chains():
    is_b = None
    for residue in chain.residues():
        if residue.name in ('ACE', 'NME'):
            continue
        # Try to match this residue to an original residue number
        try:
            resnum = int(residue.id)
        except (ValueError, TypeError):
            continue
        if resnum in original_b_resnums:
            is_b = True
            break
        elif resnum in orig_resnum_to_chain:
            is_b = False
            break
    chain_is_b[chain.index] = is_b if is_b is not None else False

# Write custom PDB output with proper formatting
# We need ATOM records (not HETATM), fixed-width format, and TER records
positions = fixer.positions
new_b_resnums = set()
resnum_counter = 0
atom_num = 0
prev_chain_idx = None

with open(args.output, "w") as out:
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if len(residues) == 0:
            continue

        # Write TER between chains (not before first chain)
        if prev_chain_idx is not None:
            out.write("TER\n")

        for residue in residues:
            resnum_counter += 1
            resname = residue.name

            # Determine protein A/B membership
            if chain_is_b.get(chain.index, False):
                new_b_resnums.add(resnum_counter)

            # Determine chain ID from original residue or inherited from chain
            if resname in ('ACE', 'NME'):
                # Inherit chain ID from the chain's original residues
                chain_id = None
                for r in residues:
                    if r.name not in ('ACE', 'NME'):
                        try:
                            orig_resnum = int(r.id)
                            chain_id = orig_resnum_to_chain.get(orig_resnum)
                            if chain_id:
                                break
                        except (ValueError, TypeError):
                            continue
                if chain_id is None:
                    chain_id = 'A'
            else:
                try:
                    orig_resnum = int(residue.id)
                    chain_id = orig_resnum_to_chain.get(orig_resnum, 'A')
                except (ValueError, TypeError):
                    chain_id = 'A'

            for atom in residue.atoms():
                atom_num += 1
                idx = atom.index
                x = positions[idx].value_in_unit(unit.angstroms)
                atomname = atom.name
                # PDBFixer names NME's methyl carbon "C", but CHARMM36 expects "CH3"
                # AMBER19SB expects "C", so only rename if --rename-nme-carbon flag is set
                if args.rename_nme_carbon and resname == 'NME' and atomname == 'C':
                    atomname = 'CH3'
                # PDBFixer names ACE's methyl carbon "CH3", but GROMOS expects "CA"
                # (GROMOS RTP defines atom NAME "CA" with atom TYPE "CH3")
                # CHARMM36 expects "CH3" so this rename is GROMOS-specific
                if args.rename_ace_carbon and resname == 'ACE' and atomname == 'CH3':
                    atomname = 'CA'
                element = atom.element.symbol if atom.element else atomname[0]

                # PDB fixed-width format
                # Columns: 1-6 record, 7-11 serial, 12 space, 13-16 name,
                #          17 altloc, 18-20 resname, 21 space, 22 chainID,
                #          23-26 resSeq, 27 iCode, 28-30 spaces,
                #          31-38 x, 39-46 y, 47-54 z, 55-60 occupancy,
                #          61-66 tempFactor, 77-78 element
                # Atom name: left-align in col 13-16 with leading space for
                # 1-3 char names (standard PDB convention)
                if len(atomname) < 4:
                    atomname_fmt = f" {atomname:<3s}"
                else:
                    atomname_fmt = f"{atomname:<4s}"

                out.write(f"ATOM  {atom_num:5d} {atomname_fmt} {resname:>3s} {chain_id}{resnum_counter:4d}    "
                          f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}"
                          f"  1.00  0.00          {element:>2s}\n")

        prev_chain_idx = chain.index

    out.write("END\n")

# Update chain_map.gs if requested
if args.chainmap:
    with open(args.chainmap, "w") as f:
        f.write("# Sequential residue numbers belonging to protein B\n")
        cap_label = "ACE caps" if args.ace_only else "ACE/NME caps"
        f.write(f"# Updated by cap_termini.py with {cap_label}\n")
        for resnum in sorted(new_b_resnums):
            f.write(f"{resnum}\n")

cap_type = "ACE caps (no NME)" if args.ace_only else "ACE/NME caps"
print(f"Added {cap_type}, wrote {args.output} ({resnum_counter} residues, {atom_num} atoms)")
if args.chainmap:
    print(f"Updated {args.chainmap} with {len(new_b_resnums)} residues for protein B")
