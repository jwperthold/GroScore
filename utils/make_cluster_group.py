#!/usr/bin/env python3
#
# make_cluster_group.py - Add Protein_Struct group to index.ndx
#
# Creates a custom index group containing Protein + structural ions + ligands
# (excluding counterions from gmx genion) for PBC clustering.
#
# Usage: python make_cluster_group.py -f emin_solv.gro -n index.ndx
#

import os
import re
import argparse

parser = argparse.ArgumentParser(description="Add Protein_Struct group to index.ndx")
parser.add_argument('-f', '--gro', type=str, required=True, help="GRO coordinate file")
parser.add_argument('-n', '--ndx', type=str, default="index.ndx", help="Index file to append to")
args = parser.parse_args()

# Read structural residue numbers from ion_residues.gs and ligand_residues.gs
struct_residues = set()
for gsfile in ["ion_residues.gs", "ligand_residues.gs"]:
    if os.path.isfile(gsfile):
        with open(gsfile) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    try:
                        struct_residues.add(int(line))
                    except ValueError:
                        pass

# Parse GRO file: collect atom numbers for Protein residues + structural ion/ligand residues
# Protein residues = standard amino acids + caps (ACE, NME)
PROTEIN_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HIE', 'HID', 'HIP', 'CYX', 'CYM',  # AMBER variants
    'HSD', 'HSE', 'HSP',  # CHARMM variants
    'HISA', 'HISB', 'HISH', 'HISD', 'HISE', 'HISP',  # GROMOS variants
    'CYSH', 'CYS1', 'CYS2',  # GROMOS CYS variants
    'ASH', 'GLH', 'ASPH', 'GLUH', 'LYSH', 'ARGN',  # protonation variants
    'ACE', 'NME', 'NHE',  # terminal caps
    'NALA', 'NARG', 'NASN', 'NASP', 'NCYS', 'NGLN', 'NGLU', 'NGLY', 'NHIS',
    'NILE', 'NLEU', 'NLYS', 'NMET', 'NPHE', 'NPRO', 'NSER', 'NTHR', 'NTRP',
    'NTYR', 'NVAL',  # GROMOS N-terminal
    'CALA', 'CARG', 'CASN', 'CASP', 'CCYS', 'CGLN', 'CGLU', 'CGLY', 'CHIS',
    'CILE', 'CLEU', 'CLYS', 'CMET', 'CPHE', 'CPRO', 'CSER', 'CTHR', 'CTRP',
    'CTYR', 'CVAL',  # GROMOS C-terminal
}

# Add NCAA residue names (if parametrized) to the protein set
if os.path.isfile("ncaa_residues.gs"):
    with open("ncaa_residues.gs") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                PROTEIN_RESNAMES.add(line)
                print(f"Added NCAA residue {line} to protein group")

atom_numbers = []
with open(args.gro) as f:
    lines = f.readlines()

for line in lines[2:-1]:  # skip header, count, and box line
    if len(line) < 20:
        continue
    try:
        # GRO fixed-width: resnum(0:5) resname(5:10) atomname(10:15) atomnum(15:20)
        resnum = int(line[0:5])
        resname = line[5:10].strip()
        atomnum = int(line[15:20])
    except (ValueError, IndexError):
        continue

    # Include if protein residue or structural ion/ligand
    if resname in PROTEIN_RESNAMES or resnum in struct_residues:
        atom_numbers.append(atomnum)

if not atom_numbers:
    print("Warning: no atoms found for Protein_Struct group")
else:
    # Append group to index.ndx
    with open(args.ndx, "a") as f:
        f.write("[ Protein_Struct ]\n")
        for i, anum in enumerate(atom_numbers):
            f.write(f"{anum:>6d}")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(atom_numbers) % 15 != 0:
            f.write("\n")
    print(f"Added Protein_Struct group: {len(atom_numbers)} atoms "
          f"(protein + {len(struct_residues)} structural ion/ligand residues)")
