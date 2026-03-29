#!/usr/bin/env python3
#
# fix_pdb.py - Fix missing atoms and non-standard residues in PDB files
#
# Usage: python fix_pdb.py -f input.pdb -o output.pdb
#

import argparse
import tempfile
import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile

# Ion residue names supported by all GroScore force fields
ION_RESIDUES = {"ZN", "CA", "MG", "CU", "CU1", "NA", "CL"}

parser = argparse.ArgumentParser(description="Fix PDB file with PDBFixer")
parser.add_argument('-f', '--file', type=str, required=True, help="Input PDB file")
parser.add_argument('-o', '--output', type=str, required=True, help="Output PDB file")
parser.add_argument('--keep-ncaa', type=str, default='', help="Comma-separated NCAA residue names to preserve (skip replacement)")
args = parser.parse_args()

# NCAA residues to keep (not replace with standard equivalents)
ncaa_keep = set(r.strip() for r in args.keep_ncaa.split(',') if r.strip())

# Separate ion lines from protein lines before PDBFixer
# PDBFixer may drop or corrupt single-atom ion residues
# Also strip TER records adjacent to ions to avoid false chain breaks
all_lines = open(args.file).readlines()
ion_lines = []
skip_lines = set()
for i, line in enumerate(all_lines):
    if line.startswith("ATOM"):
        resname = line[17:20].strip()
        if resname in ION_RESIDUES:
            ion_lines.append(line)
            skip_lines.add(i)
            # Skip TER records immediately before and after the ion
            if i > 0 and all_lines[i - 1].startswith("TER"):
                skip_lines.add(i - 1)
            if i + 1 < len(all_lines) and all_lines[i + 1].startswith("TER"):
                skip_lines.add(i + 1)

# Convert HETATM records of modified amino acids to ATOM records
# PDBFixer only processes ATOM records; HETATM modified residues get dropped,
# creating chain breaks (e.g., TRQ/TPO/SEP/MSE/HYP/MLY/CSO/PTR)
# PDBFixer's replaceNonstandardResidues() will then convert them to standard residues
MODIFIED_AA_BACKBONE = {'N', 'CA', 'C', 'O', 'CB'}
for i, line in enumerate(all_lines):
    if line.startswith("HETATM") and i not in skip_lines:
        atomname = line[12:16].strip()
        # If it has backbone atoms (N, CA, C, O), it's likely a modified amino acid
        if atomname in MODIFIED_AA_BACKBONE:
            # Convert all atoms of this residue from HETATM to ATOM
            resname = line[17:20].strip()
            chain = line[21]
            resnum = line[22:26].strip()
            for j, line2 in enumerate(all_lines):
                if line2.startswith("HETATM") and j not in skip_lines:
                    if line2[17:20].strip() == resname and line2[21] == chain and line2[22:26].strip() == resnum:
                        all_lines[j] = "ATOM  " + line2[6:]

# Write protein-only PDB for PDBFixer
with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
    tmp_path = tmp.name
    for i, line in enumerate(all_lines):
        if i not in skip_lines:
            tmp.write(line)

try:
    # Run PDBFixer on protein-only PDB
    fixer = PDBFixer(filename=tmp_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    # Filter out NCAA residues from replacement list (keep them as-is)
    if ncaa_keep and fixer.nonstandardResidues:
        kept = [(res, std) for res, std in fixer.nonstandardResidues if res.name not in ncaa_keep]
        skipped = [(res, std) for res, std in fixer.nonstandardResidues if res.name in ncaa_keep]
        if skipped:
            print(f"Keeping {len(skipped)} NCAA residue(s): {', '.join(r.name for r, _ in skipped)}")
        fixer.nonstandardResidues = kept
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Write fixed protein
    with open(args.output, "w") as out:
        PDBFile.writeFile(fixer.topology, fixer.positions, out, keepIds=True)

    # Convert NCAA HETATM back to ATOM (PDBFixer outputs non-standard residues as HETATM)
    if ncaa_keep:
        with open(args.output) as f:
            content = f.read()
        with open(args.output, 'w') as f:
            for line in content.split('\n'):
                if line.startswith('HETATM'):
                    resname = line[17:20].strip()
                    if resname in ncaa_keep:
                        line = 'ATOM  ' + line[6:]
                f.write(line + '\n')

    # Append ion lines before END record
    if ion_lines:
        with open(args.output) as f:
            content = f.read()

        with open(args.output, "w") as out:
            # Insert ions before END line
            end_idx = content.rfind("END")
            if end_idx >= 0:
                out.write(content[:end_idx])
            else:
                out.write(content)

            # Write each ion with TER separators
            for ion_line in ion_lines:
                out.write("TER\n")
                out.write(ion_line)

            if end_idx >= 0:
                out.write("\nEND\n")

        print(f"Re-inserted {len(ion_lines)} ion atom(s) into {args.output}")
finally:
    os.unlink(tmp_path)
