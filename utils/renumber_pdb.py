#!/usr/bin/env python3
#
# Renumber PDB residues sequentially across all chains
# This ensures unique residue numbers in the output GRO file
#

import os
import sys
import math
import argparse

# Ion residue names supported by all GroScore force fields
ION_RESIDUES = {"ZN", "CA", "MG", "CU", "CU1", "NA", "CL"}

# HETATM residues to skip entirely (handled separately or irrelevant)
# HOH = crystal waters (extracted to crystal_waters.pdb)
SKIP_HETATM = {"HOH"}

# Standard amino acid residue names (3-letter codes) — anything else in HETATM is a ligand
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Common variants
    "HIE", "HID", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
    "HISA", "HISB", "HISH", "HISD", "HISE", "HISP",
    "CYSH", "CYS1", "CYS2", "ASPH", "GLUH", "LYSH", "ARGN",
    # Caps
    "ACE", "NME", "NHE", "NH2",
}

parser = argparse.ArgumentParser(description="Renumber PDB residues sequentially and output chain mapping.")
parser.add_argument('-f','--pdbfile', type=str, required=True, help="Input PDB file.")
parser.add_argument('-o','--output', type=str, default="renumbered.pdb", help="Output PDB file.")
parser.add_argument('-c','--chains', type=str, required=True, help="Comma-separated chain IDs for protein B.")
args = parser.parse_args()

chains_b = set(c.strip() for c in args.chains.split(','))

if not os.path.isfile(args.pdbfile):
    print(f"Error: PDB file '{args.pdbfile}' not found.", file=sys.stderr)
    sys.exit(1)

# Pre-scan: identify HETATM residues that have backbone atoms (N, CA, C, O)
# These are modified amino acids (e.g., TRQ, TPO, SEP, MSE, HYP, MLY, CSO, PTR)
# and should be treated as protein, not ligands
MODIFIED_AA_RESIDUES = set()
with open(args.pdbfile) as f:
    for line in f:
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            atomname = line[12:16].strip()
            if resname not in ION_RESIDUES and resname not in SKIP_HETATM:
                if atomname == 'CA':
                    MODIFIED_AA_RESIDUES.add(resname)
if MODIFIED_AA_RESIDUES:
    print(f"Detected modified amino acids (HETATM with backbone): {', '.join(sorted(MODIFIED_AA_RESIDUES))}")

def is_atom_or_ion(line):
    """Check if line is an ATOM record or a HETATM record for a supported ion or modified amino acid."""
    if line.startswith("ATOM"):
        return True
    if line.startswith("HETATM"):
        resname = line[17:20].strip()
        if resname in ION_RESIDUES or resname in MODIFIED_AA_RESIDUES:
            return True
    return False

def get_resname(line):
    """Extract residue name from PDB ATOM/HETATM line."""
    return line[17:20].strip()

# First pass: collect all unique (chain, resnum) pairs, atom coordinates,
# and assign sequential numbers
seen_residues = {}  # (chain, resnum) -> new_resnum
current_resnum = 0
last_key = None
prev_chain = None
prev_orig_resnum = None
prev_is_ion = False

# Also track which new residue numbers belong to protein B
residues_b = set()

# Track which new residue numbers are ions
ion_resnums = set()

# Track gaps in residue numbering (potential chain breaks)
numbering_gaps = set()  # Set of (chain, orig_resnum) keys that follow a gap

# Collect C and N atom coordinates for peptide bond distance checking
c_atoms = {}   # (chain, resnum) -> (x, y, z) for backbone C atom
n_atoms = {}   # (chain, resnum) -> (x, y, z) for backbone N atom

with open(args.pdbfile, "r") as f:
    for line in f:
        if is_atom_or_ion(line):
            chain_id = line[21]
            resname = get_resname(line)
            is_ion = resname in ION_RESIDUES
            try:
                orig_resnum = int(line[22:26].strip())
                atomname = line[12:16].strip()
                key = (chain_id, orig_resnum)

                # Collect backbone C and N coordinates (skip for ions)
                if not is_ion:
                    if atomname == 'C':
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        c_atoms[(chain_id, orig_resnum)] = (x, y, z)
                    elif atomname == 'N':
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        n_atoms[(chain_id, orig_resnum)] = (x, y, z)

                if key != last_key:
                    # Detect gaps in original residue numbering within same chain
                    # Skip gap detection when transitioning to/from ion residues
                    if (prev_chain == chain_id and prev_orig_resnum is not None
                            and not is_ion and not prev_is_ion):
                        if orig_resnum > prev_orig_resnum + 1:
                            numbering_gaps.add(key)
                    current_resnum += 1
                    seen_residues[key] = current_resnum
                    if chain_id in chains_b:
                        residues_b.add(current_resnum)
                    if is_ion:
                        ion_resnums.add(current_resnum)
                    prev_chain = chain_id
                    prev_orig_resnum = orig_resnum
                    prev_is_ion = is_ion
                    last_key = key
            except (ValueError, IndexError):
                pass

# Determine real chain breaks by checking C-N peptide bond distance
# A peptide bond is ~1.33 Å; use 2.0 Å cutoff to be safe
PEPTIDE_BOND_CUTOFF = 2.0
chain_breaks = set()

# Build per-chain sorted residue lists for efficient lookup
chain_resnums = {}
for (c, r) in seen_residues.keys():
    chain_resnums.setdefault(c, []).append(r)
for c in chain_resnums:
    chain_resnums[c].sort()

for key in numbering_gaps:
    chain_id, orig_resnum = key
    # Find the previous residue in the same chain
    resnums = chain_resnums.get(chain_id, [])
    idx = resnums.index(orig_resnum) if orig_resnum in resnums else -1
    if idx <= 0:
        chain_breaks.add(key)
        continue
    prev_resnum = resnums[idx - 1]

    # Check C-N distance
    c_coord = c_atoms.get((chain_id, prev_resnum))
    n_coord = n_atoms.get((chain_id, orig_resnum))

    if c_coord and n_coord:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c_coord, n_coord)))
        if dist > PEPTIDE_BOND_CUTOFF:
            chain_breaks.add(key)
            print(f"Chain break: {chain_id} {prev_resnum} -> {orig_resnum} (C-N distance: {dist:.1f} A)")
        else:
            print(f"Connected (non-sequential numbering): {chain_id} {prev_resnum} -> {orig_resnum} (C-N distance: {dist:.1f} A)")
    else:
        # Cannot determine - assume chain break
        chain_breaks.add(key)

# Second pass: write renumbered PDB
# HETATM ions are converted to ATOM records for downstream compatibility
# TER records are inserted at chain breaks and before/after each ion residue
written_breaks = set()
last_key = None
last_was_ion = False
with open(args.pdbfile, "r") as fin, open(args.output, "w") as fout:
    for line in fin:
        if is_atom_or_ion(line):
            chain_id = line[21]
            resname = get_resname(line)
            is_ion = resname in ION_RESIDUES
            try:
                orig_resnum = int(line[22:26].strip())
                key = (chain_id, orig_resnum)
                new_resnum = seen_residues.get(key, orig_resnum)

                # Add TER before chain breaks
                if key in chain_breaks and key not in written_breaks:
                    fout.write("TER\n")
                    written_breaks.add(key)

                # Add TER before ion (if previous was not a TER-inducing break)
                if is_ion and last_key is not None and key != last_key and not last_was_ion:
                    if key not in written_breaks:
                        fout.write("TER\n")

                # Add TER after ion (before next protein residue)
                if not is_ion and last_was_ion and last_key is not None and key != last_key:
                    if key not in written_breaks:
                        fout.write("TER\n")

                # Write as ATOM record (convert HETATM to ATOM for ions)
                new_line = "ATOM  " + line[6:22] + f"{new_resnum:4d}" + line[26:]
                fout.write(new_line)
                last_key = key
                last_was_ion = is_ion
            except (ValueError, IndexError):
                fout.write(line)
        elif line.startswith("TER") or line.startswith("END"):
            fout.write(line)
        # Skip other HETATM, CONECT, and non-protein records

# Write chain_map.gs with the new sequential residue numbers for protein B
with open("chain_map.gs", "w") as f:
    f.write("# Sequential residue numbers belonging to protein B\n")
    f.write(f"# Generated from {args.pdbfile} for chain(s): {args.chains}\n")
    for resnum in sorted(residues_b):
        f.write(f"{resnum}\n")

# Write ion_residues.gs with sequential residue numbers of structural ions
with open("ion_residues.gs", "w") as f:
    f.write("# Sequential residue numbers of structural ions\n")
    f.write(f"# Generated from {args.pdbfile}\n")
    for resnum in sorted(ion_resnums):
        f.write(f"{resnum}\n")

num_ions = len(ion_resnums)
print(f"Renumbered {current_resnum} residues ({num_ions} ion(s)), wrote {args.output}")
print(f"Detected {len(numbering_gaps)} numbering gap(s), {len(chain_breaks)} real chain break(s) (C-N > {PEPTIDE_BOND_CUTOFF} A)")
print(f"Generated chain_map.gs with {len(residues_b)} residues for chain(s) {args.chains}")
if num_ions > 0:
    print(f"Generated ion_residues.gs with {num_ions} structural ion(s)")

# Third pass: extract ligands and crystal waters from HETATM records
# These are handled separately from the protein pipeline
ligand_lines = {}   # (resname, chain) -> [lines]
water_lines = []
with open(args.pdbfile, "r") as f:
    for line in f:
        if line.startswith("HETATM"):
            resname = get_resname(line)
            chain_id = line[21]

            if resname in ION_RESIDUES or resname in MODIFIED_AA_RESIDUES:
                continue  # Already handled above (ions and modified AAs are part of protein)
            elif resname in SKIP_HETATM:
                if resname == "HOH":
                    water_lines.append(line)
            elif resname not in STANDARD_RESIDUES:
                key = (resname, chain_id)
                if key not in ligand_lines:
                    ligand_lines[key] = []
                ligand_lines[key].append(line)

# Write ligand PDB files
ligand_info = []
for (resname, chain_id), lines in ligand_lines.items():
    lig_pdb = f"ligand_{resname}_{chain_id}.pdb"
    with open(lig_pdb, "w") as f:
        for line in lines:
            f.write(line)
        f.write("END\n")
    ligand_info.append((resname, chain_id, len(lines)))
    print(f"Extracted ligand {resname} (chain {chain_id}, {len(lines)} atoms) to {lig_pdb}")

# Write ligand_info.gs
if ligand_info:
    with open("ligand_info.gs", "w") as f:
        f.write("# Ligand residue name, chain, number of heavy atoms\n")
        for resname, chain_id, natoms in ligand_info:
            f.write(f"{resname}\t{chain_id}\t{natoms}\n")

# Write crystal waters PDB
if water_lines:
    with open("crystal_waters.pdb", "w") as f:
        for line in water_lines:
            f.write(line)
        f.write("END\n")
    print(f"Extracted {len(water_lines)} crystal water atom(s) to crystal_waters.pdb")
