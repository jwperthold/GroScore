#!/usr/bin/env python3
#
# fix_pdb.py - Fix missing atoms and non-standard residues in PDB files
#
# Usage: python fix_pdb.py -f input.pdb -o output.pdb
#

import argparse
import tempfile
import os
import sys
import warnings
import openmm
from pdbfixer import PDBFixer
from openmm.app import PDBFile

# Ion residue names supported by all GroScore force fields
ION_RESIDUES = {"BA", "CA", "CD", "CL", "CO", "CS", "CU", "CU1", "FE", "FE2", "HG", "K", "LI", "MG", "MN", "NA", "NI", "PB", "SD", "SR", "ZN"}
# PDBFixer's addMissingAtoms() minimises the atoms it adds and, where one still sits
# closer than 0.13 nm to another atom, runs Langevin dynamics whose integrator takes a
# random seed unless it is given one. Unseeded, two setups of one structure differed
# wherever side-chain atoms were missing: 14 atom lines of 1E6E's fixed.pdb (GLU A 1,
# ASP B 458, 492 and 562, the OXT of SER B 570), and with them its caps, cutout and
# results; of the 46 HADDOCK inputs 1E6E, 1GHQ, 1H1V, 1IBR, 1OPH, 2HLE and 1Z0K. The
# seed alone did not settle it: OpenMM ran PDBFixer on CUDA where present, and seeded
# runs of 2HLE (three runs, three outputs), 1Z0K and once the 1E6E extract still
# differed. With the seed, the CPU platform and one thread, as cap_termini.py runs,
# every run gives the same bytes (tests/test_fix_pdb_seed.py); 2HLE, whose many
# missing side chains need the dynamics, takes 16 s instead of 1 s.
ADD_ATOMS_SEED = 1
CPU = openmm.Platform.getPlatformByName('CPU')
CPU.setPropertyDefaultValue('Threads', '1')

parser = argparse.ArgumentParser(description="Fix PDB file with PDBFixer")
parser.add_argument('-f', '--file', type=str, required=True, help="Input PDB file")
parser.add_argument('-o', '--output', type=str, required=True, help="Output PDB file")
parser.add_argument('--keep-ncaa', type=str, default='', help="Comma-separated NCAA residue names to preserve (skip replacement)")
args = parser.parse_args()

# NCAA residues to keep (not replace with standard equivalents)
ncaa_keep = set(r.strip() for r in args.keep_ncaa.split(',') if r.strip())

# A failed run must not leave the previous run's output for cap_termini.py to use.
if os.path.exists(args.output):
    os.remove(args.output)

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
# PDBFixer only processes ATOM records for non-standard residue replacement;
# HETATM modified residues would be ignored, creating chain breaks.
# This conversion is unconditional — for AMBER+NCAA, the --keep-ncaa filter
# later prevents replacement. For other FFs, PDBFixer replaces them with
# the parent residue (e.g., TRQ → TRP).
MODIFIED_AA_BACKBONE = {'N', 'CA', 'C', 'O', 'CB'}
for i, line in enumerate(all_lines):
    if line.startswith("HETATM") and i not in skip_lines:
        atomname = line[12:16].strip()
        if atomname in MODIFIED_AA_BACKBONE:
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
    # Run PDBFixer on protein-only PDB.
    # OpenMM's PDB reader merges two residues that share chain, number and
    # insertion code into one, keeps one copy of every repeated atom and only warns
    # 'duplicate atom'. That deleted a Ser from 2FD6 chain H and from 1IQD chain B
    # (Kabat 82A-82C written without insertion codes), and cap_termini.py then
    # closed the 4.5 A backbone gap by dragging the next residue 2.3 A. A duplicate
    # atom is therefore an error, never a warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fixer = PDBFixer(filename=tmp_path, platform=CPU)   # CPU, one thread: see ADD_ATOMS_SEED
    dups = [str(w.message) for w in caught if "duplicate atom" in str(w.message)]
    if dups:
        sys.stderr.write(
            "fix_pdb: ERROR - %d duplicate atom(s) in %s: two residues share chain, residue number "
            "and insertion code, and the PDB reader would merge them into one. First: %s\n"
            "  Renumber with utils/renumber_pdb.py, which separates residues by insertion code, "
            "residue name and repeated atom names, or correct the input.\n"
            % (len(dups), args.file, dups[0].replace("WARNING: ", "")))
        sys.exit(1)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    # Filter out NCAA residues from replacement list (keep them as-is for OpenFF parametrization)
    if ncaa_keep and fixer.nonstandardResidues:
        kept = [(res, std) for res, std in fixer.nonstandardResidues if res.name not in ncaa_keep]
        skipped = [(res, std) for res, std in fixer.nonstandardResidues if res.name in ncaa_keep]
        if skipped:
            print(f"Keeping {len(skipped)} NCAA residue(s): {', '.join(r.name for r, _ in skipped)}")
        fixer.nonstandardResidues = kept
    # Add modified AAs that PDBFixer doesn't know about to its replacement list
    # PDBFixer only recognizes ~100 common modifications; others (e.g., TRQ) are missed.
    # Look up parent residue from RCSB CCD and add to the list before replacement.
    STANDARD_3 = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                   'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}
    already_listed = {res.name for res, _ in fixer.nonstandardResidues}
    unknown_ncaa = {}  # resname -> parent
    for res in fixer.topology.residues():
        rn = res.name
        if rn in STANDARD_3 or rn in ION_RESIDUES or rn in already_listed or rn in ncaa_keep:
            continue
        if rn in {'ACE', 'NME', 'NHE', 'HOH'}:
            continue
        atom_names = {a.name for a in res.atoms()}
        if {'N', 'CA', 'C', 'O'} <= atom_names and rn not in unknown_ncaa:
            unknown_ncaa[rn] = None
    if unknown_ncaa:
        import urllib.request
        for resname in list(unknown_ncaa.keys()):
            try:
                url = f"https://files.rcsb.org/ligands/download/{resname}.cif"
                cif_data = urllib.request.urlopen(url, timeout=10).read().decode()
                for line in cif_data.split('\n'):
                    if '_chem_comp.mon_nstd_parent_comp_id' in line:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[1] != '?':
                            unknown_ncaa[resname] = parts[1]
                            break
            except Exception:
                pass
        for resname, parent in unknown_ncaa.items():
            if parent and parent in STANDARD_3:
                for res in fixer.topology.residues():
                    if res.name == resname:
                        fixer.nonstandardResidues.append((res, parent))
                print(f"Added {resname} → {parent} to PDBFixer replacement list (from RCSB CCD)")
            else:
                print(f"Warning: could not determine parent for {resname}")
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=ADD_ATOMS_SEED)

    # Write fixed protein
    with open(args.output, "w") as out:
        PDBFile.writeFile(fixer.topology, fixer.positions, out, keepIds=True)

    # Convert NCAA HETATM back to ATOM and strip PDBFixer-added H atoms
    # PDBFixer's addMissingAtoms() may add wrong H to NCAAs (e.g., HG1 on
    # phosphorylated OG1 in TPO). These will be re-added correctly by pdb2gmx
    # via the HDB from parametrize_ncaa.py.
    if ncaa_keep:
        with open(args.output) as f:
            content = f.read()
        with open(args.output, 'w') as f:
            for line in content.split('\n'):
                if line.startswith(('ATOM', 'HETATM')):
                    resname = line[17:20].strip()
                    if resname in ncaa_keep:
                        # Convert HETATM to ATOM
                        if line.startswith('HETATM'):
                            line = 'ATOM  ' + line[6:]
                        # Strip H atoms — pdb2gmx with -ignh re-adds from HDB
                        atomname = line[12:16].strip()
                        if atomname.startswith('H'):
                            continue
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
