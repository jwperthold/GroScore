#!/usr/bin/env python3
#
# cap_termini.py - Add ACE/NME capping residues to all fragment termini
#
# Usage: python cap_termini.py -f input.pdb -o output.pdb [-m chain_map.gs]
#

import os
import sys
import argparse
import tempfile
from pdbfixer import PDBFixer
from openmm import unit

# Ion residue names supported by all GroScore force fields
ION_RESIDUES = {"BA", "CA", "CD", "CL", "CO", "CS", "CU", "CU1", "FE", "FE2", "HG", "K", "LI", "MG", "MN", "NA", "NI", "PB", "SD", "SR", "ZN"}

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
original_chain_header = ""  # preserve "chain(s): B,C" line for merge_ligand.py
if args.chainmap and os.path.isfile(args.chainmap):
    with open(args.chainmap, "r") as f:
        for line in f:
            if "chain(s):" in line:
                original_chain_header = line.rstrip()
            elif not line.strip().startswith("#"):
                try:
                    original_b_resnums.add(int(line.strip()))
                except (ValueError, IndexError):
                    pass

# Read original ion_residues.gs (will be updated with new residue numbers)
original_ion_resnums = set()
ion_residues_path = os.path.join(os.path.dirname(args.chainmap) if args.chainmap else ".", "ion_residues.gs")
if os.path.isfile(ion_residues_path):
    with open(ion_residues_path, "r") as f:
        for line in f:
            if not line.strip().startswith("#"):
                try:
                    original_ion_resnums.add(int(line.strip()))
                except (ValueError, IndexError):
                    pass

# Read original PDB to map residue numbers to chain IDs
# We need this to determine protein A/B membership for ACE/NME caps
# Also separate ion lines (PDBFixer cannot handle single-atom ion residues)
orig_resnum_to_chain = {}
ion_lines = []
with open(args.file, "r") as f:
    for line in f:
        if line.startswith("ATOM"):
            chain_id = line[21]
            resname = line[17:20].strip()
            try:
                resnum = int(line[22:26].strip())
                if resnum not in orig_resnum_to_chain:
                    orig_resnum_to_chain[resnum] = chain_id
                if resname in ION_RESIDUES:
                    ion_lines.append(line)
            except (ValueError, IndexError):
                pass

# Write protein-only PDB for PDBFixer (strip ions and their surrounding TER records)
# Removing TER records adjacent to ions prevents PDBFixer from seeing a false chain break
protein_only_path = None
if ion_lines:
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
    protein_only_path = tmp.name
    # Read all lines to identify which TER records are adjacent to ion ATOM lines
    all_lines = open(args.file).readlines()
    skip_lines = set()
    for i, line in enumerate(all_lines):
        if line.startswith("ATOM"):
            resname = line[17:20].strip()
            if resname in ION_RESIDUES:
                skip_lines.add(i)
                # Also skip TER records immediately before and after the ion
                if i > 0 and all_lines[i - 1].startswith("TER"):
                    skip_lines.add(i - 1)
                if i + 1 < len(all_lines) and all_lines[i + 1].startswith("TER"):
                    skip_lines.add(i + 1)
    for i, line in enumerate(all_lines):
        if i not in skip_lines:
            tmp.write(line)
    tmp.close()
    pdbfixer_input = protein_only_path
else:
    pdbfixer_input = args.file

# Load PDB with PDBFixer
fixer = PDBFixer(filename=pdbfixer_input)
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

# Strip extra terminal atoms that conflict with caps
# PDBFixer adds ACE/NME but doesn't remove N-terminal H2/H3 or C-terminal OXT
# from the adjacent residue, causing template mismatches in OpenMM
from openmm.app import Modeller as _Modeller
atoms_to_delete = []
for chain in fixer.topology.chains():
    residues = list(chain.residues())
    if len(residues) < 2:
        continue
    # If first residue is ACE, strip extra N-terminal H from second residue
    if residues[0].name == 'ACE':
        next_res = residues[1]
        for atom in next_res.atoms():
            if atom.name in ('H2', 'H3'):
                atoms_to_delete.append(atom)
    # If last residue is NME/NHE, strip OXT from second-to-last residue
    if residues[-1].name in ('NME', 'NHE'):
        prev_res = residues[-2]
        for atom in prev_res.atoms():
            if atom.name == 'OXT':
                atoms_to_delete.append(atom)
if atoms_to_delete:
    modeller_cleanup = _Modeller(fixer.topology, fixer.positions)
    modeller_cleanup.delete(atoms_to_delete)
    fixer.topology = modeller_cleanup.topology
    fixer.positions = modeller_cleanup.positions
    print(f"Removed {len(atoms_to_delete)} extra terminal atom(s) conflicting with caps")

# Brief OpenMM energy minimization of cap atoms to resolve steric clashes
# PDBFixer places ACE/NME atoms from templates without checking for clashes,
# which can result in cap hydrogens overlapping backbone atoms (e.g., ACE H1 on CA)
import openmm
from openmm import app as mmapp

cap_residue_names = {'ACE', 'NME', 'NHE'}
cap_atom_indices = set()
for atom in fixer.topology.atoms():
    if atom.residue.name in cap_residue_names:
        cap_atom_indices.add(atom.index)

if cap_atom_indices:
    # Build modeller with hydrogens for AMBER14 compatibility
    # May fail for exotic residues not in AMBER14 templates — fall back gracefully
    try:
        ff_mm = mmapp.ForceField('amber14-all.xml')
        modeller = mmapp.Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(ff_mm)

        # Build mapping: fixer atom index -> modeller atom index (by residue index + atom name)
        # Modeller.addHydrogens() inserts H atoms between existing atoms, so indices don't match
        mm_atom_lookup = {}
        for atom in modeller.topology.atoms():
            key = (atom.residue.index, atom.name)
            mm_atom_lookup[key] = atom.index

        fixer_to_mm = {}
        for atom in fixer.topology.atoms():
            key = (atom.residue.index, atom.name)
            if key in mm_atom_lookup:
                fixer_to_mm[atom.index] = mm_atom_lookup[key]

        # Identify cap atoms in the modeller topology
        cap_indices_mm = set()
        for atom in modeller.topology.atoms():
            if atom.residue.name in cap_residue_names:
                cap_indices_mm.add(atom.index)

        system = ff_mm.createSystem(modeller.topology, nonbondedMethod=mmapp.NoCutoff,
                                    constraints=None, rigidWater=False)

        # Restrain all non-cap atoms with a strong harmonic potential
        restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k", 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        mm_positions = modeller.positions
        for i in range(system.getNumParticles()):
            if i not in cap_indices_mm:
                pos = mm_positions[i]
                restraint.addParticle(i, [pos.x, pos.y, pos.z])
        system.addForce(restraint)

        integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('CPU'))
        context.setPositions(mm_positions)
        openmm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=200)

        # Map minimized positions back to fixer topology using residue+name mapping
        min_positions = context.getState(getPositions=True).getPositions()
        new_positions = list(fixer.positions)
        for fixer_idx, mm_idx in fixer_to_mm.items():
            new_positions[fixer_idx] = min_positions[mm_idx]
        fixer.positions = new_positions

        n_caps = len([r for r in fixer.topology.residues() if r.name in cap_residue_names])
        print(f"Minimized {n_caps} cap residue(s) to resolve steric clashes")
    except Exception as e:
        print(f"Warning: Cap minimization failed ({e}), using PDBFixer positions as-is")

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
new_ion_resnums = set()
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

    # Re-insert structural ion lines
    if ion_lines:
        for ion_line in ion_lines:
            resname = ion_line[17:20].strip()
            resnum = int(ion_line[22:26].strip())
            chain_id = ion_line[21]
            atomname = ion_line[12:16].strip()
            x_coord = float(ion_line[30:38])
            y_coord = float(ion_line[38:46])
            z_coord = float(ion_line[46:54])

            resnum_counter += 1
            atom_num += 1

            # Determine protein B membership
            if resnum in original_b_resnums:
                new_b_resnums.add(resnum_counter)

            # Track ion residue number mapping
            if resnum in original_ion_resnums:
                new_ion_resnums.add(resnum_counter)

            if len(atomname) < 4:
                atomname_fmt = f" {atomname:<3s}"
            else:
                atomname_fmt = f"{atomname:<4s}"

            out.write("TER\n")
            out.write(f"ATOM  {atom_num:5d} {atomname_fmt} {resname:>3s} {chain_id}{resnum_counter:4d}    "
                      f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
                      f"  1.00  0.00          {atomname[0]:>2s}\n")

    out.write("END\n")

# Clean up temp file
if protein_only_path:
    os.unlink(protein_only_path)

# Update chain_map.gs if requested
if args.chainmap:
    with open(args.chainmap, "w") as f:
        f.write("# Sequential residue numbers belonging to protein B\n")
        if original_chain_header:
            f.write(f"{original_chain_header}\n")
        cap_label = "ACE caps" if args.ace_only else "ACE/NME caps"
        f.write(f"# Updated by cap_termini.py with {cap_label}\n")
        for resnum in sorted(new_b_resnums):
            f.write(f"{resnum}\n")

# Update ion_residues.gs with new residue numbers
if new_ion_resnums and os.path.isfile(ion_residues_path):
    with open(ion_residues_path, "w") as f:
        f.write("# Sequential residue numbers of structural ions\n")
        f.write("# Updated by cap_termini.py\n")
        for resnum in sorted(new_ion_resnums):
            f.write(f"{resnum}\n")

cap_type = "ACE caps (no NME)" if args.ace_only else "ACE/NME caps"
print(f"Added {cap_type}, wrote {args.output} ({resnum_counter} residues, {atom_num} atoms)")
if args.chainmap:
    print(f"Updated {args.chainmap} with {len(new_b_resnums)} residues for protein B")
if new_ion_resnums:
    print(f"Updated {ion_residues_path} with {len(new_ion_resnums)} ion(s)")
