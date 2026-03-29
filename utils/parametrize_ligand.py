#!/usr/bin/env python3
#
# parametrize_ligand.py - Parametrize a small molecule with OpenFF for GROMACS
#
# Reads a ligand from PDB HETATM records, determines bond orders using OpenBabel,
# assigns protonation at physiological pH, parametrizes with OpenFF, and exports
# GROMACS-compatible topology (.itp) and coordinate (.gro) files.
#
# Requires: openbabel, rdkit, openff-toolkit, openff-interchange (groscore conda env)
#
# Usage: python parametrize_ligand.py -f ligand_EF6_A.pdb -r EF6
#

import os
import sys
import argparse
import tempfile

parser = argparse.ArgumentParser(description="Parametrize a small molecule with OpenFF for GROMACS.")
parser.add_argument('-f', '--file', type=str, required=True, help="Input PDB file with ligand HETATM records")
parser.add_argument('-r', '--resname', type=str, required=True, help="Residue name of the ligand")
parser.add_argument('-c', '--chain', type=str, default="", help="Chain ID (for per-instance output filenames)")
parser.add_argument('-i', '--instance', type=int, default=0, help="Instance number (for unique molecule names across same resname)")
parser.add_argument('--ph', type=float, default=7.4, help="pH for protonation state assignment (default: 7.4)")
parser.add_argument('--ff', type=str, default="openff-2.2.1.offxml", help="OpenFF force field (default: openff-2.2.1.offxml)")
args = parser.parse_args()

if not os.path.isfile(args.file):
    print(f"Error: Input file '{args.file}' not found.", file=sys.stderr)
    sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
import urllib.request

with open(args.file) as f:
    pdb_content = f.read()

mol_rdkit = None

# Strategy 1: OpenBabel bond perception from 3D coordinates
# Works for any molecule (including user-designed, non-PDB ligands)
# Especially reliable when input has hydrogen coordinates
print("OpenBabel: perceiving bond orders from 3D coordinates...")

# Capture OpenBabel warnings to detect kekulization failures
ob_log = ob.OBMessageHandler()
ob.obErrorLog.SetOutputLevel(ob.obWarning)
ob.obErrorLog.StartLogging()

conv = ob.OBConversion()
conv.SetInFormat("pdb")
mol_ob = ob.OBMol()
conv.ReadString(mol_ob, pdb_content)
mol_ob.PerceiveBondOrders()

# Check for kekulization warning from OpenBabel
ob_warnings = ob.obErrorLog.GetMessagesOfLevel(ob.obWarning)
kekulization_ok = not any("kekulize" in w.lower() for w in ob_warnings)
ob.obErrorLog.StopLogging()

mol_ob.AddHydrogens(False, True, args.ph)  # polarOnly=False, correctForPH=True

total_charge = mol_ob.GetTotalCharge()
print(f"OpenBabel: {mol_ob.NumAtoms()} atoms (with H), charge={total_charge}")

if not kekulization_ok:
    print("Warning: OpenBabel failed to kekulize aromatic bonds")

conv.SetOutFormat("sdf")
sdf_block = conv.WriteString(mol_ob)
mol_rdkit = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=True)

if mol_rdkit is None or not kekulization_ok:
    # Strategy 2: RCSB Chemical Component Dictionary template (fallback)
    # Fixes tautomer issues by using the deposited bond orders
    print("Trying RCSB template for correct bond orders...")
    pdb_mol = Chem.MolFromPDBBlock(pdb_content, removeHs=True, sanitize=True)
    try:
        url = f"https://files.rcsb.org/ligands/download/{args.resname}_ideal.sdf"
        sdf_data = urllib.request.urlopen(url, timeout=15).read().decode()
        template = Chem.MolFromMolBlock(sdf_data, removeHs=True)
        if template is not None and pdb_mol is not None:
            mol_assigned = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
            mol_rdkit = Chem.AddHs(mol_assigned, addCoords=True)
            total_charge = Chem.GetFormalCharge(mol_rdkit)
            print(f"RCSB template: {mol_rdkit.GetNumAtoms()} atoms, charge={total_charge}, "
                  f"SMILES: {Chem.MolToSmiles(mol_rdkit)}")
    except Exception as e:
        print(f"RCSB template failed: {e}")
        if mol_rdkit is None:
            print("Error: could not determine bond orders.", file=sys.stderr)
            sys.exit(1)
        else:
            print("Warning: using OpenBabel result despite kekulization issue")

# Strip PDB residue metadata (OpenFF requires all-or-none residue info)
# Round-trip through SDF to get a clean mol without PDB annotations
sdf_block = Chem.MolToMolBlock(mol_rdkit)
mol_rdkit = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=True)

# Save SDF to disk for reference/debugging
sdf_path = f"ligand_{args.resname}.sdf"
with open(sdf_path, "w") as f:
    f.write(sdf_block)
print(f"Wrote {sdf_path}")

total_charge = Chem.GetFormalCharge(mol_rdkit)

print(f"RDKit: {mol_rdkit.GetNumAtoms()} atoms, SMILES: {Chem.MolToSmiles(mol_rdkit)}")

# Step 3: Create OpenFF Molecule
from openff.toolkit import Molecule, ForceField, Topology
from openff.interchange import Interchange
import numpy as np
from openff.units import unit

offmol = Molecule.from_rdkit(mol_rdkit, allow_undefined_stereo=True)
print(f"OpenFF: {offmol.n_atoms} atoms")

# Step 4: Parametrize with OpenFF
ff = ForceField(args.ff)
topology = Topology.from_molecules([offmol])
topology.box_vectors = np.eye(3) * 5.0 * unit.nanometer  # dummy box for export

interchange = Interchange.from_smirnoff(ff, topology)
print(f"Parametrized with {args.ff}")

# Step 5: Export to GROMACS
with tempfile.TemporaryDirectory() as tmpdir:
    prefix = os.path.join(tmpdir, "ligand")
    interchange.to_gromacs(prefix)

    # Step 6: Parse monolithic .top to extract [ atomtypes ] and [ moleculetype ]
    with open(f"{prefix}.top") as f:
        top_lines = f.readlines()

    # Find section boundaries
    sections = {}
    current_section = None
    for i, line in enumerate(top_lines):
        if line.strip().startswith("["):
            section_name = line.strip().strip("[]").strip()
            current_section = section_name
            if section_name not in sections:
                sections[section_name] = {"start": i, "end": len(top_lines)}
            # Close previous section
            for s in sections:
                if s != section_name and sections[s]["end"] == len(top_lines) and sections[s]["start"] < i:
                    sections[s]["end"] = i

    # Extract [ atomtypes ] section
    atomtypes_lines = []
    if "atomtypes" in sections:
        s = sections["atomtypes"]
        atomtypes_lines = top_lines[s["start"]:s["end"]]

    # Extract everything from [ moleculetype ] to just before [ system ]
    moltype_lines = []
    in_moltype = False
    for line in top_lines:
        section = line.strip().strip("[]").strip() if line.strip().startswith("[") else None
        if section == "moleculetype":
            in_moltype = True
        elif section in ("system", "molecules", "settles", "exclusions"):
            if section in ("settles", "exclusions") and in_moltype:
                # These are part of the moleculetype
                pass
            elif section in ("system", "molecules"):
                in_moltype = False
                continue
        if in_moltype:
            moltype_lines.append(line)

    # Rename molecule type from MOL0 to a per-instance name
    # Each instance gets its own ITP because different copies of the same ligand
    # may have different atom counts (e.g., truncated crystal fragments)
    instance_suffix = f"_{args.instance}" if args.instance > 0 else ""
    mol_name = f"{args.resname}{instance_suffix}"
    atomtypes_lines = [l.replace("MOL0", mol_name) for l in atomtypes_lines]
    moltype_lines = [l.replace("MOL0", mol_name) for l in moltype_lines]

    # Write atomtypes to separate file (must be included before any moleculetype)
    atomtypes_path = f"ligand_{mol_name}_atomtypes.itp"
    with open(atomtypes_path, "w") as f:
        f.write(f"; Atom types for ligand {args.resname} (instance {args.instance})\n")
        f.write(f"; Parametrized with OpenFF {args.ff}\n\n")
        for line in atomtypes_lines:
            f.write(line)

    # Write moleculetype to separate file
    itp_path = f"ligand_{mol_name}.itp"
    with open(itp_path, "w") as f:
        f.write(f"; Molecule type for ligand {args.resname} (instance {args.instance})\n")
        f.write(f"; Parametrized with OpenFF {args.ff}\n")
        f.write(f"; Total charge: {total_charge}\n\n")
        for line in moltype_lines:
            f.write(line)

    print(f"Wrote {itp_path} ({len(atomtypes_lines)} atomtype lines, {len(moltype_lines)} moleculetype lines)")

    # Step 7: Write ligand.gro with coordinates
    chain_suffix = f"_{args.chain}" if args.chain else ""
    gro_path = f"ligand_{mol_name}{chain_suffix}.gro"
    with open(f"{prefix}.gro") as f:
        gro_lines = f.readlines()

    with open(gro_path, "w") as f:
        # Header
        f.write(f"{args.resname} ligand\n")
        # Atom count
        n_atoms_gro = int(gro_lines[1].strip())
        f.write(f"{n_atoms_gro:5d}\n")
        # Atom lines (fix residue name from MOL0 to resname)
        for line in gro_lines[2:-1]:  # skip header, count, and box line
            # GRO format: resnum+resname in first 10 chars
            # Replace MOL0 with resname
            fixed = line.replace("MOL0", f"{args.resname:>4s}"[:4])
            f.write(fixed)
        # Box line (dummy)
        f.write(gro_lines[-1])

    print(f"Wrote {gro_path} ({n_atoms_gro} atoms)")

    # Append to ligand manifest for merge_ligand.py
    with open("ligand_manifest.gs", "a") as f:
        f.write(f"{mol_name}\t{itp_path}\t{atomtypes_path}\t{gro_path}\t{n_atoms_gro}\n")
