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
parser.add_argument('--ph', type=float, default=7.4, help="pH for protonation state assignment (default: 7.4)")
parser.add_argument('--ff', type=str, default="openff-2.2.1.offxml", help="OpenFF force field (default: openff-2.2.1.offxml)")
args = parser.parse_args()

if not os.path.isfile(args.file):
    print(f"Error: Input file '{args.file}' not found.", file=sys.stderr)
    sys.exit(1)

# Step 1: OpenBabel reads PDB and perceives bond orders + protonation
from openbabel import openbabel as ob

conv = ob.OBConversion()
conv.SetInFormat("pdb")
mol_ob = ob.OBMol()

with open(args.file) as f:
    pdb_content = f.read()

conv.ReadString(mol_ob, pdb_content)
mol_ob.PerceiveBondOrders()
mol_ob.AddHydrogens(False, True, args.ph)  # polarOnly=False, correctForPH=True

total_charge = mol_ob.GetTotalCharge()
n_atoms = mol_ob.NumAtoms()
print(f"OpenBabel: {n_atoms} atoms (with H), charge={total_charge}")

# Write to SDF (preserves bond orders + 3D coordinates)
conv.SetOutFormat("sdf")
sdf_block = conv.WriteString(mol_ob)

# Step 2: RDKit reads SDF
from rdkit import Chem

mol_rdkit = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=True)
if mol_rdkit is None:
    print("Error: RDKit failed to read SDF from OpenBabel.", file=sys.stderr)
    sys.exit(1)

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

    # Rename molecule type from MOL0 to the residue name
    mol_name = args.resname
    atomtypes_lines = [l.replace("MOL0", mol_name) for l in atomtypes_lines]
    moltype_lines = [l.replace("MOL0", mol_name) for l in moltype_lines]

    # Write ligand.itp
    itp_path = f"ligand_{args.resname}.itp"
    with open(itp_path, "w") as f:
        f.write(f"; Ligand topology for {args.resname}\n")
        f.write(f"; Parametrized with OpenFF {args.ff}\n")
        f.write(f"; Total charge: {total_charge}\n\n")
        for line in atomtypes_lines:
            f.write(line)
        f.write("\n")
        for line in moltype_lines:
            f.write(line)

    print(f"Wrote {itp_path} ({len(atomtypes_lines)} atomtype lines, {len(moltype_lines)} moleculetype lines)")

    # Step 7: Write ligand.gro with coordinates
    # Read the OpenFF-generated GRO for coordinates (already in nm)
    gro_path = f"ligand_{args.resname}.gro"
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
