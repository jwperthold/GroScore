#!/usr/bin/env python3
"""
Setup benchmark structures from the HADDOCKING protein-protein affinity benchmark.
Downloads PDBs, extracts relevant chains, and creates sp.gs file.
"""

import csv
import os
import urllib.request
import sys

# Read the benchmark CSV
structures = []
with open('benchmark.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pdb_id = row['pdb_id'].upper()
        chain_id_1 = row['chain_id_1']
        chain_id_2 = row['chain_id_2']
        structures.append((pdb_id, chain_id_1, chain_id_2))

print(f"Found {len(structures)} structures in benchmark")

# Create sp.gs file
with open('sp.gs', 'w') as sp:
    sp.write("# Structure_ID  Chains_for_Protein_B\n")

    for pdb_id, chain_id_1, chain_id_2 in structures:
        # Create directory
        if not os.path.exists(pdb_id):
            os.makedirs(pdb_id)

        input_pdb = f"{pdb_id}/input.pdb"

        if os.path.exists(input_pdb):
            print(f"{pdb_id}: input.pdb already exists, skipping download")
            sp.write(f"{pdb_id}\t{chain_id_2}\n")
            continue

        # Download PDB from RCSB
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {pdb_id}...", end=" ", flush=True)

        try:
            with urllib.request.urlopen(pdb_url, timeout=30) as response:
                pdb_content = response.read().decode('utf-8')
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Extract only the chains we need
        chains_to_keep = set(chain_id_1) | set(chain_id_2)

        extracted_lines = []
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Chain ID is at position 21 (0-indexed)
                if len(line) > 21:
                    chain = line[21]
                    if chain in chains_to_keep:
                        extracted_lines.append(line)
            elif line.startswith('TER'):
                # Keep TER records for chains we're keeping
                if len(line) > 21:
                    chain = line[21]
                    if chain in chains_to_keep:
                        extracted_lines.append(line)
                elif extracted_lines:  # Keep TER if it follows our atoms
                    extracted_lines.append(line)
            elif line.startswith('END'):
                extracted_lines.append(line)

        if not extracted_lines:
            print(f"FAILED: No atoms found for chains {chains_to_keep}")
            continue

        # Write extracted PDB
        with open(input_pdb, 'w') as out:
            out.write('\n'.join(extracted_lines))
            if not extracted_lines[-1].startswith('END'):
                out.write('\nEND\n')

        print(f"OK ({len(extracted_lines)} lines, chains: {','.join(sorted(chains_to_keep))})")

        # Add to sp.gs
        sp.write(f"{pdb_id}\t{chain_id_2}\n")

print(f"\nCreated sp.gs with {len(structures)} entries")
print("Run: python ../groscore.py")
