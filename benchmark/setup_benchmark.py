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
        structures.append((pdb_id, chain_id_1, chain_id_2, dict(row)))

print(f"Found {len(structures)} structures in benchmark")

# Create sp.gs file
with open('sp.gs', 'w') as sp:
    sp.write("# Structure_ID  Chains_for_Protein_B\n")

    for pdb_id, chain_id_1, chain_id_2, row in structures:
        # Create directory
        if not os.path.exists(pdb_id):
            os.makedirs(pdb_id)

        input_pdb = f"{pdb_id}/input.pdb"

        if os.path.exists(input_pdb):
            print(f"{pdb_id}: input.pdb already exists, skipping download")
            # Convert "AB" to "A,B" for multi-character chain specifications
            chain_id_2_formatted = ','.join(chain_id_2)
            sp.write(f"{pdb_id}\t{chain_id_2_formatted}\n")
            continue

        # Download mmCIF from RCSB (has both auth and label chain IDs)
        cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        print(f"Downloading {pdb_id}...", end=" ", flush=True)

        try:
            with urllib.request.urlopen(cif_url, timeout=30) as response:
                cif_content = response.read().decode('utf-8')
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Parse mmCIF _atom_site to get auth_asym_id and label_asym_id mapping
        # and extract ATOM records for the chains we need
        # mmCIF _atom_site columns vary by file; find the column indices
        in_atom_site = False
        columns = []
        atom_records = []
        for line in cif_content.split('\n'):
            if line.startswith('_atom_site.'):
                in_atom_site = True
                columns.append(line.strip())
            elif in_atom_site:
                if line.startswith('_') or line.startswith('#') or line.startswith('loop_'):
                    in_atom_site = False
                elif line.strip():
                    atom_records.append(line)

        # Find column indices
        col_idx = {col: i for i, col in enumerate(columns)}
        auth_chain_col = col_idx.get('_atom_site.auth_asym_id')
        label_chain_col = col_idx.get('_atom_site.label_asym_id')
        group_col = col_idx.get('_atom_site.group_PDB')  # ATOM or HETATM
        atom_name_col = col_idx.get('_atom_site.label_atom_id')  # or auth_atom_id
        if atom_name_col is None:
            atom_name_col = col_idx.get('_atom_site.auth_atom_id')
        comp_col = col_idx.get('_atom_site.label_comp_id')  # residue name
        if comp_col is None:
            comp_col = col_idx.get('_atom_site.auth_comp_id')
        seq_col = col_idx.get('_atom_site.auth_seq_id')
        x_col = col_idx.get('_atom_site.Cartn_x')
        y_col = col_idx.get('_atom_site.Cartn_y')
        z_col = col_idx.get('_atom_site.Cartn_z')
        element_col = col_idx.get('_atom_site.type_symbol')
        occ_col = col_idx.get('_atom_site.occupancy')
        bfactor_col = col_idx.get('_atom_site.B_iso_or_equiv')
        alt_col = col_idx.get('_atom_site.label_alt_id')
        model_col = col_idx.get('_atom_site.pdbx_PDB_model_num')

        if auth_chain_col is None or label_chain_col is None:
            print(f"FAILED: Could not find chain ID columns in mmCIF")
            continue

        # Build chain ID mapping: label -> auth (and vice versa)
        label_to_auth = {}
        auth_to_label = {}
        for rec in atom_records:
            fields = rec.split()
            if len(fields) <= max(auth_chain_col, label_chain_col):
                continue
            auth_ch = fields[auth_chain_col]
            label_ch = fields[label_chain_col]
            if label_ch not in label_to_auth:
                label_to_auth[label_ch] = auth_ch
            if auth_ch not in auth_to_label:
                auth_to_label[auth_ch] = label_ch

        # Build per-chain sequences from auth chain IDs (protein chains only)
        aa3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                  'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                  'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
                  'MSE':'M','HSD':'H','HSE':'H','HSP':'H','HIE':'H','HID':'H','HIP':'H'}
        pdb_chain_seqs = {}
        prev_key = {}
        for rec in atom_records:
            fields = rec.split()
            if len(fields) <= max(auth_chain_col, comp_col, seq_col):
                continue
            if group_col is not None and fields[group_col] != 'ATOM':
                continue
            if model_col is not None and fields[model_col] != '1':
                continue
            auth_ch = fields[auth_chain_col]
            atom_name = fields[atom_name_col].strip('"')
            if atom_name != 'CA':
                continue
            comp = fields[comp_col].strip('"')
            seq_id = fields[seq_col]
            key = (auth_ch, seq_id)
            if key != prev_key.get(auth_ch):
                if auth_ch not in pdb_chain_seqs:
                    pdb_chain_seqs[auth_ch] = ''
                pdb_chain_seqs[auth_ch] += aa3to1.get(comp, 'X')
                prev_key[auth_ch] = key

        # Match benchmark chain IDs to PDB auth chain IDs by sequence
        chains_needed = set(chain_id_1) | set(chain_id_2)
        auth_chains = set(pdb_chain_seqs.keys())

        if chains_needed <= auth_chains:
            # Direct match — benchmark uses auth chain IDs
            chains_to_keep = chains_needed
            actual_chain_id_2 = chain_id_2
        else:
            # Need to remap: match ALL non-empty chain_X sequences to PDB chains
            # The benchmark may store sequences under the correct auth chain IDs
            # even if chain_id_1/chain_id_2 fields are wrong
            all_ref_seqs = {}
            for key in row:
                if key.startswith('chain_') and len(key) == 7 and row[key].strip():
                    ch = key[-1]
                    all_ref_seqs[ch] = row[key].strip()

            # Match each reference sequence to a PDB chain
            remap = {}
            used = set()
            for ref_ch, ref_seq in sorted(all_ref_seqs.items(), key=lambda x: -len(x[1])):
                best_score = 0
                best_pdb_ch = None
                probe = ref_seq[:min(20, len(ref_seq))]
                for pdb_ch, pdb_seq in pdb_chain_seqs.items():
                    if pdb_ch in used:
                        continue
                    if probe in pdb_seq:
                        score = len(ref_seq) / max(len(pdb_seq), 1)
                        if score > best_score:
                            best_score = score
                            best_pdb_ch = pdb_ch
                if best_pdb_ch:
                    remap[ref_ch] = best_pdb_ch
                    used.add(best_pdb_ch)

            if not remap:
                print(f"FAILED: Could not map any sequences to PDB chains {auth_chains}")
                continue

            # Rebuild chain_id_1 and chain_id_2 from the matched chains
            # chain_id_1 chains: those in remap that were referenced by chain_id_1
            # If chain_id_1/2 reference empty chains, find the correct mapping
            matched_pdb_chains = set(remap.values())

            # Identify which PDB chains belong to protein 2 (protein B)
            # The remap maps CSV chain letters → PDB auth chain IDs
            # chain_id_2 from CSV tells us which chains are protein B
            protein_b_pdb_chains = set()
            for ch in set(chain_id_2):
                if ch in remap:
                    protein_b_pdb_chains.add(remap[ch])

            # If chain_id_2 letters didn't map, the CSV chain_id_2 uses wrong IDs.
            # Match the protein_2 description to PDB chains by sequence.
            if not protein_b_pdb_chains:
                # Try: find the chain_id_2 sequences in chain_X columns
                protein_2_seq = None
                for ch in set(chain_id_2):
                    col = f"chain_{ch}"
                    if col in row and row[col].strip():
                        protein_2_seq = row[col].strip()
                        break

                if protein_2_seq:
                    probe = protein_2_seq[:20]
                    for pdb_ch, pdb_seq in pdb_chain_seqs.items():
                        if pdb_ch in matched_pdb_chains and probe in pdb_seq:
                            protein_b_pdb_chains.add(pdb_ch)
                            break

            # Still nothing: use chain_id_1 to exclude — what's left is protein B
            if not protein_b_pdb_chains:
                candidates = set()
                for ref_ch, pdb_ch in remap.items():
                    if ref_ch not in set(chain_id_1):
                        candidates.add(pdb_ch)
                # If exclusion gives all matched chains, chain_id_1 letters are also wrong
                # Fall back to: the protein with fewer chains is protein B
                if candidates and candidates != matched_pdb_chains:
                    protein_b_pdb_chains = candidates
                elif len(matched_pdb_chains) >= 2:
                    # Count chains per protein from chain_id_1/2 field lengths
                    n_chains_2 = len(set(chain_id_2))
                    # Assign the n_chains_2 smallest chains as protein B
                    chain_sizes = sorted([(ch, len(pdb_chain_seqs.get(ch, ''))) for ch in matched_pdb_chains], key=lambda x: x[1])
                    protein_b_pdb_chains = {ch for ch, _ in chain_sizes[:n_chains_2]}
                    print(f"Assigning protein B = {protein_b_pdb_chains} (smallest {n_chains_2} chain(s))", end=" ")

            if not protein_b_pdb_chains:
                # Last resort: smaller chain = protein B
                chain_sizes = [(ch, len(seq)) for ch, seq in pdb_chain_seqs.items() if ch in matched_pdb_chains]
                if len(chain_sizes) >= 2:
                    chain_sizes.sort(key=lambda x: x[1])
                    protein_b_pdb_chains = {chain_sizes[0][0]}
                    print(f"Guessing protein B = {protein_b_pdb_chains} (smaller chain)", end=" ")

            chains_to_keep = matched_pdb_chains
            actual_chain_id_2 = ''.join(sorted(protein_b_pdb_chains))

            changed = {k: v for k, v in remap.items() if k != v}
            if changed:
                print(f"Chain remap: {changed}", end=" ")

        # Write PDB from mmCIF records (using auth chain IDs)
        pdb_lines = []
        prev_chain = None
        for rec in atom_records:
            fields = rec.split()
            if len(fields) <= max(x_col, y_col, z_col, auth_chain_col):
                continue
            # Only first model
            if model_col is not None and fields[model_col] != '1':
                continue
            # Skip alternate conformations (keep first or '.')
            if alt_col is not None and fields[alt_col] not in ('.', 'A', '?'):
                continue
            auth_ch = fields[auth_chain_col]
            if auth_ch not in chains_to_keep:
                continue

            group = fields[group_col] if group_col is not None else 'ATOM'
            atom_name = fields[atom_name_col].strip('"')
            comp = fields[comp_col].strip('"')
            seq_id = fields[seq_col] if seq_col is not None else '1'
            x = float(fields[x_col])
            y = float(fields[y_col])
            z = float(fields[z_col])
            element = fields[element_col] if element_col is not None else atom_name[0]
            occ = float(fields[occ_col]) if occ_col is not None else 1.0
            bfactor = float(fields[bfactor_col]) if bfactor_col is not None else 0.0

            # Write TER between chains
            if prev_chain is not None and auth_ch != prev_chain:
                pdb_lines.append("TER")
            prev_chain = auth_ch

            # Format atom name (4 chars, left-padded for 1-3 char names)
            if len(atom_name) < 4:
                atom_name_fmt = f" {atom_name:<3s}"
            else:
                atom_name_fmt = f"{atom_name:<4s}"

            serial = len(pdb_lines) + 1
            record = f"{group:<6s}{serial:5d} {atom_name_fmt} {comp:>3s} {auth_ch}{int(seq_id):4d}    " \
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfactor:6.2f}          {element:>2s}"
            pdb_lines.append(record)

        pdb_lines.append("END")

        if len(pdb_lines) <= 1:
            print(f"FAILED: No atoms extracted for chains {chains_to_keep}")
            continue

        with open(input_pdb, 'w') as out:
            out.write('\n'.join(pdb_lines) + '\n')

        print(f"OK ({len(pdb_lines)-1} atoms, chains: {','.join(sorted(chains_to_keep))})")

        # Add to sp.gs with comma-separated chain IDs (using auth chain IDs from PDB)
        chain_id_2_formatted = ','.join(actual_chain_id_2)
        sp.write(f"{pdb_id}\t{chain_id_2_formatted}\n")

print(f"\nCreated sp.gs with {len(structures)} entries")
print("Run: python ../groscore.py")
