#!/usr/bin/env python3
"""
Setup benchmark structures from the PPB-Affinity dataset.
Downloads mmCIF files from RCSB, extracts relevant chains, and creates sp.gs file.

Reads PPB-Affinity.xlsx directly (authoritative source with proper chain assignments).
"""

import os
import sys
import csv
import argparse
import urllib.request
import openpyxl

parser = argparse.ArgumentParser(description="Setup PPB-Affinity benchmark structures.")
parser.add_argument('--include-mutants', action='store_true', help="Include mutant entries (default: wild-type only)")
parser.add_argument('--max-entries', type=int, default=0, help="Limit number of structures (0 = all)")
parser.add_argument('--max-resolution', type=float, default=0, help="Filter by resolution in Angstrom (0 = no filter)")
parser.add_argument('--source', type=str, default="", help="Filter by source dataset (e.g., 'SAbDab', 'PDBbind v2020')")
parser.add_argument('--skip-duplicate-filter', action='store_true', help="Skip duplicate entity filtering (faster but may include problematic structures)")
args = parser.parse_args()

# Read PPB-Affinity.xlsx
xlsx_path = "PPB-Affinity.xlsx"
if not os.path.isfile(xlsx_path):
    print(f"Error: {xlsx_path} not found. Download from https://zenodo.org/doi/10.5281/zenodo.11070823")
    sys.exit(1)

wb = openpyxl.load_workbook(xlsx_path, read_only=True)
ws = wb.active

# Parse xlsx rows
# Columns: 0=idx, 1=source, 2=complex_id, 3=PDB, 4=mutations, 5=ligand_chains,
#          6=receptor_chains, 7=ligand_name, 8=receptor_name, 9=KD, 10=method,
#          11=struct_method, 12=temperature, 13=resolution, 14-18=misc
structures = {}  # pdb_id -> first entry (deduplicate by PDB)
n_uppercase_replaced = 0
for row in ws.iter_rows(min_row=2, values_only=True):
    pdb_id = str(row[3]).strip().upper() if row[3] else None
    if not pdb_id or len(pdb_id) != 4:
        continue

    mutations = row[4]
    if not args.include_mutants and mutations is not None:
        continue

    source = str(row[1]).strip() if row[1] else ""
    if args.source and args.source not in source:
        continue

    resolution = row[13]
    if args.max_resolution > 0 and resolution is not None:
        try:
            if float(resolution) > args.max_resolution:
                continue
        except (ValueError, TypeError):
            pass

    # Parse chain IDs from comma-separated fields (e.g., "A, B" or "H, L")
    ligand_chains_raw = str(row[5]).strip() if row[5] else ""
    receptor_chains_raw = str(row[6]).strip() if row[6] else ""
    ligand_chains = [c.strip() for c in ligand_chains_raw.split(',') if c.strip()]
    receptor_chains = [c.strip() for c in receptor_chains_raw.split(',') if c.strip()]

    if not ligand_chains or not receptor_chains:
        continue

    kd = row[9]
    if kd is None:
        continue
    try:
        kd = float(kd)
    except (ValueError, TypeError):
        continue

    # Deduplicate by PDB ID
    # Prefer entries with all-uppercase chain IDs (avoid SAbDab lowercase convention)
    has_lowercase = any(c.islower() for c in ''.join(ligand_chains + receptor_chains))
    if pdb_id in structures:
        prev_has_lowercase = structures[pdb_id].get('_has_lowercase', False)
        if has_lowercase or not prev_has_lowercase:
            continue  # keep previous (it's already uppercase, or both are same quality)
        # Replace: current entry has all uppercase, previous had lowercase
        n_uppercase_replaced += 1

    import math
    pkd = -math.log10(kd)

    structures[pdb_id] = {
        'ligand_chains': ligand_chains,
        'receptor_chains': receptor_chains,
        '_has_lowercase': has_lowercase,
        'ligand_name': str(row[7]).strip() if row[7] else "",
        'receptor_name': str(row[8]).strip() if row[8] else "",
        'kd': kd,
        'pkd': pkd,
        'source': source,
        'resolution': resolution,
    }

    if args.max_entries > 0 and len(structures) >= args.max_entries:
        break

print(f"Found {len(structures)} structures in PPB-Affinity dataset")

# ---- Filter duplicate entity structures ----
# Case 1: Same protein entity in both A and B → wrong chain assignment → REMOVE
# Case 2: Duplicate entity on one side:
#   - If duplicate chains have inter-chain contacts → functional homodimer → KEEP
#   - If no contacts → independent copies → REMOVE
import json
import math

CONTACT_CUTOFF = 5.0   # Angstrom for CA-CA inter-chain contacts
MIN_CONTACTS = 10      # minimum contacts to consider a functional dimer

def get_entity_info(pdb_id):
    """Get chain->entity name mapping from RCSB GraphQL API."""
    gql = ('{ entry(entry_id: "%s") { polymer_entities { '
           'rcsb_polymer_entity { pdbx_description } '
           'rcsb_polymer_entity_container_identifiers { auth_asym_ids } } } }' % pdb_id)
    req = urllib.request.Request("https://data.rcsb.org/graphql",
        data=json.dumps({"query": gql}).encode(),
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())
    chain_entity = {}
    for ent in data["data"]["entry"]["polymer_entities"]:
        name = ent["rcsb_polymer_entity"]["pdbx_description"]
        for ch in ent["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]:
            chain_entity[ch] = name
    return chain_entity

def check_interchain_contacts(pdb_id, chain1, chain2):
    """Check if two chains have CA-CA contacts within cutoff. Downloads mmCIF."""
    url = "https://files.rcsb.org/download/%s.cif" % pdb_id
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        cif = resp.read().decode()
    except Exception:
        return True  # assume contacts if can't download

    lines = cif.split('\n')
    cols = [l.strip() for l in lines if l.startswith('_atom_site.')]
    idx = {}
    for i, c in enumerate(cols):
        if c == '_atom_site.auth_asym_id': idx['chain'] = i
        elif c == '_atom_site.label_atom_id': idx['atom'] = i
        elif c == '_atom_site.Cartn_x': idx['x'] = i
        elif c == '_atom_site.Cartn_y': idx['y'] = i
        elif c == '_atom_site.Cartn_z': idx['z'] = i
        elif c == '_atom_site.pdbx_PDB_model_num': idx['model'] = i

    if not all(k in idx for k in ['chain', 'atom', 'x', 'y', 'z']):
        return True

    coords1, coords2 = [], []
    in_atoms = False
    for line in lines:
        if line.startswith('_atom_site.'): in_atoms = True
        elif in_atoms:
            if line.startswith('_') or line.startswith('#') or line.startswith('loop_'): break
            fields = line.split()
            max_i = max(idx.values())
            if len(fields) <= max_i: continue
            if 'model' in idx and fields[idx['model']] != '1': continue
            if fields[idx['atom']].strip('"') != 'CA': continue
            ch = fields[idx['chain']]
            try:
                xyz = (float(fields[idx['x']]), float(fields[idx['y']]), float(fields[idx['z']]))
            except ValueError:
                continue
            if ch == chain1: coords1.append(xyz)
            elif ch == chain2: coords2.append(xyz)

    n = 0
    for x1, y1, z1 in coords1:
        for x2, y2, z2 in coords2:
            if math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) < CONTACT_CUTOFF:
                n += 1
                if n >= MIN_CONTACTS: return True
    return n >= MIN_CONTACTS

if not args.skip_duplicate_filter:
    print("\nFiltering duplicate entity structures...")
    n_case1 = 0
    n_case2_removed = 0
    n_case2_kept = 0
    to_remove = set()

    for pdb_id, info in list(structures.items()):
        c1 = set(info['ligand_chains'])
        c2 = set(info['receptor_chains'])
        if len(c1) <= 1 and len(c2) <= 1:
            continue

        try:
            chain_entity = get_entity_info(pdb_id)
        except Exception:
            continue

        a_entities = {c: chain_entity.get(c, "?") for c in c1}
        b_entities = {c: chain_entity.get(c, "?") for c in c2}

        # Case 1: same entity in both A and B
        cross = set(a_entities.values()) & set(b_entities.values()) - {"?"}
        if cross:
            to_remove.add(pdb_id)
            n_case1 += 1
            print(f"  REMOVE (case 1) {pdb_id}: same entity in A+B: {cross}")
            continue

        # Case 2: duplicate entity on one side — check contacts
        for side, entities in [("A", a_entities), ("B", b_entities)]:
            seen = {}
            for ch, name in entities.items():
                if name in seen:
                    has_contacts = check_interchain_contacts(pdb_id, seen[name], ch)
                    if not has_contacts:
                        to_remove.add(pdb_id)
                        n_case2_removed += 1
                        print(f"  REMOVE (case 2) {pdb_id}: {side} chains {seen[name]},{ch} ({name}) — no contacts")
                    else:
                        n_case2_kept += 1
                else:
                    seen[name] = ch

    for pdb_id in to_remove:
        del structures[pdb_id]

    print(f"\nDuplicate filter: removed {n_case1} (case 1) + {len(to_remove) - n_case1} (case 2), "
          f"kept {n_case2_kept} homodimers, {len(structures)} structures remain")
else:
    print("Skipping duplicate entity filter (--skip-duplicate-filter)")

# Statistics tracking
stats = {'direct_match': 0, 'dropped_lowercase': 0, 'failed_missing_uppercase': 0,
         'failed_no_protein_b': 0, 'failed_download': 0, 'failed_no_atoms': 0,
         'failed_no_auth_col': 0}
fail_details = []

# aa3to1 mapping for sequence building
aa3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
          'MSE':'M','HSD':'H','HSE':'H','HSP':'H','HIE':'H','HID':'H','HIP':'H',
          'SEC':'C','PYL':'K','CSE':'C','TPO':'T','SEP':'S','PTR':'Y'}

# Write benchmark.csv for analysis scripts
with open('benchmark.csv', 'w', newline='') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["pdb_id", "chain_id_1", "chain_id_2", "protein_1", "protein_2",
                     "kd", "pkd", "source", "resolution"])
    for pdb_id, info in structures.items():
        chain_1 = ''.join(info['ligand_chains'])
        chain_2 = ''.join(info['receptor_chains'])
        writer.writerow([pdb_id, chain_1, chain_2, info['ligand_name'], info['receptor_name'],
                         info['kd'], f"{info['pkd']:.2f}", info['source'], info['resolution']])

# Create sp.gs file and download structures
n_ok = 0
n_fail = 0
with open('sp.gs', 'w') as sp:
    sp.write("# Structure_ID  Chains_for_Protein_B\n")

    for pdb_id, info in structures.items():
        os.makedirs(pdb_id, exist_ok=True)
        input_pdb = f"{pdb_id}/input.pdb"

        # All chains needed (ligand = protein A, receptor = protein B in GroScore convention)
        all_chains = info['ligand_chains'] + info['receptor_chains']
        protein_b_chains = info['receptor_chains']

        # Download mmCIF
        cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        print(f"Downloading {pdb_id}...", end=" ", flush=True)

        try:
            with urllib.request.urlopen(cif_url, timeout=30) as response:
                cif_content = response.read().decode('utf-8')
        except Exception as e:
            print(f"FAILED: {e}")
            n_fail += 1
            stats['failed_download'] += 1
            fail_details.append((pdb_id, f"download: {e}"))
            continue

        # Parse mmCIF _atom_site
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

        col_idx = {col: i for i, col in enumerate(columns)}
        auth_chain_col = col_idx.get('_atom_site.auth_asym_id')
        group_col = col_idx.get('_atom_site.group_PDB')
        atom_name_col = col_idx.get('_atom_site.label_atom_id') or col_idx.get('_atom_site.auth_atom_id')
        comp_col = col_idx.get('_atom_site.label_comp_id') or col_idx.get('_atom_site.auth_comp_id')
        seq_col = col_idx.get('_atom_site.auth_seq_id')
        x_col = col_idx.get('_atom_site.Cartn_x')
        y_col = col_idx.get('_atom_site.Cartn_y')
        z_col = col_idx.get('_atom_site.Cartn_z')
        element_col = col_idx.get('_atom_site.type_symbol')
        occ_col = col_idx.get('_atom_site.occupancy')
        bfactor_col = col_idx.get('_atom_site.B_iso_or_equiv')
        alt_col = col_idx.get('_atom_site.label_alt_id')
        model_col = col_idx.get('_atom_site.pdbx_PDB_model_num')

        if auth_chain_col is None:
            print(f"FAILED: no auth_asym_id column in mmCIF")
            n_fail += 1
            stats['failed_no_auth_col'] += 1
            fail_details.append((pdb_id, "no auth_asym_id in CIF"))
            continue

        # Get available auth chain IDs
        pdb_chains = set()
        for rec in atom_records:
            fields = rec.split()
            if len(fields) > auth_chain_col:
                if model_col is not None and len(fields) > model_col and fields[model_col] != '1':
                    continue
                pdb_chains.add(fields[auth_chain_col])

        # Validate chain IDs against PDB
        chains_to_keep = set(all_chains)
        actual_b_chains = set(protein_b_chains)
        missing = chains_to_keep - pdb_chains

        if missing:
            missing_uppercase = {c for c in missing if c.isupper()}
            missing_lowercase = {c for c in missing if c.islower()}

            if missing_uppercase:
                print(f"FAILED: uppercase chains {missing_uppercase} not in PDB {pdb_chains}")
                n_fail += 1
                stats['failed_missing_uppercase'] += 1
                fail_details.append((pdb_id, f"uppercase chains {missing_uppercase} not in PDB"))
                continue

            # Lowercase: keep if in PDB (real chain IDs), drop if not (SAbDab convention)
            lowercase_not_in_pdb = missing_lowercase - pdb_chains

            if lowercase_not_in_pdb:
                chains_to_keep = chains_to_keep - lowercase_not_in_pdb
                actual_b_chains = actual_b_chains - lowercase_not_in_pdb
                stats['dropped_lowercase'] += 1
                print(f"(dropped SAbDab {lowercase_not_in_pdb})", end=" ")

            if not actual_b_chains:
                print(f"FAILED: no protein B chains remain after filtering")
                n_fail += 1
                stats['failed_no_protein_b'] += 1
                fail_details.append((pdb_id, f"no protein B after dropping {lowercase_not_in_pdb}"))
                continue
        else:
            stats['direct_match'] += 1

        # Write PDB from mmCIF
        pdb_lines = []
        prev_chain = None
        for rec in atom_records:
            fields = rec.split()
            required_cols = [x_col, y_col, z_col, auth_chain_col]
            if any(c is None for c in required_cols):
                continue
            if len(fields) <= max(c for c in required_cols if c is not None):
                continue
            if model_col is not None and len(fields) > model_col and fields[model_col] != '1':
                continue
            if alt_col is not None and len(fields) > alt_col and fields[alt_col] not in ('.', 'A', '?'):
                continue
            auth_ch = fields[auth_chain_col]
            if auth_ch not in chains_to_keep:
                continue

            group = fields[group_col] if group_col is not None and len(fields) > group_col else 'ATOM'
            atom_name = fields[atom_name_col].strip('"') if atom_name_col and len(fields) > atom_name_col else 'X'
            comp = fields[comp_col].strip('"') if comp_col and len(fields) > comp_col else 'UNK'
            seq_id = fields[seq_col] if seq_col and len(fields) > seq_col else '1'
            try:
                x = float(fields[x_col])
                y = float(fields[y_col])
                z = float(fields[z_col])
            except (ValueError, IndexError):
                continue
            element = fields[element_col] if element_col and len(fields) > element_col else atom_name[0]
            try:
                occ = float(fields[occ_col]) if occ_col and len(fields) > occ_col else 1.0
            except ValueError:
                occ = 1.0
            try:
                bfactor = float(fields[bfactor_col]) if bfactor_col and len(fields) > bfactor_col else 0.0
            except ValueError:
                bfactor = 0.0

            if prev_chain is not None and auth_ch != prev_chain:
                pdb_lines.append("TER")
            prev_chain = auth_ch

            if len(atom_name) < 4:
                atom_name_fmt = f" {atom_name:<3s}"
            else:
                atom_name_fmt = f"{atom_name:<4s}"

            try:
                seq_int = int(seq_id)
            except ValueError:
                seq_int = 1

            serial = len(pdb_lines) + 1
            record = (f"{group:<6s}{serial:5d} {atom_name_fmt} {comp:>3s} {auth_ch}{seq_int:4d}    "
                      f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfactor:6.2f}          {element:>2s}")
            pdb_lines.append(record)

        pdb_lines.append("END")

        if len(pdb_lines) <= 1:
            print(f"FAILED: no atoms extracted")
            n_fail += 1
            stats['failed_no_atoms'] += 1
            fail_details.append((pdb_id, "no atoms extracted"))
            continue

        with open(input_pdb, 'w') as out:
            out.write('\n'.join(pdb_lines) + '\n')

        # Write sp.gs entry
        b_chain_str = ','.join(sorted(actual_b_chains))
        sp.write(f"{pdb_id}\t{b_chain_str}\n")

        print(f"OK ({len(pdb_lines)-1} atoms, chains: {','.join(sorted(chains_to_keep))}, B={b_chain_str})")
        n_ok += 1

print(f"\n{'='*60}")
print(f"SETUP STATISTICS")
print(f"{'='*60}")
print(f"Total structures in dataset:     {len(structures)}")
print(f"Successfully set up:             {n_ok}")
print(f"Failed:                          {n_fail}")
print(f"")
print(f"Chain matching breakdown:")
print(f"  Direct match (all chains OK):  {stats['direct_match']}")
print(f"  Dropped SAbDab lowercase:      {stats['dropped_lowercase']}")
print(f"  Uppercase replaced lowercase:  {n_uppercase_replaced}")
print(f"")
print(f"Failure breakdown:")
print(f"  Missing uppercase chains:      {stats['failed_missing_uppercase']}")
print(f"  No protein B after filtering:  {stats['failed_no_protein_b']}")
print(f"  Download failed:               {stats['failed_download']}")
print(f"  No atoms extracted:            {stats['failed_no_atoms']}")
print(f"  No auth_asym_id in CIF:        {stats['failed_no_auth_col']}")
if fail_details:
    print(f"\nFailed structures:")
    for pdb_id, reason in fail_details:
        print(f"  {pdb_id}: {reason}")
print(f"\nCreated sp.gs with {n_ok} entries")
print(f"Created benchmark.csv with {len(structures)} entries")
print("Run: python ../groscore.py")
