#!/usr/bin/env python3
"""
Setup benchmark structures for heteroatom-containing protein-protein complexes.

Two subsets:
  1. Molecular glue / PROTAC ternary complexes (E3 ligase – small molecule – target)
  2. Protein-protein complexes with non-canonical amino acids (NCAAs) at the interface

Downloads mmCIF files from RCSB, extracts relevant chains, and creates sp.gs file.
"""

import os
import sys
import argparse
import urllib.request
import json

parser = argparse.ArgumentParser(description="Setup heteroatom benchmark structures.")
parser.add_argument('--max-entries', type=int, default=0, help="Limit number of structures (0 = all)")
parser.add_argument('--subset', type=str, default="all", choices=["all", "glue", "ncaa"],
                    help="Which subset to include")
parser.add_argument('--max-resolution-glue', type=float, default=3.0,
                    help="Resolution cutoff for molecular glue/PROTAC (default: 3.0 Å)")
parser.add_argument('--max-resolution-ncaa', type=float, default=2.5,
                    help="Resolution cutoff for NCAA structures (default: 2.5 Å)")
args = parser.parse_args()

# ---- Molecular glue / PROTAC structures ----
# Curated from RCSB queries + literature
# Format: (PDB_ID, protein_B_chains, description, E3_ligase)
# protein_B = target protein (pulled away from E3)
GLUE_PROTAC_STRUCTURES = [
    # (PDB_ID, protein_B_chains, protein_B_name, E3_ligase)
    # protein_B = target protein (pulled away from E3/receptor)
    # CRBN/DDB1-based molecular glues (one per unique target)
    ("5FQD", "C",      "CK1alpha",                                "CRBN"),
    ("6H0F", "D",      "GSPT1",                                   "CRBN"),
    ("8U16", "C",      "SALL4",                                   "CRBN"),
    ("9SAI", "C",      "BRD4",                                    "CRBN"),
    ("8BU1", "B,C",    "CDK12-CyclinK",                           "DDB1"),
    # VHL/ElonginBC-based PROTACs (one per unique target)
    ("5T35", "E",      "BRD4-BD2",                                "VHL"),
    ("6HAX", "E",      "SMARCA2",                                 "VHL"),
    ("6SIS", "E",      "BRD9",                                    "VHL"),
    # DCAF-based molecular glues
    ("6UD7", "C",      "RBM39",                                   "DCAF15"),
    ("6PAI", "D",      "RBM39",                                   "DCAF15"),
    ("7S4E", "D",      "WDR5",                                    "DCAF1"),
    # GID4-based PROTACs
    ("8X7H", "B",      "BRD4-BD1",                                "GID4"),
    # User-specified structures (auto-detect protein B)
    ("8G46", "C",      "DCAF1",                                   "DDB1"),
    ("8OV6", "C",      "BRD4",                                    "DCAF16"),
    # FKBP12-based molecular glues (Rui et al. RSC Chem Biol 2023)
    ("1FAP", "B",      "FRAP/mTOR",                               "FKBP12"),
    ("1TCO", "B,C",    "calcineurin",                              "FKBP12"),
    # 14-3-3 molecular glues (Rui et al. RSC Chem Biol 2023)
    ("4IHL", "P",      "RAF1-peptide",                             "14-3-3"),
    ("4JDD", "B",      "ERalpha-peptide",                          "14-3-3"),
    # Plant hormone receptor molecular glue
    ("2P1O", "C",      "IAA7",                                    "TIR1"),
    # Other diverse molecular glues (Rui et al. RSC Chem Biol 2023)
    ("1S9D", "E",      "ARNO",                                    "ARF1"),
    ("4J9Z", "R",      "calmodulin",                              "Kcnn2"),
    ("3QEL", "",       "GluN2B",                                  "GluN1"),
]

# ---- NCAA structures ----
# Query RCSB API for protein-protein complexes with modified residues at the interface

def query_rcsb(query_json, max_results=100):
    """Query RCSB search API and return list of PDB IDs."""
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query_json["request_options"] = {"paginate": {"start": 0, "rows": max_results}}
    req = urllib.request.Request(url, data=json.dumps(query_json).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        return [r["identifier"] for r in data.get("result_set", [])]
    except Exception as e:
        print(f"  RCSB query error: {e}")
        return []


def get_entry_info(pdb_id):
    """Get resolution and entity info from RCSB Data API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        data = json.loads(resp.read())
        resolution = None
        method = None
        if "rcsb_entry_info" in data:
            resolution = data["rcsb_entry_info"].get("resolution_combined", [None])[0]
        if "exptl" in data and len(data["exptl"]) > 0:
            method = data["exptl"][0].get("method", "")
        return resolution, method
    except Exception:
        return None, None


def query_ncaa_structures(ncaa_type, keyword, max_resolution=2.5, max_results=50):
    """Query RCSB for protein-protein complexes mentioning a keyword."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "full_text", "parameters": {
                    "value": keyword}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                    "operator": "greater_or_equal", "value": 2}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal", "value": max_resolution}}
            ]
        },
        "return_type": "entry",
    }
    return query_rcsb(query, max_results)


# aa3to1 mapping
aa3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
          'MSE':'M','HSD':'H','HSE':'H','HSP':'H','HIE':'H','HID':'H','HIP':'H',
          'SEC':'C','PYL':'K','CSE':'C','TPO':'T','SEP':'S','PTR':'Y',
          'HYP':'P','MLY':'K','CSO':'C','KCX':'K','CGU':'E','TRQ':'W'}


# ---- Collect structures ----
structures = {}  # pdb_id -> {chains_b, description, source, resolution}

if args.subset in ("all", "glue"):
    print("=== Molecular Glue / PROTAC Structures ===")
    for pdb_id, chains_b, desc, e3 in GLUE_PROTAC_STRUCTURES:
        resolution, method = get_entry_info(pdb_id)
        if resolution is not None and resolution > args.max_resolution_glue:
            print(f"  {pdb_id}: skipped (resolution {resolution:.1f} Å > {args.max_resolution_glue})")
            continue
        structures[pdb_id] = {
            'chains_b': chains_b,
            'description': desc,
            'source': f"glue/{e3}",
            'resolution': resolution,
            'method': method or '',
        }
        res_str = f"{resolution:.1f}" if resolution else "?"
        print(f"  {pdb_id}: {desc} ({res_str} Å)")
    print(f"  Total molecular glue/PROTAC: {sum(1 for s in structures.values() if 'glue' in s['source'])}")

if args.subset in ("all", "ncaa"):
    print("\n=== NCAA Structures (querying RCSB) ===")

    # 14-3-3 with phosphopeptide (SEP at interface)
    print("  Querying 14-3-3 + phosphoserine...")
    ids_1433 = query_ncaa_structures("SEP", "14-3-3 phosphoserine complex",
                                      args.max_resolution_ncaa, 30)
    for pdb_id in ids_1433[:10]:  # take top 10
        if pdb_id not in structures:
            resolution, method = get_entry_info(pdb_id)
            structures[pdb_id] = {
                'chains_b': '',  # determined during download
                'description': '14-3-3/phosphoserine',
                'source': 'ncaa/SEP',
                'resolution': resolution,
                'method': method or '',
            }
    print(f"    Found {len(ids_1433)}, selected {min(10, len(ids_1433))}")

    # SH2 domain with phosphotyrosine (PTR at interface)
    print("  Querying SH2 + phosphotyrosine...")
    ids_sh2 = query_ncaa_structures("PTR", "SH2 domain phosphotyrosine complex",
                                     args.max_resolution_ncaa, 30)
    for pdb_id in ids_sh2[:8]:
        if pdb_id not in structures:
            resolution, method = get_entry_info(pdb_id)
            structures[pdb_id] = {
                'chains_b': '',
                'description': 'SH2/phosphotyrosine',
                'source': 'ncaa/PTR',
                'resolution': resolution,
                'method': method or '',
            }
    print(f"    Found {len(ids_sh2)}, selected {min(8, len(ids_sh2))}")

    # Hydroxyproline-containing complexes
    print("  Querying hydroxyproline complexes...")
    ids_hyp = query_ncaa_structures("HYP", "hydroxyproline collagen complex",
                                     args.max_resolution_ncaa, 20)
    for pdb_id in ids_hyp[:5]:
        if pdb_id not in structures:
            resolution, method = get_entry_info(pdb_id)
            structures[pdb_id] = {
                'chains_b': '',
                'description': 'hydroxyproline complex',
                'source': 'ncaa/HYP',
                'resolution': resolution,
                'method': method or '',
            }
    print(f"    Found {len(ids_hyp)}, selected {min(5, len(ids_hyp))}")

    # Methylated lysine readers (chromodomain, PHD finger, etc.)
    print("  Querying methylated lysine readers...")
    ids_mly = query_ncaa_structures("MLY", "methylated lysine reader chromodomain histone",
                                     args.max_resolution_ncaa, 20)
    for pdb_id in ids_mly[:5]:
        if pdb_id not in structures:
            resolution, method = get_entry_info(pdb_id)
            structures[pdb_id] = {
                'chains_b': '',
                'description': 'methylated lysine reader',
                'source': 'ncaa/MLY',
                'resolution': resolution,
                'method': method or '',
            }
    print(f"    Found {len(ids_mly)}, selected {min(5, len(ids_mly))}")

    ncaa_count = sum(1 for s in structures.values() if 'ncaa' in s['source'])
    print(f"  Total NCAA: {ncaa_count}")

if args.max_entries > 0:
    # Limit to max_entries, keeping balanced subset
    items = list(structures.items())[:args.max_entries]
    structures = dict(items)

print(f"\nTotal structures to download: {len(structures)}")

# ---- Download and process structures ----

stats = {'ok': 0, 'failed_download': 0, 'failed_no_atoms': 0, 'failed_chain': 0}
fail_details = []

# Write benchmark.csv
with open('benchmark.csv', 'w') as csvf:
    csvf.write("pdb_id,chain_id_b,protein_b,source,resolution,method\n")
    for pdb_id, info in structures.items():
        csvf.write(f"{pdb_id},{info['chains_b']},{info['description']},"
                   f"{info['source']},{info['resolution']},{info['method']}\n")

# Download and extract structures
n_ok = 0
n_fail = 0
sp_entries = []

for pdb_id, info in structures.items():
    os.makedirs(pdb_id, exist_ok=True)
    input_pdb = f"{pdb_id}/input.pdb"

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
        print(f"FAILED: no auth_asym_id")
        n_fail += 1
        stats['failed_chain'] += 1
        fail_details.append((pdb_id, "no auth_asym_id"))
        continue

    # Get all available chains
    pdb_chains = set()
    for rec in atom_records:
        fields = rec.split()
        if len(fields) > auth_chain_col:
            if model_col is not None and len(fields) > model_col and fields[model_col] != '1':
                continue
            pdb_chains.add(fields[auth_chain_col])

    # Determine chains to keep
    chains_b_str = info['chains_b']
    if chains_b_str:
        # Predefined chains (molecular glue/PROTAC)
        chains_b = set(c.strip() for c in chains_b_str.split(','))
        chains_to_keep = pdb_chains  # keep all chains
        actual_b_chains = chains_b & pdb_chains
        if not actual_b_chains:
            print(f"FAILED: protein B chains {chains_b} not in PDB {pdb_chains}")
            n_fail += 1
            stats['failed_chain'] += 1
            fail_details.append((pdb_id, f"chains {chains_b} not found"))
            continue
    else:
        # NCAA: auto-detect chains, use smallest protein chain as B
        # Build per-chain residue counts (protein only)
        chain_res_counts = {}
        seen = set()
        for rec in atom_records:
            fields = rec.split()
            if len(fields) <= max(auth_chain_col, comp_col or 0, seq_col or 0):
                continue
            if model_col is not None and len(fields) > model_col and fields[model_col] != '1':
                continue
            ch = fields[auth_chain_col]
            comp = fields[comp_col].strip('"') if comp_col and len(fields) > comp_col else ''
            seq_id = fields[seq_col] if seq_col and len(fields) > seq_col else ''
            key = (ch, seq_id)
            if key not in seen and comp in aa3to1:
                chain_res_counts[ch] = chain_res_counts.get(ch, 0) + 1
                seen.add(key)

        if len(chain_res_counts) < 2:
            print(f"FAILED: fewer than 2 protein chains")
            n_fail += 1
            stats['failed_chain'] += 1
            fail_details.append((pdb_id, "fewer than 2 protein chains"))
            continue

        # Protein B = smallest protein chain
        sorted_chains = sorted(chain_res_counts.items(), key=lambda x: x[1])
        actual_b_chains = {sorted_chains[0][0]}
        chains_to_keep = pdb_chains
        chains_b_str = ','.join(sorted(actual_b_chains))

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
        fail_details.append((pdb_id, "no atoms"))
        continue

    with open(input_pdb, 'w') as out:
        out.write('\n'.join(pdb_lines) + '\n')

    b_chain_str = ','.join(sorted(actual_b_chains))
    sp_entries.append((pdb_id, b_chain_str))

    res_str = f"{info['resolution']:.1f}" if info['resolution'] else "?"
    print(f"OK ({len(pdb_lines)-1} atoms, B={b_chain_str}, {res_str} Å, {info['source']})")
    n_ok += 1

# Write sp.gs
with open('sp.gs', 'w') as sp:
    sp.write("# Structure_ID  Chains_for_Protein_B\n")
    for pdb_id, b_chains in sp_entries:
        sp.write(f"{pdb_id}\t{b_chains}\n")

# Statistics
print(f"\n{'='*60}")
print(f"SETUP STATISTICS")
print(f"{'='*60}")
print(f"Total structures queried:    {len(structures)}")
print(f"Successfully set up:         {n_ok}")
print(f"Failed:                      {n_fail}")
print(f"  Download failed:           {stats['failed_download']}")
print(f"  Chain issues:              {stats['failed_chain']}")
print(f"  No atoms extracted:        {stats['failed_no_atoms']}")
if fail_details:
    print(f"\nFailed structures:")
    for pdb_id, reason in fail_details:
        print(f"  {pdb_id}: {reason}")
print(f"\nCreated sp.gs with {len(sp_entries)} entries")
print(f"Created benchmark.csv with {len(structures)} entries")
print("Run: python ../groscore.py")
