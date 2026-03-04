#!/usr/bin/env python3
"""
Setup benchmark structures from the PPB-Affinity dataset.
Downloads data from Zenodo, filters entries, downloads PDBs,
extracts relevant chains, and creates sp.gs file.

Reference:
  Liu, H., Chen, P., Zhai, X. et al. PPB-Affinity: Protein-Protein Binding
  Affinity dataset for AI-based protein drug discovery. Scientific Data 11,
  1316 (2024). https://doi.org/10.1038/s41597-024-03997-4

Dataset: https://zenodo.org/doi/10.5281/zenodo.11070823

Sources in PPB-Affinity (use exact names for --source / --exclude-source):
  - "SKEMPI v2.0"              Protein-protein mutation thermodynamics (6700 entries)
  - "SAbDab"                   Antibody-antigen complexes (1055 entries)
  - "PDBbind v2020"            General protein-protein binding data (3600 entries)
  - "ATLAS"                    TCR-pMHC binding data (545 entries)
  - "Affinity Benchmark v5.5"  Curated affinity benchmark / HADDOCK (162 entries)
"""

import argparse
import csv
import math
import os
import sys
import urllib.request

try:
    from openpyxl import load_workbook
except ImportError:
    print("Error: openpyxl is required to parse the PPB-Affinity XLSX file.")
    print("Install with:  conda install openpyxl  (or: pip install openpyxl)")
    sys.exit(1)

ZENODO_RECORD_ID = "13054646"
XLSX_FILENAME = "PPB-Affinity.xlsx"
XLSX_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{XLSX_FILENAME}"


def download_file(url, filename):
    """Download a file if not already present."""
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping download")
        return True
    print(f"Downloading {filename} from Zenodo...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, filename)
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        print(f"You can manually download from: {url}")
        return False


def parse_chains(chain_str):
    """Parse chain IDs from various formats ('A', 'A,B', 'A B', 'AB')."""
    if not chain_str:
        return []
    chain_str = str(chain_str).strip()
    if ',' in chain_str:
        return [c.strip() for c in chain_str.split(',') if c.strip()]
    if ' ' in chain_str:
        return [c.strip() for c in chain_str.split() if c.strip()]
    return list(chain_str)


def find_column(header, name):
    """Find column index by name (case-insensitive, flexible matching)."""
    name_lower = name.lower()
    for idx, h in enumerate(header):
        if h.lower() == name_lower:
            return idx
    # Try partial match
    for idx, h in enumerate(header):
        if name_lower in h.lower():
            return idx
    return None


def parse_xlsx(filename, wt_only=True, sources=None, max_resolution=None,
               exclude_sources=None):
    """Parse PPB-Affinity.xlsx and return filtered entries."""
    wb = load_workbook(filename, read_only=True)
    ws = wb.active

    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h else '' for h in next(rows)]

    # Find required columns
    col_pdb = find_column(header, 'PDB')
    col_ligand = find_column(header, 'Ligand Chains')
    col_receptor = find_column(header, 'Receptor Chains')
    col_kd = find_column(header, 'KD(M)')

    for name, idx in [('PDB', col_pdb), ('Ligand Chains', col_ligand),
                       ('Receptor Chains', col_receptor), ('KD(M)', col_kd)]:
        if idx is None:
            print(f"Error: Required column '{name}' not found in XLSX")
            print(f"Available columns: {header}")
            sys.exit(1)

    # Find optional columns
    col_mutations = find_column(header, 'Mutations')
    col_source = find_column(header, 'Source')
    col_resolution = find_column(header, 'Resolution')
    col_dg = find_column(header, 'dG')
    col_ligand_name = find_column(header, 'Ligand Name')
    col_receptor_name = find_column(header, 'Receptor Name')
    col_temperature = find_column(header, 'Temperature')
    col_method = find_column(header, 'Affinity Method')

    entries = []
    seen_pdbs = set()
    skipped_mutants = 0
    skipped_source = 0
    skipped_resolution = 0
    skipped_duplicate = 0

    for row in rows:
        pdb = row[col_pdb]
        if not pdb:
            continue
        pdb = str(pdb).strip().upper()

        # Filter wild-type only
        if col_mutations is not None and wt_only:
            mutations = row[col_mutations]
            if mutations and str(mutations).strip():
                skipped_mutants += 1
                continue

        # Filter by source
        source = ''
        if col_source is not None and row[col_source]:
            source = str(row[col_source]).strip()
        if sources and source not in sources:
            skipped_source += 1
            continue
        if exclude_sources and source in exclude_sources:
            skipped_source += 1
            continue

        # Filter by resolution
        resolution = ''
        if col_resolution is not None and row[col_resolution]:
            resolution = str(row[col_resolution]).strip()
        if max_resolution and resolution:
            try:
                if float(resolution) > max_resolution:
                    skipped_resolution += 1
                    continue
            except (ValueError, TypeError):
                pass

        # Parse chains
        ligand_chains = parse_chains(row[col_ligand])
        receptor_chains = parse_chains(row[col_receptor])
        if not ligand_chains or not receptor_chains:
            continue

        # Get affinity
        kd = row[col_kd]
        if not kd:
            continue
        try:
            kd = float(kd)
        except (ValueError, TypeError):
            continue
        if kd <= 0:
            continue

        # De-duplicate by PDB ID (first entry wins)
        if pdb in seen_pdbs:
            skipped_duplicate += 1
            continue
        seen_pdbs.add(pdb)

        # Compute pKd
        pkd = -math.log10(kd)

        # Get dG
        dg = None
        if col_dg is not None and row[col_dg]:
            try:
                dg = float(row[col_dg])
            except (ValueError, TypeError):
                pass

        # Get optional fields
        def get_field(col_idx):
            if col_idx is not None and row[col_idx]:
                return str(row[col_idx]).strip()
            return ''

        entries.append({
            'pdb_id': pdb,
            'receptor_chains': receptor_chains,
            'ligand_chains': ligand_chains,
            'kd': kd,
            'pkd': pkd,
            'dg': dg,
            'receptor_name': get_field(col_receptor_name),
            'ligand_name': get_field(col_ligand_name),
            'source': source,
            'resolution': resolution,
            'temperature': get_field(col_temperature),
            'method': get_field(col_method),
        })

    wb.close()

    print(f"  Total wild-type entries: {len(entries)} unique PDBs")
    if skipped_mutants:
        print(f"  Skipped {skipped_mutants} mutant entries")
    if skipped_source:
        print(f"  Skipped {skipped_source} entries (source filter)")
    if skipped_resolution:
        print(f"  Skipped {skipped_resolution} entries (resolution filter)")
    if skipped_duplicate:
        print(f"  Skipped {skipped_duplicate} duplicate PDB entries")

    return entries


def download_pdb(pdb_id, chains_to_keep, output_path):
    """Download PDB from RCSB and extract relevant chains."""
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id}...", end=" ", flush=True)

    try:
        with urllib.request.urlopen(pdb_url, timeout=30) as response:
            pdb_content = response.read().decode('utf-8')
    except Exception as e:
        print(f"FAILED: {e}")
        return False

    extracted_lines = []
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            if len(line) > 21:
                chain = line[21]
                if chain in chains_to_keep:
                    extracted_lines.append(line)
        elif line.startswith('TER'):
            if len(line) > 21:
                chain = line[21]
                if chain in chains_to_keep:
                    extracted_lines.append(line)
            elif extracted_lines:
                extracted_lines.append(line)
        elif line.startswith('END'):
            extracted_lines.append(line)

    if not extracted_lines:
        print(f"FAILED: No atoms found for chains {','.join(sorted(chains_to_keep))}")
        return False

    with open(output_path, 'w') as out:
        out.write('\n'.join(extracted_lines))
        if not extracted_lines[-1].startswith('END'):
            out.write('\nEND\n')

    print(f"OK ({len(extracted_lines)} lines, chains: {','.join(sorted(chains_to_keep))})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Setup PPB-Affinity benchmark for GroScore',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python setup_benchmark.py                          # Wild-type only (default)
  python setup_benchmark.py --include-mutants        # Include mutant entries
  python setup_benchmark.py --source PDBbind         # Only PDBbind entries
  python setup_benchmark.py --source 'Affinity Benchmark v5.5'  # Only HADDOCK benchmark subset
  python setup_benchmark.py --exclude-source 'Affinity Benchmark v5.5'  # Exclude HADDOCK entries
  python setup_benchmark.py --max-resolution 2.5     # Resolution <= 2.5 A
  python setup_benchmark.py --max-entries 100        # Limit to 100 structures
  python setup_benchmark.py --skip-download          # Only create CSV and sp.gs

Sources: "SKEMPI v2.0", "SAbDab", "PDBbind v2020", "ATLAS", "Affinity Benchmark v5.5\"""")
    parser.add_argument('--include-mutants', action='store_true',
                        help='Include mutant entries (default: wild-type only)')
    parser.add_argument('--source', nargs='+',
                        help='Only include entries from these source(s)')
    parser.add_argument('--exclude-source', nargs='+',
                        help='Exclude entries from these source(s)')
    parser.add_argument('--max-resolution', type=float,
                        help='Maximum crystal structure resolution in Angstroms')
    parser.add_argument('--max-entries', type=int,
                        help='Maximum number of structures to include')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip PDB downloads (only create benchmark.csv and sp.gs)')
    args = parser.parse_args()

    # Download XLSX from Zenodo
    if not download_file(XLSX_URL, XLSX_FILENAME):
        sys.exit(1)

    # Parse and filter
    print(f"Parsing {XLSX_FILENAME}...")
    wt_only = not args.include_mutants
    entries = parse_xlsx(XLSX_FILENAME, wt_only=wt_only, sources=args.source,
                         max_resolution=args.max_resolution,
                         exclude_sources=args.exclude_source)

    if not entries:
        print("No entries match the specified filters.")
        sys.exit(1)

    # Sort by pKd (strongest binders first)
    entries.sort(key=lambda x: x['pkd'], reverse=True)

    # Limit entries
    if args.max_entries and len(entries) > args.max_entries:
        entries = entries[:args.max_entries]
        print(f"Limited to {args.max_entries} structures (sorted by pKd)")

    print(f"Selected {len(entries)} structures")

    # Write benchmark.csv
    with open('benchmark.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pdb_id', 'chain_id_1', 'chain_id_2', 'protein_1', 'protein_2',
                         'kd', 'pkd', 'dg', 'method', 'temperature', 'resolution', 'source'])
        for e in entries:
            chain_id_1 = ''.join(e['receptor_chains'])
            chain_id_2 = ''.join(e['ligand_chains'])
            writer.writerow([e['pdb_id'], chain_id_1, chain_id_2,
                             e['receptor_name'], e['ligand_name'],
                             e['kd'], f"{e['pkd']:.2f}",
                             f"{e['dg']:.2f}" if e['dg'] else '',
                             e['method'], e['temperature'],
                             e['resolution'], e['source']])
    print(f"Created benchmark.csv with {len(entries)} entries")

    # Create directories and download PDBs
    with open('sp.gs', 'w') as sp:
        sp.write("# Structure_ID\tChains_for_Protein_B\n")

        success = 0
        failed = []
        for e in entries:
            pdb_id = e['pdb_id']
            chains_to_keep = set(e['receptor_chains']) | set(e['ligand_chains'])

            if not os.path.exists(pdb_id):
                os.makedirs(pdb_id)

            input_pdb = f"{pdb_id}/input.pdb"

            if os.path.exists(input_pdb):
                print(f"{pdb_id}: input.pdb already exists, skipping download")
                chain_id_2_formatted = ','.join(e['ligand_chains'])
                sp.write(f"{pdb_id}\t{chain_id_2_formatted}\n")
                success += 1
                continue

            if args.skip_download:
                chain_id_2_formatted = ','.join(e['ligand_chains'])
                sp.write(f"{pdb_id}\t{chain_id_2_formatted}\n")
                success += 1
                continue

            if download_pdb(pdb_id, chains_to_keep, input_pdb):
                chain_id_2_formatted = ','.join(e['ligand_chains'])
                sp.write(f"{pdb_id}\t{chain_id_2_formatted}\n")
                success += 1
            else:
                failed.append(pdb_id)
                # Remove empty directory
                try:
                    os.rmdir(pdb_id)
                except OSError:
                    pass

    print(f"\nCreated sp.gs with {success} entries")
    if failed:
        print(f"Failed to download {len(failed)} PDBs: {', '.join(failed)}")
    if args.skip_download:
        print("PDB downloads skipped. Run without --skip-download to download PDBs.")
    print("Run: python ../groscore.py")


if __name__ == '__main__':
    main()
