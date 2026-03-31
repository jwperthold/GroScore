#!/usr/bin/env python3
"""
Filter PPB-Affinity benchmark for problematic duplicate-entity structures.

Case 1: Same protein entity appears in both protein A and B → REMOVE (wrong assignment)
Case 2: Duplicate entity on one side → check inter-chain contacts:
        - If duplicate chains have contacts with each other → functional homodimer → KEEP
        - If no contacts → independent copies → REMOVE

Reads benchmark.csv, queries RCSB for entity info, downloads CIF for
contact analysis, outputs filtered benchmark.csv.
"""

import csv
import os
import sys
import urllib.request
import json
import math

CONTACT_CUTOFF = 5.0  # Angstrom — inter-chain contact cutoff
MIN_CONTACTS = 10     # minimum contacts to consider a functional dimer

def get_entity_info(pdb_id):
    """Get chain->entity name mapping from RCSB."""
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
    """Check if two chains have contacts (CA-CA distance < cutoff).
    Downloads mmCIF and checks distances between CA atoms."""
    url = "https://files.rcsb.org/download/%s.cif" % pdb_id
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        cif = resp.read().decode()
    except Exception as e:
        print("    Warning: could not download %s (%s)" % (pdb_id, e))
        return True  # assume contacts if we can't check

    # Parse mmCIF
    lines = cif.split('\n')
    cols = [l.strip() for l in lines if l.startswith('_atom_site.')]
    auth_chain_idx = None
    atom_name_idx = None
    x_idx = y_idx = z_idx = None
    model_idx = None
    for i, c in enumerate(cols):
        if c == '_atom_site.auth_asym_id': auth_chain_idx = i
        elif c == '_atom_site.label_atom_id': atom_name_idx = i
        elif c == '_atom_site.Cartn_x': x_idx = i
        elif c == '_atom_site.Cartn_y': y_idx = i
        elif c == '_atom_site.Cartn_z': z_idx = i
        elif c == '_atom_site.pdbx_PDB_model_num': model_idx = i

    if any(v is None for v in [auth_chain_idx, atom_name_idx, x_idx, y_idx, z_idx]):
        return True  # can't parse, assume contacts

    # Collect CA coordinates per chain
    coords1 = []
    coords2 = []
    in_atoms = False
    for line in lines:
        if line.startswith('_atom_site.'):
            in_atoms = True
        elif in_atoms:
            if line.startswith('_') or line.startswith('#') or line.startswith('loop_'):
                break
            fields = line.split()
            max_idx = max(auth_chain_idx, atom_name_idx, x_idx, y_idx, z_idx)
            if len(fields) <= max_idx:
                continue
            if model_idx is not None and len(fields) > model_idx and fields[model_idx] != '1':
                continue
            if fields[atom_name_idx].strip('"') != 'CA':
                continue
            ch = fields[auth_chain_idx]
            try:
                x, y, z = float(fields[x_idx]), float(fields[y_idx]), float(fields[z_idx])
            except ValueError:
                continue
            if ch == chain1:
                coords1.append((x, y, z))
            elif ch == chain2:
                coords2.append((x, y, z))

    if not coords1 or not coords2:
        return False  # no atoms in one chain

    # Count contacts
    n_contacts = 0
    for x1, y1, z1 in coords1:
        for x2, y2, z2 in coords2:
            d = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            if d < CONTACT_CUTOFF:
                n_contacts += 1
                if n_contacts >= MIN_CONTACTS:
                    return True  # early exit

    return n_contacts >= MIN_CONTACTS


# ---- Main ----
entries = []
with open("benchmark.csv") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        entries.append(row)

print("Loaded %d entries from benchmark.csv" % len(entries))

case1_removed = []
case2_flagged = []
case2_kept = []
clean_entries = []
n_checked = 0

for i, row in enumerate(entries):
    pdb = row["pdb_id"]
    c1 = set(row["chain_id_1"].strip())
    c2 = set(row["chain_id_2"].strip())

    # Skip simple cases (single chain on each side)
    if len(c1) <= 1 and len(c2) <= 1:
        clean_entries.append(row)
        continue

    # Get entity info
    try:
        chain_entity = get_entity_info(pdb)
    except Exception:
        clean_entries.append(row)  # keep if can't check
        continue

    a_entities = {c: chain_entity.get(c, "?") for c in c1}
    b_entities = {c: chain_entity.get(c, "?") for c in c2}
    a_names = set(a_entities.values())
    b_names = set(b_entities.values())

    # Case 1: same entity in both A and B
    cross = a_names & b_names - {"?"}
    if cross:
        case1_removed.append((pdb, row["chain_id_1"], row["chain_id_2"], cross))
        print("  REMOVE (case 1) %s: same entity in A+B: %s" % (pdb, cross))
        continue

    # Case 2: duplicate entity on one side
    a_dupes = len(list(a_entities.values())) != len(set(a_entities.values())) and len(a_entities) > 1
    b_dupes = len(list(b_entities.values())) != len(set(b_entities.values())) and len(b_entities) > 1

    if a_dupes or b_dupes:
        # Find which chains are duplicates
        keep = True
        if a_dupes:
            # Find pairs of duplicate chains in A
            seen = {}
            for ch, name in a_entities.items():
                if name in seen:
                    # Check contacts between these duplicate chains
                    has_contacts = check_interchain_contacts(pdb, seen[name], ch)
                    if not has_contacts:
                        keep = False
                        case2_flagged.append((pdb, "A", seen[name], ch, name))
                        print("  REMOVE (case 2) %s: A chains %s,%s (%s) — no contacts" % (pdb, seen[name], ch, name))
                    else:
                        case2_kept.append((pdb, "A", seen[name], ch, name))
                        print("  KEEP   (case 2) %s: A chains %s,%s (%s) — functional homodimer" % (pdb, seen[name], ch, name))
                else:
                    seen[name] = ch

        if b_dupes:
            seen = {}
            for ch, name in b_entities.items():
                if name in seen:
                    has_contacts = check_interchain_contacts(pdb, seen[name], ch)
                    if not has_contacts:
                        keep = False
                        case2_flagged.append((pdb, "B", seen[name], ch, name))
                        print("  REMOVE (case 2) %s: B chains %s,%s (%s) — no contacts" % (pdb, seen[name], ch, name))
                    else:
                        case2_kept.append((pdb, "B", seen[name], ch, name))
                        print("  KEEP   (case 2) %s: B chains %s,%s (%s) — functional homodimer" % (pdb, seen[name], ch, name))
                else:
                    seen[name] = ch

        if keep:
            clean_entries.append(row)
        continue

    clean_entries.append(row)

    n_checked += 1
    if (i+1) % 200 == 0:
        print("  Progress: %d/%d entries checked..." % (i+1, len(entries)), file=sys.stderr)

# Write filtered benchmark.csv
with open("benchmark_filtered.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for row in clean_entries:
        writer.writerow(row)

print("\n" + "="*60)
print("FILTERING RESULTS")
print("="*60)
print("Original entries:        %d" % len(entries))
print("Case 1 removed:          %d (same entity in A+B)" % len(case1_removed))
print("Case 2 removed:          %d (independent copies, no contacts)" % len(case2_flagged))
print("Case 2 kept:             %d (functional homodimers)" % len(case2_kept))
print("Clean entries:           %d" % len(clean_entries))
print("\nWrote benchmark_filtered.csv")

if case1_removed:
    print("\nCase 1 removed structures:")
    for pdb, c1, c2, cross in case1_removed:
        print("  %s (A=%s B=%s): %s" % (pdb, c1, c2, cross))

if case2_flagged:
    print("\nCase 2 removed structures (independent copies):")
    for pdb, side, ch1, ch2, name in case2_flagged:
        print("  %s: %s chains %s,%s (%s)" % (pdb, side, ch1, ch2, name))
