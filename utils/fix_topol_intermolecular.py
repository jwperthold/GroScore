#!/usr/bin/env python3
#
# fix_topol_intermolecular.py - Fix topology after gmx solvate/genion
#
# gmx solvate and gmx genion append molecule entries (SOL, NA, CL) to the
# end of topol.top. If [ intermolecular_interactions ] is present, these
# entries end up inside it instead of in [ molecules ]. This script moves
# the [ intermolecular_interactions ] block back to the very end.
#

import argparse

parser = argparse.ArgumentParser(description="Fix topology intermolecular section ordering.")
parser.add_argument('-p', '--topol', type=str, default="topol.top", help="Topology file")
args = parser.parse_args()

with open(args.topol) as f:
    content = f.read()

# Find the [ intermolecular_interactions ] section
marker = "[ intermolecular_interactions ]"
idx = content.find(marker)

if idx < 0:
    # No intermolecular section — nothing to fix
    exit(0)

# Find the comment line before the marker (the "Ion coordination restraints" comment)
# Search backwards from idx for the start of the comment block
comment_start = idx
lines_before = content[:idx].rstrip()
# Check if there's a comment line immediately before
last_newline = lines_before.rfind('\n')
if last_newline >= 0:
    potential_comment = lines_before[last_newline + 1:]
    if potential_comment.strip().startswith(';'):
        comment_start = last_newline + 1

# Everything before the intermolecular section
before = content[:comment_start].rstrip('\n')

# The intermolecular section itself (from comment to end of file)
inter_and_rest = content[comment_start:]

# Split: find where the intermolecular bonds end and stray molecule entries begin
# The intermolecular section has [ bonds ] entries (lines starting with digits)
# Stray entries from gmx solvate/genion are like "SOL  19184" or "NA   10"
lines = inter_and_rest.split('\n')
inter_lines = []
stray_mol_lines = []

in_bonds = False
for line in lines:
    stripped = line.strip()

    # Detect stray molecule entries (name + count, no leading spaces typical of bonds)
    # Bond lines start with spaces + digit; molecule lines start with a word
    if in_bonds and stripped and not stripped.startswith(';') and not stripped.startswith('['):
        parts = stripped.split()
        if len(parts) == 2 and not parts[0][0].isdigit():
            # This looks like a molecule entry (e.g., "SOL  19184")
            stray_mol_lines.append(stripped)
            continue

    if '[ bonds ]' in stripped:
        in_bonds = True

    inter_lines.append(line)

if not stray_mol_lines:
    # Nothing to fix
    exit(0)

# Reconstruct: before + stray molecules in [ molecules ] + intermolecular at end
# The stray molecules need to go into [ molecules ] which is in the 'before' section
inter_block = '\n'.join(inter_lines).rstrip('\n')

with open(args.topol, 'w') as f:
    f.write(before)
    f.write('\n')
    for mol_line in stray_mol_lines:
        f.write(mol_line + '\n')
    f.write('\n')
    f.write(inter_block)
    f.write('\n')

print(f"Fixed topology: moved {len(stray_mol_lines)} molecule entry/entries before [ intermolecular_interactions ]")
