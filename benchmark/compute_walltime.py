#!/usr/bin/env python3
#
# compute_walltime.py - Calculate total wall-clock time from SLURM output files
#
# Sums GROMACS wall times from all slurm-*.out files in the current directory.
# Reports per-structure and total wall time.
#
# Usage: python compute_walltime.py [-d directory]
#

import os
import sys
import glob
import argparse
import re

parser = argparse.ArgumentParser(description="Calculate total wall-clock time from SLURM output files.")
parser.add_argument('-d', '--directory', type=str, default='.', help="Directory containing slurm-*.out files (default: current)")
args = parser.parse_args()

# Find all slurm output files
slurm_files = sorted(glob.glob(os.path.join(args.directory, "slurm-*.out")))
if not slurm_files:
    print(f"No slurm-*.out files found in {args.directory}", file=sys.stderr)
    sys.exit(1)

# Parse struct_map.gs if available (maps array index to structure ID)
struct_map = {}
struct_map_path = os.path.join(args.directory, "struct_map.gs")
if os.path.isfile(struct_map_path):
    with open(struct_map_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    struct_map[int(parts[0])] = parts[1]
                except ValueError:
                    pass

# Extract wall times per slurm file
results = []  # [(structure_id, wall_seconds, n_steps)]
total_wall = 0.0

for filepath in slurm_files:
    # Extract array index from filename: slurm-JOBID_INDEX.out
    basename = os.path.basename(filepath)
    m = re.search(r'_(\d+)\.out$', basename)
    array_idx = int(m.group(1)) if m else -1
    struct_id = struct_map.get(array_idx)

    wall_sum = 0.0
    n_steps = 0
    with open(filepath) as f:
        in_timing = False
        for line in f:
            # Extract structure ID from first "Working dir:" if not in struct_map
            if struct_id is None and "Working dir:" in line:
                struct_id = line.strip().rstrip('/').rsplit('/', 1)[-1]
            if "Core t (s)   Wall t (s)" in line:
                in_timing = True
                continue
            if in_timing and line.strip().startswith("Time:"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        wall_sum += float(parts[2])
                        n_steps += 1
                    except ValueError:
                        pass
                in_timing = False

    if struct_id is None:
        struct_id = basename
    results.append((struct_id, wall_sum, n_steps))
    total_wall += wall_sum

# Merge duplicate structures (multiple SLURM submissions/restarts)
merged = {}
for struct_id, wall, n_steps in results:
    if struct_id in merged:
        merged[struct_id] = (merged[struct_id][0] + wall, merged[struct_id][1] + n_steps)
    else:
        merged[struct_id] = (wall, n_steps)

results = [(sid, w, n) for sid, (w, n) in merged.items()]
total_wall = sum(w for _, w, _ in results)

# Sort by wall time descending
results.sort(key=lambda x: -x[1])

# Print results
print(f"{'Structure':<12} {'Wall time':>12} {'GMX steps':>10}")
print("-" * 36)
for struct_id, wall, n_steps in results:
    h = wall / 3600
    print(f"{struct_id:<12} {h:>10.2f} h {n_steps:>9d}")

print("-" * 36)
n_structures = len(results)
total_h = total_wall / 3600
print(f"{'Total':<12} {total_h:>10.2f} h {sum(r[2] for r in results):>9d}")
print(f"{'Structures':<12} {n_structures:>10d}")
print(f"{'Avg/struct':<12} {total_h / n_structures:>10.2f} h")
