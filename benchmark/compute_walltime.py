#!/usr/bin/env python3
#
# compute_walltime.py - Calculate total wall-clock time and FLOPS from SLURM output files
#
# Sums GROMACS wall times from slurm-*.out files and total FLOPS from .log files
# inside tar.gz archives. Reports per-structure and total statistics.
#
# Usage: python compute_walltime.py [-d directory]
#

import os
import sys
import glob
import argparse
import re
import tarfile

parser = argparse.ArgumentParser(description="Calculate total wall-clock time and FLOPS from SLURM output files.")
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

for filepath in slurm_files:
    basename = os.path.basename(filepath)
    m = re.search(r'_(\d+)\.out$', basename)
    array_idx = int(m.group(1)) if m else -1
    struct_id = struct_map.get(array_idx)

    wall_sum = 0.0
    n_steps = 0
    with open(filepath) as f:
        in_timing = False
        for line in f:
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

# Extract FLOPS from .log files inside tar.gz archives and unarchived directories
# GROMACS logs contain a table with "M-Number  M-Flops  % Flops" header,
# followed by per-category rows, ending with a "Total" row containing the
# total M-Flops (millions of floating point operations) for that mdrun.
flops_per_struct = {}  # struct_id -> total M-Flops

def extract_mflops_from_lines(line_iter):
    """Extract total M-Flops from GROMACS log lines (str iterator)."""
    total = 0.0
    in_table = False
    for line in line_iter:
        if 'M-Number' in line and 'M-Flops' in line:
            in_table = True
            continue
        if in_table and 'Total' in line:
            # Extract large number (M-Flops > 1e6) from the Total line
            for val in re.findall(r'[\d.]+', line):
                v = float(val)
                if v > 1e6:
                    total += v
                    break
            in_table = False
    return total

tar_files = sorted(glob.glob(os.path.join(args.directory, "*.tar.gz")))
if tar_files:
    print(f"Reading FLOPS from {len(tar_files)} archive(s)...", file=sys.stderr)
for tar_path in tar_files:
    struct_id = os.path.basename(tar_path).replace('.tar.gz', '')
    total_mflops = 0.0
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if not member.name.endswith('.log'):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                lines = (raw.decode('utf-8', errors='replace') for raw in f)
                total_mflops += extract_mflops_from_lines(lines)
    except Exception:
        pass
    if total_mflops > 0:
        flops_per_struct[struct_id] = total_mflops

# Also check unarchived structure directories for .log files
struct_dirs = [d for d in glob.glob(os.path.join(args.directory, "*")) if os.path.isdir(d)]
for struct_dir in struct_dirs:
    struct_id = os.path.basename(struct_dir)
    if struct_id in flops_per_struct:
        continue
    total_mflops = 0.0
    for log_path in glob.glob(os.path.join(struct_dir, "*.log")):
        try:
            with open(log_path) as f:
                total_mflops += extract_mflops_from_lines(f)
        except Exception:
            pass
    if total_mflops > 0:
        flops_per_struct[struct_id] = total_mflops

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

has_flops = len(flops_per_struct) > 0

# Print results
if has_flops:
    print(f"{'Structure':<12} {'Wall time':>12} {'GMX steps':>10} {'PFLOP':>10}")
    print("-" * 48)
    for struct_id, wall, n_steps in results:
        h = wall / 3600
        mflops = flops_per_struct.get(struct_id, 0)
        pflops = mflops / 1e9  # M-Flops -> PFLOP
        if pflops > 0:
            print(f"{struct_id:<12} {h:>10.2f} h {n_steps:>9d} {pflops:>10.2f}")
        else:
            print(f"{struct_id:<12} {h:>10.2f} h {n_steps:>9d} {'n/a':>10}")
else:
    print(f"{'Structure':<12} {'Wall time':>12} {'GMX steps':>10}")
    print("-" * 36)
    for struct_id, wall, n_steps in results:
        h = wall / 3600
        print(f"{struct_id:<12} {h:>10.2f} h {n_steps:>9d}")

# Summary
sep_len = 48 if has_flops else 36
print("-" * sep_len)
n_structures = len(results)
total_h = total_wall / 3600
total_pflops = sum(v / 1e9 for v in flops_per_struct.values())

if has_flops:
    print(f"{'Total':<12} {total_h:>10.2f} h {sum(r[2] for r in results):>9d} {total_pflops:>10.2f}")
else:
    print(f"{'Total':<12} {total_h:>10.2f} h {sum(r[2] for r in results):>9d}")
print(f"{'Structures':<12} {n_structures:>10d}")
print(f"{'Avg/struct':<12} {total_h / n_structures:>10.2f} h")
if has_flops:
    print(f"{'Avg PFLOP':<12} {total_pflops / n_structures:>10.2f}")
