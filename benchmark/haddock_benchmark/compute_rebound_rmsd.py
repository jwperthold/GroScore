#!/usr/bin/env python3
"""
Compute backbone RMSD between NPT-equilibrated and re-bound structures
for each cycle in each finished benchmark tar.gz.

For cycle N:
  Reference: npt_cN.gro
  Query:     bindrev_(N*2).gro
The per-cycle measurement itself lives in utils/rebound_rmsd.py, which job.run
also calls, so every production run carries the same numbers; this script only
walks the archives of a finished benchmark directory and aggregates them.

For runs made after the check became routine the values are already in the
results files (third column of results_<even>.gs) and this script is not needed.

Usage:
  python3 compute_rebound_rmsd.py <benchmark_dir> [-o rmsd_rebound.gs]
"""

import os
import sys
import glob
import tarfile
import tempfile
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'utils'))
from rebound_rmsd import rebound_rmsd   # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('benchmark_dir')
parser.add_argument('-o', '--output', default='rmsd_rebound.gs')
parser.add_argument('--gmxrc', default='/usr/local/gromacs/bin/GMXRC',
                    help="GMXRC to source before calling gmx")
args = parser.parse_args()

tgz_files = sorted(glob.glob(os.path.join(args.benchmark_dir, '*.tar.gz')))
print(f'Found {len(tgz_files)} tar.gz files in {args.benchmark_dir}')

results = []   # list of (struct_id, cycle, rmsd_ang)
errors  = []

output_path = os.path.join(args.benchmark_dir, args.output)
out_f = open(output_path, 'w')
out_f.write('# Backbone RMSD (Angstrom) between NPT-equilibrated and re-bound structures\n')
out_f.write('# Structure_ID\tCycle\tRMSD_A\n')
out_f.flush()

for tgz_path in tgz_files:
    struct_id = os.path.basename(tgz_path).replace('.tar.gz', '')
    print(f'\n=== {struct_id} ===', flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tgz_path, 'r:gz') as tf:
            tf.extractall(tmpdir)

        workdir = os.path.join(tmpdir, struct_id)
        if not os.path.isdir(workdir):
            print(f'  SKIP: no subdir {struct_id} in archive')
            errors.append((struct_id, 'no subdir'))
            continue

        tpr = 'emin_solv.tpr'
        ndx = 'index.ndx'
        missing = [f for f in (tpr, ndx) if not os.path.isfile(os.path.join(workdir, f))]
        if missing:
            print(f'  SKIP: {missing[0]} missing')
            errors.append((struct_id, f'{missing[0]} missing'))
            continue

        # Walk cycles: npt_cN pairs with bindrev_(N*2)
        struct_rmsds = []
        cycle = 1
        while True:
            push_idx = cycle * 2
            npt_gro     = f'npt_c{cycle}.gro'
            bindrev_gro = f'bindrev_{push_idx}.gro'

            if not os.path.isfile(os.path.join(workdir, npt_gro)) or \
               not os.path.isfile(os.path.join(workdir, bindrev_gro)):
                break

            print(f'  Cycle {cycle}: {npt_gro} ↔ {bindrev_gro}', flush=True)
            rmsd, err, detail = rebound_rmsd(workdir, npt_gro, bindrev_gro,
                                             tpr=tpr, ndx=ndx, tag=f'c{cycle}',
                                             gmxrc=args.gmxrc)
            if rmsd is None:
                print(f'    ERROR: {err}')
                errors.append((struct_id, f'c{cycle}: {err}'))
            else:
                print(f'    RMSD = {rmsd:.2f} Å  [{detail}]')
                struct_rmsds.append(rmsd)
                results.append((struct_id, cycle, rmsd))
                out_f.write(f'{struct_id}\t{cycle}\t{rmsd:.4f}\n')
                out_f.flush()

            cycle += 1

        if struct_rmsds:
            avg = np.mean(struct_rmsds)
            print(f'  → avg RMSD {avg:.2f} Å over {len(struct_rmsds)} cycles')

        # The archives stay untouched: rebound_rmsd() removes its own scratch
        # files, so re-compressing would only rewrite an identical archive (and
        # risk damaging it if the run were interrupted mid-write).

out_f.close()
print(f'\nResults written to {output_path}')
print(f'{len(results)} measurements across {len(set(r[0] for r in results))} structures')

if results:
    all_r = [r[2] for r in results]
    per_struct = {}
    for sid, _, r in results:
        per_struct.setdefault(sid, []).append(r)
    ps = [np.mean(v) for v in per_struct.values()]
    print(f'\nPer-cycle : mean={np.mean(all_r):.2f} Å  median={np.median(all_r):.2f} Å  std={np.std(all_r):.2f} Å')
    print(f'Per-struct: mean={np.mean(ps):.2f} Å  median={np.median(ps):.2f} Å  std={np.std(ps):.2f} Å')

if errors:
    print(f'\n{len(errors)} errors:')
    for sid, msg in errors:
        print(f'  {sid}: {msg}')
