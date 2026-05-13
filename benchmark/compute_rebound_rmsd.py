#!/usr/bin/env python3
"""
Compute backbone RMSD between NPT-equilibrated and re-bound structures
for each cycle in each finished benchmark tar.gz.

For cycle N:
  Reference: npt_cN.gro     (gathered: pbc whole → pbc cluster)
  Query:     bindrev_(N*2).gro  (pbc whole → per-chain image correction vs npt_cN)
  RMSD: Backbone fit, Backbone measurement

Usage:
  python3 compute_rebound_rmsd.py <benchmark_dir> [-o rmsd_rebound.gs]
"""

import os
import sys
import glob
import tarfile
import tempfile
import subprocess
import argparse
import numpy as np

GMXRC = '/usr/local/gromacs/bin/GMXRC'
GMX_PREFIX = f'source {GMXRC} 2>/dev/null && '

def gmx(cmd, cwd):
    full = GMX_PREFIX + cmd
    result = subprocess.run(full, shell=True, cwd=cwd, executable='/bin/bash',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode, result.stderr.decode()

# ── PBC helpers ───────────────────────────────────────────────────────────────

def count_itp_atoms(itp_path):
    """Count atom entries in the [ atoms ] section of a GROMACS .itp file."""
    in_atoms = False
    count = 0
    with open(itp_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('[') and 'atoms' in s.lower():
                in_atoms = True
                continue
            if s.startswith('[') and in_atoms:
                break
            if in_atoms and s and not s.startswith(';'):
                try:
                    int(s.split()[0])
                    count += 1
                except (ValueError, IndexError):
                    pass
    return count

def find_chain_sizes(workdir):
    """Return atom counts [n_A, n_B, …] for protein chains actually used in topol.top.

    Reads which topol_Protein_chain_*.itp files are #include'd in topol.top so
    that chain itps left over from pdb2gmx (but merged into another chain) are
    not counted twice.
    """
    topol = os.path.join(workdir, 'topol.top')
    included = []
    if os.path.isfile(topol):
        with open(topol) as f:
            for line in f:
                s = line.strip()
                if s.startswith('#include') and 'topol_Protein_chain_' in s:
                    fname = s.split('"')[1] if '"' in s else s.split("'")[1]
                    itp_path = os.path.join(workdir, fname)
                    if os.path.isfile(itp_path):
                        included.append(itp_path)

    if not included:
        # Fallback: all matching itp files
        included = sorted(glob.glob(os.path.join(workdir, 'topol_Protein_chain_*.itp')))

    sizes = []
    for itp in included:
        n = count_itp_atoms(itp)
        if n > 0:
            sizes.append(n)
    return sizes

def parse_gro_box(gro_path):
    """Return box vectors (v1, v2, v3) from the last line of a GRO file."""
    last = ''
    with open(gro_path) as f:
        for line in f:
            last = line
    parts = last.split()
    if len(parts) == 3:
        return ([float(parts[0]), 0, 0],
                [0, float(parts[1]), 0],
                [0, 0, float(parts[2])])
    # Triclinic GRO order: v1x v2y v3z v1y v1z v2x v2z v3x v3y
    v1 = [float(parts[0]), float(parts[3]), float(parts[4])]
    v2 = [float(parts[5]), float(parts[1]), float(parts[6])]
    v3 = [float(parts[7]), float(parts[8]), float(parts[2])]
    return v1, v2, v3

def nearest_image(com, ref_com, v1, v2, v3):
    """Return lattice shift t = n1*v1+n2*v2+n3*v3 (n_i ∈ {-2..2}) minimising |com+t-ref_com|."""
    best2 = float('inf')
    best = [0.0, 0.0, 0.0]
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                tx = n1*v1[0] + n2*v2[0] + n3*v3[0]
                ty = n1*v1[1] + n2*v2[1] + n3*v3[1]
                tz = n1*v1[2] + n2*v2[2] + n3*v3[2]
                d2 = ((com[0]+tx-ref_com[0])**2 +
                      (com[1]+ty-ref_com[1])**2 +
                      (com[2]+tz-ref_com[2])**2)
                if d2 < best2:
                    best2 = d2
                    best = [tx, ty, tz]
    return best

def fix_chain_images(cluster_gro, ref_gro, workdir, out_gro):
    """Correct per-chain periodic images in a pbc-cluster protein-only GRO.

    cluster_gro – protein-only GRO (pbc cluster output) of the frame to correct
    ref_gro     – protein-only reference GRO (npt_cl) or same file for self-fix

    Two modes:
      Self-fix  (cluster_gro == ref_gro): all chains anchored at their own first
                residue in cluster_gro, then sequentially propagated.  Repairs
                per-residue fragmentation of merged molecules (e.g. 1IJK β2m+peptide)
                without changing correctly placed separate-molecule chains.

      Relative-vector (cluster_gro != ref_gro): chain A is self-fixed; each
                subsequent chain X is anchored at
                  brev_chain_A_COM + (ref_chain_X_first_res - ref_chain_A_COM)
                This keeps chain X at the same spatial offset from chain A as in
                the reference frame, handling both image mismatch (1AK4) and
                merged-molecule gaps (1IJK).

    Returns (ok: bool, error_msg: str).
    """
    chain_sizes = find_chain_sizes(workdir)
    if not chain_sizes:
        return False, 'no topol_Protein_chain_*.itp found'

    n_protein = sum(chain_sizes)

    with open(os.path.join(workdir, cluster_gro)) as f:
        w_lines = f.readlines()
    with open(os.path.join(workdir, ref_gro)) as f:
        r_lines = f.readlines()

    n_w = int(w_lines[1])
    n_r = int(r_lines[1])
    if n_w < n_protein:
        return False, f'cluster_gro has {n_w} atoms, need >= {n_protein}'
    if n_r < n_protein:
        return False, f'ref_gro has {n_r} atoms, need >= {n_protein}'

    v1, v2, v3 = parse_gro_box(os.path.join(workdir, cluster_gro))
    w_prot = list(w_lines[2:2+n_protein])   # mutable
    r_prot = r_lines[2:2+n_protein]

    def xyz(line):
        return float(line[20:28]), float(line[28:36]), float(line[36:44])

    def res_com(lines_slice):
        xs = [xyz(l)[0] for l in lines_slice]
        ys = [xyz(l)[1] for l in lines_slice]
        zs = [xyz(l)[2] for l in lines_slice]
        return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

    def shift_lines(lines_slice, t):
        result = []
        for line in lines_slice:
            x, y, z = xyz(line)
            result.append('%s%8.3f%8.3f%8.3f\n' % (line[:20], x+t[0], y+t[1], z+t[2]))
        return result

    def parse_residues(atom_lines):
        boundaries = []
        cur_id = None
        cur_start = 0
        for i, line in enumerate(atom_lines):
            rid = line[0:10]
            if rid != cur_id:
                if cur_id is not None:
                    boundaries.append((cur_start, i))
                cur_id = rid
                cur_start = i
        if cur_id is not None:
            boundaries.append((cur_start, len(atom_lines)))
        return boundaries

    self_fix = (cluster_gro == ref_gro)

    # For relative-vector mode: reference chain A COM is needed for all subsequent chains
    ref_chain_A_com = None
    brev_chain_A_com = None
    if not self_fix:
        ref_chain_A_com = res_com(r_prot[:chain_sizes[0]])

    mol_start = 0
    for mol_idx, mol_size in enumerate(chain_sizes):
        mol_lines_w = w_prot[mol_start:mol_start+mol_size]
        residues = parse_residues(mol_lines_w)
        rs0, re0 = residues[0]

        if mol_idx == 0 or self_fix:
            # Self-referential: anchor at cluster_gro's own first residue
            anchor = res_com(mol_lines_w[rs0:re0])
        else:
            # Relative-vector: place chain X so it has the same offset from chain A
            # as it has in the reference frame
            mol_lines_r = r_prot[mol_start:mol_start+mol_size]
            ref_first = res_com(mol_lines_r[rs0:re0])
            anchor = tuple(brev_chain_A_com[i] + ref_first[i] - ref_chain_A_com[i]
                           for i in range(3))

        for rs, re in residues:
            res_lines = [w_prot[mol_start + rs + k] for k in range(re - rs)]
            rc = res_com(res_lines)
            t = nearest_image(rc, anchor, v1, v2, v3)
            new_lines = shift_lines(res_lines, t)
            for k, line in enumerate(new_lines):
                w_prot[mol_start + rs + k] = line
            anchor = (rc[0]+t[0], rc[1]+t[1], rc[2]+t[2])

        # After processing chain A, record its corrected COM for subsequent chains
        if mol_idx == 0 and not self_fix:
            brev_chain_A_com = res_com(w_prot[:mol_size])

        mol_start += mol_size

    out_path = os.path.join(workdir, out_gro)
    with open(out_path, 'w') as f:
        f.write('Protein chain-image corrected\n')
        f.write(f'{n_protein}\n')
        for line in w_prot:
            f.write(line)
        f.write(w_lines[-1])

    return True, ''

# ── gathering ─────────────────────────────────────────────────────────────────

def gather_gro(gro_in, tpr, ndx, gro_out, workdir, ref_gro=None):
    """Gather a single-frame GRO for RMSD comparison.

    pbc whole → pbc cluster → fix_chain_images.

    For NPT frames (ref_gro=None): self-fix mode — all chains anchored at their
    own first residue in the pbc cluster output, then sequentially propagated.
    This is identity for separate-molecule chains and repairs merged-molecule
    fragmentation (1IJK β2m+peptide).

    For bindrev frames (ref_gro = npt_cl): relative-vector mode — chain A is
    self-fixed; chains B+ are placed so their offset from chain A matches the
    npt reference frame.  Handles both different-image placement (1AK4) and
    merged-molecule gaps (1IJK).
    """
    whole = gro_in.replace('.gro', '_whole.gro')
    cluster_raw = gro_in.replace('.gro', '_clraw.gro')

    rc, err = gmx(f'echo "0" | gmx trjconv -f {gro_in} -s {tpr} '
                  f'-o {whole} -pbc whole -quiet', workdir)
    if rc != 0 or not os.path.isfile(os.path.join(workdir, whole)):
        return False, 'pbc whole failed: ' + err[-200:]

    rc, err = gmx(f'printf "Protein_Struct\\nProtein\\n" | gmx trjconv -f {whole} -s {tpr} '
                  f'-o {cluster_raw} -pbc cluster -n {ndx} -quiet', workdir)
    if rc != 0 or not os.path.isfile(os.path.join(workdir, cluster_raw)):
        return False, 'pbc cluster failed: ' + err[-200:]

    if ref_gro is not None and os.path.isfile(os.path.join(workdir, ref_gro)):
        ref = ref_gro   # relative-vector mode: chain B+ placed relative to npt chain A
    else:
        ref = cluster_raw   # self-fix mode

    ok, msg = fix_chain_images(cluster_raw, ref, workdir, gro_out)
    if not ok:
        os.rename(os.path.join(workdir, cluster_raw), os.path.join(workdir, gro_out))
        return True, ''

    try:
        os.remove(os.path.join(workdir, cluster_raw))
    except OSError:
        pass
    return True, ''

def parse_rmsd_xvg(xvg_path):
    """Return RMSD (nm) from first data line of gmx rms output."""
    with open(xvg_path) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return None

def compute_rmsd(ref_gro, query_gro, workdir, label):
    """Backbone RMSD (Å) of query_gro vs ref_gro."""
    xvg = f'rmsd_{label}.xvg'
    rc, err = gmx(f'printf "Backbone\\nBackbone\\n" | gmx rms '
                  f'-s {ref_gro} -f {query_gro} -o {xvg} -quiet', workdir)
    xvg_path = os.path.join(workdir, xvg)
    if rc != 0 or not os.path.isfile(xvg_path):
        return None, 'gmx rms failed: ' + err[-200:]
    rmsd_nm = parse_rmsd_xvg(xvg_path)
    if rmsd_nm is None:
        return None, 'could not parse xvg'
    return rmsd_nm * 10.0, ''   # nm → Å

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('benchmark_dir')
parser.add_argument('-o', '--output', default='rmsd_rebound.gs')
args = parser.parse_args()

tgz_files = sorted(glob.glob(os.path.join(args.benchmark_dir, '*.tar.gz')))
print(f'Found {len(tgz_files)} tar.gz files in {args.benchmark_dir}')

# ── intermediate files created by this script (strip before re-compressing) ──
SCRATCH = {'_whole.gro', '_cluster.gro', '_clraw.gro', 'rmsd_c'}

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
        # Extract
        with tarfile.open(tgz_path, 'r:gz') as tf:
            tf.extractall(tmpdir)

        workdir = os.path.join(tmpdir, struct_id)
        if not os.path.isdir(workdir):
            print(f'  SKIP: no subdir {struct_id} in archive')
            errors.append((struct_id, 'no subdir'))
            continue

        tpr = 'emin_solv.tpr'
        ndx = 'index.ndx'
        if not os.path.isfile(os.path.join(workdir, tpr)):
            print(f'  SKIP: {tpr} missing')
            errors.append((struct_id, f'{tpr} missing'))
            continue
        if not os.path.isfile(os.path.join(workdir, ndx)):
            print(f'  SKIP: {ndx} missing')
            errors.append((struct_id, f'{ndx} missing'))
            continue

        # Clean scratch files from old runs (if any)
        for f in os.listdir(workdir):
            if any(tag in f for tag in SCRATCH):
                os.remove(os.path.join(workdir, f))

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

            # Gather npt_cN with pbc cluster (NPT structures don't have PBC drift)
            npt_cl = f'npt_c{cycle}_cluster.gro'
            ok, err = gather_gro(npt_gro, tpr, ndx, npt_cl, workdir)
            if not ok:
                print(f'    ERROR gather npt: {err}')
                errors.append((struct_id, f'c{cycle} npt gather: {err}'))
                cycle += 1
                continue

            # Gather bindrev_K: relative-vector fix anchored to npt reference
            brev_cl = f'bindrev_{push_idx}_cluster.gro'
            ok, err = gather_gro(bindrev_gro, tpr, ndx, brev_cl, workdir, ref_gro=npt_cl)
            if not ok:
                print(f'    ERROR gather bindrev: {err}')
                errors.append((struct_id, f'c{cycle} bindrev gather: {err}'))
                cycle += 1
                continue

            # RMSD
            rmsd, err = compute_rmsd(npt_cl, brev_cl, workdir, f'c{cycle}')
            if rmsd is None:
                print(f'    ERROR rmsd: {err}')
                errors.append((struct_id, f'c{cycle} rmsd: {err}'))
            else:
                print(f'    RMSD = {rmsd:.2f} Å')
                struct_rmsds.append(rmsd)
                results.append((struct_id, cycle, rmsd))
                out_f.write(f'{struct_id}\t{cycle}\t{rmsd:.4f}\n')
                out_f.flush()

            cycle += 1

        if struct_rmsds:
            avg = np.mean(struct_rmsds)
            print(f'  → avg RMSD {avg:.2f} Å over {len(struct_rmsds)} cycles')

        # Re-compress in place
        new_tgz = tgz_path + '.tmp'
        with tarfile.open(new_tgz, 'w:gz') as tf:
            tf.add(workdir, arcname=struct_id)
        os.replace(new_tgz, tgz_path)
        print(f'  Re-compressed → {os.path.basename(tgz_path)}')

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
