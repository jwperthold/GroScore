#!/usr/bin/env python3
"""
Compute backbone RMSD between NPT-equilibrated and re-bound structures
for each cycle in each finished benchmark tar.gz.

For cycle N:
  Reference: npt_cN.gro     (pbc whole → Protein extract → min-image self-fix)
  Query:     bindrev_(N*2).gro  (same pipeline)
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

def fix_chain_images(protein_gro, workdir, out_gro, ref_chain_coms=None):
    """Correct per-chain periodic images in a protein-only GRO (from pbc whole).

    Chain A: sequential nearest-image propagation from first residue.
    Chain B+ (ref_chain_coms=None / self-fix / NPT mode):
        first residue placed at min-image from chain A COM, then propagated.
    Chain B+ (ref_chain_coms provided / cross-frame / brev mode):
        whole-chain COM placed at min-image from ref_chain_coms[mol_idx], then propagated.
        Robust because successful rebinding puts brev chain B near NPT chain B.

    Returns (ok: bool, error_msg: str, out_chain_coms: list).
    """
    chain_sizes = find_chain_sizes(workdir)
    if not chain_sizes:
        return False, 'no topol_Protein_chain_*.itp found', []

    n_protein = sum(chain_sizes)

    with open(os.path.join(workdir, protein_gro)) as f:
        lines = f.readlines()

    n = int(lines[1])
    if n < n_protein:
        return False, f'{protein_gro} has {n} atoms, need >= {n_protein}', []

    v1, v2, v3 = parse_gro_box(os.path.join(workdir, protein_gro))
    w_prot = list(lines[2:2+n_protein])

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

    chain_A_com = None
    out_chain_coms = []
    mol_start = 0
    for mol_idx, mol_size in enumerate(chain_sizes):
        mol_lines = w_prot[mol_start:mol_start+mol_size]
        residues = parse_residues(mol_lines)
        rs0, re0 = residues[0]

        if mol_idx == 0:
            anchor = res_com(mol_lines[rs0:re0])
        else:
            if ref_chain_coms is not None and mol_idx < len(ref_chain_coms):
                # Cross-frame: rigid-body shift chain B to the nearest periodic image of
                # the expected position (chain_A_com + NPT A→B relative vector).
                # Uses the raw pbc-whole COM before sequential propagation — for a clean
                # wrong-image error the raw COM is exactly one lattice vector away from
                # expected, giving d_after≈0.  The caller generates both self-fix and
                # cross-frame candidates and takes the minimum RMSD, so no additional
                # threshold is needed here.
                raw_com = res_com(mol_lines)
                expected_com = (chain_A_com[0] + ref_chain_coms[mol_idx][0] - ref_chain_coms[0][0],
                                chain_A_com[1] + ref_chain_coms[mol_idx][1] - ref_chain_coms[0][1],
                                chain_A_com[2] + ref_chain_coms[mol_idx][2] - ref_chain_coms[0][2])
                t_pre = nearest_image(raw_com, expected_com, v1, v2, v3)
                shifted = shift_lines(mol_lines, t_pre)
                for k in range(mol_size):
                    w_prot[mol_start + k] = shifted[k]
                mol_lines = shifted
            else:
                # Self-fix: place first residue at nearest-image to chain A COM.
                raw_first = res_com(mol_lines[rs0:re0])
                t_bulk = nearest_image(raw_first, chain_A_com, v1, v2, v3)
                shifted = shift_lines(mol_lines, t_bulk)
                for k in range(mol_size):
                    w_prot[mol_start + k] = shifted[k]
                mol_lines = shifted
            anchor = res_com(mol_lines[rs0:re0])

        for rs, re in residues:
            res_lines = [w_prot[mol_start+rs+k] for k in range(re-rs)]
            rc = res_com(res_lines)
            t = nearest_image(rc, anchor, v1, v2, v3)
            new_lines = shift_lines(res_lines, t)
            for k, line in enumerate(new_lines):
                w_prot[mol_start+rs+k] = line
            anchor = (rc[0]+t[0], rc[1]+t[1], rc[2]+t[2])

        out_chain_coms.append(res_com(w_prot[mol_start:mol_start+mol_size]))
        if mol_idx == 0:
            chain_A_com = out_chain_coms[0]
        mol_start += mol_size

    out_path = os.path.join(workdir, out_gro)
    with open(out_path, 'w') as f:
        f.write('Protein image-corrected\n')
        f.write(str(n_protein) + '\n')
        for line in w_prot:
            f.write(line)
        f.write(lines[-1])

    return True, '', out_chain_coms

# ── gathering ─────────────────────────────────────────────────────────────────

def gather_gro(gro_in, tpr, ndx, gro_out, workdir, ref_chain_coms=None):
    """Gather a single-frame GRO for RMSD comparison.

    pbc whole → extract Protein group → fix_chain_images.
    ref_chain_coms=None: self-fix (NPT mode).
    ref_chain_coms=list: cross-frame COM placement (brev mode).
    Returns (ok, error_msg, out_chain_coms).
    """
    whole = gro_in.replace('.gro', '_whole.gro')
    protein_raw = gro_in.replace('.gro', '_prot.gro')

    rc, err = gmx(f'echo "0" | gmx trjconv -f {gro_in} -s {tpr} '
                  f'-o {whole} -pbc whole -quiet', workdir)
    if rc != 0 or not os.path.isfile(os.path.join(workdir, whole)):
        return False, 'pbc whole failed: ' + err[-200:], []

    rc, err = gmx(f'echo "Protein" | gmx trjconv -f {whole} -s {tpr} '
                  f'-o {protein_raw} -n {ndx} -quiet', workdir)
    if rc != 0 or not os.path.isfile(os.path.join(workdir, protein_raw)):
        return False, 'protein extract failed: ' + err[-200:], []

    ok, msg, chain_coms = fix_chain_images(protein_raw, workdir, gro_out, ref_chain_coms)
    if not ok:
        os.rename(os.path.join(workdir, protein_raw), os.path.join(workdir, gro_out))
        return True, '', []

    try:
        os.remove(os.path.join(workdir, protein_raw))
    except OSError:
        pass
    return True, '', chain_coms

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
SCRATCH = {'_whole.gro', '_cluster.gro', '_clraw.gro', '_prot.gro', 'rmsd_c',
           '_cl_sf.gro', '_cl_cf.gro', '_cl_pc.gro'}

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

            # Gather npt_cN: pbc whole → Protein extract → self-fix (first-residue anchor)
            npt_cl = f'npt_c{cycle}_cluster.gro'
            ok, err, npt_chain_coms = gather_gro(npt_gro, tpr, ndx, npt_cl, workdir)
            if not ok:
                print(f'    ERROR gather npt: {err}')
                errors.append((struct_id, f'c{cycle} npt gather: {err}'))
                cycle += 1
                continue

            # Brev: pbc whole + protein extract once, then two image-fix candidates.
            brev_whole = f'bindrev_{push_idx}_whole.gro'
            brev_prot  = f'bindrev_{push_idx}_prot.gro'
            rc, err = gmx(f'echo "0" | gmx trjconv -f {bindrev_gro} -s {tpr} '
                          f'-o {brev_whole} -pbc whole -quiet', workdir)
            if rc != 0 or not os.path.isfile(os.path.join(workdir, brev_whole)):
                print(f'    ERROR brev pbc whole: {err[-100:]}')
                errors.append((struct_id, f'c{cycle} brev whole'))
                cycle += 1; continue
            rc, err = gmx(f'echo "Protein" | gmx trjconv -f {brev_whole} -s {tpr} '
                          f'-o {brev_prot} -n {ndx} -quiet', workdir)
            if rc != 0 or not os.path.isfile(os.path.join(workdir, brev_prot)):
                print(f'    ERROR brev protein extract: {err[-100:]}')
                errors.append((struct_id, f'c{cycle} brev prot'))
                cycle += 1; continue

            # Candidate A: self-fix only (first-residue anchor)
            brev_cl_sf = f'bindrev_{push_idx}_cl_sf.gro'
            fix_chain_images(brev_prot, workdir, brev_cl_sf, None)

            # Candidate B: cross-frame pre-correction (raw pbc-whole COM → expected)
            brev_cl_cf = f'bindrev_{push_idx}_cl_cf.gro'
            fix_chain_images(brev_prot, workdir, brev_cl_cf, npt_chain_coms)

            # Candidate C: standard pbc cluster (may fail for some systems)
            brev_cl_pc = f'bindrev_{push_idx}_cl_pc.gro'
            rc_pc, _ = gmx(f'printf "Protein\\nProtein\\n" | gmx trjconv '
                           f'-f {brev_whole} -s {tpr} -o {brev_cl_pc} '
                           f'-pbc cluster -n {ndx} -quiet', workdir)
            pc_ok = rc_pc == 0 and os.path.isfile(os.path.join(workdir, brev_cl_pc))

            # RMSD for each candidate; minimum = correct periodic image.
            rmsd_sf, _ = compute_rmsd(npt_cl, brev_cl_sf, workdir, f'c{cycle}_sf')
            rmsd_cf, _ = compute_rmsd(npt_cl, brev_cl_cf, workdir, f'c{cycle}_cf')
            rmsd_pc, _ = compute_rmsd(npt_cl, brev_cl_pc, workdir, f'c{cycle}_pc') \
                         if pc_ok else (None, '')
            candidates = [(v, lbl) for v, lbl in
                          [(rmsd_sf, 'sf'), (rmsd_cf, 'cf'), (rmsd_pc, 'pc')]
                          if v is not None]
            if not candidates:
                print(f'    ERROR: no valid RMSD for any candidate')
                errors.append((struct_id, f'c{cycle}: no RMSD'))
            else:
                rmsd, method = min(candidates, key=lambda x: x[0])
                sf_s = f'{rmsd_sf:.2f}' if rmsd_sf is not None else 'err'
                cf_s = f'{rmsd_cf:.2f}' if rmsd_cf is not None else 'err'
                pc_s = f'{rmsd_pc:.2f}' if rmsd_pc is not None else 'n/a'
                print(f'    RMSD = {rmsd:.2f} Å  [sf={sf_s}, cf={cf_s}, pc={pc_s}, used={method}]')
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
