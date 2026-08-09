#!/usr/bin/env python3
"""
Backbone RMSD between the bound (NPT-equilibrated) and the re-bound structure of
one cycle -- the rebinding sanity check.

A cycle pulls the two binding partners apart and pushes them back together. If
the push really re-formed the original complex, the backbone of the re-bound
structure must superimpose on the backbone of the bound reference; a large RMSD
means the partners came back in a different pose (or not at all), so the work
values of that cycle describe something other than the intended binding event.

The comparison is done on the protein only, after correcting periodic images:
`gmx trjconv -pbc whole` fixes molecules broken across the box, but says nothing
about which periodic image each chain sits in, and a chain that is a whole box
vector away from its partner would give a meaningless RMSD. Three image
corrections are therefore generated per frame and the smallest RMSD over the
3x3 grid is taken:

  sf  self-fix        : sequential nearest-image propagation, residue by residue,
                        chain B anchored to chain A
  pc  pbc cluster     : gmx trjconv -pbc cluster
  cf  combined/cross  : reference -> self-fix applied on top of pbc cluster
                        query     -> chain B placed at the image nearest to the
                                     position expected from the reference frame

Used routinely by job.run / job_fe.run (one value per cycle, written into the
results files) and by benchmark/haddock_benchmark/compute_rebound_rmsd.py.

Usage (prints the RMSD in Angstrom, or NaN if it could not be computed):
  python3 rebound_rmsd.py --ref npt_c1.gro --query bindrev_2.gro
"""

import os
import sys
import glob
import argparse
import subprocess

# ── GROMACS invocation ────────────────────────────────────────────────────────

def _gmx_env():
    """Environment for gmx calls: never write #backup# files (inode pressure)."""
    env = dict(os.environ)
    env['GMX_MAXBACKUP'] = '-1'
    return env


def gmx(cmd, cwd, gmxrc=None):
    """Run a gmx command in cwd. Returns (returncode, stderr)."""
    full = ('source %s 2>/dev/null && ' % gmxrc if gmxrc else '') + cmd
    result = subprocess.run(full, shell=True, cwd=cwd, executable='/bin/bash',
                            env=_gmx_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode, result.stderr.decode('utf-8', 'replace')

# ── periodic-image helpers ────────────────────────────────────────────────────

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


def find_chain_sizes(workdir, topol='topol.top'):
    """Return atom counts [n_A, n_B, …] for protein chains actually used in topol.top.

    Reads which topol_Protein_chain_*.itp files are #include'd in topol.top so
    that chain itps left over from pdb2gmx (but merged into another chain) are
    not counted twice.
    """
    topol_path = os.path.join(workdir, topol)
    included = []
    if os.path.isfile(topol_path):
        with open(topol_path) as f:
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
    Chain B+ (ref_chain_coms=None / self-fix / reference mode):
        first residue placed at min-image from chain A COM, then propagated.
    Chain B+ (ref_chain_coms provided / cross-frame / query mode):
        whole-chain COM placed at min-image from ref_chain_coms[mol_idx], then propagated.
        Robust because successful rebinding puts the query chain B near the
        reference chain B.

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
        return False, '%s has %d atoms, need >= %d' % (protein_gro, n, n_protein), []

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
                # the expected position (chain_A_com + reference A→B relative vector).
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

# ── RMSD ──────────────────────────────────────────────────────────────────────

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


def compute_rmsd(ref_gro, query_gro, workdir, label, gmxrc=None):
    """Backbone RMSD (Å) of query_gro vs ref_gro."""
    xvg = 'rmsd_%s.xvg' % label
    rc, err = gmx('printf "Backbone\\nBackbone\\n" | gmx rms '
                  '-s %s -f %s -o %s -quiet' % (ref_gro, query_gro, xvg),
                  workdir, gmxrc)
    xvg_path = os.path.join(workdir, xvg)
    if rc != 0 or not os.path.isfile(xvg_path):
        return None, 'gmx rms failed: ' + err[-200:]
    rmsd_nm = parse_rmsd_xvg(xvg_path)
    if rmsd_nm is None:
        return None, 'could not parse xvg'
    return rmsd_nm * 10.0, ''   # nm → Å

# ── main entry point ──────────────────────────────────────────────────────────

def rebound_rmsd(workdir, ref_gro, query_gro, tpr='emin_solv.tpr',
                 ndx='index.ndx', tag=None, cleanup=True, gmxrc=None):
    """Backbone RMSD (Å) between the bound reference and the re-bound query frame.

    Both frames are made whole, reduced to the Protein group and image-corrected
    in three ways each; the minimum of the resulting 3x3 RMSD grid is returned.

    Returns (rmsd_angstrom_or_None, error_message, detail) where detail is the
    "(ref=<mode>, query=<mode>)" combination that won.
    """
    if tag is None:
        tag = os.path.splitext(os.path.basename(query_gro))[0]

    for f in (ref_gro, query_gro, tpr, ndx):
        if not os.path.isfile(os.path.join(workdir, f)):
            return None, 'missing %s' % f, ''

    ref_base = 'rbref_%s' % tag
    qry_base = 'rbqry_%s' % tag
    scratch = []

    def path(name):
        return os.path.join(workdir, name)

    try:
        # ── reference frame ───────────────────────────────────────────────────
        ref_whole = ref_base + '_whole.gro'
        ref_prot = ref_base + '_prot.gro'
        scratch += [ref_whole, ref_prot]
        rc, err = gmx('echo "0" | gmx trjconv -f %s -s %s -o %s -pbc whole -quiet'
                      % (ref_gro, tpr, ref_whole), workdir, gmxrc)
        if rc != 0 or not os.path.isfile(path(ref_whole)):
            return None, 'ref pbc whole failed: ' + err[-200:], ''
        rc, err = gmx('echo "Protein" | gmx trjconv -f %s -s %s -o %s -n %s -quiet'
                      % (ref_whole, tpr, ref_prot, ndx), workdir, gmxrc)
        if rc != 0 or not os.path.isfile(path(ref_prot)):
            return None, 'ref protein extract failed: ' + err[-200:], ''

        # sf: self-fix propagation
        ref_sf = ref_base + '_cl_sf.gro'
        scratch.append(ref_sf)
        ok_sf, msg, ref_coms_sf = fix_chain_images(ref_prot, workdir, ref_sf, None)
        if not ok_sf:
            # Without chain sizes there is no image correction to apply; compare
            # the pbc-whole protein directly rather than reporting nothing.
            ref_sf, ref_coms_sf = ref_prot, None

        # pc: pbc cluster
        ref_pc = ref_base + '_cl_pc.gro'
        scratch.append(ref_pc)
        rc, _ = gmx('printf "Protein\\nProtein\\n" | gmx trjconv -f %s -s %s -o %s '
                    '-pbc cluster -n %s -quiet' % (ref_whole, tpr, ref_pc, ndx),
                    workdir, gmxrc)
        ref_pc_ok = rc == 0 and os.path.isfile(path(ref_pc))

        # cf: self-fix applied to the pbc-cluster output (both corrections)
        ref_cf = ref_base + '_cl_cf.gro'
        scratch.append(ref_cf)
        ref_cf_ok = False
        if ref_pc_ok and ok_sf:
            fix_chain_images(ref_pc, workdir, ref_cf, None)
            ref_cf_ok = os.path.isfile(path(ref_cf))

        # ── query frame ───────────────────────────────────────────────────────
        qry_whole = qry_base + '_whole.gro'
        qry_prot = qry_base + '_prot.gro'
        scratch += [qry_whole, qry_prot]
        rc, err = gmx('echo "0" | gmx trjconv -f %s -s %s -o %s -pbc whole -quiet'
                      % (query_gro, tpr, qry_whole), workdir, gmxrc)
        if rc != 0 or not os.path.isfile(path(qry_whole)):
            return None, 'query pbc whole failed: ' + err[-200:], ''
        rc, err = gmx('echo "Protein" | gmx trjconv -f %s -s %s -o %s -n %s -quiet'
                      % (qry_whole, tpr, qry_prot, ndx), workdir, gmxrc)
        if rc != 0 or not os.path.isfile(path(qry_prot)):
            return None, 'query protein extract failed: ' + err[-200:], ''

        qry_sf = qry_base + '_cl_sf.gro'
        qry_cf = qry_base + '_cl_cf.gro'
        scratch += [qry_sf, qry_cf]
        qry_sf_ok = qry_cf_ok = False
        if ok_sf:
            qry_sf_ok = fix_chain_images(qry_prot, workdir, qry_sf, None)[0]
            # cross-frame: guided by the reference chain COMs
            qry_cf_ok = fix_chain_images(qry_prot, workdir, qry_cf, ref_coms_sf)[0]
        if not qry_sf_ok:
            qry_sf, qry_sf_ok = qry_prot, True

        qry_pc = qry_base + '_cl_pc.gro'
        scratch.append(qry_pc)
        rc, _ = gmx('printf "Protein\\nProtein\\n" | gmx trjconv -f %s -s %s -o %s '
                    '-pbc cluster -n %s -quiet' % (qry_whole, tpr, qry_pc, ndx),
                    workdir, gmxrc)
        qry_pc_ok = rc == 0 and os.path.isfile(path(qry_pc))

        # ── 3x3 grid, minimum wins ────────────────────────────────────────────
        ref_cands = [(ref_sf, 'sf', True),
                     (ref_pc, 'pc', ref_pc_ok),
                     (ref_cf, 'cf', ref_cf_ok)]
        qry_cands = [(qry_sf, 'sf', qry_sf_ok),
                     (qry_cf, 'cf', qry_cf_ok),
                     (qry_pc, 'pc', qry_pc_ok)]
        all_rmsds = []
        for rf, rl, rok in ref_cands:
            for qf, ql, qok in qry_cands:
                if not rok or not qok:
                    continue
                label = '%s_%s_%s' % (tag, rl, ql)
                scratch.append('rmsd_%s.xvg' % label)
                v, _ = compute_rmsd(rf, qf, workdir, label, gmxrc)
                if v is not None:
                    all_rmsds.append((v, rl, ql))

        if not all_rmsds:
            return None, 'no valid RMSD for any candidate pair', ''
        rmsd, rl, ql = min(all_rmsds, key=lambda x: x[0])
        return rmsd, '', 'ref=%s, query=%s' % (rl, ql)
    finally:
        if cleanup:
            for name in scratch:
                try:
                    os.remove(path(name))
                except OSError:
                    pass

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True, help="Bound reference frame (e.g. npt_c1.gro)")
    ap.add_argument('--query', required=True, help="Re-bound frame (e.g. bindrev_2.gro)")
    ap.add_argument('-s', '--tpr', default='emin_solv.tpr', help="Run input for trjconv (default: emin_solv.tpr)")
    ap.add_argument('-n', '--ndx', default='index.ndx', help="Index file (default: index.ndx)")
    ap.add_argument('-d', '--workdir', default='.', help="Directory holding the files (default: .)")
    ap.add_argument('--tag', default=None, help="Scratch-file tag (default: derived from --query)")
    ap.add_argument('--gmxrc', default=os.environ.get('GROSCORE_GMXRC'),
                    help="GMXRC to source before calling gmx (default: rely on PATH)")
    ap.add_argument('--keep', action='store_true', help="Keep intermediate files")
    ap.add_argument('-v', '--verbose', action='store_true', help="Report the winning image-correction pair")
    args = ap.parse_args()

    rmsd, err, detail = rebound_rmsd(args.workdir, args.ref, args.query,
                                     tpr=args.tpr, ndx=args.ndx, tag=args.tag,
                                     cleanup=not args.keep, gmxrc=args.gmxrc)
    if rmsd is None:
        # NaN on stdout keeps the caller's field count intact; the reason goes to stderr.
        print("NaN")
        print("rebound_rmsd: %s" % err, file=sys.stderr)
        return
    print("%.4f" % rmsd)
    if args.verbose:
        print("rebound_rmsd: [%s]" % detail, file=sys.stderr)


if __name__ == '__main__':
    main()
