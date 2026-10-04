#!/usr/bin/env python3
"""
Backbone RMSD between the bound (NPT-equilibrated) and the re-bound structure of
one cycle: the rebinding sanity check.

A cycle pulls the two binding partners apart and pushes them back together. If
the push really re-formed the original complex, the backbone of the re-bound
structure must superimpose on the backbone of the bound reference; a large RMSD
means the partners came back in a different pose (or not at all), so the work
values of that cycle describe something other than the intended binding event.

The comparison is on the Protein group (index.ndx), and both frames are first
REASSEMBLED in space with utils/pbc_tools.assemble: a minimum spanning tree over
one anchor atom per residue, every residue placed at the periodic image nearest
the residue it hangs from. Assembled frames go to gmx rms, which fits and measures
the Backbone exactly as before, so a frame that never needed repair gives the same
number it always did: on ten 2KTF cycles of test6 old and new agree to 0.0001 A,
with the longest assembly link at 0.40-0.51 nm.

Why not by bonds or by sequence any more. A cutout partner is ONE moleculetype
made of several capped, unbonded fragments. gmx trjconv -pbc whole follows bonds
and leaves the fragments wherever mdrun wrapped them; -pbc cluster moves the
moleculetype as one unit; and the old self-fix propagated residue by residue along
the sequence to the nearest image of the previous residue, which puts a fragment
one box vector off whenever two sequence-consecutive fragments are more than half
a box apart. That is where the isolated 31-41 A values of pass 2 came from (1HE8
c10 41.2, 1I4D c3-c5 31-36, 1IJK 38-40, against continua of 5-18 A and interface
contact recovery of 0.93-0.97 in the same cycles). The minimum over the 3x3 grid
of corrections the old version took could not rescue them: two of its three
candidates per frame ran the same sequential step, and the third, -pbc cluster,
cannot split a moleculetype. A spatial tree needs no sequence and no topology, and
since it uses distances only, a drift or rotation of the whole complex between the
two frames, of any size, does not enter.

Used routinely by job.run / job_fe.run (one value per cycle, written into the
results files) and by benchmark/haddock_benchmark/compute_rebound_rmsd.py.

Usage (prints the RMSD in Angstrom, or NaN if it could not be computed):
  python3 rebound_rmsd.py --ref npt_c1.gro --query bindrev_2.gro
"""

import os
import sys
import argparse
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_tools as pbc

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


def compute_rmsd(ref_gro, query_gro, workdir, xvg, gmxrc=None):
    """Backbone RMSD (Å) of query_gro vs ref_gro, fitted by gmx rms.

    Both files are protein-only .gro files, so gmx rms builds its default Backbone
    group from residue and atom names and guesses masses from them, exactly as it
    did on the image-corrected files this script wrote before."""
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

# ── assembly ──────────────────────────────────────────────────────────────────

def assembled_protein(gro_path, prot_idx):
    """(head, assembled xyz, box, longest tree edge) of the Protein atoms of one
    frame. prot_idx is 0-based, by position in the file."""
    fr = pbc.read_gro(gro_path)
    if len(prot_idx) == 0:
        raise ValueError('the Protein group is empty')
    if prot_idx.max() >= len(fr):
        raise ValueError('%s has %d atoms, the Protein group reaches atom %d'
                         % (os.path.basename(gro_path), len(fr), prot_idx.max() + 1))
    head = [fr.head[i] for i in prot_idx]
    blocks = pbc.residue_blocks(head)
    xyz, longest = pbc.assemble(fr.xyz[prot_idx], fr.box, blocks,
                                pbc.anchor_atoms(head, blocks))
    return head, xyz, fr.box, longest

# ── main entry point ──────────────────────────────────────────────────────────

def rebound_rmsd(workdir, ref_gro, query_gro, tpr='emin_solv.tpr',
                 ndx='index.ndx', tag=None, cleanup=True, gmxrc=None):
    """Backbone RMSD (Å) between the bound reference and the re-bound query frame.

    Both frames are reduced to the Protein group of `ndx`, reassembled in space
    (pbc_tools.assemble) and handed to gmx rms. `tpr` is no longer read, since the
    assembly needs no topology; it stays in the signature so callers need not
    change.

    Returns (rmsd_angstrom_or_None, error_message, detail) where detail names the
    longest link the assembly of each frame rests on, in nm. A link far below half
    the box is the evidence that no residue was placed at a wrong image.
    """
    if tag is None:
        tag = os.path.splitext(os.path.basename(query_gro))[0]

    def path(name):
        return os.path.join(workdir, name)

    for f in (ref_gro, query_gro, ndx):
        if not os.path.isfile(path(f)):
            return None, 'missing %s' % f, ''

    ref_out = 'rbref_%s.gro' % tag
    qry_out = 'rbqry_%s.gro' % tag
    xvg = 'rmsd_%s.xvg' % tag
    scratch = [ref_out, qry_out, xvg]
    try:
        try:
            prot = np.asarray(pbc.ndx_group(path(ndx), 'Protein'), dtype=int) - 1
        except (OSError, ValueError) as e:
            return None, 'no Protein group: %s' % e, ''
        links = []
        for src, dst in ((ref_gro, ref_out), (query_gro, qry_out)):
            try:
                head, xyz, box, longest = assembled_protein(path(src), prot)
            except (OSError, ValueError, IndexError) as e:
                return None, 'could not assemble %s: %s' % (src, e), ''
            pbc.write_gro(path(dst), 'Protein, reassembled', head, xyz, box)
            links.append(longest)
        rmsd, err = compute_rmsd(ref_out, qry_out, workdir, xvg, gmxrc)
        if rmsd is None:
            return None, err, ''
        return rmsd, '', ('assembled; longest link ref %.2f nm, query %.2f nm'
                          % tuple(links))
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
    ap.add_argument('-s', '--tpr', default='emin_solv.tpr',
                    help="Accepted for compatibility; no longer read (default: emin_solv.tpr)")
    ap.add_argument('-n', '--ndx', default='index.ndx', help="Index file holding the Protein group (default: index.ndx)")
    ap.add_argument('-d', '--workdir', default='.', help="Directory holding the files (default: .)")
    ap.add_argument('--tag', default=None, help="Scratch-file tag (default: derived from --query)")
    ap.add_argument('--gmxrc', default=os.environ.get('GROSCORE_GMXRC'),
                    help="GMXRC to source before calling gmx (default: rely on PATH)")
    ap.add_argument('--keep', action='store_true', help="Keep intermediate files")
    ap.add_argument('-v', '--verbose', action='store_true', help="Report how the frames were assembled")
    args = ap.parse_args()

    try:
        rmsd, err, detail = rebound_rmsd(args.workdir, args.ref, args.query,
                                         tpr=args.tpr, ndx=args.ndx, tag=args.tag,
                                         cleanup=not args.keep, gmxrc=args.gmxrc)
    except Exception as e:                      # never abort a cycle over a diagnostic
        rmsd, err, detail = None, '%s: %s' % (type(e).__name__, e), ''
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
