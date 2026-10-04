#!/usr/bin/env python3
#
# assemble_frame.py - Put the solute of one frame back together in space.
#
# Both engines build their restraints from one equilibrated frame: job.run's
# make_disres_en.py reads npt_init_cluster.gro, job_fe.run's make_boresch.py reads
# npt_probe_cluster.gro. Both frames used to come from gmx trjconv -pbc cluster,
# which makes the cluster group whole by chaining each atom to the image nearest
# the atom before it (GROMACS 2026 calc_pbc_cluster, pbcmethods.cpp 153-158). A
# cutout partner is ONE moleculetype made of several capped, UNBONDED fragments,
# so wherever two sequence-consecutive fragments sit more than half a box apart
# the chain jumps a box vector, and every fragment after the jump goes with it.
# On real 1IJK setups this happened in both engines: the classic npt_init frame put
# chain C (residues 300-414) one lattice vector off and make_disres_en kept 407 of
# the 971 interface pairs, none of them on chain C; the FE probe frame put three
# ligand fragments (residues 300-306, 307-350, 351-416) a box vector away from where
# the whole structure has them, and 720 of 1108 contact pairs got no spring.
#
# This replaces -pbc cluster for that frame. The group (Protein_Struct by default)
# is assembled with pbc_tools.assemble, a minimum spanning tree over one anchor
# atom per residue in which every residue goes to the image nearest the residue it
# hangs from, so fragments are joined by where they are, not by their order in the
# file. Atoms are taken by POSITION, never by the .gro atom-number field. Every
# atom moves by a whole box vector or not at all, and a residue moves as one piece;
# run this after trjconv -pbc whole, which joins each residue by its bonds (a
# residue still split is made whole about its CA, else its first atom, and counted
# on stderr). Atoms outside the group, velocities, the title and the box line are
# written back untouched; of the placements the tree allows, which differ only by
# a box vector common to the whole group, the one that moves the fewest atoms is
# written, so a frame that needs no repair comes out byte for byte as it went in.
#
# Measured on real frames (v15_work/F/P). FE 1IJK, from npt_probe_whole.gro: 176
# residues moved, all eight backbone fragments at the offsets the whole editconf
# structure has (to 0.04 nm, the probe's own motion), and raw and minimum-image
# distances agree again: 1108 pairs within 0.6 nm, 144 and 576 of them on the two
# ligand fragments -pbc cluster had cut off. Classic 1IJK: make_disres_en gets 971
# interface and 243 network pairs (k 25.7), the counts of the correctly assembled
# frame, against 407 and 225 (k 61.4) from -pbc cluster. Where -pbc cluster
# happened to be right (FE 1AKJ, classic 2AJF, 1A2K, 7CEI) the restraint counts are
# identical and its frame comes back byte for byte. Tree links run 0.40-0.59 nm
# between residues, 2.2 nm to a structural chloride of 2AJF; no complex came
# closer than 2.35 nm to its own image (FE probe boxes; 3.45 nm in the classic
# boxes), far from the refusal below. 0.5 s on the 68,411-atom 1IJK probe frame.
#
# It stops (exit 1, a message on stderr, the output left alone) rather than write a
# frame it cannot vouch for:
#   * a frame it cannot read, a missing or empty group, a group reaching past the
#     frame or taking only part of a residue (only whole residues can be moved);
#   * a residue whose nearest neighbour is half the shortest lattice vector or more
#     away: its image is then not set by its neighbours;
#   * polymer heavy atoms within CONTACT_NM of each other only through a periodic
#     image: the complex touches its own copy, no placement of whole residues puts
#     every contact in one piece, and the restraint scripts would turn such a pair
#     into a spring that pulls the complex onto its own image.
#
# Usage:
#   python3 assemble_frame.py -f npt_probe_whole.gro -n index_probe.ndx \
#           -g Protein_Struct -o npt_probe_cluster.gro
#
# -o may name the input file. On success prints one line on stderr: residues moved
# and the longest link of the tree, the margin the result rests on.

import argparse, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_tools as pbc

# The heavy-atom distance at which make_disres_en.py and make_boresch.py take an
# interface spring (their interfacecutoff). A polymer pair this close only through
# a periodic image would become a spring between the complex and its own copy.
CONTACT_NM = 0.6
WATER = {"SOL", "HOH", "WAT"}


def fail(msg):
    sys.stderr.write("assemble_frame: ERROR - %s\n" % msg)
    sys.exit(1)


def read_frame(path):
    """(lines, n, xyz, box, width) of a single-frame .gro. lines are the file's
    lines verbatim, line endings included; atoms are read by position."""
    with open(path, newline="") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError("no frame")
    n = int(lines[1])
    if n < 1 or len(lines) < n + 3 or not lines[n + 2].split():
        raise ValueError("truncated: %d atoms announced, %d lines" % (n, len(lines)))
    if any(l.strip() for l in lines[n + 3:]):
        raise ValueError("more than one frame")
    atoms = lines[2:n + 2]
    w = pbc.field_width(atoms[0])
    xyz = pbc.parse_xyz(atoms, w)
    box = pbc.box_matrix(lines[n + 2].split())
    if not np.isfinite(xyz).all():
        raise ValueError("non-numeric coordinates")
    return lines, n, xyz, box, w


def image_contacts(P, box, r):
    """Pairs of points within r of each other only through a periodic image. No
    image matters beyond the group's own extent plus r, and within that range a
    pair cannot also be within r directly while the shortest lattice vector exceeds
    2r, so every such pair is counted once through t and once through -t."""
    rho = 2.0 * np.sqrt(((P - P.mean(0)) ** 2).sum(-1).max()) + r
    T = pbc.lattice_vectors(box, rho)
    if not len(T):
        return 0
    tree = cKDTree(P)
    return sum(int(tree.count_neighbors(cKDTree(P + t), r)) for t in T) // 2


def main():
    ap = argparse.ArgumentParser(
        description="Assemble one group of a .gro frame in space (residue spanning "
                    "tree); every other atom is written back unchanged.")
    ap.add_argument("-f", "--input", required=True, help="Frame to assemble (.gro).")
    ap.add_argument("-n", "--index", default="index.ndx", help="Index file (default: index.ndx).")
    ap.add_argument("-g", "--group", default="Protein_Struct",
                    help="Group to assemble, whole residues only (default: Protein_Struct).")
    ap.add_argument("-o", "--output", required=True, help="Output .gro; may be the input.")
    ap.error = lambda msg: fail(msg)               # exit 1 on every failure, usage too
    args = ap.parse_args()

    try:
        lines, n, xyz, box, w = read_frame(args.input)
    except (OSError, ValueError, IndexError) as e:
        fail("cannot read a frame from %s: %s" % (args.input, e))
    if not (np.diag(box) > 0).all() or box[0, 1] or box[0, 2] or box[1, 2]:
        fail("%s has no periodic GROMACS box (last line: %s)"
             % (args.input, lines[n + 2].strip()))
    try:
        idx = np.unique(np.asarray(pbc.ndx_group(args.index, args.group), dtype=int)) - 1
    except (OSError, ValueError) as e:
        fail(str(e))
    if not len(idx):
        fail("group %s in %s is empty" % (args.group, args.index))
    if idx[0] < 0 or idx[-1] >= n:
        fail("group %s reaches atom %d, but %s has %d atoms"
             % (args.group, (idx[-1] if idx[-1] >= n else idx[0]) + 1, args.input, n))

    head = [l[:20] for l in lines[2:n + 2]]
    ingroup = np.zeros(n, dtype=bool)
    ingroup[idx] = True
    blocks, alone = [], {}
    for s, e in pbc.residue_blocks(head):
        k = int(ingroup[s:e].sum())
        if k == e - s:
            blocks.append((s, e))
        elif k:
            fail("group %s holds %d of the %d atoms of residue %s (atoms %d-%d); only "
                 "whole residues can be moved" % (args.group, k, e - s,
                                                  head[s][:10].strip(), s + 1, e))
        elif s < idx[-1] and head[s][5:10].strip() not in WATER:
            # Not moved, but the restraint scripts read every non-water residue up
            # to the last structural one: name it, as -pbc cluster never did.
            alone[head[s][5:10].strip()] = alone.get(head[s][5:10].strip(), 0) + 1
    sel = np.concatenate([np.arange(s, e) for s, e in blocks])
    sub, at = [], 0                              # the same residues, within sel
    for s, e in blocks:
        sub.append((at, at + e - s))
        at += e - s
    hsel = [head[i] for i in sel]
    new, longest = pbc.assemble(xyz[sel], box, sub, pbc.anchor_atoms(hsel, sub))

    shortest = pbc.shortest_vector(box)
    if longest >= 0.5 * shortest:
        fail("a residue of group %s is %.3f nm from its nearest neighbour, not less than "
             "half the shortest lattice vector (%.3f nm), so its image is not determined"
             % (args.group, longest, 0.5 * shortest))

    # Every atom moved by a whole lattice vector; keep the common one that moves
    # the fewest atoms, so atoms already in place stay byte for byte.
    frac = (new - xyz[sel]) @ np.linalg.inv(box)
    shift = np.rint(frac)
    if np.abs(frac - shift).max() > 1e-6:
        fail("internal: an atom moved by %.3g of a box vector" % np.abs(frac - shift).max())
    vals, counts = np.unique(shift, axis=0, return_counts=True)
    shift -= vals[np.argmax(counts)]
    out_xyz = xyz[sel] + shift @ box

    poly = pbc.polymer_atoms(hsel, sub)
    if not len(poly):
        poly = np.arange(len(sel))
    heavy = [i for i in poly if not hsel[i][10:15].strip().lstrip("0123456789").startswith("H")]
    if shortest <= 2.0 * CONTACT_NM:
        fail("box too small: shortest lattice vector %.2f nm" % shortest)
    n_img = image_contacts(out_xyz[heavy], box, CONTACT_NM) if heavy else 0
    if n_img:
        fail("%d polymer heavy-atom pairs of group %s are within %.1f nm of each other only "
             "through a periodic image: the complex touches its own copy in this %.2f nm box"
             % (n_img, args.group, CONTACT_NM, shortest))

    moved = np.flatnonzero(np.any(shift != 0, axis=1))
    member = np.repeat(np.arange(len(sub)), [e - s for s, e in sub])
    whole = sum(1 for s, e in sub if len({tuple(v) for v in shift[s:e]}) > 1)
    out = list(lines)
    fmt = ("%%%d.%df" % (w, w - 5)) * 3
    for i in moved:
        line = lines[2 + sel[i]]
        field = fmt % tuple(out_xyz[i])
        if len(field) != 3 * w:
            fail("atom %d moves to %s, which does not fit %d-column fields"
                 % (sel[i] + 1, field.split(), w))
        out[2 + sel[i]] = line[:20] + field + line[20 + 3 * w:]
    tmp = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                       ".%s.%d.tmp" % (os.path.basename(args.output), os.getpid()))
    try:
        with open(tmp, "w", newline="") as f:
            f.writelines(out)
        os.replace(tmp, args.output)
    except OSError as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        fail("cannot write %s: %s" % (args.output, e))
    sys.stderr.write("assemble_frame: %s group %s (%d atoms, %d residues): %d residues moved "
                     "by a box vector%s; longest link %.2f nm, half the shortest lattice "
                     "vector %.2f nm%s -> %s\n"
                     % (args.input, args.group, len(sel), len(sub), len(set(member[moved])),
                        (" (%d of them also made whole)" % whole) if whole else "",
                        longest, 0.5 * shortest,
                        ("; not in the group, left in place: %s" % ", ".join(
                            "%d x %s" % (c, r) for r, c in sorted(alone.items())))
                        if alone else "", args.output))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:                       # any other failure is still exit 1
        fail("%s: %s" % (type(e).__name__, e))
