#!/usr/bin/env python3
#
# interface_qc.py - did the rebinding leg put the INTERFACE back, not just the fold?
#
# utils/rebound_rmsd.py measures backbone RMSD, which says the two partners are in
# roughly the right place relative to each other. It says nothing about whether the
# contacts re-formed or whether the side chains re-packed, and those are what the
# interface restraints exist to hold. Capping the restraint list per residue-residue
# contact is only safe if the thing being protected is measured, so this measures it.
#
# Three numbers, all against the reference structure the restraints were built on:
#
#   recovered   fraction of restrained pairs back within --tol of their reference
#               distance. The direct question: did this contact re-form?
#   formed      fraction still inside the 0.6 nm contact cutoff at all. Looser, and
#               it separates "re-formed slightly differently" from "did not re-form".
#   rms_dev     RMS of (d_query - d_ref) over the restrained pairs. This is
#               sqrt(S/N) with S the strain sum that IS the restraint work, so it is
#               the same quantity dG_intro integrates, in nm per pair.
#
# All three are INTERNAL distances, so they need no superposition and cannot be
# spoiled by a bad fit. The fourth number does need one:
#
#   sc_rmsd     RMSD of the interface side-chain atoms after least-squares fitting
#               the interface BACKBONE. Rigid-body placement is removed by the fit,
#               so what is left is repacking.
#
# Minimum-image distances are used throughout: an unclustered mdrun .gro can place
# the partners in different periodic images, which would otherwise report every
# contact as broken.
#
# Usage:
#   python3 interface_qc.py -c interface_contacts.gs --ref npt_c3.gro \
#                           --query bindrevA_3.gro [--tol 0.15]
#
# Prints one line of "key value" pairs. Exits 0 even when the interface is wrecked:
# this is diagnostic, like the RMSD check, and must never fail a cycle.

import argparse, math, os, sys
import numpy as np

ap = argparse.ArgumentParser(description="Interface contact recovery and side-chain repacking.")
ap.add_argument("-c", "--contacts", default="interface_contacts.gs",
                help="Restrained pair list from make_boresch.py.")
ap.add_argument("--ref", required=True, help="Bound reference .gro (npt_c<n>.gro).")
ap.add_argument("--query", required=True, help="Re-bound .gro (end of the last rebinding stage).")
ap.add_argument("--tol", type=float, default=0.15,
                help="A contact counts as recovered within this many nm of its "
                     "reference distance (default 0.15).")
ap.add_argument("--cutoff", type=float, default=0.6,
                help="Contact cutoff used when the restraints were chosen (default 0.6 nm).")
args = ap.parse_args()

BB = {"N", "CA", "C", "O", "OT", "OT1", "OT2", "OXT"}


def read_gro(path):
    """-> (coords by atom number, atom name by number, box matrix).

    FIXED-WIDTH columns. A split on whitespace merges resnum and resname for
    SOL entries and silently shifts everything after it."""
    xyz, name = {}, {}
    with open(path) as f:
        f.readline()
        n = int(f.readline())
        for _ in range(n):
            l = f.readline()
            a = int(l[15:20])
            name[a] = l[10:15].strip()
            xyz[a] = (float(l[20:28]), float(l[28:36]), float(l[36:44]))
        box = [float(v) for v in f.readline().split()]
    box += [0.0] * (9 - len(box))
    #      v1                v2                v3
    M = np.array([[box[0], box[3], box[4]],
                  [box[5], box[1], box[6]],
                  [box[7], box[8], box[2]]], float)
    return {k: np.array(v) for k, v in xyz.items()}, name, M


def min_image(dv, M):
    """Shortest vector equivalent to dv under the box M (rows are box vectors).

    Triclinic-safe by brute force over the 27 neighbouring images, which is
    exact and costs nothing at these sizes."""
    best, bn = dv, float(np.dot(dv, dv))
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == j == k == 0:
                    continue
                t = dv + i * M[0] + j * M[1] + k * M[2]
                n = float(np.dot(t, t))
                if n < bn:
                    best, bn = t, n
    return best


def dist(xyz, a, b, M):
    return float(np.linalg.norm(min_image(xyz[a] - xyz[b], M)))


def kabsch_rmsd(P, Q):
    """RMSD of P onto Q after optimal rigid superposition, plus the transform."""
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - Pc, Q - Qc
    V, S, Wt = np.linalg.svd(P0.T @ Q0)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return R, Pc, Qc


def out(**kw):
    print(" ".join("%s %s" % (k, v) for k, v in kw.items()))


if not os.path.isfile(args.contacts):
    sys.stderr.write("interface_qc: no %s; nothing to check\n" % args.contacts)
    out(recovered="nan", formed="nan", rms_dev="nan", sc_rmsd="nan", npairs=0)
    sys.exit(0)

pairs = []
for line in open(args.contacts):
    if line.startswith("#") or not line.strip():
        continue
    f = line.split()
    if len(f) < 9:
        continue
    pairs.append((int(f[0]), int(f[1]), float(f[2]),
                  int(f[4]), f[5], int(f[7]), f[8]))

try:
    qx, qname, qM = read_gro(args.query)
    rx, rname, rM = read_gro(args.ref)
except (OSError, ValueError) as e:
    sys.stderr.write("interface_qc: could not read a structure (%s)\n" % e)
    out(recovered="nan", formed="nan", rms_dev="nan", sc_rmsd="nan", npairs=0)
    sys.exit(0)

# --- contact recovery, superposition-free ---------------------------------
dev, formed = [], 0
for a, b, dref, _ra, _na, _rb, _nb in pairs:
    if a not in qx or b not in qx:
        continue
    dq = dist(qx, a, b, qM)
    dev.append(dq - dref)
    if dq <= args.cutoff:
        formed += 1
dev = np.array(dev, float)
n = len(dev)
if n == 0:
    out(recovered="nan", formed="nan", rms_dev="nan", sc_rmsd="nan", npairs=0)
    sys.exit(0)

recovered = float(np.mean(np.abs(dev) <= args.tol))
rms_dev = float(np.sqrt(np.mean(dev ** 2)))

# --- side-chain repacking, after fitting the interface backbone -----------
# The atoms are the interface residues' own atoms, taken from the same file, so
# the numbering matches by construction.
iface_res = set()
for _a, _b, _d, ra, _na, rb, _nb in pairs:
    iface_res.add(ra); iface_res.add(rb)

bb_a, sc_a = [], []
with open(args.query) as f:
    f.readline(); nat = int(f.readline())
    for _ in range(nat):
        l = f.readline()
        rnum = int(l[0:5]); an = l[10:15].strip(); a = int(l[15:20])
        if rnum in iface_res and a in rx:
            (bb_a if an in BB else sc_a).append(a)


def unwrap(xyz, atoms, M):
    """Atoms gathered into ONE periodic image, around the first of them.

    Without this the fit is meaningless: an mdrun .gro wraps by molecule, so an
    interface spanning a boundary comes back with some residues a box vector
    away and the RMSD reads tens of Angstrom. The contact distances above are
    already minimum-imaged; this is the same correction for the coordinates the
    superposition needs. Safe because an interface is 1-2 nm across against an
    8 nm box, so gathering about one atom cannot fold in a genuine separation."""
    origin = xyz[atoms[0]]
    return np.array([origin + min_image(xyz[a] - origin, M) for a in atoms])


sc_rmsd = float("nan")
if len(bb_a) >= 3 and sc_a:
    order = bb_a + sc_a
    P = unwrap(qx, order, qM)
    Q = unwrap(rx, order, rM)
    nbb = len(bb_a)
    R, Pc, Qc = kabsch_rmsd(P[:nbb], Q[:nbb])
    Ps = (P[nbb:] - Pc) @ R + Qc
    sc_rmsd = float(np.sqrt(np.mean(np.sum((Ps - Q[nbb:]) ** 2, axis=1)))) * 10.0  # Angstrom

out(recovered="%.4f" % recovered,
    formed="%.4f" % (formed / float(n)),
    rms_dev="%.4f" % rms_dev,
    sc_rmsd=("%.3f" % sc_rmsd) if np.isfinite(sc_rmsd) else "nan",
    npairs=n)
