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
# Atoms are read by POSITION in the file, never by the .gro atom-number field.
# GROMACS writes that field modulo 100000, and keying atoms by it let solvent atom
# 100000+k overwrite solute atom k: in pass 2 (bm_fe_amber_2) the 19 production
# systems above 99,999 atoms whose wrap reached the interface reported recovered
# 0.001 and rms_dev 4.93 nm in every cycle, against 0.91 and 0.084 nm from the
# PBC-aware distances grompp lists for the same frames. The interface residues are
# likewise the residues that CONTAIN a restrained atom, found by position, not by a
# residue number that wraps the same way.
#
# Distances are true minimum images (utils/pbc_tools.py): an unclustered mdrun .gro
# can put the partners, and a cutout's unbonded fragments, in different periodic
# images, and a 27-neighbour search on the raw difference is only a minimum image
# when that difference is already within one box vector of it. 1E6J cycle 3 read
# rms_dev 4.581 nm that way, against 0.059 nm from grompp. The side-chain fit gets
# the same treatment, through pbc_tools.assemble.
#
# Usage:
#   python3 interface_qc.py -c interface_contacts.gs --ref npt_c3.gro \
#                           --query bindrevA_3.gro [--tol 0.15]
#
# Prints one line of "key value" pairs. Exits 0 even when the interface is wrecked:
# this is diagnostic, like the RMSD check, and must never fail a cycle.

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_tools as pbc

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


def kabsch_rmsd(P, Q):
    """Optimal rigid superposition of P onto Q: (R, Pc, Qc) with (P - Pc) @ R + Qc ~ Q."""
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - Pc, Q - Qc
    V, S, Wt = np.linalg.svd(P0.T @ Q0)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return R, Pc, Qc


def out(**kw):
    print(" ".join("%s %s" % (k, v) for k, v in kw.items()))


def nothing():
    out(recovered="nan", formed="nan", rms_dev="nan", sc_rmsd="nan", npairs=0)
    sys.exit(0)


if not os.path.isfile(args.contacts):
    sys.stderr.write("interface_qc: no %s; nothing to check\n" % args.contacts)
    nothing()

# Atom numbers in the contact list are make_boresch.py's, read off the probe system,
# whose solute comes first and sits far below 100000 atoms, so they ARE positions
# (1-based) in every frame of the same system.
pairs = []
for line in open(args.contacts):
    if line.startswith("#") or not line.strip():
        continue
    f = line.split()
    if len(f) < 9:
        continue
    try:
        pairs.append((int(f[0]), int(f[1]), float(f[2])))
    except ValueError:
        continue

try:
    q = pbc.read_gro(args.query)
    r = pbc.read_gro(args.ref)
except (OSError, ValueError, IndexError) as e:
    sys.stderr.write("interface_qc: could not read a structure (%s)\n" % e)
    nothing()


def measure(q, r, pairs):
    """The four numbers, or None when no restrained pair lies inside the frame."""
    # --- contact recovery, superposition-free -----------------------------
    nq = len(q)
    use = [(a - 1, b - 1, d) for a, b, d in pairs if 1 <= a <= nq and 1 <= b <= nq]
    if not use:
        return None
    ia = np.array([u[0] for u in use])
    ib = np.array([u[1] for u in use])
    dref = np.array([u[2] for u in use])
    dq = np.sqrt((pbc.min_image(q.xyz[ia] - q.xyz[ib], q.box) ** 2).sum(-1))
    dev = dq - dref
    res = dict(n=len(dev), recovered=float(np.mean(np.abs(dev) <= args.tol)),
               formed=float(np.mean(dq <= args.cutoff)),
               rms_dev=float(np.sqrt(np.mean(dev ** 2))), sc_rmsd=float("nan"))

    # --- side-chain repacking, after fitting the interface backbone -------
    # Every residue holding a restrained atom, taken whole from the query file.
    # The reference is the same system, so the same positions are the same atoms.
    blocks = pbc.residue_blocks(q.head)
    member = np.empty(nq, dtype=int)
    for k, (s, e) in enumerate(blocks):
        member[s:e] = k
    iface = sorted(set(member[ia]) | set(member[ib]))
    sel = np.concatenate([np.arange(*blocks[k]) for k in iface])
    if sel.max() >= len(r):
        return res
    # Both frames are assembled the same way before the fit. Gathering about one
    # atom, as this used to, is only safe while the whole interface lies within
    # half a box of that atom; the spanning tree has no such limit.
    sub, start = [], 0
    for k in iface:
        s, e = blocks[k]
        sub.append((start, start + e - s))
        start += e - s
    head = [q.head[i] for i in sel]
    anchors = pbc.anchor_atoms(head, sub)
    P, _ = pbc.assemble(q.xyz[sel], q.box, sub, anchors)
    Q, _ = pbc.assemble(r.xyz[sel], r.box, sub, anchors)
    isbb = np.array([h[10:15].strip() in BB for h in head])
    if isbb.sum() >= 3 and (~isbb).any():
        R, Pc, Qc = kabsch_rmsd(P[isbb], Q[isbb])
        Ps = (P[~isbb] - Pc) @ R + Qc
        res["sc_rmsd"] = float(np.sqrt(np.mean(np.sum((Ps - Q[~isbb]) ** 2, axis=1)))) * 10.0  # A
    return res


try:
    res = measure(q, r, pairs)
except Exception as e:                    # diagnostic: never fail a cycle over it
    sys.stderr.write("interface_qc: %s: %s\n" % (type(e).__name__, e))
    res = None
if res is None:
    nothing()
out(recovered="%.4f" % res["recovered"],
    formed="%.4f" % res["formed"],
    rms_dev="%.4f" % res["rms_dev"],
    sc_rmsd=("%.3f" % res["sc_rmsd"]) if np.isfinite(res["sc_rmsd"]) else "nan",
    npairs=res["n"])
