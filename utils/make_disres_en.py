#!/usr/bin/env python3
#
# make_disres_en.py - interface and elastic-network pull restraints for the
# classic engine (job.run stage 0), from the equilibrated frame
# npt_init_cluster.gro. Prints the number of interface restraints; appends the
# a_<atom> groups to index.ndx and the pull blocks to bind.mdp, nptrev.mdp and
# bindrev.mdp.
#
# Exit status 1, with nothing written, when
#   - the frame cannot be read or a side has no atoms,
#   - the interface set is empty: the pull block would then carry only the
#     network, integrate.py -nr 0 prints -0.0 per cycle and the structure scores
#     0 kJ/mol with no flag (audit R3),
#   - the frame is split across the periodic boundary, i.e. a pair counted by
#     the minimum image is not counted by the plain distance (or the complex
#     touches its own periodic image, equally fatal for the pull). trjconv -pbc
#     cluster chains a moleculetype atom by atom and left botrocetin chain C of
#     1IJK one box vector off; the plain distances kept 378-477 of the 1020-1213
#     interface restraints, with rc 0 (audit R1). Every distance is now measured
#     both ways, so such a frame stops as DISRES_FAIL instead of scoring a
#     different restraint set: on the audit's three 1IJK cluster frames it reports
#     390 by plain distance, 1030 by minimum image, the whole frame's 1030.
#     job.run assembles the frame first (utils/assemble_frame.py), so the check
#     only fires when that failed.
#
# The written references are the plain distances: a pair the check accepted is
# within 0.9 nm, far inside half the shortest box vector, so its plain distance
# IS its minimum-image distance. On an intact frame the output is byte-identical
# to v1.4 except where the selection fixes below apply (R4 ions and GROMOS cap
# carbons out of the network, R5 Mn2+ back in the interface, R6 CHARMM36
# C-terminal anchors).

import os, sys, argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import pbc_tools as pbc


#------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate distance restraints for pulling and elastic network.")
parser.add_argument('-f','--input', type=str, default="npt_cluster.gro", help="Input coordinate file (default: npt_cluster.gro)")
parser.add_argument('-m','--chainmap', type=str, required=True, help="Chain map file containing residue numbers for protein B.")
args=parser.parse_args()

def fail(msg):
  sys.stderr.write("make_disres_en: ERROR - %s\n" % msg)
  sys.exit(1)

#------------------------------------------------------

def read_chain_map(filepath):
  """Read chain_map.gs file and return set of residue numbers belonging to protein B."""
  residues_b = set()
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          try:
            residues_b.add(int(line.strip()))
          except (ValueError, IndexError):
            pass
  return residues_b

residues_b = read_chain_map(args.chainmap)

#------------------------------------------------------

# cutoff and elastic network parameters
interfacecutoff = 0.6
en_min = 0.4
en_max = 0.9
enk = 250

# Same list as renumber_pdb.py / cap_termini.py: single-atom ion residues.
ION_RESIDUES = {"BA", "CA", "CD", "CL", "CO", "CS", "CU", "CU1", "FE", "FE2", "HG", "K",
                "LI", "MG", "MN", "NA", "NI", "PB", "SD", "SR", "ZN"}

# Water residue name in every GroScore frame (pdb2gmx renames HOH, solvate and
# merge_crystal_waters.py write SOL).
WATER = "SOL"

# Atom names that mark a fragment end without a cap next to it: the C-terminal
# carboxyl of GROMOS (OT), CHARMM36 (OT1/OT2, its COOH patch, which job.run
# selects) and AMBER (OXT), and the N-terminal H2 of GROMOS/AMBER. With only
# 'OT' and 'H2', no CHARMM36 C-terminal cut end was ever an anchor: 1A2K got 165
# network springs and 3 of 10 C-terminal ends covered (by proximity only),
# against 335 and 9 of 10 under AMBER (audit R6). With OT1/OT2 the CHARMM36 1A2K
# build has 346 springs where it had 158, its AMBER build 336.
TERMINAL_ATOMS = {"OT", "OT1", "OT2", "OXT", "H2"}

# Fail loudly on a missing input. Falling through left every selection below
# empty and still printed a restraint count, so the caller recorded a successful
# setup with no restraints in it.
if not os.path.isfile(args.input):
  fail("coordinate file %s does not exist" % args.input)
try:
  frame = pbc.read_gro(args.input)
except (OSError, ValueError) as e:
  fail("cannot read %s: %s" % (args.input, e))

# Fixed .gro columns, atoms by POSITION (pbc_tools). The split-based parse this
# replaces merged the residue number into a digit-leading residue name: residue
# 368 '1PE' read as 3681 and the ligand fell out as 'solvent' (audit R8).
head, xyz, box = frame.head, frame.xyz, frame.box

# Structural atoms = the solute block, everything before the first water. The
# old rule, every residue number up to max(chain_map, ion and ligand residues),
# dropped a receptor chain that a --no-cutout input lists after protein B: 1IQD
# with chains in the order A, C, B kept 520-545 of 1402-1420 interface
# restraints, rc 0 (audit R7). Counterions come after the water (genion), and
# merge_crystal_waters.py appends crystal waters after the protein, ions and
# ligands, so the solute block is exactly protein + structural ions + ligands.
# Solvated and ionised (216-217k atoms, order protein, SOL, NA, CL), the two
# no-cutout 1IQD builds give 1466 restraints in the original chain order (v1.4
# the same) and 1455 in the order A, C, B (v1.4: 562).
n_solute = len(head)
for i, h in enumerate(head):
  if h[5:10].strip() == WATER:
    n_solute = i
    break

try:
  blocks = pbc.residue_blocks(head[:n_solute])
  res_num = [int(head[s][0:5]) for s, _ in blocks]
except ValueError as e:
  fail("unreadable residue number in %s: %s" % (args.input, e))
res_name = [head[s][5:10].strip() for s, _ in blocks]
atom_name = [h[10:15].strip() for h in head[:n_solute]]
res_atoms = [atom_name[s:e] for s, e in blocks]

# Protein residues carry N, CA and C. That excludes the ions and the caps, which
# also have atoms named CA: a Ca2+ ion (residue CA, atom CA) and, under GROMOS,
# the ACE methyl that cap_termini.py --rename-ace-carbon names CA. Both used to
# become network nodes: 1H1V tied Ca2+ 600 to 11-12 backbone CAs at k 250, and
# 1A2K under GROMOS had 8 ACE carbons in its network (audit R4; 372 springs, 331
# without them).
is_protein = [{"N", "CA", "C"} <= set(names) for names in res_atoms]
is_ion = [rn in ION_RESIDUES and len(names) == 1 for rn, names in zip(res_name, res_atoms)]

# Per side: residue (block) indices and atom indices, in file order.
side_res = {1: [], 2: []}
for k, rn in enumerate(res_num):
  side_res[2 if rn in residues_b else 1].append(k)

def side_atoms(ks):
  return [i for k in ks for i in range(*blocks[k])]

atoms1 = side_atoms(side_res[1])
atoms2 = side_atoms(side_res[2])
if not atoms1 or not atoms2:
  fail("%s gave %d receptor and %d ligand atoms (check chain_map.gs)"
       % (args.input, len(atoms1), len(atoms2)))

block_of = np.empty(n_solute, dtype=int)
for k, (s, e) in enumerate(blocks):
  block_of[s:e] = k

def interface_atom(i):
  """Heavy atoms; not the MN* dummy masses of virtual sites. An ion's lone atom is
  always kept: the prefix test removed the Mn2+ of 1T6B, the CMG2 MIDAS metal
  coordinated by anthrax PA Asp176 OD2 at 0.19 nm, so the cross-interface metal
  bridge got none of its 5 springs (audit R5), and the H test would do the same to
  an Hg2+ (atom HG)."""
  if is_ion[block_of[i]]:
    return True
  n = atom_name[i]
  return n[0] != "H" and n[:2] != "MN"

valid1 = np.array([i for i in atoms1 if interface_atom(i)], dtype=int)
valid2 = np.array([i for i in atoms2 if interface_atom(i)], dtype=int)

#------------------------------------------------------
# Minimum-image bookkeeping

# Lattice coefficients searched after both points were reduced into the brick
# about the origin. Two points of that brick differ by at most one brick edge per
# coordinate; with GROMACS's box rules (|b_x|, |c_x| <= a_x/2, |c_y| <= b_y/2) a
# lattice vector that brings them within 0.9 nm therefore has |i| <= 2 and |j|,
# |k| <= 1. All of -2..2 is searched anyway.
_SHIFTS = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3)
                    for k in range(-2, 3)], dtype=float)
SLACK = 1e-3   # nm; candidates only, the decision is exact (below)

def periodic(box):
  return bool(np.any(np.diag(box) > 0))

def mi_matrix(P, Q):
  """Minimum-image distances between P[i] and Q[j], and whether that image is
  NOT the raw difference (a lattice shift was needed). Rows in chunks: only used
  on the network's CA sets, a few hundred atoms per side."""
  d = np.empty((len(P), len(Q)))
  shifted = np.zeros((len(P), len(Q)), dtype=bool)
  if not periodic(box):
    return cdist(P, Q), shifted
  for r in range(0, len(P), 64):
    raw = Q[None, :, :] - P[r:r + 64, None, :]
    v = pbc.min_image(raw, box)
    d[r:r + 64] = np.sqrt((v * v).sum(-1))
    shifted[r:r + 64] = np.any(v != raw, axis=-1)
  return d, shifted

def shifted_pairs(P, Q, rmax):
  """Pairs (i, j) within rmax only by a lattice shift: the minimum image of
  Q[j] - P[i] is not the raw difference and is within rmax. These are the pairs a
  split frame hides from plain distances. An image within rmax is unique (two
  images of one atom are a box vector apart, far more than 2*rmax), and a pair
  within rmax by its raw difference is never shifted, so on an intact frame the
  result is empty whatever the rounding of distances right at rmax."""
  if not periodic(box) or len(P) == 0 or len(Q) == 0:
    return []
  Pr = pbc.reduce_vectors(P, box)
  Qr = pbc.reduce_vectors(Q, box)
  tree = cKDTree(Pr)
  cand = set()
  for n in _SHIFTS:
    hits = cKDTree(Qr + n @ box).query_ball_tree(tree, rmax + SLACK)
    for j, ii in enumerate(hits):
      for i in ii:
        cand.add((i, j))
  if not cand:
    return []
  I, J = np.array(sorted(cand)).T
  raw = Q[J] - P[I]
  v = pbc.min_image(raw, box)
  d = np.sqrt((v * v).sum(-1))
  keep = np.any(v != raw, axis=-1) & (d <= rmax)
  return list(zip(I[keep], J[keep]))

#------------------------------------------------------
# Interface: cross-side heavy-atom pairs within 0.6 nm

coords1 = xyz[valid1]
coords2 = xyz[valid2]
all_distances = cdist(coords1, coords2)
ii, jj = np.nonzero(all_distances <= interfacecutoff)   # row-major: i, then j
interdis = [(valid1[a], valid2[b], all_distances[a, b]) for a, b in zip(ii, jj)]
numinterdis = len(interdis)
split_inter = shifted_pairs(coords1, coords2, interfacecutoff)

#------------------------------------------------------
# Elastic network

def build_elastic_network(ks):
  """Elastic network of one side (residue blocks ks): CA pairs at 0.4-0.9 nm among
  the CAs within 0.9 nm of a fragment-end anchor.

  Returns (pairs, keep, mismatch): pairs as (i, j, distance) with i, j indices
  into keep, the kept CA atom indices (file order), and the minimum-image counts
  (kept CAs, pairs) when they differ from the plain ones, else None.

  Anchors are the CA of the residue next to a cap (ACE before it, NME after it)
  and of a residue carrying a terminal atom (TERMINAL_ATOMS, for C-terminal cut
  ends under --ace-only caps). The first and the last anchor of the side are
  skipped (protanchor[1:lena-1] of the original), so a single-fragment side has no
  network."""
  num_to_k = {res_num[k]: k for k in ks}
  anchor_res = set()
  for k in ks:
    if TERMINAL_ATOMS & set(res_atoms[k]):
      anchor_res.add(k)
    if res_name[k] == "ACE":
      nb = num_to_k.get(res_num[k] + 1)
    elif res_name[k] == "NME":
      nb = num_to_k.get(res_num[k] - 1)
    else:
      continue
    if nb is not None and res_name[nb] not in ("ACE", "NME"):
      anchor_res.add(nb)

  ca = [i for k in ks if is_protein[k] for i in range(*blocks[k]) if atom_name[i] == "CA"]
  anchors = [i for i in ca if block_of[i] in anchor_res]
  anchors = anchors[1:-1] if len(anchors) >= 2 else []
  if not anchors or not ca:
    return [], [], None

  ca = np.array(ca, dtype=int)
  plain = cdist(xyz[ca], xyz[anchors])
  dmi, sh = mi_matrix(xyz[ca], xyz[anchors])
  keep_plain = np.any(plain <= en_max, axis=1)
  keep_mi = np.any(np.where(sh, dmi, plain) <= en_max, axis=1)

  def pairs_of(mask, minimum_image):
    kept = ca[mask]
    if len(kept) < 2:
      return kept, []
    d = cdist(xyz[kept], xyz[kept])
    if minimum_image:
      dmi2, sh2 = mi_matrix(xyz[kept], xyz[kept])
      d = np.where(sh2, dmi2, d)
    a, b = np.nonzero(np.triu((d >= en_min) & (d <= en_max), 1))
    return kept, [(x, y, d[x, y]) for x, y in zip(a, b)]

  keep, pairs = pairs_of(keep_plain, False)
  keep_m, pairs_m = pairs_of(keep_mi, True)
  same = (np.array_equal(keep, keep_m)
          and [(x, y) for x, y, _ in pairs] == [(x, y) for x, y, _ in pairs_m])
  return pairs, list(keep), (None if same else (len(keep_m), len(pairs_m)))

en1dis, protkeep1, mism1 = build_elastic_network(side_res[1])
en2dis, protkeep2, mism2 = build_elastic_network(side_res[2])
numen1dis = len(en1dis)
numen2dis = len(en2dis)

#------------------------------------------------------
# Refuse a split frame or an empty interface before anything is written

if split_inter or mism1 or mism2:
  msg = ["%s is split across the periodic boundary: some pairs are close only through "
         "a periodic image (part of the complex sits a box vector away from the rest, "
         "or the complex touches its own image)." % args.input,
         "  interface pairs within %.1f nm: %d by plain distance, %d by minimum image"
         % (interfacecutoff, numinterdis, numinterdis + len(split_inter))]
  for name, en, keep, m in (("A", en1dis, protkeep1, mism1), ("B", en2dis, protkeep2, mism2)):
    if m:
      msg.append("  elastic network side %s: %d kept CAs and %d pairs by plain distance, "
                 "%d and %d by minimum image" % (name, len(keep), len(en), m[0], m[1]))
  msg.append("  The frame must be assembled (utils/assemble_frame.py) before the "
             "restraints are generated; no restraint written.")
  fail("\n".join(msg))

if numinterdis == 0:
  fail("no cross-side heavy-atom pair within %.1f nm in %s (also by minimum image): "
       "an empty interface set would integrate to 0 kJ/mol in every cycle"
       % (interfacecutoff, args.input))

#------------------------------------------------------
# Write index groups and pull blocks. Atom numbers are positions + 1, which is
# what the .gro number field holds below its 100000 wrap.

def anum(i):
  return i + 1

# Write index file
with open("index.ndx", "a") as index:
  # Inter-protein distance restraint groups
  for i, j, dist in interdis:
    for a in (anum(i), anum(j)):
      index.write(f"[ a_{a} ]\n")
      index.write(f"{a}\n")

  # Elastic network groups, prot1 then prot2
  for en, keep in ((en1dis, protkeep1), (en2dis, protkeep2)):
    for i, j, dist in en:
      for a in (anum(keep[i]), anum(keep[j])):
        index.write(f"[ a_{a} ]\n")
        index.write(f"{a}\n")

# Helper function to write MDP pull configurations
def write_mdp_config(filename, k, rate_inter, init_offset_inter):
  """Write pull configuration to MDP file."""
  with open(filename, "a") as f:
    total_groups = (numinterdis + numen1dis + numen2dis) * 2
    total_coords = numinterdis + numen1dis + numen2dis

    f.write(f"pull-ngroups            = {total_groups:.0f}\n")
    f.write(f"pull-ncoords            = {total_coords:.0f}\n")
    f.write("\n")

    groupcount = 1
    coordcount = 1

    # Inter-protein restraints, then the network of prot1 and of prot2
    blocks_out = [(anum(i), anum(j), dist + init_offset_inter, rate_inter, k)
                  for i, j, dist in interdis]
    for en, keep in ((en1dis, protkeep1), (en2dis, protkeep2)):
      blocks_out += [(anum(keep[i]), anum(keep[j]), dist, 0, enk) for i, j, dist in en]

    for atomnum1, atomnum2, init, rate, kk in blocks_out:
      f.write(f"pull-group{groupcount:.0f}-name        = a_{float(atomnum1):.0f}\n")
      g1 = groupcount
      groupcount += 1
      f.write(f"pull-group{groupcount:.0f}-name        = a_{float(atomnum2):.0f}\n")
      g2 = groupcount
      groupcount += 1

      f.write(f"pull-coord{coordcount:.0f}-type        = umbrella\n")
      f.write(f"pull-coord{coordcount:.0f}-geometry    = distance\n")
      f.write(f"pull-coord{coordcount:.0f}-dim         = Y Y Y\n")
      f.write(f"pull-coord{coordcount:.0f}-start       = no\n")
      f.write(f"pull-coord{coordcount:.0f}-rate        = {rate}\n")
      f.write(f"pull-coord{coordcount:.0f}-groups      = {g1:.0f} {g2:.0f}\n")
      f.write(f"pull-coord{coordcount:.0f}-init        = {init:.8f}\n")
      f.write(f"pull-coord{coordcount:.0f}-k           = {kk:.8f}\n")
      f.write("\n")
      coordcount += 1

# Calculate k value
k = 25000.0 / numinterdis

# Write all MDP files
write_mdp_config("bind.mdp", k, rate_inter=0.0002, init_offset_inter=0)
write_mdp_config("nptrev.mdp", k, rate_inter=0, init_offset_inter=1.0)
write_mdp_config("bindrev.mdp", k, rate_inter=-0.0002, init_offset_inter=1.0)

print(str(numinterdis))
