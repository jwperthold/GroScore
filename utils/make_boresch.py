#!/usr/bin/env python3
#
# make_boresch.py - Generate the pull configuration for the free-energy
# (groscore_fe) protocol.
#
# This is the FE-variant counterpart of make_disres_en.py. It writes a single,
# self-contained COM-pull block into each leg .mdp so that all restraint
# generation lives in one place with consistent pull-group numbering.
#
# Two leg families are produced:
#
#   1. Unbinding / rebinding leg  (bind_fe / nptrev_fe / bindrev_fe)
#      - Interface atom-atom umbrella restraints: behave exactly as in the
#        current engine (reference moves outward at pull-rate -> mechanical
#        separation work in pullf.xvg), with the ONLY addition being kB = 0 so
#        they fade out toward the unbound state (dhdl contribution).
#      - Boresch restraints (6 coords built on backbone-COM triad groups):
#        the distance r mirrors the interface restraints (same outward rate),
#        k switched 0 -> K_r; the two angles and three dihedrals have fixed
#        references (rate 0), k switched 0 -> K_ang. Orientation is handed over
#        to the Boresch frame as lambda -> 1.
#      - Elastic network: unchanged, fixed force constant (no lambda dependence).
#
#   2. Bound-state restraint leg  (boundfwd / boundrev)
#      - Interface restraints introduced 0 -> full in the bound state, NO
#        pulling (rate 0). Pure dhdl. Gives the free energy of introducing the
#        restraints in the bound ensemble. No Boresch here.
#
# The Boresch standard-state term is computed analytically (Boresch 2003 eq.32)
# and written to boresch_analytical.gs for groscore_fe.py.
#
# Work bookkeeping for the unbinding/rebinding leg:
#   W_total = W_pull (interface + Boresch-r, the moving coords) + W_dhdl (all
#   force-constant switching). The moving coords are written FIRST so the pull
#   integrator can sum the first (numinterdis + 1) force columns.
#
# Usage:
#   python make_boresch.py -f npt_init_cluster.gro -m chain_map.gs > numpertres.gs
#

import os, sys, re, argparse, math
import numpy as np
from scipy.spatial.distance import cdist

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate Boresch + interface + elastic-network pull config for the FE protocol.")
parser.add_argument('-f', '--input', type=str, default="npt_init_cluster.gro", help="Input coordinate file (default: npt_init_cluster.gro)")
parser.add_argument('-m', '--chainmap', type=str, required=True, help="Chain map file containing residue numbers for protein B (ligand side).")
parser.add_argument('-T', '--temp', type=float, default=310.0, help="Temperature in K for the analytical term (default: 310).")
parser.add_argument('--pull-dist', type=float, default=1.0, help="Maximum COM-COM separation added during unbinding, in nm (default: 1.0).")
parser.add_argument('--pull-rate', type=float, default=0.0002, help="Pull rate in nm/ps (default: 0.0002).")
args = parser.parse_args()

#------------------------------------------------------
# Physical constants and fixed force constants (OpenFE ABFE defaults)

R_GAS = 0.00831446261815324          # kJ/mol/K
RT = R_GAS * args.temp               # kJ/mol
V0 = 1.6605390671                    # nm^3, standard-state volume (1 mol/L)

K_R = 4184.0                         # kJ/mol/nm^2  (= 10 kcal/mol/A^2)
K_ANG_RAD = 334.72                   # kJ/mol/rad^2 (= 80 kcal/mol/rad^2), used in eq.32
DEG2RAD = math.pi / 180.0
K_ANG_DEG = K_ANG_RAD * DEG2RAD**2   # kJ/mol/deg^2, used in the GROMACS mdp

# Interface / elastic-network parameters (identical to make_disres_en.py)
interfacecutoff = 0.6
en_min = 0.4
en_max = 0.9
enk = 250.0

# Approximate backbone atom masses (for mass-weighted COM, matching GROMACS)
ATOM_MASS = {"N": 14.007, "CA": 12.011, "C": 12.011}
BACKBONE = ("N", "CA", "C")

#------------------------------------------------------

def read_chain_map(filepath):
  """Return the set of residue numbers belonging to protein B (ligand side)."""
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

# Ion / ligand residue numbers -> highest structural residue number (everything
# above is counterion / bulk solvent added by gmx genion).
extra_residues = set()
for gs_file in ["ion_residues.gs", "ligand_residues.gs"]:
  gs_path = os.path.join(os.path.dirname(args.chainmap), gs_file)
  if os.path.isfile(gs_path):
    for line in open(gs_path):
      if not line.strip().startswith("#"):
        try:
          extra_residues.add(int(line.strip()))
        except (ValueError, IndexError):
          pass
all_structural = residues_b | extra_residues
max_structural_resnum = max(all_structural) if all_structural else 0

#------------------------------------------------------
# Parse the coordinate file. Store per atom: resname (with resnum), resnum,
# atomname, atomnum (int), coords. Split into receptor (prot1) and ligand (prot2)
# exactly as make_disres_en.py does.

prot1_data = []  # receptor: [(resname, resnum, atomname, atomnum, x, y, z), ...]
prot2_data = []  # ligand / protein B

# GRO fixed-width columns: resnum(0:5) resname(5:10) atomname(10:15) atomnum(15:20)
# x(20:28) y(28:36) z(36:44). The GRO atom-number field wraps modulo 100000, so for
# systems > 99999 atoms it cannot be used as the absolute atom index (that would put
# pull groups on the wrong atoms in index.ndx). The atom's line position (1-based)
# is the true topology index, so use that; also parse by fixed columns rather than
# by split (which breaks when fields touch or for SOL entries).
if os.path.isfile(args.input):
  with open(args.input, "r") as f:
    gro_lines = f.readlines()
  try:
    natoms = int(gro_lines[1])
  except (IndexError, ValueError):
    natoms = 0
  for i in range(natoms):
    if 2 + i >= len(gro_lines):
      break
    line = gro_lines[2 + i]
    try:
      resnum = int(line[0:5])
      resname_field = line[0:10].strip()      # resnum+resname, e.g. "577GLN"
      resname = line[5:10].strip()
      atomname = line[10:15].strip()
      atomnum = i + 1                          # absolute 1-based index (wrap-safe)
      x, y, z = float(line[20:28]), float(line[28:36]), float(line[36:44])
    except (ValueError, IndexError):
      continue
    if resname == "SOL" or resnum > max_structural_resnum:
      continue
    rec = (resname_field, resnum, atomname, atomnum, x, y, z)
    if resnum not in residues_b:
      prot1_data.append(rec)
    else:
      prot2_data.append(rec)

len1 = len(prot1_data)
len2 = len(prot2_data)

prot1_coords = np.array([(d[4], d[5], d[6]) for d in prot1_data], dtype=np.float64) if len1 else np.empty((0, 3))
prot2_coords = np.array([(d[4], d[5], d[6]) for d in prot2_data], dtype=np.float64) if len2 else np.empty((0, 3))

#======================================================
# PART 1 - Interface atom-atom restraints (identical selection to make_disres_en)
#======================================================

prot1_valid = np.array([i for i in range(len1) if prot1_data[i][2][0] != "H" and prot1_data[i][2][:2] != "MN"])
prot2_valid = np.array([i for i in range(len2) if prot2_data[i][2][0] != "H" and prot2_data[i][2][:2] != "MN"])

interdis = []  # (i, j, dist) indices into prot1_data / prot2_data
if len(prot1_valid) > 0 and len(prot2_valid) > 0:
  d = cdist(prot1_coords[prot1_valid], prot2_coords[prot2_valid])
  for i_idx, i in enumerate(prot1_valid):
    for j_idx, j in enumerate(prot2_valid):
      if d[i_idx, j_idx] <= interfacecutoff:
        interdis.append((i, j, d[i_idx, j_idx]))
numinterdis = len(interdis)

#======================================================
# PART 2 - Elastic network (identical to make_disres_en.build_elastic_network)
#======================================================

def build_elastic_network(prot_data, prot_coords):
  prot_len = len(prot_data)
  anchor_resnames = set()
  resname_to_type = {}
  for i in range(prot_len):
    resname_to_type[prot_data[i][0]] = prot_data[i][0][-3:]
  for i in range(prot_len):
    if prot_data[i][2] in ("OT", "H2"):
      anchor_resnames.add(prot_data[i][0])
  resnum_to_resname = {}
  for resname in resname_to_type:
    s = re.search(r"\d+", resname)
    if s:
      resnum_to_resname[int(s.group(0))] = resname
  for resname, res3 in resname_to_type.items():
    s = re.search(r"\d+", resname)
    if not s:
      continue
    num = int(s.group(0))
    if res3 == "ACE":
      nxt = resnum_to_resname.get(num + 1)
      if nxt and resname_to_type.get(nxt) not in ("ACE", "NME"):
        anchor_resnames.add(nxt)
    elif res3 == "NME":
      prv = resnum_to_resname.get(num - 1)
      if prv and resname_to_type.get(prv) not in ("ACE", "NME"):
        anchor_resnames.add(prv)
  anchor_indices = [i for i in range(prot_len) if prot_data[i][2] == "CA" and prot_data[i][0] in anchor_resnames]
  anchor_indices = anchor_indices[1:-1] if len(anchor_indices) >= 2 else []
  if not anchor_indices:
    return [], []
  anchor_coords = prot_coords[anchor_indices]
  ca_indices = [i for i in range(prot_len) if prot_data[i][2] == "CA"]
  if not ca_indices:
    return [], []
  ca_coords = prot_coords[ca_indices]
  keep_mask = np.any(cdist(ca_coords, anchor_coords) <= 0.9, axis=1)
  protkeep = [ca_indices[i] for i in range(len(ca_indices)) if keep_mask[i]]
  if len(protkeep) < 2:
    return [], protkeep
  keep_distances = cdist(prot_coords[protkeep], prot_coords[protkeep])
  en_pairs = []
  for i in range(len(protkeep)):
    for j in range(i + 1, len(protkeep)):
      if en_min <= keep_distances[i, j] <= en_max:
        en_pairs.append((i, j, keep_distances[i, j]))
  return en_pairs, protkeep

en1dis, protkeep1 = build_elastic_network(prot1_data, prot1_coords)
en2dis, protkeep2 = build_elastic_network(prot2_data, prot2_coords)
numen1dis = len(en1dis)
numen2dis = len(en2dis)

#======================================================
# PART 3 - Boresch anchor selection (snapshot-only heuristic, backbone-COM triads)
#======================================================

def residue_backbone_groups(prot_data):
  """Return {resnum: {'atoms': [atomnum,...], 'com': np.array, 'ca': np.array}}
  for residues that carry a full N/CA/C backbone (i.e. real amino acids)."""
  by_res = {}
  for (resname, resnum, atomname, atomnum, x, y, z) in prot_data:
    if atomname in BACKBONE:
      by_res.setdefault(resnum, {})[atomname] = (atomnum, np.array([x, y, z]))
  groups = {}
  for resnum, atoms in by_res.items():
    if all(a in atoms for a in BACKBONE):
      masses = np.array([ATOM_MASS[a] for a in BACKBONE])
      coords = np.array([atoms[a][1] for a in BACKBONE])
      com = (masses[:, None] * coords).sum(axis=0) / masses.sum()
      groups[resnum] = {
        "atoms": [atoms[a][0] for a in BACKBONE],
        "com": com,
        "ca": atoms["CA"][1],
      }
  return groups

rec_groups = residue_backbone_groups(prot1_data)
lig_groups = residue_backbone_groups(prot2_data)

def burial_scores(groups, coords):
  """Rigidity proxy: count of same-protein heavy atoms within 1.0 nm of the CA."""
  scores = {}
  cas = {rn: g["ca"] for rn, g in groups.items()}
  if len(coords) == 0:
    return {rn: 0 for rn in groups}
  for rn, ca in cas.items():
    scores[rn] = int(np.count_nonzero(np.linalg.norm(coords - ca, axis=1) <= 1.0))
  return scores

rec_burial = burial_scores(rec_groups, prot1_coords)
lig_burial = burial_scores(lig_groups, prot2_coords)

def angle_deg(a, b, c):
  """Angle at b (degrees) between vectors b->a and b->c."""
  v1, v2 = a - b, c - b
  cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
  return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))

def dihedral_deg(a, b, c, d):
  """Dihedral a-b-c-d in (-180, 180] degrees."""
  b1, b2, b3 = b - a, c - b, d - c
  n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
  m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
  x = np.dot(n1, n2)
  y = np.dot(m1, n2)
  return math.degrees(math.atan2(y, x))

def tri_area(p, q, r):
  return 0.5 * np.linalg.norm(np.cross(q - p, r - p))

def select_boresch_anchors():
  """Pick receptor (P1,P2,P3) and ligand (L1,L2,L3) anchor residues.

  Greedy snapshot heuristic:
    L1 : buried ligand residue closest to the receptor (defines r with P3)
    P3 : buried receptor residue near L1, COM-COM distance in [0.5, 1.2] nm
    L2 : buried ligand residue maximizing lever arm from L1 with theta_B in window
    P2 : buried receptor residue maximizing lever arm from P3 with theta_A in window
    L3 : buried ligand residue maximizing non-collinearity of (L1,L2,L3)
    P1 : buried receptor residue maximizing non-collinearity of (P3,P2,P1)
  Angle acceptance window keeps the analytical eq.32 valid (away from 0/180). All
  anchor lever arms are capped at ARM_MAX so the Boresch frame stays local: no
  coordinate vector may approach half the box, or GROMACS aborts the pull with a
  minimum-image error (seen on large complexes where the maximum-spread pick put
  P2/P3 ~6 nm apart).
  """
  if len(rec_groups) < 3 or len(lig_groups) < 3:
    return None

  ANG_LO, ANG_HI = 45.0, 135.0
  ARM_MAX = 1.2   # nm; keep every anchor within ~1.2 nm of its reference so no
                  # Boresch coordinate vector nears the minimum-image limit

  rec_med = np.median(list(rec_burial.values())) if rec_burial else 0
  lig_med = np.median(list(lig_burial.values())) if lig_burial else 0
  rec_pool = [rn for rn in rec_groups if rec_burial[rn] >= rec_med] or list(rec_groups)
  lig_pool = [rn for rn in lig_groups if lig_burial[rn] >= lig_med] or list(lig_groups)

  # L1: ligand residue in the pool closest to any receptor anchor
  rec_com_arr = np.array([rec_groups[rn]["com"] for rn in rec_pool])
  best_L1, best_d = None, 1e9
  for rn in lig_pool:
    dmin = np.min(np.linalg.norm(rec_com_arr - lig_groups[rn]["com"], axis=1))
    if dmin < best_d:
      best_d, best_L1 = dmin, rn
  L1 = best_L1
  L1c = lig_groups[L1]["com"]

  # P3: receptor residue with COM-COM distance to L1 in [0.5, 1.2] nm, else closest
  cand = [(np.linalg.norm(rec_groups[rn]["com"] - L1c), rn) for rn in rec_pool]
  in_range = [(d, rn) for d, rn in cand if 0.5 <= d <= 1.2]
  P3 = min(in_range or cand)[1]
  P3c = rec_groups[P3]["com"]

  def pick_lever(pool, exclude, vertex_c, other_c, want_angle):
    """From pool pick residue maximizing lever arm from vertex_c; if want_angle,
    prefer angle(other_c, vertex_c, cand) inside the window, else closest to 90."""
    best, best_key = None, None
    for rn in pool:
      if rn in exclude:
        continue
      c = (lig_groups.get(rn) or rec_groups.get(rn))["com"]
      arm = np.linalg.norm(c - vertex_c)
      if arm < 0.3 or arm > ARM_MAX:      # keep the lever arm local (min-image safe)
        continue
      ang = angle_deg(other_c, vertex_c, c)
      in_win = ANG_LO <= ang <= ANG_HI
      key = (1 if in_win else 0, arm if in_win else -abs(ang - 90.0))
      if best_key is None or key > best_key:
        best_key, best = key, rn
    return best

  # L2 defines theta_B (P3 - L1 - L2); P2 defines theta_A (P2 - P3 - L1)
  L2 = pick_lever(lig_pool, {L1}, L1c, P3c, want_angle=True)
  P2 = pick_lever(rec_pool, {P3}, P3c, L1c, want_angle=True)
  if L2 is None or P2 is None:
    return None
  L2c, P2c = lig_groups[L2]["com"], rec_groups[P2]["com"]

  def pick_noncollinear(pool, exclude, p, q):
    # Prefer a residue local to both existing anchors (within ARM_MAX) that
    # maximizes non-collinearity; relax the locality cap only if none qualifies.
    for local in (True, False):
      best, best_area = None, -1.0
      for rn in pool:
        if rn in exclude:
          continue
        c = (lig_groups.get(rn) or rec_groups.get(rn))["com"]
        if local and (np.linalg.norm(c - p) > ARM_MAX or np.linalg.norm(c - q) > ARM_MAX):
          continue
        a = tri_area(p, q, c)
        if a > best_area:
          best_area, best = a, rn
      if best is not None:
        return best
    return None

  L3 = pick_noncollinear(lig_pool, {L1, L2}, L1c, L2c)
  P1 = pick_noncollinear(rec_pool, {P3, P2}, P3c, P2c)
  if L3 is None or P1 is None:
    return None

  return {"P1": P1, "P2": P2, "P3": P3, "L1": L1, "L2": L2, "L3": L3}

anchors = select_boresch_anchors()
if anchors is None:
  sys.stderr.write("make_boresch: ERROR - could not select Boresch anchors "
                   "(need >=3 backbone residues per side).\n")
  # Emit a marker so job_fe.run can fall back / flag the structure.
  with open("boresch_failed.gs", "w") as f:
    f.write("BORESCH_ANCHOR_SELECTION_FAILED\n")
  print("0")
  sys.exit(1)

def gc(name):
  return (rec_groups.get(anchors[name]) or lig_groups.get(anchors[name]))["com"]

P1c, P2c, P3c = gc("P1"), gc("P2"), gc("P3")
L1c, L2c, L3c = gc("L1"), gc("L2"), gc("L3")

# Reference geometry (Boresch definition)
ref_r   = float(np.linalg.norm(P3c - L1c))          # nm
ref_thA = angle_deg(P2c, P3c, L1c)                  # deg  theta_A = angle(P2,P3,L1)
ref_thB = angle_deg(P3c, L1c, L2c)                  # deg  theta_B = angle(P3,L1,L2)
ref_phA = dihedral_deg(P1c, P2c, P3c, L1c)          # deg  phi_A
ref_phB = dihedral_deg(P2c, P3c, L1c, L2c)          # deg  phi_B
ref_phC = dihedral_deg(P3c, L1c, L2c, L3c)          # deg  phi_C

#======================================================
# PART 4 - Analytical standard-state term (Boresch 2003, eq. 32)
#
#   dG_release = -RT ln [ (8 pi^2 V0 sqrt(Kr KthA KthB KphA KphB KphC))
#                         / (r0^2 sin(thA0) sin(thB0) (2 pi RT)^3) ]
#
# Convention: dG_release is the free energy of REMOVING the Boresch restraint
# from the (decoupled) ligand and letting it explore the standard-state volume.
# dG_intro (adding the restraint) = -dG_release. groscore_fe.py applies the sign
# appropriate to the thermodynamic cycle.
#======================================================

# r0 for the standard-state release is the UNBOUND reference distance. The Boresch
# distance coordinate is pulled out by args.pull_dist during unbinding, so the
# restraint is released at (ref_r + pull_dist), not the bound ref_r. Using ref_r
# here would be inconsistent with the pull work already spent moving it out, and
# would leave the thermodynamic cycle unclosed. Only the distance moves; the
# angle/dihedral references (rate 0) keep their measured values.
r0_release = ref_r + args.pull_dist
thA0 = ref_thA * DEG2RAD
thB0 = ref_thB * DEG2RAD
prodK = K_R * (K_ANG_RAD ** 5)   # Kr * KthA*KthB*KphA*KphB*KphC (all angles equal)
numerator = 8.0 * math.pi**2 * V0 * math.sqrt(prodK)
denominator = (r0_release**2) * math.sin(thA0) * math.sin(thB0) * (2.0 * math.pi * RT)**3
dG_release = -RT * math.log(numerator / denominator)   # kJ/mol

with open("boresch_analytical.gs", "w") as f:
  f.write("# Boresch standard-state analytical term (eq. 32)\n")
  f.write("# dG_release: free energy of removing the Boresch restraint to the\n")
  f.write("# standard state (1 mol/L). dG_intro = -dG_release.\n")
  f.write("# quantity            value        unit\n")
  f.write("dG_release_kJ_mol     %.6f\n" % dG_release)
  f.write("temperature_K         %.2f\n" % args.temp)
  f.write("ref_r_bound_nm        %.6f\n" % ref_r)
  f.write("r0_release_nm         %.6f\n" % r0_release)
  f.write("pull_dist_nm          %.6f\n" % args.pull_dist)
  f.write("ref_thetaA_deg        %.4f\n" % ref_thA)
  f.write("ref_thetaB_deg        %.4f\n" % ref_thB)
  f.write("ref_phiA_deg          %.4f\n" % ref_phA)
  f.write("ref_phiB_deg          %.4f\n" % ref_phB)
  f.write("ref_phiC_deg          %.4f\n" % ref_phC)

with open("boresch_anchors.gs", "w") as f:
  f.write("# Boresch anchor residues and backbone-COM atom groups\n")
  f.write("# role  resnum  atomnums(N,CA,C)\n")
  for role in ("P1", "P2", "P3", "L1", "L2", "L3"):
    g = rec_groups.get(anchors[role]) or lig_groups.get(anchors[role])
    f.write("%s  %d  %s\n" % (role, anchors[role], ",".join(str(a) for a in g["atoms"])))

#======================================================
# PART 5 - Index groups
#======================================================

def write_index_groups():
  with open("index.ndx", "a") as index:
    # Single-atom groups for interface + elastic network (matching make_disres_en)
    for i, j, _ in interdis:
      for anum in (prot1_data[i][3], prot2_data[j][3]):
        index.write("[ a_%d ]\n%d\n" % (anum, anum))
    for i, j, _ in en1dis:
      for anum in (prot1_data[protkeep1[i]][3], prot1_data[protkeep1[j]][3]):
        index.write("[ a_%d ]\n%d\n" % (anum, anum))
    for i, j, _ in en2dis:
      for anum in (prot2_data[protkeep2[i]][3], prot2_data[protkeep2[j]][3]):
        index.write("[ a_%d ]\n%d\n" % (anum, anum))
    # Multi-atom backbone-COM triad groups for the Boresch anchors
    for role in ("P1", "P2", "P3", "L1", "L2", "L3"):
      g = rec_groups.get(anchors[role]) or lig_groups.get(anchors[role])
      index.write("[ bor_%s ]\n%s\n" % (role, " ".join(str(a) for a in g["atoms"])))

write_index_groups()

#======================================================
# PART 6 - Pull-block writers
#
# A "coord" is a dict with: name-list of ndx groups, geometry, dim, init, rate,
# k (state A), kB (state B). Groups are declared once and referenced by index.
#======================================================

k_inter = 25000.0 / numinterdis if numinterdis > 0 else 0.0

def boresch_group_ndx(role):
  return "bor_%s" % role

def build_coords(family, direction):
  """Return (pull_groups, coords) for a leg.

  family    : 'unbind'  -> interface(moving, k->0) + EN(fixed) + Boresch(0->full)
              'bound'   -> interface(fixed, 0->full) + EN(fixed), no Boresch
  direction : 'fwd' (init at bound geometry) or 'rev' (init at unbound geometry,
              rate negated). For 'bound' both use rate 0; direction only flips
              which endpoint init sits at (irrelevant, kept for symmetry).
  """
  pull_groups = []          # list of ndx group names (index = position+1)
  group_index = {}          # ndx name -> pull-group index

  def gidx(ndx_name):
    if ndx_name not in group_index:
      pull_groups.append(ndx_name)
      group_index[ndx_name] = len(pull_groups)
    return group_index[ndx_name]

  coords = []
  sign = 1.0 if direction == "fwd" else -1.0
  off = 0.0 if direction == "fwd" else args.pull_dist   # rev starts at unbound

  if family == "unbind":
    # 1) Interface restraints FIRST (moving) so pull integrator sums them first.
    rate = sign * args.pull_rate
    for i, j, dist in interdis:
      g1 = gidx("a_%d" % prot1_data[i][3])
      g2 = gidx("a_%d" % prot2_data[j][3])
      coords.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                         init=dist + off, rate=rate, k=k_inter, kB=0.0))
    # 2) Boresch distance r (moving) - the coord that takes over the pulling.
    gP3 = gidx(boresch_group_ndx("P3"))
    gL1 = gidx(boresch_group_ndx("L1"))
    coords.append(dict(geometry="distance", dim="Y Y Y", groups=[gP3, gL1],
                       init=ref_r + off, rate=rate, k=0.0, kB=K_R))
    # 3) Elastic network (fixed, no lambda dependence).
    for i, j, dist in en1dis:
      g1 = gidx("a_%d" % prot1_data[protkeep1[i]][3])
      g2 = gidx("a_%d" % prot1_data[protkeep1[j]][3])
      coords.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                         init=dist, rate=0.0, k=enk, kB=enk))
    for i, j, dist in en2dis:
      g1 = gidx("a_%d" % prot2_data[protkeep2[i]][3])
      g2 = gidx("a_%d" % prot2_data[protkeep2[j]][3])
      coords.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                         init=dist, rate=0.0, k=enk, kB=enk))
    # 4) Boresch angles + dihedrals (fixed reference, k switched 0 -> full).
    gP2 = gidx(boresch_group_ndx("P2"))
    gP1 = gidx(boresch_group_ndx("P1"))
    gL2 = gidx(boresch_group_ndx("L2"))
    gL3 = gidx(boresch_group_ndx("L3"))
    # theta_A = angle(P2, P3, L1): vectors P3->P2 and P3->L1
    coords.append(dict(geometry="angle", dim="Y Y Y", groups=[gP3, gP2, gP3, gL1],
                       init=ref_thA, rate=0.0, k=0.0, kB=K_ANG_DEG))
    # theta_B = angle(P3, L1, L2): vectors L1->P3 and L1->L2
    coords.append(dict(geometry="angle", dim="Y Y Y", groups=[gL1, gP3, gL1, gL2],
                       init=ref_thB, rate=0.0, k=0.0, kB=K_ANG_DEG))
    # phi_A = dihedral(P1, P2, P3, L1): vectors P1->P2, P2->P3, P3->L1
    coords.append(dict(geometry="dihedral", dim="Y Y Y",
                       groups=[gP1, gP2, gP2, gP3, gP3, gL1],
                       init=ref_phA, rate=0.0, k=0.0, kB=K_ANG_DEG))
    # phi_B = dihedral(P2, P3, L1, L2)
    coords.append(dict(geometry="dihedral", dim="Y Y Y",
                       groups=[gP2, gP3, gP3, gL1, gL1, gL2],
                       init=ref_phB, rate=0.0, k=0.0, kB=K_ANG_DEG))
    # phi_C = dihedral(P3, L1, L2, L3)
    coords.append(dict(geometry="dihedral", dim="Y Y Y",
                       groups=[gP3, gL1, gL1, gL2, gL2, gL3],
                       init=ref_phC, rate=0.0, k=0.0, kB=K_ANG_DEG))

  elif family == "bound":
    # Interface restraints introduced 0 -> full, no pulling (rate 0).
    for i, j, dist in interdis:
      g1 = gidx("a_%d" % prot1_data[i][3])
      g2 = gidx("a_%d" % prot2_data[j][3])
      coords.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                         init=dist, rate=0.0, k=0.0, kB=k_inter))
    for i, j, dist in en1dis:
      g1 = gidx("a_%d" % prot1_data[protkeep1[i]][3])
      g2 = gidx("a_%d" % prot1_data[protkeep1[j]][3])
      coords.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                         init=dist, rate=0.0, k=enk, kB=enk))
    for i, j, dist in en2dis:
      g1 = gidx("a_%d" % prot2_data[protkeep2[i]][3])
      g2 = gidx("a_%d" % prot2_data[protkeep2[j]][3])
      coords.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                         init=dist, rate=0.0, k=enk, kB=enk))

  return pull_groups, coords

def write_pull_block(filename, pull_groups, coords):
  with open(filename, "a") as f:
    f.write("\n")
    f.write("pull-ngroups            = %d\n" % len(pull_groups))
    f.write("pull-ncoords            = %d\n" % len(coords))
    f.write("\n")
    for gi, ndx_name in enumerate(pull_groups, start=1):
      f.write("pull-group%d-name        = %s\n" % (gi, ndx_name))
    f.write("\n")
    for ci, c in enumerate(coords, start=1):
      f.write("pull-coord%d-type        = umbrella\n" % ci)
      f.write("pull-coord%d-geometry    = %s\n" % (ci, c["geometry"]))
      f.write("pull-coord%d-dim         = %s\n" % (ci, c["dim"]))
      f.write("pull-coord%d-groups      = %s\n" % (ci, " ".join(str(g) for g in c["groups"])))
      f.write("pull-coord%d-start       = no\n" % ci)
      f.write("pull-coord%d-init        = %.8f\n" % (ci, c["init"]))
      f.write("pull-coord%d-rate        = %.8f\n" % (ci, c["rate"]))
      f.write("pull-coord%d-k           = %.8f\n" % (ci, c["k"]))
      f.write("pull-coord%d-kB          = %.8f\n" % (ci, c["kB"]))
      f.write("\n")

# Unbinding / rebinding leg
pg_fwd, co_fwd = build_coords("unbind", "fwd")
pg_rev, co_rev = build_coords("unbind", "rev")
write_pull_block("bind_fe.mdp", pg_fwd, co_fwd)     # forward: unbinding (lambda 0->1)
write_pull_block("bindrev_fe.mdp", pg_rev, co_rev)  # reverse: rebinding (lambda 1->0)

# Hold leg at the unbound restrained state (lambda = 1): same coords as the
# reverse leg but with zero rate; job_fe.run pins init-lambda = 1, delta-lambda = 0.
pg_hold, co_hold = build_coords("unbind", "rev")
for c in co_hold:
  c["rate"] = 0.0
write_pull_block("nptrev_fe.mdp", pg_hold, co_hold)

# Bound-state restraint leg
pg_b_fwd, co_b_fwd = build_coords("bound", "fwd")
pg_b_rev, co_b_rev = build_coords("bound", "rev")
write_pull_block("boundfwd.mdp", pg_b_fwd, co_b_fwd)
write_pull_block("boundrev.mdp", pg_b_rev, co_b_rev)

#======================================================
# Output: number of moving pull coords whose forces feed the pull integrator.
# Interface restraints (numinterdis) + the single Boresch distance coord = +1.
# (Elastic network and Boresch angles/dihedrals have rate 0 -> no pull work.)
#======================================================
print(numinterdis + 1)
