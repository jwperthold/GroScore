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
#   1. Unbinding / rebinding leg  (bind_fe / bindrev_fe)
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
#   1b. Hold leg  (nptrev_fe): the Boresch coordinates ONLY, at zero rate. The
#      interface coordinates are inert there (kB = 0 at lambda = 1) but GROMACS
#      still enforces its 0.49 * box distance check on them, which killed 8 of 14
#      holds. See the comment at the write_pull_block call for the detail.
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
#   python make_boresch.py -f npt_probe_cluster.gro -m chain_map.gs > numpertres.gs
#

import os, sys, re, argparse, math, itertools, subprocess, traceback, shutil
import numpy as np
from scipy.spatial.distance import cdist

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate Boresch + interface + elastic-network pull config for the FE protocol.")
parser.add_argument('-f', '--input', type=str, default="npt_probe_cluster.gro",
                    help="Reference structure the restraints are measured from "
                         "(default: npt_probe_cluster.gro, the probe equilibration "
                         "clustered). --topol and --readback-index must match it.")
parser.add_argument('-m', '--chainmap', type=str, required=True, help="Chain map file containing residue numbers for protein B (ligand side).")
parser.add_argument('-T', '--temp', type=float, default=310.0, help="Temperature in K for the analytical term (default: 310).")
parser.add_argument('--pull-dist', type=float, default=1.0, help="Maximum COM-COM separation added during unbinding, in nm (default: 1.0).")
parser.add_argument('--pull-rate', type=float, default=0.00005, help="Pull rate in nm/ps (default: 0.00005, i.e. 1.0 nm over the 20 ns unbinding leg).")
parser.add_argument('--traj', type=str, default="npt_probe.xtc",
                    help="Equilibration trajectory used to measure backbone "
                         "rigidity for anchor selection (default: npt_probe.xtc). "
                         "If missing, the burial heuristic is used instead.")
parser.add_argument('--tpr', type=str, default="npt_probe.tpr",
                    help="Run input matching --traj (default: npt_probe.tpr). Also "
                         "the source of the atom masses the COMs are taken with.")
parser.add_argument('--topol', type=str, default="topol.top",
                    help="Topology for the pull readback check (default: topol.top). "
                         "The check grompps the emitted forward pull block at "
                         "nsteps=0 and confirms GROMACS reproduces the six Boresch "
                         "references it was given, so it must match -f.")
parser.add_argument('--index', type=str, nargs='+', default=["index.ndx"],
                    help="Index file(s) the generated a_*/bor_* groups are written "
                         "into (default: index.ndx). More than one because the "
                         "reference structure and the production legs can live in "
                         "different systems: the groups are SOLUTE atom numbers, "
                         "which are identical in both, since solvate appends water "
                         "after the solute and genion replaces waters.")
parser.add_argument('--readback-index', dest='readback_index', type=str, default=None,
                    help="Index for the readback grompp; must match -f and --topol "
                         "(default: the first --index).")
parser.add_argument('--box-from', dest='box_from', type=str, default=None,
                    help="Coordinate file supplying the BOX the pull-limit gates "
                         "are scaled against (default: -f). Pass the PRODUCTION "
                         "box here whenever -f comes from a different system: the "
                         "anchor arm cap, the cross-triad rejection and the "
                         "BORESCH_R_TOO_LARGE abort all scale with 0.49*min|box|, "
                         "so a smaller reference box silently selects different "
                         "anchors and can abort on a constraint the real run does "
                         "not have.")
parser.add_argument('--no-readback', dest='readback', action='store_false',
                    help="Skip the GROMACS pull readback check.")
args = parser.parse_args()
if args.readback_index is None:
  args.readback_index = args.index[0]

#------------------------------------------------------
# Physical constants and fixed force constants (OpenFE ABFE defaults)

R_GAS = 0.00831446261815324          # kJ/mol/K
RT = R_GAS * args.temp               # kJ/mol
V0 = 1.6605390671                    # nm^3, standard-state volume (1 mol/L)

K_R = 4184.0                         # kJ/mol/nm^2  (= 10 kcal/mol/A^2)
K_ANG_RAD = 334.72                   # kJ/mol/rad^2 (= 80 kcal/mol/rad^2)
DEG2RAD = math.pi / 180.0

# GROMACS mixes units for angle pull coordinates: pull-coordN-init and -rate are
# read in DEGREES, but pull-coordN-k and -kB are read in kJ/mol/rad^2 and are NOT
# converted. See docs/user-guide/mdp-options.rst, "Note that for angles the force
# constant is expressed in terms of radians (while pull-coord1-init and
# pull-coord1-rate are expressed in degrees)", and pull.cpp, where
# pull_conversion_factor_userinput2internal() is applied to init and rate only.
#
# This file used to convert K_ANG_RAD to kJ/mol/deg^2 for the mdp while eq.32
# kept the rad^2 value. GROMACS read that number as rad^2, so every angular
# Boresch restraint was (180/pi)^2 = 3283x too weak: 0.102 instead of 334.72
# kJ/mol/rad^2, an RMS fluctuation of 288 deg rather than 5 deg, i.e. no
# orientational restraint at all. The distance restraint was unaffected, being
# in kJ/mol/nm^2. K_ANG_RAD therefore goes into the mdp unchanged.

ANG_LO, ANG_HI = 45.0, 135.0   # Boresch angle window, keeps eq.32 valid

# Interface / elastic-network parameters (identical to make_disres_en.py)
interfacecutoff = 0.6
en_min = 0.4
en_max = 0.9
enk = 250.0

# Fallback only. The real masses come from the tpr, see mass_of() below: these
# element masses are NOT what GROMACS uses for the pull COM. Every settings tree
# sets mass-repartition-factor = 3, which grompp applies before the pull code
# sees the system (backbone N becomes 11.994, CA 9.994), and GROMOS54A8 is
# united-atom where CA is CH1 at 13.019. Using this table put the reference
# angles up to 0.96 degrees away from what GROMACS measures.
ATOM_MASS = {"N": 14.007, "CA": 12.011, "C": 12.011}
BACKBONE = ("N", "CA", "C")

TPR_MASS = {}    # 1-based atom number -> mass, loaded from the tpr if available


def load_tpr_masses(tpr):
  """{atomnum: mass} straight out of the tpr, so the COMs match GROMACS's own.

  gmx dump prints atoms per MOLECULETYPE with atom[i] restarting at 0 in each,
  and the system is then assembled from molblocks. Mapping atom[i] to global
  i + 1 therefore only works for the first moltype and lets SOL overwrite the
  protein's first atoms. Expand the molblocks instead.

  Empty on any failure, in which case mass_of() falls back to ATOM_MASS."""
  if not (tpr and os.path.isfile(tpr) and shutil.which("gmx")):
    return {}
  try:
    r = subprocess.run(["gmx", "dump", "-s", tpr],
                       capture_output=True, text=True)
    if r.returncode != 0:
      return {}
    per_moltype, blocks = {}, []
    cur_mt, cur_blk = None, None
    for line in r.stdout.split("\n"):
      m = re.match(r"\s*moltype \((\d+)\):", line)
      if m:
        cur_mt = int(m.group(1))
        per_moltype.setdefault(cur_mt, [])
        continue
      m = re.match(r"\s*molblock \((\d+)\):", line)
      if m:
        cur_blk = {"moltype": None, "n": None}
        blocks.append(cur_blk)
        continue
      if cur_blk is not None and cur_blk["n"] is None:
        m = re.match(r"\s*moltype\s*=\s*(\d+)", line)
        if m:
          cur_blk["moltype"] = int(m.group(1)); continue
        m = re.match(r"\s*#molecules\s*=\s*(\d+)", line)
        if m:
          cur_blk["n"] = int(m.group(1)); continue
      if cur_mt is not None:
        m = re.search(r"atom\[\s*\d+\]=\{[^}]*?\bm=\s*([-0-9.eE+]+)", line)
        if m:
          per_moltype[cur_mt].append(float(m.group(1)))
    if not per_moltype or not blocks:
      return {}
    out, g = {}, 1
    for b in blocks:
      masses = per_moltype.get(b["moltype"], [])
      for _ in range(b["n"] or 0):
        for mass in masses:
          out[g] = mass
          g += 1
    return out
  except (OSError, ValueError):
    return {}


def mass_of(atomnum, atomname):
  return TPR_MASS.get(atomnum) or ATOM_MASS[atomname]

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


def abort(reason, message):
  """Stop with a marker job_fe.run understands, not with a bare traceback.

  Every failure here leaves the caller holding an empty numpertres.gs and five
  .mdp files whose pull block was never appended, which grompp only notices once
  the cycles are already running. Write the marker first."""
  sys.stderr.write("make_boresch: ERROR - %s\n" % message)
  with open("boresch_failed.gs", "w") as f:
    f.write(reason + "\n")
  print("0")
  sys.exit(1)


# A missing input file used to fall through this block silently, leaving every
# downstream selection empty instead of failing.
if not os.path.isfile(args.input):
  abort("INPUT_STRUCTURE_MISSING",
        "coordinate file %s does not exist" % args.input)

with open(args.input, "r") as f:
  for line in f:
    if not line.strip().startswith("#"):
      left = line[:15]
      right = line[15:]
      tmp = left.split() + right.split()
      try:
        s = re.search(r"\d+(\.\d+)?", tmp[0])
        resnum = int(s.group(0))
        atomname = tmp[1]
        atomnum = int(tmp[2])
        x, y, z = float(tmp[3]), float(tmp[4]), float(tmp[5])
        res3 = re.sub(r'\d+', '', tmp[0])
        if res3 == "SOL" or resnum > max_structural_resnum:
          continue
        rec = (tmp[0], resnum, atomname, atomnum, x, y, z)
        if resnum not in residues_b:
          prot1_data.append(rec)
        else:
          prot2_data.append(rec)
      except (ValueError, IndexError, AttributeError):
        pass

# Load before anything takes a COM. The probe tpr already exists when job_fe.run
# calls this script, and it is the same system, already repartitioned.
TPR_MASS = load_tpr_masses(args.tpr)
if TPR_MASS:
  sys.stderr.write("make_boresch: masses read from %s (%d atoms)\n"
                   % (args.tpr, len(TPR_MASS)))
else:
  sys.stderr.write("make_boresch: WARNING - no masses from %s, falling back to "
                   "element masses; COMs will not match GROMACS exactly\n"
                   % args.tpr)


def box_vector_norms(box_line):
  """Lengths of the three box vectors from a .gro box line.

  The line is v1x v2y v3z v1y v1z v2x v2z v3x v3y, so the first three fields are
  the DIAGONAL, not the vector lengths. On a rhombic dodecahedron editconf writes
  (d,0,0), (0,d,0), (d/2, d/2, d/sqrt(2)): all three vectors have norm d, while
  the z diagonal is d/sqrt(2), 29% smaller.

  GROMACS enforces 0.49 * min |v_i| over the dims a pull coordinate enables, the
  vector NORM. Verified by grompp rather than from the source: on the 2KTF box
  the abort reads "larger than 0.49 times the box size (4.045396)", which is
  0.49 * 8.25591 exactly, and a hand-built triclinic box 6 6 3 0 0 0 0 3 3
  (norms 6, 6, sqrt(27), min diagonal 3) aborts at 2.546115 = 0.49 * sqrt(27),
  where a min-diagonal rule would have predicted 1.47. The check fires at grompp,
  not only at mdrun, and applies at k = 0.
  """
  f = [float(v) for v in box_line.split()]
  f += [0.0] * (9 - len(f))
  v = [(f[0], f[3], f[4]), (f[5], f[1], f[6]), (f[7], f[8], f[2])]
  return [math.sqrt(sum(c * c for c in u)) for u in v]

# The BOX comes from --box-from, NOT from -f, because the reference structure and
# the box the legs run in need not be the same system. -f supplies geometry; the
# box supplies a limit that four separate gates scale with:
#
#   arm_max      = 0.35 * min(BOX_VEC)   candidate anchor arms
#   cross-triad rejection, relaxed_cap and the BORESCH_R_TOO_LARGE abort, all
#   at R_LIMIT_FRAC * PULL_LIMIT
#
# Taking those from the reference file was harmless while the reference came from
# the production box. It stopped being harmless when the reference moved to the
# probe run, whose box is ~19% smaller: on 2KTF the limit falls 4.045 -> 3.29 nm,
# which tightens arm_max 2.89 -> 2.33, selects DIFFERENT anchors, and so silently
# changes ref_r, the five angles and dG_release. The abort is worse than silent:
# 2KTF's r0_release of 1.882 nm sits 22% under the production threshold and 4.7%
# under the probe one, so a slightly longer complex would fail setup for a box
# constraint the production run does not have.
_box_src = args.box_from or args.input
BOX_VEC = []
try:
  _l = [x for x in open(_box_src).read().split("\n") if x.strip()]
  BOX_VEC = box_vector_norms(_l[-1])
except (ValueError, IndexError, OSError):
  BOX_VEC = []
if not BOX_VEC:
  sys.stderr.write("make_boresch: WARNING - no box read from %s; the pull-limit "
                   "gates are disabled and anchor arms fall back to ARM_MAX\n"
                   % _box_src)

# The distance every pull coordinate is measured against. 4.045 nm on 2KTF.
PULL_LIMIT = 0.49 * min(BOX_VEC) if BOX_VEC else None

# Fraction of that limit the Boresch distance is allowed to reach at full
# extension. Named once because THREE places need the same number and two of them
# used to disagree: the cross-triad search filtered at 0.9 while the final guard
# aborted at 0.6, so a triad in between passed selection and then killed the run.
# 0.6 leaves room for the interface pairs, which travel further than the Boresch
# r does -- test4 measured a largest checked pair of 2.995 nm against a 4.101 nm
# limit, 73%, while its own r0_release sat at 54%.
R_LIMIT_FRAC = 0.6

# The file can be present and still yield nothing, e.g. an empty chain_map.gs
# leaves max_structural_resnum at 0 and filters every atom out. That produced
# the same degenerate arrays downstream as a missing file.
if not prot1_data or not prot2_data:
  abort("EMPTY_STRUCTURE",
        "%s gave %d receptor and %d ligand atoms (check chain_map.gs)"
        % (args.input, len(prot1_data), len(prot2_data)))

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

#------------------------------------------------------
# Cap the restraints per residue-residue contact.
#
# The 0.6 nm rule restrains every cross-partner heavy-atom pair, which on 2KTF is
# 461 springs describing only 47 residue-residue contacts among 34 residues:
# 9.8 per contact on average and up to 31 on one, with LEU9 alone carrying 103.
# That is the measured redundancy behind "574 restraints behave like ~9 effective
# independent modes".
#
# It is not free. At lambda = 1 the interface force constant is zero, so the
# restraints hold nothing but themselves, equipartition gives <S> = N*RT/k, and
# substituting into dH/dlambda = -0.5*k*S the FORCE CONSTANT CANCELS:
#
#     dH/dlambda(lambda = 1) = -N*RT/2
#
# Every restrained degree of freedom costs RT/2 no matter how stiff it is.
# Predicted -594.1 kJ/mol at N = 461; test4 measured -602 +- 238. That term
# largely cancels between dG_intro and dG_unbind, but its FLUCTUATION does not,
# so redundant springs buy noise and nothing else.
#
# Capping per CONTACT rather than thinning globally is what keeps the interface
# intact: every contact and every residue survives any cap >= 1, so the total
# restoring force (fixed at 25000 kJ/mol/nm^2 by k = 25000/N, whatever N is) stays
# spread over the same footprint. A global distance cutoff or a random subset
# would instead delete whole contacts.
#
# Within a contact the kept springs are chosen to SPAN it rather than cluster:
# the shortest pair first, then greedily the candidate furthest (by midpoint)
# from everything already kept, with a bonus for involving a side chain. Both
# matter. Several springs on one contact attached at different atoms restrain the
# relative ORIENTATION of the two residues, not just their separation, which is
# why the cap is 4 and not 1. And 84.6% of the uncapped restraints involve a side
# chain, so a rule that quietly kept backbone pairs would discard exactly the
# packing information the restraints exist to hold.
#
# OFF SINCE test6 MEASURED IT. The reasoning above is sound about redundancy and
# wrong about what redundancy costs. test5 and test6 differ in this setting and
# almost nothing else, and the bound-state leg reads the difference cleanly since
# no Boresch coordinate appears in it:
#
#     restraints          664 (k = 37.7)      188 (k = 133.0)
#     round-trip diss     129.6 +- 11.2       190.9 +- 10.4     4.0 sigma WORSE
#     work overlap        47 of 94            0                 BAR returns nothing
#     rebinding RMSD      2.31 A              2.81 A
#     cycles over 3 A     4 of 47             21 of 49
#
# The N*RT/2 argument holds only when each spring's reference is where its pair
# actually sits. It is not: there is a standing 2.7 A mismatch between the probe
# snapshot and each production replica (see the trajectory-mean references below),
# and a spring 3.5x stiffer turns the same mismatch into 3.5x the work and 3.5x
# the spread. Cutting N at fixed sum-k therefore buys noise, which is the opposite
# of what this comment predicted.
#
# The selection code is kept, tested and one assignment away from returning, since
# a cap may well be right once the references are in the right place. It is not
# right today.
MAX_PER_CONTACT = None
BB_ATOMS = {"N", "CA", "C", "O", "OT", "OT1", "OT2", "OXT"}

def _cap_per_contact(cands, cap):
  """Keep at most `cap` restraints per residue-residue contact, spanning it."""
  by_contact = {}
  for rec in cands:
    i, j, dist = rec
    key = (prot1_data[i][1], prot2_data[j][1])       # (resnum_A, resnum_B)
    by_contact.setdefault(key, []).append(rec)

  def midpoint(rec):
    i, j, _ = rec
    a = np.array(prot1_data[i][4:7], float)
    b = np.array(prot2_data[j][4:7], float)
    return 0.5 * (a + b)

  def has_sidechain(rec):
    i, j, _ = rec
    return (prot1_data[i][2] not in BB_ATOMS) or (prot2_data[j][2] not in BB_ATOMS)

  kept = []
  for key in sorted(by_contact):
    group = sorted(by_contact[key], key=lambda r: r[2])   # shortest first
    if len(group) <= cap:
      kept.extend(group)
      continue
    chosen = [group[0]]
    mids = [midpoint(group[0])]
    while len(chosen) < cap:
      best, best_score = None, None
      for rec in group:
        if rec in chosen:
          continue
        m = midpoint(rec)
        spread = min(float(np.linalg.norm(m - c)) for c in mids)
        # The bonus is a spatial length, so it competes with spread on its own
        # scale: 0.05 nm is enough to break a tie between a backbone pair and a
        # side-chain pair, and never enough to override a genuinely better spread.
        score = spread + (0.05 if has_sidechain(rec) else 0.0)
        if best_score is None or score > best_score:
          best, best_score = rec, score
      chosen.append(best)
      mids.append(midpoint(best))
    kept.extend(chosen)
  return kept

_n_uncapped = len(interdis)
_n_contacts = len({(prot1_data[i][1], prot2_data[j][1]) for i, j, _ in interdis})
if MAX_PER_CONTACT and _n_uncapped:
  interdis = _cap_per_contact(interdis, MAX_PER_CONTACT)
  _sc = sum(1 for i, j, _ in interdis
            if prot1_data[i][2] not in BB_ATOMS or prot2_data[j][2] not in BB_ATOMS)
  sys.stderr.write("make_boresch: interface restraints %d -> %d, cap %d per "
                   "residue-residue contact (%d contacts, all kept); %.1f%% still "
                   "involve a side chain; k rises %.1f -> %.1f kJ/mol/nm^2 so the "
                   "total stays 25000\n"
                   % (_n_uncapped, len(interdis), MAX_PER_CONTACT, _n_contacts,
                      100.0 * _sc / max(len(interdis), 1),
                      25000.0 / max(_n_uncapped, 1), 25000.0 / max(len(interdis), 1)))

numinterdis = len(interdis)

# WHICH pairs are restrained is settled here; WHERE each spring sits is not. The
# reference distances above come from one frame, and reference_on_ensemble()
# further down replaces them with the mean over the probe trajectory before any
# pull block is built. interface_contacts.gs is written there too, so the file the
# QC reads records the references actually used rather than the snapshot ones.

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
      masses = np.array([mass_of(atoms[a][0], a) for a in BACKBONE])
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
  """Dihedral a-b-c-d in (-180, 180] degrees, IUPAC sign.

  The sign MUST match what GROMACS pull's `dihedral` geometry reports, since the
  value written to pull-coordN-init is compared against it every step. IUPAC is
  sign(phi) = sign(b1 . (b2 x b3)).

  Note the negation. With m1 = n1 x b2_hat, (n1 x b2h).n2 = -(n1 x n2).b2h, so
  atan2(y, x) is the exact mirror of the IUPAC angle. Without the minus sign all
  three Boresch phi references were written as mirror-image targets: verified
  against a zero-step grompp probe on the reference structure, where r, theta_A
  and theta_B agreed to six significant figures while phi_A/B/C came back
  sign-flipped (+164.879 vs -164.879, -88.6506 vs +88.6492, -94.3341 vs
  +94.3331). That put 3145 kJ/mol into dH/dlambda at t = 0 and drove theta_B
  into the pull-frame singularity, where the 1/sin(theta_B) lever tore the
  anchor apart. write_pull_block now reads its own output back to catch this
  class of error at setup instead of 10 ns into a leg.
  """
  b1, b2, b3 = b - a, c - b, d - c
  n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
  m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
  x = np.dot(n1, n2)
  y = np.dot(m1, n2)
  return -math.degrees(math.atan2(y, x))

def tri_area(p, q, r):
  return 0.5 * np.linalg.norm(np.cross(q - p, r - p))

#======================================================
# PART 3b - Anchor groups chosen from MEASURED backbone rigidity
#
# The Boresch coordinates fix the relative placement of two anchor triads, but
# say nothing about the shape of either triad: those edges are intramolecular and
# held only by the elastic network. Single-residue N/CA/C centroids turned out to
# move 0.11-0.35 nm during a hold, which on 1.0-1.3 nm arms tilts each frame by
# 10-17 degrees and let the proteins rotate 22-52 degrees relative to one another
# with every Boresch coordinate satisfied.
#
# A group's COM error eps translates into a frame orientation error of about
# eps/L for an arm of length L, so the fix has two halves: quieter COMs and
# longer arms. Averaging more atoms only suppresses UNCORRELATED motion (as
# 1/sqrt(N)); a flexible loop moves collectively and averages to nothing. So
# groups are selected on their measured COM RMSF rather than on their size, and
# the arm cap follows the box instead of a fixed 1.2 nm.
#======================================================

TRAJ_SKIP_PS = 1000.0   # discard as equilibration before measuring
R_GROUP = 0.70          # nm, radius of a candidate group about its seed CA
N_MIN_ATOMS = 18        # backbone-only, so 18 atoms is 6 residues
# Escalating ceilings on a group's COM RMSF, in nm, tried IN ORDER, first success
# wins. eq.32 assumes the anchor points sit in rigid bodies, so this is a physical
# requirement and not a tuning knob: the tightest ceiling that yields an anchor
# set is the right one, and each later entry is a concession to a structure that
# cannot supply groups that quiet. The burial heuristic is only reached when every
# entry fails, since burial carries no fluctuation information at all.
#
# Was (0.045, 0.060, 0.080). Two reasons for the finer, longer ladder:
#
#   GRANULARITY. Because the first success wins, the step size IS the precision
#   with which the accepted rigidity is chosen. On 2KTF's protein B the old
#   ladder jumped 0.060 -> 0.080 and took the loosest rung available; anything
#   that would have worked at 0.065 or 0.075 was never tried, and the anchors
#   accepted were floppier than the structure required.
#
#   HEADROOM. The rung after the last one is the burial heuristic, which carries
#   no fluctuation information at all and historically produced 22-52 degrees of
#   relative frame rotation against the 8 degree limit. Three more rungs before
#   that cliff is cheap: a round takes seconds, and ROT_MAX_DEG still decides
#   acceptance, so a loose ceiling that passes the rotation check has earned it
#   while one that does not is rejected regardless of which rung found it.
EPS_LADDER = (0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.110)
ARM_MIN = 0.60          # nm, shortest useful lever arm
MIN_SIN = 0.35          # sin of the angle between a triad's two edges
ROT_MAX_DEG = 8.0       # acceptance: relative frame rotation over the trajectory
TOP_PER_SIDE = 30       # triads kept per protein before the cross-protein search
RMSF_SPLIT = 1.0        # nm, per-atom RMSF above which a side is PBC-split
MIN_FRAMES = 20         # frames needed before the measurement means anything


def kabsch(P, Q):
  """Rotation and translation taking P onto Q."""
  pc, qc = P.mean(0), Q.mean(0)
  U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
  d = np.sign(np.linalg.det(Vt.T @ U.T))
  R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
  return R, qc - R @ pc


def read_multi_gro(path, natoms, with_box=False):
  """(nframes, natoms, 3) from a multi-frame .gro, or None.

  with_box also returns (nframes, 3, 3) box matrices, rows being the box vectors.
  Cross-protein distances need them: the trajectory is only -pbc whole, so the two
  partners can sit in different periodic images and a raw subtraction is then
  wrong by a box vector. Intra-protein callers do not, hence the default."""
  if natoms <= 0:
    return (None, None) if with_box else None
  L = open(path).read().split("\n")
  out, box, i = [], [], 0
  while i < len(L) - 1 and L[i].strip():
    n = int(L[i + 1])
    if n != natoms:
      return (None, None) if with_box else None
    if i + 2 + n > len(L):     # truncated write: keep the frames we have
      break
    out.append([[float(L[i + 2 + k][20:28]), float(L[i + 2 + k][28:36]),
                 float(L[i + 2 + k][36:44])] for k in range(n)])
    b = [float(v) for v in L[i + 2 + n].split()] + [0.0] * 9
    #                 v1              v2              v3
    box.append([[b[0], b[3], b[4]], [b[5], b[1], b[6]], [b[7], b[8], b[2]]])
    i += n + 3
  if not out:
    return (None, None) if with_box else None
  arr = np.asarray(out, dtype=float).reshape(len(out), natoms, 3)
  if not with_box:
    return arr
  return arr, np.asarray(box, dtype=float)


# The 27 translations of a triclinic cell, built once. Brute force over them is
# exact for any box shape and costs nothing at these sizes; the alternative,
# rounding by the inverse box, is only correct for cells that are nearly
# rectangular and fails quietly on the dodecahedra this pipeline usually builds.
_IMG = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                 for k in (-1, 0, 1)], dtype=float)


def min_image_dist(dv, M):
  """Shortest distance equivalent to displacement dv under box M.

  dv is (..., 3) and M is (3, 3) or (..., 3, 3), so a whole trajectory of pair
  vectors goes through in one call."""
  shifts = np.einsum("ij,...jk->...ik", _IMG, M)        # (..., 27, 3)
  cand = dv[..., None, :] + shifts
  return np.sqrt((cand ** 2).sum(-1)).min(-1)


def load_backbone_trajectory(atom_numbers):
  """Backbone coordinates over time, or None if the trajectory is unavailable.

  trjconv renumbers a subset on output, so the index is written here and the
  row order is known by construction.

  -pbc whole, not cluster: trjconv refuses to cluster a backbone-only index
  ("Molecule N marked for clustering but not atom 1 in it"). whole is what the
  per-protein fit below needs anyway, since that fit removes rigid-body motion
  and so does not care where the other protein sits. The cross-protein reference
  geometry does care, and is taken from args.input instead, which job_fe.run has
  already run through -pbc cluster."""
  if not atom_numbers:
    return None
  if not (os.path.isfile(args.traj) and os.path.isfile(args.tpr)):
    return None
  ndx, gro = ".bb_sel.ndx", ".bb_traj.gro"
  try:
    with open(ndx, "w") as f:
      f.write("[ bbsel ]\n" + " ".join(str(a) for a in atom_numbers) + "\n")
    cmd = ["gmx", "trjconv", "-s", args.tpr, "-f", args.traj, "-n", ndx,
           "-o", gro, "-b", "%g" % TRAJ_SKIP_PS, "-pbc", "whole"]
    r = subprocess.run(cmd, input="bbsel\n", capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(gro):
      return None
    return read_multi_gro(gro, len(atom_numbers))
  except (OSError, ValueError, IndexError):
    return None
  finally:
    for f in (ndx, gro):
      if os.path.isfile(f):
        os.remove(f)


def reference_on_ensemble(pairs):
  """Re-reference each interface spring to its MEAN distance over the probe run.

  A harmonic restraint costs <(d - r0)^2>, which is minimised at r0 = <d> and by
  nothing else, so the ensemble mean is not a heuristic here: it is the choice
  that makes the switch cheapest, for any spring constant.

  What it was before is one frame. npt_probe.gro is the last step of the 10 ns
  probe, and every production replica then equilibrates 10 ns from emin_vac under
  its own seed and is measured against that single snapshot. Reading dH/dlambda at
  the first frame of boundfwd gives 0.5*k*sum(d - r0)^2 exactly, and on test5 it
  came to 1082 +- 954 kJ/mol: a 2.72 +- 1.13 A mismatch per pair, ranging 1.16 to
  6.02 A across cycles, explaining 51% of the W_intro variance on a leg that
  carries 80% of the total. Splitting the residual strain three ways puts 40% in a
  systematic offset shared by every replica, and that is the part this removes.

  Distances are minimum-imaged. trjconv is asked for -pbc whole, which makes each
  molecule whole but leaves the two partners free to sit in different images;
  clustering a cross-protein subset is what the backbone reader above had to avoid,
  and at a 0.6 nm contact against a box of several nm the minimum image is not an
  approximation, it is the same number.

  Returns (pairs, note) with the distances replaced, or the input untouched and a
  note saying why, since a missing trajectory must degrade to the old behaviour
  rather than abort a setup."""
  if not pairs:
    return pairs, "no interface pairs"
  anums = sorted({prot1_data[i][3] for i, _, _ in pairs} |
                 {prot2_data[j][3] for _, j, _ in pairs})
  if not (os.path.isfile(args.traj) and os.path.isfile(args.tpr)):
    return pairs, "no %s/%s; keeping the snapshot references" % (args.traj, args.tpr)
  ndx, gro = ".if_sel.ndx", ".if_traj.gro"
  try:
    with open(ndx, "w") as f:
      f.write("[ ifsel ]\n" + " ".join(str(a) for a in anums) + "\n")
    r = subprocess.run(["gmx", "trjconv", "-s", args.tpr, "-f", args.traj,
                        "-n", ndx, "-o", gro, "-b", "%g" % TRAJ_SKIP_PS,
                        "-pbc", "whole"],
                       input="ifsel\n", capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(gro):
      return pairs, "trjconv failed; keeping the snapshot references"
    X, M = read_multi_gro(gro, len(anums), with_box=True)
  except (OSError, ValueError, IndexError) as e:
    return pairs, "%s; keeping the snapshot references" % e
  finally:
    for f in (ndx, gro):
      if os.path.isfile(f):
        os.remove(f)
  if X is None or len(X) < MIN_FRAMES:
    return pairs, ("only %s frames, need %d; keeping the snapshot references"
                   % (0 if X is None else len(X), MIN_FRAMES))
  row = {a: k for k, a in enumerate(anums)}
  ia = np.array([row[prot1_data[i][3]] for i, _, _ in pairs])
  ib = np.array([row[prot2_data[j][3]] for _, j, _ in pairs])
  d = min_image_dist(X[:, ia, :] - X[:, ib, :], M[:, None, :, :])   # (frames, pairs)
  mean, sd = d.mean(0), d.std(0)
  snap = np.array([p[2] for p in pairs])
  shift = mean - snap
  sys.stderr.write(
      "make_boresch: interface references re-measured on %d frames of %s -- moved "
      "%+.3f A on average, rms %.3f A, worst %.3f A. Residual spread about the new "
      "reference is %.3f A rms, which is the part no choice of reference can "
      "remove. Removing the offset takes %.0f kJ/mol of systematic strain out of "
      "the bound leg.\n"
      % (len(X), args.traj, 10.0 * shift.mean(),
         10.0 * float(np.sqrt((shift ** 2).mean())), 10.0 * float(np.abs(shift).max()),
         10.0 * float(sd.mean()),
         0.5 * (25000.0 / len(pairs)) * float((shift ** 2).sum())))
  return ([(i, j, float(m)) for (i, j, _), m in zip(pairs, mean)],
          "mean over %d frames" % len(X))


def measure_rigidity(prot_data, traj, row_of):
  """Per-protein fitted trajectory and per-atom RMSF, both over backbone atoms."""
  anums = [rec[3] for rec in prot_data if rec[2] in BACKBONE and rec[3] in row_of]
  rows = [row_of[a] for a in anums]
  X = traj[:, rows, :]
  ref = X[0]
  fit = np.empty_like(X)
  for k in range(len(X)):
    R, t = kabsch(X[k], ref)
    fit[k] = X[k] @ R.T + t
  rmsf = np.sqrt(((fit - fit.mean(0)) ** 2).sum(-1).mean(0))
  return anums, fit, dict(zip(anums, rmsf))


def residue_index(prot_data, anums):
  """Per-residue backbone lookup shared by the group builders."""
  idx_of = {a: i for i, a in enumerate(anums)}
  xyz_of = {rec[3]: np.array(rec[4:7], dtype=float) for rec in prot_data}
  by_res, ca_of = {}, {}
  for (rn3, resnum, atomname, atomnum, x, y, z) in prot_data:
    if atomname in BACKBONE and atomnum in idx_of:
      by_res.setdefault(resnum, {})[atomname] = atomnum
      if atomname == "CA":
        ca_of[resnum] = np.array([x, y, z])
  full = [r for r in by_res if all(a in by_res[r] for a in BACKBONE)]
  return by_res, ca_of, idx_of, xyz_of, full


def make_group(residues, by_res, idx_of, xyz_of, fit):
  """A pull group from a residue list, with its COM, COM time series and RMSF.

  The reference COM comes from args.input via xyz_of, not from com_t[0]. The
  trajectory is only -pbc whole, so the two proteins can sit in different periodic
  images in any given frame, and every cross-protein quantity built from it (r,
  the two angles, the three dihedrals, dG_release) would then be wrong.
  args.input is clustered, and is also the snapshot the interface restraints and
  the elastic network use, so the whole restraint set refers to one structure.
  eps and relative_rotation stay on com_t: both are insensitive to imaging."""
  atoms = [by_res[r][a] for r in residues for a in BACKBONE]
  if len(atoms) < N_MIN_ATOMS:
    return None
  cols = [idx_of[a] for a in atoms]
  w = np.array([mass_of(by_res[r][a], a) for r in residues for a in BACKBONE])
  com_t = (fit[:, cols, :] * w[None, :, None]).sum(1) / w.sum()
  eps = float(np.sqrt(((com_t - com_t.mean(0)) ** 2).sum(-1).mean()))
  xyz = np.array([xyz_of[a] for a in atoms])
  return {"atoms": atoms, "com": (xyz * w[:, None]).sum(0) / w.sum(),
          "com_t": com_t, "eps": eps, "nres": len(residues),
          "residues": list(residues)}


def build_rigid_groups(prot_data, anums, fit, eps_max, label=""):
  """Candidate groups, keyed by seed residue, filtered on measured COM RMSF.

  These OVERLAP: a residue within R_GROUP of two seeds belongs to both. They are
  candidates for ranking only; the triad that wins is rebuilt disjointly by
  partition_triad before anything is written."""
  by_res, ca_of, idx_of, xyz_of, full = residue_index(prot_data, anums)
  groups = {}
  n_small, all_eps = 0, []
  for seed in full:
    near = [r for r in full if np.linalg.norm(ca_of[r] - ca_of[seed]) <= R_GROUP]
    g = make_group(near, by_res, idx_of, xyz_of, fit)
    if g is None:
      n_small += 1
      continue
    all_eps.append(g["eps"])
    if g["eps"] > eps_max:
      continue
    groups[seed] = g
  # thin out overlapping seeds: keep the quietest of any cluster of nearby COMs
  keep = {}
  for s in sorted(groups, key=lambda s: groups[s]["eps"]):
    c = groups[s]["com"]
    if all(np.linalg.norm(c - groups[k]["com"]) > 0.5 for k in keep):
      keep[s] = groups[s]
  # Say which filter did the cutting. Reaching fewer than 3 groups sends the run
  # back to the burial heuristic, and without this the reason was invisible.
  sys.stderr.write("make_boresch: %s: %d residues, %d seeds too small, "
                   "%d measured (eps median %.3f, min %.3f nm), %d under "
                   "eps_max %.3f, %d after 0.5 nm thinning\n"
                   % (label, len(full), n_small, len(all_eps),
                      float(np.median(all_eps)) if all_eps else float("nan"),
                      min(all_eps) if all_eps else float("nan"),
                      len(groups), eps_max, len(keep)))
  return keep


def partition_triad(seeds, prot_data, anums, fit, eps_max):
  """Rebuild a triad so that no atom belongs to two of its groups.

  The candidate groups overlap heavily: on 2KTF the selected L2 and L3 shared 12
  of 24 atoms and P2/P3 shared 18 of 33, because the 0.5 nm thinning is on COM
  separation and says nothing about membership. Shared atoms pull the two COMs
  toward each other, which shortens the lever arm, and the frame error goes as
  eps/L, so the overlap costs accuracy twice: a shorter L and a COM that moves
  with its neighbour.

  Each residue in the union of the three neighbourhoods is assigned to its NEAREST
  seed, which is the even split the geometry allows and which pushes the three
  COMs apart rather than pulling them together. Returns None if the split starves
  a group below N_MIN_ATOMS or pushes one over eps_max, in which case the caller
  moves on to the next-ranked triad."""
  by_res, ca_of, idx_of, xyz_of, full = residue_index(prot_data, anums)
  owned = {s: [] for s in seeds}
  for r in full:
    d = [(np.linalg.norm(ca_of[r] - ca_of[s]), s) for s in seeds]
    dist, s = min(d)
    if dist <= R_GROUP:
      owned[s].append(r)
  out = {}
  for s in seeds:
    g = make_group(sorted(owned[s]), by_res, idx_of, xyz_of, fit)
    if g is None or g["eps"] > eps_max:
      return None
    out[s] = g
  # Disjoint by construction, but assert it: a silent overlap here would be a
  # correlated pair of Boresch coordinates that eq.32 does not model.
  seen = set()
  for s in seeds:
    a = set(out[s]["atoms"])
    if a & seen:
      return None
    seen |= a
  return out


def rank_triads(groups, arm_max):
  """Best triads for one protein: quiet COMs, long arms, well-conditioned frame."""
  keys = list(groups)
  out = []
  for a, b, c in itertools.combinations(keys, 3):
    ca, cb, cc = (groups[k]["com"] for k in (a, b, c))
    e1, e2, e3 = cb - ca, cc - ca, cc - cb
    arms = [np.linalg.norm(v) for v in (e1, e2, e3)]
    if min(arms) < ARM_MIN or max(arms) > arm_max:
      continue
    sin = np.linalg.norm(np.cross(e1, e2)) / (arms[0] * arms[1])
    if sin < MIN_SIN:
      continue
    eps = max(groups[k]["eps"] for k in (a, b, c))
    # eps/L is the frame error; reward long arms and penalise noisy COMs
    out.append((eps / min(arms) / sin, (a, b, c)))
  out.sort()
  return [t for _, t in out[:TOP_PER_SIDE]]


def frame_of(c1, c2, c3):
  e1 = c2 - c1
  e1 = e1 / np.linalg.norm(e1)
  t = c3 - c1
  e2 = t - np.dot(t, e1) * e1
  e2 = e2 / np.linalg.norm(e2)
  return np.stack([e1, e2, np.cross(e1, e2)], axis=1)


def relative_rotation(rec_g, lig_g, P, L):
  """RMS relative frame rotation over the trajectory, in degrees."""
  A = [frame_of(*[rec_g[k]["com_t"][f] for k in P]) for f in range(len(rec_g[P[0]]["com_t"]))]
  B = [frame_of(*[lig_g[k]["com_t"][f] for k in L]) for f in range(len(lig_g[L[0]]["com_t"]))]
  rel = [a.T @ b for a, b in zip(A, B)]
  ang = [math.degrees(math.acos(np.clip((np.trace(rel[0].T @ R) - 1) / 2.0, -1, 1)))
         for R in rel]
  return float(np.sqrt(np.mean(np.square(ang))))


def try_measured_anchors():
  """Anchors chosen from measured rigidity, or None to fall back to burial.

  Returns (rec_groups, lig_groups, anchors, frame_rotation_deg)."""
  bb = [rec[3] for rec in prot1_data + prot2_data if rec[2] in BACKBONE]
  traj = load_backbone_trajectory(bb)
  # ndim and the atom axis both matter, not just the frame count: an empty
  # selection gives trjconv a legal zero-element group, which it happily writes
  # as N frames of 0 atoms. That shape passed a frame-count-only check and then
  # crashed the indexing below instead of falling back.
  if (traj is None or traj.ndim != 3 or traj.shape[1] == 0
      or traj.shape[0] < MIN_FRAMES):
    sys.stderr.write("make_boresch: no usable equilibration trajectory "
                     "(%s, shape %s), falling back to the burial heuristic\n"
                     % (args.traj, None if traj is None else traj.shape))
    return None
  row_of = {a: i for i, a in enumerate(bb)}
  a1, fit1, r1 = measure_rigidity(prot1_data, traj, row_of)
  a2, fit2, r2 = measure_rigidity(prot2_data, traj, row_of)
  # A side that spans several chains can come out of -pbc whole with its chains
  # in different images. The self-fit is then meaningless in every frame, and it
  # shows up as backbone RMSF of order the box rather than of order 0.1 nm.
  worst = max([max(r1.values()) if r1 else 0.0,
               max(r2.values()) if r2 else 0.0])
  if worst > RMSF_SPLIT:
    sys.stderr.write("make_boresch: backbone RMSF reaches %.2f nm, so a protein "
                     "is split across periodic images in the trajectory; "
                     "falling back to the burial heuristic\n" % worst)
    return None
  arm_max = 0.35 * min(BOX_VEC) if BOX_VEC else 1.2

  def ang(a, b, c):
    v1, v2 = a - b, c - b
    return math.degrees(math.acos(np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))

  def disjoint_triads(T, prot_data, anums, fit, eps_max, label):
    """Rebuild each ranked triad with its atoms uniquely owned, then re-check.

    The candidate groups overlap, so the arms and the conditioning that
    rank_triads measured are not the ones the restraint will actually see. Splits
    move the COMs apart, which can only lengthen the arms, but it also changes eps
    and can starve a group, so everything is re-tested on the split geometry."""
    out = []
    for t in T:
      d = partition_triad(t, prot_data, anums, fit, eps_max)
      if d is None:
        continue
      c = [d[k]["com"] for k in t]
      e = [float(np.linalg.norm(c[1] - c[0])), float(np.linalg.norm(c[2] - c[0])),
           float(np.linalg.norm(c[2] - c[1]))]
      if min(e) < ARM_MIN or max(e) > arm_max:
        continue
      sin = float(np.linalg.norm(np.cross(c[1] - c[0], c[2] - c[0])) / (e[0] * e[1]))
      if sin < MIN_SIN:
        continue
      out.append((tuple(t), d))
    sys.stderr.write("make_boresch: %s: %d ranked triads, %d survive the "
                     "disjoint split\n" % (label, len(T), len(out)))
    return out

  def search(eps_max):
    """One round at a given rigidity threshold. (rot, anchors, g1, g2) or None."""
    g1 = build_rigid_groups(prot1_data, a1, fit1, eps_max, "receptor")
    g2 = build_rigid_groups(prot2_data, a2, fit2, eps_max, "ligand")
    if len(g1) < 3 or len(g2) < 3:
      sys.stderr.write("make_boresch: too few rigid groups at eps_max %.3f\n"
                       % eps_max)
      return None
    T1, T2 = rank_triads(g1, arm_max), rank_triads(g2, arm_max)
    if not T1 or not T2:
      sys.stderr.write("make_boresch: no triad satisfied the arm/conditioning "
                       "limits (arm_max %.2f nm) at eps_max %.3f\n"
                       % (arm_max, eps_max))
      return None
    D1 = disjoint_triads(T1, prot1_data, a1, fit1, eps_max, "receptor")
    D2 = disjoint_triads(T2, prot2_data, a2, fit2, eps_max, "ligand")
    if not D1 or not D2:
      return None

    best = None
    for Pt, d1 in D1:
      for Lt, d2 in D2:
        # P3 and L1 carry r: take the closest cross pair of the two triads.
        pairs = [(np.linalg.norm(d1[p]["com"] - d2[l]["com"]), p, l)
                 for p in Pt for l in Lt]
        r_cross, P3, L1 = min(pairs)
        # ARM_MIN/arm_max bound the three INTRA-protein edges of each triad;
        # rank_triads never sees a cross pair, so without this r was whatever the
        # closest approach of two rigidity-selected triads happened to be. Over
        # 2KTF's own residues the worst valid combination gives
        # r + pull_dist = 4.408 nm against a 4.045 nm limit.
        #
        # R_LIMIT_FRAC, the SAME budget the final guard aborts on. It used to be
        # 0.9 here against 0.6 there, which left a dead band: a triad landing
        # between the two passed selection and then killed the whole setup with
        # BORESCH_R_TOO_LARGE. test5 selected r_cross + pull_dist = 2.526 nm
        # against a 2.432 threshold, 4% over, and aborted after the search had
        # already reported success. Searching under the budget the guard enforces
        # means the search either returns a usable triad or reports honestly that
        # this rigidity ceiling has none.
        if PULL_LIMIT and r_cross + args.pull_dist > R_LIMIT_FRAC * PULL_LIMIT:
          continue
        rest_p = [k for k in Pt if k != P3]
        rest_l = [k for k in Lt if k != L1]
        for P2, P1 in (rest_p, rest_p[::-1]):
          for L2, L3 in (rest_l, rest_l[::-1]):
            thA = ang(d1[P2]["com"], d1[P3]["com"], d2[L1]["com"])
            thB = ang(d1[P3]["com"], d2[L1]["com"], d2[L2]["com"])
            if not (ANG_LO <= thA <= ANG_HI and ANG_LO <= thB <= ANG_HI):
              continue
            rot = relative_rotation(d1, d2, (P1, P2, P3), (L1, L2, L3))
            if best is None or rot < best[0]:
              best = (rot, {"P1": P1, "P2": P2, "P3": P3,
                            "L1": L1, "L2": L2, "L3": L3}, d1, d2)
    if best is None:
      sys.stderr.write("make_boresch: no anchor set met the angle window at "
                       "eps_max %.3f\n" % eps_max)
    return best

  # Escalate the rigidity threshold rather than giving up on the measured path.
  # EPS_MAX is an absolute cut on a quantity that moves with the system, the force
  # field and the barostat: on 2KTF the receptor median went 0.030 -> 0.048 nm
  # between two runs of the same structure. A side whose groups all sit slightly
  # above the cut is still better served by its own quietest measured groups than
  # by the burial heuristic, which has no fluctuation information at all.
  #
  # Serial: take the FIRST round that yields an anchor set and stop.
  #
  # Do not be tempted to run every round and keep the one with the least frame
  # rotation. eq.32 derives the standard-state term for anchor points held fixed
  # in two RIGID bodies, so a group's COM RMSF is a direct measure of how well the
  # derivation applies, not merely a knob controlling how many candidates there
  # are. Relative frame rotation is a different quantity: two floppy triads that
  # happen to co-rotate score well on it while still breaking the rigid-body
  # premise underneath the analytical term. So the tightest threshold that works
  # wins, and a looser one is only ever a concession to a structure that cannot
  # supply rigid enough groups.
  #
  # It is also much cheaper. A round is seconds; running the whole ladder and
  # scoring every result cost 2m32s on 2KTF, and the extra rounds are wasted work
  # on every structure that succeeds at 0.045.
  #
  # The concrete cost of choosing this way, measured on 2KTF: 0.045 accepts a set
  # rotating 6.6 deg where 0.060 would have found 4.5 deg. Both clear ROT_MAX_DEG,
  # and the 0.045 groups are the more rigid ones, which is the trade being made
  # deliberately.
  chosen, chosen_eps = None, None
  for eps_max in EPS_LADDER:
    sys.stderr.write("make_boresch: --- anchor search at eps_max %.3f nm ---\n"
                     % eps_max)
    r = search(eps_max)
    if r is not None:
      chosen, chosen_eps = r, eps_max
      break
    sys.stderr.write("make_boresch: eps_max %.3f found nothing, relaxing\n"
                     % eps_max)
  if chosen is None:
    sys.stderr.write("make_boresch: no measured anchor set at any eps_max in %s, "
                     "falling back to the burial heuristic\n" % (EPS_LADDER,))
    return None

  rot, anch, g1, g2 = chosen
  sys.stderr.write("make_boresch: accepted the first round that succeeded, "
                   "eps_max %.3f\n" % chosen_eps)
  sys.stderr.write("make_boresch: selected anchors give %.1f deg RMS relative "
                   "frame rotation over the equilibration (limit %.1f)\n"
                   % (rot, ROT_MAX_DEG))
  if rot > ROT_MAX_DEG:
    sys.stderr.write("make_boresch: WARNING - the best anchor set still rotates "
                     "%.1f deg; the Boresch frame will not hold orientation "
                     "well and dG_release will overstate the confinement\n" % rot)
  return g1, g2, anch, rot


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
  # The docstring promises [0.5, 1.2] nm, but an empty in_range falls through to
  # the global minimum with no cap at all, and the burial filter can exclude the
  # surface residues that actually form the interface. Say so when it happens:
  # r0_release is guarded further down, but silence here made an unbounded ref_r
  # look like a deliberate choice.
  if not in_range and cand:
    sys.stderr.write("make_boresch: no pooled receptor group 0.5-1.2 nm from L1; "
                     "P3 falls back to the global minimum at %.3f nm\n"
                     % min(cand)[0])
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
    # The relaxed pass used to have NO cap, and since it ranks by tri_area, which
    # grows with distance from the line pq, it actively selected the most distant
    # non-collinear residue available. That left P1-P2 (phi_A) and L2-L3 (phi_C)
    # unbounded. Keep the retry, since removing it turns these cases into hard
    # aborts, but bound the relaxed pass by the box instead of by nothing.
    relaxed_cap = R_LIMIT_FRAC * PULL_LIMIT if PULL_LIMIT else ARM_MAX
    for cap in (ARM_MAX, relaxed_cap):
      best, best_area = None, -1.0
      for rn in pool:
        if rn in exclude:
          continue
        c = (lig_groups.get(rn) or rec_groups.get(rn))["com"]
        if np.linalg.norm(c - p) > cap or np.linalg.norm(c - q) > cap:
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

# The burial heuristic underneath only survives if the measured path can fail
# without taking the process with it. Its own return-None paths were guarded;
# an exception was not, and one killed a setup that the fallback would have
# completed.
try:
  _measured = try_measured_anchors()
except Exception:
  sys.stderr.write("make_boresch: measured anchor selection raised, falling "
                   "back to the burial heuristic\n" + traceback.format_exc())
  _measured = None
if _measured:
  rec_groups, lig_groups, anchors, FRAME_ROT = _measured
else:
  anchors, FRAME_ROT = select_boresch_anchors(), float('nan')
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

# Covers both anchor modes, including every burial fallback that relaxes its own
# cap. GROMACS aborts at grompp when any checked pair exceeds PULL_LIMIT, so
# without this the failure surfaced as a dead cycle rather than a failed setup.
# 60% leaves room for the fluctuation on top of the switched reference: 2KTF sits
# at 1.882 against a 2.427 threshold.
if PULL_LIMIT and r0_release > R_LIMIT_FRAC * PULL_LIMIT:
  abort("BORESCH_R_TOO_LARGE",
        "ref_r %.3f + pull_dist %.3f = %.3f nm exceeds 60%% of the GROMACS pull "
        "limit %.3f nm (0.49 * shortest box vector); increase the editconf "
        "padding or reselect anchors"
        % (ref_r, args.pull_dist, r0_release, PULL_LIMIT))

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
  # Recorded so the work integration always uses the rate this structure was
  # actually run with: integrate.py multiplies the time integral of the pull
  # force by the rate, so a mismatch silently rescales every pull work.
  f.write("pull_rate_nm_ps       %.8f\n" % args.pull_rate)
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

def strip_generated_groups(path):
  """Drop a_* and bor_* groups left by an earlier run before appending again.

  job_fe.run regenerates index.ndx with make_ndx immediately before calling this
  script, so in the normal flow there is nothing to strip. A standalone re-run in
  an existing directory has no such luck, and appending a second copy leaves
  grompp resolving a pull group name to whichever duplicate it saw first."""
  if not os.path.isfile(path):
    return
  out, skipping = [], False
  for line in open(path):
    m = re.match(r"\s*\[\s*(\S+)\s*\]", line)
    if m:
      skipping = m.group(1).startswith("a_") or m.group(1).startswith("bor_")
    if not skipping:
      out.append(line)
  with open(path, "w") as f:
    f.writelines(out)


def write_index_groups():
  """Append the generated groups to every index file named by --index.

  The groups are SOLUTE atom numbers, so the same definitions are valid in every
  system built from the same solute: the production legs read index.ndx while the
  readback grompps in whichever system the reference structure came from."""
  for _p in args.index:
    strip_generated_groups(_p)
  for _p in args.index:
    _write_index_groups_to(_p)


def _write_index_groups_to(path):
  with open(path, "a") as index:
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

# GROMACS needs a reference atom to make a multi-atom pull group whole across
# periodic boundaries. Left unset it uses the middle ENTRY OF THE INDEX LIST,
# which is an arbitrary atom once a group spans several residues. Pick the atom
# closest to the group's own centre instead, which is the safest reference and
# the one least likely to sit on a boundary.
PBCATOM = {}

def compute_pbcatoms():
  for role in ("P1", "P2", "P3", "L1", "L2", "L3"):
    g = rec_groups.get(anchors[role]) or lig_groups.get(anchors[role])
    xyz, anums = [], []
    for src in (prot1_data, prot2_data):
      for rec in src:
        if rec[3] in g["atoms"]:
          xyz.append((rec[4], rec[5], rec[6])); anums.append(rec[3])
    if not xyz:
      continue
    a = np.asarray(xyz)
    PBCATOM["bor_%s" % role] = int(anums[int(np.argmin(
        np.linalg.norm(a - a.mean(axis=0), axis=1)))])

write_index_groups()
compute_pbcatoms()

#======================================================
# PART 6 - Pull-block writers
#
# A "coord" is a dict with: name-list of ndx groups, geometry, dim, init, rate,
# k (state A), kB (state B). Groups are declared once and referenced by index.
#======================================================

# Both geometry corrections are applied here, before any block is built, so every
# leg that follows sees the same references.
_snapdis = [p[2] for p in interdis]
interdis, _ref_note = reference_on_ensemble(interdis)
if not _ref_note.startswith("mean over"):
  sys.stderr.write("make_boresch: interface references NOT re-measured (%s)\n"
                   % _ref_note)

# The contact list the rebinding QC checks recovery against, written now that the
# references are final. utils/interface_qc.py reads it rather than reparsing the
# pull block, and takes ref_nm from column 3; the snapshot distance is kept as a
# tenth column so the shift stays auditable after the fact.
with open("interface_contacts.gs", "w") as _f:
  _f.write("# atomA atomB ref_nm resA resnumA nameA resB resnumB nameB snap_nm\n")
  _f.write("# uncapped %d, kept %d, cap %s per residue-residue contact; "
           "ref_nm is the %s\n"
           % (_n_uncapped, numinterdis, MAX_PER_CONTACT, _ref_note))
  for (i, j, dist), _snap in zip(interdis, _snapdis):
    a, b = prot1_data[i], prot2_data[j]
    _f.write("%d %d %.6f %s %d %s %s %d %s %.6f\n"
             % (a[3], b[3], dist, a[0], a[1], a[2], b[0], b[1], b[2], _snap))

#------------------------------------------------------
# The path each interface reference travels during the pull.
#
# Every interface coordinate used to be written with the SAME rate, so every
# reference grew by u while the partner actually underwent a rigid translation of u
# along the pull axis. Those agree only for pairs parallel to the axis. On 2KTF's
# real geometry the disagreement reaches 2.51 A rms and 7.25 A on the worst pair,
# and it is a strain of 786 kJ/mol at u = 1 that no physics asked for: the springs
# spend the whole ramp demanding a configuration no rigid motion can reach. Stage C
# pays 42.8 kJ/mol of dhdl hysteresis for it (measured by decomposing dH/dlambda
# per coordinate, where the Boresch coordinates account for 0.20), and stage A's
# pull friction peaks exactly where this term is already worth tens of kJ/mol.
#
# So each reference now follows |r_i + u*n| instead of |r_i| + u, with n the pull
# axis. A GROMACS pull rate is constant within a leg, so what is written is the
# chord of that curve across the leg -- and because the ramp is already staged,
# that is one chord per stage, which brings the worst mismatch anywhere on the path
# from 2.51 A down to 0.12 A.
#
# This does NOT remove the pulling. The springs still have to lag behind their
# references to drag the partner out, and that lag is the friction the method is
# measuring. What goes away is the separate demand that the interface deform.
PULL_AXIS = (L1c - P3c) / np.linalg.norm(L1c - P3c)

_pair_vec = np.array([np.array(prot2_data[j][4:7], dtype=float)
                      - np.array(prot1_data[i][4:7], dtype=float)
                      for i, j, _ in interdis]) if interdis else np.empty((0, 3))


def pair_displacement(u):
  """How far each interface reference should have moved once the partner has
  translated u*pull_dist along the pull axis. Zero at u = 0 by construction, so it
  composes with the ensemble-mean reference rather than replacing it."""
  if not len(_pair_vec):
    return np.empty(0)
  d0 = np.linalg.norm(_pair_vec, axis=1)
  return np.linalg.norm(_pair_vec + (u * args.pull_dist) * PULL_AXIS, axis=1) - d0


k_inter = 25000.0 / numinterdis if numinterdis > 0 else 0.0

def boresch_group_ndx(role):
  return "bor_%s" % role

def leg_ps(filename):
  """Length of a leg in ps, read from the mdp it is about to be written into.

  The unbinding leg is split into stages of different length, so each stage needs
  its own pull rate. Deriving that rate from the mdp's own nsteps*dt means the
  rate and the leg length cannot drift apart: they are the same number by
  construction, rather than two constants that have to be kept in step by hand.
  Returns None if the file is unreadable, and the caller then refuses to write."""
  try:
    ns = dt = None
    for line in open(filename):
      k = line.split("=")[0].strip()
      if k == "nsteps":
        ns = int(float(line.split("=")[1].split(";")[0]))
      elif k == "dt":
        dt = float(line.split("=")[1].split(";")[0])
    return None if (ns is None or dt is None) else ns * dt
  except (OSError, ValueError, IndexError):
    return None


def build_coords(family, direction, u_from=None, u_to=None, ps=None):
  """Return (pull_groups, coords) for a leg.

  family    : 'unbind'  -> interface(moving, k->0) + EN(fixed) + Boresch(0->full)
              'bound'   -> interface(fixed, 0->full) + EN(fixed), no Boresch
  direction : 'fwd' or 'rev'. With u_from/u_to given this only selects the
              family's endpoint conventions; the geometry comes from u_from/u_to.

  u_from/u_to are the fraction of --pull-dist the reference has travelled at the
  start and end of THIS stage, so a stage covering 0.3 -> 1.0 starts its moving
  coordinates already 0.3*pull_dist out and moves the remaining 0.7 over `ps`.
  pull-coordN-start = no makes init absolute, so this offset is mandatory: without
  it stage B would jump back to the bound geometry. u_from == u_to gives rate 0,
  which is what makes a hold contribute no work.

  Omitting them reproduces the old single-stage behaviour (0 -> 1 forward,
  1 -> 0 reverse) at the constant --pull-rate.
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
  if u_from is None:                       # legacy single-stage behaviour
    u_from = 0.0 if direction == "fwd" else 1.0
    u_to = 1.0 if direction == "fwd" else 0.0
    stage_rate = sign * args.pull_rate
  else:
    # Rate follows from the distance this stage covers and the time it is given.
    stage_rate = 0.0 if (u_to == u_from or not ps) else \
                 (u_to - u_from) * args.pull_dist / ps
  off = u_from * args.pull_dist            # where this stage's references start

  if family == "unbind":
    # 1) Interface restraints FIRST (moving) so pull integrator sums them first.
    #
    # Each one follows its own chord of |r_i + u*n| across this stage rather than
    # the common scalar rate; see the note at PULL_AXIS. weight is that chord as a
    # fraction of the stage's nominal displacement, and it is what integrate.py
    # needs to turn the summed pull force back into work -- with one rate per
    # coordinate, rate_common * integral(sum F) is no longer the work, and
    # sum(rate_i * integral(F_i)) is.
    _leg_ps = ps if ps else (args.pull_dist / abs(args.pull_rate)
                             if args.pull_rate else 0.0)
    _d_from, _d_to = pair_displacement(u_from), pair_displacement(u_to)
    _nominal = (u_to - u_from) * args.pull_dist
    for n, (i, j, dist) in enumerate(interdis):
      g1 = gidx("a_%d" % prot1_data[i][3])
      g2 = gidx("a_%d" % prot2_data[j][3])
      _move = float(_d_to[n] - _d_from[n])
      coords.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                         init=dist + float(_d_from[n]),
                         rate=(_move / _leg_ps if _leg_ps else 0.0),
                         k=k_inter, kB=0.0,
                         weight=(_move / _nominal if _nominal else 1.0)))
    # 2) Boresch distance r (moving) - the coord that takes over the pulling.
    # This one IS the pull axis, so its reference does grow by the full nominal
    # amount and its weight is exactly 1: the correction above is about the
    # interface coordinates, which point every other way.
    gP3 = gidx(boresch_group_ndx("P3"))
    gL1 = gidx(boresch_group_ndx("L1"))
    coords.append(dict(geometry="distance", dim="Y Y Y", groups=[gP3, gL1],
                       init=ref_r + off, rate=stage_rate, k=0.0, kB=K_R,
                       boresch=True, weight=1.0))
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
                       init=ref_thA, rate=0.0, k=0.0, kB=K_ANG_RAD, boresch=True))
    # theta_B = angle(P3, L1, L2): vectors L1->P3 and L1->L2
    coords.append(dict(geometry="angle", dim="Y Y Y", groups=[gL1, gP3, gL1, gL2],
                       init=ref_thB, rate=0.0, k=0.0, kB=K_ANG_RAD, boresch=True))
    # phi_A = dihedral(P1, P2, P3, L1): vectors P1->P2, P2->P3, P3->L1
    coords.append(dict(geometry="dihedral", dim="Y Y Y",
                       groups=[gP1, gP2, gP2, gP3, gP3, gL1],
                       init=ref_phA, rate=0.0, k=0.0, kB=K_ANG_RAD, boresch=True))
    # phi_B = dihedral(P2, P3, L1, L2)
    coords.append(dict(geometry="dihedral", dim="Y Y Y",
                       groups=[gP2, gP3, gP3, gL1, gL1, gL2],
                       init=ref_phB, rate=0.0, k=0.0, kB=K_ANG_RAD, boresch=True))
    # phi_C = dihedral(P3, L1, L2, L3)
    coords.append(dict(geometry="dihedral", dim="Y Y Y",
                       groups=[gP3, gL1, gL1, gL2, gL2, gL3],
                       init=ref_phC, rate=0.0, k=0.0, kB=K_ANG_RAD, boresch=True))

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

def write_pull_weights(legname, coords):
  """One weight per force column integrate.py sums, for this leg.

  The pull work is sum_i rate_i * integral(F_i dt). While every rate was the same
  that factored into rate * integral(sum F), which is what integrate.py computes
  from a single -r. Per-coordinate rates break the factorisation, so the sum has to
  be weighted: w_i = rate_i / rate_nominal, and rate_nominal * sum(w_i F_i) is the
  work again. The weights are ratios of displacements, so they do not depend on
  direction -- a reverse leg retraces the same chords -- and job_fe.run keeps
  passing the rate magnitude exactly as before.

  Uniform weights reproduce the old arithmetic exactly; tests/test_pull_weights.py
  asserts that on real data rather than by inspection."""
  n = numinterdis + 1                       # what job_fe.run passes as -nr
  w = [float(c.get("weight", 1.0)) for c in coords[:n]]
  with open("pull_weights_%s.gs" % legname, "w") as f:
    f.write("# per-coordinate pull-work weight w = rate_i / rate_nominal, one per\n"
            "# summed force column, in column order. integrate.py -R reads this.\n")
    for x in w:
      f.write("%.10g\n" % x)


def write_pull_block(filename, pull_groups, coords):
  # Truncate any block a previous run left behind. This appends, and job_fe.run
  # re-copies the templates immediately before calling this script, so production
  # is safe; a standalone re-run in an existing directory is not, and stacking a
  # second block only fails later, at the cycle's grompp. Same non-idempotency as
  # the index.ndx append.
  if os.path.isfile(filename):
    kept = []
    for line in open(filename):
      if line.lstrip().startswith("pull-ngroups"):
        break
      kept.append(line)
    with open(filename, "w") as f:
      f.writelines(kept)
  with open(filename, "a") as f:
    f.write("\n")
    f.write("pull-ngroups            = %d\n" % len(pull_groups))
    f.write("pull-ncoords            = %d\n" % len(coords))
    f.write("\n")
    for gi, ndx_name in enumerate(pull_groups, start=1):
      f.write("pull-group%d-name        = %s\n" % (gi, ndx_name))
      if ndx_name in PBCATOM:
        f.write("pull-group%d-pbcatom     = %d\n" % (gi, PBCATOM[ndx_name]))
    f.write("\n")
    for ci, c in enumerate(coords, start=1):
      f.write("pull-coord%d-type        = umbrella\n" % ci)
      f.write("pull-coord%d-geometry    = %s\n" % (ci, c["geometry"]))
      f.write("pull-coord%d-dim         = %s\n" % (ci, c["dim"]))
      f.write("pull-coord%d-groups      = %s\n" % (ci, " ".join(str(g) for g in c["groups"])))
      f.write("pull-coord%d-start       = no\n" % ci)
      f.write("pull-coord%d-init        = %.8f\n" % (ci, c["init"]))
      # %.10g, not %.8f: a stage rate is span/time and need not be representable
      # in 8 decimals. 0.3 nm over 13000 ps is 2.3076923e-5, which %.8f rounds to
      # 2.308e-5 and which then overshoots the stage's own end reference by
      # 4e-5 nm. Small, but it puts a discontinuity at every stage boundary for
      # no reason, and the same rounding would reach the work through the rate
      # recorded in boresch_analytical.gs.
      f.write("pull-coord%d-rate        = %.10g\n" % (ci, c["rate"]))
      f.write("pull-coord%d-k           = %.8f\n" % (ci, c["k"]))
      f.write("pull-coord%d-kB          = %.8f\n" % (ci, c["kB"]))
      f.write("\n")

# Unbinding / rebinding ramp, in stages with an equilibrium hold at every internal
# boundary. WHICH stages exist and how long they are is not decided here: it comes
# from utils/fe_protocol.py, the single definition of the cycle. This loop only
# knows how to turn one stage description into one pull block.
#
# Each stage gets its own rate, DERIVED from its own mdp's nsteps*dt rather than
# passed in, so the mdp and the rate cannot drift apart, and its own init offset,
# because pull-coordN-start = no makes init absolute and each stage has to resume
# where the previous one stopped.
#
# The holds carry rate 0 and delta-lambda 0, so they contribute no work and the
# stage works still sum exactly to the work of the whole ramp. They exist so the
# stages either side can be estimated separately: adding their per-cycle works
# would rebuild the unstaged distribution and gain nothing. Unlike the unbound
# hold below, the internal holds KEEP their interface coordinates, because at
# lambda < 1 the force constant is still (1 - lambda) of full and they are doing
# real work holding the partly separated interface.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_protocol as P

stage_rates = {}
# Where the six Boresch coordinates ended up inside the FIRST stage's block,
# captured from the coords themselves rather than assumed. The readback below used
# to take the last six columns, which is only the right answer when the elastic
# network is empty: build_coords emits interface coords, then the Boresch
# distance, then the elastic network, then the five angles and dihedrals. A
# partner with three or more terminal-anchor residues produces a non-empty
# network, and the readback would then compare the Boresch r against an
# elastic-network pair distance and abort a setup that is in fact correct.
boresch_idx = []
first_stage_mdp = None
for _leg in P.legs():
  if _leg["kind"] not in ("stage", "hold"):
    continue
  _name, _u0, _u1 = _leg["mdp"], _leg["u_from"], _leg["u_to"]
  _ps = leg_ps(_name)
  if _ps is None:
    abort("STAGE_MDP_MISSING",
          "cannot read nsteps/dt from %s, so its pull rate is undefined" % _name)
  _dir = "fwd" if _u1 >= _u0 else "rev"
  _pg, _co = build_coords("unbind", _dir, u_from=_u0, u_to=_u1, ps=_ps)
  if _leg["kind"] == "hold" and _u0 >= args.pull_dist:
    # The HOLD at the far end drops its interface coordinates; see the long note
    # below. Every other hold keeps them. The kind test is load-bearing: the first
    # reverse stage also starts at u = pull_dist, and stripping its interface
    # coordinates would leave it pulling nothing at rate zero.
    _co = [c for c in _co if c.get("boresch")]
    for c in _co:
      c["rate"] = 0.0
  write_pull_block(_name, _pg, _co)
  # The per-coordinate weights integrate.py needs for this leg, one line per
  # summed force column, in the order the columns appear. Written per stage
  # because the chords differ per stage, and only for stages: a hold does no work,
  # writes no pull-force file and is never integrated, and the far-end hold has
  # dropped its interface coordinates entirely, so a weights file for it would
  # carry a length that disagrees with -nr and abort the day something did read it.
  if _leg["kind"] == "stage":
    write_pull_weights(_leg["name"], _co)
  if _leg["kind"] == "stage" and first_stage_mdp is None:
    first_stage_mdp = _name
    boresch_idx = [i for i, c in enumerate(_co) if c.get("boresch")]
  stage_rates[_leg["name"]] = (0.0 if _u1 == _u0
                               else (_u1 - _u0) * args.pull_dist / _ps)

# Each stage's rate is recorded next to the geometry, for the same reason the
# single rate always was: integrate.py turns the time integral of the pull force
# into work by multiplying by the rate, so a stage integrated at another stage's
# rate is silently rescaled. With one rate that could have been a constant; with
# one per stage it cannot. Appended rather than written above because the rates
# are only known once each stage's mdp has been read.
with open("boresch_analytical.gs", "a") as f:
  f.write("u_boundaries          %s\n"
          % " ".join("%.4f" % b for b in P.boundaries()))
  for _leg in P.legs():
    if _leg["kind"] not in ("stage", "hold"):
      continue
    f.write("stage_rate_nm_ps  %-20s %.10g  # u %.2f -> %.2f over %g ps\n"
            % (_leg["name"], stage_rates[_leg["name"]],
               _leg["u_from"], _leg["u_to"], leg_ps(_leg["mdp"]) or 0.0))

# Hold leg at the unbound restrained state (lambda = 1): only the Boresch
# coordinates, with zero rate. job_fe.run pins init-lambda = 1, delta-lambda = 0.
#
# The interface coordinates are deliberately left out. At lambda = 1 their force
# constant is kB = 0, so they restrain nothing, but GROMACS still evaluates every
# pull coordinate and aborts if any pair of its groups exceeds 0.49 * box:
#
#   Fatal error: Distance between pull groups 73 and 75 (3.772877 nm) is larger
#   than 0.49 times the box size (3.770459).
#
# Once the partners separate the interface surfaces relax apart, and with several
# hundred inert pairs one of them eventually crosses that limit: 8 of 14 holds
# died this way. The check has nothing to do with the physics of the hold, since
# no force is computed from those coordinates. Dropping them removes the failure
# mode outright rather than making it rarer, and the hold gets cheaper as a
# bonus. Verified that grompp accepts the reduced pull setup from the unbinding
# leg's checkpoint, that mdrun runs on it, and that the rebinding leg still
# grompps from the hold's checkpoint with the full block restored.
#
# Sharpening the mechanism, because it reads as if a restraint were removed. Their
# force constant is already kB = 0 at lambda = 1, so they restrained nothing here
# either way; what the hold loses is only the CHECK. The interface surfaces keep
# relaxing apart for the full 5 ns, and the rebinding leg grompps from this leg's
# checkpoint with the full block restored, so whatever they drifted to is what
# bindrev is checked against at t = 0. Measured over 8 cycles on 2KTF: bindfwd
# peaks at 2.40-3.07 nm (always at its end), the hold runs on unchecked to
# 2.88-3.88 nm, and bindrev starts at 2.41-3.56 nm against a 4.045 nm limit before
# decaying within 400 ps. The first frame of bindrev, not the unbinding leg, is
# what the box padding in job_fe.run has to cover.
#
# Do not "fix" this by tethering the dropped pairs during the hold. A flat-bottom
# tether present only in nptrev is absent from the lambda = 1 Hamiltonian of the
# stages either side of it, so it would bias the starting distribution handed to
# the reverse leg, which is the equilibrium end state the Crooks analysis assumes.
#
# The block itself is written by the loop above, which drops the interface
# coordinates from whichever hold sits at u = pull_dist.

# Bound-state restraint leg
pg_b_fwd, co_b_fwd = build_coords("bound", "fwd")
pg_b_rev, co_b_rev = build_coords("bound", "rev")
write_pull_block("boundfwd.mdp", pg_b_fwd, co_b_fwd)
write_pull_block("boundrev.mdp", pg_b_rev, co_b_rev)


#======================================================
# PART 6 - Read the pull block back through GROMACS
#
# Everything above is open loop: the six Boresch coordinates are measured with
# this file's own helpers, written into four .mdp files, and never verified. A
# one-character sign error in dihedral_deg therefore survived into production and
# cost days of GPU time, because a mirror-image phi reference is not visibly
# wrong anywhere. It only shows up as strain once mdrun starts.
#
# So grompp the forward block at nsteps = 0 against the same structure the
# references were measured from, run zero steps, and compare GROMACS's own pullx
# against what was emitted. This catches sign conventions, group ordering, unit
# errors and index drift in one test, for about 6 s per setup.
#
# The FORWARD block, not the hold: bindrev_fe and nptrev_fe carry
# init = ref_r + pull_dist, so on the bound reference their distance coordinate
# legitimately differs from its init by pull_dist and the check would misfire.
#======================================================

CHECK_TOL_DEG = 0.5     # 0.19 sigma at K_ANG_RAD; mass repartitioning alone is 0.96
CHECK_TOL_NM = 0.005


def wrap180(x):
  return (x + 180.0) % 360.0 - 180.0


def verify_pull_block(mdp):
  """Re-measure the emitted pull coordinates with GROMACS. None if not run."""
  if not args.readback:
    return None
  if not os.path.isfile(args.topol):
    sys.stderr.write("make_boresch: skipping the pull readback, no %s\n"
                     % args.topol)
    return None
  if not shutil.which("gmx"):
    sys.stderr.write("make_boresch: skipping the pull readback, gmx not on PATH\n")
    return None
  work = ".pullcheck"
  try:
    os.makedirs(work, exist_ok=True)
    probe = os.path.join(work, "probe.mdp")
    with open(probe, "w") as f:
      for line in open(mdp):
        k = line.split("=")[0].strip()
        # pull-print-com1/2 are the obsolete spellings; grompp refuses a file
        # carrying both those and the modern pull-print-com that we append.
        if k in ("nsteps", "pull-nstxout", "nstxout-compressed", "nstlog",
                 "nstenergy", "continuation", "gen_vel", "free-energy",
                 "init-lambda", "delta-lambda", "nstdhdl",
                 "pull-print-com", "pull-print-com1", "pull-print-com2"):
          continue
        f.write(line)
      f.write("\nnsteps = 0\npull-nstxout = 1\npull-print-com = no\n"
              "continuation = yes\ngen_vel = no\nfree-energy = no\n")
    tpr = os.path.join(work, "probe.tpr")
    r = subprocess.run(["gmx", "grompp", "-f", probe, "-c", args.input,
                        "-r", args.input, "-p", args.topol, "-n", args.readback_index,
                        "-o", tpr, "-maxwarn", "20"],
                       capture_output=True, text=True)
    if r.returncode != 0:
      sys.stderr.write("make_boresch: pull readback grompp failed:\n%s\n"
                       % r.stderr[-2000:])
      return False
    r = subprocess.run(["gmx", "mdrun", "-s", tpr, "-nt", "1", "-nb", "cpu",
                        "-deffnm", os.path.join(work, "probe")],
                       capture_output=True, text=True)
    xvg = os.path.join(work, "probe_pullx.xvg")
    if r.returncode != 0 or not os.path.isfile(xvg):
      sys.stderr.write("make_boresch: pull readback mdrun failed:\n%s\n"
                       % r.stderr[-2000:])
      return False
    row = None
    for line in open(xvg):
      if not line.startswith(("#", "@")):
        row = [float(v) for v in line.split()]
        break
    if not row:
      sys.stderr.write("make_boresch: pull readback produced no pullx row\n")
      return False
    return row[1:]
  except (OSError, ValueError) as e:
    sys.stderr.write("make_boresch: pull readback error: %s\n" % e)
    return False
  finally:
    shutil.rmtree(work, ignore_errors=True)


# The FIRST stage is the block whose inits ARE the bound references (u_from = 0),
# so it is the only stage the readback can compare against the reference
# structure. Any later stage legitimately sits u_from*pull_dist away and would
# false-positive.
_read = verify_pull_block(first_stage_mdp)
if _read is False:
  abort("PULL_READBACK_FAILED",
        "could not read the emitted pull block back through GROMACS")
elif _read is not None:
  # The six Boresch coordinates, in emission order, located by the marker
  # build_coords set on them rather than by counting back from the end.
  names = ["r", "theta_A", "theta_B", "phi_A", "phi_B", "phi_C"]
  want = [ref_r, ref_thA, ref_thB, ref_phA, ref_phB, ref_phC]
  if len(boresch_idx) != 6:
    abort("PULL_READBACK_FAILED",
          "expected 6 Boresch coordinates in %s, found %d"
          % (first_stage_mdp, len(boresch_idx)))
  got = [_read[i] for i in boresch_idx] if max(boresch_idx) < len(_read) else []
  bad = []
  for nm, w, g in zip(names, want, got):
    if nm == "r":
      d = abs(g - w)
      ok = d <= CHECK_TOL_NM
    else:
      d = abs(wrap180(g - w))
      ok = d <= CHECK_TOL_DEG
    sys.stderr.write("make_boresch: readback %-8s emitted %10.4f  gromacs "
                     "%10.4f  diff %8.4f%s\n"
                     % (nm, w, g, d, "" if ok else "   MISMATCH"))
    if not ok:
      bad.append("%s off by %.4f" % (nm, d))
  if len(got) < 6:
    abort("PULL_READBACK_FAILED",
          "pullx returned %d columns, too few to hold the Boresch coordinates "
          "at %s" % (len(_read), boresch_idx))
  if bad:
    abort("PULL_READBACK_MISMATCH",
          "GROMACS does not reproduce the emitted Boresch references (%s). A "
          "sign or ordering convention in this file disagrees with the pull "
          "code; do not run on this setup." % "; ".join(bad))

#======================================================
# Output: number of moving pull coords whose forces feed the pull integrator.
# Interface restraints (numinterdis) + the single Boresch distance coord = +1.
# (Elastic network and Boresch angles/dihedrals have rate 0 -> no pull work.)
#======================================================
print(numinterdis + 1)
