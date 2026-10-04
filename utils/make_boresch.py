#!/usr/bin/env python3
#
# make_boresch.py - Generate the pull configuration for the free-energy
# (groscore_fe) protocol.
#
# This is the FE-variant counterpart of make_disres_en.py. It writes a single,
# self-contained COM-pull block into each leg .mdp so that all restraint
# generation lives in one place with consistent pull-group numbering.
#
# Two leg families are produced, and the elastic network once more on its own.
# WHICH legs carry which family is fe_protocol.legs()'s "block" key, never the leg
# name and never the leg kind; see the leg loop in PART 6.
#
#   1. Unbinding / rebinding leg  (bind_fe / bindrev_fe, and the ramp holds
#      between them; block "unbind")
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
#   1b. Hold leg at the far end  (nptrev_fe): the six Boresch coordinates AND the
#      elastic network, at zero rate, without the interface coordinates. Those are
#      inert there (kB = 0 at lambda = 1) but GROMACS still enforces its
#      0.49 * box distance check on them, which killed 8 of 14 holds. The network
#      is not inert: it is part of the lambda = 1 Hamiltonian of the stages on
#      either side, and v1.3 dropped it together with the springs, so it switched
#      back on unrecorded at the start of bindrevI with a median 153 kJ/mol of
#      strain (39 structures of the second HADDOCK pass, per-structure medians up
#      to 918). See the note at the far-end strip for the detail.
#
#   2. Bound-state restraint leg  (boundfwd / boundrev, AND the holds inside the
#      bound lambda schedule, holdbfwd* / holdbrev*; block "bound")
#      - Interface restraints introduced 0 -> full in the bound state, NO
#        pulling (rate 0). Pure dhdl. Gives the free energy of introducing the
#        restraints in the bound ensemble. No Boresch here.
#      - Elastic network: the same coordinates as family 1.
#      The holds carry this block because they sit at lambda 0.25 of THIS
#      schedule. v1.3 picked blocks by kind and gave them the unbind block
#      (springs at 0.75 k_inter instead of 0.25, plus the Boresch at 25%), so the
#      Hamiltonian jumped at hold entry and exit by 6.4 and 7.3 kJ/mol of
#      unrecorded work on average, structure dependent.
#
#   3. Bound-state NPT  (npt_c_fe.mdp, the per-cycle 20 ns equilibration that
#      boundfwd1 starts from): the elastic network ONLY, k = kB = enk, rate 0, no
#      free-energy block. Without it the network switched on unrecorded at t = 0
#      of boundfwd1 with a median 299 kJ/mol (single cycles up to 33,549), and 20
#      of 920 cycles died at that grompp because a network pair had drifted beyond
#      0.49 box during npt_c. A structure without a network gets 'pull = no'
#      there, i.e. exactly the npt_fe physics. The network does NOT make a partner
#      one body: it joins only CAs near cut ends, so the setup log counts the
#      bodies it leaves per partner (see PART 2).
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
# -f must be ONE PIECE: job_fe.run writes npt_probe_cluster.gro with
# utils/assemble_frame.py, and PART 0 stops the setup (INPUT_FRAME_SPLIT) if any
# polymer residue of -f is not where its contacts put it.
#

import os, sys, re, argparse, math, itertools, subprocess, traceback, shutil
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_tools

#------------------------------------------------------

# WIDTH CEILING ON AN INTERFACE SPRING, nm. The contact cutoff above is applied to
# the MEAN distance over the pooled probe replicas, and the mean cannot see how
# much of the time a pair is actually in contact. Measured across test13/14/15,
# 27-40% of the springs that survive the mean cutoff are outside it in more than a
# quarter of the frames, and r(mean, sd) is only +0.27 to +0.32, so filtering on
# the mean does not remove them: they sit at short mean distance and swing.
#
# They are not free. A harmonic spring referenced to r0 = <d> costs var(d), so
#
#     <W_intro> = 0.5 * sum_k * mean_i(sd_i^2)
#
# which does NOT depend on the number of springs: dropping the widest lowers the
# bound-leg work and its fluctuation even though k = sum_k/N rises to compensate.
# The widest 10% of pairs carry 33-48% of sum(sd^2) on those runs.
#
# 0.15 nm is where that trade is best: it keeps 92-95% of the springs, so the
# interface is still pinned at nearly as many points, while cutting the modelled
# bound-leg work to 63-78%. Dropping on intermittency directly instead removes a
# third of the interface for a similar gain, which is a worse trade.
SD_MAX_DEFAULT = 0.15
SD_MIN_KEEP = 60          # never let the width ceiling take the set below this

parser = argparse.ArgumentParser(description="Generate Boresch + interface + elastic-network pull config for the FE protocol.")
parser.add_argument('-f', '--input', type=str, default="npt_probe_cluster.gro",
                    help="Reference structure the restraints are measured from "
                         "(default: npt_probe_cluster.gro, the last probe frame "
                         "assembled in one piece by utils/assemble_frame.py; a "
                         "frame whose polymer is split across the periodic "
                         "boundary stops the setup with INPUT_FRAME_SPLIT). "
                         "--topol and --readback-index must match it.")
parser.add_argument('-m', '--chainmap', type=str, required=True, help="Chain map file containing residue numbers for protein B (ligand side).")
parser.add_argument('-T', '--temp', type=float, default=310.0, help="Temperature in K for the analytical term (default: 310).")
parser.add_argument('--pull-dist', type=float, default=1.0, help="Maximum COM-COM separation added during unbinding, in nm (default: 1.0).")
parser.add_argument('--pull-rate', type=float, default=0.00005, help="Pull rate in nm/ps (default: 0.00005, i.e. 1.0 nm over the 20 ns unbinding leg).")
parser.add_argument('--sd-max', dest='sd_max', type=float, default=SD_MAX_DEFAULT,
                    help="Drop interface pairs whose distance standard deviation "
                         "over the pooled probe replicas exceeds this, in nm "
                         "(default: %g). A pair wider than this is not a contact "
                         "being restrained, it is an average of a contact and a "
                         "gap. Pass 0 or a negative value to keep every pair; the "
                         "spread is reported either way."
                         % SD_MAX_DEFAULT)
parser.add_argument('--sum-k', dest='sum_k', type=float, default=12500.0, help="Total interface restraint stiffness in kJ/mol/nm^2, split evenly over however many springs there are (k = sum_k/N). It sets the work of switching the interface restraints on, which is 0.5*sum_k*<(d-r0)^2> and is the largest single term in the bound leg; it also sets how hard the springs pull before the Boresch takes over, so it cannot go arbitrarily low. The ramp boundaries in fe_protocol.py were measured at this value and do not transfer to another one unchanged. Default 12500.")
parser.add_argument('--traj', type=str, nargs='+', default=["npt_probe.xtc"],
                    help="Equilibration trajectory used to measure backbone "
                         "rigidity for anchor selection (default: npt_probe.xtc). "
                         "If missing, the burial heuristic is used instead.")
parser.add_argument('--tpr', type=str, nargs='+', default=["npt_probe.tpr"],
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

# One tpr per trajectory. A single tpr with several trajectories is the common case
# (same system, different seeds), so it is broadcast rather than rejected; anything
# else is a mistake worth stopping for, because trjconv would otherwise read the
# wrong topology for every replica after the first and say nothing.
if len(args.tpr) == 1 and len(args.traj) > 1:
    args.tpr = args.tpr * len(args.traj)
if len(args.tpr) != len(args.traj):
    sys.stderr.write("make_boresch: %d --traj against %d --tpr; give one tpr, or "
                     "one per trajectory\n" % (len(args.traj), len(args.tpr)))
    sys.exit(2)
TRAJ_PAIRS = [(t, u) for t, u in zip(args.traj, args.tpr)
              if os.path.isfile(t) and os.path.isfile(u)]

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

# THE CANDIDATE PASS IS DELIBERATELY WIDER THAN THE CONTACT CUTOFF, so that which
# pairs end up restrained is decided by the ENSEMBLE and not by the one frame the
# candidates are enumerated from.
#
# The 0.6 nm cutoff is applied to the ensemble mean (see finalise_interface), but
# the candidates it is applied to used to be enumerated at 0.6 nm in
# npt_probe1_cluster.gro. So a pair sitting at 0.62 nm in that frame whose mean is
# 0.55 was never considered, and the frame is a draw: across eight runs of one
# structure it yielded 556 to 918 candidates and 228 to 322 springs, a 41% range in
# the spring count that set dG_intro (r = +0.91 with the count) and, because
# k = sum_k/N, the stiffness of every spring with it (r = -0.73 with stage E's
# dissipation). It cancels in dG_bind, at r = -0.10, so it was never a bias -- but
# it is why the per-channel intervals are 4-5x too small and it fed the stage that
# loses BAR.
#
# The cut must be wide enough that it is NON-BINDING, not merely wider. That is a
# claim about each interface, not a general one, so finalise_interface reports the
# margin between the widest RESTRAINED pair's frame distance and this cut, and warns
# when there is none.
#
# 0.9 -> 1.2 AFTER THAT DIAGNOSTIC FIRED. 0.9 came from the frame-to-mean shift over
# eight runs (+1.1 to +2.0 A on average, rms 1.6 to 2.7 A) plus the +0.3 nm reach of
# the pairs the mean cutoff already rejects. It held on test27 with 0.107 nm to
# spare, and then, on the five runs at that setting:
#
#     test29  margin 0.205      test31  margin 0.013   <- binding
#     test30  margin 0.155      test32  margin 0.019   <- binding
#     test33  margin 0.097
#
# Two of five had restrained pairs at 0.887 and 0.881 nm against a 0.9 cut, so for
# those the frame was deciding the spring set again -- exactly what the wide pass
# exists to stop. test32 also came out with the fewest springs of the five (289) and
# the stiffest set (43.25), which is what a binding cut looks like downstream.
#
# 1.2 gives 0.6 nm of headroom against an rms shift of 0.16 to 0.27 nm, i.e. well
# past anything measured, and costs roughly double the candidates. That is affordable
# only because the trajectory pass is chunked (see CHUNK_BYTES); without it the cut
# would be chosen on what fits in memory rather than on the physics.
#
# WHAT THE WIDE PASS BUYS, measured. At 0.9 on test27's own probes the spring set
# went from 228 at k = 54.82 to 281 at k = 44.48, of which 53 -- 19% of the final
# set -- could not have been candidates before. Over test29-33 the sets are 344 +- 38
# springs at k = 36.7 against 268 +- 33 at k = 47.2 for the eight runs before it. The
# anchor search is untouched by any of it and reproduced test27's triad exactly, down
# to r 1.4407 -> 1.4602 nm and 4.9 deg of frame rotation.
CANDIDATE_CUT = 1.2

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

# Fixed .gro columns and atoms by POSITION (pbc_tools.read_gro), and the solute is
# everything before the first water, as in make_disres_en.py. The split parse read
# the residue field '   331PE' (residue 33, ligand 1PE) as residue 331, so the
# ligand fell out as solvent or onto the wrong side, and its rule of every residue
# number up to max(chain_map, ion and ligand residues) dropped a receptor chain
# that a --no-cutout input lists after protein B (v15_work review PY-4, audit R7
# and R8 of the classic engine). Counterions come after the water (genion), and
# crystal waters are appended after the protein, ions and ligands. The residue
# field stays '<resnum><resname>', the key everything below groups residues by.
try:
  _frame = pbc_tools.read_gro(args.input)
except (OSError, ValueError) as e:
  abort("INPUT_STRUCTURE_UNREADABLE", "cannot read %s: %s" % (args.input, e))
_n_solute = next((i for i, h in enumerate(_frame.head) if h[5:10].strip() == "SOL"),
                 len(_frame.head))
for _i in range(_n_solute):
  _h = _frame.head[_i]
  try:
    _resnum = int(_h[0:5])
  except ValueError:
    abort("INPUT_STRUCTURE_UNREADABLE",
          "unreadable residue number in line %d of %s" % (_i + 3, args.input))
  rec = ("%d%s" % (_resnum, _h[5:10].strip()), _resnum, _h[10:15].strip(), _i + 1) \
      + tuple(float(v) for v in _frame.xyz[_i])
  (prot2_data if _resnum in residues_b else prot1_data).append(rec)

# Load before anything takes a COM. The probe tpr already exists when job_fe.run
# calls this script, and it is the same system, already repartitioned.
TPR_MASS = load_tpr_masses(args.tpr[0])
if TPR_MASS:
  sys.stderr.write("make_boresch: masses read from %s (%d atoms)\n"
                   % (args.tpr[0], len(TPR_MASS)))
else:
  sys.stderr.write("make_boresch: WARNING - no masses from %s, falling back to "
                   "element masses; COMs will not match GROMACS exactly\n"
                   % args.tpr[0])


def box_matrix(box_line):
  """The three box vectors of a .gro box line, as the rows of a 3x3 list.

  The line is v1x v2y v3z v1y v1z v2x v2z v3x v3y; a rectangular box writes only
  the first three and the rest are zero."""
  f = [float(v) for v in box_line.split()]
  f += [0.0] * (9 - len(f))
  return [[f[0], f[3], f[4]], [f[5], f[1], f[6]], [f[7], f[8], f[2]]]


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
  return [math.sqrt(sum(c * c for c in u)) for u in box_matrix(box_line)]

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

# The box of -f ITSELF, which the single-frame geometry is minimum-imaged in. Not
# BOX_VEC: that is the production box the limits scale with, while -f comes from
# the probe system. An unreadable box line gives the zero matrix, under which
# min_image_vec returns its input unchanged, i.e. the raw coordinates as before.
try:
  _l = [x for x in open(args.input).read().split("\n") if x.strip()]
  INPUT_BOX = np.asarray(box_matrix(_l[-1]), dtype=float)
except (ValueError, IndexError, OSError):
  INPUT_BOX = np.zeros((3, 3))

# Fraction of that limit the Boresch distance is allowed to reach at full
# extension. Named once because THREE places need the same number and two of them
# used to disagree: the cross-triad search filtered at 0.9 while the final guard
# aborted at 0.6, so a triad in between passed selection and then killed the run.
# 0.6 leaves room for the interface pairs, which travel further than the Boresch
# r does -- test4 measured a largest checked pair of 2.995 nm against a 4.101 nm
# limit, 73%, while its own r0_release sat at 54%.
R_LIMIT_FRAC = 0.6

# The file can be present and still yield nothing for one side, e.g. an empty
# chain_map.gs puts every solute atom on the receptor side. That produced the same
# degenerate arrays downstream as a missing file.
if not prot1_data or not prot2_data:
  abort("EMPTY_STRUCTURE",
        "%s gave %d receptor and %d ligand atoms (check chain_map.gs)"
        % (args.input, len(prot1_data), len(prot2_data)))

#======================================================
# Periodic images
#======================================================

# The 27 single translations of a triclinic cell, built once.
_IMG = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                 for k in (-1, 0, 1)], dtype=float)


def to_brick(dv, M):
  """dv moved by whole box vectors into the brick |dz| <= cz/2, |dy| <= by/2,
  |dx| <= ax/2, the reduction GROMACS's own pbc_dx starts from. GROMACS boxes are
  lower triangular (v1 = (ax,0,0), v2 = (bx,by,0), v3 = (cx,cy,cz)), so z is
  settled by v3 alone, then y by v2, then x by v1. A zero diagonal (no periodicity
  along that axis) leaves that component alone."""
  for d in (2, 1, 0):
    den = np.where(M[..., d, d] == 0, np.inf, M[..., d, d])
    dv = dv - np.round(dv[..., d] / den)[..., None] * M[..., d, :]
  return dv


# THE BRICK FIRST, THEN THE 27 NEIGHBOURS. The search over single translations used
# to start from the raw difference, and it is exact only when that difference is
# within one box vector per axis of the true one. A -pbc whole frame does not
# promise that: each molecule is made whole outward from an atom mdrun had put in
# the cell, so two atoms of different molecules, or of two unbonded fragments of
# one cutout, can sit two lattice vectors apart. Measured on short vectors
# displaced by up to two lattice vectors per axis in the xy-square dodecahedron
# these setups use, the raw search returned the wrong image for 156,817 of 200,000
# draws and the brick-first search for none. Over 200,000 arbitrary vectors up to
# 30 nm the brick-first search also agreed with an exhaustive +-3 search from the
# brick, in both dodecahedra, the octahedron, a rectangular box and the skewed box
# [[10,0,0],[5,6,0],[-5,3,5]]. tests/test_boresch_pbc.py repeats the comparison.
#
# NOT EXACT IN EVERY BOX GROMACS ACCEPTS. In a flatter box that grompp's box check
# still allows, the brick plus 27 neighbours can miss the minimum for a vector
# longer than about half the smallest diagonal element: in [[10,0,0],[5,6,0],
# [5,3,2]] it returned a longer image for 217 of 12,000 vectors, up to 0.68 nm too
# long, the shortest failing one 3.02 nm (review R7). Every box this pipeline
# builds is an editconf dodecahedron scaled isotropically by C-rescale, where it is
# exact.
def min_image_vec(dv, M):
  """The minimum image of dv under box M, exact in the boxes this pipeline builds
  (see the note above). min_image_dist takes its norm; angles and dihedrals need
  the vector itself. dv is (..., 3) and M is (3, 3) or (..., 3, 3), broadcast
  against each other."""
  dv = to_brick(dv, M)
  shifts = np.einsum("ij,...jk->...ik", _IMG, M)        # (..., 27, 3)
  cand = dv[..., None, :] + shifts
  k = np.argmin((cand ** 2).sum(-1), axis=-1)
  return np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]


def min_image_dist(dv, M):
  """Shortest distance equivalent to displacement dv under box M.

  dv is (..., 3) and M is (3, 3) or (..., 3, 3), so a whole trajectory of pair
  vectors goes through in one call."""
  dv = to_brick(dv, M)
  shifts = np.einsum("ij,...jk->...ik", _IMG, M)        # (..., 27, 3)
  cand = dv[..., None, :] + shifts
  return np.sqrt((cand ** 2).sum(-1)).min(-1)


def shortest_lattice_vector(M):
  """Length of the shortest non-zero lattice vector of box M (rows a, b, c), or 0
  for a box with a zero diagonal, i.e. no periodicity to speak of. Searched over
  every combination of up to two of each box vector, which contains the shortest
  one of every box GROMACS builds."""
  M = np.asarray(M, dtype=float)
  if not np.all(np.diag(M) > 0):
    return 0.0
  c = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3)
                for k in range(-2, 3) if (i, j, k) != (0, 0, 0)], dtype=float)
  return float(np.sqrt(((c @ M) ** 2).sum(-1)).min())


def min_image_cdist(A, B, M):
  """Pairwise distances between the rows of A and of B under the minimum image of
  box M, laid out as cdist(A, B).

  A difference no longer than half the shortest lattice vector is its own minimum
  image, since every other image of it is at least that much longer; cdist's own
  number is kept for those, so a frame that is one piece gets bit for bit the
  distances it always got. Only the longer differences are re-measured with
  min_image_dist, in chunks; on a frame in one piece those are pairs across the
  complex, far beyond every cutoff they are tested against. A box with a zero
  diagonal gives plain cdist."""
  d = cdist(A, B)
  half = 0.5 * shortest_lattice_vector(M)
  if half <= 0 or not d.size:
    return d
  ii, jj = np.nonzero(d > half)
  blk = max(1, (256 << 20) // (len(_IMG) * 3 * 8))
  for s in range(0, len(ii), blk):
    a, b = ii[s:s + blk], jj[s:s + blk]
    d[a, b] = min_image_dist(B[b] - A[a], M)
  return d


def nearest_image(X, ref, M):
  """X (..., 3) moved by whole box vectors to its image nearest ref. A point that
  is already there comes back bit for bit, because its shift is the difference of
  two identical vectors."""
  X = np.asarray(X, dtype=float)
  dv = X - ref
  return X + (min_image_vec(dv, M) - dv)


FRAG_BOND_NM = 0.2      # backbone neighbours closer than this are covalently bonded


def backbone_fragments(xyz, M):
  """Pieces of one side's backbone that share no backbone bond, as arrays of row
  indices, largest first (input order among equals).

  The rows are N, CA, C of each residue in input order (caps contribute their N
  or C), so two backbone atoms can only be bonded if they are neighbours in that
  order, and neighbours are bonded exactly when they sit within FRAG_BOND_NM:
  N-CA is 0.147 nm, CA-C 0.153, C-N 0.133, and no two non-bonded heavy atoms come
  within 0.25 nm. Distances are minimum-imaged in M, so a fragment that -f itself
  shows split is still one fragment."""
  if len(xyz) == 0:
    return []
  gap = min_image_dist(xyz[1:] - xyz[:-1], M) > FRAG_BOND_NM
  pieces = np.split(np.arange(len(xyz)), np.flatnonzero(gap) + 1)
  return sorted(pieces, key=len, reverse=True)


#======================================================
# PART 0 - -f must be one piece
#
# Every restraint below is measured on -f: the interface candidates and the elastic
# network as distances, the anchor groups as COMs, the pull axis and every spring's
# pull path as vectors. Until this check they were raw coordinate differences, and
# -f came from gmx trjconv -pbc cluster, which makes a moleculetype whole atom by
# atom along the sequence. A cutout partner is ONE moleculetype of several capped,
# unbonded fragments, so a sequence step between two fragments that is longer than
# half the box puts every later fragment of that partner one box vector off. On the
# real 1IJK probe the step is NME299 -> ACE300, 7.28 nm in a 9.90 nm box, and ligand
# residues 300-416 sat one box vector from the rest. Raw distances then found 8076
# candidate pairs within 1.2 nm where the minimum image finds 19366, all 435 springs
# landed on ligand residues 201-299, 720 of the 1108 contact pairs inside 0.6 nm got
# none, and the ligand network lost the 5 pairs that are its only link between
# 170-299 and 300-416. Nothing in the log said so (review FR1).
#
# job_fe.run now writes -f with utils/assemble_frame.py: pbc_tools.assemble, a
# minimum spanning tree over one anchor atom per residue, places every residue at
# the image nearest the residue it hangs from, so contact decides the image and the
# sequence order does not. The same assembly is run here on the solute of -f, for
# two things:
#
#   * THE GUARD. If it moves any POLYMER residue, -f is not one piece and the setup
#     stops with INPUT_FRAME_SPLIT. A frame from assemble_frame.py never trips it.
#     The test is by contact and not by fragment centroid: a centroid offset from
#     the root fragment is ambiguous beyond half a box, and in a frame that IS one
#     piece a fragment can sit that far from the root's centroid (1RLB: a 5-residue
#     fragment 5.15 nm from it against a probe half-box of 4.94 nm, in 23% of
#     orientations; review R2), so a centroid test would stop correct setups.
#   * THE COORDINATES. Everything below reads the assembled positions, so a ligand
#     or an ion that -f has in another image (non-polymer residues are moved with a
#     note, not stopped) is measured where its contacts are. Atoms the assembly does
#     not move keep their coordinates bit for bit.
#
# On top of that the distances that become restraints are minimum-imaged (PART 1
# and 2, the anchor-group COMs and the springs' pull paths), because the minimum
# image is what GROMACS pulls on. On a frame in one piece it is the same number,
# and with it the candidate, spring and network sets no longer depend on how -f is
# wrapped even if this guard were bypassed: on the split 1IJK frame, with the guard
# and the assembly both switched off, the 1150 springs and their references and the
# 226 network pairs came out as on the assembled frame. A
# vector between two arbitrary residues, which the anchor search and the burial
# heuristic take, has no minimum image to trust, because the complex is wider than
# half its probe box (CA extent 7.41 nm against 4.95 on 1IJK, 4.10 against 3.33 on
# 2KTF); only the assembled frame gives those right.
#======================================================

def assemble_input(data, M):
  """Box-vector coefficients that put every residue of `data` where its contacts
  put it.

  data is prot1_data + prot2_data, M the box of -f. A residue is a run of atoms
  with the same residue field, in input order. pbc_tools.assemble joins them by a
  spanning tree over one anchor atom per residue (its CA, else its first atom), the
  assembly measure_box, interface_qc, rebound_rmsd and assemble_frame.py use, here
  rooted at the first residue of the largest polymer fragment so that the bulk of
  the complex stays where -f has it.

  Returns (n, polymer, blocks, longest): n the (atoms, 3) integer coefficients of
  the box vectors each atom is moved by, all zero for an atom already in place;
  polymer[k] whether residue k is polymer (an N/CA/C triple, or a cap); blocks the
  residues as (start, end); longest the longest link of the tree in nm, nan for a
  box with no periodicity, where nothing is assembled."""
  xyz = np.array([rec[4:7] for rec in data], dtype=float)
  blocks, start = [], 0
  for i in range(1, len(data) + 1):
    if i == len(data) or data[i][:2] != data[start][:2]:
      blocks.append((start, i))
      start = i
  polymer = [{"N", "CA", "C"} <= {data[i][2] for i in range(s, e)}
             or re.sub(r"^\d+", "", data[s][0]) in pbc_tools.CAP_RESIDUES
             for s, e in blocks]
  n = np.zeros((len(data), 3), dtype=int)
  if not np.all(np.diag(M) > 0):
    return n, polymer, blocks, float("nan")
  anchors = [next((i for i in range(s, e) if data[i][2] == "CA"), s) for s, e in blocks]
  block_of = np.empty(len(data), dtype=int)
  for k, (s, e) in enumerate(blocks):
    block_of[s:e] = k
  rows = np.array([i for k, (s, e) in enumerate(blocks) if polymer[k]
                   for i in range(s, e) if data[i][2] in BACKBONE], dtype=int)
  root = 0
  if len(rows):
    root = int(block_of[rows[backbone_fragments(xyz[rows], M)[0][0]]])
  order = [root] + [k for k in range(len(blocks)) if k != root]
  asm, longest = pbc_tools.assemble(xyz, M, [blocks[k] for k in order],
                                    np.array([anchors[k] for k in order]))
  n = np.rint((asm - xyz) @ np.linalg.inv(M)).astype(int)
  return n, polymer, blocks, longest


def describe_moved(data, n, blocks, which, len_first):
  """'ligand 300-416 by (0,-1,1)'-style runs of the residues `which`, one run per
  side, box-vector shift and stretch of consecutive residue numbers."""
  runs = []
  for k in which:
    s = blocks[k][0]
    key = ("receptor" if s < len_first else "ligand", tuple(int(v) for v in n[s]))
    if runs and runs[-1][0] == key and data[s][1] == runs[-1][2] + 1:
      runs[-1][2] = data[s][1]
    else:
      runs.append([key, data[s][1], data[s][1]])
  out = ["%s %s by (%d,%d,%d)" % ((key[0], "%d" % lo if lo == hi else "%d-%d" % (lo, hi))
                                  + key[1]) for key, lo, hi in runs]
  return "; ".join(out[:8]) + ("; and %d more" % (len(out) - 8) if len(out) > 8 else "")


def apply_shifts(data, n, M):
  """data with every atom moved by its box vectors, untouched where n is zero."""
  L = n @ M
  return [rec[:4] + tuple(float(v) for v in np.array(rec[4:7], dtype=float) + L[i])
          if n[i].any() else rec for i, rec in enumerate(data)]


_all_data = prot1_data + prot2_data
_shift_n, _polymer, _blocks, _longest = assemble_input(_all_data, INPUT_BOX)
_moved = [k for k, (s, e) in enumerate(_blocks) if _shift_n[s:e].any()]
_moved_poly = [k for k in _moved if _polymer[k]]
if _moved_poly:
  abort("INPUT_FRAME_SPLIT",
        "%s is not one piece: %d polymer residue(s) sit whole box vectors from "
        "where their contacts put them (%s; longest contact link %.3f nm). "
        "job_fe.run writes -f with utils/assemble_frame.py, which never leaves a "
        "frame like this; gmx trjconv -pbc cluster does, for unbonded fragments "
        "farther apart than half the box"
        % (args.input, len(_moved_poly),
           describe_moved(_all_data, _shift_n, _blocks, _moved_poly, len(prot1_data)),
           _longest))
if _moved:
  sys.stderr.write("make_boresch: %d non-polymer residue(s) of %s sat whole box "
                   "vectors from their contacts and are measured where the contacts "
                   "put them: %s\n"
                   % (len(_moved), args.input,
                      describe_moved(_all_data, _shift_n, _blocks, _moved, len(prot1_data))))
  _all_data = apply_shifts(_all_data, _shift_n, INPUT_BOX)
  prot1_data, prot2_data = _all_data[:len(prot1_data)], _all_data[len(prot1_data):]
if math.isnan(_longest):
  sys.stderr.write("make_boresch: WARNING - %s has no periodic box line, so it is "
                   "taken as it stands and nothing is minimum-imaged\n" % args.input)
else:
  sys.stderr.write("make_boresch: %s is one piece: %d residues joined by contact, "
                   "longest link %.3f nm\n" % (args.input, len(_blocks), _longest))

len1 = len(prot1_data)
len2 = len(prot2_data)

prot1_coords = np.array([(d[4], d[5], d[6]) for d in prot1_data], dtype=np.float64) if len1 else np.empty((0, 3))
prot2_coords = np.array([(d[4], d[5], d[6]) for d in prot2_data], dtype=np.float64) if len2 else np.empty((0, 3))

#======================================================
# PART 1 - Interface atom-atom restraints (identical selection to make_disres_en:
# heavy atoms, not the MN* dummy masses of virtual sites, and an ion's lone atom
# always)
#======================================================

# Same list as make_disres_en.py, renumber_pdb.py and cap_termini.py: single-atom
# ion residues.
ION_RESIDUES = {"BA", "CA", "CD", "CL", "CO", "CS", "CU", "CU1", "FE", "FE2", "HG", "K",
                "LI", "MG", "MN", "NA", "NI", "PB", "SD", "SR", "ZN"}


def residue_atom_names(prot_data):
  """{residue field ('<resnum><resname>'): set of its atom names}."""
  names = {}
  for rec in prot_data:
    names.setdefault(rec[0], set()).add(rec[2])
  return names


def interface_atoms(prot_data):
  """Indices of the atoms that may carry an interface spring. The name tests drop
  hydrogens and the MN* dummy masses of virtual sites, and with them the lone atom
  of an Mn2+ (MN) or Hg2+ (HG) ion; an ion is therefore kept by its residue, as in
  make_disres_en.py. Without that the FE engine left out the Mn2+ of 1T6B, the metal
  bridging the two partners, that the classic engine restrains (review PY-3)."""
  ions = {rn for rn, nm in residue_atom_names(prot_data).items()
          if len(nm) == 1 and re.sub(r"^\d+", "", rn) in ION_RESIDUES}
  return np.array([i for i, rec in enumerate(prot_data)
                   if rec[0] in ions or (rec[2][0] != "H" and rec[2][:2] != "MN")], dtype=int)


prot1_valid = interface_atoms(prot1_data)
prot2_valid = interface_atoms(prot2_data)

# CANDIDATES, at CANDIDATE_CUT rather than interfacecutoff. The contact cutoff is
# applied later, to the ensemble mean, so that this frame decides only what gets
# MEASURED and the trajectory decides what gets restrained.
#
# MINIMUM-IMAGED in -f's own box, the distance GROMACS will pull on, so the set
# does not depend on how -f is wrapped. Raw distances on the split 1IJK frame of the
# first v1.5 round found 8076 pairs within 1.2 nm against 19366 and put all 435
# springs on one ligand fragment (PART 0). On a frame in one piece the two agree bit
# for bit; see min_image_cdist.
interdis = []  # (i, j, dist) indices into prot1_data / prot2_data
if len(prot1_valid) > 0 and len(prot2_valid) > 0:
  d = min_image_cdist(prot1_coords[prot1_valid], prot2_coords[prot2_valid], INPUT_BOX)
  for i_idx, i in enumerate(prot1_valid):
    for j_idx, j in enumerate(prot2_valid):
      if d[i_idx, j_idx] <= CANDIDATE_CUT:
        interdis.append((i, j, d[i_idx, j_idx]))
  _n_at_contact = int((d <= interfacecutoff).sum())
  sys.stderr.write(
      "make_boresch: %d candidate pairs within %.2f nm in the reference frame; "
      "%d of them are already inside the %.2f nm contact cutoff there, so the "
      "wider pass adds %d pairs for the ensemble to rule on\n"
      % (len(interdis), CANDIDATE_CUT, _n_at_contact, interfacecutoff,
         len(interdis) - _n_at_contact))

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
# restoring force (fixed at sum_k by k = sum_k/N, whatever N is) stays
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
    return a + 0.5 * min_image_vec(b - a, INPUT_BOX)

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

_n_snapshot = len(interdis)
# The frame distance each candidate was picked on, kept so interface_contacts.gs
# can record both ends of the shift after the references move.
_snap_of = {(i, j): dist for i, j, dist in interdis}

# NOTHING IS SETTLED YET. What the loop above produced is a CANDIDATE list, chosen
# by one frame's distances, and both the reference each spring gets and the final
# membership of the list are decided later, in finalise_interface(), once the
# probe trajectory has been read. Deciding them here would mean selecting on one
# basis and referencing on another, which is what test7 did and cost 40% of its
# springs to pairs that are not contacts on average. numinterdis, the cap and
# interface_contacts.gs all belong to that step and are set there.

#======================================================
# PART 2 - Elastic network (the selection of make_disres_en.build_elastic_network,
# with its distances minimum-imaged like the interface candidates above)
#
# WHAT THE NETWORK HOLDS, AND WHAT IT DOES NOT. Its nodes are the CAs of amino-acid
# residues (those with N, CA and C) within 0.9 nm of a cut-end residue (the residue
# next to an ACE or NME cap, or one carrying a terminal atom, all but the first and
# last of them) and its pairs are those nodes 0.4-0.9 nm apart. It therefore ties
# together fragment ends that happen to lie near each other and nothing else; it
# does not make a partner one body. On the real 1AKJ setup the receptor's fragments
# form two bodies under it, residues 1-261 and the tail 262-347 that carries 250 of
# the 1257 interface contacts, and the two ligand chains share no pair (critic, D3).
# On 1IJK the ligand is one body only through 5 pairs from residues 202-206 to 372,
# the pairs a split -f lost. In npt_c and nptrev the network holds each body
# together, not the bodies to each other, so bodies can drift apart there. This is
# not redesigned here; network_bodies() puts the number of bodies per partner and
# their residue spans in the setup log, so the pilot shows where it matters.
#======================================================

# Atom names that mark a fragment end without a cap next to it, as in
# make_disres_en.py: the C-terminal carboxyl of GROMOS (OT), CHARMM36 (OT1/OT2, its
# COOH patch) and AMBER (OXT), and the N-terminal H2. With only OT and H2 no
# CHARMM36 C-terminal cut end was an anchor: the CHARMM36 1A2K setup got 158
# network pairs where the classic engine gets 346 (review PY-2).
TERMINAL_ATOMS = {"OT", "OT1", "OT2", "OXT", "H2"}


def amino_acid_residues(prot_data):
  """Residue fields of the residues with N, CA and C. That leaves out the ions and
  the caps, which have atoms named CA too: a Ca2+ ion (residue CA, atom CA) and,
  under GROMOS, the ACE methyl that cap_termini.py --rename-ace-carbon names CA.
  Both used to become network nodes and anchors, as in make_disres_en.py before its
  audit R4: the GROMOS 1A2K setup had 8 ACE carbons in a network of 372 pairs where
  the classic engine has 331 (review PY-1)."""
  return {rn for rn, nm in residue_atom_names(prot_data).items() if {"N", "CA", "C"} <= nm}


def polymer_backbone(prot_data):
  """Atom numbers of the backbone atoms (N, CA, C) of the polymer: the amino-acid
  residues and the caps, which contribute their N or C to the backbone chain. Not
  an ion's atom named CA, which as a backbone row would join the self-fit, break the
  chain into spurious fragments and could trip RMSF_SPLIT if the ion moves."""
  poly = amino_acid_residues(prot_data) | {
      rec[0] for rec in prot_data if re.sub(r"^\d+", "", rec[0]) in pbc_tools.CAP_RESIDUES}
  return [rec[3] for rec in prot_data if rec[0] in poly and rec[2] in BACKBONE]


def build_elastic_network(prot_data, prot_coords):
  prot_len = len(prot_data)
  amino = amino_acid_residues(prot_data)
  anchor_resnames = set()
  resname_to_type = {}
  resnum_of = {}
  for i in range(prot_len):
    resname_to_type[prot_data[i][0]] = prot_data[i][0][-3:]
    resnum_of[prot_data[i][0]] = prot_data[i][1]
  for i in range(prot_len):
    if prot_data[i][2] in TERMINAL_ATOMS:
      anchor_resnames.add(prot_data[i][0])
  # Residue numbers from the parsed number field, not from the digits of the
  # residue field, which for '331PE' (residue 33, ligand 1PE) would give 331.
  resnum_to_resname = {num: rn for rn, num in resnum_of.items()}
  for resname, res3 in resname_to_type.items():
    num = resnum_of[resname]
    if res3 == "ACE":
      nxt = resnum_to_resname.get(num + 1)
      if nxt and resname_to_type.get(nxt) not in ("ACE", "NME"):
        anchor_resnames.add(nxt)
    elif res3 == "NME":
      prv = resnum_to_resname.get(num - 1)
      if prv and resname_to_type.get(prv) not in ("ACE", "NME"):
        anchor_resnames.add(prv)
  anchor_indices = [i for i in range(prot_len) if prot_data[i][2] == "CA"
                    and prot_data[i][0] in anchor_resnames and prot_data[i][0] in amino]
  anchor_indices = anchor_indices[1:-1] if len(anchor_indices) >= 2 else []
  if not anchor_indices:
    return [], []
  anchor_coords = prot_coords[anchor_indices]
  ca_indices = [i for i in range(prot_len) if prot_data[i][2] == "CA" and prot_data[i][0] in amino]
  if not ca_indices:
    return [], []
  ca_coords = prot_coords[ca_indices]
  keep_mask = np.any(min_image_cdist(ca_coords, anchor_coords, INPUT_BOX) <= 0.9, axis=1)
  protkeep = [ca_indices[i] for i in range(len(ca_indices)) if keep_mask[i]]
  if len(protkeep) < 2:
    return [], protkeep
  keep_distances = min_image_cdist(prot_coords[protkeep], prot_coords[protkeep], INPUT_BOX)
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

SS_BOND_NM = 0.25       # SG-SG of a disulfide is 0.20 nm, of two free cysteines >= 0.35


def residue_spans(resnums):
  """'1-261,270-280' for a set of residue numbers."""
  out, lo, prev = [], None, None
  for r in sorted(resnums):
    if lo is None:
      lo = prev = r
    elif r == prev + 1:
      prev = r
    else:
      out.append("%d" % lo if lo == prev else "%d-%d" % (lo, prev))
      lo = prev = r
  if lo is not None:
    out.append("%d" % lo if lo == prev else "%d-%d" % (lo, prev))
  return ",".join(out)


def network_bodies(prot_data, keep, pairs):
  """The bodies one partner's polymer makes under the elastic network.

  Its fragments are backbone_fragments of the polymer residues (an N/CA/C triple,
  or a cap) in -f, and two fragments are one body when a network pair or a
  disulfide joins them. Returns (fragments, network pairs between fragments,
  disulfides between fragments, bodies as sorted lists of residue numbers)."""
  names = {}
  for rec in prot_data:
    names.setdefault(rec[1], set()).add(rec[2])
  poly = {r for r, nm in names.items() if {"N", "CA", "C"} <= nm} | \
         {rec[1] for rec in prot_data if re.sub(r"^\d+", "", rec[0]) in pbc_tools.CAP_RESIDUES}
  rows = [k for k, rec in enumerate(prot_data) if rec[1] in poly and rec[2] in BACKBONE]
  if not rows:
    return 0, 0, 0, []
  frags = backbone_fragments(np.array([prot_data[k][4:7] for k in rows], dtype=float),
                             INPUT_BOX)
  frag_of = {}
  for f, piece in enumerate(frags):
    for r in piece:
      frag_of.setdefault(prot_data[rows[r]][1], f)
  parent = list(range(len(frags)))

  def find(x):
    while parent[x] != x:
      x = parent[x]
    return x

  def join(a, b):
    """Put fragments a and b in one body; True if they are two fragments."""
    if a is None or b is None or a == b:
      return False
    parent[find(a)] = find(b)
    return True

  n_en = sum(1 for i, j, _ in pairs
             if join(frag_of.get(prot_data[keep[i]][1]), frag_of.get(prot_data[keep[j]][1])))
  sg = [rec for rec in prot_data if rec[2] == "SG" and rec[1] in frag_of]
  n_ss = 0
  if len(sg) > 1:
    xyz = np.array([rec[4:7] for rec in sg], dtype=float)
    d = min_image_cdist(xyz, xyz, INPUT_BOX)
    for p, q in zip(*np.nonzero(np.triu(d <= SS_BOND_NM, 1))):
      n_ss += join(frag_of[sg[p][1]], frag_of[sg[q][1]])
  bodies = {}
  for r, f in frag_of.items():
    bodies.setdefault(find(f), []).append(r)
  return len(frags), n_en, n_ss, sorted(sorted(b) for b in bodies.values())


# One line per partner, so the pilot can see which structures run npt_c and nptrev
# as several network bodies (see the note at the top of PART 2).
for _label, _data, _keep, _pairs in (("receptor", prot1_data, protkeep1, en1dis),
                                     ("ligand", prot2_data, protkeep2, en2dis)):
  _nf, _nen, _nss, _bodies = network_bodies(_data, _keep, _pairs)
  sys.stderr.write("make_boresch: elastic network, %s: %d pairs; %d backbone "
                   "fragment(s), %d network pair(s) and %d disulfide(s) between them, "
                   "%d bod%s: %s%s\n"
                   % (_label, len(_pairs), _nf, _nen, _nss, len(_bodies),
                      "y" if len(_bodies) == 1 else "ies",
                      " | ".join(residue_spans(b) for b in _bodies),
                      "" if len(_bodies) < 2 else
                      "; npt_c and nptrev hold each body together, not the bodies "
                      "to each other"))

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

# Discard as equilibration before measuring, PER REPLICA. Derived from the probe
# length rather than set here, so lengthening the probe lengthens the discard with
# it: the point of running N of them for twice this long is that everything is
# measured on the second half of each, after they have forgotten the shared
# minimisation they all started from.
import fe_protocol as P
TRAJ_SKIP_PS = P.NPT_INIT_PS * P.PROBE_SKIP_FRAC

# ONE PULL DISTANCE, IN ONE UNIT. fe_protocol gives every u_from/u_to in nm (lambda
# is u/PULL_DIST there), and build_coords, pair_displacement, the stage rates and
# the far-end hold test use them as nm. The first v1.5 round read them as fractions
# of --pull-dist and multiplied by it once more, which agreed only while both
# numbers were 1.0: with PULL_DIST = 1.5, the ramp rescaled to end at 1.5 nm and
# --pull-dist 1.5, nptrev's Boresch r sat at ref_r + 2.25 nm while eq.32 released at
# ref_r + 1.50, and bindfwdI moved its references 0.8505 nm in a 0.567 nm stage
# (review R4). --pull-dist must still equal PULL_DIST: eq.32 releases the restraint
# at ref_r + --pull-dist, which is the state the legs end in only if the ramp ends
# at that distance too.
if abs(args.pull_dist - P.PULL_DIST) > 1e-9:
  abort("PULL_DIST_MISMATCH",
        "--pull-dist %g nm differs from fe_protocol.PULL_DIST %g nm. The ramp ends "
        "at PULL_DIST and eq.32 releases the Boresch restraint at ref_r + "
        "--pull-dist, so the two must be the same distance"
        % (args.pull_dist, P.PULL_DIST))

# The bound-state NPT every cycle runs before boundfwd1. job_fe.run copies its
# template into the structure directory with the leg mdps, and the elastic-network
# block is written into it at the end of PART 6. Checked here, before anything is
# measured, because a setup that cannot write it must not finish: npt_c would run
# without the network and switch it on at boundfwd1, the v1.3 defect.
NPT_C_MDP = "npt_c_fe.mdp"

# Which pull block each leg carries, keyed by fe_protocol.legs()'s "block". Every
# leg must name one, and every leg's mdp must already be here: write_pull_block
# appends, so a missing file would otherwise be CREATED holding nothing but a pull
# block, and only the cycle's grompp would notice.
PULL_BLOCKS = ("unbind", "bound")
_unblocked = [l.get("name", "?") for l in P.legs() if l.get("block") not in PULL_BLOCKS]
if _unblocked:
  abort("PROTOCOL_BLOCK_MISSING",
        "fe_protocol.legs() gives no valid 'block' (one of %s) for %s; the pull "
        "block a leg carries is never inferred from its name or kind"
        % ("/".join(PULL_BLOCKS), ", ".join(_unblocked)))
_absent = sorted({l["mdp"] for l in P.legs() if not os.path.isfile(l["mdp"])})
if _absent:
  abort("LEG_MDP_MISSING",
        "leg mdp(s) not in %s: %s" % (os.getcwd(), " ".join(_absent)))
if not os.path.isfile(NPT_C_MDP):
  abort("NPT_C_MDP_MISSING",
        "%s is not in %s; job_fe.run copies it with the leg mdps, and npt_c must "
        "carry the elastic network" % (NPT_C_MDP, os.getcwd()))

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

def write_ndx_group(path, name, atoms):
  """One index group, 15 atom numbers per line, which is the layout gmx itself
  writes. NOT one line: a single line of a few thousand numbers takes trjconv down
  with a glibc malloc assertion (GROMACS 2026.0, measured 2026-09-09: 3800 atoms on
  one line is fine, 4000 aborts, the same 4000 wrapped is fine). Every reader below
  used to write one line and swallow the abort, so on six HADDOCK setups the whole
  ensemble pass silently returned nothing and 20,000-40,000 candidates became
  springs at k = 0.3-0.6. The backbone selection never crossed the limit; the
  interface selection on a big interface does."""
  atoms = list(atoms)
  with open(path, "w") as f:
    f.write("[ %s ]\n" % name)
    for i in range(0, len(atoms), 15):
      f.write(" ".join(str(a) for a in atoms[i:i + 15]) + "\n")


def trjconv_failed(r, traj, gro, what):
  """True when a trjconv pass produced nothing -- and SAYS SO on stderr, with the
  last line trjconv wrote. The callers fall back when this is True, which is right;
  falling back without a word is what hid the index bug above for a whole run."""
  if r.returncode == 0 and os.path.isfile(gro):
    return False
  err = [ln for ln in (r.stderr or "").splitlines() if ln.strip()]
  sys.stderr.write("make_boresch: trjconv (%s) on %s exited %d%s; this replica is "
                   "skipped\n" % (what, traj, r.returncode,
                                  (": " + err[-1].strip()[:200]) if err else ""))
  return True



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
    box.append(box_matrix(L[i + 2 + n]))
    i += n + 3
  if not out:
    return (None, None) if with_box else None
  arr = np.asarray(out, dtype=float).reshape(len(out), natoms, 3)
  if not with_box:
    return arr
  return arr, np.asarray(box, dtype=float)


# Every atom's coordinates in args.input, by atom number, as PART 0 left them (one
# piece). The reference structure the anchor groups, their pbcatoms and the
# snapshot geometry are all taken from.
XYZ_OF = {rec[3]: np.array(rec[4:7], dtype=float) for rec in prot1_data + prot2_data}
# Each atom's position in args.input, the order pbcatom_of takes members in.
INPUT_ORDER = {rec[3]: k for k, rec in enumerate(prot1_data + prot2_data)}


def pbcatom_of(atoms):
  """The reference atom GROMACS is given for a multi-atom pull group: the member
  closest to the group's own centre in args.input (see PBCATOM in PART 5, which
  writes exactly this into every mdp). Members are taken in input order, the order
  that choice has always used, so a tie resolves the same way. They are put at
  their images nearest the first member before the centre is taken, so the choice
  does not depend on how -f is wrapped; for a group in one piece, which every
  anchor group is, nothing moves."""
  anums = sorted((a for a in set(atoms) if a in XYZ_OF), key=INPUT_ORDER.get)
  if not anums:
    return None
  a = np.array([XYZ_OF[x] for x in anums])
  a = nearest_image(a, a[0], INPUT_BOX)
  return int(anums[int(np.argmin(np.linalg.norm(a - a.mean(axis=0), axis=1)))])


def gromacs_com(X, M, w, pbc_row):
  """COM of one pull group per frame, taken the way GROMACS takes it for a group
  with a pbcatom: every member moved to its periodic image closest to the pbcatom,
  then the mass-weighted mean. X is (frames, natoms, 3) for the group's members,
  M is (frames, 3, 3), w the masses and pbc_row the pbcatom's row in X.

  A plain mean of the raw coordinates is what this replaces, and it is only right
  while every member already sits in one image. -pbc whole cannot promise that for
  a cutout: a group of residues from two unbonded fragments of one moleculetype is
  split whenever trjconv leaves the fragments in different images, and its COM
  then lands between the images, nowhere near the residues."""
  ref = X[:, pbc_row, :]
  d = min_image_vec(X - ref[:, None, :], M[:, None, :, :])
  return ref + (d * w[None, :, None]).sum(1) / w.sum()


def load_backbone_trajectory(atom_numbers):
  """Backbone coordinates and box over time, or (None, None) if the trajectory is
  unavailable.

  trjconv renumbers a subset on output, so the index is written here and the
  row order is known by construction.

  -pbc whole, not cluster: trjconv refuses to cluster a backbone-only index
  ("Molecule N marked for clustering but not atom 1 in it"). whole makes every
  covalently connected piece whole and nothing more, so a side built from several
  unbonded cutout fragments can come out with its fragments in different images.
  The boxes are returned so reassemble_backbone() can put them back together
  before the per-protein fit; the cross-protein geometry is minimum-imaged where
  it is computed.

  ALL REPLICAS ARE CONCATENATED, so anchor rigidity, the eps ranking and the frame
  rotation are ensemble quantities rather than properties of whichever single
  equilibration happened to run. That is the whole reason there are five of them."""
  if not atom_numbers or not TRAJ_PAIRS:
    return None, None
  ndx, gro = ".bb_sel.ndx", ".bb_traj.gro"
  out, boxes = [], []
  try:
    write_ndx_group(ndx, "bbsel", atom_numbers)
    for traj, tpr in TRAJ_PAIRS:
      if os.path.isfile(gro):
        os.remove(gro)
      cmd = ["gmx", "trjconv", "-s", tpr, "-f", traj, "-n", ndx,
             "-o", gro, "-b", "%g" % TRAJ_SKIP_PS, "-pbc", "whole"]
      r = subprocess.run(cmd, input="bbsel\n", capture_output=True, text=True)
      if trjconv_failed(r, traj, gro, "backbone"):
        continue
      part, pbox = read_multi_gro(gro, len(atom_numbers), with_box=True)
      if part is not None:
        out.append(part); boxes.append(pbox)
    if not out:
      return None, None
    return np.concatenate(out, axis=0), np.concatenate(boxes, axis=0)
  except (OSError, ValueError, IndexError):
    return None, None
  finally:
    for f in (ndx, gro):
      if os.path.isfile(f):
        os.remove(f)


# PUTTING THE BACKBONE BACK TOGETHER before it is fitted. A cutout side is a merge
# of capped, unbonded fragments, and -pbc whole leaves each fragment where mdrun
# wrapped it, so in some frames a fragment sits a box vector away from the rest.
# The per-side self-fit then reads that as motion: in 40 of the 46 setups of the
# second HADDOCK pass the backbone RMSF came out at 2.9-6.9 nm against the 1.0 nm
# guard (RMSF_SPLIT), the measured anchor search was abandoned, and every one of
# those structures was anchored by the burial heuristic, which carries no
# fluctuation information at all. All 36 setups with two or more merge points fell
# back; of the 10 with at most one, 6 got measured anchors.
#
# BY CONTACT, NOT BY CENTROID. Each fragment is moved by whole box vectors only, so
# no distance inside a fragment changes. Which box vector is decided by one contact
# chosen in -f, which PART 0 has made one piece: the fragment's backbone atom
# closest to anything already placed, and the atom it is closest to. In every frame
# the fragment goes to the image that puts that pair at its minimum image. A
# contact is well under 1 nm in a packed complex and stays so in every frame, far
# inside half a box, so the choice is never close, however the complex tumbles and
# however far the fragment sits from the root. The first v1.5 version placed each
# fragment by its centroid's offset from the root's, after fitting the root to -f,
# and that is ambiguous once the offset nears half a box, which happens in
# complexes that are one piece: a synthetic rigid side in the real 2KTF probe box
# (half-box 3.36 nm), a 5-residue fragment 3.6 nm from the root, came out non-physical
# in 198-200 of 200 frames, at RMSF 0.88 nm with the root group's eps 0.238 nm
# (above every EPS_LADDER rung) under 30 degrees of tumbling, and at 1.42 nm, over
# the guard, under 90 (review R2). Either way a rigid side fell back to burial.
#
# Both partners go into one tree, rooted at the largest fragment, so a fragment
# that touches only the other partner hangs from that contact rather than from a
# distant piece of its own side. The per-side fit downstream takes out each side's
# own translation, so the partners' relative image does not matter there. A
# fragment that really leaves its partner (the probe runs without the elastic
# network) is not hidden: it is put at its nearest image and its RMSF still trips
# the guard.


def contact_links(xyz, M, frags):
  """How to put fragments together by contact, decided on one frame.

  xyz (n, 3) is that frame and M its box; frags are the fragments as arrays of row
  indices, the root first. Prim's algorithm: the unplaced fragment closest to
  anything already placed is attached next, through its closest pair. Returns
  [(f, a, b, d)] in placement order: fragment f (an index into frags) hangs from
  row a of an already placed fragment through its own row b, d nm apart at the
  minimum image."""
  best, placed, links = {}, {0}, []

  def offer(g):
    for f in range(len(frags)):
      if f in placed:
        continue
      sub = min_image_cdist(xyz[frags[g]], xyz[frags[f]], M)
      i, j = np.unravel_index(int(np.argmin(sub)), sub.shape)
      cand = (float(sub[i, j]), int(frags[g][i]), int(frags[f][j]))
      if f not in best or cand[0] < best[f][0]:
        best[f] = cand

  if len(frags) < 2:
    return links
  offer(0)
  while best:
    f = min(best, key=lambda q: (best[q][0], q))
    d, a, b = best.pop(f)
    links.append((f, a, b, d))
    placed.add(f)
    offer(f)
  return links


def reassemble_backbone(X, M, ref, ref_box, n_first):
  """Backbone frames with every fragment moved by whole box vectors to the image
  its contact in -f puts it at.

  X is (frames, n, 3) in input order, M (frames, 3, 3), ref the same atoms in -f
  (one piece, PART 0) and ref_box its box. The first n_first rows are one side, the
  receptor, and the rest the other; fragments are found per side, so none spans
  both. Returns (X, fragments per side, frames per side in which that side's
  fragments were not all in one image). Frames that need no move come back bit for
  bit."""
  frags, side_of = [], []
  for s, rows in enumerate((np.arange(n_first), np.arange(n_first, len(ref)))):
    for f in backbone_fragments(ref[rows], ref_box):
      frags.append(rows[f])
      side_of.append(s)
  nfrag = [side_of.count(0), side_of.count(1)]
  if len(frags) < 2:
    return X, nfrag, [0, 0]
  # Largest first, the receptor's first among equals: the root is the bulk.
  order = sorted(range(len(frags)), key=lambda k: -len(frags[k]))
  frags, side_of = [frags[k] for k in order], [side_of[k] for k in order]
  out = np.array(X, dtype=float, copy=True)
  shift = np.zeros((len(X), len(frags), 3))
  for f, a, b, _d in contact_links(ref, ref_box, frags):
    dv = out[:, b, :] - out[:, a, :]
    s = min_image_vec(dv, M) - dv                 # exactly 0 where already in place
    out[:, frags[f], :] += s[:, None, :]
    shift[:, f, :] = s
  moved = []
  for side in (0, 1):
    ks = [k for k in range(len(frags)) if side_of[k] == side]
    sh = shift[:, ks, :]
    moved.append(int(np.any(np.abs(sh - sh[:, :1, :]) > 1e-6, axis=(1, 2)).sum())
                 if len(ks) > 1 else 0)
  return out, nfrag, moved


# --sd-max, or None when the user disabled it with a non-positive value. Bound
# here rather than read from args inside the routine so that the routine stays
# testable without an argv.
SD_MAX_NM = args.sd_max if (args.sd_max or 0) > 0 else None


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
  if not TRAJ_PAIRS:
    return pairs, "no usable --traj/--tpr; keeping the snapshot references"
  ndx, gro = ".if_sel.ndx", ".if_traj.gro"
  Xs, Ms, nrep = [], [], 0
  try:
    write_ndx_group(ndx, "ifsel", anums)
    for traj, tpr in TRAJ_PAIRS:
      if os.path.isfile(gro):
        os.remove(gro)
      r = subprocess.run(["gmx", "trjconv", "-s", tpr, "-f", traj,
                          "-n", ndx, "-o", gro, "-b", "%g" % TRAJ_SKIP_PS,
                          "-pbc", "whole"],
                         input="ifsel\n", capture_output=True, text=True)
      if trjconv_failed(r, traj, gro, "interface"):
        continue
      Xp, Mp = read_multi_gro(gro, len(anums), with_box=True)
      if Xp is not None:
        Xs.append(Xp); Ms.append(Mp); nrep += 1
  except (OSError, ValueError, IndexError) as e:
    return pairs, "%s; keeping the snapshot references" % e
  finally:
    for f in (ndx, gro):
      if os.path.isfile(f):
        os.remove(f)
  X = np.concatenate(Xs, axis=0) if Xs else None
  M = np.concatenate(Ms, axis=0) if Ms else None
  if X is None or len(X) < MIN_FRAMES:
    return pairs, ("only %s frames, need %d; keeping the snapshot references"
                   % (0 if X is None else len(X), MIN_FRAMES))
  row = {a: k for k, a in enumerate(anums)}
  ia = np.array([row[prot1_data[i][3]] for i, _, _ in pairs])
  ib = np.array([row[prot2_data[j][3]] for _, j, _ in pairs])
  # IN CHUNKS OVER PAIRS. min_image_dist builds a (frames, pairs, 27, 3) array of
  # image candidates, so this one call sizes the whole setup's memory: at 5005
  # pooled frames it is 3.2 MB per pair, which was 2.9 GB at the old 900-candidate
  # pass and would be 8 GB now that the pass is wide enough to be non-binding.
  # Chunking bounds it at CHUNK_BYTES whatever the candidate count, which is what
  # lets CANDIDATE_CUT be chosen on physics rather than on what fits.
  CHUNK_BYTES = 512 << 20
  per_pair = max(len(X) * len(_IMG) * 3 * 8, 1)
  blk = max(1, min(len(pairs), CHUNK_BYTES // per_pair))
  d = np.empty((len(X), len(pairs)), dtype=np.float64)
  for s in range(0, len(pairs), blk):
    e = min(s + blk, len(pairs))
    d[:, s:e] = min_image_dist(X[:, ia[s:e], :] - X[:, ib[s:e], :],
                               M[:, None, :, :])
  mean, sd = d.mean(0), d.std(0)
  snap = np.array([p[2] for p in pairs])
  shift = mean - snap
  sys.stderr.write(
      "make_boresch: interface references re-measured on %d frames pooled over "
      "%d replica(s) -- moved "
      "%+.3f A on average, rms %.3f A, worst %.3f A. Residual spread about the new "
      "reference is %.3f A rms, which is the part no choice of reference can "
      "remove. Removing the offset takes %.0f kJ/mol of systematic strain out of "
      "the bound leg.\n"
      % (len(X), nrep, 10.0 * shift.mean(),
         10.0 * float(np.sqrt((shift ** 2).mean())), 10.0 * float(np.abs(shift).max()),
         10.0 * float(sd.mean()),
         0.5 * (args.sum_k / len(pairs)) * float((shift ** 2).sum())))
  # THE SPREAD, not just the mean. A harmonic spring costs <(d-r0)^2> = var(d) at
  # r0 = <d>, so the per-pair variance IS that pair's contribution to the bound-leg
  # work and to its fluctuation. Pooling five replicas makes a new failure mode
  # possible that one replica could not show: a pair whose MEAN sits inside the
  # contact cutoff while it spends much of its time far outside it. Such a pair is
  # not a contact being restrained, it is an average of a contact and a gap, and it
  # pays full price in both work and noise.
  #
  # Reported rather than filtered, because how much intermittency is acceptable is
  # a modelling choice and dropping pairs raises k on the ones that remain.
  frac_out = (d > interfacecutoff).mean(0)
  order = np.argsort(sd)[::-1]
  q = np.percentile(sd, [50, 90, 99])
  worst = sd[order[0]]
  top10 = max(1, len(sd) // 10)
  share = float((sd[order[:top10]] ** 2).sum() / max((sd ** 2).sum(), 1e-12))
  n_int = int((frac_out > 0.25).sum())
  n_in = int(((mean <= interfacecutoff) & (frac_out > 0.25)).sum())
  sys.stderr.write(
      "make_boresch: per-pair distance spread over the pooled frames -- sd median "
      "%.3f A, p90 %.3f A, p99 %.3f A, worst %.3f A. The widest 10%% of pairs carry "
      "%.0f%% of sum(sd^2), which is what the bound leg pays. %d pairs are outside "
      "the %.2f nm cutoff in more than a quarter of frames, %d of them while their "
      "MEAN is still inside it: those are intermittent contacts, not restrained "
      "ones.\n"
      % (10 * q[0], 10 * q[1], 10 * q[2], 10 * worst, 100 * share,
         n_int, interfacecutoff, n_in))
  if SD_MAX_NM is not None:
    drop = sd > SD_MAX_NM
    # A ceiling that would leave almost nothing is refused rather than obeyed. The
    # springs exist to hold the pose while the Boresch set takes over, and an
    # interface pinned at a handful of points is a worse failure than one pinned at
    # some wide ones. On test13/14/15 this ceiling drops 5-8%, nowhere near the
    # floor, so tripping it means the interface is unlike anything measured here
    # and the ceiling is the wrong tool for it.
    #
    # COUNTED AMONG THE PAIRS THAT WILL ACTUALLY BE RESTRAINED, not among all
    # candidates. The two were the same number while candidates were enumerated at
    # the contact cutoff; now that the pass is wide enough to be non-binding, most
    # candidates are destined for the mean cutoff below and counting them here
    # would let the floor be cleared by pairs that never become springs -- the
    # guard would go quiet exactly as the set it protects got small.
    n_left = int((sd <= SD_MAX_NM)[mean <= interfacecutoff].sum())
    if drop.all() or n_left < SD_MIN_KEEP:
      sys.stderr.write(
          "make_boresch: --sd-max %.3f nm would leave only %d springs inside the "
          "%.2f nm contact cutoff, below the floor of %d; KEEPING ALL PAIRS. This "
          "interface is much more mobile than the ceiling assumes -- check the "
          "spread line above before trusting the bound leg.\n"
          % (SD_MAX_NM, n_left, interfacecutoff, SD_MIN_KEEP))
    elif drop.any():
      # The saving is quoted against the pairs the MEAN cutoff would have kept
      # anyway, not against all candidates. Most wide pairs also sit far out, so
      # the two filters overlap heavily and quoting the whole candidate set would
      # credit this ceiling with removals the cutoff below was going to make. On
      # test13/14/15 the honest figure is 22-37%; the candidate-set figure is
      # 84-93%, and would be nonsense.
      surv = mean <= interfacecutoff
      s_all = float((sd[surv] ** 2).sum())
      s_cut = float((sd[surv & drop] ** 2).sum())
      sys.stderr.write(
          "make_boresch: --sd-max %.3f nm drops %d of %d candidates (%.0f%%). Of "
          "the pairs the contact cutoff would have kept regardless, it removes "
          "%d of %d and %.0f%% of their sum(sd^2), which is the part of the bound "
          "leg's work and fluctuation this actually saves.\n"
          % (SD_MAX_NM, int(drop.sum()), len(sd), 100.0 * drop.mean(),
             int((surv & drop).sum()), int(surv.sum()),
             100.0 * s_cut / max(s_all, 1e-12)))
      keep = ~drop
      pairs = [p for p, k in zip(pairs, keep) if k]
      mean = mean[keep]
  return ([(i, j, float(m)) for (i, j, _), m in zip(pairs, mean)],
          "mean over %d frames from %d replica(s)" % (len(X), nrep))


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
  args.input is one piece (PART 0 stops the setup otherwise), and it is also the
  snapshot the interface restraints and the elastic network are measured on, so
  the whole restraint set refers to one structure. The COM is still taken about
  the group's pbcatom, as GROMACS takes it, so it does not depend on how -f is
  wrapped either; for a group in one piece that is the plain weighted mean, bit
  for bit. eps and relative_rotation stay on com_t: both are insensitive to
  imaging."""
  atoms = [by_res[r][a] for r in residues for a in BACKBONE]
  if len(atoms) < N_MIN_ATOMS:
    return None
  cols = [idx_of[a] for a in atoms]
  w = np.array([mass_of(by_res[r][a], a) for r in residues for a in BACKBONE])
  com_t = (fit[:, cols, :] * w[None, :, None]).sum(1) / w.sum()
  eps = float(np.sqrt(((com_t - com_t.mean(0)) ** 2).sum(-1).mean()))
  xyz = np.array([xyz_of[a] for a in atoms])
  xyz = nearest_image(xyz, xyz[atoms.index(pbcatom_of(atoms))], INPUT_BOX)
  atomnames = [a for _r in residues for a in BACKBONE]
  return {"atoms": atoms, "atomnames": atomnames,
          "com": (xyz * w[:, None]).sum(0) / w.sum(),
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


def group_weights(g):
  """Mass weights for a group's atoms, whatever built it.

  The measured path's make_group records "atomnames"; the burial fallback builds
  its triads straight from BACKBONE and records none, so reading the key blindly
  turned every burial-path setup into a KeyError the moment the reference geometry
  started being averaged over the probe ensemble. It surfaced only when a bug
  forced 2KTF down the fallback, which is exactly the kind of path that is never
  exercised until it matters.
  """
  a = g["atoms"]
  names = g.get("atomnames")
  if not names or len(names) != len(a):
    names = [BACKBONE[k % len(BACKBONE)] for k in range(len(a))]
  return np.array([mass_of(x, nm) for x, nm in zip(a, names)])


def ensemble_cross_r(cand):
  """Mean cross distance over the pooled probe frames, for many candidate pairs.

  WHY THIS EXISTS. The anchor search rejects a triad whose r + pull_dist exceeds
  the box budget, and the final guard aborts on the same budget -- but until now
  the search measured r on the SNAPSHOT while the guard measured it on the
  ENSEMBLE mean, and ensemble averaging moves r. On test23 the snapshot gave
  1.306 nm, which passes (2.306 < 2.467), and the pooled mean gave 1.490, which
  does not (2.490 > 2.467): the setup was selected, ran its five 20 ns probes, and
  then aborted, losing the whole run. Searching on the quantity the guard enforces
  closes that dead band, exactly as matching R_LIMIT_FRAC closed the earlier one
  between 0.9 and 0.6.

  `cand` maps a key to a group dict with "atoms" and "atomnames". Returns (COM
  per key, (frames, 3) each; boxes, (frames, 3, 3)), from which the caller takes
  every cross distance, or None if the trajectory cannot be read, in which case
  the caller keeps the snapshot behaviour.

  One trjconv pass over the union of candidate atoms serves every pair, so the
  cost is one extraction per replica per search round, not one per pair. Each atom
  is extracted ONCE however many groups hold it: the groups are kept per triad
  (see search), which on 1IJK makes 180 groups of largely the same atoms where
  there were 52, and repeating them would multiply the frames read at 5005 per
  setup by that much.
  """
  if not TRAJ_PAIRS or not cand:
    return None
  keys = sorted(cand)
  anums = sorted({a for k in keys for a in cand[k]["atoms"]})
  row = {a: i for i, a in enumerate(anums)}
  ndx, gro = ".xr_sel.ndx", ".xr_traj.gro"
  Xs, Ms = [], []
  try:
    write_ndx_group(ndx, "xrsel", anums)
    for traj, tpr in TRAJ_PAIRS:
      if os.path.isfile(gro):
        os.remove(gro)
      r = subprocess.run(["gmx", "trjconv", "-s", tpr, "-f", traj, "-n", ndx,
                          "-o", gro, "-b", "%g" % TRAJ_SKIP_PS, "-pbc", "whole"],
                         input="xrsel\n", capture_output=True, text=True)
      if trjconv_failed(r, traj, gro, "anchor candidates"):
        continue
      Xp, Mp = read_multi_gro(gro, len(anums), with_box=True)
      if Xp is not None:
        Xs.append(Xp); Ms.append(Mp)
  except (OSError, ValueError, IndexError):
    return None
  finally:
    for f in (ndx, gro):
      if os.path.isfile(f):
        os.remove(f)
  if not Xs:
    return None
  X = np.concatenate(Xs, axis=0)
  M = np.concatenate(Ms, axis=0)
  if len(X) < MIN_FRAMES:
    return None
  # Each COM the way GROMACS will take it once the group is a pull group, about
  # the pbcatom PBCATOM would give it. The final guard measures r on exactly these
  # COMs (ensemble_boresch_geometry), so the search has to as well, or the dead
  # band this function closed reopens for any group whose atoms -pbc whole split.
  com = {}
  for k in keys:
    a = list(cand[k]["atoms"])
    w = group_weights(cand[k])
    com[k] = gromacs_com(X[:, [row[x] for x in a], :], M, w, a.index(pbcatom_of(a)))
  return com, M


def snapshot_cross_r(gp, gl):
  """r between two groups in -f, taken exactly as snapshot_boresch_geometry takes
  it from P3 and L1: each COM about its group's pbcatom, the vector between them
  minimum-imaged in -f's box. For the pair that wins, this IS SNAP_REF["r"], the
  value the pose gate falls back to."""
  com = []
  for g in (gp, gl):
    a = list(g["atoms"])
    X = np.array([[XYZ_OF[x] for x in a]], dtype=float)
    com.append(gromacs_com(X, INPUT_BOX[None], group_weights(g), a.index(pbcatom_of(a))))
  return float(np.linalg.norm(min_image_vec(com[1] - com[0], INPUT_BOX[None]), axis=-1)[0])


def climb_ladder(search, ladder=EPS_LADDER, rot_max=ROT_MAX_DEG, log=None):
  """Walk an eps ladder and return (result, eps) for the set that is accepted.

  Split out of try_measured_anchors so the ACCEPTANCE POLICY can be exercised on
  its own: the search itself needs a trajectory, five probe replicas and a tpr,
  and a policy that is only ever read cannot be checked against the sentence that
  describes it. `search(eps)` returns None or a tuple whose first element is the
  relative frame rotation in degrees, smaller being better.

  The policy, in one line: the first rung that yields a set UNDER rot_max wins;
  if none does, the best rung's set is used and the caller warns."""
  best, best_eps = None, None
  for eps in ladder:
    if log:
      log("make_boresch: --- anchor search at eps_max %.3f nm ---\n" % eps)
    r = search(eps)
    if r is None:
      if log:
        log("make_boresch: eps_max %.3f found nothing, relaxing\n" % eps)
      continue
    if best is None or r[0] < best[0]:
      best, best_eps = r, eps
    if r[0] <= rot_max:
      break
    if log:
      log("make_boresch: eps_max %.3f gives %.1f deg of relative frame rotation, "
          "over the %.1f limit; trying the next rung\n" % (eps, r[0], rot_max))
  return best, best_eps


def try_measured_anchors():
  """Anchors chosen from measured rigidity, or None to fall back to burial.

  Returns (rec_groups, lig_groups, anchors, frame_rotation_deg)."""
  bb1 = polymer_backbone(prot1_data)
  bb2 = polymer_backbone(prot2_data)
  bb = bb1 + bb2
  traj, boxes = load_backbone_trajectory(bb)
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
  # Every fragment back into one image before anything is fitted, both partners in
  # one tree; see reassemble_backbone. The per-side fit below then treats each side
  # as one body, while the partners stay free relative to each other.
  ref = np.array([XYZ_OF[a] for a in bb])
  traj, nfrag, nmoved = reassemble_backbone(traj, boxes, ref, INPUT_BOX, len(bb1))
  for label, k, anums in (("receptor", 0, bb1), ("ligand", 1, bb2)):
    if anums:
      sys.stderr.write("make_boresch: %s backbone is %d fragment(s); %d of %d frames "
                       "had a fragment in another periodic image and were "
                       "reassembled before the self-fit\n"
                       % (label, nfrag[k], nmoved[k], len(traj)))
  row_of = {a: i for i, a in enumerate(bb)}
  a1, fit1, r1 = measure_rigidity(prot1_data, traj, row_of)
  a2, fit2, r2 = measure_rigidity(prot2_data, traj, row_of)
  # The fragments are reassembled above, so RMSF of order the box no longer means
  # a split image. What still reaches it is real: a piece of one side moving that
  # far relative to the rest during the probe (which runs without the elastic
  # network), or a fragment that strayed more than half a box from where -f puts
  # it. Either way the side is not one rigid body and the measured search does not
  # apply.
  worst = max([max(r1.values()) if r1 else 0.0,
               max(r2.values()) if r2 else 0.0])
  if worst > RMSF_SPLIT:
    sys.stderr.write("make_boresch: backbone RMSF reaches %.2f nm even with every "
                     "fragment reassembled, so part of a partner moves that far "
                     "relative to the rest; falling back to the burial heuristic\n"
                     % worst)
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
    """One round at a given rigidity threshold. (rot, anchors, g1, g2, (ensemble
    r, snapshot r)) or None."""
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

    # The cross distance is measured on the ENSEMBLE, not on the snapshot,
    # because that is what the final guard aborts on. One trjconv pass over every
    # candidate anchor group in this round serves all pairs; if the trajectory
    # cannot be read we fall back to the snapshot, which is the old behaviour.
    #
    # EVERY GROUP AS ITS OWN TRIAD BUILT IT. partition_triad splits a seed's
    # neighbourhood against the other two seeds of the triad, so a seed shared by
    # several triads has a different group, and a different COM, in each. Keyed by
    # seed alone, a combination's r was that of whichever triad registered the seed
    # last, not that of the groups the guard then measures.
    cand = {}
    for i, (Pt, d1) in enumerate(D1):
      for p in Pt:
        cand[("P", i, p)] = d1[p]
    for j, (Lt, d2) in enumerate(D2):
      for l in Lt:
        cand[("L", j, l)] = d2[l]
    ens = ensemble_cross_r(cand)
    if ens is None:
      sys.stderr.write("make_boresch: cross distances from the SNAPSHOT (no usable "
                       "trajectory); the box budget is checked on that\n")
    else:
      com_e, M_e = ens
      sys.stderr.write("make_boresch: cross distances from %d pooled frames over "
                       "%d candidate groups\n" % (len(M_e), len(cand)))

    def cross_r(i, p, j, l):
      if ens is None:
        return snapshot_cross_r(D1[i][1][p], D2[j][1][l])
      return float(np.mean(np.linalg.norm(
          min_image_vec(com_e[("L", j, l)] - com_e[("P", i, p)], M_e), axis=1)))

    best = None
    n_over = 0
    for i, (Pt, d1) in enumerate(D1):
      for j, (Lt, d2) in enumerate(D2):
        # P3 and L1 carry r: take the closest cross pair of the two triads.
        pairs = [(cross_r(i, p, j, l), p, l) for p in Pt for l in Lt]
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
        #
        # ON THE LARGER OF THE TWO r THE SETUP CAN WRITE. The guard checks the r
        # that is finally emitted: the ensemble mean, or the snapshot's when the
        # pose gate replaces the mean, which it does for all six coordinates when
        # any one of them fails. Budgeting the ensemble r alone let a triad through
        # here whose snapshot r did not fit: 2KTF in a 7.15 nm box (budget 2.102
        # nm) with the gate set to 2 deg accepted r 1.091 here and then stopped
        # with BORESCH_R_TOO_LARGE on the snapshot's 1.112, after the five probes
        # (review R1). Which of the two is written is only known once the gate has
        # seen all six ensemble means, so the budget takes the larger.
        r_snap = snapshot_cross_r(d1[P3], d2[L1])
        if PULL_LIMIT and max(r_cross, r_snap) + args.pull_dist > R_LIMIT_FRAC * PULL_LIMIT:
          n_over += 1
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
                            "L1": L1, "L2": L2, "L3": L3}, d1, d2, (r_cross, r_snap))
    if best is None:
      sys.stderr.write("make_boresch: no anchor set met the angle window at "
                       "eps_max %.3f (%d combinations were over the box budget "
                       "of %.3f nm)\n"
                       % (eps_max, n_over,
                          R_LIMIT_FRAC * PULL_LIMIT if PULL_LIMIT else float('inf')))
    elif n_over:
      sys.stderr.write("make_boresch: %d triad combinations rejected for exceeding "
                       "the box budget (the larger of the ensemble and snapshot r, "
                       "plus pull_dist, > %.3f nm); the accepted set sits at %.3f "
                       "(snapshot %.3f)\n"
                       % (n_over, R_LIMIT_FRAC * PULL_LIMIT, best[4][0], best[4][1]))
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
  # ROT_MAX_DEG IS AN ACCEPTANCE TEST, NOT A LABEL. The note above already says so
  # -- "a loose ceiling that passes the rotation check has earned it while one that
  # does not is rejected regardless of which rung found it" -- but the code used to
  # print the limit and then keep the set anyway. Two of eight runs of the same
  # structure took a set the stated criterion rejects (test15 at 8.1 deg, test25 at
  # 8.9) with two and three rungs still untried, and those two are the dissipation
  # tail: 61.3 and 75.7 kJ/mol per cycle against 56.8-64.9 for the six under
  # 5.5 deg, with test25 carrying the worst ramp stage in the whole set at 24.7.
  # Over the eight, rotation correlates +0.68 with the cycle's total dissipation
  # and +0.64 with stage E's.
  #
  # Relaxing eps admits MORE groups, so the pool the search minimises rotation over
  # grows monotonically and the best achievable rotation can only improve -- except
  # through the 0.5 nm thinning and TOP_PER_SIDE, which can evict a good group when
  # more candidates appear. Hence "keep looking, remember the best so far" rather
  # than "take the last rung": a later rung is used only if it is actually better.
  #
  # First success that CLEARS the limit still wins, so nothing changes for a
  # structure whose tightest rung already passes, which is six of the eight.
  chosen, chosen_eps = climb_ladder(search, log=sys.stderr.write)
  if chosen is None:
    sys.stderr.write("make_boresch: no measured anchor set at any eps_max in %s, "
                     "falling back to the burial heuristic\n" % (EPS_LADDER,))
    return None

  rot, anch, g1, g2 = chosen[:4]
  sys.stderr.write("make_boresch: accepted eps_max %.3f\n" % chosen_eps)
  sys.stderr.write("make_boresch: selected anchors give %.1f deg RMS relative "
                   "frame rotation over the equilibration (limit %.1f)\n"
                   % (rot, ROT_MAX_DEG))
  if rot > ROT_MAX_DEG:
    sys.stderr.write("make_boresch: WARNING - no rung of %s met the %.1f deg limit; "
                     "the best was %.1f deg at eps_max %.3f. The Boresch frame will "
                     "not hold orientation well and dG_release will overstate the "
                     "confinement\n" % (EPS_LADDER, ROT_MAX_DEG, rot, chosen_eps))
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


def circular_mean_deg(a):
  """Mean of angles in degrees, taken on the circle.

  A PLAIN MEAN IS WRONG HERE AND WRONG SILENTLY. Dihedrals live on (-180, 180],
  so two frames at +179 and -179 are 2 degrees apart and average to 180, while
  arithmetic gives 0 -- a reference pointing the opposite way, which is exactly
  the shape of the dihedral_deg sign bug that voided every work integral before
  b4c267f. Averaging the unit vectors instead has no wrap to get wrong.

  Returns the mean and the circular standard deviation, both in degrees; the
  spread is what says whether averaging these frames was meaningful at all."""
  r = np.radians(np.asarray(a, float))
  c, s_ = np.cos(r).mean(), np.sin(r).mean()
  # CLAMPED: on identical inputs the resultant comes out at 1 + 1e-16, and then
  # log is positive and the sqrt below is a domain error rather than a spread of
  # zero. Perfect agreement is the easiest case to hit and the easiest to miss.
  R = min(1.0, math.hypot(c, s_))
  mean = math.degrees(math.atan2(s_, c))
  sd = math.degrees(math.sqrt(-2.0 * math.log(R))) if R > 1e-12 else float('nan')
  return mean, sd


def wrap180(x):
  return (x + 180.0) % 360.0 - 180.0


# THE BORESCH GEOMETRY IS TAKEN THE WAY GROMACS EVALUATES IT, in the snapshot and in
# every ensemble frame alike. Two rules, both from the pull code:
#
#   * a multi-atom group's COM is taken with every member at its image closest to
#     the group's pbcatom (gromacs_com, with the pbcatom PBCATOM writes);
#   * each coordinate is built from vectors BETWEEN group COMs, and each of those
#     vectors is minimum-imaged on its own, from the first group of a pair to the
#     second. Read off the groups build_coords writes:
#
#       r        [P3, L1]                 P3->L1
#       theta_A  [P3, P2, P3, L1]         P3->P2, P3->L1           angle at P3
#       theta_B  [L1, P3, L1, L2]         L1->P3, L1->L2           angle at L1
#       phi_A    [P1, P2, P2, P3, P3, L1] P1->P2, P2->P3, P3->L1
#       phi_B    [P2, P3, P3, L1, L1, L2] P2->P3, P3->L1, L1->L2
#       phi_C    [P3, L1, L1, L2, L2, L3] P3->L1, L1->L2, L2->L3
#
# v1.4 took raw COMs of -pbc whole frames and minimum-imaged only P3->L1, carrying
# the rest of each triad along as raw offsets. That is right only while every
# group and each partner's triad sit in one image, which a cutout made of unbonded
# fragments does not promise. 1AKJ, 1E6J and 1IQD of the second HADDOCK pass got a
# dihedral reference 103-115 degrees from their snapshot pose (1AKJ phi_A snapshot
# -118.4, emitted 138.7, production -113 +- 4), stage I then dissipated 172-449
# kJ/mol per direction against a median of 18.8, and their pKD intervals came out
# at +-3.8 to +-25 against a median of +-2.1. On split synthetic frames the v1.4
# arithmetic moves the references by 29-68 degrees (tests/test_boresch_pbc.py), and
# on 51 raw 2KTF probe frames it disagrees with GROMACS's own pull evaluation by
# up to 180 degrees, where this one agrees to 5e-4.
BORESCH_ROLES = ("P1", "P2", "P3", "L1", "L2", "L3")
BORESCH_VECTORS = (("P3", "L1"), ("P3", "P2"), ("L1", "P3"), ("L1", "L2"),
                   ("P1", "P2"), ("P2", "P3"), ("L2", "L3"))
# The six coordinates in the order build_coords emits them.
BORESCH_KEYS = ("r", "thA", "thB", "phA", "phB", "phC")
BORESCH_NAMES = {"r": "r", "thA": "theta_A", "thB": "theta_B",
                 "phA": "phi_A", "phB": "phi_B", "phC": "phi_C"}


def boresch_series(com, M):
  """The six Boresch coordinates per frame from per-frame group COMs.

  com maps role -> (frames, 3); M is (frames, 3, 3). Returns a dict of arrays: r
  in nm, theta_A/theta_B (thA, thB) and phi_A/B/C (phA, phB, phC) in degrees.
  angle_deg and dihedral_deg take POINTS, so each coordinate is handed a chain of
  points built from its own minimum-imaged vectors, starting at the origin: the
  same two definitions the snapshot always used, now fed the vectors GROMACS
  uses. Their argument order is the one the readback verified against GROMACS:
  angle_deg(a, b, c) is the angle AT b, so theta_A is the angle at P3 between P2
  and L1 and theta_B the angle at L1 between P3 and L2."""
  v = {p: min_image_vec(com[p[1]] - com[p[0]], M) for p in BORESCH_VECTORS}
  o = np.zeros(3)
  n = len(v[("P3", "L1")])

  def dihedral(b1, b2, b3):
    return np.array([dihedral_deg(o, b1[i], b1[i] + b2[i], b1[i] + b2[i] + b3[i])
                     for i in range(n)])

  return {
      "r": np.linalg.norm(v[("P3", "L1")], axis=-1),
      "thA": np.array([angle_deg(v[("P3", "P2")][i], o, v[("P3", "L1")][i])
                       for i in range(n)]),
      "thB": np.array([angle_deg(v[("L1", "P3")][i], o, v[("L1", "L2")][i])
                       for i in range(n)]),
      "phA": dihedral(v[("P1", "P2")], v[("P2", "P3")], v[("P3", "L1")]),
      "phB": dihedral(v[("P2", "P3")], v[("P3", "L1")], v[("L1", "L2")]),
      "phC": dihedral(v[("P3", "L1")], v[("L1", "L2")], v[("L2", "L3")]),
  }


def role_slices(groups):
  """(atom numbers of all six groups concatenated, role -> slice into them)."""
  anums, slices = [], {}
  for role in BORESCH_ROLES:
    a = groups[role]["atoms"]
    slices[role] = slice(len(anums), len(anums) + len(a))
    anums += list(a)
  return anums, slices


def group_coms(groups, X, M, slices):
  """Per-frame COM of every role, about the pbcatom PBCATOM gives that group."""
  com = {}
  for role in BORESCH_ROLES:
    a = list(groups[role]["atoms"])
    w = group_weights(groups[role])
    com[role] = gromacs_com(X[:, slices[role], :], M, w, a.index(pbcatom_of(a)))
  return com


def snapshot_boresch_geometry(groups):
  """The six coordinates in -f alone, taken exactly as every ensemble frame is:
  COMs about each group's pbcatom and minimum-imaged vectors, in -f's own box.
  Returns (dict of six floats, the minimum-imaged P3->L1 vector)."""
  anums, slices = role_slices(groups)
  X = np.array([[XYZ_OF[a] for a in anums]], dtype=float)
  M = INPUT_BOX[None, :, :]
  com = group_coms(groups, X, M, slices)
  s = boresch_series(com, M)
  return ({k: float(s[k][0]) for k in s},
          min_image_vec(com["L1"] - com["P3"], M)[0])


# THE POSE GATE. An ensemble mean is only a reference if it is a pose: a circular
# mean over frames that disagree about the orientation lands where no frame is.
# Each of the five angles and dihedrals is compared with the bound pose of -f by
# circular difference, and r by plain difference; if any one is off by more than
# its limit, ALL SIX snapshot values are emitted, so the restraint still describes
# one consistent pose of the bound complex rather than a mix of two.
#
# 15 degrees is the limit the second HADDOCK pass's QC recommended for the emitted
# reference against the bound geometry; the four structures it caught were 53-114
# degrees off, the largest offset among the other 39 was 27 (1OPH phi_A). 0.12 nm
# is just above the largest |r_ensemble - r_snapshot| any of the 46 setups showed,
# 0.115 nm (1R0R); r carries the pull and the box budget, so a mean that moves it
# further than that is not the distance of any frame the anchors were chosen on.
POSE_GATE_DEG = 15.0
POSE_GATE_NM = 0.12


def pose_gate(ens, snap):
  """[(coordinate, ensemble value, snapshot value, |difference|)] for every
  coordinate on which the ensemble mean is further from the snapshot than the
  gate allows. Empty means the mean is a pose."""
  bad = []
  d = abs(ens["r"] - snap["r"])
  if d > POSE_GATE_NM:
    bad.append(("r", ens["r"], snap["r"], d))
  for key in ("thA", "thB", "phA", "phB", "phC"):
    d = abs(wrap180(ens[key] - snap[key]))
    if d > POSE_GATE_DEG:
      bad.append((key, ens[key], snap[key], d))
  return bad


def choose_reference(ens, snap):
  """(the six references to emit, what pose_gate found). The ensemble mean if it
  passes the gate, otherwise the snapshot for all six coordinates."""
  bad = pose_gate(ens, snap)
  return (dict(snap) if bad else dict(ens)), bad


def ensemble_boresch_geometry(groups):
  """Boresch reference geometry averaged over every probe replica.

  `groups` maps role -> the group dict make_group built, whose "atoms" are the
  backbone atom numbers of that anchor cluster. Their COMs are recomputed frame by
  frame from the pooled trajectory, which is the same measurement the single
  snapshot used to supply -- only now it is an average over the ensemble the
  production replicas are drawn from rather than one arbitrary configuration.

  trjconv gives -pbc whole, which leaves the two partners, and the unbonded
  fragments of one partner, free to sit in different periodic images. Every COM
  and every vector is therefore taken the way GROMACS takes it; see the note at
  BORESCH_VECTORS.

  Returns (dict of the six references, dict of their spreads, nframes) or
  (None, None, 0) if the trajectory cannot be read.
  """
  anums, slices = role_slices(groups)
  if not TRAJ_PAIRS:
    return None, None, 0
  ndx, gro = ".bor_sel.ndx", ".bor_traj.gro"
  Xs, Ms = [], []
  try:
    write_ndx_group(ndx, "borsel", anums)
    for traj, tpr in TRAJ_PAIRS:
      if os.path.isfile(gro):
        os.remove(gro)
      r = subprocess.run(["gmx", "trjconv", "-s", tpr, "-f", traj, "-n", ndx,
                          "-o", gro, "-b", "%g" % TRAJ_SKIP_PS, "-pbc", "whole"],
                         input="borsel\n", capture_output=True, text=True)
      if trjconv_failed(r, traj, gro, "Boresch groups"):
        continue
      Xp, Mp = read_multi_gro(gro, len(anums), with_box=True)
      if Xp is not None:
        Xs.append(Xp); Ms.append(Mp)
  except (OSError, ValueError, IndexError):
    return None, None, 0
  finally:
    for f in (ndx, gro):
      if os.path.isfile(f):
        os.remove(f)
  if not Xs:
    return None, None, 0
  X = np.concatenate(Xs, axis=0)
  M = np.concatenate(Ms, axis=0)
  if len(X) < MIN_FRAMES:
    return None, None, len(X)

  # Mass-weighted COM per role per frame, with the same weights make_group used,
  # each about its group's pbcatom; then the six coordinates from the same
  # minimum-imaged vectors and the same definitions as the snapshot.
  s = boresch_series(group_coms(groups, X, M, slices), M)

  # Angles are confined to [0, 180] and average arithmetically; dihedrals wrap and
  # do not. Getting that backwards is silent, so the two are not treated alike.
  ref = {"r": float(s["r"].mean()),
         "thA": float(s["thA"].mean()), "thB": float(s["thB"].mean())}
  sd = {"r": float(s["r"].std(ddof=1)),
        "thA": float(s["thA"].std(ddof=1)), "thB": float(s["thB"].std(ddof=1))}
  for key in ("phA", "phB", "phC"):
    ref[key], sd[key] = circular_mean_deg(s[key])
  return ref, sd, len(X)

def _grp(name):
  return rec_groups.get(anchors[name]) or lig_groups.get(anchors[name])

def gc(name):
  return _grp(name)["com"]

P1c, P2c, P3c = gc("P1"), gc("P2"), gc("P3")
L1c, L2c, L3c = gc("L1"), gc("L2"), gc("L3")

# Reference geometry (Boresch definition), AVERAGED OVER THE PROBE ENSEMBLE.
#
# It used to come from one clustered snapshot, and that is the same defect the
# interface references had: a restraint referenced to one configuration charges
# every production replica for the difference, and the difference is a draw. Three
# repeats of one protocol scattered 14.4 kJ/mol on dG_bind against a within-run
# bootstrap of 4.4, because all 50 cycles of a run share one setup and resampling
# cycles cannot see it.
#
# The snapshot values are still computed, and are used if the trajectory cannot be
# read, so a setup with no probe still produces a consistent restraint set -- the
# old behaviour, named as such rather than arrived at silently. They are also what
# the pose gate below falls back to.
_ANCHOR_GROUPS = {role: _grp(role) for role in BORESCH_ROLES}

# The single-frame geometry of args.input, kept whatever the references become.
# It is what GROMACS can be asked to reproduce, so it is what the readback below
# compares against; see the note there. Taken like every ensemble frame, about
# each group's pbcatom and with minimum-imaged vectors in -f's own box.
SNAP_REF, _snap_p3l1 = snapshot_boresch_geometry(_ANCHOR_GROUPS)

# The pull axis is P3->L1 of the snapshot, the difference of the two anchor COMs,
# which leaves every interface path bit for bit what v1.4 wrote. -f is one piece
# (PART 0), so that difference is the minimum-imaged vector GROMACS measures; it is
# taken from the minimum image should the two ever disagree.
#
# The first v1.5 round warned here when the anchor geometry on raw COMs and the
# minimum-imaged one disagreed, and said the warning also caught a split -f. It
# could not: it looked at the six anchor groups only, and the split 1IJK frame
# passed it in silence with ligand residues 300-416 a box vector off (review FR2).
# PART 0 checks every polymer residue instead.
_PULL_VEC = (L1c - P3c) if np.linalg.norm((L1c - P3c) - _snap_p3l1) <= 1e-6 \
            else _snap_p3l1

ref_r, ref_thA, ref_thB = SNAP_REF["r"], SNAP_REF["thA"], SNAP_REF["thB"]
ref_phA, ref_phB, ref_phC = SNAP_REF["phA"], SNAP_REF["phB"], SNAP_REF["phC"]
REF_SOURCE = "single reference frame"

_ens, _esd, _enf = ensemble_boresch_geometry(_ANCHOR_GROUPS)
if _ens:
  # The three dihedral shifts are DIFFERENCES OF ANGLES and have to be wrapped to
  # (-180, 180] like everything else about a dihedral. Unwrapped, test32 reported
  # phi_A as having "moved 353.6 deg" when it moved -6.4, which reads as the
  # reference having been thrown away and is the same wrap the circular mean exists
  # to handle. Diagnostic only, but a diagnostic nobody can believe is worse than
  # none: this line is what a reader checks before trusting dG_release.
  sys.stderr.write(
      "make_boresch: Boresch geometry averaged over %d pooled frames from %d "
      "replica(s). r %.4f -> %.4f nm (sd %.4f); angles/dihedrals moved "
      "%.1f/%.1f/%.1f/%.1f/%.1f deg with circular sd %.1f/%.1f/%.1f on the "
      "dihedrals. A dihedral sd near 90 deg means the frames disagree about the "
      "orientation and the mean is not a pose.\n"
      % (_enf, len(TRAJ_PAIRS), SNAP_REF["r"], _ens["r"], _esd["r"],
         _ens["thA"] - SNAP_REF["thA"], _ens["thB"] - SNAP_REF["thB"],
         wrap180(_ens["phA"] - SNAP_REF["phA"]), wrap180(_ens["phB"] - SNAP_REF["phB"]),
         wrap180(_ens["phC"] - SNAP_REF["phC"]),
         _esd["phA"], _esd["phB"], _esd["phC"]))
  _ref, _gate = choose_reference(_ens, SNAP_REF)
  if _gate:
    for _k, _e, _s, _d in _gate:
      _u, _lim = ("nm", POSE_GATE_NM) if _k == "r" else ("deg", POSE_GATE_DEG)
      sys.stderr.write("make_boresch: POSE GATE - %s: ensemble mean %.4f %s, bound "
                       "pose (snapshot) %.4f %s, %.4f %s apart against a %g %s "
                       "limit\n" % (BORESCH_NAMES[_k], _e, _u, _s, _u, _d, _u,
                                    _lim, _u))
    sys.stderr.write("make_boresch: the ensemble mean is not a pose of the bound "
                     "complex; the SNAPSHOT geometry is emitted for all six "
                     "Boresch coordinates instead\n")
    REF_SOURCE = ("single reference frame, the ensemble mean failed the pose "
                  "gate on %s" % ",".join(BORESCH_NAMES[g[0]] for g in _gate))
  else:
    REF_SOURCE = "mean over %d frames from %d replica(s)" % (_enf, len(TRAJ_PAIRS))
  ref_r, ref_thA, ref_thB = _ref["r"], _ref["thA"], _ref["thB"]
  ref_phA, ref_phB, ref_phC = _ref["phA"], _ref["phB"], _ref["phC"]
else:
  sys.stderr.write("make_boresch: Boresch geometry NOT averaged (%d usable frames); "
                   "using the single reference frame\n" % _enf)

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
# at 1.882 against a 2.427 threshold. A triad from the measured search cannot
# reach it any more, since the search budgets the larger of the ensemble and the
# snapshot r of the very groups measured here; the burial path has no such budget.
if PULL_LIMIT and r0_release > R_LIMIT_FRAC * PULL_LIMIT:
  abort("BORESCH_R_TOO_LARGE",
        "ref_r %.3f (%s) + pull_dist %.3f = %.3f nm exceeds 60%% of the GROMACS "
        "pull limit %.3f nm (0.49 * shortest box vector); increase the editconf "
        "padding or reselect anchors"
        % (ref_r, REF_SOURCE, args.pull_dist, r0_release, PULL_LIMIT))

thA0 = ref_thA * DEG2RAD
thB0 = ref_thB * DEG2RAD
prodK = K_R * (K_ANG_RAD ** 5)   # Kr * KthA*KthB*KphA*KphB*KphC (all angles equal)
numerator = 8.0 * math.pi**2 * V0 * math.sqrt(prodK)
denominator = (r0_release**2) * math.sin(thA0) * math.sin(thB0) * (2.0 * math.pi * RT)**3
dG_release = -RT * math.log(numerator / denominator)   # kJ/mol

#======================================================
# PART 4b - Settle the interface restraint set
#
# Selection and referencing are ONE decision and are made together here, after the
# anchor search (so an aborted setup does not pay for a trajectory pass) and before
# anything reads numinterdis.
#
# test7 split them and it showed. Candidates were picked by a single frame at
# 0.6 nm, then referenced to the trajectory mean, and the mean is a different
# number: it moved +0.86 A on average and up to +5.4 A. 253 of 637 springs -- 40%
# -- ended up referenced BEYOND the 0.6 nm contact cutoff, one of them at 1.11 nm.
# Those are not contacts. They were pairs that happened to be close in the frame
# that was looked at, they carried the largest reference shifts, and every one of
# them diluted the stiffness budget: k was 39.25 where the 384 real contacts alone
# would have given 65.10 kJ/mol/nm^2.
#
# It also silently broke the QC. interface_qc.py reports `formed` as the fraction
# of restrained pairs inside 0.6 nm, so a set that is 40% beyond that cutoff by
# construction has a ceiling of 60.3%. test7 read 0.581 and looked like poor
# rebinding; against its real ceiling it had recovered 96% of what was reachable.
#
# So the cutoff is now applied to the same distance the spring is referenced to.
#
# AND THE CANDIDATE PASS IS NOW WIDER THAN THE CUTOFF, which is the other half of
# the same fix and used to be the note here saying it had not been done: "a pair
# sitting at 0.62 nm in that frame whose mean is 0.55 nm is a genuine contact that
# never gets considered ... only one of them has been measured to cost anything."
# It has now been measured. Across eight runs of 2KTF under one protocol the frame
# yielded 556 to 918 candidates and 228 to 322 springs, and the spring count sets
# dG_intro at r = +0.91 and, through k = sum_k/N, stage E's dissipation at r =
# -0.73. dG_bind is unaffected (r = -0.10), so this was never a bias; it is why the
# per-channel intervals are 4-5x too small and why the stiffness of the whole set
# was being drawn at random. See CANDIDATE_CUT.
def finalise_interface(cands):
  """-> (pairs, note). Re-reference on the ensemble, then re-apply the cutoff to
  the reference that resulted, then cap. In that order, because each step should
  see what the one before it decided."""
  pairs, note = reference_on_ensemble(cands)
  if not note.startswith("mean over"):
    # No trajectory: the references are the frame distances. Those were accepted
    # at CANDIDATE_CUT, which is WIDER than the contact cutoff, so the cutoff has
    # to be applied here or every candidate becomes a spring. The old comment
    # said "the cutoff already accepted them"; that was true when candidates
    # were enumerated at 0.6 nm and stopped being true when the pass widened,
    # and nothing noticed until six HADDOCK setups ran with 20,000-40,000
    # springs (2026-09-09). Same trap as SD_MIN_KEEP: widen a filter's input
    # and every guard downstream of it silently loosens.
    keep = [p for p in pairs if p[2] <= interfacecutoff]
    sys.stderr.write("make_boresch: interface references NOT re-measured (%s); "
                     "applying the %.2f nm contact cutoff to the frame distances "
                     "instead, %d of %d candidates kept -- the test7 geometry, "
                     "self-consistent but unmeasured\n"
                     % (note, interfacecutoff, len(keep), len(pairs)))
    if not keep:
      abort("NO_INTERFACE_CONTACTS",
            "no candidate pair inside %.2f nm in the reference frame" % interfacecutoff)
    return keep, "single reference frame"

  keep = [p for p in pairs if p[2] <= interfacecutoff]
  dropped = len(pairs) - len(keep)
  if dropped:
    far = sorted((p[2] for p in pairs if p[2] > interfacecutoff), reverse=True)
    sys.stderr.write(
        "make_boresch: %d of %d candidates have a MEAN distance beyond the %.2f nm "
        "contact cutoff and are not restrained (worst %.3f nm). %d springs remain, "
        "so k is %.2f rather than %.2f kJ/mol/nm^2 for the same %.0f total.\n"
        % (dropped, len(pairs), interfacecutoff, far[0], len(keep),
           args.sum_k / max(len(keep), 1), args.sum_k / max(len(pairs), 1),
           args.sum_k))

  # HOW CLOSE DID THE CANDIDATE PASS COME TO BINDING? Two numbers: how many springs
  # the old 0.6 nm pass would have missed, and how much room is left between the
  # widest RESTRAINED pair's frame distance and the cut.
  #
  # READ THE MARGIN AS PROXIMITY, NOT AS LOSS. It says how near the widest spring
  # sat to the edge, which is not the same as pairs having fallen off it: a narrow
  # margin only means a slightly different draw might have lost some. This wording
  # is second-hand experience. At CANDIDATE_CUT = 0.9 test31 and test32 reported
  # 0.013 and 0.019 nm and the line here said, flatly, to raise the cut. Raising it
  # to 1.2 and re-running both on their own five probe replicas returned spring sets
  # IDENTICAL to the 0.9 ones -- 321 at k = 38.94 and 289 at k = 43.25, to the pair.
  # Nothing had been lost; only the headroom was thin. The direct test is to widen
  # and compare the spring count, and that is what the line now asks for.
  gained = [p for p in keep if _snap_of.get((p[0], p[1]), 0.0) > interfacecutoff]
  widest = max((_snap_of.get((p[0], p[1]), 0.0) for p in keep), default=0.0)
  sys.stderr.write(
      "make_boresch: %d of the %d springs (%.0f%%) sat beyond %.2f nm in the "
      "reference frame and would not have been candidates at all under the old "
      "pass. The widest restrained pair was at %.3f nm in that frame, leaving "
      "%.3f nm of margin to the %.2f nm candidate cut%s\n"
      % (len(gained), len(keep), 100.0 * len(gained) / max(len(keep), 1),
         interfacecutoff, widest, CANDIDATE_CUT - widest, CANDIDATE_CUT,
         "" if CANDIDATE_CUT - widest > 0.05 else
         " -- thin enough to check rather than trust. Re-run with a wider "
         "CANDIDATE_CUT and compare the spring count: unchanged means nothing was "
         "being lost off the edge and only the headroom was small."))
  if not keep:
    abort("NO_INTERFACE_CONTACTS",
          "all %d candidate pairs have a mean distance beyond %.2f nm, so the "
          "interface has no restrainable contact at all" % (len(pairs), interfacecutoff))
  return keep, note


interdis, _ref_note = finalise_interface(interdis)

# The cap, if it is ever switched back on, applies to the settled set: capping
# candidates would spend the budget on pairs the step above then discards.
_n_uncapped = len(interdis)
_n_contacts = len({(prot1_data[i][1], prot2_data[j][1]) for i, j, _ in interdis})
if MAX_PER_CONTACT and _n_uncapped:
  interdis = _cap_per_contact(interdis, MAX_PER_CONTACT)
  _sc = sum(1 for i, j, _ in interdis
            if prot1_data[i][2] not in BB_ATOMS or prot2_data[j][2] not in BB_ATOMS)
  sys.stderr.write("make_boresch: interface restraints %d -> %d, cap %d per "
                   "residue-residue contact (%d contacts, all kept); %.1f%% still "
                   "involve a side chain; k rises %.1f -> %.1f kJ/mol/nm^2 so the "
                   "total stays %.0f\n"
                   % (_n_uncapped, len(interdis), MAX_PER_CONTACT, _n_contacts,
                      100.0 * _sc / max(len(interdis), 1),
                      args.sum_k / max(_n_uncapped, 1),
                      args.sum_k / max(len(interdis), 1), args.sum_k))

numinterdis = len(interdis)
sys.stderr.write("make_boresch: %d interface springs at k = %.2f kJ/mol/nm^2 "
                 "(%d candidates enumerated at %.2f nm, settled on the ensemble)\n"
                 % (numinterdis, args.sum_k / max(numinterdis, 1), _n_snapshot,
                    CANDIDATE_CUT))

# The contact list the rebinding QC checks recovery against, written now that the
# set and its references are both final. utils/interface_qc.py reads it rather than
# reparsing the pull block and takes ref_nm from column 3; the frame distance the
# candidate was picked on is kept as a tenth column so the shift stays auditable.
with open("interface_contacts.gs", "w") as _f:
  _f.write("# atomA atomB ref_nm resA resnumA nameA resB resnumB nameB frame_nm\n")
  _f.write("# candidates %d, restrained %d, cap %s per residue-residue contact; "
           "ref_nm is the %s, and the %.2f nm cutoff is applied to IT\n"
           % (_n_snapshot, numinterdis, MAX_PER_CONTACT, _ref_note, interfacecutoff))
  for i, j, dist in interdis:
    a, b = prot1_data[i], prot2_data[j]
    _f.write("%d %d %.6f %s %d %s %s %d %s %.6f\n"
             % (a[3], b[3], dist, a[0], a[1], a[2], b[0], b[1], b[2],
                _snap_of.get((i, j), float("nan"))))

with open("boresch_analytical.gs", "w") as f:
  f.write("# Boresch standard-state analytical term (eq. 32)\n")
  f.write("# dG_release: free energy of removing the Boresch restraint to the\n")
  f.write("# standard state (1 mol/L). dG_intro = -dG_release.\n")
  f.write("# quantity            value        unit\n")
  f.write("dG_release_kJ_mol     %.6f\n" % dG_release)
  f.write("temperature_K         %.2f\n" % args.temp)
  f.write("ref_r_bound_nm        %.6f\n" % ref_r)
  # Where the geometry came from, in the same file as the geometry. Two runs that
  # disagree are only diagnosable if each says what it measured itself on.
  f.write("reference_source      %s\n" % REF_SOURCE.replace(" ", "_"))
  f.write("n_probe_replicas      %d\n" % len(TRAJ_PAIRS))
  f.write("r0_release_nm         %.6f\n" % r0_release)
  f.write("pull_dist_nm          %.6f\n" % args.pull_dist)
  # Recorded so the work integration always uses the rate this structure was
  # actually run with: integrate.py multiplies the time integral of the pull
  # force by the rate, so a mismatch silently rescales every pull work.
  f.write("pull_rate_nm_ps       %.8f\n" % args.pull_rate)
  # Recorded next to the geometry because it scales every work in the bound leg:
  # a cycle re-integrated later, or two runs compared, must be readable without
  # having to remember which sum-k the directory was set up with.
  f.write("sum_k_kJ_mol_nm2      %.4f\n" % args.sum_k)
  f.write("n_interface_springs   %d\n" % numinterdis)
  # Recorded for the same reason as sum_k: it selects WHICH springs exist, so two
  # directories built at different values do not have comparable bound legs.
  f.write("sd_max_nm             %s\n"
          % ("%.4f" % SD_MAX_NM if SD_MAX_NM is not None else "off"))
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
# the one least likely to sit on a boundary. pbcatom_of() is that choice, and the
# Boresch geometry above unwraps every group about the same atom, so the
# references are measured the way GROMACS will evaluate them.
PBCATOM = {}

def compute_pbcatoms():
  for role in ("P1", "P2", "P3", "L1", "L2", "L3"):
    g = rec_groups.get(anchors[role]) or lig_groups.get(anchors[role])
    a = pbcatom_of(g["atoms"])
    if a is not None:
      PBCATOM["bor_%s" % role] = a

write_index_groups()
compute_pbcatoms()

#======================================================
# PART 6 - Pull-block writers
#
# A "coord" is a dict with: name-list of ndx groups, geometry, dim, init, rate,
# k (state A), kB (state B). Groups are declared once and referenced by index.
#======================================================

# interdis is already final here: PART 4b settled both which pairs are restrained
# and where each one sits, so every leg below sees the same set and the same
# references. All that is left is the path they travel.
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
# _PULL_VEC is the snapshot P3->L1, the raw difference unless -f splits it (see
# where it is set).
PULL_AXIS = _PULL_VEC / np.linalg.norm(_PULL_VEC)

# Each spring's pair vector in -f, minimum-imaged like its distance, since that is
# the vector GROMACS pulls along. A spring is a contact, far shorter than half the
# box, so on a frame in one piece this is the plain difference bit for bit.
_pair_vec = min_image_vec(np.array([np.array(prot2_data[j][4:7], dtype=float)
                                    - np.array(prot1_data[i][4:7], dtype=float)
                                    for i, j, _ in interdis]), INPUT_BOX) \
            if interdis else np.empty((0, 3))


def pair_displacement(u):
  """How far each interface reference should have moved once the partner has
  translated u nm along the pull axis. Zero at u = 0 by construction, so it
  composes with the ensemble-mean reference rather than replacing it."""
  if not len(_pair_vec):
    return np.empty(0)
  d0 = np.linalg.norm(_pair_vec, axis=1)
  return np.linalg.norm(_pair_vec + u * PULL_AXIS, axis=1) - d0


# THE COMPOSED REFERENCE CAN CROSS ZERO. An interface reference travels
#
#     ref_i(u) = dist_i + |r_i + u*L*n| - |r_i|
#
# where dist_i is the ensemble mean over the pooled probe replicas and r_i is the
# pair vector in the single reference frame, the last frame of replica 1. The two
# disagree in length. For a pair whose frame vector points against the pull axis
# the second term falls by up to |r_i| - |r_i x n| before it rises again, and when
# the frame put the pair further apart than its mean by more than its offset from
# the axis, that fall exceeds dist_i and the reference goes negative. GROMACS
# refuses that wherever it happens: grompp rejects a negative pull-coordN-init, and
# mdrun stops at the first step where init + rate*t < 0 with "Pull reference
# distance for coordinate N (-0.000000) needs to be non-negative". The value prints
# as -0.000000 because it is one step of rate*dt below zero. The crossing step is
# fixed by the mdp, so the leg dies at the same step in every cycle and the retry
# dies with it.
#
# The second HADDOCK pass (v1.3) lost 1IBR (pull-coord780 in stage G: frame 0.553
# nm against a mean of 0.369, 11 degrees off antiparallel), 2B42 (pull-coord666, G)
# and 1BVN (pull-coord357, H) this way in every cycle, and the first pass lost 1AKJ
# (pull-coord164, G). 1OPH came within 0.020 nm of it.
#
# A FLOOR AT THE STAGE BOUNDARIES IS ENOUGH. A GROMACS reference moves in a straight
# line within a stage, and a chord between two values at or above the floor stays
# above it. It leaves u = 0 alone (dist_i is a heavy-atom contact distance) and
# u = 1 does not matter (kB = 0 there), so the end states and the free energy are
# unchanged. Only the path of a pair that would otherwise demand a negative distance
# is reshaped, at a point where its spring is already down to (1 - lambda) of full
# strength. A chord that stays above the floor keeps its arithmetic bit for bit.
#
# NOT ZERO. init is written to 1e-8 nm and rate to ten significant digits, and mdrun
# evaluates init + rate*t with both read in single precision, so a chord aimed at
# exactly 0 ends one rounding error below it about half the time and dies on its
# last step. 0.005 nm is below the lowest boundary reference of every structure that
# ran (0.012 nm, 1IBR at u = 0.379), so the 43 structures that completed the ramp get
# the same pull blocks, and the stages the three lost structures completed are
# unchanged.
PULL_REF_FLOOR_NM = 0.005


def interface_chord(dist, d_from, d_to):
  """(init, move) of one interface reference across one stage: the chord of
  dist + pair_displacement(u) between the stage's two boundaries, floored at
  PULL_REF_FLOOR_NM. A chord that never reaches the floor keeps the original
  arithmetic exactly."""
  a, b = dist + d_from, dist + d_to
  if a >= PULL_REF_FLOOR_NM and b >= PULL_REF_FLOOR_NM:
    return a, d_to - d_from
  a = max(a, PULL_REF_FLOOR_NM)
  return a, max(b, PULL_REF_FLOOR_NM) - a


def report_ref_floor():
  """One line in the setup log: the lowest interface reference at any stage
  boundary, and how many coordinates the floor holds up."""
  low, worst = set(), None
  for u in P.boundaries():
    d = pair_displacement(u)
    for n, (_i, _j, dist) in enumerate(interdis):
      v = dist + float(d[n])
      if worst is None or v < worst[0]:
        worst = (v, n + 1, u)
      if v < PULL_REF_FLOOR_NM:
        low.add(n + 1)
  if worst is not None:
    sys.stderr.write("make_boresch: lowest interface reference on the ramp %.4f nm "
                     "(pull-coord%d at u = %.3f); %d coordinate(s) floored at %.3f nm\n"
                     % (worst[0], worst[1], worst[2], len(low), PULL_REF_FLOOR_NM))


# Sum-k is a budget split over however many springs there are, so N and k never
# move independently and the collective stiffness is whatever --sum-k says.
#
# WHAT IT BUYS AND WHAT IT COSTS. The bound leg's work is 0.5*sum_k*<(d-r0)^2>, so
# both the work and its cycle-to-cycle spread are proportional to it, and that leg
# carries 80% of the dG_bind variance. Against that, the springs are the only thing
# pulling at the start of the ramp -- the Boresch distance coordinate has k = 0 at
# lambda = 0 and only reaches full strength at the far end -- and a given pulling
# force needs a lag of F/sum_k, so a softer set has to stretch further to do the
# same job. Measured on test5 at the stage-A friction peak, the genuine lag is
# 0.67 A at 25000 and scales inversely:
#
#     sum_k    25000   18000   12500    8000    5000
#     lag       0.67    0.93    1.34    2.10    3.36  A
#
# For scale, test5 ran with 2.5 A of imposed reference error on top of that lag and
# still rebound (2.31 A RMSD, 43/47 cycles under 3 A). The pull-path correction
# above removes that error, so spending part of the recovered headroom on a softer
# spring is a trade the same interface has already shown it can absorb -- but it is
# a trade, and it is why this is a knob with a recorded value rather than a new
# default.
k_inter = args.sum_k / numinterdis if numinterdis > 0 else 0.0

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


def elastic_network_coords(gidx):
  """The elastic network as pull coordinates: every intra-partner CA pair of
  build_elastic_network, k = kB = enk and rate 0, so it is the same restraint at
  every lambda. Marked en=True, which is how the far-end hold keeps it.

  One definition for every block that carries it (both leg families, nptrev and
  npt_c_fe.mdp), so 'the same network' holds by construction. gidx registers the
  pull groups in the caller's numbering, receptor pairs first, each pair's first
  atom first, the order the leg blocks have always used."""
  out = []
  for data, keep, pairs in ((prot1_data, protkeep1, en1dis),
                            (prot2_data, protkeep2, en2dis)):
    for i, j, dist in pairs:
      g1 = gidx("a_%d" % data[keep[i]][3])
      g2 = gidx("a_%d" % data[keep[j]][3])
      out.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                      init=dist, rate=0.0, k=enk, kB=enk, en=True))
  return out


def build_coords(family, direction, u_from=None, u_to=None, ps=None):
  """Return (pull_groups, coords) for a leg.

  family    : 'unbind'  -> interface(moving, k->0) + EN(fixed) + Boresch(0->full)
              'bound'   -> interface(fixed, 0->full) + EN(fixed), no Boresch
              'en'      -> EN(fixed) only, for npt_c_fe.mdp
  direction : 'fwd' or 'rev'. With u_from/u_to given this only selects the
              family's endpoint conventions; the geometry comes from u_from/u_to.

  u_from/u_to are the distance in nm the reference has travelled at the start and
  end of THIS stage, as fe_protocol gives them, so a stage covering 0.3 -> 1.0
  starts its moving coordinates already 0.3 nm out and moves the remaining 0.7 nm
  over `ps`. pull-coordN-start = no makes init absolute, so this offset is
  mandatory: without it stage B would jump back to the bound geometry. u_from ==
  u_to gives rate 0, which is what makes a hold contribute no work. They are never
  scaled by --pull-dist (see ONE PULL DISTANCE).

  Omitting them reproduces the old single-stage behaviour (0 -> --pull-dist
  forward, --pull-dist -> 0 reverse) at the constant --pull-rate.
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
    u_from = 0.0 if direction == "fwd" else args.pull_dist
    u_to = args.pull_dist if direction == "fwd" else 0.0
    stage_rate = sign * args.pull_rate
  else:
    # Rate follows from the distance this stage covers and the time it is given.
    stage_rate = 0.0 if (u_to == u_from or not ps) else (u_to - u_from) / ps
  off = u_from                             # where this stage's references start

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
    _nominal = u_to - u_from
    for n, (i, j, dist) in enumerate(interdis):
      g1 = gidx("a_%d" % prot1_data[i][3])
      g2 = gidx("a_%d" % prot2_data[j][3])
      _init, _move = interface_chord(dist, float(_d_from[n]), float(_d_to[n]))
      coords.append(dict(geometry="distance", dim="Y Y Y", groups=[g1, g2],
                         init=_init,
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
    coords += elastic_network_coords(gidx)
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
    coords += elastic_network_coords(gidx)

  elif family == "en":
    coords += elastic_network_coords(gidx)

  else:
    raise ValueError("unknown pull-block family %r" % family)

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


# What an mdp's pull-* lines become while its pull code is off.
_PULL_OFF = "; off while pull = no: "


def set_pull_switch(filename, on):
  """Turn the pull code of an mdp on or off and leave everything else alone.

  Off drops any pull block (from pull-ngroups on, as write_pull_block does) and
  comments out every other pull-* line: with pull = no, grompp warns once for each
  pull-* key it still meets, five warnings for the standard print header alone,
  which would eat half of the -maxwarn 10 every cycle grompp runs with. On restores
  those lines. The 'pull' line itself is written either way, and appended if the
  file has none, so the switch never depends on GROMACS's default."""
  out, seen = [], False
  for line in open(filename):
    if line.lstrip().startswith("pull-ngroups"):
      break
    if line.startswith(_PULL_OFF):
      line = line[len(_PULL_OFF):]
    key = line.split(";")[0].split("=")[0].strip().lower().replace("_", "-")
    if key == "pull":
      line = "pull                     = %s\n" % ("yes" if on else "no")
      seen = True
    elif key.startswith("pull-") and not on:
      line = _PULL_OFF + line
    out.append(line)
  if out and not out[-1].endswith("\n"):
    out[-1] += "\n"
  if not seen:
    out.append("pull                     = %s\n" % ("yes" if on else "no"))
  with open(filename, "w") as f:
    f.writelines(out)


# WHICH BLOCK A LEG CARRIES is fe_protocol.legs()'s "block", checked above to be
# one of PULL_BLOCKS for every leg, and nothing else: not the name, not the kind.
# v1.3 chose by kind ("every stage and hold gets the unbind block"), which handed
# the two holds inside the bound lambda schedule, holdbfwd1 and holdbrev1 at
# lambda 0.25, the unbind block: springs at 0.75 k_inter instead of 0.25 k_inter,
# plus the six Boresch coordinates at a quarter strength. The Hamiltonian then
# jumped at every hold entry and exit, and in the 39 structures of the second
# HADDOCK pass with a sound Boresch reference that switch did 6.4 (forward) and
# 7.3 (reverse) kJ/mol of unrecorded work on average, range 0.2 to 26.9, so the
# protocol's "holds do no work" was false exactly where dG_intro is measured.
#
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
report_ref_floor()
stage_rates = {}
# The FIRST stage's mdp and u_from, for the GROMACS readback below. It finds the
# six Boresch coordinates in that mdp by their groups (written_boresch). The
# readback once took the last six columns, which is only the right answer when the
# elastic network is empty: build_coords emits interface coords, then the Boresch
# distance, then the elastic network, then the five angles and dihedrals, and with
# a network it compared the Boresch r against a network pair distance and aborted
# a setup that was in fact correct.
first_stage_mdp, first_stage_u = None, 0.0
for _leg in P.legs():
  if _leg["block"] != "unbind":
    continue
  _name, _u0, _u1 = _leg["mdp"], _leg["u_from"], _leg["u_to"]
  _ps = leg_ps(_name)
  if _ps is None:
    abort("STAGE_MDP_MISSING",
          "cannot read nsteps/dt from %s, so its pull rate is undefined" % _name)
  _dir = "fwd" if _u1 >= _u0 else "rev"
  _pg, _co = build_coords("unbind", _dir, u_from=_u0, u_to=_u1, ps=_ps)
  if _leg["kind"] == "hold" and _u0 >= P.PULL_DIST:
    # The HOLD at the far end drops its interface coordinates and keeps the six
    # Boresch coordinates and the elastic network; see the long note below. Every
    # other hold keeps everything. The kind test is load-bearing: the first
    # reverse stage also starts at u = PULL_DIST, and stripping its interface
    # coordinates would leave it pulling nothing at rate zero.
    _co = [c for c in _co if c.get("boresch") or c.get("en")]
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
    first_stage_mdp, first_stage_u = _name, _u0
  stage_rates[_leg["name"]] = 0.0 if _u1 == _u0 else (_u1 - _u0) / _ps

# Bound-state restraint block, for every leg whose block is "bound": the bound
# sub-legs and the holds between them. The block is identical for all of them,
# interface coords at k = 0 -> kB = k_inter, rate 0, the elastic network, no
# Boresch, because only init-lambda and delta-lambda distinguish them, and those
# come from make_fe_mdps.py. A hold thus sits at its lambda of the SAME
# Hamiltonian as the sub-legs either side, 0.25 k_inter at lambda 0.25, and the
# switch into and out of it does no work. So this loops over whatever fe_protocol
# declares rather than naming files, and a reshaping of BOUND needs no edit here.
pg_bound, co_bound = build_coords("bound", "fwd")
for _leg in P.legs():
  if _leg["block"] != "bound":
    continue
  if _leg["u_from"] != _leg["u_to"]:
    abort("PROTOCOL_BLOCK_INVALID",
          "%s carries the bound block but moves u %g -> %g; the bound block has "
          "no moving coordinate" % (_leg["name"], _leg["u_from"], _leg["u_to"]))
  write_pull_block(_leg["mdp"], pg_bound, co_bound)
  if _leg["kind"] in ("stage", "hold"):
    stage_rates[_leg["name"]] = 0.0

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

# Hold leg at the unbound restrained state (lambda = 1): the six Boresch
# coordinates and the elastic network, with zero rate, and no interface
# coordinates. job_fe.run pins init-lambda = 1, delta-lambda = 0.
#
# THE ELASTIC NETWORK STAYS, because the hold must sample the lambda = 1
# Hamiltonian of the stages on either side of it, and the network is part of that
# Hamiltonian at every lambda (k = kB = enk). Until v1.5 the strip below kept only
# the Boresch coordinates and so dropped the network together with the springs,
# with nothing in this note to justify it. Over 5 ns the network pairs then
# drifted, and at t = 0 of bindrevI the network came back at full strength
# with a median 153 kJ/mol of strain (median over 39 structures of the second
# HADDOCK pass, per-structure medians 7 to 918; 32 of 718 cycles above 1000),
# where the network sits at 40-52 kJ/mol whenever it stays on. That is unrecorded
# work on the first state of the reverse ramp, the one the Crooks analysis takes
# as equilibrium. The network's pairs are intra-partner CA pairs at 0.4-0.9 nm held
# at 250 kJ/mol/nm^2, so they cannot approach the 0.49 * box limit that is the
# reason the interface coordinates go.
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
# The same argument is why the elastic network stays: what nptrev samples must be
# H(lambda = 1), and dropping the springs changes nothing in it (k = 0 there)
# while dropping the network did.
#
# The block itself is written by the loop above, which drops the interface
# coordinates from whichever hold sits at u = PULL_DIST.

# THE BOUND-STATE NPT CARRIES THE ELASTIC NETWORK. npt_c is the 20 ns equilibration
# every cycle runs before boundfwd1, so its end is the start of the forward bound
# leg, and the Crooks analysis takes that start as drawn from H(lambda = 0) of the
# bound family: interface springs at k = 0, i.e. absent, plus the network at full
# strength. Until v1.5 npt_c ran the npt_fe template with no pull code at all, so
# the network was off for 20 ns and switched on at t = 0 of boundfwd1, unrecorded:
# a median 299 kJ/mol over the 39 structures with a network in the second HADDOCK
# pass (per-structure medians 9 to 5569, single cycles up to 33,549), relaxing to
# 60 by the end of boundfwd1. Worse, the network pairs drifted freely, and in 20 of
# 920 cycles one drifted beyond 0.49 box, so the boundfwd1 grompp refused the
# network coordinate and the cycle was lost (1RLB 7, 1OPH 6, 1H1V 2, 1HE8 2, 1AK4,
# 1IQD, 1T6B).
#
# EXACTLY the network the legs carry (elastic_network_coords, the same pairs and
# references), k = kB = enk, rate 0. npt_c_fe.mdp has no free-energy block, so
# GROMACS uses k, which is why kB is the same number. A structure without a
# network gets pull = no, which leaves npt_c exactly the npt_fe physics. The probe
# NPT (npt_init_fe.mdp) stays without it: the network is built from the probe.
#
# What this holds is each network BODY (see PART 2), not each partner: two bodies
# of one partner that no network pair or disulfide joins stay as free to drift
# apart in npt_c as before, and the setup log says how many each partner has.
pg_en, co_en = build_coords("en", "fwd")
set_pull_switch(NPT_C_MDP, bool(co_en))
if co_en:
  write_pull_block(NPT_C_MDP, pg_en, co_en)
sys.stderr.write("make_boresch: %s carries %s\n"
                 % (NPT_C_MDP, "the elastic network, %d coordinates at k = kB = "
                    "%g kJ/mol/nm^2" % (len(co_en), enk) if co_en else
                    "no pull code (pull = no): this structure has no elastic "
                    "network"))


#======================================================
# PART 6 - Read the pull blocks back: from the mdps, and through GROMACS
#
# Everything above is open loop: the six Boresch coordinates are measured with
# this file's own helpers, written into the leg .mdp files, and never verified. A
# one-character sign error in dihedral_deg therefore survived into production and
# cost days of GPU time, because a mirror-image phi reference is not visibly
# wrong anywhere. It only shows up as strain once mdrun starts.
#
# So two things are read back. Every leg's mdp, as text: each Boresch coordinate,
# found by its geometry and its bor_ groups, must carry the reference this setup
# computed (r displaced by the leg's u_from), the leg's rate and the full force
# constant as kB; a bound leg and npt_c_fe.mdp must carry none. Then the forward
# block through GROMACS: grompp it at nsteps = 0 against the structure the
# references were measured from, run zero steps, and take from pullx both what
# GROMACS measures and the reference it read from the mdp. This catches sign
# conventions, group ordering, unit errors and index drift, for about 6 s per
# setup.
#
# The FORWARD block, not the hold: bindrev_fe and nptrev_fe carry
# init = ref_r + pull_dist, so on the bound reference their distance coordinate
# legitimately differs from its init by pull_dist and the check would misfire.
#======================================================

CHECK_TOL_DEG = 0.5     # 0.19 sigma at K_ANG_RAD; mass repartitioning alone is 0.96
CHECK_TOL_NM = 0.005
# How closely GROMACS's reference column can be compared with the reference this
# setup computed: pullx prints six significant digits (143.546 for 143.5455630) and
# the tpr keeps init in single precision. A swapped, mirrored or misplaced
# reference is off by degrees.
READ_TOL_DEG = 0.002
READ_TOL_NM = 2e-5

# Which Boresch coordinate a pull coordinate is, by its geometry and the bor_
# groups it names in order; the vectors are those of BORESCH_VECTORS.
BORESCH_SIGNATURES = {
    ("distance", ("bor_P3", "bor_L1")): "r",
    ("angle", ("bor_P3", "bor_P2", "bor_P3", "bor_L1")): "thA",
    ("angle", ("bor_L1", "bor_P3", "bor_L1", "bor_L2")): "thB",
    ("dihedral", ("bor_P1", "bor_P2", "bor_P2", "bor_P3", "bor_P3", "bor_L1")): "phA",
    ("dihedral", ("bor_P2", "bor_P3", "bor_P3", "bor_L1", "bor_L1", "bor_L2")): "phB",
    ("dihedral", ("bor_P3", "bor_L1", "bor_L1", "bor_L2", "bor_L2", "bor_L3")): "phC",
}


def written_boresch(path):
  """The Boresch coordinates an mdp carries, read from its text: ({key:
  (coordinate number, init, rate, k, kB)}, [what is wrong]). Wrong is a bor_ group
  in a coordinate of no Boresch shape, or a Boresch coordinate written twice."""
  kv = {}
  for line in open(path):
    s = line.split(";")[0]
    if "=" in s:
      k, v = (x.strip() for x in s.split("=", 1))
      kv[k.lower().replace("_", "-")] = v
  name = {i: kv.get("pull-group%d-name" % i)
          for i in range(1, int(kv.get("pull-ngroups") or 0) + 1)}
  out, bad = {}, []
  for c in range(1, int(kv.get("pull-ncoords") or 0) + 1):
    groups = tuple(name.get(int(g)) or "?" for g in kv.get("pull-coord%d-groups" % c, "").split())
    if not any(g.startswith("bor_") for g in groups):
      continue
    geom = kv.get("pull-coord%d-geometry" % c)
    key = BORESCH_SIGNATURES.get((geom, groups))
    if key is None or key in out:
      bad.append("pull-coord%d (%s %s)" % (c, geom, " ".join(groups)))
      continue
    out[key] = (c,) + tuple(float(kv["pull-coord%d-%s" % (c, f)])
                            for f in ("init", "rate", "k", "kb"))
  return out, bad


def check_written_legs(ref):
  """Every leg's Boresch coordinates as its mdp carries them, against ref, the
  references this setup computed (keyed like SNAP_REF). Returns what disagrees."""
  bad = []
  for leg in P.legs():
    got, odd = written_boresch(leg["mdp"])
    bad += ["%s: %s" % (leg["mdp"], o) for o in odd]
    if leg["block"] != "unbind":
      if got:
        bad.append("%s carries %s, but a %s leg has no Boresch coordinate"
                   % (leg["mdp"], "/".join(BORESCH_NAMES[k] for k in got), leg["block"]))
      continue
    lacks = [BORESCH_NAMES[k] for k in BORESCH_KEYS if k not in got]
    if lacks:
      bad.append("%s lacks %s" % (leg["mdp"], "/".join(lacks)))
      continue
    for key in BORESCH_KEYS:
      c, init, rate, k, kB = got[key]
      want = ref[key] + (leg["u_from"] if key == "r" else 0.0)
      want_rate = stage_rates[leg["name"]] if key == "r" else 0.0
      want_kB = K_R if key == "r" else K_ANG_RAD
      if abs(init - want) > 1e-7:
        bad.append("%s pull-coord%d (%s): init %.8f, the reference is %.8f"
                   % (leg["mdp"], c, BORESCH_NAMES[key], init, want))
      if abs(rate - want_rate) > 1e-8 * abs(want_rate) + 1e-15:
        bad.append("%s pull-coord%d (%s): rate %.10g, the leg's is %.10g"
                   % (leg["mdp"], c, BORESCH_NAMES[key], rate, want_rate))
      if k != 0.0 or abs(kB - want_kB) > 1e-6:
        bad.append("%s pull-coord%d (%s): k %g -> kB %g, not 0 -> %g"
                   % (leg["mdp"], c, BORESCH_NAMES[key], k, kB, want_kB))
  got, odd = written_boresch(NPT_C_MDP)
  if got or odd:
    bad.append("%s carries Boresch coordinates; it must carry the network only"
               % NPT_C_MDP)
  return bad


def readback_verdict(got, carried, snap, intended):
  """What GROMACS read and measured on -f, against this file's snapshot and its
  references. got is GROMACS's measured value of each of the six coordinates in
  BORESCH_KEYS order, carried GROMACS's reference column for each, i.e. the
  pull-coordN-init grompp read from the leg's mdp; snap and intended are dicts
  keyed like SNAP_REF, intended being the references this setup computed for that
  leg.

  Three separate questions. CONVENTION: does GROMACS get the snapshot geometry
  this file computes from the same structure, to CHECK_TOL? WRITTEN: does the leg
  carry the references this setup computed, to READ_TOL? POSE: is what the leg
  carries within the pose gate's limits of the bound pose GROMACS measures, with
  CHECK_TOL on top, since the gate compares against the snapshot and the snapshot
  is only known to agree with GROMACS to CHECK_TOL? The pose is asked of what
  GROMACS read, not of this file's numbers, so a writer that put a wrong number
  into the mdp fails it whatever the gate saw.

  Returns (log lines, convention failures, written failures, pose failures)."""
  lines, conv, init, pose = [], [], [], []
  for key, g, w in zip(BORESCH_KEYS, got, carried):
    s, i = snap[key], intended[key]
    if key == "r":
      d, dw, dp = abs(g - s), abs(w - i), abs(w - g)
      tol, rtol, ptol = CHECK_TOL_NM, READ_TOL_NM, POSE_GATE_NM + CHECK_TOL_NM
    else:
      d, dw, dp = abs(wrap180(g - s)), abs(wrap180(w - i)), abs(wrap180(w - g))
      tol, rtol, ptol = CHECK_TOL_DEG, READ_TOL_DEG, POSE_GATE_DEG + CHECK_TOL_DEG
    nm = BORESCH_NAMES[key]
    lines.append("make_boresch: readback %-8s snapshot %10.4f  gromacs %10.4f  "
                 "diff %8.4f%s   (written %10.4f, %.4f from gromacs%s%s)\n"
                 % (nm, s, g, d, "" if d <= tol else "   MISMATCH", w, dp,
                    "" if dp <= ptol else "   NOT A POSE",
                    "" if dw <= rtol else "   NOT THE REFERENCE %.4f" % i))
    if d > tol:
      conv.append("%s off by %.4f" % (nm, d))
    if dw > rtol:
      init.append("%s written %.4f, the reference is %.4f" % (nm, w, i))
    if dp > ptol:
      pose.append("%s written %.4f, %.4f from GROMACS's %.4f (limit %g)"
                  % (nm, w, dp, g, ptol))
  return lines, conv, init, pose


def verify_pull_block(mdp):
  """Re-measure an mdp's pull coordinates with GROMACS on -f: {coordinate number:
  (measured value, GROMACS's reference or None)}. None if not run, False if it
  failed."""
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
                 "pull-print-com", "pull-print-com1", "pull-print-com2",
                 "pull-print-ref-value"):
          continue
        f.write(line)
      # pull-print-ref-value adds a column "N ref" after each coordinate's "N":
      # the reference GROMACS itself read from the mdp, which is what the legs
      # will run with.
      f.write("\nnsteps = 0\npull-nstxout = 1\npull-print-com = no\n"
              "pull-print-ref-value = yes\n"
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
    row, col_val, col_ref = None, {}, {}
    for line in open(xvg):
      m = re.match(r'@\s*s(\d+)\s+legend\s+"(\d+)( ref)?"', line)
      if m:
        (col_ref if m.group(3) else col_val)[int(m.group(2))] = int(m.group(1)) + 1
      elif not line.startswith(("#", "@")):
        row = [float(v) for v in line.split()]
        break
    if not row or not col_val:
      sys.stderr.write("make_boresch: pull readback produced no pullx row\n")
      return False
    return {c: (row[k], row[col_ref[c]] if col_ref.get(c, len(row)) < len(row) else None)
            for c, k in col_val.items() if k < len(row)}
  except (OSError, ValueError) as e:
    sys.stderr.write("make_boresch: pull readback error: %s\n" % e)
    return False
  finally:
    shutil.rmtree(work, ignore_errors=True)


# The FIRST stage is the block whose inits ARE the bound references (u_from = 0),
# so it is the only stage the readback can compare against the reference
# structure. Any later stage legitimately sits u_from nm away and would
# false-positive.
#
# WHAT THIS CHECK IS FOR, and what it is NOT for. It exists to catch a sign or
# ordering convention in this file disagreeing with the pull code -- the shape of
# the dihedral_deg bug that voided every work integral before b4c267f. That is a
# question about how a coordinate is COMPUTED, so it is asked of one structure:
# does GROMACS get the same six numbers from args.input that the functions above
# get from args.input?
#
# It is NOT a check that the emitted reference EQUALS args.input. Since the
# references became ensemble means they legitimately do not, and comparing the two
# at the convention tolerance made the guard fire on every setup: on test13 the
# six "mismatches" were exactly the ensemble shift, to four decimals. So the
# convention is checked against SNAP_REF, the single-frame geometry.
#
# WHAT IT LEFT UNCHECKED UNTIL v1.5, and why 1AKJ went through it. The emitted
# reference was printed beside the snapshot "for context" and compared with
# nothing. 1AKJ's setup log in the second HADDOCK pass carried phi_A snapshot
# -118.4, emitted 138.7, and phi_B 139.5 against -116.0, each about 104 degrees
# apart, and the setup completed; so did 1E6J (115) and 1IQD (111). The convention
# check passed there, and rightly: this file and GROMACS agreed about the snapshot
# of -f. The fault sat in the ensemble mean, which v1.4 built
# from raw COMs of -pbc whole frames with only P3->L1 imaged (see BORESCH_VECTORS),
# and no check ever put an emitted value next to anything GROMACS measured.
#
# So the second question became whether every EMITTED reference lies within the
# pose gate's limits of the bound pose as GROMACS measures it on -f. But it was
# asked of the Python references, ref_r and the rest, never of what the mdps
# carry, and pullx only gave GROMACS's measurement of -f: a build_coords that wrote
# theta_A's init as theta_B's, theta_B's as theta_A's and phi_A mirrored ran on 2KTF
# to exit 0, with theta_A written 98.58 against a measured 75.60 and phi_A 70
# degrees off, while the log printed the right numbers as 'emitted' (review R3).
#
# Hence three questions now, see readback_verdict, after every leg's mdp has been
# read back as text: the CONVENTION, as before; WRITTEN, whether GROMACS read the
# references this setup computed out of the first stage's mdp; and the POSE,
# whether what GROMACS read lies within the gate's limits of what it measures. The
# gate guarantees the pose by construction for what this file computed; GROMACS
# confirms it for what the legs will run with. The ensemble means themselves are
# pinned by tests/test_boresch_pbc.py: the snapshot and every ensemble frame go
# through the same boresch_series, so a convention verified here is a convention
# verified for both.
_intended = dict(r=ref_r, thA=ref_thA, thB=ref_thB,
                 phA=ref_phA, phB=ref_phB, phC=ref_phC)
_problems = check_written_legs(_intended)
if _problems:
  abort("PULL_INIT_MISMATCH",
        "the leg mdps do not carry the Boresch references this setup computed: %s%s"
        % ("; ".join(_problems[:6]),
           "; and %d more" % (len(_problems) - 6) if len(_problems) > 6 else ""))
sys.stderr.write("make_boresch: every leg mdp carries the computed Boresch "
                 "references (read back from %d mdps)\n"
                 % (len({l["mdp"] for l in P.legs()}) + 1))
_read = verify_pull_block(first_stage_mdp)
if _read is False:
  abort("PULL_READBACK_FAILED",
        "could not read the emitted pull block back through GROMACS")
elif _read is not None:
  # The six Boresch coordinates as the mdp has them, found by their groups.
  _cols = [written_boresch(first_stage_mdp)[0][k][0] for k in BORESCH_KEYS]
  if any(c not in _read or _read[c][1] is None for c in _cols):
    abort("PULL_READBACK_FAILED",
          "pullx has no value or no reference column for the Boresch coordinates "
          "%s of %s" % (_cols, first_stage_mdp))
  got = [_read[c][0] for c in _cols]
  carried = [_read[c][1] for c in _cols]
  _first = dict(_intended, r=ref_r + first_stage_u)
  _lines, _conv, _init, _pose = readback_verdict(got, carried, SNAP_REF, _first)
  for _ln in _lines:
    sys.stderr.write(_ln)
  if _conv:
    abort("PULL_READBACK_MISMATCH",
          "GROMACS does not reproduce this file's own geometry for the reference "
          "structure (%s). A sign or ordering convention here disagrees with the "
          "pull code; do not run on this setup. This compares the SNAPSHOT "
          "values; the written references are checked separately." % "; ".join(_conv))
  if _init:
    abort("PULL_INIT_MISMATCH",
          "GROMACS read Boresch references from %s that are not the ones this "
          "setup computed (%s)" % (first_stage_mdp, "; ".join(_init)))
  if _pose:
    abort("BORESCH_REFERENCE_NOT_A_POSE",
          "the Boresch references the legs carry are not a pose of the bound "
          "complex as GROMACS measures it on %s (%s); the pose gate should have "
          "replaced them, so do not run on this setup" % (args.input, "; ".join(_pose)))

#======================================================
# Output: number of moving pull coords whose forces feed the pull integrator.
# Interface restraints (numinterdis) + the single Boresch distance coord = +1.
# (Elastic network and Boresch angles/dihedrals have rate 0 -> no pull work.)
#======================================================
print(numinterdis + 1)
