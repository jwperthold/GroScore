#!/usr/bin/env python3
#
# groscore_fe.py - Free-energy (absolute-binding) variant of GroScore.
#
# Unlike the classic engine (groscore.py), which reports a relative pull-work
# "score", this variant estimates an absolute binding free energy from a
# thermodynamic cycle built out of bidirectional non-equilibrium switching:
#
#   unrestrained-bound
#        | +dG_intro     (bound-state restraint introduction; dhdl)
#   interface-restrained-bound
#        | +dG_unbind     (interface -> Boresch handoff + separation by
#        |                 fe_protocol.PULL_DIST, 1.0 nm; pull-force work + dhdl work)
#   Boresch-restrained-unbound
#        | +dG_release    (analytical Boresch standard-state term, eq. 32)
#   unrestrained-unbound (1 mol/L standard state)
#
#   dG_bind = -(dG_intro + dG_unbind + dG_release)
#
# Each simulated leg is run forward and reverse per cycle; the free energies are
# estimated with BAR, and with the Crooks-Gaussian-Intersection (CGI) and simple
# average estimators alongside it, reusing the machinery of the classic engine.
#
# The unbinding ramp is run in STAGES with an equilibrium hold at every internal
# boundary, in both directions. How many stages and where the boundaries sit is
# defined in utils/fe_protocol.py and nowhere else. The stages are scored as
# separate Crooks processes, each dissipating a fraction of the total, so each has
# work overlap where the whole ramp has none, and dG_unbind is their sum. That is
# only as sound as the holds are at equilibrating the interface, and they are
# measured not to be (score_structure states the numbers). The
# holds do no work, so the stage works also sum to the work of the whole ramp,
# which is scored too as the assumption-free cross-check (dG_unbind_1s).
#
# job_fe.run writes, per completed cycle, one row to results_fe.d/<id>_c<N>.gs:
# STRUCT_ID, the cycle, the forward bound sub-leg works, the pull and dhdl works of
# every ramp stage forward and then reverse, the reverse bound sub-leg works, and
# the rebound RMSD. That is FE_RESULT_NF = 43 fields in v1.5; fe_protocol.works()
# names the work fields in row order, with the reverse parts in the order they run.
# The setup writes, per structure, one line to results_analytical.d/<id>.gs:
#   STRUCT_ID dG_release_kJ_mol
# All works are in kJ/mol. A legacy single results_fe.gs or results_analytical.gs
# is still read, and so are rows of earlier stage counts, whose layout follows from
# the row width. See read_works.
#
# SIGN CONVENTIONS (physical work, all kJ/mol):
#   * dhdl work (integrate_dhdl.py) is the physical switching work along the ramp:
#       forward  ramp lambda 0->1 : W_fwd = int_0^1 <dH/dl> dl
#       reverse  ramp lambda 1->0 : W_rev = int_1^0 <dH/dl> dl  (opposite sign)
#   * pull work (integrate.py) returns -rate * int F dt, where the rate is passed
#     via -r (job_fe.run reads it back from boresch_analytical.gs, so a cycle is
#     always integrated with the rate it was actually run at).
#     Mapping to physical work W = rate_actual * int F dt:
#       forward leg (rate +|r|): W_pull_fwd = -(integrate.py output)
#       reverse leg (rate -|r|): W_pull_rev = +(integrate.py output)
#   * A forward/reverse pair is combined with Crooks by feeding
#       pulls  = W_forward
#       pushes = -W_reverse
#     so that (mean(pulls)+mean(pushes))/2 -> dG and the CGI intersection -> dG.
#
# Every scoring pass also writes fe_works.png -- the forward and sign-aligned
# reverse work histogram of each leg, with the avg and CGI rules drawn on -- and
# prints the Gaussian/near-equilibrium consistency of each leg, which says
# whether those CGI crossings are measured or extrapolated. Diagnostic only:
# neither changes a number in scores_fe.gs.
#
# NOTE: the relative sign of the pull-work and dhdl-work terms, and the pull-force
# sign convention of this GROMACS build, MUST be validated on the first real
# bidirectional run using that diagnostic (forward and reverse work histograms
# should overlap around dG). See the SIGN_* constants below.

import math, os, re, sys, argparse, shutil, glob, subprocess
from statistics import NormalDist
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="GroScore FE: absolute binding free energy via Boresch restraints.")
parser.add_argument('-n', '--numruns', type=int, default=None,
                    help="TOTAL bidirectional cycles wanted per structure. Remembered "
                         "in run_config.gs, so later invocations in the same directory "
                         "inherit it and only the first run needs to say it (default: "
                         "the remembered value, or DEFAULT_NUMRUNS = 50 for a fresh "
                         "directory, because BAR needs work overlap and no shorter "
                         "prefix of a real run has produced one). Re-run with a larger "
                         "value plus --restart to add cycles: only the cycles without "
                         "a complete result are submitted.")
parser.add_argument('--sum-k', dest='sum_k', type=float, default=None,
                    help="Total interface restraint stiffness in kJ/mol/nm^2, split "
                         "evenly over however many springs the interface has. "
                         "Remembered in run_config.gs the same way as --numruns, so "
                         "two run directories can differ in this and nothing else "
                         "without either of them editing tracked code. It scales the "
                         "work of switching the interface restraints on, which is the "
                         "largest term in the bound leg and carries most of the "
                         "dG_bind variance; it also sets how hard those springs pull "
                         "before the Boresch takes over, so lowering it trades "
                         "restraint noise for pulling authority (default: the "
                         "remembered value, or DEFAULT_SUM_K = 12500 for a fresh "
                         "directory).")
parser.add_argument('--sd-max', dest='sd_max', type=float, default=None,
                    help="Ceiling on an interface pair's distance standard deviation "
                         "over the pooled probe replicas, in nm; wider pairs get no "
                         "spring. Remembered in run_config.gs like --sum-k, and for "
                         "the same reason: it selects WHICH springs exist, so two "
                         "directories built at different values do not have "
                         "comparable bound legs. 0 keeps every pair (default: the "
                         "remembered value, or DEFAULT_SD_MAX for a fresh directory).")
parser.add_argument('-s', '--structparams', type=str, default="sp.gs", help="Structure parameter file (default: sp.gs).")
parser.add_argument('-ff', '--forcefield', type=str, default=None,
                    choices=["gromos54a8", "gromos54a8_rf", "charmm36", "amber19sb_opc", "amber19sb_opc3"],
                    help="Force field (default: amber19sb_opc3). Remembered in "
                         "run_config.gs: a directory whose cycles were built under "
                         "one force field cannot have the rest built under another.")
# store_const with default None rather than store_false: a remembered setting can
# only be honoured if "the user said nothing" is distinguishable from "the user
# asked for the default", which store_false cannot express. The positive form is
# added so a directory can be changed back, not only away.
parser.add_argument('--no-cutout', dest='cutout', action='store_const', const=False,
                    default=None, help="Disable interface cutout.")
parser.add_argument('--cutout', dest='cutout', action='store_const', const=True,
                    default=None, help="Enable interface cutout (the default).")
parser.add_argument('--no-ligand-param', dest='ligand_param', action='store_const',
                    const=False, default=None,
                    help="Run without small-molecule ligands: none is parametrized or "
                         "merged, and each is named in the job log. Without it, CHARMM36 "
                         "and GROMOS 54A8 stop every structure with a ligand in the "
                         "simulated region as LIGAND_FAIL. Remembered in run_config.gs.")
parser.add_argument('--ligand-param', dest='ligand_param', action='store_const',
                    const=True, default=None,
                    help="Enable OpenFF small-molecule parametrization (the default).")
parser.add_argument('--slurm', type=str, default="workstation", help="SLURM template name from slurm/ (default: workstation).")
parser.add_argument('--run-local', dest='run_local', action='store_true',
                    help="Run on this machine instead of submitting to SLURM. Setup jobs and "
                         "cycles are started in the background and spread round-robin over the "
                         "local GPUs, for single multi-GPU workstations. Requires --ngpus.")
parser.add_argument('--ngpus', type=int, default=0, metavar='N',
                    help="Number of GPUs to distribute local jobs over (mandatory with --run-local).")
parser.add_argument('--jobs-per-gpu', dest='jobs_per_gpu', type=int, default=8, metavar='N',
                    help="Concurrent jobs per GPU in --run-local mode (default: 8); the rest "
                         "queue and start as slots free up. One cutout-sized system leaves the "
                         "GPU idle during CPU-side work, so stacking jobs raises aggregate "
                         "throughput. Lower it if you run out of GPU or host memory; "
                         "0 starts every job at once.")
parser.add_argument('--threads-per-job', dest='threads_per_job', type=int, default=1, metavar='N',
                    help="CPU threads per local job, i.e. gmx mdrun -nt (default: 1).")
parser.add_argument('--restart', action='store_true',
                    help="Submit again for an existing run: re-queues missing/failed cycles "
                         "and, with a larger -n, adds new ones.")
parser.add_argument('--inject-job-run', action='store_true',
                    help="Install this job_fe.run into archived (.tar.gz) structures "
                         "that get cycles submitted. Without it an archived "
                         "structure is NOT submitted: the job.run inside its "
                         "archive is whatever ran it last and may predate this code.")
parser.add_argument('--temp', type=float, default=None,
                    help="Temperature in K (default: 310). Remembered in "
                         "run_config.gs: it sets RT for pKD and for BAR, so scoring "
                         "the same works at two temperatures gives two answers.")
parser.add_argument('--n-boot-bar', dest='n_boot_bar', type=int, default=5000, metavar='N',
                    help="Bootstrap rows for the BAR confidence intervals (default: 5000). "
                         "avg and CGI always use 50000; BAR is a root-find per row rather "
                         "than a closed form, so it runs on a prefix of the same resample. "
                         "5000 rows know their own CI to about 1%%, which is well inside the "
                         "CI itself. Raise it if a run shows the BAR CI moving materially "
                         "between 5000 and 50000.")
parser.add_argument('--rmsd-warn', type=float, default=10.0, metavar='A',
                    help="Warn about cycles whose re-bound backbone RMSD exceeds this "
                         "value in Angstrom (default: 10.0). Free energies are always "
                         "reported; the check only flags them.")
parser.add_argument('--array-throttle', type=int, default=0, metavar='N',
                    help="Max cycle-array tasks running concurrently per structure "
                         "(SLURM %%N). Use 1 on a single-GPU workstation; 0 = no limit (default).")
parser.add_argument('--sequential', action='store_true',
                    help="Legacy submission: one job per structure running all cycles "
                         "sequentially, instead of a setup job + per-cycle job array.")
args = parser.parse_args()

if args.run_local and args.ngpus < 1:
  parser.error("--run-local needs --ngpus N: the number of GPUs the jobs are distributed over")
if args.ngpus and not args.run_local:
  parser.error("--ngpus only applies to --run-local; SLURM allocates GPUs itself")

#------------------------------------------------------
#
# The requested cycle count has to outlive the invocation that asked for it.
# Everything downstream keys off it: which cycles count as missing, and
# GROSCORE_NUMCYCLES, which job.run compares against the .done markers to decide
# a structure is finished and can be tarred up. Falling back to the built-in
# default on a later run would therefore both stop the run growing past that
# number and tell a cycle job it was the target -- enough, with that many markers
# present, to archive a structure mid-run. So it is remembered here rather than
# defaulted, and a fresh directory records it immediately.
#
# 50, NOT the classic engine's 5. This engine's answer is a BAR estimate, and BAR
# returns nothing at all without forward/reverse work overlap. On test5 no prefix
# of the run produced a number: five, ten, twenty, forty cycles all came back
# BAR_NO_OVERLAP and only the full 47 scored. A five-cycle FE run is not a coarse
# answer, it is no answer, so defaulting to 5 here would only ever mean a wasted
# 400 ns. groscore.py keeps 5 because its score is a calibrated pull work that a
# handful of cycles genuinely does estimate.
RUN_CONFIG = "run_config.gs"
DEFAULT_NUMRUNS = 50

# Total interface restraint stiffness, kJ/mol/nm^2, split evenly over however many
# springs the interface has. THE VALUE LIVES HERE, and every run directory records
# the one it used, so nobody has to go looking for it: groscore_fe.py always passes
# an explicit --sum-k down to make_boresch.py, whose own default is a fallback for
# running that script by hand and is pinned equal to this one by tests/test_sum_k.py.
#
# 12500 rather than the 25000 this started at. Two reasons, and the second is the
# one that makes it more than a preference:
#
#   * it is what test8 and test10 measured. Halving the budget took stage A's work
#     overlap from 3 of 100 to 54 and brought the staged sum and the one-shot ramp
#     into agreement to 0.61 kJ/mol, because both ends of the bound leg scale with
#     it and so does the gap between them.
#   * THE RAMP BOUNDARIES ASSUME IT. Every stage boundary and every leg time in
#     fe_protocol comes from a friction profile measured on test12, which ran at
#     12500. Friction depends on stiffness -- stiffer springs drag harder -- so
#     running those boundaries at 25000 would apply a calibration taken under
#     conditions that no longer hold. The two constants are coupled whether or not
#     anyone remembers it, which is why this one says so.
#
# Not written as argparse's default= for the same reason DEFAULT_NUMRUNS is not:
# the parser has to be able to tell "the user asked for 12500" from "the user asked
# for nothing", or a directory set up at some other value would be silently reset
# by the next invocation that omitted the flag.
DEFAULT_SUM_K = 12500.0

# Width ceiling on an interface spring, nm. Pinned equal to make_boresch.py's own
# SD_MAX_DEFAULT by tests/test_sd_max.py, exactly as DEFAULT_SUM_K is pinned to its
# --sum-k default: this file records the value in run_config.gs and job_fe.run
# passes it on, so the two must not drift. 0 means keep every pair.
#
# It belongs in the remembered set for the same reason sum_k does, and slightly
# more sharply: sum_k scales the springs, this decides WHICH of them exist, so a
# directory that built half its cycles at one value and half at another has two
# different bound legs pooled into one number.
DEFAULT_SD_MAX = 0.15


def read_run_config():
  cfg = {}
  if os.path.isfile(RUN_CONFIG):
    try:
      with open(RUN_CONFIG) as f:
        for line in f:
          if line.strip().startswith("#"):
            continue
          tmp = line.split()
          if len(tmp) >= 2:
            cfg[tmp[0]] = tmp[1]
    except OSError:
      pass
  return cfg


def write_run_config(**kw):
  cfg = read_run_config()
  cfg.update({k: str(v) for k, v in kw.items()})
  with open(RUN_CONFIG, "w") as f:
    f.write("# GroScore-FE run settings, remembered between invocations in this\n")
    f.write("# directory. Delete a line to fall back to the command-line default.\n")
    for key in sorted(cfg):
      f.write("%s\t%s\n" % (key, cfg[key]))


# EVERY SETTING THAT CHANGES WHAT IS SIMULATED IS REMEMBERED, from this one table.
#
# A run directory is filled in over days by repeated invocations -- topping up
# cycles, restarting failures, re-scoring -- and each of those is a fresh command
# line. Anything that alters what is simulated must therefore come from the
# DIRECTORY rather than from whichever command line happened to run last, or the
# cycles in one directory stop being samples of one thing. That is not hypothetical:
# sum_k selects the springs, sd_max selects which of them survive, the force field
# selects the topology, and a --restart that silently omitted any of them would
# build the remaining cycles against a different system and pool them anyway.
#
# The rule for each entry: an explicit value wins and WARNS if it differs from what
# the directory already has; no value takes the remembered one; a fresh directory
# takes the default and RECORDS it, so a directory always states its own settings
# rather than relying on a default that may move under it.
#
# What is deliberately NOT here: everything about how the work is scheduled or
# reported rather than what it is. --slurm, --run-local, --ngpus, --jobs-per-gpu,
# --threads-per-job, --array-throttle, --sequential, --restart, --inject-job-run
# are scheduling; -s is which structures; --n-boot-bar and --rmsd-warn only change
# how finished works are summarised and can be varied freely on the same data.
#
#         attr           key            parse   format             default
REMEMBERED = [
  ("numruns",      "numruns",      int,   lambda v: "%d" % v, lambda: DEFAULT_NUMRUNS,
   "how many cycles this directory is aiming for"),
  ("sum_k",        "sum_k",        float, lambda v: "%g" % v, lambda: DEFAULT_SUM_K,
   "the interface stiffness budget, so works from two values are not comparable"),
  ("sd_max",       "sd_max",       float, lambda v: "%g" % v, lambda: DEFAULT_SD_MAX,
   "which interface pairs get a spring at all, so the bound legs differ"),
  ("forcefield",   "forcefield",   str,   lambda v: "%s" % v, lambda: "amber19sb_opc3",
   "the force field, so the topology itself differs"),
  ("temp",         "temp",         float, lambda v: "%g" % v, lambda: 310.0,
   "the temperature RT is taken at, so pKD and BAR both move"),
  ("cutout",       "cutout",       lambda x: x == "1", lambda v: "1" if v else "0",
   lambda: True, "whether the interface is cut out, so the system differs"),
  ("ligand_param", "ligand_param", lambda x: x == "1", lambda v: "1" if v else "0",
   lambda: True, "whether small molecules are parametrised, so the topology differs"),
]

_cfg = read_run_config()
REMEMBERED_FROM_CFG = set()
_to_write = {}
for _attr, _key, _parse, _fmt, _default, _why in REMEMBERED:
  _given = getattr(args, _attr)
  if _given is None:
    try:
      setattr(args, _attr, _parse(_cfg[_key]))
      REMEMBERED_FROM_CFG.add(_attr)
    except (KeyError, ValueError):
      setattr(args, _attr, _default())
  elif _key in _cfg and _cfg[_key] != _fmt(_given):
    if _key == "numruns":
      # A new cycle target changes no work: a larger -n only raises the total the
      # jobs run to (GROSCORE_NUMCYCLES), which is how cycles are added. The warning
      # below said the works of the two values were not comparable (review D10).
      print("Cycle target %s in %s, now %s; recorded." % (_cfg[_key], RUN_CONFIG, _fmt(_given)))
    else:
      print("WARNING: this directory was set up with %s = %s and you asked for %s."
            % (_key, _cfg[_key], _fmt(_given)))
      print("  That is %s." % _why)
      print("  Cycles already built keep the old value; only structures whose setup")
      print("  re-runs will pick up the new one, and works from the two are not")
      print("  comparable. Use a fresh directory unless that is what you meant.")
  _now = _fmt(getattr(args, _attr))
  if _cfg.get(_key) != _now:
    _to_write[_key] = _now
if _to_write:
  write_run_config(**_to_write)

# Kept as names because the submission summary reads them.
NUMRUNS_REMEMBERED = "numruns" in REMEMBERED_FROM_CFG
SUM_K_REMEMBERED = "sum_k" in REMEMBERED_FROM_CFG

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from local_runner import launch_local, print_local_status
import estimators as est
import fe_protocol as P

# Display names for the bound sub-legs, taken from the protocol so a reshaping of
# BOUND renames the channels and the columns with it. A run from before the bound
# leg was staged has one sub-leg and keeps the old single-channel presentation.
BOUND_NAMES = [b[0] for b in P.BOUND]

RT = 0.00831446261815324 * args.temp  # kJ/mol


def check_ref_t():
  """Warn if --temp disagrees with the temperature the mdps actually run at.

  --temp is a post-processing knob only: it sets RT for pKD and for BAR, and it
  never reaches grompp, where ref_t is fixed in the templates. So --temp 300 on a
  310 K simulation silently gives both a wrong kT, with nothing to notice it.
  Warn rather than abort, since scoring an archived run whose settings tree has
  moved on is legitimate."""
  mdp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "settings", args.forcefield, P.legs()[0]["mdp"])
  try:
    with open(mdp) as f:
      for line in f:
        if line.strip().startswith("ref_t"):
          vals = [float(v) for v in line.split("=")[1].split()]
          if any(abs(v - args.temp) > 0.5 for v in vals):
            print("WARNING: --temp %.1f K but %s runs at ref_t = %s. RT is used "
                  "for pKD and for BAR, so both will be wrong by that ratio."
                  % (args.temp, mdp, " ".join("%g" % v for v in vals)))
          return
  except (OSError, ValueError, IndexError):
    pass                                  # missing or unparseable: not fatal

# How much of the rebinding-QC warning is printed. The full picture is always in
# scores_fe.gs; the console only needs enough to know what to look at.
MAX_FLAGGED_SHOWN = 10   # structures listed in the warning
MAX_CYCLES_SHOWN  = 5    # bad cycles listed per structure

# LOW_RECOVERY, the per-structure rebinding flag beside HIGH_RMSD (decision of
# 2026-10-04): the median fraction of restrained contacts back within tolerance at
# the end of rebinding (interface_qc.py, results_qc.d) is below this. Backbone RMSD
# says nothing about the interface: the 20 pass-2 cycles above 10 A had recovery
# 0.760-0.974, and Spearman(RMSD, rms_dev) over 450 valid cycles is 0.00. Recovery
# had an energetic consequence in pass 2: 0.1 less of it moved a typical
# structure's dG_intro + dG_unbind by about 9 kJ/mol, cycle for cycle, mostly
# through reverse bound legs that started from the rebound state, which v1.5 no
# longer does (bm_fe_amber_2/_ana/digest/qc.txt, highrmsd_not_rebinding and
# rebinding_strain_bias). 0.85 is the cut qc.txt's qc_filter_sensitivity used: pass
# 2 had a pooled median of 0.934 and five structure medians below 0.85 (1AKJ 1BJ1
# 1GRN 1WQ1 2JEL, 0.78-0.83). A constant, not an option: it is pre-registered for
# the pilot readout and must not move after the data are in. Neither flag changes a
# score. Pass-2 results_qc.d rows of the 19 structures above 99,999 atoms read about
# 0.001 from the atom-number wrap fixed in v1.5's interface_qc, so a re-score of
# pass 2 flags all of those as well.
RECOVERY_WARN = 0.85

# Sign toggles to flip during first-run validation if the diagnostic requires it.
SIGN_PULL_FWD = -1.0   # W_pull_fwd = SIGN_PULL_FWD * integrate.py(fwd)
SIGN_PULL_REV = +1.0   # W_pull_rev = SIGN_PULL_REV * integrate.py(rev)

#------------------------------------------------------

def readstructparams(filepath):
  ids, chains = [], []
  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          try:
            ids.append(tmp[0]); chains.append(tmp[1])
          except (IndexError, AttributeError):
            pass
  return (ids, chains) if len(ids) == len(chains) and ids else ([], [])

#------------------------------------------------------

def _stream_avg(fwd, rev, axis=None):
  """Crooks average estimator of one stream: (mean(fwd) + mean(-rev)) / 2."""
  return (fwd.mean(axis) - rev.mean(axis)) / 2.0

def _stream_cgi(fwd, rev):
  """CGI intersection of one stream. fwd/rev are 2-D (n_replicates, n_cycles);
  pushes = -rev (mean flips sign, variance unchanged). Returns one value per row,
  NaN where the two Gaussians are degenerate."""
  ap, vp = fwd.mean(1), fwd.var(1)
  aq, vq = -rev.mean(1), rev.var(1)
  out = np.full(fwd.shape[0], np.nan)
  m = (vp > 0) & (vq > 0) & (vp != vq)
  if np.any(m):
    apm, vpm, aqm, vqm = ap[m], vp[m], aq[m], vq[m]
    dinv = 1.0 / vpm - 1.0 / vqm
    t1 = apm / vpm - aqm / vqm
    # Discriminant of Goette & Grubmueller eq. (12). Their log is of the SIGMA
    # ratio with a factor 2; ln of the VARIANCE ratio already absorbs it, so no
    # extra 2 here. See groscore.py calculate_scores and tests/test_cgi.py.
    t2 = np.sqrt((apm - aqm)**2 / (vpm * vqm) + dinv * np.log(vqm / vpm))
    s1 = (t1 + t2) / dinv
    s2 = (t1 - t2) / dinv
    mid = (apm + aqm) / 2.0
    pick = np.where(np.abs(mid - s1) > np.abs(mid - s2), s2, s1)
    # Degenerate-crossing fallback (Goette & Grubmueller p. 449): when neither
    # root lies between the two means the Gaussians are too close to locate a
    # meaningful intersection, and the mean of both is used instead. See the
    # long comment in groscore.py calculate_scores.
    lo, hi = np.minimum(apm, aqm), np.maximum(apm, aqm)
    out[m] = np.where((pick < lo) | (pick > hi), mid, pick)
  return out

def score_structure(W_intro, W_remove, Wtot_f, Wtot_r, dG_release,
                     stages=None, bound=None, n_boot=50000, n_boot_bar=5000,
                     seed=12345):
  """Joint cycle-level bootstrap for one structure.

  Point estimates and 95% CIs for dG_intro, dG_unbind and dG_bind under all three
  estimators (BAR, average, CGI). The bootstrap resamples CYCLES (the sampling
  unit) with a SHARED index across every stream, so the dG_bind CI correctly
  includes the covariance between its components (all are estimated from the same
  cycles) rather than assuming independence. Forward/reverse works are paired by
  cycle. dG_release is analytical and treated as exact (contributes no error).

    bound  stream: forward = W_intro (restraints on),  reverse = W_remove (off)
    unbind stream: forward = Wtot_f  (unbinding),      reverse = Wtot_r (rebinding)
    dG_bind = -(dG_intro + dG_unbind + dG_release)

  STAGED UNBINDING. `stages`, when given, is [(letter, Wf, Wr), ...], the same
  unbinding channel resolved into the ramp stages, which are separated by
  equilibrium holds in both directions. ANY NUMBER of stages is accepted; the
  count comes from the data, not from this file. Two estimates of the same
  dG_unbind then exist and BOTH are reported:

    staged     sum of the per-stage estimates, each on its own forward/reverse
               pair. Each stage dissipates a fraction of the total, so each has
               far more work overlap than the whole ramp does; that is the reason
               the ramp is split at all, and the staged BAR sum is the headline
               dG_bind as shipped.
    one-shot   the whole ramp from the summed works (`Wtot_f`, `Wtot_r`). The hold
               segments are stationary and do exactly zero work, so the stage
               works sum to the work of the full protocol EXACTLY, and Crooks
               applies to that protocol whatever happens in the middle of it.

  The one-shot value needs no assumption about the holds. The staged value needs
  each stage's forward and reverse runs to start from the equilibrium of their own
  end state, which at an internal boundary means the holds must have equilibrated
  the interface, and they are measured not to. In pass 2 the
  forward and reverse states at the same internal ramp boundary differed by a
  median 7-27 kJ/mol in full-strength spring energy over the 39 sound structures,
  and the 350 ps holds closed a median of -6 to 49% of that gap, mostly about a
  tenth (bm_fe_amber_2/_ana/rlx_ramp.py; fe_protocol.py, WHAT THE TAU ARGUMENT DOES
  NOT COVER). The holds settle dH/dlambda but not the interface, so there the
  per-stage channels are not valid Crooks pairs, and the staged BAR and CGI sums
  carry that bias however tight their intervals. The gap between staged and
  one-shot is therefore not redundancy: it measures how far the holds are from
  equilibrium, which is why both are columns. The avg estimate is linear in the
  works, so staged and one-shot avg are the same number, the whole-ramp average.
  This changes no estimator and not the headline: which estimate the pilot verdict
  and the paper read is still open.
  """
  Wi = np.asarray(W_intro, float); Wr = np.asarray(W_remove, float)
  Wf = np.asarray(Wtot_f, float);  Wv = np.asarray(Wtot_r, float)
  ncyc = len(Wi)
  nan = float('nan')

  st = [(L, np.asarray(f, float), np.asarray(v, float)) for L, f, v in (stages or [])]
  # The bound leg is staged on exactly the same terms as the ramp: sub-legs with an
  # equilibrium hold between them, each its own Crooks process, summed. Wi/Wr stay
  # the works of the WHOLE switch and become the one-shot cross-check on that sum,
  # the same role Wtot_f/Wtot_r play for the ramp.
  bd = [(N, np.asarray(f, float), np.asarray(v, float)) for N, f, v in (bound or [])]

  r = dict(n=ncyc,
           intro_avg=nan, intro_avg_ci=nan, intro_cgi=nan, intro_cgi_ci=nan,
           unb_avg=nan, unb_avg_ci=nan, unb_cgi=nan, unb_cgi_ci=nan,
           bind_avg=nan, bind_avg_ci=nan, bind_cgi=nan, bind_cgi_ci=nan,
           intro_bar=nan, intro_bar_ci=nan, unb_bar=nan, unb_bar_ci=nan,
           bind_bar=nan, bind_bar_ci=nan, intro_bar_note="", unb_bar_note="",
           unb1s_bar=nan, unb1s_bar_ci=nan, unb1s_avg=nan,
           intro1s_bar=nan, intro1s_bar_ci=nan, intro1s_avg=nan,
           stages=[L for L, _, _ in st], staged=bool(st),
           bound=[N for N, _, _ in bd], bsplit=bool(bd))
  # One set of per-stage slots per stage actually present.
  for L, _, _ in st:
    r["unb%s_bar" % L] = nan
    r["unb%s_bar_ci" % L] = nan
    r["unb%s_bar_note" % L] = ""
  for N, _, _ in bd:
    r["intro%s_bar" % N] = nan
    r["intro%s_bar_ci" % N] = nan
    r["intro%s_bar_note" % N] = ""
  if ncyc == 0:
    return r

  # Point estimates from the full data.
  r['intro1s_avg'] = float(_stream_avg(Wi, Wr))
  r['intro_avg']   = (float(sum(_stream_avg(f, v) for _, f, v in bd))
                      if bd else r['intro1s_avg'])
  r['unb1s_avg'] = float(_stream_avg(Wf, Wv))
  r['unb_avg']   = (float(sum(_stream_avg(f, v) for _, f, v in st))
                    if st else r['unb1s_avg'])
  r['bind_avg']  = -(r['intro_avg'] + r['unb_avg'] + dG_release)
  if ncyc >= 3:                                   # CGI needs the per-cycle variance
    # The STAGED sum over the bound sub-legs, like intro_avg above and like the
    # bootstrap that gives this column its CI. It used to be the one-shot CGI of the
    # summed switch, so point and interval described two different estimators: on
    # pass 2 (bm_fe_amber_2/_ana/digest/corr.txt F2) they differed by up to
    # 8.04 kJ/mol (1E6E, 1.35 pKD), and 4.27 in pass 1 (1A2K).
    if bd:
      r['intro_cgi'] = float(sum(_stream_cgi(f[None, :], v[None, :])[0]
                                 for _, f, v in bd))
    else:
      r['intro_cgi'] = float(_stream_cgi(Wi[None, :], Wr[None, :])[0])
    if not st:
      r['unb_cgi'] = float(_stream_cgi(Wf[None, :], Wv[None, :])[0])
    else:
      r['unb_cgi'] = float(sum(_stream_cgi(f[None, :], v[None, :])[0]
                               for _, f, v in st))
    if np.isfinite(r['intro_cgi']) and np.isfinite(r['unb_cgi']):
      r['bind_cgi'] = -(r['intro_cgi'] + r['unb_cgi'] + dG_release)

  # BAR, the headline estimator. Every stream passes its reverse works RAW: that
  # is estimators.py's convention and this file's, so unlike _stream_avg and
  # _stream_cgi, which negate internally, nothing is flipped here.
  #
  # The overlap guard inside est.bar runs before the solve and returns NaN plus a
  # reason rather than a number, because a degenerate BAR is indistinguishable
  # from a good one by inspection: at the dissipation the whole unbinding ramp
  # runs at, the solver returns a confident value with the wrong sign. A leg with
  # no overlap therefore has no BAR column at all, which is the intended outcome,
  # and is exactly what unb1s_bar is expected to keep reading while the stage
  # columns beside it do not.
  r['intro1s_bar'], intro1s_note      = est.bar(Wi, Wr, RT)
  if not bd:
    r['intro_bar'], r['intro_bar_note'] = r['intro1s_bar'], intro1s_note
  else:
    # Same rule as the stages: name the sub-leg that failed, not just the channel.
    bvals = []
    for N, f, v in bd:
      b, note = est.bar(f, v, RT)
      r["intro%s_bar" % N], r["intro%s_bar_note" % N] = b, note
      bvals.append(b)
    if all(np.isfinite(x) for x in bvals):
      r['intro_bar'] = float(sum(bvals))
  r['unb1s_bar'], unb1s_note          = est.bar(Wf, Wv, RT)
  if not st:
    r['unb_bar'], r['unb_bar_note'] = r['unb1s_bar'], unb1s_note
  else:
    # Name the stage that failed, not just the channel: "the unbinding BAR is
    # missing" is only actionable if it says which stage. The per-stage tokens are
    # kept apart here and joined by the writer, which owns the channel suffix.
    vals = []
    for L, f, v in st:
      b, note = est.bar(f, v, RT)
      r["unb%s_bar" % L], r["unb%s_bar_note" % L] = b, note
      vals.append(b)
    if all(np.isfinite(v) for v in vals):
      r['unb_bar'] = float(sum(vals))
  if np.isfinite(r['intro_bar']) and np.isfinite(r['unb_bar']):
    r['bind_bar'] = -(r['intro_bar'] + r['unb_bar'] + dG_release)

  if ncyc < 2:
    return r

  # Joint bootstrap: one shared cycle-index resample drives every stream.
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, ncyc, size=(n_boot, ncyc))
  Wi_b, Wr_b, Wf_b, Wv_b = Wi[idx], Wr[idx], Wf[idx], Wv[idx]

  ia = (sum(_stream_avg(f[idx], v[idx], axis=1) for _, f, v in bd)
        if bd else _stream_avg(Wi_b, Wr_b, axis=1))
  ua = (sum(_stream_avg(f[idx], v[idx], axis=1) for _, f, v in st)
        if st else _stream_avg(Wf_b, Wv_b, axis=1))
  r['intro_avg_ci'] = 1.96 * float(np.std(ia))
  r['unb_avg_ci']   = 1.96 * float(np.std(ua))
  r['bind_avg_ci']  = 1.96 * float(np.std(-(ia + ua + dG_release)))

  if ncyc >= 3:
    ic = (sum(_stream_cgi(f[idx], v[idx]) for _, f, v in bd)
          if bd else _stream_cgi(Wi_b, Wr_b))
    uc = (sum(_stream_cgi(f[idx], v[idx]) for _, f, v in st)
          if st else _stream_cgi(Wf_b, Wv_b))
    if np.isfinite(ic).sum() > 1:
      r['intro_cgi_ci'] = 1.96 * float(np.nanstd(ic))
    if np.isfinite(uc).sum() > 1:
      r['unb_cgi_ci'] = 1.96 * float(np.nanstd(uc))
    both = np.isfinite(ic) & np.isfinite(uc)
    if both.sum() > 1:
      r['bind_cgi_ci'] = 1.96 * float(np.std(-(ic[both] + uc[both] + dG_release)))

  # BAR CIs, on a PREFIX of the same index. avg and CGI are closed forms over a
  # resampled row and cost microseconds for all 50000; BAR is a root-find per row
  # and costs about 1000x that, so it runs on the first n_boot_bar rows only. The
  # bootstrap standard deviation has relative error 1/sqrt(2B), so 5000 rows know
  # their own CI to 1%, which is far inside the CI itself.
  #
  # A prefix, not a fresh draw: the same cycles must resample together across all
  # streams, or bind_bar_ci silently becomes the independent-error combination
  # instead of carrying the intro/unbind covariance. That covariance is the reason
  # this bootstrap is joint at all, and with the ramp staged it also carries the
  # covariance BETWEEN the stages, which share every cycle by construction.
  nb = min(n_boot_bar, n_boot)
  jdx = idx[:nb]
  def boot(f, v, ok):
    return est.bar_bootstrap(f, v, RT, jdx) if ok else None
  i1 = boot(Wi, Wr, np.isfinite(r['intro1s_bar']))
  if i1 is not None and np.isfinite(i1).sum() > 1:
    r['intro1s_bar_ci'] = 1.96 * float(np.nanstd(i1))
  if not bd:
    ib = i1
  else:
    bper = []
    for N, f, v in bd:
      b = boot(f, v, np.isfinite(r["intro%s_bar" % N]))
      if b is not None and np.isfinite(b).sum() > 1:
        r["intro%s_bar_ci" % N] = 1.96 * float(np.nanstd(b))
      bper.append(b)
    # As for the stages: the sum is undefined if any sub-leg is unresolvable, and
    # dG_intro is already NaN in that case.
    ib = None if any(b is None for b in bper) else sum(bper)
  if ib is not None and np.isfinite(ib).sum() > 1:
    r['intro_bar_ci'] = 1.96 * float(np.nanstd(ib))
  s1 = boot(Wf, Wv, np.isfinite(r['unb1s_bar']))
  if s1 is not None and np.isfinite(s1).sum() > 1:
    r['unb1s_bar_ci'] = 1.96 * float(np.nanstd(s1))

  if not st:
    ub = s1
  else:
    per = []
    for L, f, v in st:
      b = boot(f, v, np.isfinite(r["unb%s_bar" % L]))
      if b is not None and np.isfinite(b).sum() > 1:
        r["unb%s_bar_ci" % L] = 1.96 * float(np.nanstd(b))
      per.append(b)
    # The staged CI needs every stage, since the sum is undefined if one is not
    # resolvable; a missing stage leaves dG_unbind itself NaN anyway.
    ub = None if any(b is None for b in per) else sum(per)
  if ub is not None and np.isfinite(ub).sum() > 1:
    r['unb_bar_ci'] = 1.96 * float(np.nanstd(ub))
  if ib is not None and ub is not None:
    both = np.isfinite(ib) & np.isfinite(ub)
    if both.sum() > 1:
      r['bind_bar_ci'] = 1.96 * float(np.std(-(ib[both] + ub[both] + dG_release)))
  return r

#------------------------------------------------------

def compact_ranges(nums):
  """[1,2,3,7,9,10] -> '1-3,7,9-10' (SLURM array index specification)."""
  parts, start, prev = [], None, None
  for n in sorted(nums):
    if start is None:
      start = prev = n
    elif n == prev + 1:
      prev = n
    else:
      parts.append(str(start) if start == prev else "%d-%d" % (start, prev))
      start = prev = n
  if start is not None:
    parts.append(str(start) if start == prev else "%d-%d" % (start, prev))
  return ",".join(parts)

def sbatch_parsable(path):
  """Submit a job script and return its job id (None if submission failed)."""
  try:
    out = subprocess.run(["sbatch", "--parsable", path],
                         capture_output=True, text=True, check=True).stdout.strip()
  except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
    print("  sbatch failed for %s: %s" % (path, e))
    return None
  return out.split(";")[0] if out else None

# The per-structure leg mdps, i.e. every mdp make_boresch.py writes a pull block
# into. Named once because setup_is_complete checks them and then deletes them,
# and a list that drifted between those two would leave a structure that fails its
# own check forever without ever clearing what makes it fail.
LEG_MDPS = tuple(sorted({l["mdp"] for l in P.legs()}))

# npt_c's per-structure mdp (v1.5). It ships with 'pull = yes' and no coordinates,
# which grompp refuses, so a usable copy is one make_boresch.py has finished: it
# appended the elastic network (a pull-ngroups line) or switched pull off for a
# structure without one ('pull = no'). The same test as job_fe.run's setup
# post-condition, so the runner and this file agree on what a finished setup is.
NPT_C_MDP = "npt_c_fe.mdp"
_PULL_OFF = re.compile(r"^pull\s*=\s*no(\s|;|$)")

#------------------------------------------------------
#
# PROTOCOL IDENTITY. A run directory, and every structure in it, must hold the
# cycles of ONE protocol, and nothing in a result row says which: v1.5 kept the
# 43-field row of v1.4, so once written a v1.4 row and a v1.5 row look the same. A
# pass-2 directory reused for the pilot therefore counted its v1.4 rows as finished
# v1.5 cycles and pooled them with new ones (review CYC-4: structure directories
# recreated inside bm_fe_amber_2 got 'all 20 cycles complete' and the scorer
# reported pass-2 numbers), and its set-up structures got cycle tasks that all stop
# at job_fe.run's npt_c_fe.mdp guard (CYC-3). So the identity is kept beside the
# rows, in two places, each written once and only compared afterwards:
#
#   run_config.gs, key protocol_id: fe_protocol.PROTOCOL_ID at the FIRST submission
#       into a directory and never later, so a directory that already holds work of
#       an unknown protocol cannot acquire an id by being touched again
#   <sid>/protocol_id.gs: written by job_fe.run when a setup completes; a cycle
#       refuses to run unless it matches the PROTOCOL_ID of the code running it
#
# PROTOCOL_ID hashes the leg table and the works layout, so it moves whenever what a
# cycle simulates or what a row means moves.
PROTOCOL_KEY = "protocol_id"
PROTOCOL_FILE = "protocol_id.gs"

def _protocol_token(text):
  """The id in the text of a protocol_id.gs: the first token of its first line that
  is not a comment, or None."""
  for line in text.splitlines():
    tok = line.split()
    if tok and not tok[0].startswith("#"):
      return tok[0]
  return None

def read_protocol_id(path):
  """The id in the protocol_id.gs at `path`, or None if it is absent or empty."""
  try:
    with open(path) as f:
      return _protocol_token(f.read())
  except OSError:
    return None

def _foreign(done, pid, want):
  """Why a setup with this (setup.done present, protocol_id.gs content) is not one of
  protocol `want`, or None. v1.4 and earlier wrote no protocol_id.gs, so a finished
  setup without one predates v1.5. A protocol_id.gs naming another id marks the
  directory as that protocol's whether or not its setup finished: a setup rerun on
  top of it would reuse that protocol's box, ligands and probe."""
  if pid is not None and pid != want:
    return "set up for protocol %s" % pid
  if pid is None and done:
    return "setup.done but no %s: set up before v1.5" % PROTOCOL_FILE
  return None

def foreign_setup(d, want):
  """_foreign() for the structure directory d."""
  return _foreign(os.path.isfile(os.path.join(d, "setup.done")),
                  read_protocol_id(os.path.join(d, PROTOCOL_FILE)), want)

def run_dir_history():
  """What in the current directory shows that work was already submitted or run here.

  Empty is what makes a submission the FIRST one, the only point at which the
  directory's protocol id is recorded. Each entry is written only by an earlier
  submission, setup, cycle or scoring pass. run_config.gs is not one of them: every
  invocation writes it the moment this module is imported."""
  seen = [f for f in ("struct_map.gs", "array_submit.run", "results_fe.gs",
                      "results_analytical.gs", "scores_fe.gs") if os.path.isfile(f)]
  seen += ["%s/" % d for d in ("results_fe.d", "results_analytical.d",
                               "results_qc.d", "slurm_fe")
           if os.path.isdir(d) and os.listdir(d)]
  try:
    with open("results_0.gs") as f:
      # main() creates it holding only its header line just before a submission
      if any(l.strip() and not l.lstrip().startswith("#") for l in f):
        seen.append("results_0.gs")
  except OSError:
    pass
  return seen

def protocol_gate():
  """None if this directory may take cycles of fe_protocol.PROTOCOL_ID, else the
  lines of the refusal, the first of them a one-line summary.

  The id is recorded here, by the first submission into a directory with no history.
  A directory with history but no recorded id was started before v1.5: its rows,
  setups and checkpoints belong to an older protocol, and any cycle submitted into
  it would be counted as done by its rows and pooled with them, or would continue
  its checkpoints into v1.5 legs (CYC-1). A directory recorded under another id
  holds cycles of a protocol this code does not run. Either way nothing is
  submitted, and nothing in the directory is touched."""
  rec = read_run_config().get(PROTOCOL_KEY)
  cur = P.PROTOCOL_ID
  if rec == cur:
    return None
  if rec is None:
    hist = run_dir_history()
    if not hist:
      write_run_config(**{PROTOCOL_KEY: cur})
      print("Recorded FE protocol %s in %s: this directory takes cycles of that "
            "protocol only.\n" % (cur, RUN_CONFIG))
      return None
    return ["ERROR: nothing was submitted: this run directory predates v1.5.",
            "  It already holds work (%s), but %s records no %s, so its rows,"
            % (", ".join(hist), RUN_CONFIG, PROTOCOL_KEY),
            "  setups and checkpoints belong to an older FE protocol. Its rows look",
            "  exactly like this protocol's, so new cycles here would be counted as done",
            "  by them and pooled with them, and its set-up structures would carry old",
            "  checkpoints into new legs. Start a fresh run directory for this code",
            "  (sp.gs and the structure inputs only). What is here is still scored,",
            "  below and on every later invocation here; only submitting is refused."]
  return ["ERROR: nothing was submitted: this run directory holds another FE protocol.",
          "  %s records %s %s, and this code runs %s. Cycles of two"
          % (RUN_CONFIG, PROTOCOL_KEY, rec, cur),
          "  protocols must never be pooled. Start a fresh run directory for this code.",
          "  What is here is still scored, below and on every later invocation here;",
          "  only submitting is refused."]

def protocol_warning():
  """The warning a scoring pass prints when this directory's recorded protocol is not
  the one this code runs, as lines; [] when it is."""
  rec = read_run_config().get(PROTOCOL_KEY)
  if rec == P.PROTOCOL_ID:
    return []
  if rec is None:
    return ["WARNING: %s records no %s, so this run directory predates v1.5"
            % (RUN_CONFIG, PROTOCOL_KEY),
            "  and its rows cannot be tied to any protocol. They are scored as they",
            "  are; do not pool these numbers with v1.5 results."]
  return ["WARNING: this run directory holds FE protocol %s, and this code runs %s."
          % (rec, P.PROTOCOL_ID),
          "  Its rows are scored with this code's row layout and estimators."]

def setup_problems(sid):
  """What the setup in ./sid lacks of what the cycles consume, [] if nothing.

  Looks and changes nothing; setup_is_complete decides what to do about it."""
  d = "./%s" % sid
  bad = []
  for f in ("numpertres.gs", "boresch_analytical.gs", "boresch_anchors.gs"):
    p = os.path.join(d, f)
    if not os.path.isfile(p) or os.path.getsize(p) == 0:
      bad.append(f)
  for m in LEG_MDPS:
    p = os.path.join(d, m)
    if not os.path.isfile(p):
      bad.append(m)
      continue
    with open(p) as fh:
      if not any(l.startswith("pull-ngroups") for l in fh):
        bad.append("%s (no pull block)" % m)
  p = os.path.join(d, NPT_C_MDP)
  if not os.path.isfile(p):
    bad.append(NPT_C_MDP)
  else:
    with open(p) as fh:
      if not any(l.startswith("pull-ngroups") or _PULL_OFF.match(l) for l in fh):
        bad.append("%s (no elastic-network block and not pull = no)" % NPT_C_MDP)
  return bad

def setup_is_complete(sid):
  """True only if setup.done is backed by the artefacts the cycles consume, and the
  setup is one of THIS protocol.

  setup.done used to be enough on its own, which made a bad setup permanent: the
  marker suppressed the setup job here, and job_fe.run exits early on the same
  marker, so nothing could ever redo the work. A directory left with a 0-byte
  numpertres.gs and leg .mdp files carrying no pull block reproduced its failure
  on every resubmission. When the marker is not backed up, clear it and the
  half-written restraints so the setup job can run again.

  A setup of another protocol, or from before v1.5, is never complete here, and it
  is never cleared either. A v1.4 directory passed the old test, since every leg mdp
  it names carries a pull block there (CYC-3), and clearing its setup.done would
  rebuild only the restraints, on the v1.4 box, ligands and probe, after which its
  unfinished cycles continue their v1.4 checkpoints into v1.5 legs (CYC-1).
  setup_and_submit holds such a structure back before it gets here."""
  d = "./%s" % sid
  if not os.path.isfile(os.path.join(d, "setup.done")):
    return False
  if foreign_setup(d, P.PROTOCOL_ID):
    return False
  bad = setup_problems(sid)
  if not bad:
    return True
  print("  %s: setup.done is present but the restraints are not (%s); "
        "redoing setup." % (sid, ", ".join(bad)))
  for f in ("setup.done", "numpertres.gs", "numpertres.tmp",
            "boresch_failed.gs", NPT_C_MDP) + LEG_MDPS:
    try:
      os.remove(os.path.join(d, f))
    except OSError:
      pass
  return False

# Stage-0 verdicts the structure itself decides: its input, its loops, its ligand
# chemistry. A new setup would reach the same one, so a structure whose last
# results_0.gs status is one of these gets no job. A --restart used to submit its
# setup again, and its cycle array, whose tasks only read the status and exited
# (bm_fe_amber_3a: 1GRN, 1WQ1, 1RLB, three setups and 60 empty tasks). The other
# failures (FAILED, PROBE_FAIL, BORESCH_FAIL, ...) can be a step that a new attempt
# gets through, and are submitted again as before.
STAGE0_FINAL = ("LIGAND_FAIL", "BROKEN", "ENTANGLED")

# ligand_failed.gs lines that record a crash or a kill rather than a verdict on the
# ligand: parametrize_ligand.py's first line, and the runners' line for one that
# exited without recording a reason. A LIGAND_FAIL with such a line, or without any
# line (merge_ligand.py or fuse_glycans.py crashed), may pass on another attempt, so
# it is not held.
LIGAND_CRASH = ("parametrisation started but did not finish", "without recording a reason")


def ligand_fail_is_final(sid):
    """True when ./sid/ligand_failed.gs holds lines and none records a crash or a kill."""
    try:
        with open(os.path.join(".", sid, "ligand_failed.gs")) as f:
            lines = [l for l in f if l.strip()]
    except OSError:
        return False
    return bool(lines) and not any(t in l for l in lines for t in LIGAND_CRASH)

def hold_reason(sid, have, wants, status=None):
  """(TOKEN, detail) if the structure directory ./sid must get no job, else None.

  `have` is the set of cycles results_fe.d already holds for it, `wants` whether
  anything would be submitted and `status` its last results_0.gs status, if that is
  a failure (read_status). Looks and changes nothing.

    OTHER_PROTOCOL      its setup belongs to another protocol or predates v1.5.
                        Held even with nothing to submit: this code must not even
                        install its job.run there.
    STAGE0_FINAL        its setup ended in one of STAGE0_FINAL, which another
                        setup would repeat; a LIGAND_FAIL only when ligand_failed.gs
                        holds a verdict, not a crash (ligand_fail_is_final).
    ROWS_WITHOUT_SETUP  results_fe.d holds rows for it, but its directory holds no
                        record of any setup, neither setup.done nor protocol_id.gs:
                        a structure directory recreated inside a run directory that
                        already ran it (CYC-4). The setup it would get starts from a
                        new probe and builds new restraints, and its cycles would be
                        counted as done by the old rows and pooled with them.
                        job_fe.run cannot see this, since the rows live up here.

  A directory whose protocol_id.gs names this protocol is not held, with or without
  setup.done: there job_fe.run decides. It lets a setup run again over existing
  cycles only while the solvated system it built is still there, and then rebuilds
  the same restraints (make_boresch.py has no random input), which is also the case
  setup_is_complete creates when it clears setup.done over a missing restraint."""
  d = "./%s" % sid
  why = foreign_setup(d, P.PROTOCOL_ID)
  if why:
    return ("OTHER_PROTOCOL", why)
  if status in STAGE0_FINAL and (status != "LIGAND_FAIL" or ligand_fail_is_final(sid)):
    return ("STAGE0_FINAL", "last stage-0 status %s" % status)
  if (have and wants and not os.path.isfile(os.path.join(d, "setup.done"))
      and not os.path.isfile(os.path.join(d, PROTOCOL_FILE))):
    return ("ROWS_WITHOUT_SETUP", "%d result rows, but no setup.done and no %s"
            % (len(have), PROTOCOL_FILE))
  return None

def inject_job_run(sid, job_src):
  """Install job_src as job.run inside ./<sid>.tar.gz. None on success.

  (TOKEN, detail) when the archived setup belongs to another protocol or predates
  v1.5, by the same test hold_reason applies to a directory; the archive is then
  left exactly as it was. Rewriting streams the whole archive anyway, so reading
  its protocol_id.gs on the way costs nothing extra. The new job.run goes straight
  from job_src into the archive: the old code staged it in ./<sid>/, the very path
  an unpacked structure lives at, and a failure part way left that directory
  behind to be taken for one."""
  import io, tarfile
  arc, new = "./%s.tar.gz" % sid, "./%s_new.tar.gz" % sid
  done, pid = False, None

  def executable(ti):
    ti.mode = 0o755
    return ti

  try:
    with tarfile.open(arc, "r:gz") as old_tar, tarfile.open(new, "w:gz") as new_tar:
      for m in old_tar:
        name = m.name[2:] if m.name.startswith("./") else m.name
        if name.endswith("/job.run"):
          continue
        if name == "%s/setup.done" % sid:
          done = True
        if m.isreg() and name == "%s/%s" % (sid, PROTOCOL_FILE):
          data = old_tar.extractfile(m).read()
          pid = _protocol_token(data.decode("utf-8", "replace"))
          new_tar.addfile(m, io.BytesIO(data))
          continue
        new_tar.addfile(m, old_tar.extractfile(m) if m.isreg() else None)
      new_tar.add(job_src, arcname="./%s/job.run" % sid, filter=executable)
  except (OSError, tarfile.TarError) as e:
    try:
      os.remove(new)
    except OSError:
      pass
    return ("INJECT_FAILED", "%s.tar.gz not rewritten: %s" % (sid, e))
  why = _foreign(done, pid, P.PROTOCOL_ID)
  if why:
    os.remove(new)
    return ("OTHER_PROTOCOL", "archived, " + why)
  os.replace(new, arc)
  print("Injected job_fe.run into %s.tar.gz" % sid)
  return None

# What to do about each reason a structure was held back, printed once per reason
# that occurred, after the per-structure list.
HOLD_HELP = {
  "OTHER_PROTOCOL": (
    "Their setup belongs to another FE protocol, or predates v1.5: the box, ligands,",
    "probe and restraints are that protocol's, and unfinished cycles would carry its",
    "checkpoints into this protocol's legs. Set them up again from their inputs, in",
    "a fresh run directory. Do not just remove setup.done: the setup would rebuild",
    "only the restraints, on the old box, ligands and probe."),
  "STAGE0_FINAL": (
    "Their setup ended in a verdict on the structure itself, which a new setup would",
    "repeat; the reason is in its setup log (for LIGAND_FAIL also ligand_failed.gs; a",
    "LIGAND_FAIL that records a crash or a kill there is submitted again instead).",
    "To set one up again on purpose, after fixing its input or the code that stopped",
    "it, append the line '<ID> SETUP_RUNNING' to results_0.gs and run this again."),
  "ROWS_WITHOUT_SETUP": (
    "They have result rows in results_fe.d, but their directory holds no setup at",
    "all, as after recreating it. A new setup builds new restraints on a new probe,",
    "and its cycles would be counted and pooled with these rows. Move their rows",
    "out of results_fe.d/ and results_qc.d/ first, or use a fresh run directory."),
  "ARCHIVED": (
    "They are archived with cycles still missing. The job.run inside an archive is",
    "the one that last ran there and may predate this code. Pass --inject-job-run",
    "to install this job_fe.run into the archive and submit the missing cycles."),
  "INJECT_FAILED": (
    "Their archive could not be rewritten and was left as it was.",),
}

def report_held(held):
  """Print the structures held back from submission, [(sid, TOKEN, detail), ...]."""
  if not held:
    return
  print("NOT SUBMITTED: %d structure%s held back and left untouched."
        % (len(held), "" if len(held) == 1 else "s"))
  for token in HOLD_HELP:
    hits = [(s, d) for s, t, d in held if t == token]
    if not hits:
      continue
    print("  %s:" % token)
    for s, d in hits:
      print("    %-10s %s" % (s, d))
    for line in HOLD_HELP[token]:
      print("    " + line)
  print("")

def unarchived_complete(sid, numruns):
  """True when ./sid holds results_fe_c<N>.done for every cycle 1..numruns but is
  still a directory, with no <sid>.tar.gz beside it.

  job_fe.run archives a structure only in the task that finishes its last cycle, so
  one whose archive failed there stayed a directory that no later run tried again:
  with every cycle done, --restart had nothing to submit for it (1AKJ in
  bm_fe_amber_3a, whose tar met a full disk). Such a directory gets a
  job_fe.run --archive job. One with a <sid>.tar.gz beside it is reported and left
  to the user, as job_fe.run --archive refuses it too."""
  d = "./%s" % sid
  if not os.path.isdir(d) or os.path.exists("./%s.tar.gz" % sid) or numruns < 1:
    return False
  return all(os.path.isfile(os.path.join(d, "results_fe_c%d.done" % c))
             for c in range(1, numruns + 1))

def unmarked_cycles(sid, numruns):
  """Cycles 1..numruns of ./sid without results_fe_c<N>.done, in a directory whose
  setup completed and that has marked at least one cycle done; [] otherwise.

  Called when every cycle has a complete row. job_fe.run writes the row first and
  the .done after it, so a task killed between the two (or a touch on a full disk)
  leaves a complete row that nothing marks done, and the structure was never
  archived. A cycle task for it writes the .done (and archives after the last).
  The directory must be the one the rows came from: a structure directory
  recreated inside a run directory that already ran it has rows and no .done at
  all, and must not get cycles (CYC-4, hold_reason)."""
  d = "./%s" % sid
  if not os.path.isfile(os.path.join(d, "setup.done")) \
     or not glob.glob(os.path.join(d, "results_fe_c*.done")):
    return []
  return [c for c in range(1, numruns + 1)
          if not os.path.isfile(os.path.join(d, "results_fe_c%d.done" % c))]

def setup_and_submit(structids, structchains):
  """Write per-structure run.gs, install job_fe.run, build and submit the jobs.

  Called only once protocol_gate() has let this directory take cycles. A structure
  whose continuation would mix protocols or setups is held back and left untouched
  (hold_reason, inject_job_run), and so is an archived one without
  --inject-job-run; report_held lists them all, with what to do, at the end."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  job_src = os.path.join(script_dir, "job_fe.run")
  if not os.path.isfile(job_src):
    print("Error: job_fe.run not found in %s" % script_dir); sys.exit(1)
  # Checked before anything is written, so a mistyped --slurm leaves no structure
  # half prepared.
  slurm_template = os.path.join(script_dir, "slurm", args.slurm + ".sh")
  if not os.path.isfile(slurm_template):
    print("Error: SLURM template not found: %s" % slurm_template); sys.exit(1)
  with open(slurm_template) as t:
    template = t.read().rstrip("\n")

  with open("struct_map.gs", "w") as f:
    f.write("# Array_Index Structure_ID\n")
    for i, sid in enumerate(structids):
      f.write("%d %s\n" % (i, sid))

  # -n is the TOTAL number of cycles wanted, so re-running with a larger -n adds
  # cycles. Only the cycles that have no complete result yet are submitted, so
  # topping up a structure costs one array task per missing cycle rather than
  # re-walking everything. Cycles whose result contains NaN count as missing and
  # are recomputed. The total is passed via GROSCORE_NUMCYCLES because run.gs
  # cannot be rewritten inside an archived structure.
  done_cycles = {}
  for sid, rows_ in read_works("results_fe.gs").items():
    done_cycles[sid] = {int(r[0]) for r in rows_}

  stage0 = read_status("results_0.gs", structids)
  live, held = [], []
  for i, sid in enumerate(structids):
    have = done_cycles.get(sid, set())
    lack = [c for c in range(1, args.numruns + 1) if c not in have]
    # --sequential runs whatever a structure lacks inside one job, so there every
    # structure has work; the array path submits only the missing cycles.
    wants = args.sequential or bool(lack)
    if os.path.isdir("./%s" % sid):
      why = hold_reason(sid, have, wants, stage0.get(sid))
      if why:
        held.append((sid,) + why)
        continue
      print("Setting up %s." % sid)
      if os.path.exists("./%s.tar.gz" % sid):
        print("  Both %s/ and %s.tar.gz exist; the directory is used. Compare them and"
              " remove one by hand." % (sid, sid))
      if not args.restart:
        with open("./%s/run.gs" % sid, "w") as f:
          f.write("%s %d %d %s %d\n" % (structchains[i], args.numruns * 2,
                                        1 if args.cutout else 0, args.forcefield,
                                        1 if args.ligand_param else 0))
      # Install atomically: copy to a temp file, then rename over job.run. A
      # rename gives the new file a fresh inode, so if a previous SLURM job is
      # still executing the old job.run its bash keeps reading the old inode and
      # finishes cleanly. An in-place copy (shutil.copy onto the live path)
      # truncates and rewrites the same inode, corrupting a running script
      # mid-execution (observed: interleaved cycle-based and PUSH_IDX naming).
      dst = "./%s/job.run" % sid
      tmp = "./%s/.job.run.%d.tmp" % (sid, os.getpid())
      shutil.copy(job_src, tmp)
      os.chmod(tmp, 0o755)
      os.replace(tmp, dst)
      live.append(sid)
    elif os.path.isfile("./%s.tar.gz" % sid):
      # An archive with work to do is submitted only with this code's runner in it.
      # The setup job unpacks it and its cycles then run the job.run inside, which
      # is whatever ran there last: v1.4's would run the v1.5 legs as a plain chain,
      # npt_c without the network and boundrev2 straight after bindrevA (CYC-2). Up
      # to v1.4 this branch printed that it was skipping the archive and then
      # submitted it anyway.
      if not lack:
        # Nothing to run. The array path below says so; --sequential, which would
        # unpack and re-archive it for nothing, leaves it out.
        if args.sequential:
          print("  %s: all %d cycles complete, nothing to submit." % (sid, args.numruns))
        else:
          live.append(sid)
        continue
      if not args.inject_job_run:
        held.append((sid, "ARCHIVED", "%d of %d cycles missing" % (len(lack), args.numruns)))
        continue
      why = inject_job_run(sid, job_src)
      if why:
        held.append((sid,) + why)
        continue
      live.append(sid)
    else:
      print("Structure %s: directory doesn't exist." % sid)
      with open("results_0.gs", "a") as f:
        f.write("%s NODIR\n" % sid)

  if not live:
    print("No structures to submit.\n")
    report_held(held)
    return

  def extract_snippet(sid):
    # A structure archived after finishing its cycles is unpacked again when more
    # cycles are added. .archive.lock is inside the tarball, so drop it or the
    # completed structure could never be re-archived.
    return ("if [[ ! -d \"%s\" && -f \"%s.tar.gz\" ]]; then\n"
            "  tar -xzf \"%s.tar.gz\"\n  rm \"%s.tar.gz\"\n"
            "  rm -rf \"%s/.archive.lock\"\nfi\n" % (sid, sid, sid, sid, sid))

  if args.sequential:
    # Legacy layout: one task per structure, running setup + all cycles
    # sequentially inside a single job. The total goes along as GROSCORE_NUMCYCLES,
    # as on the array path: --restart never rewrites run.gs, so without it
    # '-n <more> --restart --sequential' ran the old total and added no cycle
    # (review PY-5).
    if args.run_local:
      jobs = [{"name": sid, "dir": sid, "argv": ["./job.run"],
               "env": {"GROSCORE_NUMCYCLES": str(args.numruns)},
               "archive": "%s.tar.gz" % sid, "log": "job_local.out"} for sid in live]
      launch_local(jobs, args.ngpus, args.jobs_per_gpu, args.threads_per_job)
      print("")
      report_held(held)
      return
    # The array runs over struct_map.gs indices, so it lists the live ones only: it
    # used to span 0..N-1, which ran every held-back structure as well.
    keep = set(live)
    idx = [i for i, sid in enumerate(structids) if sid in keep]
    with open("array_submit.run", "w") as f:
      f.write(template + "\n")
      f.write("#SBATCH --array=%s\n\n" % compact_ranges(idx))
      f.write("export GROSCORE_NUMCYCLES=%d\n" % args.numruns)
      f.write("STRUCT_ID=$(awk -v idx=\"$SLURM_ARRAY_TASK_ID\" '$1 == idx {print $2}' struct_map.gs)\n")
      # .archive.lock goes, as in extract_snippet: archives made before job_fe.run
      # left it out carry it, and it would stop the structure from being archived.
      f.write("if [[ ! -d \"$STRUCT_ID\" && -f \"${STRUCT_ID}.tar.gz\" ]]; then\n")
      f.write("  tar -xzf \"${STRUCT_ID}.tar.gz\"\n  rm \"${STRUCT_ID}.tar.gz\"\n"
              "  rm -rf \"${STRUCT_ID}/.archive.lock\"\nfi\n")
      f.write("cd $STRUCT_ID\n./job.run\n")
    # Flushed first: sbatch writes to the same stdout, and into a redirected log its
    # line otherwise lands above everything printed so far.
    sys.stdout.flush()
    os.system("sbatch array_submit.run")
    print("Submitted 1 array of %d structures (sequential cycles).\n" % len(idx))
    report_held(held)
    return

  # Default: per structure, a one-off setup job plus a job array over cycles.
  # The cycles are embarrassingly parallel, but every cycle needs the restraints
  # (elastic network + interface + Boresch) that the setup defines, so the array
  # depends on the setup job. `afterany` rather than `afterok`: on setup failure
  # the tasks still start and exit cleanly after reading the status, instead of
  # sitting pending forever with an unsatisfiable dependency. job.run --cycle
  # additionally waits on setup.done, covering the case where a task is released
  # or started before the setup files are in place.
  if not args.run_local:
    os.makedirs("slurm_fe", exist_ok=True)
  throttle = ("%%%d" % args.array_throttle) if args.array_throttle > 0 else ""

  local_jobs = []
  n_setup = n_array = n_archive = 0
  for sid in live:
    have = done_cycles.get(sid, set())
    missing = [c for c in range(1, args.numruns + 1) if c not in have]
    if not missing and os.path.isdir("./%s" % sid):
      nomark = unmarked_cycles(sid, args.numruns)
      if nomark:
        print("  %s: complete rows but no results_fe_c<N>.done for cycle(s) %s;"
              " submitting them to write it." % (sid, compact_ranges(nomark)))
        missing = nomark
    if not missing:
      busy = [p for p in ("%s.tar.gz.part" % sid, "%s/.archive.lock" % sid)
              if os.path.exists(p)]
      if busy and os.path.isdir("./%s" % sid):
        print("  %s: all %d cycles complete; an archive is under way or was interrupted"
              " (%s). If no task archives it, remove %s by hand."
              % (sid, args.numruns, " and ".join(busy), " and ".join(busy)))
        continue
      if not unarchived_complete(sid, args.numruns):
        print("  %s: all %d cycles complete, nothing to submit." % (sid, args.numruns))
        continue
      # Every cycle done, never archived: an archive-only job (unarchived_complete).
      # Locally its log goes beside the directory, which the job removes.
      if args.run_local:
        local_jobs.append({"name": "archive_%s" % sid, "dir": sid,
                           "argv": ["./job.run", "--archive"],
                           "env": {"GROSCORE_NUMCYCLES": str(args.numruns)},
                           "log": "../job_local_archive_%s.out" % sid})
        n_archive += 1
        print("  %s: all %d cycles complete but not archived; queueing its archive."
              % (sid, args.numruns))
        continue
      archive_path = os.path.join("slurm_fe", "archive_%s.run" % sid)
      with open(archive_path, "w") as f:
        f.write(template + "\n")
        f.write("#SBATCH --job-name=fearch_%s\n" % sid)
        # A tar and a gzip -t, not a cycle: a limit far below the template's, so the
        # scheduler can fit it in. The largest bm_fe_amber_3a structure is 48 GB.
        f.write("#SBATCH --time=08:00:00\n\n")
        f.write("export GROSCORE_NUMCYCLES=%d\n" % args.numruns)
        f.write("cd %s || exit 1\n./job.run --archive\n" % sid)
      sys.stdout.flush()
      if os.system("sbatch %s" % archive_path) == 0:
        n_archive += 1
        print("  %s: all %d cycles complete but not archived; submitting its archive job."
              % (sid, args.numruns))
      continue

    if args.run_local:
      # Same shape as the SLURM path: one setup job, then the missing cycles
      # depending on it. The dependency is "after any", like --dependency=afterany
      # above, so a cycle whose setup failed still starts, reads the stage-0
      # status and exits cleanly instead of waiting forever.
      deps = []
      if not setup_is_complete(sid):
        local_jobs.append({"name": "setup_%s" % sid, "dir": sid,
                           "argv": ["./job.run", "--setup"],
                           "archive": "%s.tar.gz" % sid, "log": "job_local.out"})
        deps = ["setup_%s" % sid]
        n_setup += 1
      for c in missing:
        local_jobs.append({"name": "cycle_%s_c%d" % (sid, c), "dir": sid,
                           "argv": ["./job.run", "--cycle", str(c)], "deps": deps,
                           "env": {"GROSCORE_NUMCYCLES": str(args.numruns)},
                           "archive": "%s.tar.gz" % sid,
                           "log": "job_local_c%d.out" % c})
      n_array += len(missing)
      print("  %s: queueing %d of %d cycles (%s)"
            % (sid, len(missing), args.numruns, compact_ranges(missing)))
      continue

    setup_path = os.path.join("slurm_fe", "setup_%s.run" % sid)
    cycles_path = os.path.join("slurm_fe", "cycles_%s.run" % sid)
    with open(setup_path, "w") as f:
      f.write(template + "\n")
      f.write("#SBATCH --job-name=fesetup_%s\n\n" % sid)
      f.write(extract_snippet(sid))
      f.write("cd %s || exit 1\n./job.run --setup\n" % sid)
    with open(cycles_path, "w") as f:
      f.write(template + "\n")
      f.write("#SBATCH --job-name=fecyc_%s\n" % sid)
      f.write("#SBATCH --array=%s%s\n\n" % (compact_ranges(missing), throttle))
      f.write("export GROSCORE_NUMCYCLES=%d\n" % args.numruns)
      f.write("cd %s || exit 1\n./job.run --cycle $SLURM_ARRAY_TASK_ID\n" % sid)

    # A structure whose setup already completed needs no new setup job and no
    # dependency. An archived structure has no directory, so its setup job runs
    # (it unpacks the tarball, then exits early since setup.done is inside).
    dep = ""
    if not setup_is_complete(sid):
      jid = sbatch_parsable(setup_path)
      if jid:
        dep = "--dependency=afterany:%s " % jid
        n_setup += 1
    sys.stdout.flush()            # sbatch's own line after this file's, as above
    if os.system("sbatch %s%s" % (dep, cycles_path)) == 0:
      n_array += 1
      print("  %s: submitting %d of %d cycles (%s)"
            % (sid, len(missing), args.numruns, compact_ranges(missing)))

  if args.run_local:
    # array-throttle is a SLURM array feature; locally the pool size is the cap.
    if args.array_throttle > 0:
      print("  (--array-throttle is ignored locally; --jobs-per-gpu sets the limit)")
    launch_local(local_jobs, args.ngpus, args.jobs_per_gpu, args.threads_per_job)
    print("")
    report_held(held)
    return

  print("Submitted %d setup job(s) + %d cycle array(s)%s, target %d cycles/structure%s.\n"
        % (n_setup, n_array,
           " + %d archive job(s)" % n_archive if n_archive else "", args.numruns,
           " (max %d concurrent)" % args.array_throttle if args.array_throttle > 0 else ""))
  report_held(held)

#------------------------------------------------------

def read_status(filepath, structids):
  """Read results_0.gs -> {struct_id: status} for non-OK stage-0 outcomes.

  The file is APPEND-ONLY across setup attempts, so a structure that failed and was
  then fixed carries both lines. Only the last one for a structure describes the
  current state; keeping the last FAILING one instead makes an old failure permanent
  and reports a working directory as broken. SETUP_RUNNING and OK are both
  "not finished, not failed" and are not statuses to report. Everything else
  (FAILED, PROBE_FAIL, LIGAND_FAIL from v1.5 on, ...) keeps the structure out of
  scoring under its own name, with no list here to keep in step.
  """
  last = {}
  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        if line.strip().startswith("#"):
          continue
        tmp = line.split()
        if len(tmp) >= 2 and tmp[0] in structids:
          last[tmp[0]] = tmp[1]
  return {sid: st for sid, st in last.items()
          if st not in ("OK", "SETUP_RUNNING")}

def read_analytical(filepath, andir="results_analytical.d"):
  """-> {struct_id: dG_release_kJ_mol}, merging the per-structure files written by
  the setup job with any legacy single results_analytical.gs."""
  vals = {}

  def take(line):
    if line.strip().startswith("#"):
      return
    tmp = line.split()
    if len(tmp) >= 2:
      try:
        vals[tmp[0]] = float(tmp[1])
      except ValueError:
        pass

  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        take(line)
  for path in sorted(glob.glob(os.path.join(andir, "*.gs"))):
    try:
      with open(path) as f:
        for line in f:
          take(line)
    except OSError:
      pass

  # Last resort: read the value straight out of each unarchived structure's
  # boresch_analytical.gs. The top-level record is only a cache -- if it is
  # cleared (or was written by an older job.run that gated it behind a marker),
  # every structure would otherwise score PENDING even though the value is right
  # there in the structure directory.
  for path in sorted(glob.glob(os.path.join("*", "boresch_analytical.gs"))):
    sid = os.path.basename(os.path.dirname(path))
    if sid in vals:
      continue
    try:
      with open(path) as f:
        for line in f:
          tmp = line.split()
          if len(tmp) >= 2 and tmp[0] == "dG_release_kJ_mol":
            try:
              vals[sid] = float(tmp[1])
            except ValueError:
              pass
            break
    except OSError:
      pass

  # Archived structures: the directory is gone, so recover the value from the
  # tarball. Without this a structure that finished all its cycles and was then
  # archived scores PENDING forever. Scanning a gzip stream is slow, so the
  # result is cached into andir and the scan happens at most once.
  archived = [os.path.basename(p)[:-7] for p in sorted(glob.glob("*.tar.gz"))]
  archived = [s for s in archived if s not in vals]
  if archived:
    import tarfile
    for sid in archived:
      found = None
      try:
        with tarfile.open("%s.tar.gz" % sid, "r:gz") as tar:
          for m in tar:
            if not m.name.endswith("boresch_analytical.gs"):
              continue
            fh = tar.extractfile(m)
            if fh is not None:
              for raw in fh:
                tmp = raw.decode("utf-8", "replace").split()
                if len(tmp) >= 2 and tmp[0] == "dG_release_kJ_mol":
                  try:
                    found = float(tmp[1])
                  except ValueError:
                    pass
                  break
            break
      except (OSError, tarfile.TarError):
        continue
      if found is not None:
        vals[sid] = found
        try:
          os.makedirs(andir, exist_ok=True)
          with open(os.path.join(andir, "%s.gs" % sid), "w") as f:
            f.write("%s %.6f\n" % (sid, found))
        except OSError:
          pass
  return vals

def read_works(filepath, workdir="results_fe.d"):
  """-> {struct_id: [ (cycle,
                       [(Wf, Wr), ...one per BOUND sub-leg],
                       rmsd, nstages,
                       [(Wf_pull, Wf_dhdl, Wr_pull, Wr_dhdl), ...one per stage]),
                      ... ]}, keeping only rows whose WORK values are all numeric.

  ANY NUMBER OF BOUND SUB-LEGS AND RAMP STAGES is accepted, because both have been
  reshaped repeatedly and will be again. A row is

      STRUCT_ID cycle <1 work per bound sub-leg> <4 per stage> <1 per sub-leg> RMSD

  with both groups laid out mirror-symmetrically about the centre: forward parts in
  run order, then the reverse parts in the order they RUN, i.e. reversed. So a
  sub-leg pairs with the field counted from its group's far end, and the same
  arithmetic reads the bound leg and the ramp.

      1 bound, 1 stage   ->  9 fields   the unstaged ramp, before 2026-08-14
      1 bound, 2 stages  -> 13 fields   split at u = 0.3
      1 bound, 3 stages  -> 17 fields   split at u = 0.3 and u = 0.5
      2 bound, 6 stages  -> 31 fields   2026-08-17 to 08-18 (_PAST)
      2 bound, 5 stages  -> 27 fields   until 2026-08-25 (_PAST)
      2 bound, 9 stages  -> 43 fields   current, fe_protocol.result_nf()

  The width of the CURRENT protocol is taken from fe_protocol and the shipped
  widths of earlier two-sub-leg protocols from _PAST; anything else falls back to
  the old nstages = (NF - 5) / 4 rule, which assumed exactly one bound sub-leg per
  direction and cannot express a split bound leg. Rows of any of those
  widths are read, so a directory from an older protocol still scores, and the
  layout travels with each row so a structure whose cycles straddle a reshaping is
  scored on what its rows contain rather than on what the current protocol expects.

  The RMSD is the rebinding sanity check (last column, Angstrom); it is diagnostic
  only, so a row whose RMSD is missing or NaN is still a valid result and is kept
  with nan in that slot.

  Reads the per-cycle files written by the cycle array (results_fe.d/<sid>_c<n>.gs,
  one line each -- parallel tasks must not append to a shared file) as well as any
  legacy single results_fe.gs."""
  works = {}

  # (nbound, nstages) for a row of a given width. The CURRENT protocol is asked
  # first, by width, because the old rule below cannot express a split bound leg:
  # it hardcodes exactly two non-stage work fields, so a 31-field row gives
  # (31-5)/4 = 6.5, no match, and the row is dropped WITHOUT A WORD. That is the
  # same silent-drop shape as the completeness gate and the ["nan"] * 19 literal,
  # and it is why the layout is asked of fe_protocol rather than inferred.
  #
  # _PAST holds widths that SHIPPED under an earlier protocol and that the old rule
  # cannot express either, because reshaping the ramp changes the width and the
  # protocol only ever knows its current one. The old rule then survives as the last
  # fallback, so every directory written before the bound leg was staged still parses
  # and still scores. Each width listed is unique to one (nbound, nstages) pair, so
  # the order matters only for widths more than one rule could claim.
  _CUR = {P.result_nf(): (P.n_bound(), P.n_stages())}
  # ONE LINE. Both _layout rules are lifted out by regex so the two copies of this
  # arithmetic can be checked against each other, and that regex reads a line.
  _PAST = {31: (2, 6), 27: (2, 5)}   # six-stage ramp 08-17..08-18, five-stage ..08-25

  def _layout(nf):
    if nf in _CUR:
      return _CUR[nf]
    if nf in _PAST:
      return _PAST[nf]
    if nf >= 9 and (nf - 5) % 4 == 0:
      return 1, (nf - 5) // 4
    return None

  def take(line):
    if line.strip().startswith("#"):
      return
    tmp = line.split()
    # With RMSD the row is nbound*2 + nstages*4 + 3 wide; without it, one less.
    # Try the RMSD-bearing width first, since every writer since 2026-07 emits one.
    for nf, has_rmsd in ((len(tmp), True), (len(tmp) + 1, False)):
      lay = _layout(nf)
      if lay:
        nbound, nstages = lay
        break
    else:
      return
    nw = 2 * nbound + 4 * nstages        # work fields, excluding the trailing RMSD
    if len(tmp) < 2 + nw:
      return
    try:
      cyc = int(tmp[1])
      vals = [float(x) for x in tmp[2:2 + nw]]
    except ValueError:
      return
    if any(math.isnan(v) for v in vals):
      return
    # Both groups are laid out mirror-symmetrically about the centre of the row:
    # forward sub-legs in run order, then the ramp, then the reverse sub-legs in
    # the order they RUN, which is the reverse of the forward order. So sub-leg i
    # pairs with the i-th field counted from its group's far end, and the same
    # arithmetic serves the bound leg and the ramp.
    bound = [(vals[i], vals[nw - 1 - i]) for i in range(nbound)]
    body = vals[nbound:nw - nbound]
    stages = []
    for i in range(nstages):
      fp, fd = body[2 * i], body[2 * i + 1]
      j = 2 * nstages + 2 * (nstages - 1 - i)
      rp, rd = body[j], body[j + 1]
      stages.append((fp, fd, rp, rd))
    rmsd = float('nan')
    if has_rmsd and len(tmp) >= 3 + nw:
      try:
        rmsd = float(tmp[2 + nw])
      except ValueError:
        pass
    works.setdefault(tmp[0], []).append((cyc, bound, rmsd, nstages, stages))

  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        take(line)
  for path in sorted(glob.glob(os.path.join(workdir, "*.gs"))):
    try:
      with open(path) as f:
        for line in f:
          take(line)
    except OSError:
      pass
  return works

def read_interface_qc(workdir="results_qc.d"):
  """-> {struct_id: [(cycle, {metric: value}), ...]} from utils/interface_qc.py.

  A SIDECAR, deliberately not part of the result row. read_works decodes the bound
  sub-leg and ramp stage counts from the row width (_layout), so appended QC fields
  would match no layout and read_works would silently reject every row. It is
  also diagnostic: a cycle with a wrecked interface is still a valid work
  measurement and must keep contributing to the free energy.

  Lines are `STRUCT cycle key value key value ...`. Absent directory, absent
  file, or a run predating the check all give {} rather than an error."""
  out = {}
  if not os.path.isdir(workdir):
    return out
  for path in sorted(glob.glob(os.path.join(workdir, "*.gs"))):
    try:
      for line in open(path):
        f = line.split()
        if len(f) < 4 or line.startswith("#"):
          continue
        try:
          cyc = int(f[1])
        except ValueError:
          continue
        rec = {}
        for k, v in zip(f[2::2], f[3::2]):
          try:
            rec[k] = float(v)
          except ValueError:
            pass
        if rec:
          out.setdefault(f[0], []).append((cyc, rec))
    except OSError:
      pass
  return out


#------------------------------------------------------
#
# Work-distribution diagnostic.
#
# Everything below turns the per-cycle works into the Crooks picture: a figure
# of the forward and sign-aligned reverse histogram of every leg, and a table
# saying whether the CGI crossing drawn on it is a measured quantity or an
# extrapolation. It runs as part of every scoring pass, because the free
# energies in scores_fe.gs cannot be read without it.

# Palette: categorical slots 1-2 (validated, CVD dE 24.7), plus chrome/ink tokens.
FWD, REV = "#2a78d6", "#eb6834"
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

WORKS_PLOT = "fe_works.png"
CONV_PLOT = "fe_convergence.png"

# Cycle counts at which the convergence figure re-scores. Every count from
# CONV_MIN is drawn while there are few of them and the estimate is still moving;
# past CONV_DENSE the grid thins to CONV_POINTS in total, because a full re-score
# is not free and the curve is flat by then. The LAST point is always the full set,
# so the right-hand end of the figure is the number in scores_fe.gs and not a
# nearby approximation to it.
CONV_MIN = 5            # below this a bootstrap CI is not evidence either way
CONV_DENSE = 25
CONV_POINTS = 32

# Intermediate points use a cheaper bootstrap than the table does. The relative
# error on a bootstrap standard deviation is 1/sqrt(2B), so 2.2% at B = 1000,
# which is a band width no eye resolves.
#
# IT IS THE BAR BOOTSTRAP THAT COSTS, not the avg/CGI one, and by an order of
# magnitude: measured per score_structure call on a 49-cycle three-stage run,
#
#   n_boot 50000, n_boot_bar 5000   2023 ms
#   n_boot  5000, n_boot_bar 5000   1884 ms      thinning n_boot buys ~7%
#   n_boot  5000, n_boot_bar 1000    427 ms      thinning n_boot_bar buys 4.4x
#
# because avg and CGI are closed-form over a resampled array while BAR solves its
# implicit equation by bisection on every row. So the knob that matters here is
# the second one. The LAST point is always re-run at the table's own settings, so
# the published number and the right-hand end of the curve are the same
# computation rather than merely close.
CONV_NBOOT = 5000
CONV_NBOOT_BAR = 1000

# The figure is a development diagnostic: it answers "has this protocol settled",
# which is a question about a handful of structures, not about a whole benchmark.
# One page of them is drawn and the rest are counted out loud rather than silently
# skipped, because a bounded cost that lies about its coverage is worse than a
# slow one.
CONV_MAX_STRUCTS = 16          # one page, matching ROWS_PER_PAGE below

# One figure row per structure, so a single image would grow without bound with
# the run. Agg itself does not stop this -- it will happily write 122400 px of
# height (200 rows) -- but everything downstream does, and the cost is linear in
# rows throughout: measured at 11 in x 180 dpi and 3.4 in per row, about 0.4 s
# and 9 MB of peak RSS per row, so 200 structures on one page is ~80 s and
# ~1.8 GB. What actually bounds a page is what can open it: GPU textures and
# most image viewers cap a dimension around 16384 px, and PIL refuses over
# 89 Mpx (~45000 px tall here) as a decompression bomb.
#
# 16 rows lands at 1980 x 9792 px and ~2 MB, comfortably inside all three, and
# renders in ~6 s from ~200 MB. Structures are paginated at that size, all of
# them, in sp.gs order; see _page_paths() for how the files are named.
ROWS_PER_PAGE = 16

# A metric computed from too few cycles is not evidence either way, and a flag
# raised on one is noise dressed up as a finding. Each check therefore reports
# one of three states -- ok, flagged, or n/a -- instead of collapsing "no data"
# into "failed": a leg with a single cycle would otherwise come back failed
# purely because 0/0 is not a number.
N_MIN_NORM = 3      # Shapiro-Wilk is defined from three samples up
N_MIN_OVL = 3       # below this a min..max range is not a range
N_MIN_SEP = 4       # _sep_limit's extreme-value argument needs a tail to reach
CHECKS = ("FD", "OVL", "SEP")

# Family-wise false-positive rate allowed per leg. FD runs one normality test on
# each of the leg's two work distributions, so the per-test level is Bonferroni-
# corrected to ALPHA/2 and the leg as a whole still misfires only ALPHA of the
# time when the works really are Gaussian.
ALPHA = 0.05

def _sep_limit(n):
  """Largest mean gap, in pooled sigma, that n cycles per direction can span.

  Two equally wide Gaussians cross halfway between their means, i.e. sep/2 sigma
  out in either tail, so "is the crossing sampled?" is really "does a run of n
  cycles reach sep/2 sigma?". The most extreme of n standard normals sits near
  z_n = Phi^-1(1 - 1/n), so the crossing leaves the sampled region once
  sep > 2 * z_n -- tightening when there are few cycles and relaxing when there
  are many, which no flat threshold can do.

  The limit follows n the whole way up, with no ceiling: a run long enough to
  put samples out at 5 sigma has genuinely measured a 5-sigma crossing, and
  capping it would report an estimate as extrapolated on the strength of a
  round number rather than of anything about the run."""
  if n < N_MIN_SEP:
    return float('nan')            # too few cycles to speak of a tail at all
  return 2.0 * NormalDist().inv_cdf(1.0 - 1.0 / n)

def _band(value, lo, hi):
  """ok / flag / n/a for a metric that must sit inside [lo, hi].

  A nan bound means the band itself could not be formed, which is as much a
  reason to abstain as a nan value."""
  if not (np.isfinite(value) and not (math.isnan(lo) or math.isnan(hi))):
    return "n/a"
  return "ok" if lo <= value <= hi else "flag"

def _normality(x):
  """Shapiro-Wilk p-value for one work distribution, or nan if it cannot be run.

  Shapiro-Wilk is the standard omnibus test of normality and the most powerful
  one at the sample sizes a cycle count gives; it is defined from n = 3 up,
  unlike the skew/kurtosis tests, which need n >= 20 before their asymptotics
  mean anything. scipy is imported here rather than at module scope so that
  submitting jobs never pays for it."""
  from scipy.stats import shapiro
  d = np.asarray(x, float)
  if len(d) < N_MIN_NORM or d.std() <= 0:
    return float('nan')
  return float(shapiro(d).pvalue)

def _fd_check(p_f, p_r):
  """FD state from the two per-distribution normality p-values.

  Each work distribution is tested on its own, which is the only way to ask
  whether IT is Gaussian: the previous ratio test compared the dissipation
  against (sf^2+sr^2)/4RT, i.e. it tested the Gaussian assumption, Crooks and
  linear response jointly and could not say which of the three had failed.

  Both directions must be testable, since a leg with one degenerate distribution
  says nothing about the pair."""
  ps = [p for p in (p_f, p_r) if np.isfinite(p)]
  if len(ps) < 2:
    return "n/a"
  return "flag" if min(ps) < ALPHA / 2.0 else "ok"

def _ovl_check(inside, n):
  """OVL state: do any sampled works lie where both directions have support?

  This is the blunt form of the question sep only proxies, and it survives things
  sep does not. sep = 2*diss/sigma_pooled puts a quadratic-in-deviations scale in
  the denominator, so ONE extreme work inflates sigma, drags sep down and quietly
  clears the check while the distributions are exactly as far apart as before. A
  count can only move by one per outlier, out of 2n.

  Flagged only at exactly zero, which is the assumption-free statement: no
  estimator has data. A handful is not much better -- the estimator's weight
  concentrates in the overlap region, so a few samples there carry the whole
  answer -- but that is a graded question, and sep is the graded measure. The
  count is always printed so a marginal 3 of 100 is visible rather than hidden
  behind a verdict."""
  if n < N_MIN_OVL or not np.isfinite(inside):
    return "n/a"
  return "flag" if inside == 0 else "ok"


def gaussian_check(st):
  """Near-equilibrium consistency of one leg.

  CGI models both work distributions as Gaussians and reads dG off where they
  cross, so three things have to hold:

    FD   each distribution is separately consistent with a Gaussian, by a
         Shapiro-Wilk test of that distribution alone. CGI's model is fitted to
         each one individually, so that is where the assumption has to be tested
    OVL  some sampled works actually lie where both directions have support. No
         estimator can do better than the data in the crossing region, and this
         asks whether that region contains any at all
    SEP  the histograms sit no further apart than the cycles can span -- at most
         2 * z_n pooled sigma, z_n = Phi^-1(1 - 1/n) -- the graded version of the
         same question, in units of the distributions' own spread

  OVL and SEP overlap in intent but not in failure mode: sep divides by a scale,
  so one extreme work can inflate sigma and clear it while nothing has improved,
  where a count moves by at most one. Read OVL first.

  Each is referred to the sampling noise of n cycles, so none fires on a
  departure that n cycles would produce by chance. A check whose input is
  undefined at that n reports n/a and takes part in no verdict."""
  sd_f, sd_r, diss, n = st['sd_f'], st['sd_r'], st['diss'], st['n']
  pred = (sd_f ** 2 + sd_r ** 2) / (4.0 * RT)
  pooled = math.sqrt((sd_f ** 2 + sd_r ** 2) / 2.0)
  # Descriptive only, no longer a flag: the linear-response prediction for the
  # dissipation. Far from 1 says the leg is out of the near-equilibrium regime,
  # which is worth seeing even though it does not by itself impugn either
  # distribution's shape.
  ratio = diss / pred if pred > 0 else float('nan')
  widths = sd_f / sd_r if sd_r > 0 else float('nan')
  sep = 2.0 * diss / pooled if pooled > 0 else float('nan')   # gap = 2*diss
  lim = _sep_limit(n)

  state = {'FD': _fd_check(st['p_fwd'], st['p_rev']),
           'OVL': _ovl_check(st.get('inside', float('nan')), n),
           # One-sided: a negative sep means the two means have crossed over,
           # which puts the crossing between them -- sampled, not extrapolated.
           'SEP': _band(sep, -math.inf, lim)}
  flags = [c for c in CHECKS if state[c] == "flag"]
  skipped = [c for c in CHECKS if state[c] == "n/a"]

  if len(skipped) == len(CHECKS):
    verdict = "n/a (n=%d)" % n
  else:
    verdict = ("+".join(flags) if flags else "OK")
    if skipped:
      verdict += " (%s n/a)" % ",".join(skipped)
  st.update(pred=pred, ratio=ratio, widths=widths, sep=sep, sep_lim=lim,
            state=state, flags=flags, skipped=skipped, verdict=verdict)
  return st

def _why_na(check, st):
  """Why a check abstained on this leg, in a few words."""
  if check == "SEP":
    return "n < %d" % N_MIN_SEP
  if check == "OVL":
    return "n < %d" % N_MIN_OVL
  if st['n'] < N_MIN_NORM:
    return "n < %d" % N_MIN_NORM
  return "a distribution has zero width"

def _cell(x, width=8, prec=2):
  """Format a metric right-aligned in `width`, or 'n/a' if it has none."""
  return ("%*.*f" % (width, prec, x)) if np.isfinite(x) else "n/a".rjust(width)

def _pcell(p, width=7):
  """A p-value, kept readable across the decades a normality test spans."""
  if not np.isfinite(p):
    return "n/a".rjust(width)
  return ("%.1e" % p if p < 1e-3 else "%.3f" % p).rjust(width)

def leg_stats(fwd, rev):
  """Estimates and widths of one leg, drawing nothing.

  `rev` is the RAW reverse work, matching _stream_avg/_stream_cgi; the
  sign-aligned reverse -rev is what the figure plots and what sd_r describes
  (negating does not change a width). Same estimators the scores are built from,
  so the rules on the figure are the numbers in scores_fe.gs rather than a
  second implementation of them."""
  f = np.asarray(fwd, float)
  v = np.asarray(rev, float)
  ral = -v                                   # sign-aligned reverse, as plotted
  # How many of the 2n sampled works land inside the OTHER direction's observed
  # range, and -- when none do -- how much empty work separates the two ranges.
  # Both are order statistics, so no distributional assumption enters.
  #
  # est.overlap_count is the SAME function that decides whether BAR is reported,
  # so the OVL verdict on the figure, the ovl column in the Gaussian report and
  # BAR_NO_OVERLAP in scores_fe.gs cannot disagree about what overlap means.
  inside = est.overlap_count(f, v)
  gap = max(0.0, max(f.min(), ral.min()) - min(f.max(), ral.max())) if len(f) else float('nan')
  bar_v, bar_note = est.bar(f, v, RT)
  return {'n': len(f),
          'avg': float(_stream_avg(f, v)),
          'cgi': float(_stream_cgi(f[None, :], v[None, :])[0]) if len(f) >= 3
                 else float('nan'),
          'bar': bar_v, 'bar_note': bar_note,
          'diss': float((f.mean() + v.mean()) / 2.0),   # per-direction dissipation
          'sd_f': float(f.std()), 'sd_r': float(v.std()),
          'inside': inside, 'gap': float(gap),
          # Normality is a property of each sample, and negating one of them
          # cannot change its shape, so the raw reverse works are tested as-is.
          'p_fwd': _normality(f), 'p_rev': _normality(v)}

# THE v1.5 ACCEPTANCE CHECK on the bound leg. A sub-leg whose forward and reverse
# runs both start from equilibrium has mean dissipation (<W_f> + <W_r>)/2 >= 0,
# because <W> >= dG holds for each direction on its own. Below zero by more than
# its noise, the two directions are not time-reverse pairs, and dG_intro is biased
# by whatever separates them however tight its interval is. Pass 2 had sub-leg 2
# negative in 39 of 43 structures (median -4.46 kJ/mol) and sub-leg 1 in 30
# (bm_fe_amber_2/_ana/digest/qc.txt, extra findings); the likely causes were the
# reverse bound legs starting from a strained rebound state and the bound holds
# running the unbinding Hamiltonian, both changed in v1.5. This is the check that
# the change worked, so it is reported for every structure rather than left to
# the diss column of the Gaussian table, where pass 2 printed the negative values
# without anyone flagging them.
#
# THE CUT IS A STUDENT t QUANTILE, NOT 2 SE (decision of 2026-10-04 on review QC-3).
# se comes from the same n cycles as diss, so at zero true dissipation diss/se
# follows t with n - 1 degrees of freedom, not a normal. The 2-se cut this check
# shipped with therefore fired far more often than the 2.3% it suggests: 9.5% per
# sub-leg at n = 3, 5.8% at 5, 3.9% at 10 and 2.9% at 20 (20,000 draws per n through
# this function, v15_work/RV/qc/negdiss_fpr.py, matching the t distribution's
# 9.2/5.8/3.8/3.0%), so a pilot read at three to five cycles would have flagged
# 11-18% of exactly reversible structures. Cut at the lower end of the two-sided 95%
# t interval instead, a sub-leg with exactly zero dissipation is flagged 2.5% of the
# time at every n, and the report states how many flags that predicts, so the count
# can be read against chance rather than against zero. Ten cycles at least: the sd
# behind se is then good to about a quarter (1/sqrt(2(n-1)) = 0.24), where at three
# it is good to a half, and an sd from three cycles is the fluke this check must
# not rest on.
NEG_DISS_CONF = 0.95   # two-sided t interval; NEG_DISS is its lower tail, 2.5%
N_MIN_DISS = 10        # below this many cycles there is no verdict

def neg_diss_crit(n):
  """The t below which a sub-leg of n cycles is NEG_DISS: the (1 + CONF)/2 quantile
  of Student's t with n - 1 degrees of freedom, 2.262 at n = 10 and 2.093 at n = 20.
  nan below N_MIN_DISS. scipy is imported here, as in _normality, so that
  submitting jobs never pays for it."""
  if n < N_MIN_DISS:
    return float('nan')
  from scipy.stats import t
  return float(t.ppf(0.5 + NEG_DISS_CONF / 2.0, n - 1))

def neg_diss_flag(diss, se, crit):
  """NEG_DISS: the mean dissipation lies more than crit standard errors below
  zero, strictly. None when there is no verdict (crit is nan)."""
  if not np.isfinite(crit):
    return None
  return bool(diss < -crit * se)

def bound_dissipation(bound_w):
  """[(name, n, diss, se, crit, flagged), ...] for bound_w = [(name, fwd, rev), ...].

  diss is the mean over cycles of d_c = (W_f + W_r)/2 with W_r the RAW reverse
  work, i.e. (mean f - mean b)/2 with b = -W_r the sign-aligned reverse. se is the
  standard error of that mean from the per-cycle d_c, so it is paired by cycle,
  the unit every bootstrap in this file resamples, and carries whatever
  correlation the two directions of one cycle have. crit is neg_diss_crit(n), and
  flagged is None below N_MIN_DISS cycles."""
  out = []
  for name, f, v in bound_w:
    d = (np.asarray(f, float) + np.asarray(v, float)) / 2.0
    n = len(d)
    diss = float(d.mean()) if n else float('nan')
    se = float(d.std(ddof=1) / math.sqrt(n)) if n >= 2 else float('nan')
    crit = neg_diss_crit(n)
    out.append((name, n, diss, se, crit, neg_diss_flag(diss, se, crit)))
  return out

def print_bound_dissipation(per_struct):
  """The acceptance-check block of the report. per_struct is [(sid, rows), ...]
  with rows from bound_dissipation()."""
  if not per_struct:
    return
  alpha = (1.0 - NEG_DISS_CONF) / 2.0         # one tail of the two-sided interval
  print("")
  print("Bound-leg dissipation per sub-leg, (<W_f> + <W_r>)/2 paired by cycle "
        "(v1.5 acceptance check):")
  print("  With both directions started from equilibrium its expectation cannot be")
  print("  negative. NEG_DISS marks a sub-leg whose mean lies more than t_crit standard")
  print("  errors below zero, t_crit the two-sided %g%% Student t quantile at n - 1"
        % (100.0 * NEG_DISS_CONF))
  print("  degrees of freedom, so a sub-leg with exactly zero dissipation is flagged")
  print("  %g%% of the time. Below %d cycles there is no verdict (n/a). A flagged"
        % (100.0 * alpha, N_MIN_DISS))
  print("  sub-leg's forward and reverse runs are not time-reverse pairs, and dG_intro")
  print("  is biased however tight its interval.")
  print("  %-10s %-9s %4s %9s %9s %8s %7s   %s"
        % ("structure", "sub-leg", "n", "diss", "se", "z", "t_crit", "flag"))
  flagged, tested_structs, tested, hits = [], 0, 0, 0
  for sid, rows in per_struct:
    hit = judged = False
    for name, n, diss, se, crit, flag in rows:
      z = diss / se if (np.isfinite(se) and se > 0) else float('nan')
      tok = "n/a" if flag is None else ("NEG_DISS" if flag else "-")
      if flag is not None:
        judged = True
        tested += 1
        hits += int(flag)
      hit = hit or bool(flag)
      print("  %-10s %-9s %4d %9s %9s %8s %7s   %s"
            % (sid, name, n, _cell(diss, 9), _cell(se, 9), _cell(z, 8),
               _cell(crit, 7, 3), tok))
    tested_structs += int(judged)
    if hit:
      flagged.append(sid)
  untested = len(per_struct) - tested_structs
  more = (" (%d more with fewer than %d cycles)" % (untested, N_MIN_DISS)
          if untested else "")
  if flagged:
    print("  NEG_DISS in %d of %d tested structures%s: %s."
          % (len(flagged), tested_structs, more, ", ".join(flagged)))
  else:
    print("  NEG_DISS in none of %d tested structures%s." % (tested_structs, more))
  # Expected flags at zero dissipation, by linearity: exact whatever the correlation
  # between the two sub-legs of one structure, which is why it is counted in
  # sub-legs. It also bounds the expected number of flagged structures.
  if tested:
    print("  Chance alone, with zero true dissipation in every one of the %d tested"
          % tested)
    print("  sub-legs, flags %.2f of them (%g%% each); observed %d."
          % (alpha * tested, 100.0 * alpha, hits))

def check_legs(legs):
  """[(sid, leg_name, stats), ...] for every leg of every structure, in order.

  `legs` is [(sid, [(name, fwd, rev, plot), ...]), ...]. EVERY channel is checked,
  including the ones the figure does not draw: on a staged run the whole undivided
  ramp is reported here even though only its stages get panels, because its
  overlap and dissipation are what the split is measured against."""
  out = []
  for sid, channels in legs:
    for name, fwd, rev, _plot in channels:
      out.append((sid, name, gaussian_check(leg_stats(fwd, rev))))
  return out

def _panel(ax, fwd, rev, title, nbins, st):
  """Forward vs sign-aligned reverse histogram with the estimate rules drawn on.

  `st` is this leg's leg_stats(), so the figure and the table cannot disagree."""
  f = np.asarray(fwd, float)
  rev_al = -np.asarray(rev, float)

  lo = min(f.min(), rev_al.min())
  hi = max(f.max(), rev_al.max())
  pad = 0.05 * (hi - lo) if hi > lo else 1.0
  bins = np.linspace(lo - pad, hi + pad, nbins + 1)
  binw = bins[1] - bins[0]

  series = []
  for d, color, label in ((f, FWD, "forward"), (rev_al, REV, "reverse (−W)")):
    ax.hist(d, bins=bins, color=color, alpha=0.45, edgecolor=color, linewidth=1.5,
            label="%s  n=%d, μ=%.1f, σ=%.1f" % (label, len(d), d.mean(), d.std()))
    series.append((d, color))

  avg, cgi, diss = st['avg'], st['cgi'], st['diss']
  ax.axvline(avg, color=INK, lw=2.0, label="avg  %.1f" % avg)
  if np.isfinite(cgi):
    ax.axvline(cgi, color=INK, lw=2.0, ls="--", label="CGI  %.1f" % cgi)
  # BAR is the reported estimate, so it is drawn even though it usually sits close
  # to the other two: where it does NOT, the leg has not converged and the figure
  # should say so. A suppressed BAR is labelled with its reason rather than
  # silently omitted, so an absent rule cannot be mistaken for an absent leg.
  if np.isfinite(st.get('bar', float('nan'))):
    ax.axvline(st['bar'], color=INK, lw=2.0, ls=":", label="BAR  %.1f" % st['bar'])
  elif st.get('bar_note'):
    ax.plot([], [], ' ', label="BAR  %s" % st['bar_note'].replace("BAR_", "").lower())

  # Gaussian fits -- these are exactly what CGI intersects, so drawing them shows
  # whether the reported crossing sits inside the sampled region or is
  # extrapolated into a gap where neither distribution has data.
  x0, x1 = ax.get_xlim()
  grid = np.linspace(x0, x1, 400)
  for d, color in series:
    sd = d.std()
    if sd <= 0 or len(d) < 2:
      continue
    pdf = np.exp(-0.5 * ((grid - d.mean()) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    ax.plot(grid, pdf * len(d) * binw, color=color, lw=2.0)
  ax.set_xlim(x0, x1)

  ax.set_title(title, fontsize=10, color=INK, fontweight="bold", loc="left")
  ax.set_xlabel("work [kJ/mol]", fontsize=9, color=SECONDARY)
  ax.set_ylabel("cycles", fontsize=9, color=SECONDARY)
  ax.tick_params(labelsize=8, colors=MUTED, length=3)
  ax.grid(axis="y", color=GRID, lw=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
  ax.text(0.98, 0.97, "dissipation %.0f kJ/mol" % diss, transform=ax.transAxes,
          ha="right", va="top", fontsize=8, color=SECONDARY)
  # Opaque-ish legend surface: the avg/CGI rules would otherwise strike through
  # the label text wherever they cross the legend box.
  leg = ax.legend(fontsize=7.5, frameon=True, labelcolor=SECONDARY, loc="upper left",
                  facecolor=SURFACE, edgecolor="none", framealpha=0.92)
  leg.set_zorder(5)

def _page_paths(npages):
  """Output paths for `npages` pages of the work-distribution figure.

  A single page keeps the plain WORKS_PLOT name, so the common case of a handful
  of structures is unchanged; more than one numbers them fe_works_01.png etc."""
  if npages <= 1:
    return [WORKS_PLOT]
  stem, ext = os.path.splitext(WORKS_PLOT)
  return ["%s_%02d%s" % (stem, i + 1, ext) for i in range(npages)]

def _is_page(name):
  """Is `name` a file some run of _page_paths() could have written?"""
  stem, ext = os.path.splitext(WORKS_PLOT)
  if not name.endswith(ext):
    return False
  body = name[:-len(ext)]
  return body == stem or (body.startswith(stem + "_") and body[len(stem) + 1:].isdigit())

def _clear_stale_pages(keep):
  """Delete pages left behind by a previous, longer run.

  Dropping from three pages to two -- or from several back to a single
  fe_works.png -- would otherwise leave files that still look like current
  output. Only names this module itself could have written are removed."""
  keep = set(keep)
  stem, ext = os.path.splitext(WORKS_PLOT)
  for path in glob.glob("%s*%s" % (stem, ext)):
    if path not in keep and _is_page(os.path.basename(path)):
      try:
        os.remove(path)
      except OSError:
        pass

def _fit_suptitle(fig, text, size=13, **kw):
  """Add a figure title, shrunk if it would not fit the canvas.

  Both figures size themselves from their contents -- columns per channel, columns
  per structure -- so the canvas can end up narrower than a fixed-size title needs,
  and matplotlib does not wrap or scale it: it draws it clipped at both ends. A
  single-structure convergence page is 6.4 in wide and lost the G and the I off its
  title that way.

  Measured against the renderer rather than estimated from the character count,
  because the width depends on the face and the weight and this must hold for any
  title anyone later writes. Only ever shrinks."""
  t = fig.suptitle(text, fontsize=size, **kw)
  avail = fig.get_size_inches()[0] * fig.dpi * 0.96
  try:
    w = t.get_window_extent(fig.canvas.get_renderer()).width
  except (AttributeError, RuntimeError):
    return t                                   # no renderer yet: leave it alone
  if w > avail:
    t.set_fontsize(max(7.0, size * avail / w))
  return t


def _conv_grid(n):
  """Cycle counts to re-score at, ending exactly on n."""
  if n < CONV_MIN:
    return []
  grid = list(range(CONV_MIN, min(n, CONV_DENSE) + 1))
  if n > CONV_DENSE:
    # linspace rather than a computed stride: a stride has to be floored at 1, and
    # then a span shorter than the point budget silently produces MORE points than
    # the budget rather than fewer. Spacing by count instead bounds it by
    # construction, which is what CONV_POINTS is supposed to mean.
    rest = max(0, CONV_POINTS - len(grid))
    if rest:
      grid += [int(round(x)) for x in np.linspace(CONV_DENSE + 1, n, rest)]
    grid.append(n)
  return sorted(set(g for g in grid if CONV_MIN <= g <= n))


def plot_convergence(conv, n_boot_bar):
  """dG_bind against the number of cycles it was computed from, per estimator.

  The three estimators are drawn SEPARATELY rather than as one headline with a
  band, because their disagreement is the diagnostic. BAR makes no distributional
  assumption, CGI assumes both work histograms are Gaussian, avg assumes the
  dissipation is symmetric; where a leg has converged they land together, and
  where it has not they fan out. A curve that is still sloping at the right-hand
  edge has not converged whatever its interval says, and a run whose three curves
  are still separated there has a leg that is not sampling, not a small n.

  Every point is the SUM OVER LEGS -- dG_intro plus the staged ramp plus the
  analytical release -- and never the one-shot ramp, which is a different estimate
  of the same thing and belongs in the table rather than on top of these.

  Each point re-runs the full joint bootstrap on the first m cycles, so the band
  is the same quantity as the CI in scores_fe.gs and the last point is that number
  exactly. Structures are drawn one per panel in sp.gs order, paginated like the
  work figure."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  series = (("BAR", "bind_bar", "bind_bar_ci", FWD, "-",  2.0),
            ("avg", "bind_avg", "bind_avg_ci", REV, "--", 1.6),
            ("CGI", "bind_cgi", "bind_cgi_ci", SECONDARY, ":", 1.6))

  drawn = []
  for sid, Wi, Wr, Wtf, Wtr, stage_w, rel, staged, bound_w in conv[:CONV_MAX_STRUCTS]:
    grid = _conv_grid(len(Wi))
    if len(grid) < 2:
      continue
    curves = {k: ([], [], []) for _, k, _, _, _, _ in series}
    for m in grid:
      sw = [(L, f[:m], v[:m]) for L, f, v in stage_w] if staged else None
      bw = [(N, f[:m], v[:m]) for N, f, v in bound_w] if bound_w else None
      try:
        # the endpoint is the published number, so it gets the published bootstrap
        cheap = {} if m == grid[-1] else {"n_boot": CONV_NBOOT,
                                          "n_boot_bar": CONV_NBOOT_BAR}
        r = score_structure(Wi[:m], Wr[:m], Wtf[:m], Wtr[:m], rel, stages=sw,
                            bound=bw,
                            **({"n_boot_bar": n_boot_bar} if not cheap else cheap))
      except (ValueError, FloatingPointError):
        continue
      for _lab, key, cik, _c, _ls, _lw in series:
        curves[key][0].append(m)
        curves[key][1].append(r.get(key, float('nan')))
        curves[key][2].append(r.get(cik, float('nan')))
    if any(np.isfinite(curves[k][1]).any() for _, k, _, _, _, _ in series):
      drawn.append((sid, curves))
  if not drawn:
    return None

  pages = [drawn[i:i + ROWS_PER_PAGE] for i in range(0, len(drawn), ROWS_PER_PAGE)]
  paths = ([CONV_PLOT] if len(pages) == 1 else
           ["%s_p%d%s" % (os.path.splitext(CONV_PLOT)[0], i + 1,
                          os.path.splitext(CONV_PLOT)[1]) for i in range(len(pages))])
  for path, page in zip(paths, pages):
    ncol = min(2, len(page))
    nrow = (len(page) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 3.6 * nrow),
                             squeeze=False, facecolor=SURFACE)
    flat = [a for row in axes for a in row]
    for ax in flat:
      ax.set_visible(False)
    for ax, (sid, curves) in zip(flat, page):
      ax.set_visible(True)
      ax.set_facecolor(SURFACE)
      lo, hi, late = [], [], []
      for lab, key, cik, colour, ls, lw in series:
        m, v, c = (np.asarray(x, float) for x in curves[key])
        ok = np.isfinite(v)
        if not ok.any():
          continue
        # A marker as well as a line: BAR is often only defined for the last few
        # cycle counts, and a two-point line segment is easy to miss when what it
        # is telling you is that the leg had no overlap until then.
        ax.plot(m[ok], v[ok], ls, color=colour, lw=lw, label=lab, zorder=3,
                marker="o" if ok.sum() <= 6 else None, ms=3.2)
        band = np.isfinite(c) & ok
        if band.any():
          ax.fill_between(m[band], (v - c)[band], (v + c)[band],
                          color=colour, alpha=0.13, lw=0, zorder=1)
        # the settled value, so the eye has the endpoint to judge the slope against
        ax.axhline(v[ok][-1], color=colour, lw=0.6, alpha=0.35, zorder=0)
        # y-limits from the SECOND HALF only. The first few cycle counts carry
        # enormous intervals that are not information, and letting them set the
        # scale flattens the part of the curve anyone is reading.
        half = m[ok] >= (m[ok][0] + m[ok][-1]) / 2.0
        if half.any():
          vv, cc = v[ok][half], np.nan_to_num(c[ok][half], nan=0.0)
          late.append((np.nanmin(vv - cc), np.nanmax(vv + cc)))
        if key == "bind_bar" and not ok.all():
          first = int(m[ok][0])
          ax.annotate("BAR first resolves at %d cycles" % first,
                      xy=(0.02, 0.06), xycoords="axes fraction",
                      fontsize=7.5, color=colour, alpha=0.9)
      if late:
        lo = min(x for x, _ in late); hi = max(y for _, y in late)
        pad = 0.18 * max(hi - lo, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
      ax.set_title("%s — dG_bind against cycles used" % sid,
                   fontsize=10, color=INK, loc="left")
      ax.set_xlabel("cycles", fontsize=9, color=SECONDARY)
      ax.set_ylabel("dG_bind  (kJ/mol)", fontsize=9, color=SECONDARY)
      ax.grid(True, color=GRID, lw=0.6, zorder=0)
      for s in ax.spines.values():
        s.set_color(AXIS)
      ax.tick_params(colors=SECONDARY, labelsize=8)
      ax.legend(fontsize=8, frameon=False, labelcolor=SECONDARY, ncol=3,
                loc="upper right", borderaxespad=0.3)
    _fit_suptitle(fig, "GroScore-FE convergence — staged sum over legs, "
                       "shaded 95% CI",
                  fontweight="bold", color=INK, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
  return paths


def plot_works(legs, stats):
  """Plot every structure's work distributions; return the page paths written.

  `legs` is [(sid, [(name, fwd, rev, plot), ...]), ...], one entry per structure
  with finished cycles and already in sp.gs order, and `stats` the matching
  check_legs() output. Pages preserve that order and are filled ROWS_PER_PAGE at a
  time, so a structure keeps its place as long as sp.gs does. matplotlib is
  imported here rather than at module scope so that submitting jobs never pays for
  it.

  A row gets one panel per channel with plot=True, so a staged run draws its ramp
  stages side by side: whether the split bought the overlap it was meant to is a
  question about the stages, and the answer is the picture. The number of
  columns is the widest row on the PAGE, so a page mixing staged and unstaged
  structures leaves the spare cells empty rather than misaligning the grid."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  TITLES = {"restraints": "bound-state restraints (dhdl)",
            "bound 1":    "bound-state restraints, sub-leg 1 (dhdl)",
            "bound 2":    "bound-state restraints, sub-leg 2 (dhdl)",
            "bound 1+2":  "bound-state restraints, sub-legs summed (dhdl)",
            "unbind A":   "unbinding stage A (pull + dhdl)",
            "unbind B":   "unbinding stage B (pull + dhdl)",
            "unbind A+B": "whole ramp, stages summed (pull + dhdl)",
            "unbind/rebind": "unbinding / rebinding (pull + dhdl)"}

  by_leg = {(sid, name): st for sid, name, st in stats}
  pages = [legs[i:i + ROWS_PER_PAGE] for i in range(0, len(legs), ROWS_PER_PAGE)]
  paths = _page_paths(len(pages))
  _clear_stale_pages(paths)

  for pageno, (path, page) in enumerate(zip(paths, pages), start=1):
    drawn = [[ch for ch in channels if ch[3]] for _sid, channels in page]
    ncol = max(len(d) for d in drawn)
    fig, axes = plt.subplots(len(page), ncol,
                             figsize=(5.5 * ncol, 3.4 * len(page)),
                             squeeze=False, facecolor=SURFACE)
    for row, ((sid, _channels), shown) in enumerate(zip(page, drawn)):
      nb = max(6, min(25, len(shown[0][1]) // 3 + 4))
      for ax in axes[row]:
        ax.set_facecolor(SURFACE)
        ax.set_visible(False)
      for col, (name, fwd, rev, _p) in enumerate(shown):
        axes[row][col].set_visible(True)
        _panel(axes[row][col], fwd, rev,
               "%s — %s" % (sid, TITLES.get(name, name)), nb,
               by_leg[(sid, name)])

    title = "GroScore-FE leg work distributions — forward vs sign-aligned reverse"
    if len(pages) > 1:
      title += "   (page %d of %d)" % (pageno, len(pages))
    _fit_suptitle(fig, title, fontweight="bold", color=INK, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
  return paths

# Long-form explanation of each flag, printed only for the flags that actually
# fired. The maths is short enough to state in full, and the failure modes are
# distinct enough that "which one fired" changes what you should do next.
FLAG_HELP = {
"FD": [
 "FD: a work distribution is not Gaussian",
 "",
 "  CGI fits one Gaussian per distribution, so each is tested on its own with a",
 "  Shapiro-Wilk test. The leg is flagged if either p falls below ALPHA/2. The",
 "  factor of two is a Bonferroni correction for running two tests per leg.",
 "",
 "  Switching too fast skews the works from the tail inward. The rare low-work",
 "  cycles that carry dG are exponentially unlikely, n cycles never reach them,",
 "  and the fitted Gaussian comes out too narrow and too far out. Use dG_avg",
 "  rather than CGI on a flagged leg.",
 "",
 "  A pass is weak evidence. Shapiro-Wilk has little power at n = 16, so an",
 "  unflagged leg is only 'not detectably non-Gaussian'.",
],
"OVL": [
 "OVL: no sampled work lies where both directions have support",
 "",
 "  Every bidirectional estimator (avg, CGI, BAR, Crooks) reads dG off the region",
 "  where P_f(W) and P_r(-W) are both populated. This counts how many of the 2n",
 "  works fall inside the other direction's observed min..max range. Zero means",
 "  the answer comes from the fitted model's extrapolated tails, not from data.",
 "",
 "  The gap is reported in RT. It says how deep into an exponentially unlikely",
 "  tail a cycle would have to land, which is why more cycles rarely helps once",
 "  the gap is large.",
 "",
 "  Only exactly zero is flagged, because that statement needs no assumption",
 "  about shape. Read the count as well; a handful is little better than none.",
 "",
 "  A flagged leg has NO BAR value: scores_fe.gs carries nan and a Note of",
 "  BAR_NO_OVERLAP_INTRO or BAR_NO_OVERLAP_UNBIND. That is the same count as this",
 "  flag, not a second opinion. avg and CGI are still reported, because they are",
 "  model extrapolations by construction and were never claiming otherwise; BAR",
 "  is refused instead of reported because it looks like a measurement. Its",
 "  solver returns a confident finite number on separated data, with the wrong",
 "  sign at the dissipation the unbinding leg currently runs at.",
],
"SEP": [
 "SEP: the histograms sit further apart than the cycles can span",
 "",
 "  sep = 2*diss / pooled sigma, the mean gap in units of the distributions own",
 "  spread. Two equally wide Gaussians cross halfway between their means, so the",
 "  crossing lies sep/2 sigma into either tail, and it is measured only if the",
 "  cycles reach that far. The most extreme of n samples sits near",
 "  z_n = Phi^-1(1 - 1/n), which gives sep_max = 2*z_n: 2.3 at n=8, 3.1 at n=16,",
 "  4.7 at n=100. There is no ceiling, since a run that puts samples out at",
 "  5 sigma has genuinely measured a 5-sigma crossing.",
 "",
 "  sep divides by a scale, so one extreme work can inflate sigma and clear the",
 "  check while the distributions stay exactly as far apart. OVL asks the same",
 "  question without that weakness. Read OVL first and use sep as the graded",
 "  measure.",
 "",
 "  This is not specific to CGI. BAR and Crooks need the same overlap. The scale",
 "  is RT: a few RT of dissipation converges, tens of RT needs exponentially many",
 "  cycles.",
],
}

def print_gaussian_report(stats):
  """Console table and, for whatever fired, the long form of each flag.

  CGI fits a Gaussian to each work distribution and reports where the two cross.
  That crossing is dG only if the works really are Gaussian, comparably wide, and
  overlapping; this tests all three. It is diagnostic only and changes no result."""
  print("")
  print("Gaussian / near-equilibrium consistency  (RT = %.3f kJ/mol at %.1f K)"
        % (RT, args.temp))
  print("-" * 78)
  print("CGI reads dG off the crossing of the two fitted Gaussians. That crossing is dG")
  print("only if each distribution is Gaussian (FD) and the two overlap (OVL, with SEP as")
  print("the graded form). A check the cycle count cannot answer reports n/a. These are")
  print("diagnostic and change no result.")
  print("")
  print("  diss        ( <W_f> + <W_r> ) / 2        dissipated work; dG cancels out")
  print("  ratio       diss / ( (sf^2+sr^2)/4RT )  linear response predicts 1; descriptive")
  print("  sf/sr       widths of the forward / sign-aligned reverse works")
  print("  p_fwd,p_rev Shapiro-Wilk of each distribution ALONE; FD flags below %.3f"
        % (ALPHA / 2.0))
  print("  ovl         works inside the OTHER direction's range, of 2n; OVL flags at 0")
  print("  sep         2 * diss / pooled sigma, vs sep_max = 2 * Phi^-1( 1 - 1/n )")
  print("")
  print("  %-10s %-13s %4s %9s %7s %7s %7s %7s %8s %6s %8s %8s   %s"
        % ("structure", "leg", "n", "diss", "ratio", "sf/sr", "p_fwd", "p_rev",
           "ovl", "sep", "sep_max", "diss/RT", "verdict"))
  for sid, leg, st in stats:
    ovl = ("%d/%d" % (st['inside'], 2 * st['n'])) if 'inside' in st else "n/a"
    print("  %-10s %-13s %4d %9s %7s %7s %7s %7s %8s %6s %8s %8s   %s"
          % (sid, leg, st['n'], _cell(st['diss']), _cell(st['ratio'], 7),
             _cell(st['widths'], 7), _pcell(st['p_fwd']), _pcell(st['p_rev']),
             ovl.rjust(8), _cell(st['sep'], 6, 1), _cell(st['sep_lim'], 8, 1),
             _cell(st['diss'] / RT, 8, 1), st['verdict']))

  bad = [(sid, leg, st) for sid, leg, st in stats if st['flags']]
  thin = [(sid, leg, st) for sid, leg, st in stats if st['skipped']]
  tested = [(sid, leg, st) for sid, leg, st in stats if len(st['skipped']) < len(CHECKS)]
  print("")
  if not bad and not tested:
    # Not the pass below: with every check abstaining nothing was measured, and a
    # one-cycle run printed that its crossings were measured (review E1).
    print("No leg had the cycles for any check, so no CGI crossing is tested yet (see")
    print("n/a below).")
  elif not bad:
    print("No leg failed a check it had the cycles to answer. Where testable, both work")
    print("distributions are Gaussian and the two overlap, so the CGI crossings are")
    print("measured rather than extrapolated.")
  else:
    print("%d of %d legs failed at least one check." % (len(bad), len(stats)))

    # One block per flag that fired anywhere, each listing the legs it applies to
    # with the number that triggered it.
    for flag in CHECKS:
      hits = [(sid, leg, st) for sid, leg, st in bad if flag in st['flags']]
      if not hits:
        continue
      print("")
      print("-" * 78)
      for line in FLAG_HELP[flag]:
        print(line)
      print("")
      print("  affected legs:")
      for sid, leg, st in hits:
        if flag == "FD":
          which = "forward" if st['p_fwd'] <= st['p_rev'] else "reverse"
          detail = "Shapiro-Wilk p_fwd=%s p_rev=%s at n=%d (%s rejects below %.3f)" % (
              _pcell(st['p_fwd'], 0), _pcell(st['p_rev'], 0), st['n'], which,
              ALPHA / 2.0)
        elif flag == "OVL":
          detail = "0 of %d works in the other's range; %.1f kJ/mol (%.1f RT) of empty gap" % (
              2 * st['n'], st['gap'], st['gap'] / RT)
        else:
          detail = "sep %.1f sigma vs %.1f reachable at n=%d (diss %.1f = %.0f RT)" % (
              st['sep'], st['sep_lim'], st['n'], st['diss'], st['diss'] / RT)
        print("    %-10s %-13s  %s" % (sid, leg, detail))

    print("")
    print("-" * 78)
    print("What to do about it")
    print("")
    print("  Use dG_avg rather than CGI on a flagged leg. It assumes nothing about shape,")
    print("  while a crossing drawn from unsampled tails can land anywhere.")
    print("")
    print("  More cycles will not clear these flags. They shrink the CI around whatever")
    print("  the estimator converges to, but do not reduce the hysteresis. That needs a")
    print("  slower switch, i.e. a longer stage: each stage's rate is DERIVED from its")
    print("  own mdp as stage_span / stage_time, so lengthening the stage in the RAMP")
    print("  table of utils/fe_protocol.py and regenerating with utils/make_fe_mdps.py")
    print("  lowers the rate to match. There is no rate to set by hand; --pull-rate")
    print("  only survives as the legacy single-ramp fallback.")
    print("")
    print("  Run utils/fe_leg_efficiency.py before spending anything. It measures whether")
    print("  this leg's dissipation really falls as 1/t and prices both options.")

  # Skipped checks are reported separately from failed ones: a leg that has not
  # run enough cycles has not been judged, and saying so is the only honest
  # summary. This block also fires when nothing was flagged.
  if thin:
    print("")
    print("-" * 78)
    print("n/a: the check has no answer on this leg")
    print("")
    print("  Minimum cycles are %d for FD, which is Shapiro-Wilk's own minimum, %d for OVL"
          % (N_MIN_NORM, N_MIN_OVL))
    print("  to have a min..max range, and %d for SEP to have a tail. FD also abstains when"
          % N_MIN_SEP)
    print("  a distribution has collapsed to a single repeated value.")
    print("")
    print("  An abstention is neither a pass nor a failure. The leg is untested on that")
    print("  point, and only more cycles will change that.")
    print("")
    print("  untested legs:")
    for sid, leg, st in thin:
      grouped = {}
      for check in st['skipped']:
        grouped.setdefault(_why_na(check, st), []).append(check)
      for reason, checks in grouped.items():
        print("    %-10s %-13s  %-11s  %s" % (sid, leg, ",".join(checks), reason))

#------------------------------------------------------

def recovery_by_cycle(iqc_rows):
  """{cycle: recovered} from one structure's read_interface_qc() rows, the last line
  of a cycle winning as for the result rows, finite values only."""
  out = {}
  for cyc, rec in iqc_rows:
    v = rec.get("recovered", float('nan'))
    if np.isfinite(v):
      out[cyc] = v
    else:
      out.pop(cyc, None)
  return out

def score(structids):
  check_ref_t()          # RT feeds pKD and BAR; warn if it does not match the mdp
  # First, so the report opens with it: a directory whose rows cannot be tied to the
  # protocol this code runs is still scored, but says so.
  for line in protocol_warning():
    print(line)
  rec_protocol = read_run_config().get(PROTOCOL_KEY)
  status = read_status("results_0.gs", structids)
  analytical = read_analytical("results_analytical.gs")
  works = read_works("results_fe.gs")
  iqc = read_interface_qc()

  rows = []  # (sid, result_dict_or_None, dG_release_or_None, ncyc, note)
  why_not = {}      # sid -> why a structure with result rows got no score
  bad_cycles = {}   # sid -> [(cycle, rmsd), …] that failed the rebinding check
  all_rmsds = []    # every measured cycle, for the summary statistics
  low_rec = {}      # sid -> (median recovery, cycles) below RECOVERY_WARN
  n_rec = 0         # structures with a recovery median at all
  legs = []         # (sid, [(name, fwd, rev, plot), ...]) for the diagnostic
  conv = []         # (sid, Wi, Wr, Wtf, Wtr, stage_w, dG_release, staged, bound_w)
  bdiss = []        # (sid, bound_dissipation rows), the v1.5 acceptance check
  for sid in structids:
    if sid in status:
      rows.append((sid, None, None, 0, status[sid]))
      continue
    cycles = works.get(sid, [])
    # deduplicate by cycle index (restart safety), keep last occurrence
    by_cycle = {}
    for row in cycles:
      by_cycle[row[0]] = row
    cycles = [by_cycle[c] for c in sorted(by_cycle)]

    # PROTOCOL IDENTITY (see protocol_gate). Where the directory records its
    # protocol, a structure whose own directory says it was set up for another one,
    # or before v1.5, is never scored, whatever rows it has: they would be pooled
    # into one table, and maybe into one structure, with this protocol's. Only
    # unpacked directories are checked. An archived structure keeps protocol_id.gs
    # inside a tarball of up to 13 GB (test29's 2KTF.tar.gz is 12.7), minutes to
    # stream on every scoring pass; its rows are vouched for by what wrote them
    # instead, since every cycle refuses to run unless protocol_id.gs matches its
    # code, and nothing is submitted into this directory under any other id.
    if rec_protocol is not None and os.path.isdir(sid):
      why = foreign_setup(sid, rec_protocol)
      if why:
        rows.append((sid, None, None, len(cycles), "PROTOCOL_MISMATCH"))
        why_not[sid] = "%s; this run directory holds %s" % (why, rec_protocol)
        continue

    # ONE ROW LAYOUT PER STRUCTURE. Cycles whose rows carry different bound sub-leg
    # or ramp stage counts come from differently shaped protocols: their stages
    # cannot be paired, and neither can their summed works, since each shape
    # dissipates differently and a pool of two work distributions is not the work
    # distribution of either protocol. Such a structure used to be promised a
    # score from the summed works and instead stopped the whole scoring pass with
    # an IndexError in score_structure's bootstrap (review QC-6). It is now not
    # scored, and the report says which cycles carry which layout.
    layouts = {}
    for c in cycles:
      layouts.setdefault((len(c[1]), c[3]), []).append(c[0])
    if len(layouts) > 1:
      rows.append((sid, None, None, len(cycles), "MIXED_LAYOUT"))
      why_not[sid] = "; ".join("%d bound + %d stages in cycles %s"
                               % (nb, ns, compact_ranges(cs))
                               for (nb, ns), cs in sorted(layouts.items()))
      continue
    if cycles:
      nbound, nstages = next(iter(layouts))
      # Bound sub-leg works. One channel per sub-leg, exactly as for the ramp; a
      # run from before the bound leg was staged has one of them and behaves as it
      # always did.
      bound_w = []
      for i in range(nbound):
        f = [c[1][i][0] for c in cycles]
        v = [c[1][i][1] for c in cycles]
        bound_w.append((BOUND_NAMES[i] if i < len(BOUND_NAMES) else str(i + 1), f, v))
      # The sub-leg works sum to the work of the whole switch, since the holds
      # between them do none; that sum is what dG_intro is compared against.
      W_intro  = [sum(x) for x in zip(*[f for _, f, _ in bound_w])] if bound_w else []
      W_remove = [sum(x) for x in zip(*[v for _, _, v in bound_w])] if bound_w else []
      if bound_w:
        bdiss.append((sid, bound_dissipation(bound_w)))

      # How many ramp stages these cycles carry: one count, checked above.
      staged = nstages > 1

      # Physical stage works. Each stage's forward and reverse pair up directly:
      # read_works has already undone the protocol's storage order, in which the
      # reverse stages appear in the order they run rather than in stage order.
      # The letters come from the protocol, not from a string here. A literal
      # "ABCDEFGH" survived the ramp going from three stages to five twice over and
      # then silently ran out at nine, naming stage I as "9" in every column header
      # and every plot while the arithmetic stayed right -- the same shape as the
      # parallel hardcoded lists fe_protocol exists to remove. A row from a ramp
      # LONGER than the current one still has to be named, so the fallback stays.
      LETTERS = P.stage_letters()
      stage_w = []
      for i in range(nstages):
        f = [SIGN_PULL_FWD * c[4][i][0] + c[4][i][1] for c in cycles]
        v = [SIGN_PULL_REV * c[4][i][2] + c[4][i][3] for c in cycles]
        stage_w.append((LETTERS[i] if i < len(LETTERS) else str(i + 1), f, v))

      # The hold segments do zero work, so these sums are the works of the full
      # ramp exactly, and they are what the one-shot estimate and the work-overlap
      # diagnostic both run on.
      Wtot_f = [sum(x) for x in zip(*[f for _, f, _ in stage_w])] if stage_w else []
      Wtot_r = [sum(x) for x in zip(*[v for _, _, v in stage_w])] if stage_w else []

      # The diagnostic needs the works alone, so a structure whose analytical
      # dG_release has not landed yet is still plotted and checked -- an
      # unfinished run is exactly when the convergence picture is wanted.
      #
      # The whole ramp is CHECKED but not PLOTTED on a staged run: it is the same
      # free energy the stage panels already show, its histograms are expected to
      # be far apart, and its numbers still appear in the table below as the
      # baseline the split is measured against. It is named for the arithmetic
      # ("unbind A+B") rather than for a leg, because no simulation of that name
      # runs and the table must not read as reporting one. On an unstaged run the
      # ramp really is one leg, and there it keeps the leg's name.
      # The bound leg is drawn per sub-leg when it is split, for the same reason
      # the ramp is: whether the split bought the overlap it was meant to is a
      # question about the sub-legs, and the answer is the picture.
      if len(bound_w) > 1:
        chan = [("bound %s" % n, f, v, True) for n, f, v in bound_w]
        chan.append(("bound %s" % "+".join(n for n, _, _ in bound_w),
                     W_intro, W_remove, False))
      else:
        chan = [("restraints", W_intro, W_remove, True)]
      if staged:
        chan += [("unbind %s" % L, f, v, True) for L, f, v in stage_w]
        chan.append(("unbind %s" % "+".join(L for L, _, _ in stage_w),
                     Wtot_f, Wtot_r, False))
      else:
        chan.append(("unbind/rebind", Wtot_f, Wtot_r, True))
      legs.append((sid, chan))

    if not cycles or sid not in analytical:
      rows.append((sid, None, None, len(cycles), "PENDING"))
      continue

    dG_release = analytical[sid]
    r = score_structure(W_intro, W_remove, Wtot_f, Wtot_r, dG_release,
                        stages=stage_w if staged else None,
                        bound=bound_w if len(bound_w) > 1 else None,
                        n_boot_bar=args.n_boot_bar)
    # Everything the convergence figure needs to re-score prefixes of this
    # structure. Collected here rather than recomputed there so the two can only
    # ever be looking at the same cycles in the same order.
    conv.append((sid, W_intro, W_remove, Wtot_f, Wtot_r, stage_w, dG_release,
                 staged, bound_w if len(bound_w) > 1 else None))

    # WORK OVERLAP, the variable that says when to distrust BAR on this row.
    #
    # BAR is the headline and it is the estimator with a measured overlap bias:
    # over three protocols on 2KTF the BAR-minus-avg gap ran +5.52 / +3.56 / +1.55
    # at mean overlaps of 35 / 44 / 43%, r = -0.82, always in the same direction
    # (BAR reads LESS negative when the histograms barely meet). Regressed per run
    # over sixteen runs, BAR carries a dissipation slope of +0.201 +- 0.122 where
    # avg and cgi are flat. None of that is visible in the free energies or in
    # their intervals, and at validation scale nobody is going to read setup logs
    # for hundreds of structures, so the two numbers that predict it go in the row.
    #
    # Both run over EVERY BAR channel, bound sub-legs included, because that is
    # what the dG_bind in the same row is a sum over and a channel left out of the
    # summary is a channel that can fail unwatched. The consequence to know when
    # reading them: the bound legs are near-reversible by construction and sit at
    # 90-98%, so they compress the MEAN towards the middle. **The MIN is the
    # discriminating one** -- it is what goes to zero when a structure loses BAR,
    # and on test27 it reads exactly 0.00 against a mean of 47%.
    #
    # Diagnostic only. A low-overlap structure still reports its free energies,
    # exactly as a high-RMSD one does; this says which rows to believe.
    #
    # Below N_MIN_OVL cycles a channel has no min..max range for a work to fall in,
    # so the count is 0 whatever the works do. That is nan here, as in the Gaussian
    # table, and not the 0.00 of a structure that really lost its overlap: a
    # one-cycle row read 0.00 in the column the pilot sorts on (review E1).
    ov = []
    for _n, _f, _v in (list(bound_w) + list(stage_w)):
      if len(_f) < N_MIN_OVL:
        ov.append(float('nan'))
        continue
      c = est.overlap_count(_f, _v)
      ov.append(100.0 * c / (2 * len(_f)))
    r['ovl_mean'] = float(np.mean(ov)) if ov else float('nan')
    r['ovl_min'] = float(np.min(ov)) if ov else float('nan')

    # Rebinding sanity check: did the reverse ramp put the partners back into the
    # bound pose? Since v1.5 no leg starts from that state (boundrev2 branches from
    # holdfwd0), so a high value marks a dissipative reverse ramp, not a cycle that
    # failed to close. Diagnostic only: the free energies are reported either way.
    rmsds = [(c[0], c[2]) for c in cycles if not math.isnan(c[2])]
    r['rmsd_mean'] = float(np.mean([v for _, v in rmsds])) if rmsds else float('nan')
    r['rmsd_max']  = float(np.max([v for _, v in rmsds])) if rmsds else float('nan')
    all_rmsds.extend(v for _, v in rmsds)
    bad = [(c, v) for c, v in rmsds if v > args.rmsd_warn]
    if bad:
      bad_cycles[sid] = bad
    # Contact recovery at the end of rebinding, per structure, beside HIGH_RMSD:
    # the median over the cycles scored here, so it describes the same cycles the
    # free energy and the RMSD come from. Diagnostic only, like HIGH_RMSD.
    rec = recovery_by_cycle(iqc.get(sid, []))
    rec = [rec[c[0]] for c in cycles if c[0] in rec]
    if rec:
      n_rec += 1
      med = float(np.median(rec))
      if med < RECOVERY_WARN:
        low_rec[sid] = (med, len(rec))
    flags = []
    if not np.isfinite(r['bind_avg']):
      # Rows, but no number. avg is the estimator defined on any data, so when it
      # is not finite nothing is, and the report must say why rather than leave
      # the structure out of both 'Scored' and 'Not scored' (review QC-5).
      flags.append("NO_SCORE")
      why_not[sid] = ("dG_release is %s" % dG_release if not np.isfinite(dG_release)
                      else "a work average is not finite")
    if bad:
      flags.append("HIGH_RMSD")
    if sid in low_rec:
      flags.append("LOW_RECOVERY")
    rows.append((sid, r, dG_release, len(cycles), ",".join(flags)))

  # Report binding free energies in kJ/mol and as pKD (never kcal/mol).
  # dG_bind = -RT ln(Ka) = RT ln(KD)  =>  pKD = -log10(KD) = -dG_bind / (RT ln 10).
  RTLN10 = RT * math.log(10.0)
  def pkd(x):
    return -x / RTLN10 if (x is not None and not (isinstance(x, float) and math.isnan(x))) else float('nan')
  def pkd_ci(ci):   # pKD is linear in dG_bind, so its CI just rescales by RT ln10
    return ci / RTLN10 if (ci is not None and np.isfinite(ci)) else float('nan')

  def cell(x):
    return ("%.2f" % x) if (x is not None and np.isfinite(x)) else "nan"

  rows_valid = [row for row in rows if row[1] is not None and np.isfinite(row[1]['bind_avg'])]
  rows_valid.sort(key=lambda row: row[1]['bind_avg'])

  # dG_bind, dG_intro and dG_unbind are each reported under all three estimators,
  # each with its own 95% CI. BAR leads because it is the headline as shipped: it
  # makes no distributional assumption, where CGI assumes both work histograms are
  # Gaussian and reads dG off where they cross. avg and CGI are retained so the
  # three can be compared per structure, which is the cheapest available check on
  # whether a leg has converged. A BAR column reading nan is not a failure to
  # compute; it means the leg had no forward/reverse overlap, and Note says which.
  # The dG_bind CIs come from the joint cycle bootstrap (they include the
  # dG_intro/dG_unbind covariance, so they are NOT the quadrature of the
  # component CIs).
  # dG_unbind_* is the STAGED estimate, the sum over the ramp stages. The columns
  # after it are the audit trail for that sum: each stage separately, and
  # dG_unbind_1s, the same free energy taken from the whole ramp in one shot. 1s
  # assumes nothing about the holds and is expected to read nan for want of
  # overlap, which is why the ramp is staged; where it does resolve, 1s minus staged
  # measures how far the holds are from equilibrium, and they are measured to be
  # far from it (score_structure). dG_unbind_avg is the same
  # number staged or not, since avg is linear in the works.
  # RMSD_mean/RMSD_max are the rebinding sanity check (Angstrom); a structure with
  # any cycle above --rmsd-warn additionally carries HIGH_RMSD in Note, and one
  # whose median contact recovery is below RECOVERY_WARN carries LOW_RECOVERY. A
  # structure with rows but no finite score carries NO_SCORE.
  #
  # N_NUMERIC is DERIVED, not written out. It used to be a literal 19 repeated in
  # the pending-row branch below, which silently had to be kept in step with this
  # string by hand.
  # The per-stage block is as wide as the ramp has stages, which comes from the
  # data rather than from this file. Taken from the widest structure scored, so a
  # directory holding runs of two different protocols still produces one
  # rectangular table; a structure with fewer stages leaves the extra pairs nan.
  STAGE_LETTERS = []
  for _sid, _r, _gr, _n, _note in rows:
    if _r and len(_r.get('stages', [])) > len(STAGE_LETTERS):
      STAGE_LETTERS = list(_r['stages'])
  stage_cols = "  ".join("dG_unb%s_bar  dG_unb%s_bar_CI" % (L, L) for L in STAGE_LETTERS)
  # Same treatment for the bound sub-legs: as wide as the widest structure scored,
  # taken from the data rather than from this file, so a directory holding runs of
  # two protocols still produces one rectangular table.
  BOUND_COLS = []
  for _sid, _r, _gr, _n, _note in rows:
    if _r and len(_r.get('bound', [])) > len(BOUND_COLS):
      BOUND_COLS = list(_r['bound'])
  bound_cols = "  ".join("dG_intro%s_bar  dG_intro%s_bar_CI" % (N, N)
                         for N in BOUND_COLS)
  cols = ("dGbind_bar  dGbind_bar_CI  pKD_bar  pKD_bar_CI  "
          "dGbind_avg  dGbind_avg_CI  pKD_avg  pKD_avg_CI  "
          "dGbind_cgi  dGbind_cgi_CI  pKD_cgi  pKD_cgi_CI  "
          "dG_intro_bar  dG_intro_bar_CI  dG_intro_avg  dG_intro_avg_CI  dG_intro_cgi  dG_intro_cgi_CI  "
          "dG_unbind_bar  dG_unbind_bar_CI  dG_unbind_avg  dG_unbind_avg_CI  dG_unbind_cgi  dG_unbind_cgi_CI  "
          + (bound_cols + "  " if bound_cols else "")
          + (stage_cols + "  " if stage_cols else "")
          + "dG_intro_1s_bar  dG_intro_1s_bar_CI  "
            "dG_unbind_1s_bar  dG_unbind_1s_bar_CI  "
            "dG_release  Overlap_mean_pct  Overlap_min_pct  RMSD_mean_A  RMSD_max_A  Ncycles  Note")
  N_NUMERIC = len(cols.split()) - 2          # every column but Ncycles and Note
  with open("scores_fe.gs", "w") as f:
    f.write("# GroScore-FE absolute binding free energies (kJ/mol; pKD dimensionless, T=%.1f K)\n" % args.temp)
    f.write("# Structure_ID  " + "  ".join(cols.split()) + "\n")
    # An empty Note is written as "-" rather than left blank so that every row has
    # the same field count under ANY splitting rule, whitespace included. A blank
    # last field survives a split on tabs and vanishes on a split on whitespace,
    # which is the sort of difference that is only noticed downstream.
    for sid, r, gr, n, note in rows:
      if r is None:
        f.write("\t".join([sid] + ["nan"] * N_NUMERIC + [str(n), note or "-"]) + "\n")
      else:
        # A BAR leg that was suppressed says so in Note, appended to whatever is
        # already there, comma-joined so a single grep finds either token. The
        # unbinding tokens keep _UNBIND in them and add the stage, so grepping
        # for the channel still catches both stages.
        # dG_unbind_1s is NOT flagged: it is the whole undivided ramp, it is
        # expected to have no overlap, and a token on every row would say nothing.
        bar_notes = []
        if r.get('intro_bar_note'):
          bar_notes.append(r['intro_bar_note'] + "_INTRO")
        if r.get('unb_bar_note'):
          bar_notes.append(r['unb_bar_note'] + "_UNBIND")
        for L in r.get('stages', []):
          if r.get("unb%s_bar_note" % L):
            bar_notes.append(r["unb%s_bar_note" % L] + "_UNBIND_" + L)
        for N in r.get('bound', []):
          if r.get("intro%s_bar_note" % N):
            bar_notes.append(r["intro%s_bar_note" % N] + "_INTRO_" + N)
        if bar_notes:
          note = ",".join(([note] if note else []) + bar_notes)
        note = note or "-"
        vals = [cell(r['bind_bar']), cell(r['bind_bar_ci']), cell(pkd(r['bind_bar'])), cell(pkd_ci(r['bind_bar_ci'])),
                cell(r['bind_avg']), cell(r['bind_avg_ci']), cell(pkd(r['bind_avg'])), cell(pkd_ci(r['bind_avg_ci'])),
                cell(r['bind_cgi']), cell(r['bind_cgi_ci']), cell(pkd(r['bind_cgi'])), cell(pkd_ci(r['bind_cgi_ci'])),
                cell(r['intro_bar']), cell(r['intro_bar_ci']),
                cell(r['intro_avg']), cell(r['intro_avg_ci']),
                cell(r['intro_cgi']), cell(r['intro_cgi_ci']),
                cell(r['unb_bar']), cell(r['unb_bar_ci']),
                cell(r['unb_avg']), cell(r['unb_avg_ci']),
                cell(r['unb_cgi']), cell(r['unb_cgi_ci'])]
        # One pair per COLUMN, not per sub-leg this structure has, so a run
        # mixing protocols still writes a rectangular table.
        for N in BOUND_COLS:
          vals += [cell(r.get("intro%s_bar" % N, float('nan'))),
                   cell(r.get("intro%s_bar_ci" % N, float('nan')))]
        for L in STAGE_LETTERS:
          vals += [cell(r.get("unb%s_bar" % L, float('nan'))),
                   cell(r.get("unb%s_bar_ci" % L, float('nan')))]
        vals += [cell(r['intro1s_bar']), cell(r['intro1s_bar_ci']),
                 cell(r['unb1s_bar']), cell(r['unb1s_bar_ci']),
                 cell(gr), cell(r['ovl_mean']), cell(r['ovl_min']),
                 cell(r['rmsd_mean']), cell(r['rmsd_max'])]
        f.write("\t".join([sid] + vals + [str(n), note]) + "\n")

  done = len(rows_valid)
  print("Scored %d/%d structures with complete cycles. Wrote scores_fe.gs." % (done, len(structids)))
  # Every structure without a score, by reason: a status the setup wrote (FAILED,
  # PROBE_FAIL, LIGAND_FAIL, ...), no rows yet (PENDING), rows that must not be
  # pooled (PROTOCOL_MISMATCH, MIXED_LAYOUT), and rows that gave no finite number
  # (NO_SCORE). A structure with a status is never scored, whatever result rows an
  # earlier setup may have left behind: read_status runs first. The NO_SCORE kind
  # used to be named neither under 'Scored' nor here, with '-' as its Note (review
  # QC-5). Where a reason has detail, it follows on a line per structure.
  valid = {row[0] for row in rows_valid}
  unscored = {}
  for sid, r, _gr, _n, note in rows:
    if sid in valid:
      continue
    token = (note or "PENDING") if r is None else note.split(",")[0]
    unscored.setdefault(token, []).append(sid)
  if unscored:
    print("Not scored: %s." % "; ".join("%s %s" % (k, ", ".join(v))
                                        for k, v in sorted(unscored.items())))
    for token, sids in sorted(unscored.items()):
      for sid in sids:
        if sid in why_not:
          print("  %s %s: %s" % (token, sid, why_not[sid]))

  # ── rebinding sanity check summary ──────────────────────────────────────────
  # Every cycle separates the partners and pushes them back. The backbone RMSD
  # between the bound equilibration and the end of the reverse ramp says whether
  # that ramp re-formed the bound pose. Since v1.5 nothing starts from that state
  # (boundrev2 branches from holdfwd0), so a high value marks a dissipative reverse
  # path rather than a cycle that failed to close. Contact recovery, the other
  # per-structure flag, follows it.
  print("")
  print("Rebinding sanity check (backbone RMSD of the re-bound structure, warn > %.1f A):"
        % args.rmsd_warn)
  if not all_rmsds:
    print("  No RMSD values available yet (no cycle has finished, or the runs predate this check).")
  else:
    vals = np.asarray(all_rmsds, float)
    print("  %d simulations analyzed: mean %.2f A, median %.2f A, max %.2f A"
          % (vals.size, float(vals.mean()), float(np.median(vals)), float(vals.max())))
    if not bad_cycles:
      print("  All measured cycles re-bound within the threshold.")
    else:
      print("")
      print("  WARNING: %d structure%s did not re-bind properly in at least one cycle."
            % (len(bad_cycles), "s" if len(bad_cycles) > 1 else ""))
      print("  Their free energies are reported but should be treated with caution")
      print("  (flagged HIGH_RMSD in scores_fe.gs):")
      # Worst first and truncated -- scores_fe.gs holds the full list.
      worst = sorted(bad_cycles.items(), key=lambda kv: max(v for _, v in kv[1]), reverse=True)
      for sid, bad in worst[:MAX_FLAGGED_SHOWN]:
        detail = ", ".join("c%d=%.1f" % (c, v) for c, v in bad[:MAX_CYCLES_SHOWN])
        if len(bad) > MAX_CYCLES_SHOWN:
          detail += ", +%d more" % (len(bad) - MAX_CYCLES_SHOWN)
        print("    %-20s %s" % (sid, detail))

      if len(worst) > MAX_FLAGGED_SHOWN:
        print("    ... and %d more (grep HIGH_RMSD scores_fe.gs for the full list)"
              % (len(worst) - MAX_FLAGGED_SHOWN))
  if n_rec:
    print("")
    print("  Contact recovery per structure, the median over its scored cycles of the")
    print("  restrained contacts back within tolerance (flag below %.2f):" % RECOVERY_WARN)
    if not low_rec:
      print("  LOW_RECOVERY in none of %d structures." % n_rec)
    else:
      print("  LOW_RECOVERY in %d of %d structures (flagged in scores_fe.gs; no score"
            % (len(low_rec), n_rec))
      print("  changes):")
      lowest = sorted(low_rec.items(), key=lambda kv: kv[1][0])
      for sid, (med, k) in lowest[:MAX_FLAGGED_SHOWN]:
        print("    %-20s median %.3f over %d cycles" % (sid, med, k))
      if len(lowest) > MAX_FLAGGED_SHOWN:
        print("    ... and %d more (grep LOW_RECOVERY scores_fe.gs for the full list)"
              % (len(lowest) - MAX_FLAGGED_SHOWN))
  print("")
  # ── interface recovery ──────────────────────────────────────────────────────
  # The RMSD above says the partners came back to roughly the right place. This
  # says whether the CONTACTS re-formed and the side chains re-packed, which is
  # what the interface restraints hold and what capping the restraint list per
  # residue-residue contact puts at risk. Pooled over every cycle and diagnostic
  # only; the per-structure flag on it is LOW_RECOVERY above, and no free energy
  # changes either way.
  if iqc:
    rows_q = [(sid, c, r) for sid, lst in iqc.items() for c, r in lst]
    print("")
    print("Interface recovery at the end of rebinding (%d cycles):" % len(rows_q))
    def _col(key):
      v = [r[key] for _s, _c, r in rows_q if key in r and np.isfinite(r[key])]
      return np.asarray(v, float) if v else None
    for key, label, unit in (
        ("recovered", "contacts back within tolerance", ""),
        ("formed",    "contacts still inside the cutoff", ""),
        ("rms_dev",   "rms distance deviation per pair", " nm"),
        ("sc_rmsd",   "interface side-chain RMSD", " A")):
      v = _col(key)
      if v is None:
        continue
      print("  %-34s mean %6.3f%s   median %6.3f   worst %6.3f"
            % (label, float(v.mean()), unit, float(np.median(v)),
               float(v.min() if key in ("recovered", "formed") else v.max())))
    n_p = _col("npairs")
    if n_p is not None:
      print("  restrained pairs checked           %d" % int(np.median(n_p)))
    print("  A reduction in the restraint count is safe only if these hold up;")
    print("  they exist so that can be measured rather than argued.")
  print_bound_dissipation(bdiss)
  print("")

  # ── work distributions and their Crooks/Gaussian consistency ────────────────
  # The rebinding checks say whether the reverse ramp re-formed the bound pose;
  # this says whether the numbers extracted from the works converged. Both are
  # diagnostic only.
  if not legs:
    print("No cycle has finished yet, so there are no work distributions to check.")
    print("")
    return
  stats = check_legs(legs)
  cpaths = plot_convergence(conv, args.n_boot_bar) if conv else None
  if cpaths:
    print("Wrote %s — dG_bind against cycles used, BAR / avg / CGI separately,"
          % (cpaths[0] if len(cpaths) == 1 else "%s .. %s" % (cpaths[0], cpaths[-1])))
    print("  each with its own 95%% CI. Three curves that have not come together")
    print("  by the right-hand edge mean a leg is not sampling, not that n is small.")
    if len(conv) > CONV_MAX_STRUCTS:
      print("  Drawn for the first %d of %d scored structures in sp.gs order: this"
            % (CONV_MAX_STRUCTS, len(conv)))
      print("  figure is a protocol diagnostic and re-scores every prefix, so it is")
      print("  bounded rather than run over a whole benchmark.")
  paths = plot_works(legs, stats)
  if len(paths) == 1:
    print("Wrote %s (%d structure%s: %s)."
          % (paths[0], len(legs), "" if len(legs) == 1 else "s",
             ", ".join(sid for sid, *_ in legs)))
  else:
    print("Wrote %s .. %s (%d structures in sp.gs order, %d per page)."
          % (paths[0], paths[-1], len(legs), ROWS_PER_PAGE))
  print_gaussian_report(stats)
  print("")

#------------------------------------------------------

def main():
  print("")
  print("#################################")
  print("#                               #")
  print("#          GroScore-FE          #")
  print("#                               #")
  print("#################################")
  print("")

  structids, structchains = readstructparams(args.structparams)
  if not structids:
    print("Error: No valid structures found in %s" % args.structparams); sys.exit(1)
  # Say where the target came from: silently inheriting or silently defaulting a
  # number that decides when a structure gets archived is worth one line.
  print("GroScore-FE: %d structures, %d bidirectional cycles each%s.\n"
        % (len(structids), args.numruns,
           " (remembered from %s; override with -n)" % RUN_CONFIG
           if NUMRUNS_REMEMBERED else ""))

  # Submit jobs on first invocation (or on --restart), into a directory whose
  # protocol is this code's. A refusal touches nothing and submits nothing; what is
  # here is still scored below, and the invocation then exits 1, because what it
  # was asked to do did not happen.
  refused = None
  if args.restart or not os.path.isfile("struct_map.gs"):
    refused = protocol_gate()
    if refused:
      print("\n".join(refused) + "\n")
    else:
      if not os.path.isfile("results_0.gs"):
        with open("results_0.gs", "w") as f:
          f.write("# Stage-0 setup status\n")
      setup_and_submit(structids, structchains)

  # Progress of a --run-local run. No-op when the jobs went to SLURM instead.
  print_local_status()

  # Score whatever results are currently available.
  score(structids)
  if refused:
    # Again at the end, where it is read: the report above can run to pages.
    print(refused[0])
    sys.exit(1)

if __name__ == "__main__":
  main()
