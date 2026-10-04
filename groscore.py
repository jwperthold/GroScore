#!/usr/bin/env python3
#

import string, math, array
import os, sys, glob, re, time, argparse, shutil
import numpy as np
from scipy import stats

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Input files for GroScore")
parser.add_argument('-n','--numruns', type=int, default=5, required=False, help="Number of pull/push cycles to perform (default: 5)")
parser.add_argument('-s','--structparams', type=str, default="sp.gs", required=False, help="GroSscore strucutre parameter file")
parser.add_argument('-ff','--forcefield', type=str, default="amber19sb_opc3", choices=["gromos54a8", "gromos54a8_rf", "charmm36", "amber19sb_opc", "amber19sb_opc3"], help="Force field to use (default: amber19sb_opc3)")
parser.add_argument('--no-cutout', dest='cutout', action='store_false', help="Disable interface cutout, use full protein structure")
parser.add_argument('--no-ligand-param', dest='ligand_param', action='store_false', help="Run without small-molecule ligands: none is parametrized or merged, and each is named in the job log. Without it, CHARMM36 and GROMOS 54A8 stop every structure with a ligand in the simulated region as LIGAND_FAIL.")
parser.add_argument('--slurm', type=str, default="workstation", help="SLURM template name from slurm/ directory (default: workstation)")
parser.add_argument('--run-local', dest='run_local', action='store_true',
                    help="Run on this machine instead of submitting to SLURM. Jobs are "
                         "started in the background and spread round-robin over the local "
                         "GPUs, for single multi-GPU workstations. Requires --ngpus.")
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
parser.add_argument('--restart', action='store_true', help="Restart: resubmit jobs even if run.gs exists. The submitted jobs run -n cycles in total, so a larger -n adds cycles.")
parser.add_argument('--inject-job-run', action='store_true', help="Inject fresh job.run into archived (.tar.gz) structures and submit them; without it an archive is not submitted")
parser.add_argument('--rmsd-warn', type=float, default=10.0, metavar='A',
                    help="Warn about cycles whose re-bound backbone RMSD exceeds this "
                         "value in Angstrom (default: 10.0). Scores are always reported; "
                         "the check only flags them.")
parser.set_defaults(cutout=True, ligand_param=True)

args=parser.parse_args()

if args.run_local and args.ngpus < 1:
  parser.error("--run-local needs --ngpus N: the number of GPUs the jobs are distributed over")
if args.ngpus and not args.run_local:
  parser.error("--ngpus only applies to --run-local; SLURM allocates GPUs itself")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from local_runner import launch_local, print_local_status
import estimators as est

#------------------------------------------------------
# Temperature, for BAR only.
#
# Deliberately NOT a --temp flag. groscore_fe.py has one and it never reaches
# grompp: ref_t is fixed in the mdp templates, so passing --temp 300 there would
# silently hand the estimator the wrong kT for a 310 K simulation. Nothing in the
# classic pipeline can change the simulation temperature either, so a flag here
# would only create a second copy of that trap. The value is checked against the
# mdp at startup instead, so the two cannot drift apart unnoticed.
T_REF = 310.0
RT = 0.00831446261815324 * T_REF          # kJ/mol


def check_ref_t(forcefield):
  """Warn, never abort, if the force field's mdp does not run at T_REF."""
  mdp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "settings", forcefield, "bind.mdp")
  try:
    with open(mdp) as f:
      for line in f:
        if line.strip().startswith("ref_t"):
          vals = [float(v) for v in line.split("=")[1].split()]
          if any(abs(v - T_REF) > 0.5 for v in vals):
            print("WARNING: %s has ref_t = %s but groscore.py assumes %.1f K for "
                  "the BAR column; that column will be wrong."
                  % (mdp, " ".join("%g" % v for v in vals), T_REF))
          return
  except (OSError, ValueError, IndexError):
    pass                                  # missing or unparseable: not fatal

#------------------------------------------------------

def readstructparams(filepath):
  ids = []
  chains = []
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          try:
            ids.append(tmp[0])
            chains.append(tmp[1])
          except (IndexError, AttributeError):
            pass
  if len(ids) == len(chains) and len(ids) > 0:
    return ids, chains
  else:
    return [], []

#------------------------------------------------------

def read_status(filepath):
  """({struct_id: status}, number of malformed lines) from results_0.gs.

  results_0.gs is append-only across setups and restarts, so a structure can have
  many lines, and only the LAST one describes the directory as it is now: an old
  OK must not hide a later failure, and an old failure must not outlive a setup
  that has since succeeded. Until v1.5 a single line with one token made the whole
  file unreadable (the reader returned False and the caller crashed unpacking it,
  so no score file was written for any structure), and a structure whose latest
  line was a failure was still scored from leftover cycle rows (v15_work
  digest_audit SCORER-6). Every line job.run and groscore.py write is 'ID STATUS'
  with an upper-case status, so anything else is a torn append and is skipped.
  """
  status, bad = {}, 0
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        tmp = line.split()
        if not tmp or tmp[0].startswith("#"):
          continue
        if len(tmp) != 2 or not re.fullmatch(r"[A-Z_]+", tmp[1]):
          bad += 1
          continue
        status[tmp[0]] = tmp[1]
  return status, bad


def scoreable(status):
  """A structure is scored only while its last stage-0 status is OK. None (no line
  at all, e.g. results_0.gs was cleared) keeps the pre-v1.5 behaviour of scoring
  whatever cycles exist."""
  return status is None or status == "OK"

#------------------------------------------------------

def to_float(token):
  try:
    v = float(token)
  except (ValueError, TypeError):
    return float('nan')
  return v if np.isfinite(v) else float('nan')


def read_legacy(filepath, push):
  """[(struct_id, work, rmsd)] from one append-only results_<n>.gs (job.run before
  v1.5). Malformed rows are counted, not used.

  A push row is 'ID work RMSD'. job.run before v1.5 wrote the work with an
  unguarded command substitution, so a failed integration left 'ID RMSD', and the
  rebinding RMSD in Angstrom was read as the push work (v15_work digest_audit
  SCORER-4: 'X1 -100.0 2.10' followed by 'X1 2.10' scored -148.9 instead of
  -200.0). Push files written before the RMSD column existed (82c2ffd, the
  published benchmark) hold only two-field rows. So the rule is per file: once
  any row of a push file has three fields, a two-field row in it is an empty work
  and is skipped.
  """
  rows, bad = [], 0
  if not os.path.isfile(filepath):
    return rows, bad
  with open(filepath, "r") as f:
    lines = [l.split() for l in f if l.strip() and not l.strip().startswith("#")]
  three = push and any(len(t) >= 3 for t in lines)
  for tmp in lines:
    if len(tmp) < 2 or (three and len(tmp) < 3):
      bad += 1
      continue
    rmsd = to_float(tmp[2]) if push and len(tmp) >= 3 else float('nan')
    rows.append((tmp[0], to_float(tmp[1]), rmsd))
  return rows, bad


def read_percycle(dirpath, known):
  """Per-cycle results written by job.run since v1.5 (one file per structure and
  cycle, written with '>', so a re-walk or a new setup replaces rather than appends).

  Returns ({struct_id: {cycle: (pull, push, rmsd)}}, set of struct_ids under this
  layout, [malformed file names]). <ID>_c<N>.gs holds the one line
  'ID N pull push RMSD'; <ID>_setup.gs is written by job.run once the restraint set
  of a setup exists, before any of its cycles has a row. Either puts the structure
  under the per-cycle layout, and from then on its legacy results_<n>.gs rows are
  never read: those belong to an earlier setup in the same run directory (v15_work
  digest_audit runner R3). The file name must match the row, so a structure whose
  ID is a prefix of another's cannot pick up the other's rows.
  """
  rows, marked, bad = {}, set(), []
  if not os.path.isdir(dirpath):
    return rows, marked, bad
  for name in sorted(os.listdir(dirpath)):
    # The cycle suffix is matched at the END, greedily, so an ID may itself
    # contain '_c<digits>'; anything else (job.run's .tmp files) is not a result.
    m = re.fullmatch(r"(.+)_c(\d+)\.gs", name) or re.fullmatch(r"(.+)_(setup)\.gs", name)
    if not m or m.group(1) not in known:
      continue                                      # not a structure of this sp.gs
    sid = m.group(1)
    try:
      with open(os.path.join(dirpath, name)) as f:
        tmp = f.read().split()
    except OSError:
      bad.append(name)
      continue
    if m.group(2) == "setup":
      if tmp[:1] == [sid]:
        marked.add(sid)
      else:
        bad.append(name)
      continue
    if len(tmp) != 5 or tmp[0] != sid or tmp[1] != m.group(2):
      bad.append(name)
      continue
    marked.add(sid)
    rows.setdefault(sid, {})[int(tmp[1])] = tuple(to_float(t) for t in tmp[2:5])
  return rows, marked, bad

#------------------------------------------------------

def t975(df):
  """Two-sided 95% quantile of Student's t with df degrees of freedom."""
  return float(stats.t.ppf(0.975, df))


def avg_ci95(pulls, pushes):
  """Half-width of the 95% interval of the Crooks average; nan below two cycles.

  The score is the mean over cycles of s_i = (pull_i + push_i)/2, and the cycles are
  independent by construction (each starts from emin_solv.gro with fresh
  velocities), so this is the paired Student t interval of that mean,
  t(0.975, n-1) * sd(s, ddof=1) / sqrt(n). Pairing keeps the covariance between a
  push and the pull of its own cycle, which resampling the two lists separately
  threw away. Until v1.5 the column was 1.96 times the SD of that independent
  plug-in bootstrap: exactly 0.0 for one cycle, and at the default five cycles an
  88% interval about 15% too narrow (v15_work digest_audit SCORER-3; coverage over
  1500 Gaussian replicates n=2 0.70, 3 0.82, 5 0.88, 10 0.91, 20 0.93).
  """
  s = (np.asarray(pulls, dtype=float) + np.asarray(pushes, dtype=float)) / 2.0
  if s.size < 2:
    return float('nan')
  return t975(s.size - 1) * float(np.std(s, ddof=1)) / math.sqrt(s.size)


def cgi_crossing(mp, vp, mq, vq, fallback=True):
  """Crooks Gaussian Intersection, vectorised: where N(mp, vp) and N(mq, vq) cross.

  Goette & Grubmueller, J Comput Chem 30, 447-456 (2009), eq. (12), taking the root
  closest to the midpoint of the means. Equating the two log densities gives
  a x^2 - 2 b x + c = 0 with a = 1/vp - 1/vq, b = mp/vp - mq/vq and
  c = mp^2/vp - mq^2/vq + ln(vp/vq); its discriminant b^2 - a c is the paper's
      (mp - mq)^2 / (vp vq) + (1/vp - 1/vq) ln(vq/vp)
  (the logarithm of the VARIANCE ratio, which absorbs the paper's factor 2 on
  ln(sigma2/sigma1); until 2026-08-10 the 2 was applied twice, see
  tests/test_cgi.py). The roots are taken as the numerically stable pair q/a and
  c/q, q = b + sign(b) sqrt(disc), rather than (b +- sqrt(disc))/a: equal variances
  then give the midpoint, which is the exact crossing of two Gaussians of one
  width, instead of 0/0, and widths equal to within rounding no longer
  ill-condition the answer (sd 30/30 had given -256.0 for a true -210.0).

  fallback applies the paper's p. 449 rule: "If both Gaussians were too close to
  compute a proper intersection point, i.e. if Wf and -Wr were closer to each other
  than to one of the two intersection values, we empirically chose the mean of both
  as the best estimate." Operationally, neither root between the two means. It is
  applied to finite roots only. A crossing that does not exist (a zero variance,
  identical distributions) is nan; the old test 'not (lo <= cgi <= hi)' was also
  true for nan and replaced it by the mean without a word.
  """
  mp, vp, mq, vq = (np.asarray(x, dtype=float) for x in (mp, vp, mq, vq))
  with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
    a = 1.0/vp - 1.0/vq
    b = mp/vp - mq/vq
    c = mp*mp/vp - mq*mq/vq + np.log(vp/vq)
    disc = (mp - mq)**2/(vp*vq) + a*np.log(vq/vp)
    q = b + np.where(b < 0, -1.0, 1.0) * np.sqrt(disc)
    r1 = q / a
    r2 = c / q
    mid = (mp + mq) / 2.0
    d1 = np.abs(r1 - mid)
    d2 = np.abs(r2 - mid)
    d1 = np.where(np.isfinite(d1), d1, np.inf)
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    out = np.where(d1 < d2, r1, r2)
    if fallback:
      out = np.where(np.isfinite(out) & ((out < np.minimum(mp, mq)) | (out > np.maximum(mp, mq))),
                     mid, out)
    out = np.where(np.isfinite(out), out, np.nan)
  return out if out.ndim else float(out)


# The CGI fits a width to each work distribution, and two widths from a handful of
# cycles are too noisy to cross: the threshold groscore.py has always used (n > 19).
CGI_MIN_N = 20
CGI_BOOT = 20000          # seeded, so the run-to-run spread no longer matters


def cgi_score(pulls, pushes):
  """(CGI, CI95, note): note is '' for a finite CGI, else the reason it is nan.

  OVERLAP GATE, the one BAR has (estimators.overlap_count == 0). In the classic
  regime the pulls and the pushes lie about 132 RT apart and share no
  range, so the two fitted Gaussians cross somewhere in their extrapolated tails,
  at about (sd_push*mean_pull + sd_pull*mean_push)/(sd_pull + sd_push): each mean
  weighted by the OTHER side's width, a value set by which side happens to be
  narrower, not by any sampled work. On 24 cycles 340 kJ/mol apart every
  avg was -230.0 while the CGI read -330.4, -233.3 or -129.6 depending only on the
  width ratio, each with a CI95 of 24 that hid it (v15_work digest_audit
  SCORER-2). Without overlap the CGI is nan with CGI_NO_OVERLAP.

  CI95: whole cycles resampled (paired, as for BAR), each resample gated the same
  way, the SD inflated by sqrt(n/(n-1)) for the plug-in bias and multiplied by
  t(0.975, n-1), so that for a mean it would reproduce avg_ci95's interval.
  """
  p = np.asarray(pulls, dtype=float)
  q = np.asarray(pushes, dtype=float)
  n = p.size
  if n < CGI_MIN_N:
    return float('nan'), float('nan'), "CGI_N_LT_%d" % CGI_MIN_N
  if est.overlap_count(p, -q) == 0:
    return float('nan'), float('nan'), "CGI_NO_OVERLAP"
  value = cgi_crossing(p.mean(), p.var(), q.mean(), q.var())
  if not np.isfinite(value):
    return float('nan'), float('nan'), "CGI_DEGENERATE"
  idx = np.random.default_rng(12345).integers(0, n, size=(CGI_BOOT, n))
  P, Q = p[idx], q[idx]
  b = cgi_crossing(P.mean(1), P.var(1), Q.mean(1), Q.var(1))
  n_in = (((P >= Q.min(1)[:, None]) & (P <= Q.max(1)[:, None])).sum(1)
          + ((Q >= P.min(1)[:, None]) & (Q <= P.max(1)[:, None])).sum(1))
  b = b[(n_in > 0) & np.isfinite(b)]
  ci95 = (t975(n - 1) * float(np.std(b)) * math.sqrt(n / (n - 1.0))
          if b.size > 1 else float('nan'))
  return float(value), ci95, ""

#------------------------------------------------------

def complete_cycles(frenstruct, i):
  """[(cycle_idx, pull, push)] of structure i's cycles with both legs finite."""
  out = []
  for c in range(frenstruct.shape[1] // 2):
    pull_val, push_val = frenstruct[i, 2*c], frenstruct[i, 2*c + 1]
    if not np.isnan(pull_val) and not np.isnan(push_val):
      out.append((c, pull_val, push_val))
  return out


def calculate_scores(frenstruct, structids, numstructs, num_cycles, use_max_data=False,
                     rmsdstruct=None, scored=None):
  """Calculate scores for structures with at least num_cycles complete cycles.

  Args:
    frenstruct: Array of free energy values [numstructs x (numruns*2)]
    structids: List of structure IDs
    numstructs: Number of structures
    num_cycles: Number of cycles to use (or minimum if use_max_data=True)
    use_max_data: If True, use all available data; if False, use only first num_cycles
    rmsdstruct: Optional array of re-bound backbone RMSDs [numstructs x numruns],
      summarised over exactly the cycles that entered the score
    scored: Optional per-structure booleans; a structure marked False (its last
      stage-0 status is a failure) is never scored, whatever rows it has left

  Returns:
    fren: List of (struct_id, avg_score, ci95, num_cycles_used, rmsd_mean, rmsd_max,
      bar, bar_ci95, bar_note, note) tuples (only structures with >= num_cycles)
    frencgi: Same, with the CGI score and its note
  """
  fren = []
  frencgi = []

  for i in range(numstructs):
    if scored is not None and not scored[i]:
      continue
    # Collect complete cycles (matching pull-push pairs)
    cycles = complete_cycles(frenstruct, i)
    num_complete_cycles = len(cycles)

    # Skip this structure if it doesn't have enough complete cycles
    if num_complete_cycles < num_cycles:
      continue

    # For convergence tracking, use first num_cycles complete cycles
    # For max data, use all complete cycles
    if not use_max_data and num_complete_cycles > num_cycles:
      cycles_to_use = cycles[:num_cycles]
      num_cycles_used = num_cycles
    else:
      cycles_to_use = cycles
      num_cycles_used = num_complete_cycles

    # Extract pulls and pushes from selected cycles
    pulls = [cycle[1] for cycle in cycles_to_use]
    pushes = [cycle[2] for cycle in cycles_to_use]

    # Re-bound RMSD over exactly those cycles (see rebound_rmsd.py)
    rmsd_mean, rmsd_max = rmsd_stats(rmsdstruct, i, [c[0] for c in cycles_to_use])

    avg_score = (np.average(pulls) + np.average(pushes)) / 2.0
    avg_ci = avg_ci95(pulls, pushes)
    avg_note = "" if np.isfinite(avg_ci) else "CI95_N_LT_2"
    cgi, cgi_ci, cgi_note = cgi_score(pulls, pushes)

    # BAR, as a comparison column only: the classic score stays the average.
    #
    # NOTE the negation. job.run integrates both legs at +rate, so a pull is
    # -W_unbind and a push is +W_rebind: both are estimates of dG_bind, the pull
    # from below and the push from above. est.bar(fwd, rev) wants a Crooks pair,
    # the reverse work in its own sign, and pulls with negated pushes are the
    # unbinding pair (W_unbind, W_rebind) negated as a whole. With as many
    # forward as reverse works, which complete cycles guarantee, BAR is odd under
    # that joint negation, bar(-x, -y) = -bar(x, y), so this returns
    # -dG_unbind = dG_bind, with the Score's sign. tests/test_bar.py pins both
    # directions.
    #
    # Expect BAR_NO_OVERLAP on essentially every structure: forward and reverse
    # works are separated by ~132 RT at the current pull rate, and BAR cannot
    # measure a crossing region that contains no samples. The suppression is the
    # informative part, so it is recorded per structure rather than left in prose.
    bar_score, bar_note = est.bar(pulls, [-p for p in pushes], RT)
    bar_ci95 = float('nan')
    if not np.isnan(bar_score) and len(pulls) >= est.BAR_MIN_N:
      idx = np.random.default_rng(12345).integers(0, len(pulls),
                                                  size=(5000, len(pulls)))
      b = est.bar_bootstrap(pulls, [-p for p in pushes], RT, idx)
      if np.isfinite(b).sum() > 1:
        bar_ci95 = 1.96 * float(np.nanstd(b))

    fren.append((structids[i], avg_score, avg_ci, num_cycles_used, rmsd_mean, rmsd_max,
                 bar_score, bar_ci95, bar_note, avg_note))
    frencgi.append((structids[i], cgi, cgi_ci, num_cycles_used, rmsd_mean, rmsd_max,
                    bar_score, bar_ci95, bar_note, cgi_note))

  return fren, frencgi

#------------------------------------------------------

def rmsd_stats(rmsdstruct, struct_idx, cycle_indices):
  """(mean, max) re-bound RMSD over the given cycles of one structure."""
  if rmsdstruct is None or not len(cycle_indices):
    return float('nan'), float('nan')
  vals = [rmsdstruct[struct_idx, c] for c in cycle_indices
          if c < rmsdstruct.shape[1] and not np.isnan(rmsdstruct[struct_idx, c])]
  if not vals:
    return float('nan'), float('nan')
  return float(np.mean(vals)), float(np.max(vals))

#------------------------------------------------------

# How much of the rebinding-QC warning is printed. The full picture is always in
# the score files; the console only needs enough to know what to look at.
MAX_FLAGGED_SHOWN = 10   # structures listed in the warning
MAX_CYCLES_SHOWN  = 5    # bad cycles listed per structure

def rmsd_cell(value):
  return "nan" if np.isnan(value) else "%.2f"%value

def rmsd_flag(rmsd_max, threshold):
  """Verdict of the rebinding sanity check for one structure."""
  if np.isnan(rmsd_max):
    return "nan"
  return "HIGH_RMSD" if rmsd_max > threshold else "OK"

#------------------------------------------------------

def write_score_file(path, header, structids, scored, struct_status, threshold):
  """Write one scores_*.gs file.

  scored: {struct_id: (score, ci95, ncycles, rmsd_mean, rmsd_max, bar, bar_ci,
  bar_note, score_note)} for the structures that produced data; every other
  structure is written with its stage-0 status so the file always lists all of
  sp.gs.

  BAR and Score_note are APPENDED, never inserted. Every consumer of these files
  reads parts[0] and parts[1] under a len(parts) >= 2 guard, and
  ppb_convergence_plot.py also parts[2], the CI95 (benchmark/capri_benchmark/
  capri_common.py, haddock_benchmark's analyze_correlation.py and
  analyze_convergence.py, ppb_benchmark's ppb_table.py, ppb_subset_scatter.py and
  ppb_convergence_plot.py), so appending breaks none of them and inserting would
  break all of them. The Score column stays the average or CGI as before; BAR is a
  comparison column and is not the classic engine's score. Score_note says why a
  Score or its CI95 is nan (CGI_N_LT_20, CGI_NO_OVERLAP, CGI_DEGENERATE,
  CI95_N_LT_2), nan when there is nothing to say.
  """
  with open(path, "w") as f:
    f.write(header)
    f.write("# Structure_ID  Score  CI95  Cycles_Used  RMSD_mean_A  RMSD_max_A  "
            "RMSD_flag  BAR  BAR_CI95  BAR_note  Score_note\n")
    for struct_id in structids:
      if struct_id in scored:
        score, ci95, nc, rmean, rmax, bar, bar_ci, bar_note, note = scored[struct_id]
        score_cols = ("nan\tnan" if np.isnan(score) else "%.1f\t%.1f"%(score, ci95))
        bar_cols = ("nan\tnan" if np.isnan(bar) else "%.1f\t%.1f"%(bar, bar_ci))
        f.write("%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n"%(struct_id, score_cols, nc,
                                                rmsd_cell(rmean), rmsd_cell(rmax),
                                                rmsd_flag(rmax, threshold),
                                                bar_cols, bar_note or "nan", note or "nan"))
      else:
        status = struct_status.get(struct_id, "nan")
        f.write("%s\t%s\t%s\t0\tnan\tnan\tnan\tnan\tnan\tnan\tnan\n"%(struct_id, status, status))

#------------------------------------------------------

def array_spec(indices):
  """SLURM --array value for a sorted list of indices, consecutive runs collapsed."""
  parts, start = [], None
  for k, i in enumerate(indices):
    if start is None:
      start = i
    if k == len(indices) - 1 or indices[k + 1] != i + 1:
      parts.append(str(start) if start == i else "%d-%d" % (start, i))
      start = None
  return ",".join(parts)

#------------------------------------------------------

print("")
print("##################################")
print("#                                #")
print("#            GroScore            #")
print("#                                #")
print("##################################")
print("")

structids, structchains = readstructparams(args.structparams)
numstructs = len(structids)
if numstructs == 0:
  print("Error: No valid structures found in " + args.structparams)
  exit(1)
print("Reading input parameters finished.")
print("GroScore will calculate a binding free energy estimate for " + str(numstructs) + " structures.")
print("Each structure will undergo " + str(args.numruns) + " independent equilibration cycles (each cycle = 1 pull + 1 push).")
if args.restart:
  print("RESTART MODE: Will resubmit jobs even if run.gs exists.")
print("")

#------------------------------------------------------
# Setup and submission

if args.restart or not os.path.isfile("results_0.gs"):
  if not os.path.isfile("results_0.gs"):
    f = open("results_0.gs", "a")
    f.write("# Results for simulation fitness:\n")
    f.close()
  # Write structure ID mapping file for job.run
  f = open("struct_map.gs", "w")
  f.write("# Array_Index Structure_ID\n")
  i = 0
  while i < numstructs:
    f.write("%d %s\n"%(i, structids[i]))
    i += 1
  f.close()
  # Write run.gs and copy job.run for each structure
  script_dir = os.path.dirname(os.path.abspath(__file__))
  job_run_src = os.path.join(script_dir, "job.run")
  if not os.path.isfile(job_run_src):
    print("Error: job.run not found in %s"%script_dir)
    exit(1)
  live = []  # struct_map indices of the structures submitted: a directory, or an archive given the current job.run
  i = 0
  while i < numstructs:
    if os.path.exists("./%s"%structids[i]):
      live.append(i)
      print("Setting up %s."%structids[i])
      # Only write run.gs if NOT in restart mode
      if not args.restart:
        f = open("./%s/run.gs"%structids[i], "w")
        cutout_flag = 1 if args.cutout else 0
        # MAXRUNS = numruns * 2 because each cycle has one pull (odd) and one push (even)
        ligand_param_flag = 1 if args.ligand_param else 0
        f.write("%s %d %d %s %d\n"%(structchains[i],args.numruns*2,cutout_flag,args.forcefield,ligand_param_flag))
        f.close()
      # Copy job.run to structure directory and make executable
      job_run_dst = "./%s/job.run"%structids[i]
      shutil.copy(job_run_src, job_run_dst)
      os.chmod(job_run_dst, 0o755)
    elif os.path.isfile("./%s.tar.gz"%structids[i]):
      if args.inject_job_run:
        live.append(i)
        sys.stdout.write("Setting up %s.tar.gz. "%structids[i])
        sys.stdout.flush()
        # Archived structure: inject fresh job.run into archive
        # Uses Python tarfile to stream gz→gz without full decompression on disk
        import tarfile
        tmpdir = "./%s"%structids[i]
        os.makedirs(tmpdir, exist_ok=True)
        shutil.copy(job_run_src, os.path.join(tmpdir, "job.run"))
        os.chmod(os.path.join(tmpdir, "job.run"), 0o755)
        archive_path = "./%s.tar.gz"%structids[i]
        new_archive_path = "./%s_new.tar.gz"%structids[i]
        with tarfile.open(archive_path, "r:gz") as old_tar:
          with tarfile.open(new_archive_path, "w:gz") as new_tar:
            for member in old_tar:
              if member.name.endswith("/job.run"):
                continue  # skip old job.run
              new_tar.addfile(member, old_tar.extractfile(member) if member.isreg() else None)
            new_tar.add(os.path.join(tmpdir, "job.run"), arcname="./%s/job.run"%structids[i])
        os.replace(new_archive_path, archive_path)
        print("Done.")
        shutil.rmtree(tmpdir)
      else:
        # NOT submitted. Through v1.4 this printed 'Skipping' and submitted the
        # archive anyway: the array covered every structure, so the job.run inside
        # the tarball re-walked its finished cycles with whatever utils the checkout
        # now held, re-integrated them with a freshly regenerated restraint count,
        # and could never add a cycle, because it read the count from the run.gs
        # inside the archive (v15_work digest_audit runner R6, SCORER-5 and the
        # re-walk extra finding).
        print("%s.tar.gz is archived and is NOT submitted: the job.run inside it "
              "predates this code. Pass --inject-job-run to run it with this code "
              "(needed to add cycles with a larger -n)."%structids[i])
    else:
      print("Structure %s: directory doesn't exist."%structids[i])
      f = open("results_0.gs", "a")
      f.write("%s NODIR\n"%structids[i])
      f.close()
    i += 1
  # How many cycles the jobs run in total. job.run takes GROSCORE_NUMCYCLES over
  # the count in run.gs, which --restart never rewrites and which an archived
  # structure keeps inside its tarball, so this is what lets '-n <more> --restart'
  # add cycles (the job_fe.run mechanism).
  if not live:
    print("Nothing to submit.")
  elif args.run_local:
    # Single workstation, no scheduler: one job per structure, executed by
    # utils/local_runner.py in a bounded pool with one GPU pinned per slot.
    jobs = [{"name": structids[i], "dir": structids[i], "argv": ["./job.run"],
             "archive": "%s.tar.gz"%structids[i], "log": "job_local.out",
             "env": {"GROSCORE_NUMCYCLES": str(args.numruns)}} for i in live]
    launch_local(jobs, args.ngpus, args.jobs_per_gpu, args.threads_per_job)
    print("")
  else:
    #SBATCH array
    slurm_template = os.path.join(script_dir, "slurm", args.slurm + ".sh")
    if not os.path.isfile(slurm_template):
      print("Error: SLURM template not found: %s"%slurm_template)
      exit(1)
    with open(slurm_template, "r") as tmpl:
      template_lines = tmpl.read()
    f = open("array_submit.run", "w")
    f.write(template_lines.rstrip("\n") + "\n")
    # Only the submitted structures, by their struct_map.gs index.
    f.write("#SBATCH --array=%s\n"%array_spec(live))
    f.write("\n")
    f.write("export GROSCORE_NUMCYCLES=%d\n"%args.numruns)
    f.write("# Read structure ID from mapping file\n")
    f.write("STRUCT_ID=$(awk -v idx=\"$SLURM_ARRAY_TASK_ID\" '$1 == idx {print $2}' struct_map.gs)\n")
    # The tarball goes only after a complete extraction. A failed one used to be
    # followed by 'rm' anyway, losing what the archive still held, and the next
    # task then ran job.run on a partial directory (v15_work digest_audit
    # runner R4).
    f.write("# Extract archived structure if needed (for restarts)\n")
    f.write("if [[ ! -d \"$STRUCT_ID\" && -f \"${STRUCT_ID}.tar.gz\" ]]; then\n")
    f.write("  if tar -xzf \"${STRUCT_ID}.tar.gz\"; then\n")
    f.write("    rm \"${STRUCT_ID}.tar.gz\"\n")
    f.write("  else\n")
    f.write("    echo \"Could not extract ${STRUCT_ID}.tar.gz; it is kept, the partial directory removed.\"\n")
    f.write("    rm -rf \"$STRUCT_ID\"\n")
    f.write("    exit 1\n")
    f.write("  fi\n")
    f.write("fi\n")
    f.write("cd \"$STRUCT_ID\" || exit 1\n")
    f.write("./job.run\n")
    f.close()
    # Flushed first: sbatch writes to the same stdout, and into a redirected log its
    # line otherwise lands above everything printed so far.
    sys.stdout.flush()
    os.system("sbatch array_submit.run")
    print("Submitted all simulation jobs.")
    print("")

#------------------------------------------------------
# Stage 0: the last status of every structure

laststatus, nbad = read_status("results_0.gs")
if nbad:
  print("WARNING: %d malformed line%s in results_0.gs skipped." % (nbad, "s" if nbad > 1 else ""))
calcstruct = np.zeros(shape=(numstructs))
struct_status = {}  # struct_id -> last status when it is not OK (BROKEN, ENTANGLED, NODIR, ...)
scored = []
for l in range(numstructs):
  st = laststatus.get(structids[l])
  scored.append(scoreable(st))
  calcstruct[l] = 1 if scoreable(st) else 0
  if st is not None and st != "OK":
    struct_status[structids[l]] = st
if os.path.isfile("results_0.gs"):
  np.savetxt("calcstruct.gs",calcstruct,delimiter="\t")

#------------------------------------------------------
# Cycle data: per-cycle files (job.run since v1.5), else the legacy results_<n>.gs

frenstruct = np.zeros(shape=(numstructs,args.numruns*2))
frenstruct[:,:] = "NaN"
# Re-bound backbone RMSD per structure and cycle (Angstrom). Purely diagnostic: it
# never changes a score, it only flags cycles in which the partners failed to
# return to the bound pose.
rmsdstruct = np.zeros(shape=(numstructs,args.numruns))
rmsdstruct[:,:] = "NaN"
index = {s: l for l, s in enumerate(structids)}

percycle, marked, badfiles = read_percycle("results.d", set(structids))
if badfiles:
  print("WARNING: %d malformed file%s in results.d skipped: %s" % (
        len(badfiles), "s" if len(badfiles) > 1 else "", ", ".join(badfiles[:5])
        + (", ..." if len(badfiles) > 5 else "")))
for sid, cyc in percycle.items():
  for c, (pull, push, rmsd) in cyc.items():
    if 1 <= c <= args.numruns:
      frenstruct[index[sid], 2*c - 2] = pull
      frenstruct[index[sid], 2*c - 1] = push
      rmsdstruct[index[sid], c - 1] = rmsd

nbad = 0
conflicts = {}      # struct_id -> first (file, old, new) where a later finite row changed a work
for j in range(1, args.numruns*2 + 1):
  rows, bad = read_legacy("results_%d.gs" % j, push=(j % 2 == 0))
  nbad += bad
  for sid, val, rmsd in rows:
    if sid not in index or sid in marked:
      continue
    l = index[sid]
    if not np.isnan(val):
      old = frenstruct[l, j-1]
      if not np.isnan(old) and old != val and sid not in conflicts:
        conflicts[sid] = ("results_%d.gs" % j, old, val)
      frenstruct[l, j-1] = val
    if j % 2 == 0:
      rmsdstruct[l, j//2 - 1] = rmsd
if nbad:
  print("WARNING: %d malformed row%s in results_<n>.gs skipped (an empty work field)."
        % (nbad, "s" if nbad > 1 else ""))
if conflicts:
  # Legacy files are append-only and shared by every setup a structure ever had in
  # this run directory; differing works for one cycle mean two setups or a
  # re-integration are mixed (v15_work digest_audit runner R3). The last is used.
  print("WARNING: %d structure%s with differing works for one cycle in results_<n>.gs "
        "(the last is used):" % (len(conflicts), "s" if len(conflicts) > 1 else ""))
  for sid in list(conflicts)[:MAX_FLAGGED_SHOWN]:
    fname, old, new = conflicts[sid]
    print("    %-20s %s: %.4f then %.4f" % (sid, fname, old, new))
np.savetxt("frenstruct.gs",frenstruct,delimiter="\t")
np.savetxt("rmsdstruct.gs",rmsdstruct,delimiter="\t")

#------------------------------------------------------
# Scores

ncomplete = [len(complete_cycles(frenstruct, l)) if scored[l] else 0 for l in range(numstructs)]
max_cycles = max(ncomplete) if ncomplete else 0
if os.path.isfile("results_0.gs"):
  sys.stdout.write("Calculating scores... ")
  sys.stdout.flush()
  if max_cycles > 0:
    check_ref_t(args.forcefield)    # BAR needs kT; warn if the mdp disagrees

  # Write score files for each cycle threshold (1 to the most cycles any
  # structure has complete); the main files below list every structure even
  # before any cycle has finished.
  for cycle_threshold in range(1, max_cycles + 1):
    # Calculate scores using only first cycle_threshold cycles (for convergence tracking)
    fren, frencgi = calculate_scores(frenstruct, structids, numstructs, cycle_threshold,
                                     use_max_data=False, rmsdstruct=rmsdstruct, scored=scored)

    # Write scores for this cycle threshold (all structures, nan for missing)
    avg_c = {s: rest for s, *rest in fren}
    cgi_c = {s: rest for s, *rest in frencgi}
    cyc_header = "# Scores using first %d cycle%s\n"%(cycle_threshold,
                                                      "s" if cycle_threshold > 1 else "")
    write_score_file("scores_avg_c%d.gs"%cycle_threshold, cyc_header,
                     structids, avg_c, struct_status, args.rmsd_warn)
    write_score_file("scores_cgi_c%d.gs"%cycle_threshold, cyc_header,
                     structids, cgi_c, struct_status, args.rmsd_warn)

  # Update main score files (all structures using their maximum available data)
  # Include all structures with at least 1 complete cycle, each using all its available data
  fren_max, frencgi_max = calculate_scores(frenstruct, structids, numstructs, num_cycles=1,
                                           use_max_data=True, rmsdstruct=rmsdstruct,
                                           scored=scored)

  # Build lookup dicts for scores (keyed by struct_id)
  avg_scores = {s: rest for s, *rest in fren_max}
  cgi_scores = {s: rest for s, *rest in frencgi_max}

  # Write all structures (matching sp.gs order), nan for missing data
  max_header = "# Scores for all structures (each using maximum available data)\n"
  write_score_file("scores_avg.gs", max_header, structids, avg_scores,
                   struct_status, args.rmsd_warn)
  write_score_file("scores_cgi.gs", max_header, structids, cgi_scores,
                   struct_status, args.rmsd_warn)

  print("Done!")

  # Check if all expected cycles are complete
  nok = sum(1 for l in range(numstructs) if scored[l])
  nall = sum(1 for l in range(numstructs) if scored[l] and ncomplete[l] >= args.numruns)
  print("")
  if nok and nall == nok:
    print("All %d requested cycles are complete!"%args.numruns)
  else:
    print("%d of %d structures with stage-0 status OK have all %d requested cycles complete."
          %(nall, nok, args.numruns))
  print("")

#------------------------------------------------------
# Rebinding sanity check
#
# Every cycle pulls the partners apart and pushes them back together; job.run
# measures how well the re-bound structure superimposes on the bound state the
# cycle started from. Cycles that did not find their way back describe a
# different (or no) binding event, so their work values are suspect even though
# they are still scored and reported.

# Progress of a --run-local run. No-op when the jobs went to SLURM instead.
print("")
print_local_status()

print("")
print("Rebinding sanity check (backbone RMSD of the re-bound structure, warn > %.1f A):"%args.rmsd_warn)
allvals = rmsdstruct[~np.isnan(rmsdstruct)]
if allvals.size == 0:
  print("  No RMSD values available yet (no cycle has finished, or the runs predate this check).")
else:
  print("  %d simulations analyzed: mean %.2f A, median %.2f A, max %.2f A"
        %(allvals.size, np.mean(allvals), np.median(allvals), np.max(allvals)))
  flagged = []
  for i in range(numstructs):
    bad = [(c+1, rmsdstruct[i,c]) for c in range(rmsdstruct.shape[1])
           if not np.isnan(rmsdstruct[i,c]) and rmsdstruct[i,c] > args.rmsd_warn]
    if bad:
      flagged.append((structids[i], bad))
  if not flagged:
    print("  All measured cycles re-bound within the threshold.")
  else:
    print("")
    print("  WARNING: %d structure%s did not re-bind properly in at least one cycle."
          %(len(flagged), "s" if len(flagged) > 1 else ""))
    print("  Their scores are reported but should be treated with caution")
    print("  (flagged HIGH_RMSD in scores_avg.gs / scores_cgi.gs):")
    # Worst first and truncated: a large screen can flag hundreds of poses, and a
    # wall of them buries the summary above. The score files hold the full list.
    flagged.sort(key=lambda x: max(v for _, v in x[1]), reverse=True)
    for struct_id, bad in flagged[:MAX_FLAGGED_SHOWN]:
      detail = ", ".join("c%d=%.1f"%(c, v) for c, v in bad[:MAX_CYCLES_SHOWN])
      if len(bad) > MAX_CYCLES_SHOWN:
        detail += ", +%d more"%(len(bad) - MAX_CYCLES_SHOWN)
      print("    %-20s %s"%(struct_id, detail))
    if len(flagged) > MAX_FLAGGED_SHOWN:
      print("    ... and %d more (grep HIGH_RMSD scores_avg.gs for the full list)"
            %(len(flagged) - MAX_FLAGGED_SHOWN))
print("")
