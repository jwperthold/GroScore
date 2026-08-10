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
#        | +dG_unbind     (interface -> Boresch handoff + separation to 1.5 nm;
#        |                 pull-force work + dhdl work)
#   Boresch-restrained-unbound
#        | +dG_release    (analytical Boresch standard-state term, eq. 32)
#   unrestrained-unbound (1 mol/L standard state)
#
#   dG_bind = -(dG_intro + dG_unbind + dG_release)
#
# Each simulated leg is run forward and reverse per cycle; the free energies are
# estimated with the Crooks-Gaussian-Intersection (CGI) and the simple average
# estimator, reusing the machinery of the classic engine.
#
# job_fe.run writes, per completed cycle, a line to results_fe.gs:
#   STRUCT_ID cycle W_intro Wunbind_pull Wunbind_dhdl Wrebind_pull Wrebind_dhdl W_remove
# and, per structure, a line to results_analytical.gs:
#   STRUCT_ID dG_release_kJ_mol
# All works are in kJ/mol.
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
# NOTE: the relative sign of the pull-work and dhdl-work terms, and the pull-force
# sign convention of this GROMACS build, MUST be validated on the first real
# bidirectional run using the standard CGI diagnostic (forward and reverse work
# histograms should overlap around dG). See the SIGN_* constants below.

import math, os, sys, argparse, shutil, glob, subprocess
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="GroScore FE: absolute binding free energy via Boresch restraints.")
parser.add_argument('-n', '--numruns', type=int, default=5,
                    help="TOTAL bidirectional cycles wanted per structure (default: 5). "
                         "Re-run with a larger value plus --restart to add cycles: only "
                         "the cycles without a complete result are submitted.")
parser.add_argument('-s', '--structparams', type=str, default="sp.gs", help="Structure parameter file (default: sp.gs).")
parser.add_argument('-ff', '--forcefield', type=str, default="amber19sb_opc3",
                    choices=["gromos54a8", "charmm36", "amber19sb_opc", "amber19sb_opc3"],
                    help="Force field (default: amber19sb_opc3).")
parser.add_argument('--no-cutout', dest='cutout', action='store_false', help="Disable interface cutout.")
parser.add_argument('--no-ligand-param', dest='ligand_param', action='store_false', help="Disable OpenFF small-molecule parametrization.")
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
parser.add_argument('--inject-job-run', action='store_true', help="Inject fresh job_fe.run into archived (.tar.gz) structures.")
parser.add_argument('--temp', type=float, default=310.0, help="Temperature in K (default: 310).")
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
parser.set_defaults(cutout=True, ligand_param=True)
args = parser.parse_args()

if args.run_local and args.ngpus < 1:
  parser.error("--run-local needs --ngpus N: the number of GPUs the jobs are distributed over")
if args.ngpus and not args.run_local:
  parser.error("--ngpus only applies to --run-local; SLURM allocates GPUs itself")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from local_runner import launch_local, print_local_status

RT = 0.00831446261815324 * args.temp  # kJ/mol

# How much of the rebinding-QC warning is printed. The full picture is always in
# scores_fe.gs; the console only needs enough to know what to look at.
MAX_FLAGGED_SHOWN = 10   # structures listed in the warning
MAX_CYCLES_SHOWN  = 5    # bad cycles listed per structure

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
    out[m] = np.where(np.abs(mid - s1) > np.abs(mid - s2), s2, s1)
  return out

def score_structure(W_intro, W_remove, Wtot_f, Wtot_r, dG_release,
                     n_boot=50000, seed=12345):
  """Joint cycle-level bootstrap for one structure.

  Point estimates and 95% CIs for dG_intro, dG_unbind and dG_bind under BOTH the
  average (Crooks) and CGI estimators. The bootstrap resamples CYCLES (the
  sampling unit) with a SHARED index across the bound and unbinding streams, so
  the dG_bind CI correctly includes the covariance between dG_intro and dG_unbind
  (both are estimated from the same cycles) rather than assuming independence.
  Forward/reverse works are paired by cycle. dG_release is analytical and treated
  as exact (contributes no error).

    bound  stream: forward = W_intro (restraints on),  reverse = W_remove (off)
    unbind stream: forward = Wtot_f  (unbinding),        reverse = Wtot_r (rebinding)
    dG_bind = -(dG_intro + dG_unbind + dG_release)
  """
  Wi = np.asarray(W_intro, float); Wr = np.asarray(W_remove, float)
  Wf = np.asarray(Wtot_f, float);  Wv = np.asarray(Wtot_r, float)
  ncyc = len(Wi)
  nan = float('nan')

  r = dict(n=ncyc,
           intro_avg=nan, intro_avg_ci=nan, intro_cgi=nan, intro_cgi_ci=nan,
           unb_avg=nan, unb_avg_ci=nan, unb_cgi=nan, unb_cgi_ci=nan,
           bind_avg=nan, bind_avg_ci=nan, bind_cgi=nan, bind_cgi_ci=nan)
  if ncyc == 0:
    return r

  # Point estimates from the full data.
  r['intro_avg'] = float(_stream_avg(Wi, Wr))
  r['unb_avg']   = float(_stream_avg(Wf, Wv))
  r['bind_avg']  = -(r['intro_avg'] + r['unb_avg'] + dG_release)
  if ncyc >= 3:                                   # CGI needs the per-cycle variance
    r['intro_cgi'] = float(_stream_cgi(Wi[None, :], Wr[None, :])[0])
    r['unb_cgi']   = float(_stream_cgi(Wf[None, :], Wv[None, :])[0])
    if np.isfinite(r['intro_cgi']) and np.isfinite(r['unb_cgi']):
      r['bind_cgi'] = -(r['intro_cgi'] + r['unb_cgi'] + dG_release)
  if ncyc < 2:
    return r

  # Joint bootstrap: one shared cycle-index resample drives both streams.
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, ncyc, size=(n_boot, ncyc))
  Wi_b, Wr_b, Wf_b, Wv_b = Wi[idx], Wr[idx], Wf[idx], Wv[idx]

  ia = _stream_avg(Wi_b, Wr_b, axis=1)
  ua = _stream_avg(Wf_b, Wv_b, axis=1)
  r['intro_avg_ci'] = 1.96 * float(np.std(ia))
  r['unb_avg_ci']   = 1.96 * float(np.std(ua))
  r['bind_avg_ci']  = 1.96 * float(np.std(-(ia + ua + dG_release)))

  if ncyc >= 3:
    ic = _stream_cgi(Wi_b, Wr_b)
    uc = _stream_cgi(Wf_b, Wv_b)
    if np.isfinite(ic).sum() > 1:
      r['intro_cgi_ci'] = 1.96 * float(np.nanstd(ic))
    if np.isfinite(uc).sum() > 1:
      r['unb_cgi_ci'] = 1.96 * float(np.nanstd(uc))
    both = np.isfinite(ic) & np.isfinite(uc)
    if both.sum() > 1:
      r['bind_cgi_ci'] = 1.96 * float(np.std(-(ic[both] + uc[both] + dG_release)))
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

def setup_and_submit(structids, structchains):
  """Write per-structure run.gs, copy job_fe.run, build and submit the jobs."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  job_src = os.path.join(script_dir, "job_fe.run")
  if not os.path.isfile(job_src):
    print("Error: job_fe.run not found in %s" % script_dir); sys.exit(1)

  with open("struct_map.gs", "w") as f:
    f.write("# Array_Index Structure_ID\n")
    for i, sid in enumerate(structids):
      f.write("%d %s\n" % (i, sid))

  for i, sid in enumerate(structids):
    if os.path.isdir("./%s" % sid):
      print("Setting up %s." % sid)
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
    elif os.path.isfile("./%s.tar.gz" % sid):
      if args.inject_job_run:
        import tarfile
        tmpdir = "./%s" % sid
        os.makedirs(tmpdir, exist_ok=True)
        shutil.copy(job_src, os.path.join(tmpdir, "job.run"))
        os.chmod(os.path.join(tmpdir, "job.run"), 0o755)
        arc = "./%s.tar.gz" % sid
        new = "./%s_new.tar.gz" % sid
        with tarfile.open(arc, "r:gz") as old_tar, tarfile.open(new, "w:gz") as new_tar:
          for m in old_tar:
            if m.name.endswith("/job.run"):
              continue
            new_tar.addfile(m, old_tar.extractfile(m) if m.isreg() else None)
          new_tar.add(os.path.join(tmpdir, "job.run"), arcname="./%s/job.run" % sid)
        os.replace(new, arc)
        shutil.rmtree(tmpdir)
        print("Injected job_fe.run into %s.tar.gz" % sid)
      else:
        print("Skipping %s.tar.gz (archived, use --inject-job-run)." % sid)
    else:
      print("Structure %s: directory doesn't exist." % sid)
      with open("results_0.gs", "a") as f:
        f.write("%s NODIR\n" % sid)

  slurm_template = os.path.join(script_dir, "slurm", args.slurm + ".sh")
  if not os.path.isfile(slurm_template):
    print("Error: SLURM template not found: %s" % slurm_template); sys.exit(1)
  with open(slurm_template) as t:
    template = t.read().rstrip("\n")

  live = [sid for sid in structids
          if os.path.isdir("./%s" % sid) or os.path.isfile("./%s.tar.gz" % sid)]
  if not live:
    print("No structures to submit.\n"); return

  def extract_snippet(sid):
    # A structure archived after finishing its cycles is unpacked again when more
    # cycles are added. .archive.lock is inside the tarball, so drop it or the
    # completed structure could never be re-archived.
    return ("if [[ ! -d \"%s\" && -f \"%s.tar.gz\" ]]; then\n"
            "  tar -xzf \"%s.tar.gz\"\n  rm \"%s.tar.gz\"\n"
            "  rm -rf \"%s/.archive.lock\"\nfi\n" % (sid, sid, sid, sid, sid))

  if args.sequential:
    # Legacy layout: one task per structure, running setup + all cycles
    # sequentially inside a single job.
    if args.run_local:
      jobs = [{"name": sid, "dir": sid, "argv": ["./job.run"],
               "archive": "%s.tar.gz" % sid, "log": "job_local.out"} for sid in live]
      launch_local(jobs, args.ngpus, args.jobs_per_gpu, args.threads_per_job)
      print("")
      return
    with open("array_submit.run", "w") as f:
      f.write(template + "\n")
      f.write("#SBATCH --array=0-%d\n\n" % (len(structids) - 1))
      f.write("STRUCT_ID=$(awk -v idx=\"$SLURM_ARRAY_TASK_ID\" '$1 == idx {print $2}' struct_map.gs)\n")
      f.write("if [[ ! -d \"$STRUCT_ID\" && -f \"${STRUCT_ID}.tar.gz\" ]]; then\n")
      f.write("  tar -xzf \"${STRUCT_ID}.tar.gz\"\n  rm \"${STRUCT_ID}.tar.gz\"\nfi\n")
      f.write("cd $STRUCT_ID\n./job.run\n")
    os.system("sbatch array_submit.run")
    print("Submitted 1 array of %d structures (sequential cycles).\n" % len(structids))
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

  # -n is the TOTAL number of cycles wanted, so re-running with a larger -n adds
  # cycles. Only the cycles that have no complete result yet are submitted, so
  # topping up a structure costs one array task per missing cycle rather than
  # re-walking everything. Cycles whose result contains NaN count as missing and
  # are recomputed. The total is passed via GROSCORE_NUMCYCLES because run.gs
  # cannot be rewritten inside an archived structure.
  done_cycles = {}
  for sid, rows_ in read_works("results_fe.gs").items():
    done_cycles[sid] = {int(r[0]) for r in rows_}

  local_jobs = []
  n_setup = n_array = 0
  for sid in live:
    have = done_cycles.get(sid, set())
    missing = [c for c in range(1, args.numruns + 1) if c not in have]
    if not missing:
      print("  %s: all %d cycles complete, nothing to submit." % (sid, args.numruns))
      continue

    if args.run_local:
      # Same shape as the SLURM path: one setup job, then the missing cycles
      # depending on it. The dependency is "after any", like --dependency=afterany
      # above, so a cycle whose setup failed still starts, reads the stage-0
      # status and exits cleanly instead of waiting forever.
      deps = []
      if not os.path.isfile("./%s/setup.done" % sid):
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
    if not os.path.isfile("./%s/setup.done" % sid):
      jid = sbatch_parsable(setup_path)
      if jid:
        dep = "--dependency=afterany:%s " % jid
        n_setup += 1
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
    return

  print("Submitted %d setup job(s) + %d cycle array(s), target %d cycles/structure%s.\n"
        % (n_setup, n_array, args.numruns,
           " (max %d concurrent)" % args.array_throttle if args.array_throttle > 0 else ""))

#------------------------------------------------------

def read_status(filepath, structids):
  """Read results_0.gs -> {struct_id: status} for non-OK stage-0 outcomes."""
  status = {}
  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        if line.strip().startswith("#"):
          continue
        tmp = line.split()
        if len(tmp) >= 2 and tmp[0] in structids and tmp[1] != "OK":
          status[tmp[0]] = tmp[1]
  return status

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
  """-> {struct_id: [ (cycle, W_intro, Wu_pull, Wu_dhdl, Wr_pull, Wr_dhdl,
  W_remove, rebound_rmsd), ... ]}, keeping only rows whose WORK values are all
  numeric.

  The RMSD is the rebinding sanity check (9th column, Angstrom); it is diagnostic
  only, so a row whose RMSD is missing or NaN is still a valid result and is kept
  with nan in that slot.

  Reads the per-cycle files written by the cycle array (results_fe.d/<sid>_c<n>.gs,
  one line each -- parallel tasks must not append to a shared file) as well as any
  legacy single results_fe.gs, so runs started before the split still score."""
  works = {}

  def take(line):
    if line.strip().startswith("#"):
      return
    tmp = line.split()
    if len(tmp) < 8:
      return
    try:
      cyc = int(tmp[1])
      vals = [float(x) for x in tmp[2:8]]
    except ValueError:
      return
    if any(math.isnan(v) for v in vals):
      return
    rmsd = float('nan')
    if len(tmp) >= 9:
      try:
        rmsd = float(tmp[8])
      except ValueError:
        pass
    works.setdefault(tmp[0], []).append((cyc, *vals, rmsd))

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

#------------------------------------------------------

def score(structids):
  status = read_status("results_0.gs", structids)
  analytical = read_analytical("results_analytical.gs")
  works = read_works("results_fe.gs")

  rows = []  # (sid, result_dict_or_None, dG_release_or_None, ncyc, note)
  bad_cycles = {}   # sid -> [(cycle, rmsd), …] that failed the rebinding check
  all_rmsds = []    # every measured cycle, for the summary statistics
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
    if not cycles or sid not in analytical:
      rows.append((sid, None, None, len(cycles), "PENDING"))
      continue

    W_intro   = [c[1] for c in cycles]
    Wu_pull   = [c[2] for c in cycles]
    Wu_dhdl   = [c[3] for c in cycles]
    Wr_pull   = [c[4] for c in cycles]
    Wr_dhdl   = [c[5] for c in cycles]
    W_remove  = [c[6] for c in cycles]

    # Physical total works for the unbinding/rebinding stream.
    Wtot_f = [SIGN_PULL_FWD * up + ud for up, ud in zip(Wu_pull, Wu_dhdl)]
    Wtot_r = [SIGN_PULL_REV * rp + rd for rp, rd in zip(Wr_pull, Wr_dhdl)]

    dG_release = analytical[sid]
    r = score_structure(W_intro, W_remove, Wtot_f, Wtot_r, dG_release)

    # Rebinding sanity check: the thermodynamic cycle only closes if the
    # rebinding leg put the partners back into the pose the bound leg started
    # from. Diagnostic only -- the free energies are reported either way.
    rmsds = [(c[0], c[7]) for c in cycles if not math.isnan(c[7])]
    r['rmsd_mean'] = float(np.mean([v for _, v in rmsds])) if rmsds else float('nan')
    r['rmsd_max']  = float(np.max([v for _, v in rmsds])) if rmsds else float('nan')
    all_rmsds.extend(v for _, v in rmsds)
    bad = [(c, v) for c, v in rmsds if v > args.rmsd_warn]
    if bad:
      bad_cycles[sid] = bad
    rows.append((sid, r, dG_release, len(cycles), "HIGH_RMSD" if bad else ""))

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

  # dG_bind, dG_intro and dG_unbind are each reported under BOTH the average and
  # CGI estimators, each with its own 95% CI. The dG_bind CIs come from the joint
  # cycle bootstrap (they include the dG_intro/dG_unbind covariance, so they are
  # NOT simply the quadrature of the component CIs).
  # RMSD_mean/RMSD_max are the rebinding sanity check (Angstrom); a structure with
  # any cycle above --rmsd-warn additionally carries HIGH_RMSD in Note.
  cols = ("dGbind_avg  dGbind_avg_CI  pKD_avg  pKD_avg_CI  dGbind_cgi  dGbind_cgi_CI  pKD_cgi  pKD_cgi_CI  "
          "dG_intro_avg  dG_intro_avg_CI  dG_intro_cgi  dG_intro_cgi_CI  "
          "dG_unbind_avg  dG_unbind_avg_CI  dG_unbind_cgi  dG_unbind_cgi_CI  "
          "dG_release  RMSD_mean_A  RMSD_max_A  Ncycles  Note")
  with open("scores_fe.gs", "w") as f:
    f.write("# GroScore-FE absolute binding free energies (kJ/mol; pKD dimensionless, T=%.1f K)\n" % args.temp)
    f.write("# Structure_ID  " + "  ".join(cols.split()) + "\n")
    for sid, r, gr, n, note in rows:
      if r is None:
        f.write("\t".join([sid] + ["nan"] * 19 + [str(n), note or ""]) + "\n")
      else:
        vals = [cell(r['bind_avg']), cell(r['bind_avg_ci']), cell(pkd(r['bind_avg'])), cell(pkd_ci(r['bind_avg_ci'])),
                cell(r['bind_cgi']), cell(r['bind_cgi_ci']), cell(pkd(r['bind_cgi'])), cell(pkd_ci(r['bind_cgi_ci'])),
                cell(r['intro_avg']), cell(r['intro_avg_ci']),
                cell(r['intro_cgi']), cell(r['intro_cgi_ci']),
                cell(r['unb_avg']), cell(r['unb_avg_ci']),
                cell(r['unb_cgi']), cell(r['unb_cgi_ci']),
                cell(gr), cell(r['rmsd_mean']), cell(r['rmsd_max'])]
        f.write("\t".join([sid] + vals + [str(n), note]) + "\n")

  done = len(rows_valid)
  print("Scored %d/%d structures with complete cycles. Wrote scores_fe.gs." % (done, len(structids)))

  # ── rebinding sanity check summary ──────────────────────────────────────────
  # Every cycle separates the partners and pushes them back; the backbone RMSD
  # between the bound equilibration and the end of the rebinding leg says whether
  # the cycle actually closed on the state it started from.
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
  print("GroScore-FE: %d structures, %d bidirectional cycles each.\n" % (len(structids), args.numruns))

  # Submit jobs on first invocation (or on --restart).
  if args.restart or not os.path.isfile("struct_map.gs"):
    if not os.path.isfile("results_0.gs"):
      with open("results_0.gs", "w") as f:
        f.write("# Stage-0 setup status\n")
    setup_and_submit(structids, structchains)

  # Progress of a --run-local run. No-op when the jobs went to SLURM instead.
  print_local_status()

  # Score whatever results are currently available.
  score(structids)

if __name__ == "__main__":
  main()
