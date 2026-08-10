#!/usr/bin/env python3
#

import string, math, array
import os, sys, glob, re, time, argparse, shutil
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Input files for GroScore")
parser.add_argument('-n','--numruns', type=int, default=5, required=False, help="Number of pull/push cycles to perform (default: 5)")
parser.add_argument('-s','--structparams', type=str, default="sp.gs", required=False, help="GroSscore strucutre parameter file")
parser.add_argument('-ff','--forcefield', type=str, default="amber19sb_opc3", choices=["gromos54a8", "charmm36", "amber19sb_opc", "amber19sb_opc3"], help="Force field to use (default: amber19sb_opc3)")
parser.add_argument('--no-cutout', dest='cutout', action='store_false', help="Disable interface cutout, use full protein structure")
parser.add_argument('--no-ligand-param', dest='ligand_param', action='store_false', help="Disable small molecule parametrization with OpenFF (AMBER forcefields)")
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
parser.add_argument('--restart', action='store_true', help="Restart: resubmit jobs even if run.gs exists")
parser.add_argument('--inject-job-run', action='store_true', help="Inject fresh job.run into archived (.tar.gz) structures")
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

def readtwocolumns(filepath):
  ids = []
  vals = []
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          try:
            ids.append(tmp[0])
            vals.append(tmp[1])
          except (IndexError, AttributeError):
            pass
  if len(ids) == len(vals):
    return ids, vals
  else:
    return False

#------------------------------------------------------

def readtwocolumnsfloat(filepath):
  ids = []
  vals = []
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          if len(tmp) < 2:
            continue
          try:
            ids.append(tmp[0])
            vals.append(float(tmp[1]))
          except (ValueError, TypeError):
            ids.append(tmp[0])
            vals.append(float('nan'))
  if len(ids) == len(vals):
    return ids, vals
  else:
    return False

#------------------------------------------------------

def readcolumnfloat(filepath, col):
  """Read (id, float(column `col`)) pairs from a results file.

  Used for the re-bound RMSD, which job.run appends as a third column to the
  push results files. Rows written before the column existed, or whose value is
  the literal NaN, yield nan rather than being dropped.
  """
  ids = []
  vals = []
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if line.strip().startswith("#"):
          continue
        tmp = line.split()
        if len(tmp) < 2:
          continue
        ids.append(tmp[0])
        try:
          vals.append(float(tmp[col]))
        except (IndexError, ValueError, TypeError):
          vals.append(float('nan'))
  return ids, vals

#------------------------------------------------------

def countlines(filepath):
  i = 0
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          i += 1
  return i

#------------------------------------------------------

def bootstrap_score(pulls, pushes, n_bootstrap=1000, method='avg'):
  """Calculate bootstrap standard error for a score (vectorized).

  Args:
    pulls: List of pull free energy values
    pushes: List of push free energy values
    n_bootstrap: Number of bootstrap iterations (default: 1000)
    method: Scoring method ('avg' or 'cgi')

  Returns:
    Standard error of the score
  """
  if len(pulls) == 0 or len(pushes) == 0:
    return float('nan')

  pulls_arr = np.array(pulls)
  pushes_arr = np.array(pushes)
  n_pulls = len(pulls)
  n_pushes = len(pushes)

  # Generate all bootstrap samples at once (n_bootstrap x n_samples)
  boot_pulls_idx = np.random.randint(0, n_pulls, size=(n_bootstrap, n_pulls))
  boot_pushes_idx = np.random.randint(0, n_pushes, size=(n_bootstrap, n_pushes))
  boot_pulls_all = pulls_arr[boot_pulls_idx]
  boot_pushes_all = pushes_arr[boot_pushes_idx]

  if method == 'avg':
    # Vectorized average method - compute all bootstrap scores at once
    avgpulls_all = np.mean(boot_pulls_all, axis=1)
    avgpushes_all = np.mean(boot_pushes_all, axis=1)
    bootstrap_scores = (avgpulls_all + avgpushes_all) / 2.0
    return np.std(bootstrap_scores)

  elif method == 'cgi' and len(pulls) > 2 and len(pushes) > 2:
    # Vectorized CGI method - compute statistics for all bootstrap samples
    avgpulls_all = np.mean(boot_pulls_all, axis=1)
    varpulls_all = np.var(boot_pulls_all, axis=1)
    avgpushes_all = np.mean(boot_pushes_all, axis=1)
    varpushes_all = np.var(boot_pushes_all, axis=1)

    # Filter valid samples (positive variances, different variances)
    valid_mask = (varpulls_all > 0) & (varpushes_all > 0) & (varpulls_all != varpushes_all)

    if not np.any(valid_mask):
      return float('nan')

    # Extract valid samples
    avgpulls = avgpulls_all[valid_mask]
    varpulls = varpulls_all[valid_mask]
    avgpushes = avgpushes_all[valid_mask]
    varpushes = varpushes_all[valid_mask]

    # Vectorized CGI calculation
    inv_varpulls = 1.0 / varpulls
    inv_varpushes = 1.0 / varpushes
    diff_inv_var = inv_varpulls - inv_varpushes

    term1 = avgpulls * inv_varpulls - avgpushes * inv_varpushes
    # Discriminant of Goette & Grubmueller eq. (12); see calculate_scores. The
    # log is of the variance ratio here, which already absorbs the paper's
    # factor 2 on ln(sigma2/sigma1).
    term2_sqrt = np.sqrt(
      (avgpulls - avgpushes)**2 / (varpulls * varpushes) +
      diff_inv_var * np.log(varpushes / varpulls)
    )

    tmpcgi = (term1 + term2_sqrt) / diff_inv_var
    tmpcgii = (term1 - term2_sqrt) / diff_inv_var

    # Choose solution closest to average
    avg_mid = (avgpulls + avgpushes) / 2.0
    disti = np.abs(avg_mid - tmpcgi)
    distii = np.abs(avg_mid - tmpcgii)

    bootstrap_scores = np.where(disti > distii, tmpcgii, tmpcgi)
    return np.std(bootstrap_scores)

  return float('nan')

#------------------------------------------------------

def calculate_scores(frenstruct, structids, numstructs, num_cycles, use_max_data=False,
                     rmsdstruct=None):
  """Calculate scores for structures with at least num_cycles complete cycles.

  Args:
    frenstruct: Array of free energy values [numstructs x (numruns*2)]
    structids: List of structure IDs
    numstructs: Number of structures
    num_cycles: Number of cycles to use (or minimum if use_max_data=True)
    use_max_data: If True, use all available data; if False, use only first num_cycles
    rmsdstruct: Optional array of re-bound backbone RMSDs [numstructs x numruns],
      summarised over exactly the cycles that entered the score

  Returns:
    fren: List of (struct_id, avg_score, ci95, num_cycles_used, rmsd_mean, rmsd_max) tuples
      (only structures with >= num_cycles)
    frencgi: Same, with the CGI score
  """
  fren = []
  frencgi = []
  max_idx = num_cycles * 2 if not use_max_data else frenstruct.shape[1]

  for i in range(numstructs):
    # Collect complete cycles (matching pull-push pairs)
    complete_cycles = []
    max_cycles = frenstruct.shape[1] // 2

    for cycle_idx in range(max_cycles):
      pull_idx = cycle_idx * 2
      push_idx = cycle_idx * 2 + 1

      if pull_idx < frenstruct.shape[1] and push_idx < frenstruct.shape[1]:
        pull_val = frenstruct[i, pull_idx]
        push_val = frenstruct[i, push_idx]

        # Only include if BOTH pull and push exist for this cycle
        if not np.isnan(pull_val) and not np.isnan(push_val):
          complete_cycles.append((cycle_idx, pull_val, push_val))

    num_complete_cycles = len(complete_cycles)

    # Skip this structure if it doesn't have enough complete cycles
    if num_complete_cycles < num_cycles:
      continue

    # For convergence tracking, use first num_cycles complete cycles
    # For max data, use all complete cycles
    if not use_max_data and num_complete_cycles > num_cycles:
      cycles_to_use = complete_cycles[:num_cycles]
      num_cycles_used = num_cycles
    else:
      cycles_to_use = complete_cycles
      num_cycles_used = num_complete_cycles

    # Extract pulls and pushes from selected cycles
    pulls = [cycle[1] for cycle in cycles_to_use]
    pushes = [cycle[2] for cycle in cycles_to_use]

    # Re-bound RMSD over exactly those cycles (see rebound_rmsd.py)
    rmsd_mean, rmsd_max = rmsd_stats(rmsdstruct, i, [c[0] for c in cycles_to_use])

    avg_score = float('nan')
    avg_ci95 = float('nan')
    cgi_score = float('nan')
    cgi_ci95 = float('nan')

    # Calculate average score if we have data
    if len(pulls) > 0 and len(pushes) > 0:
      avgpulls = np.average(pulls)
      avgpushes = np.average(pushes)
      avg_score = (avgpulls + avgpushes) / 2.0

      # Bootstrap error estimation for average method
      avg_stderr = bootstrap_score(pulls, pushes, n_bootstrap=50000, method='avg')
      if not np.isnan(avg_stderr):
        avg_ci95 = 1.96 * avg_stderr

    # Calculate CGI score if we have enough data
    if len(pulls) > 19 and len(pushes) > 19:
      avgpulls = np.average(pulls)
      varpulls = np.var(pulls)
      avgpushes = np.average(pushes)
      varpushes = np.var(pushes)

      # Crooks Gaussian Intersection: the work at which the two fitted Gaussians
      # cross. Goette & Grubmueller, J Comput Chem 30, 447-456 (2009), eq. (12):
      #
      #   dF = [ Wf/s1^2 - (-Wr)/s2^2 +- sqrt( (Wf+Wr)^2/(s1^2 s2^2)
      #                                        + 2 (1/s1^2 - 1/s2^2) ln(s2/s1) ) ]
      #        / (1/s1^2 - 1/s2^2)
      #
      # NOTE the paper's logarithm is of the STANDARD DEVIATION ratio. Because
      # 2*ln(s2/s1) = ln(v2/v1), the factor 2 belongs to the sigma form only.
      # Until 2026-08-10 this code kept the 2 while taking the log of the
      # VARIANCE ratio, applying it twice; the reported crossing then drifted off
      # the true intersection whenever the two work distributions differed in
      # width (~2.5 kJ/mol at sigma ratio 2, ~6.8 at ratio 5). See tests/test_cgi.py.
      tmpcgi = (avgpulls/varpulls - avgpushes/varpushes + math.sqrt(1.0/(varpulls*varpushes) * (avgpulls-avgpushes)**2.0 + (1.0/varpulls - 1.0/varpushes) * math.log(varpushes/varpulls))) / (1.0/varpulls - 1.0/varpushes)
      tmpcgii = (avgpulls/varpulls - avgpushes/varpushes - math.sqrt(1.0/(varpulls*varpushes) * (avgpulls-avgpushes)**2.0 + (1.0/varpulls - 1.0/varpushes) * math.log(varpushes/varpulls))) / (1.0/varpulls - 1.0/varpushes)
      disti = math.fabs((avgpulls+avgpushes)/2.0 - tmpcgi)
      distii = math.fabs((avgpulls+avgpushes)/2.0 - tmpcgii)

      if disti > distii:
        cgi_score = tmpcgii
      else:
        cgi_score = tmpcgi

      # Bootstrap error estimation for CGI method
      cgi_stderr = bootstrap_score(pulls, pushes, n_bootstrap=50000, method='cgi')
      if not np.isnan(cgi_stderr):
        cgi_ci95 = 1.96 * cgi_stderr

    fren.append((structids[i], avg_score, avg_ci95, num_cycles_used, rmsd_mean, rmsd_max))
    frencgi.append((structids[i], cgi_score, cgi_ci95, num_cycles_used, rmsd_mean, rmsd_max))

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

  scored: {struct_id: (score, ci95, ncycles, rmsd_mean, rmsd_max)} for the
  structures that produced data; every other structure is written with its
  stage-0 status so the file always lists all of sp.gs.
  """
  with open(path, "w") as f:
    f.write(header)
    f.write("# Structure_ID  Score  CI95  Cycles_Used  RMSD_mean_A  RMSD_max_A  RMSD_flag\n")
    for struct_id in structids:
      if struct_id in scored:
        score, ci95, nc, rmean, rmax = scored[struct_id]
        score_cols = ("nan\tnan" if np.isnan(score) else "%.1f\t%.1f"%(score, ci95))
        f.write("%s\t%s\t%d\t%s\t%s\t%s\n"%(struct_id, score_cols, nc,
                                            rmsd_cell(rmean), rmsd_cell(rmax),
                                            rmsd_flag(rmax, threshold)))
      else:
        status = struct_status.get(struct_id, "nan")
        f.write("%s\t%s\t%s\t0\tnan\tnan\tnan\n"%(struct_id, status, status))

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
calcstruct = np.zeros(shape=(numstructs))
calcstruct[:] = 1.0
struct_status = {}  # struct_id -> failure reason (BROKEN, ENTANGLED, NODIR) or None for OK
frenstruct = np.zeros(shape=(numstructs,args.numruns*2))
frenstruct[:,:] = "NaN"
# Re-bound backbone RMSD per structure and cycle (Angstrom), third column of the
# push results files. Purely diagnostic: it never changes a score, it only flags
# cycles in which the partners failed to return to the bound pose.
rmsdstruct = np.zeros(shape=(numstructs,args.numruns))
rmsdstruct[:,:] = "NaN"
print("Reading input parameters finished.")
print("GroScore will calculate a binding free energy estimate for " + str(numstructs) + " structures.")
print("Each structure will undergo " + str(args.numruns) + " independent equilibration cycles (each cycle = 1 pull + 1 push).")
if args.restart:
  print("RESTART MODE: Will resubmit jobs even if run.gs exists.")
print("")

j = 0
while j <= args.numruns*2:
  # setup simulations
  if j == 0 and (args.restart or not os.path.isfile("results_%.0f.gs"%j)):
    if not os.path.isfile("results_%.0f.gs"%j):
      f = open("results_%.0f.gs"%j, "a")
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
    live = []  # structures that exist as a directory or an archive, i.e. are runnable
    i = 0
    while i < numstructs:
      if os.path.exists("./%s"%structids[i]):
        live.append(structids[i])
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
        live.append(structids[i])
        if args.inject_job_run:
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
          print("Skipping %s.tar.gz (archived, use --inject-job-run to update)."%structids[i])
      else:
        print("Structure %s: directory doesn't exist."%structids[i])
        f = open("results_0.gs", "a")
        f.write("%s NODIR\n"%structids[i])
        f.close()
      i += 1
    if args.run_local:
      # Single workstation, no scheduler: one job per structure, executed by
      # utils/local_runner.py in a bounded pool with one GPU pinned per slot.
      jobs = [{"name": sid, "dir": sid, "argv": ["./job.run"],
               "archive": "%s.tar.gz"%sid, "log": "job_local.out"} for sid in live]
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
      f.write("#SBATCH --array=0-%d\n"%(numstructs-1))
      f.write("\n")
      f.write("# Read structure ID from mapping file\n")
      f.write("STRUCT_ID=$(awk -v idx=\"$SLURM_ARRAY_TASK_ID\" '$1 == idx {print $2}' struct_map.gs)\n")
      f.write("# Extract archived structure if needed (for restarts)\n")
      f.write("if [[ ! -d \"$STRUCT_ID\" && -f \"${STRUCT_ID}.tar.gz\" ]]; then\n")
      f.write("  tar -xzf \"${STRUCT_ID}.tar.gz\"\n")
      f.write("  rm \"${STRUCT_ID}.tar.gz\"\n")
      f.write("fi\n")
      f.write("cd $STRUCT_ID\n")
      f.write("./job.run\n")
      f.close()
      os.system("sbatch array_submit.run")
      print("Submitted all simulation jobs.")
      print("")
  # stage 0
  if j == 0 and os.path.isfile("results_%.0f.gs"%j):
    results1, results2 = readtwocolumns("results_%.0f.gs"%(j))
    i = 0
    while i < len(results1):
      if results2[i] == "OK":
        l = 0
        while l < numstructs:
          if structids[l] == results1[i]:
            calcstruct[l] = 1
          l += 1
      else:
        l = 0
        while l < numstructs:
          if structids[l] == results1[i]:
            calcstruct[l] = 0
            struct_status[results1[i]] = results2[i]
          l += 1
      i += 1
    np.savetxt("calcstruct.gs",calcstruct,delimiter="\t")
  # stage > 0
  elif os.path.isfile("results_%.0f.gs"%j):
    # read in this stage j
    results1, results2 = readtwocolumnsfloat("results_%.0f.gs"%(j))
    k = 0
    while k < len(results1):
      l = 0
      while l < numstructs:
        if structids[l] == results1[k]:
          try:
            if not np.isnan(results2[k]):
              frenstruct[l,j-1] = results2[k]
          except (IndexError, AttributeError, ValueError, TypeError):
            print("Error parsing file results_" + str(j) + ".gs at line " + str(k+1) + "!")
        l += 1
      k += 1
    np.savetxt("frenstruct.gs",frenstruct,delimiter="\t")

    # Re-bound RMSD of this cycle (push files only, third column)
    if j % 2 == 0:
      rmsd1, rmsd2 = readcolumnfloat("results_%.0f.gs"%(j), 2)
      for k in range(len(rmsd1)):
        for l in range(numstructs):
          if structids[l] == rmsd1[k]:
            rmsdstruct[l,j//2-1] = rmsd2[k]
      np.savetxt("rmsdstruct.gs",rmsdstruct,delimiter="\t")

    # Check if we have a complete cycle (pull + push pair)
    # j is the result number (1-indexed), so j=2,4,6,... means a cycle just completed
    if j >= 2 and j % 2 == 0:
      current_cycle = j // 2

      sys.stdout.write("\rCalculating scores for cycle %d... "%current_cycle)
      sys.stdout.flush()

      # Write score files for each cycle threshold (1 to current_cycle)
      for cycle_threshold in range(1, current_cycle + 1):
        # Calculate scores using only first cycle_threshold cycles (for convergence tracking)
        fren, frencgi = calculate_scores(frenstruct, structids, numstructs, cycle_threshold,
                                         use_max_data=False, rmsdstruct=rmsdstruct)

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
                                               use_max_data=True, rmsdstruct=rmsdstruct)

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
      if j == args.numruns*2:
        print("")
        print("All %d requested cycles are complete!"%args.numruns)
        print("")
  j += 1

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
