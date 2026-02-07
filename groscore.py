#!/usr/bin/env python3
#

import string, math, array
import os, sys, glob, re, time, argparse, shutil
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Input files for GroScore")
parser.add_argument('-s','--structparams', type=str, default="sp.gs", required=False, help="GroSscore strucutre parameter file")
parser.add_argument('-n','--numruns', type=int, default=5, required=False, help="Number of pull/push cycles to perform (default: 5)")
parser.add_argument('--cutout', dest='cutout', action='store_true', help="Enable interface cutout (default)")
parser.add_argument('--no-cutout', dest='cutout', action='store_false', help="Disable interface cutout, use full protein structure")
parser.add_argument('-ff','--forcefield', type=str, default="charmm36", choices=["gromos54a7", "charmm36", "amber19sb"], help="Force field to use (default: charmm36)")
parser.add_argument('--restart', action='store_true', help="Restart: resubmit jobs even if run.gs exists")
parser.set_defaults(cutout=True)
args=parser.parse_args()

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
          try:
            ids.append(tmp[0])
            vals.append(float(tmp[1]))
          except (IndexError, AttributeError):
            pass
          except ValueError:
            ids.append(tmp[0])
            vals.append("NaN")
  if len(ids) == len(vals):
    return ids, vals
  else:
    return False

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
    term2_sqrt = np.sqrt(
      (avgpulls - avgpushes)**2 / (varpulls * varpushes) +
      2.0 * diff_inv_var * np.log(varpushes / varpulls)
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

def calculate_scores(frenstruct, structids, numstructs, num_cycles, use_max_data=False):
  """Calculate scores for structures with at least num_cycles complete cycles.

  Args:
    frenstruct: Array of free energy values [numstructs x (numruns*2)]
    structids: List of structure IDs
    numstructs: Number of structures
    num_cycles: Number of cycles to use (or minimum if use_max_data=True)
    use_max_data: If True, use all available data; if False, use only first num_cycles

  Returns:
    fren: List of (struct_id, avg_score, ci95, num_cycles_used) tuples (only structures with >= num_cycles)
    frencgi: List of (struct_id, cgi_score, ci95, num_cycles_used) tuples (only structures with >= num_cycles)
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
          complete_cycles.append((pull_val, push_val))

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
    pulls = [cycle[0] for cycle in cycles_to_use]
    pushes = [cycle[1] for cycle in cycles_to_use]

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
    if len(pulls) > 2 and len(pushes) > 2:
      avgpulls = np.average(pulls)
      varpulls = np.var(pulls)
      avgpushes = np.average(pushes)
      varpushes = np.var(pushes)

      # Crooks Gaussian Intersection
      tmpcgi = (avgpulls/varpulls - avgpushes/varpushes + math.sqrt(1.0/(varpulls*varpushes) * (avgpulls-avgpushes)**2.0 + 2.0 * (1.0/varpulls - 1.0/varpushes) * math.log(varpushes/varpulls))) / (1.0/varpulls - 1.0/varpushes)
      tmpcgii = (avgpulls/varpulls - avgpushes/varpushes - math.sqrt(1.0/(varpulls*varpushes) * (avgpulls-avgpushes)**2.0 + 2.0 * (1.0/varpulls - 1.0/varpushes) * math.log(varpushes/varpulls))) / (1.0/varpulls - 1.0/varpushes)
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

    fren.append((structids[i], avg_score, avg_ci95, num_cycles_used))
    frencgi.append((structids[i], cgi_score, cgi_ci95, num_cycles_used))

  return fren, frencgi

#------------------------------------------------------

print("")
print("#################################")
print("#                               #")
print("#         GroScore 0.96         #")
print("#                               #")
print("#################################")
print("")

structids, structchains = readstructparams(args.structparams)
numstructs = len(structids)
if numstructs == 0:
  print("Error: No valid structures found in " + args.structparams)
  exit(1)
calcstruct = np.zeros(shape=(numstructs))
calcstruct[:] = 1.0
frenstruct = np.zeros(shape=(numstructs,args.numruns*2))
frenstruct[:,:] = "NaN"
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
    i = 0
    while i < numstructs:
      if os.path.exists("./%s"%structids[i]):
        f = open("./%s/run.gs"%structids[i], "w")
        cutout_flag = 1 if args.cutout else 0
        # MAXRUNS = numruns * 2 because each cycle has one pull (odd) and one push (even)
        f.write("%s %d %d %s\n"%(structchains[i],args.numruns*2,cutout_flag,args.forcefield))
        f.close()
        # Copy job.run to structure directory and make executable
        job_run_dst = "./%s/job.run"%structids[i]
        shutil.copy(job_run_src, job_run_dst)
        os.chmod(job_run_dst, 0o755)
      else:
        print("Structure %s: directory doesn't exist."%structids[i])
        f = open("results_0.gs", "a")
        f.write("%s NODIR\n"%structids[i])
        f.close()
      i += 1
    #SBATCH array
    f = open("array_submit.run", "w")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH -J gs_0.94\n")
    f.write("#SBATCH -n 4\n")
    f.write("#SBATCH --mem=4G\n")
    f.write("#SBATCH --array=0-%d\n"%(numstructs-1))
    f.write("\n")
    f.write("# Read structure ID from mapping file\n")
    f.write("STRUCT_ID=$(awk -v idx=\"$SLURM_ARRAY_TASK_ID\" '$1 == idx {print $2}' struct_map.gs)\n")
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
          l += 1
      i += 1
    np.savetxt("calcstruct.gs",calcstruct,delimiter="\t")
  # stage > 0
  elif os.path.isfile("results_%.0f.gs"%j):
    # read in this stage j
    try:
      results1, results2 = readtwocolumnsfloat("results_%.0f.gs"%(j))
    except TypeError:
      print("Error reading file results_" + str(j) + ".gs!")
      exit()
    k = 0
    while k < len(results1):
      try:
        l = 0
        while l < numstructs:
          if structids[l] == results1[k]:
            frenstruct[l,j-1] = results2[k]
          l += 1
      except (IndexError, AttributeError, ValueError):
        print("Error parsing file results_" + str(j) + ".gs at line " + str(k+1) + "!")
      k += 1
    np.savetxt("frenstruct.gs",frenstruct,delimiter="\t")

    # Check if we have a complete cycle (pull + push pair)
    # j is the result number (1-indexed), so j=2,4,6,... means a cycle just completed
    if j >= 2 and j % 2 == 0:
      current_cycle = j // 2

      sys.stdout.write("\rCalculating scores for cycle %d... "%current_cycle)
      sys.stdout.flush()

      # Write score files for each cycle threshold (1 to current_cycle)
      for cycle_threshold in range(1, current_cycle + 1):
        # Calculate scores using only first cycle_threshold cycles (for convergence tracking)
        fren, frencgi = calculate_scores(frenstruct, structids, numstructs, cycle_threshold, use_max_data=False)

        # Sort by score (NaN values go to end)
        fren.sort(key=lambda x: (math.isnan(x[1]), x[1]))
        frencgi.sort(key=lambda x: (math.isnan(x[1]), x[1]))

        # Write scores for this cycle threshold
        with open("scores_avg_c%d.gs"%cycle_threshold, "w") as f:
          f.write("# Scores using first %d cycle%s (structures with >=%d cycles)\n"%(cycle_threshold, "s" if cycle_threshold > 1 else "", cycle_threshold))
          f.write("# Structure_ID  Score  CI95  Cycles_Used\n")
          for struct_id, score, ci95, num_cycles in fren:
            if not np.isnan(score):
              f.write("%s\t%.1f\t%.1f\t%d\n"%(struct_id, score, ci95, num_cycles))
            else:
              f.write("%s\tnan\tnan\t%d\n"%(struct_id, num_cycles))

        with open("scores_cgi_c%d.gs"%cycle_threshold, "w") as f:
          f.write("# Scores using first %d cycle%s (structures with >=%d cycles)\n"%(cycle_threshold, "s" if cycle_threshold > 1 else "", cycle_threshold))
          f.write("# Structure_ID  Score  CI95  Cycles_Used\n")
          for struct_id, score, ci95, num_cycles in frencgi:
            if not np.isnan(score):
              f.write("%s\t%.1f\t%.1f\t%d\n"%(struct_id, score, ci95, num_cycles))
            else:
              f.write("%s\tnan\tnan\t%d\n"%(struct_id, num_cycles))

      # Update main score files (all structures using their maximum available data)
      # Include all structures with at least 1 complete cycle, each using all its available data
      fren_max, frencgi_max = calculate_scores(frenstruct, structids, numstructs, num_cycles=1, use_max_data=True)
      fren_max.sort(key=lambda x: (math.isnan(x[1]), x[1]))
      frencgi_max.sort(key=lambda x: (math.isnan(x[1]), x[1]))

      with open("scores_avg.gs", "w") as f:
        f.write("# Scores for all structures (each using maximum available data)\n")
        f.write("# Structure_ID  Score  CI95  Cycles_Used\n")
        for struct_id, score, ci95, num_cycles in fren_max:
          if not np.isnan(score):
            f.write("%s\t%.1f\t%.1f\t%d\n"%(struct_id, score, ci95, num_cycles))
          else:
            f.write("%s\tnan\tnan\t%d\n"%(struct_id, num_cycles))

      with open("scores_cgi.gs", "w") as f:
        f.write("# Scores for all structures (each using maximum available data)\n")
        f.write("# Structure_ID  Score  CI95  Cycles_Used\n")
        for struct_id, score, ci95, num_cycles in frencgi_max:
          if not np.isnan(score):
            f.write("%s\t%.1f\t%.1f\t%d\n"%(struct_id, score, ci95, num_cycles))
          else:
            f.write("%s\tnan\tnan\t%d\n"%(struct_id, num_cycles))

      print("Done!")

      # Check if all expected cycles are complete
      if j == args.numruns*2:
        print("")
        print("All %d requested cycles are complete!"%args.numruns)
        print("")
  j += 1
