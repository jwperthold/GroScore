#!/usr/bin/env python3
#

#################################
#                               #
#         GroScore 0.91         #
#                               #
#################################

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
  """Calculate bootstrap standard error for a score.

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
  bootstrap_scores = []

  for _ in range(n_bootstrap):
    # Resample with replacement
    boot_pulls = np.random.choice(pulls_arr, size=n_pulls, replace=True)
    boot_pushes = np.random.choice(pushes_arr, size=n_pushes, replace=True)

    if method == 'avg':
      # Simple average method
      boot_score = (np.mean(boot_pulls) + np.mean(boot_pushes)) / 2.0
      bootstrap_scores.append(boot_score)

    elif method == 'cgi' and len(pulls) > 2 and len(pushes) > 2:
      # Crooks Gaussian Intersection
      try:
        avgpulls = np.mean(boot_pulls)
        varpulls = np.var(boot_pulls)
        avgpushes = np.mean(boot_pushes)
        varpushes = np.var(boot_pushes)

        if varpulls > 0 and varpushes > 0 and varpulls != varpushes:
          tmpcgi = (avgpulls/varpulls - avgpushes/varpushes + math.sqrt(1.0/(varpulls*varpushes) * (avgpulls-avgpushes)**2.0 + 2.0 * (1.0/varpulls - 1.0/varpushes) * math.log(varpushes/varpulls))) / (1.0/varpulls - 1.0/varpushes)
          tmpcgii = (avgpulls/varpulls - avgpushes/varpushes - math.sqrt(1.0/(varpulls*varpushes) * (avgpulls-avgpushes)**2.0 + 2.0 * (1.0/varpulls - 1.0/varpushes) * math.log(varpushes/varpulls))) / (1.0/varpulls - 1.0/varpushes)
          disti = math.fabs((avgpulls+avgpushes)/2.0 - tmpcgi)
          distii = math.fabs((avgpulls+avgpushes)/2.0 - tmpcgii)

          if disti > distii:
            boot_score = tmpcgii
          else:
            boot_score = tmpcgi
          bootstrap_scores.append(boot_score)
      except (ValueError, ZeroDivisionError):
        # Skip this bootstrap sample if calculation fails
        pass

  if len(bootstrap_scores) > 0:
    return np.std(bootstrap_scores)
  else:
    return float('nan')

#------------------------------------------------------

def calculate_scores(frenstruct, structids, numstructs, num_cycles):
  """Calculate scores using data from the first num_cycles cycles.

  Args:
    frenstruct: Array of free energy values [numstructs x (numruns*2)]
    structids: List of structure IDs
    numstructs: Number of structures
    num_cycles: Number of complete cycles to include (each cycle = pull + push)

  Returns:
    fren: List of (struct_id, avg_score, ci95) tuples
    frencgi: List of (struct_id, cgi_score, ci95) tuples
  """
  fren = []
  frencgi = []
  max_idx = num_cycles * 2  # Each cycle has 2 results (pull + push)

  for i in range(numstructs):
    pulls = []
    pushes = []

    for k in range(min(max_idx, frenstruct.shape[1])):
      # pulling (odd result numbers = even indices: 0,2,4,...)
      if k % 2 == 0 and not np.isnan(frenstruct[i,k]):
        pulls.append(frenstruct[i,k])
      # pushing (even result numbers = odd indices: 1,3,5,...)
      elif k % 2 != 0 and not np.isnan(frenstruct[i,k]):
        pushes.append(frenstruct[i,k])

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
      avg_stderr = bootstrap_score(pulls, pushes, n_bootstrap=1000, method='avg')
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
      cgi_stderr = bootstrap_score(pulls, pushes, n_bootstrap=1000, method='cgi')
      if not np.isnan(cgi_stderr):
        cgi_ci95 = 1.96 * cgi_stderr

    fren.append((structids[i], avg_score, avg_ci95))
    frencgi.append((structids[i], cgi_score, cgi_ci95))

  return fren, frencgi

#------------------------------------------------------

print("")
print("#################################")
print("#                               #")
print("#         GroScore 0.91         #")
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
print("")

j = 0
while j <= args.numruns*2:
  # setup simulations
  if j == 0 and not os.path.isfile("results_%.0f.gs"%j):
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
    f.write("#SBATCH -J gs_0.91\n")
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

      # Calculate cumulative scores using all cycles up to current_cycle
      sys.stdout.write("\rCalculating scores for cycle %d... "%current_cycle)
      sys.stdout.flush()

      fren, frencgi = calculate_scores(frenstruct, structids, numstructs, current_cycle)

      # Sort by score (NaN values go to end)
      fren.sort(key=lambda x: (math.isnan(x[1]), x[1]))
      frencgi.sort(key=lambda x: (math.isnan(x[1]), x[1]))

      # Write cumulative scores for this cycle
      cycle_label = "cycle 1" if current_cycle == 1 else "cycles 1-%d"%current_cycle

      with open("scores_avg_c%d.gs"%current_cycle, "w") as f:
        f.write("# Cumulative scores using %s\n"%cycle_label)
        f.write("# Structure_ID  Score  CI95\n")
        for struct_id, score, ci95 in fren:
          if not np.isnan(score):
            f.write("%s\t%.1f\t%.1f\n"%(struct_id, score, ci95))
          else:
            f.write("%s\tnan\tnan\n"%struct_id)

      with open("scores_cgi_c%d.gs"%current_cycle, "w") as f:
        f.write("# Cumulative scores using %s\n"%cycle_label)
        f.write("# Structure_ID  Score  CI95\n")
        for struct_id, score, ci95 in frencgi:
          if not np.isnan(score):
            f.write("%s\t%.1f\t%.1f\n"%(struct_id, score, ci95))
          else:
            f.write("%s\tnan\tnan\n"%struct_id)

      # Also update main score files (always reflect latest complete data)
      with open("scores_avg.gs", "w") as f:
        f.write("# Cumulative scores using %s\n"%cycle_label)
        f.write("# Structure_ID  Score  CI95\n")
        for struct_id, score, ci95 in fren:
          if not np.isnan(score):
            f.write("%s\t%.1f\t%.1f\n"%(struct_id, score, ci95))
          else:
            f.write("%s\tnan\tnan\n"%struct_id)

      with open("scores_cgi.gs", "w") as f:
        f.write("# Cumulative scores using %s\n"%cycle_label)
        f.write("# Structure_ID  Score  CI95\n")
        for struct_id, score, ci95 in frencgi:
          if not np.isnan(score):
            f.write("%s\t%.1f\t%.1f\n"%(struct_id, score, ci95))
          else:
            f.write("%s\tnan\tnan\n"%struct_id)

      print("Done!")

      # Check if all expected cycles are complete
      if j == args.numruns*2:
        print("")
        print("All %d requested cycles are complete!"%args.numruns)
        print("")
  j += 1
