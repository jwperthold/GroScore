#!/usr/bin/env python3
#

#################################
#                               #
#         GroScore 0.85         #
#                               #
#################################

import string, math, array
import os, sys, glob, re, time, argparse, shutil
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Input files for GroScore")
parser.add_argument('-s','--structparams', type=str, default="sp.gs", required=False, help="GroSscore strucutre parameter file")
parser.add_argument('-n','--numruns', type=int, default=10, required=True, help="Number of runs GroSscore should perform")
parser.add_argument('--cutout', dest='cutout', action='store_true', help="Enable interface cutout (default)")
parser.add_argument('--no-cutout', dest='cutout', action='store_false', help="Disable interface cutout, use full protein structure")
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

print("")
print("#################################")
print("#                               #")
print("#         GroScore 0.85         #")
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
frenstruct = np.zeros(shape=(numstructs,args.numruns))
frenstruct[:,:] = "NaN"
print("Reading input parameters finished.")
print("GroScore will calculate a binding free energy estimate for " + str(numstructs) + " structures.")
print("")

j = 0
while j <= args.numruns:
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
        f.write("%s %d %d\n"%(structchains[i],args.numruns,cutout_flag))
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
    f.write("#SBATCH -J gs_0.85\n")
    f.write("#SBATCH -n 2\n")
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
    if j == args.numruns:
      # check if last stage has finished
      i = 0
      ishould = int(np.sum(calcstruct))
      seen = set()
      if os.path.isfile("results_%.0f.gs"%(args.numruns)):
        with open("results_%.0f.gs"%(args.numruns), "r") as f:
          for line in f:
            if not line.strip().startswith("#"):
              tmp = line.split()
              try:
                struct_id = tmp[0]
                if struct_id in structids:
                  idx = structids.index(struct_id)
                  if calcstruct[idx] == 1 and struct_id not in seen:
                    seen.add(struct_id)
                    i += 1
              except (IndexError, AttributeError):
                pass
        if i == ishould:
          print("All simulations are done!")
        else:
          print("Simulations are not done yet!")
          k = 0
          while k < numstructs:
            if calcstruct[k] == 1 and structids[k] not in seen:
              print("Missing results for structure %s!"%structids[k])
            k += 1
      # now do ranking
      fren = []  # list of (struct_id, score)
      frencgi = []  # list of (struct_id, score)
      i = 0
      while i < numstructs:
        k = 0
        pulls = []
        pushes = []
        while k < frenstruct.shape[1]:
          # pulling
          if k%2 == 0.0 and np.isnan(frenstruct[i,k]) == False:
            # the dW direction in the file is -1*unbinding=binding
            pulls.append(frenstruct[i,k])
          # pushing
          if k%2 != 0.0 and np.isnan(frenstruct[i,k]) == False:
            # the dW direction in the file is -1*-1*binding=binding
            pushes.append(frenstruct[i,k])
          k += 1
        # fren
        avgpulls = 0
        varpulls = 0
        avgpushes = 0
        varpushes = 0
        avg_score = float('nan')
        cgi_score = float('nan')
        if len(pulls) > 0 and len(pushes) > 0:
          # pulls
          avgpulls = np.average(pulls)
          # pushes
          avgpushes = np.average(pushes)
          # combine fwd + rev with averages
          avg_score = (avgpulls + avgpushes) / 2.0
        if len(pulls) > 2 and len(pushes) > 2:
          # pulls
          varpulls = np.var(pulls)
          # pushes
          varpushes = np.var(pushes)
          # combine fwd + rev with gaussian intersection
          tmpcgi = (avgpulls/varpulls - avgpushes/varpushes + math.sqrt(1.0/(varpulls*varpushes) * (avgpulls-avgpushes)**2.0 + 2.0 * (1.0/varpulls - 1.0/varpushes) * math.log(varpushes/varpulls))) / (1.0/varpulls - 1.0/varpushes)
          tmpcgii = (avgpulls/varpulls - avgpushes/varpushes - math.sqrt(1.0/(varpulls*varpushes) * (avgpulls-avgpushes)**2.0 + 2.0 * (1.0/varpulls - 1.0/varpushes) * math.log(varpushes/varpulls))) / (1.0/varpulls - 1.0/varpushes)
          disti = math.fabs((avgpulls+avgpushes)/2.0 - tmpcgi)
          distii = math.fabs((avgpulls+avgpushes)/2.0 - tmpcgii)
          if disti > distii:
            cgi_score = tmpcgii
          else:
            cgi_score = tmpcgi
        fren.append((structids[i], avg_score))
        frencgi.append((structids[i], cgi_score))
        sys.stdout.write("\rCalculating results - " + str(round((float(i)+1.0)/numstructs*100.0,1)) + "%")
        sys.stdout.flush()
        i += 1
      print("")
      print("")
      # Sort by score (NaN values go to end)
      fren.sort(key=lambda x: (math.isnan(x[1]), x[1]))
      frencgi.sort(key=lambda x: (math.isnan(x[1]), x[1]))
      # Write output files
      with open("scores_avg.gs", "w") as f:
        for struct_id, score in fren:
          f.write("%s\t%s\n"%(struct_id, score))
      with open("scores_cgi.gs", "w") as f:
        for struct_id, score in frencgi:
          f.write("%s\t%s\n"%(struct_id, score))
  j += 1
