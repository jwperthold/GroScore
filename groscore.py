#!/usr/bin/env python3
#

#################################
#                               #
#         GroScore 0.83         #
#                               #
#################################

import string, math, array
import os, sys, glob, re, time, argparse
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Input files for GroScore")
parser.add_argument('-s','--structparams', type=str, default="sp.gs", required=False, help="GroSscore strucutre parameter file")
parser.add_argument('-n','--numruns', type=int, default=10, required=True, help="Number of runs GroSscore should perform")
args=parser.parse_args()

#------------------------------------------------------

def readstructparams(filepath):
  num = []
  chains = []
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          try:
            num.append(int(tmp[0]))
            chains.append(tmp[1])
          except (ValueError, IndexError, AttributeError):
            i = 0
  if len(num) == len(chains):
    return num, chains
  else:
    return False

#------------------------------------------------------

def readtwocolumns(filepath):
  num = []
  num2 = []
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          try:
            num.append(int(tmp[0]))
            num2.append(tmp[1])
          except (ValueError, IndexError, AttributeError):
            i = 0
  if len(num) == len(num2):
    return num, num2
  else:
    return False

#------------------------------------------------------

def readtwocolumnsfloat(filepath):
  num = []
  num2 = []
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          try:
            num.append(int(tmp[0]))
            num2.append(float(tmp[1]))
          except (IndexError, AttributeError):
            i = 0
          except ValueError:
            num.append(int(tmp[0]))
            num2.append("NaN")
  if len(num) == len(num2):
    return num, num2
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
print("#         GroScore 0.83         #")
print("#                               #")
print("#################################")
print("")

structnums, structchains = readstructparams(args.structparams)
numstructs = len(structnums)
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
    i = 0
    while i < numstructs:
      if os.path.exists("./%d"%structnums[i]):
        f = open("./%d/run.gs"%structnums[i], "a")
        f.write("%s %d\n"%(structchains[i],args.numruns))
        f.close()
      else:
        print("Structure %d: directory doesn't exist."%structnums[i])
        f = open("results_0.gs", "a")
        f.write("%d NODIR\n"%structnums[i])
        f.close()
      i += 1
    #SBATCH array
    f = open("array_submit.run", "a")
    f.write("#!/bin/bash\n#\n#SBATCH -J gs_0.83\n#SBATCH -n 4\n#SBATCH --partition NGN\n#SBATCH --gres mps:33\n#SBATCH --mail-type=FAIL\n#SBATCH --mail-user=jan@ackergarten.at\n#SBATCH --array=%d-%d\n./job.run"%(structnums[0],structnums[-1]))
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
          if structnums[l] == results1[i]:
            calcstruct[l] = 1
          l += 1
      else:
        l = 0
        while l < numstructs:
          if structnums[l] == results1[i]:
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
          if structnums[l] == results1[k]:
            frenstruct[l,j-1] = results2[k]
          l += 1
      except (IndexError, AttributeError, ValueError):
        print("Error parsing file results_" + str(j) + ".gs at line " + str(k+1) + "!")
      k += 1
    np.savetxt("frenstruct.gs",frenstruct,delimiter="\t")
    if j == args.numruns:
      # check if last stage has finsihed
      i = 0
      ishould = np.sum(calcstruct)
      seen = np.zeros(numstructs)
      if os.path.isfile("results_%.0f.gs"%(args.numruns)):
        with open("results_%.0f.gs"%(args.numruns), "r") as f:
          for line in f:
            if not line.strip().startswith("#"):
              tmp = line.split()
            try:
              test = int(tmp[0])
              if calcstruct[test-1] == 1 and seen[test-1] == 0.0:
                seen[test-1] = 1.0
                i += 1
            except (ValueError, IndexError, AttributeError):
              xyz = 0
        if i == ishould:
          print("All simulations are done!")
        else:
          print("Simulations are not done yet!")
          k = 0
          while k < seen.shape[0]:
            if calcstruct[k] == 1 and seen[k] == 0:
              print("Missing results for structure " +  str(k+1) + "!")
            k += 1
      # now do ranking
      fren = np.zeros(shape=(numstructs,2))
      frencgi = np.zeros(shape=(numstructs,2))
      fren[:,0] = structnums
      frencgi[:,0] = structnums
      fren[:,1] = "NaN"
      frencgi[:,1] = "NaN"
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
        if len(pulls) > 0 and len(pushes) > 0:
          # pulls
          avgpulls = np.average(pulls)
          # pushes
          avgpushes = np.average(pushes)
          # combine fwd + rev with averages
          fren[i,1] = (avgpulls + avgpushes) / 2.0
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
            frencgi[i,1] = tmpcgii
          else:
            frencgi[i,1] = tmpcgi
        sys.stdout.write("\rCalculating results - " + str(round((float(i)+1.0)/numstructs*100.0,1)) + "%")
        sys.stdout.flush()
        i += 1
      print("")
      print("")
      fren = fren[fren[:,1].argsort()]
      frencgi = frencgi[frencgi[:,1].argsort()]
      np.savetxt("scores_avg.gs",fren,delimiter="\t")
      np.savetxt("scores_cgi.gs",frencgi,delimiter="\t")
  j += 1
