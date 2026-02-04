#!/usr/bin/env python3
#

#################################
#                               #
#         GroScore 0.82         #
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
  start = []
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          try:
            num.append(int(tmp[0]))
            start.append(int(tmp[1]))
          except (ValueError, IndexError, AttributeError):
            i = 0
  if len(num) == len(start):
    params = np.zeros(shape=(len(num),3))
    i = 0
    while i < len(num):
      params[i,0] = num[i]
      params[i,1] = start[i]
      i += 1
    return params
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
print("#         GroScore 0.82         #")
print("#                               #")
print("#################################")
print("")

structparams = readstructparams(args.structparams)
calcstruct = np.zeros(shape=(structparams.shape[0]))
calcstruct[:] = 1.0
frenstruct = np.zeros(shape=(structparams.shape[0],args.numruns))
frenstruct[:,:] = "NaN"
print("Reading input parameters finished.")
print("GroScore will calculate a binding free energy estimate for " + str(structparams.shape[0]) + " structures.")
print("")

j = 0
while j <= args.numruns:
  # setup simulations
  if j == 0 and not os.path.isfile("results_%.0f.gs"%j):
    f = open("results_%.0f.gs"%j, "a")
    f.write("# Results for simulation fitness:\n")
    f.close()
    i = 0
    while i < structparams.shape[0]:
      if os.path.exists("./%.0f"%structparams[i,0]):
        f = open("./%.0f/run.gs"%structparams[i,0], "a")
        f.write("%.0f %.0f\n"%(structparams[i,1],args.numruns))
        f.close()
      else:
        print("Structure %.0f: directory doesn't exist."%structparams[i,0])
        f = open("results_0.gs", "a")
        f.write("%.0f NODIR\n"%structparams[i,0])
        f.close()
      i += 1
    #SBATCH array
    f = open("array_submit.run", "a")
    f.write("#!/bin/bash\n#\n#SBATCH -J gs_0.82\n#SBATCH -n 4\n#SBATCH --partition NGN\n#SBATCH --gres mps:33\n#SBATCH --mail-type=FAIL\n#SBATCH --mail-user=jan@ackergarten.at\n#SBATCH --array=%.0f-%.0f\n./job.run"%(structparams[0,0],structparams[structparams.shape[0]-1,0]))
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
        while l < structparams.shape[0]:
          if structparams[l,0] == results1[i]:
            calcstruct[l] = 1
          l += 1
      else:
        l = 0
        while l < structparams.shape[0]:
          if structparams[l,0] == results1[i]:
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
        while l < structparams.shape[0]:
          if structparams[l,0] == results1[k]:
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
      seen = np.zeros(structparams.shape[0])
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
      fren = np.zeros(shape=(structparams.shape[0],2))
      frencgi = np.zeros(shape=(structparams.shape[0],2))
      fren[:,0] = structparams[:,0]
      frencgi[:,0] = structparams[:,0]
      fren[:,1] = "NaN"
      frencgi[:,1] = "NaN"
      i = 0
      while i < structparams.shape[0]:
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
        sys.stdout.write("\rCalculating results - " + str(round((float(i)+1.0)/structparams.shape[0]*100.0,1)) + "%")
        sys.stdout.flush()
        i += 1
      print("")
      print("")
      fren = fren[fren[:,1].argsort()]
      frencgi = frencgi[frencgi[:,1].argsort()]
      np.savetxt("scores_avg.gs",fren,delimiter="\t")
      np.savetxt("scores_cgi.gs",frencgi,delimiter="\t")
  j += 1
