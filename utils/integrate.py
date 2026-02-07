#!/usr/bin/env python3
#

import string
import math
import array
import os, sys, glob, re, time, argparse
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Do force integration.")
parser.add_argument('-f','--file', type=str, default="bind_pullf.xvg", required=True, help="Forces.")
parser.add_argument('-nr','--numpertres', type=int, default=666, required=True, help="Number of perturbed restraints to integrate.")
args=parser.parse_args()

#------------------------------------------------------

# Extract base name from input file for output naming
# E.g., "bindfwd_1_pullf.xvg" -> "bindfwd_1"
input_basename = os.path.basename(args.file)
if input_basename.endswith("_pullf.xvg"):
  base_name = input_basename[:-10]  # Remove "_pullf.xvg"
elif input_basename.endswith(".xvg"):
  base_name = input_basename[:-4]   # Remove ".xvg"
else:
  base_name = input_basename

# Output file names based on input file
dg_file = base_name + "_DG.dat"
dgdt_file = base_name + "_dGdt.dat"

#------------------------------------------------------

# read fren file
temp = []
forces = []
erro = []
if os.path.isfile(args.file):
  with open(args.file, "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        if not line.strip().startswith("@"):
          tmp = line.split()
          try:
            temp.append(float(tmp[0]))
            i = 1
            f = 0
            while i <= args.numpertres:
              f += float(tmp[i])
              i += 1
            forces.append(f)
          except (ValueError, IndexError):
            i=0

DG = 0.0
i = 0
f = open(dg_file, "w")
fb = open(dgdt_file, "w")
while i < len(temp) - 1:
  DG += (temp[i+1] - temp[i]) * (forces[i] + forces[i+1]) / 2
  f.write(str(temp[i] + (temp[i+1] - temp[i])/2.0)+"\t"+str(DG*0.0002)+"\n")
  fb.write(str(temp[i] + (temp[i+1] - temp[i])/2.0)+"\t"+str((forces[i] + forces[i+1]) / 2)+"\n")
  i += 1
f.close()
fb.close()

print(-1.0*DG*0.0002)

