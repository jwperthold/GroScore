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
parser.add_argument('-r','--rate', type=float, default=0.0002, required=False, help="Pull rate in nm/ps used in the simulation (default: 0.0002). The trapezoidal sum below is the time integral of the pull force; multiplying by the rate converts it to work, so this MUST match the rate in the .mdp or the work is scaled wrongly.")
parser.add_argument('-R','--weights', type=str, default=None, required=False, help="File of per-coordinate weights, one per summed force column. The work is sum_i rate_i*integral(F_i), which equals rate*integral(sum F) only while every coordinate moves at the same rate. When they do not -- interface references now follow their own chord of the pull path -- pass the weights w_i = rate_i/rate and the sum is weighted instead. Omitted, every weight is 1 and the arithmetic is unchanged.")
args=parser.parse_args()

#------------------------------------------------------

# Per-coordinate weights, or None for the unweighted sum. A wrong-length file is
# fatal rather than padded: it means the weights and the pull block disagree about
# how many coordinates there are, and silently integrating the ones that happen to
# line up would produce a plausible number from the wrong columns.
weights = None
if args.weights:
  if not os.path.isfile(args.weights):
    sys.stderr.write("integrate.py: no such weights file: %s\n" % args.weights)
    sys.exit(2)
  weights = []
  with open(args.weights) as fh:
    for line in fh:
      line = line.strip()
      if line and not line.startswith("#"):
        weights.append(float(line))
  if len(weights) != args.numpertres:
    sys.stderr.write("integrate.py: %s holds %d weights but -nr is %d; refusing to "
                     "integrate rather than guess which columns they belong to\n"
                     % (args.weights, len(weights), args.numpertres))
    sys.exit(2)

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
              f += float(tmp[i]) * (1.0 if weights is None else weights[i - 1])
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
  f.write(str(temp[i] + (temp[i+1] - temp[i])/2.0)+"\t"+str(DG*args.rate)+"\n")
  fb.write(str(temp[i] + (temp[i+1] - temp[i])/2.0)+"\t"+str((forces[i] + forces[i+1]) / 2)+"\n")
  i += 1
f.close()
fb.close()

print(-1.0*DG*args.rate)

