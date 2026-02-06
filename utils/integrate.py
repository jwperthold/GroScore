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
f = open("DG.dat", "a")
fb = open("dGdt.dat", "a")
while i < len(temp) - 1:
  DG += (temp[i+1] - temp[i]) * (forces[i] + forces[i+1]) / 2
  f.write(str(temp[i] + (temp[i+1] - temp[i])/2.0)+"\t"+str(DG*0.0002)+"\n")
  fb.write(str(temp[i] + (temp[i+1] - temp[i])/2.0)+"\t"+str((forces[i] + forces[i+1]) / 2)+"\n")
  i += 1
f.write("\n")
fb.write("\n")
f.close()
fb.close()

print(-1.0*DG*0.0002)

