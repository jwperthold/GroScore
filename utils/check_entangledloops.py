#!/usr/bin/env python3
#
# PLEASE REPORT BUGS, QUESTIONS AND COMMENTS TO JAN.PERTHOLD@BOKU.AC.AT
#

import string
import math
import array
import os, sys, glob, re, time, argparse
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Calulate necesaary rotations to align CA-COMS with the z-axis.")
parser.add_argument('-f','--confile', type=str, default="conf.gro", required=True, help="GROMACS coordinate file.")
parser.add_argument('-s','--startb', type=int, default="24", required=True, help="Start residue for protein B.")
args=parser.parse_args()

#------------------------------------------------------

def getdistance(x1, y1, z1, x2, y2, z2):
  """ Gives the distance between two points in 3D space.
  """  
  return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))

def getmeanpos(x1, y1, z1, x2, y2, z2):
  """ Returns the mean position of two points in 3D space.
  """  
  return 0.5*(x1+x2), 0.5*(y1+y2), 0.5*(z1+z2)

#------------------------------------------------------

# read coordinate file - only CAs
num = []
num2 = []
xC = []
xC2 = []
yC = []
yC2 = []
zC = []
zC2 = []
if os.path.isfile(args.confile):
  with open(args.confile, "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        tmp = line.split()
        try:
          s = re.search(r"\d+(\.\d+)?", tmp[0])
          tmp[0] = s.group(0)
          if int(tmp[0]) < args.startb and tmp[1] == "CA":
            num.append(int(tmp[2]))
            xC.append(float(tmp[3]))
            yC.append(float(tmp[4]))
            zC.append(float(tmp[5]))
          if int(tmp[0]) >= args.startb and tmp[1] == "CA":
            num2.append(int(tmp[2]))
            xC2.append(float(tmp[3]))
            yC2.append(float(tmp[4]))
            zC2.append(float(tmp[5]))
        except (ValueError, IndexError, AttributeError):
          i=0

score = 0

i = 5
outeroutermax = 0
while i < 21:
  #all ms of first partner
  outermax = 0
  j = 0
  while j < len(num)-i:
    #all threads of first partner
    outer = 0
    #if outer thread ends are within 1 nm
    if getdistance(xC[j],yC[j],zC[j],xC[i+j-1],yC[i+j-1],zC[i+j-1]) < 1.0:
      k = j
      while k < i+j:
        l = 5
        innermax = 0
        while l < 21:
          #all ms of second partner
          m = 0
          while m < len(num2)-l:
            #all threads of second partner
            #if inner thread ends are within 1 nm AND Cas of the central residues of both threads are within 1 nm
            if getdistance(xC2[m],yC2[m],zC2[m],xC2[l+m-1],yC2[l+m-1],zC2[l+m-1]) < 1.0 and getdistance(xC[i//2+j-1],yC[i//2+j-1],zC[i//2+j-1],xC2[l//2+m-1],yC2[l//2+m-1],zC2[l//2+m-1]) < 1.0:
              inner = 0
              n = m
              while n < l+m:
                a1, a2, a3 = getmeanpos(xC[k],yC[k],zC[k],xC[k+1],yC[k+1],zC[k+1])
                b1, b2, b3 = getmeanpos(xC2[n],yC2[n],zC2[n],xC2[n+1],yC2[n+1],zC2[n+1])
                dif = np.array([a1,a2,a3]) - np.array([b1,b2,b3])
                inner += np.dot((dif / (np.linalg.norm(dif)**3)), np.cross(np.array([xC[k+1]-xC[k],yC[k+1]-yC[k],zC[k+1]-zC[k]]),np.array([xC2[n+1]-xC2[n],yC2[n+1]-yC2[n],zC2[n+1]-zC2[n]])))
                n += 1
              if innermax < abs(inner):
                innermax = abs(inner) 
            m += 1
          l += 1
        outer += innermax
        k += 1
    if outermax < abs(outer):
      outermax = abs(outer)
    j += 1
  if outeroutermax < outermax:
    outeroutermax = outermax
  i += 1

if (1/(4*3.14159265359)*outeroutermax) > 1.0:
  print("1")
else:
  print("0")
