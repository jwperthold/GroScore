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

parser = argparse.ArgumentParser(description="Cut-out protein around the interface.")
parser.add_argument('-m','--chainmap', type=str, required=True, help="Chain map file containing residue numbers for protein B.")
args=parser.parse_args()

#------------------------------------------------------

def read_chain_map(filepath):
  """Read chain_map.gs file and return set of residue numbers belonging to protein B."""
  residues_b = set()
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      for line in f:
        if not line.strip().startswith("#"):
          try:
            residues_b.add(int(line.strip()))
          except (ValueError, IndexError):
            pass
  return residues_b

residues_b = read_chain_map(args.chainmap)

#------------------------------------------------------

def getdistance(x1, y1, z1, x2, y2, z2):
  """ Gives the distance between two points in 3D space.
  """  
  return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))

#------------------------------------------------------

# cutoff parameters
interfacecutoff = 0.6
keepcutoff = 2.0

# read coordinate file
# prot1[residue, atomname, xC, yC, zC]
# prot2[residue, atomname, xC, yC, zC]
prot1 = np.empty(shape=(1000000, 5), dtype=object)
prot2 = np.empty(shape=(1000000, 5), dtype=object)
len1 = 0
len2 = 0
if os.path.isfile("conf.gro"):
  with open("conf.gro", "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        left = line[:15] 
        right = line[15:] 
        tmp = left.split() + right.split()
        try:
          s = re.search(r"\d+(\.\d+)?", tmp[0])
          resnum = s.group(0)
          if int(resnum) not in residues_b and tmp[1] != "CL" and tmp[1][0] != "H" and tmp[1][:2] != "MN":
            prot1[len1] = [tmp[0], tmp[1], np.float64(tmp[3]), np.float64(tmp[4]), np.float64(tmp[5])]
            len1 += 1
          if int(resnum) in residues_b and tmp[1] != "CL" and tmp[1][0] != "H" and tmp[1][:2] != "MN":
            prot2[len2] = [tmp[0], tmp[1], np.float64(tmp[3]), np.float64(tmp[4]), np.float64(tmp[5])]
            len2 += 1
        except (ValueError, IndexError, AttributeError):
          i = 0

# calculate inter protein distances
# interdis[atom1, atom2, distance]
interdis = np.empty(shape=(1000000, 3))
numinterdis = 0
tempdis = 0.0
i = 0
while i < len1:
  j = 0
  while j < len2:
    tempdis = getdistance(prot1[i,2], prot1[i,3], prot1[i,4], prot2[j,2], prot2[j,3], prot2[j,4])
    if tempdis <= interfacecutoff:
      interdis[numinterdis] = [i, j, tempdis]
      numinterdis += 1
    j += 1
  i += 1

# find residues around interface
keep1 = []
keep2 = []
laterkeep1 = []
laterkeep2 = []
protkeep1 = np.empty(shape=(1000000, 5), dtype=object)
protkeep2 = np.empty(shape=(1000000, 5), dtype=object)
protlaterkeep1 = np.empty(shape=(1000000, 5), dtype=object)
protlaterkeep2 = np.empty(shape=(1000000, 5), dtype=object)
lenk1 = 0
lenk2 = 0
lenlk1 = 0
lenlk2 = 0
# prot1
i = 0
while i < len1:
  j = 0
  while j < numinterdis:
    # distance between atom and interface atom of the same protein
    tempdis = getdistance(prot1[i,2], prot1[i,3], prot1[i,4], prot1[int(interdis[j,0]),2], prot1[int(interdis[j,0]),3], prot1[int(interdis[j,0]),4])
    if tempdis <= keepcutoff:
      keep1.append(prot1[i,0])
      break
    j += 1
  i += 1
# get all coordinates to keep for check whether the kept atoms are within elastic network distance
i = 0
while i < len1:
  j = 0
  while j < len(keep1):
    if prot1[i,0] == keep1[j]:
      if prot1[i,1] == "CA":
        protkeep1[lenk1] = prot1[i]
        lenk1 += 1
    j += 1
  i += 1
# check whether the kept atoms are within elastic network distance
i = 0
while i < lenk1:
  j = i + 1
  while j < lenk1:
    tempdis = getdistance(protkeep1[i,2], protkeep1[i,3], protkeep1[i,4], protkeep1[j,2], protkeep1[j,3], protkeep1[j,4])
    if tempdis <= 0.9 and tempdis >= 0.4:
      laterkeep1.append(protkeep1[i,0])
      break
    j += 1
  i += 1
# get all coordinates to keep
i = 0
while i < len1:
  j = 0
  while j < len(laterkeep1):
    if prot1[i,0] == laterkeep1[j]:
      protlaterkeep1[lenlk1] = prot1[i]
      lenlk1 += 1
      break
    j += 1
  i += 1
# prot2
i = 0
while i < len2:
  j = 0
  while j < numinterdis:
    # distance between atom and interface atom of the same protein
    tempdis = getdistance(prot2[i,2], prot2[i,3], prot2[i,4], prot2[int(interdis[j,1]),2], prot2[int(interdis[j,1]),3], prot2[int(interdis[j,1]),4])
    if tempdis <= keepcutoff:
      keep2.append(prot2[i,0])
      break
    j += 1
  i += 1
# get all coordinates to keep for check whether the kept atoms are within elastic network distance
i = 0
while i < len2:
  j = 0
  while j < len(keep2):
    if prot2[i,0] == keep2[j]:
      if prot2[i,1] == "CA":
        protkeep2[lenk2] = prot2[i]
        lenk2 += 1
    j += 1
  i += 1
# check whether the kept atoms are within elastic network distance
i = 0
while i < lenk2:
  j = i + 1
  while j < lenk2:
    tempdis = getdistance(protkeep2[i,2], protkeep2[i,3], protkeep2[i,4], protkeep2[j,2], protkeep2[j,3], protkeep2[j,4])
    if tempdis <= 0.9 and tempdis >= 0.4:
      laterkeep2.append(protkeep2[i,0])
      break
    j += 1
  i += 1
# get all coordinates to keep
i = 0
while i < len2:
  j = 0
  while j < len(laterkeep2):
    if prot2[i,0] == laterkeep2[j]:
      protlaterkeep2[lenlk2] = prot2[i]
      lenlk2 += 1
      break
    j += 1
  i += 1

# write cutout pdb file
pdbfile = open("cutout.pdb", "a")
i = 0
while i < (lenlk1 + lenlk2):
  if i < lenlk1:
    if i > 0:
      if int(protlaterkeep1[i,0][:-3]) > int(protlaterkeep1[i-1,0][:-3]) + 1:
        pdbfile.write("{0: <5}".format("TER"))
        pdbfile.write("{0: >6}".format(str(i+1)))
        pdbfile.write("   ")
        pdbfile.write("{0: <3}".format(" "))
        pdbfile.write(" ") #Alternate location indicator
        pdbfile.write("{0: >2}".format(" "))
        pdbfile.write(" ")
        pdbfile.write("A")
        pdbfile.write("{0: >4}".format(protlaterkeep1[i-1,0][:-3]))
        pdbfile.write("\n")
    pdbfile.write("{0: <5}".format("ATOM"))
    pdbfile.write("{0: >6}".format(str(i+1)))
    pdbfile.write("  ")
    pdbfile.write("{0: <3}".format(protlaterkeep1[i,1]))
    pdbfile.write(" ") #Alternate location indicator
    pdbfile.write("{0: >2}".format(protlaterkeep1[i,0][-3:]))
    pdbfile.write(" ")
    pdbfile.write("A")
    pdbfile.write("{0: >4}".format(protlaterkeep1[i,0][:-3]))
    pdbfile.write("    ") #Code for insertions of residues
    pdbfile.write("{:8.3f}".format(protlaterkeep1[i,2]*10))
    pdbfile.write("{:8.3f}".format(protlaterkeep1[i,3]*10))
    pdbfile.write("{:8.3f}".format(protlaterkeep1[i,4]*10))
    pdbfile.write("{0: >6}".format("1.00"))
    pdbfile.write("{0: >6}".format("0.00"))
    pdbfile.write("          ") #Segment identifier
    pdbfile.write("{0: >2}".format(protlaterkeep1[i,1][0]))
    pdbfile.write("\n")
  else:
    if i-lenlk1 == 0:
        pdbfile.write("{0: <5}".format("TER"))
        pdbfile.write("{0: >6}".format(str(i+1)))
        pdbfile.write("   ")
        pdbfile.write("{0: <3}".format(" "))
        pdbfile.write(" ") #Alternate location indicator
        pdbfile.write("{0: >2}".format(" "))
        pdbfile.write(" ")
        pdbfile.write("A")
        pdbfile.write("{0: >4}".format(protlaterkeep1[i-1,0][:-3]))
        pdbfile.write("\n")
    if i-lenlk1 > 0:
      if int(protlaterkeep2[i-lenlk1,0][:-3]) > int(protlaterkeep2[i-lenlk1-1,0][:-3]) + 1:
        pdbfile.write("{0: <5}".format("TER"))
        pdbfile.write("{0: >6}".format(str(i+1)))
        pdbfile.write("   ")
        pdbfile.write("{0: <3}".format(" "))
        pdbfile.write(" ") #Alternate location indicator
        pdbfile.write("{0: >2}".format(" "))
        pdbfile.write(" ")
        pdbfile.write("B")
        pdbfile.write("{0: >4}".format(protlaterkeep2[i-lenlk1-1,0][:-3]))
        pdbfile.write("\n")
    pdbfile.write("{0: <5}".format("ATOM"))
    pdbfile.write("{0: >6}".format(str(i+1)))
    pdbfile.write("  ")
    pdbfile.write("{0: <3}".format(protlaterkeep2[i-lenlk1,1]))
    pdbfile.write(" ") #Alternate location indicator
    pdbfile.write("{0: >2}".format(protlaterkeep2[i-lenlk1,0][-3:]))
    pdbfile.write(" ")
    pdbfile.write("B")
    pdbfile.write("{0: >4}".format(protlaterkeep2[i-lenlk1,0][:-3]))
    pdbfile.write("    ") #Code for insertions of residues
    pdbfile.write("{:8.3f}".format(protlaterkeep2[i-lenlk1,2]*10))
    pdbfile.write("{:8.3f}".format(protlaterkeep2[i-lenlk1,3]*10))
    pdbfile.write("{:8.3f}".format(protlaterkeep2[i-lenlk1,4]*10))
    pdbfile.write("{0: >6}".format("1.00"))
    pdbfile.write("{0: >6}".format("0.00"))
    pdbfile.write("          ") #Segment identifier
    pdbfile.write("{0: >2}".format(protlaterkeep2[i-lenlk1,1][0]))
    pdbfile.write("\n")
  i += 1
pdbfile.close()
