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

parser = argparse.ArgumentParser(description="Generate distance restarints for pulling and elastic network.")
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
  return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))

#------------------------------------------------------

# cutoff and elstic network parameters
interfacecutoff = 0.6
enk = 250

# read coordinate file
# prot1[residue, atomname, atom, xC, yC, zC]
# prot2[residue, atomname, atom, xC, yC, zC]
prot1 = np.empty(shape=(1000000, 6), dtype=object)
prot2 = np.empty(shape=(1000000, 6), dtype=object)
len1 = 0
len2 = 0
if os.path.isfile("npt_center_prot.gro"):
  with open("npt_center_prot.gro", "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        left = line[:15] 
        right = line[15:] 
        tmp = left.split() + right.split()
        try:
          s = re.search(r"\d+(\.\d+)?", tmp[0])
          resnum = s.group(0)
          if int(resnum) not in residues_b and tmp[1] != "CL":
            prot1[len1] = [tmp[0], tmp[1], tmp[2], np.float64(tmp[3]), np.float64(tmp[4]), np.float64(tmp[5])]
            len1 += 1
          if int(resnum) in residues_b and tmp[1] != "CL":
            prot2[len2] = [tmp[0], tmp[1], tmp[2], np.float64(tmp[3]), np.float64(tmp[4]), np.float64(tmp[5])]
            len2 += 1
        except (ValueError, IndexError, AttributeError):
          i = 0

# calculate inter protein distance restraints
# interdis[atom1, atom2, distance]
interdis = np.empty(shape=(1000000, 3))
numinterdis = 0
i = 0
while i < len1:
  if prot1[i,1][0] != "H" and prot1[i,1][:2] != "MN":
    j = 0
    while j < len2:
      if prot2[j,1][0] != "H" and prot2[j,1][:2] != "MN":
        tempdis = getdistance(prot1[i,3], prot1[i,4], prot1[i,5], prot2[j,3], prot2[j,4], prot2[j,5])
        if tempdis <= interfacecutoff:
          interdis[numinterdis] = [i, j, tempdis]
          numinterdis += 1
      j += 1
  i += 1
# calculate elastic networks
protanchor1 = np.empty(shape=(1000000, 6), dtype=object)
protanchor2 = np.empty(shape=(1000000, 6), dtype=object)
protkeep1 = np.empty(shape=(1000000, 6), dtype=object)
protkeep2 = np.empty(shape=(1000000, 6), dtype=object)
lena1 = 0
lena2 = 0
lenk1 = 0
lenk2 = 0
# prot1 - find anchor CA atoms to restrain (terminal residues except first and last residue of chain if they only have one end)
i = 0
while i < len1:
  if prot1[i,1] == "OT":
    res = prot1[i,0]
    j = 0
    while j < len1:
      if prot1[j,1] == "CA" and prot1[j,0] == res:
        protanchor1[lena1] = prot1[j]
        lena1 += 1
      j += 1
  if prot1[i,1] == "H2":
    res = prot1[i,0]
    j = 0
    while j < len1:
      if prot1[j,1] == "CA" and prot1[j,0] == res:
        protanchor1[lena1] = prot1[j]
        lena1 += 1
      j += 1
  i += 1
protanchor1 = protanchor1[1:lena1-1]
lena1 -= 2
# prot1 - now find CA atoms in en distance around anchor CA atoms and restrain them
i = 0
while i < len1:
  if prot1[i,1] == "CA":
    j = 0
    while j < lena1:
      tempdis = getdistance(prot1[i,3], prot1[i,4], prot1[i,5], protanchor1[j,3], protanchor1[j,4], protanchor1[j,5])
      if tempdis <= 0.9:
        protkeep1[lenk1] = prot1[i]
        lenk1 += 1
        break
      j += 1
  i += 1
protkeep1 = protkeep1[:lenk1]
# prot1 - now build en
en1dis = np.empty(shape=(1000000, 3))
numen1dis = 0
i = 0
while i < lenk1:
  j = i + 1
  while j < lenk1:
    tempdis = getdistance(protkeep1[i,3], protkeep1[i,4], protkeep1[i,5], protkeep1[j,3], protkeep1[j,4], protkeep1[j,5])
    if tempdis <= 0.9 and tempdis >= 0.4:
      en1dis[numen1dis] = [i, j, tempdis]
      numen1dis += 1
    j += 1
  i += 1
en1dis = en1dis[:numen1dis]
# prot2 - find anchor CA atoms to restrain (terminal residues except first and last residue of chain if they only have one end)
i = 0
while i < len1:
  if prot2[i,1] == "OT":
    res = prot2[i,0]
    j = 0
    while j < len1:
      if prot2[j,1] == "CA" and prot2[j,0] == res:
        protanchor2[lena2] = prot2[j]
        lena2 += 1
      j += 1
  if prot2[i,1] == "H2":
    res = prot2[i,0]
    j = 0
    while j < len1:
      if prot2[j,1] == "CA" and prot2[j,0] == res:
        protanchor2[lena2] = prot2[j]
        lena2 += 1
      j += 1
  i += 1
protanchor2 = protanchor2[1:lena2-1]
lena2 -= 2
# prot2 - now find CA atoms in en distance around anchor CA atoms and restrain them
i = 0
while i < len1:
  if prot2[i,1] == "CA":
    j = 0
    while j < lena2:
      tempdis = getdistance(prot2[i,3], prot2[i,4], prot2[i,5], protanchor2[j,3], protanchor2[j,4], protanchor2[j,5])
      if tempdis <= 0.9:
        protkeep2[lenk2] = prot2[i]
        lenk2 += 1
        break
      j += 1
  i += 1
protkeep2 = protkeep2[:lenk2]
# prot2 - now build en
en2dis = np.empty(shape=(1000000, 3))
numen2dis = 0
i = 0
while i < lenk2:
  j = i + 1
  while j < lenk2:
    tempdis = getdistance(protkeep2[i,3], protkeep2[i,4], protkeep2[i,5], protkeep2[j,3], protkeep2[j,4], protkeep2[j,5])
    if tempdis <= 0.9 and tempdis >= 0.4:
      en2dis[numen2dis] = [i, j, tempdis]
      numen2dis += 1
    j += 1
  i += 1
en2dis = en2dis[:numen2dis]

print(str(numinterdis))
# write inter protein distance restraints groups to index file
index = open("index.ndx", "a")
i = 0
while i < numinterdis:
  index.write("[ a_" + prot1[int(interdis[i,0]),2] + " ]" + "\n")
  index.write(prot1[int(interdis[i,0]),2] + "\n")
  index.write("[ a_" + prot2[int(interdis[i,1]),2] + " ]" + "\n")
  index.write(prot2[int(interdis[i,1]),2] + "\n")
  i += 1
i = 0
while i < numen1dis:
  index.write("[ a_" + protkeep1[int(en1dis[i,0]),2] + " ]" + "\n")
  index.write(protkeep1[int(en1dis[i,0]),2] + "\n")
  index.write("[ a_" + protkeep1[int(en1dis[i,1]),2] + " ]" + "\n")
  index.write(protkeep1[int(en1dis[i,1]),2] + "\n")
  i += 1
i = 0
while i < numen2dis:
  index.write("[ a_" + protkeep2[int(en2dis[i,0]),2] + " ]" + "\n")
  index.write(protkeep2[int(en2dis[i,0]),2] + "\n")
  index.write("[ a_" + protkeep2[int(en2dis[i,1]),2] + " ]" + "\n")
  index.write(protkeep2[int(en2dis[i,1]),2] + "\n")
  i += 1
index.close()
# write to nptfwd.mdp
k = 25000.0/numinterdis
f = open("nptfwd.mdp", "a")
f.write("pull-ngroups            = %.0f\n"%(numinterdis*2+numen1dis*2+numen2dis*2))
f.write("pull-ncoords            = %.0f\n"%(numinterdis+numen1dis+numen2dis))
f.write("\n")
groupcount = 1
coordcount = 1
j = 0
while j < numinterdis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(prot1[int(interdis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(prot2[int(interdis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,interdis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,k))
  f.write("\n")
  coordcount += 1
  j += 1
j = 0
while j < numen1dis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep1[int(en1dis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep1[int(en1dis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,en1dis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,enk))
  f.write("\n")
  coordcount += 1
  j += 1
j = 0
while j < numen2dis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep2[int(en2dis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep2[int(en2dis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,en2dis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,enk))
  f.write("\n")
  coordcount += 1
  j += 1
# write to bind.mdp
k = 25000.0/numinterdis
f = open("bind.mdp", "a")
f.write("pull-ngroups            = %.0f\n"%(numinterdis*2+numen1dis*2+numen2dis*2))
f.write("pull-ncoords            = %.0f\n"%(numinterdis+numen1dis+numen2dis))
f.write("\n")
groupcount = 1
coordcount = 1
j = 0
while j < numinterdis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(prot1[int(interdis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(prot2[int(interdis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0.0002\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,interdis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,k))
  f.write("\n")
  coordcount += 1
  j += 1
j = 0
while j < numen1dis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep1[int(en1dis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep1[int(en1dis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,en1dis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,enk))
  f.write("\n")
  coordcount += 1
  j += 1
j = 0
while j < numen2dis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep2[int(en2dis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep2[int(en2dis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,en2dis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,enk))
  f.write("\n")
  coordcount += 1
  j += 1
f.close()
# write to nptrev.mdp
k = 25000.0/numinterdis
f = open("nptrev.mdp", "a")
f.write("pull-ngroups            = %.0f\n"%(numinterdis*2+numen1dis*2+numen2dis*2))
f.write("pull-ncoords            = %.0f\n"%(numinterdis+numen1dis+numen2dis))
f.write("\n")
groupcount = 1
coordcount = 1
j = 0
while j < numinterdis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(prot1[int(interdis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(prot2[int(interdis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,interdis[j,2]+1.0))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,k))
  f.write("\n")
  coordcount += 1
  j += 1
j = 0
while j < numen1dis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep1[int(en1dis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep1[int(en1dis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,en1dis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,enk))
  f.write("\n")
  coordcount += 1
  j += 1
j = 0
while j < numen2dis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep2[int(en2dis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep2[int(en2dis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,en2dis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,enk))
  f.write("\n")
  coordcount += 1
  j += 1
f.close()
# write to bindrev.mdp
k = 25000.0/numinterdis
f = open("bindrev.mdp", "a")
f.write("pull-ngroups            = %.0f\n"%(numinterdis*2+numen1dis*2+numen2dis*2))
f.write("pull-ncoords            = %.0f\n"%(numinterdis+numen1dis+numen2dis))
f.write("\n")
groupcount = 1
coordcount = 1
j = 0
while j < numinterdis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(prot1[int(interdis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(prot2[int(interdis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = -0.0002\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,interdis[j,2]+1.0))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,k))
  f.write("\n")
  coordcount += 1
  j += 1
j = 0
while j < numen1dis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep1[int(en1dis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep1[int(en1dis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,en1dis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,enk))
  f.write("\n")
  coordcount += 1
  j += 1
j = 0
while j < numen2dis:
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep2[int(en2dis[j,0]),2])))
  g1 = groupcount
  groupcount += 1
  f.write("pull-group%.0f-name        = a_%.0f\n"%(groupcount,float(protkeep2[int(en2dis[j,1]),2])))
  g2 = groupcount
  groupcount += 1
  f.write("pull-coord%.0f-type        = umbrella\n"%(coordcount))
  f.write("pull-coord%.0f-geometry    = distance\n"%(coordcount))
  f.write("pull-coord%.0f-dim         = Y Y Y\n"%(coordcount))
  f.write("pull-coord%.0f-start       = no\n"%(coordcount))
  f.write("pull-coord%.0f-rate        = 0\n"%(coordcount))
  f.write("pull-coord%.0f-groups      = %.0f %.0f\n"%(coordcount,g1,g2))
  f.write("pull-coord%.0f-init        = %.8f\n"%(coordcount,en2dis[j,2]))
  f.write("pull-coord%.0f-k           = %.8f\n"%(coordcount,enk))
  f.write("\n")
  coordcount += 1
  j += 1
f.close()
