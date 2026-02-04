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
            num.append(int(tmp[0]))
            xC.append(float(tmp[3]))
            yC.append(float(tmp[4]))
            zC.append(float(tmp[5]))
          if int(tmp[0]) >= args.startb and tmp[1] == "CA":
            num2.append(int(tmp[0]))
            xC2.append(float(tmp[3]))
            yC2.append(float(tmp[4]))
            zC2.append(float(tmp[5]))
        except (ValueError, IndexError, AttributeError):
          i=0

foundbrokenloops = []
foundbrokenloops2 = []

i = 0
while i < len(num)-1:
  if getdistance(xC[i],yC[i],zC[i],xC[i+1],yC[i+1],zC[i+1]) > 0.4:
    #loop is broken
    foundbrokenloops.append(num[i])
    foundbrokenloops.append(num[i+1])
  i += 1

i = 0
while i < len(num2)-1:
  if getdistance(xC2[i],yC2[i],zC2[i],xC2[i+1],yC2[i+1],zC2[i+1]) > 0.4:
    #loop is broken
    foundbrokenloops2.append(num2[i])
    foundbrokenloops2.append(num2[i+1])
  i += 1

#reload file with all atoms
del num[:]
del num2[:]
del xC[:]
del xC2[:]
del yC[:]
del yC2[:]
del zC[:]
del zC2[:]
linecount = 0
if os.path.isfile(args.confile):
  with open(args.confile, "r") as f:
    for line in f:
      if not line.strip().startswith("#"):
        tmp = line.split()
        try:
          s = re.search(r"\d+(\.\d+)?", tmp[0])
          tmp[0] = s.group(0)
          if int(tmp[0]) < args.startb and linecount > 1:
            num.append(int(tmp[0]))
            xC.append(float(tmp[3]))
            yC.append(float(tmp[4]))
            zC.append(float(tmp[5]))
          if int(tmp[0]) >= args.startb and linecount > 1:
            num2.append(int(tmp[0]))
            xC2.append(float(tmp[3]))
            yC2.append(float(tmp[4]))
            zC2.append(float(tmp[5]))
        except (ValueError, IndexError, AttributeError):
          i=0
        linecount += 1

#print str(foundbrokenloops)
#print str(foundbrokenloops2)

#first partner
bothnear = 0
i = 0
while i < len(foundbrokenloops):
  #search in first end
  j = 0
  while j < len(num):
    if num[j] == foundbrokenloops[i]:
      k = 0
      while k < len(num2):
        if getdistance(xC[j],yC[j],zC[j],xC2[k],yC2[k],zC2[k]) < 0.4:
          #print str(num2[k])
          bothnear = 1
          break
        k += 1
    if bothnear == 1:
      break
    j += 1
  #search in second end if we found something in first end
  if bothnear == 1:
    j = 0
    while j < len(num):
      if num[j] == foundbrokenloops[i+1]:
        k = 0
        while k < len(num2):
          if getdistance(xC[j],yC[j],zC[j],xC2[k],yC2[k],zC2[k]) < 0.4:
            #print str(num2[k])
            bothnear = 2
            break
          k += 1
      if bothnear == 2:
        break
      j += 1
  i += 2

#second partner
if bothnear != 2:
  bothnear = 0
  i = 0
  while i < len(foundbrokenloops2):
    #search in first end
    j = 0
    while j < len(num2):
      if num2[j] == foundbrokenloops2[i]:
        k = 0
        while k < len(num):
          if getdistance(xC[k],yC[k],zC[k],xC2[j],yC2[j],zC2[j]) < 0.4:
            #print str(num[k])
            bothnear = 1
            break
          k += 1
      if bothnear == 1:
        break
      j += 1
    #search in second end if we found something in first end
    if bothnear == 1:
      j = 0
      while j < len(num2):
        if num2[j] == foundbrokenloops2[i+1]:
          k = 0
          while k < len(num):
            if getdistance(xC[k],yC[k],zC[k],xC2[j],yC2[j],zC2[j]) < 0.4:
              #print str(num[k])
              bothnear = 2
              break
            k += 1
        if bothnear == 2:
          break
        j += 1
    i += 2

if bothnear == 2:
  print("1")
else:
  print("0")
