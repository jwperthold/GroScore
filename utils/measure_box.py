#!/usr/bin/env python3
#
# measure_box.py - Size the production box from the equilibration, instead of
# from a constant.
#
# The GROMACS pull code aborts when any checked pair of pull groups is further
# apart than 0.49 * the shortest box VECTOR. The padding that satisfies that has
# always been a hard-coded number in job_fe.run (1.5, then 1.8), derived from one
# structure and applied to every complex. This measures it per structure.
#
# The rule: the production box vector is
#
#     L = D_max + 2 * pad
#
# where D_max is the LARGEST DISTANCE BETWEEN ANY TWO SOLUTE ATOMS, taken as the
# running maximum over the whole equilibration, and pad is the clearance wanted on
# each side (default 1.5 nm).
#
# Three things this gets right that a naive version does not:
#
#   * running maximum, not the frame being re-solvated. On 2KTF the running max is
#     5.3441 nm and the last frame is 4.6078, a 0.74 nm difference that is the
#     whole margin.
#   * the solute group must be the actual solute. Protein_Struct measures 5.34 nm
#     on 2KTF; !Water, which is the group name a script might reach for, measures
#     22.9 nm in an 8.26 nm box because it includes counterions spread through the
#     cell. That would ask for a 25 nm box.
#   * PBC. The chains are separate [molecules], so the frame has to be clustered
#     before any distance is taken, or the answer is the box vector rather than
#     the solute.
#
# The equilibration is run in a SMALL box, since the bound complex never needs the
# production box: on 2KTF the largest checked pull pair in the bound reference is
# 1.18 nm against a limit of 3.29 nm even at -d 1.00. The measurement is what the
# small box is for.
#
# Usage:
#   python3 measure_box.py -f npt_init.xtc -s npt_init.tpr -n index.ndx \
#           -g Protein_Struct -c npt_init.gro --pad 1.5
#
# Prints the editconf -d value to use for the production box on stdout, and the
# measurement on stderr.

import os, sys, re, argparse, subprocess, shutil, tempfile
import numpy as np

p = argparse.ArgumentParser(description="Size the production box from an equilibration trajectory.")
p.add_argument('-f', '--traj', default="npt_init.xtc", help="Equilibration trajectory.")
p.add_argument('-s', '--tpr', default="npt_init.tpr", help="Run input matching --traj.")
p.add_argument('-n', '--index', default="index.ndx", help="Index file.")
p.add_argument('-g', '--group', default="Protein_Struct",
               help="Solute group. Must be the solute only: !Water includes "
                    "counterions and is meaningless here (default: Protein_Struct).")
p.add_argument('-c', '--struct', default=None,
               help="Structure the production box will actually be built from. Its "
                    "own diameter is measured with editconf so the -d that comes "
                    "back produces the requested L on THAT file.")
p.add_argument('--pad', type=float, default=1.5,
               help="Clearance per side, in nm. L = D_max + 2*pad (default: 1.5).")
p.add_argument('--compression', type=float, default=0.9932,
               help="Expected NPT box shrinkage, pre-paid so the equilibrated box "
                    "still meets the target (default: 0.9932, measured on 2KTF).")
p.add_argument('--min-d', type=float, default=0.0,
               help="Floor on the returned padding, e.g. the solvation floor.")
p.add_argument('-b', '--begin', type=float, default=0.0, help="Skip to this time, ps.")
args = p.parse_args()


def fail(msg):
  sys.stderr.write("measure_box: ERROR - %s\n" % msg)
  sys.exit(1)


def read_multi_gro(path):
  """Yield (natoms, coords) per frame from a multi-frame .gro."""
  L = open(path).read().split("\n")
  i = 0
  while i < len(L) - 1 and L[i].strip():
    n = int(L[i + 1])
    if i + 2 + n > len(L):
      break
    xyz = np.empty((n, 3), dtype=float)
    for k in range(n):
      ln = L[i + 2 + k]
      xyz[k] = (float(ln[20:28]), float(ln[28:36]), float(ln[36:44]))
    yield n, xyz
    i += n + 3


def max_pairwise(xyz):
  """Exact largest distance between any two points.

  The convex hull carries the diameter, so hull vertices are enough and are far
  cheaper than the full N^2. Falls back to brute force if scipy is unavailable or
  the hull is degenerate (collinear/coplanar input)."""
  pts = xyz
  try:
    from scipy.spatial import ConvexHull
    if len(xyz) > 8:
      pts = xyz[ConvexHull(xyz).vertices]
  except Exception:
    pts = xyz
  d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
  return float(np.sqrt(d2.max()))


def gro_diameter(path):
  """Diameter of a single-frame .gro, i.e. its largest atom-atom distance.

  This is exactly the quantity editconf calls "diameter" and pads with -d
  (verified: editconf reports 4.713 on 2KTF's emin_vac.gro and brute force gives
  4.71254). Computed here rather than parsed from editconf, which only prints the
  diameter line for -bt dodecahedron and would make this depend on the box type
  of a probe run."""
  for natoms, xyz in read_multi_gro(path):
    return max_pairwise(xyz)
  return None


for f in (args.traj, args.tpr, args.index):
  if not os.path.isfile(f):
    fail("%s not found" % f)
if not shutil.which("gmx"):
  fail("gmx not on PATH")

# Cluster the solute before measuring: the two chains are separate [molecules], so
# an unclustered frame reports the box vector, not the solute. On 2KTF the raw
# npt_init.gro gives 10.400 nm against 4.608 clustered.
work = tempfile.mkdtemp(prefix="measure_box.")
sub = os.path.join(work, "solute.gro")
try:
  cmd = ["gmx", "trjconv", "-s", args.tpr, "-f", args.traj, "-n", args.index,
         "-o", sub, "-pbc", "cluster", "-b", "%g" % args.begin]
  r = subprocess.run(cmd, input="%s\n%s\n" % (args.group, args.group),
                     capture_output=True, text=True)
  if r.returncode != 0 or not os.path.isfile(sub):
    fail("trjconv -pbc cluster failed for group %s:\n%s" % (args.group, r.stderr[-1500:]))

  dmax, n_frames, per_frame = 0.0, 0, []
  for natoms, xyz in read_multi_gro(sub):
    if natoms == 0:
      continue
    d = max_pairwise(xyz)
    per_frame.append(d)
    dmax = max(dmax, d)
    n_frames += 1
finally:
  shutil.rmtree(work, ignore_errors=True)

if n_frames == 0:
  fail("no usable frames in %s" % args.traj)

a = np.array(per_frame)
sys.stderr.write("measure_box: group %s, %d frames, largest solute atom-atom "
                 "distance: max %.4f mean %.4f median %.4f min %.4f std %.4f nm\n"
                 % (args.group, n_frames, a.max(), a.mean(), np.median(a),
                    a.min(), a.std()))
sys.stderr.write("measure_box: last frame %.4f nm, running max reached %.4f nm "
                 "(taking the running max: the last frame is not the worst case)\n"
                 % (a[-1], dmax))

# Target box vector, with the NPT shrinkage pre-paid so the EQUILIBRATED box still
# meets it rather than the freshly built one.
L_target = (dmax + 2.0 * args.pad) / args.compression
sys.stderr.write("measure_box: L = D_max %.4f + 2 * pad %.2f = %.4f nm, "
                 "/ compression %.4f -> build at %.4f nm (pull limit 0.49L = "
                 "%.4f after shrinkage)\n"
                 % (dmax, args.pad, dmax + 2.0 * args.pad, args.compression,
                    L_target, 0.49 * L_target * args.compression))

if args.struct:
  if not os.path.isfile(args.struct):
    fail("%s not found" % args.struct)
  d_struct = gro_diameter(args.struct)
  if d_struct is None:
    fail("could not read any frame from %s" % args.struct)
  pad_d = (L_target - d_struct) / 2.0
  sys.stderr.write("measure_box: %s has diameter %.4f nm, so editconf -d %.3f "
                   "gives L = %.4f\n" % (args.struct, d_struct, pad_d, L_target))
else:
  pad_d = args.pad

if pad_d < args.min_d:
  sys.stderr.write("measure_box: raising -d %.3f to the floor %.3f\n"
                   % (pad_d, args.min_d))
  pad_d = args.min_d

print("%.3f" % pad_d)
