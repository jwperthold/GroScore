#!/usr/bin/env python3
#
# Work of one pull leg from its GROMACS pull-force file: the rate times the time
# integral (trapezoid rule on the recorded times) of the summed forces of the first
# -nr coordinates, printed with a minus sign.
#
# NO PARTIAL OR EMPTY LEG IS EVER PRINTED AS A WORK. Through v1.4 this script
# integrated whatever rows parsed: a missing, empty, header-only or one-row file
# printed -0.0 with exit 0, a leg that died part-way printed the work of the part
# that ran, and a row cut inside the integrated columns crashed it with an
# IndexError and empty output, which job.run then wrote as an empty field
# (v15_work digest_audit: runner R2, scorer SCORER-1 and SCORER-4, shared SH7). A
# real mdrun SIGKILLed 4-8 s into a 1IJK pull leg left one row and printed -0.0;
# one killed at 40 s left a last row of 553 of 633 fields, cut in the
# elastic-network tail, and printed a work over 0..t_crash. GROMACS leaves such
# files on every abort that bypasses stdio flushing (std::_Exit in gmx_fatal,
# SIGKILL, SIGSEGV), cut at a 4096-byte block.
#
# Now every failure prints NaN on stdout, says why on stderr and exits non-zero,
# so a caller that only reads stdout still gets a field no score accepts:
#   - a missing file, or fewer than two complete rows;
#   - a row before the last that is not a complete row of numbers (its field
#     count differs from the coordinate count, or a field does not parse), or any
#     non-finite force: the file is not one leg's output;
#   - fewer force columns than -nr, or --ncols disagreeing with the file's legends:
#     the leg ran with another restraint set than the one it is integrated with;
#   - time that runs backwards;
#   - with --tend, a last row more than one output interval short of the leg's
#     end: the leg did not finish;
#   - with --last-ps, a last complete row anywhere but where the leg writes its
#     last row (floor(nsteps / pull-nstfout) * pull-nstfout * dt).
# Only the LAST row may be incomplete, the one the writer was cut in: it is
# dropped, never padded. An unterminated last line counts as incomplete even when
# its fields parse, because its last number may itself be cut.
#
# --tend has to allow one output interval, since mdrun writes pull output only on
# multiples of pull-nstfout: a leg whose nsteps is not one ends short of nsteps*dt
# (bindfwdD, 2290 of 2291 ps). One interval is also exactly what a record that lost
# its last row falls short by, and the last row is the one a disk that fills as the
# leg ends takes: mdrun flushes its output files at the final checkpoint, before it
# writes the last row (the full disk of bm_fe_amber_3a, 2026-10-13, seen in the dhdl
# records). --last-ps names that row's time, so such a record is refused.
#
# On a complete file the arithmetic is the one this script always had, in the same
# order, so every value it printed for a finished leg is reproduced exactly.
#

import os, sys, math, argparse

#------------------------------------------------------

parser = argparse.ArgumentParser(description="Do force integration.")
parser.add_argument('-f','--file', type=str, default="bind_pullf.xvg", required=True, help="Forces.")
parser.add_argument('-nr','--numpertres', type=int, default=666, required=True, help="Number of perturbed restraints to integrate: the first NR force columns.")
parser.add_argument('-r','--rate', type=float, default=0.0002, required=False, help="Pull rate in nm/ps used in the simulation (default: 0.0002). The trapezoidal sum below is the time integral of the pull force; multiplying by the rate converts it to work, so this MUST match the rate in the .mdp or the work is scaled wrongly.")
parser.add_argument('-R','--weights', type=str, default=None, required=False, help="File of per-coordinate weights, one per summed force column. The work is sum_i rate_i*integral(F_i), which equals rate*integral(sum F) only while every coordinate moves at the same rate. When they do not -- interface references now follow their own chord of the pull path -- pass the weights w_i = rate_i/rate and the sum is weighted instead. Omitted, every weight is 1 and the arithmetic is unchanged.")
parser.add_argument('--tend', type=float, default=None, required=False, help="Length of the leg in ps (nsteps*dt). The last complete row must lie within one output interval of it, or the leg did not finish and no work is printed. GROMACS writes pull output only on multiples of pull-nstfout and adds no row at the last step, so one interval is exactly what a finished leg can fall short by.")
parser.add_argument('--last-ps', type=float, default=None, required=False, metavar='PS', help="Time of the last row the leg writes: floor(nsteps / pull-nstfout) * pull-nstfout * dt (tinit 0). The last complete row must be there, to within the four decimals of the time column: one before it means the record lost its tail (a single lost row passes --tend), one after it that the leg ran longer than its mdp says. Either way no work is printed.")
parser.add_argument('--ncols', type=int, default=None, required=False, help="Number of pull coordinates the leg was set up with (pull-ncoords of its mdp). Every row must hold exactly that many force columns, and the file's legends must agree: a leg from another restraint set is refused rather than integrated over the wrong columns.")
args=parser.parse_args()

#------------------------------------------------------

def fail(why, code=1):
  """No work: NaN on stdout for callers that only read stdout, the reason on stderr."""
  sys.stderr.write("integrate.py: %s: %s; no work printed\n" % (args.file, why))
  print("NaN")
  sys.exit(code)

if args.numpertres < 1:
  fail("-nr is %d, and a work needs at least one force column" % args.numpertres, 2)

# Per-coordinate weights, or None for the unweighted sum. A wrong-length file is
# fatal rather than padded: it means the weights and the pull block disagree about
# how many coordinates there are, and silently integrating the ones that happen to
# line up would produce a plausible number from the wrong columns.
weights = None
if args.weights:
  if not os.path.isfile(args.weights):
    sys.stderr.write("integrate.py: no such weights file: %s\n" % args.weights)
    print("NaN")
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
    print("NaN")
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

if not os.path.isfile(args.file):
  fail("no such file")
with open(args.file, "r") as fh:
  lines = fh.read().split("\n")

# Every data row as (fields, terminated by a newline). Each '@ sN legend' line
# names one pull coordinate; the other '@ legend ...' lines are plot layout.
nleg = 0
rows = []
for k, line in enumerate(lines):
  s = line.strip()
  if not s or s.startswith("#"):
    continue
  if s.startswith("@"):
    tok = s.split()
    if len(tok) >= 3 and tok[1][:1] == "s" and tok[1][1:].isdigit() and tok[2] == "legend":
      nleg += 1
    continue
  rows.append((s.split(), k < len(lines) - 1))

if args.ncols is not None and nleg and nleg != args.ncols:
  fail("the file holds %d pull coordinates but --ncols is %d, so the leg ran with another "
       "restraint set than the one it is integrated with" % (nleg, args.ncols))
if args.ncols is not None:
  width = 1 + args.ncols
elif nleg:
  width = 1 + nleg
else:
  width = len(rows[0][0]) if rows else 0
if rows and width < 1 + args.numpertres:
  fail("rows hold %d force columns but -nr is %d" % (width - 1, args.numpertres))

temp = []
forces = []
cut = False
for n, (tok, terminated) in enumerate(rows):
  vals = None
  if terminated and len(tok) == width:
    try:
      vals = [float(x) for x in tok]
    except ValueError:
      vals = None
  if vals is None:
    if n == len(rows) - 1:
      cut = True
      continue          # the row the writer was cut in
    fail("row %d is not a complete row of %d numbers (%d fields)" % (n + 1, width, len(tok)))
  if not all(math.isfinite(v) for v in vals):
    fail("row %d holds a non-finite number" % (n + 1))
  f = 0
  i = 1
  while i <= args.numpertres:
    f += vals[i] * (1.0 if weights is None else weights[i - 1])
    i += 1
  temp.append(vals[0])
  forces.append(f)

if len(temp) < 2:
  fail("%d complete row(s); a work needs at least two" % len(temp))
back = [i for i in range(len(temp) - 1) if temp[i + 1] < temp[i]]
if back:
  fail("time runs backwards after row %d (%g -> %g ps)"
       % (back[0] + 1, temp[back[0]], temp[back[0] + 1]))
steps = sorted(temp[i + 1] - temp[i] for i in range(len(temp) - 1))
interval = steps[len(steps) // 2]
if args.tend is not None:
  if temp[-1] < args.tend - interval - 1e-6 * max(1.0, abs(args.tend)):
    fail("the last complete row is at %g ps of a %g ps leg (rows every %g ps): the leg did "
         "not finish" % (temp[-1], args.tend, interval))
if args.last_ps is not None:
  # Times carry four decimals: 2e-4 ps, plus 0.1% of the interval.
  tol = 2e-4 + 1e-3 * interval
  dropped = ", its last line, cut, was dropped" if cut else ""
  if temp[-1] < args.last_ps - tol:
    fail("the last complete row is at %g ps, before the leg's last row at %g ps (rows every "
         "%g ps%s): the record lost its tail" % (temp[-1], args.last_ps, interval, dropped))
  if temp[-1] > args.last_ps + tol:
    fail("the last complete row is at %g ps, after the leg's last row at %g ps: the leg ran "
         "longer than its mdp says" % (temp[-1], args.last_ps))

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
