#!/usr/bin/env python3
#
# integrate_dhdl.py - Non-equilibrium switching work from a GROMACS dhdl.xvg.
#
# For the FE protocol the force-constant switching (interface restraints off,
# Boresch restraints on) is driven by a linear lambda ramp (delta-lambda). The
# alchemical work along that ramp is
#
#     W_dhdl = integral_{lambda_start}^{lambda_end} <dH/dlambda> dlambda
#
# GROMACS writes dH/dlambda vs time in dhdl.xvg (dhdl-derivatives = yes). Because
# delta-lambda ramps lambda linearly in time and the records are evenly spaced in
# time, lambda is linear in row index; we reconstruct it as a ramp from
# lambda_start to lambda_end and integrate all derivative columns (summed, since a
# single scalar lambda drives every component) by the trapezoidal rule.
#
# The result is the external switching work in kJ/mol, with the SAME sign meaning
# as the pull-force work from integrate.py, so groscore_fe.py can add the two
# per trajectory before the Crooks-Gaussian-Intersection analysis.
#
# Usage:
#   python integrate_dhdl.py -f bindfwd_1.xvg --direction fwd
#   python integrate_dhdl.py -f bindrev_2.xvg --direction rev
#

import os, sys, re, argparse
import numpy as np

parser = argparse.ArgumentParser(description="Integrate dH/dlambda work from a GROMACS dhdl.xvg.")
parser.add_argument('-f', '--file', type=str, required=True, help="dhdl .xvg file.")
parser.add_argument('--direction', type=str, choices=["fwd", "rev"], default="fwd",
                    help="fwd: lambda ramps lambda_start->lambda_end (default 0->1); "
                         "rev: 1->0.")
parser.add_argument('--lambda-start', type=float, default=None,
                    help="Override starting lambda (default 0 for fwd, 1 for rev).")
parser.add_argument('--lambda-end', type=float, default=None,
                    help="Override ending lambda (default 1 for fwd, 0 for rev).")
parser.add_argument('--leg-ps', type=float, default=None, metavar='PS',
                    help="Simulated length of the leg in ps (nsteps * dt). Given "
                         "this, lambda is placed from the TIME column rather than "
                         "spread evenly over the recorded rows, which is the only "
                         "correct thing to do when the last record does not land "
                         "on the last step: mdrun writes dhdl on multiples of "
                         "nstdhdl and does not force a final row at nsteps, so a "
                         "leg whose nsteps is not a multiple of nstdhdl otherwise "
                         "has its whole ramp stretched onto a shorter run of rows.")
args = parser.parse_args()

#------------------------------------------------------

def parse_xvg(path):
  """Return (legends, data) where legends maps data-column index (0-based, after
  the time column) to its s-legend string, and data is an (nrows, ncols) array
  including the leading time column."""
  legends = {}
  rows = []
  with open(path) as f:
    for line in f:
      s = line.strip()
      if not s:
        continue
      if s.startswith("@"):
        m = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', s)
        if m:
          legends[int(m.group(1))] = m.group(2)
        continue
      if s.startswith("#") or s.startswith("&"):
        continue
      parts = s.split()
      try:
        rows.append([float(p) for p in parts])
      except ValueError:
        continue
  if not rows:
    return legends, np.empty((0, 0))
  width = min(len(r) for r in rows)
  data = np.array([r[:width] for r in rows], dtype=np.float64)
  return legends, data

#------------------------------------------------------

if not os.path.isfile(args.file):
  sys.stderr.write("integrate_dhdl: file not found: %s\n" % args.file)
  print("NaN")
  sys.exit(1)

legends, data = parse_xvg(args.file)
if data.shape[0] < 2 or data.shape[1] < 2:
  sys.stderr.write("integrate_dhdl: not enough data in %s\n" % args.file)
  print("NaN")
  sys.exit(1)

# Column 0 is time. Data columns 1.. correspond to s0, s1, ... in `legends`.
# Identify dH/dlambda derivative columns by their legend text; fall back to all
# non-time columns if legends are absent.
deriv_cols = []
for col in range(1, data.shape[1]):
  leg = legends.get(col - 1, "")
  if re.search(r'd\s*[HVG].*/\s*d\s*\\?x?l', leg) or "dl" in leg.lower() or "dvdl" in leg.lower():
    deriv_cols.append(col)
if not deriv_cols:
  # No recognizable legends: assume every column after time is a derivative.
  deriv_cols = list(range(1, data.shape[1]))

dvdl = data[:, deriv_cols].sum(axis=1)
n = len(dvdl)

# Reconstruct the linear lambda ramp.
lam_start = args.lambda_start
lam_end = args.lambda_end
if lam_start is None:
  lam_start = 0.0 if args.direction == "fwd" else 1.0
if lam_end is None:
  lam_end = 1.0 if args.direction == "fwd" else 0.0

if args.leg_ps and args.leg_ps > 0:
  # lambda from the clock, which is what GROMACS itself does: delta-lambda is per
  # STEP, so lambda(t) = lambda_start + (lambda_end - lambda_start) * t / leg_ps
  # whatever subset of steps happened to be recorded. A leg that stopped
  # recording early then integrates over the span it actually covered, and the
  # uncovered sliver is reported rather than silently absorbed.
  t = data[:, 0]
  lam = lam_start + (lam_end - lam_start) * (t - t[0]) / args.leg_ps
  short = abs(lam[-1] - lam_end)
  if short > 1e-9:
    frac = 100.0 * short / max(abs(lam_end - lam_start), 1e-12)
    sys.stderr.write("integrate_dhdl: %s stops at lambda %.6f, %.2f%% short of "
                     "%.6f; nsteps is probably not a multiple of nstdhdl. The "
                     "work covers only the recorded span.\n"
                     % (os.path.basename(args.file), lam[-1], frac, lam_end))
else:
  # No leg length given: fall back to spreading the ramp evenly over the rows.
  # Exact whenever the last record lands on the last step, which is the case for
  # every mdp this repo generates (nstdhdl 500 divides every nsteps), and wrong
  # by nsteps/(nstdhdl*floor(nsteps/nstdhdl)) when it does not.
  lam = np.linspace(lam_start, lam_end, n)

# W = integral <dH/dlambda> dlambda (trapezoidal). The sign of dlambda is
# negative for the reverse ramp, giving the reverse-process work automatically.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))
W = float(_trapz(dvdl, lam))

# Write a per-step running integral for inspection, mirroring integrate.py.
base = os.path.basename(args.file)
base = base[:-4] if base.endswith(".xvg") else base
with open(base + "_Wdhdl.dat", "w") as out:
  running = 0.0
  for i in range(1, n):
    running += (lam[i] - lam[i - 1]) * (dvdl[i] + dvdl[i - 1]) / 2.0
    out.write("%g\t%g\n" % (lam[i], running))

print(W)
