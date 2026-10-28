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
# A RECORD IS INTEGRATED ONLY WHEN IT IS WHOLE. On 2026-10-13 a full /gpfs
# (bm_fe_amber_3a) dropped whole 8 KiB stdio buffers from dhdl files while mdrun
# carried on, since GROMACS checks those writes only at its checkpoints, and what
# was left still parsed. 1I4D bindrevH_17 lost its header and 0-754 ps and began
# with a one-number fragment; 1IBR bindrevB_12 lost 1548-2328 ps mid-file and kept
# one orphan number; in 1OPH bindrevI_19 the fragment "15 -299.31186" read as a row
# at t = 15 ps between rows near 2000 ps. This script skipped what it could not
# parse, cut every row to the narrowest one and integrated the rest: the first two
# came out NaN only because a one-number fragment left no derivative column, and
# the third came out 169.30 instead of about 190, a number like any other. Now every
# data line must be a row of finite numbers, all rows one width, the last one ending
# in a line end, time must rise strictly and by one record step throughout, and with
# --leg-ps the rows must start at t = 0 and end at --last-ps (where mdrun writes its
# last row; without it, within one step of the leg's end). Anything else prints NaN,
# exits 1 and names the line on stderr. A lost last row matters: mdrun flushes its
# files at the final checkpoint, before it prints that step's dhdl row, so a disk
# that fills then loses exactly that row.
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
parser.add_argument('--last-ps', type=float, default=None, metavar='PS',
                    help="Time of the last record mdrun writes, "
                         "floor(nsteps/nstdhdl) * nstdhdl * dt (job_fe.run passes "
                         "it). Given this, the record must end there exactly, so a "
                         "lost last row is refused; without it, anywhere within one "
                         "record step of --leg-ps is accepted.")

#------------------------------------------------------

def parse_xvg(path):
  """Return (legends, rows, bad). legends maps data-column index (0-based, after
  the time column) to its s-legend string; rows is [(line number, [floats])], one
  per data line, the time column included; bad is the first data line that is not
  a row of numbers, (line number, what follows 'line N' in the message), or None.
  mdrun ends every row with a line end, so a last data line without one was cut: it
  counts as bad too."""
  legends = {}
  rows = []
  bad = None
  last_nl = True
  with open(path) as f:
    for lineno, line in enumerate(f, 1):
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
      last_nl = line.endswith("\n")
      try:
        rows.append((lineno, [float(p) for p in s.split()]))
      except ValueError:
        if bad is None:
          bad = (lineno, " is not a row of numbers: '%s'" % s[:60])
  if bad is None and rows and not last_nl:
    bad = (rows[-1][0], ", the last, has no line end: the file was cut there")
  return legends, rows, bad

def record_problem(rows, bad, leg_ps=None, last_ps=None):
  """Why the rows of parse_xvg are not one whole dhdl record, or None if they are.

  Times are written with four decimals, so steps are compared to within 2e-4 ps
  plus 0.1% of the step. With leg_ps the record must start at t = 0, as every leg
  does (grompp: 'Starting time for run is 0 ps'), and must not run past leg_ps.
  Its end: with last_ps, exactly there; mdrun writes dhdl on multiples of nstdhdl
  only, so a leg whose nsteps is not one ends short of leg_ps (bindfwdD and
  bindfwdE, 1 ps). Without last_ps, anywhere within one record step of leg_ps,
  which cannot tell a lost last row from such a leg."""
  if bad is not None:
    return "line %d%s" % bad
  if len(rows) < 2:
    return "not enough data"
  widths = [len(v) for _, v in rows]
  width = max(sorted(set(widths)), key=widths.count)
  odd = [(n, len(v)) for n, v in rows if len(v) != width]
  if odd:
    more = " (and %d more such line%s)" % (len(odd) - 1, "s" if len(odd) > 2 else "") \
           if len(odd) > 1 else ""
    return ("line %d has %d field%s where the other rows have %d: a broken or spliced "
            "row%s" % (odd[0][0], odd[0][1], "" if odd[0][1] == 1 else "s", width, more))
  if width < 2:
    return "not enough data"
  # nan and inf parse as numbers, and a nan time would pass every check below.
  finite = np.all(np.isfinite(np.array([v for _, v in rows], dtype=np.float64)), axis=1)
  if not finite.all():
    return "line %d holds a value that is not finite" % rows[int(np.argmin(finite))][0]
  t = np.array([v[0] for _, v in rows])
  d = np.diff(t)
  back = np.nonzero(d <= 0)[0]
  if back.size:
    i = int(back[0]) + 1
    return ("line %d: time %g ps does not follow %g ps (line %d)"
            % (rows[i][0], t[i], t[i - 1], rows[i - 1][0]))
  step = float(np.median(d))
  tol = 2e-4 + 1e-3 * step
  off = np.nonzero(np.abs(d - step) > tol)[0]
  if off.size:
    i = int(off[0]) + 1
    lost = int(round(d[i - 1] / step)) - 1
    gap = (" (a gap of about %d row%s)" % (lost, "" if lost == 1 else "s")) if lost >= 1 else ""
    return ("line %d: time goes from %g to %g ps, where the record step is %g ps%s"
            % (rows[i][0], t[i - 1], t[i], step, gap))
  if leg_ps and leg_ps > 0:
    if abs(t[0]) > tol:
      return "the record starts at %g ps, not at 0: its head is missing" % t[0]
    if t[-1] > leg_ps + tol:
      return "the record ends at %g ps, after the end of the leg at %g ps" % (t[-1], leg_ps)
    if last_ps is not None and last_ps > 0:
      if t[-1] < last_ps - tol:
        return ("the record ends at %g ps, before its last row at %g ps: its tail is "
                "missing" % (t[-1], last_ps))
      if t[-1] > last_ps + tol:
        return "the record ends at %g ps, after its last row at %g ps" % (t[-1], last_ps)
    elif leg_ps - t[-1] > step + tol:
      return ("the record ends at %g ps, more than one record step (%g ps) before the "
              "end of the leg at %g ps: its tail is missing" % (t[-1], step, leg_ps))
  return None

#------------------------------------------------------

def main():
  args = parser.parse_args()

  if not os.path.isfile(args.file):
    sys.stderr.write("integrate_dhdl: file not found: %s\n" % args.file)
    print("NaN")
    sys.exit(1)

  # The running integral of an earlier integration of this file goes first, so that
  # none is left beside a record that is now refused (fe_leg_efficiency.py reads it).
  base = os.path.basename(args.file)
  base = base[:-4] if base.endswith(".xvg") else base
  try:
    os.remove(base + "_Wdhdl.dat")
  except OSError:
    pass

  legends, rows, bad = parse_xvg(args.file)
  why = record_problem(rows, bad, args.leg_ps, args.last_ps)
  if why == "not enough data":
    sys.stderr.write("integrate_dhdl: not enough data in %s\n" % args.file)
    print("NaN")
    sys.exit(1)
  if why:
    sys.stderr.write("integrate_dhdl: %s: %s; no work is computed from it.\n"
                     % (args.file, why))
    print("NaN")
    sys.exit(1)
  data = np.array([v for _, v in rows], dtype=np.float64)

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
    # Exact whenever the last record lands on the last step, and wrong by
    # nsteps/(nstdhdl*floor(nsteps/nstdhdl)) when it does not, as for bindfwdD/E
    # and bindrevD/E (nsteps 572750 and 687250, nstdhdl 500); job_fe.run therefore
    # always passes --leg-ps.
    lam = np.linspace(lam_start, lam_end, n)

  # W = integral <dH/dlambda> dlambda (trapezoidal). The sign of dlambda is
  # negative for the reverse ramp, giving the reverse-process work automatically.
  _trapz = getattr(np, "trapezoid", getattr(np, "trapz"))
  W = float(_trapz(dvdl, lam))

  # Write a per-step running integral for inspection, mirroring integrate.py.
  with open(base + "_Wdhdl.dat", "w") as out:
    running = 0.0
    for i in range(1, n):
      running += (lam[i] - lam[i - 1]) * (dvdl[i] + dvdl[i - 1]) / 2.0
      out.write("%g\t%g\n" % (lam[i], running))

  print(W)

if __name__ == "__main__":
  main()
