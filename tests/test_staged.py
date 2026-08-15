#!/usr/bin/env python3
"""Regression test for the staged unbinding ramp.

The ramp runs as stage A (lambda 0 -> U_SPLIT) and stage B (U_SPLIT -> 1) with an
equilibrium hold between them, so that each stage dissipates a fraction of the
total and each has forward/reverse work overlap where the whole ramp has none.
Two properties carry that design, and neither is visible in a finished number:

  1. ADDITIVITY. The stage works must sum to the work of the whole ramp, or the
     staged and one-shot estimates are not two estimates of the same quantity and
     the cross-check in scores_fe.gs means nothing. The hold does no work, so the
     identity should be exact. The dhdl channel is where it can silently fail:
     integrate_dhdl.py rebuilds the lambda ramp from --direction alone, which
     assumes the leg spans the WHOLE of lambda, so a stage integrated with the
     default span comes out 1/U_SPLIT times too large and still looks like a
     work. Section 1 checks the identity through integrate_dhdl.py itself and
     then checks that the wrong span really would have been wrong, so the fix in
     job_fe.run cannot be dropped unnoticed.

  2. THE STAGED ESTIMATE. Section 2 runs groscore_fe.py end to end on synthetic
     Crooks pairs whose dG is known, built so the stages overlap and their sum
     does not: BAR must recover each stage, report their sum as dG_unbind, and
     report NOTHING for the one-shot column. It also pins the row layout (every
     row the same width, including PENDING) and the joint bootstrap (the dG_bind
     CI must not be the quadrature of its parts).

Run:  python3 tests/test_staged.py
"""

import math, os, re, shutil, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "utils"))
import fe_protocol as P
RT = 0.00831446261815324 * 310.0
# The first stage's span, i.e. the fraction of lambda a stage-A dhdl file covers.
U_SPLIT = P.RAMP[0][2] / P.PULL_DIST

failures = []

def check(name, ok, detail=""):
  print("  %-56s %s%s" % (name, "PASS" if ok else "FAIL",
                          ("  " + detail) if detail else ""))
  if not ok:
    failures.append(name)

work = tempfile.mkdtemp(prefix="groscore_staged_")

#------------------------------------------------------
# 1. the stage works add up
#------------------------------------------------------

print("")
print("1. staged dhdl works sum to the unstaged one")

INTEGRATE = os.path.join(REPO, "utils", "integrate_dhdl.py")

def dvdl(lam):
  """dH/dlambda with real structure in it: a steep rise near lambda=0, as the
  interface comes apart, decaying away. Asymmetric on purpose, so a wrong lambda
  span cannot pass by cancellation."""
  return 900.0 * np.exp(-6.0 * lam) + 40.0 * np.sin(9.0 * lam) - 120.0

def write_xvg(path, lam, dt=2.0):
  with open(path, "w") as f:
    f.write("# synthetic dhdl\n")
    f.write('@ s0 legend "dH/dl fep-lambda = 0.0000"\n')
    for i, l in enumerate(lam):
      f.write("%.4f  %.6f\n" % (i * dt, dvdl(l)))

def integrate(path, direction, span=True, l0=None, l1=None):
  cmd = [sys.executable, INTEGRATE, "-f", path, "--direction", direction]
  if span:
    cmd += ["--lambda-start", str(l0), "--lambda-end", str(l1)]
  out = subprocess.run(cmd, capture_output=True, text=True)
  if out.returncode != 0:
    sys.exit("integrate_dhdl.py failed on %s:\n%s" % (path, out.stderr))
  return float(out.stdout.split()[0])

NA, NB = 7501, 2501                      # 15 ns and 5 ns at 2 ps output
lamA = np.linspace(0.0, U_SPLIT, NA)
lamB = np.linspace(U_SPLIT, 1.0, NB)
lamW = np.linspace(0.0, 1.0, NA + NB - 1)
for name, lam in (("A", lamA), ("B", lamB), ("whole", lamW),
                  ("rB", lamB[::-1]), ("rA", lamA[::-1])):
  write_xvg(os.path.join(work, name + ".xvg"), lam)

WA = integrate(os.path.join(work, "A.xvg"), "fwd", True, 0.0, U_SPLIT)
WB = integrate(os.path.join(work, "B.xvg"), "fwd", True, U_SPLIT, 1.0)
WW = integrate(os.path.join(work, "whole.xvg"), "fwd", True, 0.0, 1.0)
fine = np.linspace(0.0, 1.0, 200001)
exact = float(np.trapezoid(dvdl(fine), fine))

print("   W_A %.3f  +  W_B %.3f  =  %.3f      whole %.3f   analytic %.3f"
      % (WA, WB, WA + WB, WW, exact))
check("stage works sum to the whole-ramp work", abs((WA + WB) - WW) < 0.05,
      "difference %.4g" % abs((WA + WB) - WW))
check("the whole-ramp work is the true integral", abs(WW - exact) < 0.05)

WA_default = integrate(os.path.join(work, "A.xvg"), "fwd", span=False)
check("the default lambda span really would be wrong",
      abs(WA_default / WA - 1.0 / U_SPLIT) < 0.01,
      "%.3f, i.e. %.2fx too large" % (WA_default, WA_default / WA))

RB = integrate(os.path.join(work, "rB.xvg"), "rev", True, 1.0, U_SPLIT)
RA = integrate(os.path.join(work, "rA.xvg"), "rev", True, U_SPLIT, 0.0)
check("reverse stages sum to minus the forward total",
      abs((RB + RA) + WW) < 0.05, "%.3f vs %.3f" % (RB + RA, -WW))

# The span alone is not sufficient. mdrun writes dhdl on multiples of nstdhdl and
# does not force a final record at nsteps, so a leg whose nsteps is not a multiple
# of nstdhdl stops recording early. Without --leg-ps the ramp is then stretched
# onto the rows that WERE recorded and the work is inflated by
# nsteps/(nstdhdl*floor(nsteps/nstdhdl)); with it, lambda comes off the clock and
# the integral covers the span actually run. Every shipped mdp divides evenly, so
# this is latent -- which is exactly why it needs a test.
LEG_PS, REC_PS = 1050.0, 1000.0
t_short = np.arange(0.0, REC_PS + 1e-9, 100.0)
lam_short = U_SPLIT * t_short / LEG_PS
short_path = os.path.join(work, "short.xvg")
with open(short_path, "w") as f:
  f.write('@ s0 legend "dH/dl fep-lambda = 0.0000"\n')
  for ti, li in zip(t_short, lam_short):
    f.write("%.4f  %.6f\n" % (ti, dvdl(li)))

def integrate_legps(path, l0, l1, leg_ps=None):
  cmd = [sys.executable, INTEGRATE, "-f", path, "--direction", "fwd",
         "--lambda-start", str(l0), "--lambda-end", str(l1)]
  if leg_ps is not None:
    cmd += ["--leg-ps", str(leg_ps)]
  out = subprocess.run(cmd, capture_output=True, text=True)
  return float(out.stdout.split()[0]), out.stderr

truth_short = float(np.trapezoid(dvdl(lam_short), lam_short))
w_stretch, _ = integrate_legps(short_path, 0.0, U_SPLIT)
w_clock, err_short = integrate_legps(short_path, 0.0, U_SPLIT, LEG_PS)
check("--leg-ps integrates the span actually recorded",
      abs(w_clock - truth_short) < 1e-6, "%.4f" % w_clock)
check("without it the ramp is stretched",
      abs(w_stretch / truth_short - LEG_PS / REC_PS) < 0.002,
      "%.4f, i.e. %.4fx" % (w_stretch, w_stretch / truth_short))
check("and a short leg says so on stderr", "short of" in err_short)
w_even, err_even = integrate_legps(os.path.join(work, "A.xvg"), 0.0, U_SPLIT, 15000.0)
check("--leg-ps changes nothing when the rows reach the last step",
      abs(w_even - WA) < 1e-9 and "short of" not in err_even)

# The spans job_fe.run passes come out of the mdps, so the mdps must say what the
# protocol says. Checked on the shipped set rather than a copy.
print("")
print("   lambda spans of the generated mdps:")
MDP_DIR = os.path.join(REPO, "settings", "amber19sb_opc3")
def mdp_span(name):
  vals = {}
  try:
    for line in open(os.path.join(MDP_DIR, name)):
      if "=" not in line:
        continue
      k, v = line.split("=", 1)
      k = k.strip()
      if k in ("init-lambda", "delta-lambda", "nsteps"):
        vals[k] = float(v.strip())
  except OSError:
    return None
  if len(vals) != 3:
    return None
  return (vals["init-lambda"],
          vals["init-lambda"] + vals["delta-lambda"] * vals["nsteps"])

EXPECT = [("npt_fe.mdp", None, None)]
EXPECT = [(l["mdp"], l["lam_from"], l["lam_to"]) for l in P.legs()]
for name, l0, l1 in EXPECT:
  got = mdp_span(name)
  check("   %-18s %.2f -> %.2f" % (name, l0, l1),
        got is not None and abs(got[0] - l0) < 1e-9 and abs(got[1] - l1) < 1e-9,
        "got %s" % (("%.4f -> %.4f" % got) if got else "no lambda block"))

#------------------------------------------------------
# 2. the staged estimate, end to end through groscore_fe.py
#------------------------------------------------------

print("")
print("2. staged scoring end to end")

rng = np.random.default_rng(7)
NCYC = 24

def crooks(dG, diss, n):
  """A forward/reverse work pair satisfying Crooks by construction: Gaussian at
  the linear-response width sigma^2 = 2 RT * dissipation. The reverse is returned
  IN ITS OWN SIGN, which is what job_fe.run stores."""
  s = math.sqrt(2 * RT * diss)
  return rng.normal(dG + diss, s, n), rng.normal(-dG + diss, s, n)

# Stage A carries most of the free energy and most of the dissipation; B is the
# easy tail. Both stages overlap; their SUM does not, which is the case the split
# exists for and the one the test has to reproduce.
dGA, dGB, dGi = 62.0, 18.0, -5.0
WA_f, WA_r = crooks(dGA, 8.0 * RT, NCYC)
WB_f, WB_r = crooks(dGB, 4.0 * RT, NCYC)
Wi, Wrm = crooks(dGi, 3.0 * RT, NCYC)

# job_fe.run stores the pull channel PRE-sign (groscore_fe.py applies SIGN_PULL_*:
# -1 forward, +1 reverse) and the dhdl channel already physical, so the works are
# split between the two channels and the pull half inverted. That way the row
# exercises the sign round trip instead of assuming it.
SIGN_PULL_FWD, SIGN_PULL_REV = -1.0, +1.0
def channels(w, sign):
  dhdl = 0.4 * w
  return (w - dhdl) / sign, dhdl

pA_f, dA_f = channels(WA_f, SIGN_PULL_FWD)
pB_f, dB_f = channels(WB_f, SIGN_PULL_FWD)
pB_r, dB_r = channels(WB_r, SIGN_PULL_REV)
pA_r, dA_r = channels(WA_r, SIGN_PULL_REV)

run = os.path.join(work, "run")
os.makedirs(os.path.join(run, "results_fe.d"))
for sid in ("STAGED", "LEGACY", "PEND"):
  os.makedirs(os.path.join(run, sid))

with open(os.path.join(run, "results_analytical.gs"), "w") as f:
  for sid in ("STAGED", "LEGACY", "PEND"):
    f.write("%s   -34.5\n" % sid)

for c in range(NCYC):
  with open(os.path.join(run, "results_fe.d", "STAGED_c%d.gs" % (c + 1)), "w") as f:
    f.write("STAGED %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.3f\n"
            % (c + 1, Wi[c], pA_f[c], dA_f[c], pB_f[c], dB_f[c],
               pB_r[c], dB_r[c], pA_r[c], dA_r[c], Wrm[c], 1.5))
  # The same cycle as a pre-staging row: one ramp, i.e. the stages summed.
  lp_f, ld_f = channels(WA_f[c] + WB_f[c], SIGN_PULL_FWD)
  lp_r, ld_r = channels(WA_r[c] + WB_r[c], SIGN_PULL_REV)
  with open(os.path.join(run, "results_fe.d", "LEGACY_c%d.gs" % (c + 1)), "w") as f:
    f.write("LEGACY %d %.6f %.6f %.6f %.6f %.6f %.6f %.3f\n"
            % (c + 1, Wi[c], lp_f, ld_f, lp_r, ld_r, Wrm[c], 1.5))

with open(os.path.join(run, "sp.gs"), "w") as f:
  f.write("# id file chainA chainB\n")
  for sid in ("STAGED", "LEGACY", "PEND"):
    f.write("%s %s.pdb A B\n" % (sid, sid))
open(os.path.join(run, "results_0.gs"), "w").close()
with open(os.path.join(run, "run_config.gs"), "w") as f:
  f.write("numruns %d\n" % NCYC)

proc = subprocess.run([sys.executable, os.path.join(REPO, "groscore_fe.py"),
                       "--restart", "-n", str(NCYC)],
                      cwd=run, capture_output=True, text=True,
                      env=dict(os.environ, MPLBACKEND="Agg"))
scores = os.path.join(run, "scores_fe.gs")
if not os.path.isfile(scores):
  print((proc.stdout + proc.stderr)[-3000:])
  sys.exit("groscore_fe.py wrote no scores_fe.gs")

lines = [l.rstrip("\n") for l in open(scores)]
header = [l for l in lines if l.startswith("# Structure_ID")][0].split()[1:]
body = {l.split()[0]: l.split() for l in lines if not l.startswith("#")}

def col(sid, name):
  v = body[sid][header.index(name)]
  try:
    return float(v)
  except ValueError:
    return v

# header keeps its leading Structure_ID, so its indices are the row's indices.
ncol = len(header)
for sid in ("STAGED", "LEGACY", "PEND"):
  check("%s row is as wide as the header" % sid, len(body.get(sid, [])) == ncol,
        "%d vs %d" % (len(body.get(sid, [])), ncol))

a, b = col("STAGED", "dG_unbA_bar"), col("STAGED", "dG_unbB_bar")
u, s1 = col("STAGED", "dG_unbind_bar"), col("STAGED", "dG_unbind_1s_bar")
print("   dG_A %.2f (true %.1f)   dG_B %.2f (true %.1f)   staged %.2f (true %.1f)"
      % (a, dGA, b, dGB, u, dGA + dGB))
check("BAR recovers stage A", abs(a - dGA) < 3.0)
check("BAR recovers stage B", abs(b - dGB) < 3.0)
check("dG_unbind_bar is exactly the stage sum", abs(u - (a + b)) < 1e-6)
check("the staged sum recovers the total", abs(u - (dGA + dGB)) < 4.0)
check("the one-shot column is suppressed for want of overlap",
      not np.isfinite(s1), "the case the split exists for")

check("legacy rows leave the per-stage columns empty",
      not np.isfinite(col("LEGACY", "dG_unbA_bar")))
lu, l1 = col("LEGACY", "dG_unbind_bar"), col("LEGACY", "dG_unbind_1s_bar")
check("legacy dG_unbind is its own one-shot value",
      (not np.isfinite(lu) and not np.isfinite(l1)) or abs(lu - l1) < 1e-9)
check("a legacy leg with no overlap says so in Note",
      "BAR_NO_OVERLAP" in str(col("LEGACY", "Note")))

# avg is defined whatever the overlap, so it is what witnesses the additivity
# identity on the scoring side: stage-summed and unstaged must be the same number.
check("stage-summed avg equals the unstaged avg",
      abs(col("STAGED", "dG_unbind_avg") - col("LEGACY", "dG_unbind_avg")) < 1e-6,
      "%.4f vs %.4f" % (col("STAGED", "dG_unbind_avg"),
                        col("LEGACY", "dG_unbind_avg")))

ci_a, ci_b = col("STAGED", "dG_unbA_bar_CI"), col("STAGED", "dG_unbB_bar_CI")
ci_u = col("STAGED", "dG_unbind_bar_CI")
quad = math.sqrt(ci_a ** 2 + ci_b ** 2)
check("the joint CI is not the quadrature of the stage CIs",
      np.isfinite(ci_u) and abs(ci_u - quad) > 1e-3,
      "joint %.3f vs quadrature %.3f: the difference IS the covariance"
      % (ci_u, quad))
check("dG_bind and its CI are finite",
      np.isfinite(col("STAGED", "dGbind_bar")) and
      np.isfinite(col("STAGED", "dGbind_bar_CI")))

#------------------------------------------------------
# 3. the completeness gate covers every work in the row
#------------------------------------------------------
#
# job_fe.run decides a cycle is finished, writes its .done marker and lets the
# structure archive on the strength of this one awk expression. It used to check
# fields 3-8, which covered the whole of a 9-field row and then silently stopped
# covering the last four works when the ramp was split and the row grew to 13. A
# cycle that lost its rebinding stage A was marked complete, counted toward the
# requested total, and then dropped by read_works for those very NaNs, which
# loses it for good once the structure is tarred. The gate is extracted from the
# script itself rather than restated here, so this cannot pass against a copy.

print("")
print("3. the cycle-completeness gate")

job = open(os.path.join(REPO, "job_fe.run")).read()
m = re.search(r"complete_row\(\)\s*\{(.*?)\n\s*\}", job, re.S)
check("complete_row() is defined in job_fe.run", bool(m))
check("the runner takes its width from fe_protocol, not from a literal",
      "fe_protocol.py" in job and not re.search(r"^FE_RESULT_NF=\d", job, re.M))

if m:
  awk_src = re.search(r"'(.*?)'", m.group(1), re.S).group(1)
  width = P.result_nf()
  nworks = len(P.works())
  check("the protocol's row width matches its own work list",
        width == nworks + 3, "%d fields, %d works" % (width, nworks))

  def gate(row):
    p = os.path.join(work, "row.gs")
    open(p, "w").write(row + "\n")
    return subprocess.run(["awk", "-v", "want=%d" % width, awk_src, p]).returncode == 0

  full = "X 1 " + " ".join("%g" % (i + 1) for i in range(nworks)) + " 1.5"
  check("a complete row is accepted", gate(full))
  check("a NaN RMSD alone is still accepted",
        gate("X 1 " + " ".join("%g" % (i + 1) for i in range(nworks)) + " NaN"))
  # EVERY work position must be seen. The gate once checked only fields 3-8,
  # which covered a 9-field row exactly and silently stopped covering the tail the
  # moment the row grew. The range is derived now, so this loop grows with it.
  for pos in range(3, 3 + nworks):
    f = full.split()
    f[pos - 1] = "NaN"
    check("NaN in field %-2d is rejected%s"
          % (pos, "  (outside the old 3-8 window)" if pos > 8 else ""),
          not gate(" ".join(f)))
  check("a narrower row from an earlier protocol is recomputed",
        not gate("X 1 1 2 3 4 5 6 1.5"))
  check("a truncated row is recomputed",
        not gate("X 1 " + " ".join("%g" % (i + 1) for i in range(nworks))))

shutil.rmtree(work, ignore_errors=True)

print("")
if failures:
  print("FAILED: " + ", ".join(failures))
  sys.exit(1)
print("All staged-ramp tests passed.")
