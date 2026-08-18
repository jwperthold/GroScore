#!/usr/bin/env python3
"""The ensemble references: circular means, pooled probes, and the split bound leg.

Three things here have silent failure modes, which is why they are pinned:

  * A dihedral averaged arithmetically across the +/-180 wrap gives a reference
    pointing the OTHER WAY and no error. That is the exact shape of the
    dihedral_deg sign bug that voided every work integral produced before b4c267f.
  * A result row whose width no rule recognises is DROPPED by read_works without a
    word, so a protocol change that widens the row can turn a finished run into
    PENDING. The current width must parse and older widths must still parse.
  * The bound leg is now split, and its sub-leg works must pair forward-to-reverse
    across the centre of the row. Getting the pairing backwards is arithmetically
    silent and thermodynamically wrong.

Standalone, no pytest: python3 tests/test_ensemble_refs.py
"""
import os, sys, math, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "utils"))
sys.path.insert(0, REPO)

failures = []


def check(name, ok, detail=""):
    print("  %-64s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


# make_boresch.py parses argv at import, so the routine is loaded from source
# rather than imported. Reading it out of the shipped file is the point: a test
# that reimplemented the mean would pass while the shipped one was wrong.
src = open(os.path.join(REPO, "utils", "make_boresch.py")).read()
i = src.index("def circular_mean_deg")
j = src.index("\ndef ", i + 1)
ns = {"np": np, "math": math}
exec(compile(src[i:j], "circular_mean_deg", "exec"), ns)
circular_mean_deg = ns["circular_mean_deg"]

print("\n[1] circular mean, where the arithmetic one is wrong")
m, sd = circular_mean_deg([179.0, -179.0])
check("+179 and -179 average to 180, not 0", abs(abs(m) - 180.0) < 1e-6,
      "got %.4f" % m)
m, _ = circular_mean_deg([170.0, -170.0])
check("+170 and -170 average to 180", abs(abs(m) - 180.0) < 1e-6, "got %.4f" % m)
m, _ = circular_mean_deg([10.0, -10.0])
check("+10 and -10 still average to 0", abs(m) < 1e-6, "got %.4f" % m)
m, _ = circular_mean_deg([30.0, 60.0, 90.0])
check("an unwrapped run averages as usual", abs(m - 60.0) < 1e-6, "got %.4f" % m)
m, sd = circular_mean_deg([45.0] * 20)
check("no spread on identical input", sd < 1e-6, "got %.4g" % sd)
_, sd_tight = circular_mean_deg([40.0, 45.0, 50.0])
_, sd_wide = circular_mean_deg([0.0, 80.0, 160.0])
check("spread grows with disagreement", sd_tight < sd_wide,
      "%.2f vs %.2f" % (sd_tight, sd_wide))
# A PERFECTLY uniform circle has no mean direction at all, and nan is the honest
# report -- a finite number there would be an artefact of the arithmetic.
_, sd_none = circular_mean_deg([0.0, 120.0, 240.0])
check("a uniform circle reports no spread rather than inventing one",
      math.isnan(sd_none), "%.4g" % sd_none)

print("\n[2] the shipped code uses it for dihedrals and NOT for the angles")
seg = src[src.index("def ensemble_boresch_geometry"):]
seg = seg[:seg.index("\ndef ", 1)] if "\ndef " in seg[1:] else seg
for key in ("phA", "phB", "phC"):
    check("%s goes through circular_mean_deg" % key,
          ('("%s"' % key) in seg and "circular_mean_deg" in seg)
check("thA/thB use the arithmetic mean, being confined to [0,180]",
      'ref = {"r": float(r_t.mean())' in seg and '"thA": float(thA_t.mean())' in seg)
check("the ligand side is minimum-imaged before any angle is taken",
      "min_image_vec(com[\"L1\"] - com[\"P3\"], M)" in seg)

# The ensemble and the snapshot must define the six coordinates IDENTICALLY. They
# did not at first: angle_deg(a, b, c) is the angle at b, and the ensemble had the
# first two arguments swapped, which moved theta_A and theta_B by 66-78 degrees
# while the dihedrals -- whose order was right -- stayed within 5. Only the
# printed shift gave it away, so the order is asserted here rather than trusted.
import re as _re
def _args(text, pat):
    m = _re.search(pat, text)
    return tuple(x.strip() for x in m.group(1).split(",")) if m else None
snap_thA = _args(src, r"ref_thA = angle_deg\(([^)]*)\)")
snap_thB = _args(src, r"ref_thB = angle_deg\(([^)]*)\)")
ens_thA = _args(seg, r"thA_t = np\.array\(\[angle_deg\(([^)]*)\)")
ens_thB = _args(seg, r"thB_t = np\.array\(\[angle_deg\(([^)]*)\)")
def _strip(t):
    return tuple(x.replace("[i]", "").rstrip("c") for x in t) if t else None
check("theta_A has the same argument order in both definitions",
      _strip(snap_thA) == _strip(ens_thA), "%s vs %s" % (snap_thA, ens_thA))
check("theta_B has the same argument order in both definitions",
      _strip(snap_thB) == _strip(ens_thB), "%s vs %s" % (snap_thB, ens_thB))
for nm, series in (("phA", "phA_t"), ("phB", "phB_t"), ("phC", "phC_t")):
    sn = _args(src, r"ref_%s = dihedral_deg\(([^)]*)\)" % nm)
    en = _args(seg, r"%s = np\.array\(\[dihedral_deg\(([^)]*)\)" % series)
    check("%s has the same argument order in both definitions" % nm,
          _strip(sn) == _strip(en), "%s vs %s" % (sn, en))

print("\n[2b] the pull readback checks the CONVENTION, not the reference value")
# The readback grompps args.input and asks GROMACS for the six coordinates. Once
# the references became ensemble means they no longer equal any single frame, so a
# readback that compares GROMACS against the EMITTED values fires on every setup.
# It did: test13's six "mismatches" were the ensemble shift to four decimals, and
# the whole run died at BORESCH_FAIL with 50 cycle tasks already queued. The check
# must compare against the snapshot geometry of the same structure.
check("the snapshot geometry is kept under its own name",
      "SNAP_REF = dict(" in src)
_rb = src[src.index("_read = verify_pull_block("):]
check("the readback compares against SNAP_REF",
      "want = [SNAP_REF[" in _rb, _rb[:0])
check("and NOT against the emitted references",
      "want = [ref_r, ref_thA" not in _rb)
check("the emitted value is still printed, for context",
      "emitted = [ref_r, ref_thA" in _rb and "(emitted %10.4f)" in _rb)
check("verify_pull_block still measures the reference structure itself",
      '"-c", args.input' in src)

print("\n[3] probe replicas are pooled, and the discard is derived")
import fe_protocol as P
check("N_PROBES is more than one", P.N_PROBES > 1, str(P.N_PROBES))
check("half of each replica is discarded", abs(P.PROBE_SKIP_FRAC - 0.5) < 1e-9)
check("TRAJ_SKIP_PS is derived from the probe length, not written twice",
      "TRAJ_SKIP_PS = P.NPT_INIT_PS * P.PROBE_SKIP_FRAC" in src)
check("--traj and --tpr both take a list",
      "'--traj', type=str, nargs='+'" in src and "'--tpr', type=str, nargs='+'" in src)
check("a single tpr is broadcast over several trajectories",
      "args.tpr = args.tpr * len(args.traj)" in src)
check("the backbone trajectory concatenates every replica",
      "np.concatenate(out, axis=0)" in src)
mb = open(os.path.join(REPO, "utils", "measure_box.py")).read()
check("measure_box takes the running max over every replica",
      "nargs='+'" in mb and "for traj, tpr in zip(args.traj, args.tpr)" in mb)

print("\n[4] the split bound leg reads back as it was written")
sys.argv = [sys.argv[0]]
import groscore_fe as G

check("the protocol declares more than one bound sub-leg", P.n_bound() > 1,
      str(P.n_bound()))
check("its lambda spans are contiguous 0 -> 1",
      abs(P.BOUND[0][1]) < 1e-12 and abs(P.BOUND[-1][2] - 1.0) < 1e-12)

# A synthetic row of the current width, with every field distinct, so a pairing
# error cannot hide behind equal values.
nf = P.result_nf()
row = ["X", "1"] + ["%d" % (i + 1) for i in range(nf - 3)] + ["2.5"]
import tempfile, shutil
d = tempfile.mkdtemp(prefix="ensrefs_")
open(os.path.join(d, "X_c1.gs"), "w").write(" ".join(row) + "\n")
w = G.read_works(os.path.join(d, "nope.gs"), d)
check("a current-width row parses at all", "X" in w, "width %d" % nf)
if "X" in w:
    c = sorted(w["X"])[0]
    nb, nst = len(c[1]), c[3]
    check("with the right sub-leg and stage counts",
          nb == P.n_bound() and nst == P.n_stages(), "%d/%d" % (nb, nst))
    nw = 2 * nb + 4 * nst
    # sub-leg i must pair field i with field nw-1-i, counting from the far end
    ok = all(c[1][i] == (float(i + 1), float(nw - i)) for i in range(nb))
    check("bound sub-legs pair across the centre of the row", ok, str(c[1]))
    check("the RMSD is the last field", abs(c[2] - 2.5) < 1e-9, str(c[2]))
shutil.rmtree(d, ignore_errors=True)

print("\n[5] every older row width still parses")
for nb_old, nst_old in ((1, 1), (1, 2), (1, 3)):
    nf_old = 3 + 2 * nb_old + 4 * nst_old
    d = tempfile.mkdtemp(prefix="ensrefs_old_")
    row = ["Y", "1"] + ["%d" % (i + 1) for i in range(nf_old - 3)] + ["1.5"]
    open(os.path.join(d, "Y_c1.gs"), "w").write(" ".join(row) + "\n")
    w = G.read_works(os.path.join(d, "nope.gs"), d)
    ok = "Y" in w and len(sorted(w["Y"])[0][1]) == nb_old \
         and sorted(w["Y"])[0][3] == nst_old
    check("legacy %d-field row (%d bound, %d stages)" % (nf_old, nb_old, nst_old), ok)
    shutil.rmtree(d, ignore_errors=True)

print("\n[6] the holds are what the measurement said they should be")
check("mid-ramp holds are shorter than the handoffs",
      P.RAMP_HOLD_PS < P.HANDOFF_HOLD_PS,
      "%g vs %g" % (P.RAMP_HOLD_PS, P.HANDOFF_HOLD_PS))
holds = {l["name"]: l["ps"] for l in P.legs() if l["kind"] == "hold"}
check("holdfwd0 and holdrev0 keep the handoff length",
      holds.get("holdfwd0") == P.HANDOFF_HOLD_PS
      and holds.get("holdrev0") == P.HANDOFF_HOLD_PS, str(holds))
check("every hold is at least 20x the 16 ps relaxation time measured on the ramp",
      min(holds.values()) >= 20 * 16.0, "shortest %g ps" % min(holds.values()))

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all ensemble-reference checks passed")
