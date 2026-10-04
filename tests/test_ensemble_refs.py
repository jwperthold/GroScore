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
  * Since v1.5 the snapshot and the ensemble share one definition of the six
    Boresch coordinates, built from minimum-imaged vectors between COMs taken about
    each group's pbcatom, and the readback compares what the legs CARRY with
    GROMACS as well as the snapshot: GROMACS's reference column, i.e. the
    pull-coordN-init it read, against the references computed and against its own
    measurement of the pose. v1.4 imaged one vector of the seven and checked the
    emitted values against nothing, and three setups of the second HADDOCK pass left
    with references 103-115 degrees from the bound pose; the first v1.5 round checked
    the Python values, not the mdps (review R3).
  * The reverse bound legs start from the forward handoff, so holdrev0 is gone and
    holdfwd0 alone keeps the handoff length.

Standalone, no pytest: python3 tests/test_ensemble_refs.py
"""
import os, re, sys, math, importlib.util
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
def _fn(name):
    """A top-level def of make_boresch.py, up to the next top-level def; empty if
    the file has none, so a missing routine fails its checks instead of the run."""
    i = src.find("\ndef %s(" % name) + 1
    if i == 0:
        return ""
    j = src.find("\ndef ", i)
    return src[i:j if j > 0 else len(src)]
seg = _fn("ensemble_boresch_geometry")
check("phA/phB/phC go through circular_mean_deg",
      'for key in ("phA", "phB", "phC"):' in seg
      and "circular_mean_deg(s[key])" in seg)
check("thA/thB use the arithmetic mean, being confined to [0,180]",
      'ref = {"r": float(s["r"].mean())' in seg and '"thA": float(s["thA"].mean())' in seg)

# v1.5: EVERY inter-group vector is minimum-imaged, and every COM is taken about
# its group's pbcatom, which is how GROMACS evaluates the coordinates. v1.4 imaged
# only P3->L1 and carried the rest of each triad along as raw offsets of -pbc whole
# frames, which put 1AKJ's, 1E6J's and 1IQD's references 103-115 degrees from any
# pose the complex adopts (digest qc.txt boresch_reference_corrupted).
ser = _fn("boresch_series")
check("boresch_series minimum-images every vector it uses",
      "min_image_vec(com[p[1]] - com[p[0]], M) for p in BORESCH_VECTORS" in ser)
m = re.search(r"^BORESCH_VECTORS = (\(.*?\))\n(?=\S)", src, re.M | re.S)
vecs = set(eval(m.group(1))) if m else set()
want = {("P3", "L1"), ("P3", "P2"), ("L1", "P3"), ("L1", "L2"),
        ("P1", "P2"), ("P2", "P3"), ("L2", "L3")}
check("and those are exactly the seven directed pairs GROMACS uses",
      vecs == want, str(sorted(vecs)))
check("the ensemble takes its COMs about each group's pbcatom",
      "group_coms(groups, X, M, slices)" in seg
      and "gromacs_com(" in _fn("group_coms") and "pbcatom_of(a)" in _fn("group_coms"))

# The ensemble and the snapshot must define the six coordinates IDENTICALLY. They
# did not at first: angle_deg(a, b, c) is the angle at b, and the ensemble had the
# first two arguments swapped, which moved theta_A and theta_B by 66-78 degrees
# while the dihedrals, whose order was right, stayed within 5. Both now go
# through boresch_series, and boresch_series is pinned NUMERICALLY to the point
# definitions the readback verified against GROMACS (angle_deg(P2,P3,L1),
# angle_deg(P3,L1,L2), dihedral_deg(P1,P2,P3,L1), (P2,P3,L1,L2), (P3,L1,L2,L3)).
check("the snapshot goes through boresch_series",
      "boresch_series(com, M)" in _fn("snapshot_boresch_geometry"))
check("and so does the ensemble", "boresch_series(group_coms(" in seg)
import re as _re
ns2 = {"np": np, "math": math}
for name in ("angle_deg", "dihedral_deg", "to_brick", "min_image_vec", "boresch_series"):
    exec(compile(_fn(name), name, "exec"), ns2)
exec(compile("_IMG = " + _re.search(r"^_IMG = (.*?)\n(?=\S)", src, _re.M | _re.S).group(1),
             "_IMG", "exec"), ns2)
ns2["BORESCH_VECTORS"] = tuple(sorted(want))
rng = np.random.default_rng(11)
worst = 0.0 if "boresch_series" in ns2 else float("inf")
for _ in range(50 if "boresch_series" in ns2 else 0):
    pts = {r: rng.normal(scale=0.8, size=3) for r in ("P1", "P2", "P3", "L1", "L2", "L3")}
    com = {r: v[None, :] for r, v in pts.items()}
    s = ns2["boresch_series"](com, np.diag([30.0, 30.0, 30.0])[None])
    A, D, p = ns2["angle_deg"], ns2["dihedral_deg"], pts
    ref = {"r": np.linalg.norm(p["L1"] - p["P3"]),
           "thA": A(p["P2"], p["P3"], p["L1"]), "thB": A(p["P3"], p["L1"], p["L2"]),
           "phA": D(p["P1"], p["P2"], p["P3"], p["L1"]),
           "phB": D(p["P2"], p["P3"], p["L1"], p["L2"]),
           "phC": D(p["P3"], p["L1"], p["L2"], p["L3"])}
    for k in ref:
        worst = max(worst, abs(float(s[k][0]) - ref[k]))
check("boresch_series equals the verified point definitions on 50 random poses "
      "(worst %.1e)" % worst, worst < 1e-9, "%.3g" % worst)

print("\n[2b] the pull readback checks the CONVENTION and the POSE, separately")
# The readback grompps args.input and asks GROMACS for the six coordinates. Once
# the references became ensemble means they no longer equal any single frame, so a
# readback that compares GROMACS against the EMITTED values at the convention
# tolerance fires on every setup. It did: test13's six "mismatches" were the
# ensemble shift to four decimals, and the whole run died at BORESCH_FAIL with 50
# cycle tasks already queued. So the convention is checked against the snapshot.
# But v1.4 then compared the emitted values with NOTHING, and 1AKJ left setup with
# phi_A emitted 138.7 against a snapshot of -118.4 printed beside it. Since v1.5 what
# the legs carry is compared with GROMACS, at the pose gate's tolerance; the
# behaviour is pinned by tests/test_boresch_pbc.py [7], the wiring here.
check("the snapshot geometry is kept under its own name",
      "SNAP_REF, _snap_p3l1 = snapshot_boresch_geometry(" in src)
_rb = src[src.index("_read = verify_pull_block("):]
check("the readback hands GROMACS's measured values and reference column over, "
      "with SNAP_REF and the computed references",
      "readback_verdict(got, carried, SNAP_REF, _first)" in _rb)
rv = _fn("readback_verdict")
check("convention against the snapshot, written against the computed reference, "
      "pose against GROMACS",
      "d, dw, dp = abs(wrap180(g - s)), abs(wrap180(w - i)), abs(wrap180(w - g))" in rv
      and "tol, rtol, ptol = CHECK_TOL_DEG, READ_TOL_DEG, POSE_GATE_DEG + CHECK_TOL_DEG" in rv)
check("each failure aborts with its own code",
      "PULL_READBACK_MISMATCH" in _rb and "PULL_INIT_MISMATCH" in _rb
      and "BORESCH_REFERENCE_NOT_A_POSE" in _rb)
check("the written value is printed beside GROMACS's", "(written %10.4f" in rv)
check("GROMACS is asked for its reference column", "pull-print-ref-value = yes" in src)
check("every leg's mdp is read back as text before GROMACS is asked",
      0 <= src.find("_problems = check_written_legs(_intended)")
      < src.find("_read = verify_pull_block(first_stage_mdp)"))
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
check("measure_box pools the frames of every replica",
      "nargs='+'" in mb and "for traj, tpr in zip(args.traj, args.tpr)" in mb)

print("\n[4] the split bound leg reads back as it was written")
sys.argv = [sys.argv[0]]
import tempfile, shutil
# groscore_fe reads and writes run_config.gs in the working directory when it is
# imported. Imported from the caller's directory, this test wrote one there (into
# the repository root when run as python3 tests/test_ensemble_refs.py), and a
# stale one there handed the import its settings: the one in the repository root
# on 2026-10-04 said sum_k 25000, not the default 12500. So the import runs in a
# scratch directory, and the caller's directory is restored.
_caller = os.getcwd()
_rc_path = os.path.join(_caller, "run_config.gs")
_rc_before = open(_rc_path, "rb").read() if os.path.isfile(_rc_path) else None
_scratch = tempfile.mkdtemp(prefix="ensrefs_import_")
os.chdir(_scratch)
try:
    import groscore_fe as G
finally:
    os.chdir(_caller)
    shutil.rmtree(_scratch, ignore_errors=True)
check("importing groscore_fe leaves the caller's run_config.gs as it was",
      (open(_rc_path, "rb").read() if os.path.isfile(_rc_path) else None) == _rc_before,
      "absent before" if _rc_before is None else "present before")

check("the protocol declares more than one bound sub-leg", P.n_bound() > 1,
      str(P.n_bound()))
check("its lambda spans are contiguous 0 -> 1",
      abs(P.BOUND[0][1]) < 1e-12 and abs(P.BOUND[-1][2] - 1.0) < 1e-12)

# A synthetic row of the current width, with every field distinct, so a pairing
# error cannot hide behind equal values.
nf = P.result_nf()
row = ["X", "1"] + ["%d" % (i + 1) for i in range(nf - 3)] + ["2.5"]
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
# v1.5 removed holdrev0: the reverse bound legs branch from the end of holdfwd0, the
# forward handoff, because the 1 ns holdrev0 removed a median 8% of a 14 kJ/mol
# strain gap the rebound interface carried at u = 0 (bm_fe_amber_2/_ana/rlx_handoff.py).
check("holdfwd0 keeps the handoff length",
      holds.get("holdfwd0") == P.HANDOFF_HOLD_PS, str(holds))
check("and no leg named holdrev0 exists any more",
      "holdrev0" not in [l["name"] for l in P.legs()])
check("every hold is at least 20x the 16 ps relaxation time measured on the ramp",
      min(holds.values()) >= 20 * 16.0, "shortest %g ps" % min(holds.values()))

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all ensemble-reference checks passed")
