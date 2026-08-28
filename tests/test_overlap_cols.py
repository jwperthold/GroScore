#!/usr/bin/env python3
"""scores_fe.gs must carry the overlap that says when to distrust its own BAR.

BAR is the headline and it is the estimator with a measured overlap bias. Over
three protocols on 2KTF the BAR-minus-avg gap ran +5.52 / +3.56 / +1.55 at mean
ramp overlaps of 35 / 44 / 43%, r = -0.82, always in the same direction: BAR reads
LESS negative when the histograms barely meet. Per run over sixteen runs, BAR
carries a dissipation slope of +0.201 +- 0.122 where avg and cgi are flat.

None of that is visible in the free energies or in their intervals. It lived in the
setup and slurm logs, which nobody is going to read for hundreds of structures, so
the two numbers that predict it belong in the row.

  Overlap_mean_pct   tracks the BIAS
  Overlap_min_pct    predicts the FAILURE -- one empty channel nan's the estimate,
                     and it reads 0.00 on exactly the run that lost BAR

Standalone, no pytest: python3 tests/test_overlap_cols.py
"""
import os, re, subprocess, sys, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(ROOT, "groscore_fe.py")
src = open(FE).read()
failures = []


def check(name, ok, detail=""):
    print("  %-68s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


print("\n[1] the columns exist and sit beside the other diagnostics")
m = re.search(r'"dG_release  ([^"]*?)Ncycles  Note"', src)
check("the tail of the column list is readable", m is not None)
tail = m.group(1).split() if m else []
for c in ("Overlap_mean_pct", "Overlap_min_pct", "RMSD_mean_A", "RMSD_max_A"):
    check("%s is a column" % c, c in tail, str(tail))
if tail:
    check("overlap comes before the RMSD pair, with the other per-row diagnostics",
          tail.index("Overlap_mean_pct") < tail.index("RMSD_mean_A"), str(tail))
check("and both are written into the row, in header order",
      re.search(r"cell\(r\['ovl_mean'\]\), cell\(r\['ovl_min'\]\)", src) is not None)

print("\n[2] they are computed over EVERY BAR channel, not just the ramp")
blk = src[src.index("ov = []"):]
blk = blk[:blk.index("# Rebinding sanity check")]
check("bound sub-legs and ramp stages both feed it",
      "list(bound_w) + list(stage_w)" in blk, blk[:200])
check("as a percentage of 2n, matching est.overlap_count",
      "100.0 * c / (2 * len(_f))" in blk)
check("mean and min, not one or the other",
      "np.mean(ov)" in blk and "np.min(ov)" in blk)
check("an empty channel list gives nan rather than raising",
      "if ov else float('nan')" in blk)

print("\n[3] end to end on real works, including the run that lost BAR")
CASES = [("test29", -39.65, False), ("test27", None, True)]
for name, want_bar, expect_zero_min in CASES:
    d = os.path.join(ROOT, name)
    if not os.path.isdir(os.path.join(d, "results_fe.d")):
        print("  %-68s skip" % ("%s not present" % name))
        continue
    w = tempfile.mkdtemp(prefix="ovl_")
    try:
        shutil.copytree(os.path.join(d, "results_fe.d"), os.path.join(w, "results_fe.d"))
        for extra in ("results_analytical.d", "results_qc.d"):
            if os.path.isdir(os.path.join(d, extra)):
                shutil.copytree(os.path.join(d, extra), os.path.join(w, extra))
        for f in ("results_0.gs", "run_config.gs", "sp.gs"):
            if os.path.isfile(os.path.join(d, f)):
                shutil.copy(os.path.join(d, f), w)
        # the scorer skips a structure whose directory is absent (NODIR)
        os.makedirs(os.path.join(w, "2KTF"), exist_ok=True)
        subprocess.run([sys.executable, FE, "--sequential"], cwd=w,
                       capture_output=True, text=True, timeout=900)
        L = [l.rstrip("\n") for l in open(os.path.join(w, "scores_fe.gs"))]
        h = [l for l in L if l.startswith("#") and "Structure_ID" in l][0].lstrip("#").split()
        row = [l for l in L if not l.startswith("#") and l.strip()][0].split("\t")
        D = dict(zip(h, row))
        check("%s: header and row have the same width (%d)" % (name, len(h)),
              len(h) == len(row), "%d vs %d" % (len(h), len(row)))
        mean, mn = float(D["Overlap_mean_pct"]), float(D["Overlap_min_pct"])
        check("%s: overlap columns are populated (mean %.1f, min %.1f)"
              % (name, mean, mn), mean == mean and mn == mn)
        check("%s: min <= mean, as they must be" % name, mn <= mean + 1e-9,
              "%s vs %s" % (mn, mean))
        check("%s: both inside 0-100%%" % name, 0 <= mn and mean <= 100)
        if expect_zero_min:
            check("%s lost BAR and its min overlap is exactly 0" % name, mn == 0.0,
                  str(mn))
            check("  and the Note says which channel", "NO_OVERLAP" in D["Note"],
                  D["Note"])
            check("  while the mean alone would NOT have flagged it (%.0f%%)" % mean,
                  mean > 20, str(mean))
        else:
            check("%s kept BAR and its min overlap is above 0" % name, mn > 0, str(mn))
        if want_bar is not None:
            check("%s: the free energy is unchanged by adding the columns (%.2f)"
                  % (name, want_bar),
                  abs(float(D["dGbind_bar"]) - want_bar) < 0.005, D["dGbind_bar"])
    finally:
        shutil.rmtree(w, ignore_errors=True)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all overlap-column checks passed")
