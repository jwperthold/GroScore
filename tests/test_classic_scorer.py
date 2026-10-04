#!/usr/bin/env python3
"""groscore.py, the classic scorer and submitter: what it reads and what it reports.

The findings behind each section are in v15_work/digest_audit (scorer and runner
lenses of the v1.4 audit, every one reproduced by a verifier).

  [1] CI95 (SCORER-3). The column is the paired Student t interval of the mean
      per-cycle score s_i = (pull_i + push_i)/2: nan for one cycle (v1.4: 0.0),
      t(0.975, 1) * SE = 25.4 for two cycles whose s differ by 4, and
      t(0.975, 4) * SE = 5.6 for five (v1.4: 1.96 times the SD of an unseeded
      bootstrap that resampled pulls and pushes independently, an 88% interval at
      five cycles). Pull and push anti-correlated within their cycles give equal
      s_i and a CI95 of exactly 0.0, which only a paired interval can give.
  [2] Stage 0 (SCORER-6). The LAST status line of a structure decides: a later
      FAILED or SETUP_RUNNING stops it being scored from rows left by an earlier
      setup, a later OK lifts an old failure, a malformed or torn line is skipped
      (v1.4: one single-token line crashed the reader, no score file at all).
  [3] Per-cycle results (runner R3, SCORER-4). results.d/<ID>_c<N>.gs rows are
      read; once a structure has a row or a <ID>_setup.gs marker there, its
      legacy results_<n>.gs rows are never read, so works and RMSDs of an earlier
      setup in the same run directory cannot fill its cycles. A structure without
      either is scored from the legacy files as before, except that in a push file
      with three-field rows a two-field row is an empty work, not the RMSD as the
      work. Push files from before the RMSD column (two fields throughout) still
      read. An ID that is a prefix of another ID never picks up the other's rows.
  [4] Submission (runner R4 and R6, SCORER-5). The array covers only what is
      submitted: an archive is submitted only with --inject-job-run (its old
      job.run could not add cycles and re-integrated its finished ones), the
      generated script exports GROSCORE_NUMCYCLES = -n, removes a tarball only
      after a complete extraction, and never runs job.run outside a structure
      directory.

Every expected number is written out, never computed from the code under test.

Standalone, no pytest: python3 tests/test_classic_scorer.py
"""
import os, shutil, subprocess, sys, tarfile, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROSCORE = os.path.join(ROOT, "groscore.py")
failures = []


def check(name, ok, detail=""):
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:240]))
    if not ok:
        failures.append(name)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def run_groscore(d, *extra, env=None):
    return subprocess.run([sys.executable, GROSCORE] + list(extra), cwd=d,
                          capture_output=True, text=True, env=env)


def rows(path):
    """{struct_id: fields} of a score file."""
    out = {}
    if os.path.isfile(path):
        for line in open(path):
            if not line.startswith("#") and line.strip():
                tok = line.split()
                out[tok[0]] = tok
    return out


def percycle(d, sid, cycle, pull, push, rmsd="1.00"):
    write(os.path.join(d, "results.d", "%s_c%d.gs" % (sid, cycle)),
          "%s %d %s %s %s\n" % (sid, cycle, pull, push, rmsd))


# ---------------------------------------------------------------------------
print("\n[1] CI95: the paired t interval of the per-cycle mean")
d = tempfile.mkdtemp(prefix="cls_ci_")
write(os.path.join(d, "sp.gs"), "N1\tB\nN2\tB\nN5\tB\nANTI\tB\n")
write(os.path.join(d, "results_0.gs"), "# stage 0\nN1 OK\nN2 OK\nN5 OK\nANTI OK\n")
percycle(d, "N1", 1, -100, -60)
for c, (p, q) in enumerate(((-100, -60), (-96, -56)), 1):
    percycle(d, "N2", c, p, q)
for c, (p, q) in enumerate(((-100, -60), (-100, -60), (-90, -50), (-100, -60), (-100, -60)), 1):
    percycle(d, "N5", c, p, q)
for c, (p, q) in enumerate(((-100, -60), (-90, -70), (-110, -50)), 1):
    percycle(d, "ANTI", c, p, q)
r = run_groscore(d, "-n", "5")
avg = rows(os.path.join(d, "scores_avg.gs"))
check("groscore.py ran and wrote scores_avg.gs", r.returncode == 0 and avg, r.stderr[-300:])
got = {k: avg.get(k, [None] * 11) for k in ("N1", "N2", "N5", "ANTI")}
check("1 cycle: score -80.0, CI95 nan (v1.4: 0.0), note CI95_N_LT_2",
      got["N1"][1:4] == ["-80.0", "nan", "1"] and got["N1"][10:11] == ["CI95_N_LT_2"], got["N1"])
check("2 cycles, s = -80/-76: score -78.0, CI95 12.706 * 2.0 = 25.4",
      got["N2"][1:4] == ["-78.0", "25.4", "2"] and got["N2"][10:11] == ["nan"], got["N2"])
check("5 cycles, s = -80 x4, -70: score -78.0, CI95 2.776 * 2.0 = 5.6",
      got["N5"][1:4] == ["-78.0", "5.6", "5"], got["N5"])
check("pull and push anti-correlated, s constant -80: CI95 0.0 (paired)",
      got["ANTI"][1:4] == ["-80.0", "0.0", "3"], got["ANTI"])
c1 = rows(os.path.join(d, "scores_avg_c1.gs"))
check("scores_avg_c1.gs: one cycle is never a 0.0 interval",
      all(c1.get(k, [None] * 3)[2] == "nan" for k in ("N1", "N2", "N5", "ANTI")),
      {k: c1.get(k) for k in ("N1", "N2")})
c2 = rows(os.path.join(d, "scores_avg_c2.gs"))
check("scores_avg_c2.gs: N2 at 25.4, N1 not scored (one cycle)",
      c2.get("N2", [None] * 3)[1:3] == ["-78.0", "25.4"] and c2.get("N1", [None] * 4)[3] == "0",
      (c2.get("N2"), c2.get("N1")))
check("score files exist for 1..5 cycles and no further",
      all(os.path.isfile(os.path.join(d, "scores_avg_c%d.gs" % k)) for k in range(1, 6))
      and not os.path.isfile(os.path.join(d, "scores_avg_c6.gs")))
shutil.rmtree(d, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\n[2] stage 0: the last status decides")
d = tempfile.mkdtemp(prefix="cls_st_")
write(os.path.join(d, "sp.gs"), "".join("%s\tB\n" % s for s in ("S1", "S2", "S3", "S4", "S5", "S6")))
write(os.path.join(d, "results_0.gs"),
      "# stage 0\nS1 OK\nS2 OK\nS2 FAILED\nS3 FAILED\nS3 OK\nS4 OK\nS4 SETUP_RUNNING\n"
      "S5\nS6 OK\nS6 OKS7 DISRES_FAIL\n")
for s in ("S1", "S2", "S3", "S4", "S5", "S6"):
    percycle(d, s, 1, -100, -60)
r = run_groscore(d, "-n", "1")
avg = rows(os.path.join(d, "scores_avg.gs"))
check("a single-token line does not stop scoring (v1.4: TypeError, no file)",
      r.returncode == 0 and len(avg) == 6, r.stderr[-300:])
check("  and is reported", "malformed line" in r.stdout, r.stdout[-400:])
check("OK: scored", avg.get("S1", [None] * 2)[1] == "-80.0", avg.get("S1"))
check("OK then FAILED: not scored from the old rows, row says FAILED",
      avg.get("S2", [None] * 4)[1:4] == ["FAILED", "FAILED", "0"], avg.get("S2"))
check("FAILED then OK: scored", avg.get("S3", [None] * 2)[1] == "-80.0", avg.get("S3"))
check("OK then SETUP_RUNNING: not scored while a new setup runs",
      avg.get("S4", [None] * 2)[1] == "SETUP_RUNNING", avg.get("S4"))
check("only a malformed line: no status, scored as before", avg.get("S5", [None] * 2)[1] == "-80.0",
      avg.get("S5"))
check("a torn append ('S6 OKS7 DISRES_FAIL') is skipped, S6 keeps OK",
      avg.get("S6", [None] * 2)[1] == "-80.0", avg.get("S6"))
shutil.rmtree(d, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\n[3] per-cycle results, and legacy run directories")
d = tempfile.mkdtemp(prefix="cls_rd_")
ids = ("MARK", "NEW", "OLD", "RMSDW", "V082", "Z", "Z_c1", "BADROW", "FAILC")
write(os.path.join(d, "sp.gs"), "".join("%s\tB\n" % s for s in ids))
write(os.path.join(d, "results_0.gs"), "# stage 0\n" + "".join("%s OK\n" % s for s in ids))
# Legacy append-only files left by an earlier setup of MARK and NEW, plus OLD,
# RMSDW and V082, which never ran under the v1.5 job.run.
write(os.path.join(d, "results_1.gs"),
      "MARK -150.0\nNEW -500.0\nOLD -100.0\nRMSDW -280.0\n")
write(os.path.join(d, "results_2.gs"),
      "MARK 90.0 2.50\nNEW 500.0 9.99\nOLD -60.0 1.50\nRMSDW -120.0 3.41\nRMSDW 3.41\n")
write(os.path.join(d, "results_3.gs"), "V082 -100.0\n")
write(os.path.join(d, "results_4.gs"), "V082 -60.0\n")      # 82c2ffd layout: no RMSD column
write(os.path.join(d, "results.d", "MARK_setup.gs"), "MARK numpertres 407\n")
percycle(d, "NEW", 1, -100, -60, "0.50")
percycle(d, "NEW", 2, -120, -40, "0.70")
percycle(d, "Z", 1, -100, -60)
percycle(d, "Z_c1", 1, -200, -100)
write(os.path.join(d, "results.d", "BADROW_c1.gs"), "BADROW 2 -100 -60 1.0\n")
percycle(d, "FAILC", 1, "NaN", "NaN", "NaN")
percycle(d, "FAILC", 2, -100, -60)
write(os.path.join(d, "results.d", "NEW_c3.gs.tmp123"), "NEW 3 -999 -999 1.0\n")
r = run_groscore(d, "-n", "3")
avg = rows(os.path.join(d, "scores_avg.gs"))
g = lambda k: avg.get(k, [None] * 11)
check("groscore.py ran", r.returncode == 0 and avg, r.stderr[-300:])
check("a setup marker alone: the old setup's legacy rows are not scored",
      g("MARK")[1] == "nan" and g("MARK")[3] == "0", g("MARK"))
check("per-cycle rows win over legacy rows of the same cycle: -80.0, 2 cycles",
      g("NEW")[1] == "-80.0" and g("NEW")[3] == "2", g("NEW"))
check("  and their RMSDs: mean 0.60, max 0.70", g("NEW")[4:6] == ["0.60", "0.70"], g("NEW"))
check("a legacy-only structure is scored from results_<n>.gs as before (-80.0)",
      g("OLD")[1] == "-80.0" and g("OLD")[4] == "1.50", g("OLD"))
check("'ID RMSD' in a three-field push file is no work: -200.0 (v1.4: -138.3)",
      g("RMSDW")[1] == "-200.0" and g("RMSDW")[3] == "1", g("RMSDW"))
check("  and is reported", "malformed row" in r.stdout, r.stdout[-500:])
check("a two-field push file (82c2ffd) still reads: cycle 2 of V082 at -80.0",
      g("V082")[1] == "-80.0" and g("V082")[3] == "1", g("V082"))
check("Z reads Z_c1.gs only (-80.0), Z_c1 reads Z_c1_c1.gs (-150.0)",
      g("Z")[1] == "-80.0" and g("Z_c1")[1] == "-150.0", (g("Z"), g("Z_c1")))
check("a row whose cycle is not its file's is skipped and reported",
      g("BADROW")[3] == "0" and "BADROW_c1.gs" in r.stdout, (g("BADROW"), r.stdout[-300:]))
check("a NaN row (failed cycle) is not a cycle: FAILC uses cycle 2 only",
      g("FAILC")[1] == "-80.0" and g("FAILC")[3] == "1", g("FAILC"))
check("a .tmp file is not a row (NEW keeps 2 cycles)", g("NEW")[3] == "2", g("NEW"))
fren = [l.split() for l in open(os.path.join(d, "frenstruct.gs"))] if \
    os.path.isfile(os.path.join(d, "frenstruct.gs")) else []
check("frenstruct.gs carries NEW's per-cycle works",
      len(fren) == len(ids) and [float(x) for x in fren[1][:4]] == [-100.0, -60.0, -120.0, -40.0],
      fren[1] if len(fren) > 1 else fren)
shutil.rmtree(d, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\n[4] submission: only what runs, with GROSCORE_NUMCYCLES")
d = tempfile.mkdtemp(prefix="cls_sub_")
fake = os.path.join(d, ".fakebin")
write(os.path.join(fake, "sbatch"), '#!/bin/bash\necho "$*" >> "$SBATCH_LOG"\n')
os.chmod(os.path.join(fake, "sbatch"), 0o755)
env = dict(os.environ, PATH=fake + ":" + os.environ["PATH"],
           SBATCH_LOG=os.path.join(d, "sbatch.log"))
run = os.path.join(d, "run")
write(os.path.join(run, "sp.gs"), "DIR1\tB\nARCH\tB\nGONE\tB\nDIR2\tB\n")
for s in ("DIR1", "DIR2"):
    write(os.path.join(run, s, "input.pdb"), "END\n")
tmp_arch = os.path.join(d, "arch_src")
write(os.path.join(tmp_arch, "ARCH", "run.gs"), "B 4 1 amber19sb_opc3 1\n")
write(os.path.join(tmp_arch, "ARCH", "job.run"), "#!/bin/bash\necho OLD_JOB_RUN\n")
with tarfile.open(os.path.join(run, "ARCH.tar.gz"), "w:gz") as t:
    t.add(os.path.join(tmp_arch, "ARCH"), arcname="ARCH")
r = run_groscore(run, "-n", "3", env=env)
sub = open(os.path.join(run, "array_submit.run")).read() if \
    os.path.isfile(os.path.join(run, "array_submit.run")) else ""
check("fresh submission: sbatch called once", os.path.isfile(env["SBATCH_LOG"])
      and open(env["SBATCH_LOG"]).read().split() == ["array_submit.run"], r.stdout[-300:])
check("  the array covers DIR1 and DIR2 only (indices 0 and 3)", "#SBATCH --array=0,3\n" in sub,
      [l for l in sub.splitlines() if "--array" in l])
check("  the archive is NOT submitted, and the log says so",
      "ARCH.tar.gz is archived and is NOT submitted" in r.stdout, r.stdout[-400:])
check("  GONE is recorded NODIR", "GONE NODIR" in open(os.path.join(run, "results_0.gs")).read())
check("  the script exports GROSCORE_NUMCYCLES=3", "export GROSCORE_NUMCYCLES=3\n" in sub)
check("  run.gs and job.run were written for DIR1",
      open(os.path.join(run, "DIR1", "run.gs")).read() == "B 6 1 amber19sb_opc3 1\n"
      and os.access(os.path.join(run, "DIR1", "job.run"), os.X_OK))
open(env["SBATCH_LOG"], "w").close()
r = run_groscore(run, "-n", "5", "--restart", "--inject-job-run", env=env)
sub = open(os.path.join(run, "array_submit.run")).read()
check("-n 5 --restart --inject-job-run: the archive is submitted (array 0-1,3)",
      "#SBATCH --array=0-1,3\n" in sub, [l for l in sub.splitlines() if "--array" in l])
check("  with GROSCORE_NUMCYCLES=5, while run.gs keeps its 3 cycles",
      "export GROSCORE_NUMCYCLES=5\n" in sub
      and open(os.path.join(run, "DIR1", "run.gs")).read() == "B 6 1 amber19sb_opc3 1\n")
with tarfile.open(os.path.join(run, "ARCH.tar.gz")) as t:
    names = t.getnames()
    jr = [n for n in names if n.endswith("job.run")]
    body = t.extractfile(jr[0]).read().decode() if len(jr) == 1 else ""
check("  and the archive now holds this checkout's job.run",
      len(jr) == 1 and "GROSCORE_NUMCYCLES" in body, jr)

# The generated script itself, driven with a stub job.run.
def drive(task_id, prepare):
    w = os.path.join(d, "drive%s" % task_id)
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    prepare(w)
    shutil.copy(os.path.join(run, "struct_map.gs"), w)
    body = sub.split("#SBATCH --array=")[1].split("\n", 1)[1]
    rr = subprocess.run(["bash", "-c", body], cwd=w, capture_output=True, text=True,
                        env=dict(os.environ, SLURM_ARRAY_TASK_ID=str(task_id)))
    return rr, w


def good_archive(w):
    src = os.path.join(w, "src", "ARCH")
    write(os.path.join(src, "job.run"), '#!/bin/bash\necho "RAN numcycles=$GROSCORE_NUMCYCLES"\n')
    os.chmod(os.path.join(src, "job.run"), 0o755)
    with tarfile.open(os.path.join(w, "ARCH.tar.gz"), "w:gz") as t:
        t.add(src, arcname="ARCH")
    shutil.rmtree(os.path.join(w, "src"))


def bad_archive(w):
    good_archive(w)
    data = open(os.path.join(w, "ARCH.tar.gz"), "rb").read()
    open(os.path.join(w, "ARCH.tar.gz"), "wb").write(data[:len(data) // 2])


rr, w = drive(1, good_archive)
check("a good archive is extracted, the tarball removed, job.run run with -n",
      "RAN numcycles=5" in rr.stdout and not os.path.exists(os.path.join(w, "ARCH.tar.gz")),
      (rr.stdout, rr.stderr[-200:]))
rr, w = drive(1, bad_archive)
check("a truncated archive: exit 1, tarball kept, no directory, job.run not run",
      rr.returncode == 1 and os.path.isfile(os.path.join(w, "ARCH.tar.gz"))
      and not os.path.isdir(os.path.join(w, "ARCH")) and "RAN" not in rr.stdout,
      (rr.returncode, rr.stdout[-200:]))
def stray(w):
    write(os.path.join(w, "job.run"), "#!/bin/bash\necho STRAY\n")
    os.chmod(os.path.join(w, "job.run"), 0o755)


rr, w = drive(1, stray)
check("no directory and no archive: exit 1, nothing run in the run directory",
      rr.returncode == 1 and "STRAY" not in rr.stdout, (rr.returncode, rr.stdout))
shutil.rmtree(d, ignore_errors=True)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all classic scorer checks passed")
