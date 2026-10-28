#!/usr/bin/env python3
"""Submission never mixes protocols: the run directory's protocol identity, the
per-structure holds, and the archived-structure rule of groscore_fe.py.

v1.5 kept the 43-field row of v1.4, so nothing in a row says which protocol made
it. The review of 2026-10-04 reproduced four ways that went wrong at submission:
a pre-v1.5 directory with setup.done counted as complete and got cycles that all
stop (CYC-3); recreated structure directories inside a pass-2 run directory
counted the old rows as done v1.5 cycles and pooled them (CYC-4); archived
structures were reported 'Skipping' and submitted anyway, to run their old job.run
(CYC-2a); and --sequential ran its array over every index, held or not. What is
pinned, against a fake sbatch that records every call:

  [1] A FRESH run directory records fe_protocol.PROTOCOL_ID in run_config.gs at its
      first submission, and a later scoring pass prints no protocol warning.
  [2] A directory with history and no recorded id (pass 2), with or without
      --restart, and [3] one recorded under another id: nothing is submitted,
      nothing in the directory is touched (job.run, run.gs, struct_map.gs, no
      slurm_fe/), no id is recorded, the rows are still scored with a WARNING, the
      refusal is the first and the last thing printed, and the exit status is 1.
  [4] Under the directory's own id, structure by structure: a setup.done without
      protocol_id.gs, or a protocol_id.gs of another id, is held (OTHER_PROTOCOL)
      and keeps its setup.done and job.run; result rows for a directory that holds
      no setup record at all, as after recreating it, are held
      (ROWS_WITHOUT_SETUP); an archive with missing cycles is held without
      --inject-job-run (ARCHIVED) and stays byte-identical; npt_c_fe.mdp without
      a network block or 'pull = no' makes a setup incomplete. A directory whose
      protocol_id.gs names this protocol is left to job_fe.run's own guard on
      re-running a setup over existing cycles: with rows and an incomplete
      setup it is cleared and resubmitted, protocol_id.gs kept, and so is one
      whose setup.done was removed. The rest is submitted exactly as before: a
      setup job plus the missing cycles, or the cycles alone after a complete
      setup.
  [5] --inject-job-run installs this job_fe.run (executable) into an archive of
      this protocol and submits it; an archive set up before v1.5 or for another
      protocol is held, byte-identical, and leaves no directory or _new.tar.gz.
  [6] --sequential runs its array over the submitted structures' indices only,
      and passes the total as GROSCORE_NUMCYCLES, on SLURM and to the local
      runner, so -n <more> --restart --sequential adds cycles.
  [7] setup_is_complete itself: False for a foreign setup without clearing it,
      True for a complete one, and npt_c_fe.mdp judged as job_fe.run's setup
      post-condition judges it.
  [8] A structure whose last stage-0 status is LIGAND_FAIL, BROKEN or ENTANGLED
      is held (STAGE0_FINAL) and gets no job.run: --restart used to submit its
      setup and cycles again. A LIGAND_FAIL counts only when ligand_failed.gs
      says why: one that records a crash or a kill, or has no line, is submitted.
      PROBE_FAIL is still submitted, and so is a structure whose last line is
      SETUP_RUNNING again.
  [9] A directory with every cycle done (results_fe_c<N>.done) and no archive gets
      a job_fe.run --archive job, on SLURM (slurm_fe/archive_<ID>.run, a short time
      limit, GROSCORE_NUMCYCLES) and to the local runner (logging beside the
      directory it removes): its archive used to be made only by the task that
      finished the last cycle, so a failed one (1AKJ in bm_fe_amber_3a) was never
      tried again. A directory whose rows are all complete but a .done is missing
      gets those cycles, which write it. One with an archive under way or killed
      (.archive.lock) gets nothing and is reported; so is one with an <ID>.tar.gz
      beside it. The --sequential unpack removes .archive.lock, as the others do.

Run:  python3 tests/test_fe_submit_protocol.py
"""
import hashlib, io, json, os, shutil, stat, subprocess, sys, tarfile, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(REPO, "groscore_fe.py")
JOB = os.path.join(REPO, "job_fe.run")
sys.path.insert(0, os.path.join(REPO, "utils"))
import fe_protocol as P

failures = []


def check(name, ok, detail=""):
    print("  %-68s %s%s" % (name, "PASS" if ok else "FAIL",
                             "" if ok or not detail else "  " + str(detail)))
    if not ok:
        failures.append(name)


CUR = P.PROTOCOL_ID
OTHER = "fe-0.0-deadbeef"
NAMES = [w[0] for w in P.works()]
LEG_MDPS = sorted({l["mdp"] for l in P.legs()})
OLD_JOB = "#!/bin/bash\necho 'v1.4 job.run'\n"
work = tempfile.mkdtemp(prefix="fe_submit_")

# A fake sbatch: one log line per call with its arguments, a job id for --parsable.
fakebin = os.path.join(work, "bin")
os.makedirs(fakebin)
with open(os.path.join(fakebin, "sbatch"), "w") as f:
    f.write('#!/bin/sh\necho "$*" >> "$SBATCH_LOG"\n'
            'case " $* " in *" --parsable "*) echo 4242;; esac\nexit 0\n')
os.chmod(os.path.join(fakebin, "sbatch"), 0o755)


def row(sid, c):
    return "%s %d %s 1.50" % (sid, c, " ".join("1.0" for _ in NAMES))


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def put(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def setup_files(d, protocol=CUR, nptc="pull                     = no\n", done=True):
    """A finished setup as job_fe.run leaves it; protocol None writes no
    protocol_id.gs (v1.4 wrote none)."""
    for name in ("numpertres.gs", "boresch_analytical.gs", "boresch_anchors.gs"):
        put(os.path.join(d, name), "x 1\n")
    for m in LEG_MDPS:
        put(os.path.join(d, m), "; leg\npull-ngroups             = 3\n")
    if nptc is not None:
        put(os.path.join(d, "npt_c_fe.mdp"), "; npt_c\n" + nptc)
    if protocol is not None:
        put(os.path.join(d, "protocol_id.gs"), protocol + "\n")
    if done:
        put(os.path.join(d, "setup.done"), "")
    put(os.path.join(d, "job.run"), OLD_JOB)
    put(os.path.join(d, "run.gs"), "B 8 1 amber19sb_opc3 1\n")


def make_archive(run, sid, protocol=CUR):
    src = os.path.join(work, "arcsrc", sid)
    setup_files(src, protocol=protocol)
    with tarfile.open(os.path.join(run, "%s.tar.gz" % sid), "w:gz") as t:
        t.add(src, arcname=sid)
    shutil.rmtree(os.path.join(work, "arcsrc"))


def new_run(name, sids, cfg, rows=None, history=True):
    run = os.path.join(work, name)
    os.makedirs(os.path.join(run, "results_fe.d"))
    put(os.path.join(run, "sp.gs"), "# id chains\n" + "".join("%s B\n" % s for s in sids))
    put(os.path.join(run, "run_config.gs"), cfg)
    for sid, cycles in (rows or {}).items():
        for c in cycles:
            put(os.path.join(run, "results_fe.d", "%s_c%d.gs" % (sid, c)), row(sid, c) + "\n")
    if history:
        put(os.path.join(run, "struct_map.gs"),
            "# Array_Index Structure_ID\n" + "".join("%d %s\n" % x for x in enumerate(sids)))
        put(os.path.join(run, "results_0.gs"),
            "# Stage-0 setup status\n" + "".join("%s OK\n" % s for s in sids))
    return run


def gs(run, *argv):
    log = os.path.join(run, "sbatch.log")
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ, MPLBACKEND="Agg", SBATCH_LOG=log,
               PATH=fakebin + os.pathsep + os.environ.get("PATH", ""))
    p = subprocess.run([sys.executable, FE] + list(argv), cwd=run, capture_output=True,
                       text=True, env=env, timeout=900)
    calls = open(log).read().splitlines() if os.path.exists(log) else []
    return p, calls


def cfg_of(run):
    out = {}
    for line in open(os.path.join(run, "run_config.gs")):
        t = line.split()
        if len(t) >= 2 and not line.startswith("#"):
            out[t[0]] = t[1]
    return out


def lines(p):
    return [l for l in p.stdout.splitlines() if l.strip()]


print("\n[1] a fresh run directory records its protocol at the first submission")
run = new_run("fresh", ["FRESH"], "", history=False)
os.makedirs(os.path.join(run, "FRESH"))
put(os.path.join(run, "FRESH", "input.pdb"), "END\n")
p, calls = gs(run, "-n", "2")
check("rc 0", p.returncode == 0, p.stdout[-300:] + p.stderr[-300:])
check("run_config.gs records protocol_id %s" % CUR, cfg_of(run).get("protocol_id") == CUR,
      cfg_of(run))
check("and says so", "Recorded FE protocol %s" % CUR in p.stdout)
check("the setup job and its cycles were submitted",
      calls == ["--parsable slurm_fe/setup_FRESH.run",
                "--dependency=afterany:4242 slurm_fe/cycles_FRESH.run"], calls)
check("job_fe.run was installed", open(os.path.join(run, "FRESH", "job.run")).read()
      == open(JOB).read())
p, calls = gs(run)
check("a later scoring pass: rc 0, no protocol WARNING, nothing submitted",
      p.returncode == 0 and not calls
      and not any(l.startswith("WARNING") and "protocol" in l for l in lines(p)),
      [l for l in lines(p) if l.startswith("WARNING")] + calls)

print("\n[2] a directory that predates v1.5 is refused and left untouched")
for label, argv, keep_map in (("--restart", ["--restart", "-n", "4"], True),
                              ("no struct_map.gs, no flag", [], False)):
    run = new_run("pre15_%d" % keep_map, ["OLD", "NEW"], "numruns\t4\n",
                  rows={"OLD": [1, 2]})
    if not keep_map:
        os.remove(os.path.join(run, "struct_map.gs"))
    setup_files(os.path.join(run, "OLD"), protocol=None, nptc=None)
    os.makedirs(os.path.join(run, "NEW"))
    before = {f: open(os.path.join(run, f)).read()
              for f in ("OLD/job.run", "OLD/run.gs", "results_0.gs")
              + (("struct_map.gs",) if keep_map else ())}
    p, calls = gs(run, *argv)
    L = lines(p)
    head = "ERROR: nothing was submitted: this run directory predates v1.5."
    check("%s: exit status 1" % label, p.returncode == 1, p.returncode)
    check("%s: nothing went to sbatch" % label, not calls, calls)
    check("%s: the refusal is printed first and last" % label,
          head in L[:15] and L[-1] == head, L[-1:])
    check("%s: no file touched, no slurm_fe/, no array_submit.run, no NEW/job.run" % label,
          all(open(os.path.join(run, f)).read() == t for f, t in before.items())
          and not os.path.exists(os.path.join(run, "slurm_fe"))
          and not os.path.exists(os.path.join(run, "array_submit.run"))
          and not os.path.exists(os.path.join(run, "NEW", "job.run"))
          and os.path.exists(os.path.join(run, "struct_map.gs")) == keep_map)
    check("%s: no protocol_id recorded" % label, "protocol_id" not in cfg_of(run),
          cfg_of(run))
    check("%s: the rows are still scored, under a WARNING" % label,
          os.path.isfile(os.path.join(run, "scores_fe.gs"))
          and any(l.startswith("WARNING") and "records no protocol_id" in l for l in L))

print("\n[3] a directory recorded under another protocol is refused")
run = new_run("other", ["X"], "numruns\t4\nprotocol_id\t%s\n" % OTHER, rows={"X": [1]})
os.makedirs(os.path.join(run, "X"))
p, calls = gs(run, "--restart")
L = lines(p)
check("exit status 1, nothing submitted", p.returncode == 1 and not calls, calls)
check("the refusal names both ids",
      any(OTHER in l and CUR in l for l in L) and L[-1].startswith("ERROR: nothing was"),
      L[-1:])
check("the recorded id stays", cfg_of(run).get("protocol_id") == OTHER)

print("\n[4] structure by structure, under the directory's own id")
SIDS = ["OLD", "OTHERID", "ROWS", "FRESH", "DONE", "BADNPTC", "BADROWS", "NODONE",
        "ARCH", "ARCHDONE"]
run = new_run("own", SIDS, "numruns\t4\nprotocol_id\t%s\n" % CUR,
              rows={"ROWS": [1, 2], "DONE": [1, 2], "BADROWS": [1, 2], "NODONE": [1, 2],
                    "ARCH": [1, 2], "ARCHDONE": [1, 2, 3, 4]})
setup_files(os.path.join(run, "OLD"), protocol=None)
setup_files(os.path.join(run, "OTHERID"), protocol=OTHER)
os.makedirs(os.path.join(run, "ROWS"))
os.makedirs(os.path.join(run, "FRESH"))
setup_files(os.path.join(run, "DONE"))
setup_files(os.path.join(run, "BADNPTC"), nptc="pull                     = yes\n")
setup_files(os.path.join(run, "BADROWS"), nptc="pull                     = yes\n")
setup_files(os.path.join(run, "NODONE"), done=False)
make_archive(run, "ARCH")
make_archive(run, "ARCHDONE")
arch_md5 = md5(os.path.join(run, "ARCH.tar.gz"))
p, calls = gs(run, "--restart", "-n", "4")
out = p.stdout
check("rc 0", p.returncode == 0, p.stderr[-300:])
check("submitted: FRESH, DONE (3-4, no setup), BADNPTC, BADROWS and NODONE",
      calls == ["--parsable slurm_fe/setup_FRESH.run",
                "--dependency=afterany:4242 slurm_fe/cycles_FRESH.run",
                "slurm_fe/cycles_DONE.run",
                "--parsable slurm_fe/setup_BADNPTC.run",
                "--dependency=afterany:4242 slurm_fe/cycles_BADNPTC.run",
                "--parsable slurm_fe/setup_BADROWS.run",
                "--dependency=afterany:4242 slurm_fe/cycles_BADROWS.run",
                "--parsable slurm_fe/setup_NODONE.run",
                "--dependency=afterany:4242 slurm_fe/cycles_NODONE.run"], calls)
check("DONE's array is its missing cycles, 3-4",
      "#SBATCH --array=3-4" in open(os.path.join(run, "slurm_fe", "cycles_DONE.run")).read())
check("the summary: 4 structures held back",
      "NOT SUBMITTED: 4 structures held back and left untouched." in out,
      [l for l in out.splitlines() if "NOT SUBMITTED" in l])


def held(token, sid):
    """sid is listed under token in the NOT SUBMITTED block."""
    blk = out[out.find("NOT SUBMITTED"):].splitlines()
    try:
        i = blk.index("  %s:" % token)
    except ValueError:
        return False
    for l in blk[i + 1:]:
        if l.startswith("  ") and not l.startswith("    "):
            break
        if l.split()[:1] == [sid]:
            return True
    return False


for token, sid in (("OTHER_PROTOCOL", "OLD"), ("OTHER_PROTOCOL", "OTHERID"),
                   ("ROWS_WITHOUT_SETUP", "ROWS"), ("ARCHIVED", "ARCH")):
    check("%s is held as %s" % (sid, token), held(token, sid))
for token, sid in (("ROWS_WITHOUT_SETUP", "BADROWS"), ("ROWS_WITHOUT_SETUP", "NODONE")):
    check("%s, with this protocol's protocol_id.gs, is not held" % sid,
          not held(token, sid))
for sid in ("OLD", "OTHERID"):
    d = os.path.join(run, sid)
    check("%s keeps setup.done and its old job.run" % sid,
          os.path.exists(os.path.join(d, "setup.done"))
          and open(os.path.join(d, "job.run")).read() == OLD_JOB)
for sid in ("BADNPTC", "BADROWS"):
    d = os.path.join(run, sid)
    check("%s (npt_c_fe.mdp without network or pull = no) cleared for a redo" % sid,
          not os.path.exists(os.path.join(d, "setup.done"))
          and not os.path.exists(os.path.join(d, "npt_c_fe.mdp"))
          and os.path.exists(os.path.join(d, "protocol_id.gs")))
check("ARCH is byte-identical and was not unpacked",
      md5(os.path.join(run, "ARCH.tar.gz")) == arch_md5
      and not os.path.exists(os.path.join(run, "ARCH")))
check("ARCHDONE: all cycles complete, not held",
      "ARCHDONE: all 4 cycles complete, nothing to submit." in out
      and not held("ARCHIVED", "ARCHDONE"))
check("the old 'Skipping ... use --inject-job-run' line is gone",
      "Skipping" not in out)

print("\n[5] --inject-job-run checks the archive's protocol on the way through")
run = new_run("inject", ["ARCH", "ARCHOLD", "ARCHOTHER"],
              "numruns\t4\nprotocol_id\t%s\n" % CUR,
              rows={s: [1, 2] for s in ("ARCH", "ARCHOLD", "ARCHOTHER")})
make_archive(run, "ARCH")
make_archive(run, "ARCHOLD", protocol=None)
make_archive(run, "ARCHOTHER", protocol=OTHER)
sums = {s: md5(os.path.join(run, "%s.tar.gz" % s)) for s in ("ARCHOLD", "ARCHOTHER")}
p, calls = gs(run, "--restart", "-n", "4", "--inject-job-run")
out = p.stdout
check("rc 0", p.returncode == 0, p.stderr[-300:])
with tarfile.open(os.path.join(run, "ARCH.tar.gz")) as t:
    mem = {(m.name[2:] if m.name.startswith("./") else m.name): m for m in t}
    jobs = [n for n in mem if n.endswith("/job.run")]
    body = t.extractfile(mem["ARCH/job.run"]).read() if "ARCH/job.run" in mem else b""
check("ARCH: exactly one job.run inside, this job_fe.run",
      jobs == ["ARCH/job.run"] and body == open(JOB, "rb").read(), jobs)
check("ARCH: and it is executable",
      "ARCH/job.run" in mem and mem["ARCH/job.run"].mode & stat.S_IXUSR)
check("ARCH: setup.done and protocol_id.gs survive the rewrite",
      "ARCH/setup.done" in mem and "ARCH/protocol_id.gs" in mem, sorted(mem)[:6])
check("ARCH was submitted: its setup job (to unpack it) and cycles 3-4",
      calls == ["--parsable slurm_fe/setup_ARCH.run",
                "--dependency=afterany:4242 slurm_fe/cycles_ARCH.run"], calls)
for sid in ("ARCHOLD", "ARCHOTHER"):
    check("%s: held as OTHER_PROTOCOL, byte-identical, no %s/ and no _new.tar.gz"
          % (sid, sid),
          held("OTHER_PROTOCOL", sid)
          and md5(os.path.join(run, "%s.tar.gz" % sid)) == sums[sid]
          and not os.path.exists(os.path.join(run, sid))
          and not os.path.exists(os.path.join(run, "%s_new.tar.gz" % sid)))

print("\n[6] --sequential runs its array over the submitted indices only")
run = new_run("seq", ["OLD", "A", "GONE", "B"], "numruns\t4\nprotocol_id\t%s\n" % CUR)
setup_files(os.path.join(run, "OLD"), protocol=None)
os.makedirs(os.path.join(run, "A"))
os.makedirs(os.path.join(run, "B"))
p, calls = gs(run, "--restart", "--sequential")
arr = [l for l in open(os.path.join(run, "array_submit.run")).read().splitlines()
       if l.startswith("#SBATCH --array")] if os.path.exists(
           os.path.join(run, "array_submit.run")) else []
check("the array is 1,3 (A and B), not 0-3", arr == ["#SBATCH --array=1,3"], arr)
check("submitted once, OLD held", calls == ["array_submit.run"]
      and "Submitted 1 array of 2 structures" in p.stdout, calls)
# The total goes along as GROSCORE_NUMCYCLES, as on the array path: --restart never
# rewrites run.gs, so '-n <more> --restart --sequential' used to run the old total
# (review PY-5).
seq_script = open(os.path.join(run, "array_submit.run")).read()
check("the array exports the recorded total (GROSCORE_NUMCYCLES=4)",
      "export GROSCORE_NUMCYCLES=4\n" in seq_script, seq_script[-300:])
check("its unpack removes .archive.lock, as the setup job's does",
      '  rm \"${STRUCT_ID}.tar.gz\"\n  rm -rf \"${STRUCT_ID}/.archive.lock\"\nfi\n' in seq_script,
      seq_script[-300:])
p, calls = gs(run, "-n", "6", "--restart", "--sequential")
seq_script = open(os.path.join(run, "array_submit.run")).read()
check("-n 6 --restart --sequential: the array exports GROSCORE_NUMCYCLES=6",
      "export GROSCORE_NUMCYCLES=6\n" in seq_script and calls == ["array_submit.run"],
      (calls, seq_script[-300:]))
# --run-local --sequential: the job list handed to the local runner, read in-process
# with launch_local replaced, so no runner is started.
runl = new_run("seq_local", ["A"], "numruns\t4\nprotocol_id\t%s\n" % CUR)
os.makedirs(os.path.join(runl, "A"))
put(os.path.join(work, "seq_local_driver.py"),
    "import json, sys\n"
    "sys.argv = ['groscore_fe.py', '--restart', '--sequential', '--run-local', '--ngpus', '1']\n"
    "sys.path.insert(0, %r)\n"
    "import groscore_fe as G\n"
    "got = []\n"
    "G.launch_local = lambda jobs, *a, **k: got.extend(jobs) or True\n"
    "G.setup_and_submit(['A'], ['B'])\n"
    "print('JOBS ' + json.dumps(got))\n" % REPO)
p = subprocess.run([sys.executable, os.path.join(work, "seq_local_driver.py")], cwd=runl,
                   capture_output=True, text=True, timeout=600)
jl = [json.loads(l[5:]) for l in p.stdout.splitlines() if l.startswith("JOBS ")]
check("--run-local --sequential: the local job carries GROSCORE_NUMCYCLES=4",
      jl and len(jl[0]) == 1 and jl[0][0].get("env", {}).get("GROSCORE_NUMCYCLES") == "4",
      (jl, p.stderr[-300:]))

print("\n[7] setup_is_complete")
scratch = os.path.join(work, "inproc")
os.makedirs(scratch)
here = os.getcwd()
os.chdir(scratch)
sys.argv = [sys.argv[0]]
sys.path.insert(0, REPO)
G = None
try:
    import groscore_fe as G
except Exception as e:                                    # pragma: no cover
    check("groscore_fe imports", False, "%s: %s" % (type(e).__name__, e))
if G is not None:
    setup_files("OLD", protocol=None)
    setup_files("OTHERID", protocol=OTHER)
    setup_files("DONE")
    setup_files("NOCOMMENT", nptc="pull = no ; no network pairs\n")
    setup_files("NETWORK", nptc="pull                     = yes\npull-ngroups = 2\n")
    setup_files("TEMPLATE", nptc="pull                     = yes\n")
    setup_files("NOPE", nptc="pull = nope\n")
    setup_files("NONPTC", nptc=None)
    check("a v1.4 setup (setup.done, no protocol_id.gs) is not complete",
          G.setup_is_complete("OLD") is False)
    check("and is not cleared: setup.done and the mdps stay",
          os.path.exists("OLD/setup.done") and os.path.exists("OLD/" + LEG_MDPS[0]))
    check("another protocol's setup is not complete, and stays",
          G.setup_is_complete("OTHERID") is False and os.path.exists("OTHERID/setup.done"))
    check("a complete v1.5 setup with 'pull = no' is complete", G.setup_is_complete("DONE"))
    check("'pull = no' followed by a comment counts", G.setup_is_complete("NOCOMMENT"))
    check("a network block (pull-ngroups) counts", G.setup_is_complete("NETWORK"))
    for sid in ("TEMPLATE", "NOPE", "NONPTC"):
        check("%s: not complete, and cleared for a redo" % sid,
              G.setup_is_complete(sid) is False and not os.path.exists(sid + "/setup.done")
              and not os.path.exists(sid + "/npt_c_fe.mdp"))
os.chdir(here)

print("\n[8] a setup that ended in a verdict on the structure is not submitted again")
# bm_fe_amber_3a: --restart submitted the setups of 1GRN, 1WQ1 and 1RLB again, which
# stopped as LIGAND_FAIL once more, and their 60 cycle tasks only read the status.
S0 = ["LIG", "BRK", "ENT", "PROBE", "REDO", "KILLED", "NOLINE"]
run = new_run("stage0", S0, "numruns\t2\nprotocol_id\t%s\n" % CUR)
put(os.path.join(run, "results_0.gs"),
    "# Stage-0 setup status\nLIG SETUP_RUNNING\nLIG LIGAND_FAIL\nBRK BROKEN\n"
    "ENT ENTANGLED\nPROBE PROBE_FAIL\nREDO LIGAND_FAIL\nREDO SETUP_RUNNING\n"
    "KILLED LIGAND_FAIL\nNOLINE LIGAND_FAIL\n")
for sid in S0:
    os.makedirs(os.path.join(run, sid))
# A LIGAND_FAIL is a verdict only when ligand_failed.gs says why: the AlF3 line is one,
# a parametrisation killed mid-run is not, and neither is a LIGAND_FAIL without a line
# (merge_ligand.py or fuse_glycans.py crashed).
put(os.path.join(run, "LIG", "ligand_failed.gs"),
    "AF3 R 169 openff-2.2.1.offxml has no parameters for Al (it types only H C), so no ligand containing it "
    "can be parametrised\n")
put(os.path.join(run, "KILLED", "ligand_failed.gs"),
    "SO4 A 401 openff-2.2.1.offxml has no parameters for Al (it types only H C), so no ligand containing it "
    "can be parametrised\nGDP A 198 parametrisation started but did not finish (crashed or killed)\n")
p, calls = gs(run, "--restart", "-n", "2")
out = p.stdout
check("rc 0", p.returncode == 0, p.stderr[-300:])
check("submitted: PROBE (retried), REDO (last line SETUP_RUNNING), and the crashed KILLED and NOLINE",
      calls == ["--parsable slurm_fe/setup_PROBE.run",
                "--dependency=afterany:4242 slurm_fe/cycles_PROBE.run",
                "--parsable slurm_fe/setup_REDO.run",
                "--dependency=afterany:4242 slurm_fe/cycles_REDO.run",
                "--parsable slurm_fe/setup_KILLED.run",
                "--dependency=afterany:4242 slurm_fe/cycles_KILLED.run",
                "--parsable slurm_fe/setup_NOLINE.run",
                "--dependency=afterany:4242 slurm_fe/cycles_NOLINE.run"], calls)
for sid in ("LIG", "BRK", "ENT"):
    check("%s is held as STAGE0_FINAL, gets no job.run and no slurm_fe file" % sid,
          held("STAGE0_FINAL", sid)
          and not os.path.exists(os.path.join(run, sid, "job.run"))
          and not os.path.exists(os.path.join(run, "slurm_fe", "setup_%s.run" % sid)))
check("the summary: 3 structures held back, with how to set one up again",
      "NOT SUBMITTED: 3 structures held back and left untouched." in out
      and "'<ID> SETUP_RUNNING'" in out,
      [l for l in out.splitlines() if "NOT SUBMITTED" in l])

print("\n[9] a finished structure that is still a directory gets an archive job")
S9 = ["COMPLETE", "NOMARK", "BOTH", "PARTIAL", "BUSY", "RECREATED"]
run = new_run("archive", S9, "numruns\t4\nprotocol_id\t%s\n" % CUR,
              rows={"COMPLETE": [1, 2, 3, 4], "NOMARK": [1, 2, 3, 4], "BOTH": [1, 2, 3, 4],
                    "PARTIAL": [1, 2], "BUSY": [1, 2, 3, 4], "RECREATED": [1, 2, 3, 4]})
for sid in S9:
    setup_files(os.path.join(run, sid))
for sid, cycles in (("COMPLETE", (1, 2, 3, 4)), ("NOMARK", (1, 2, 3)),
                    ("BOTH", (1, 2, 3, 4)), ("PARTIAL", (1, 2)), ("BUSY", (1, 2, 3, 4))):
    for c in cycles:
        put(os.path.join(run, sid, "results_fe_c%d.done" % c), "")
os.makedirs(os.path.join(run, "BUSY", ".archive.lock"))
put(os.path.join(run, "BOTH.tar.gz"), "an older archive\n")
both_md5 = md5(os.path.join(run, "BOTH.tar.gz"))
p, calls = gs(run, "--restart", "-n", "4")
out = p.stdout
check("rc 0", p.returncode == 0, p.stderr[-300:])
check("submitted: COMPLETE's archive job, NOMARK's cycle 4, PARTIAL's cycles 3-4",
      calls == ["slurm_fe/archive_COMPLETE.run", "slurm_fe/cycles_NOMARK.run",
                "slurm_fe/cycles_PARTIAL.run"], calls)
script = open(os.path.join(run, "slurm_fe", "archive_COMPLETE.run")).read() \
    if os.path.exists(os.path.join(run, "slurm_fe", "archive_COMPLETE.run")) else ""
check("the archive job: the template, its name, a short limit, the total, --archive",
      script.startswith(open(os.path.join(REPO, "slurm", "workstation.sh")).read().rstrip("\n") + "\n")
      and "#SBATCH --job-name=fearch_COMPLETE\n#SBATCH --time=08:00:00\n\n" in script
      and script.endswith("export GROSCORE_NUMCYCLES=4\ncd COMPLETE || exit 1\n./job.run --archive\n"),
      script)
check("COMPLETE says why it gets one",
      "COMPLETE: all 4 cycles complete but not archived; submitting its archive job." in out)
check("NOMARK (rows complete, .done of cycle 4 missing): cycle 4, to write it; no archive job",
      "NOMARK: complete rows but no results_fe_c<N>.done for cycle(s) 4; submitting them to write it."
      in out and "#SBATCH --array=4\n" in open(os.path.join(run, "slurm_fe", "cycles_NOMARK.run")).read()
      and not os.path.exists(os.path.join(run, "slurm_fe", "archive_NOMARK.run")))
check("RECREATED (rows, but no .done at all: not the directory they came from): nothing",
      "RECREATED: all 4 cycles complete, nothing to submit." in out
      and not os.path.exists(os.path.join(run, "slurm_fe", "cycles_RECREATED.run")))
check("BOTH (directory and BOTH.tar.gz): reported, no job, the archive untouched",
      "BOTH: all 4 cycles complete, nothing to submit." in out
      and "Both BOTH/ and BOTH.tar.gz exist; the directory is used. Compare them and remove one by hand."
      in out and not os.path.exists(os.path.join(run, "slurm_fe", "archive_BOTH.run"))
      and md5(os.path.join(run, "BOTH.tar.gz")) == both_md5)
check("BUSY (.archive.lock): reported as under way or interrupted, no job",
      "BUSY: all 4 cycles complete; an archive is under way or was interrupted (BUSY/.archive.lock)."
      " If no task archives it, remove BUSY/.archive.lock by hand." in out
      and not os.path.exists(os.path.join(run, "slurm_fe", "archive_BUSY.run")), out[-600:])
check("the summary counts the archive job",
      "Submitted 0 setup job(s) + 2 cycle array(s) + 1 archive job(s), target 4" in out,
      [l for l in out.splitlines() if l.startswith("Submitted")])
# --run-local: the job list handed to the local runner, launch_local replaced.
runl = new_run("archive_local", ["COMPLETE"], "numruns\t4\nprotocol_id\t%s\n" % CUR,
               rows={"COMPLETE": [1, 2, 3, 4]})
setup_files(os.path.join(runl, "COMPLETE"))
for c in (1, 2, 3, 4):
    put(os.path.join(runl, "COMPLETE", "results_fe_c%d.done" % c), "")
put(os.path.join(work, "archive_local_driver.py"),
    "import json, sys\n"
    "sys.argv = ['groscore_fe.py', '--restart', '-n', '4', '--run-local', '--ngpus', '1']\n"
    "sys.path.insert(0, %r)\n"
    "import groscore_fe as G\n"
    "got = []\n"
    "G.launch_local = lambda jobs, *a, **k: got.extend(jobs) or True\n"
    "G.setup_and_submit(['COMPLETE'], ['B'])\n"
    "print('JOBS ' + json.dumps(got))\n" % REPO)
p = subprocess.run([sys.executable, os.path.join(work, "archive_local_driver.py")], cwd=runl,
                   capture_output=True, text=True, timeout=600)
jl = [json.loads(l[5:]) for l in p.stdout.splitlines() if l.startswith("JOBS ")]
check("--run-local: one local job, ./job.run --archive with GROSCORE_NUMCYCLES=4,"
      " logging beside the directory it removes",
      jl == [[{"name": "archive_COMPLETE", "dir": "COMPLETE", "argv": ["./job.run", "--archive"],
               "env": {"GROSCORE_NUMCYCLES": "4"}, "log": "../job_local_archive_COMPLETE.out"}]],
      (jl, p.stderr[-300:]))

shutil.rmtree(work, ignore_errors=True)
print("")
if failures:
    print("FAILED (%d): %s" % (len(failures), "; ".join(failures)))
    sys.exit(1)
print("All submission protocol checks passed.")
