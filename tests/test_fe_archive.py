#!/usr/bin/env python3
"""job_fe.run archives a finished structure only behind a tested archive, says so,
can archive one that was left a directory, and says where and when it ran.

bm_fe_amber_3a: the archive step printed nothing on success, never tested the
gzip, and ran only in the task that finished the last cycle, so 1AKJ, whose tar met
a full disk, stayed a directory that no --restart touched again. Its logs named no
node and held no time.

  [1] archive_structure, extracted and driven in a scratch run directory: a
      complete structure becomes <ID>.tar.gz that gzip -t passes, written as .part
      and renamed, job_run.lock and .archive.lock left out, results_analytical.d
      written, one line with the size, also in the run directory's archive.log; an
      incomplete one, or one whose .archive.lock is taken, is left alone with the
      reason; a failing gzip test or tar keeps the directory, leaves no .part and no
      marker, and keeps an older <ID>.tar.gz; a directory that cannot be removed
      completely is reported as such. archive_abort says how far an interrupted
      archive got, at each stage.
  [2] job_fe.run --archive end to end: archives a complete structure (exit 0),
      refuses (exit 1, directory untouched) an incomplete one, one with an
      <ID>.tar.gz beside it, and one whose cycle lock a live process holds; a lock
      left by a dead process is reclaimed. The total: GROSCORE_NUMCYCLES, else
      numruns in ../run_config.gs, else run.gs, and it says which. TERM while it
      tars removes the .part and the marker and exits 143.
  [3] Every task prints a start and an end line with time, host, SLURM job and the
      GPUs nvidia-smi lists; a failing, hanging or missing nvidia-smi is said so.
      A TERM during a cycle task's wait for its setup ends it at once, and the end
      line says exit 143.
  [4] The cycle path archives through the same function; there is one tar in the
      script; every EXIT trap releases its locks and then prints the end line, and
      every INT/TERM trap that releases a lock calls archive_abort first.

Standalone, no pytest: python3 tests/test_fe_archive.py
"""
import gzip, hashlib, os, re, shutil, signal, subprocess, sys, tarfile, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = os.path.join(ROOT, "job_fe.run")
SRC = open(JOB).read()
HOST = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
REAL_GZIP = shutil.which("gzip")
fails = 0


def check(name, ok, detail=""):
    global fails
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:400]))
    if not ok:
        fails += 1


def put(path, text="x\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def exe(path, text):
    put(path, text)
    os.chmod(path, 0o755)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def structure(run, sid, done=(1, 2), runs="A 4 1 amber19sb_opc3 1"):
    """A finished structure directory: run.gs (2 cycles), .done markers, the
    analytical term, some output, a held job_run.lock."""
    d = os.path.join(run, sid)
    put(os.path.join(d, "run.gs"), runs + "\n")
    put(os.path.join(d, "boresch_analytical.gs"), "dG_release_kJ_mol -33.5\n")
    put(os.path.join(d, "bindrevA_1_dhdl.xvg.gz"), "data\n" * 200)
    for c in done:
        put(os.path.join(d, "results_fe_c%d.done" % c), "")
    put(os.path.join(d, "job_run.lock", "owner"), "%s:1:-\n" % HOST)
    shutil.copy(JOB, os.path.join(d, "job.run"))
    os.chmod(os.path.join(d, "job.run"), 0o755)
    return d


def members(path):
    with tarfile.open(path) as t:
        return t.getnames()


GPU_OK = ('#!/bin/bash\necho "$*" >> "${NVLOG:-/dev/null}"\n'
          'echo "0, NVIDIA A40, 00000000:41:00.0, 1 MiB, 46068 MiB, 0 %"\n')
GPU_LOST = ('#!/bin/bash\n'
            'echo "Unable to determine the device handle for GPU0000:41:00.0: Unknown Error"\n'
            'exit 255\n')


with tempfile.TemporaryDirectory() as tmp:
    co = os.path.join(tmp, "co")
    os.makedirs(os.path.join(co, "utils"))
    shutil.copy(os.path.join(ROOT, "utils", "fe_protocol.py"), os.path.join(co, "utils"))
    gpubin = os.path.join(tmp, "gpubin")
    exe(os.path.join(gpubin, "nvidia-smi"), GPU_OK)

    def env(*bins, **extra):
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("SLURM_", "GROSCORE_", "CUDA_VISIBLE"))}
        e.update(PATH=os.pathsep.join(list(bins) + [gpubin, os.environ["PATH"]]),
                 PYTHONDONTWRITEBYTECODE="1")
        e.update(extra)
        return e

    # ---------------------------------------------------------------------
    print("\n[1] archive_structure itself")
    def fn(name):
        m = re.search(r"^%s\(\) \{\n.*?^\}\n" % name, SRC, re.S | re.M)
        return m.group(0) if m else ""

    func = fn("archive_structure")
    check("archive_structure, archive_note and archive_abort are top-level functions",
          func and fn("archive_note") and fn("archive_abort"))
    PRE = ('ARCHIVE_WHY=""; ARCHIVE_PART=""; ARCHIVE_LOCK=""; ARCHIVE_LOG=""; ARCHIVE_STAGE="";'
           ' ARCHIVE_HAD_OLD=""\n' + fn("archive_note") + fn("archive_abort") + func)

    def drive(run, sid, numcycles=2, bins=()):
        script = (PRE + "STRUCT_ID=%s; NUMCYCLES=%d\ncd %s\narchive_structure\nrc=$?\n"
                  'echo "RC=$rc PWD=$PWD WHY=$ARCHIVE_WHY"\n' % (sid, numcycles, os.path.join(run, sid)))
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              env=env(*bins))

    def alog(run):
        p = os.path.join(run, "archive.log")
        return open(p).read() if os.path.exists(p) else ""

    run = os.path.realpath(os.path.join(tmp, "r1"))
    d = structure(run, "S")
    os.makedirs(os.path.join(d, ".archive.lock.unrelated"))
    r = drive(run, "S")
    arc = os.path.join(run, "S.tar.gz")
    check("complete: returns 0 and leaves the shell in the run directory",
          "RC=0 PWD=%s " % run in r.stdout, r.stdout + r.stderr)
    check("  S.tar.gz exists, gzip -t passes, the directory is gone, no .part",
          os.path.isfile(arc) and subprocess.run([REAL_GZIP, "-t", arc]).returncode == 0
          and not os.path.isdir(d) and not os.path.exists(arc + ".part"))
    names = members(arc) if os.path.isfile(arc) else []
    check("  the archive holds the cycles and leaves job_run.lock and .archive.lock out",
          "S/results_fe_c2.done" in names and "S/.archive.lock.unrelated" in names
          and not [n for n in names if n.endswith(("job_run.lock", ".archive.lock"))
                   or "job_run.lock/" in n or ".archive.lock/" in n], names)
    size = os.path.getsize(arc) if os.path.isfile(arc) else -1
    line = "Archived S to S.tar.gz (%d bytes, gzip test passed); directory removed." % size
    check("  one line says so, with the size in bytes", line in r.stdout, r.stdout)
    check("  and archive.log in the run directory has it, with time and host",
          re.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4} %s %s\n$"
                   % (re.escape(HOST), re.escape(line)), alog(run)), alog(run))
    check("  results_analytical.d/S.gs is written",
          open(os.path.join(run, "results_analytical.d", "S.gs")).read() == "S -33.5\n")

    run = os.path.join(tmp, "r2")
    d = structure(run, "S", done=(1,))
    r = drive(run, "S")
    check("incomplete: returns 2 with the count, nothing written",
          "RC=2" in r.stdout and "1 of 2 cycles have their results_fe_c<N>.done" in r.stdout
          and os.listdir(run) == ["S"] and not os.path.isdir(os.path.join(d, ".archive.lock")),
          (r.stdout, os.listdir(run)))
    d = structure(os.path.join(tmp, "r3"), "S")
    os.makedirs(os.path.join(d, ".archive.lock"))
    r = drive(os.path.join(tmp, "r3"), "S")
    check("another task's .archive.lock: returns 2, says what to remove if it is stale",
          "RC=2" in r.stdout and "another task holds S/.archive.lock" in r.stdout
          and "remove S/.archive.lock and any S.tar.gz.part by hand" in r.stdout
          and os.listdir(os.path.join(tmp, "r3")) == ["S"], r.stdout)

    badgz = os.path.join(tmp, "badgz")
    exe(os.path.join(badgz, "gzip"),
        '#!/bin/bash\nfor a in "$@"; do if [[ $a == -t ]]; then\n'
        '  echo "gzip: ${@: -1}: unexpected end of file" >&2; exit 1; fi; done\n'
        'exec %s "$@"\n' % REAL_GZIP)
    run = os.path.join(tmp, "r4")
    d = structure(run, "S")
    put(os.path.join(run, "S.tar.gz"), "an older archive\n")
    old = md5(os.path.join(run, "S.tar.gz"))
    r = drive(run, "S", bins=(badgz,))
    check("a failing gzip test: returns 1 and keeps the directory",
          "RC=1" in r.stdout and os.path.isdir(d)
          and "Archive failed or empty for S (tar, the gzip test or the rename failed" in r.stdout,
          r.stdout + r.stderr)
    check("  gzip's complaint is in the log", "unexpected end of file" in r.stderr, r.stderr)
    check("  no .part, no .archive.lock, and the older S.tar.gz is untouched",
          not os.path.exists(os.path.join(run, "S.tar.gz.part"))
          and not os.path.exists(os.path.join(d, ".archive.lock"))
          and md5(os.path.join(run, "S.tar.gz")) == old)
    check("  archive.log has the failure", "Archive failed or empty for S" in alog(run), alog(run))
    badtar = os.path.join(tmp, "badtar")
    exe(os.path.join(badtar, "tar"), '#!/bin/bash\necho "tar: write error" >&2\nexit 2\n')
    run = os.path.join(tmp, "r5")
    d = structure(run, "S")
    r = drive(run, "S", bins=(badtar,))
    # results_analytical.d is written before the tar, on purpose: the scorer needs the
    # term whether or not the archive works.
    check("a failing tar: returns 1, keeps the directory, leaves no archive and no marker",
          "RC=1" in r.stdout and os.path.isdir(d)
          and sorted(os.listdir(run)) == ["S", "archive.log", "results_analytical.d"]
          and not os.path.exists(os.path.join(d, ".archive.lock")), (r.stdout, os.listdir(run)))

    # A directory that rm cannot empty (a read-only subdirectory here; on NFS, a
    # file another process still holds open).
    run = os.path.join(tmp, "r6")
    d = structure(run, "S")
    put(os.path.join(d, "ro", "f"), "x\n")
    os.chmod(os.path.join(d, "ro"), 0o555)
    r = drive(run, "S")
    os.chmod(os.path.join(d, "ro"), 0o755)
    check("a directory not removed completely: archived, and said so",
          "RC=0" in r.stdout and os.path.isfile(os.path.join(run, "S.tar.gz"))
          and "passed), but S/ could not be removed completely; remove it by hand." in r.stdout,
          r.stdout)

    # archive_abort at each stage (the INT and TERM traps call it).
    def abort(stage, part=True, final=False, had_old=False):
        base = tempfile.mkdtemp(dir=tmp)
        os.makedirs(os.path.join(base, "S", ".archive.lock"))
        if part:
            put(os.path.join(base, "S.tar.gz.part"))
        if final:
            put(os.path.join(base, "S.tar.gz"))
        script = (PRE + "STRUCT_ID=S; cd %s; ARCHIVE_STAGE=%s; ARCHIVE_PART=$PWD/S.tar.gz.part;"
                  " ARCHIVE_LOCK=$PWD/S/.archive.lock; ARCHIVE_LOG=$PWD/archive.log;"
                  " ARCHIVE_HAD_OLD=%s\narchive_abort\n" % (base, stage, "1" if had_old else ""))
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env())
        return (r.stdout, os.path.exists(os.path.join(base, "S.tar.gz.part")),
                os.path.isdir(os.path.join(base, "S", ".archive.lock")), alog(base))

    out, part, lock, lg = abort("prep", part=False)
    check("abort before the tar: the marker removed, 'before its tar'",
          "interrupted before its tar; directory kept." in out and not lock and out.strip() in lg, out)
    out, part, lock, lg = abort("tar")
    check("abort during the tar: .part and marker removed",
          "S.tar.gz.part removed, directory kept." in out and not part and not lock, out)
    out, part, lock, lg = abort("tar", part=False, final=True)
    check("abort right after the rename: the archive is said to be whole, marker kept",
          "after S.tar.gz passed its gzip test; S/ was not removed" in out and lock, out)
    out, part, lock, lg = abort("tar", part=True, final=True, had_old=True)
    check("  but an older S.tar.gz beside a .part is not taken for the new one",
          "S.tar.gz.part removed, directory kept." in out and not part, out)
    out, part, lock, lg = abort("rm", part=False, final=True)
    check("abort while removing the directory: the archive is complete, remove the rest",
          "S.tar.gz is complete (gzip test passed), but removing S/ was interrupted." in out, out)
    out, part, lock, lg = abort("", part=True)
    check("no archive under way: archive_abort does nothing", out == "" and part and lock, out)

    # ---------------------------------------------------------------------
    print("\n[2] job_fe.run --archive")

    def job(run, sid, *bins, popen=False, **extra):
        argv = ["bash", "./job.run", "--archive"]
        kw = dict(cwd=os.path.join(run, sid), env=env(*bins, **extra), text=True)
        if popen:
            return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True, **kw)
        return subprocess.run(argv, capture_output=True, timeout=300, **kw)

    def fresh(name, **kw):
        run = os.path.join(co, name)
        d = structure(run, "S", **kw)
        shutil.rmtree(os.path.join(d, "job_run.lock"))
        return run, d

    run, d = fresh("a1")
    r = job(run, "S")
    arc = os.path.join(run, "S.tar.gz")
    check("complete: exit 0, archived, the directory gone",
          r.returncode == 0 and os.path.isfile(arc) and not os.path.isdir(d),
          (r.returncode, r.stdout[-400:], r.stderr[-400:]))
    check("  the success line is printed", "Archived S to S.tar.gz (" in r.stdout, r.stdout[-400:])
    check("  neither its job_run.lock nor .archive.lock is in the archive",
          os.path.isfile(arc) and not [n for n in members(arc) if "lock" in n],
          members(arc) if os.path.isfile(arc) else None)
    check("  the total came from run.gs (no run_config.gs here), and it says so",
          "Archiving S: 2 cycles wanted (run.gs)." in r.stdout, r.stdout[-600:])

    run, d = fresh("a2", done=(1,))
    r = job(run, "S")
    check("incomplete: exit 1, says why, directory untouched, lock released",
          r.returncode == 1 and "Not archived: 1 of 2 cycles have their results_fe_c<N>.done." in r.stdout
          and os.path.isdir(d) and not os.path.exists(os.path.join(d, "job_run.lock"))
          and os.listdir(run) == ["S"], (r.returncode, r.stdout[-300:]))
    # run.gs keeps the -n of the first submission; run_config.gs follows -n.
    put(os.path.join(run, "run_config.gs"), "# remembered\nnumruns\t3\n")
    r = job(run, "S")
    check("  numruns 3 in ../run_config.gs wins over run.gs (2): 1 of 3",
          r.returncode == 1 and "3 cycles wanted (numruns in ../run_config.gs)" in r.stdout
          and "1 of 3 cycles" in r.stdout, r.stdout[-400:])
    r = job(run, "S", GROSCORE_NUMCYCLES="1")
    check("  GROSCORE_NUMCYCLES=1 wins over both: archived",
          r.returncode == 0 and "1 cycles wanted (GROSCORE_NUMCYCLES)" in r.stdout
          and os.path.isfile(os.path.join(run, "S.tar.gz")), r.stdout[-300:])

    run, d = fresh("a3")
    put(os.path.join(run, "S.tar.gz"), "older\n")
    old = md5(os.path.join(run, "S.tar.gz"))
    r = job(run, "S")
    check("an S.tar.gz beside it: exit 1, both untouched",
          r.returncode == 1 and "Not archived: ../S.tar.gz exists already" in r.stdout
          and os.path.isdir(d) and md5(os.path.join(run, "S.tar.gz")) == old, r.stdout[-300:])

    run, d = fresh("a4")
    put(os.path.join(d, "results_fe_c2.lock", "owner"), "%s:%d:-\n" % (HOST, os.getpid()))
    r = job(run, "S")
    check("a cycle lock held by a live process: exit 1, directory untouched",
          r.returncode == 1 and "results_fe_c2.lock is held by %s:%d:-" % (HOST, os.getpid()) in r.stdout
          and os.path.isdir(os.path.join(d, "results_fe_c2.lock")), r.stdout[-300:])
    p = subprocess.Popen(["true"])
    p.wait()
    shutil.rmtree(os.path.join(d, "results_fe_c2.lock"))
    put(os.path.join(d, "results_fe_c2.lock", "owner"), "%s:%d:-\n" % (HOST, p.pid))
    r = job(run, "S")
    check("a cycle lock left by a dead process: reclaimed, archived, not in the archive",
          r.returncode == 0 and "reclaiming stale lock" in r.stdout
          and os.path.isfile(os.path.join(run, "S.tar.gz"))
          and not [n for n in members(os.path.join(run, "S.tar.gz"))
                   if "results_fe_c2.lock" in n or "job_run.lock" in n],
          r.stdout[-400:])

    slowtar = os.path.join(tmp, "slowtar")
    exe(os.path.join(slowtar, "tar"), '#!/bin/bash\necho partial > "$2"\nexec sleep 60\n')
    run, d = fresh("a5")
    p = job(run, "S", slowtar, popen=True)
    part = os.path.join(run, "S.tar.gz.part")
    for _ in range(200):
        if os.path.exists(part):
            break
        time.sleep(0.05)
    time.sleep(0.2)
    os.killpg(p.pid, signal.SIGTERM)
    out, err = p.communicate(timeout=60)
    check("TERM while it tars: exit 143, .part and .archive.lock removed, directory kept",
          p.returncode == 143 and not os.path.exists(part)
          and not os.path.exists(os.path.join(d, ".archive.lock")) and os.path.isdir(d)
          and not os.path.exists(os.path.join(d, "job_run.lock")),
          (p.returncode, os.listdir(run), out[-300:]))
    check("  and it says so", "Archive of S interrupted; S.tar.gz.part removed, directory kept." in out,
          out[-300:])
    check("  the end line still comes, with exit 143",
          re.search(r"^Task end: .* on %s, exit 143, after \d+ s$" % re.escape(HOST), out, re.M),
          out[-300:])

    # ---------------------------------------------------------------------
    print("\n[3] start and end lines")
    run, d = fresh("b1")
    r = job(run, "S", NVLOG=os.path.join(tmp, "nv.log"))
    first = r.stdout.splitlines()[:3]
    check("the first line: time, host, mode and directory",
          first and re.match(r"^Task start: \d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4} on %s, "
                             r"job_fe\.run archive in %s$" % (re.escape(HOST), re.escape(d)), first[0]),
          first)
    check("  then the SLURM job and the GPU variables, unset here",
          len(first) > 1 and first[1] == "  SLURM job -, CUDA_VISIBLE_DEVICES=unset, GROSCORE_GPU_ID=unset",
          first)
    check("  then each GPU nvidia-smi lists",
          len(first) > 2 and first[2] == "  GPU 0, NVIDIA A40, 00000000:41:00.0, 1 MiB, 46068 MiB, 0 %",
          first)
    check("  queried by index, name, bus id, memory and load, as CSV",
          "--query-gpu=index,name,pci.bus_id,memory.used,memory.total,utilization.gpu"
          in open(os.path.join(tmp, "nv.log")).read())
    check("the end line: time, host, exit status and duration, then the GPUs again",
          re.search(r"^Task end: \d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4} on %s, exit 0, after \d+ s\n"
                    r"  SLURM job -, .*\n  GPU 0, NVIDIA A40" % re.escape(HOST), r.stdout, re.M),
          r.stdout[-400:])
    run, d = fresh("b2", done=(1,))
    lost = os.path.join(tmp, "lostgpu")
    exe(os.path.join(lost, "nvidia-smi"), GPU_LOST)
    r = job(run, "S", lost, SLURM_ARRAY_JOB_ID="8807218", SLURM_ARRAY_TASK_ID="17",
            SLURM_JOB_ID="8807250", CUDA_VISIBLE_DEVICES="0")
    check("an array task names its job as <array>_<task>, with CUDA_VISIBLE_DEVICES",
          "  SLURM job 8807218_17, CUDA_VISIBLE_DEVICES=0, GROSCORE_GPU_ID=unset" in r.stdout,
          r.stdout[:300])
    check("a failing nvidia-smi is printed with its exit status and message",
          "  GPU: nvidia-smi failed (exit 255): Unable to determine the device handle for "
          "GPU0000:41:00.0: Unknown Error" in r.stdout, r.stdout[:400])
    check("  and a refused --archive ends with 'exit 1'",
          re.search(r"^Task end: .*, exit 1, after \d+ s$", r.stdout, re.M), r.stdout[-300:])
    r = subprocess.run(["bash", "-c", 'SLURM_JOB_ID=4242; ' + re.search(
        r"^task_banner\(\) \{\n.*?^\}\n", SRC, re.S | re.M).group(0) +
        'gpu_report() { :; }\ntask_banner start x'], capture_output=True, text=True, env=env())
    check("a plain SLURM job is named by SLURM_JOB_ID", "  SLURM job 4242," in r.stdout, r.stdout)

    gr = fn("gpu_report")
    hang = os.path.join(tmp, "hanggpu")
    exe(os.path.join(hang, "nvidia-smi"), "#!/bin/bash\ntrap '' TERM\nexec sleep 120\n")
    t0 = time.time()
    r = subprocess.run(["bash", "-c", gr + "gpu_report"], capture_output=True, text=True,
                       env=env(hang), timeout=120)
    took = time.time() - t0
    check("an nvidia-smi that does not answer: left behind after 30 s, and said so",
          r.stdout == "  GPU: nvidia-smi did not answer within 30 s\n" and 29 <= took < 45,
          (r.stdout, round(took, 1)))
    r = subprocess.run([shutil.which("bash"), "-c", gr + "gpu_report"], capture_output=True,
                       text=True, env=dict(env(), PATH="/nonexistent"))
    check("no nvidia-smi at all: said so",
          r.stdout == "  GPU: no nvidia-smi on this host\n", (r.stdout, r.stderr))

    # A cycle task waiting for its setup (here forever: no setup.done) is stopped by
    # a TERM to bash alone, as the local runner sends it: at once, and its end line
    # says 143. Without a TERM trap it read 'exit 1', the status of the last test.
    run, d = fresh("b3", done=())
    p = subprocess.Popen(["bash", "./job.run", "--cycle", "1"], cwd=d, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                         env=env(GROSCORE_SETUP_POLL="30", GROSCORE_SETUP_WAIT="600"))
    time.sleep(3)                     # well inside the first 30 s sleep of the wait
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    try:
        rc = p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        rc = None
    took = time.time() - t0
    try:
        os.killpg(p.pid, signal.SIGKILL)  # the wait's sleep, left running, holds the pipe
    except ProcessLookupError:
        pass
    out, err = p.communicate(timeout=60)
    check("TERM during the setup wait: exit 143 within 3 s",
          "waiting for shared setup of S" in out and rc == 143 and took < 3,
          (rc, round(took, 1), out[-300:]))
    check("  and the end line says exit 143",
          re.search(r"^Task end: .*, exit 143, after \d+ s$", out, re.M), out[-300:])

    # ---------------------------------------------------------------------
    print("\n[4] one archive step, and the traps")
    check("the cycle path calls archive_structure, then says why it did not archive",
          re.search(r'^  archive_structure\n  if \[\[ \$\? -eq 2 && "\$ARCHIVE_WHY" == "another task"\* \]\]; then$',
                    SRC, re.M) is not None)
    check("and the script has exactly one tar -czf", SRC.count("tar -czf") == 1, SRC.count("tar -czf"))
    traps = re.findall(r"^\s*trap '([^']*)' (EXIT|INT|TERM)$", SRC, re.M)
    exits = [t for t, s in traps if s == "EXIT"]
    check("three EXIT traps, each releasing first and printing the end line last",
          len(exits) == 3 and all(t.startswith("_rc=$?; ") and
                                  t.endswith('task_banner end "exit $_rc, after ${SECONDS} s"')
                                  for t in exits), exits)
    sig = [t for t, s in traps if s in ("INT", "TERM") and "release_" in t]
    check("every INT/TERM trap that releases a lock calls archive_abort before it",
          len(sig) == 4 and all(t.index("archive_abort") < t.index("release_") for t in sig), sig)
    r = subprocess.run(["bash", "-n", JOB], capture_output=True, text=True)
    check("bash -n job_fe.run", r.returncode == 0, r.stderr[:300])

print()
sys.exit(1 if fails else 0)
