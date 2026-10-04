#!/usr/bin/env python3
"""utils/local_runner.py unpacks an archived structure completely or not at all.

The runner unpacked a structure's tarball in place. An extraction that failed part
way (a damaged or truncated tarball, a full disk) left a partial directory behind,
and since the archive is only unpacked when the directory is missing, the next
start took that directory for the structure and ran job.run on it (v15_work
integrator, leftover of runner R4). Pinned, with the real runner on a job list:

  [1] a truncated tarball: no structure directory is left, the archive is kept,
      the runner says why, and the job fails without running job.run.
  [2] an intact tarball: unpacked, the archive removed, job.run run in it.

Run:  python3 tests/test_local_runner_extract.py
"""
import io, json, os, shutil, subprocess, sys, tarfile, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, "utils", "local_runner.py")
failures = []


def check(name, ok, detail=""):
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:300]))
    if not ok:
        failures.append(name)


def archive(run, sid):
    """<sid>.tar.gz of <sid>/ holding job.run (writes ran.txt) and 400 kB of noise."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        body = b"#!/bin/bash\necho ran > ran.txt\n"
        ti = tarfile.TarInfo("%s/job.run" % sid)
        ti.size, ti.mode = len(body), 0o755
        t.addfile(ti, io.BytesIO(body))
        noise = os.urandom(400000)
        ti = tarfile.TarInfo("%s/noise.bin" % sid)
        ti.size = len(noise)
        t.addfile(ti, io.BytesIO(noise))
    data = buf.getvalue()
    with open(os.path.join(run, "%s.tar.gz" % sid), "wb") as f:
        f.write(data)
    return len(data)


def run_runner(run, sid):
    with open(os.path.join(run, "local_jobs.json"), "w") as f:
        json.dump({"ngpus": 1, "slots": 1, "threads": 1,
                   "jobs": [{"name": sid, "dir": sid, "argv": ["./job.run"],
                             "archive": "%s.tar.gz" % sid, "log": "job_local.out"}]}, f)
    p = subprocess.run([sys.executable, RUNNER, os.path.join(run, "local_jobs.json")], cwd=run,
                       capture_output=True, text=True, timeout=300)
    log = os.path.join(run, "local_runner.log")
    return p, (open(log).read() if os.path.isfile(log) else p.stdout + p.stderr)


work = tempfile.mkdtemp(prefix="local_runner_extract_")
try:
    print("\n[1] a truncated tarball")
    run = os.path.join(work, "bad")
    os.makedirs(run)
    n = archive(run, "S1")
    with open(os.path.join(run, "S1.tar.gz"), "r+b") as f:     # cut inside the noise member
        f.truncate(n // 2)
    p, log = run_runner(run, "S1")
    check("no S1/ is left behind (it would be taken for the structure)",
          not os.path.exists(os.path.join(run, "S1")), sorted(os.listdir(run)))
    check("  and no unpacking directory either", not [x for x in os.listdir(run) if x.startswith(".unpack")],
          sorted(os.listdir(run)))
    check("  the archive is kept", os.path.isfile(os.path.join(run, "S1.tar.gz")))
    check("  the runner says it could not unpack it, and the job failed",
          "could not unpack S1.tar.gz" in log and "S1: failed" in log and p.returncode == 1, log[-400:])
    p, log = run_runner(run, "S1")
    check("  a second start does the same, and job.run never ran",
          not os.path.exists(os.path.join(run, "S1")) and "could not unpack S1.tar.gz" in log, log[-300:])

    print("\n[2] an intact tarball")
    run = os.path.join(work, "good")
    os.makedirs(run)
    archive(run, "S2")
    p, log = run_runner(run, "S2")
    check("unpacked, archive removed, job.run run in it",
          os.path.isfile(os.path.join(run, "S2", "ran.txt")) and not os.path.exists(os.path.join(run, "S2.tar.gz"))
          and "S2: done (exit 0)" in log, (sorted(os.listdir(run)), log[-300:]))
    check("  no unpacking directory left", not [x for x in os.listdir(run) if x.startswith(".unpack")],
          sorted(os.listdir(run)))
finally:
    shutil.rmtree(work, ignore_errors=True)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all local-runner extraction checks passed")
