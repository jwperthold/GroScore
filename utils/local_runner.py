#!/usr/bin/env python3
#
# Local (non-SLURM) multi-GPU job runner for GroScore.
#
# Replaces the SLURM job array when GroScore is driven with --run-local, for the
# single fat workstations that GPU cloud providers rent out (4-8 GPUs in one box,
# no scheduler). The orchestrator writes a job list, this runner executes it in a
# bounded pool of worker slots, and every slot is pinned to one GPU through the
# GROSCORE_GPU_ID environment variable that job.run turns into `gmx mdrun -gpu_id`.
#
# Design notes:
#   * One worker thread per slot, slot k always on GPU (k % ngpus): the GPU a job
#     lands on is stable for the whole job, so nvidia-smi stays interpretable and
#     a job never migrates between devices mid-run.
#   * Jobs are pulled from a shared queue instead of being statically assigned to
#     a GPU, so one slow structure cannot leave its GPU idle while others queue.
#     With as many slots as jobs this degenerates to "start everything at once,
#     GPU = job index % ngpus".
#   * Dependencies are honoured "after any", matching the --dependency=afterany
#     the SLURM path uses for GroScore-FE: a cycle still starts when its setup
#     failed, reads the failure status and exits cleanly, rather than hanging.
#   * The runner is detached from the launching terminal (own session, output to
#     a log file), so closing the shell does not kill the simulations.

import argparse, glob, json, os, shutil, signal, subprocess, sys, tarfile, threading, time

PID_FILE    = "local_runner.pid"
JOB_FILE    = "local_jobs.json"
LOG_FILE    = "local_runner.log"
STATUS_FILE = "local_status.gs"

#------------------------------------------------------

def detected_gpus():
  """Number of GPUs nvidia-smi reports, or None if it cannot be asked."""
  try:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                         timeout=30, check=True).stdout
  except (subprocess.SubprocessError, FileNotFoundError, OSError):
    return None
  return sum(1 for line in out.splitlines() if line.startswith("GPU "))

#------------------------------------------------------

def runner_pid(rundir="."):
  """PID of the runner working in `rundir`, or None if none is alive."""
  path = os.path.join(rundir, PID_FILE)
  if not os.path.isfile(path):
    return None
  try:
    with open(path) as f:
      pid = int(f.read().split()[0])
  except (ValueError, IndexError, OSError):
    return None
  try:
    os.kill(pid, 0)                       # signal 0: existence check only
  except OSError:
    return None                           # stale file from a killed runner
  return pid

#------------------------------------------------------

def launch_local(jobs, ngpus, jobs_per_gpu, threads_per_job, rundir="."):
  """Write the job list and start a detached runner for it.

  jobs: list of dicts as documented in Job.__init__.
  Returns True if a runner was started.
  """
  if not jobs:
    print("No jobs to run locally.")
    return False

  alive = runner_pid(rundir)
  if alive is not None:
    print("A local runner is already active (pid %d)." % alive)
    print("Wait for it to finish, or stop it with 'kill %d', before resubmitting." % alive)
    return False

  slots = len(jobs) if jobs_per_gpu <= 0 else ngpus * jobs_per_gpu
  slots = max(1, min(slots, len(jobs)))

  available = detected_gpus()
  if available is not None and ngpus > available:
    print("Warning: --ngpus %d was requested but nvidia-smi reports %d GPU(s)."
          % (ngpus, available))
    print("         Jobs assigned to a non-existent device will fail in mdrun.")
  if slots > 64:
    print("Warning: %d concurrent jobs. Each is a full GROMACS process with its own"
          % slots)
    print("         system in memory; lower --jobs-per-gpu if the machine swaps.")

  spec = {"ngpus": ngpus, "slots": slots, "threads": threads_per_job, "jobs": jobs}
  job_path = os.path.join(rundir, JOB_FILE)
  with open(job_path, "w") as f:
    json.dump(spec, f, indent=1)

  runner = os.path.abspath(__file__)
  log_path = os.path.join(rundir, LOG_FILE)
  log = open(log_path, "a")
  try:
    proc = subprocess.Popen([sys.executable, runner, job_path],
                            cwd=os.path.abspath(rundir),
                            stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            start_new_session=True)   # survives the terminal, like nohup
  except OSError as e:
    print("Error: could not start the local runner: %s" % e)
    log.close()
    return False
  log.close()

  print("Started local runner (pid %d): %d job(s) over %d GPU(s), %d concurrent, "
        "%d thread(s) each." % (proc.pid, len(jobs), ngpus, slots, threads_per_job))
  print("Progress: %s and %s" % (LOG_FILE, STATUS_FILE))
  return True

#------------------------------------------------------

def incomplete_cycles(rundir=".", resultsdir="results_fe.d"):
  """[(struct_id, cycle), ...] whose row exists but has a non-numeric work.

  Mirrors the completeness rule job_fe.run uses to decide whether to write a
  .done marker: fields 3-8 are the six works and must all be numeric. Field 9 is
  the rebinding RMSD, which is diagnostic only -- a cycle whose RMSD could not be
  computed is still a valid result."""
  out = []
  for path in sorted(glob.glob(os.path.join(rundir, resultsdir, "*.gs"))):
    try:
      for line in open(path):
        f = line.split()
        if len(f) < 8 or line.lstrip().startswith("#"):
          continue
        works = f[2:8]
        if any(w.lower() == "nan" for w in works):
          out.append((f[0], f[1]))
    except OSError:
      pass
  return out


def print_local_status(rundir="."):
  """One-line progress report for a run that is being driven locally."""
  path = os.path.join(rundir, STATUS_FILE)
  if not os.path.isfile(path):
    return
  st = {}
  with open(path) as f:
    for line in f:
      if line.strip().startswith("#"):
        continue
      tmp = line.split()
      if len(tmp) >= 2:
        st[tmp[0]] = tmp[1]
  alive = runner_pid(rundir)
  state = ("runner active (pid %d)" % alive) if alive else "runner finished"
  print("Local run: %s - %s/%s jobs done, %s running, %s queued, %s failed."
        % (state, st.get("done", "?"), st.get("total", "?"), st.get("running", "?"),
           st.get("pending", "?"), st.get("failed", "?")))

  # "done" only means the process exited 0. A cycle whose legs died partway still
  # writes its row -- with NaN works -- so it can be repaired without re-running
  # the MD, and that row is invisible in the counts above. Surface it here, or the
  # only symptom is Ncycles quietly lagging the done count.
  partial = incomplete_cycles(rundir)
  if partial:
    shown = ", ".join("%s c%s" % (sid, c) for sid, c in partial[:6])
    more = "" if len(partial) <= 6 else ", +%d more" % (len(partial) - 6)
    print("             %d finished cycle%s wrote an incomplete result (%s%s);"
          % (len(partial), "" if len(partial) == 1 else "s", shown, more))
    print("             they are queued for repair on the next run.")

#------------------------------------------------------

class Job:
  """One unit of work: a command run inside one structure directory.

  name:    unique id, referenced by other jobs' `deps`
  dir:     structure directory, relative to the run directory
  argv:    command to execute there (e.g. ["./job.run", "--cycle", "3"])
  deps:    names that must have finished first (success not required)
  env:     extra environment variables
  archive: tarball to unpack if `dir` does not exist
  log:     file inside `dir` that receives stdout/stderr
  """

  def __init__(self, spec):
    self.name    = spec["name"]
    self.dir     = spec.get("dir", ".")
    self.argv    = list(spec["argv"])
    self.deps    = list(spec.get("deps", []))
    self.env     = dict(spec.get("env", {}))
    self.archive = spec.get("archive")
    self.log     = spec.get("log", "local_%s.out" % spec["name"])
    self.state   = "pending"
    self.rc      = None
    self.gpu     = None

#------------------------------------------------------

class Runner:

  def __init__(self, spec, rundir):
    self.jobs    = [Job(j) for j in spec["jobs"]]
    self.ngpus   = max(1, int(spec.get("ngpus", 1)))
    self.slots   = max(1, int(spec.get("slots", 1)))
    self.threads = max(1, int(spec.get("threads", 1)))
    self.rundir  = rundir
    self.cond    = threading.Condition()      # guards job states, running, procs
    self.extract = threading.Lock()           # one tarball unpacked at a time
    self.status  = threading.Lock()           # one writer of the status file
    self.running = 0
    self.stopping = False
    self.procs   = {}
    self.started = time.time()
    # State is tracked incrementally rather than recomputed by scanning self.jobs:
    # a CAPRI-sized screen is tens of thousands of jobs, and every worker wake-up
    # would otherwise walk the whole list.
    self.fin     = set()                      # names of jobs in a terminal state
    self.counts  = {"pending": len(self.jobs), "running": 0,
                    "done": 0, "failed": 0, "skipped": 0}
    self.cursor  = 0                          # no job below this index is pending

  #----------------------------------------------------

  def say(self, msg):
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)

  #----------------------------------------------------

  def _set_state(self, job, state):
    """Move a job to a new state, keeping the counters in sync. Caller holds cond."""
    self.counts[job.state] -= 1
    job.state = state
    self.counts[state] = self.counts.get(state, 0) + 1
    if state in ("done", "failed", "skipped"):
      self.fin.add(job.name)

  def _next_runnable(self):
    """First pending job whose dependencies have all finished. Caller holds cond."""
    # Jobs only ever leave "pending", so everything below the cursor can be skipped
    # for good; the scan past it is only long while dependencies block the queue.
    while self.cursor < len(self.jobs) and self.jobs[self.cursor].state != "pending":
      self.cursor += 1
    for i in range(self.cursor, len(self.jobs)):
      job = self.jobs[i]
      if job.state == "pending" and all(d in self.fin for d in job.deps):
        return job
    return None

  def _pending(self):
    return self.counts["pending"]

  #----------------------------------------------------

  def _extract(self, job, workdir):
    """Unpack an archived structure. Serialised: several cycles of the same
    structure can become runnable at the same instant."""
    with self.extract:
      if os.path.isdir(workdir):
        return
      arc = os.path.join(self.rundir, job.archive)
      if not os.path.isfile(arc):
        return
      self.say("%s: unpacking %s" % (job.name, job.archive))
      with tarfile.open(arc, "r:gz") as tar:
        try:
          tar.extractall(self.rundir, filter="data")   # Python >= 3.12
        except TypeError:
          tar.extractall(self.rundir)
      os.remove(arc)
      # .archive.lock is the election marker job.run uses to decide who archives
      # the structure; it lives inside the tarball, so a re-opened structure must
      # lose it or it could never be archived again.
      lock = os.path.join(workdir, ".archive.lock")
      if os.path.isdir(lock):
        shutil.rmtree(lock, ignore_errors=True)

  #----------------------------------------------------

  def _run(self, job, gpu):
    workdir = os.path.join(self.rundir, job.dir)
    if job.archive and not os.path.isdir(workdir):
      self._extract(job, workdir)
    if not os.path.isdir(workdir):
      self.say("%s: directory %s does not exist, skipped" % (job.name, job.dir))
      return 127

    env = dict(os.environ)
    env["GROSCORE_GPU_ID"] = str(gpu)      # -> gmx mdrun -gpu_id in job.run
    env["GROSCORE_NT"]     = str(self.threads)
    env.update(job.env)

    try:
      out = open(os.path.join(workdir, job.log), "a")
    except OSError as e:
      self.say("%s: cannot open log: %s" % (job.name, e))
      return 126
    try:
      out.write("\n===== %s on GPU %d, %s =====\n"
                % (job.name, gpu, time.strftime("%Y-%m-%d %H:%M:%S")))
      out.flush()
      try:
        proc = subprocess.Popen(job.argv, cwd=workdir, env=env, stdout=out,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
      except OSError as e:
        self.say("%s: cannot start %s: %s" % (job.name, " ".join(job.argv), e))
        return 126
      with self.cond:
        self.procs[job.name] = proc
      rc = proc.wait()
    finally:
      out.close()
      with self.cond:
        self.procs.pop(job.name, None)
    return rc

  #----------------------------------------------------

  def _worker(self, slot):
    gpu = slot % self.ngpus
    while True:
      with self.cond:
        while True:
          if self.stopping:
            return
          job = self._next_runnable()
          if job is not None:
            break
          if self._pending() == 0:
            return
          if self.running == 0:
            # Pending jobs, nothing running, nothing runnable: the remaining
            # dependencies can never be satisfied (unknown or circular name).
            for j in self.jobs:
              if j.state == "pending":
                self._set_state(j, "skipped")
                self.say("%s: skipped, dependencies %s can never run"
                         % (j.name, ",".join(j.deps)))
            self.cond.notify_all()
            return
          self.cond.wait()
        self._set_state(job, "running")
        job.gpu = gpu
        self.running += 1
      self.say("%s: started on GPU %d (slot %d)" % (job.name, gpu, slot))

      # Whatever happens, the job must reach a terminal state and the slot must be
      # released: an exception escaping here would kill the worker thread and leave
      # the job "running" forever, silently shrinking the pool.
      rc = 125
      try:
        self._write_status()
        rc = self._run(job, gpu)
      except Exception as e:
        self.say("%s: runner error: %r" % (job.name, e))
      finally:
        with self.cond:
          job.rc = rc
          self._set_state(job, "done" if rc == 0 else "failed")
          self.running -= 1
          self.cond.notify_all()
        self.say("%s: %s (exit %s)" % (job.name, job.state, rc))
        try:
          self._write_status()
        except Exception as e:
          self.say("could not write %s: %r" % (STATUS_FILE, e))

  #----------------------------------------------------

  def _write_status(self):
    with self.cond:
      counts = dict(self.counts)
      body = ("# GroScore local runner status\n"
              "pid\t%d\n"
              "started\t%s\n"
              "gpus\t%d\n"
              "slots\t%d\n"
              "threads\t%d\n"
              "total\t%d\n"
              "done\t%d\n"
              "failed\t%d\n"
              "skipped\t%d\n"
              "running\t%d\n"
              "pending\t%d\n"
              % (os.getpid(), time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(self.started)),
                 self.ngpus, self.slots, self.threads, len(self.jobs),
                 counts["done"], counts["failed"], counts["skipped"],
                 counts["running"], counts["pending"]))
    # Written through a private temp file so a reader never sees a half-written
    # status, and serialised because every slot writes this on every transition:
    # two threads sharing one temp name would have the first os.replace pull the
    # file out from under the second.
    path = os.path.join(self.rundir, STATUS_FILE)
    with self.status:
      tmp = "%s.%d.tmp" % (path, threading.get_ident())
      with open(tmp, "w") as f:
        f.write(body)
      os.replace(tmp, path)

  #----------------------------------------------------

  def shutdown(self, signum, frame):
    with self.cond:
      if self.stopping:
        return
      self.stopping = True
      procs = list(self.procs.values())
      self.cond.notify_all()
    self.say("Received signal %d, terminating %d running job(s)." % (signum, len(procs)))
    for p in procs:
      try:
        p.terminate()
      except OSError:
        pass

  #----------------------------------------------------

  def run(self):
    signal.signal(signal.SIGTERM, self.shutdown)
    signal.signal(signal.SIGINT, self.shutdown)
    self.say("GroScore local runner: %d job(s), %d slot(s) over %d GPU(s), "
             "%d thread(s) per job." % (len(self.jobs), self.slots, self.ngpus,
                                        self.threads))
    self._write_status()

    workers = [threading.Thread(target=self._worker, args=(k,), daemon=True)
               for k in range(self.slots)]
    for w in workers:
      w.start()
    for w in workers:
      w.join()

    self._write_status()
    ok = [j for j in self.jobs if j.state == "done"]
    bad = [j for j in self.jobs if j.state in ("failed", "skipped")]
    self.say("Finished in %.1f h: %d succeeded, %d failed or skipped."
             % ((time.time() - self.started) / 3600.0, len(ok), len(bad)))
    for j in bad:
      self.say("  %-30s %s (exit %s)" % (j.name, j.state, j.rc))
    return 0 if not bad else 1

#------------------------------------------------------

def main():
  ap = argparse.ArgumentParser(
    description="Run GroScore jobs on the local machine, distributed over GPUs.")
  ap.add_argument("jobfile", nargs="?", default=JOB_FILE,
                  help="Job list written by groscore.py --run-local (default: %s)" % JOB_FILE)
  a = ap.parse_args()

  if not os.path.isfile(a.jobfile):
    print("Error: job file %s not found." % a.jobfile)
    return 1
  with open(a.jobfile) as f:
    spec = json.load(f)
  rundir = os.path.dirname(os.path.abspath(a.jobfile))

  pid_path = os.path.join(rundir, PID_FILE)
  with open(pid_path, "w") as f:
    f.write("%d\n" % os.getpid())
  try:
    rc = Runner(spec, rundir).run()
  finally:
    try:
      os.remove(pid_path)
    except OSError:
      pass
  return rc

if __name__ == "__main__":
  sys.exit(main())
