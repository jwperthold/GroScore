#!/usr/bin/env python3
"""The FE protocol names itself, and the runner pairing is checked both ways.

v1.5 kept the leg names and the 43-field result row, so nothing in a structure
directory or a row said which protocol made it. The review of 2026-10-04 found three
ways to mix protocols without a word: a pre-v1.5 directory set up again went on with
its v1.4 cycles in v1.5 legs (CYC-1), an archived v1.4 job.run ran the v1.5 table as a
plain chain (CYC-2), and v1.4 rows counted as finished v1.5 cycles (CYC-4).
fe_protocol.PROTOCOL_ID is the answer they share (contract F5).

  [1] THE ID of this table is written out below, so any change to the cycle has to
      change this file too, on purpose.
  [2] IT IS A FUNCTION OF THE TABLE ALONE: the same in fresh interpreters under
      different hash seeds.
  [3] WHAT IT COVERS. A one-leg length, a stage boundary, boundrev2's branch, a
      hold's pull block, the npt_c length and the row layout each change it; the
      setup shape (probe count, probe length) and the trajectory interval do not,
      since they change how restraints are measured, not what a cycle runs.
  [4] THE EXPORT. --shell carries FE_PROTOCOL_ID, and its FIRST line exits 1 unless
      FE_RUNNER_HONOURS_START_FROM=1: an old job.run never declares it, so it stops
      before defining anything (CYC-2b). The summary prints the id.
  [5] THE RECORD. job_fe.run writes the id into protocol_id.gs in exactly one
      place, at setup completion, and before setup.done, so a cycle task released
      by setup.done always finds it.

Standalone, no pytest: python3 tests/test_protocol_id.py
"""
import os, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO = os.path.join(ROOT, "utils", "fe_protocol.py")
JOB = os.path.join(ROOT, "job_fe.run")

WANT_ID = "fe-1.5-f78ecb6a"
failures = []


def check(name, ok, detail=""):
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:300]))
    if not ok:
        failures.append(name)


def id_of(src, seed="0"):
    """PROTOCOL_ID of a fe_protocol.py source text, from a fresh interpreter."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "fe_protocol.py"), "w") as f:
            f.write(src)
        r = subprocess.run([sys.executable, "-c",
                            "import fe_protocol as P; print(P.PROTOCOL_ID)"],
                           cwd=d, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=seed,
                                    PYTHONDONTWRITEBYTECODE="1"))
        return r.stdout.strip() if r.returncode == 0 else "ERROR " + r.stderr[-200:]


SRC = open(PROTO).read()

print("\n[1] the id of this table")
got = id_of(SRC)
check("PROTOCOL_ID is %s" % WANT_ID, got == WANT_ID, got)
check("  of the form fe-1.5-<8 hex>", re.match(r"^fe-1\.5-[0-9a-f]{8}$", got or "") is not None,
      got)

print("\n[2] a function of the table alone")
ids = {id_of(SRC, seed) for seed in ("0", "1", "4242", "random")}
check("the same id in fresh interpreters under four hash seeds", ids == {WANT_ID}, ids)

print("\n[3] what it covers, and what it leaves out")
# (what, old text, new text, must the id change?)
EDITS = [
    ("a ramp hold 1 ps longer (RAMP_HOLD_PS 350 -> 351)",
     "RAMP_HOLD_PS    = 350.0", "RAMP_HOLD_PS    = 351.0", True),
    ("one stage boundary moved (B ends at 0.167)",
     '("B", 0.120, 0.166, 2898.0),\n    ("C", 0.166,',
     '("B", 0.120, 0.167, 2898.0),\n    ("C", 0.167,', True),
    ("boundrev2 no longer branching from holdfwd0",
     'start_from="holdfwd0" if i == len(BOUND) - 1 else None', "start_from=None", True),
    ("the bound holds back on the unbind block (the v1.4 defect)",
     'False, False, "bound", lam=(l0, l0))', 'False, False, "unbind", lam=(l0, l0))', True),
    ("npt_c 1 ns shorter (NPT_PS 20000 -> 19000)",
     "NPT_PS          = 20000.0", "NPT_PS          = 19000.0", True),
    ("the row layout: W_remove fields renamed",
     '"W_remove%s" % l["stage"]', '"W_rem%s" % l["stage"]', True),
    ("the probe count (N_PROBES 5 -> 4)",
     "N_PROBES        = 5", "N_PROBES        = 4", False),
    ("the probe length (NPT_INIT_PS 20000 -> 10000)",
     "NPT_INIT_PS     = 20000.0", "NPT_INIT_PS     = 10000.0", False),
    ("the trajectory interval of the legs (XOUT_PS 400 -> 500)",
     "XOUT_PS         = 400.0", "XOUT_PS         = 500.0", False),
]
for what, old, new, changes in EDITS:
    if old not in SRC:
        check("%s: %s" % ("changes the id" if changes else "leaves the id", what), False,
              "edit target not found in fe_protocol.py")
        continue
    alt = id_of(SRC.replace(old, new, 1))
    ok = alt.startswith("fe-1.5-") and ((alt != WANT_ID) == changes)
    check("%s: %s" % ("changes the id" if changes else "leaves the id", what), ok, alt)

print("\n[4] the export and the summary")
r = subprocess.run([sys.executable, PROTO, "--shell"], capture_output=True, text=True)
lines = r.stdout.splitlines() if r.returncode == 0 else []
check("--shell runs", r.returncode == 0, r.stderr[-200:])
check("  its first line is the runner check",
      bool(lines) and lines[0].startswith('[[ "${FE_RUNNER_HONOURS_START_FROM:-0}" == 1 ]] ||'),
      lines[:1])
check("  and it exports FE_PROTOCOL_ID=\"%s\"" % WANT_ID,
      ('FE_PROTOCOL_ID="%s"' % WANT_ID) in lines, [l for l in lines if "PROTOCOL" in l])
exp = r.stdout
r_old = subprocess.run(["bash", "-c", 'eval "$1"; echo "DEFINED ${#FE_LEGS[@]} $FE_PROTOCOL_ID"',
                        "_", exp], capture_output=True, text=True)
check("an eval without the declaration (an old job.run) exits 1, defining nothing",
      r_old.returncode == 1 and "DEFINED" not in r_old.stdout and "FATAL" in r_old.stderr,
      (r_old.returncode, r_old.stdout[-80:], r_old.stderr[-120:]))
r_new = subprocess.run(["bash", "-c", 'FE_RUNNER_HONOURS_START_FROM=1; eval "$1"; '
                        'echo "DEFINED ${#FE_LEGS[@]} $FE_PROTOCOL_ID"', "_", exp],
                       capture_output=True, text=True)
check("  with it, the 42 legs and the id are defined",
      r_new.returncode == 0 and ("DEFINED 42 %s" % WANT_ID) in r_new.stdout,
      (r_new.returncode, r_new.stdout[-80:], r_new.stderr[-120:]))
r = subprocess.run([sys.executable, PROTO], capture_output=True, text=True)
check("the summary prints the id", ("protocol id: %s" % WANT_ID) in r.stdout, r.stdout[:200])

print("\n[5] job_fe.run records it once, at setup completion, before setup.done")
job = open(JOB).read().splitlines()
writes = [i for i, l in enumerate(job, 1) if re.search(r">\s*protocol_id\.gs", l)]
touch = [i for i, l in enumerate(job, 1) if re.match(r"^\s*touch setup\.done\s*$", l)]
check("protocol_id.gs is written in exactly one place", len(writes) == 1, writes)
check("  as the current id", len(writes) == 1
      and 'echo "$FE_PROTOCOL_ID" > protocol_id.gs' in job[writes[0] - 1],
      [job[i - 1] for i in writes])
check("  on the line before the only touch of setup.done",
      len(writes) == 1 and len(touch) == 1 and touch[0] == writes[0] + 1, (writes, touch))

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all protocol-id checks passed")
