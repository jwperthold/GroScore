#!/usr/bin/env python3
"""The probe replicas get the same retry as the cycle legs, and can.

20 of the 22 PROBE_FAILs on the first HADDOCK FE pass (2026-09-09) were one probe
replica of five dying mid-run to the same blow-up that the cycle legs survive
through run_leg_mdrun (202 retries, 166 rescued). The probe called mdrun directly.
run_leg_mdrun was also defined INSIDE the cycle section, so it could not simply be
called from Stage 0; it now lives at top level, before either.

Standalone, no pytest: python3 tests/test_probe_retry.py
"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = os.path.join(ROOT, "job_fe.run")
src = open(JOB).read()
lines = src.splitlines()
fails = 0

def check(name, ok, detail=""):
    global fails
    print("  %-70s %s" % (name, "pass" if ok else "FAIL  %s" % detail))
    if not ok: fails += 1

def line_of(pat):
    for i, ln in enumerate(lines, 1):
        if re.search(pat, ln): return i
    return None

print("\n[1] the function is where both callers can see it")
d = line_of(r"^run_leg_mdrun\(\) \{$")
check("run_leg_mdrun is defined at column 0 (top level)", d is not None)
check("and only once", src.count("run_leg_mdrun() {") == 1, str(src.count("run_leg_mdrun() {")))
stage0 = line_of(r"# Stage 0: Prepare system")
cyc = line_of(r'^if \[\[ "\$MODE" == "cycle" \]\]; then$')
check("defined before Stage 0 begins", d is not None and stage0 is not None and d < stage0, "%s vs %s" % (d, stage0))
check("defined before the cycle section begins", d is not None and cyc is not None and d < cyc, "%s vs %s" % (d, cyc))

print("\n[2] the probe replicas use it")
u = line_of(r"^\s+run_leg_mdrun npt_probe\$\{rep\} 600m")
check("the probe NPT is run through run_leg_mdrun", u is not None)
check("with the same 600 min ceiling as npt_c", u is not None and "600m" in lines[u - 1])
check("no bare 'gmx mdrun ... npt_probe' call remains",
      re.search(r"gmx mdrun[^\n]*-deffnm npt_probe", src) is None)
check("the PROBE_FAIL path still exists for a replica that fails twice",
      'echo $STRUCT_ID "PROBE_FAIL" >> ../results_0.gs' in src)
check("the cycle legs still use it too", src.count("run_leg_mdrun npt_c${cycle} 600m") == 1)

print("\n[3] the file still parses")
r = subprocess.run(["bash", "-n", JOB], capture_output=True, text=True)
check("bash -n job_fe.run", r.returncode == 0, r.stderr.strip()[:200])

print("\n[4] the function behaves: succeeds first time, retries once, gives up")
# Extract the function verbatim and drive it with a fake gmx on PATH.
m = re.search(r"^run_leg_mdrun\(\) \{\n.*?^\}\n", src, re.S | re.M)
# It starts mdrun through run_bg, which records the running leg in the locks
# (review S3); run_bg and its proc_start come with it.
helpers = "".join(h.group(0) for h in (re.search(r"^%s\(\) \{\n.*?^\}\n" % n, src, re.S | re.M)
                                       for n in ("proc_start", "run_bg")) if h)
check("function body extractable", m is not None and helpers.count("() {") == 2)
if m:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fake = os.path.join(d, "gmx")
        # a gmx that fails N times (N in ./fails) before writing the .gro
        open(fake, "w").write("#!/bin/bash\n"
            "out=''; while [[ $# -gt 0 ]]; do [[ $1 == -deffnm ]] && out=$2; shift; done\n"
            "n=$(cat fails 2>/dev/null || echo 0)\n"
            "if [[ $n -gt 0 ]]; then echo $((n-1)) > fails; exit 1; fi\n"
            "touch ${out}.gro\n")
        os.chmod(fake, 0o755)
        script = ("set -u\nnt=1; GPUOPT=''\n" + helpers + m.group(0) +
                  "cd %s\necho 0 > fails; run_leg_mdrun a 1m; echo rc_a=$? \n"
                  "rm -f a.gro; echo 1 > fails; run_leg_mdrun b 1m; echo rc_b=$?\n"
                  "rm -f b.gro; echo 5 > fails; run_leg_mdrun c 1m; echo rc_c=$?\n" % d)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                           env=dict(os.environ, PATH=d + ":" + os.environ["PATH"]))
        out = r.stdout
        check("clean run returns 0", "rc_a=0" in out, out[-300:])
        check("one failure is retried and returns 0", "rc_b=0" in out and "attempt 1 of 2" in out, out[-300:])
        check("two failures give up with 1", "rc_c=1" in out, out[-300:])
        check("no error under set -u with no extra args", r.returncode == 0, r.stderr[-300:])

print()
sys.exit(1 if fails else 0)
