#!/usr/bin/env python3
"""The probe replicas get the same retry as the cycle legs, and can.

20 of the 22 PROBE_FAILs on the first HADDOCK FE pass (2026-09-09) were one probe
replica of five dying mid-run to the same blow-up that the cycle legs survive
through run_leg_mdrun (202 retries, 166 rescued). The probe called mdrun directly.
run_leg_mdrun was also defined INSIDE the cycle section, so it could not simply be
called from Stage 0; it now lives at top level, before either. Every probe stage,
the NVT ladder included, now goes through it.

Every attempt starts from step 0 of the leg's tpr: an unfinished run is never
continued from its checkpoint (bm_fe_amber_3a, where a full disk dropped xvg blocks
from runs that went on), and a leg without a tpr is not started at all.

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
u = line_of(r"^\s+run_leg_mdrun \$\{pout\} ")
check("every probe stage is run through run_leg_mdrun", u is not None
      and "for pstage in nvt_1 nvt_2 nvt_3 nvt_4 nvt_5 npt; do" in src)
check("npt_probe with the same 600 min ceiling as npt_c", u is not None and "600m" in lines[u - 1]
      and "eq_stage npt_c${cycle} ./npt_c_fe.mdp nvt_5_c${cycle} 600m" in src)
check("no bare 'gmx mdrun' of a probe or cycle NVT/NPT stage remains",
      re.search(r"gmx mdrun[^\n]*-deffnm (npt_probe|nvt_\S*probe|nvt_\S*_c)", src) is None)
check("the PROBE_FAIL path still exists for a replica that fails twice",
      'echo $STRUCT_ID "PROBE_FAIL" >> ../results_0.gs' in src)
check("the cycle legs still use it too", src.count("run_leg_mdrun ${out} 180m") == 1
      and src.count("run_leg_mdrun $out $tmo") == 1)

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
        # a gmx that logs its arguments and fails N times (N in ./fails), leaving a
        # checkpoint as a dead mdrun does, before writing the .gro
        open(fake, "w").write("#!/bin/bash\n"
            'echo "$*" >> calls.log\n'
            "out=''; while [[ $# -gt 0 ]]; do [[ $1 == -deffnm ]] && out=$2; shift; done\n"
            "n=$(cat fails 2>/dev/null || echo 0)\n"
            "if [[ $n -gt 0 ]]; then echo $((n-1)) > fails; touch ${out}.cpt; exit 1; fi\n"
            "touch ${out}.gro ${out}.cpt\n")
        os.chmod(fake, 0o755)
        script = ("set -u\nnt=1; GPUOPT=''\n" + helpers + m.group(0) +
                  "cd %s\ntouch a.tpr b.tpr c.tpr\n"
                  "echo 0 > fails; run_leg_mdrun a 1m; echo rc_a=$? \n"
                  "rm -f a.gro; echo 1 > fails; run_leg_mdrun b 1m; echo rc_b=$?\n"
                  "rm -f b.gro; echo 5 > fails; run_leg_mdrun c 1m; echo rc_c=$?\n"
                  "echo 0 > fails; run_leg_mdrun d 1m; echo rc_d=$?; if [[ -f d.gro ]]; then echo d_ran; fi\n"
                  % d)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                           env=dict(os.environ, PATH=d + ":" + os.environ["PATH"]))
        out = r.stdout
        calls = open(os.path.join(d, "calls.log")).read().splitlines() \
            if os.path.exists(os.path.join(d, "calls.log")) else []
        check("clean run returns 0", "rc_a=0" in out, out[-300:])
        check("one failure is retried and returns 0", "rc_b=0" in out and "attempt 1 of 2" in out, out[-300:])
        check("  the retry starts from step 0, not from the checkpoint the dead run left",
              "retrying from step 0" in out and calls and not [c for c in calls if "-cpi" in c], calls)
        check("two failures give up with 1", "rc_c=1" in out, out[-300:])
        check("no tpr (a failed grompp): not started, returns 1",
              "rc_d=1" in out and "d_ran" not in out and "not started" in out
              and not [c for c in calls if "-deffnm d" in c], out[-300:])
        check("no error under set -u with no extra args", r.returncode == 0, r.stderr[-300:])

print()
sys.exit(1 if fails else 0)
