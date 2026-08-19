#!/usr/bin/env python3
"""Every setting that changes what is SIMULATED must come from the directory.

A run directory is filled in over days by repeated invocations -- topping up
cycles, restarting failures, re-scoring -- and each is a fresh command line.
Anything that alters what is simulated therefore has to come from the DIRECTORY
rather than from whichever command line ran last, or the cycles in one directory
stop being samples of one thing.

What must hold for EACH remembered setting, not just for sum_k:

  * a fresh directory records it, default included, so no invocation can reach a
    default that has since moved
  * a later invocation with no flag inherits it, and does not warn
  * an explicit DIFFERENT value warns and says the works are not comparable
  * an explicit SAME value does not warn
  * booleans can express "unset", or a remembered false could never be honoured

Standalone, no pytest: python3 tests/test_remembered.py
"""
import os, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(ROOT, "groscore_fe.py")
JOB = os.path.join(ROOT, "job_fe.run")
failures = []


def check(name, ok, detail=""):
    print("  %-66s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def fresh():
    d = tempfile.mkdtemp(prefix="rem_")
    with open(os.path.join(d, "sp.gs"), "w") as f:
        f.write("# Structure_ID  Chains_for_Protein_B\n2KTF\tB---\n")
    return d


def run(cwd, *extra):
    return subprocess.run([sys.executable, FE, "--sequential"] + list(extra),
                          capture_output=True, text=True, cwd=cwd, timeout=180)


def cfg(cwd):
    p = os.path.join(cwd, "run_config.gs")
    out = {}
    if os.path.isfile(p):
        for line in open(p):
            if line.startswith("#"):
                continue
            t = line.split()
            if len(t) >= 2:
                out[t[0]] = t[1]
    return out


src = open(FE).read()

print("\n[1] the set is declared in one table, not scattered")
m = re.search(r"^REMEMBERED = \[(.*?)^\]", src, re.M | re.S)
check("groscore_fe.py declares a REMEMBERED table", m is not None)
keys = re.findall(r'^\s*\("(\w+)",\s*"(\w+)"', m.group(1), re.M) if m else []
have = {k for _, k in keys}
print("     table holds: %s" % ", ".join(sorted(have)))
for k in ("numruns", "sum_k", "sd_max", "forcefield", "temp", "cutout", "ligand_param"):
    check("%s is remembered" % k, k in have)
for k in ("slurm", "ngpus", "array_throttle", "n_boot_bar", "rmsd_warn",
          "structparams", "restart", "sequential", "run_local"):
    check("%s is NOT remembered (scheduling or reporting only)" % k, k not in have)

print("\n[2] a fresh directory records every one of them, defaults included")
d = fresh()
run(d)
c = cfg(d)
for k in sorted(have):
    check("fresh directory states %s (= %s)" % (k, c.get(k)), k in c, str(c))

print("\n[3] the defaults are the ones the code names, and agree with make_boresch")
mb = open(os.path.join(ROOT, "utils", "make_boresch.py")).read()
for name, key in (("DEFAULT_SUM_K", "sum_k"), ("DEFAULT_SD_MAX", "sd_max")):
    mm = re.search(r"^%s\s*=\s*([0-9.]+)" % name, src, re.M)
    check("%s is named at module level" % name, mm is not None)
    if mm:
        check("and run_config.gs got it (%s)" % mm.group(1),
              abs(float(c.get(key, "nan")) - float(mm.group(1))) < 1e-12,
              "%s vs %s" % (c.get(key), mm.group(1)))
sd_mb = re.search(r"^SD_MAX_DEFAULT\s*=\s*([0-9.]+)", mb, re.M)
sd_fe = re.search(r"^DEFAULT_SD_MAX\s*=\s*([0-9.]+)", src, re.M)
check("groscore_fe and make_boresch agree on the sd ceiling",
      sd_mb and sd_fe and abs(float(sd_mb.group(1)) - float(sd_fe.group(1))) < 1e-12,
      "%s vs %s" % (sd_mb and sd_mb.group(1), sd_fe and sd_fe.group(1)))

print("\n[4] inherit silently, warn on a real change, stay quiet on the same value")
for flag, first, other in (("--sum-k", "9000", "8000"),
                           ("--sd-max", "0.20", "0.30"),
                           ("--temp", "300", "298"),
                           ("-ff", "charmm36", "gromos54a8")):
    d = fresh()
    run(d, flag, first)
    key = {"--sum-k": "sum_k", "--sd-max": "sd_max", "--temp": "temp",
           "-ff": "forcefield"}[flag]
    got = cfg(d).get(key)
    check("%s %s is recorded" % (flag, first), got is not None, str(cfg(d)))
    # Look for THIS warning specifically. Any "WARNING" is too broad: --temp also
    # trips check_ref_t, which fires because 300 K disagrees with the mdps' ref_t
    # and is a different and entirely correct complaint.
    MARK = "this directory was set up with"
    r = run(d)
    check("%s survives an invocation with no flag" % flag,
          cfg(d).get(key) == got, "%s -> %s" % (got, cfg(d).get(key)))
    check("and that invocation does not warn about a change",
          MARK not in r.stdout + r.stderr)
    r = run(d, flag, first)
    check("re-stating the same value does not warn about a change",
          MARK not in r.stdout + r.stderr, (r.stdout + r.stderr)[-200:])
    r = run(d, flag, other)
    said = r.stdout + r.stderr
    check("changing it warns and says works are not comparable",
          MARK in said and "comparable" in said, said[-300:])
    # Numbers are written back with %g, so 0.20 is reported as 0.2. Compare the
    # formatted forms rather than the strings that were typed.
    def _fmt(x):
        try:
            return "%g" % float(x)
        except ValueError:
            return x
    check("and the warning names the setting and both values",
          key in said and _fmt(first) in said and _fmt(other) in said, said[-300:])
    shutil.rmtree(d, ignore_errors=True)

print("\n[5] booleans can express 'unset', so a remembered false is honoured")
check("--no-cutout uses store_const with default None, not store_false",
      re.search(r"'--no-cutout'[^)]*store_const[^)]*default=None", src, re.S) is not None)
check("a positive --cutout exists so it can be changed back",
      re.search(r"'--cutout'[^)]*store_const[^)]*const=True", src, re.S) is not None)
check("set_defaults no longer overrides them",
      "set_defaults(cutout=True" not in src)
d = fresh()
run(d, "--no-cutout")
check("--no-cutout records cutout=0", cfg(d).get("cutout") == "0", str(cfg(d)))
run(d)
check("and a later bare invocation keeps it off", cfg(d).get("cutout") == "0",
      str(cfg(d)))
r = run(d, "--cutout")
check("--cutout turns it back on, with a warning",
      cfg(d).get("cutout") == "1" and "WARNING" in r.stdout + r.stderr, str(cfg(d)))
shutil.rmtree(d, ignore_errors=True)

print("\n[6] job_fe.run reads back what was written, for both restraint settings")
job = open(JOB).read()
for key, flag in (("sum_k", "--sum-k"), ("sd_max", "--sd-max")):
    m2 = re.search(r"%s=\$\(awk\s+('[^']*')\s+\.\./run_config\.gs" % key.upper(), job)
    check("job_fe.run reads %s from ../run_config.gs" % key, m2 is not None)
    if m2:
        d2 = fresh()
        run(d2)
        got = subprocess.run(["awk", m2.group(1).strip("'"),
                              os.path.join(d2, "run_config.gs")],
                             capture_output=True, text=True).stdout.strip()
        check("its awk yields the recorded %s (%s)" % (key, got),
              got == cfg(d2).get(key), "%s vs %s" % (got, cfg(d2).get(key)))
        shutil.rmtree(d2, ignore_errors=True)
    check("and passes it on as %s" % flag, "%s $%s" % (flag, key.upper()) in job)
check("both options reach the make_boresch call line",
      "$SUM_K_OPT $SD_MAX_OPT" in job)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all remembered-setting checks passed")
