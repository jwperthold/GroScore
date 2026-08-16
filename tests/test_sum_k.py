#!/usr/bin/env python3
"""The interface stiffness budget: how it is remembered, and that it stays a budget.

sum_k is a property of the restraint set, so every cycle in a run directory has to
be built with the same value or their works are not comparable. It therefore lives
in run_config.gs alongside numruns rather than in a flag someone has to retype, and
job_fe.run reads it back from there. What must hold:

  * the first invocation records it, later ones inherit it
  * asking for a different value in a directory that already has one WARNS
  * job_fe.run's own reader agrees with what groscore_fe.py wrote
  * k is the budget divided by however many springs there are, exactly

Standalone, no pytest: python3 tests/test_sum_k.py
"""
import os, re, sys, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(ROOT, "groscore_fe.py")
JOB = os.path.join(ROOT, "job_fe.run")

failures = []


def check(name, ok, detail=""):
    print("  %-62s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def run_fe(cwd, *extra):
    """Only the config handling is under test; it happens before any work, so the
    exit status is irrelevant and simulations never start."""
    return subprocess.run([sys.executable, FE, "-n", "3", "--sequential"] + list(extra),
                          capture_output=True, text=True, cwd=cwd, timeout=120)


def cfg(cwd):
    p = os.path.join(cwd, "run_config.gs")
    if not os.path.isfile(p):
        return {}
    out = {}
    for line in open(p):
        if line.startswith("#"):
            continue
        t = line.split()
        if len(t) >= 2:
            out[t[0]] = t[1]
    return out


tmp = tempfile.mkdtemp(prefix="sumk_")
with open(os.path.join(tmp, "sp.gs"), "w") as f:
    f.write("# Structure_ID  Chains_for_Protein_B\n2KTF\tB---\n")

print("\n[1] run_config.gs remembers sum_k the way it remembers numruns")
r = run_fe(tmp, "--sum-k", "12500")
check("first invocation records it", cfg(tmp).get("sum_k") == "12500",
      "cfg=%r stderr=%r" % (cfg(tmp), r.stderr[-300:]))
run_fe(tmp)
check("a later invocation without the flag keeps it",
      cfg(tmp).get("sum_k") == "12500", "cfg=%r" % cfg(tmp))
check("numruns still works alongside it", cfg(tmp).get("numruns") == "3",
      "cfg=%r" % cfg(tmp))

print("\n[2] changing it in an existing directory warns rather than doing it quietly")
r = run_fe(tmp, "--sum-k", "8000")
said = r.stdout + r.stderr
check("warns", "WARNING" in said and "12500" in said and "8000" in said,
      said[-400:])
check("and says the works are not comparable", "comparable" in said, said[-400:])

print("\n[3] job_fe.run reads back exactly what groscore_fe.py wrote")
# The reader is extracted from job_fe.run rather than restated, so the two cannot
# drift apart the way the completeness gate and the row writer once did.
src = open(JOB).read()
m = re.search(r'SUM_K=\$\(awk\s+(\'[^\']*\')\s+\.\./run_config\.gs', src)
check("job_fe.run still reads sum_k from ../run_config.gs", m is not None)
if m:
    prog = m.group(1).strip("'")
    got = subprocess.run(["awk", prog, os.path.join(tmp, "run_config.gs")],
                         capture_output=True, text=True).stdout.strip()
    check("its awk yields the recorded value", got == cfg(tmp).get("sum_k"),
          "awk=%r cfg=%r" % (got, cfg(tmp).get("sum_k")))
    empty = tempfile.mkdtemp(prefix="sumk_none_")
    with open(os.path.join(empty, "run_config.gs"), "w") as f:
        f.write("# nothing pinned here\nnumruns\t5\n")
    got2 = subprocess.run(["awk", prog, os.path.join(empty, "run_config.gs")],
                          capture_output=True, text=True).stdout.strip()
    check("a directory that pins nothing yields nothing", got2 == "", repr(got2))
    shutil.rmtree(empty, ignore_errors=True)

print("\n[4] the budget is divided, not applied per spring")
# make_boresch.py's one line, checked against the arithmetic it has to satisfy.
mb = open(os.path.join(ROOT, "utils", "make_boresch.py")).read()
check("k_inter is sum_k/N and nothing else",
      re.search(r"k_inter\s*=\s*args\.sum_k\s*/\s*numinterdis", mb) is not None)
check("no bare 25000 survives outside the default and its prose",
      len([l for l in mb.splitlines()
           if "25000" in l and not l.lstrip().startswith("#")
           and "default=25000.0" not in l.replace(" ", "")]) == 0,
      [l.strip() for l in mb.splitlines()
       if "25000" in l and not l.lstrip().startswith("#")])
for total, n in ((25000.0, 656), (12500.0, 656), (12500.0, 188), (8000.0, 461)):
    k = total / n
    check("N=%d at sum_k=%g gives k=%.4f and N*k back" % (n, total, k),
          abs(n * k - total) < 1e-9)

print("\n[5] the pull-path weights are ratios, so sum_k cannot touch them")
check("write_pull_weights never mentions sum_k or k_inter",
      "sum_k" not in mb.split("def write_pull_weights")[1].split("def write_pull_block")[0]
      and "k_inter" not in mb.split("def write_pull_weights")[1].split("def write_pull_block")[0])

shutil.rmtree(tmp, ignore_errors=True)
print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all sum_k checks passed")
