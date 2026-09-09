#!/usr/bin/env python3
"""Every force field constrains ALL bonds, at 4 fs with heavy hydrogens, and every
tree agrees with itself and with the others about it.

Until v1.2 the all-atom trees constrained h-bonds only, so that the constraint
chains stayed short enough for GPU LINCS and the update ran on the GPU (1.29x on
2KTF). On the 46-structure HADDOCK FE pass of 2026-09-09 that setting blew up at
0.5 to 14 crashes per us depending on the structure, scaling with protein size and
not with water, while GROMOS 54A8 on all-bonds had none in 25 us. At 4 fs the
heavy-atom stretches (~20 fs period) are integrated at five steps per period, and
the Feenstra/GROMACS recipe for 4-5 fs is all-bonds plus heavy hydrogens. v1.3 puts
every tree on it and pays the 1.29x.

A tree that drifts back by inheritance -- the FE mdps are generated from the base
mdps -- would blow up silently again, so this pins the setting in every sampling
mdp, and pins the integrator settings that justify it alongside.

Standalone, no pytest: python3 tests/test_constraints.py
"""
import glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = os.path.join(ROOT, "settings")
WANT = {"constraints": "all-bonds", "dt": "0.004", "mass-repartition-factor": "3",
        "lincs_order": "6", "lincs_iter": "1"}
fails = 0

def check(name, ok, detail=""):
    global fails
    print("  %-70s %s" % (name, "pass" if ok else "FAIL  %s" % detail))
    if not ok: fails += 1

def mdp(path):
    d = {}
    for ln in open(path):
        ln = ln.split(";")[0].strip()
        if "=" in ln:
            k, v = ln.split("=", 1)
            d[k.strip().replace("-", "_")] = v.strip()
    return d

trees = sorted(d for d in glob.glob(os.path.join(SET, "*")) if os.path.isdir(d))
check("there are settings trees", len(trees) >= 5, str(len(trees)))
for t in trees:
    name = os.path.basename(t)
    print("\n[%s]" % name)
    sampling = []
    for f in sorted(glob.glob(os.path.join(t, "*.mdp"))):
        d = mdp(f)
        if d.get("integrator") == "md":
            sampling.append((os.path.basename(f), d))
    check("has sampling mdps", len(sampling) >= 50, str(len(sampling)))
    for key, want in WANT.items():
        k = key.replace("-", "_")
        wrong = [n for n, d in sampling if d.get(k) != want]
        check("every sampling mdp has %s = %s" % (key, want), not wrong,
              "%d wrong, e.g. %s" % (len(wrong), wrong[:3]))
    mins = [os.path.basename(f) for f in glob.glob(os.path.join(t, "*.mdp"))
            if mdp(f).get("integrator") in ("steep", "cg")]
    check("minimisations do not constrain h-bonds only",
          not any(mdp(os.path.join(t, m)).get("constraints") == "h-bonds" for m in mins))

print("\n[cross-tree]")
src = open(os.path.join(ROOT, "README.md")).read()
check("README's force-field table says all-bonds for every row",
      "| h-bonds |" not in src and src.count("| all-bonds |") >= 4, str(src.count("| all-bonds |")))
gen = open(os.path.join(ROOT, "utils", "make_fe_mdps.py")).read()
check("the generator does not set constraints itself (it inherits from the base mdp)",
      re.search(r"constraints\s*=\s*h-bonds", gen) is None)

print()
sys.exit(1 if fails else 0)
