#!/usr/bin/env python3
"""A force field must be run with the electrostatics it was parametrised with, and
every mdp that SAMPLES must agree with the rest of its tree about that.

GROMOS 54A8 is a reaction-field force field. Run under PME it scored 2KTF at
dG_bind = -64.2 +- 3.1 kJ/mol against -26.1 published for the same force field
(Perthold & Oostenbrink, JCTC 2017, 13, 5697) and -25.1/-28 from experiment.
AMBER19SB/OPC3 gave -39.9 on the same structure under the identical protocol, so
the 38 kJ/mol was not the method and not the sampling: all five GROMOS runs
returned BAR with better overlap than the AMBER arm (min 24-36% against 8-25%) and
rebound to 2.5-2.9 A.

settings/gromos54a8 is therefore kept AS IT WAS, PME and all, because test34-38
were run with it and a settings tree is the record of how a run was done.
settings/gromos54a8_rf is the reaction-field variant, and the paper's numbers
verbatim: "a cutoff sphere of 1.4 nm ... a reaction-field contribution with a
relative dielectric permittivity of 61 beyond the cutoff sphere".

WHY THE MINIMISATIONS ARE EXEMPT, and it is not to make this pass. Every tree ships
its energy minimisations on reaction field at 1.4 nm regardless of what its dynamics
use, gromos54a8 included. That is defensible for the reason the rule exists at all:
the rule is that the ensemble the free energy is computed from must come from ONE
Hamiltonian, and a minimiser draws no ensemble. It relaxes clashes so that solvate,
genion and the NVT ladder have something sane to start from, and the ladder then
equilibrates under the production Hamiltonian. So the exemption is by INTEGRATOR,
not by filename: anything that samples is checked, and a filename list would go
stale the moment a minimisation is renamed.

Standalone, no pytest: python3 tests/test_electrostatics.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = os.path.join(ROOT, "settings")
# What each tree must run. gromos54a8 is the as-shipped PME tree that test34-38
# used; gromos54a8_rf is the variant. AMBER and CHARMM are lattice-sum force fields
# and must stay on PME.
EXPECT = {
    "gromos54a8":     ("PME", {}, 1.4),
    "gromos54a8_rf":  ("Reaction-Field", {"epsilon_rf": 61.0, "epsilon_r": 1.0}, 1.4),
    "amber19sb_opc":  ("PME", {}, 1.0),
    "amber19sb_opc3": ("PME", {}, 1.0),
    "charmm36":       ("PME", {}, 1.2),
}
MINIMISERS = {"steep", "cg", "l-bfgs", "steepest"}
failures = []


def check(name, ok, detail=""):
    print("  %-68s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def settings(path):
    """mdp options, keys normalised the way GROMACS normalises them: it treats
    '-' and '_' as the same character, so epsilon-rf and epsilon_rf are one key.
    Reading them as two is how I once concluded epsilon_rf was never set."""
    out = {}
    for line in open(path):
        line = line.split(";")[0]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().lower().replace("-", "_")] = v.strip()
    return out


def samples(s):
    """Does this mdp draw an ensemble, or only relax a geometry?"""
    return s.get("integrator", "md").strip().lower() not in MINIMISERS


for ff in sorted(EXPECT):
    d = os.path.join(SET, ff)
    if not os.path.isdir(d):
        check("[%s] tree exists" % ff, False, d)
        continue
    want_ct, want_eps, want_rc = EXPECT[ff]
    all_mdps = sorted(f for f in os.listdir(d) if f.endswith(".mdp"))
    opts = {f: settings(os.path.join(d, f)) for f in all_mdps}
    mdps = [f for f in all_mdps if samples(opts[f])]
    print("\n[%s] %d mdps, %d sample, %d minimise"
          % (ff, len(all_mdps), len(mdps), len(all_mdps) - len(mdps)))
    check("it has minimisations, so the exemption is doing something",
          len(mdps) < len(all_mdps), str(len(all_mdps)))

    seen_ct, seen_rc, offenders, eps_bad, cut_bad = set(), set(), [], [], []
    for fn in mdps:
        s = opts[fn]
        ct = s.get("coulombtype", "")
        seen_ct.add(ct.lower())
        if ct.lower() != want_ct.lower():
            offenders.append("%s=%s" % (fn, ct))
        for k, v in want_eps.items():
            try:
                if abs(float(s.get(k, "nan")) - v) > 1e-9:
                    eps_bad.append("%s: %s=%s" % (fn, k, s.get(k)))
            except ValueError:
                eps_bad.append("%s: %s=%s" % (fn, k, s.get(k)))
        rs = [s.get(k) for k in ("rcoulomb", "rvdw", "rlist") if k in s]
        for r in rs:
            try:
                if abs(float(r) - want_rc) > 1e-9:
                    cut_bad.append("%s: %s" % (fn, r))
            except ValueError:
                cut_bad.append("%s: %s" % (fn, r))
        if rs:
            seen_rc.add(tuple(rs))

    check("every sampling mdp uses %s" % want_ct, not offenders,
          ", ".join(offenders[:5]) + (" ..." if len(offenders) > 5 else ""))
    check("and the sampling files agree with each other (%d coulombtype%s)"
          % (len(seen_ct), "" if len(seen_ct) == 1 else "s"),
          len(seen_ct) == 1, str(sorted(seen_ct)))
    if want_eps:
        check("epsilon_rf = %g and epsilon_r = %g everywhere"
              % (want_eps["epsilon_rf"], want_eps["epsilon_r"]),
              not eps_bad, ", ".join(eps_bad[:4]))
    check("the cutoff sphere is %.1f nm throughout" % want_rc, not cut_bad,
          ", ".join(cut_bad[:4]))
    check("and rcoulomb, rvdw and rlist agree with each other",
          len(seen_rc) <= 1, str(sorted(seen_rc)[:3]))

print("\n[variant] gromos54a8_rf differs from gromos54a8 in the electrostatics ONLY")
a, b = os.path.join(SET, "gromos54a8"), os.path.join(SET, "gromos54a8_rf")
if os.path.isdir(a) and os.path.isdir(b):
    fa = sorted(f for f in os.listdir(a) if f.endswith(".mdp"))
    fb = sorted(f for f in os.listdir(b) if f.endswith(".mdp"))
    check("the two trees hold the same file names", fa == fb,
          str(set(fa) ^ set(fb)))
    ELE = {"coulombtype", "epsilon_rf", "epsilon_r"}
    diffs = set()
    for fn in fa:
        if fn not in fb:
            continue
        sa, sb = settings(os.path.join(a, fn)), settings(os.path.join(b, fn))
        for k in set(sa) | set(sb):
            if sa.get(k) != sb.get(k):
                diffs.add(k)
    check("and every differing option is an electrostatics one (%s)"
          % ", ".join(sorted(diffs)), diffs <= ELE, str(sorted(diffs - ELE)))
    check("the timestep, thermostat, barostat and constraints are untouched",
          not (diffs & {"dt", "tcoupl", "pcoupl", "ref_t", "tau_t", "tau_p",
                        "constraints", "mass_repartition_factor", "nsteps"}),
          str(sorted(diffs)))

print("\n[registration] a new tree is useless if nothing can select it")
for f, pat in (("groscore_fe.py", r'choices=\[([^\]]*)\]'),
               ("groscore.py", r'choices=\[([^\]]*)\]')):
    src = open(os.path.join(ROOT, f)).read()
    m = re.search(pat, src)
    check("%s offers gromos54a8_rf" % f,
          m is not None and "gromos54a8_rf" in m.group(1),
          m.group(1) if m else "no choices=")
mk = open(os.path.join(ROOT, "utils", "make_fe_mdps.py")).read()
check("make_fe_mdps generates for it", "gromos54a8_rf" in mk)

print("\n[base name] a variant suffix must not silently change WHICH force field")
# ten of these in each job file: pdb2gmx's -ff, GMXLIB, the NCAA rename table, the
# ion protonation map, residuetypes.dat. All of them are == "gromos54a8" tests that
# a tree called gromos54a8_rf would fall straight through, into the AMBER branch.
for f in ("job_fe.run", "job.run"):
    src = open(os.path.join(ROOT, f)).read()
    lines = src.splitlines()
    check("%s defines FF_BASE" % f, any(l.startswith("FF_BASE=") for l in lines))
    check("  by stripping the variant suffix",
          'FF_BASE="${FORCEFIELD%_rf}"' in src)
    check("  no equality test on $FORCEFIELD survives",
          '"$FORCEFIELD" ==' not in src,
          [l for l in lines if '"$FORCEFIELD" ==' in l][:2])
    check("  utils are given the base name, not the tree name",
          "--ff $FORCEFIELD" not in src)
    check("  but MDP_DIR still uses the tree name",
          'settings/$FORCEFIELD' in src)
    d = next(i for i, l in enumerate(lines) if l.startswith("FF_BASE="))
    u = [i for i, l in enumerate(lines)
         if "FF_BASE" in l and not l.lstrip().startswith("#") and i != d]
    check("  and it is defined before its first use (line %d, first use %d)"
          % (d + 1, (min(u) + 1) if u else -1), bool(u) and min(u) > d, str(u[:3]))

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all electrostatics checks passed")
