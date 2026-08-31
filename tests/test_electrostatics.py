#!/usr/bin/env python3
"""A force field must be run with the electrostatics it was parametrised with,
and every mdp in its tree must agree about that.

GROMOS 54A8 is a reaction-field force field. The tree ran it under PME, and on
2KTF that gave dG_bind = -64.2 +- 3.1 kJ/mol against -26.1 published for the same
force field (Perthold & Oostenbrink, JCTC 2017, 13, 5697) and -25.1/-28 from
experiment. AMBER19SB/OPC3 under PME gave -39.9 on the same structure with the same
protocol, so the 38 kJ/mol was not the method and not the sampling: all five GROMOS
runs returned BAR with better overlap than the AMBER arm (min 24-36% against 8-25%)
and rebound to 2.5-2.9 A.

The paper's settings, verbatim: "Calculation of nonbonded electrostatic and
Lennard-Jones interactions was done within a cutoff sphere of 1.4 nm. For the
calculation of electrostatic interactions, a reaction-field contribution with a
relative dielectric permittivity of 61 beyond the cutoff sphere was added."

THE SECOND CHECK IS THE MORE GENERAL ONE. The tree was not merely wrong, it was
INCONSISTENT: the three energy minimisations ran reaction-field at epsilon_rf 61
while the 54 dynamics files ran PME. The probe equilibration measures the interface
references, the anchor rigidity and the box that the production legs then use, so a
tree that disagrees with itself about the Hamiltonian is measuring one system and
simulating another. Nothing anywhere would have reported that.

Standalone, no pytest: python3 tests/test_electrostatics.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = os.path.join(ROOT, "settings")
# What each force field was parametrised with. GROMOS is reaction-field; the
# AMBER and CHARMM trees are lattice-sum force fields and must stay on PME.
EXPECT = {
    "gromos54a8": ("Reaction-Field", {"epsilon_rf": 61.0, "epsilon_r": 1.0}, 1.4),
    "amber19sb_opc": ("PME", {}, 1.0),
    "amber19sb_opc3": ("PME", {}, 1.0),
    "charmm36": ("PME", {}, 1.2),
}

# THE ONE FILE OUTSIDE THE RULE, and the reason it is outside it. emin_vac runs
# before solvation, on conf_vacbox.gro, purely to relax clashes so that solvate and
# genion have something sane to work with. Nothing is sampled from it and no free
# energy is computed from it, so it does not have to share the production
# Hamiltonian -- and in fact every tree ships the same one, reaction-field at 1.4 nm.
# It is named here rather than pattern-matched so that any OTHER file drifting away
# from its tree still fails.
EXEMPT = {"emin_vac.mdp"}
failures = []


def check(name, ok, detail=""):
    print("  %-68s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def settings(path):
    """mdp options, keys normalised the way GROMACS normalises them."""
    out = {}
    for line in open(path):
        line = line.split(";")[0]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().lower().replace("-", "_")] = v.strip()
    return out


for ff in sorted(EXPECT):
    d = os.path.join(SET, ff)
    if not os.path.isdir(d):
        print("\n[%s] tree absent, skipped" % ff)
        continue
    want_ct, want_eps, want_rc = EXPECT[ff]
    mdps = sorted(f for f in os.listdir(d)
                  if f.endswith(".mdp") and f not in EXEMPT)
    print("\n[%s] %d mdps" % (ff, len(mdps)))
    seen_ct, seen_rc, offenders, eps_bad, cut_bad = set(), set(), [], [], []
    for fn in mdps:
        s = settings(os.path.join(d, fn))
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
        # rcoulomb, rvdw and rlist must be the one cutoff sphere the FF assumes
        rs = [s.get(k) for k in ("rcoulomb", "rvdw", "rlist") if k in s]
        for r in rs:
            try:
                if abs(float(r) - want_rc) > 1e-9:
                    cut_bad.append("%s: %s" % (fn, r))
            except ValueError:
                cut_bad.append("%s: %s" % (fn, r))
        if rs:
            seen_rc.add(tuple(rs))

    check("every mdp uses %s" % want_ct, not offenders,
          ", ".join(offenders[:5]) + (" ..." if len(offenders) > 5 else ""))
    # THE INCONSISTENCY CHECK: one tree, one Hamiltonian. This is what was broken.
    check("and the tree agrees with ITSELF (%d distinct coulombtype%s)"
          % (len(seen_ct), "" if len(seen_ct) == 1 else "s"),
          len(seen_ct) == 1, str(sorted(seen_ct)))
    if want_eps:
        check("epsilon_rf = %g and epsilon_r = %g everywhere"
              % (want_eps["epsilon_rf"], want_eps["epsilon_r"]),
              not eps_bad, ", ".join(eps_bad[:4]))
    check("the cutoff sphere is %.1f nm in every file" % want_rc, not cut_bad,
          ", ".join(cut_bad[:4]))
    check("and rcoulomb, rvdw and rlist agree with each other",
          len(seen_rc) <= 1, str(sorted(seen_rc)[:3]))

print("\n[cross-tree] the change did not leak into the lattice-sum force fields")
for ff in sorted(EXPECT):
    d = os.path.join(SET, ff)
    if not os.path.isdir(d):
        continue
    blob = "".join(open(os.path.join(d, f)).read()
                   for f in os.listdir(d)
                   if f.endswith(".mdp") and f not in EXEMPT).lower()
    if EXPECT[ff][0] == "PME":
        check("%s has no reaction-field anywhere" % ff, "reaction-field" not in blob)
    else:
        check("%s has no PME anywhere" % ff,
              not re.search(r"coulombtype\s*=\s*pme", blob))

print("\n[provenance] the reason is recorded where the next person will look")
gro = os.path.join(SET, "gromos54a8", "npt.mdp")
if os.path.isfile(gro):
    head = open(gro).read()
    check("the mdp says why it is reaction-field", "REACTION FIELD" in head)
    check("and cites the paper the number is compared against",
          "5697" in head and "Oostenbrink" in head)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all electrostatics checks passed")
