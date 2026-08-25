#!/usr/bin/env python3
"""ROT_MAX_DEG decides acceptance. It used to only decide what got printed.

The eps ladder relaxes a ceiling on a candidate group's COM RMSF, and the note
above it in make_boresch.py already stated the policy: "a loose ceiling that
passes the rotation check has earned it while one that does not is rejected
regardless of which rung found it". The code took the first rung that returned
ANY triad, printed the rotation next to the limit, and used the set whether or
not it cleared it.

What that cost, over eight runs of 2KTF under one protocol:

    run      eps rung   rotation   rungs left   cycle dissipation
    test15      0.065      8.1 deg          4              61.3
    test25      0.075      8.9 deg          3              75.7
    six others  .045-.065  4.3-6.4          -         56.8-64.9

test25 also carries the worst ramp stage measured anywhere in the set (24.7
kJ/mol in stage E against a median of 14.4). Across the eight, rotation
correlates +0.68 with the cycle's total dissipation and +0.64 with stage E's.

The ladder is monotone in the candidate POOL -- a looser eps admits every group
the tighter one did -- so continuing can only find an equal or better minimum,
apart from the 0.5 nm thinning and TOP_PER_SIDE evicting a group when more
candidates appear. That is why the policy is "remember the best" rather than
"take the last rung tried".

Standalone, no pytest: python3 tests/test_rot_limit.py
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))

# make_boresch parses argv and reads files at import; only the policy is wanted,
# so it is compiled out of the source rather than imported.
src = open(os.path.join(ROOT, "utils", "make_boresch.py")).read()


def const(name):
    """The module cannot be imported (it parses argv and reads files at import),
    so module-level constants are read out of the source."""
    line = [l for l in src.splitlines() if l.startswith(name + " ")][0]
    return eval(line.split("=", 1)[1].split("#")[0].strip())


LADDER = const("EPS_LADDER")
ROT_MAX = const("ROT_MAX_DEG")
i = src.index("def climb_ladder(")
j = src.index("\ndef ", i + 1)
ns = {"EPS_LADDER": LADDER, "ROT_MAX_DEG": ROT_MAX}
exec(compile(src[i:j], "climb_ladder", "exec"), ns)
climb = ns["climb_ladder"]
failures = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def stub(table):
    """search(eps) -> (rot, 'set-at-<eps>') from a dict; missing eps means None."""
    seen = []

    def search(eps):
        seen.append(eps)
        r = table.get(eps)
        return None if r is None else (r, "set-at-%.3f" % eps)
    return search, seen


print("\n[1] the first rung that CLEARS the limit wins, and stops the ladder")
s, seen = stub({0.045: 6.6, 0.055: 4.0})
best, eps = climb(s, LADDER, 8.0)
check("a passing first rung is taken", eps == 0.045 and best[0] == 6.6, str((best, eps)))
check("and no further rung is searched", seen == [0.045], str(seen))

print("\n[2] a rung that FAILS the limit does not end the search")
s, seen = stub({0.045: 8.9, 0.055: 5.2, 0.065: 4.1})
best, eps = climb(s, LADDER, 8.0)
check("the over-limit rung is not accepted", eps != 0.045, str((best, eps)))
check("the next passing rung is", eps == 0.055 and best[0] == 5.2, str((best, eps)))
check("and the ladder stops there, not at the global minimum",
      seen == [0.045, 0.055], str(seen))

print("\n[3] test25's real shape: three rungs empty, then 8.9, then better")
s, seen = stub({0.075: 8.9, 0.085: 7.4, 0.095: 3.0})
best, eps = climb(s, LADDER, 8.0)
check("empty rungs are skipped without being accepted", 0.045 in seen and eps != 0.045)
check("8.9 deg no longer ends it", eps != 0.075, str((best, eps)))
check("7.4 deg does, being the first under 8.0", eps == 0.085 and best[0] == 7.4,
      str((best, eps)))

print("\n[4] when NO rung clears the limit, the best one is used, not the last")
s, seen = stub({0.045: 22.0, 0.065: 9.1, 0.085: 14.0, 0.110: 30.0})
best, eps = climb(s, LADDER, 8.0)
check("every rung was tried", len(seen) == len(LADDER), str(seen))
check("the BEST is returned, not the last", eps == 0.065 and best[0] == 9.1,
      str((best, eps)))
check("and it is over the limit, so the caller must warn", best[0] > 8.0)

print("\n[5] no rung yields anything -> None, so burial is reached")
s, seen = stub({})
best, eps = climb(s, LADDER, 8.0)
check("returns None", best is None and eps is None, str((best, eps)))
check("after trying the whole ladder", seen == list(LADDER), str(seen))

print("\n[6] exactly at the limit counts as passing")
s, seen = stub({0.045: 8.0})
best, eps = climb(s, LADDER, 8.0)
check("8.0 deg against a limit of 8.0 is accepted", eps == 0.045, str((best, eps)))
check("and stops the ladder", seen == [0.045], str(seen))

print("\n[7] the caller still warns, and only when nothing cleared the limit")
after = src[src.index("chosen, chosen_eps = climb_ladder("):]
after = after[:after.index("\ndef ")]
check("try_measured_anchors delegates to climb_ladder",
      "climb_ladder(search, log=sys.stderr.write)" in after)
check("it warns when the accepted set is over the limit",
      "if rot > ROT_MAX_DEG:" in after and "WARNING" in after)
check("and the warning says the whole ladder was tried",
      "no rung of" in after, after[-400:])
check("the old unconditional first-success break is gone",
      "accepted the first round that succeeded" not in src)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all rotation-limit checks passed")
