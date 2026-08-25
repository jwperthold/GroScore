#!/usr/bin/env python3
"""The nine-stage ramp must stay a SUBDIVISION of the five-stage one.

This is the property the whole 2026-08-25 change rests on, and it is invisible in
the table. Cutting a stage at x and giving the halves time in proportion to their
spans leaves the pulling RATE unchanged across the cut, which is why the cut point
could be read off runs that had already been done instead of being fitted -- and
fitting is what cost test19-23 two of four BAR estimates.

Two things have to hold for that argument to keep applying, and both are one careless
edit away from being false while the file still looks right:

  * the five boundaries of the 2026-08-20 table must still be boundaries, so the
    stages measured at those rates are still the stages being run
  * the rate must be CONSTANT between consecutive old boundaries, so each parent's
    halves really do run at their parent's rate

A time nudged by hand, a span rounded to fewer decimals, or a boundary "tidied" all
break the second one silently: every stage still runs, the works still parse, the
row is still the right width, and the split point stops being the measured one.

Also pinned: the widths of every shipped ramp stay readable, because a reshaped ramp
changes the row width and the reader that does not know the old width drops every row
of an old directory WITHOUT A WORD.

Standalone, no pytest: python3 tests/test_subdivision.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "utils"))
import fe_protocol as P

# The table this one subdivides: five stages, equal 5200 ps, shipped as e586bcc and
# measured over test13/14/15 and test24-28. Written out rather than imported because
# the point of the test is that the current table still agrees with THIS one.
PARENT = [("A", 0.0, 0.12), ("B", 0.12, 0.20), ("C", 0.20, 0.31),
          ("D", 0.31, 0.49), ("E", 0.49, 1.00)]
PARENT_PS = 5200.0
failures = []


def check(name, ok, detail=""):
    print("  %-68s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def rate(s):
    """nm per ps, the number that goes into pull-coordN-rate."""
    return (s[2] - s[1]) * P.PULL_DIST / s[3]


print("\n[1] every boundary of the five-stage table is still a boundary")
bnd = P.boundaries()
for _, a, b in PARENT:
    check("u = %.2f survives" % b, any(abs(x - b) < 1e-9 for x in bnd), str(bnd))
check("and the ramp still spans 0 -> PULL_DIST",
      abs(bnd[0]) < 1e-12 and abs(bnd[-1] - P.PULL_DIST) < 1e-12, str(bnd))

print("\n[2] the rate is constant inside each parent, so each cut is at fixed rate")
scale = None
for name, a, b in PARENT:
    kids = [s for s in P.RAMP if s[1] >= a - 1e-9 and s[2] <= b + 1e-9]
    check("parent %s is covered by %d stage(s)" % (name, len(kids)),
          bool(kids) and abs(sum(k[2] - k[1] for k in kids) - (b - a)) < 1e-9,
          str(kids))
    if not kids:
        continue
    rates = [rate(k) for k in kids]
    lo, hi = min(rates), max(rates)
    check("  and they all run at one rate (%.4e, spread %.2f%%)"
          % (rates[0], 100 * (hi - lo) / hi),
          (hi - lo) / hi < 1e-3, str(rates))
    # Every parent must have been slowed by the SAME factor to pay for the holds;
    # a single parent retimed on its own is exactly the move that needs a density.
    f = (PARENT_PS / (b - a)) * rates[0] / P.PULL_DIST
    if scale is None:
        scale = f
    check("  and by the same factor as every other parent (%.4f)" % f,
          abs(f - scale) < 2e-3, "%.6f vs %.6f" % (f, scale))

print("\n[3] the cut points are the measured ones")
MEASURED = {"B": 0.166, "C": 0.250, "D": 0.379, "E": 0.622}   # sd 0.014/0.011/0.013/0.029
for name, x in sorted(MEASURED.items()):
    check("parent %s is cut at %.3f" % (name, x),
          any(abs(u - x) < 1e-9 for u in bnd), str(bnd))
check("parent A is NOT cut (n_sigma 1.51, it is not near the cliff)",
      not any(0.0 < u < 0.12 - 1e-9 for u in bnd), str(bnd))
check("so the ramp has exactly nine stages", P.n_stages() == 9, str(P.n_stages()))

print("\n[4] subdividing did not change the cycle")
check("still 100 ns per cycle", abs(P.cycle_ps() - 100000.0) < 1.0,
      "%.1f ps" % P.cycle_ps())
check("the bound leg is untouched", [b[:3] for b in P.BOUND] ==
      [("1", 0.0, 0.25), ("2", 0.25, 1.0)], str(P.BOUND))
holds = [l["ps"] for l in P.legs() if l["kind"] == "hold"]
check("the handoffs are still 1 ns", holds.count(P.HANDOFF_HOLD_PS) == 4, str(holds))
check("and the eight new-and-old mid-ramp holds pay for the boundaries",
      holds.count(P.RAMP_HOLD_PS) == 16, str(holds))

print("\n[5] rows from every shipped ramp still parse")
fe = open(os.path.join(ROOT, "groscore_fe.py")).read()
le = open(os.path.join(ROOT, "utils", "fe_leg_efficiency.py")).read()
for who, src in (("groscore_fe", fe), ("fe_leg_efficiency", le)):
    m = re.search(r"^\s*_PAST = (\{.*?\})", src, re.M)
    check("%s declares _PAST on one line" % who, m is not None)
    if not m:
        continue
    past = eval(m.group(1))
    check("  it remembers the six-stage row (31 fields)", past.get(31) == (2, 6),
          str(past))
    check("  and the five-stage row (27 fields), which just stopped being current",
          past.get(27) == (2, 5), str(past))
    check("  and does NOT claim the current width, which comes from the protocol",
          P.result_nf() not in past, "%d in %s" % (P.result_nf(), past))
check("the current row is 43 fields", P.result_nf() == 43, str(P.result_nf()))

print("\n[6] nothing downstream hardcodes the stage letters any more")
check("groscore_fe takes them from the protocol",
      "LETTERS = P.stage_letters()" in fe)
check('the "ABCDEFGH" literal that would have run out at nine is gone',
      '"ABCDEFGH"' not in fe.replace('# "ABCDEFGH" survived', ""))
check("and the protocol supplies nine of them",
      len(P.stage_letters()) == 9 and P.stage_letters()[-1] == "I",
      str(P.stage_letters()))

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all subdivision checks passed")
