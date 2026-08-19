#!/usr/bin/env python3
"""fe_leg_efficiency must read the same rows groscore_fe reads.

It keeps its own reader, and that reader kept the old nstages = (NF - 5) / 4 rule
after the bound leg was split. That rule hardcodes exactly two non-stage work
fields, so a 27-field row gives 5.5, matches nothing, and every row is silently
rejected: the tool reported "no complete cycles" for the entire five-stage
protocol. Two readers of one format is the arrangement that produced this, so what
is pinned here is that they AGREE.

Standalone, no pytest: python3 tests/test_leg_efficiency_layout.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.argv = [sys.argv[0]]
import fe_protocol as P

failures = []


def check(name, ok, detail=""):
    print("  %-64s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


fe = open(os.path.join(ROOT, "groscore_fe.py")).read()
le = open(os.path.join(ROOT, "utils", "fe_leg_efficiency.py")).read()


def layout_of(src, alias):
    """Extract the _layout rule from a source file and run it.

    The two files import fe_protocol under different aliases, so the extracted
    lines are normalised onto one name rather than rewritten per file --
    rewriting "P." turned "_P." into "__P." and broke the test itself.
    """
    ns = {"P": P}
    i = src.index("def _layout(")
    j = src.index("\n\n", i)
    pre = re.search(r"^(\s*)_CUR = .*$", src[:i], re.M).group(0).strip()
    past = re.search(r"^\s*_PAST = .*$", src[:i], re.M)
    exec(pre.replace("_P.", "P."), ns)
    if past:
        exec(past.group(0).strip(), ns)
    else:
        ns["_PAST"] = {}
    body = "\n".join(l[2:] if l.startswith("  ") else l
                     for l in src[i:j].splitlines())
    exec(body, ns)
    return ns["_layout"]


print("\n[1] both files expose a width -> (nbound, nstages) rule")
check("groscore_fe has _layout", "def _layout(" in fe)
check("fe_leg_efficiency has _layout", "def _layout(" in le)
check("and fe_leg_efficiency no longer uses the bare (NF - 5) rule as its gate",
      "if nf < 9 or (nf - 5) % 4 != 0" not in le)

print("\n[2] they agree on every width that has ever shipped, and on nonsense")
try:
    a = layout_of(fe, "P")
    b = layout_of(le, "_P")
except Exception as e:                                     # noqa: BLE001
    check("both rules are extractable", False, repr(e))
    a = b = None

if a and b:
    widths = [9, 13, 17, 21, 25, 27, 31, 35, P.result_nf()]
    for w in widths:
        check("width %2d resolves the same in both: %s" % (w, a(w)), a(w) == b(w),
              "%s vs %s" % (a(w), b(w)))
    for w in (0, 5, 8, 10, 26, 28, 30):
        check("width %2d is rejected the same way (%s)" % (w, a(w)), a(w) == b(w),
              "%s vs %s" % (a(w), b(w)))
    check("the CURRENT protocol width parses at all", a(P.result_nf()) is not None)
    check("and resolves to the protocol's own shape",
          a(P.result_nf()) == (P.n_bound(), P.n_stages()), str(a(P.result_nf())))

print("\n[3] the split bound leg is summed, not indexed by its ends")
check("the bound branch sums the sub-legs", "sum(works[c][i] for i in range(nb))" in le)
check("and mirrors the reverse parts across the row",
      "works[c][nw - 1 - i]" in le)
check("stage indices start after the bound block",
      "j = nb + 2 * i" in le and "k = nb + 2 * ns" in le)
check("NBOUND is recorded per cycle alongside NSTAGES",
      "NBOUND[int(f[1])] = nb" in le)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all leg-efficiency layout checks passed")
