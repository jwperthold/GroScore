#!/usr/bin/env python3
"""The anchor search must measure r the way the final guard measures it.

The search rejects a triad whose r + pull_dist exceeds the box budget, and the
final guard aborts on the same budget. Those agreed on the FRACTION (an earlier
fix) but not on the QUANTITY: the search used the snapshot cross distance while
the guard used the ensemble mean, and pooling five probe replicas moves r. test23
selected a triad at 1.306 nm (passes, 2.306 < 2.467), averaged to 1.490 (fails,
2.490), and died after its five 20 ns probes had already run -- the whole setup
lost to a dead band between two measurements of the same thing.

Also pinned here: two bugs made while fixing that, because both are the kind that
only appear on a path nobody exercises.

  * a helper defined AFTER its call site. Python binds at call time, so
    ensemble_cross_r sitting below try_measured_anchors raised NameError, which
    the caller swallowed into "falling back to the burial heuristic".
  * that fallback then hit a pre-existing KeyError: the burial path builds groups
    with no "atomnames", which ensemble_boresch_geometry had been reading blindly
    ever since the reference geometry started being ensemble-averaged.

Standalone, no pytest: python3 tests/test_anchor_box.py
"""
import os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB = os.path.join(ROOT, "utils", "make_boresch.py")
src = open(MB).read()
failures = []


def check(name, ok, detail=""):
    print("  %-68s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


print("\n[1] the search measures the ENSEMBLE cross distance")
check("ensemble_cross_r exists", "def ensemble_cross_r(" in src)
seg = src[src.index("def search(eps_max)"):]
seg = seg[:seg.index("\n  # Escalate")] if "\n  # Escalate" in seg else seg
check("search calls it", "ensemble_cross_r(cand)" in seg)
check("and uses it for the cross pair, not the snapshot norm",
      "cross_r(p, l, d1, d2)" in seg)
check("falling back to the snapshot only when there is no trajectory",
      "if ens is None" in seg and 'd1[p]["com"] - d2[l]["com"]' in seg)

print("\n[2] definition order -- the helper must exist when search runs")
i_def = src.index("def ensemble_cross_r(")
i_use = src.index("def try_measured_anchors(")
check("ensemble_cross_r is defined before try_measured_anchors",
      i_def < i_use, "def at %d, user at %d" % (i_def, i_use))
i_gw = src.index("def group_weights(")
check("group_weights too", i_gw < i_def)

print("\n[3] the search budget and the final guard are the same test")
sel = re.search(r"r_cross \+ args\.pull_dist > (\S+) \* PULL_LIMIT", src)
grd = re.search(r"r0_release > (\S+) \* PULL_LIMIT", src)
check("the search scales by a named fraction", sel is not None)
check("the guard scales by one too", grd is not None)
if sel and grd:
    check("and it is the SAME name (%s)" % sel.group(1),
          sel.group(1) == grd.group(1), "%s vs %s" % (sel.group(1), grd.group(1)))
check("r0_release is ref_r + pull_dist, i.e. what the search compares",
      re.search(r"r0_release = ref_r \+ args\.pull_dist", src) is not None)

print("\n[4] group_weights tolerates a group the burial path built")
ns = {"np": np, "BACKBONE": ("N", "CA", "C"),
      "mass_of": lambda a, nm: {"N": 14.0, "CA": 12.0, "C": 12.0}.get(nm, 1.0)}
i = src.index("def group_weights(")
j = src.index("\ndef ", i + 1)
exec(compile(src[i:j], "group_weights", "exec"), ns)
gw = ns["group_weights"]
full = {"atoms": [1, 2, 3], "atomnames": ["N", "CA", "C"]}
bare = {"atoms": [1, 2, 3]}                      # what the burial path produces
check("a group WITH atomnames is weighted by them",
      list(gw(full)) == [14.0, 12.0, 12.0], str(gw(full)))
check("a group WITHOUT them does not raise", gw(bare) is not None)
check("and falls back to the BACKBONE cycle",
      list(gw(bare)) == [14.0, 12.0, 12.0], str(gw(bare)))
six = {"atoms": [1, 2, 3, 4, 5, 6]}
check("the cycle repeats for a multi-residue group",
      list(gw(six)) == [14.0, 12.0, 12.0, 14.0, 12.0, 12.0], str(gw(six)))
odd = {"atoms": [1, 2, 3], "atomnames": ["N", "CA"]}     # wrong length
check("a mismatched atomnames length is ignored rather than zipped short",
      len(gw(odd)) == 3, str(gw(odd)))

print("\n[5] neither reader assumes the key is present any more")
check("ensemble_boresch_geometry uses group_weights",
      'w = group_weights(groups[role])' in src)
check("ensemble_cross_r uses group_weights",
      'w = group_weights(cand[k])' in src)
check('no bare ["atomnames"] lookup survives',
      '["atomnames"])' not in src.replace('"atomnames": atomnames', ''))

print("\n[6] the rejection is reported, not silent")
check("the count of over-budget combinations is printed",
      "box budget" in src and "n_over" in src)
check("and the accepted set's own r with it",
      re.search(r"the accepted set.*sits at", src, re.S) is not None)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all anchor-box checks passed")
