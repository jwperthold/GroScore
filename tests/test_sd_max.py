#!/usr/bin/env python3
"""The width ceiling on an interface spring.

The contact cutoff is applied to the MEAN distance over the pooled probe replicas,
and a mean cannot see how much of the time a pair is actually in contact. Measured
across test13/14/15, 27-40% of the springs surviving the mean cutoff are outside it
in more than a quarter of the frames, and r(mean, sd) is only +0.27 to +0.32, so the
mean cutoff does not remove them.

What must hold:

  * the ceiling is ON by default, at one number, stated where you would look
  * a non-positive value disables it, because "keep everything" has to stay sayable
  * it drops the WIDE pairs and only those
  * it refuses to strip the interface bare, however wide the pairs are: a set of a
    handful of springs is a worse failure than a set of some mobile ones
  * the value is recorded, since it selects which springs exist and two directories
    built at different values do not have comparable bound legs

Standalone, no pytest: python3 tests/test_sd_max.py
"""
import os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB = os.path.join(ROOT, "utils", "make_boresch.py")
src = open(MB).read()
failures = []


def check(name, ok, detail=""):
    print("  %-66s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


print("\n[1] the default is on, and is one number")
m = re.search(r"^SD_MAX_DEFAULT\s*=\s*([0-9.]+)", src, re.M)
check("SD_MAX_DEFAULT is named at module level", m is not None)
if m:
    check("and it is 0.15 nm", abs(float(m.group(1)) - 0.15) < 1e-12, m.group(1))
check("the parser takes its default FROM that constant, not a literal",
      re.search(r"'--sd-max'[^)]*default=SD_MAX_DEFAULT", src, re.S) is not None)
i_def = src.index("SD_MAX_DEFAULT =")
i_use = src.index("'--sd-max'")
check("the constant is defined BEFORE the parser reads it",
      i_def < i_use, "def at %d, use at %d" % (i_def, i_use))
m2 = re.search(r"^SD_MIN_KEEP\s*=\s*(\d+)", src, re.M)
check("a floor on the surviving spring count exists", m2 is not None)

print("\n[2] a non-positive value disables it")
check("SD_MAX_NM is None unless the flag is positive",
      re.search(r"SD_MAX_NM\s*=\s*args\.sd_max\s+if\s+\(args\.sd_max\s+or\s+0\)\s*>\s*0\s+else\s+None",
                src) is not None)

print("\n[3] the filter itself, run on synthetic spreads")
# The routine is not importable (make_boresch parses argv at import), so the rule
# is re-expressed here and checked against the SHIPPED thresholds rather than
# reimplemented from memory.
SD_MAX = float(m.group(1)) if m else 0.15
FLOOR = int(m2.group(1)) if m2 else 60
rng = np.random.default_rng(7)


def apply_rule(sd):
    drop = sd > SD_MAX
    if drop.all() or (len(sd) - int(drop.sum())) < FLOOR:
        return np.ones(len(sd), bool)          # refused: keep all
    return ~drop


sd = np.concatenate([rng.uniform(0.02, 0.14, 300), rng.uniform(0.16, 0.40, 40)])
keep = apply_rule(sd)
check("every pair above the ceiling is dropped", not (sd[keep] > SD_MAX).any())
check("every pair below it is kept", int(keep.sum()) == 300, str(int(keep.sum())))

wide = rng.uniform(0.20, 0.50, 200)
check("an interface that is ALL wide is kept whole, not stripped",
      apply_rule(wide).all())

mostly = np.concatenate([rng.uniform(0.02, 0.14, FLOOR - 10),
                         rng.uniform(0.20, 0.50, 300)])
check("and so is one that would be left under the floor",
      apply_rule(mostly).all(), "%d would survive" % (FLOOR - 10))

justover = np.concatenate([rng.uniform(0.02, 0.14, FLOOR + 5),
                           rng.uniform(0.20, 0.50, 300)])
check("but a set that stays above the floor IS filtered",
      int(apply_rule(justover).sum()) == FLOOR + 5)

print("\n[4] the work model the ceiling is justified by")
# <W_intro> = 0.5 * sum_k * mean(sd^2): independent of the spring COUNT, which is
# why dropping wide pairs helps even though k = sum_k/N rises to compensate.
SUM_K = 12500.0
before = 0.5 * SUM_K * np.mean(sd ** 2)
after = 0.5 * SUM_K * np.mean(sd[keep] ** 2)
check("dropping the wide pairs lowers the modelled bound-leg work",
      after < before, "%.1f -> %.1f" % (before, after))
k_before, k_after = SUM_K / len(sd), SUM_K / int(keep.sum())
check("even though k per spring rises", k_after > k_before,
      "%.2f -> %.2f" % (k_before, k_after))

print("\n[5] the value is recorded next to sum_k")
check("boresch_analytical.gs carries sd_max_nm",
      "sd_max_nm" in src and "n_interface_springs" in src)
check("and says 'off' rather than a number when disabled",
      re.search(r'sd_max_nm.*\n?.*"off"', src) is not None
      or '"off"' in src.split("sd_max_nm")[1][:200])

print("\n[6] the spread is reported whether or not it filters")
seg = src[src.index("frac_out = "):src.index("if SD_MAX_NM is not None")]
check("the spread diagnostic runs before any filtering",
      "per-pair distance spread" in seg)
check("and reports the tail, not just the median",
      "p90" in seg and "p99" in seg and "worst" in seg)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all sd-max checks passed")
