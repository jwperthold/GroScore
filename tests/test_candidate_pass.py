#!/usr/bin/env python3
"""Which pairs get restrained must be decided by the ensemble, not by one frame.

The contact cutoff has been applied to the ensemble mean since test7. What was
still a single frame was the CANDIDATE pass: pairs were enumerated at 0.6 nm in
npt_probe1_cluster.gro and only then measured, so a pair at 0.62 nm in that frame
whose mean is 0.55 was never considered. The old comment said so and said it had
not been measured to cost anything. Measured across eight runs of 2KTF under one
protocol, it costs this:

    candidates   556 - 918        a 65% range from one frame
    springs      228 - 322        a 41% range
    -> dG_intro  r = +0.91 with the spring count
    -> diss(E)   r = -0.73, because k = sum_k/N ties stiffness to the count
    -> dG_bind   r = -0.10        so it was never a bias

Enumerating wide and letting the mean decide removes the draw. Three things have to
hold for that to be true rather than merely intended, and each is checked here:

  * the pass must be WIDER than the cutoff, and the enumeration must use the wide
    number while the cutoff still applies to the mean
  * the run must SAY whether the pass was actually non-binding on this interface,
    since 0.9 nm is a measurement on 2KTF and not a law
  * the width-ceiling floor must count SPRINGS, not candidates. This one is the
    trap: the two counts were identical while candidates were enumerated at the
    cutoff, and widening the pass silently inflates the candidate count, so a floor
    counted over candidates goes quiet exactly as the set it protects gets small.

Standalone, no pytest: python3 tests/test_candidate_pass.py
"""
import os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "utils", "make_boresch.py")).read()
failures = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def const(name):
    m = re.search(r"^%s\s*=\s*([0-9.]+)" % name, src, re.M)
    return float(m.group(1)) if m else None


CUT = const("interfacecutoff")
CAND = const("CANDIDATE_CUT")

print("\n[1] the candidate pass is wider than the contact cutoff")
check("CANDIDATE_CUT exists", CAND is not None)
check("interfacecutoff is still %.2f nm" % (CUT or -1), CUT == 0.6, str(CUT))
check("and the candidate pass is wider (%.2f > %.2f)" % (CAND or -1, CUT or -1),
      CAND is not None and CAND > CUT, "%s vs %s" % (CAND, CUT))
check("wide enough to cover the +0.3 nm frame-to-mean shifts already observed",
      CAND is not None and CAND - CUT >= 0.25, str(CAND))

print("\n[2] the enumeration uses the wide cut, the cutoff still uses the mean")
enum = src[src.index("interdis = []"):]
enum = enum[:enum.index("_n_at_contact")]
check("the frame enumeration tests CANDIDATE_CUT",
      "d[i_idx, j_idx] <= CANDIDATE_CUT" in enum, enum[-300:])
check("and not the contact cutoff", "<= interfacecutoff" not in enum)
fin = src[src.index("def finalise_interface("):]
fin = fin[:fin.index("\ninterdis, _ref_note")]
check("finalise_interface still applies interfacecutoff to the settled reference",
      "keep = [p for p in pairs if p[2] <= interfacecutoff]" in fin)
check("and that reference is the ensemble mean, not the frame",
      "pairs, note = reference_on_ensemble(cands)" in fin)

print("\n[3] the run reports whether the pass was non-binding HERE")
check("it counts the springs the old 0.6 nm pass would have missed",
      "would not have been candidates at all" in fin)
check("it reports the margin to the candidate cut",
      "CANDIDATE_CUT - widest" in fin)
check("and warns when that margin is gone",
      "WHICH IS NOT MARGIN" in fin and "Raise CANDIDATE_CUT" in fin)

print("\n[4] the width-ceiling floor counts SPRINGS, not candidates")
m = re.search(r"^\s*n_left = (.*)$", src, re.M)
check("the floor's count is a single extractable expression", m is not None)
if m:
    expr = m.group(1)
    # 100 candidates. 30 are inside the contact cutoff and would be springs; of
    # those only 20 are narrow enough to survive the ceiling, which is under a
    # floor of 25. The other 70 are far pairs that never become springs, and 65 of
    # them are narrow -- so counting candidates gives 85 and clears the floor while
    # the interface is left pinned at 20 points.
    ns = {"np": np}
    ns["mean"] = np.array([0.5] * 30 + [0.8] * 70)
    ns["sd"] = np.array([0.1] * 20 + [0.9] * 10 + [0.1] * 65 + [0.9] * 5)
    ns["SD_MAX_NM"] = 0.15
    ns["interfacecutoff"] = 0.6
    got = eval(expr, ns)
    check("it counts only pairs inside the contact cutoff (20, not 85)",
          int(got) == 20, "%s -> %s" % (expr, got))
    old = int((ns["sd"] <= ns["SD_MAX_NM"]).sum())
    check("  the candidate-wide count would have been %d and cleared a floor of 25"
          % old, old == 85 and old > 25)
    floor = re.search(r"if drop\.all\(\) or (.*?):", src)
    check("and the floor test uses that count", floor is not None
          and "n_left" in floor.group(1), floor.group(1) if floor else "")
check("the message says which cutoff the count is inside",
      "springs inside the" in src)

print("\n[5] the trajectory pass is bounded in memory, so the cut is free to be wide")
ref = src[src.index("def reference_on_ensemble("):]
ref = ref[:ref.index("\ndef ")]
check("the distance loop is chunked over pairs", "for s in range(0, len(pairs), blk)" in ref)
check("with a declared byte budget", "CHUNK_BYTES" in ref)
mb = re.search(r"CHUNK_BYTES = (\d+) << 20", ref)
check("the budget is stated in MB", mb is not None, ref[:200])
if mb:
    budget = int(mb.group(1)) << 20
    # min_image_dist builds (frames, pairs, 27, 3) float64.
    for frames, pairs in ((5005, 2500), (5005, 250), (200, 40000)):
        per_pair = frames * 27 * 3 * 8
        blk = max(1, min(pairs, budget // per_pair))
        used = blk * per_pair
        check("  %d frames x %d pairs -> %d per chunk, %.0f MB"
              % (frames, pairs, blk, used / 2 ** 20),
              used <= budget or blk == 1, str(used))
check("and the unchunked whole-array call is gone",
      "min_image_dist(X[:, ia, :] - X[:, ib, :]" not in src)

print("\n[6] the stale note claiming this was not done is gone")
check('"the candidate pass still uses one frame" no longer appears',
      "the candidate pass still uses one frame" not in src)
check("and the spring-count line no longer says the frame chose them",
      "candidates from the reference frame" not in src)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all candidate-pass checks passed")
