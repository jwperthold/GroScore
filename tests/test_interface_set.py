#!/usr/bin/env python3
"""The restrained set and its references must be decided on the same basis.

test7 decided them separately: candidates by one frame at 0.6 nm, references by
the trajectory mean. The mean is a different number -- it moved +0.86 A on average
and up to +5.4 A -- so 253 of 637 springs ended up referenced BEYOND the contact
cutoff, one at 1.11 nm. Those are not contacts, they diluted the stiffness budget,
and they silently capped interface_qc's `formed` at 60.3% by construction, which
made a healthy run look like a failing one.

What must hold, on whatever setup directories are present:

  * every restrained pair's REFERENCE is inside the contact cutoff
  * the frame distance is kept alongside it, so the shift stays auditable
  * the pipeline order in make_boresch.py is select -> re-reference -> re-cut ->
    cap, since each step should see what the one before it decided
  * dropping is never silent

Standalone, no pytest: python3 tests/test_interface_set.py
"""
import os, re, sys, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB = os.path.join(ROOT, "utils", "make_boresch.py")
CUTOFF = 0.6

failures = []


def check(name, ok, detail=""):
    print("  %-64s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


src = open(MB).read()

print("\n[1] the pipeline is ordered select -> re-reference -> re-cut -> cap")
pos = {}
for key, pat in (("candidates", r"^_n_snapshot = len\(interdis\)"),
                 ("finalise",   r"^interdis, _ref_note = finalise_interface\("),
                 ("cap",        r"^if MAX_PER_CONTACT and _n_uncapped:"),
                 ("numinterdis", r"^numinterdis = len\(interdis\)"),
                 ("k_inter",    r"^k_inter = args\.sum_k / numinterdis")):
    m = re.search(pat, src, re.M)
    check("make_boresch.py still has the %s step" % key, m is not None)
    pos[key] = m.start() if m else -1
if all(v >= 0 for v in pos.values()):
    order = ["candidates", "finalise", "cap", "numinterdis", "k_inter"]
    check("and they appear in that order",
          all(pos[a] < pos[b] for a, b in zip(order, order[1:])), pos)

print("\n[2] the cutoff is applied to the reference, not to the frame distance")
m = re.search(r"keep = \[p for p in pairs if p\[2\] <= interfacecutoff\]", src)
check("finalise_interface re-cuts on the re-referenced distance", m is not None)
check("and aborts rather than proceeding with nothing",
      "NO_INTERFACE_CONTACTS" in src)
check("a drop is always reported", re.search(
    r"beyond the %.*contact cutoff\s*\"?\s*\n?.*not restrained", src, re.S) is not None
    or "are not restrained" in src)

print("\n[3] nothing reads the set before it is settled")
for name, pat in (("the index writer", r"def _write_index_groups_to"),
                  ("boresch_analytical.gs", r'open\("boresch_analytical\.gs", "w"\)'),
                  ("interface_contacts.gs", r'open\("interface_contacts\.gs", "w"\)'),
                  ("build_coords", r"^def build_coords")):
    m = re.search(pat, src, re.M)
    if m:
        check("%s comes after finalise_interface" % name,
              m.start() > pos["finalise"], "%d vs %d" % (m.start(), pos["finalise"]))

print("\n[4] every setup directory present obeys the invariant")
found = 0
for cf in sorted(glob.glob(os.path.join(ROOT, "test*", "*", "interface_contacts.gs"))) + \
          sorted(glob.glob("/tmp/p3*/interface_contacts.gs")):
    hdr = ""
    ref, frame = [], []
    for line in open(cf):
        if line.startswith("#"):
            hdr += line
            continue
        if not line.strip():
            continue
        f = line.split()
        ref.append(float(f[2]))
        frame.append(float(f[9]) if len(f) > 9 else float("nan"))
    if not ref:
        continue
    found += 1
    ref = np.array(ref); frame = np.array(frame)
    tag = os.path.relpath(cf, ROOT) if cf.startswith(ROOT) else cf
    mean_ref = "mean over" in hdr
    applied = "cutoff is applied to IT" in hdr
    over = (ref > CUTOFF + 1e-9).sum()
    print(f"      {tag}")
    print(f"        {len(ref)} pairs, refs {ref.min():.3f}-{ref.max():.3f} nm, "
          f"{over} beyond {CUTOFF} nm"
          + ("   [pre-fix setup]" if not applied else ""))
    if applied:
        check("  %s: no reference beyond the cutoff" % tag, over == 0, "%d over" % over)
        check("  %s: frame distance recorded too" % tag,
              np.isfinite(frame).all())
        check("  %s: formed has a 100%% ceiling" % tag,
              (ref <= CUTOFF + 1e-9).all())
    elif mean_ref:
        # a directory built by the half-finished version; this is the defect
        print(f"        ^ selected on the frame, referenced on the mean: "
              f"{100.0*over/len(ref):.0f}% of springs are not contacts")
check("at least one setup was available to check", found > 0)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all interface-set checks passed")
