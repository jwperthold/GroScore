#!/usr/bin/env python3
"""The stage-0 gates check_brokenloop.py and check_entangledloops.py read every
atom, take CA atoms from protein residues only, and never pass on a failure.

Audit SH6 (digest_audit/audit_shared.txt) and the runner audit's extra finding:
both gates parsed .gro lines with str.split(), which merges atom name and number
from atom 10000 on ('CA10000'); those atoms dropped out silently (1HE8 conf.gro:
9999 of 14958 atoms read, none of protein A's 166 CA). Both read Ca2+ ions
(residue CA, atom CA) as chain CA atoms. A missing frame gave empty lists and the
verdict 0, and the runners test '-eq 0', which also passes on empty output.
Pinned here on synthetic frames (CA 0.38 nm apart, N and C 0.15 nm off each CA):

  1. check_brokenloop: a 0.80 nm chain break whose two ends sit 0.30 nm below the
     other protein prints 1, also when 10002 atoms come first (v1.4: 0 there).
     A continuous chain followed by a Ca2+ ion 0.80 nm on and 0.30 nm from the
     other protein prints 0 (v1.4: 1, the ion read as the next CA).
  2. check_entangledloops: two chains, each carrying a 12-CA ring of radius 0.6
     nm, rings interlocked (the gate's linking number 1.195) print 1, also when
     10002 atoms come first (v1.4: 0 there); rings 2.6 nm apart print 0.
  3. Both: a missing frame and a side without CA atoms print 2 and exit 1 (v1.4:
     0, exit 0).

Run:  python3 tests/test_setup_gates.py
"""

import math, os, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BROKEN = os.path.join(REPO, "utils", "check_brokenloop.py")
ENTANGLED = os.path.join(REPO, "utils", "check_entangledloops.py")
failures = []


def check(name, ok, detail=""):
    detail = " | ".join(l.strip() for l in str(detail).splitlines() if l.strip())
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


class Frame(object):
    """Atoms in file order; residues numbered from 1; side B residues recorded."""

    def __init__(self):
        self.atoms, self.side_b, self.res = [], set(), 0

    def filler(self, n):
        """n atoms in 3-atom non-protein residues, 20 nm away from everything."""
        for i in range(n):
            if i % 3 == 0:
                self.res += 1
            self.atoms.append((self.res, "UNK", "C%d" % (i % 3 + 1),
                               20.0 + 0.1 * (i % 50), 20.0 + 0.1 * (i // 50 % 50), 20.0 + 0.1 * (i // 2500)))

    def residue(self, side, ca, name="ALA"):
        self.res += 1
        x, y, z = ca
        for an, dz in (("N", -0.15), ("CA", 0.0), ("C", 0.15)):
            self.atoms.append((self.res, name, an, x, y, z + dz))
        if side == "B":
            self.side_b.add(self.res)

    def ion(self, side, pos, name="CA"):
        self.res += 1
        self.atoms.append((self.res, name, name) + tuple(pos))
        if side == "B":
            self.side_b.add(self.res)

    def write(self, d, tag):
        g = os.path.join(d, tag + ".gro")
        with open(g, "w") as f:
            f.write("gate test\n%5d\n" % len(self.atoms))
            for i, (r, rn, an, x, y, z) in enumerate(self.atoms):
                f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (r % 100000, rn, an, (i + 1) % 100000, x, y, z))
            f.write("  30.00000  30.00000  30.00000\n")
        cm = os.path.join(d, tag + ".gs")
        with open(cm, "w") as f:
            f.write("# side B\n" + "".join("%d\n" % r for r in sorted(self.side_b)))
        return g, cm


def gate(script, g, cm):
    p = subprocess.run([sys.executable, script, "-f", g, "-m", cm], capture_output=True, text=True)
    return p.stdout.strip(), p.returncode, p.stderr.strip()


def broken_frame(filler=0, ion_instead=False):
    """Side A: 5 CAs, then (break) 3 CAs, 0.80 nm across the break; or, with
    ion_instead, 8 continuous CAs and a Ca2+ 0.80 nm past the last one. Side B: a
    continuous chain 0.30 nm above the two ends (y 0.30)."""
    f = Frame()
    f.filler(filler)
    xs = [1.0 + 0.38 * k for k in range(5)] + [1.0 + 0.38 * 4 + 0.80 + 0.38 * k for k in range(3)]
    if ion_instead:
        xs = [1.0 + 0.38 * k for k in range(8)]
    for x in xs:
        f.residue("A", (x, 0.0, 0.0))
    end = xs[4] if not ion_instead else xs[-1]
    if ion_instead:
        f.ion("A", (end + 0.80, 0.0, 0.0))
    nb = int(round((0.80 + 0.76) / 0.38)) + 1
    for k in range(nb):
        f.residue("B", (end - 0.38 + 0.38 * k, 0.30, 0.0))
    return f


def ring_chain(f, side, center, plane, n_ring=12, r=0.6, lead=3, step=0.38):
    c = np.array(center, float)
    if plane == "xy":
        u, v, ld = np.array([1, 0, 0.]), np.array([0, 1, 0.]), np.array([0, 0, 1.])
    else:
        u, v, ld = np.array([1, 0, 0.]), np.array([0, 0, 1.]), np.array([0, 1, 0.])
    ring = [c + r * (math.cos(2 * math.pi * k / n_ring) * u + math.sin(2 * math.pi * k / n_ring) * v)
            for k in range(n_ring)]
    pts = [ring[0] + ld * step * (lead - i) for i in range(lead)] + ring + \
          [ring[-1] + ld * step * (i + 1) for i in range(lead)]
    for p in pts:
        f.residue(side, tuple(np.round(p, 3)))


def ring_frame(dx, filler=0):
    f = Frame()
    f.filler(filler)
    ring_chain(f, "A", (5.0, 5.0, 5.0), "xy")
    ring_chain(f, "B", (5.0 + dx, 5.0, 5.0), "xz")
    return f


def main():
    for s in (BROKEN, ENTANGLED):
        if not os.path.isfile(s):
            print("missing %s" % s)
            sys.exit(1)
    d = tempfile.mkdtemp(prefix="gates_")
    try:
        print("[1] check_brokenloop.py")
        out, rc, err = gate(BROKEN, *broken_frame().write(d, "b1"))
        check("[1] chain break 0.80 nm, both ends 0.30 nm from B: prints 1", (out, rc) == ("1", 0), "%r rc %d %s" % (out, rc, err))
        f = broken_frame(filler=10002)
        out, rc, err = gate(BROKEN, *f.write(d, "b2"))
        check("[1] the same after 10002 atoms (atom numbers >= 10003): prints 1",
              (out, rc) == ("1", 0) and len(f.atoms) > 10000, "%r rc %d %s" % (out, rc, err))
        out, rc, err = gate(BROKEN, *broken_frame(ion_instead=True).write(d, "b3"))
        check("[1] continuous chain + Ca2+ 0.80 nm on, 0.30 nm from B: prints 0", (out, rc) == ("0", 0), "%r rc %d %s" % (out, rc, err))

        print("[2] check_entangledloops.py")
        out, rc, err = gate(ENTANGLED, *ring_frame(0.6).write(d, "e1"))
        check("[2] interlocked rings: prints 1", (out, rc) == ("1", 0), "%r rc %d %s" % (out, rc, err))
        f = ring_frame(0.6, filler=10002)
        out, rc, err = gate(ENTANGLED, *f.write(d, "e2"))
        check("[2] interlocked rings after 10002 atoms: prints 1",
              (out, rc) == ("1", 0) and len(f.atoms) > 10000, "%r rc %d %s" % (out, rc, err))
        out, rc, err = gate(ENTANGLED, *ring_frame(2.6).write(d, "e3"))
        check("[2] rings 2.6 nm apart: prints 0", (out, rc) == ("0", 0), "%r rc %d %s" % (out, rc, err))

        print("[3] failures print 2 and exit 1")
        g, cm = broken_frame().write(d, "f1")
        empty_cm = os.path.join(d, "empty.gs")
        open(empty_cm, "w").write("# no side B\n")
        for name, script in (("check_brokenloop", BROKEN), ("check_entangledloops", ENTANGLED)):
            out, rc, err = gate(script, os.path.join(d, "missing.gro"), cm)
            check("[3] %s: missing frame -> 2, exit 1" % name, (out, rc) == ("2", 1), "%r rc %d" % (out, rc))
            out, rc, err = gate(script, g, empty_cm)
            check("[3] %s: no CA on side B -> 2, exit 1" % name, (out, rc) == ("2", 1), "%r rc %d" % (out, rc))
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print()
    if failures:
        print("%d check(s) FAILED:" % len(failures))
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
