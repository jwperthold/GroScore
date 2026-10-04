#!/usr/bin/env python3
"""make_cutout.py builds fragments from protein residues only, extends them only
along their own chain, never counts caps, and fills the gaps the extension leaves.

Audit SH8 (digest_audit/audit_shared.txt): a Ca2+ ion (residue CA, atom CA) passed
the CA test and became a one-residue 'fragment'; its extension walked residue
numbers until it reached the end of the receptor chain and pulled in 1BVN's LYS
496 - LEU 497, a dipeptide 4 nm from the interface. Caps counted toward the
five-residue minimum, so 4-residue fragments stayed (1AK4, 1KXQ, 7CEI). Gaps were
filled before the extension only, so a gap the extension narrowed to one residue
got an NME and an ACE built into it (1BVN 413). Pinned on a synthetic full model
(side A: chains A1 2-41, A2 44-53, A3 56-85, each between ACE and NME, and a
Ca2+; side B: three 3-residue chains, each touching one CB of A1 at 0.50 nm). The
keep step (2.0 nm from a same-side interface atom) was recomputed independently:
A2 and A3 run 1.89-1.92 nm below the contacts, and every residue at the edge of a
window lies at least 0.03 nm inside or outside 2.0 nm.

  1. The Ca2+ (residue 102), 0.50 nm off A1's CA 10 at the interface, is in the
     cutout as an ion, and nothing comes with it: v1.4 extended it by A3's last
     residues 83-85 and its NME 86.
  2. A2 passes the interface for 4 residues, 44-47, right after its ACE 43: the
     fragment becomes 44-48 (v1.4: ACE 43 counted, 4 residues left once the caps
     were removed).
  3. A3 passes it for 4 residues (59-62) and, 4 residues on, for 3 (67-69); the
     extension takes 58 and 66, 70, leaving the 3-residue gap 63-65, which is
     filled: A3 is one fragment 58-70 (v1.4: 58-62 and 66-70, capped apart).
  4. The rest is as v1.4 cut it: A1 5-36, side B 88-90, 93-95, 98-100, no caps.

Run:  python3 tests/test_setup_cutout.py
"""

import os, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "utils", "make_cutout.py")
failures = []


def check(name, ok, detail=""):
    detail = " | ".join(l.strip() for l in str(detail).splitlines() if l.strip())
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


def build():
    """[(chain, resnum, resname, [(atom, x, y, z)])] in file order (nm), and the
    side-B residue numbers. Backbone along x, CA 0.38 nm apart, N and C 0.15 nm
    off the CA along z, O 0.1 nm along -y."""
    res, side_b = [], set()
    num = [0]

    def add(chain, name, atoms, side="A"):
        num[0] += 1
        res.append((chain, num[0], name, atoms))
        if side == "B":
            side_b.add(num[0])
        return num[0]

    def chain(ch, x0, y, z, n, side="A", extra=None):
        """ACE, n ALA, NME along +x from x0. extra: {offset k: [(atom, x, y, z)]}."""
        add(ch, "ACE", [("CH3", x0 - 0.38, y, z), ("C", x0 - 0.24, y, z), ("O", x0 - 0.24, y - 0.1, z)], side)
        nums = []
        for k in range(n):
            x = x0 + 0.38 * k
            at = [("N", x, y, z - 0.15), ("CA", x, y, z), ("C", x, y, z + 0.15), ("O", x, y - 0.1, z + 0.15)]
            at += (extra or {}).get(k, [])
            nums.append(add(ch, "ALA", at, side))
        xe = x0 + 0.38 * (n - 1)
        add(ch, "NME", [("N", xe + 0.24, y, z), ("C", xe + 0.38, y, z)], side)
        return nums

    # The three interface contacts: a CB of A1 (CA at y 0) and a CB of a side-B
    # chain 0.50 nm above it. Each A1 CB is placed so that the chain below it
    # (y -1.60) has exactly the residues listed within the 2.0 nm keep radius:
    # half-window w in x, height 1.60 + y_cb with (1.60 + y_cb)^2 + w^2 = 4.
    contacts = [(2.97, 0.288),    # w 0.660: A2 CAs 2.40-3.54 (44-47)
                (8.19, 0.288),    # w 0.660: A3 CAs 7.62-8.76 (59-62)
                (11.04, 0.317)]   # w 0.570: A3 CAs 10.66-11.42 (67-69)
    extra = {8: [("CB", 2.97, 0.288, 0.0)], 22: [("CB", 8.19, 0.288, 0.0)], 29: [("CB", 11.04, 0.317, 0.0)]}
    chain("A", 0.0, 0.0, 0.0, 40, extra=extra)          # A1: ACE 1, 2-41, NME 42
    chain("C", 2.40, -1.60, 0.0, 10)                    # A2: ACE 43, 44-53, NME 54
    chain("D", 6.48, -1.60, 0.0, 30)                    # A3: ACE 55, 56-85, NME 86
    for xc, yc in contacts:                             # B: 87-91, 92-96, 97-101
        chain("B", xc - 0.38, yc + 0.75, 0.0, 3, side="B", extra={1: [("CB", xc, yc + 0.50, 0.0)]})
    # Ca2+ on side A, 0.50 nm below A1's CA at x 3.04 (residue 10)
    add("A", "CA", [("CA", 3.04, -0.50, 0.0)])          # 102
    return res, side_b


def write_inputs(d, res, side_b):
    with open(os.path.join(d, "conf.gro"), "w") as f:
        atoms = [(r, n, a, x, y, z) for _, r, n, al in res for a, x, y, z in al]
        f.write("cutout test\n%5d\n" % len(atoms))
        for i, (r, n, a, x, y, z) in enumerate(atoms):
            f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (r, n, a, i + 1, x, y, z))
        f.write("  30.00000  30.00000  30.00000\n")
    with open(os.path.join(d, "fixed_capped.pdb"), "w") as f:
        serial, prev = 0, None
        for ch, r, n, al in res:
            key = (ch, n == "CA")
            if prev is not None and (key != prev or n == "ACE"):
                f.write("TER\n")
            prev = key
            for a, x, y, z in al:
                serial += 1
                f.write("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00\n" % (
                    serial, (" " + a) if len(a) < 4 else a, n, ch, r, 10 * x, 10 * y, 10 * z))
        f.write("TER\nEND\n")
    with open(os.path.join(d, "chain_map.gs"), "w") as f:
        f.write("# side B\n" + "".join("%d\n" % r for r in sorted(side_b)))


def cut(res, side_b):
    d = tempfile.mkdtemp(prefix="cutout_")
    try:
        write_inputs(d, res, side_b)
        p = subprocess.run([sys.executable, SCRIPT, "-m", "chain_map.gs"], cwd=d, capture_output=True, text=True)
        frags, cur, prev = [], [], None
        if p.returncode == 0 and os.path.isfile(os.path.join(d, "cutout.pdb")):
            for l in open(os.path.join(d, "cutout.pdb")):
                if l.startswith("TER"):
                    if cur:
                        frags.append(cur)
                    cur, prev = [], None
                elif l.startswith("ATOM"):
                    r = int(l[22:26])
                    if r != prev:
                        cur.append((r, l[17:20].strip()))
                    prev = r
            if cur:
                frags.append(cur)
        return p.returncode, frags, p.stdout + p.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)


def runs(frags):
    """Fragments as (first, last, n) of residue numbers."""
    return [(f[0][0], f[-1][0], len(f)) for f in frags]


def main():
    if not os.path.isfile(SCRIPT):
        print("missing %s" % SCRIPT)
        sys.exit(1)
    res, side_b = build()
    rc, frags, log = cut(res, side_b)
    check("make_cutout rc 0", rc == 0, log[-300:])
    r = runs(frags)
    print("  fragments: %s" % r)
    names = {num: name for f in frags for num, name in f}
    check("[1] the Ca2+ (residue 102) is in the cutout", names.get(102) == "CA", str(r))
    check("[1] nothing of A3's end (83-86) came with it", not any(83 <= n <= 86 for n in names), str(r))
    check("[2] A2 fragment 44-48, five residues (no ACE 43)", (44, 48, 5) in r and 43 not in names, str(r))
    check("[3] A3 one fragment 58-70 (gap 63-65 filled)", (58, 70, 13) in r, str(r))
    check("[4] fragments exactly: A1 5-36, A2 44-48, A3 58-70, Ca2+, B 88-90, 93-95, 98-100",
          sorted(r) == sorted([(5, 36, 32), (44, 48, 5), (58, 70, 13), (102, 102, 1),
                               (88, 90, 3), (93, 95, 3), (98, 100, 3)]), str(r))
    check("[4] no cap residue in the cutout", not any(n in ("ACE", "NME") for n in names.values()))

    print()
    if failures:
        print("%d check(s) FAILED:" % len(failures))
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
