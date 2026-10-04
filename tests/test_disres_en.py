#!/usr/bin/env python3
"""make_disres_en.py: the classic pull restraints refuse split frames and select
the right atoms.

The v1.4 audit (digest_audit/audit_restraints.txt) found the classic restraint
generator silently wrong in eight ways. Pinned here on a synthetic frame whose
every pair was counted by hand and by an independent brute-force census over all
lattice images (fixed numbers below, never computed from the code under test):

  1. Intact frame, AMBER caps: exactly 6 interface pairs, k = 25000/6, the
     network 3 + 2 CA pairs. v1.4 found 1 (items 6, 8 and 9 lose the other 5).
     Interface cutoff from both sides: CB-probe pairs at 0.599 and exactly
     0.600 nm are restrained, 0.601 nm is not.
  2. R1/F2, split frame: one side-B fragment one box vector away (as trjconv
     -pbc cluster left 1IJK's chain C) drops the CB-CB contact from the plain
     distances. The script must stop (exit 1, nothing written) and report
     '5 by plain distance, 6 by minimum image'; v1.4 wrote 5 springs, rc 0.
  3. R3: all of side B one box vector away gives 0 plain pairs; v1.4 printed 0
     with rc 0 and the structure scored 0 kJ/mol. Now exit 1, '0 by plain
     distance, 6 by minimum image'. Side B out of reach of every image: exit 1,
     'no cross-side heavy-atom pair'.
  4. The split test is by minimum image with the same 0.6 nm cutoff: in an 8 nm
     box a probe 8.599 or 8.600 nm above its partner (0.599 / 0.600 through the
     image) is a split, 8.601 is not (then the interface is empty).
  5. R4: the Ca2+ ion (residue CA, atom CA) 0.50-0.63 nm from three kept CAs is
     not a network node (v1.4: 3 springs on it), nor is a GROMOS ACE methyl named
     CA (v1.4: side B got 3 network pairs instead of 2).
  6. R5: the Mn2+ ion 0.25 nm from a side-A OD1 is restrained to it (v1.4 dropped
     every atom named MN*); a virtual-site dummy MN1 in a protein residue is not.
  7. R6: CHARMM36 C-terminal cut ends (OT1/OT2) are network anchors, so side B
     gets the same 2 pairs as with AMBER NME caps (v1.4: 0).
  8. R7: a receptor fragment listed after protein B and after the ligand is
     restrained (its 2 probe pairs and its network pair); the water and the NA/CL
     counterions after it never are, though they sit 0.30-0.59 nm from both sides.
  9. R8: the side-B ligand named 1PE ('   261PE' in .gro columns) is restrained to
     the side-A OG it binds (2 pairs). The split parse read residue 261: v1.4
     dropped it as solvent, and with item 8's rule alone it would land on side A,
     261 not being in chain_map.gs.

Run:  python3 tests/test_disres_en.py
"""

import math, os, re, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "utils", "make_disres_en.py")
failures = []

DODECA = 8.0


def check(name, ok, detail=""):
    detail = " | ".join(l.strip() for l in str(detail).splitlines() if l.strip())
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


def dodecahedron(d):
    """GROMACS xy-square rhombic dodecahedron, rows a, b, c."""
    return np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [d / 2, d / 2, d * math.sqrt(2) / 2]])


def build(ff="amber", shift=None):
    """The synthetic frame. ff: amber (ACE/NME caps), gromos (ACE whose methyl is
    named CA, C-terminal OT), charmm (ACE, C-terminal OT1/OT2). shift: None, 'B1'
    (that fragment by box vector c), 'B' (every side-B residue by a) or 'far'
    (side B 2.5 nm along z, out of reach). Returns (atoms, side_b, ions, ligands,
    box) with atoms as [resnum, resname, atomname, x, y, z] in file order."""
    box = dodecahedron(DODECA)
    atoms, side_b, ions, ligs = [], set(), set(), set()
    num = [0]
    groups = {}

    def add(name, alist, side, group=None):
        num[0] += 1
        for an, x, y, z in alist:
            atoms.append([num[0], name, an, x, y, z])
        if side == "B":
            side_b.add(num[0])
        if group:
            groups.setdefault(group, []).append(num[0])
        return num[0]

    def ace(x, y, z, s, side, group):
        if ff == "gromos":
            add("ACE", [("CA", x - 0.38, y, z), ("C", x - 0.24, y, z), ("O", x - 0.24, y - 0.1 * s, z - 0.1)], side, group)
        else:
            h = ("H1", "H2", "H3") if ff == "amber" else ("HH31", "HH32", "HH33")
            add("ACE", [(h[0], x - 0.45, y - 0.08 * s, z), ("CH3", x - 0.38, y, z), (h[1], x - 0.45, y - 0.05 * s, z + 0.08),
                        (h[2], x - 0.45, y - 0.05 * s, z - 0.08), ("C", x - 0.24, y, z), ("O", x - 0.24, y - 0.1 * s, z - 0.1)],
                side, group)

    def nme(x, y, z, s, side, group):
        if ff == "amber":
            add("NME", [("N", x + 0.24, y, z), ("H", x + 0.24, y - 0.1 * s, z), ("C", x + 0.38, y, z),
                        ("H1", x + 0.45, y - 0.08 * s, z), ("H2", x + 0.45, y - 0.05 * s, z + 0.08),
                        ("H3", x + 0.45, y - 0.05 * s, z - 0.08)], side, group)

    def fragment(side, group, x0, y, z, extra=None, cbdy=0.0):
        """Three ALA, CA 0.38 nm apart along x; side A's O and H point to -y,
        side B's to +y. extra: {residue k: [(name, dx, dy, dz)]} relative to CA."""
        s = 1.0 if side == "A" else -1.0
        ace(x0, y, z, s, side, group)
        for k in range(3):
            x = x0 + 0.38 * k
            al = [("N", x, y, z - 0.15), ("H", x, y - 0.1 * s, z - 0.15), ("CA", x, y, z), ("HA", x, y - 0.1 * s, z),
                  ("C", x, y, z + 0.15), ("O", x, y - 0.10 * s, z + 0.25)]
            for n, dx, dy, dz in (extra or {}).get(k, []):
                al.append((n, x + dx, y + dy, z + dz))
            if k == 2 and ff == "gromos":
                al.append(("OT", x + 0.08, y - 0.10 * s, z + 0.30))
            if k == 2 and ff == "charmm":
                al += [("OT1", x + 0.08, y - 0.10 * s, z + 0.30), ("OT2", x + 0.12, y - 0.12 * s, z + 0.10),
                       ("HT2", x + 0.16, y - 0.14 * s, z + 0.06)]
            add("ALA", al, side, group)
        nme(x0 + 0.76, y, z, s, side, group)

    # side A fragment 1: OG of its first residue holds the side-B ligand, OD1 of its
    # last residue the Mn2+, MN1 is a vsite dummy
    fragment("A", "A1", 1.00, 1.00, 1.00, extra={0: [("OG", 0.0, 0.45, 0.0)], 1: [("MN1", 0.0, 0.70, 0.0)],
                                                 2: [("OD1", 0.0, 0.45, 0.0)]})
    # side A fragment 2 and side B fragment 1 face each other 1 nm apart: CB-CB 0.50
    fragment("A", "A2", 1.00, 3.00, 1.00, extra={1: [("CB", 0.0, 0.25, 0.0)]})
    fragment("B", "B1", 1.00, 4.00, 1.00, extra={1: [("CB", 0.0, -0.25, 0.0)]})
    fragment("B", "B2", 1.00, 6.00, 3.00)
    ions.add(add("CA", [("CA", 1.38, 3.00, 0.50)], "A", "ion"))
    ions.add(add("MN", [("MN", 1.76, 1.70, 1.00)], "B", "ion"))
    for k, y in enumerate((0.599, 0.600, 0.601)):
        add("PRB", [("C1", 3.00 + 0.38 * k, y, 3.00)], "B", "probe")
    ligs.add(add("1PE", [("C1", 1.00, 1.85, 1.00), ("O1", 1.00, 2.00, 1.00)], "B", "lig"))
    # side A fragment 3, listed after protein B: CBs at y 0.000 below the probes
    fragment("A", "A3", 3.00, -0.30, 3.00, extra={k: [("CB", 0.0, 0.30, 0.0)] for k in range(3)})
    for w in range(2):
        zz = 1.30 + 0.30 * w
        add("SOL", [("OW", 1.38, 3.50, zz), ("HW1", 1.38, 3.55, zz + 0.08), ("HW2", 1.38, 3.45, zz + 0.08)], "-", "sol")
    add("NA", [("NA", 1.76, 3.30, 1.00)], "-", "sol")
    add("CL", [("CL", 1.76, 3.70, 1.00)], "-", "sol")

    if shift:
        move = {"B1": set(groups["B1"]), "B": set(side_b), "far": set(side_b)}[shift]
        vec = {"B1": box[2], "B": box[0], "far": np.array([0.0, 0.0, 2.5])}[shift]
        for a in atoms:
            if a[0] in move:
                a[3:] = list(np.array(a[3:]) + vec)
    return atoms, side_b, ions, ligs, box


def write_gro(path, atoms, box):
    with open(path, "w") as f:
        f.write("synthetic classic frame\n%5d\n" % len(atoms))
        for i, (rn, res, an, x, y, z) in enumerate(atoms):
            f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (rn, res, an, i + 1, x, y, z))
        f.write("%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f\n" % (
            box[0, 0], box[1, 1], box[2, 2], box[0, 1], box[0, 2], box[1, 0], box[1, 2], box[2, 0], box[2, 1]))


def run(atoms, side_b, ions, ligs, box):
    """Run the script in a fresh directory. Returns (rc, stdout, stderr, coords,
    untouched) with coords the pull coordinates of bind.mdp as ((resnum, resname,
    atom), (resnum, resname, atom), init, k) and untouched True when index.ndx and
    the three mdps are still empty."""
    d = tempfile.mkdtemp(prefix="disres_en_")
    try:
        write_gro(os.path.join(d, "frame.gro"), atoms, box)
        for name, nums in (("chain_map.gs", side_b), ("ion_residues.gs", ions), ("ligand_residues.gs", ligs)):
            with open(os.path.join(d, name), "w") as f:
                f.write("# test\n" + "".join("%d\n" % n for n in sorted(nums)))
        for name in ("index.ndx", "bind.mdp", "nptrev.mdp", "bindrev.mdp"):
            open(os.path.join(d, name), "w").close()
        p = subprocess.run([sys.executable, SCRIPT, "-f", "frame.gro", "-m", "chain_map.gs"],
                           cwd=d, capture_output=True, text=True)
        txt = open(os.path.join(d, "bind.mdp")).read()
        untouched = all(os.path.getsize(os.path.join(d, n)) == 0
                        for n in ("index.ndx", "bind.mdp", "nptrev.mdp", "bindrev.mdp"))
        grp = {int(g): int(a) for g, a in re.findall(r"pull-group(\d+)-name\s*=\s*a_(\d+)", txt)}
        coords = []
        for c, g1, g2 in re.findall(r"pull-coord(\d+)-groups\s*=\s*(\d+) (\d+)", txt):
            init = float(re.search(r"pull-coord%s-init\s*=\s*(\S+)" % c, txt).group(1))
            k = re.search(r"pull-coord%s-k\s*=\s*(\S+)" % c, txt).group(1)
            ident = [tuple(atoms[grp[int(g)] - 1][:3]) for g in (g1, g2)]
            coords.append((ident[0], ident[1], init, k))
        return p.returncode, p.stdout.strip(), p.stderr, coords, untouched
    finally:
        shutil.rmtree(d, ignore_errors=True)


def pair_names(coords):
    return sorted((a[0], a[2], b[0], b[2]) for a, b, _, _ in coords)


def network(coords, n_inter, side_b):
    """Network pairs by side: residue numbers of the two CAs."""
    out = {"A": [], "B": []}
    for a, b, _, _ in coords[n_inter:]:
        out["B" if a[0] in side_b else "A"].append((a[0], a[2], b[0], b[2]))
    return {s: sorted(v) for s, v in out.items()}


def main():
    if not os.path.isfile(SCRIPT):
        print("missing %s" % SCRIPT)
        sys.exit(1)

    # -------------------------------------------------------------------- 1, 5-9
    print("[1] intact frame, AMBER caps")
    atoms, B, I, L, box = build("amber")
    rc, out, err, coords, _ = run(atoms, B, I, L, box)
    check("[1] rc 0, prints 6 interface restraints", rc == 0 and out == "6", "rc %d out %r %s" % (rc, out, err.strip()[:200]))
    inter = coords[:6] if out == "6" else coords
    want = sorted([(8, "CB", 13, "CB"),              # A2 CB - B1 CB, 0.50 nm
                   (4, "OD1", 22, "MN"),             # R5: A1 OD1 - Mn2+, 0.25 nm
                   (28, "CB", 23, "C1"),             # R7: A3 CB - probe at 0.599
                   (29, "CB", 24, "C1"),             #     A3 CB - probe at exactly 0.600
                   (2, "OG", 26, "C1"),              # R8: A1 OG - side-B 1PE C1, 0.40 nm
                   (2, "OG", 26, "O1")])             #     A1 OG - side-B 1PE O1, 0.55 nm
    check("[1] the six interface pairs, no others", pair_names(inter) == want, str(pair_names(inter)))
    ks = {k for _, _, _, k in inter}
    check("[1] interface k = 25000/6 = 4166.66666667", ks == {"4166.66666667"}, str(ks))
    inits = {(a[0], b[0]): i for a, b, i, _ in inter}
    check("[1] probe references 0.59900000 and 0.60000000",
          abs(inits.get((28, 23), -1) - 0.599) < 1e-9 and abs(inits.get((29, 24), -1) - 0.6) < 1e-9, str(inits))
    check("[1] cutoff: probe at 0.601 nm (residue 25) not restrained", all(b[0] != 25 for _, b, _, _ in coords))
    net = network(coords, 6, B)
    check("[1] network side A: CA 2-4, 7-9, 28-30 (3 pairs)",
          net["A"] == [(2, "CA", 4, "CA"), (7, "CA", 9, "CA"), (28, "CA", 30, "CA")], str(net["A"]))
    check("[1] network side B: CA 12-14, 17-19 (2 pairs)",
          net["B"] == [(12, "CA", 14, "CA"), (17, "CA", 19, "CA")], str(net["B"]))
    check("[5] R4: no network spring on the Ca2+ ion (residue 21)",
          all(a[0] != 21 and b[0] != 21 for a, b, _, _ in coords[6:]))
    check("[6] R5: the vsite dummy MN1 (residue 3) is in no pair", all("MN1" not in (a[2], b[2]) for a, b, _, _ in coords))
    check("[8] R7: fragment 3 (after protein B) has its 2 probe pairs",
          (28, "CB", 23, "C1") in pair_names(inter) and (29, "CB", 24, "C1") in pair_names(inter))
    check("[8] R7/sentinels: no pair on water (32, 33) or NA/CL (34, 35)",
          all(a[0] < 32 and b[0] < 32 for a, b, _, _ in coords))
    check("[9] R8: side-B ligand 1PE (residue 26) on side B, bound to A1 OG",
          (2, "OG", 26, "C1") in pair_names(inter) and (2, "OG", 26, "O1") in pair_names(inter))

    print("[5] R4 under GROMOS names (ACE methyl named CA, C-terminal OT)")
    atoms, B, I, L, box = build("gromos")
    rc, out, err, coords, _ = run(atoms, B, I, L, box)
    check("[5] gromos: rc 0, 6 interface restraints", rc == 0 and out == "6", "rc %d out %r %s" % (rc, out, err.strip()[:200]))
    net = network(coords, 6, B)
    resname = {a[0]: a[1] for a in atoms}
    check("[5] gromos: no network pair on an ACE carbon",
          all(resname[r] != "ACE" for p in net["A"] + net["B"] for r in (p[0], p[2])), str(net))
    check("[5] gromos: network side B = CA 10-12, 14-16 (2 pairs, v1.4 had 3)",
          net["B"] == [(10, "CA", 12, "CA"), (14, "CA", 16, "CA")], str(net["B"]))
    check("[5] gromos: network side A = CA 2-4, 6-8, 24-26",
          net["A"] == [(2, "CA", 4, "CA"), (6, "CA", 8, "CA"), (24, "CA", 26, "CA")], str(net["A"]))

    print("[7] R6 under CHARMM36 names (ACE only, C-terminal OT1/OT2)")
    atoms, B, I, L, box = build("charmm")
    rc, out, err, coords, _ = run(atoms, B, I, L, box)
    check("[7] charmm: rc 0, 6 interface restraints", rc == 0 and out == "6", "rc %d out %r %s" % (rc, out, err.strip()[:200]))
    net = network(coords, 6, B)
    check("[7] charmm: network side B = CA 10-12, 14-16 (v1.4: none)",
          net["B"] == [(10, "CA", 12, "CA"), (14, "CA", 16, "CA")], str(net["B"]))
    check("[7] charmm: network side A = CA 2-4, 6-8, 24-26",
          net["A"] == [(2, "CA", 4, "CA"), (6, "CA", 8, "CA"), (24, "CA", 26, "CA")], str(net["A"]))

    # -------------------------------------------------------------------- 2-3
    print("[2] R1/F2: side-B fragment 1 one box vector (c) away")
    atoms, B, I, L, box = build("amber", shift="B1")
    rc, out, err, coords, untouched = run(atoms, B, I, L, box)
    check("[2] exit 1, nothing printed", rc == 1 and out == "", "rc %d out %r" % (rc, out))
    check("[2] reports 5 by plain distance, 6 by minimum image",
          "5 by plain distance, 6 by minimum image" in err, err.strip()[-300:])
    check("[2] index.ndx and the three mdps untouched", untouched)

    print("[3] R3: all of side B one box vector (a) away; side B out of reach")
    atoms, B, I, L, box = build("amber", shift="B")
    rc, out, err, coords, untouched = run(atoms, B, I, L, box)
    check("[3] whole side shifted: exit 1, nothing written", rc == 1 and out == "" and untouched, "rc %d out %r" % (rc, out))
    check("[3] reports 0 by plain distance, 6 by minimum image",
          "0 by plain distance, 6 by minimum image" in err, err.strip()[-300:])
    atoms, B, I, L, box = build("amber", shift="far")
    rc, out, err, coords, untouched = run(atoms, B, I, L, box)
    check("[3] side B out of reach: exit 1, 'no cross-side heavy-atom pair'",
          rc == 1 and untouched and "no cross-side heavy-atom pair" in err, "rc %d %s" % (rc, err.strip()[-200:]))

    # -------------------------------------------------------------------- 4
    print("[4] minimum-image cutoff of the split test, 8 nm cubic box")
    for ytop, want_split in ((8.599, True), (8.600, True), (8.601, False)):
        box8 = np.diag([8.0, 8.0, 8.0])
        mini = [[1, "ALA", "N", 2.000, -0.300, 1.850], [1, "ALA", "CA", 2.000, -0.300, 2.000],
                [1, "ALA", "C", 2.000, -0.300, 2.150], [1, "ALA", "O", 2.000, -0.400, 2.250],
                [1, "ALA", "CB", 2.000, 0.000, 2.000], [2, "PRB", "C1", 2.000, ytop, 2.000]]
        rc, out, err, coords, untouched = run(mini, {2}, set(), set(), box8)
        if want_split:
            ok = rc == 1 and untouched and "0 by plain distance, 1 by minimum image" in err
        else:
            ok = rc == 1 and untouched and "no cross-side heavy-atom pair" in err and "split" not in err
        check("[4] probe at y %.3f (%.3f through the image): %s" % (ytop, ytop - 8.0, "split" if want_split else "empty, no split"),
              ok, "rc %d %s" % (rc, err.strip()[-160:]))

    print()
    if failures:
        print("%d check(s) FAILED:" % len(failures))
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
