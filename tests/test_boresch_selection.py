#!/usr/bin/env python3
"""make_boresch.py selects the interface candidates and the elastic network by the
rules of make_disres_en.py, so the FE and the classic engine restrain the same
atoms of a frame.

The classic generator's audit (R4-R8) fixed five selection rules in
make_disres_en.py; the FE generator kept the old ones (v15_work review PY-1 to
PY-4). On the hand-counted synthetic frame of tests/test_disres_en.py, under AMBER,
GROMOS and CHARMM36 atom names, the FE interface pairs within 0.6 nm and the FE
network pairs must be the classic sets, and these facts of the frame must hold:

  PY-1  the Ca2+ ion (residue CA, atom CA) is no network node or anchor, nor is a
        GROMOS ACE methyl named CA (side B: CA 10-12 and 14-16, 2 pairs, not 3),
        and the ion is no backbone row of the anchor search (polymer_backbone)
  PY-2  CHARMM36 C-terminal cut ends (OT1/OT2) are network anchors: side B gets
        CA 10-12 and 14-16 (before: none)
  PY-3  the Mn2+ ion (atom MN) 0.25 nm from a side-A OD1 is an interface
        candidate; the virtual-site dummy MN1 is not
  PY-4  the side-B ligand 1PE ('   261PE' in .gro columns under AMBER names) is
        read on side B and bound to the side-A OG; the receptor fragment listed
        after protein B keeps its probe pairs; the solute is every atom before the
        first water, so no water and no NA/CL is read

The frame sits in a 30 nm box, so that it is one piece by contact (in the 8 nm
box of test_disres_en.py one fragment's nearest contact is through an image, which
make_boresch.py's PART 0 rightly calls a split frame).

Run:  python3 tests/test_boresch_selection.py
"""
import contextlib, io, os, re, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(REPO, "utils")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_disres_en as T          # the synthetic frame; its checks run only as __main__

failures = []


def check(name, ok, detail=""):
    print("  %-74s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:400]))
    if not ok:
        failures.append(name)


SRC = open(os.path.join(UTILS, "make_boresch.py")).read()
_lines = SRC.split("\n")
_cut = next(i for i, l in enumerate(_lines) if l.startswith("# PART 3 - Boresch anchor selection"))
TOP = "\n".join(_lines[:_cut - 1]) + "\n"     # PART 0-2: parse, interface candidates, network


def write_case(d, atoms, side_b, ions, ligs, box):
    T.write_gro(os.path.join(d, "frame.gro"), atoms, box)
    for name, nums in (("chain_map.gs", side_b), ("ion_residues.gs", ions), ("ligand_residues.gs", ligs)):
        with open(os.path.join(d, name), "w") as f:
            f.write("# test\n" + "".join("%d\n" % n for n in sorted(nums)))


def classic(atoms, side_b, ions, ligs, box):
    """rc, interface pairs and network pairs of make_disres_en.py, by atom number."""
    d = tempfile.mkdtemp(prefix="bsel_classic_")
    try:
        write_case(d, atoms, side_b, ions, ligs, box)
        for name in ("index.ndx", "bind.mdp", "nptrev.mdp", "bindrev.mdp"):
            open(os.path.join(d, name), "w").close()
        p = subprocess.run([sys.executable, os.path.join(UTILS, "make_disres_en.py"), "-f", "frame.gro",
                            "-m", "chain_map.gs"], cwd=d, capture_output=True, text=True)
        txt = open(os.path.join(d, "bind.mdp")).read()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    grp = {int(g): int(a) for g, a in re.findall(r"pull-group(\d+)-name\s*=\s*a_(\d+)", txt)}
    inter, net = set(), set()
    for c, g1, g2 in re.findall(r"pull-coord(\d+)-groups\s*=\s*(\d+) (\d+)", txt):
        k = float(re.search(r"pull-coord%s-k\s*=\s*(\S+)" % c, txt).group(1))
        pair = (grp[int(g1)], grp[int(g2)])
        if k == 250.0:                      # the network's enk; the interface k is 25000/6
            net.add(frozenset(pair))
        else:
            inter.add(pair)
    return p.returncode, inter, net


def fe(atoms, side_b, ions, ligs, box):
    """make_boresch.py PART 0-2 on the frame: (namespace, interface pairs within
    0.6 nm, network pairs), by atom number."""
    d = tempfile.mkdtemp(prefix="bsel_fe_")
    ns = {"__file__": os.path.join(UTILS, "make_boresch.py"), "__name__": "make_boresch_top"}
    argv, cwd = sys.argv[:], os.getcwd()
    try:
        write_case(d, atoms, side_b, ions, ligs, box)
        sys.argv = ["make_boresch.py", "-f", "frame.gro", "-m", "chain_map.gs", "--tpr", "none.tpr",
                    "--traj", "none.xtc"]
        os.chdir(d)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(TOP, "make_boresch.py", "exec"), ns)
            except SystemExit as e:
                ns["_exit"] = (e.code, err.getvalue())
    finally:
        sys.argv = argv
        os.chdir(cwd)
        shutil.rmtree(d, ignore_errors=True)
    if "_exit" in ns:
        return ns, None, None
    p1, p2 = ns["prot1_data"], ns["prot2_data"]
    inter = {(p1[i][3], p2[j][3]) for i, j, x in ns["interdis"] if x <= 0.6}
    k1, k2 = ns["protkeep1"], ns["protkeep2"]
    net = ({frozenset((p1[k1[i]][3], p1[k1[j]][3])) for i, j, _ in ns["en1dis"]}
           | {frozenset((p2[k2[i]][3], p2[k2[j]][3])) for i, j, _ in ns["en2dis"]})
    return ns, inter, net


def named(atoms, pairs):
    """Pairs of atom numbers as sorted (resnum, atom name, resnum, atom name)."""
    out = []
    for p in pairs:
        a, b = sorted(p)
        out.append((atoms[a - 1][0], atoms[a - 1][2], atoms[b - 1][0], atoms[b - 1][2]))
    return sorted(out)


for ff in ("amber", "gromos", "charmm"):
    print("\n[%s] the FE selection on the synthetic frame of test_disres_en.py" % ff)
    atoms, B, I, L, _box8 = T.build(ff)
    box = T.dodecahedron(30.0)
    res_of = lambda name: [a[0] for a in atoms if a[1] == name]
    holder = lambda atom: next(a[0] for a in atoms if a[2] == atom)
    ca_ion, mn_ion, lig = res_of("CA")[0], res_of("MN")[0], res_of("1PE")[0]
    first_sol = min(i + 1 for i, a in enumerate(atoms) if a[1] == "SOL")
    a3 = {a[0] for i, a in enumerate(atoms) if a[0] > lig and i + 1 < first_sol}
    rc, ci, cn = classic(atoms, B, I, L, box)
    ns, fi, fn = fe(atoms, B, I, L, box)
    if fi is None:
        check("%s: make_boresch.py PART 0-2 run through" % ff, False, ns.get("_exit"))
        continue
    check("%s: make_disres_en.py rc 0 on the same frame, 6 interface pairs" % ff,
          rc == 0 and len(ci) == 6, (rc, len(ci)))
    check("%s: FE interface pairs within 0.6 nm == classic" % ff, fi == ci,
          "FE only %s, classic only %s" % (named(atoms, fi - ci), named(atoms, ci - fi)))
    check("%s: FE network pairs == classic" % ff, fn == cn,
          "FE only %s, classic only %s" % (named(atoms, fn - cn), named(atoms, cn - fn)))
    nodes = {r for p in named(atoms, fn) for r in (p[0], p[2])}
    resname = {a[0]: a[1] for a in atoms}
    check("%s: PY-1 no network node on the Ca2+ ion (residue %d) or an ACE carbon" % (ff, ca_ion),
          ca_ion not in nodes and all(resname[r] != "ACE" for r in nodes), named(atoms, fn))
    ion_atom = next(i + 1 for i, a in enumerate(atoms) if a[0] == ca_ion)
    bb = ns["polymer_backbone"](ns["prot1_data"]) if "polymer_backbone" in ns else []
    check("%s: PY-1 the Ca2+ atom is no backbone row of the anchor search" % ff,
          ion_atom not in bb and len(bb) > 0, (ion_atom, len(bb)))
    fnames = named(atoms, fi)
    check("%s: PY-3 Mn2+ (residue %d) is a candidate of the side-A OD1" % (ff, mn_ion),
          (holder("OD1"), "OD1", mn_ion, "MN") in fnames, fnames)
    check("%s: PY-3 the vsite dummy MN1 is in no pair" % ff,
          all("MN1" not in (p[1], p[3]) for p in fnames), fnames)
    check("%s: PY-4 ligand 1PE (residue %d) read on side B, bound to the side-A OG" % (ff, lig),
          (holder("OG"), "OG", lig, "C1") in fnames and (holder("OG"), "OG", lig, "O1") in fnames
          and lig in {r[1] for r in ns["prot2_data"]}, fnames)
    check("%s: PY-4 the receptor fragment after protein B keeps its probe pairs" % ff,
          sum(1 for p in fnames if {resname[p[0]], resname[p[2]]} == {"ALA", "PRB"}
              and (p[0] in a3 or p[2] in a3)) == 2, fnames)
    read = [rec[3] for rec in ns["prot1_data"] + ns["prot2_data"]]
    check("%s: PY-4 the solute is every atom before the first water, no NA/CL" % ff,
          sorted(read) == list(range(1, first_sol)), (len(read), first_sol))
    net_b = [p for p in named(atoms, fn) if p[0] in B]
    want_b = [(12, "CA", 14, "CA"), (17, "CA", 19, "CA")] if ff == "amber" \
        else [(10, "CA", 12, "CA"), (14, "CA", 16, "CA")]
    check("%s: side B network %s (PY-1 gromos: before 3, PY-2 charmm: before none)"
          % (ff, ", ".join("%d-%d" % (p[0], p[2]) for p in want_b)), net_b == want_b, net_b)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all FE selection checks passed")
