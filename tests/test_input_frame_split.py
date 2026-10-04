#!/usr/bin/env python3
"""-f must be one piece, and the restraints must not depend on how it is wrapped.

THE FAILURE (review FR1). make_boresch.py took the interface candidates and the
elastic network from raw distances in -f, npt_probe_cluster.gro, which gmx trjconv
-pbc cluster had made whole atom by atom along the sequence. A cutout partner is one
moleculetype of several unbonded fragments, so a sequence step longer than half the
box put every later fragment a box vector off. On the real 1IJK probe ligand
residues 300-416 sat one box vector away: 8076 candidate pairs within 1.2 nm instead
of 19366, all 435 springs on residues 201-299, 720 of 1108 contact pairs without a
spring, 5 network pairs lost, and nothing in the log.

WHAT IS PINNED, on synthetic complexes (two partners of capped ALA strands) run
through the whole script, no GROMACS:

  [1] a frame in one piece runs and says so.
  [2] a ligand fragment one box vector off stops the setup with INPUT_FRAME_SPLIT
      before any mdp gets a pull block, and the message names the residues and the
      box vector.
  [3] so does the whole ligand partner one box vector off, and
  [4] a fragment split INSIDE itself.
  [5] no false alarm: a frame in one piece whose small fragment sits 4.4 nm from the
      root fragment's centroid in an 8 nm box, beyond half its shortest lattice
      vector, runs. A test by fragment centroid would stop it (review R2's 1RLB
      case).
  [6] an ion one box vector off does not stop the setup: it is measured where its
      contacts put it, and every restraint comes out as with the ion in place.
  [7] with the guard switched off in a scratch copy, the split frames of [2] and
      [4] give the restraints of the whole frame, every number of every output to
      1e-6; with the in-memory assembly switched off as well, the minimum-imaged
      distances alone still give the same candidate count, springs and network.
  [8] min_image_cdist is cdist bit for bit wherever a pair vector is its own
      minimum image, and the exact minimum image elsewhere; shortest_lattice_vector
      finds a combination shorter than every box vector.

Fails on the first v1.5 round and on 431510c (no guard: [2]-[4] run on; the split
frames lose springs and network pairs; the helpers of [8] do not exist), passes now.

Standalone, no pytest: python3 tests/test_input_frame_split.py
"""
import atexit, math, os, re, shutil, subprocess, sys, tempfile
import numpy as np
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(ROOT, "utils")
TPL = os.path.join(ROOT, "settings", "amber19sb_opc3")
sys.path.insert(0, UTILS)
import fe_protocol as P

failures = []


def check(name, ok, detail=""):
    print("  %-76s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


# ------------------------------------------------------------------ complexes
def strand(res, x0, y, z, nala, dz):
    """ACE, nala ALA, NME along x from x0: [(resnum, resname, atom, x, y, z)], next resnum."""
    out = [(res, "ACE", "CH3", x0 - 0.36, y, z), (res, "ACE", "C", x0 - 0.12, y, z),
           (res, "ACE", "O", x0 - 0.12, y - 0.12, z)]
    res += 1
    for j in range(nala):
        x = x0 + 0.36 * j
        out += [(res, "ALA", "N", x, y, z), (res, "ALA", "CA", x + 0.12, y + 0.08, z),
                (res, "ALA", "C", x + 0.24, y, z), (res, "ALA", "O", x + 0.24, y - 0.12, z),
                (res, "ALA", "CB", x + 0.12, y + 0.2, z + dz)]
        res += 1
    out += [(res, "NME", "N", x0 + 0.36 * nala, y, z), (res, "NME", "C", x0 + 0.36 * nala + 0.12, y + 0.1, z)]
    return out, res + 1


def complex_a():
    """Receptor: two strands of five ALA 0.55 nm apart, then a Mg2+ ion; ligand: the
    same 0.5 nm above. Returns (atoms, ligand residues, ion residues, box)."""
    atoms, res = [], 1
    for y in (0.0, 0.55):
        a, res = strand(res, 0.0, y, 0.0, 5, 0.1)
        atoms += a
    lig = []
    for y in (0.0, 0.55):
        a, res2 = strand(res, 0.0, y, 0.5, 5, -0.1)
        lig += list(range(res, res2))
        atoms += a
        res = res2
    atoms.append((res, "MG", "MG", 0.9, 0.27, 0.25))
    return atoms, lig, [res], np.diag([12.0, 12.0, 12.0])


def complex_b():
    """Receptor: a root strand of six ALA (residues 1-8 with its caps), a strand of
    three 0.3 nm past its end and another of three 0.3 nm past that (residues
    14-18), whose centroid sits 4.2 nm from the root's; ligand: a strand of five
    0.5 nm above the root. 6.4 nm long in an 8 nm box: 1.6 nm of solvent to the
    image, so every contact is its own minimum image."""
    atoms, res = [], 1
    for x0, n in ((0.0, 6), (2.94, 3), (4.80, 3)):
        a, res = strand(res, x0, 0.0, 0.0, n, 0.1)
        atoms += a
    a, res2 = strand(res, 0.0, 0.0, 0.5, 5, -0.1)
    return atoms + a, list(range(res, res2)), [], np.diag([8.0, 8.0, 8.0])


def write(d, atoms, lig, ions, box, moves=()):
    """syn.gro, chain_map.gs, ion_residues.gs and the templates in d. moves: (set of
    residue numbers, lattice coefficients) applied on the way out, as a wrapped
    frame would have them."""
    shift = {}
    for resset, ijk in moves:
        for r in resset:
            shift[r] = np.asarray(ijk, float) @ box
    with open(os.path.join(d, "syn.gro"), "w") as f:
        f.write("synthetic complex\n%5d\n" % len(atoms))
        for k, (r, rn, nm, x, y, z) in enumerate(atoms, 1):
            v = np.array([x, y, z]) + 4.0 + shift.get(r, 0.0)
            f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (r, rn, nm, k, v[0], v[1], v[2]))
        f.write("%10.5f%10.5f%10.5f\n" % (box[0, 0], box[1, 1], box[2, 2]))
    open(os.path.join(d, "chain_map.gs"), "w").write("".join("%d\n" % r for r in lig))
    open(os.path.join(d, "ion_residues.gs"), "w").write("".join("%d\n" % r for r in ions))
    for m in sorted({l["mdp"] for l in P.legs()}) + ["npt_c_fe.mdp"]:
        shutil.copy(os.path.join(TPL, m), d)


def run(atoms, lig, ions, box, moves=(), utils=UTILS):
    """(dir, exit code, stderr) of make_boresch on the complex, moved as asked."""
    d = tempfile.mkdtemp(prefix="ifsplit_")
    TMP.append(d)
    write(d, atoms, lig, ions, box, moves)
    r = subprocess.run([sys.executable, os.path.join(utils, "make_boresch.py"), "-f", "syn.gro",
                        "-m", "chain_map.gs", "--pull-dist", "%g" % P.PULL_DIST,
                        "--traj", "none.xtc", "--tpr", "none.tpr", "--index", "index.ndx",
                        "--no-readback"], cwd=d, capture_output=True, text=True, timeout=600)
    return d, r.returncode, r.stderr


def marker(d):
    p = os.path.join(d, "boresch_failed.gs")
    return open(p).read().strip() if os.path.isfile(p) else ""


OUTPUTS = sorted({l["mdp"] for l in P.legs()}) + ["npt_c_fe.mdp", "index.ndx",
                                                    "interface_contacts.gs",
                                                    "boresch_anchors.gs", "boresch_analytical.gs"]
NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def same_outputs(a, b, files=OUTPUTS, tol=1e-6):
    """Files that differ between two runs: tokens compared exactly, numbers to tol."""
    bad = []
    for f in files:
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            bad.append(f + " (missing)")
            continue
        ta, tb = open(pa).read().split(), open(pb).read().split()
        if len(ta) != len(tb) or any(
                (abs(float(x) - float(y)) > tol) if (NUM.match(x) and NUM.match(y)) else x != y
                for x, y in zip(ta, tb)):
            bad.append(f)
    return bad


def candidates(err):
    m = re.search(r"(\d+) candidate pairs within", err)
    return int(m.group(1)) if m else None


def network_pairs(d):
    """The network coordinates npt_c_fe.mdp carries, as group pairs with init."""
    kv = {}
    for ln in open(os.path.join(d, "npt_c_fe.mdp")):
        s = ln.split(";")[0]
        if "=" in s:
            k, v = (x.strip() for x in s.split("=", 1))
            kv[k] = v
    g = {i: kv["pull-group%d-name" % i] for i in range(1, int(kv.get("pull-ngroups", 0)) + 1)}
    return sorted((tuple(g[int(x)] for x in kv["pull-coord%d-groups" % c].split()),
                   round(float(kv["pull-coord%d-init" % c]), 6))
                  for c in range(1, int(kv.get("pull-ncoords", 0)) + 1))


def springs(d):
    out = []
    for ln in open(os.path.join(d, "interface_contacts.gs")):
        if ln.strip() and not ln.startswith("#"):
            f = ln.split()
            out.append((int(f[0]), int(f[1]), round(float(f[2]), 6)))
    return sorted(out)


def patched(pairs):
    """utils/ with text replacements in make_boresch.py, in a scratch copy."""
    u = tempfile.mkdtemp(prefix="ifsplit_utils_")
    TMP.append(u)
    for f in ("make_boresch.py", "fe_protocol.py", "pbc_tools.py"):
        if os.path.isfile(os.path.join(UTILS, f)):
            shutil.copy(os.path.join(UTILS, f), u)
    p = os.path.join(u, "make_boresch.py")
    s = open(p).read()
    for a, b in pairs:
        if s.count(a) != 1:
            return None
        s = s.replace(a, b)
    open(p, "w").write(s)
    return u


TMP = []            # every temporary directory, removed at exit even after a crash
atexit.register(lambda: [shutil.rmtree(x, ignore_errors=True) for x in TMP])
A, LIG, IONS, BOX = complex_a()
lig_res = set(LIG)
frag2 = set(range(LIG[0] + 7, LIG[-1] + 1))          # the ligand's second strand
rec_frag1 = set(range(1, 8))                          # the receptor's first strand

# ------------------------------------------------------------------ [1]
print("\n[1] a frame in one piece")
d0, rc0, err0 = run(A, LIG, IONS, BOX)
check("runs (exit %d)" % rc0, rc0 == 0, err0[-400:])
check("and says -f is one piece", "syn.gro is one piece" in err0, err0[:300])

# ------------------------------------------------------------------ [2]-[4]
print("\n[2]-[4] a polymer piece in another image stops the setup")
cases = [("[2] a ligand strand one box vector off", [(frag2, (0, -1, 0))],
          "ligand %d-%d by (0,1,0)" % (min(frag2), max(frag2))),
         ("[3] the whole ligand one box vector off", [(lig_res, (1, 0, 0))],
          "ligand %d-%d by (-1,0,0)" % (min(lig_res), max(lig_res))),
         ("[4] a receptor strand split inside itself", [({5, 6, 7}, (0, 0, 1))],
          "receptor 5-7 by (0,0,-1)")]
split_dirs = {}
for label, moves, named in cases:
    d, rc, err = run(A, LIG, IONS, BOX, moves)
    split_dirs[label] = moves
    check("%s: INPUT_FRAME_SPLIT (%r, exit %d)" % (label, marker(d), rc),
          rc != 0 and marker(d) == "INPUT_FRAME_SPLIT", err[-300:])
    check("    naming %s" % named, named in err, err[-300:])
    check("    before any mdp got a pull block",
          not any("pull-ngroups" in open(os.path.join(d, m)).read() for m in OUTPUTS[:3]))

# ------------------------------------------------------------------ [5]
print("\n[5] no false alarm on a long complex in one piece")
B, LIGB, IONSB, BOXB = complex_b()
xyz = np.array([a[3:] for a in B])
res = np.array([a[0] for a in B])
c_root = xyz[(res >= 1) & (res <= 8)].mean(0)
c_far = xyz[(res >= 14) & (res <= 18)].mean(0)
gap = min(cdist(xyz, xyz + np.array(v, float) * 8.0).min()
          for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)))
print("      far fragment %.2f nm from the root's centroid; half the shortest lattice "
      "vector %.2f nm; %.2f nm of solvent to the nearest image"
      % (np.linalg.norm(c_far - c_root), 4.0, gap))
check("the far fragment is beyond half a box from the root's centroid, the complex "
      "well clear of its image", np.linalg.norm(c_far - c_root) > 4.1 and gap > 1.2)
d5, rc5, err5 = run(B, LIGB, IONSB, BOXB)
check("it runs (exit %d) and says -f is one piece" % rc5,
      rc5 == 0 and "syn.gro is one piece" in err5, err5[-300:])

# ------------------------------------------------------------------ [6]
print("\n[6] an ion in another image is measured where its contacts are")
d6, rc6, err6 = run(A, LIG, IONS, BOX, [(set(IONS), (0, 1, 0))])
check("runs (exit %d), with a note naming it" % rc6,
      rc6 == 0 and re.search(r"1 non-polymer residue\(s\) of syn.gro .*receptor %d by "
                             r"\(0,-1,0\)" % IONS[0], err6) is not None, err6[-400:])
bad = same_outputs(d0, d6)
check("every restraint as with the ion in place (%d files differ)" % len(bad), not bad, str(bad))

# ------------------------------------------------------------------ [7]
print("\n[7] the restraints do not depend on how -f is wrapped")
OFF = [('abort("INPUT_FRAME_SPLIT",',
        '(lambda r, m: sys.stderr.write("make_boresch: SWITCHED OFF %s\\n" % r))("INPUT_FRAME_SPLIT",')]
NOSHIFT = OFF + [("if _moved:\n  sys.stderr.write(\"make_boresch: %d non-polymer",
                  "if False:\n  sys.stderr.write(\"make_boresch: %d non-polymer")]
u_off, u_mi = patched(OFF), patched(NOSHIFT)
check("the guard and the assembly can be switched off in a scratch copy",
      u_off is not None and u_mi is not None)
if u_off and u_mi:
    for label, moves, _ in cases[:1] + cases[2:]:
        d, rc, err = run(A, LIG, IONS, BOX, moves, utils=u_off)
        bad = same_outputs(d0, d)
        check("%s, guard off: every output as for the whole frame (%d differ)"
              % (label[:3], len(bad)), rc == 0 and not bad, "rc %d %s" % (rc, bad))
        d, rc, err = run(A, LIG, IONS, BOX, moves, utils=u_mi)
        check("%s, assembly off too: %s candidates as for the whole frame (%s)"
              % (label[:3], candidates(err), candidates(err0)),
              rc == 0 and candidates(err) == candidates(err0) is not None, err[-200:])
        check("    the same springs with the same references, and the same network",
              rc == 0 and springs(d) == springs(d0) and network_pairs(d) == network_pairs(d0))

# ------------------------------------------------------------------ [8]
print("\n[8] min_image_cdist and shortest_lattice_vector")
src = open(os.path.join(UTILS, "make_boresch.py")).read()


def grab(name):
    i = src.find("\ndef %s(" % name)
    if i < 0:
        return None
    out = []
    for k, ln in enumerate(src[i + 1:].split("\n")):
        if k and ln and not ln[0].isspace() and not ln.startswith("#"):
            break
        out.append(ln)
    return "\n".join(out)


ns = {"np": np, "cdist": cdist}
m = re.search(r"^_IMG = (.*?)\n(?=\S)", src, re.M | re.S)
if m:
    exec(compile("_IMG = " + m.group(1), "_IMG", "exec"), ns)
for fn in ("to_brick", "min_image_dist", "shortest_lattice_vector", "min_image_cdist"):
    code = grab(fn)
    if code:
        exec(compile(code, fn, "exec"), ns)
if all(k in ns for k in ("min_image_cdist", "shortest_lattice_vector")):
    D = 7.0
    dod = np.array([[D, 0, 0], [0, D, 0], [D / 2, D / 2, D * math.sqrt(2) / 2]])
    check("shortest_lattice_vector: dodecahedron 7.0", abs(ns["shortest_lattice_vector"](dod) - 7.0) < 1e-12)
    flat = np.array([[10.0, 0, 0], [5.0, 6.0, 0], [5.0, 3.0, 2.0]])
    got = ns["shortest_lattice_vector"](flat)
    check("shortest_lattice_vector: c - b = (0,-3,2), %.4f, shorter than every box vector"
          % got, abs(got - math.sqrt(13.0)) < 1e-12)
    check("shortest_lattice_vector: 0 for a box with a zero diagonal",
          ns["shortest_lattice_vector"](np.diag([5.0, 0.0, 5.0])) == 0.0)
    rng = np.random.default_rng(11)
    a = rng.uniform(0, 4, (300, 3))                  # a 4 nm cloud in a 7 nm box
    b = rng.uniform(0, 4, (200, 3))
    shift = rng.integers(-2, 3, (200, 3)).astype(float) @ dod
    got = ns["min_image_cdist"](a, b + shift, dod)
    raw = cdist(a, b)
    lat = np.array([(i, j, k) for i in range(-3, 4) for j in range(-3, 4)
                    for k in range(-3, 4)], float) @ dod
    want = np.array([[np.min(np.linalg.norm(b[j] + shift[j] + lat - a[i], axis=1))
                      for j in range(len(b))] for i in range(len(a))])
    check("every distance is the minimum image although b sits up to two lattice "
          "vectors away (worst %.1e nm)" % np.abs(got - want).max(), np.abs(got - want).max() < 1e-9)
    short = raw <= 3.5
    check("and, unshifted, cdist bit for bit on the %d pairs shorter than half the "
          "shortest lattice vector" % short.sum(),
          np.array_equal(ns["min_image_cdist"](a, b, dod)[short], raw[short]))
else:
    check("make_boresch.py defines min_image_cdist and shortest_lattice_vector", False, "absent")

for x in TMP:
    shutil.rmtree(x, ignore_errors=True)
print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all input-frame checks passed")
