#!/usr/bin/env python3
"""The Boresch reference must be the geometry GROMACS evaluates, whatever images
the frames put the atoms in, and a reference that is not a pose must never leave
setup.

THE FAILURE. 1AKJ, 1E6J and 1IQD of the second HADDOCK pass left setup with Boresch
references 103-115 degrees from any pose the bound complex adopts (1AKJ phi_A:
snapshot -118.4, emitted 138.7, production -113 +- 4). Stage I then dissipated
172-449 kJ/mol per direction against a median of 18.8, and their pKD intervals
reached +-25. ensemble_boresch_geometry took raw COMs of -pbc whole frames and
minimum-imaged only P3->L1, carrying P2, P1, L2 and L3 along as raw offsets. A
cutout made of unbonded fragments comes out of -pbc whole with groups and triads
split across images, and every such frame entered the mean as a different pose.
The readback compared GROMACS with the snapshot only, so the emitted values were
printed and never checked.

WHAT IS PINNED, each on synthetic data with fixed numbers:

  [1] min_image_vec finds the minimum image of a vector that is up to two lattice
      vectors from it, in the xy-square dodecahedron these setups use (the raw
      27-neighbour search of v1.4 does not), and agrees with an exhaustive search
      from the brick in five GROMACS box shapes.
  [2] gromacs_com puts a group back together about its pbcatom.
  [3] the SHIPPED ensemble_boresch_geometry, fed frames in which a group, a triad,
      a whole partner and one group two lattice vectors away are split across the
      boundary of a dodecahedron box, returns the reference of the unsplit frames
      to 1e-3 deg and 1e-5 nm. The trajectory reader is stubbed; nothing else is.
  [4] the snapshot geometry is taken the same way.
  [5] the pose gate fires just above 15 deg (circular, also across +-180) and
      0.12 nm and stays silent at and just below both.
  [6] a gate that fires emits the snapshot for ALL six coordinates.
  [7] the readback asks three questions of what GROMACS read and measured on -f:
      the CONVENTION (GROMACS against the snapshot, 0.5 deg / 0.005 nm); WRITTEN,
      whether GROMACS's reference column, i.e. the pull-coordN-init the leg
      carries, is the reference this setup computed (to pullx's six significant
      digits); and the POSE, whether what the leg carries lies within the gate's
      limits of what GROMACS MEASURES. The pose is asked of GROMACS's measurement,
      not of the snapshot: with GROMACS 0.4 deg from the snapshot, a reference
      15.3 deg from the snapshot on the other side fails and one 15.7 deg away on
      the same side passes (review R5; a pose check against the snapshot passed
      the first round's test). A writer that swaps theta_A and theta_B fails
      WRITTEN however close the two are to a pose (review R3).

Fails on 431510c ([1] and [3] behaviourally, the rest because the functions do not
exist), [7] fails on the first v1.5 round, and all passes now.

Standalone, no pytest: python3 tests/test_boresch_pbc.py
"""
import math, os, re, shutil, sys, tempfile
from types import SimpleNamespace
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "utils", "make_boresch.py")).read()
failures = []


def check(name, ok, detail=""):
    print("  %-72s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def grab(name):
    """A top-level def of make_boresch.py, up to the next top-level statement that
    is not a comment, or None if the file has no such def."""
    i = src.find("\ndef %s(" % name)
    if i < 0:
        return None
    out = []
    for k, ln in enumerate(src[i + 1:].split("\n")):
        if k and ln and not ln[0].isspace() and not ln.startswith("#"):
            break
        out.append(ln)
    return "\n".join(out)


def const(name):
    """A top-level assignment, possibly over several lines, or None."""
    m = re.search(r"^%s = (.*?)\n(?=\S)" % name, src, re.M | re.S)
    return m.group(1) if m else None


ns = {"np": np, "math": math, "sys": sys, "os": os}
for name in ("_IMG", "BORESCH_ROLES", "BORESCH_VECTORS", "BORESCH_KEYS",
             "BORESCH_NAMES", "POSE_GATE_DEG", "POSE_GATE_NM", "CHECK_TOL_DEG",
             "CHECK_TOL_NM", "READ_TOL_DEG", "READ_TOL_NM"):
    c = const(name)
    if c is not None:
        exec(compile("%s = %s" % (name, c), name, "exec"), ns)
for fn in ("to_brick", "min_image_vec", "nearest_image", "angle_deg", "dihedral_deg",
           "circular_mean_deg", "wrap180", "group_weights", "pbcatom_of",
           "gromacs_com", "boresch_series", "role_slices", "group_coms",
           "snapshot_boresch_geometry", "pose_gate", "choose_reference",
           "readback_verdict", "ensemble_boresch_geometry"):
    code = grab(fn)
    if code is not None:
        exec(compile(code, fn, "exec"), ns)


def have(*names):
    missing = [n for n in names if n not in ns]
    if missing:
        check("make_boresch.py defines %s" % ", ".join(missing), False, "absent")
    return not missing


def ln(v):
    return np.linalg.norm(v, axis=-1)


def dodecahedron(d):
    """The xy-square rhombic dodecahedron editconf builds, box vectors as rows."""
    return np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [d / 2, d / 2, d * math.sqrt(2) / 2]])


def exhaustive(dv, M, reach=3):
    """Minimum image by brute force over +-reach lattice vectors from the brick:
    the reference [1] is checked against, written here and not taken from the
    code under test."""
    for d in (2, 1, 0):
        dv = dv - np.round(dv[..., d] / M[d, d])[..., None] * M[d]
    rng_ = range(-reach, reach + 1)
    img = np.array([(i, j, k) for i in rng_ for j in rng_ for k in rng_], float) @ M
    cand = dv[..., None, :] + img
    k = np.argmin((cand ** 2).sum(-1), axis=-1)
    return np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]


# ------------------------------------------------------------------ [1]
print("\n[1] minimum image: right for vectors two lattice vectors away")
if have("min_image_vec", "_IMG"):
    rng = np.random.default_rng(2026)
    M = dodecahedron(7.0)
    lim = 0.45 * 7.0
    t = rng.normal(size=(20000, 3)); t /= ln(t)[:, None]
    t *= (lim * rng.uniform(0, 1, 20000) ** (1 / 3))[:, None]
    for nmax, label in ((1, "one"), (2, "two")):
        n = rng.integers(-nmax, nmax + 1, size=(20000, 3)).astype(float)
        got = ns["min_image_vec"](t + n @ M, M)
        bad = int(np.sum(ln(got - t) > 1e-9))
        check("up to %s lattice vector(s) per axis: %d of 20000 wrong" % (label, bad),
              bad == 0, str(bad))
    boxes = {"dodecahedron xy-square": dodecahedron(10.0),
             "dodecahedron xy-hexagon": np.array([[10.0, 0, 0], [5.0, 10 * math.sqrt(3) / 2, 0],
                                                  [5.0, 10 * math.sqrt(3) / 6, 10 * math.sqrt(6) / 3]]),
             "truncated octahedron": np.array([[10.0, 0, 0], [10 / 3, 20 * math.sqrt(2) / 3, 0],
                                               [-10 / 3, 10 * math.sqrt(2) / 3, 10 * math.sqrt(6) / 3]]),
             "rectangular": np.diag([10.0, 8.0, 13.0]),
             # skewed, though not the flattest box grompp accepts: in flatter ones
             # the 27-neighbour search can miss (review R7, see min_image_vec)
             "skewed [[10,0,0],[5,6,0],[-5,3,5]]": np.array([[10.0, 0, 0], [5.0, 6.0, 0], [-5.0, 3.0, 5.0]])}
    for name, M in boxes.items():
        dv = rng.uniform(-30, 30, size=(20000, 3))
        bad = int(np.sum(np.abs(ln(ns["min_image_vec"](dv, M)) - ln(exhaustive(dv, M))) > 1e-9))
        check("%s: agrees with the +-3 search on 20000 vectors to 30 nm (%d off)"
              % (name, bad), bad == 0, str(bad))
    # a (frames, n, 3) array against (frames, 1, 3, 3) boxes, the shape the callers use
    Mf = np.stack([dodecahedron(7.0), dodecahedron(7.3)])[:, None]
    v = np.array([[[0.3, -0.2, 0.1], [14.3, 7.0, 0.0]], [[0.3, -0.2, 0.1], [-6.9, 0.1, 0.2]]])
    got = ns["min_image_vec"](v, Mf)
    check("broadcasts over frames and atoms", got.shape == (2, 2, 3)
          and np.allclose(got[0, 1], [0.3, 0.0, 0.0]) and np.allclose(got[1, 1], [0.4, 0.1, 0.2]),
          str(got))

# ------------------------------------------------------------------ the pose
# Six anchor groups of a bound pose, each a few residues of N/CA/C. Atom numbers
# are 1-based and run P1..L3, the order the trajectory reader returns them in.
MASS = {"N": 14.007, "CA": 12.011, "C": 12.011}
ns["BACKBONE"] = ("N", "CA", "C")
ns["mass_of"] = lambda a, nm: MASS[nm]
CENTRES = {"P1": (-1.0, 0.8, -0.6), "P2": (-0.3, 1.0, 0.2), "P3": (0.0, 0.0, 0.0),
           "L1": (1.1, 0.1, 0.0), "L2": (1.5, -0.9, 0.3), "L3": (2.2, -0.2, -0.7)}
NRES = {"P1": 4, "P2": 5, "P3": 6, "L1": 4, "L2": 3, "L3": 5}
rng = np.random.default_rng(7)
groups, xyz0, anum = {}, {}, 1
for role in ("P1", "P2", "P3", "L1", "L2", "L3"):
    n = 3 * NRES[role]
    atoms = list(range(anum, anum + n)); anum += n
    groups[role] = {"atoms": atoms, "atomnames": ["N", "CA", "C"] * NRES[role]}
    for a in atoms:
        xyz0[a] = np.array(CENTRES[role]) + rng.uniform(-0.3, 0.3, 3) + np.array([3.0, 3.0, 2.0])
ns["prot1_data"] = [("ALA%d" % a, a, nm, a) + tuple(xyz0[a]) for r in ("P1", "P2", "P3")
                    for a, nm in zip(groups[r]["atoms"], groups[r]["atomnames"])]
ns["prot2_data"] = [("ALA%d" % a, a, nm, a) + tuple(xyz0[a]) for r in ("L1", "L2", "L3")
                    for a, nm in zip(groups[r]["atoms"], groups[r]["atomnames"])]
ns["XYZ_OF"] = dict(xyz0)
ORDER = [a for r in ("P1", "P2", "P3", "L1", "L2", "L3") for a in groups[r]["atoms"]]
slot = {a: i for i, a in enumerate(ORDER)}
BOX = dodecahedron(7.0)
ns["INPUT_ORDER"] = {rec[3]: k for k, rec in enumerate(ns["prot1_data"] + ns["prot2_data"])}
ns["INPUT_BOX"] = BOX


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


# 30 frames: the whole complex rotated and moved, each atom jiggled, so the
# frames are different poses of one bound complex and none is split.
NF = 30
whole = np.empty((NF, len(ORDER), 3))
base = np.array([xyz0[a] for a in ORDER])
for f in range(NF):
    R = rot(rng.normal(size=3), rng.uniform(0, 0.15))
    whole[f] = (base - base.mean(0)) @ R.T + base.mean(0) + rng.normal(0, 0.15, 3) \
        + rng.normal(0, 0.02, base.shape)


def lattice(i, j, k):
    return i * BOX[0] + j * BOX[1] + k * BOX[2]


# The same frames as -pbc whole can hand them over for a cutout: a group split
# atom by atom, a whole partner in another image, one ligand group (a separate
# fragment) shifted against its own triad, and one receptor group TWO lattice
# vectors away, which a search from the raw difference cannot reach.
split = whole.copy()
P2 = [slot[a] for a in groups["P2"]["atoms"]]
LIG = [slot[a] for r in ("L1", "L2", "L3") for a in groups[r]["atoms"]]
L3 = [slot[a] for a in groups["L3"]["atoms"]]
P1 = [slot[a] for a in groups["P1"]["atoms"]]
for f in range(NF):
    for q, i in enumerate(P2):
        if (q + f) % 3 == 0:
            split[f, i] += lattice(1, 0, 0) if q % 2 else lattice(0, -1, 1)
    if f % 2:
        split[f, LIG] += lattice(0, 1, -1)
    if f % 3 == 1:
        split[f, L3] += lattice(-1, 0, 0)
    if f % 4 == 2:
        split[f, P1] += lattice(2, 0, 0)

# ------------------------------------------------------------------ [2]
print("\n[2] a group's COM is taken about its pbcatom, as GROMACS takes it")
if have("gromacs_com", "pbcatom_of"):
    w = np.array([MASS[nm] for nm in groups["P2"]["atomnames"]])
    pb = groups["P2"]["atoms"].index(ns["pbcatom_of"](groups["P2"]["atoms"]))
    Mf = np.repeat(BOX[None], NF, axis=0)
    c_whole = ns["gromacs_com"](whole[:, P2, :], Mf, w, pb)
    c_split = ns["gromacs_com"](split[:, P2, :], Mf, w, pb)
    raw = (split[:, P2, :] * w[None, :, None]).sum(1) / w.sum()
    d = ln(ns["min_image_vec"](c_split - c_whole, Mf))
    d_raw = ln(ns["min_image_vec"](raw - c_whole, Mf))
    check("split group: COM within 1e-9 nm of the whole group's (worst %.1e)"
          % d.max(), d.max() < 1e-9, "%.3g" % d.max())
    check("  where the plain mean lands up to %.2f nm away" % d_raw.max(), d_raw.max() > 1.0)
    # The pbcatom itself, which every mdp is written with, must not depend on how -f
    # is wrapped either: the first round took the members' raw centre, which a split
    # group puts between the images.
    whole_pb = {r: ns["pbcatom_of"](groups[r]["atoms"]) for r in groups}
    keep = (ns["XYZ_OF"], ns["prot1_data"], ns["prot2_data"])
    sx = dict(xyz0)
    for r in groups:
        for q, a in enumerate(groups[r]["atoms"]):
            if q % 2:
                sx[a] = sx[a] + lattice(1, -1, 0) * (1 if r < "M" else -1)
    ns["XYZ_OF"] = sx
    ns["prot1_data"] = [rec[:4] + tuple(sx[rec[3]]) for rec in keep[1]]
    ns["prot2_data"] = [rec[:4] + tuple(sx[rec[3]]) for rec in keep[2]]
    split_pb = {r: ns["pbcatom_of"](groups[r]["atoms"]) for r in groups}
    ns["XYZ_OF"], ns["prot1_data"], ns["prot2_data"] = keep
    check("every group's pbcatom is the same atom when -f splits each group",
          split_pb == whole_pb, "%s vs %s" % (split_pb, whole_pb))

# ------------------------------------------------------------------ [3]
print("\n[3] the shipped ensemble_boresch_geometry on split frames")
if have("ensemble_boresch_geometry"):
    feed = {}

    def read_multi_gro(path, natoms, with_box=False):
        X = feed["X"]
        assert natoms == X.shape[1], (natoms, X.shape)
        return (X, np.repeat(BOX[None], len(X), axis=0)) if with_box else X

    ns.update(read_multi_gro=read_multi_gro,
              subprocess=SimpleNamespace(run=lambda *a, **k: SimpleNamespace(returncode=0, stderr="")),
              trjconv_failed=lambda r, traj, gro, what: False,
              write_ndx_group=lambda path, name, atoms: None,
              TRAJ_PAIRS=[("probe.xtc", "probe.tpr")], TRAJ_SKIP_PS=0.0, MIN_FRAMES=20)
    here = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="borpbc_")
    os.chdir(tmp)
    try:
        feed["X"] = whole
        ref_w, sd_w, n_w = ns["ensemble_boresch_geometry"](groups)
        feed["X"] = split
        ref_s, sd_s, n_s = ns["ensemble_boresch_geometry"](groups)
    finally:
        os.chdir(here)
        shutil.rmtree(tmp, ignore_errors=True)
    print("      whole: r %.5f thA %.3f thB %.3f phA %.3f phB %.3f phC %.3f"
          % tuple(ref_w[k] for k in ("r", "thA", "thB", "phA", "phB", "phC")))
    print("      split: r %.5f thA %.3f thB %.3f phA %.3f phB %.3f phC %.3f"
          % tuple(ref_s[k] for k in ("r", "thA", "thB", "phA", "phB", "phC")))
    check("both read %d frames" % NF, n_w == n_s == NF, "%s %s" % (n_w, n_s))
    dr = abs(ref_s["r"] - ref_w["r"])
    check("r of the split frames equals the whole frames' to 1e-5 nm (%.1e)" % dr,
          dr <= 1e-5, "%.3g" % dr)
    for k in ("thA", "thB", "phA", "phB", "phC"):
        dd = abs(((ref_s[k] - ref_w[k]) + 180.0) % 360.0 - 180.0)
        check("%s of the split frames equals the whole frames' to 1e-3 deg (%.1e)" % (k, dd),
              dd <= 1e-3, "%.3g" % dd)
    # and the whole-frame reference is the pose itself, not just self-consistent
    A, D = ns["angle_deg"], ns["dihedral_deg"]
    c = {r: np.mean([xyz0[a] * MASS[nm] for a, nm in zip(groups[r]["atoms"], groups[r]["atomnames"])], 0)
         / np.mean([MASS[nm] for nm in groups[r]["atomnames"]]) for r in groups}
    pose = {"r": np.linalg.norm(c["L1"] - c["P3"]), "thA": A(c["P2"], c["P3"], c["L1"]),
            "thB": A(c["P3"], c["L1"], c["L2"]), "phA": D(c["P1"], c["P2"], c["P3"], c["L1"]),
            "phB": D(c["P2"], c["P3"], c["L1"], c["L2"]), "phC": D(c["P3"], c["L1"], c["L2"], c["L3"])}
    off = max(abs(((ref_w[k] - pose[k]) + 180) % 360 - 180) for k in ("thA", "thB", "phA", "phB", "phC"))
    check("and the reference is the bound pose (within %.1f deg of the unrotated one)" % off,
          off < 5.0, "%.2f" % off)

# ------------------------------------------------------------------ [4]
print("\n[4] the snapshot geometry is taken the same way")
if have("snapshot_boresch_geometry", "boresch_series"):
    ns["INPUT_BOX"] = BOX
    snap_w, _ = ns["snapshot_boresch_geometry"](groups)
    shifted = dict(xyz0)
    for q, a in enumerate(groups["P2"]["atoms"] + groups["L3"]["atoms"]):
        if q % 2:
            shifted[a] = shifted[a] + lattice(1, -1, 1)
    for a in groups["L1"]["atoms"]:
        shifted[a] = shifted[a] + lattice(-2, 0, 0)
    ns["XYZ_OF"] = shifted
    snap_s, _ = ns["snapshot_boresch_geometry"](groups)
    ns["XYZ_OF"] = dict(xyz0)
    worst = max(abs(snap_s["r"] - snap_w["r"]) * 1e3,
                *[abs(((snap_s[k] - snap_w[k]) + 180) % 360 - 180)
                  for k in ("thA", "thB", "phA", "phB", "phC")])
    check("a snapshot with split groups gives the unsplit geometry (worst %.1e)" % worst,
          worst < 1e-6, "%.3g" % worst)
    off = max(abs(((snap_w[k] - pose[k]) + 180) % 360 - 180) for k in ("thA", "thB", "phA", "phB", "phC"))
    check("and it is the pose's own geometry (%.1e deg)" % off, off < 1e-6, "%.3g" % off)

# ------------------------------------------------------------------ [5]
print("\n[5] the pose gate: 15 deg circular, 0.12 nm, from both sides")
if have("pose_gate"):
    S = {"r": 0.5, "thA": 100.0, "thB": 60.0, "phA": 175.0, "phB": -175.0, "phC": 0.0}

    def fired(**chg):
        e = dict(S); e.update(chg)
        return [b[0] for b in ns["pose_gate"](e, S)]

    check("identical: silent", fired() == [])
    for k, below, at, above in (("thA", 114.99, 115.0, 115.01),
                                ("thB", 45.01, 45.0, 44.99),
                                ("phC", -14.99, -15.0, -15.01)):
        check("%s %.2f deg off: silent" % (k, abs(below - S[k])), fired(**{k: below}) == [])
        check("%s exactly 15 deg off: silent" % k, fired(**{k: at}) == [])
        check("%s %.2f deg off: fires on %s only" % (k, abs(above - S[k]), k),
              fired(**{k: above}) == [k], str(fired(**{k: above})))
    # across the wrap: 175 -> -170 is 15 deg, not 345
    check("phA 175 -> -170.01 (14.99 deg across +-180): silent", fired(phA=-170.01) == [])
    check("phA 175 -> -170.00 (15 deg across +-180): silent", fired(phA=-170.0) == [])
    check("phA 175 -> -169.99 (15.01 deg across +-180): fires", fired(phA=-169.99) == ["phA"])
    check("phB -175 -> 170.01 (14.99 deg across +-180): silent", fired(phB=170.01) == [])
    check("phB -175 -> 169.99 (15.01 deg across +-180): fires", fired(phB=169.99) == ["phB"])
    check("r 0.5 -> 0.6199 nm: silent", fired(r=0.6199) == [])
    check("r 0.5 -> 0.62 nm (exactly 0.12): silent", fired(r=0.62) == [])
    check("r 0.5 -> 0.6201 nm: fires", fired(r=0.6201) == ["r"])
    check("r 0.5 -> 0.3801 nm: silent", fired(r=0.3801) == [])
    check("r 0.5 -> 0.3799 nm: fires", fired(r=0.3799) == ["r"])
    check("1AKJ's phi_A (snapshot -118.4, emitted 138.7) fires",
          [b[0] for b in ns["pose_gate"](dict(S, phA=138.7), dict(S, phA=-118.4))] == ["phA"])

# ------------------------------------------------------------------ [6]
print("\n[6] a gate that fires emits the snapshot for all six coordinates")
if have("choose_reference"):
    S = {"r": 1.10, "thA": 75.6, "thB": 96.4, "phA": -118.4, "phB": 139.5, "phC": -97.0}
    E = {"r": 1.09, "thA": 74.0, "thB": 98.6, "phA": 138.7, "phB": -116.0, "phC": -99.0}
    ref, bad = ns["choose_reference"](E, S)
    check("1AKJ-like mean: the snapshot is emitted, all six of it", ref == S, str(ref))
    check("  and the two offending dihedrals are named", [b[0] for b in bad] == ["phA", "phB"],
          str(bad))
    E2 = dict(S, r=1.09, thA=74.0, phC=-99.0)
    ref2, bad2 = ns["choose_reference"](E2, S)
    check("a mean within the gate is emitted as it is", ref2 == E2 and bad2 == [], str(ref2))

# ------------------------------------------------------------------ [7]
print("\n[7] the readback: convention, what the legs carry, and the pose against GROMACS")
if have("readback_verdict", "BORESCH_KEYS"):
    K = ns["BORESCH_KEYS"]
    S = {"r": 1.1122, "thA": 75.6032, "thB": 96.3657, "phA": -118.4, "phB": 139.5, "phC": -96.95}

    def verdict(got=None, carried=None, intended=None):
        """readback_verdict with GROMACS reproducing the snapshot unless told
        otherwise, and the legs carrying exactly what was intended."""
        got = dict(S, **(got or {}))
        intended = dict(S, **(intended or {}))
        carried = dict(intended, **(carried or {}))
        _, conv, init, pose = ns["readback_verdict"]([got[k] for k in K],
                                                     [carried[k] for k in K], S, intended)
        return conv, init, pose

    conv, init, pose_ = verdict(intended=dict(phA=138.7, phB=-116.0))
    check("1AKJ's phi_A/phi_B: two pose failures, no convention or written failure",
          len(pose_) == 2 and not conv and not init and pose_[0].startswith("phi_A"),
          str((conv, init, pose_)))
    for off, n in ((15.4, 0), (15.6, 1)):
        _, _, p = verdict(intended=dict(phA=-118.4 + off))
        check("carried %.1f deg from GROMACS (limit 15 + 0.5): %d pose failure(s)" % (off, n),
              len(p) == n, str(p))
    for off, n in ((0.124, 0), (0.126, 1)):
        _, _, p = verdict(intended=dict(r=1.1122 + off))
        check("carried r %.3f nm from GROMACS (limit 0.12 + 0.005): %d pose failure(s)"
              % (off, n), len(p) == n, str(p))
    # R5: GROMACS 0.4 deg (0.004 nm) from the snapshot, inside the convention
    # tolerance, so measuring the pose from the snapshot instead would flip both.
    c, _, p = verdict(got=dict(phA=-118.4 + 0.4), intended=dict(phA=-118.4 - 15.3))
    check("GROMACS +0.4 deg off the snapshot, carried -15.3 deg off it: 15.7 deg from "
          "GROMACS, a pose failure", len(p) == 1 and not c, str((c, p)))
    c, _, p = verdict(got=dict(phA=-118.4 + 0.4), intended=dict(phA=-118.4 + 15.7))
    check("GROMACS +0.4 deg off the snapshot, carried +15.7 deg off it: 15.3 deg from "
          "GROMACS, passes", not p and not c, str((c, p)))
    c, _, p = verdict(got=dict(r=1.1122 + 0.004), intended=dict(r=1.1122 - 0.122))
    check("GROMACS r +0.004 nm, carried -0.122 nm: 0.126 nm from GROMACS, a pose failure",
          len(p) == 1 and not c, str((c, p)))
    c, _, p = verdict(got=dict(r=1.1122 + 0.004), intended=dict(r=1.1122 + 0.127))
    check("GROMACS r +0.004 nm, carried +0.127 nm: 0.123 nm from GROMACS, passes",
          not p and not c, str((c, p)))
    c, i, p = verdict(got=dict(thB=96.3657 + 0.6))
    check("GROMACS 0.6 deg from the snapshot: a convention failure only",
          len(c) == 1 and not i and not p, str((c, i, p)))
    # WRITTEN: what GROMACS read from the mdp against what this setup computed.
    # A swap of two angles 10 deg apart is still a pose and only this catches it.
    g2 = dict(thA=80.0, thB=90.0)
    _, i, p = verdict(got=g2, intended=g2, carried=dict(thA=90.0, thB=80.0))
    check("theta_A and theta_B swapped in the mdp, 10 deg apart: two written "
          "failures, no pose failure", len(i) == 2 and not p, str((i, p)))
    _, i, p = verdict(carried=dict(phA=118.4))
    check("phi_A written mirrored: a written failure, and a pose failure, since what "
          "the legs carry is no pose whatever was intended",
          len(i) == 1 and "phi_A" in i[0] and len(p) == 1 and "phi_A" in p[0], str((i, p)))
    for d, n in ((0.0019, 0), (0.0021, 1)):
        _, i, _ = verdict(carried=dict(phB=139.5 + d))
        check("carried %.4f deg off the reference (pullx precision 0.002): %d written "
              "failure(s)" % (d, n), len(i) == n, str(i))
    for d, n in ((1.9e-5, 0), (2.1e-5, 1)):
        _, i, _ = verdict(carried=dict(r=1.1122 + d))
        check("carried r %.1e nm off the reference (0.00002): %d written failure(s)"
              % (d, n), len(i) == n, str(i))
    rb = src[src.index("_read = verify_pull_block("):]
    check("the module aborts on each failure with its own code",
          'abort("PULL_READBACK_MISMATCH"' in rb and 'abort("PULL_INIT_MISMATCH"' in rb
          and 'abort("BORESCH_REFERENCE_NOT_A_POSE"' in rb)
    check("and GROMACS's measured and reference columns are both handed over",
          "readback_verdict(got, carried, SNAP_REF, _first)" in rb)
    blk = src[src.index("_ref, _gate = choose_reference(_ens, SNAP_REF)"):]
    blk = blk[:blk.index("\nelse:")]
    check("and the emitted references come from choose_reference",
          'ref_r, ref_thA, ref_thB = _ref["r"], _ref["thA"], _ref["thB"]' in blk
          and 'ref_phA, ref_phB, ref_phC = _ref["phA"], _ref["phB"], _ref["phC"]' in blk)

# ------------------------------------------------------------------ [8]
print("\n[8] verify_pull_block reads GROMACS's value AND reference columns by legend")
for fn in ("verify_pull_block",):
    code = grab(fn)
    if code is not None:
        exec(compile(code, fn, "exec"), ns)
if have("verify_pull_block"):
    seen = {}
    # pullx as GROMACS 2026 writes it with pull-print-ref-value = yes: "N" and "N ref"
    # per coordinate; coordinate 3 here has no reference column.
    XVG = ('# GROMACS pullx\n@    title "Pull COM"\n@ s0 legend "1"\n@ s1 legend "1 ref"\n'
           '@ s2 legend "2"\n@ s3 legend "2 ref"\n@ s4 legend "3"\n'
           '0.0000 1.11224 1.09083 75.6032 73.9837 146.51\n')

    def fake_run(cmd, **kw):
        if cmd[:2] == ["gmx", "grompp"]:
            seen["mdp"] = open(cmd[cmd.index("-f") + 1]).read()
        elif cmd[:2] == ["gmx", "mdrun"]:
            open(cmd[cmd.index("-deffnm") + 1] + "_pullx.xvg", "w").write(XVG)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    here = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="borpbc_")
    os.chdir(tmp)
    try:
        open("leg.mdp", "w").write("nsteps = 5000\npull-print-ref-value = no\npull = yes\n")
        open("topol.top", "w").write("; stub\n")
        ns.update(args=SimpleNamespace(readback=True, topol="topol.top", input="f.gro",
                                       readback_index="i.ndx"),
                  subprocess=SimpleNamespace(run=fake_run), re=re,
                  shutil=SimpleNamespace(which=lambda x: "/usr/bin/gmx", rmtree=shutil.rmtree))
        got = ns["verify_pull_block"]("leg.mdp")
    finally:
        os.chdir(here)
        shutil.rmtree(tmp, ignore_errors=True)
    check("coordinate N's value from legend 'N' and its reference from 'N ref'",
          got == {1: (1.11224, 1.09083), 2: (75.6032, 73.9837), 3: (146.51, None)}, str(got))
    m = seen.get("mdp", "")
    check("the probe asks for the reference column, once, and keeps nsteps out",
          m.count("pull-print-ref-value") == 1 and "pull-print-ref-value = yes" in m
          and "nsteps = 0" in m and "nsteps = 5000" not in m, m[-200:])

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all Boresch PBC checks passed")
