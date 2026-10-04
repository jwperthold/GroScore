#!/usr/bin/env python3
"""A fragmented side must be put back together, by contact, before its rigidity is
measured.

THE FAILURE. A cutout side is a merge of capped, unbonded fragments, and trjconv
-pbc whole leaves each fragment in the image mdrun wrapped it into. The per-side
self-fit in make_boresch.py then reads a fragment a box vector away as motion: in 40
of the 46 setups of the second HADDOCK pass the backbone RMSF came out at 2.9-6.9 nm
against the 1.0 nm guard, the measured anchor search was abandoned, and the burial
heuristic, which carries no fluctuation information, anchored every one of them
(digest setup.txt S7). The first v1.5 repair placed each fragment by its centroid's
offset from the root's, which is ambiguous once that offset nears half a box: a
rigid side with a fragment 3.6 nm from the root in the real 2KTF probe box came out
non-physical in 198-200 of 200 frames and fell back to burial (review R2). Its test
could not tell (review R6): three strands within 0.6 nm of each other, no fragment
split inside -f, and a drift-only reassembly passed it.

WHAT IS PINNED, on synthetic sides in xy-square dodecahedra, over 60 frames that
TUMBLE through up to 180 degrees, drift by up to 3 nm and put fragments in other
images, up to two lattice vectors away:

  [1] backbone_fragments finds the pieces, largest first, also when -f has one of
      them in another image and when -f splits one INSIDE itself across the
      boundary, which raw neighbour gaps would cut in two.
  [2] reassemble_backbone moves fragments by whole box vectors only, one vector per
      fragment per frame, and gives back the whole frames up to one lattice vector
      per frame, so the self-fit RMSF is the noise put in (0.05 nm per axis), not
      box-sized. The side's far fragment sits 3.4 nm from the root's centroid in a
      6.0 nm box, beyond half its shortest lattice vector (3.0 nm), where a
      placement by centroid offset picks the wrong image.
  [3] the per-side count of frames that needed putting together is exact.
  [4] whole frames come back bit for bit, and a side of one fragment unchanged.
  [5] a fragment that really drifts away from the rest is NOT hidden: its RMSF still
      trips the 1.0 nm guard.
  [6] both partners go into one tree: a ligand fragment that touches only the
      receptor, 4.2 nm from the ligand's bulk in an 8.0 nm box (half its shortest
      lattice vector 4.0 nm), is placed by that contact.
  [7] try_measured_anchors reassembles the backbone before measure_rigidity,
      against -f in -f's own box.

Fails on the first v1.5 round (its reassemble_side places by centroid, and the
functions tested here do not exist there) and on 431510c, and passes now.

Standalone, no pytest: python3 tests/test_anchor_fragments.py
"""
import math, os, re, sys
import numpy as np
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "utils", "make_boresch.py")).read()
failures = []


def check(name, ok, detail=""):
    print("  %-76s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


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


ns = {"np": np, "math": math, "cdist": cdist}
for name in ("_IMG", "FRAG_BOND_NM", "RMSF_SPLIT"):
    m = re.search(r"^%s = (.*?)\n(?=\S)" % name, src, re.M | re.S)
    if m:
        exec(compile("%s = %s" % (name, m.group(1)), name, "exec"), ns)
for fn in ("to_brick", "min_image_vec", "min_image_dist", "shortest_lattice_vector",
           "min_image_cdist", "backbone_fragments", "contact_links", "reassemble_backbone"):
    code = grab(fn)
    if code is not None:
        exec(compile(code, fn, "exec"), ns)
have = all(n in ns for n in ("backbone_fragments", "reassemble_backbone", "contact_links",
                             "FRAG_BOND_NM"))
check("make_boresch.py defines backbone_fragments, contact_links, reassemble_backbone",
      have, "absent")
RMSF_SPLIT = ns.get("RMSF_SPLIT", 1.0)
LAT = [(1, 0, 0), (0, -1, 1), (-1, 1, 0), (2, 0, 0), (0, 0, -1), (-2, 1, 1), (1, 1, -1)]


def dodecahedron(d):
    return np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [d / 2, d / 2, d * math.sqrt(2) / 2]])


def lattice(box, ijk):
    return np.asarray(ijk, float) @ box


def strand(nres, origin, axis=(1, 0, 0), up=(0, 0, 1)):
    """N, CA, C of nres residues along axis: N-CA 0.146, CA-C 0.152, C-N 0.133 nm,
    the CA raised along up so the chain zigzags like a strand."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    up = np.asarray(up, float) / np.linalg.norm(up)
    out, p = [], np.asarray(origin, float)
    for _ in range(nres):
        n = p
        ca = n + 0.110 * axis + 0.096 * up       # |.| = 0.146
        c = ca + 0.120 * axis - 0.093 * up       # |.| = 0.152
        out += [n, ca, c]
        p = c + 0.133 * axis                     # next N
    return out


def meander(nres, nstrands, origin):
    """ONE continuous backbone folded into nstrands antiparallel strands of nres
    residues, each turn a residue running along +y, so the strands lie 0.564 nm
    apart and every neighbour in the chain is bonded."""
    out, p = [], np.asarray(origin, float)
    for s in range(nstrands):
        seg = strand(nres, p, (1, 0, 0) if s % 2 == 0 else (-1, 0, 0))
        out += seg
        if s < nstrands - 1:
            n = seg[-1] + np.array([0.0, 0.133, 0.0])
            ca = n + np.array([0.0, 0.146, 0.0])
            c = ca + np.array([0.0, 0.152, 0.0])
            out += [n, ca, c]
            p = c + np.array([0.0, 0.133, 0.0])
    return out


def build(pieces, shift=(1.0, 1.0, 1.0)):
    """(coordinates, row index arrays per piece) of pieces given as atom lists."""
    xyz = np.array([a for p in pieces for a in p], dtype=float) + np.asarray(shift)
    b = np.cumsum([0] + [len(p) for p in pieces])
    return xyz, [np.arange(b[i], b[i + 1]) for i in range(len(pieces))]


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def frames(x0, pieces, box, drift_piece=None, nf=60, seed=5, wrap=True, together=None):
    """(frames as -pbc whole can hand them over, the same frames whole, boxes, the
    lattice shift each piece got in each frame). Every frame is the structure
    TUMBLED by up to 180 degrees about its centroid, moved by a random walk of up to
    3 nm and jiggled by 0.05 nm per axis. Then, in a cycle of four frames: none
    moved; the pieces in `together` (default all) moved by one lattice vector, in
    another image but not split; and twice, pieces moved by different lattice
    vectors, some by none."""
    together = range(len(pieces)) if together is None else together
    rng = np.random.default_rng(seed)
    whole = np.empty((nf,) + x0.shape)
    c0 = x0.mean(0)
    walk = np.zeros(3)
    for f in range(nf):
        walk = np.clip(walk + rng.normal(0, 0.4, 3), -3.0, 3.0)
        R = rot(rng.normal(size=3), rng.uniform(0, math.pi))
        x = x0.copy()
        if drift_piece is not None:              # one piece leaves, 0.1 nm a frame
            x[pieces[drift_piece]] += np.array([0.0, 0.1 * f, 0.0])
        whole[f] = (x - c0) @ R.T + c0 + walk + rng.normal(0, 0.05, x.shape)
    wrapped, shifts = whole.copy(), np.zeros((nf, len(pieces), 3), dtype=int)
    if wrap:
        for f in range(nf):
            for p in range(len(pieces)):
                if f % 4 == 1 and p in together:
                    s = LAT[f % len(LAT)]
                elif f % 4 > 1 and (f + p) % 3:
                    s = LAT[(f * 5 + p * 3) % len(LAT)]
                else:
                    continue
                wrapped[f, pieces[p]] += lattice(box, s)
                shifts[f, p] = s
    return wrapped, whole, np.repeat(box[None], nf, axis=0), shifts


def self_fit_rmsf(X):
    """Per-atom RMSF after fitting every frame onto frame 0, written here (Kabsch)
    and not taken from the code under test."""
    ref = X[0]
    fit = np.empty_like(X)
    for k in range(len(X)):
        P, Q = X[k] - X[k].mean(0), ref - ref.mean(0)
        U, _, Vt = np.linalg.svd(P.T @ Q)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        fit[k] = P @ R.T + ref.mean(0)
    return np.sqrt(((fit - fit.mean(0)) ** 2).sum(-1).mean(0))


def one_lattice_vector_per_frame(out, whole, box):
    """True if every frame of out is the whole frame moved by ONE lattice vector."""
    frac = np.einsum("fai,ij->faj", out - whole, np.linalg.inv(box))
    n = np.round(frac)
    return np.allclose(frac, n, atol=1e-6) and bool(np.all(n == n[:, :1, :]))


def frames_split(shifts, cols):
    """Frames in which the pieces `cols` were put in more than one image."""
    return int(np.sum([len({tuple(s) for s in shifts[f][cols]}) > 1 for f in range(len(shifts))]))


# ---------------------------------------------------------------- the side
# The root: one chain of four antiparallel strands of five residues (23 residues
# with the turns). A bridge of four residues 0.4 nm past its end, and a far
# fragment of three residues 0.4 nm past the bridge, whose centroid sits 3.4 nm from
# the root's. 4.8 nm long in a 6.0 nm box, so 1.2 nm of solvent to its image.
BOX = dodecahedron(6.0)
HALF = 3.0                           # half the shortest lattice vector of BOX
side, rfr = build([meander(5, 4, (0.0, 0.0, 0.0)), strand(4, (2.08, 0.85, 0.0)),
                   strand(3, (3.80, 0.85, 0.0))])
c_root, c_far = side[rfr[0]].mean(0), side[rfr[2]].mean(0)
print("      far fragment centroid %.2f nm from the root's; half the shortest lattice "
      "vector %.2f nm" % (np.linalg.norm(c_far - c_root), HALF))
check("the side is in the regime a centroid placement gets wrong",
      np.linalg.norm(c_far - c_root) > HALF + 0.3)

# -f: the side whole except two things. The far fragment one lattice vector away,
# -a, where its nearest neighbour in RAW coordinates is the root's start, 1.2 nm
# off, although the two are 4.8 nm apart in the complex; only the minimum image
# finds its real contact, the bridge. And the bridge split INSIDE itself: its last
# two residues another lattice vector off.
ref = side.copy()
ref[rfr[2]] += lattice(BOX, (-1, 0, 0))
ref[rfr[1][-6:]] += lattice(BOX, (-1, 0, 1))
d_root, d_bridge = cdist(ref[rfr[2]], ref[rfr[0]]).min(), cdist(ref[rfr[2]], ref[rfr[1]]).min()
check("in -f's raw coordinates the far fragment is nearer the root (%.2f nm) than "
      "the bridge (%.2f nm)" % (d_root, d_bridge), d_root < 1.5 < d_bridge)
wrapped, whole, M, shifts = frames(side, rfr, BOX)
print("      as handed over, the self-fit RMSF reaches %.2f nm (the guard is %.1f)"
      % (self_fit_rmsf(wrapped).max(), RMSF_SPLIT))

print("\n[1] the fragments are found, largest first, by minimum-imaged gaps")
if have:
    fr = ns["backbone_fragments"](ref, BOX)
    check("three fragments of %s atoms, the bridge whole although -f splits it"
          % [len(f) for f in fr], [list(f) for f in fr] == [list(t) for t in rfr],
          str([len(f) for f in fr]))
    check("the four strands of the root are one chain",
          [len(f) for f in ns["backbone_fragments"](side[rfr[0]], BOX)] == [len(rfr[0])])

print("\n[2] put together by contact: one lattice vector per frame, the noise as RMSF")
if have:
    out, nfrag, nmoved = ns["reassemble_backbone"](wrapped, M, ref, BOX, len(side))
    print("      fragments per side %s, frames moved per side %s of %d" % (nfrag, nmoved, len(out)))
    shift = out - wrapped
    frac = np.einsum("fai,ij->faj", shift, np.linalg.inv(BOX))
    check("every atom moved by an integer combination of box vectors",
          np.allclose(frac, np.round(frac), atol=1e-6))
    check("all atoms of one fragment by the same one, so no fragment is bent",
          all(np.allclose(shift[:, f, :], shift[:, f[:1], :], atol=1e-9) for f in rfr))
    check("every frame is its whole frame up to one lattice vector",
          one_lattice_vector_per_frame(out, whole, BOX))
    r_out, r_whole = self_fit_rmsf(out).max(), self_fit_rmsf(whole).max()
    check("RMSF after reassembly %.3f nm = the whole frames' %.3f nm" % (r_out, r_whole),
          abs(r_out - r_whole) < 1e-9, "%.4f vs %.4f" % (r_out, r_whole))
    check("and under the %.1f nm guard" % RMSF_SPLIT, r_out < RMSF_SPLIT)

print("\n[3] the per-side count of frames that needed it")
if have:
    want = frames_split(shifts, [0, 1, 2])
    check("receptor: %d frames had its fragments in more than one image (%s counted)"
          % (want, nmoved), nfrag == [3, 0] and nmoved == [want, 0], "%s %s" % (nfrag, nmoved))

print("\n[4] what needs no move is not touched")
if have:
    _, wh, Mw, _ = frames(side, rfr, BOX, wrap=False)
    o, nf_, nm_ = ns["reassemble_backbone"](wh, Mw, side, BOX, len(side))
    check("whole frames come back bit for bit, 0 frames moved",
          np.array_equal(o, wh) and nm_ == [0, 0], "%s" % nm_)
    one = wrapped[:, rfr[0], :]
    o1, n1, m1 = ns["reassemble_backbone"](one, M, side[rfr[0]], BOX, len(rfr[0]))
    check("a side of one fragment: same coordinates, 1 fragment, 0 frames moved",
          np.array_equal(o1, one) and n1 == [1, 0] and m1 == [0, 0], "%s %s" % (n1, m1))

print("\n[5] a fragment that really leaves is not hidden")
if have:
    wr2, wh2, M2, _ = frames(side, rfr, BOX, drift_piece=1, seed=9)
    out2, _, _ = ns["reassemble_backbone"](wr2, M2, ref, BOX, len(side))
    r2 = self_fit_rmsf(out2)
    check("the drifting bridge still reaches %.2f nm, over the guard"
          % r2[rfr[1]].max(), r2[rfr[1]].max() > RMSF_SPLIT)

print("\n[6] both partners in one tree")
# The receptor: the root chain and three more fragments in a row out to x = 6.9 nm;
# the ligand: a chain of three strands 0.5 nm over the root, and a fragment of three
# residues 0.5 nm over the receptor's last one, 4.2 nm from the ligand's bulk. In an
# 8.0 nm box its nearest own-side contact is past half the shortest lattice vector,
# where the minimum image is the wrong one; its receptor contact is not.
BOX8 = dodecahedron(8.0)
rec8 = [meander(5, 4, (0.0, 0.0, 0.0)), strand(4, (2.08, 0.85, 0.0)),
        strand(4, (3.80, 0.85, 0.0)), strand(3, (5.92, 0.85, 0.0))]
lig8 = [meander(5, 3, (0.0, 0.28, 0.5)), strand(3, (5.92, 0.85, 0.5))]
both, pieces = build(rec8 + lig8)
nrec = sum(len(p) for p in rec8)
lbulk, lfrag = pieces[4], pieces[5]
d_own = cdist(both[lfrag], both[lbulk]).min()
d_rec = cdist(both[lfrag], both[:nrec]).min()
print("      ligand fragment: %.2f nm from its own bulk, %.2f nm from the receptor"
      % (d_own, d_rec))
check("its own-side contact is past half a box (4.0 nm), its receptor contact is not",
      d_own > 4.0 and d_rec < 0.6)
gap = min(cdist(both, both + lattice(BOX8, (i, j, k))).min()
          for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0))
check("the complex keeps %.2f nm of solvent to its nearest image, over 1 nm" % gap, gap > 1.0)
if have:
    # every fourth frame has the whole ligand in another image than the receptor, as
    # -pbc whole often leaves the partners: moved, but not split
    wr3, wh3, M3, sh3 = frames(both, pieces, BOX8, seed=13, together=[4, 5])
    out3, nf3, nm3 = ns["reassemble_backbone"](wr3, M3, both, BOX8, nrec)
    check("every frame of the complex is its whole frame up to one lattice vector",
          one_lattice_vector_per_frame(out3, wh3, BOX8))
    want = [frames_split(sh3, [0, 1, 2, 3]), frames_split(sh3, [4, 5])]
    check("fragments per side %s, frames per side %s (expected [4, 2], %s)"
          % (nf3, nm3, want), nf3 == [4, 2] and nm3 == want)

print("\n[7] try_measured_anchors reassembles the backbone before the self-fit")
t = grab("try_measured_anchors") or ""
i_r = t.find("reassemble_backbone(")
i_m = t.find("measure_rigidity(prot1_data")
check("reassemble_backbone is called before measure_rigidity", 0 <= i_r < i_m,
      "%d %d" % (i_r, i_m))
check("on both sides at once, against -f in -f's own box",
      "reassemble_backbone(traj, boxes, ref, INPUT_BOX, len(bb1))" in t
      and "ref = np.array([XYZ_OF[a] for a in bb])" in t)
check("with the trajectory's own boxes", "traj, boxes = load_backbone_trajectory(bb)" in t)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all fragment-reassembly checks passed")
