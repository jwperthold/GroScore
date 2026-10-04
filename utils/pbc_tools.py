#!/usr/bin/env python3
"""Periodic-boundary geometry for the diagnostics, numpy only.

Shared by utils/interface_qc.py, utils/rebound_rmsd.py, utils/measure_box.py and
utils/assemble_frame.py. All of them read solute coordinates out of mdrun frames,
and in the 2026-09 HADDOCK pass 2 (bm_fe_amber_2/_ana/digest) the first three
failed on the same three properties of those frames. Each is handled here once.

1. THE ATOM-NUMBER FIELD WRAPS. A .gro carries the atom number in a 5-digit field
   that GROMACS writes modulo 100000. 22 of 46 production systems are larger than
   that, and interface_qc.py, which keyed atoms by that field, let solvent atom
   100000+k overwrite solute atom k in 19 of them: contact recovery 0.001 where
   GROMACS's own pull listing gives 0.91. Atoms are read by POSITION here, and
   nothing is looked up by the atom-number or residue-number field, which wraps the
   same way.

2. A RAW DIFFERENCE IS NOT A MINIMUM IMAGE. Testing the 27 neighbouring images of a
   raw coordinate difference finds the minimum image only when the difference is
   already within one box vector of it. 1E6J cycle 3 read rms_dev 4.581 nm against
   0.059 nm from grompp, the signature of pairs left a box vector apart.
   min_image() first reduces the difference into the box the way GROMACS does
   (along c, then b, then a) and only then searches the neighbours, and further
   still in the flat triclinic cells where 27 neighbours are not enough, so it
   returns the shortest image in every box GROMACS can write.

3. A CUTOUT PARTNER IS NOT ONE PIECE. make_cutout.py builds each partner as one
   moleculetype made of several capped, UNBONDED fragments, and in 40 of 46 pass-2
   setups mdrun left those fragments in different periodic images. Bonds cannot
   rejoin them (gmx trjconv -pbc whole follows bonds), and nearest-image
   propagation along the sequence breaks wherever two sequence-consecutive
   fragments sit more than half a box apart, which is where the 31-41 A rebound
   RMSDs of 1HE8, 1I4D and 1IJK came from. -pbc cluster is such a propagation: it
   chains every atom of a moleculetype to the image nearest the atom before it
   before moving the whole as one unit, which put chain C of the classic 1IJK
   restraint frame, and three ligand fragments of its FE probe frame, a box vector
   away from the rest.
   assemble() rejoins residues by SPACE instead: a minimum spanning tree over one
   anchor atom per residue, each residue placed at the image nearest the residue it
   hangs from. In a packed complex every residue has a neighbour within about 1 nm
   (the longest tree edge, which the callers print, is 0.40-0.59 nm on real 2KTF,
   1AKJ and 1IJK frames; a structural chloride of 2AJF hangs 2.2 nm from the nearest
   CA), while the nearest periodic copy of any residue is at least the solvent gap
   between the complex and its image away, 2 nm even in the 1 nm probe box (2.35 nm
   the closest on those frames). Every tree edge is therefore far inside half a box
   and the choice of image is never a close call.
   The tree uses distances only, so a whole-complex drift or rotation of any size
   between two frames does not enter.

follow() is the same idea along a trajectory: each frame is placed at the image
nearest the previous, already assembled one, atom by atom, which is what
gmx trjconv -pbc nojump does, but starting from an assembled first frame rather
than from whatever images the first frame happened to have.
"""

import re
import numpy as np

# Lattice translations i*a + j*b + k*c with i, j, k in {-1, 0, 1}, as coefficients.
# Multiplied by the box per call, because an NPT trajectory has a box per frame.
_NEIGHBOURS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                        for k in (-1, 0, 1)], dtype=float)

# Residues that belong to the polymer without carrying an N/CA/C triple: the caps
# cap_termini.py adds (ACE, NME) and every other cap the GROMACS 2026
# residuetypes.dat lists as Protein: NHE (amber19sb.ff), NAC, CT3 (CHARMM) and NH2,
# an amidated C-terminus as peptides are often deposited (gromos54a8.ff has an
# [ NH2 ] entry). Left out, a cap at the extreme of the complex would drop out of
# the diameter measure_box.py sizes the box from.
CAP_RESIDUES = {"ACE", "NME", "NHE", "NAC", "CT3", "NH2"}


# ---------------------------------------------------------------- .gro reading

def box_matrix(fields):
    """3x3 box matrix, ROWS the box vectors a, b, c, from the numbers on a .gro box
    line: a_x b_y c_z [a_y a_z b_x b_z c_x c_y]. A rectangular box has three."""
    v = [float(x) for x in list(fields)[:9]]
    v += [0.0] * (9 - len(v))
    return np.array([[v[0], v[3], v[4]],
                     [v[5], v[1], v[6]],
                     [v[7], v[8], v[2]]], dtype=float)


class Frame(object):
    """One .gro frame. Atoms are indexed by POSITION, 0-based.

    head  the first 20 columns of every atom line (residue number, residue name,
          atom name, atom number), kept verbatim so a selection can be written back
          out unchanged; None on trajectory frames read without names
    xyz   (n, 3) coordinates, nm
    box   (3, 3), rows a, b, c"""

    def __init__(self, title, head, xyz, box):
        self.title, self.head, self.xyz, self.box = title, head, xyz, box

    def __len__(self):
        return len(self.xyz)

    @property
    def resname(self):
        return [h[5:10].strip() for h in self.head]

    @property
    def name(self):
        return [h[10:15].strip() for h in self.head]

    @property
    def time(self):
        """Frame time in ps from a title written by trjconv or mdrun, else None."""
        m = re.search(r"\bt=\s*([-+0-9.eE]+)", self.title or "")
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None


def field_width(line):
    """Width of one coordinate field. %8.3f by default, but trjconv -ndec writes
    more decimals, and GROMACS's own reader takes the width from the distance
    between the first two decimal points; so does this. The field then holds
    width - 5 decimals."""
    p1 = line.find(".", 20)
    p2 = line.find(".", p1 + 1) if p1 >= 0 else -1
    return (p2 - p1) if (p1 >= 0 and p2 > p1) else 8


def parse_xyz(lines, w):
    """(n, 3) coordinates from .gro atom lines whose fields are w columns wide."""
    raw = "".join(l[20:20 + 3 * w].ljust(3 * w) for l in lines).encode()
    return np.frombuffer(raw, dtype="S%d" % w).astype(float).reshape(-1, 3)


def iter_gro(path, names=True):
    """Yield every frame of a (multi-frame) .gro, streaming, so a long trajectory is
    never held in memory. With names=False only the first frame keeps its atom
    heads; a trajectory written by trjconv repeats the same ones in every frame."""
    with open(path) as f:
        first = True
        while True:
            title = f.readline()
            if not title:
                return
            nline = f.readline()
            if not nline.strip():
                return
            n = int(nline)
            lines = [f.readline() for _ in range(n)]
            boxline = f.readline()
            if (n and not lines[-1]) or not boxline.strip():
                return                                # truncated last frame
            xyz = parse_xyz(lines, field_width(lines[0])) if n else np.empty((0, 3))
            head = [l[:20] for l in lines] if (names or first) else None
            yield Frame(title.rstrip("\n"), head, xyz, box_matrix(boxline.split()))
            first = False


def read_gro(path):
    """The first frame of a .gro. ValueError if there is none."""
    for fr in iter_gro(path):
        return fr
    raise ValueError("no frame in %s" % path)


def write_gro(path, title, head, xyz, box):
    """Write one frame. head supplies columns 1-20 verbatim; coordinates %8.3f."""
    with open(path, "w") as f:
        f.write("%s\n%d\n" % (title, len(xyz)))
        for h, (x, y, z) in zip(head, xyz):
            f.write("%-20s%8.3f%8.3f%8.3f\n" % (h[:20], x, y, z))
        b = box
        vals = [b[0, 0], b[1, 1], b[2, 2]]
        off = [b[0, 1], b[0, 2], b[1, 0], b[1, 2], b[2, 0], b[2, 1]]
        if any(abs(v) > 0 for v in off):
            vals += off
        f.write("".join("%10.5f" % v for v in vals) + "\n")


def ndx_group(path, name):
    """1-based atom indices of the FIRST group called `name` (case-insensitive) in a
    GROMACS index file. make_boresch.py appends many groups with repeated names, so
    first-wins is the rule that matches what gmx tools select. ValueError if absent."""
    want, cur, out = name.strip().lower(), None, None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("["):
                if out is not None:
                    break
                cur = s.strip("[] \t").lower()
                if cur == want:
                    out = []
                continue
            if out is not None and s:
                out.extend(int(t) for t in s.split())
    if out is None:
        raise ValueError("no [ %s ] group in %s" % (name, path))
    return out


# ---------------------------------------------------------------- images

def reduce_vectors(dv, box):
    """Shift difference vectors into the brick |x| <= a_x/2, |y| <= b_y/2,
    |z| <= c_z/2 by whole box vectors: along c, then b, then a.

    The order matters. c is the only box vector with a z component and b the only
    one left with a y component, so each step fixes its coordinate without
    disturbing the one before. This is GROMACS's own first step, and it already
    recovers any displacement shorter than half a brick edge exactly, which is all
    follow() needs. A zero box edge means no periodicity along that vector."""
    dv = np.array(dv, dtype=float)                    # always a copy
    for m in (2, 1, 0):
        L = box[m, m]
        if L > 0:
            dv -= np.round(dv[..., m] / L)[..., None] * box[m]
    return dv


def exact_radius(box):
    """Length below which the 27-neighbour search of min_image() is provably exact.

    The minimum image m of a brick-reduced r is r + i*a + j*b + k*c. Its z
    component gives |k| c_z <= |m| + c_z/2, so |k| <= 1 while |m| < 1.5 c_z; then
    y gives |j| <= 1 while |m| < 1.5 b_y - |c_y|, and x gives |i| <= 1 while
    |m| < 1.5 a_x - |b_x| - |c_x|. In the xy-square rhombic dodecahedron both
    engines build this is the box vector L, above every minimum image there is
    (the cell's corners sit at L/sqrt(2)); in the truncated octahedron it is 0.83 L
    against corners at 0.65 L. In the xy-hexagon dodecahedron (0.5 L against
    corners at 0.71 L) and in flat triclinic cells it lies below the longest
    minimum images, and min_image() searches further for those."""
    a, b, c = box
    return min(1.5 * c[2], 1.5 * b[1] - abs(c[1]), 1.5 * a[0] - abs(b[0]) - abs(c[0]))


def _lattice(box, rho, half):
    """Lattice vectors i*a + j*b + k*c over the coefficient ranges that can matter at
    length rho: |k| c_z <= rho, then |j| b_y <= rho + |k c_y|, then
    |i| a_x <= rho + |j b_x| + |k c_x|, each widened by `half` box steps (0.5 when
    the vectors are added to a brick-reduced difference, 0 for bare lattice
    vectors)."""
    a, b, c = box
    K = int(np.floor(rho / c[2] + half))
    J = int(np.floor((rho + K * abs(c[1])) / b[1] + half))
    I = int(np.floor((rho + J * abs(b[0]) + K * abs(c[0])) / a[0] + half))
    ijk = np.array([(i, j, k) for i in range(-I, I + 1) for j in range(-J, J + 1)
                    for k in range(-K, K + 1)], dtype=float)
    return ijk @ box


def lattice_vectors(box, rmax):
    """Every nonzero lattice vector no longer than rmax, as rows."""
    T = _lattice(box, rmax, 0.0)
    n = np.sqrt((T * T).sum(-1))
    return T[(n > 0) & (n <= rmax)]


def shortest_vector(box):
    """Length of the shortest nonzero lattice vector. In a flat triclinic cell a sum
    of box vectors can be shorter than any of them, and half this length, not half
    the shortest box vector, is how far a minimum image is certain to be unique."""
    return float(np.sqrt((lattice_vectors(box, np.linalg.norm(box, axis=1).min()) ** 2)
                         .sum(-1)).min())


def min_image(dv, box):
    """The shortest vector equivalent to each dv under the box (rows a, b, c, the
    lower-triangular form GROMACS writes), for any dv in any such box.

    reduce_vectors() first, so the raw difference may be any number of box vectors
    away; then the 27 neighbours of the reduced vector, because in a triclinic box
    the brick is not the Wigner-Seitz cell and the reduced vector can still be one
    lattice step from its minimum image. That is the search GROMACS makes after the
    same reduction, and it is exact for every minimum image shorter than
    exact_radius(), which covers every vector in the rhombic dodecahedra both
    engines build. It is not exact in every cell GROMACS accepts: the review
    (QC-7) found flat triclinic cells such as rows (11.898, 0, 0),
    (-5.949, 11.423, 0), (-5.949, -5.712, 3.316) where it returned images up to
    1.134 nm too long. A result at or beyond exact_radius() is therefore searched
    again over every lattice step that could still shorten it (the bounds of
    _lattice). In the pipeline's cells that second search never runs; checked
    against an exhaustive search it changed no result in rectangular boxes,
    dodecahedra or octahedra, only in flat cells like that one."""
    dv = reduce_vectors(dv, box)
    cand = dv[..., None, :] + _NEIGHBOURS @ box       # (..., 27, 3)
    d2 = (cand * cand).sum(-1)
    k = np.argmin(d2, axis=-1)
    out = np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]
    if (np.diag(box) > 0).all():
        got2 = np.take_along_axis(d2, k[..., None], axis=-1)[..., 0]
        far = got2 >= exact_radius(box) ** 2
        if far.any():
            T = _lattice(box, float(np.sqrt(got2[far].max())), 0.5)
            c2 = dv[far][:, None, :] + T[None]
            j = np.argmin((c2 * c2).sum(-1), axis=-1)
            out[far] = c2[np.arange(len(j)), j]
    return out


# ---------------------------------------------------------------- residues

def residue_blocks(head):
    """[(start, end), ...] for consecutive runs of atoms sharing columns 1-10
    (residue number and name) of their .gro lines, i.e. residues by POSITION.
    A wrapped residue number only ever repeats 100000 residues later, so it cannot
    merge two neighbours."""
    out, start = [], 0
    for i in range(1, len(head) + 1):
        if i == len(head) or head[i][:10] != head[start][:10]:
            out.append((start, i))
            start = i
    return out


def anchor_atoms(head, blocks, prefer="CA"):
    """One anchor atom per residue: its CA if it has one, else its first atom."""
    out = []
    for s, e in blocks:
        a = s
        for i in range(s, e):
            if head[i][10:15].strip() == prefer:
                a = i
                break
        out.append(a)
    return np.array(out, dtype=int)


def polymer_atoms(head, blocks=None):
    """Indices of atoms in POLYMER residues: those carrying atoms named N, CA and C
    (every amino acid, NCAAs included, whatever their residue name) plus the cap
    residues, which have no such triple. Ligands, ions and waters have none of the
    three names together, a calcium ion's lone atom CA included."""
    if blocks is None:
        blocks = residue_blocks(head)
    keep = []
    for s, e in blocks:
        names = {head[i][10:15].strip() for i in range(s, e)}
        if {"N", "CA", "C"} <= names or head[s][5:10].strip() in CAP_RESIDUES:
            keep.extend(range(s, e))
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------- assembly

def spanning_tree(P, box):
    """Place points so every one sits at the image nearest the point it hangs from
    in a minimum spanning tree over minimum-image distances (Prim's algorithm).

    Returns (positions, longest tree edge in nm). The longest edge is the margin
    the result rests on: as long as it is far below half the shortest lattice
    vector (shortest_vector), no point could have been placed at another image."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    pos = P.copy()
    if n < 2:
        return pos, 0.0
    todo = np.arange(1, n)
    d = min_image(P[todo] - pos[0], box)
    best = np.sqrt((d * d).sum(-1))                  # distance of each to the tree
    cand = pos[0] + d                                # where it would go
    longest = 0.0
    while len(todo):
        k = int(np.argmin(best))
        node = todo[k]
        pos[node] = cand[k]
        longest = max(longest, float(best[k]))
        keep = np.ones(len(todo), dtype=bool)
        keep[k] = False
        todo, best, cand = todo[keep], best[keep], cand[keep]
        if not len(todo):
            break
        d = min_image(P[todo] - pos[node], box)
        dn = np.sqrt((d * d).sum(-1))
        upd = dn < best
        best[upd] = dn[upd]
        cand[upd] = pos[node] + d[upd]
    return pos, longest


def assemble(xyz, box, blocks, anchors=None):
    """Rejoin a solute that mdrun left in several periodic images.

    blocks   residues as (start, end) ranges covering every atom of xyz; a residue
             is far smaller than half a box, so it is made whole about its anchor
    anchors  one atom index per block (default: the first atom of each)

    Residues are joined by a spanning tree over their anchors (see the module
    docstring for why that cannot pick a wrong image in a packed complex), and each
    atom is then placed at the image nearest its own anchor. Returns
    (assembled xyz, longest tree edge in nm)."""
    xyz = np.asarray(xyz, dtype=float)
    if anchors is None:
        anchors = np.array([s for s, _ in blocks], dtype=int)
    member = np.empty(len(xyz), dtype=int)
    for k, (s, e) in enumerate(blocks):
        member[s:e] = k
    A = xyz[anchors]
    pos, longest = spanning_tree(A, box)
    return pos[member] + min_image(xyz - A[member], box), longest


def follow(prev, xyz, box):
    """xyz placed atom by atom at the image nearest `prev`, the previous frame
    already assembled. Exact while no atom moves more than half a brick edge
    between two frames, which at the 10 ps the probe writes is never close."""
    prev = np.asarray(prev, dtype=float)
    return prev + reduce_vectors(np.asarray(xyz, dtype=float) - prev, box)
