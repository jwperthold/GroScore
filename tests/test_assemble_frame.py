#!/usr/bin/env python3
"""utils/assemble_frame.py rebuilds the restraint reference frame the way the
restraint scripts need it, and touches nothing else.

Both engines took their restraint frame from gmx trjconv -pbc cluster, which chains
each atom to the image nearest the one before it. A cutout partner is one
moleculetype of unbonded fragments, so a sequence step longer than half the box
sends every later fragment a box vector away: on real 1IJK setups that dropped
chain C from the classic interface (407 of 971 pairs kept) and three ligand
fragments from the FE springs (720 of 1108 contact pairs). Pinned here, through the
command line both runners use:

  1. A frame split the way -pbc cluster and mdrun leave it (two fragments after a
     3.5 nm sequence jump one box vector off, the fragment holding the tree's root,
     another fragment, a zinc and a ligand in other images, one residue broken
     across the boundary; atom-number field scrambled, velocities on every line)
     comes back as the whole frame BYTE FOR BYTE: solvent, velocities, title and
     box untouched, the 35 repaired residues moved by box vectors only, the
     largest set of atoms sharing an image left where it was, the longest tree
     link 0.50 nm (the strand spacing), a non-water residue outside the group
     named on stderr.
  2. A whole frame comes out byte-identical, 0 residues moved; the split frame
     written with the 9-column fields of trjconv -ndec 4 comes back as the whole
     one byte for byte as well; -o may be the input.
  3. Every failure exits 1 and leaves the output alone: unreadable, truncated,
     missing or two-frame input, no box, missing group, a group past the frame or
     holding part of a residue, a usage error.
  4. The two refusals, each just below and just above its limit at the file's
     0.001 nm resolution (exactly AT the limit the outcome rests on binary
     rounding of decimal input, so it is not pinned), each leaving the output
     alone: a residue 2.998 nm from everything passes and 3.002 nm fails in a box
     whose half shortest lattice vector is 3.000 nm; a chain whose ends come
     0.601 nm close through the box passes and 0.599 nm fails, the 0.6 nm the
     restraint scripts take springs at.

No gmx needed. Run:  python3 tests/test_assemble_frame.py
"""

import os, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AF = os.path.join(REPO, "utils", "assemble_frame.py")
failures = []


def check(name, ok, detail=""):
    print("  %-66s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


# A rhombic dodecahedron of 6.000 nm with c_z written as 4.243, so every box vector
# lies on the 0.001 nm grid and a shift by box vectors is exact in the .gro text.
BOX = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [3.0, 3.0, 4.243]])
BOXLINE = "".join("%10.5f" % v for v in (6.0, 6.0, 4.243, 0, 0, 0, 0, 3.0, 3.0)) + "\n"
A, B, C = BOX


def gro_text(atoms, xyz, vel, w=8, boxline=BOXLINE, title="synthetic frame t= 100.000"):
    """atoms: [(resnum, resname, atomname)]; atom numbers deliberately scrambled,
    since the tool must go by position."""
    d = w - 5
    out = [title + "\n", "%5d\n" % len(atoms)]
    for i, ((rn, res, an), x, v) in enumerate(zip(atoms, xyz, vel)):
        out.append("%5d%-5s%5s%5d" % (rn % 100000, res, an, (7919 * (i + 1)) % 100000)
                   + ("%%%d.%df" % (w, d)) * 3 % tuple(x)
                   + ("%%%d.%df" % (w, d + 1)) * 3 % tuple(v) + "\n")
    return "".join(out) + boxline


def write(path, text):
    with open(path, "w", newline="") as f:
        f.write(text)


def read(path):
    with open(path, newline="") as f:
        return f.read()


def ndx(path, groups):
    with open(path, "w") as f:
        for name, idx in groups:
            f.write("[ %s ]\n" % name)
            for s in range(0, len(idx), 15):
                f.write(" ".join("%d" % (i + 1) for i in idx[s:s + 15]) + "\n")


def run(d, *args):
    r = subprocess.run([sys.executable, AF] + list(args), cwd=d, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------- the complex
# Eight 8-residue strands 0.500 nm apart, each an unbonded fragment: partner A at
# strand positions 0, 7, 3 (its first two fragments 3.5 nm apart in sequence order,
# more than half the box), partner B at 1, 2, 4, 5, 6. A non-water residue UNK sits
# outside the group among its residues; a zinc and a three-atom ligand are in it;
# then 150 waters wrapped atom by atom and a counter-ion.
OFF = {"N": (0.040, -0.130, 0.030), "CA": (0.0, 0.0, 0.0), "C": (-0.040, 0.130, -0.030),
       "O": (-0.120, 0.130, -0.030), "H": (0.100, -0.150, 0.030)}
T0 = np.array([2.000, 2.000, 1.500])
atoms, X, frag = [], [], []
rn = 0
for k in (0, 7, 3, 1, 2, 4, 5, 6):
    for r in range(8):
        rn += 1
        for an, o in OFF.items():
            atoms.append((rn, "ALA", an))
            X.append(T0 + [0.5 * k, 0.38 * r, 0.0] + np.array(o))
            frag.append(len(set(frag)) if r == 0 and an == "N" else (frag[-1] if frag else 0))
NPROT = len(atoms)                                       # 320
atoms.append((rn + 1, "UNK", "X1"))
X.append(T0 + [1.75, -0.40, 0.0])
atoms.append((rn + 2, "ZN", "ZN"))
X.append(T0 + [1.75, 1.33, -0.300])
for an, o in (("C1", (0.0, 0.0, 0.350)), ("C2", (0.100, 0.0, 0.420)), ("O1", (0.200, 0.0, 0.350))):
    atoms.append((rn + 3, "LIG", an))
    X.append(T0 + [2.5, 1.52, 0.0] + np.array(o))
GROUP = list(range(NPROT)) + [NPROT + 1, NPROT + 2, NPROT + 3, NPROT + 4]   # all but UNK
rng = np.random.default_rng(20261004)
nwat = 0
for gx in np.arange(0.3, 6.0, 1.2):
    for gy in np.arange(0.3, 6.0, 1.2):
        for gz in np.arange(0.2, 4.2, 0.7):
            nwat += 1
            for an, o in (("OW", (0, 0, 0)), ("HW1", (0.096, 0.0, 0.0)),
                          ("HW2", (-0.024, 0.093, 0.0))):
                atoms.append((rn + 3 + nwat, "SOL", an))
                X.append(np.array([gx, gy, gz]) + o + [0.11 * nwat % 1.0, 0.0, 0.0])
atoms.append((rn + 4 + nwat, "NA", "NA"))
X.append(np.array([5.900, 5.900, 4.200]))
X = np.round(np.array(X), 3)
for i in range(NPROT + 5, len(X)):                     # waters wrapped atom by atom
    for m in (2, 1, 0):
        X[i] -= np.floor(X[i, m] / BOX[m, m]) * BOX[m]
X = np.round(X, 3)
VEL = np.round(rng.normal(0.0, 0.4, X.shape), 4)
frag = np.array(frag)

# The split: A's second and third fragments one box vector off (what -pbc cluster's
# chain does after the 3.5 nm jump), A's first fragment, which holds the tree's root
# residue, B's second fragment, the zinc and the ligand in four other images, and
# one atom of residue 27 across the boundary. The largest set of atoms sharing one
# image is B's other four fragments, so that is the placement the output keeps.
SPLIT = X.copy()
SPLIT[:NPROT][frag == 0] += B - A
SPLIT[:NPROT][(frag == 1) | (frag == 2)] += A
SPLIT[:NPROT][frag == 4] += C - B
SPLIT[NPROT + 1] -= C
SPLIT[NPROT + 2:NPROT + 5] -= A
o27 = [i for i, (r_, _, an) in enumerate(atoms) if r_ == 27 and an == "O"][0]
SPLIT[o27] += B
SPLIT = np.round(SPLIT, 3)

work = tempfile.mkdtemp(prefix="assemble_frame_")
W = lambda *p: os.path.join(work, *p)
whole_txt = gro_text(atoms, X, VEL)
write(W("whole.gro"), whole_txt)
write(W("split.gro"), gro_text(atoms, SPLIT, VEL))
ndx(W("index.ndx"), [("System", list(range(len(atoms)))), ("Protein_Struct", GROUP)])

print("\n1. a frame split the way -pbc cluster and mdrun leave it")
rc, out, err = run(work, "-f", "split.gro", "-n", "index.ndx", "-g", "Protein_Struct",
                   "-o", "out.gro")
check("exit 0, nothing on stdout, one line on stderr",
      rc == 0 and not out and len(err.strip().splitlines()) == 1, err.strip()[-300:])
check("the output is the whole frame byte for byte",
      os.path.isfile(W("out.gro")) and read(W("out.gro")) == whole_txt)
check("35 residues moved, 1 of them also made whole",
      "35 residues moved by a box vector (1 of them also made whole)" in err, err.strip()[-200:])
check("longest link 0.50 nm (the strand spacing), half the lattice 3.00 nm",
      "longest link 0.50 nm, half the shortest lattice vector 3.00 nm" in err)
check("324 group atoms in 66 residues; UNK named as left in place",
      "(324 atoms, 66 residues)" in err and "left in place: 1 x UNK" in err)

print("\n2. a whole frame is left alone; -ndec fields; -o = -f")
rc, out, err = run(work, "-f", "whole.gro", "-n", "index.ndx", "-o", "out_whole.gro")
check("whole frame: byte-identical, 0 residues moved",
      rc == 0 and read(W("out_whole.gro")) == whole_txt and ": 0 residues moved" in err,
      err.strip()[-200:])
whole9 = gro_text(atoms, X, VEL, w=9)
write(W("split9.gro"), gro_text(atoms, SPLIT, VEL, w=9))
rc, out, err = run(work, "-f", "split9.gro", "-n", "index.ndx", "-o", "out9.gro")
check("9-column fields (trjconv -ndec 4): the whole frame byte for byte",
      rc == 0 and read(W("out9.gro")) == whole9, err.strip()[-200:])
shutil.copy(W("split.gro"), W("inplace.gro"))
rc, out, err = run(work, "-f", "inplace.gro", "-n", "index.ndx", "-o", "inplace.gro")
check("-o equal to -f: repaired in place", rc == 0 and read(W("inplace.gro")) == whole_txt)
check("no scratch file left in the directory",
      not [f for f in os.listdir(work) if f.endswith(".tmp")], str(os.listdir(work)))

print("\n3. failures exit 1 and leave the output alone")
SENTINEL = "not to be touched\n"


def refused(label, files, *args):
    """files: {name: text} written first; the run must exit 1, say why on stderr
    and leave out_keep.gro as it was."""
    for name, text in files.items():
        write(W(name), text)
    write(W("out_keep.gro"), SENTINEL)
    rc, out, err = run(work, *(list(args) + ["-o", "out_keep.gro"]))
    check(label, rc == 1 and "assemble_frame: ERROR" in err and read(W("out_keep.gro")) == SENTINEL,
          "rc %d: %s" % (rc, err.strip()[-160:]))


lines = whole_txt.splitlines(True)
refused("unreadable frame", {"bad.gro": "garbage\n"}, "-f", "bad.gro", "-n", "index.ndx")
refused("truncated frame", {"trunc.gro": "".join(lines[:-10])},
        "-f", "trunc.gro", "-n", "index.ndx")
refused("two frames in one file", {"two.gro": whole_txt + whole_txt},
        "-f", "two.gro", "-n", "index.ndx")
refused("no periodic box", {"nobox.gro": "".join(lines[:-1]) + "   0.00000   0.00000   0.00000\n"},
        "-f", "nobox.gro", "-n", "index.ndx")
refused("missing input file", {}, "-f", "nothere.gro", "-n", "index.ndx")
refused("missing group", {}, "-f", "split.gro", "-n", "index.ndx", "-g", "Nope")
ndx(W("past.ndx"), [("Protein_Struct", GROUP + [len(atoms) + 5])])
refused("group reaching past the frame", {}, "-f", "split.gro", "-n", "past.ndx")
ndx(W("part.ndx"), [("Protein_Struct", [i for i in GROUP if i != o27])])
refused("group holding part of a residue", {}, "-f", "split.gro", "-n", "part.ndx")
refused("usage error (no -f)", {}, "-n", "index.ndx")

print("\n4. the two refusals, just below and just above their limits")
# A lone residue and a zinc along the xy diagonal, where the cell reaches 4.24 nm:
# 2.998 nm passes, 3.002 nm (0.5 of the 6.000 nm shortest lattice vector is 3.000)
# fails.
for label, dxy, want in (("residue 2.998 nm from everything: assembled", 2.120, 0),
                         ("residue 3.002 nm from everything: refused", 2.123, 1)):
    at2 = [(1, "ALA", "N"), (1, "ALA", "CA"), (1, "ALA", "C"), (2, "ZN", "ZN")]
    x2 = np.array([[1.040, 0.870, 2.030], [1.000, 1.000, 2.000], [0.960, 1.130, 1.970],
                   [1.000 + dxy, 1.000 + dxy, 2.000]])
    write(W("lone.gro"), gro_text(at2, x2, np.zeros_like(x2)))
    ndx(W("lone.ndx"), [("Protein_Struct", [0, 1, 2, 3])])
    write(W("out_keep.gro"), SENTINEL)
    rc, out, err = run(work, "-f", "lone.gro", "-n", "lone.ndx", "-o", "out_keep.gro")
    check(label, rc == want and (want == 0 or ("not determined" in err
                                               and read(W("out_keep.gro")) == SENTINEL)),
          "rc %d: %s" % (rc, err.strip()[-160:]))
# A straight chain along a: N(i) 0.38 nm apart, and the last residue's C placed so
# that it comes 0.599 or 0.601 nm close to the first N through the box.
for label, cx, want in (("chain ends 0.601 nm apart through the box: assembled", 5.599, 0),
                        ("chain ends 0.599 nm apart through the box: refused", 5.601, 1)):
    at3, x3 = [], []
    for i in range(14):
        for an, dx in (("N", -0.120), ("CA", 0.0), ("C", 0.120)):
            at3.append((i + 1, "ALA", an))
            x3.append([0.320 + 0.380 * i + dx, 3.000, 2.000])
    for an, x in (("N", 5.500), ("CA", 5.560), ("C", cx)):
        at3.append((15, "ALA", an))
        x3.append([x, 3.000, 2.000])
    x3 = np.array(x3)
    write(W("chain.gro"), gro_text(at3, x3, np.zeros_like(x3)))
    ndx(W("chain.ndx"), [("Protein_Struct", list(range(len(at3))))])
    write(W("out_keep.gro"), SENTINEL)
    rc, out, err = run(work, "-f", "chain.gro", "-n", "chain.ndx", "-o", "out_keep.gro")
    check(label, rc == want and (want == 0 or ("1 polymer heavy-atom pairs" in err
                                               and "touches its own copy" in err
                                               and read(W("out_keep.gro")) == SENTINEL)),
          "rc %d: %s" % (rc, err.strip()[-160:]))

shutil.rmtree(work, ignore_errors=True)
print("")
if failures:
    print("FAILED (%d): %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("All assemble_frame checks passed.")
