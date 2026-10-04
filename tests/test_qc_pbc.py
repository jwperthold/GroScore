#!/usr/bin/env python3
"""The rebinding diagnostics read mdrun frames the way mdrun writes them.

Pass 2 (bm_fe_amber_2/_ana/digest: setup.txt S10, qc.txt iqc_wrap_artefact,
rmsd_image_artefact) found both rebinding diagnostics measuring artefacts of the
file format and of the periodic cell rather than the structure. Pinned here:

  A. interface_qc.py reads atoms by POSITION. The .gro atom-number field wraps at
     100000, and keying atoms by it let solvent atom 100000+k overwrite solute atom
     k: 19 structures read recovered 0.001 in every cycle where grompp's own
     PBC-aware listing gives 0.91. Two synthetic 100,600-atom frames whose solvent
     numbers wrap onto the solute, DIFFERENT frames (the query's solute moved
     rigidly and re-wrapped atom by atom, its solvent arranged differently, every
     CG displaced 0.1 nm in the body frame) must read full recovery and the
     side-chain RMSD those displacements give by construction, sqrt(0.1^2 / 2) nm
     = 0.707 A (CB stays, CG moves), the value the same solute pair gives without
     solvent. A frame compared with itself would read 0 whatever solvent atoms
     entered the fit (review QC-1: three sabotages passed that way).
  B. interface_qc.py takes TRUE minimum images in a triclinic cell. A 27-image
     search on the raw difference fails once a pair is more than one box vector
     from its minimum image (1E6J c3: rms_dev 4.581 nm against grompp's 0.059). A
     rhombic dodecahedron frame with the interface wrapped atom by atom across the
     cell boundary, and one fragment two box vectors off, must read the same as
     the unwrapped frame; and a query rotated 37 degrees, whose interface reaches
     5.3 nm from its first atom in a box of inradius 3.5 nm, must read 0.707 A,
     which pins the spanning-tree assembly of the side-chain fit (gathering about
     the first atom puts its far end at different images in the two frames).
  C. rebound_rmsd.py reassembles cutout fragments in space. A cutout partner is one
     moleculetype of unbonded fragments; when two sequence-consecutive fragments
     sit more than half a box apart, residue-by-residue propagation puts one a box
     vector off (the 31-41 A values of 1HE8, 1I4D, 1IJK). A synthetic complex with
     exactly that layout, fragments scattered over images and the whole body
     rotated and drifted by more than half a box, must give the RMSD of the
     unscattered pair: 0 for a rigid motion, and 0.8819 A for one displaced
     fragment, which an independent mass-weighted fit with GROMACS's own guessed
     masses reproduces below. The old propagation fails on this layout even with
     nothing scattered (12.9 A for the rigid motion), since its first two chain-A
     fragments are 3.5 nm apart in a 6 nm box; on real 2KTF cycles, which have no
     such gap, old and new agree to 0.0001 A.
  D. pbc_tools.min_image against a brute-force lattice search, on the cell shapes
     GROMACS writes and on two flat triclinic cells GROMACS accepts, where
     reduction plus 27 neighbours returned images up to 1.13 nm too long (the
     cell of review QC-7, and one that needs every term of exact_radius).

C needs gmx (grompp for the fixture's run input, gmx rms inside rebound_rmsd).

Run:  python3 tests/test_qc_pbc.py
"""

import math, os, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IQC = os.path.join(REPO, "utils", "interface_qc.py")
RMSD = os.path.join(REPO, "utils", "rebound_rmsd.py")
failures = []


def check(name, ok, detail=""):
    print("  %-66s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


def dodecahedron(d):
    """GROMACS's xy-square rhombic dodecahedron, rows a, b, c."""
    return np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [d / 2, d / 2, d * math.sqrt(2) / 2]])


def wrap_atoms(x, box):
    """Every atom into the brick, as mdrun's put_atoms_in_box does: along c, b, a."""
    x = np.array(x, float)
    for m in (2, 1, 0):
        x -= np.floor(x[:, m] / box[m, m])[:, None] * box[m]
    return x


def write_gro(path, atoms, xyz, box, title="synthetic"):
    """atoms: [(resnum, resname, atomname)], numbers written modulo 100000 the way
    GROMACS writes them."""
    with open(path, "w") as f:
        f.write("%s\n%d\n" % (title, len(atoms)))
        for i, ((rn, res, an), (x, y, z)) in enumerate(zip(atoms, xyz)):
            f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
                    % (rn % 100000, res, an, (i + 1) % 100000, x, y, z))
        f.write("".join("%10.5f" % v for v in (box[0, 0], box[1, 1], box[2, 2], box[0, 1],
                                                   box[0, 2], box[1, 0], box[1, 2],
                                                   box[2, 0], box[2, 1])) + "\n")


def iqc(d, ref, query):
    r = subprocess.run([sys.executable, IQC, "-c", "contacts.gs", "--ref", ref,
                        "--query", query], cwd=d, capture_output=True, text=True)
    t = r.stdout.split()
    return dict(zip(t[0::2], t[1::2])), r


def num(res, key):
    try:
        return float(res.get(key, "nan"))
    except ValueError:
        return float("nan")


def intact(res):
    """Every contact back, no strain, side chains in place. The tolerances are the
    rounding of %8.3f coordinates, a box vector shifted in and out again."""
    return (num(res, "recovered") == 1.0 and num(res, "formed") == 1.0
            and num(res, "rms_dev") <= 0.0005 and num(res, "sc_rmsd") <= 0.005)


work = tempfile.mkdtemp(prefix="qc_pbc_")

# ---------------------------------------------------------------- interface fixture
# Two partners of 50 residues each, six atoms per residue, on a 10 x 5 grid; B sits
# 0.9 nm above A, so every CG of A has the CG of its B counterpart 0.4 nm away.
OFF = {"N": (-0.12, 0.0, 0.0), "CA": (0.0, 0.0, 0.0), "C": (0.12, 0.0, 0.0),
       "O": (0.15, 0.1, 0.0), "CB": (0.0, 0.0, 0.15), "CG": (0.0, 0.05, 0.25)}
NAMES = list(OFF)
solute_atoms, solute_xyz = [], []
for part, z0, sgn in (("A", 0.0, 1.0), ("B", 0.9, -1.0)):
    for k in range(50):
        rn = k + 1 if part == "A" else k + 51
        ca = np.array([0.5 * (k % 10), 0.5 * (k // 10), z0])
        for an in NAMES:
            o = np.array(OFF[an]) * np.array([1.0, 1.0, sgn])
            solute_atoms.append((rn, "ALA", an))
            solute_xyz.append(ca + o)
solute_xyz = np.array(solute_xyz)
NS = len(solute_atoms)                                   # 600
cg = [i for i, (_, _, an) in enumerate(solute_atoms) if an == "CG"]
pairs = list(zip(cg[:50], cg[50:]))                      # 0-based positions
with open(os.path.join(work, "contacts.gs"), "w") as f:
    f.write("# atomA atomB ref_nm resA resnumA nameA resB resnumB nameB frame_nm\n")
    for a, b in pairs:
        d = np.linalg.norm(solute_xyz[a] - solute_xyz[b])
        f.write("%d %d %.6f ALA%d %d CG ALA%d %d CG %.6f\n"
                % (a + 1, b + 1, d, a // 6 + 1, a // 6 + 1, b // 6 + 1, b // 6 + 1, d))

def rotation(axis, deg):
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


# The query's solute: every CG moved 0.1 nm along the body's x axis, so each CG-CG
# pair keeps its length while the side chains change; CB stays. After the backbone
# fit that is an RMSD of sqrt((100 * 0 + 100 * 0.1^2) / 200) nm over the 200
# side-chain atoms of the 100 interface residues.
FIXED_SC = 0.7071                                       # A
solute_def = solute_xyz.copy()
solute_def[[i for i, (_, _, an) in enumerate(solute_atoms) if an == "CG"]] += [0.1, 0.0, 0.0]


def moved(x, R, T):
    """x rotated by R about the solute's centroid, then translated by T."""
    c = solute_xyz.mean(axis=0)
    return (x - c) @ R.T + c + T


print("\nA. interface_qc reads atoms by position (100,600 atoms, numbers wrapped)")
big = dodecahedron(12.0)
grid = np.array([(x, y, z) for z in np.arange(3.5, 8.4, 0.18)
                 for y in np.arange(0.05, 11.9, 0.18) for x in np.arange(0.05, 11.9, 0.18)])
solvent = grid[:100000]
atoms_big = solute_atoms + [(101 + i, "SOL", "OW") for i in range(len(solvent))]
ref_big = solute_xyz + np.array([3.0, 4.0, 1.0])
qry_big = wrap_atoms(moved(solute_def, rotation([1, 2, 3], 50), [5.3, -2.1, 3.7]), big)
write_gro(os.path.join(work, "big.gro"), atoms_big, np.vstack([ref_big, solvent]), big)
# The query's solvent sits elsewhere, so any solvent atom read in place of a solute
# atom no longer cancels between the two frames.
write_gro(os.path.join(work, "bigq.gro"), atoms_big, np.vstack([qry_big, solvent + 0.09]), big)
write_gro(os.path.join(work, "small.gro"), solute_atoms, ref_big, big)
write_gro(os.path.join(work, "smallq.gro"), solute_atoms, qry_big, big)
res, proc = iqc(work, "big.gro", "bigq.gro")
res_s, _ = iqc(work, "small.gro", "smallq.gro")
with open(os.path.join(work, "bigq.gro")) as f:
    lines = f.readlines()
check("the frame really wraps: atom 100001 is written as 1",
      len(atoms_big) == 100600 and NS == 600 and lines[2 + 100000][15:20] == "    1")
check("recovered 1.0000 (old: solvent overwrote the solute)",
      num(res, "recovered") == 1.0, str(res) + proc.stderr[-200:])
check("formed 1.0000 and rms_dev 0.0000",
      num(res, "formed") == 1.0 and num(res, "rms_dev") <= 0.0005, str(res))
check("side-chain RMSD %.3f A, from solute atoms only" % FIXED_SC,
      abs(num(res, "sc_rmsd") - FIXED_SC) <= 0.005, str(res))
check("the same as the solute pair alone, without solvent",
      abs(num(res_s, "sc_rmsd") - FIXED_SC) <= 0.005
      and abs(num(res, "sc_rmsd") - num(res_s, "sc_rmsd")) <= 0.0015, "%s / %s" % (res, res_s))
check("all 50 restrained pairs measured", res.get("npairs") == "50", str(res))

print("\nB. interface_qc takes true minimum images in a dodecahedron")
box = dodecahedron(7.0)
ref_x = solute_xyz + np.array([2.0, 2.0, 1.0])
write_gro(os.path.join(work, "ref.gro"), solute_atoms, ref_x, box)
# The interface placed across the cell's corner and wrapped atom by atom, as a
# frame written straight from mdrun's state is: every pair that straddles a face
# is split by one box vector...
q = wrap_atoms(ref_x + np.array([5.6, 5.1, 4.4]), box)
write_gro(os.path.join(work, "split1.gro"), solute_atoms, q, box)
res1, _ = iqc(work, "ref.gro", "split1.gro")
check("pairs split by one box vector: every contact intact", intact(res1), str(res1))
# ...and B residues 61-70 two box vectors off, which whole-molecule output of an
# unbonded fragment in a triclinic cell can leave.
q2 = q.copy()
far = [i for i, (rn, _, _) in enumerate(solute_atoms) if 61 <= rn <= 70]
q2[far] += 2.0 * box[0] + box[2]
write_gro(os.path.join(work, "split2.gro"), solute_atoms, q2, box)
res2, _ = iqc(work, "ref.gro", "split2.gro")
check("a fragment two box vectors off: every contact intact", intact(res2), str(res2))
# A genuine change must still show: B lifted 0.2 nm off A.
q3 = ref_x.copy()
q3[NS // 2:] += np.array([0.0, 0.0, 0.2])
write_gro(os.path.join(work, "lift.gro"), solute_atoms, q3, box)
res3, _ = iqc(work, "ref.gro", "lift.gro")
check("B lifted 0.2 nm: no contact within 0.15, rms_dev 0.2000",
      num(res3, "recovered") == 0.0 and abs(num(res3, "rms_dev") - 0.2) <= 0.0005, str(res3))
# The interface reaches 5.3 nm from its first atom (A's first N to B's far corner),
# past even the 4.95 nm corners of this cell, so no gathering about one atom can
# keep it whole; rotated 37 degrees about z, not a symmetry of the lattice, the two
# frames would also come apart at different images.
far = max(np.linalg.norm(solute_xyz[i] - solute_xyz[0]) for i in range(NS))
q4 = wrap_atoms(moved(solute_def, rotation([0, 0, 1], 37), [3.3, 1.2, 3.2]), box)
write_gro(os.path.join(work, "rot.gro"), solute_atoms, q4, box)
res4, _ = iqc(work, "ref.gro", "rot.gro")
check("interface %.2f nm from its first atom, query rotated: sc %.3f A" % (far, FIXED_SC),
      far > 4.95 and num(res4, "recovered") == 1.0 and num(res4, "rms_dev") <= 0.0005
      and abs(num(res4, "sc_rmsd") - FIXED_SC) <= 0.005, str(res4))

# ---------------------------------------------------------------- rebound_rmsd fixture
print("\nC. rebound_rmsd reassembles fragments in space")
# A flat sheet of eight 8-residue strands, 0.5 nm apart, in a 6 nm dodecahedron.
# Chain A is three unbonded fragments at strand positions 0, 7 and 3, so its first
# two fragments are 3.5 nm apart in sequence order, more than half the 6 nm box;
# chain B's five fragments fill the strands between them.
FRAG_A, FRAG_B = [0, 7, 3], [1, 2, 4, 5, 6]
D_BOX = 6.0


def sheet():
    atoms, xyz, frag = [], [], []
    rn = 0
    for chain, pos in (("A", FRAG_A), ("B", FRAG_B)):
        for fi, k in enumerate(pos):
            for r in range(8):
                rn += 1
                ca = np.array([0.5 * k, 0.38 * r, 0.0])
                for an, o in (("N", (0.04, -0.13, 0.03)), ("CA", (0.0, 0.0, 0.0)),
                              ("C", (-0.04, 0.13, -0.03))):
                    atoms.append((rn, "ALA", an))
                    xyz.append(ca + np.array(o))
                    frag.append(len(set(frag)) if not frag else
                                (frag[-1] if r or an != "N" else frag[-1] + 1))
    return atoms, np.array(xyz), np.array(frag)


S_ATOMS, S_XYZ, S_FRAG = sheet()
NA = 3 * 8 * len(FRAG_A)


def itp(name, first, last):
    lines = ["[ moleculetype ]", "%s 3" % name, "", "[ atoms ]"]
    for i in range(first, last):
        rn, res, an = S_ATOMS[i]
        t, m = ("NX", 14.007) if an == "N" else ("CX", 12.011)
        lines.append("%6d %4s %5d %5s %5s %5d 0.0 %8.3f" % (i - first + 1, t, rn, res, an,
                                                          i - first + 1, m))
    lines += ["", "[ bonds ]"]
    for i in range(first, last - 1):
        if S_FRAG[i] == S_FRAG[i + 1]:                   # bonded only within a fragment
            lines.append("%6d %6d 1 0.14 100000" % (i - first + 1, i - first + 2))
    return "\n".join(lines) + "\n"


def gmx_fixture(d):
    """topol.top + chain itps + index.ndx + emin_solv.tpr, the files the rebound
    check finds in a structure directory. grompp needs a box: the reference frame."""
    open(os.path.join(d, "topol_Protein_chain_A.itp"), "w").write(itp("Protein_chain_A", 0, NA))
    open(os.path.join(d, "topol_Protein_chain_B.itp"), "w").write(
        itp("Protein_chain_B", NA, len(S_ATOMS)))
    open(os.path.join(d, "topol.top"), "w").write(
        "[ defaults ]\n1 2 no 1.0 1.0\n\n[ atomtypes ]\n"
        "NX 7 14.007 0.0 A 0.30 0.10\nCX 6 12.011 0.0 A 0.30 0.10\n\n"
        '#include "topol_Protein_chain_A.itp"\n#include "topol_Protein_chain_B.itp"\n\n'
        "[ system ]\nsynthetic sheet\n\n[ molecules ]\n"
        "Protein_chain_A 1\nProtein_chain_B 1\n")
    open(os.path.join(d, "fix.mdp"), "w").write(
        "integrator = md\nnsteps = 0\ncutoff-scheme = Verlet\npbc = xyz\n"
        "coulombtype = Cut-off\nrcoulomb = 0.9\nrvdw = 0.9\n")
    n = len(S_ATOMS)
    with open(os.path.join(d, "index.ndx"), "w") as f:
        for g in ("System", "Protein"):
            f.write("[ %s ]\n" % g)
            f.write("\n".join(" ".join("%d" % (i + 1) for i in range(s, min(s + 15, n)))
                              for s in range(0, n, 15)) + "\n")
    r = subprocess.run(["gmx", "grompp", "-f", "fix.mdp", "-c", "ref.gro", "-p", "topol.top",
                        "-o", "emin_solv.tpr", "-maxwarn", "20"],
                       cwd=d, capture_output=True, text=True,
                       env=dict(os.environ, GMX_MAXBACKUP="-1"))
    return r.returncode == 0 and os.path.isfile(os.path.join(d, "emin_solv.tpr")), r.stderr


def place(X, R, T, shifts, box):
    """Body coordinates X rotated, translated, then each fragment moved by its own
    lattice vector: whole fragments in scattered images, as mdrun writes them."""
    Y = X @ R.T + T
    for k, s in enumerate(shifts):
        Y[S_FRAG == k] += np.asarray(s, float) @ box
    return Y


def rmsd_cli(d, ref, query, tag):
    r = subprocess.run([sys.executable, RMSD, "--ref", ref, "--query", query,
                        "-s", "emin_solv.tpr", "-n", "index.ndx", "--tag", tag, "-v"],
                       cwd=d, capture_output=True, text=True)
    try:
        return float(r.stdout.split()[0]), r.stderr.strip()
    except (ValueError, IndexError):
        return float("nan"), r.stderr.strip()


def weighted_fit_rmsd(P, Q, w):
    """Mass-weighted RMSD after the optimal fit of P onto Q, in A."""
    w = np.asarray(w, float) / np.sum(w)
    pc, qc = w @ P, w @ Q
    P0, Q0 = P - pc, Q - qc
    V, _, Wt = np.linalg.svd((P0 * w[:, None]).T @ Q0)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(V @ Wt))])
    Rm = V @ D @ Wt
    return 10.0 * math.sqrt(np.sum(w * np.sum((P0 @ Rm - Q0) ** 2, axis=1)))


sbox = dodecahedron(D_BOX)
rot = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 120 deg about (1,1,1)
I3 = np.eye(3)
T_REF = np.array([1.0, 1.5, 2.0])
T_QRY = np.array([4.1, -3.3, 5.2])                     # more than half a box, every axis
NFRAG = int(S_FRAG.max()) + 1
SH_REF = [(0, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (-1, 1, 0), (0, -1, 1), (1, 1, -1)]
SH_QRY = [(1, 0, 0), (0, 0, 0), (-1, 1, 1), (2, 0, -1), (0, -1, 0), (1, 1, 0), (-1, 0, 1), (0, 2, -1)]
assert len(SH_REF) == len(SH_QRY) == NFRAG
# The deformed query: fragment 2 of chain A slid 0.3 nm along its strand.
X_DEF = S_XYZ.copy()
X_DEF[S_FRAG == 1] += np.array([0.0, 0.30, 0.0])
FIXED_DEFORMED = 0.8819         # A; mass-weighted fit of the exact coordinates

c_dir = os.path.join(work, "rmsd")
os.makedirs(c_dir)
write_gro(os.path.join(c_dir, "ref.gro"), S_ATOMS, place(S_XYZ, I3, T_REF, SH_REF, sbox), sbox)
ok_fix, err = gmx_fixture(c_dir)
check("fixture: grompp builds a run input for the fragmented complex", ok_fix,
      "" if ok_fix else err[-300:])
for name, X, R, T, sh in (("ref_whole", S_XYZ, I3, T_REF, [(0, 0, 0)] * NFRAG),
                          ("rigid_whole", S_XYZ, rot, T_QRY, [(0, 0, 0)] * NFRAG),
                          ("rigid_scatter", S_XYZ, rot, T_QRY, SH_QRY),
                          ("def_whole", X_DEF, rot, T_QRY, [(0, 0, 0)] * NFRAG),
                          ("def_scatter", X_DEF, rot, T_QRY, SH_QRY)):
    write_gro(os.path.join(c_dir, name + ".gro"), S_ATOMS, place(X, R, T, sh, sbox), sbox)
if ok_fix:
    v0, e0 = rmsd_cli(c_dir, "ref_whole.gro", "rigid_whole.gro", "t0")
    v1, e1 = rmsd_cli(c_dir, "ref.gro", "rigid_scatter.gro", "t1")
    check("rigid motion, nothing scattered: 0.000 A", abs(v0) < 0.02, "%.4f %s" % (v0, e0[-120:]))
    check("rigid motion, both frames scattered, 120 deg and > L/2 drift: 0.000 A",
          abs(v1) < 0.02, "%.4f %s" % (v1, e1[-160:]))
    v2, e2 = rmsd_cli(c_dir, "ref_whole.gro", "def_whole.gro", "t2")
    v3, e3 = rmsd_cli(c_dir, "ref.gro", "def_scatter.gro", "t3")
    check("one fragment displaced, unscattered: %.4f A" % FIXED_DEFORMED,
          abs(v2 - FIXED_DEFORMED) < 0.001, "%.4f" % v2)
    check("the same, scattered: equal to the unscattered value",
          abs(v3 - v2) < 0.001 and abs(v3 - FIXED_DEFORMED) < 0.001, "%.4f %s" % (v3, e3[-160:]))
    # GROMACS guesses united-atom masses from the names in a .gro: N 15.0147,
    # CA 13.019 (atommass.dat, AAA entries), C 12.0107. The fixed value is that
    # weighted fit on the exact coordinates; the files carry 3 decimals.
    m = [{"N": 15.0147, "CA": 13.019, "C": 12.0107}[an] for _, _, an in S_ATOMS]
    w_ind = weighted_fit_rmsd(X_DEF @ rot.T + T_QRY, S_XYZ, m)
    check("an independent mass-weighted fit gives %.4f A too" % FIXED_DEFORMED,
          abs(w_ind - FIXED_DEFORMED) < 0.001, "%.5f" % w_ind)
    left = [f for f in os.listdir(c_dir) if f.startswith(("rbref_", "rbqry_", "rmsd_"))]
    check("no scratch files left behind", not left, str(left))
    # 'never abort, nan on failure': an index without a Protein group.
    open(os.path.join(c_dir, "bad.ndx"), "w").write("[ System ]\n1 2 3\n")
    r = subprocess.run([sys.executable, RMSD, "--ref", "ref.gro", "--query", "def_scatter.gro",
                        "-n", "bad.ndx"], cwd=c_dir, capture_output=True, text=True)
    check("a failure prints NaN and exits 0", r.returncode == 0 and r.stdout.strip() == "NaN",
          "%d %r" % (r.returncode, r.stdout.strip()))

# ---------------------------------------------------------------- pbc_tools
print("\nD. pbc_tools.min_image against a brute-force lattice search")
sys.path.insert(0, os.path.join(REPO, "utils"))
try:
    import pbc_tools as pbc
except ImportError as e:
    pbc = None
    check("pbc_tools imports", False, str(e))
if pbc is not None:
    rng = np.random.default_rng(20261003)
    d7 = 7.0
    cells = {"dodecahedron": dodecahedron(d7),
             "rectangular": np.diag([5.0, 6.0, 7.0]),
             "triclinic": np.array([[6.0, 0, 0], [-2.9, 6.5, 0], [2.8, -3.1, 7.2]]),
             "octahedron": np.array([[d7, 0, 0], [d7 / 3, 2 * math.sqrt(2) / 3 * d7, 0],
                                     [-d7 / 3, math.sqrt(2) / 3 * d7, math.sqrt(6) / 3 * d7]]),
             # GROMACS-legal (|b_x|, |c_x| <= a_x/2, |c_y| <= b_y/2) but flat: its
             # shortest lattice vector, a + b + c, is 6.60 nm against rows of 8.89
             # nm and more, and the 27 neighbours missed images up to 1.13 nm.
             "flat triclinic (QC-7)": np.array([[11.898, 0, 0], [-5.949, 11.423, 0],
                                                [-5.949, -5.712, 3.316]]),
             # Also legal, flat along y: the 27 neighbours miss images from 3.84 nm
             # on, below 1.5 b_y = 4.5 nm, so only the |c_y| term of exact_radius
             # (3.09 nm here) sends them on to the second search.
             "flat triclinic, wide a": np.array([[13.92, 0, 0], [3.659, 3.0, 0],
                                                 [0.277, -1.414, 4.253]])}
    co = np.array([(i, j, k) for i in range(-6, 7) for j in range(-6, 7)
                   for k in range(-6, 7)], float)
    for name, B in cells.items():
        dv = rng.uniform(-40.0, 40.0, (4000, 3))
        got = pbc.min_image(dv, B)
        base = dv - np.round(dv @ np.linalg.inv(B)) @ B
        best = np.concatenate([np.sqrt(((base[s:s + 400, None, :] + (co @ B)[None]) ** 2)
                                       .sum(-1)).min(axis=1) for s in range(0, len(dv), 400)])
        frac = (got - dv) @ np.linalg.inv(B)
        check("%s: shortest image of 4000 differences up to 40 nm" % name,
              np.abs(np.linalg.norm(got, axis=1) - best).max() < 1e-9
              and np.abs(frac - np.round(frac)).max() < 1e-9,
              "worst excess %.3g nm" % (np.linalg.norm(got, axis=1) - best).max())

shutil.rmtree(work, ignore_errors=True)
print("")
if failures:
    print("FAILED (%d): %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("All PBC diagnostic checks passed.")
