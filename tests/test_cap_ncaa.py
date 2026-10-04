#!/usr/bin/env python3
"""cap_termini.py relaxes the caps of a structure that holds residues amber14 has
no template for: fused glycans (NG1), NCAAs, and the CYM a cutout carries without
hydrogens.

OpenMM bonds only the residues it knows, and amber14-all has templates only for
those. Before v1.5 such a residue got neither internal bonds nor the peptide bond
from the residue before it, createSystem failed on that neighbour ('No template
found for residue 34 (THR) ... missing 1 C atom'), cap_termini.py caught the
exception and wrote EVERY cap of the structure at PDBFixer's raw position. Now the
bonds OpenMM left out are made (inside a residue without any bond, by covalent
distance r1 + r2 + 0.35 A; the peptide bond C(i)-N(i+1) below 0.2 nm, or next to
a cap) and a template generator gives each residue that still matches no template
one from its own atoms and bonds: N, CA, C, O amber14 backbone types, every other
atom an uncharged Lennard-Jones sphere of its element, everything massless.

Cap contacts below are heavy-atom pairs between a cap atom and an atom of another
residue, the cap's own C-N junction bond left out (the measure of the glycan
design's critic, v15_work/G/critic). Pinned:

  1. The fused 2AJF full model (the glycan prototype's fixed.pdb: four NG1, each
     ASN + NAG, 22 heavy atoms): no 'Cap minimization failed', one generic
     template for the four NG1, 92 bonds made (23 per NG1), no cap contact below
     1.8 A (pre-change: 7, the closest 1.11 A; now the closest is 1.83 A), and all
     6328 non-cap atoms, the 88 NG1 atoms among them, keep their coordinate text.
  2. Its cutout: ACE 237, the cap in front of NG1 238, at least 3 A from every
     sugar atom (pre-change 1.26 A, inside the NAG ring; now 5.63 A), at most one
     contact below 1.8 A (pre-change 16, down to 0.85 A; the one left is an NME N
     1.08 A from the carbonyl O before it, the pattern behind every sub-1.8 A
     contact the unchanged benchmark structures keep, e.g. 0.84 A in the 1S1Q
     cutout), all 3216 non-cap atoms in place.
  3. The 1E6E cutout (four CYM without hydrogens, no NCAA) and the 2MTA cutout
     (TRQ and one CYM): both minimise (pre-change: 'No template found for residue
     440 (ALA)' and '250 (SER)'), 24 bonds made in each, contacts below 1.8 A at
     most 1 and 2 (pre-change 9 and 20), all 3839 and 3196 non-cap atoms in place.
  4. HIC (4-methylhistidine, 2BTF) twice under one name, complete (11 heavy atoms)
     and truncated (8): two generic templates, 20 bonds made, all 105 atoms in
     place, and two runs with different hash seeds and thread settings write the
     same bytes. With the GROMOS cap flags (--ace-only --rename-ace-carbon) the
     one-copy fragment is templated and its ACE minimised as well.
  5. A fluorinated NCAA (PFF, 4-fluoro-Phe) and one with an element outside the
     Lennard-Jones table (BPX, a para boronic acid): both templated, 28 bonds made,
     minimised. So is PFF whose F, renamed Q1 with a blank element column, has no
     element at all.
  6. The two numeric rules from both sides. F 1.66 A from CZ is bonded, 1.70 A is
     not (C 0.76 + F 0.57 + 0.35 = 1.68 A): 13 against 12 bonds made. HIC's N 1.90
     A from GLU's C gets the peptide bond, 2.10 A does not (0.2 nm): 12 against 11
     bonds, and at 2.10 A the GLU, its C now without an external bond, is
     templated generically too.
  7. The same PHE fragments as in 5 with standard residues: byte for byte the
     output of the pre-change script (tests/data/caps/phe_pair_capped_ref.pdb),
     with no bond made and no template. This check passes before the change by
     construction; it pins that inputs without such residues are untouched.
  8. A failure that remains (injected: loading any generic template raises) is a
     warning that names the residue ('no generic template for HIC A 72: ...'),
     rc 0, and the capped PDB is still written with PDBFixer's caps. Pre-change
     the warning named only the neighbour, by its index ('residue 2 (GLU)').

Inputs: tests/data/caps holds the fused 2AJF fixed.pdb and cutout.pdb of the glycan
prototype (v15_work/G/design/run/2AJF_deposited) and the 1E6E and 2MTA cutout.pdb
of the v1.5 audit run (v15_work/R/shared/root/bench), gzipped byte for byte; the
fragments in 4-8 are cut from the same audit run's 2BTF and 1GCQ fixed.pdb. PDB
data are CC0. The pre-change numbers were measured with the v1.5 cap_termini.py as
it stood before this change (sha256 a3bfdc7d), OpenMM 8.5.0, CPU platform, one
thread; so was the reference of 7, which a different OpenMM build may round
differently.

Run:  python3 tests/test_cap_ncaa.py   (about 2 min: four whole structures)
"""

import gzip, os, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "utils", "cap_termini.py")
DATA = os.path.join(REPO, "tests", "data", "caps")
failures = []

ASN_ATOMS = {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"}
CAPS = ("ACE", "NME")


def check(name, ok, detail=""):
    detail = " | ".join(l.strip() for l in str(detail).splitlines() if l.strip())
    print("  %-72s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


def run(pdb_text, env_extra=None, flags=(), pre=None):
    """One cap_termini.py run in a fresh directory: (rc, capped text, log). With
    `pre`, that Python code runs first in the same process (a fault to inject)."""
    d = tempfile.mkdtemp(prefix="capncaa_")
    try:
        open(os.path.join(d, "input.pdb"), "w").write(pdb_text)
        open(os.path.join(d, "chain_map.gs"), "w").write("# Sequential residue numbers belonging to protein B\n")
        env = dict(os.environ, **(env_extra or {}))
        argv = [SCRIPT, "-f", "input.pdb", "-o", "capped.pdb", "-m", "chain_map.gs"] + list(flags)
        cmd = [sys.executable] + argv
        if pre:
            cmd = [sys.executable, "-c", pre + "\nimport runpy, sys\nsys.argv = %r\n"
                   "runpy.run_path(%r, run_name='__main__')\n" % (argv, SCRIPT)]
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=env)
        out = open(os.path.join(d, "capped.pdb")).read() if os.path.isfile(os.path.join(d, "capped.pdb")) else ""
        return p.returncode, out, p.stdout + p.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)


def residues(text):
    """[(resname, chain, resnum, [(atom, 'x y z text')])] in file order."""
    out, key = [], None
    for l in text.splitlines():
        if l.startswith(("ATOM", "HETATM")):
            k = (l[17:20], l[21], l[22:27])
            if k != key:
                key = k
                out.append((l[17:20].strip(), l[21], int(l[22:26]), []))
            out[-1][3].append((l[12:16].strip(), l[30:54]))
    return out


def xyz(s):
    return np.array([float(s[0:8]), float(s[8:16]), float(s[16:24])])


def cap_contacts(text):
    """Distances (A) between a cap heavy atom and a heavy atom of another residue,
    below 2.2 A, the cap's C-N junction bond left out; sorted."""
    rows = [(r[0], i, a, xyz(s)) for i, r in enumerate(residues(text)) for a, s in r[3] if not a.startswith("H")]
    X = np.array([r[3] for r in rows])
    out = []
    for i, (rn, ri, a, x) in enumerate(rows):
        if rn not in CAPS:
            continue
        d = np.linalg.norm(X - x, axis=1)
        for j in np.nonzero(d < 2.2)[0]:
            if rows[j][1] == ri:
                continue
            if {a, rows[j][2]} == {"C", "N"} and abs(rows[j][1] - ri) == 1:
                continue
            out.append(float(d[j]))
    return sorted(out)


def unmoved(inp, out):
    """(kept, moved, missing): every atom of every multi-atom input residue against
    the same atom of the matching non-cap output residue. Ions (one atom) are
    written after the chains and are left out; OXT, H2 and H3 next to a cap may
    be removed by design, and an atom PDBFixer renamed in place (the cutouts' ILE
    CD, written as CD1 before and after this change) counts as kept."""
    res_in = [r for r in residues(inp) if len(r[3]) > 1]
    res_out = [r for r in residues(out) if r[0] not in CAPS and len(r[3]) > 1]
    kept = moved = missing = 0
    for ri, ro in zip(res_in, res_out):
        have = dict(ro[3])
        texts = set(have.values())
        for name, s in ri[3]:
            if have.get(name) == s or (name not in have and s in texts):
                kept += 1
            elif name in have:
                moved += 1
            else:
                missing += name not in ("OXT", "H2", "H3")
    if len(res_in) != len(res_out):
        missing += 1000
    return kept, moved, missing


def lines(log, prefix):
    return [l.strip() for l in log.splitlines() if l.strip().startswith(prefix)]


def logged(log):
    """(failed, templates line, bonds made) of a cap_termini.py log."""
    t = lines(log, "Cap minimization: generic templates for ")
    b = lines(log, "Cap minimization: made ")
    return ("Cap minimization failed" in log, t[0][len("Cap minimization: generic templates for "):] if t else "",
            int(b[0].split()[3]) if b else 0)


def vendored(name):
    with gzip.open(os.path.join(DATA, name), "rt") as f:
        return f.read()


# ---------------------------------------------------------------- fragments
# 2BTF chain A 70-76 (fixed.pdb of the audit run), HIC 72 complete.
HIC_FRAGMENT = """\
ATOM    506  N   ILE A  70      33.873  57.282  51.694  1.00  0.00           N
ATOM    507  CA  ILE A  70      34.077  57.658  53.064  1.00  0.00           C
ATOM    508  C   ILE A  70      34.684  56.422  53.658  1.00  0.00           C
ATOM    509  O   ILE A  70      34.200  55.323  53.401  1.00  0.00           O
ATOM    510  CB  ILE A  70      32.704  58.026  53.670  1.00  0.00           C
ATOM    511  CG1 ILE A  70      31.945  59.068  52.885  1.00  0.00           C
ATOM    512  CG2 ILE A  70      32.931  58.675  54.970  1.00  0.00           C
ATOM    513  CD1 ILE A  70      30.392  58.974  53.048  1.00  0.00           C
ATOM    514  N   GLU A  71      35.677  56.580  54.513  1.00  0.00           N
ATOM    515  CA  GLU A  71      36.445  55.471  55.022  1.00  0.00           C
ATOM    516  C   GLU A  71      36.951  55.750  56.424  1.00  0.00           C
ATOM    517  O   GLU A  71      37.711  56.689  56.629  1.00  0.00           O
ATOM    518  CB  GLU A  71      37.539  55.307  54.040  1.00  0.00           C
ATOM    519  CG  GLU A  71      38.525  54.216  54.314  1.00  0.00           C
ATOM    520  CD  GLU A  71      39.340  53.937  53.073  1.00  0.00           C
ATOM    521  OE1 GLU A  71      38.708  53.577  52.049  1.00  0.00           O
ATOM    522  OE2 GLU A  71      40.572  54.100  53.147  1.00  0.00           O
ATOM    523  N   HIC A  72      36.576  54.962  57.432  1.00  0.00           N
ATOM    524  CA  HIC A  72      36.905  55.235  58.813  1.00  0.00           C
ATOM    525  C   HIC A  72      36.440  56.637  59.173  1.00  0.00           C
ATOM    526  O   HIC A  72      37.023  57.318  60.017  1.00  0.00           O
ATOM    527  CB  HIC A  72      38.391  55.124  59.068  1.00  0.00           C
ATOM    528  CG  HIC A  72      38.811  53.664  58.958  1.00  0.00           C
ATOM    529  ND1 HIC A  72      38.271  52.576  59.534  1.00  0.00           N
ATOM    530  CD2 HIC A  72      39.822  53.216  58.134  1.00  0.00           C
ATOM    531  CE1 HIC A  72      38.895  51.516  59.085  1.00  0.00           C
ATOM    532  NE2 HIC A  72      39.836  51.904  58.245  1.00  0.00           N
ATOM    533  CZ  HIC A  72      40.825  51.000  57.582  1.00  0.00           C
ATOM    534  N   GLY A  73      35.315  57.019  58.547  1.00  0.00           N
ATOM    535  CA  GLY A  73      34.696  58.318  58.733  1.00  0.00           C
ATOM    536  C   GLY A  73      35.227  59.451  57.866  1.00  0.00           C
ATOM    537  O   GLY A  73      34.512  60.448  57.733  1.00  0.00           O
ATOM    538  N   ILE A  74      36.435  59.317  57.281  1.00  0.00           N
ATOM    539  CA  ILE A  74      37.135  60.362  56.548  1.00  0.00           C
ATOM    540  C   ILE A  74      36.986  60.164  55.040  1.00  0.00           C
ATOM    541  O   ILE A  74      36.992  59.016  54.609  1.00  0.00           O
ATOM    542  CB  ILE A  74      38.608  60.320  56.954  1.00  0.00           C
ATOM    543  CG1 ILE A  74      38.764  60.301  58.477  1.00  0.00           C
ATOM    544  CG2 ILE A  74      39.318  61.500  56.275  1.00  0.00           C
ATOM    545  CD1 ILE A  74      38.149  61.413  59.326  1.00  0.00           C
ATOM    546  N   VAL A  75      36.883  61.192  54.174  1.00  0.00           N
ATOM    547  CA  VAL A  75      36.640  61.002  52.745  1.00  0.00           C
ATOM    548  C   VAL A  75      38.008  60.891  52.109  1.00  0.00           C
ATOM    549  O   VAL A  75      38.922  61.692  52.361  1.00  0.00           O
ATOM    550  CB  VAL A  75      35.810  62.197  52.175  1.00  0.00           C
ATOM    551  CG1 VAL A  75      35.731  62.202  50.657  1.00  0.00           C
ATOM    552  CG2 VAL A  75      34.373  62.051  52.676  1.00  0.00           C
ATOM    553  N   THR A  76      38.119  59.732  51.428  1.00  0.00           N
ATOM    554  CA  THR A  76      39.331  59.344  50.716  1.00  0.00           C
ATOM    555  C   THR A  76      39.202  59.671  49.237  1.00  0.00           C
ATOM    556  O   THR A  76      40.193  59.886  48.534  1.00  0.00           O
ATOM    557  CB  THR A  76      39.615  57.824  50.884  1.00  0.00           C
ATOM    558  OG1 THR A  76      38.496  57.020  50.453  1.00  0.00           O
ATOM    559  CG2 THR A  76      39.937  57.557  52.352  1.00  0.00           C
"""

# 1GCQ chain B 22-26 (fixed.pdb of the audit run), PHE 24 in the middle.
PHE_FRAGMENT = """\
ATOM    178  N   GLY B  22       8.248  17.532  43.525  1.00  0.00           N
ATOM    179  CA  GLY B  22       8.882  17.288  44.807  1.00  0.00           C
ATOM    180  C   GLY B  22       8.992  15.816  45.167  1.00  0.00           C
ATOM    181  O   GLY B  22       9.834  15.455  45.996  1.00  0.00           O
ATOM    182  N   ASP B  23       8.168  14.959  44.571  1.00  0.00           N
ATOM    183  CA  ASP B  23       8.227  13.524  44.860  1.00  0.00           C
ATOM    184  C   ASP B  23       9.524  12.894  44.399  1.00  0.00           C
ATOM    185  O   ASP B  23      10.087  13.277  43.372  1.00  0.00           O
ATOM    186  CB  ASP B  23       7.110  12.760  44.159  1.00  0.00           C
ATOM    187  CG  ASP B  23       5.776  12.929  44.822  1.00  0.00           C
ATOM    188  OD1 ASP B  23       5.706  13.466  45.934  1.00  0.00           O
ATOM    189  OD2 ASP B  23       4.776  12.528  44.211  1.00  0.00           O
ATOM    190  N   PHE B  24       9.985  11.903  45.150  1.00  0.00           N
ATOM    191  CA  PHE B  24      11.188  11.180  44.785  1.00  0.00           C
ATOM    192  C   PHE B  24      10.736   9.900  44.103  1.00  0.00           C
ATOM    193  O   PHE B  24       9.825   9.209  44.576  1.00  0.00           O
ATOM    194  CB  PHE B  24      12.055  10.915  46.012  1.00  0.00           C
ATOM    195  CG  PHE B  24      12.645  12.163  46.587  1.00  0.00           C
ATOM    196  CD1 PHE B  24      13.687  12.811  45.934  1.00  0.00           C
ATOM    197  CD2 PHE B  24      12.138  12.717  47.759  1.00  0.00           C
ATOM    198  CE1 PHE B  24      14.221  13.991  46.428  1.00  0.00           C
ATOM    199  CE2 PHE B  24      12.669  13.911  48.272  1.00  0.00           C
ATOM    200  CZ  PHE B  24      13.711  14.546  47.604  1.00  0.00           C
ATOM    201  N   ILE B  25      11.366   9.601  42.975  1.00  0.00           N
ATOM    202  CA  ILE B  25      11.026   8.441  42.172  1.00  0.00           C
ATOM    203  C   ILE B  25      12.212   7.512  42.069  1.00  0.00           C
ATOM    204  O   ILE B  25      13.338   7.948  41.840  1.00  0.00           O
ATOM    205  CB  ILE B  25      10.615   8.906  40.746  1.00  0.00           C
ATOM    206  CG1 ILE B  25       9.386   9.827  40.834  1.00  0.00           C
ATOM    207  CG2 ILE B  25      10.348   7.706  39.832  1.00  0.00           C
ATOM    208  CD1 ILE B  25       9.208  10.755  39.643  1.00  0.00           C
ATOM    209  N   HIS B  26      11.956   6.231  42.290  1.00  0.00           N
ATOM    210  CA  HIS B  26      12.990   5.219  42.197  1.00  0.00           C
ATOM    211  C   HIS B  26      13.019   4.824  40.731  1.00  0.00           C
ATOM    212  O   HIS B  26      12.047   4.243  40.232  1.00  0.00           O
ATOM    213  CB  HIS B  26      12.607   4.000  43.037  1.00  0.00           C
ATOM    214  CG  HIS B  26      13.676   2.953  43.101  1.00  0.00           C
ATOM    215  ND1 HIS B  26      14.379   2.679  44.259  1.00  0.00           N
ATOM    216  CD2 HIS B  26      14.169   2.121  42.156  1.00  0.00           C
ATOM    217  CE1 HIS B  26      15.259   1.727  44.018  1.00  0.00           C
ATOM    218  NE2 HIS B  26      15.155   1.369  42.751  1.00  0.00           N
"""

# Para substituents of PHE 24 (and of its copy 30 A along x), on the CG-CZ axis
# and in the ring plane: F 1.35 A from CZ; B 1.57 A from CZ with two O 1.36 A off
# at 120 deg; F 1.66 A and 1.70 A from CZ for the bonding limit of 1.68 A.
F_135 = (14.225, 15.694, 48.094)
B_157, OB1, OB2 = (44.308, 15.881, 48.174), (43.807, 16.416, 49.319), (45.327, 16.503, 47.522)
F_166 = (14.343, 15.958, 48.207)
F_170 = (14.358, 15.992, 48.221)


def atom_line(name, res, chain, num, x, el):
    nf = (" %-3s" % name) if len(name) < 4 else name
    return ("ATOM  %5d %s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
            % (0, nf, res, chain, num, x[0], x[1], x[2], el))


def pdb(*fragments):
    """Fragments (lists of ATOM lines) as one PDB, TER between them, serials renumbered."""
    out, n = [], 0
    for frag in fragments:
        for l in frag:
            n += 1
            out.append(l[:6] + "%5d" % n + l[11:])
        out.append("TER")
    return "\n".join(out + ["END"]) + "\n"


def frag(text, chain=None, dnum=0, shift=(0.0, 0.0, 0.0), rename=None, drop=(), add=(), move_res=None):
    """ATOM lines of `text`, optionally on another chain, renumbered by dnum, moved
    by shift (A); rename {resnum: name}; drop {(resnum, atom)}; add [(resnum, line)]
    appended in order behind that residue's last atom (resnum before dnum);
    move_res (resnum, vector): one residue moved rigidly."""
    out = []
    for l in text.splitlines():
        num = int(l[22:26])
        name = l[12:16].strip()
        if (num, name) in drop:
            continue
        x = xyz(l[30:54]) + np.array(shift)
        if move_res and move_res[0] == num:
            x = x + move_res[1]
        res = (rename or {}).get(num, l[17:20])
        out.append(l[:17] + "%3s" % res + " " + (chain or l[21]) + "%4d" % (num + dnum) + l[26:30]
                   + "%8.3f%8.3f%8.3f" % tuple(x) + l[54:])
    for after, line in add:
        idx = max(i for i, l in enumerate(out) if int(l[22:26]) == after + dnum)
        out.insert(idx + 1, line)
    return out


def main():
    if not os.path.isfile(SCRIPT):
        print("missing %s" % SCRIPT)
        sys.exit(1)

    print("[1] the fused 2AJF full model: four NG1 (ASN + NAG)")
    inp = vendored("2AJF_fused_fixed.pdb.gz")
    rc, out, log = run(inp)
    failed, templ, made = logged(log)
    check("[1] rc 0, no 'Cap minimization failed'", rc == 0 and not failed,
          next((l for l in log.splitlines() if "failed" in l), "")[:160])
    check("[1] one generic template for the four NG1", templ == "NG1", templ)
    check("[1] 92 bonds made, 23 per NG1 (22 inside, 1 peptide)", made == 92, str(made))
    c = cap_contacts(out)
    check("[1] no cap contact below 1.8 A (pre-change 7, closest 1.11 A)",
          out != "" and not [d for d in c if d < 1.8], "below 2.2 A: %s" % " ".join("%.2f" % d for d in c))
    kept, moved, missing = unmoved(inp, out)
    ng1 = sum(1 for r in residues(out) if r[0] == "NG1" for _ in r[3])
    check("[1] all 6328 non-cap atoms keep their coordinate text, the 88 NG1 atoms too",
          moved == 0 and missing == 0 and kept == 6328 and ng1 == 88,
          "kept %d, moved %d, missing %d, NG1 atoms %d" % (kept, moved, missing, ng1))

    print("[2] its cutout: ACE 237 in front of NG1 238")
    inp = vendored("2AJF_fused_cutout.pdb.gz")
    rc, out, log = run(inp)
    failed, templ, made = logged(log)
    check("[2] rc 0, no 'Cap minimization failed', generic template for NG1",
          rc == 0 and not failed and templ == "NG1", templ)
    res = residues(out)
    pairs = []
    for prev, cur in zip(res, res[1:]):
        if prev[0] == "ACE" and cur[0] == "NG1":
            sugar = [xyz(s) for a, s in cur[3] if a not in ASN_ATOMS]
            pairs.append((prev[2], cur[2], min(np.linalg.norm(xyz(s) - g) for _, s in prev[3] for g in sugar)))
    check("[2] ACE 237 at least 3 A from every sugar atom of NG1 238 (pre-change 1.26 A)",
          len(pairs) == 1 and pairs[0][:2] == (237, 238) and pairs[0][2] >= 3.0,
          " ".join("ACE %d - NG1 %d: %.2f A" % p for p in pairs))
    c = cap_contacts(out)
    check("[2] at most 1 cap contact below 1.8 A (pre-change 16, down to 0.85 A)",
          out != "" and len([d for d in c if d < 1.8]) <= 1, " ".join("%.2f" % d for d in c))
    kept, moved, missing = unmoved(inp, out)
    check("[2] all 3216 non-cap atoms keep their coordinate text", moved == 0 and missing == 0 and kept == 3216,
          "kept %d, moved %d, missing %d" % (kept, moved, missing))

    print("[3] cutouts with CYM without hydrogens: 1E6E (4 CYM), 2MTA (TRQ, CYM)")
    for name, want_t, want_b, most, before, natoms in (("1E6E_cutout.pdb.gz", "CYM", 24, 1, 9, 3839),
                                                       ("2MTA_cutout.pdb.gz", "TRQ, CYM", 24, 2, 20, 3196)):
        inp = vendored(name)
        rc, out, log = run(inp)
        failed, templ, made = logged(log)
        tag = name.split("_")[0]
        check("[3] %s: rc 0, no 'Cap minimization failed'" % tag, rc == 0 and not failed,
              next((l for l in log.splitlines() if "failed" in l), "")[:160])
        check("[3] %s: generic templates for %s, %d bonds made" % (tag, want_t, want_b),
              templ == want_t and made == want_b, "%s / %d" % (templ, made))
        c = cap_contacts(out)
        check("[3] %s: at most %d cap contact(s) below 1.8 A (pre-change %d)" % (tag, most, before),
              out != "" and len([d for d in c if d < 1.8]) <= most, " ".join("%.2f" % d for d in c))
        kept, moved, missing = unmoved(inp, out)
        check("[3] %s: all %d non-cap atoms keep their coordinate text" % (tag, natoms),
              moved == 0 and missing == 0 and kept == natoms, "kept %d, moved %d, missing %d" % (kept, moved, missing))

    print("[4] one NCAA name, two atom counts: HIC complete and truncated")
    hic_pair = pdb(frag(HIC_FRAGMENT),
                   frag(HIC_FRAGMENT, chain="B", dnum=100, shift=(30.0, 0.0, 0.0),
                        drop={(72, "CE1"), (72, "NE2"), (72, "CZ")}))
    rc1, out1, log1 = run(hic_pair, {"PYTHONHASHSEED": "1", "OPENMM_CPU_THREADS": "8"})
    rc2, out2, log2 = run(hic_pair, {"PYTHONHASHSEED": "2"})
    failed, templ, made = logged(log1)
    check("[4] rc 0, no 'Cap minimization failed'", rc1 == 0 and rc2 == 0 and not failed,
          next((l for l in log1.splitlines() if "failed" in l), "")[:160])
    check("[4] two generic templates for HIC, 20 bonds made (11 + 1, 7 + 1)",
          templ == "HIC (2 templates)" and made == 20, "%s / %d" % (templ, made))
    check("[4] the caps were minimised", "Minimized 4 cap residue(s)" in log1)
    kept, moved, missing = unmoved(hic_pair, out1)
    check("[4] all 105 input atoms keep their coordinate text", kept == 105 and moved == 0 and missing == 0,
          "kept %d, moved %d, missing %d" % (kept, moved, missing))
    check("[4] byte-identical output under hash seeds 1 / 2, threads 8 / default", out1 == out2 and out1 != "")
    rc, out, log = run(pdb(frag(HIC_FRAGMENT)), flags=["--ace-only", "--rename-ace-carbon"])
    failed, templ, made = logged(log)
    check("[4] GROMOS caps (--ace-only): HIC templated, 12 bonds made, the ACE minimised",
          rc == 0 and not failed and templ == "HIC" and made == 12 and "Minimized 1 cap residue(s)" in log,
          "%s / %d" % (templ, made))

    print("[5] a fluorinated NCAA and one with an element outside the table")
    halo = pdb(frag(PHE_FRAGMENT, rename={24: "PFF"}, add=[(24, atom_line("F", "PFF", "B", 24, F_135, "F"))]),
               frag(PHE_FRAGMENT, chain="C", dnum=100, shift=(30.0, 0.0, 0.0), rename={24: "BPX"},
                    add=[(24, atom_line("B", "BPX", "C", 124, B_157, "B")),
                         (24, atom_line("OB1", "BPX", "C", 124, OB1, "O")),
                         (24, atom_line("OB2", "BPX", "C", 124, OB2, "O"))]))
    rc, out, log = run(halo)
    failed, templ, made = logged(log)
    check("[5] rc 0, no 'Cap minimization failed', caps minimised",
          rc == 0 and not failed and "Minimized 4 cap residue(s)" in log,
          next((l for l in log.splitlines() if "failed" in l), "")[:160])
    check("[5] generic templates for PFF and BPX, 28 bonds made (12 + 1, 14 + 1)",
          templ == "PFF, BPX" and made == 28, "%s / %d" % (templ, made))
    kept, moved, missing = unmoved(halo, out)
    check("[5] all 86 input atoms keep their coordinate text", kept == 86 and moved == 0 and missing == 0,
          "kept %d, moved %d, missing %d" % (kept, moved, missing))
    # PFF's F renamed Q1 with a blank element column: OpenMM reads no element for it.
    rc, out, log = run(pdb(frag(PHE_FRAGMENT, rename={24: "PFF"},
                                add=[(24, atom_line("Q1", "PFF", "B", 24, F_135, ""))])))
    failed, templ, made = logged(log)
    check("[5] an atom without an element: PFF templated, 13 bonds made, caps minimised",
          rc == 0 and not failed and templ == "PFF" and made == 13 and "Minimized 2 cap residue(s)" in log,
          "%s / %d" % (templ, made))

    print("[6] the bonding limits from both sides")
    for f_xyz, want, label in ((F_166, 13, "F 1.66 A from CZ: bonded"), (F_170, 12, "F 1.70 A from CZ: not bonded")):
        rc, out, log = run(pdb(frag(PHE_FRAGMENT, rename={24: "PFF"},
                                    add=[(24, atom_line("F", "PFF", "B", 24, f_xyz, "F"))])))
        failed, templ, made = logged(log)
        check("[6] %s, %d bonds made" % (label, want), rc == 0 and not failed and made == want and templ == "PFF",
              "%s / %d" % (templ, made))
    # HIC 72 moved rigidly along GLU 71 C -> HIC 72 N (1.333 A in 2BTF) to 1.90 / 2.10 A.
    c71, n72 = np.array([36.951, 55.750, 56.424]), np.array([36.576, 54.962, 57.432])
    u = (n72 - c71) / np.linalg.norm(n72 - c71)
    for dist, want, want_t, label in ((1.90, 12, "HIC", "C-N 1.90 A: peptide bond made"),
                                      (2.10, 11, "GLU, HIC", "C-N 2.10 A: no peptide bond, GLU templated")):
        moved_pdb = pdb(frag(HIC_FRAGMENT, move_res=(72, (dist - np.linalg.norm(n72 - c71)) * u)))
        rc, out, log = run(moved_pdb)
        failed, templ, made = logged(log)
        got = [l for l in moved_pdb.splitlines() if l[17:20] == "HIC" and l[12:16] == " N  "][0]
        d = np.linalg.norm(xyz(got[30:54]) - c71)
        check("[6] %s (%.3f A), %d bonds made" % (label, d, want),
              rc == 0 and not failed and made == want and templ == want_t, "%s / %d" % (templ, made))

    print("[7] standard residues: byte for byte the pre-change output")
    phe_pair = pdb(frag(PHE_FRAGMENT), frag(PHE_FRAGMENT, chain="C", dnum=100, shift=(30.0, 0.0, 0.0)))
    rc, out, log = run(phe_pair)
    ref = open(os.path.join(DATA, "phe_pair_capped_ref.pdb")).read()
    first = next((i for i, (a, b) in enumerate(zip(out.splitlines(), ref.splitlines())) if a != b), None)
    check("[7] identical to tests/data/caps/phe_pair_capped_ref.pdb", rc == 0 and out == ref,
          "first difference at line %s" % (first + 1 if first is not None else "end"))
    check("[7] no bond made, no generic template",
          not lines(log, "Cap minimization: made") and not lines(log, "Cap minimization: generic"))

    print("[8] a failure that remains: warned, the residue named, the output written")
    # Any force field loaded from a string buffer, i.e. every generic template, fails.
    inject = ("import io, openmm.app as app\n_load = app.ForceField.loadFile\n"
              "def _fail(self, files, *a, **k):\n"
              "    if isinstance(files, io.StringIO):\n"
              "        raise ValueError('injected failure')\n"
              "    return _load(self, files, *a, **k)\n"
              "app.ForceField.loadFile = _fail\n")
    rc, out, log = run(pdb(frag(HIC_FRAGMENT)), pre=inject)
    warn = lines(log, "Warning: Cap minimization failed")
    check("[8] rc 0 and one warning naming HIC A 72 with the reason",
          rc == 0 and len(warn) == 1 and "no generic template for HIC A 72: injected failure" in warn[0],
          (warn or [log[-200:]])[0][-120:])
    res = residues(out)
    check("[8] the output is written: 9 residues (ACE, 7, NME), 59 atoms, caps as PDBFixer put them",
          [r[0] for r in res] == ["ACE", "ILE", "GLU", "HIC", "GLY", "ILE", "VAL", "THR", "NME"]
          and sum(len(r[3]) for r in res) == 59, " ".join(r[0] for r in res))

    print()
    if failures:
        print("%d check(s) FAILED:" % len(failures))
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
