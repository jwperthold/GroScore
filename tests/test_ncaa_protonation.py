#!/usr/bin/env python3
"""Existing NCAAs get their pH 7.4 states, by the ligand path's rules (ncaa review P1).

parametrize_ncaa.py built every ordinary NCAA from its CCD entry, the neutral
molecule, and RDKit's AddHs kept it neutral: phosphoserine, -threonine and
-tyrosine came out at charge 0 instead of -2, ornithine and 2,3-diaminopropionic
acid at 0 instead of +1. The hydrogens now come from OpenBabel's pH model on the CCD
hydrogen counts (parametrize_ligand.ob_protonate) and the pH 7.4 acid rules follow
(parametrize_ligand.ionize_acids), as for ligands and fused glycan residues. A CB
next to an amine N types its hydrogens HP, amber19sb's convention for LYS and LYN
(verifier V4), with the XC-CT-HP angle amber19sb lacks. Pinned on ALA-X-ALA peptides
built from the CCD ideal coordinates of tests/data/ccd (CC0), the network refused:

  [charge]  ncaa.rtp charges, as written, sum exactly to SEP -2, TPO -2, PTR -2,
            ORN +1, DPP +1; no hydrogen on the phosphate oxygens, three on the
            ORN NE and the DPP NG amine.
  [types]   DPP HB1/HB2 are HP (CB bonded to the amine NG), ORN HB HC, SEP and TPO
            HB H1; ncaa_bonded.itp holds 'HP CT XC' at 109.5 deg, 418.4.
  [gromacs] pdb2gmx (-ignh -missing) and grompp -maxwarn 0 accept the five chains,
            whose qtot are -2, -2, -2, +1 and +1.

Run:  python3 tests/test_ncaa_protonation.py   (about 5 min)
"""
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(REPO, "utils")
SCRIPT = os.path.join(UTILS, "parametrize_ncaa.py")
CCD = os.path.join(REPO, "tests", "data", "ccd")
MDP = os.path.join(REPO, "settings", "amber19sb_opc3", "emin_vac.mdp")
sys.path.insert(0, UTILS)
import ccd_cache  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print("  %-74s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:400]))
    if not ok:
        failures.append(name)


# residue: (chain, residue charge at pH 7.4)
WANT = {"SEP": ("A", -2), "TPO": ("B", -2), "PTR": ("C", -2), "ORN": ("D", 1), "DPP": ("E", 1)}


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def add(a, v, s):
    return tuple(x + s * y for x, y in zip(a, v))


def perp(v):
    return unit((v[1], -v[0], 0.0) if abs(v[2]) < 0.9 else (0.0, v[2], -v[1]))


def peptide(code, chain, shift):
    """ATOM records of ALA 1, <code> 2, ALA 3: the NCAA's heavy atoms at its CCD ideal
    coordinates (OXT left out), the ALAs placed along its N-H and C-OXT directions.
    Only the topology matters here; nothing is minimised."""
    cif = ccd_cache.parse_cif(open(os.path.join(CCD, code + ".cif")).read())
    ids = cif["_chem_comp_atom.atom_id"]
    els = cif["_chem_comp_atom.type_symbol"]
    xyz = {n: tuple(float(cif["_chem_comp_atom.pdbx_model_Cartn_%s_ideal" % c][i]) + shift for c in "xyz")
           for i, n in enumerate(ids)}
    u1, u2 = unit(tuple(h - n for h, n in zip(xyz["H"], xyz["N"]))), unit(tuple(o - c for o, c in zip(xyz["OXT"], xyz["C"])))
    c0 = add(xyz["N"], u1, 1.33)
    ca0 = add(c0, u1, 1.52)
    ala1 = [("N", add(ca0, u1, 1.46)), ("CA", ca0), ("C", c0), ("O", add(c0, perp(u1), 1.23)), ("CB", add(ca0, perp(u1), 1.53))]
    n3 = add(xyz["C"], u2, 1.33)
    ca3 = add(n3, u2, 1.46)
    c3 = add(ca3, u2, 1.52)
    ala3 = [("N", n3), ("CA", ca3), ("C", c3), ("O", add(c3, perp(u2), 1.23)), ("CB", add(ca3, perp(u2), 1.53))]
    mid = [(n, xyz[n], e) for n, e in zip(ids, els) if e not in ("H", "D") and n != "OXT"]
    rows = []
    for resnum, res, atoms in ((1, "ALA", [(n, p, n[0]) for n, p in ala1]), (2, code, mid),
                               (3, "ALA", [(n, p, n[0]) for n, p in ala3])):
        for n, p, e in atoms:
            rows.append("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                        % (len(rows) + 1, (" " + n) if len(n) < 4 else n, res, chain, resnum, p[0], p[1], p[2], e))
    return rows


work = tempfile.mkdtemp(prefix="ncaa_protonation_")
try:
    rows = []
    for k, (code, (chain, _)) in enumerate(WANT.items()):
        rows += peptide(code, chain, 40.0 * k)
        rows.append("TER")
    with open(os.path.join(work, "fixed.pdb"), "w") as f:
        f.write("\n".join(rows) + "\nEND\n")
    env = {k: v for k, v in os.environ.items() if k != "GROSCORE_CCD_CACHE"}
    # No network: every request goes to a closed local port and is refused at once.
    env.update(GROSCORE_CCD_CACHE=CCD, https_proxy="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
               http_proxy="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9", no_proxy="", NO_PROXY="")
    p = subprocess.run([sys.executable, SCRIPT, "-f", "fixed.pdb", "--gmx-ff", "amber19sb"], cwd=work, env=env,
                       capture_output=True, text=True, timeout=1800)
    log = p.stdout + p.stderr
    with open(os.path.join(work, "pn.log"), "w") as f:
        f.write(log)
    check("parametrize_ncaa.py: exit 0, no ncaa_failed.gs", p.returncode == 0
          and not os.path.isfile(os.path.join(work, "ncaa_failed.gs")), log[-600:])
    rtp_path = os.path.join(work, "amber19sb.ff", "ncaa.rtp")
    rtp = open(rtp_path).read().split("\n") if os.path.isfile(rtp_path) else []

    def entry(res):
        """([(name, type, charge text)], {heavy atom: [hydrogens]}) of [ res ]."""
        try:
            i = rtp.index("[ %s ]" % res)
        except ValueError:
            return [], {}
        atoms, hs, sec = [], {}, None
        for line in rtp[i + 1:]:
            s = line.strip()
            if not s.startswith("[") and (line.startswith("[") or s.startswith(";")):
                break
            if s.startswith("["):
                if s.strip("[] ") not in ("atoms", "bonds", "impropers", "cmap"):
                    break
                sec = s.strip("[] ")
            elif s and sec == "atoms":
                t = s.split()
                atoms.append((t[0], t[1], t[2]))
            elif s and sec == "bonds":
                a, b = s.split()[:2]
                for x, y in ((a, b), (b, a)):
                    if y.startswith("H") and not x.startswith(("H", "-", "+")):
                        hs.setdefault(x, []).append(y)
        return atoms, hs

    print("\n[charge] pH 7.4 states, the written charges summing exactly")
    E = {res: entry(res) for res in WANT}
    for res, (chain, q) in WANT.items():
        atoms = E[res][0]
        total = sum((Decimal(c) for _, _, c in atoms), Decimal(0))
        check("%s: %d atoms in ncaa.rtp, charges sum to %+d exactly" % (res, len(atoms), q),
              atoms and total == q, total)
    for res in ("SEP", "TPO", "PTR"):
        hs = E[res][1]
        check("%s: no hydrogen on OP1, OP2 or OP3" % res, E[res][0] and not any(hs.get(o) for o in ("O1P", "O2P", "O3P",
                                                                                      "OP1", "OP2", "OP3")), hs)
    check("ORN: three hydrogens on NE", len(E["ORN"][1].get("NE", [])) == 3, E["ORN"][1])
    check("DPP: three hydrogens on NG", len(E["DPP"][1].get("NG", [])) == 3, E["DPP"][1])

    print("\n[types] CB hydrogens: HP next to an amine N")
    hb = {res: sorted(t for n, t, _ in E[res][0] if n.startswith("HB")) for res in WANT}
    check("DPP HB1/HB2 HP (CB bonded to the amine NG)", hb["DPP"] == ["HP", "HP"], hb["DPP"])
    check("ORN HB HC (CB bonded to carbons only)", hb["ORN"] == ["HC", "HC"], hb["ORN"])
    check("SEP HB H1, TPO HB H1 (CB bonded to O)", hb["SEP"] == ["H1", "H1"] and hb["TPO"] == ["H1"], hb)
    bonded = os.path.join(work, "amber19sb.ff", "ncaa_bonded.itp")
    ang = [l.split() for l in open(bonded)] if os.path.isfile(bonded) else []
    check("ncaa_bonded.itp: 'HP CT XC 1 109.5000 418.4000'", ["HP", "CT", "XC", "1", "109.5000", "418.4000"] in ang,
          [a for a in ang if "HP" in a][:3])

    print("\n[gromacs] pdb2gmx and grompp -maxwarn 0")
    genv = dict(os.environ, GMXLIB=work)

    def gmx(*args):
        return subprocess.run(("gmx",) + args, cwd=work, env=genv, capture_output=True, text=True, timeout=900)
    r = gmx("pdb2gmx", "-f", "fixed.pdb", "-o", "conf.gro", "-p", "topol.top", "-water", "none", "-ff", "amber19sb",
            "-ignh", "-missing")
    check("pdb2gmx: exit 0", r.returncode == 0, (r.stdout + r.stderr)[-600:])
    qtot = []
    for ch in "ABCDE":
        itp = os.path.join(work, "topol_Protein_chain_%s.itp" % ch)
        q = re.findall(r"qtot\s+(\S+)", open(itp).read()) if os.path.isfile(itp) else []
        qtot.append(round(float(q[-1]), 4) if q else None)
    check("chain qtot -2, -2, -2, +1, +1", qtot == [-2.0, -2.0, -2.0, 1.0, 1.0], qtot)
    if r.returncode == 0:
        gmx("editconf", "-f", "conf.gro", "-o", "box.gro", "-bt", "cubic", "-d", "1.5")
        r = gmx("grompp", "-f", MDP, "-c", "box.gro", "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "0")
        check("grompp -maxwarn 0: exit 0", r.returncode == 0, (r.stdout + r.stderr)[-600:])
finally:
    shutil.rmtree(work, ignore_errors=True)

print("\n%s" % ("ALL PASS" if not failures else "%d FAILED" % len(failures)))
sys.exit(1 if failures else 0)
