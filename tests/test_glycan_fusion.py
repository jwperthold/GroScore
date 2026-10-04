#!/usr/bin/env python3
"""A glycan bonded to Asn, Ser or Thr is fused with it into one residue (utils/fuse_glycans.py).

1MAH (NAG on ASN A 350) and 2AJF (NAGs on ASN A 53, A 322, A 546 and E 330) stopped
as LIGAND_FAIL: merge_ligand.py found the sugars covalently bonded to their Asn, a
link the ligand path cannot model. fuse_glycans.py makes each glycan and its amino
acid one non-canonical residue for the NCAA path and writes its chemistry into
glycans.json. Pinned on small extracts of deposited structures (tests/data/glycans,
PDB data CC0) with CCD entries from forcefield/ccd_cache and tests/data/ccd, and the
network refused (urlopen raises, every call is counted):

  [1MAH]     one definition NG1: parent ASN, link ND2-C1A, 22 heavy atoms whose H
             counts sum to 19 (ND2 keeps 1); instance A 350 + NAG A 544 at 1.451 A;
             H2A on C2A and HN2A on N2A; fused.pdb has no HETATM NAG and 22
             contiguous ATOM records of NG1 A 350, every other line as in the input;
             the log says 'fused NG1: ASN A 350 + NAG A 544 (ND2-C1 1.45 A)' and the
             stereo comparison; renumber_pdb.py extracts no ligand from fused.pdb and
             ligand_NAG_A.pdb from the input; only NAG and ASN are looked up.
  [2AJF]     one NG1 with four instances, links 1.454, 1.452, 1.457 and 1.452 A; the
             ZN, CL and waters stay as they are.
  [refuse]   O6 deleted: 'missing O6'; an SO4 1.60 A from O6: 'not all saccharides';
             the input is kept byte for byte and every member gets a
             glycan_unfused.gs line; a deposited O1 1.40 A from C1 is dropped and
             logged, and the glycan fused with 22 atoms.
  [cap]      5XJE --max-sugars 3: NG1 = ASN C 162 + NAG F 1 + FUC F 2 (32 heavy, 29
             H), NG2 = ASN C 45 + NAG G 1 + NAG G 2 + BMA G 3 (47, 42), both 9-sugar
             Fc glycans refused ('9 sugars, more than the cap of 3');
             $GROSCORE_GLYCAN_MAX_SUGARS 2 refuses the 3-sugar glycan and 3 fuses it,
             --max-sugars wins over it, and a bad value leaves no cap. Without
             either there is no cap: the 9-sugar Fc glycans are NG1 (118 heavy atoms,
             108 H) on A 297 and B 297, beside NG2 and NG3, and nothing is left.
  [names]    an input holding a residue NG1 gets NG2; the name generator skips NGL
             and SGB; 3AVE --max-sugars 12 tags NAG C1=A, NAG C2=B, FUC C8=C, BMA
             C3=D, MAN C4=E, MAN C6=F, NAG C5=G, NAG C7=H and likewise in chain D,
             one definition of 107 heavy atoms; no fused name is ever looked up in
             the CCD (NG1 is a real entry, the decoy in tests/data/ccd).
  [O-linked] 5A2I SER P 4 + A2G: SG1 with OG h=0; 5A2K THR P 4 + A2G: TG1 with OG1
             h=0 and CG2 h=3 (its HG21-HG23 kept clear of sugar names); 5A2L A2G on
             CYS SG: 'attached to CYS P 4 SG'.
  [sialic]   4BYH glycan C, --max-sugars 12: SIA is tag J, bonded through C2 with
             O2 left out, named C10J/O1AJ/O1BJ, and its C11 methyl and O1B hydrogens
             take HX bases (H11J1 and HO1BJ have 5 characters).
  [stereo]   5T78: an alpha GalNAc deposited under the beta code NGA is fused, and
             'C1 inverted against NGA' is logged as a WARNING (review F11).
  [altloc]   1MAH with altlocs A/B on ASN A 350 and NAG A 544, and on the NAG only:
             altloc A kept, 22 NG1 atoms, no altloc or HETATM NAG record left, no
             ligand file after renumber_pdb.py (review F2); a link that is a bond
             only in altloc B is refused.
  [F9]       2BTF ACE caps and HIC, 1S1Q MSE, 2MTA TRQ: no CCD lookup, no line, a
             byte copy; an SO4 bonded to SER OG is looked up, is no glycan, and gets
             no line (review F9).
  [review]   the fusion review's cases. 1YPZ NAG L 1, whose deposited O1 sits 1.51 A
             from ND2, is fused with O1 dropped (a leaving atom makes no link, F3);
             a glycan bonded to the protein only through its O1 is refused, saying
             so. 1FZC: the free C1 of NAG K 2 lies 1.85 A from O4 of NAG K 1 (a
             deposited link longer than the covalent rule); the link is joined and
             logged, and NG1 = ASN B 364 + NAG K 1, NAG K 2. With an O1 of NAG K 2
             deposited 1.40 A from C1 the link is ambiguous: not joined, NAG K 1 not
             fused, and both sugars get that reason (F1). 2FD6: NAG C 1 C1 1.99 A
             from ASN U 172 ND2, its O1 drawn on ND2 (0.79 A): the site link is
             joined, NG1 = ASN U 172 + NAG C 1, NAG C 2, no ligand left. 2PO6 NAG E 301 on OD1 of ASN E 42 (a
             flipped amide) is fused as NG1 with OD1 and ND2 swapped back (F4). O4
             and O6 named the other way round are refused, the two stretched bonds
             named (F5). A NAG whose C1 lies 1.53 A from C6 of NAG A 544 is refused:
             no hydroxyl O (F8). NAG A 544 written as ATOM records is fused as the
             HETATM one is, and when it is refused its records become HETATM, so
             renumber_pdb.py extracts it (F7).
  [fallback] (cap 3) --exclude-file with 'NG1 <reason>': the 5XJE F glycan stays HETATM, its
             sugars' lines say 'NG1 could not be parametrised: <reason>', NG2 keeps
             its name on this and a further run; a chemistry the first pass did not
             fuse is not fused by the fallback either, since parametrize_ncaa.py made
             no parameters for it (review R2); an existing glycans.json lends its
             names to unchanged chemistries and keeps the others' names reserved; a
             stale glycan_unfused.gs goes.
  [cli]      exit 0 on a structure, 1 on an unreadable input; the residue classes
             are renumber_pdb.py's, and ccd_cache.GLYCAN_PARENTS are SITES.

Standalone, no pytest: python3 tests/test_glycan_fusion.py (about 10 s)
"""
import ast
import contextlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(ROOT, "utils")
DATA = os.path.join(ROOT, "tests", "data", "glycans")
CCD_DATA = os.path.join(ROOT, "tests", "data", "ccd")
FG = os.path.join(UTILS, "fuse_glycans.py")
RN = os.path.join(UTILS, "renumber_pdb.py")
fails = 0


def check(name, ok, detail=""):
    global fails
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:400]))
    if not ok:
        fails += 1


if not os.path.isfile(FG):
    check("utils/fuse_glycans.py exists", False, FG)
    print("\n1 FAILED")
    sys.exit(1)

os.environ["GROSCORE_CCD_CACHE"] = CCD_DATA
os.environ.pop("GROSCORE_GLYCAN_MAX_SUGARS", None)
spec = importlib.util.spec_from_file_location("fuse_glycans_under_test", FG)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
C = F.ccd_cache
net, looked = [], []


def no_network(*a, **k):
    net.append(a[0] if a else k)
    raise OSError("network disabled in this test")


C.urllib.request.urlopen = no_network
real_fetch = C.fetch


def spy_fetch(cid, *a, **k):
    looked.append(cid)
    return real_fetch(cid, *a, **k)


C.fetch = spy_fetch
tmp = tempfile.mkdtemp(prefix="test_glycan_fusion_")
PROXY = {k: "http://127.0.0.1:9" for k in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY")}


def fixture(name):
    with open(os.path.join(DATA, name)) as f:
        return f.read()


def fuse(name, text, *args, stale=None):
    """F.main in a directory of its own: (dir, rc, log, glycans.json, glycan_unfused.gs lines)."""
    d = os.path.join(tmp, name)
    os.makedirs(d)
    with open(os.path.join(d, "input.pdb"), "w") as f:
        f.write(text)
    if stale:
        with open(os.path.join(d, "glycan_unfused.gs"), "w") as f:
            f.write(stale)
    cwd, out = os.getcwd(), io.StringIO()
    os.chdir(d)
    try:
        with contextlib.redirect_stdout(out):
            rc = F.main(["-f", "input.pdb", "-o", "fused.pdb", "-d", "glycans.json"] + list(args))
    finally:
        os.chdir(cwd)
    j = json.load(open(os.path.join(d, "glycans.json"))) if os.path.isfile(os.path.join(d, "glycans.json")) else {}
    u = os.path.join(d, "glycan_unfused.gs")
    return d, rc, out.getvalue(), j, ([l.rstrip("\n") for l in open(u)] if os.path.isfile(u) else [])


def again(d, *args):
    """F.main once more in an existing directory (the fallback re-run)."""
    cwd, out = os.getcwd(), io.StringIO()
    os.chdir(d)
    try:
        with contextlib.redirect_stdout(out):
            rc = F.main(["-f", "input.pdb", "-o", "fused.pdb", "-d", "glycans.json"] + list(args))
    finally:
        os.chdir(cwd)
    u = os.path.join(d, "glycan_unfused.gs")
    return rc, out.getvalue(), json.load(open(os.path.join(d, "glycans.json"))), (
        [l.rstrip("\n") for l in open(u)] if os.path.isfile(u) else [])


def renumber(d, pdb):
    """renumber_pdb.py on d/pdb in a fresh subdirectory: (residues, ligand files)."""
    w = os.path.join(d, "rn_" + pdb.split(".")[0])
    os.makedirs(w)
    shutil.copy(os.path.join(d, pdb), os.path.join(w, pdb))
    r = subprocess.run([sys.executable, RN, "-f", pdb, "-o", "renumbered.pdb", "-c", "X"], cwd=w,
                       capture_output=True, text=True, timeout=120)
    m = re.search(r"Renumbered (\d+) residues", r.stdout)
    return (int(m.group(1)) if m else None), sorted(f for f in os.listdir(w) if f.startswith("ligand_")
                                                    and f.endswith(".pdb"))


def lines_of(d, name="fused.pdb"):
    return open(os.path.join(d, name)).read().splitlines()


def same_bytes(d):
    return open(os.path.join(d, "input.pdb"), "rb").read() == open(os.path.join(d, "fused.pdb"), "rb").read()


def coords(text, res, chain, num):
    return {l[12:16].strip(): (float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in text.splitlines()
            if l.startswith(("ATOM", "HETATM")) and l[17:20] == res and l[21] == chain and l[22:26].strip() == num}


def hetatm(serial, name, res, chain, num, xyz, el, alt=" ", occ=1.0):
    nm = (" " + name) if len(name) < 4 else name
    return "HETATM%5d %-4s%1s%3s %1s%4s    %8.3f%8.3f%8.3f%6.2f 30.00          %2s" % (
        serial, nm, alt, res, chain, num, xyz[0], xyz[1], xyz[2], occ, el)


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def add(a, b, s=1.0):
    return tuple(x + s * y for x, y in zip(a, b))


def tetra(v):
    """Four tetrahedral unit vectors, the first along v (a sulfate's S-O directions)."""
    v = unit(v)
    e1 = unit((v[1], -v[0], 0.0) if abs(v[2]) < 0.9 else (0.0, v[2], -v[1]))
    e2 = (v[1] * e1[2] - v[2] * e1[1], v[2] * e1[0] - v[0] * e1[2], v[0] * e1[1] - v[1] * e1[0])
    return [v] + [unit(tuple(-c / 3.0 + 0.9428 * (math.cos(p) * a + math.sin(p) * b) for c, a, b in zip(v, e1, e2)))
                  for p in (0.0, 2.0944, 4.1888)]


def sulfate(serial, chain, num, oxygen, partner):
    """SO4 records with S 1.60 A from `oxygen`, away from `partner`, S-O 1.48 A."""
    v = unit(sub(oxygen, partner))
    s = add(oxygen, v, 1.60)
    return [hetatm(serial, "S", "SO4", chain, num, s, "S")] + [
        hetatm(serial + 1 + k, "O%d" % (k + 1), "SO4", chain, num, add(s, x, 1.48), "O") for k, x in enumerate(tetra(v))]


def before_end(text, extra):
    """text with the extra records inserted before its END line."""
    i = text.rindex("END")
    return text[:i] + "\n".join(extra) + "\n" + text[i:]


def residue(j, k=0):
    return j["residues"][k] if j.get("residues") and len(j["residues"]) > k else {}


MAH = fixture("1MAH_glycan.pdb")

print("\n[1MAH] NAG A 544 on ASN A 350")
del looked[:]
d, rc, log, j, unf = fuse("1mah", MAH, stale="NAG A 544 a line from an earlier run\n")
r = residue(j)
check("exit 0, one definition NG1, parent ASN, link ND2-C1A",
      rc == 0 and len(j.get("residues", [])) == 1 and r.get("resname") == "NG1" and r.get("parent") == "ASN"
      and r.get("link") == ["ND2", "C1A"], "%s %s" % (rc, {k: r.get(k) for k in ("resname", "parent", "link")}))
hs = {a["name"]: a["h"] for a in r.get("atoms", [])}
check("22 heavy atoms whose H counts sum to 19; ND2 keeps 1 H, C1A 1",
      len(hs) == 22 and sum(hs.values()) == 19 and hs.get("ND2") == 1 and hs.get("C1A") == 1, str(hs))
inst = r.get("instances", [{}])
check("one instance: A 350, NAG A 544, link 1.451 A",
      len(inst) == 1 and (inst[0]["chain"], inst[0]["resnum"], inst[0]["icode"], inst[0]["sugars"], inst[0]["link_A"])
      == ("A", "350", "", ["NAG A 544"], 1.451), str(inst))
hn = r.get("hnames", {})
check("hnames: H2A on C2A and HN2A on N2A (one name each)", hn.get("C2A") == ["H2A"] and hn.get("N2A") == ["HN2A"],
      str(hn))
check("bonds: ND2-C1A single, and the acetamide C7A=O7A double",
      ["ND2", "C1A", 1] in r.get("bonds", []) and ["C7A", "O7A", 2] in r.get("bonds", []), str(r.get("bonds", [])[-4:]))
fl, il = lines_of(d), lines_of(d, "input.pdb")
ng = [i for i, l in enumerate(fl) if l.startswith("ATOM") and l[17:26] == "NG1 A 350"]
check("fused.pdb: no HETATM NAG, 22 contiguous ATOM records of NG1 A 350",
      not [l for l in fl if l.startswith("HETATM") and l[17:20] == "NAG"] and len(ng) == 22
      and ng == list(range(ng[0], ng[0] + 22)), "%d records at %s" % (len(ng), ng[:3]))
check("every other line of the input is unchanged and in order",
      [l for l in fl if l[17:26] != "NG1 A 350"] == [l for l in il if l[17:26] not in ("ASN A 350", "NAG A 544")],
      "")
check("the sugar records carry the tagged names and element columns (' C1A', 'O7A', 'O')",
      any(l[12:16] == " C1A" and l[76:78] == " C" for l in fl) and any(l[12:16] == " O7A" and l[76:78] == " O"
                                                                    for l in fl), "")
check("log: 'fused NG1: ASN A 350 + NAG A 544 (ND2-C1 1.45 A)'",
      "fused NG1: ASN A 350 + NAG A 544 (ND2-C1 1.45 A)" in log, log[-300:])
check("log: the stereo comparison 'NAG A 544 (A): C1 C2 C3 C4 C5 as in NAG'",
      "  stereo NAG A 544 (A): C1 C2 C3 C4 C5 as in NAG" in log, log[-300:])
check("a stale glycan_unfused.gs is removed (nothing left as ligands)", not unf, str(unf))
check("CCD lookups: NAG and ASN only, no network", looked == ["NAG", "ASN"] and not net, "%s %s" % (looked, net))
n_f, lig_f = renumber(d, "fused.pdb")
n_i, lig_i = renumber(d, "input.pdb")
check("renumber_pdb.py: fused.pdb gives 11 residues and no ligand file", n_f == 11 and not lig_f, "%s %s" % (n_f, lig_f))
check("  input.pdb gives 11 residues and ligand_NAG_A.pdb", n_i == 11 and lig_i == ["ligand_NAG_A.pdb"], "%s %s" % (n_i, lig_i))

print("\n[2AJF] four NAGs, one definition")
d, rc, log, j, unf = fuse("2ajf", fixture("2AJF_glycans.pdb"))
r = residue(j)
got = [(i["chain"], i["resnum"], i["sugars"], i["link_A"]) for i in r.get("instances", [])]
check("one definition NG1 with 4 instances, links 1.454, 1.452, 1.457, 1.452 A",
      len(j.get("residues", [])) == 1 and r.get("resname") == "NG1"
      and got == [("A", "53", ["NAG A 701"], 1.454), ("A", "322", ["NAG A 702"], 1.452),
                  ("A", "546", ["NAG A 703"], 1.457), ("E", "330", ["NAG E 601"], 1.452)], str(got))
het_in = [l for l in lines_of(d, "input.pdb") if l.startswith("HETATM") and l[17:20] != "NAG"]
check("ZN, CL and the waters stay as they are, no HETATM NAG left",
      [l for l in lines_of(d) if l.startswith("HETATM")] == het_in and len(het_in) == 4, str(len(het_in)))

print("\n[refuse] not fused, left untouched, every member named")
no_o6 = "\n".join(l for l in MAH.splitlines() if not (l[17:20] == "NAG" and l[12:16] == " O6 ")) + "\n"
d, rc, log, j, unf = fuse("no_o6", no_o6)
check("O6 deleted: not fused, the input byte for byte", rc == 0 and not j.get("residues") and same_bytes(d), log[-300:])
check("  'NAG A 544 NAG A 544: missing O6' in glycan_unfused.gs", unf == ["NAG A 544 NAG A 544: missing O6"], str(unf))
nag = coords(MAH, "NAG", "A", "544")
asn = coords(MAH, "ASN", "A", "350")
d, rc, log, j, unf = fuse("so4", before_end(MAH, sulfate(4600, "A", "545", nag["O6"], nag["C6"])))
check("an SO4 with S 1.60 A from O6: not fused, the input byte for byte",
      rc == 0 and not j.get("residues") and same_bytes(d), log[-300:])
check("  both members: 'not all saccharides: SO4 A 545 (NON-POLYMER)'",
      unf == ["NAG A 544 not all saccharides: SO4 A 545 (NON-POLYMER)",
              "SO4 A 545 not all saccharides: SO4 A 545 (NON-POLYMER)"], str(unf))
w = unit(tuple(-c for c in add(add(unit(sub(asn["ND2"], nag["C1"])), unit(sub(nag["O5"], nag["C1"]))),
                                unit(sub(nag["C2"], nag["C1"])))))
o1 = hetatm(4593, "O1", "NAG", "A", "544", add(nag["C1"], w, 1.40), "O")
with_o1 = MAH.replace(next(l for l in MAH.splitlines() if l[17:20] == "NAG" and l[12:16] == " C1 "),
                      next(l for l in MAH.splitlines() if l[17:20] == "NAG" and l[12:16] == " C1 ") + "\n" + o1)
d, rc, log, j, unf = fuse("o1", with_o1)
check("a deposited O1 1.40 A from C1: fused with 22 atoms",
      len(residue(j).get("atoms", [])) == 22 and not unf, "%s %s" % (len(residue(j).get("atoms", [])), unf))
check("  O1 dropped and logged, no O1 record in fused.pdb",
      "dropped the deposited leaving atom NAG A 544 O1" in log and not [l for l in lines_of(d) if l[12:16] in (" O1 ", " O1A")],
      log[-300:])

print("\n[cap] 5XJE")
XJE = fixture("5XJE_glycans.pdb")
d, rc, log, j, unf = fuse("5xje", XJE, "--max-sugars", "3")
got = {x["resname"]: (len(x["atoms"]), sum(a["h"] for a in x["atoms"]), x["instances"][0]["sugars"],
                      x["instances"][0]["chain"] + " " + x["instances"][0]["resnum"]) for x in j.get("residues", [])}
check("NG1 = ASN C 162 + NAG F 1 + FUC F 2: 32 heavy atoms, 29 H",
      got.get("NG1") == (32, 29, ["NAG F 1", "FUC F 2"], "C 162"), str(got.get("NG1")))
check("NG2 = ASN C 45 + NAG G 1 + NAG G 2 + BMA G 3: 47 heavy atoms, 42 H",
      got.get("NG2") == (47, 42, ["NAG G 1", "NAG G 2", "BMA G 3"], "C 45"), str(got.get("NG2")))
cap = [l for l in unf if l.endswith(" 9 sugars, more than the cap of 3")]
check("both 9-sugar Fc glycans refused: 18 lines '9 sugars, more than the cap of 3'",
      len(unf) == 18 and len(cap) == 18 and {l.split()[1] for l in cap} == {"D", "E"}, str(unf[:3]))
check("their records stay HETATM: 18 sugar residues in fused.pdb",
      len({l[17:26] for l in lines_of(d) if l.startswith("HETATM") and l[21] in "DE"}) == 18, "")


def cap_run(name, env, *args):
    if env is None:
        os.environ.pop("GROSCORE_GLYCAN_MAX_SUGARS", None)
    else:
        os.environ["GROSCORE_GLYCAN_MAX_SUGARS"] = env
    try:
        d, rc, log, j, unf = fuse(name, XJE, *args)
    finally:
        os.environ.pop("GROSCORE_GLYCAN_MAX_SUGARS", None)
    return sorted(x["resname"] for x in j.get("residues", [])), [l for l in unf if l.startswith(("NAG G", "BMA G"))], log


names, gl, log = cap_run("env2", "2")
check("$GROSCORE_GLYCAN_MAX_SUGARS=2: the 2-sugar glycan fused, the 3-sugar one refused",
      names == ["NG1"] and len(gl) == 3 and all(l.endswith("3 sugars, more than the cap of 2") for l in gl), str(gl))
names, gl, log = cap_run("env3", "3")
check("$GROSCORE_GLYCAN_MAX_SUGARS=3: both fused", names == ["NG1", "NG2"] and not gl, str(gl))
names, gl, log = cap_run("flag2", "3", "--max-sugars", "2")
check("--max-sugars 2 wins over $GROSCORE_GLYCAN_MAX_SUGARS=3", names == ["NG1"] and len(gl) == 3, str(names))
names, gl, log = cap_run("envbad", "four")
check("a value that is no whole number: said, and no cap (all four glycans fused, NG1-NG3)",
      names == ["NG1", "NG2", "NG3"] and "is not a whole number of at least 1; no cap" in log, log[:300])
d, rc, log, j, unf = fuse("5xje_nocap", XJE)
got = {x["resname"]: (len(x["atoms"]), sum(a["h"] for a in x["atoms"]), [i["chain"] + " " + i["resnum"] for i in x["instances"]])
       for x in j.get("residues", [])}
check("no cap by default: the 9-sugar Fc glycans fused as NG1 (118 heavy, 108 H) on A 297 and B 297",
      got.get("NG1") == (118, 108, ["A 297", "B 297"]) and len(residue(j).get("sugars", [])) == 9, str(got.get("NG1")))
check("  NG2 (32, 29) on C 162 and NG3 (47, 42) on C 45, nothing left as ligands, max_sugars null",
      got.get("NG2") == (32, 29, ["C 162"]) and got.get("NG3") == (47, 42, ["C 45"]) and not unf
      and j.get("max_sugars", 0) is None, "%s %s" % (got, unf[:2]))
check("  the log says 'no cap on the number of sugars'",
      "fuse_glycans: 4 glycan(s) bonded to the protein; no cap on the number of sugars" in log, log[:200])

print("\n[names] residue names and sugar tags")
del looked[:]
far = [hetatm(9001 + k, n, "NG1", "X", "999", (200.0 + k, 200.0, 200.0), el)
       for k, (n, el) in enumerate((("C1", "C"), ("O1", "O"), ("P", "P")))]
d, rc, log, j, unf = fuse("ng1_taken", before_end(MAH, far))
check("an input holding a residue NG1 gets NG2 for its glycan",
      [x["resname"] for x in j.get("residues", [])] == ["NG2"] and "fused NG2: ASN A 350" in log, log[-300:])
check("  and the CCD is not asked for NG1 (NAG, ASN only)", looked == ["NAG", "ASN"], str(looked))
taken = F.taken_names()
check("the generator skips NGL (inside NGLN/NGLU/NGLY of aminoacids.hdb) and SGB (CHARMM36)",
      F.next_name("NG", taken | {"NG" + c for c in "123456789ABCDEFGHIJK"}) == "NGM"
      and F.next_name("SG", taken | {"SG" + c for c in "123456789A"}) == "SGC", sorted(taken))
measured, F.TAKEN_MEASURED = F.TAKEN_MEASURED, set()
try:
    scanned = F.taken_names()
finally:
    F.TAKEN_MEASURED = measured
gmx_hdb = any(os.path.isfile(os.path.join(t, "amber19sb.ff", "aminoacids.hdb")) for t in F.gromacs_top_dirs())
check("  the scan itself finds SGB in forcefield/%s" % (" and NGL in GROMACS's amber19sb.ff" if gmx_hdb else ""),
      "SGB" in scanned and ("NGL" in scanned or not gmx_hdb), sorted(scanned))
d, rc, log, j, unf = fuse("3ave", fixture("3AVE_glycans.pdb"), "--max-sugars", "12")
r = residue(j)
tags = [(s["tag"], s["ccd"]) for s in r.get("sugars", [])]
check("3AVE: one definition of 107 heavy atoms, two instances",
      len(j.get("residues", [])) == 1 and len(r.get("atoms", [])) == 107 and len(r.get("instances", [])) == 2,
      "%d %d" % (len(r.get("atoms", [])), len(r.get("instances", []))))
check("  tags A-H: NAG NAG FUC BMA MAN MAN NAG NAG",
      tags == list(zip("ABCDEFGH", ("NAG", "NAG", "FUC", "BMA", "MAN", "MAN", "NAG", "NAG"))), str(tags))
check("  chain C: NAG C1=A, C2=B, FUC C8=C, BMA C3=D, MAN C4=E, MAN C6=F, NAG C5=G, NAG C7=H; chain D alike",
      [i["sugars"] for i in r.get("instances", [])] == [
          ["%s %s %d" % (x, c, n) for x, n in (("NAG", 1), ("NAG", 2), ("FUC", 8), ("BMA", 3), ("MAN", 4), ("MAN", 6),
                                                ("NAG", 5), ("NAG", 7))] for c in "CD"], str([i["sugars"] for i in r.get("instances", [])]))
check("no fused name was ever looked up in the CCD",
      not [x for x in looked if re.fullmatch(r"[NST]G[1-9A-Z]", x)], str(sorted(set(looked))))

print("\n[O-linked] Ser and Thr")
d, rc, log, j, unf = fuse("5a2i", fixture("5A2I_tn_ser.pdb"))
r = residue(j)
hs = {a["name"]: a["h"] for a in r.get("atoms", [])}
check("5A2I SER P 4 + A2G P 1009: SG1, link OG-C1A, OG h=0, CB h=2",
      r.get("resname") == "SG1" and r.get("link") == ["OG", "C1A"] and hs.get("OG") == 0 and hs.get("CB") == 2,
      "%s %s %s" % (r.get("resname"), r.get("link"), hs))
d, rc, log, j, unf = fuse("5a2k", fixture("5A2K_tn_thr.pdb"))
r = residue(j)
hs = {a["name"]: a["h"] for a in r.get("atoms", [])}
sugar_h = {x for v in r.get("hnames", {}).values() for x in v}
check("5A2K THR P 4 + A2G P 10: TG1, link OG1-C1A, OG1 h=0, CG2 h=3",
      r.get("resname") == "TG1" and r.get("link") == ["OG1", "C1A"] and hs.get("OG1") == 0 and hs.get("CG2") == 3,
      "%s %s %s" % (r.get("resname"), r.get("link"), hs))
check("  HG21-HG23 (CG2) and HB stay free of sugar hydrogen names",
      not sugar_h & {"HG21", "HG22", "HG23", "HB", "HA", "H"} and len(sugar_h) == 14, sorted(sugar_h))
d, rc, log, j, unf = fuse("5a2l", fixture("5A2L_s_linked.pdb"))
check("5A2L A2G on CYS SG: refused, 'attached to CYS P 4 SG'",
      not j.get("residues") and unf == ["A2G P 1009 attached to CYS P 4 SG, not to ASN ND2, SER OG or THR OG1"]
      and same_bytes(d), str(unf))

print("\n[sialic] 4BYH, a sialylated glycan")
d, rc, log, j, unf = fuse("4byh", fixture("4BYH_sialyl.pdb"), "--max-sugars", "12")
r = residue(j)
names = {a["name"] for a in r.get("atoms", [])}
check("one definition: 10 sugars, SIA is tag J",
      [s["ccd"] for s in r.get("sugars", [])][-1:] == ["SIA"] and r.get("sugars", [{}])[-1].get("tag") == "J"
      and len(r.get("sugars", [])) == 10, str([(s["tag"], s["ccd"]) for s in r.get("sugars", [])]))
check("  SIA bonded through C2 to GAL O6 (O6I-C2J), its O2 left out",
      ["O6I", "C2J", 1] in r.get("bonds", []) and "O2J" not in names and {"C10J", "O1AJ", "O1BJ"} <= names,
      str([b for b in r.get("bonds", []) if "J" in b[1] and b[0][-1] != "J"]))
check("  the C11 methyl takes HX01-HX03 and the O1B hydrogen HX1",
      r.get("hnames", {}).get("C11J") == ["HX01", "HX02", "HX03"] and r.get("hnames", {}).get("O1BJ") == ["HX1"],
      str({k: v for k, v in r.get("hnames", {}).items() if k.endswith("J")}))
check("  138 heavy atoms, 125 H; SIA stereo 'C2 C4 C5 C6 C7 C8 as in SIA'",
      len(names) == 138 and sum(a["h"] for a in r.get("atoms", [])) == 125
      and "stereo SIA C 7 (J): C2 C4 C5 C6 C7 C8 as in SIA" in log, "%d %s" % (len(names), log[-200:]))

print("\n[stereo] a deposited anomer that contradicts its code is logged")
d, rc, log, j, unf = fuse("5t78", fixture("5T78_nga.pdb"))
check("5T78 NGA F 101 on THR F 4: fused as TG1",
      residue(j).get("resname") == "TG1" and "fused TG1: THR F 4 + NGA F 101" in log, log[-300:])
check("  'WARNING stereo NGA F 101 (A): C1 inverted against NGA; C2 C3 C4 C5 as in NGA'",
      "  WARNING stereo NGA F 101 (A): C1 inverted against NGA; C2 C3 C4 C5 as in NGA" in log, log[-300:])

print("\n[altloc] one conformer per fused unit")


def altlocs(text, which, shift=0.35):
    out = []
    for l in text.splitlines():
        if l.startswith(("ATOM", "HETATM")) and (l[17:20], l[21], l[22:26].strip()) in which:
            out.append(l[:16] + "A" + l[17:54] + "  0.60" + l[60:])
            out.append(l[:16] + "B" + l[17:30] + "%8.3f" % (float(l[30:38]) + shift) + l[38:54] + "  0.40" + l[60:])
        else:
            out.append(l)
    return "\n".join(out) + "\n"


for label_, which in (("altlocs on ASN A 350 and NAG A 544", {("ASN", "A", "350"), ("NAG", "A", "544")}),
                      ("altlocs on NAG A 544 only", {("NAG", "A", "544")})):
    text = altlocs(MAH, which)
    d, rc, log, j, unf = fuse("alt_%d" % len(which), text)
    fl = lines_of(d)
    ng = [l for l in fl if l[17:26] == "NG1 A 350"]
    n_f, lig = renumber(d, "fused.pdb")
    check("%s: fused, altloc A kept, 22 NG1 records, none with an altloc" % label_,
          len(residue(j).get("atoms", [])) == 22 and len(ng) == 22 and all(l[16] == " " for l in ng)
          and "altloc A kept for ASN A 350" in log, log[-300:])
    check("  no HETATM NAG, no ASN A 350 record left; renumber_pdb.py extracts no ligand",
          not [l for l in fl if l[17:20] == "NAG" or l[17:26] == "ASN A 350"] and not lig and n_f == 11,
          "%s %s" % (n_f, lig))
    b_xyz = {m[30:54] for m in text.splitlines() if m[16:17] == "B"}
    a_xyz = {m[30:54] for m in text.splitlines() if m[16:17] == "A"}
    check("  every altloc record kept is altloc A's (%d), none is B's" % len([l for l in ng if l[30:54] in a_xyz]),
          not [l for l in ng if l[30:54] in b_xyz] and len([l for l in ng if l[30:54] in a_xyz]) == len(
              [m for m in text.splitlines() if m[16:17] == "A" and m[76:78].strip() != "H"]), "")
d0 = sub(nag["C1"], asn["ND2"])
only_b = []
for l in MAH.splitlines():
    if l[17:26] == "NAG A 544":
        x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
        moved = add((x, y, z), unit(d0), 2.5)
        only_b.append(l[:16] + "A" + l[17:30] + "%8.3f%8.3f%8.3f" % moved + l[54:])
        only_b.append(l[:16] + "B" + l[17:])
    else:
        only_b.append(l)
d, rc, log, j, unf = fuse("alt_b_only", "\n".join(only_b) + "\n")
check("a link that is a bond only in altloc B: refused, the input byte for byte",
      not j.get("residues") and same_bytes(d)
      and unf == ["NAG A 544 the link to ASN A 350 ND2 is not a bond in altloc A, the conformer OpenMM keeps"], str(unf))

print("\n[F9] covalent neighbours that are not glycans")
for name in ("2BTF_caps.pdb", "1S1Q_mse.pdb", "2MTA_trq.pdb"):
    del looked[:]
    d, rc, log, j, unf = fuse("f9_" + name[:4], fixture(name))
    check("%s: no CCD lookup, no line, a byte copy" % name[:-4],
          rc == 0 and not looked and not unf and same_bytes(d) and "NOT fused" not in log, "%s %s" % (looked, unf))
SER = fixture("5A2I_tn_ser.pdb")
ser = coords(SER, "SER", "P", "4")
no_sugar = before_end("\n".join(l for l in SER.splitlines() if l[17:20] != "A2G") + "\n",
                      sulfate(900, "P", "900", ser["OG"], ser["CB"]))
del looked[:]
d, rc, log, j, unf = fuse("f9_so4", no_sugar)
check("an SO4 bonded to SER P 4 OG: looked up, no glycan, no line, a byte copy",
      looked == ["SO4"] and not unf and same_bytes(d)
      and "bonded to the protein, not a glycan (no saccharide): SO4 P 900" in log, "%s %s %s" % (looked, unf, log[:200]))

print("\n[review] the fusion review's cases")
RING = ("C1", "C2", "C3", "C4", "C5", "O5")


def nearest_other(extra, but="C1"):
    """The shortest distance from an atom of the `extra` records other than `but` to an atom of 1MAH."""
    old = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in MAH.splitlines() if l.startswith(("ATOM", "HETATM"))]
    return min(math.dist((float(l[30:38]), float(l[38:46]), float(l[46:54])), p) for l in extra if l[12:16].strip() != but
               for p in old)


def nag_copy(num, anchor, d_link, clear=2.4):
    """NAG A 544's records as NAG A <num>, turned and moved so that its C1 lies d_link A
    from `anchor` (a point at an atom of NAG A 544), on the line from that ring's centre
    through the anchor, with the copy's own ring further out, and spun about that line
    in 30-degree steps until no other copy atom is within `clear` A of 1MAH. Rotations
    only, so the copy keeps its stereocentres."""
    cen = tuple(sum(nag[n][k] for n in RING) / len(RING) for k in range(3))
    d = unit(sub(anchor, cen))
    t = add(anchor, d, d_link)
    n = unit(add(unit(sub(cen, nag["C1"])), d))     # turns the copy's ring onto d
    flip = lambda v: sub(tuple(2.0 * sum(n[k] * v[k] for k in range(3)) * n[i] for i in range(3)), v)

    def spin(v, a):                                 # Rodrigues, about d
        c, s, dv = math.cos(a), math.sin(a), sum(d[k] * v[k] for k in range(3))
        cross = (d[1] * v[2] - d[2] * v[1], d[2] * v[0] - d[0] * v[2], d[0] * v[1] - d[1] * v[0])
        return tuple(v[i] * c + cross[i] * s + d[i] * dv * (1 - c) for i in range(3))
    src = [l for l in MAH.splitlines() if l[17:26] == "NAG A 544"]
    for step in range(12):
        out = []
        for k, l in enumerate(src):
            q = add(t, spin(flip(sub((float(l[30:38]), float(l[38:46]), float(l[46:54])), nag["C1"])),
                            math.radians(30 * step)))
            out.append(hetatm(4700 + k, l[12:16].strip(), "NAG", "A", str(num), q, l[76:78].strip()))
        if nearest_other(out) > clear:
            return out
    return out


d, rc, log, j, unf = fuse("1ypz", fixture("1YPZ_o1.pdb"))
r = residue(j)
check("1YPZ: NAG L 1 with its O1 1.51 A from ND2 is fused (F3): NG1 = ASN G 133 + NAG L 1, NAG L 2, MAN L 3",
      r.get("resname") == "NG1" and r.get("instances", [{}])[0].get("sugars") == ["NAG L 1", "NAG L 2", "MAN L 3"]
      and not unf, "%s %s" % (log[-300:], unf))
check("  O1 dropped and logged, no O1 record in fused.pdb",
      "dropped the deposited leaving atom NAG L 1 O1" in log
      and not [l for l in lines_of(d) if l[12:16].strip().startswith("O1") and l[17:20] == "NG1"], log[-300:])
out = unit(sub(nag["C1"], asn["ND2"]))
moved = []
for l in MAH.splitlines():
    if l[17:26] == "NAG A 544":
        p = add((float(l[30:38]), float(l[38:46]), float(l[46:54])), out, 0.85)
        l = l[:30] + "%8.3f%8.3f%8.3f" % p + l[54:]
    moved.append(l)
c1 = add(nag["C1"], out, 0.85)                          # C1 now 2.30 A from ND2: no bond
u = unit(sub(asn["ND2"], c1))
o5 = sub(add(nag["O5"], out, 0.85), c1)
v = unit(sub(tuple(-x for x in o5), tuple(sum(o5[k] * u[k] for k in range(3)) * -u[i] for i in range(3))))
t39 = math.radians(39.1)                                # puts O1 1.50 A from ND2 at 1.40 A from C1
o1 = add(c1, add(tuple(math.cos(t39) * x for x in u), tuple(math.sin(t39) * x for x in v)), 1.40)
only_o1 = "\n".join(moved).replace(next(l for l in moved if l[17:26] == "NAG A 544" and l[12:16] == " C1 "),
                                   next(l for l in moved if l[17:26] == "NAG A 544" and l[12:16] == " C1 ") + "\n"
                                   + hetatm(4593, "O1", "NAG", "A", "544", o1, "O")) + "\n"
d, rc, log, j, unf = fuse("only_o1", only_o1)
check("C1 2.30 A from ND2, the deposited O1 1.50 A from it: refused, 'only through ... leaving atom O1'",
      not j.get("residues") and unf == ["NAG A 544 bonded to the protein only through the CCD leaving atom O1 "
                                        "(NAG A 544 O1-ASN A 350 ND2), which the link replaces"]
      and abs(math.dist(o1, asn["ND2"]) - 1.50) < 0.01, "%s %.2f" % (unf, math.dist(o1, asn["ND2"])))
FZC = fixture("1FZC_partial.pdb")
d, rc, log, j, unf = fuse("1fzc", FZC)
r = residue(j)
check("1FZC: the 1.85 A link joined, 'joined stretched link NAG K 1 O4-NAG K 2 C1 1.85 A'",
      "joined stretched link NAG K 1 O4-NAG K 2 C1 1.85 A" in log, log[-400:])
check("  NG1 = ASN B 364 + NAG K 1, NAG K 2, bond O4A-C1B, nothing left as ligands",
      r.get("resname") == "NG1" and r.get("instances", [{}])[0].get("sugars") == ["NAG K 1", "NAG K 2"]
      and ["O4A", "C1B", 1] in r.get("bonds", []) and not unf, "%s %s" % (unf, log[-300:]))
check("  the instance records it: stretched_links ['NAG K 1 O4-NAG K 2 C1 1.85 A']",
      r.get("instances", [{}])[0].get("stretched_links") == ["NAG K 1 O4-NAG K 2 C1 1.85 A"], str(r.get("instances")))
# NAG K 2's O1 deposited as an atom of its own, 1.40 A from C1 away from NAG K 1 O4:
# C1 is then a reducing end next to O4, no link, and that stays refused.
fzc2 = coords(FZC, "NAG", "K", "2")
fzc1 = coords(FZC, "NAG", "K", "1")
o1 = add(fzc2["C1"], unit(sub(fzc2["C1"], fzc1["O4"])), 1.40)
fzc_o1 = FZC.replace(next(l for l in FZC.splitlines() if l[17:26] == "NAG K   2" and l[12:16] == " C1 "),
                     next(l for l in FZC.splitlines() if l[17:26] == "NAG K   2" and l[12:16] == " C1 ") + "\n"
                     + hetatm(11131, "O1", "NAG", "K", "2", o1, "O"))
d, rc, log, j, unf = fuse("1fzc_o1", fzc_o1)
why_fzc = ("NAG K 2 C1 lies 1.85 A from NAG K 1 O4, a glycosidic link longer than the covalent rule that is not "
           "joined (its leaving atom O1 is deposited, 1.40 A from C1)")
check("1FZC with NAG K 2 O1 deposited 1.40 A from C1: not joined, NAG K 1 not fused (F1)",
      not j.get("residues") and same_bytes(d) and "NOT fused NAG K 1: " in log and "joined stretched link" not in log,
      log[-400:])
check("  both NAG K 1 and NAG K 2 have a glycan_unfused.gs line with the reason",
      sorted(unf) == ["NAG K 1 " + why_fzc, "NAG K 2 " + why_fzc], str(unf))
d, rc, log, j, unf = fuse("2fd6", fixture("2FD6_site.pdb"))
r = residue(j)
check("2FD6: the site link joined, 'joined stretched link ASN U 172 ND2-NAG C 1 C1 1.99 A'",
      "joined stretched link ASN U 172 ND2-NAG C 1 C1 1.99 A" in log, log[-400:])
check("  NG1 = ASN U 172 + NAG C 1, NAG C 2, link ND2-C1A 1.99 A, both O1 dropped, nothing left",
      r.get("resname") == "NG1" and r.get("link") == ["ND2", "C1A"]
      and r.get("instances", [{}])[0].get("sugars") == ["NAG C 1", "NAG C 2"]
      and round(r.get("instances", [{}])[0].get("link_A", 0), 2) == 1.99 and not unf
      and "dropped the deposited leaving atom NAG C 1 O1" in log and "dropped the deposited leaving atom NAG C 2 O1" in log,
      "%s %s" % (unf, log[-400:]))
n_f, lig_f = renumber(d, "fused.pdb")
check("  renumber_pdb.py extracts no ligand from fused.pdb", not lig_f and n_f is not None, "%s %s" % (n_f, lig_f))
d, rc, log, j, unf = fuse("2po6", fixture("2PO6_flipped.pdb"))
r = residue(j)
po6 = coords(fixture("2PO6_flipped.pdb"), "ASN", "E", "42")
ng = coords(open(os.path.join(d, "fused.pdb")).read() if rc == 0 else "", "NG1", "E", "42")
check("2PO6: NAG E 301 on OD1 of ASN E 42 is fused as NG1, link ND2-C1A 1.34 A (F4)",
      r.get("resname") == "NG1" and r.get("link") == ["ND2", "C1A"]
      and round(r.get("instances", [{}])[0].get("link_A", 0), 2) == 1.34,
      "%s %s" % (r.get("instances", [{}])[0].get("link_A"), log[-300:]))
check("  the log says OD1 and ND2 were swapped", "ASN E 42: OD1 and ND2 swapped" in log, log[-400:])
check("  fused.pdb: ND2 where the deposit has OD1 and the other way round, elements N and O",
      ng.get("ND2") == po6.get("OD1") and ng.get("OD1") == po6.get("ND2")
      and [l[76:78] for l in lines_of(d) if l[17:26] == "NG1 E  42" and l[12:16] in (" OD1", " ND2")] == [" O", " N"],
      "%s %s" % (ng.get("ND2"), po6.get("OD1")))
swapped = "\n".join(l[:12] + {" O4 ": " O6 ", " O6 ": " O4 "}.get(l[12:16], l[12:16]) + l[16:] if l[17:26] == "NAG A 544"
                    else l for l in MAH.splitlines()) + "\n"
d, rc, log, j, unf = fuse("o4o6", swapped)
check("O4 and O6 named the other way round: refused, both stretched bonds named (F5)",
      not j.get("residues") and len(unf) == 1 and "C4-O4 3.15 A" in unf[0] and "C6-O6 3.06 A" in unf[0]
      and "do not fit the CCD entry's bonds" in unf[0], str(unf))
cc = nag_copy(545, nag["C6"], 1.53)
d, rc, log, j, unf = fuse("c_link", before_end(MAH, cc))
check("a NAG whose C1 lies 1.53 A from C6 of NAG A 544, nothing else within 2.3 A: refused (F8)",
      not j.get("residues") and nearest_other(cc) > 2.3
      and "NAG A 545 is linked to NAG A 544 C6, a C, not to a hydroxyl O" in (unf[0] if unf else ""),
      "%s %.2f" % (unf, nearest_other(cc)))
atom_nag = "\n".join(("ATOM  " + l[6:]) if l[17:26] == "NAG A 544" else l for l in MAH.splitlines()) + "\n"
d, rc, log, j, unf = fuse("atom_nag", atom_nag)
d_het = os.path.join(tmp, "1mah")
check("NAG A 544 as ATOM records: fused as NG1 like the HETATM one, the same fused.pdb (F7)",
      residue(j).get("resname") == "NG1" and open(os.path.join(d, "fused.pdb")).read()
      == open(os.path.join(d_het, "fused.pdb")).read(), log[-300:])
atom_no_o6 = "\n".join(("ATOM  " + l[6:]) if l[17:26] == "NAG A 544" else l for l in no_o6.splitlines()) + "\n"
d, rc, log, j, unf = fuse("atom_no_o6", atom_no_o6)
n_f, lig_f = renumber(d, "fused.pdb")
check("  refused (O6 missing): its records become HETATM in fused.pdb, renumber_pdb.py extracts the NAG",
      not j.get("residues") and unf == ["NAG A 544 NAG A 544: missing O6"]
      and lines_of(d) == [("HETATM" + l[6:]) if l[17:26] == "NAG A 544" else l for l in lines_of(d, "input.pdb")]
      and lig_f == ["ligand_NAG_A.pdb"], "%s %s %s" % (unf, lig_f, log[-200:]))

print("\n[fallback] a fused residue that failed to parametrise")
# The fallback cases keep the cap of 3 they were written for: the 9-sugar glycans stay
# ligands, and the 2- and 3-sugar ones are NG1 and NG2.
CAP3 = ("--max-sugars", "3")
d, rc, log, j, unf = fuse("fallback", XJE, *CAP3)
with open(os.path.join(d, "glycan_failed.gs"), "w") as f:
    f.write("NG1 AM1-BCC did not converge within the time cap\n")
rc, log, j, unf = again(d, "--exclude-file", "glycan_failed.gs", *CAP3)
fl = lines_of(d)
check("re-run with --exclude-file: NG1's glycan stays HETATM, ASN C 162 stays ASN",
      [x["resname"] for x in j["residues"]] == ["NG2"] and [x["resname"] for x in j["excluded"]] == ["NG1"]
      and any(l.startswith("HETATM") and l[17:26] == "NAG F   1" for l in fl)
      and any(l[17:26] == "ASN C 162" for l in fl), str([x["resname"] for x in j["residues"]]))
check("  its sugars' lines: 'NG1 could not be parametrised: AM1-BCC did not converge ...'",
      [l for l in unf if l.split()[1] == "F"] == [
          "NAG F 1 NG1 could not be parametrised: AM1-BCC did not converge within the time cap",
          "FUC F 2 NG1 could not be parametrised: AM1-BCC did not converge within the time cap"], str(unf[-2:]))
check("  NG2 keeps its name and atoms (ASN C 45 + NAG G 1 + NAG G 2 + BMA G 3)",
      j["residues"][0]["instances"][0]["sugars"] == ["NAG G 1", "NAG G 2", "BMA G 3"]
      and any(l[17:26] == "NG2 C  45" for l in fl), "")
rc, log, j, unf = again(d, "--exclude-file", "glycan_failed.gs", *CAP3)
check("  a further run gives the same names", [x["resname"] for x in j["residues"]] == ["NG2"]
      and [x["resname"] for x in j["excluded"]] == ["NG1"], str([x["resname"] for x in j["residues"]]))
d, rc, log, j, unf = fuse("fallback_new", XJE, *CAP3)
j["residues"] = [x for x in j["residues"] if x["resname"] != "NG2"]     # as if the first pass had not fused it
json.dump(j, open(os.path.join(d, "glycans.json"), "w"))
with open(os.path.join(d, "glycan_failed.gs"), "w") as f:
    f.write("NG1 AM1-BCC did not converge within the time cap\n")
rc, log, j, unf = again(d, "--exclude-file", "glycan_failed.gs", *CAP3)
check("  a chemistry the first pass did not fuse is not fused by the fallback (review R2)",
      j["residues"] == [] and [x["resname"] for x in j["excluded"]] == ["NG1"]
      and [l for l in unf if l.split()[1] == "G"] == [
          "%s G %d the first pass did not fuse this chemistry, so parametrize_ncaa.py made no parameters for it" % (s, n)
          for s, n in (("NAG", 1), ("NAG", 2), ("BMA", 3))], str(unf[-3:]))
d, rc, log, j, unf = fuse("old_names", MAH)
j["residues"][0]["resname"] = "NG5"
other = json.loads(json.dumps(j["residues"][0]))
other.update(resname="NG1", parent="SER")              # another chemistry holding NG1
j["residues"].append(other)
json.dump(j, open(os.path.join(d, "glycans.json"), "w"))
rc, log, j, unf = again(d)
check("an existing glycans.json lends its name to an unchanged chemistry (NG5 stays NG5)",
      [x["resname"] for x in j["residues"]] == ["NG5"] and "fused NG5: ASN A 350" in log, log[-200:])
j["residues"][0]["atoms"][0]["h"] = 2                  # the old NG5 is now another chemistry
j["residues"].append(other)
json.dump(j, open(os.path.join(d, "glycans.json"), "w"))
rc, log, j, unf = again(d)
check("  a changed chemistry gets a new name, clear of the old file's NG1 and NG5: NG2",
      [x["resname"] for x in j["residues"]] == ["NG2"], str([x["resname"] for x in j["residues"]]))

print("\n[cli] the script as the runners call it")
d = os.path.join(tmp, "cli")
os.makedirs(d)
shutil.copy(os.path.join(DATA, "1MAH_glycan.pdb"), os.path.join(d, "input.pdb"))
env = dict(os.environ, GROSCORE_CCD_CACHE=CCD_DATA, **PROXY)
p = subprocess.run([sys.executable, FG, "-f", "input.pdb", "-o", "fused.pdb", "-d", "glycans.json"], cwd=d, env=env,
                   capture_output=True, text=True, timeout=120)
check("python3 fuse_glycans.py on 1MAH: exit 0, the log line, fused.pdb and glycans.json",
      p.returncode == 0 and "fused NG1: ASN A 350 + NAG A 544 (ND2-C1 1.45 A)" in p.stdout
      and os.path.isfile(os.path.join(d, "fused.pdb")) and os.path.isfile(os.path.join(d, "glycans.json")),
      p.stdout[-300:] + p.stderr[-300:])
p = subprocess.run([sys.executable, FG, "-f", "missing.pdb", "-o", "fused.pdb", "-d", "glycans.json"], cwd=d, env=env,
                   capture_output=True, text=True, timeout=120)
check("an unreadable input: exit 1, 'cannot read missing.pdb'", p.returncode == 1 and "cannot read missing.pdb" in p.stderr,
      "%s %s" % (p.returncode, p.stderr[-200:]))
src = ast.parse(open(RN).read())
sets = {t.id: ast.literal_eval(n.value) for n in src.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name) and t.id in ("STANDARD_RESIDUES", "METAL_CLUSTERS", "ION_RESIDUES")}
check("renumber_pdb.py's STANDARD_RESIDUES, METAL_CLUSTERS and ION_RESIDUES are the ones used here",
      sets.get("STANDARD_RESIDUES") == F.STANDARD_RESIDUES and sets.get("METAL_CLUSTERS") == F.METAL_CLUSTERS
      and sets.get("ION_RESIDUES") == C.ION_RESIDUES, str(sorted(sets)))
check("ccd_cache.GLYCAN_PARENTS are the SITES (ASN, SER, THR)", set(C.GLYCAN_PARENTS) == set(F.SITES), str(C.GLYCAN_PARENTS))
check("no network call in the whole file", not net, str(net))

shutil.rmtree(tmp, ignore_errors=True)
print("\n%s" % ("ALL PASS" if fails == 0 else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
