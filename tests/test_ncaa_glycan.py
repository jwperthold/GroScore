#!/usr/bin/env python3
"""parametrize_ncaa.py builds a fused glycan residue from its glycans.json definition
and never looks its name up in the CCD.

utils/fuse_glycans.py makes a glycan covalently bonded to ASN ND2, SER OG or THR OG1
one residue with that amino acid (NG<k>, SG<k>, TG<k>) and defines its chemistry in
glycans.json. The NCAA path read such a name as a CCD code. NG1 is a real entry
(GalNAc 1-phosphate, parent A2G), so the fused 1MAH residue got parent A2G, OpenBabel
built it at +1, the 19-atom RCSB template failed, and NG1 went to ncaa_failed.gs, the
parent-replacement path (design run 1MAH_orig). Pinned with the network refused (a
sitecustomize makes urlopen raise in every child process and logs the attempt) and
the real NG1 entry as a decoy in GROSCORE_CCD_CACHE:

  [decoy]   ccd_parent(): the decoy cache gives NG1 the parent A2G; given the
            definitions, NG1 gets ASN and its CCD entry is not read.
  [merge]   merge_hdb() drops a stale entry of the same name with exactly its count
            lines, keeps every other entry, NGLN among them, and appends the new ones.
  [names]   fused_h_names() on a sialic-acid-like definition after protonation: the
            amino acid's hydrogens get the NCAA names (H, HA), the sugar's keep the
            definition's (H1A, HX01-HX03), the carboxylate's HX1 is dropped, and a
            hydrogen gained by N5A, which the definition gives none, takes the free
            base HX2, not H5A, the name of C5A's hydrogen (critic F5).
  [types]   assign_types_and_charges() on a Ser-like residue whose charges have 7
            decimals: written with the rtp's 6 decimals they sum exactly to the
            formal charge, the residual on CA (critic F7: rounded one by one they
            summed to -2e-6 e here and to -1.4e-5 e on the fused 1MAH residue); the CB
            hydrogens are H1 next to an oxygen, as amber19sb types SER and THR, and
            HC next to a carbon (critic F12c); H1 next to S, N or F as well, as
            amber19sb types the CB hydrogens of CYS (ncaa review F3: a Cys-derived
            NCAA such as CME kept HC), and H2 next to two of them.
  [rtp]     generate_rtp() writes the side-chain impropers in one order whatever
            order OpenFF lists them in (ncaa review F4: two runs on one input gave
            ncaa.rtp files that differed in that order only).
  [wall]    GROSCORE_GLYCAN_AM1BCC_WALL_S is the whole AM1-BCC time of a fused
            residue: with 100 s, a deposited conformer that fails after 95 s leaves
            5 s to the seed-1 retry, which is stopped there (runner review R3: each
            got the full wall, 185 s in all, and the residue was fused). Unset,
            there is no wall: a deposited conformer failing after 5000 s and a
            seed-1 run of 4000 s give charges after 9000 s (the ligand path's
            3600 s cap stopped the first attempt), and the ligand path keeps 3600 s.
  [1MAH]    the fragment around NG1 = ASN A 350 + NAG A 544: parent ASN, the CMAP
            'N-NG1 XC-NG1 C-NG1' copied from ASN, no network request, neither
            OpenBabel perception nor the RCSB template nor the MCS, and no failure
            file, though an earlier attempt left an ncaa_failed.gs and a
            glycan_failed.gs naming NG1 (critic F3: they were never removed); [ NG1 ] has
            41 uniquely named atoms whose written charges sum to 0 within 1e-6, the
            bond ND2 C1A, H2A on C2A and HN2A on N2A, and the CA correction stays
            within 0.05 e (prototype +0.011 e); one NG1 hdb entry of 15 lines, though
            the local force field held a stale one (critic F12a); pdb2gmx builds the
            rtp's 41 atoms, grompp with emin_vac.mdp gives no warning, and ND2-C1A is
            1.43-1.48 A after EM.
  [NGL]     a residue named NGL, part of NGLN, NGLU and NGLY in amber19sb's
            aminoacids.hdb, gets its own hdb entry (design 5(e): the substring test
            skipped it) and pdb2gmx builds all 14 of its atoms. NGL is defined here as
            the plain in-chain Asn, so only the name is under test.
  [amine]   the 1MAH residue without its acetyl, a glucosamine whose N2A carries 2
            hydrogens (named HX01 HX02 in the definition): OpenBabel's pH model makes
            it NH3+, the residue charge +1, and the gained group takes the free base
            HX1 (HX11-HX13), so that the hdb line '3 4 HX1 N2A ...' rebuilds it; the
            written charges sum to +1 within 1e-6 and pdb2gmx builds the rtp's 37
            atoms (critic F5: a protonated amine failed through an unnamed hydrogen).
  [fail]    a fused residue that cannot be built goes to glycan_failed.gs as
            'RESNAME reason', never to ncaa_failed.gs, while an ordinary NCAA without
            a CCD entry still goes to ncaa_failed.gs; hydrogen names no hdb line can
            rebuild (H61A H62A for the C6A pair) stop the residue within a minute,
            before any AM1-BCC run (critic F5); and GROSCORE_GLYCAN_AM1BCC_WALL_S
            bounds the first AM1-BCC attempt (1 s here), so that a run can force the
            glycan fallback (critic F4).
  [sialyl]  a synthetic sialyl-Tn, Thr-GalNAc(a1-O)-Neu5Ac(a2-6) (TG1): residue charge
            -1; the carboxylate keeps no H on O1AB or O1BB, its CCD hydrogen name HX1
            dropped before any sqm run (critic F5: the definition's names stopped
            every sialylated glycan after its AM1-BCC run); 77 uniquely named atoms;
            written charges sum to -1 within 1e-6; the CB hydrogen typed H1; one hdb
            line per parent of hydrogens; pdb2gmx builds the rtp's 77 atoms, grompp
            gives no warning, and both glycosidic bonds stay 1.38-1.50 A through EM.

Fixtures (tests/data/ncaa_glycan, PDB and CCD data are CC0): the 1MAH fragment is
residues 338-342 of fixed.pdb after fuse_glycans.py, renumber_pdb.py and fix_pdb.py
--keep-ncaa NG1, renumbered 1-5. The sialyl-Tn peptide Ala-Thr-Ala was assembled from
the CCD ideal coordinates of ALA, THR, A2G and SIA (each leaving atom on the atom
that replaces it), embedded with ETKDG and relaxed with MMFF, then fused and fixed the
same way; fuse_glycans.py logs its stereocentres as those of A2G and SIA.

The parametrisations run side by side, and TG1's AM1-BCC on 89 atoms sets the time:
616-642 s at a load of 5-17 on 24 cores, 759 s at 25 (the design's 89-atom 5XJE
residue took 532 s); the whole test took 621 and 647 s.

Run:  python3 tests/test_ncaa_glycan.py
"""
import ast, json, math, os, shutil, subprocess, sys, tempfile, threading, time
from decimal import Decimal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(REPO, "utils")
SCRIPT = os.path.join(UTILS, "parametrize_ncaa.py")
DATA = os.path.join(REPO, "tests", "data")
FIX = os.path.join(DATA, "ncaa_glycan")
DECOY = os.path.join(DATA, "ccd", "NG1.cif")
MDP = os.path.join(REPO, "settings", "amber19sb_opc3", "emin_vac.mdp")
WALL_ENV = "GROSCORE_GLYCAN_AM1BCC_WALL_S"
failures = []


def check(name, ok, detail=""):
    print("  %-74s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:400]))
    if not ok:
        failures.append(name)


def read(path):
    with open(path) as f:
        return f.read()


SRC = read(SCRIPT)
TREE = ast.parse(SRC).body
FUNCS = {n.name: n for n in TREE if isinstance(n, ast.FunctionDef)}
sys.path.insert(0, UTILS)
import ccd_cache  # noqa: E402

net_calls = []


def no_network(*a, **k):
    net_calls.append(a[0] if a else k)
    raise OSError("network disabled in this test")


ccd_cache.urllib.request.urlopen = no_network


def load(names):
    """A namespace with the module-level assignments and functions of parametrize_ncaa.py
    named in `names` (missing ones are left out)."""
    body = [n for n in TREE if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    ns = {"ccd_cache": ccd_cache}
    exec(compile(ast.Module(body=body, type_ignores=[]), SCRIPT, "exec"), ns)
    return ns


work = tempfile.mkdtemp(prefix="ncaa_glycan_")
cache = os.path.join(work, "ccd")
os.makedirs(cache)
shutil.copy(DECOY, cache)
nonet = os.path.join(work, "nonet")
os.makedirs(nonet)
with open(os.path.join(nonet, "sitecustomize.py"), "w") as f:
    f.write("import os, urllib.request\n"
            "def _refuse(*a, **k):\n"
            "    with open(os.environ['GLYCAN_TEST_NET_LOG'], 'a') as f:\n"
            "        f.write(repr(a[0] if a else k) + '\\n')\n"
            "    raise OSError('network refused in this test')\n"
            "urllib.request.urlopen = _refuse\n")

DEF_1MAH = json.loads(read(os.path.join(FIX, "1MAH_NG1_glycans.json")))
PDB_1MAH = read(os.path.join(FIX, "1MAH_NG1_fixed.pdb"))
DEF_STN = json.loads(read(os.path.join(FIX, "sialylTn_TG1_glycans.json")))
PDB_STN = read(os.path.join(FIX, "sialylTn_TG1_fixed.pdb"))


def stock_ff():
    """The amber19sb.ff of the GROMACS installation, found as parametrize_ncaa.py finds it."""
    dirs = ['/usr/local/gromacs/share/gromacs/top', '/usr/share/gromacs/top']
    out = subprocess.run(["gmx", "--version"], capture_output=True, text=True).stdout
    for line in out.split("\n"):
        if "Data prefix" in line:
            dirs.insert(0, line.split()[-1] + "/share/gromacs/top")
    return next(os.path.join(d, "amber19sb.ff") for d in dirs if os.path.isdir(os.path.join(d, "amber19sb.ff")))


def start(name, pdb_text, defs, env_extra=None, seed=None):
    """parametrize_ncaa.py --gmx-ff amber19sb started in its own directory on fixed.pdb,
    with glycans.json (when defs is given) and the files in seed written first (None:
    a copy of the stock amber19sb.ff)."""
    d = os.path.join(work, name)
    os.makedirs(d)
    with open(os.path.join(d, "fixed.pdb"), "w") as f:
        f.write(pdb_text)
    if defs is not None:
        with open(os.path.join(d, "glycans.json"), "w") as f:
            json.dump(defs, f, indent=1)
    for rel, text in (seed or {}).items():
        if text is None:
            shutil.copytree(stock_ff(), os.path.join(d, rel))
            continue
        with open(os.path.join(d, rel), "a") as f:
            f.write(text)
    env = {k: v for k, v in os.environ.items() if k not in ("GROSCORE_CCD_CACHE", WALL_ENV)}
    env.update(GROSCORE_CCD_CACHE=cache, GLYCAN_TEST_NET_LOG=os.path.join(d, "network.log"),
               PYTHONPATH=nonet + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
               https_proxy="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
               http_proxy="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9", no_proxy="", NO_PROXY="")
    env.update(env_extra or {})
    log = open(os.path.join(d, "pn.log"), "w")
    run = {"dir": d, "log": log, "t0": time.monotonic()}
    run["proc"] = subprocess.Popen([sys.executable, SCRIPT, "-f", "fixed.pdb", "--gmx-ff", "amber19sb"], cwd=d,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)

    def waiter():            # the end time, whenever finish() gets to this run
        run["proc"].wait()
        run["end"] = time.monotonic()

    run["waiter"] = threading.Thread(target=waiter, daemon=True)
    run["waiter"].start()
    return run


def finish(run, timeout=1800):
    run["waiter"].join(timeout)
    if run["waiter"].is_alive():
        run["proc"].kill()
        run["waiter"].join()
        run["rc"] = "timeout"
    else:
        run["rc"] = run["proc"].returncode
    run["log"].close()
    run["s"] = run["end"] - run["t0"]
    run["out"] = read(os.path.join(run["dir"], "pn.log"))
    return run


def lines_of(run, name):
    path = os.path.join(run["dir"], name)
    return [l.rstrip("\n") for l in open(path) if l.strip()] if os.path.isfile(path) else None


def net_log(run):
    return lines_of(run, "network.log") or []


def no_failure_files(run):
    return lines_of(run, "ncaa_failed.gs") is None and lines_of(run, "glycan_failed.gs") is None


def rtp_entry(run, resname):
    """(comment lines before it, [(name, type, charge text)], {frozenset bond}) of [ resname ]."""
    lines = read(os.path.join(run["dir"], "amber19sb.ff", "ncaa.rtp")).split("\n")
    i = lines.index("[ %s ]" % resname)
    k = i - 1
    while k >= 0 and lines[k].startswith(";"):
        k -= 1
    atoms, bonds, sec = [], set(), None
    for line in lines[i + 1:]:
        if line.startswith("[") or line.startswith(";"):
            break
        s = line.strip()
        if s.startswith("["):
            sec = s.strip("[] ")
        elif s and sec == "atoms":
            t = s.split()
            atoms.append((t[0], t[1], t[2]))
        elif s and sec == "bonds":
            bonds.add(frozenset(s.split()))
    return lines[k + 1:i], atoms, bonds


def rtp_or_nothing(run, resname):
    try:
        return rtp_entry(run, resname)
    except (OSError, ValueError) as e:
        check("ncaa.rtp holds [ %s ]" % resname, False, e)
        return [], [], set()


def hdb_entries(run, resname):
    path = os.path.join(run["dir"], "amber19sb.ff", "aminoacids.hdb")
    lines = read(path).split("\n") if os.path.isfile(path) else []
    return [lines[i + 1:i + 1 + int(l.split()[1])] for i, l in enumerate(lines)
            if len(l.split()) == 2 and l.split()[0] == resname and l.split()[1].isdigit()]


def gromacs(run, resname, pairs):
    """pdb2gmx with the runners' -ignh -missing, grompp emin_vac.mdp -maxwarn 0, EM on 4
    threads: {'names': atom names of resname in conf.gro, 'rc': [pdb2gmx, grompp,
    mdrun], 'warnings': grompp WARNING lines, 'dist': {pair: A after EM}}."""
    d = run["dir"]
    env = dict(os.environ, GMXLIB=d)
    res = {"names": [], "rc": [], "warnings": [], "dist": {}}
    if run["rc"] != 0:
        return res

    def gmx(*args):
        return subprocess.run(("gmx",) + args, cwd=d, env=env, capture_output=True, text=True, timeout=900)

    p = gmx("pdb2gmx", "-f", "fixed.pdb", "-o", "conf.gro", "-p", "topol.top", "-water", "none",
            "-ff", "amber19sb", "-ignh", "-missing")
    res["rc"].append(p.returncode)
    if p.returncode != 0:
        return res
    res["names"] = [l[10:15].strip() for l in read(os.path.join(d, "conf.gro")).split("\n")[2:-2]
                    if l[5:10].strip() == resname]
    gmx("editconf", "-f", "conf.gro", "-o", "box.gro", "-bt", "cubic", "-d", "1.5")
    p = gmx("grompp", "-f", MDP, "-c", "box.gro", "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "0")
    res["rc"].append(p.returncode)
    res["warnings"] = [l for l in (p.stdout + p.stderr).split("\n") if l.startswith("WARNING")]
    if p.returncode != 0:
        return res
    p = gmx("mdrun", "-nt", "4", "-deffnm", "em")
    res["rc"].append(p.returncode)
    xyz = {}
    if os.path.isfile(os.path.join(d, "em.gro")):
        for l in read(os.path.join(d, "em.gro")).split("\n")[2:-2]:
            if l[5:10].strip() == resname:
                xyz[l[10:15].strip()] = [10 * float(l[20 + 8 * k:28 + 8 * k]) for k in range(3)]
    res["dist"] = {(a, b): math.dist(xyz[a], xyz[b]) for a, b in pairs if a in xyz and b in xyz}
    return res


def charge_sum(atoms):
    return sum((Decimal(q) for _, _, q in atoms), Decimal(0))


def h_by_parent(bonds):
    """{heavy atom: [its hydrogens]} from rtp bonds (hydrogen names start with H)."""
    out = {}
    for b in bonds:
        x, y = sorted(b)
        for p, h in ((x, y), (y, x)):
            if h.startswith("H") and not p.startswith("H") and not p.startswith(("-", "+")):
                out.setdefault(p, []).append(h)
    return out


# ---------------------------------------------------------------------------
print("\n[decoy] the parent of a fused residue is its definition's")
if "ccd_parent" not in FUNCS:
    check("parametrize_ncaa.py defines ccd_parent()", False)
else:
    ns = load({"ccd_parent"})
    os.environ["GROSCORE_CCD_CACHE"] = cache
    fetched = []
    real_fetch = ccd_cache.fetch

    def spy(ccd_id, *a, **k):
        fetched.append(ccd_id)
        return real_fetch(ccd_id, *a, **k)

    ccd_cache.fetch = spy
    try:
        got = ns["ccd_parent"]("NG1")
        check("the decoy: the real NG1 entry in the cache names parent A2G", got == ("A2G", None), got)
        fetched.clear()
        try:
            got = ns["ccd_parent"]("NG1", {"NG1": DEF_1MAH["residues"][0]})
        except TypeError as e:
            got = e
        check("given the definitions, NG1 gets the definition's parent ASN", got == ("ASN", None), got)
        check("  and its CCD entry is not read", fetched == [], fetched)
    finally:
        ccd_cache.fetch = real_fetch
        os.environ.pop("GROSCORE_CCD_CACHE", None)
    check("  with no network request", not net_calls, net_calls)

# ---------------------------------------------------------------------------
print("\n[merge] merge_hdb() replaces an entry of the same name, word by word")
if "merge_hdb" not in FUNCS:
    check("parametrize_ncaa.py defines merge_hdb()", False)
else:
    ns = load({"merge_hdb"})
    old = ("ACE\t1\n3\t4\tH\tCH3\tC\tO\t\n"
           "NG1\t3\n1\t1\tH\tN\t-C\tCA\n; a comment inside the entry\n\n1\t5\tHA\tCA\tN\tC\tCB\n"
           "2\t6\tHB\tCB\tCA\tCG\n"
           "NGLN \t2\n1\t1\tH\tN\t-C\tCA\n1\t5\tHA\tCA\tN\tC\tCB\n")
    text, replaced = ns["merge_hdb"](old, ["NG1\t1\n1\t1\tH\tN\t-C\tCA", "NGL\t1\n1\t1\tH\tN\t-C\tCA"])
    want = ("ACE\t1\n3\t4\tH\tCH3\tC\tO\t\n"
            "NGLN \t2\n1\t1\tH\tN\t-C\tCA\n1\t5\tHA\tCA\tN\tC\tCB\n"
            "\nNG1\t1\n1\t1\tH\tN\t-C\tCA\n"
            "\nNGL\t1\n1\t1\tH\tN\t-C\tCA\n")
    check("the stale NG1 entry and its 3 count lines go, NGLN stays, NG1 and NGL follow", text == want,
          repr(text))
    check("  and NG1 is reported as replaced, NGL as new", replaced == ["NG1"], replaced)

# ---------------------------------------------------------------------------
print("\n[names] hydrogens named as they are after protonation")
ns = load({"HX_POOL", "assign_h_names", "fused_h_names"})
if "fused_h_names" not in ns:
    check("parametrize_ncaa.py defines fused_h_names()", False)
else:
    d = {"atoms": [{"name": n, "h": h} for n, h in (("N", 1), ("CA", 1), ("C1A", 1), ("C5A", 1), ("N5A", 0),
                                                       ("O1BA", 1), ("C11A", 3))],
         "hnames": {"C1A": ["H1A"], "C5A": ["H5A"], "O1BA": ["HX1"], "C11A": ["HX01", "HX02", "HX03"]}}
    parent = {10: "N", 11: "CA", 12: "C1A", 13: "C5A", 14: "N5A", 15: "C11A", 16: "C11A", 17: "C11A"}
    try:
        got, notes = ns["fused_h_names"](d, parent)
    except Exception as e:
        got, notes = "%s: %s" % (type(e).__name__, e), []
    check("H, HA, H1A, H5A, HX2 on N5A, HX01-HX03", got == {10: "H", 11: "HA", 12: "H1A", 13: "H5A", 14: "HX2",
                                                           15: "HX01", 16: "HX02", 17: "HX03"}, got)
    check("  the dropped and the gained name in the notes",
          sorted(notes) == ["N5A gained hydrogens at the pH (0 -> 1): named HX2",
                            "O1BA keeps 0 of 1 hydrogens at the pH: HX1 dropped"], notes)

# ---------------------------------------------------------------------------
print("\n[types] written charges sum exactly to the formal charge; CB hydrogens next to O are H1")
ns = load({"BACKBONE_FIXED_CHARGES", "BACKBONE_TYPES", "CB_TYPE", "CB_H_TYPE", "CB_O_H_TYPE", "CB_H_TYPES",
           "ELECTRON_WITHDRAWING", "build_adjacency", "assign_types_and_charges"})
# A Ser-like residue as OpenFF exports it: OpenFF charges with 7 decimals.
ATOMS = [("N", "n", 7, -0.3), ("H", "hn", 1, 0.2), ("CA", "c4", 6, 0.0123454), ("HA", "h1", 1, 0.0987654),
         ("CB", "c4", 6, 0.1111114), ("HB1", "hc", 1, 0.0555554), ("HB2", "hc", 1, 0.0555554),
         ("OG", "oh", 8, -0.6543214), ("HG", "ho", 1, 0.4012344), ("C", "c", 6, 0.5), ("O", "o", 8, -0.5)]
BONDS = [(0, 1), (0, 2), (2, 3), (2, 4), (2, 9), (4, 5), (4, 6), (4, 7), (7, 8), (9, 10)]
for og, og_type, og_z, want_hb in (("OG", "oh", 8, "H1"), ("CG", "c3", 6, "HC"), ("SG", "ss", 16, "H1"),
                                   ("NG", "n3", 7, "H1"), ("FG", "f", 9, "H1")):
    atoms = [(og, og_type, og_z, q) if n == "OG" else (n, t, z, q) for n, t, z, q in ATOMS]
    top = {"atomtypes": [{"name": t, "at_num": z} for _, t, z, _ in atoms],
           "atoms": [{"type": t, "charge": q} for _, t, _, q in atoms],
           "bonds": [(i, j, 1, 0.1, 1000.0) for i, j in BONDS]}
    heavy = {i: a[0] for i, a in enumerate(atoms) if a[2] > 1}
    h_par = {1: "N", 3: "CA", 5: "CB", 6: "CB", 8: og}
    h_nam = {1: "H", 3: "HA", 5: "HB1", 6: "HB2", 8: "H" + og}
    try:
        type_map, charge_map, _ = ns["assign_types_and_charges"](heavy, h_par, h_nam, top, "SXX", 0)
    except Exception as e:
        check("assign_types_and_charges() runs (CB on %s)" % og, False, "%s: %s" % (type(e).__name__, e))
        continue
    written = {i: "%12.6f" % q for i, q in charge_map.items()}
    total = sum((Decimal(s) for s in written.values()), Decimal(0))
    if og == "OG":
        check("the 11 charges as written (6 decimals) sum to 0 exactly", len(written) == 11 and total == 0, total)
        check("  the residual on CA alone (CA 0.046501, HA 0.098765, OG -0.654321)",
              [written[i].strip() for i in (2, 3, 7)] == ["0.046501", "0.098765", "-0.654321"],
              [written[i].strip() for i in (2, 3, 7)])
    check("the CB hydrogens are %s when CB is bonded to %s" % (want_hb, og),
          [type_map[5], type_map[6]] == [want_hb, want_hb], [type_map[5], type_map[6]])
# Two electron-withdrawing neighbours: OG and a CL in the place of HB2.
atoms = [("CL", "cl", 17, -0.0555554) if k == 6 else a for k, a in enumerate(ATOMS)]
top = {"atomtypes": [{"name": t, "at_num": z} for _, t, z, _ in atoms],
       "atoms": [{"type": t, "charge": q} for _, t, _, q in atoms],
       "bonds": [(i, j, 1, 0.1, 1000.0) for i, j in BONDS]}
try:
    type_map, charge_map, _ = ns["assign_types_and_charges"]({i: a[0] for i, a in enumerate(atoms) if a[2] > 1},
                                                             {1: "N", 3: "CA", 5: "CB", 8: "OG"},
                                                             {1: "H", 3: "HA", 5: "HB1", 8: "HG"}, top, "SXX", 0)
    got = type_map[5]
except Exception as e:
    got = "%s: %s" % (type(e).__name__, e)
check("the CB hydrogen is H2 when CB is bonded to OG and CL", got == "H2", got)

# ---------------------------------------------------------------------------
print("\n[rtp] the side-chain impropers in one order")
ns = load({"BACKBONE_TYPES", "BACKBONE_NAMES", "generate_rtp"})
names = ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"]
amap = dict(enumerate(names))
fake = {"bonds": [(0, 1, 1, 0.1, 1.0), (1, 2, 1, 0.1, 1.0), (2, 3, 1, 0.1, 1.0), (1, 4, 1, 0.1, 1.0),
                  (4, 5, 1, 0.1, 1.0), (5, 6, 1, 0.1, 1.0), (5, 7, 1, 0.1, 1.0)]}
imps = [(5, 4, 6, 7, 4, 180.0, 4.6, 2), (4, 6, 5, 7, 4, 180.0, 4.6, 2), (7, 6, 5, 4, 4, 180.0, 4.6, 2)]
out = []
for order in (imps, imps[::-1], imps[1:] + imps[:1]):
    try:
        out.append(ns["generate_rtp"]("NXX", amap, {}, {}, names, {i: "T" for i in amap}, {i: 0.0 for i in amap},
                                      dict(fake, impropers=list(order)), set(amap), {}))
    except Exception as e:
        out.append("%s: %s" % (type(e).__name__, e))
check("three orders of the same three impropers give one rtp text", len(set(out)) == 1, out[:2])
# the [ impropers ] block: the two peptide impropers, then the side chain's
side = out[0].split(" [ impropers ]\n")[-1].split("\n [ cmap ]")[0].split("\n")[2:]
check("  and the side chain's are sorted", len(side) == 3 and side == sorted(side), side)

# ---------------------------------------------------------------------------
print("\n[wall] %s is the whole AM1-BCC time of a fused residue" % WALL_ENV)
ns = load({"GLYCAN_WALL_ENV", "fused_interchange"})
ns["os"] = os
import parametrize_ligand as PLM  # noqa: E402  (standard library at import)
now, calls = [0.0], []


def fake_attempt(offmol, conformer, timeout=None):
    """sqm on a fake clock: the deposited conformer fails after 95 s, a seeded one
    converges after 90 s, and an attempt past `timeout` is stopped there with a
    TimeoutError, as am1bcc_attempt() stops it."""
    run = 95.0 if conformer == "deposited" else 90.0
    calls.append((conformer, timeout))
    if timeout is not None and run > timeout:
        now[0] += timeout
        raise TimeoutError("AM1-BCC did not finish within %.0f s" % timeout)
    now[0] += run
    if conformer == "deposited":
        raise RuntimeError("Cannot properly run sqm (fake: no convergence)")
    return [0.0]


class FakeMol(object):
    conformers = ["deposited"]


saved = (PLM.am1bcc_attempt, PLM.seeded_conformer, PLM.make_interchange, PLM.AM1BCC_RETRY_WALL_S)
PLM.am1bcc_attempt = fake_attempt
PLM.seeded_conformer = lambda offmol, seed: "seed%d" % seed
PLM.make_interchange = lambda ff, offmol: PLM.am1bcc_charges(offmol, clock=lambda: now[0])
os.environ[WALL_ENV] = "100"
try:
    try:
        got = ns["fused_interchange"](None, FakeMol(), clock=lambda: now[0])
    except TypeError:                       # one without a clock: its own time, the fake attempts' clock
        got = ns["fused_interchange"](None, FakeMol())
except PLM.LigandFailure as e:
    got = "LigandFailure: %s" % e
except Exception as e:
    got = "%s: %s" % (type(e).__name__, e)
finally:
    PLM.am1bcc_attempt, PLM.seeded_conformer, PLM.make_interchange, PLM.AM1BCC_RETRY_WALL_S = saved
    os.environ.pop(WALL_ENV, None)
check("%s=100: at most 100 s of AM1-BCC in all (fake clock)" % WALL_ENV, now[0] <= 100.0, (now[0], calls))
check("  the seed-1 retry got the 5 s left and was stopped there: LigandFailure",
      calls == [("deposited", 100.0), ("seed1", 5.0)] and str(got).startswith("LigandFailure: AM1-BCC charges failed"),
      (calls, str(got)[:300]))
# Unset, there is no wall: a large glycan whose deposited conformer fails after 5000 s
# and whose seed-1 run converges after 4000 s more keeps its fused residue. The ligand
# path's 3600 s cap stopped the first attempt at 3600 s.
now[0] = 0.0
del calls[:]


def slow_attempt(offmol, conformer, timeout=None):
    run = 5000.0 if conformer == "deposited" else 4000.0
    calls.append((conformer, timeout))
    if timeout is not None and run > timeout:
        now[0] += timeout
        raise TimeoutError("AM1-BCC did not finish within %.0f s" % timeout)
    now[0] += run
    if conformer == "deposited":
        raise RuntimeError("Cannot properly run sqm (fake: no convergence)")
    return [0.0]


saved = (PLM.am1bcc_attempt, PLM.seeded_conformer, PLM.make_interchange, PLM.AM1BCC_RETRY_WALL_S)
PLM.am1bcc_attempt = slow_attempt
PLM.seeded_conformer = lambda offmol, seed: "seed%d" % seed
PLM.make_interchange = lambda ff, offmol: PLM.am1bcc_charges(offmol, clock=lambda: now[0])
os.environ.pop(WALL_ENV, None)
try:
    try:
        got = ns["fused_interchange"](None, FakeMol(), clock=lambda: now[0])
    except TypeError:
        got = ns["fused_interchange"](None, FakeMol())
except PLM.LigandFailure as e:
    got = "LigandFailure: %s" % e
except Exception as e:
    got = "%s: %s" % (type(e).__name__, e)
finally:
    PLM.am1bcc_attempt, PLM.seeded_conformer, PLM.make_interchange, PLM.AM1BCC_RETRY_WALL_S = saved
check("%s unset: no wall, 9000 s of AM1-BCC, charges from seed 1" % WALL_ENV,
      calls == [("deposited", None), ("seed1", None)] and now[0] == 9000.0 and isinstance(got, tuple)
      and "seed 1" in str(got[1]), (calls, now[0], str(got)[:300]))
check("  and the ligand path's wall is back afterwards (3600 s)", PLM.AM1BCC_RETRY_WALL_S == 3600.0,
      PLM.AM1BCC_RETRY_WALL_S)

# ---------------------------------------------------------------------------
# The parametrisations, side by side: TG1 first, the longest.
ngl_def = {"resname": "NGL", "family": "NG", "parent": "ASN", "link": [], "hnames": {}, "sugars": [],
           "instances": [], "atoms": [], "bonds": []}
for a in DEF_1MAH["residues"][0]["atoms"]:
    if a["name"] in ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"):
        ngl_def["atoms"].append(dict(a, h=a["h"] + (1 if a["name"] == "ND2" else 0)))
names_ngl = {a["name"] for a in ngl_def["atoms"]}
ngl_def["bonds"] = [b for b in DEF_1MAH["residues"][0]["bonds"] if b[0] in names_ngl and b[1] in names_ngl]
pdb_ngl = "".join(l.replace(" NG1 ", " NGL ") + "\n" for l in PDB_1MAH.split("\n")
                  if l and not (l[17:20] == "NG1" and l[12:16].strip() not in names_ngl))
fail_def = json.loads(json.dumps(DEF_1MAH))
for a in fail_def["residues"][0]["atoms"]:
    if a["name"] == "C1A":
        a["h"] = 0
pdb_zq9 = "".join(l.replace(" LYS A   1", " ZQ9 A   1") + "\n" for l in PDB_1MAH.split("\n") if l)
hdb_def = json.loads(json.dumps(DEF_1MAH))
hdb_def["residues"][0]["hnames"]["C6A"] = ["H61A", "H62A"]
# Glucosamine: the GlcNAc without C7A, O7A and C8A, N2A with 2 hydrogens. fuse_glycans.py
# would name them HX01 HX02 (HN2A1 has 5 characters).
ACETYL = ("C7A", "O7A", "C8A")
amine_def = json.loads(json.dumps(DEF_1MAH))
res = amine_def["residues"][0]
res["atoms"] = [dict(a, h=2) if a["name"] == "N2A" else a for a in res["atoms"] if a["name"] not in ACETYL]
res["bonds"] = [b for b in res["bonds"] if b[0] not in ACETYL and b[1] not in ACETYL]
res["hnames"] = {k: v for k, v in res["hnames"].items() if k not in ACETYL}
res["hnames"]["N2A"] = ["HX01", "HX02"]
pdb_amine = "".join(l + "\n" for l in PDB_1MAH.split("\n") if l and not (l[17:20] == "NG1" and l[12:16].strip() in ACETYL))

t_all = time.monotonic()
runs = {
    "sialyl": start("sialyl", PDB_STN, DEF_STN),
    "1MAH": start("1MAH", PDB_1MAH, DEF_1MAH,
                  seed={"amber19sb.ff": None,
                        os.path.join("amber19sb.ff", "aminoacids.hdb"): "\nNG1\t2\n1\t1\tH\tN\t-C\tCA\n"
                                                                         "1\t5\tHA\tCA\tN\tC\tCB\n",
                        "ncaa_failed.gs": "NG1\n", "glycan_failed.gs": "NG1 an earlier attempt\n"}),
    "NGL": start("NGL", pdb_ngl, {"residues": [ngl_def]}),
    "amine": start("amine", pdb_amine, amine_def),
    "route": start("route", pdb_zq9, fail_def),
    "hdbfault": start("hdbfault", PDB_1MAH, hdb_def),
    "wall": start("wall", PDB_1MAH, DEF_1MAH, env_extra={WALL_ENV: "1"}),
}

# ---------------------------------------------------------------------------
print("\n[1MAH] NG1 = ASN A 350 + NAG A 544")
r = finish(runs["1MAH"])
out = r["out"]
check("parametrize_ncaa.py exits 0", r["rc"] == 0, (r["rc"], out[-600:]))
check("  no ncaa_failed.gs and no glycan_failed.gs, though an earlier attempt left both",
      no_failure_files(r), (lines_of(r, "ncaa_failed.gs"), lines_of(r, "glycan_failed.gs")))
check("  parent ASN, its CMAP copied", "NCAA NG1: parent residue = ASN" in out
      and "Added CMAP for NG1 (copied from ASN)" in out, out[-600:])
check("  no OpenBabel perception, RCSB template or MCS", not any(
    s in out for s in ("Strategy 1", "Strategy 2", "MCS", "RCSB")), out[:800])
check("  no network request", net_log(r) == [], net_log(r))
cmap_path = os.path.join(r["dir"], "amber19sb.ff", "cmap.itp")
check("cmap.itp holds 'N-NG1 XC-NG1 C-NG1'", os.path.isfile(cmap_path) and "N-NG1 XC-NG1 C-NG1" in read(cmap_path))
comment, atoms, bonds = rtp_or_nothing(r, "NG1")
names = [a[0] for a in atoms]
check("[ NG1 ] has 41 atoms with unique names", len(names) == 41 and len(set(names)) == 41, names)
q = charge_sum(atoms)
check("  its written charges sum to 0 within 1e-6", atoms != [] and abs(q) < Decimal("1e-6"), q)
check("  the link bond ND2 C1A", frozenset(("ND2", "C1A")) in bonds)
check("  H2A on C2A and HN2A on N2A", frozenset(("C2A", "H2A")) in bonds and frozenset(("N2A", "HN2A")) in bonds)
check("  a comment names ASN and NAG", any("NG1: ASN + NAG (tag A)" in c for c in comment), comment)
corr = [float(l.split(":")[1].split()[0]) for l in out.split("\n") if "Charge correction on CA" in l]
check("  the CA correction stays within 0.05 e (prototype +0.011 e)",
      len(corr) == 1 and -0.05 < corr[0] < 0.05, corr)
hdb = hdb_entries(r, "NG1")
check("one NG1 hdb entry of 15 lines in place of the stale one", [len(e) for e in hdb] == [15],
      [len(e) for e in hdb])
g = gromacs(r, "NG1", [("ND2", "C1A")])
check("pdb2gmx builds the 41 atoms of the rtp", g["rc"][:1] == [0] and len(names) == 41
      and sorted(g["names"]) == sorted(names), (g["rc"], len(g["names"])))
check("grompp emin_vac.mdp without -maxwarn: no warning", g["rc"][1:2] == [0] and g["warnings"] == [],
      g["warnings"])
link = g["dist"].get(("ND2", "C1A"))
check("ND2-C1A after EM is 1.43-1.48 A", link is not None and 1.43 <= link <= 1.48, link)

# ---------------------------------------------------------------------------
print("\n[NGL] a residue named NGL gets its own hdb entry")
r = finish(runs["NGL"])
check("parametrize_ncaa.py exits 0 without a failure file", r["rc"] == 0 and no_failure_files(r),
      (r["rc"], r["out"][-600:]))
hdb = hdb_entries(r, "NGL")
check("one NGL hdb entry of 4 lines (N, CA, CB, ND2)", [len(e) for e in hdb] == [4], [len(e) for e in hdb])
stock = read(os.path.join(stock_ff(), "aminoacids.hdb"))
local_path = os.path.join(r["dir"], "amber19sb.ff", "aminoacids.hdb")
local = read(local_path) if os.path.isfile(local_path) else ""
check("  the stock entries kept as they were, NGLN, NGLU and NGLY among them",
      local.startswith(stock.rstrip("\n")) and all(n in local for n in ("NGLN", "NGLU", "NGLY")))
names = [a[0] for a in rtp_or_nothing(r, "NGL")[1]]
g = gromacs(r, "NGL", [])
check("pdb2gmx builds the 14 atoms of the rtp", len(names) == 14 and g["rc"][:1] == [0]
      and sorted(g["names"]) == sorted(names), (len(names), g["rc"], g["names"]))

# ---------------------------------------------------------------------------
print("\n[amine] a glucosamine N2A gains a hydrogen at pH 7.4")
r = finish(runs["amine"])
out = r["out"]
check("parametrize_ncaa.py exits 0 without a failure file", r["rc"] == 0 and no_failure_files(r),
      (r["rc"], out[-600:]))
check("  formal charge +1, the gained group named from the free base HX1",
      "formal charge +1" in out and "N2A gained hydrogens at the pH (2 -> 3): named HX11 HX12 HX13" in out,
      out[:1500])
comment, atoms, bonds = rtp_or_nothing(r, "NG1")
names = [a[0] for a in atoms]
check("[ NG1 ] has 37 atoms with unique names, HX11-HX13 on N2A",
      len(names) == 37 and len(set(names)) == 37 and sorted(h_by_parent(bonds).get("N2A", [])) ==
      ["HX11", "HX12", "HX13"], (len(names), h_by_parent(bonds).get("N2A")))
q = charge_sum(atoms)
check("  its written charges sum to +1 within 1e-6", atoms != [] and abs(q - 1) < Decimal("1e-6"), q)
hdb = hdb_entries(r, "NG1")
check("  the hdb line '3 4 HX1 N2A ...' rebuilds them",
      len(hdb) == 1 and any(l.split("\t")[:4] == ["3", "4", "HX1", "N2A"] for l in hdb[0]), hdb)
g = gromacs(r, "NG1", [])
check("pdb2gmx builds the 37 atoms of the rtp", g["rc"][:1] == [0] and len(names) == 37
      and sorted(g["names"]) == sorted(names), (g["rc"], len(g["names"])))

# ---------------------------------------------------------------------------
print("\n[fail] glycan_failed.gs for a fused residue, ncaa_failed.gs for an NCAA")
r = finish(runs["route"])
check("parametrize_ncaa.py exits 0", r["rc"] == 0, (r["rc"], r["out"][-600:]))
check("  ZQ9 (no CCD entry) in ncaa_failed.gs, alone", lines_of(r, "ncaa_failed.gs") == ["ZQ9"],
      lines_of(r, "ncaa_failed.gs"))
check("  NG1 (C1A defined without its hydrogen) in glycan_failed.gs with the reason",
      lines_of(r, "glycan_failed.gs") == ["NG1 radical atoms in the definition: C1A"],
      lines_of(r, "glycan_failed.gs"))
check("  the network was asked for ZQ9 only (the refusal works, NG1 was not looked up)",
      len(net_log(r)) >= 1 and all("ZQ9" in l for l in net_log(r)), net_log(r))
r = finish(runs["hdbfault"])
check("hydrogen names no hdb line rebuilds: NG1 in glycan_failed.gs, saying so",
      r["rc"] == 0 and lines_of(r, "glycan_failed.gs") == [
          "NG1 no hdb entry can rebuild its hydrogens: C6A: the hdb line makes H61A1 H61A2, the rtp has H61A H62A"]
      and lines_of(r, "ncaa_failed.gs") is None, (r["rc"], lines_of(r, "glycan_failed.gs"), r["out"][-600:]))
check("  before any AM1-BCC run, within 60 s", "AM1-BCC" not in r["out"] and r["s"] < 60, r["s"])
r = finish(runs["wall"])
gf = lines_of(r, "glycan_failed.gs") or []
check("%s=1: NG1 in glycan_failed.gs, AM1-BCC stopped at 1 s" % WALL_ENV,
      r["rc"] == 0 and len(gf) == 1 and gf[0].startswith("NG1 ") and "AM1-BCC" in gf[0] and "1 s" in gf[0],
      (r["rc"], gf, r["out"][-600:]))
check("  no ncaa_failed.gs, no local force field", lines_of(r, "ncaa_failed.gs") is None
      and not os.path.isdir(os.path.join(r["dir"], "amber19sb.ff")))
check("  within 90 s (an unbounded first attempt takes 100-160 s and succeeds)", r["s"] < 90, r["s"])

# ---------------------------------------------------------------------------
print("\n[sialyl] TG1 = THR + GalNAc(a1-O) + Neu5Ac(a2-6), residue charge -1")
r = finish(runs["sialyl"])
out = r["out"]
check("parametrize_ncaa.py exits 0", r["rc"] == 0, (r["rc"], out[-600:]))
check("  no ncaa_failed.gs and no glycan_failed.gs", no_failure_files(r),
      (lines_of(r, "ncaa_failed.gs"), lines_of(r, "glycan_failed.gs")))
check("  parent THR, formal charge -1", "NCAA TG1: parent residue = THR" in out
      and "formal charge -1" in out, out[-800:])
check("  the carboxylate's hydrogen name dropped before the AM1-BCC run",
      "O1BB keeps 0 of 1 hydrogens at the pH: HX1 dropped" in out
      and 0 <= out.find("HX1 dropped") < out.find("AM1-BCC"), out[:1500])
check("  no network request", net_log(r) == [], net_log(r))
comment, atoms, bonds = rtp_or_nothing(r, "TG1")
names = [a[0] for a in atoms]
h_of = h_by_parent(bonds)
check("[ TG1 ] has 77 atoms with unique names", len(names) == 77 and len(set(names)) == 77, (len(names), names))
q = charge_sum(atoms)
check("  its written charges sum to -1 within 1e-6", atoms != [] and abs(q + 1) < Decimal("1e-6"), q)
check("  no hydrogen on the carboxylate oxygens O1AB and O1BB",
      "O1AB" in names and "O1BB" in names and not h_of.get("O1AB") and not h_of.get("O1BB"),
      (h_of.get("O1AB"), h_of.get("O1BB")))
check("  the glycosidic bonds OG1 C1A and O6A C2B",
      frozenset(("OG1", "C1A")) in bonds and frozenset(("O6A", "C2B")) in bonds)
cb_h = [(a[0], a[1]) for a in atoms if a[0] in h_of.get("CB", [])]
check("  the CB hydrogen typed H1, as amber19sb types THR", len(cb_h) == 1 and cb_h[0][1] == "H1", cb_h)
hdb = hdb_entries(r, "TG1")
check("one TG1 hdb entry, one line for each of the 27 parents of hydrogens",
      len(h_of) == 27 and [len(e) for e in hdb] == [27], ([len(e) for e in hdb], len(h_of)))
g = gromacs(r, "TG1", [("OG1", "C1A"), ("O6A", "C2B")])
check("pdb2gmx builds the 77 atoms of the rtp", g["rc"][:1] == [0] and len(names) == 77
      and sorted(g["names"]) == sorted(names), (g["rc"], len(g["names"])))
check("grompp emin_vac.mdp without -maxwarn: no warning", g["rc"][1:2] == [0] and g["warnings"] == [],
      g["warnings"])
d1, d2 = g["dist"].get(("OG1", "C1A")), g["dist"].get(("O6A", "C2B"))
check("OG1-C1A and O6A-C2B after EM are 1.38-1.50 A",
      d1 is not None and d2 is not None and 1.38 <= d1 <= 1.50 and 1.38 <= d2 <= 1.50, (d1, d2))

print("\n(parametrisations: %s; %.0f s in all)"
      % (", ".join("%s %.0f s" % (k, r["s"]) for k, r in runs.items()), time.monotonic() - t_all))
print()
if failures:
    print("FAILED (%d), the runs kept in %s:" % (len(failures), work))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
shutil.rmtree(work, ignore_errors=True)
print("all fused-glycan NCAA checks passed")
