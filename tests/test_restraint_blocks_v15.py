#!/usr/bin/env python3
"""Which pull block each leg carries, and the elastic network in npt_c and nptrev.

Pinned on the mdps make_boresch.py actually writes for a synthetic complex, with
the real templates from settings/amber19sb_opc3 and the real fe_protocol:

  [1] fe_protocol.legs() names a block for every leg (contract C1): "bound" for
      the bound sub-legs and the holds inside the bound lambda schedule, "unbind"
      for the ramp stages, the ramp holds and nptrev.
  [2] holdbfwd1 and holdbrev1 carry EXACTLY the bound legs' coordinates: the same
      pairs, references, k and kB, rate 0, and no bor_ group. v1.3 gave them the
      unbind block (springs at 0.75 k_inter at their lambda 0.25 instead of 0.25,
      plus the Boresch at 25%), and every entry and exit did 6.4/7.3 kJ/mol of
      unrecorded work on average in the second HADDOCK pass.
  [3] nptrev carries the six Boresch coordinates and the WHOLE elastic network and
      no interface spring, all at rate 0. v1.3 dropped the network with the
      springs and it came back at bindrevI with a median 153 kJ/mol of strain.
  [4] npt_c_fe.mdp carries exactly the network the legs carry, k = kB = 250, rate
      0, no Boresch, no free-energy block; and a structure without a network gets
      pull = no with no live pull-* key (grompp warns once per such key). v1.3 ran
      npt_c without the network: 299 kJ/mol switched on unrecorded at boundfwd1,
      and 20 of 920 cycles lost at that grompp.
  [5] make_boresch.py stops with NPT_C_MDP_MISSING, before writing any block, when
      the structure directory has no npt_c_fe.mdp (contract C2).
  [6] the block follows the "block" key and not the leg name: a bound hold renamed
      to something without "holdb" in it still gets the bound block, and a leg
      without a block stops the setup (PROTOCOL_BLOCK_MISSING).
  [7] --pull-dist must equal fe_protocol.PULL_DIST: eq.32 releases the Boresch
      restraint at ref_r + --pull-dist, and the ramp ends at PULL_DIST
      (PULL_DIST_MISMATCH).
  [8] u is in nm everywhere (review R4): with PULL_DIST = 1.5, the ramp rescaled to
      end at 1.5 nm and --pull-dist 1.5, every stage's Boresch r starts at ref_r +
      u_from and moves u_to - u_from, no interface reference moves further than
      that, and nptrev holds r at ref_r + 1.5 nm, the r0_release eq.32 uses. The
      first v1.5 round scaled u by --pull-dist again: nptrev at ref_r + 2.25.
  [9] the leg mdps are read back as text (review R3): a run says every leg carries
      the computed Boresch references, and a writer that swaps theta_A/theta_B and
      mirrors phi_A, or one that forgets nptrev's u offset, stops the setup with
      PULL_INIT_MISMATCH. The first v1.5 round ran both to exit 0.
 [10] the setup log counts each partner's elastic-network bodies (critic, D3): one
      body when the network or a disulfide joins the fragments, two when nothing
      does.

Every property check fails on 431510c and passes now; [8]-[10] fail on the first
v1.5 round too. The lines that only confirm a run took place (and that the
synthetic complex has, or lacks, a network) pass on both. No GROMACS needed: the run
has no trajectory (snapshot references, burial anchors) and no readback through
GROMACS; the mdps are read back as text either way.

Standalone, no pytest: python3 tests/test_restraint_blocks_v15.py
"""
import atexit, os, re, shutil, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(ROOT, "utils")
TPL = os.path.join(ROOT, "settings", "amber19sb_opc3")
sys.path.insert(0, UTILS)
import fe_protocol as P

failures = []
# Every temporary directory, removed at exit even when a check raises.
TMP = []
atexit.register(lambda: [shutil.rmtree(x, ignore_errors=True) for x in TMP])


def check(name, ok, detail=""):
    print("  %-72s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


# ------------------------------------------------------------------ the complex
def write_complex(path, nfrag, gap=0.55, ss=False):
    """Two partners stacked 0.5 nm apart, each nfrag capped strands of five ALA
    (ACE ... NME), gap nm apart. Two or more fragments per side give an elastic
    network, since make_boresch anchors the network at the residues next to the
    caps; one gives none. With ss, the middle residues of the receptor's first two
    strands become CYS whose SG atoms meet halfway, 0.20 nm apart (a disulfide).
    Returns the ligand residue numbers."""
    lines, a, res, lig = [], 1, 1, []
    for side, z in (("A", 0.0), ("B", 0.5)):
        for f in range(nfrag):
            y = gap * f
            dz = 0.1 if side == "A" else -0.1
            atoms = [(res, "ACE", "CH3", -0.36, y, z), (res, "ACE", "C", -0.12, y, z),
                     (res, "ACE", "O", -0.12, y - 0.12, z)]
            if side == "B":
                lig.append(res)
            res += 1
            for j in range(5):
                x = 0.36 * j
                cys = ss and side == "A" and f < 2 and j == 2
                rn = "CYS" if cys else "ALA"
                atoms += [(res, rn, "N", x, y, z), (res, rn, "CA", x + 0.12, y + 0.08, z),
                          (res, rn, "C", x + 0.24, y, z), (res, rn, "O", x + 0.24, y - 0.12, z),
                          (res, rn, "CB", x + 0.12, y + 0.2, z + dz)]
                if cys:
                    atoms.append((res, rn, "SG", x + 0.12, gap / 2 + (-0.1 if f == 0 else 0.1), z))
                if side == "B":
                    lig.append(res)
                res += 1
            atoms += [(res, "NME", "N", 1.8, y, z), (res, "NME", "C", 1.92, y + 0.1, z)]
            if side == "B":
                lig.append(res)
            res += 1
            for r, rn, nm, x, yy, zz in atoms:
                lines.append("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (r, rn, nm, a, x + 4.0, yy + 4.0, zz + 4.0))
                a += 1
    with open(path, "w") as f:
        f.write("synthetic complex\n%5d\n" % len(lines))
        f.writelines(lines)
        f.write("  12.00000  12.00000  12.00000\n")
    return lig


HEADER = ("\n; ---- COM pulling (pull block appended by make_boresch.py) ----\n"
          "pull                     = yes\npull-print-com1          = no\n"
          "pull-print-com2          = no\npull-print-ref-value     = no\n"
          "pull-print-components    = no\npull-nstxout             = 0\n"
          "pull-nstfout             = 0\n")

WRAPPER = r'''
import runpy, sys
utils, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, utils)
import fe_protocol as P
if mode == "pd15":
    # a consistent change of the pull distance: 1.5 nm, the ramp rescaled to end there
    P.PULL_DIST = 1.5
    P.RAMP = [(s, a * 1.5, b * 1.5, ps) for s, a, b, ps in P.RAMP]
orig = P.legs
def legs():
    out = orig()
    if mode == "rename":
        # the bound-schedule holds under a name that says nothing about it
        for l in out:
            if l["name"].startswith("holdb"):
                l["name"] = l["name"].replace("holdb", "pausex")
                l["mdp"] = l["mdp"].replace("holdb", "pausex")
    elif mode == "noblock":
        for l in out:
            if l["name"] == "holdfwd3":
                l.pop("block", None)
    return out
P.legs = legs
sys.argv = [utils + "/make_boresch.py"] + sys.argv[3:]
runpy.run_path(utils + "/make_boresch.py", run_name="__main__")
'''


def setup(nfrag=2, npt_c=True, mode="plain", pull_dist="1.0", utils=UTILS, gap=0.55,
          ss=False):
    """A structure directory with the complex, the templates and (optionally)
    npt_c_fe.mdp; runs make_boresch from `utils` there. Returns (dir, exit code,
    stderr)."""
    d = tempfile.mkdtemp(prefix="blocks_v15_")
    TMP.append(d)
    lig = write_complex(os.path.join(d, "syn.gro"), nfrag, gap, ss)
    with open(os.path.join(d, "chain_map.gs"), "w") as f:
        f.write("# ligand residues\n" + "".join("%d\n" % r for r in lig))
    for l in P.legs():
        shutil.copy(os.path.join(TPL, l["mdp"]), d)
        if mode == "rename" and l["name"].startswith("holdb"):
            shutil.copy(os.path.join(TPL, l["mdp"]),
                        os.path.join(d, l["mdp"].replace("holdb", "pausex")))
    if npt_c:
        tpl = os.path.join(TPL, "npt_c_fe.mdp")
        if os.path.isfile(tpl):
            shutil.copy(tpl, d)
        else:                           # before v1.5 there was no template
            with open(os.path.join(d, "npt_c_fe.mdp"), "w") as f:
                f.write(open(os.path.join(TPL, "npt_fe.mdp")).read() + HEADER)
    wrap = os.path.join(d, ".wrap.py")
    open(wrap, "w").write(WRAPPER)
    r = subprocess.run([sys.executable, wrap, utils, mode, "-f", "syn.gro", "-m", "chain_map.gs",
                        "--pull-dist", pull_dist, "--traj", "none.xtc", "--tpr", "none.tpr",
                        "--index", "index.ndx", "--no-readback"],
                       cwd=d, capture_output=True, text=True, timeout=600)
    return d, r.returncode, r.stderr


def block(path):
    """(pull on?, [coords]) of an mdp; each coord as a dict with group NAMES."""
    kv, on = {}, None
    for line in open(path):
        s = line.split(";")[0]
        if "=" not in s:
            continue
        k, v = (x.strip() for x in s.split("=", 1))
        kv[k] = v
        if k == "pull":
            on = v
    groups = [kv["pull-group%d-name" % i] for i in range(1, int(kv.get("pull-ngroups", 0)) + 1)]
    co = []
    for c in range(1, int(kv.get("pull-ncoords", 0)) + 1):
        co.append(dict(geom=kv["pull-coord%d-geometry" % c],
                       groups=tuple(groups[int(x) - 1] for x in kv["pull-coord%d-groups" % c].split()),
                       init=kv["pull-coord%d-init" % c], rate=float(kv["pull-coord%d-rate" % c]),
                       k=float(kv["pull-coord%d-k" % c]), kB=float(kv["pull-coord%d-kB" % c])))
    return on, co


def is_bor(c):
    return any(g.startswith("bor_") for g in c["groups"])


def sig(c):
    return (c["geom"], c["groups"], c["init"], c["rate"], c["k"], c["kB"])


# ------------------------------------------------------------------ [1]
print("\n[1] fe_protocol names a block for every leg (contract C1)")
L = P.legs()
check("every leg carries a 'block'", all("block" in l for l in L),
      str([l["name"] for l in L if "block" not in l][:5]))
want = {l["name"]: ("bound" if l["kind"] == "bound" or
                    (l["kind"] == "hold" and l["u_from"] == 0.0 and l["lam_from"] == l["lam_to"]
                     and 0.0 < l["lam_from"] < 1.0) else "unbind") for l in L}
wrong = [l["name"] for l in L if l.get("block") != want[l["name"]]]
check("bound for the bound sub-legs and holdb*, unbind for every other leg",
      not wrong, str(wrong))
BOUND_HOLDS = [l["name"] for l in L if l["kind"] == "hold" and want[l["name"]] == "bound"]
print("      bound-schedule holds: %s" % " ".join(BOUND_HOLDS))

# ------------------------------------------------------------------ the main run
d, rc, err = setup(nfrag=2)
check("make_boresch.py runs on the two-fragment complex (exit %d)" % rc, rc == 0, err[-600:])
B = {l["name"]: block(os.path.join(d, l["mdp"])) for l in L} if rc == 0 else {}
EN = []
if rc == 0:
    _, cb = B["boundfwd1"]
    EN = [c for c in cb if c["k"] == 250.0 and c["kB"] == 250.0]
    IFACE = [c for c in cb if c not in EN]
    print("      boundfwd1: %d interface springs, %d elastic-network coordinates" % (len(IFACE), len(EN)))
    check("  the synthetic complex has an elastic network to test with", len(EN) > 0)

# ------------------------------------------------------------------ [2]
print("\n[2] the bound-schedule holds carry exactly the bound block")
if rc == 0:
    ref = [sig(c) for c in B["boundfwd2"][1]]
    for h in BOUND_HOLDS:
        co = B[h][1]
        check("%s: the same %d coordinates as boundfwd2 (pairs, init, rate 0, k, kB)"
              % (h, len(ref)), [sig(c) for c in co] == ref,
              "%d vs %d coords" % (len(co), len(ref)))
        check("%s: no bor_ group among its coordinates" % h, not any(is_bor(c) for c in co))
        ks = {(c["k"], c["kB"]) for c in co if c["k"] != c["kB"]}
        check("%s: interface springs at k 0 -> kB k_inter (%s)" % (h, sorted(ks)),
              len(ks) == 1 and list(ks)[0][0] == 0.0 and list(ks)[0][1] > 0.0, str(ks))

# ------------------------------------------------------------------ [3]
print("\n[3] nptrev: the Boresch coordinates and the whole network, no springs")
if rc == 0:
    _, cn = B["nptrev"]
    bor = [c for c in cn if is_bor(c)]
    rest = [c for c in cn if not is_bor(c)]
    print("      nptrev: %d Boresch, %d of the legs' %d network coordinates, %d other"
          % (len(bor), len([c for c in rest if sig(c) in {sig(e) for e in EN}]), len(EN),
             len([c for c in rest if sig(c) not in {sig(e) for e in EN}])))
    check("exactly the six Boresch coordinates and every network coordinate, at "
          "rate 0, and no interface spring",
          len(bor) == 6 and sorted(sig(c) for c in rest) == sorted(sig(e) for e in EN)
          and all(c["rate"] == 0.0 for c in cn)
          and not any(c["groups"] in {i["groups"] for i in IFACE} for c in cn))

# ------------------------------------------------------------------ [4]
print("\n[4] npt_c_fe.mdp: the network on its own, or pull = no without one")
if rc == 0:
    on, cc = block(os.path.join(d, "npt_c_fe.mdp"))
    check("pull = yes and exactly the legs' %d network coordinates (%d written)"
          % (len(EN), len(cc)),
          on == "yes" and sorted(sig(c) for c in cc) == sorted(sig(e) for e in EN))
    txt = open(os.path.join(d, "npt_c_fe.mdp")).read()
    check("all at k = kB = 250 kJ/mol/nm^2, rate 0, no Boresch group, no free-energy",
          bool(cc) and all(c["k"] == 250.0 and c["kB"] == 250.0 and c["rate"] == 0.0
                           and not is_bor(c) for c in cc)
          and re.search(r"^\s*free-energy\s*=\s*yes", txt, re.M) is None)
d1, rc1, err1 = setup(nfrag=1)
check("make_boresch.py runs on the one-fragment complex (exit %d)" % rc1, rc1 == 0, err1[-600:])
if rc1 == 0:
    on, cc = block(os.path.join(d1, "npt_c_fe.mdp"))
    _, cb1 = block(os.path.join(d1, "boundfwd1.mdp"))
    check("  which has no elastic network",
          not any(c["k"] == 250.0 and c["kB"] == 250.0 for c in cb1))
    check("npt_c_fe.mdp says pull = no", on == "no", str(on))
    live = [ln for ln in open(os.path.join(d1, "npt_c_fe.mdp"))
            if ln.split(";")[0].strip().lower().startswith("pull-")]
    check("and carries no live pull-* key (%d)" % len(live), not live, "".join(live[:3]))

# ------------------------------------------------------------------ [5]
print("\n[5] no npt_c_fe.mdp: the setup stops before writing anything")
d2, rc2, err2 = setup(nfrag=2, npt_c=False)
marker = os.path.join(d2, "boresch_failed.gs")
reason = open(marker).read().strip() if os.path.isfile(marker) else ""
check("exit code is not 0 (%d)" % rc2, rc2 != 0)
check("boresch_failed.gs says NPT_C_MDP_MISSING (%r)" % reason, reason == "NPT_C_MDP_MISSING")
check("no leg mdp received a pull block",
      not any("pull-ngroups" in open(os.path.join(d2, l["mdp"])).read() for l in L))

# ------------------------------------------------------------------ [6]
print("\n[6] the block follows the 'block' key, not the leg name")
d3, rc3, err3 = setup(nfrag=2, mode="rename")
check("make_boresch.py runs with the bound holds renamed (exit %d)" % rc3, rc3 == 0, err3[-600:])
if rc3 == 0:
    ref3 = [sig(c) for c in block(os.path.join(d3, "boundfwd2.mdp"))[1]]
    for h in BOUND_HOLDS:
        m = [l["mdp"] for l in L if l["name"] == h][0].replace("holdb", "pausex")
        got = [sig(c) for c in block(os.path.join(d3, m))[1]]
        check("%s (was %s) carries the bound block" % (m, h), got == ref3,
              "%d vs %d coords" % (len(got), len(ref3)))
d4, rc4, err4 = setup(nfrag=2, mode="noblock")
marker = os.path.join(d4, "boresch_failed.gs")
reason = open(marker).read().strip() if os.path.isfile(marker) else ""
check("a leg without 'block' stops the setup: PROTOCOL_BLOCK_MISSING (%r, exit %d)"
      % (reason, rc4), rc4 != 0 and reason == "PROTOCOL_BLOCK_MISSING")

# ------------------------------------------------------------------ [7]
print("\n[7] --pull-dist must be fe_protocol's PULL_DIST")
d5, rc5, err5 = setup(nfrag=2, pull_dist="0.8")
marker = os.path.join(d5, "boresch_failed.gs")
reason = open(marker).read().strip() if os.path.isfile(marker) else ""
check("--pull-dist 0.8 against PULL_DIST %g stops: PULL_DIST_MISMATCH (%r, exit %d)"
      % (P.PULL_DIST, reason, rc5), rc5 != 0 and reason == "PULL_DIST_MISMATCH")

# ------------------------------------------------------------------ [8]
print("\n[8] u is in nm everywhere: PULL_DIST 1.5, the ramp rescaled, --pull-dist 1.5")


def analytical(dd):
    out = {}
    for ln in open(os.path.join(dd, "boresch_analytical.gs")):
        f = ln.split()
        if len(f) >= 2 and not f[0].startswith("#"):
            out[f[0]] = f[1]
    return out


def leg_ps(path):
    kv = {}
    for ln in open(path):
        s = ln.split(";")[0]
        if "=" in s:
            k, v = (x.strip() for x in s.split("=", 1))
            kv[k] = v
    return int(float(kv["nsteps"])) * float(kv["dt"])


R_PAIR = ("bor_P3", "bor_L1")
d6, rc6, err6 = setup(nfrag=2, mode="pd15", pull_dist="1.5")
check("make_boresch.py runs with PULL_DIST 1.5 (exit %d)" % rc6, rc6 == 0, err6[-600:])
if rc6 == 0:
    an = analytical(d6)
    ref_r, r0 = float(an["ref_r_bound_nm"]), float(an["r0_release_nm"])
    print("      ref_r %.6f nm, r0_release %.6f nm" % (ref_r, r0))
    check("eq.32 releases at ref_r + 1.5 nm", abs(r0 - (ref_r + 1.5)) < 2e-6)
    # this process's fe_protocol is the real one (PULL_DIST 1.0); the run's is x 1.5
    U = [(l["mdp"], l["u_from"] * 1.5, l["u_to"] * 1.5, l["kind"]) for l in L
         if l["block"] == "unbind"]
    bad_r, bad_if, worst = [], [], 0.0
    for mdp, u0, u1, kind in U:
        ps = leg_ps(os.path.join(d6, mdp))
        _, co = block(os.path.join(d6, mdp))
        rr = [c for c in co if c["groups"] == R_PAIR and c["geom"] == "distance"]
        want_move = (u1 - u0) if kind == "stage" else 0.0
        if len(rr) != 1 or abs(float(rr[0]["init"]) - (ref_r + u0)) > 2e-6 \
                or abs(rr[0]["rate"] * ps - want_move) > 1e-6:
            bad_r.append((mdp, rr[0]["init"] if rr else None, ref_r + u0))
        for c in co:
            if not is_bor(c) and not (c["k"] == 250.0 and c["kB"] == 250.0):
                worst = max(worst, abs(c["rate"]) * ps - abs(u1 - u0))
                if abs(c["rate"]) * ps > abs(u1 - u0) + 1e-6:
                    bad_if.append(mdp)
    check("every unbind leg's Boresch r starts at ref_r + u_from and moves u_to - u_from"
          " (%d of %d legs wrong)" % (len(bad_r), len(U)), not bad_r, str(bad_r[:3]))
    check("no interface reference moves further than the stage (worst excess %+.4f nm)"
          % worst, not bad_if, str(sorted(set(bad_if))[:4]))
    _, cn = block(os.path.join(d6, "nptrev_fe.mdp"))
    rn = [float(c["init"]) for c in cn if c["groups"] == R_PAIR]
    check("nptrev holds r at r0_release (%s against %.6f)" % (rn, r0),
          len(rn) == 1 and abs(rn[0] - r0) < 2e-6)

# ------------------------------------------------------------------ [9]
print("\n[9] the leg mdps are read back: a corrupted writer stops the setup")


def patched_utils(pairs):
    """A copy of the utils make_boresch.py needs, with text replacements in it; None
    if a replacement site is not there exactly once."""
    u = tempfile.mkdtemp(prefix="blocks_v15_utils_")
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


check("a run says every leg mdp carries the computed Boresch references",
      rc == 0 and "every leg mdp carries the computed Boresch references" in err)
SWAP = [('groups=[gP3, gP2, gP3, gL1],\n                       init=ref_thA,',
         'groups=[gP3, gP2, gP3, gL1],\n                       init=ref_thB,'),
        ('groups=[gL1, gP3, gL1, gL2],\n                       init=ref_thB,',
         'groups=[gL1, gP3, gL1, gL2],\n                       init=ref_thA,'),
        ('groups=[gP1, gP2, gP2, gP3, gP3, gL1],\n                       init=ref_phA,',
         'groups=[gP1, gP2, gP2, gP3, gP3, gL1],\n                       init=-ref_phA,')]
HOLD = [('    for c in _co:\n      c["rate"] = 0.0',
         '    for c in _co:\n      c["rate"] = 0.0\n'
         '      if c.get("boresch") and c["geometry"] == "distance":\n'
         '        c["init"] = ref_r')]
# theta_A = theta_B = 90 deg in this synthetic pose, so the swap writes the same
# numbers; the mirrored phi_A is what shows (test_boresch_pbc [7] pins a swap of two
# different angles).
for label, pairs, name in (("theta_A/theta_B swapped and phi_A mirrored", SWAP, "(phi_A)"),
                           ("nptrev's r without its u offset", HOLD, "nptrev_fe.mdp")):
    u = patched_utils(pairs)
    check("the writer can be corrupted this way (%s)" % label, u is not None)
    if u is None:
        continue
    dx, rcx, errx = setup(nfrag=2, utils=u)
    mk = os.path.join(dx, "boresch_failed.gs")
    reason = open(mk).read().strip() if os.path.isfile(mk) else ""
    check("%s: PULL_INIT_MISMATCH naming %s (%r, exit %d)" % (label, name, reason, rcx),
          rcx != 0 and reason == "PULL_INIT_MISMATCH" and name in errx, errx[-300:])
    shutil.rmtree(dx, ignore_errors=True)

# ------------------------------------------------------------------ [10]
print("\n[10] the setup log counts the network bodies of each partner")
BODY = re.compile(r"elastic network, (receptor|ligand): (\d+) pairs; (\d+) backbone "
                  r"fragment\(s\), (\d+) network pair\(s\) and (\d+) disulfide\(s\) between "
                  r"them, (\d+) bod")


def bodies(text):
    return {m.group(1): tuple(int(m.group(k)) for k in range(2, 7)) for m in BODY.finditer(text)}


b0 = bodies(err)
print("      fragments 0.55 nm apart: %s" % b0)
check("joined by network pairs: 2 fragments, 1 body per partner",
      set(b0) == {"receptor", "ligand"} and all(v[1] == 2 and v[2] > 0 and v[4] == 1
                                                for v in b0.values()), str(b0))
d7, rc7, err7 = setup(nfrag=2, gap=2.5)
b7 = bodies(err7)
print("      fragments 2.5 nm apart: %s" % b7)
check("nothing joins them: 2 bodies per partner, said so (exit %d)" % rc7,
      rc7 == 0 and all(v[1] == 2 and v[2] == 0 and v[4] == 2 for v in b7.values())
      and len(b7) == 2 and "not the bodies to each other" in err7, str(b7))
d8, rc8, err8 = setup(nfrag=2, gap=2.5, ss=True)
b8 = bodies(err8)
print("      2.5 nm apart, a disulfide between the receptor's two: %s" % b8)
check("a disulfide joins the receptor's into 1 body; the ligand stays at 2 (exit %d)" % rc8,
      rc8 == 0 and b8.get("receptor", (0,) * 5)[3:] == (1, 1)
      and b8.get("ligand", (0,) * 5)[4] == 2, str(b8))

for x in TMP:
    shutil.rmtree(x, ignore_errors=True)
print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all v1.5 restraint-block checks passed")
