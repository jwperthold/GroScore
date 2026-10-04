#!/usr/bin/env python3
"""Interface pull references must stay non-negative along the whole ramp.

THE FAILURE. make_boresch.py builds each interface reference from two sources,

    ref_i(u) = dist_i + |r_i + u L n| - |r_i|

with dist_i the ensemble mean over the probe replicas and r_i the pair vector in
the single reference frame. For a pair whose frame vector points against the pull
axis and whose frame distance exceeds its mean by more than its offset from the
axis, ref_i(u) goes negative. grompp refuses a negative pull-coordN-init, and mdrun
stops at the first step where init + rate*t < 0:

    Pull reference distance for coordinate 780 (-0.000000) needs to be non-negative

That removed 1IBR (coordinate 780, stage G), 2B42 (666, G) and 1BVN (357, H) from
the second HADDOCK pass in all 20 cycles, and 1AKJ (164, G) from the first.

WHAT THIS DOES. Loads pair_displacement and build_coords (and interface_chord,
elastic_network_coords and PULL_REF_FLOOR_NM when present) from the source of
utils/make_boresch.py, since the script parses argv at import. Feeds them a
synthetic interface: the four pairs reconstructed from grompp's own pull tables in
the benchmark logs, plus 400 ordinary pairs. Runs every leg that carries the
UNBIND pull block (every ramp stage and hold; the far-end hold drops its interface
coordinates and is skipped) and evaluates each interface reference the way mdrun
does: init written '%.8f', rate '%.10g', both read back in single precision,
t = step * dt.

Fails on 431510c (28 leg/pair combinations below zero, 20 negative inits, lowest
-0.064 nm, no floor defined) and passes with the floor of PULL_REF_FLOOR_NM.

Standalone, no pytest: python3 tests/test_pull_ref_floor.py
"""
import math, os, re, sys
from types import SimpleNamespace
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))
import fe_protocol as P

DT = 0.004
failures = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def grab(src, name):
    """A top-level def, up to the next top-level statement."""
    i = src.find("\ndef %s(" % name)
    if i < 0:
        return None
    out = []
    for k, ln in enumerate(src[i + 1:].split("\n")):
        if k and ln and not ln[0].isspace() and not ln.startswith("#"):
            break
        out.append(ln)
    return "\n".join(out)


src = open(os.path.join(ROOT, "utils", "make_boresch.py")).read()

# ---------------------------------------------------------------- the interface
nhat = np.array([0.37, -0.51, 0.77]); nhat /= np.linalg.norm(nhat)
perp = np.cross(nhat, [1.0, 0.0, 0.0]); perp /= np.linalg.norm(perp)

# Reconstructed from grompp's "reference at t=0" column at every stage boundary the
# runs reached (bm_fe_amber_2/_ana/pref_fitp.py; residuals <= 0.00044 nm). dist =
# ensemble mean, R = frame distance, cos = cos(frame vector, pull axis). The
# observed series are the numbers grompp printed, to its three decimals.
RECON = [
    # name,        dist,     R,        cos,       failing stage, observed u..., observed ref...
    ("1IBR c780", 0.369129, 0.553371, -0.980756, "G",
     [0.0, 0.12, 0.166, 0.2, 0.25, 0.31, 0.379],
     [0.369, 0.252, 0.208, 0.175, 0.128, 0.072, 0.012]),
    ("2B42 c666", 0.377965, 0.460154, -0.987769, "G",
     [0.0, 0.12, 0.166, 0.2, 0.25, 0.31, 0.379],
     [0.378, 0.260, 0.215, 0.182, 0.135, 0.079, 0.022]),
    ("1BVN c357", 0.526067, 0.668610, -0.989724, "H",
     [0.0, 0.12, 0.166, 0.2, 0.25, 0.31, 0.379, 0.49],
     [0.526, 0.408, 0.362, 0.329, 0.280, 0.222, 0.156, 0.054]),
    ("1AKJ c164", 0.409234, 0.529829, -0.989235, "G",
     [0.0, 0.12, 0.166, 0.2, 0.25, 0.31, 0.379],
     [0.409, 0.291, 0.246, 0.213, 0.164, 0.107, 0.044]),
]
# What the logs say about WHEN each one died (pref_timing.py), for the printout.
EVIDENCE = {"1IBR c780": "no checkpoint in 20 cycles: t* < ~1116 ps",
            "2B42 c666": "checkpoints bound t* to 2270-2854 ps",
            "1BVN c357": "checkpoints bound t* to 771-956 ps",
            "1AKJ c164": "pass 1, every cycle"}

vecs, dists, names = [], [], []
for name, dist, R, c, *_ in RECON:
    vecs.append(R * (c * nhat + math.sqrt(1 - c * c) * perp)); dists.append(dist); names.append(name)

BOUND = [round(b, 6) for b in P.boundaries()]


def composed_min(v, dist):
    """Lowest dist + |v + u n| - |v| over the stage boundaries, computed here and not
    by the code under test."""
    return min(dist + np.linalg.norm(v + u * P.PULL_DIST * nhat) - np.linalg.norm(v)
               for u in BOUND)


rng = np.random.default_rng(20261003)
while len(vecs) < len(RECON) + 400:
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    R = rng.uniform(0.3, 1.2)
    dist = float(np.clip(R - rng.normal(0.15, 0.15), 0.25, 0.6))
    v = R * d
    if composed_min(v, dist) >= 0.0115:      # ordinary: never near the floor
        vecs.append(v); dists.append(dist); names.append("ordinary")
vecs = np.array(vecs)
NPAIR = len(vecs)
NREC = len(RECON)

prot1 = [("ALA1", 1, "CB", 1000 + k, 0.0, 0.0, 0.0) for k in range(NPAIR)]
prot2 = [("GLY2", 2, "CA", 5000 + k, 0.0, 0.0, 0.0) for k in range(NPAIR)]
interdis = [(k, k, dists[k]) for k in range(NPAIR)]

ns = dict(np=np, sys=sys, P=P,
          args=SimpleNamespace(pull_dist=P.PULL_DIST, pull_rate=0.00005),
          PULL_AXIS=nhat, _pair_vec=vecs, interdis=interdis,
          prot1_data=prot1, prot2_data=prot2, k_inter=12500.0 / NPAIR,
          boresch_group_ndx=lambda r: "bor_%s" % r, ref_r=0.8, K_R=4184.0,
          K_ANG_RAD=334.72, en1dis=[], en2dis=[], protkeep1=[], protkeep2=[],
          enk=250.0, ref_thA=90.0, ref_thB=90.0, ref_phA=-120.0, ref_phB=60.0,
          ref_phC=170.0)
m = re.search(r"^PULL_REF_FLOOR_NM\s*=\s*([0-9.eE+-]+)", src, re.M)
FLOOR = float(m.group(1)) if m else None
if FLOOR is not None:
    ns["PULL_REF_FLOOR_NM"] = FLOOR
for fn in ("pair_displacement", "interface_chord", "elastic_network_coords",
           "build_coords"):
    code = grab(src, fn)
    if code is None:
        if fn in ("interface_chord", "elastic_network_coords"):
            continue          # optional helpers: absent before the floor / v1.5
        sys.exit("make_boresch.py has no %s()" % fn)
    exec(compile(code, fn, "exec"), ns)


def unbind_block(leg):
    """Does this leg carry the unbind pull block? fe_protocol says so through the
    'block' key since v1.5. Before that every stage and hold got it, including the
    two holds inside the bound lambda schedule."""
    if "block" in leg:
        return leg["block"] == "unbind"
    return leg["kind"] in ("stage", "hold")


# ---------------------------------------------------------------- run every leg
print("\nsource: utils/make_boresch.py as it is")
legs = []
for l in P.legs():
    if not unbind_block(l) or (l["u_from"] >= P.PULL_DIST and l["kind"] == "hold"):
        continue          # the far-end hold drops its interface coordinates
    nsteps = int(round(l["ps"] / DT))
    ps = nsteps * DT      # what leg_ps() reads back from the mdp
    dirn = "fwd" if l["u_to"] >= l["u_from"] else "rev"
    _pg, co = ns["build_coords"]("unbind", dirn, u_from=l["u_from"], u_to=l["u_to"], ps=ps)
    iface = co[:NPAIR]
    init_s = ["%.8f" % c["init"] for c in iface]
    rate_s = ["%.10g" % c["rate"] for c in iface]
    i32 = np.array([float(np.float32(float(s))) for s in init_s])
    r32 = np.array([float(np.float32(float(s))) for s in rate_s])
    legs.append(dict(leg=l, nsteps=nsteps, ps=ps, co=iface, init_s=init_s, rate_s=rate_s,
                     v0=i32, vend=i32 + r32 * (nsteps * DT), i32=i32, r32=r32))


def first_negative(L, k):
    """First step at which coordinate k's reference is below zero in leg L, or None."""
    i, r = L["i32"][k], L["r32"][k]
    if i < 0:
        return 0
    if r >= 0 or L["vend"][k] >= 0:
        return None
    s = int(i / (-r * DT))
    for q in range(max(0, s - 3), min(L["nsteps"], s + 3) + 1):
        if i + r * (q * DT) < 0:
            return q
    return None


print("\n[1] the reconstruction reproduces what grompp printed at every boundary the runs reached")
for k, (name, dist, R, c, st, us, obs) in enumerate(RECON):
    got = {}
    for L in legs:
        if L["leg"]["kind"] == "stage" and L["leg"]["dirn"] == "fwd":
            got[round(L["leg"]["u_from"], 6)] = float(L["init_s"][k])
    err = max(abs(got[round(u, 6)] - o) for u, o in zip(us, obs))
    check("%s: max |init - grompp| = %.4f nm over %d boundaries" % (name, err, len(us)),
          err <= 0.001, "%.4f" % err)

print("\n[2] where the code under test sends each reconstructed reference below zero")
for k, (name, *_rest) in enumerate(RECON):
    st = RECON[k][4]
    hit = None
    for L in legs:                                   # run order
        q = first_negative(L, k)
        if q is not None:
            hit = (L, q); break
    if hit:
        L, q = hit
        kind = "grompp refuses the leg" if q == 0 else \
            "mdrun dies at step %d of %d, t = %.0f ps (%.0f%%)" % (q, L["nsteps"], q * DT,
                                                                 100.0 * q / L["nsteps"])
        print("      %s: first in %s: %s; logs: stage %s, %s"
              % (name, L["leg"]["name"], kind, st, EVIDENCE[name]))
        same = L["leg"]["name"] == "bindfwd" + st
        print("      %s  the failing leg is the one the logs show" % ("  " if same else "!!"))
    else:
        print("      %s: never below zero" % name)

print("\n[3] no interface reference goes below zero at any step of any leg, either direction")
bad = [(L["leg"]["name"], names[k]) for L in legs for k in range(NPAIR)
       if L["v0"][k] < 0 or L["vend"][k] < 0]
check("references >= 0 at every step (%d legs x %d pairs)" % (len(legs), NPAIR), not bad,
      "%d leg/pair combinations, e.g. %s" % (len(bad), bad[:4]))

print("\n[4] grompp: no leg starts with a negative pull-coordN-init")
neg = [(L["leg"]["name"], names[k]) for L in legs for k in range(NPAIR) if float(L["init_s"][k]) < 0]
check("every written init >= 0", not neg, "%d, e.g. %s" % (len(neg), neg[:4]))

print("\n[5] headroom: rounding cannot take a reference across zero at a leg's last step")
lo = min(min(L["v0"].min(), L["vend"].min()) for L in legs)
check("lowest reference at any leg start or end >= 1e-6 nm (got %.3e)" % lo, lo >= 1e-6)

print("\n[6] continuity: each leg starts where the previous one ended")
# The legs above are the ramp in run order. A protocol without the 'block' key
# also lists the two bound-family holds there; they sit at u = 0 with init = dist,
# as do holdfwd0 and the end of bindrevA, so they cannot open a jump either way.
ramp = legs
worst = 0.0
for a, b in zip(ramp, ramp[1:]):
    worst = max(worst, float(np.abs(a["vend"] - b["v0"]).max()))
check("max jump between consecutive legs %.2e nm <= 2e-7" % worst, worst <= 2e-7)

print("\n[7] Crooks: every reverse stage retraces its forward stage exactly")
fw = {L["leg"]["stage"]: L for L in legs if L["leg"]["kind"] == "stage" and L["leg"]["dirn"] == "fwd"}
rv = {L["leg"]["stage"]: L for L in legs if L["leg"]["kind"] == "stage" and L["leg"]["dirn"] == "rev"}
d_init = max(float(np.abs(rv[s]["v0"] - fw[s]["vend"]).max()) for s in fw)
d_rate = max(abs(float(rv[s]["co"][k]["rate"]) + float(fw[s]["co"][k]["rate"]))
             for s in fw for k in range(NPAIR))
check("reverse init = forward end (max %.2e nm)" % d_init, d_init <= 2e-7)
check("reverse rate = -forward rate (max %.2e nm/ps)" % d_rate, d_rate <= 1e-15)

print("\n[8] pull-work bookkeeping: weight * nominal / ps is the rate written")
dw = 0.0
for L in legs:
    if L["leg"]["kind"] != "stage":
        continue
    nominal = (L["leg"]["u_to"] - L["leg"]["u_from"]) * P.PULL_DIST
    for c in L["co"]:
        dw = max(dw, abs(c["weight"] * nominal / L["ps"] - c["rate"]) / max(abs(c["rate"]), 1e-30))
check("relative mismatch %.2e <= 1e-12" % dw, dw <= 1e-12)

print("\n[9] ordinary pairs are written exactly as before the fix")
diffs = 0
for L in legs:
    u0, u1, ps = L["leg"]["u_from"], L["leg"]["u_to"], L["ps"]
    for k in range(NREC, NPAIR):
        v = vecs[k]
        d0 = np.linalg.norm(v + u0 * P.PULL_DIST * nhat) - np.linalg.norm(v)
        d1 = np.linalg.norm(v + u1 * P.PULL_DIST * nhat) - np.linalg.norm(v)
        want_i = "%.8f" % (dists[k] + float(d0))
        want_r = "%.10g" % ((float(d1 - d0) / ps) if u1 != u0 else 0.0)
        diffs += (L["init_s"][k] != want_i) + (L["rate_s"][k] != want_r)
check("%d ordinary pairs x %d legs: init and rate strings identical (%d differ)"
      % (NPAIR - NREC, len(legs), diffs), diffs == 0)

print("\n[10] the bound end of the ramp is untouched: stage A starts every spring at dist")
a = fw["A"]
nd = sum(a["init_s"][k] != "%.8f" % dists[k] for k in range(NPAIR))
check("stage A init written as '%%.8f' %% dist for all %d pairs (%d differ)" % (NPAIR, nd),
      nd == 0)

print("\n[11] the floor itself")
check("PULL_REF_FLOOR_NM is defined", FLOOR is not None, "absent")
if FLOOR is not None:
    check("1e-6 nm < floor (%.4g) < 0.0115 nm, the lowest reference a completed "
          "stage ran with" % FLOOR, 1e-6 < FLOOR < 0.0115)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all pull-reference checks passed")
