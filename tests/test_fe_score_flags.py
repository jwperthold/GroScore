#!/usr/bin/env python3
"""The scorer's per-structure verdicts: LOW_RECOVERY beside HIGH_RMSD, every
structure without a score named with its reason, mixed row layouts reported
instead of crashing, rows of another protocol never scored, and scores_fe.gs
keeping exactly its columns.

Each property is pinned because its failure is silent or stops everything:

  [1] LOW_RECOVERY (decision of 2026-10-04): the median contact recovery from
      results_qc.d over the cycles a structure is scored on, flagged strictly
      below 0.85. Fixed medians 0.849, 0.850 and 0.851 on both sides of the cut,
      chosen so that a mean or a minimum would flag all three. A QC line for a
      cycle with no result row (c99 at 0.0) and a superseded line of cycle 1
      (0.10, then 0.90) must not count: either would move RECX's median from 0.87
      to 0.84. The flag goes into Note only: a twin with the same works and no QC
      has identical numbers in every numeric column.

  [2] NO_SCORE (review QC-5): a structure with complete rows whose dG_release is
      nan used to appear neither under 'Scored' nor under 'Not scored', with '-'
      as its Note.

  [3] MIXED_LAYOUT (review QC-6): cycles of one structure that straddle a
      reshaping of the ramp (43- and 31-field rows) or of the bound leg (43- and
      9-field rows) stopped the whole scoring pass with an IndexError, before
      scores_fe.gs was written. They now give a not-scored row naming the layouts,
      and every other structure is still scored.

  [4] PROTOCOL IDENTITY (contract F5, review CYC-4): where run_config.gs records a
      protocol_id, a structure whose directory carries a protocol_id.gs of another
      id, or setup.done without any protocol_id.gs (set up before v1.5), is not
      scored (PROTOCOL_MISMATCH). The comparison is with the RECORDED id, not with
      the code's. Without a recorded id everything is scored, under a WARNING.

  [5] THE COLUMNS. None of the above adds a column: the header is the fixed list
      below, with one pair per bound sub-leg and per ramp stage of the fixture's
      own layout, and every row, scored or not, is as wide as the header.

  [6] FEWER THAN 3 CYCLES (review E1): the overlap columns are nan, not the 0.00
      of a structure that lost BAR, and a report in which no leg could be tested
      does not call the CGI crossings measured.

Run:  python3 tests/test_fe_score_flags.py
"""
import os, shutil, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(REPO, "groscore_fe.py")
sys.path.insert(0, os.path.join(REPO, "utils"))
import fe_protocol as P

failures = []


def check(name, ok, detail=""):
    print("  %-66s %s%s" % (name, "PASS" if ok else "FAIL",
                             "" if ok or not detail else "  " + str(detail)))
    if not ok:
        failures.append(name)


NAMES = [w[0] for w in P.works()]
rng = np.random.default_rng(20261004)
OTHER = "fe-0.0-deadbeef"


def row_current(sid, c):
    """A row in the protocol's own layout: pull 0, the dhdl channel carries the work."""
    v = {}
    for L in P.stage_letters():
        v["Wu%s_pull" % L], v["Wu%s_dhdl" % L] = 0.0, 5.0 + rng.normal()
        v["Wr%s_pull" % L], v["Wr%s_dhdl" % L] = 0.0, -4.0 + rng.normal()
    for b in P.BOUND:
        v["W_intro%s" % b[0]], v["W_remove%s" % b[0]] = 10 + rng.normal(), -9 + rng.normal()
    return "%s %d %s 1.50" % (sid, c, " ".join("%.6f" % v[k] for k in NAMES))


def row_31(sid, c):
    """The six-stage ramp of 2026-08-17: 2 bound + 6 stages, 31 fields."""
    w = [10 + rng.normal(), 10 + rng.normal()]
    w += [x for _ in range(6) for x in (0.0, 5.0 + rng.normal())]
    w += [x for _ in range(6) for x in (0.0, -4.0 + rng.normal())]
    w += [-9 + rng.normal(), -9 + rng.normal()]
    return "%s %d %s 1.50" % (sid, c, " ".join("%.6f" % x for x in w))


def row_9(sid, c):
    """The unstaged protocol of before 2026-08-14: 1 bound + 1 ramp, 9 fields."""
    w = [10 + rng.normal(), 0.0, 40 + rng.normal(), 0.0, -30 + rng.normal(),
         -9 + rng.normal()]
    return "%s %d %s 1.50" % (sid, c, " ".join("%.6f" % x for x in w))


def build(run, plan, release=None, qc=None, dirs=None, cfg="numruns 6\n"):
    """plan: {sid: [row maker per cycle]}; qc: {sid: [(cycle, recovered), ...]} in
    file order; dirs: {sid: {filename: text}}."""
    os.makedirs(os.path.join(run, "results_fe.d"))
    for sid, makers in plan.items():
        for c, mk in enumerate(makers, 1):
            with open(os.path.join(run, "results_fe.d", "%s_c%d.gs" % (sid, c)), "w") as f:
                f.write(mk(sid, c) + "\n")
    with open(os.path.join(run, "sp.gs"), "w") as f:
        f.write("# Structure_ID  Chains_for_Protein_B\n")
        f.write("".join("%s B\n" % s for s in plan))
    with open(os.path.join(run, "results_analytical.gs"), "w") as f:
        f.write("".join("%s %s\n" % (s, (release or {}).get(s, "-30.0")) for s in plan))
    with open(os.path.join(run, "results_0.gs"), "w") as f:
        f.write("# Stage-0 setup status\n" + "".join("%s OK\n" % s for s in plan))
    with open(os.path.join(run, "run_config.gs"), "w") as f:
        f.write(cfg)
    open(os.path.join(run, "struct_map.gs"), "w").write("# scoring only\n")
    if qc:
        os.makedirs(os.path.join(run, "results_qc.d"))
        for sid, lines in qc.items():
            with open(os.path.join(run, "results_qc.d", "%s.gs" % sid), "w") as f:
                for c, rec in lines:
                    f.write("%s %d recovered %.4f formed 0.7000 rms_dev 0.0800 "
                            "sc_rmsd 2.000 npairs 100\n" % (sid, c, rec))
    for sid, files in (dirs or {}).items():
        os.makedirs(os.path.join(run, sid))
        for name, text in files.items():
            with open(os.path.join(run, sid, name), "w") as f:
                f.write(text)


def score(run):
    p = subprocess.run([sys.executable, FE], cwd=run, capture_output=True, text=True,
                       env=dict(os.environ, MPLBACKEND="Agg"), timeout=900)
    table = {}
    hdr = []
    path = os.path.join(run, "scores_fe.gs")
    if os.path.isfile(path):
        L = [l.rstrip("\n") for l in open(path)]
        hdr = [l for l in L if l.startswith("# Structure_ID")][0].lstrip("#").split()
        for l in L:
            if l.strip() and not l.startswith("#"):
                f = l.split("\t")
                table[f[0]] = f
    return p, hdr, table


def col(hdr, row, name):
    return row[hdr.index(name)] if row and name in hdr else None


def not_scored(out):
    """{token: [sids]} from the 'Not scored:' line."""
    line = [l for l in out.splitlines() if l.startswith("Not scored:")]
    got = {}
    if line:
        for part in line[0][len("Not scored:"):].strip().rstrip(".").split(";"):
            t = part.split(None, 1)
            if len(t) == 2:
                got[t[0]] = [s.strip() for s in t[1].split(",")]
    return got


work = tempfile.mkdtemp(prefix="fe_score_flags_")

print("\n[1] LOW_RECOVERY and [2] NO_SCORE, [5] the columns")
cur = lambda n: [row_current] * n
plan = {"GOOD": cur(6), "RECLO": cur(5), "RECLO2": cur(5), "RECAT": cur(5),
        "RECHI": cur(5), "RECX": cur(4), "NANREL": cur(6)}
run_a = os.path.join(work, "a")
# RECLO2 is RECLO's works without QC: same rows, so the numbers must be identical.
os.makedirs(run_a)
build(run_a, plan, release={"NANREL": "nan"},
      qc={"RECLO": [(1, 0.95), (2, 0.70), (3, 0.849), (4, 0.849), (5, 0.80)],
          "RECAT": [(1, 0.60), (2, 0.85), (3, 0.85), (4, 0.85), (5, 0.90)],
          "RECHI": [(1, 0.95), (2, 0.851), (3, 0.851), (4, 0.20), (5, 0.99)],
          "RECX": [(1, 0.10), (1, 0.90), (2, 0.90), (3, 0.84), (4, 0.84), (99, 0.0)]})
for c in range(1, 6):
    shutil.copy(os.path.join(run_a, "results_fe.d", "RECLO_c%d.gs" % c),
                os.path.join(run_a, "results_fe.d", "RECLO2_c%d.gs" % c))
    with open(os.path.join(run_a, "results_fe.d", "RECLO2_c%d.gs" % c)) as f:
        t = f.read().replace("RECLO ", "RECLO2 ", 1)
    with open(os.path.join(run_a, "results_fe.d", "RECLO2_c%d.gs" % c), "w") as f:
        f.write(t)
p, hdr, tab = score(run_a)
out = p.stdout
check("the scoring pass runs (rc 0) and writes scores_fe.gs", p.returncode == 0 and tab,
      p.stderr[-400:])
notes = {s: (r[-1] if r else None) for s, r in tab.items()}
check("RECLO (median 0.849) carries LOW_RECOVERY in Note",
      "LOW_RECOVERY" in (notes.get("RECLO") or "").split(","), notes.get("RECLO"))
check("RECAT (median exactly 0.850) does not",
      "LOW_RECOVERY" not in (notes.get("RECAT") or ""), notes.get("RECAT"))
check("RECHI (median 0.851, mean 0.77, min 0.20) does not",
      "LOW_RECOVERY" not in (notes.get("RECHI") or ""), notes.get("RECHI"))
check("RECX: the c99 line without a row and c1's superseded line do not count",
      "LOW_RECOVERY" not in (notes.get("RECX") or ""), notes.get("RECX"))
check("the report lists it beside the RMSD check, with its median",
      any(l.split()[:4] == ["RECLO", "median", "0.849", "over"] for l in out.splitlines()),
      [l for l in out.splitlines() if "RECLO" in l])
check("and counts it: 'LOW_RECOVERY in 1 of 4 structures'",
      "LOW_RECOVERY in 1 of 4 structures" in out,
      [l for l in out.splitlines() if "LOW_RECOVERY in" in l])
rl, rl2 = tab.get("RECLO"), tab.get("RECLO2")
numeric = [i for i, c in enumerate(hdr) if c not in ("Structure_ID", "Ncycles", "Note")]
check("the flag changes no number: RECLO equals its QC-free twin RECLO2",
      bool(rl and rl2) and all(rl[i] == rl2[i] for i in numeric),
      [(hdr[i], rl[i], rl2[i]) for i in numeric if rl and rl2 and rl[i] != rl2[i]][:3])
ns = not_scored(out)
check("NANREL is named: 'Not scored: NO_SCORE NANREL'",
      ns.get("NO_SCORE") == ["NANREL"], ns)
check("with its reason on a line of its own",
      "  NO_SCORE NANREL: dG_release is nan" in out.splitlines(),
      [l for l in out.splitlines() if "NANREL" in l][:3])
check("and NO_SCORE in its Note, not '-'",
      "NO_SCORE" in (notes.get("NANREL") or "").split(","), notes.get("NANREL"))
check("'Scored 6/7': NANREL is the one left out",
      "Scored 6/7 structures" in out, [l for l in out.splitlines() if l.startswith("Scored")])

# [5] The fixed header. The fixed parts are written out; the per-sub-leg and
# per-stage pairs follow the fixture's layout, which is the protocol's.
B = [b[0] for b in P.BOUND]
S = P.stage_letters()
want = (["Structure_ID",
         "dGbind_bar", "dGbind_bar_CI", "pKD_bar", "pKD_bar_CI",
         "dGbind_avg", "dGbind_avg_CI", "pKD_avg", "pKD_avg_CI",
         "dGbind_cgi", "dGbind_cgi_CI", "pKD_cgi", "pKD_cgi_CI",
         "dG_intro_bar", "dG_intro_bar_CI", "dG_intro_avg", "dG_intro_avg_CI",
         "dG_intro_cgi", "dG_intro_cgi_CI",
         "dG_unbind_bar", "dG_unbind_bar_CI", "dG_unbind_avg", "dG_unbind_avg_CI",
         "dG_unbind_cgi", "dG_unbind_cgi_CI"]
        + [x for n in B for x in ("dG_intro%s_bar" % n, "dG_intro%s_bar_CI" % n)]
        + [x for s in S for x in ("dG_unb%s_bar" % s, "dG_unb%s_bar_CI" % s)]
        + ["dG_intro_1s_bar", "dG_intro_1s_bar_CI", "dG_unbind_1s_bar",
           "dG_unbind_1s_bar_CI", "dG_release", "Overlap_mean_pct", "Overlap_min_pct",
           "RMSD_mean_A", "RMSD_max_A", "Ncycles", "Note"])
check("the header is exactly the v1.5 column list (%d columns)" % len(want),
      hdr == want, [c for c in hdr if c not in want] + ["missing:"]
      + [c for c in want if c not in hdr])
check("every row, scored or not, is as wide as the header",
      bool(tab) and all(len(r) == len(hdr) for r in tab.values()),
      {s: len(r) for s, r in tab.items() if len(r) != len(hdr)})

print("\n[3] MIXED_LAYOUT instead of an IndexError")
run_b = os.path.join(work, "b")
os.makedirs(run_b)
build(run_b, {"GOOD": cur(6),
              "MIXD": [row_current] * 3 + [row_31] * 3,
              "MIXB": [row_current] * 3 + [row_9] * 3})
p, hdr_b, tab = score(run_b)
out = p.stdout
check("the pass completes (rc 0) and writes scores_fe.gs",
      p.returncode == 0 and bool(tab), p.stderr[-400:])
check("no traceback", "Traceback" not in p.stderr, p.stderr[-400:])
ns = not_scored(out)
check("MIXD and MIXB are named MIXED_LAYOUT",
      sorted(ns.get("MIXED_LAYOUT", [])) == ["MIXB", "MIXD"], ns)
check("MIXD's reason names both ramp layouts and their cycles",
      "  MIXED_LAYOUT MIXD: 2 bound + 6 stages in cycles 4-6; 2 bound + %d stages in "
      "cycles 1-3" % len(S) in out.splitlines(),
      [l for l in out.splitlines() if "MIXD" in l][:3])
check("MIXB's reason names the bound-leg change too",
      "  MIXED_LAYOUT MIXB: 1 bound + 1 stages in cycles 4-6; 2 bound + %d stages in "
      "cycles 1-3" % len(S) in out.splitlines(),
      [l for l in out.splitlines() if "MIXB" in l][:3])
for s in ("MIXD", "MIXB"):
    r = tab.get(s)
    check("%s: Note MIXED_LAYOUT, Ncycles 6, every number nan" % s,
          bool(r) and r[-1] == "MIXED_LAYOUT" and r[-2] == "6"
          and all(x == "nan" for x in r[1:-2]), r and r[-3:])
check("GOOD is still scored", col(hdr_b, tab.get("GOOD"), "dGbind_avg") not in (None, "nan"),
      tab.get("GOOD", [None])[:3])
check("the mixed structures stay out of the dissipation and Gaussian tables",
      not any(l.split()[:1] in (["MIXD"], ["MIXB"]) for l in out.splitlines()),
      [l for l in out.splitlines() if l.split()[:1] in (["MIXD"], ["MIXB"])][:2])

print("\n[4] rows of another protocol are not scored")
pplan = {"PMATCH": cur(5), "POTHER": cur(5), "POLD": cur(5), "PARCH": cur(5)}
pdirs = {"PMATCH": {"setup.done": "", "protocol_id.gs": P.PROTOCOL_ID + "\n"},
         "POTHER": {"setup.done": "", "protocol_id.gs": OTHER + "\n"},
         "POLD": {"setup.done": ""}}
for label, cfg, expect_unscored, warn in (
        ("recorded = this code", "numruns 5\nprotocol_id\t%s\n" % P.PROTOCOL_ID,
         ["POLD", "POTHER"], None),
        ("nothing recorded", "numruns 5\n", [], "records no protocol_id"),
        ("recorded = %s" % OTHER, "numruns 5\nprotocol_id\t%s\n" % OTHER,
         ["PMATCH", "POLD"], "holds FE protocol %s" % OTHER)):
    run = os.path.join(work, "p%d" % len(os.listdir(work)))
    os.makedirs(run)
    build(run, pplan, dirs=pdirs, cfg=cfg)
    p, hdr_p, tab = score(run)
    out = p.stdout
    ns = not_scored(out)
    check("%s: rc 0" % label, p.returncode == 0, p.stderr[-300:])
    check("%s: not scored as PROTOCOL_MISMATCH: %s" % (label, expect_unscored or "none"),
          sorted(ns.get("PROTOCOL_MISMATCH", [])) == expect_unscored, ns)
    scored = [s for s in pplan if s not in expect_unscored]
    check("%s: scored: %s" % (label, ", ".join(scored)),
          all(col(hdr_p, tab.get(s), "dGbind_avg") not in (None, "nan") for s in scored),
          {s: col(hdr_p, tab.get(s), "dGbind_avg") for s in scored})
    for s in expect_unscored:
        r = tab.get(s)
        check("%s: %s has Note PROTOCOL_MISMATCH and 5 rows counted" % (label, s),
              bool(r) and r[-1] == "PROTOCOL_MISMATCH" and r[-2] == "5", r and r[-2:])
    if warn:
        check("%s: the report opens with a WARNING (%s)" % (label, warn),
              any(l.startswith("WARNING") and warn in l for l in out.splitlines()[:30]),
              [l for l in out.splitlines() if l.startswith("WARNING")])
    else:
        check("%s: no protocol WARNING" % label,
              not any("protocol" in l and l.startswith("WARNING") for l in out.splitlines()))
    if "POLD" in expect_unscored:
        check("%s: POLD's reason says it predates v1.5" % label,
              any(l.startswith("  PROTOCOL_MISMATCH POLD:") and "before v1.5" in l
                  for l in out.splitlines()),
              [l for l in out.splitlines() if "POLD" in l][:2])

print("\n[6] too few cycles to have an overlap: nan, and no 'measured' verdict")
# Below 3 cycles a channel has no min..max range, so its overlap count is 0 whatever
# the works do. Such a row read Overlap 0.00 (the value of a real loss of BAR) while
# the report called its untested crossings measured (review E1).
run_e = os.path.join(work, "e")
os.makedirs(run_e)
build(run_e, {"ONE": cur(1), "TWO": cur(2)}, cfg="numruns 2\n")
p, hdr_e, tab = score(run_e)
out = p.stdout
for s in ("ONE", "TWO"):
    check("%s: Overlap_mean_pct and Overlap_min_pct are nan, not 0.00" % s,
          col(hdr_e, tab.get(s), "Overlap_mean_pct") == "nan"
          and col(hdr_e, tab.get(s), "Overlap_min_pct") == "nan",
          (col(hdr_e, tab.get(s), "Overlap_mean_pct"), col(hdr_e, tab.get(s), "Overlap_min_pct")))
check("no leg tested: the report does not call the crossings measured",
      "measured rather than extrapolated" not in out
      and "No leg had the cycles for any check" in out,
      [l for l in out.splitlines() if "No leg" in l])
run_f = os.path.join(work, "f")
os.makedirs(run_f)
build(run_f, {"SIX": cur(6)})
p, hdr_f, tab = score(run_f)
mean6 = col(hdr_f, tab.get("SIX"), "Overlap_mean_pct")
check("6 cycles: the overlap columns are numbers (%s)" % mean6,
      mean6 not in (None, "nan") and col(hdr_f, tab.get("SIX"), "Overlap_min_pct") not in (None, "nan"),
      tab.get("SIX", [])[-6:])

shutil.rmtree(work, ignore_errors=True)
print("")
if failures:
    print("FAILED (%d): %s" % (len(failures), "; ".join(failures)))
    sys.exit(1)
print("All scorer flag checks passed.")
