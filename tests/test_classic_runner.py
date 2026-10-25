#!/usr/bin/env python3
"""job.run, the classic runner, driven end to end with a fake gmx.

The whole script runs, setup and cycles, in a scratch run directory. gmx is a fake
with GROMACS's file semantics (grompp refuses a missing -c or -t; mdrun needs its
tpr, and a leg that dies leaves its force file up to the crash, maybe a checkpoint,
and no .gro), every utility it calls is a stub that logs its arguments, except
integrate.py, which is the real one behind a logging wrapper. A full pull force
file integrates to -100.0 and a full push to -60.0 (4 interface columns, 2
elastic-network columns, 5000 ps every 2 ps). The findings behind each section are
in v15_work/digest_audit (runner and scorer lenses) and digest_review (critic).

  [1] a fresh run: OK only once the restraint set exists (SETUP_RUNNING before);
      one row per cycle in results.d with '>', 'ID N pull push RMSD'; a .done per
      cycle; the restraint frame assembled with assemble_frame.py after trjconv
      -pbc whole, never -pbc cluster (R1); integrate.py told the leg length and the
      coordinate count; archived without job_run.lock. Under AMBER19SB with
      ligands on, fuse_glycans.py runs first and renumber_pdb.py reads fused.pdb
      (decision of 2026-10-04; tests/test_glycan_runner.py has the rest).
  [2] a pull that dies twice with a checkpoint (R2, SCORER-1): retried from step
      0, never from the checkpoint the dead run left (an unfinished run is never
      continued), never integrated, nothing grompped from it, the cycle's row NaN,
      no .done, no archive, exit 1. The next run runs it again from step 0 and
      finishes the cycle without touching the finished one (no gmx call, no
      integration).
  [3] a push that dies, a push whose force file stops at 3000 of 5000 ps although
      the leg wrote its .gro, and a leg rescued by the in-job retry.
  [4] a legacy splice: nptrev and bindrev left by v1.4, built from the dead pull's
      checkpoint, are older than the pull's finished .gro and are redone.
  [5] a re-walk with a changed restraint count (re-walk extra finding): with
      GROSCORE_NUMCYCLES=3 and make_disres_en now giving 9, the restraint set is
      not regenerated, cycles 1-2 are neither re-run nor re-integrated and their
      rows stay byte-identical, cycle 3 is added (R6: -n <more> adds cycles).
  [6] a re-setup in an existing run directory (R3): the old setup's rows go at the
      start, the new cycle 1 that fails is NaN, cycle 2 carries the new RMSD, and
      groscore.py scores one cycle.
  [7] ligands (decision of 2026-10-04, F4): merge_ligand.py gets --ligand-param
      on/off-ff/off-user; a ligand_failed.gs line stops the setup as LIGAND_FAIL
      before any minimisation; a parametrisation that fails without recording a
      reason is recorded by job.run and stops it too. Under off-ff and off-user no
      glycan is fused: renumber_pdb.py reads input.pdb.
  [8] stage-0 statuses (R5, SCORER-6, F2): 0 interface restraints, make_disres_en
      exit 1, assemble_frame exit 1 and an empty broken-loop verdict each stop with
      their status and no OK; no cycle starts. A frozen restraint set whose count
      reads 0 stops as DISRES_FAIL before any leg.
  [9] the lock (R8, --run-local extra finding): a live owner stops the job (exit
      1, no gmx call), a dead one is reclaimed, and a fresh directory holding the
      files the submitter writes (run.gs, job.run, job_local.out) runs. A SIGTERM to
      the script stops the running leg at once and ends the job (exit 143, lock
      released, nothing started after it) instead of releasing the lock and
      carrying on, or waiting for the leg to end (review S3). A job.run killed hard
      leaves its leg running, and that leg keeps the lock until it has ended.
 [10] the archive (R4): a tar that fails keeps the directory and leaves no tarball.
 [11] job.run's leg() on its own: a source leg with a checkpoint but no .gro starts
      nothing (no grompp from a dead leg's checkpoint, whoever calls leg()).
 [12] a directory copied without its file times (every tpr older than its source)
      keeps its finished legs (review S6).
 [13] the FeS force-field branch of both runners fires on the FE2 and SD residues
      of a cluster, not on the SD atom of a methionine (review E3).

Standalone, no pytest: python3 tests/test_classic_runner.py
"""
import gzip, os, re, shutil, signal, socket, subprocess, sys, tarfile, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []


def check(name, ok, detail=""):
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:300]))
    if not ok:
        failures.append(name)


FAKE_GMX = r'''#!/bin/bash
echo "$*" >> "$GMXLOG"
sub=$1; shift
args=("$@")
get() { local i; for ((i=0; i<${#args[@]}; i++)); do [[ "${args[$i]}" == "$1" ]] && { echo "${args[$((i+1))]}"; return; }; done; }
case "$sub" in
  grompp)
    c=$(get -c); t=$(get -t); o=$(get -o)
    [[ -n "$c" && ! -f "$c" ]] && { echo "Fatal error: File '$c' does not exist" >&2; exit 1; }
    [[ -n "$t" && ! -f "$t" ]] && { echo "Fatal error: File '$t' does not exist" >&2; exit 1; }
    touch "${o%.tpr}.tpr" ;;
  mdrun) exec python3 "$FAKE_MDRUN" "$@" ;;
  pdb2gmx) cat > /dev/null; echo "fake $sub" > "$(get -o)"; touch topol.top ;;
  make_ndx) cat > /dev/null; echo "[ System ]" > "$(get -o)" ;;
  editconf|solvate|genion|trjconv) cat > /dev/null; echo "fake $sub" > "$(get -o)" ;;
esac
exit 0
'''

# Scenario lines in <structure>/scenario.txt:
#   NAME die FRACTION CPT COUNT [cut]   the first COUNT attempts die at FRACTION of
#                                       the leg, leaving a checkpoint if CPT is yes
#   NAME short                          the leg finishes but its force file stops at 60%
#   NAME slow SECONDS                   the leg takes SECONDS, then finishes
FAKE_MDRUN = r'''
import os, sys, time
args = sys.argv[1:]
get = lambda f: args[args.index(f) + 1] if f in args else None
name = get("-deffnm")
if not name or not os.path.isfile(name + ".tpr"):
    sys.stderr.write("Fatal error: Can not open file %s.tpr\n" % name)
    sys.exit(1)
mode, frac, cpt = "finish", 1.0, "yes"
if os.path.isfile("scenario.txt"):
    for line in open("scenario.txt"):
        t = line.split()
        if not t or t[0] != name:
            continue
        if t[1] == "short":
            mode = "short"
        elif t[1] == "slow":                   # NAME slow SECONDS: a leg that takes a while
            open(".slow_" + name, "w").close()
            time.sleep(float(t[2]))
        else:
            mark = ".died_" + name
            n = int(open(mark).read()) if os.path.isfile(mark) else 0
            if n < int(t[4]):
                open(mark, "w").write(str(n + 1))
                mode, frac, cpt = ("die-cut" if t[5:] == ["cut"] else "die"), float(t[2]), t[3]
        break
if name.startswith(("bindfwd_", "bindrev_")) and frac > 0:
    npr = int(open("numpertres.gs").read().split()[0])
    nco = npr + 2
    for l in open("bind.mdp"):
        if l.split("=")[0].strip() == "pull-ncoords":
            nco = int(l.split("=")[1].split(";")[0])
    f = (100.0 if name.startswith("bindfwd_") else 60.0) / npr
    tend = 5000.0 * (0.6 if mode == "short" else frac)
    with open(name + ".xvg", "w") as o:
        o.write('# fake pull force\n@    title "Pull force"\n')
        for k in range(nco):
            o.write('@ s%d legend "%d"\n' % (k, k + 1))
        t = 0.0
        while t <= tend + 1e-9:
            o.write("%.4f" % t + ("\t%g" % f) * npr + "\t7" * (nco - npr) + "\n")
            t += 2.0
        if mode == "die-cut":
            o.write("%.4f" % t + ("\t%g" % f) * (npr // 2))
if mode.startswith("die"):
    if cpt == "yes":
        open(name + ".cpt", "w").close()
    sys.exit(1)
open(name + ".gro", "w").close()
open(name + ".cpt", "w").close()
'''

STUB_HEAD = r'''#!/usr/bin/env python3
import os, shutil, sys
with open(os.environ["UTILLOG"], "a") as _f:
    _f.write(os.path.basename(sys.argv[0]) + " " + " ".join(sys.argv[1:]) + "\n")
arg = lambda f: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else None
env = lambda k, d="": os.environ.get(k, d)
'''

STUBS = {
    # The glycan fusion with nothing to fuse: fused.pdb is input.pdb and glycans.json
    # defines no residue (utils/fuse_glycans.py always writes both).
    "fuse_glycans.py": r'''
shutil.copy(arg("-f"), arg("-o"))
open(arg("-d"), "w").write('{"residues": [], "report": {"fused": [], "not_fused": []}}\n')
if arg("--unfused"):
    open(arg("--unfused"), "w").close()
''',
    "renumber_pdb.py": r'''
open(arg("-o"), "w").close()
open("chain_map.gs", "w").write("# Protein B chain(s): B\n2\n")
open("ion_residues.gs", "w").close()
for lig in filter(None, env("FAKE_LIGANDS").split(",")):
    r, c, n = lig.split("_")
    open("ligand_%s_%s.pdb" % (r, c), "w").write(
        "HETATM    1  C1  %-3s %s%4s       0.000   0.000   0.000  1.00  0.00           C\n" % (r, c, n))
''',
    "fix_pdb.py": 'open(arg("-o"), "w").close()\n',
    "cap_termini.py": 'open(arg("-o"), "w").close()\n',
    "fix_ion_protonation.py": "",
    "parametrize_ncaa.py": "",
    "parametrize_ligand.py": 'sys.exit(int(env("FAKE_PARAM_RC", "0")))\n',
    "check_brokenloop.py": 'v = env("FAKE_BL", "0")\nif v != "EMPTY":\n    print(v)\n',
    "check_entangledloops.py": 'print("0")\n',
    "make_cutout.py": 'open("cutout.pdb", "w").close()\n',
    "merge_ligand.py": r'''
lf = env("FAKE_LIGFAIL")
if lf and lf != "keep":
    open("ligand_failed.gs", "w").write(lf + "\n")
elif not lf and os.path.isfile("ligand_failed.gs"):
    os.remove("ligand_failed.gs")
''',
    "merge_crystal_waters.py": "",
    "make_ion_restraints.py": "",
    "retype_ions.py": "",
    "fix_topol_intermolecular.py": "",
    "make_cluster_group.py": "",
    "assemble_frame.py": r'''
rc = int(env("FAKE_AF_RC", "0"))
if rc:
    sys.stderr.write("assemble_frame: ERROR - fake failure\n")
    sys.exit(rc)
shutil.copy(arg("-f"), arg("-o"))
''',
    "make_disres_en.py": r'''
if int(env("FAKE_MDE_RC", "0")):
    sys.exit(int(env("FAKE_MDE_RC")))
npr, nen = int(env("FAKE_NPR", "4")), 2
for m in ("bind.mdp", "nptrev.mdp", "bindrev.mdp"):
    with open(m, "a") as f:
        f.write("pull-ngroups            = %d\npull-ncoords            = %d\n" % (2 * (npr + nen), npr + nen))
print(npr)
''',
    "rebound_rmsd.py": 'print(env("FAKE_RMSD", "1.2500"))\n',
    # The real integrate.py, behind a wrapper that logs the call.
    "integrate.py": 'os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(sys.argv[0]), "integrate_real.py")] + sys.argv[1:])\n',
}


def make_root():
    root = tempfile.mkdtemp(prefix="classic_runner_")
    os.makedirs(os.path.join(root, "utils"))
    os.makedirs(os.path.join(root, "run"))
    os.makedirs(os.path.join(root, "bin"))
    for name, body in STUBS.items():
        with open(os.path.join(root, "utils", name), "w") as f:
            f.write(STUB_HEAD + body)
    shutil.copy(os.path.join(ROOT, "utils", "integrate.py"), os.path.join(root, "utils", "integrate_real.py"))
    os.symlink(os.path.join(ROOT, "settings"), os.path.join(root, "settings"))
    with open(os.path.join(root, "bin", "gmx"), "w") as f:
        f.write(FAKE_GMX)
    os.chmod(os.path.join(root, "bin", "gmx"), 0o755)
    with open(os.path.join(root, "fake_mdrun.py"), "w") as f:
        f.write(FAKE_MDRUN)
    return root


def new_structure(root, runs="B 4 1 amber19sb_opc3 1", sid="S1"):
    d = os.path.join(root, "run", sid)
    os.makedirs(d)
    open(os.path.join(d, "input.pdb"), "w").write("END\n")
    open(os.path.join(d, "run.gs"), "w").write(runs + "\n")
    shutil.copy(os.path.join(ROOT, "job.run"), os.path.join(d, "job.run"))
    os.chmod(os.path.join(d, "job.run"), 0o755)
    return d


def run_job(root, scenario=None, sid="S1", path_prefix=None, backdate=True, **fake):
    """Run job.run in <root>/run/<sid>. backdate: age every file there by 3 h first, as
    a resubmission long after the setup finds them (v1.4's 70-minute guard then lets
    the job through; equal ages are not 'older' to the stale-tpr test)."""
    d = os.path.join(root, "run", sid)
    if scenario is not None:
        open(os.path.join(d, "scenario.txt"), "w").write(scenario)
    if backdate:
        old = time.time() - 3 * 3600
        for dirpath, dirs, files in os.walk(d):
            for name in files + dirs:
                os.utime(os.path.join(dirpath, name), (old, old))
    for log in ("gmx.log", "util.log"):
        open(os.path.join(root, log), "w").close()
    r = subprocess.run(["bash", "job.run"], cwd=d, env=job_env(root, path_prefix, fake),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
    return r


def job_env(root, path_prefix=None, fake=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GROSCORE_", "FAKE_", "SLURM_"))}
    env.update(PATH=(path_prefix + ":" if path_prefix else "") + os.path.join(root, "bin") + ":" + os.environ["PATH"],
               GMXLOG=os.path.join(root, "gmx.log"), UTILLOG=os.path.join(root, "util.log"),
               FAKE_MDRUN=os.path.join(root, "fake_mdrun.py"))
    env.update({k: str(v) for k, v in (fake or {}).items()})
    return env


def log(root, which):
    return [l.rstrip("\n") for l in open(os.path.join(root, which + ".log"))]


def grompps(root):
    out = []
    for l in log(root, "gmx"):
        t = l.split()
        if t and t[0] == "grompp" and "-o" in t:
            out.append(t[t.index("-o") + 1])
    return out


def mdruns(root):
    return [l for l in log(root, "gmx") if l.startswith("mdrun ")]


def integrated(root):
    return [l.split()[l.split().index("-f") + 1] for l in log(root, "util") if l.startswith("integrate.py ")]


def row(root, cycle, sid="S1"):
    p = os.path.join(root, "run", "results.d", "%s_c%d.gs" % (sid, cycle))
    return open(p).read() if os.path.isfile(p) else None


def statuses(root, sid="S1"):
    p = os.path.join(root, "run", "results_0.gs")
    if not os.path.isfile(p):
        return []
    return [l.split()[1] for l in open(p) if l.split()[:1] == [sid] and len(l.split()) == 2]


def works(text):
    """(pull, push, rmsd) of a row as floats, nan for NaN; None if malformed."""
    t = (text or "").split()
    if len(t) != 5:
        return None
    try:
        return tuple(float(x) for x in t[2:])
    except ValueError:
        return None


def near(x, v):
    return x is not None and abs(x - v) < 1e-6


def is_nan(x):
    return x is not None and x != x


def archived(root, sid="S1"):
    return (os.path.isfile(os.path.join(root, "run", sid + ".tar.gz"))
            and not os.path.isdir(os.path.join(root, "run", sid)))


def unarchive(root, sid="S1"):
    run = os.path.join(root, "run")
    with tarfile.open(os.path.join(run, sid + ".tar.gz")) as t:
        t.extractall(run)
    os.remove(os.path.join(run, sid + ".tar.gz"))


def sdir(root, sid="S1"):
    return os.path.join(root, "run", sid)


# ---------------------------------------------------------------------------
print("\n[1] a fresh run of two cycles")
rootA = make_root()
new_structure(rootA)
r = run_job(rootA)
c1, c2 = works(row(rootA, 1)), works(row(rootA, 2))
check("exit 0", r.returncode == 0, (r.returncode, r.stdout[-400:], r.stderr[-300:]))
check("results.d/S1_c1.gs is 'S1 1 -100.0 -60.0 1.25'",
      (row(rootA, 1) or "").split()[:2] == ["S1", "1"] and c1 and near(c1[0], -100) and near(c1[1], -60)
      and near(c1[2], 1.25), row(rootA, 1))
check("  and c2 the same, one line each",
      c2 and near(c2[0], -100) and near(c2[1], -60) and (row(rootA, 2) or "").count("\n") == 1, row(rootA, 2))
check("results_0.gs: SETUP_RUNNING, then OK last", statuses(rootA)[:1] == ["SETUP_RUNNING"]
      and statuses(rootA)[-1:] == ["OK"] and statuses(rootA).count("OK") == 1, statuses(rootA))
setup_gs = os.path.join(rootA, "run", "results.d", "S1_setup.gs")
check("results.d/S1_setup.gs names the structure and its 4 restraints",
      os.path.isfile(setup_gs) and open(setup_gs).read().split() == ["S1", "numpertres", "4"])
ul = log(rootA, "util")
check("merge_ligand.py ran with --ligand-param on",
      any(l.startswith("merge_ligand.py") and "--ligand-param on" in l for l in ul),
      [l for l in ul if l.startswith("merge_ligand")])
fz = [i for i, l in enumerate(ul) if l.startswith("fuse_glycans.py ")]
rn = [i for i, l in enumerate(ul) if l.startswith("renumber_pdb.py ")]
check("the glycans are fused first: fuse_glycans.py, then renumber_pdb.py -f fused.pdb",
      len(fz) == 1 and len(rn) == 1 and fz[0] < rn[0] and ul[rn[0]].startswith("renumber_pdb.py -f fused.pdb"),
      [ul[i] for i in fz + rn])
af = [l for l in ul if l.startswith("assemble_frame.py")]
check("the restraint frame: assemble_frame.py on the -pbc whole frame, Protein_Struct",
      len(af) == 1 and "-f npt_init_whole.gro" in af[0] and "-g Protein_Struct" in af[0], af)
gl = log(rootA, "gmx")
check("  and no trjconv -pbc cluster anywhere", not [l for l in gl if "-pbc cluster" in l],
      [l for l in gl if "trjconv" in l])
check("  make_disres_en.py read the assembled frame",
      any(l.startswith("make_disres_en.py -f npt_init_cluster.gro") for l in ul))
ints = [l for l in ul if l.startswith("integrate.py")]
check("integrate.py: 4 calls, each with -nr 4 --tend 5000 --ncols 6",
      len(ints) == 4 and all("-nr 4" in l and "--tend 5000" in l and "--ncols 6" in l for l in ints), ints)
check("archived: S1.tar.gz, no directory", archived(rootA))
if archived(rootA):
    with tarfile.open(os.path.join(rootA, "run", "S1.tar.gz")) as t:
        names = t.getnames()
    check("  with a .done per cycle and without job_run.lock",
          "S1/results_c1.done" in names and "S1/results_c2.done" in names
          and not [n for n in names if "job_run.lock" in n], [n for n in names if "lock" in n or "done" in n])
else:
    check("  with a .done per cycle and without job_run.lock", False, "not archived")

# ---------------------------------------------------------------------------
print("\n[2] a pull leg that dies twice with a checkpoint, then the restart")
rootB = make_root()
new_structure(rootB)
r = run_job(rootB, scenario="bindfwd_1 die 0.4 yes 2\n")
md = [l for l in mdruns(rootB) if "-deffnm bindfwd_1" in l]
check("two attempts, both from step 0, never from the dead run's checkpoint",
      len(md) == 2 and not [l for l in md if "-cpi" in l], md)
check("nothing is grompped from the dead pull (no nptrev_2)", "nptrev_2" not in grompps(rootB),
      grompps(rootB))
check("its force file is never integrated (v1.4: -40.0)", "bindfwd_1.xvg" not in integrated(rootB),
      integrated(rootB))
w1 = works(row(rootB, 1))
check("cycle 1's row is 'S1 1 NaN NaN NaN'", w1 is not None and all(is_nan(x) for x in w1), row(rootB, 1))
check("  and it has no .done", not os.path.isfile(os.path.join(sdir(rootB), "results_c1.done")))
check("cycle 2 still runs and completes", works(row(rootB, 2)) and near(works(row(rootB, 2))[0], -100),
      row(rootB, 2))
check("no archive, exit 1", not archived(rootB) and r.returncode == 1, (r.returncode, r.stdout[-300:]))
if archived(rootB):
    unarchive(rootB)
open(os.path.join(sdir(rootB), "scenario.txt"), "w").close()
r = run_job(rootB)
md = [l for l in mdruns(rootB) if "-deffnm bindfwd_1" in l]
check("restart: the pull runs again from step 0, not from its checkpoint",
      len(md) == 1 and "-cpi" not in md[0], md)
check("  nptrev_2 and bindrev_2 are built from the finished pull",
      grompps(rootB) == ["nptrev_2", "bindrev_2"], grompps(rootB))
w1 = works(row(rootB, 1))
check("  cycle 1 now -100.0 / -60.0", w1 and near(w1[0], -100) and near(w1[1], -60), row(rootB, 1))
check("  the finished cycle 2 is not touched: no gmx call, no integration",
      not [l for l in log(rootB, "gmx") if "_c2" in l or "bindfwd_3" in l or "bindrev_4" in l]
      and integrated(rootB) == ["bindfwd_1.xvg", "bindrev_2.xvg"], integrated(rootB))
check("  archived, exit 0", archived(rootB) and r.returncode == 0, (r.returncode, r.stdout[-300:]))

# ---------------------------------------------------------------------------
print("\n[3] a dead push, a short force file, a retry that rescues")
rootD = make_root()
new_structure(rootD)
r = run_job(rootD, scenario="bindrev_2 die 0.4 no 2\n")
w1 = works(row(rootD, 1))
check("a dead push: row NaN, its partial force file never integrated (v1.4: -24.0)",
      w1 is not None and all(is_nan(x) for x in w1) and "bindrev_2.xvg" not in integrated(rootD),
      (row(rootD, 1), integrated(rootD)))
check("  exit 1, no archive", r.returncode == 1 and not archived(rootD), r.returncode)
rootE = make_root()
new_structure(rootE)
r = run_job(rootE, scenario="bindrev_2 short\n")
w1 = works(row(rootE, 1))
check("a push that wrote its .gro but stops at 3000 ps: push NaN (v1.4: -36.0)",
      w1 is not None and near(w1[0], -100) and is_nan(w1[1]), row(rootE, 1))
check("  no .done for it, exit 1", not os.path.isfile(os.path.join(sdir(rootE), "results_c1.done"))
      and r.returncode == 1, r.returncode)
rootK = make_root()
new_structure(rootK)
r = run_job(rootK, scenario="bindfwd_1 die 0.4 yes 1 cut\n")
md = [l for l in mdruns(rootK) if "-deffnm bindfwd_1" in l]
w1 = works(row(rootK, 1))
check("a pull that dies once is rescued by the retry, from step 0",
      len(md) == 2 and "-cpi" not in md[1] and w1 and near(w1[0], -100), (md, row(rootK, 1)))
check("  exit 0, archived", r.returncode == 0 and archived(rootK), r.returncode)

# ---------------------------------------------------------------------------
print("\n[4] a v1.4 splice: a push built from the dead pull's checkpoint is redone")
rootM = make_root()
new_structure(rootM)
run_job(rootM, scenario="bindfwd_1 die 0.4 yes 2\n")
os.makedirs(sdir(rootM), exist_ok=True)
open(os.path.join(sdir(rootM), "scenario.txt"), "w").close()
# What v1.4 left after such a pull: nptrev and bindrev built from its checkpoint and
# run to the end, the push's force file integrated and gzipped. The backdating in
# run_job makes them as old as the dead pull, older than the pull's .gro to come.
for f in ("nptrev_2.tpr", "nptrev_2.cpt", "nptrev_2.gro", "bindrev_2.tpr", "bindrev_2.cpt",
          "bindrev_2.gro"):
    open(os.path.join(sdir(rootM), f), "w").close()
with gzip.open(os.path.join(sdir(rootM), "bindrev_2.xvg.gz"), "wt") as f:
    f.write("# the push force file v1.4 integrated and gzipped\n0.0000\t1\t1\t1\t1\t7\t7\n")
r = run_job(rootM)
g = grompps(rootM)
check("nptrev_2 and bindrev_2 are grompped again from the finished pull", g == ["nptrev_2", "bindrev_2"], g)
md = [l for l in mdruns(rootM) if "-deffnm nptrev_2" in l or "-deffnm bindrev_2" in l]
check("  and run from scratch, not from their stale checkpoints",
      len(md) == 2 and not [l for l in md if "-cpi" in l], md)
check("  saying why", "nptrev_2.tpr is older than bindfwd_1.gro" in r.stdout, r.stdout[-500:])
w1 = works(row(rootM, 1))
check("  cycle 1 -100.0 / -60.0", w1 and near(w1[0], -100) and near(w1[1], -60), row(rootM, 1))

# ---------------------------------------------------------------------------
print("\n[5] a re-walk with a changed restraint count, and a third cycle")
before = (row(rootA, 1), row(rootA, 2))
if archived(rootA):
    unarchive(rootA)
if before[0] is not None:
    os.utime(os.path.join(rootA, "run", "results.d", "S1_c1.gs"), (1.0e9, 1.0e9))
r = run_job(rootA, GROSCORE_NUMCYCLES=3, FAKE_NPR=9)
ul = log(rootA, "util")
check("make_disres_en.py is not called again", not [l for l in ul if l.startswith("make_disres_en")],
      [l for l in ul if l.startswith("make_disres_en")])
sg = os.path.join(rootA, "run", "results.d", "S1_setup.gs")
check("the restraint count stays 4 (results.d/S1_setup.gs)",
      os.path.isfile(sg) and open(sg).read().split() == ["S1", "numpertres", "4"])
old_legs = {"%s_c%d" % (s, c) for s in ("nvt_1", "nvt_2", "nvt_3", "nvt_4", "nvt_5") for c in (1, 2)} \
    | {"npt_c1", "npt_c2", "bindfwd_1", "bindfwd_3", "nptrev_2", "nptrev_4", "bindrev_2", "bindrev_4"}
ran = [l.split()[l.split().index("-deffnm") + 1] for l in mdruns(rootA)]
check("cycles 1-2: no leg re-run, no re-integration",
      not (set(ran) & old_legs) and integrated(rootA) == ["bindfwd_5.xvg", "bindrev_6.xvg"],
      (sorted(set(ran) & old_legs), integrated(rootA)))
check("  their rows are byte-identical and were not rewritten",
      before[0] is not None and (row(rootA, 1), row(rootA, 2)) == before
      and os.path.getmtime(os.path.join(rootA, "run", "results.d", "S1_c1.gs")) == 1.0e9)
w3 = works(row(rootA, 3))
check("GROSCORE_NUMCYCLES=3 adds cycle 3 (-100.0 / -60.0, -nr 4)", w3 and near(w3[0], -100)
      and near(w3[1], -60) and all("-nr 4" in l for l in ul if l.startswith("integrate.py")),
      (row(rootA, 3), [l for l in ul if l.startswith("integrate.py")]))
check("  and the structure archives again, exit 0", archived(rootA) and r.returncode == 0, r.returncode)

# ---------------------------------------------------------------------------
print("\n[6] the structure set up again in the same run directory")
for leftover in (os.path.join(rootA, "run", "S1.tar.gz"), sdir(rootA)):
    if os.path.isdir(leftover):
        shutil.rmtree(leftover)
    elif os.path.exists(leftover):
        os.remove(leftover)
new_structure(rootA)
r = run_job(rootA, scenario="bindfwd_1 die 0 no 2\n", FAKE_RMSD="0.7500")
st = statuses(rootA)
check("results_0.gs: the new setup says SETUP_RUNNING, then OK", st[-2:] == ["SETUP_RUNNING", "OK"], st)
w1, w2 = works(row(rootA, 1)), works(row(rootA, 2))
check("cycle 1 of the new setup failed: its row is NaN, not the old -100/-60",
      w1 is not None and all(is_nan(x) for x in w1), row(rootA, 1))
check("cycle 2 carries the new setup's RMSD 0.75, not the old 1.25", w2 and near(w2[2], 0.75), row(rootA, 2))
check("the old cycle 3 row is gone", row(rootA, 3) is None, row(rootA, 3))
open(os.path.join(rootA, "run", "sp.gs"), "w").write("S1\tB\n")
g = subprocess.run([sys.executable, os.path.join(ROOT, "groscore.py"), "-n", "3"],
                   cwd=os.path.join(rootA, "run"), capture_output=True, text=True)
sc = [l.split() for l in open(os.path.join(rootA, "run", "scores_avg.gs"))
      if l.startswith("S1")] if os.path.isfile(os.path.join(rootA, "run", "scores_avg.gs")) else []
check("groscore.py scores the new setup's one cycle: -80.0, 1 cycle, RMSD 0.75",
      sc and sc[0][1] == "-80.0" and sc[0][3] == "1" and sc[0][4] == "0.75", (sc, g.stderr[-300:]))

# ---------------------------------------------------------------------------
print("\n[7] ligands")
rootH = make_root()
new_structure(rootH)
r = run_job(rootH, FAKE_LIGANDS="GNP_B_1167", FAKE_LIGFAIL="GNP B 1167 AM1-BCC failed (sqm)")
check("a ligand_failed.gs line: LIGAND_FAIL, exit 0", statuses(rootH)[-1:] == ["LIGAND_FAIL"]
      and r.returncode == 0, (statuses(rootH), r.returncode))
check("  the line is printed, and no minimisation was started",
      "GNP B 1167 AM1-BCC failed" in r.stdout and "emin_vac" not in grompps(rootH), grompps(rootH))
check("  parametrize_ligand.py ran for the ligand",
      any(l.startswith("parametrize_ligand.py -f ligand_GNP_B.pdb") for l in log(rootH, "util")))
rootH2 = make_root()
new_structure(rootH2)
r = run_job(rootH2, FAKE_LIGANDS="ATP_A_501", FAKE_PARAM_RC=1, FAKE_LIGFAIL="keep")
lf = os.path.join(sdir(rootH2), "ligand_failed.gs")
check("a parametrisation that dies without a reason is recorded by job.run",
      os.path.isfile(lf) and open(lf).read().startswith("ATP A 501 parametrize_ligand.py exited 1"),
      open(lf).read() if os.path.isfile(lf) else "no ligand_failed.gs")
check("  and stops the structure as LIGAND_FAIL", statuses(rootH2)[-1:] == ["LIGAND_FAIL"], statuses(rootH2))
rootH3 = make_root()
new_structure(rootH3, runs="B 4 1 gromos54a8 1")
r = run_job(rootH3, FAKE_LIGANDS="SO4_A_401",
            FAKE_LIGFAIL="SO4 A 401 not parametrised: force field gromos54a8 has no ligand parametrisation")
ul = log(rootH3, "util")
check("GROMOS 54A8: merge_ligand.py --ligand-param off-ff, no parametrisation",
      any(l.startswith("merge_ligand.py") and "--ligand-param off-ff" in l for l in ul)
      and not [l for l in ul if l.startswith("parametrize_ligand")], [l for l in ul if "ligand" in l])
check("  and no glycan fusion: renumber_pdb.py reads input.pdb",
      not [l for l in ul if l.startswith("fuse_glycans.py")]
      and [l for l in ul if l.startswith("renumber_pdb.py")][:1] == ["renumber_pdb.py -f input.pdb -o renumbered.pdb -c B"],
      [l for l in ul if l.startswith(("fuse_glycans", "renumber_pdb"))])
check("  and the in-region ligand stops it as LIGAND_FAIL", statuses(rootH3)[-1:] == ["LIGAND_FAIL"],
      statuses(rootH3))
rootH4 = make_root()
new_structure(rootH4, runs="B 2 1 amber19sb_opc3 0")
r = run_job(rootH4, FAKE_LIGANDS="SO4_A_401")
ul = log(rootH4, "util")
check("--no-ligand-param: --ligand-param off-user, no parametrisation, the run goes on",
      any(l.startswith("merge_ligand.py") and "--ligand-param off-user" in l for l in ul)
      and not [l for l in ul if l.startswith("parametrize_ligand")] and statuses(rootH4)[-1:] == ["OK"],
      (statuses(rootH4), [l for l in ul if "ligand" in l]))
check("  and no glycan fusion: renumber_pdb.py reads input.pdb",
      not [l for l in ul if l.startswith("fuse_glycans.py")]
      and [l for l in ul if l.startswith("renumber_pdb.py")][:1] == ["renumber_pdb.py -f input.pdb -o renumbered.pdb -c B"],
      [l for l in ul if l.startswith(("fuse_glycans", "renumber_pdb"))])

# ---------------------------------------------------------------------------
print("\n[8] stage-0 verdicts: each stops with its status and no OK")
for what, fake, status in (("make_disres_en.py finds 0 interface pairs (v1.4: scored 0.0)",
                            {"FAKE_NPR": 0}, "DISRES_FAIL"),
                           ("make_disres_en.py exits 1 (empty set or split frame)",
                            {"FAKE_MDE_RC": 1}, "DISRES_FAIL"),
                           ("assemble_frame.py exits 1", {"FAKE_AF_RC": 1}, "ASSEMBLE_FAIL"),
                           ("check_brokenloop.py prints nothing (v1.4: passed)",
                            {"FAKE_BL": "EMPTY"}, "FAILED")):
    root = make_root()
    new_structure(root)
    r = run_job(root, **fake)
    st = statuses(root)
    check("%s -> %s" % (what, status), st[-1:] == [status] and "OK" not in st
          and not [g for g in grompps(root) if g.startswith("bindfwd")] and r.returncode == 0,
          (st, r.returncode, r.stdout[-300:]))
    shutil.rmtree(root, ignore_errors=True)
rootZ = make_root()
new_structure(rootZ)
run_job(rootZ, scenario="bindrev_2 die 0.4 no 2\n")
if archived(rootZ):
    unarchive(rootZ)
open(os.path.join(sdir(rootZ), "numpertres.gs"), "w").write("0\n")
r = run_job(rootZ)
check("pull legs exist but numpertres.gs says 0 -> DISRES_FAIL, no leg runs",
      statuses(rootZ)[-1:] == ["DISRES_FAIL"] and "numpertres.gs" in r.stdout and not mdruns(rootZ),
      (statuses(rootZ), r.stdout[-300:]))
shutil.rmtree(rootZ, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\n[9] one job.run per directory")
def locked_structure(pid):
    root = make_root()
    d = new_structure(root)
    os.makedirs(os.path.join(d, "job_run.lock"))
    open(os.path.join(d, "job_run.lock", "owner"), "w").write("%s:%d:-\n" % (socket.gethostname(), pid))
    return root


rootL = locked_structure(os.getpid())
r = run_job(rootL)
check("a live owner: exit 1, 'Safety check', no gmx call, no status",
      r.returncode == 1 and "Safety check" in r.stdout and not log(rootL, "gmx") and not statuses(rootL),
      (r.returncode, r.stdout[-300:]))
shutil.rmtree(rootL, ignore_errors=True)
dead = subprocess.Popen(["true"])
dead.wait()
rootL = locked_structure(dead.pid)
r = run_job(rootL)
check("a dead owner on this host: reclaimed, the run completes",
      "reclaiming stale lock" in r.stdout and r.returncode == 0 and archived(rootL),
      (r.returncode, r.stdout[-300:]))
rootR = make_root()
d = new_structure(rootR)
open(os.path.join(d, "job_local.out"), "w").write("===== S1 on GPU 0 =====\n")
r = run_job(rootR, backdate=False)
check("a fresh --run-local directory (run.gs, job.run, job_local.out) runs (v1.4: exit 0, nothing)",
      r.returncode == 0 and archived(rootR) and "70 minutes" not in r.stdout, (r.returncode, r.stdout[-300:]))
# SIGTERM to the script alone (a local runner's terminate(), not to its mdrun): the
# job must stop the running leg at once (review S3: it waited until the leg had
# run to its end, up to LEG_TIMEOUT), release the lock, and start nothing.
rootS = make_root()
d = new_structure(rootS)
open(os.path.join(d, "scenario.txt"), "w").write("nvt_1_c1 slow 20\n")
for lg in ("gmx.log", "util.log"):
    open(os.path.join(rootS, lg), "w").close()
p = subprocess.Popen(["bash", "job.run"], cwd=d, env=job_env(rootS), stdin=subprocess.DEVNULL,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
t0 = time.time()
while not os.path.exists(os.path.join(d, ".slow_nvt_1_c1")) and time.time() - t0 < 120 \
        and p.poll() is None:
    time.sleep(0.1)
time.sleep(0.5)
t_term = time.time()
p.send_signal(signal.SIGTERM)
out, _ = p.communicate(timeout=120)
took = time.time() - t_term
check("SIGTERM during a 20 s leg: exit 143 within 10 s, nothing started after it",
      p.returncode == 143 and took < 10 and "nvt_2_c1" not in grompps(rootS),
      (p.returncode, "%.1f s" % took, grompps(rootS)[-3:]))
check("  the leg was stopped, not run to its end (no nvt_1_c1.gro)",
      not os.path.exists(os.path.join(d, "nvt_1_c1.gro")))
check("  and job_run.lock released", not os.path.exists(os.path.join(d, "job_run.lock")))
shutil.rmtree(rootS, ignore_errors=True)
# job.run killed hard (SIGKILL, as `pkill -9` or an OOM kill) while its pull leg runs:
# the leg runs on under timeout. The next start must not run the same leg beside it
# (review S3: it reclaimed the lock from the dead pid and ran a second bindfwd_1).
rootO = make_root()
d = new_structure(rootO)
open(os.path.join(d, "scenario.txt"), "w").write("bindfwd_1 slow 12\n")
for lg in ("gmx.log", "util.log"):
    open(os.path.join(rootO, lg), "w").close()
p = subprocess.Popen(["bash", "job.run"], cwd=d, env=job_env(rootO), stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
while not os.path.exists(os.path.join(d, ".slow_bindfwd_1")) and time.time() - t0 < 120 \
        and p.poll() is None:
    time.sleep(0.1)
p.kill()
p.wait()
r = run_job(rootO, backdate=False)
check("job.run SIGKILLed during its pull: the next start exits 1 ('Safety check')",
      r.returncode == 1 and "Safety check" in r.stdout, (r.returncode, r.stdout[-300:]))
check("  because the leg it left still runs, and starts no second bindfwd_1",
      "the leg it started still runs" in r.stdout and not [l for l in mdruns(rootO) if "-deffnm bindfwd_1" in l],
      (r.stdout[-300:], mdruns(rootO)))
t0 = time.time()
while os.path.isdir(d) and not os.path.exists(os.path.join(d, "bindfwd_1.gro")) and time.time() - t0 < 60:
    time.sleep(0.2)
time.sleep(1.0)
if os.path.isdir(d):
    r = run_job(rootO, backdate=False)
    check("once that leg has ended, the next start reclaims the lock and completes",
          r.returncode == 0 and "reclaiming stale lock" in r.stdout and archived(rootO)
          and not [l for l in mdruns(rootO) if "-deffnm bindfwd_1" in l], (r.returncode, r.stdout[-300:]))
else:
    check("once that leg has ended, the next start reclaims the lock and completes", False,
          "the second start already ran the structure to its archive")
shutil.rmtree(rootO, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\n[10] the archive step")
rootT = make_root()
new_structure(rootT)
tbin = os.path.join(rootT, "tarbin")
os.makedirs(tbin)
with open(os.path.join(tbin, "tar"), "w") as f:
    f.write('#!/bin/bash\nif [[ "$1" == "-czf" ]]; then head -c 100 /dev/urandom > "$2"; '
            'echo "tar: Disk quota exceeded" >&2; exit 2; fi\nexec /usr/bin/tar "$@"\n')
os.chmod(os.path.join(tbin, "tar"), 0o755)
r = run_job(rootT, path_prefix=tbin)
check("a tar that fails: the directory stays, no tarball (v1.4: directory deleted)",
      os.path.isdir(sdir(rootT)) and not os.path.exists(os.path.join(rootT, "run", "S1.tar.gz")),
      (os.listdir(os.path.join(rootT, "run")), r.stdout[-200:]))
check("  and says so", "Archive failed or empty for S1" in r.stdout, r.stdout[-300:])

# ---------------------------------------------------------------------------
print("\n[11] leg() on its own: nothing starts from a leg without its .gro")
JOB = open(os.path.join(ROOT, "job.run")).read()


def func(name):
    """The bash function `name` of job.run, verbatim, or None."""
    lines = JOB.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln == "%s() {\n" % name:
            for j in range(i + 1, len(lines)):
                if lines[j] == "}\n":
                    return "".join(lines[i:j + 1])
    return None


fns = [func(n) for n in ("run_leg_mdrun", "require_gro", "leg")]
check("run_leg_mdrun, require_gro and leg are top-level functions of job.run", None not in fns)
if None not in fns:
    rootF = make_root()
    w = os.path.join(rootF, "w")
    os.makedirs(w)
    for f in ("bindfwd_1.tpr", "bindfwd_1.cpt", "nptrev.mdp"):     # a pull that died
        open(os.path.join(w, f), "w").close()
    open(os.path.join(rootF, "gmx.log"), "w").close()
    env = dict(os.environ, PATH=os.path.join(rootF, "bin") + ":" + os.environ["PATH"],
               GMXLOG=os.path.join(rootF, "gmx.log"), FAKE_MDRUN=os.path.join(rootF, "fake_mdrun.py"))
    rr = subprocess.run(["bash", "-c", "nt=1; GPUOPT=''\n" + "".join(fns)
                         + "leg nptrev_2 nptrev.mdp bindfwd_1 60m -n index.ndx\necho rc=$?\n"],
                        cwd=w, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    check("leg nptrev_2 from a bindfwd_1 with .cpt but no .gro: rc 1, no grompp, no mdrun",
          "rc=1" in rr.stdout and not log(rootF, "gmx") and not os.path.exists(os.path.join(w, "nptrev_2.tpr")),
          (rr.stdout, log(rootF, "gmx")))
    check("  naming the leg that did not finish", "bindfwd_1 produced no bindfwd_1.gro" in rr.stdout,
          rr.stdout)
    shutil.rmtree(rootF, ignore_errors=True)
else:
    check("leg nptrev_2 from a bindfwd_1 with .cpt but no .gro: rc 1, no grompp, no mdrun", False,
          "functions not found")
    check("  naming the leg that did not finish", False, "functions not found")

# ---------------------------------------------------------------------------
print("\n[12] a run directory copied without its file times keeps its finished legs")
# cp -r (or scp -r without -p) gives files the times of the copy, and every tpr can
# come out older than the .gro of the leg it starts from. Judged by file times, the
# whole finished cycle of a directory without .done markers (as v1.4 left it) was
# run again (review S6); a tpr built here records which .gro it was built after.
rootC = make_root()
new_structure(rootC)
run_job(rootC, scenario="bindfwd_3 die 0.4 yes 2\n")       # cycle 2 unfinished: no archive
sd = sdir(rootC)
for f in (os.path.join(sd, "results_c1.done"), os.path.join(rootC, "run", "results.d", "S1_c1.gs")):
    if os.path.exists(f):
        os.remove(f)
open(os.path.join(sd, "scenario.txt"), "w").close()
for f in os.listdir(sd):
    t = 1.0e9 if f.endswith(".tpr") else 1.0e9 + 100.0
    os.utime(os.path.join(sd, f), (t, t))
r = run_job(rootC, backdate=False)
c1_legs = {"nvt_%d_c1" % k for k in range(1, 6)} | {"npt_c1", "bindfwd_1", "nptrev_2", "bindrev_2"}
ran = {l.split()[l.split().index("-deffnm") + 1] for l in mdruns(rootC)}
check("every tpr older than its source: no leg of the finished cycle 1 runs again",
      not (ran & c1_legs) and "is older than" not in r.stdout, (sorted(ran & c1_legs), r.stdout[-300:]))
w1 = works(row(rootC, 1))
check("  its row is integrated again from its own force files (-100 / -60)",
      w1 and near(w1[0], -100) and near(w1[1], -60), row(rootC, 1))
check("  and the unfinished cycle 2 is completed, exit 0", r.returncode == 0 and archived(rootC),
      (r.returncode, r.stdout[-300:]))

# ---------------------------------------------------------------------------
print("\n[13] the FeS force-field branch fires on FeS residues, not on methionine")
# Its test was a grep for ' SD ', which matches the SD atom of every methionine: each
# such AMBER19SB structure got a local copy of the force field (review E3).
fes_dir = tempfile.mkdtemp(prefix="fes_branch_")
open(os.path.join(fes_dir, "met.pdb"), "w").write(
    "ATOM      1  SD  MET A   1      10.000  10.000  10.000  1.00  0.00           S\n"
    "ATOM      2  CE  MET A   1      11.500  10.000  10.000  1.00  0.00           C\nEND\n")
open(os.path.join(fes_dir, "sd.pdb"), "w").write(
    "HETATM    1 SD    SD A 300      10.000  10.000  10.000  1.00  0.00           S\nEND\n")
open(os.path.join(fes_dir, "fe2.pdb"), "w").write(
    "HETATM    1 FE   FE2 A 301      10.000  10.000  10.000  1.00  0.00          FE\nEND\n")
for runner in ("job.run", "job_fe.run"):
    src = open(os.path.join(ROOT, runner)).read()
    seg = src[src.index("# Add FeS cluster ion entries"):]
    cond = re.search(r"\n  if (.*?); then\n", seg, re.S)
    got = {}
    for pdbname in ("met.pdb", "sd.pdb", "fe2.pdb"):
        rr = subprocess.run(["bash", "-c", "PDB2GMX_INPUT=%s\nif %s; then echo FES; else echo NO; fi"
                             % (pdbname, cond.group(1) if cond else "false")],
                            cwd=fes_dir, capture_output=True, text=True)
        got[pdbname] = rr.stdout.strip()
    check("%s: methionine no, SD and FE2 residues yes" % runner,
          got == {"met.pdb": "NO", "sd.pdb": "FES", "fe2.pdb": "FES"}, got)
shutil.rmtree(fes_dir, ignore_errors=True)

for root in (rootA, rootB, rootD, rootE, rootK, rootM, rootH, rootH2, rootH3, rootH4, rootL, rootR, rootT, rootC):
    shutil.rmtree(root, ignore_errors=True)

r = subprocess.run(["bash", "-n", os.path.join(ROOT, "job.run")], capture_output=True, text=True)
check("bash -n job.run", r.returncode == 0, r.stderr[:200])

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all classic runner checks passed")
