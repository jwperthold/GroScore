#!/usr/bin/env python3
"""job_fe.run end to end: the guards around its setup, its protocol and its cycles.

The real job_fe.run runs in a scratch checkout whose helper scripts are stubs that
log their arguments and write what the runner looks for, with a fake gmx that does
the same, so every path below is the runner's own control flow, in seconds.

  [1] A FRESH --sequential --run-local START RUNS. The legacy safety check counted
      files modified within 70 minutes, the ones the submitter writes included:
      run.gs and job.run from groscore_fe.py and job_local.out, which
      utils/local_runner.py opens before it starts job.run. So the job exited 0 at
      once and the runner logged it 'done (exit 0)' (audit of 2026-10-04). Driven
      here through the real local runner, the structure is set up, runs its cycle
      and archives. On the way: the reference frame comes from assemble_frame.py,
      not from trjconv -pbc cluster (contract F1; review FR1, 1IJK lost 720 of 1108
      contact pairs to a split -f); protocol_id.gs records the protocol (F5); the
      holds' <hold>_<c>.xvg are gzipped and kept; the setup lock is not archived.
  [2] ONE SETUP OR LEGACY JOB PER DIRECTORY. While a live job holds job_run.lock a
      second legacy or --setup job exits 1 and touches nothing.
  [3] A lock left by a dead job on this host is reclaimed, and released on exit.
  [4] LEGACY MODE HONOURS setup.done (review CYC-1). A pre-v1.5 directory (no
      npt_c_fe.mdp, no protocol_id.gs) with cycle 1 finished and cycle 2 killed in
      npt_c: legacy mode used to re-run the setup, which rebuilt the restraints on
      the old probe, and cycle 2 then went on from its v1.4 checkpoint. Now the
      setup is left alone, cycle 1 is integrated and cycle 2 stops, with no gmx
      call at all.
  [5] A SETUP NEVER VOUCHES FOR CYCLES IT DID NOT START. In the same directory
      with setup.done removed, as the old guard advised, --setup refuses
      (SETUP_FAIL) instead of rebuilding the restraints under the cycles, and the
      waiting cycle tasks then exit on that status. A v1.5 directory whose marker
      groscore_fe.py cleared is set up again as before.
  [6] A CYCLE RUNS ONLY UNDER ITS SETUP'S PROTOCOL: protocol_id.gs of another
      protocol, or none, stops it before any gmx call; the right one runs, also
      while a setup or legacy job holds job_run.lock (cycle tasks never take it).
  [7] A failed assembly stops the setup as ASSEMBLE_FAIL, before make_boresch.py.
  [8] A make_boresch.py abort prints its reason from boresch_failed.gs, not a fixed
      "anchor selection failed" (review CYC-6).
  [9] merge_ligand.py gets --ligand-param on (AMBER19SB), off-user
      (--no-ligand-param) or off-ff (CHARMM36, GROMOS 54A8), and the ligands are
      parametrised only when on (contract F4); so are the glycans fused
      (fuse_glycans.py, then renumber_pdb.py -f fused.pdb; decision of 2026-10-04).
      A parametrisation that exits non-zero without recording a reason gets its
      line from the runner, as in job.run, and stops the structure.
 [10] A RE-RUN AFTER PROBE_FAIL RESUMES THE PROBE (review S4): after an unreadable
      box padding it measures again on the finished replicas, after a replica that
      died twice it runs that replica again from step 0, never from the checkpoint
      the dead run left; neither redoes the setup or a finished replica.

Standalone, no pytest: python3 tests/test_fe_runner_guards.py
"""
import json, os, re, shutil, subprocess, sys, tarfile, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = os.path.join(ROOT, "job_fe.run")
sys.path.insert(0, os.path.join(ROOT, "utils"))
import fe_protocol as P            # scaffolding only: which files a cycle leaves

WANT_ID = "fe-1.5-f78ecb6a"         # tests/test_protocol_id.py pins how it is made
OTHER_ID = "fe-1.5-00000000"
HOST = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
failures = []


def check(name, ok, detail=""):
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:300]))
    if not ok:
        failures.append(name)


FAKE_GMX = r"""#!/bin/bash
echo "$*" >> "$GMXLOG"
sub=$1; shift
out=""; deffnm=""; dhdl=""; pf=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) out=$2; shift ;;
    -deffnm) deffnm=$2; shift ;;
    -dhdl) dhdl=$2; shift ;;
    -pf) pf=$2; shift ;;
  esac
  shift
done
case "$sub" in
  grompp) if [[ -n "$out" ]]; then [[ "$out" == *.tpr ]] || out="$out.tpr"; echo "fake tpr" > "$out"; fi ;;
  mdrun)
    if [[ -n "$deffnm" ]]; then
      echo "fake cpt" > "$deffnm.cpt"; echo "fake gro" > "$deffnm.gro"
      # without -dhdl, mdrun -deffnm writes dH/dlambda to <deffnm>.xvg
      if [[ -n "$dhdl" ]]; then echo "0 0" > "$dhdl"; else echo "0 0" > "$deffnm.xvg"; fi
      [[ -n "$pf" ]] && echo "0 0" > "$pf"
    fi ;;
  pdb2gmx) [[ -n "$out" ]] && echo "fake gro" > "$out"; echo "fake top" > topol.top ;;
  *) [[ -n "$out" ]] && echo "fake $sub" > "$out" ;;
esac
exit 0
"""

STUB_HEAD = (
    "import os, sys\n"
    "with open(os.environ['STUBLOG'], 'a') as _f:\n"
    "    _f.write('%s %s\\n' % (os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])))\n"
    "def arg(k):\n"
    "    a = sys.argv[1:]\n"
    "    return a[a.index(k) + 1] if k in a else None\n")

MAKE_BORESCH = r"""
reason = os.environ.get("MB_ABORT")
if reason:
    sys.stderr.write("make_boresch: ERROR - stub abort %s\n" % reason)
    open("boresch_failed.gs", "w").write(reason + "\nsecond line\n")
    print("0"); sys.exit(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re, fe_protocol as P
for m in sorted({l["mdp"] for l in P.legs()}):
    if os.path.isfile(m):
        open(m, "a").write("pull-ngroups             = 2\npull-ncoords             = 1\n")
if os.path.isfile("npt_c_fe.mdp"):
    t = re.sub(r"(?m)^pull(\s+)= yes", r"pull\1= no", open("npt_c_fe.mdp").read())
    open("npt_c_fe.mdp", "w").write(t)
with open("boresch_analytical.gs", "w") as f:
    f.write("dG_release_kJ_mol -30.0\npull_rate_nm_ps 0.00005\n")
    for l in P.legs():
        if l["kind"] == "stage":
            f.write("stage_rate_nm_ps %s %.10g\n" % (l["name"], (l["u_to"] - l["u_from"]) / l["ps"]))
open("boresch_anchors.gs", "w").write("anchors stub\n")
print("100")
"""

ASSEMBLE = r"""
if os.environ.get("ASSEMBLE_RC") == "1":
    sys.stderr.write("assemble_frame: ERROR - stub cannot assemble %s\n" % arg("-g"))
    sys.exit(1)
open(arg("-o"), "w").write("assembled from %s\n" % arg("-f"))
sys.stderr.write("assemble_frame: 0 residues moved, longest link 0.40 nm\n")
"""

MERGE = r"""
if os.environ.get("MERGE_FAIL"):
    open("ligand_failed.gs", "w").write("SO4 A 401 stub: not in topol.top\n")
"""

# The glycan fusion with nothing to fuse: fused.pdb is input.pdb and glycans.json
# defines no residue (utils/fuse_glycans.py always writes both).
FUSE = r"""
import shutil
shutil.copy(arg("-f"), arg("-o"))
open(arg("-d"), "w").write('{"residues": [], "report": {"fused": [], "not_fused": []}}\n')
if arg("--unfused"):
    open(arg("--unfused"), "w").close()
"""

# The ligand files of a run come from its renumbering (FAKE_LIGANDS, e.g. 'SO4_A'):
# the runner removes every ligand_*.pdb right before renumber_pdb.py, so a file put
# there beforehand is an earlier pass's and is gone.
RENUMBER = r"""
for lig in filter(None, os.environ.get("FAKE_LIGANDS", "").split(",")):
    open("ligand_%s.pdb" % lig, "w").write("HETATM    1  S   SO4 A 401       0.000   0.000   0.000  1.00  0.00           S\n")
"""

# PL_RC: parametrize_ligand.py exits with that status, after writing the line
# PL_REASON for its instance when that is set, without any line otherwise.
PARAM_LIG = r"""
if os.environ.get("PL_REASON"):
    open("ligand_failed.gs", "a").write("SO4 A 401 %s\n" % os.environ["PL_REASON"])
if os.environ.get("PL_RC"):
    sys.exit(int(os.environ["PL_RC"]))
"""

STUBS = {
    "check_brokenloop.py": 'print("0")\n', "check_entangledloops.py": 'print("0")\n',
    "measure_box.py": 'print("1.500")\n', "make_boresch.py": MAKE_BORESCH,
    "assemble_frame.py": ASSEMBLE, "merge_ligand.py": MERGE,
    "integrate.py": 'print("1.0")\n', "integrate_dhdl.py": 'print("1.0")\n',
    "rebound_rmsd.py": 'print("1.0")\n',
    "interface_qc.py": 'print("recovered 1.0 formed 1.0 rms_dev 0.0 sc_rmsd 0.0 npairs 1")\n',
    "fuse_glycans.py": FUSE, "renumber_pdb.py": RENUMBER, "parametrize_ligand.py": PARAM_LIG,
}
for name in ("fix_pdb.py", "cap_termini.py", "fix_ion_protonation.py",
             "parametrize_ncaa.py", "make_cutout.py",
             "merge_crystal_waters.py", "make_ion_restraints.py",
             "fix_topol_intermolecular.py", "make_cluster_group.py"):
    STUBS[name] = "pass\n"


class Sandbox:
    def __init__(self, base):
        self.base = base
        self.co = os.path.join(base, "co")
        self.run = os.path.join(self.co, "run")
        u = os.path.join(self.co, "utils")
        os.makedirs(u)
        os.makedirs(self.run)
        shutil.copy(os.path.join(ROOT, "utils", "fe_protocol.py"), u)
        for name, body in STUBS.items():
            with open(os.path.join(u, name), "w") as f:
                f.write(STUB_HEAD + body)
        os.symlink(os.path.join(ROOT, "settings"), os.path.join(self.co, "settings"))
        self.bin = os.path.join(base, "bin")
        os.makedirs(self.bin)
        with open(os.path.join(self.bin, "gmx"), "w") as f:
            f.write(FAKE_GMX)
        os.chmod(os.path.join(self.bin, "gmx"), 0o755)
        self.gmxlog = os.path.join(base, "gmx.log")
        self.stublog = os.path.join(base, "stub.log")

    def env(self, **extra):
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("SLURM_", "GROSCORE_"))}
        e.update(PATH=self.bin + ":" + os.environ["PATH"], GMXLOG=self.gmxlog,
                 STUBLOG=self.stublog, GROSCORE_NT="1", PYTHONDONTWRITEBYTECODE="1")
        e.update(extra)
        return e

    def reset_logs(self):
        for p in (self.gmxlog, self.stublog):
            open(p, "w").close()

    def gmx(self):
        return open(self.gmxlog).read().splitlines()

    def stubs(self):
        return open(self.stublog).read().splitlines()

    def status(self, sid):
        p = os.path.join(self.run, "results_0.gs")
        if not os.path.isfile(p):
            return []
        return [l.split()[1] for l in open(p) if l.split()[:1] == [sid]]

    def struct(self, sid, runs="A 2 1 amber19sb_opc3 1"):
        s = os.path.join(self.run, sid)
        os.makedirs(s)
        shutil.copy(JOB, os.path.join(s, "job.run"))
        os.chmod(os.path.join(s, "job.run"), 0o755)
        with open(os.path.join(s, "run.gs"), "w") as f:
            f.write(runs + "\n")
        with open(os.path.join(s, "input.pdb"), "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        return s

    def job(self, s, args=(), **extra):
        self.reset_logs()
        return subprocess.run(["bash", "./job.run"] + list(args), cwd=s,
                              env=self.env(**extra), capture_output=True, text=True,
                              timeout=900)


def touch(d, *names, text="dummy\n"):
    for n in names:
        with open(os.path.join(d, n), "w") as f:
            f.write(text)


def crafted(sb, sid, v15, proto, setup_done=True, cycle1=True, cycle2_nptc=True,
            restraints=True, runs="A 4 1 amber19sb_opc3 1"):
    """A structure directory as a finished setup leaves it, made by hand. v15: it
    has npt_c_fe.mdp; proto: protocol_id.gs text or None; cycle1: every leg of
    cycle 1 finished; cycle2_nptc: cycle 2 killed in npt_c (its .tpr and .cpt, no
    .gro), as a walltime kill leaves it; restraints: leg mdps, numpertres.gs and the
    Boresch files exist."""
    s = sb.struct(sid, runs)
    touch(s, "conf.gro", "conf_cutout.gro", "conf_vacbox.gro", "emin_vac.gro",
          "emin_solv.gro", "emin_solv.tpr", "topol.top", "index.ndx", "index_probe.ndx",
          "npt_probe.gro", "npt_probe.tpr", "npt_probe_whole.gro", "npt_probe1.xtc",
          "npt_probe1.tpr", "box.gro", "chain_map.gs")
    touch(s, "boxpad.gs", text="1.500\n")
    tree = os.path.join(ROOT, "settings", "amber19sb_opc3")
    legs = P.legs()
    if restraints:
        for m in sorted({l["mdp"] for l in legs}):
            shutil.copy(os.path.join(tree, m), s)
            with open(os.path.join(s, m), "a") as f:
                f.write("pull-ngroups             = 2\n")
        touch(s, "numpertres.gs", text="100\n")
        touch(s, "boresch_anchors.gs", text="anchors\n")
        with open(os.path.join(s, "boresch_analytical.gs"), "w") as f:
            f.write("dG_release_kJ_mol -30.0\npull_rate_nm_ps 0.00005\n")
            for l in legs:
                if l["kind"] == "stage":
                    f.write("stage_rate_nm_ps %s %.10g\n"
                            % (l["name"], (l["u_to"] - l["u_from"]) / l["ps"]))
        if v15:
            t = open(os.path.join(tree, "npt_c_fe.mdp")).read().replace(
                "pull                     = yes", "pull                     = no")
            with open(os.path.join(s, "npt_c_fe.mdp"), "w") as f:
                f.write(t)
    if setup_done:
        touch(s, "setup.done", text="")
    if proto is not None:
        touch(s, "protocol_id.gs", text=proto + "\n")
    if cycle1:
        touch(s, "nvt_1_c1.tpr", "npt_c1.tpr", "npt_c1.cpt", "npt_c1.gro")
        for l in legs:
            touch(s, *["%s_1.%s" % (l["name"], e) for e in ("tpr", "cpt", "gro")])
            if l["pull"]:
                touch(s, "%s_1_pullf.xvg" % l["name"], text="0 0\n")
            if l["dhdl"]:
                touch(s, "%s_1_dhdl.xvg" % l["name"], text="0 0\n")
    if cycle2_nptc:
        for k in range(1, 6):
            touch(s, *["nvt_%d_c2.%s" % (k, e) for e in ("tpr", "cpt", "gro")])
        touch(s, "npt_c2.tpr", "npt_c2.cpt")
    return s


def age(s, hours=3):
    """Date every file of s back, as a directory that has sat since its last job.
    The file-time guard of older runners then lets a legacy job through, so what
    such a runner does in the directory is what a test of it sees."""
    t = time.time() - hours * 3600
    for n in os.listdir(s):
        os.utime(os.path.join(s, n), (t, t), follow_symlinks=False)


def fields(path):
    return len(open(path).read().split()) if os.path.isfile(path) else 0


def ls(d, n=5):
    """The first n entries of d for a failure message; d may have been archived away."""
    return sorted(os.listdir(d))[:n] if os.path.isdir(d) else "%s is gone" % os.path.basename(d)


with tempfile.TemporaryDirectory() as tmp:
    sb = Sandbox(tmp)
    run = sb.run

    # -----------------------------------------------------------------------
    print("\n[1] a fresh --sequential --run-local start, through the real local runner")
    sb.struct("S1", "A 2 1 amber19sb_opc3 1")
    jobs = {"ngpus": 1, "slots": 1, "threads": 1,
            "jobs": [{"name": "S1", "dir": "S1", "argv": ["./job.run"],
                      "archive": "S1.tar.gz", "log": "job_local.out"}]}
    with open(os.path.join(run, "local_jobs.json"), "w") as f:
        json.dump(jobs, f)
    sb.reset_logs()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "utils", "local_runner.py"),
                        os.path.join(run, "local_jobs.json")], cwd=run, env=sb.env(),
                       capture_output=True, text=True, timeout=900)
    rlog = open(os.path.join(run, "local_runner.log")).read() \
        if os.path.isfile(os.path.join(run, "local_runner.log")) else r.stdout
    st = sb.status("S1")
    check("the job starts its setup (SETUP_RUNNING, then OK)",
          st[:1] == ["SETUP_RUNNING"] and "OK" in st, st)
    arc = os.path.join(run, "S1.tar.gz")
    names, member = [], {}
    if os.path.isfile(arc):
        with tarfile.open(arc) as t:
            names = t.getnames()
            for n in ("S1/job_local.out", "S1/protocol_id.gs", "S1/npt_probe_cluster.gro"):
                if n in names:
                    member[n] = t.extractfile(n).read().decode()
    out = member.get("S1/job_local.out", "")
    check("  and runs to the end: cycle 1 done, structure archived",
          os.path.isfile(arc) and not os.path.isdir(os.path.join(run, "S1"))
          and "S1/results_fe_c1.done" in names, (os.path.isfile(arc), names[:3]))
    check("  the runner logs it done, exit 0", "S1: done (exit 0)" in rlog, rlog[-300:])
    check("  and no 70-minute safety exit in its log",
          "within the last 70 minutes" not in out and "Setup complete for S1" in out,
          out[-300:])
    check("  the 43-field row is written",
          fields(os.path.join(run, "results_fe.d", "S1_c1.gs")) == 43)
    check("  protocol_id.gs records %s" % WANT_ID,
          member.get("S1/protocol_id.gs", "").strip() == WANT_ID,
          member.get("S1/protocol_id.gs"))
    check("  job_run.lock is not in the archive", not any("job_run.lock" in n for n in names),
          [n for n in names if "lock" in n])
    holds = ["S1/%s_1.xvg.gz" % h for h in ("holdfwd0", "holdbfwd1", "holdbrev1", "nptrev",
                                             "holdrev3")]
    check("  the holds' dH/dlambda files are kept, gzipped",
          all(h in names for h in holds)
          and not any(n.endswith(("holdfwd0_1.xvg", "nptrev_1.xvg")) for n in names),
          [n for n in names if "hold" in n and "xvg" in n][:6])
    asm = [l for l in sb.stubs() if l.startswith("assemble_frame.py ")]
    check("  the reference frame is assembled by assemble_frame.py from the whole frame",
          asm == ["assemble_frame.py -f npt_probe_whole.gro -n index_probe.ndx "
                  "-g Protein_Struct -o npt_probe_cluster.tmp.gro"], asm)
    check("  and that is the frame make_boresch.py reads",
          member.get("S1/npt_probe_cluster.gro", "").startswith(
              "assembled from npt_probe_whole.gro")
          and any(l.startswith("make_boresch.py -f npt_probe_cluster.gro")
                  for l in sb.stubs()), member.get("S1/npt_probe_cluster.gro"))
    g = sb.gmx()
    check("  trjconv -pbc cluster is not used",
          not [l for l in g if "-pbc cluster" in l], [l for l in g if "cluster" in l][:3])
    check("  the .pdb copy is made from the assembled frame",
          [l for l in g if l.startswith("trjconv") and "-o npt_probe_cluster.pdb" in l
           and "-f npt_probe_cluster.gro" in l] != [],
          [l for l in g if "npt_probe_cluster.pdb" in l])

    # -----------------------------------------------------------------------
    print("\n[2] one setup or legacy job per directory")
    s2 = sb.struct("S2")
    os.makedirs(os.path.join(s2, "job_run.lock"))
    owner = "%s:%d:-" % (HOST, os.getpid())
    touch(s2, "job_run.lock/owner", text=owner + "\n")
    for args, what in (((), "legacy"), (("--setup",), "--setup")):
        r = sb.job(s2, args)
        check("%s while a live job holds job_run.lock: exit 1, says so" % what,
              r.returncode == 1 and "holds job_run.lock" in r.stdout,
              (r.returncode, r.stdout[-200:]))
        check("  no status line, no gmx call, no helper run",
              sb.status("S2") == [] and sb.gmx() == [] and sb.stubs() == [],
              (sb.status("S2"), sb.gmx()[:2], sb.stubs()[:2]))
        check("  and the holder's lock is left as it was",
              open(os.path.join(s2, "job_run.lock", "owner")).read().strip() == owner)

    # -----------------------------------------------------------------------
    print("\n[3] a dead job's lock is reclaimed and released on exit")
    s3 = sb.struct("S3")
    os.makedirs(os.path.join(s3, "job_run.lock"))
    touch(s3, "job_run.lock/owner", text="%s:4999999:-\n" % HOST)
    r = sb.job(s3, (), MERGE_FAIL="1", FAKE_LIGANDS="SO4_A")
    check("legacy run over a dead owner's lock goes on (reclaimed)",
          "reclaiming stale lock" in r.stdout and sb.status("S3")[:1] == ["SETUP_RUNNING"],
          (r.stdout[-200:], sb.status("S3")))
    check("  stops on its LIGAND_FAIL, exit 0, and releases the lock",
          r.returncode == 0 and sb.status("S3")[-1:] == ["LIGAND_FAIL"]
          and not os.path.exists(os.path.join(s3, "job_run.lock")),
          (r.returncode, sb.status("S3"), ls(s3)))

    # -----------------------------------------------------------------------
    print("\n[4] legacy mode in a pre-v1.5 directory leaves its setup alone")
    s4 = crafted(sb, "S4", v15=False, proto=None)
    age(s4)
    r = sb.job(s4, (), GROSCORE_NUMCYCLES="2")
    st = sb.stubs()
    check("the setup is not re-run (no make_boresch.py, no assemble_frame.py)",
          not [l for l in st if l.startswith(("make_boresch", "assemble_frame"))],
          [l.split()[0] for l in st])
    check("  no npt_c_fe.mdp appears, no SETUP_RUNNING line",
          not os.path.exists(os.path.join(s4, "npt_c_fe.mdp")) and sb.status("S4") == [],
          sb.status("S4"))
    check("  it warns that the structure predates v1.5", "WARNING" in r.stdout
          and "fresh directory" in r.stdout, r.stdout[-400:])
    check("  cycle 1, every leg done, is integrated: 43-field row and .done",
          fields(os.path.join(run, "results_fe.d", "S4_c1.gs")) == 43
          and os.path.isfile(os.path.join(s4, "results_fe_c1.done")))
    check("  cycle 2, killed in npt_c, stops and names protocol_id.gs",
          "Cycle 2 FAILED" in r.stdout and "protocol_id.gs" in r.stdout
          and not os.path.isfile(os.path.join(s4, "results_fe_c2.done")), r.stdout[-400:])
    check("  with no gmx call at all, and exit 1", sb.gmx() == [] and r.returncode == 1,
          (r.returncode, sb.gmx()[:3]))
    # The cycle phase sets its own EXIT trap; it must still release the setup lock.
    check("  and job_run.lock is released when the legacy job ends after its cycles",
          os.path.isdir(s4) and not os.path.exists(os.path.join(s4, "job_run.lock")), ls(s4))

    # -----------------------------------------------------------------------
    print("\n[5] a setup never vouches for cycles it did not start")
    s5 = crafted(sb, "S5", v15=False, proto=None, setup_done=False)
    r = sb.job(s5, ("--setup",))
    check("--setup after setup.done was removed, cycles present: refused, SETUP_FAIL",
          r.returncode == 0 and "Stage 0 refused" in r.stdout
          and sb.status("S5")[-1:] == ["SETUP_FAIL"], (r.returncode, sb.status("S5"),
                                                       r.stdout[-300:]))
    check("  nothing rebuilt: no helper, no gmx call, no setup.done, no protocol_id.gs",
          sb.stubs() == [] and sb.gmx() == []
          and not os.path.exists(os.path.join(s5, "setup.done"))
          and not os.path.exists(os.path.join(s5, "protocol_id.gs")),
          (sb.stubs()[:3], sb.gmx()[:3]))
    r = sb.job(s5, ("--cycle", "2"), GROSCORE_NUMCYCLES="2", GROSCORE_SETUP_POLL="1",
               GROSCORE_SETUP_WAIT="5")
    check("  and a cycle task waiting on it exits 0 on that status, no gmx call",
          r.returncode == 0 and "failed (SETUP_FAIL)" in r.stdout and sb.gmx() == [],
          (r.returncode, r.stdout[-200:]))
    s5b = crafted(sb, "S5b", v15=True, proto=WANT_ID, setup_done=False)
    r = sb.job(s5b, ("--setup",))
    check("a v1.5 directory whose setup.done was cleared is set up again as before",
          r.returncode == 0 and "Setup complete for S5b" in r.stdout
          and any(l.startswith("make_boresch.py") for l in sb.stubs())
          and os.path.isfile(os.path.join(s5b, "setup.done"))
          and open(os.path.join(s5b, "protocol_id.gs")).read().strip() == WANT_ID,
          (r.returncode, r.stdout[-300:]))

    # -----------------------------------------------------------------------
    print("\n[6] a cycle runs only under its setup's protocol")
    for sid, proto, what in (("S6a", OTHER_ID, "set up for protocol %s" % OTHER_ID),
                             ("S6b", None, "no protocol_id.gs")):
        s6 = crafted(sb, sid, v15=True, proto=proto, cycle2_nptc=False)
        r = sb.job(s6, ("--cycle", "2"), GROSCORE_NUMCYCLES="2")
        check("protocol_id.gs %s: cycle 2 stops, exit 1, no gmx call"
              % ("of another protocol" if proto else "absent"),
              r.returncode == 1 and what in r.stdout and sb.gmx() == []
              and not os.path.isfile(os.path.join(s6, "results_fe_c2.done")),
              (r.returncode, r.stdout[-300:], sb.gmx()[:2]))
    s6 = crafted(sb, "S6c", v15=True, proto=WANT_ID, cycle2_nptc=False)
    os.makedirs(os.path.join(s6, ".archive.lock"))       # keep it for inspection
    os.makedirs(os.path.join(s6, "job_run.lock"))
    touch(s6, "job_run.lock/owner", text=owner + "\n")
    r = sb.job(s6, ("--cycle", "2"), GROSCORE_NUMCYCLES="2")
    g = sb.gmx()
    check("protocol_id.gs of this protocol: cycle 2 runs to its .done",
          r.returncode == 0 and os.path.isfile(os.path.join(s6, "results_fe_c2.done"))
          and fields(os.path.join(run, "results_fe.d", "S6c_c2.gs")) == 43,
          (r.returncode, r.stdout[-300:]))
    check("  npt_c from ./npt_c_fe.mdp, boundrev2 from the end of holdfwd0",
          any("-f ./npt_c_fe.mdp" in l and "-o npt_c2" in l for l in g)
          and any("-c holdfwd0_2.tpr -t holdfwd0_2.cpt" in l and "-o boundrev2_2" in l
                  for l in g), [l for l in g if "boundrev2_2" in l][:1])
    check("  while a setup job held job_run.lock, which it left alone",
          open(os.path.join(s6, "job_run.lock", "owner")).read().strip() == owner)

    # -----------------------------------------------------------------------
    print("\n[7] an assembly failure stops the setup")
    s7 = crafted(sb, "S7", v15=True, proto=None, setup_done=False, cycle1=False,
                 cycle2_nptc=False, restraints=False)
    r = sb.job(s7, ("--setup",), ASSEMBLE_RC="1")
    check("assemble_frame.py exits 1: ASSEMBLE_FAIL, exit 0",
          r.returncode == 0 and sb.status("S7")[-1:] == ["ASSEMBLE_FAIL"]
          and "assemble_frame.py exited 1" in r.stdout, (r.returncode, sb.status("S7")))
    check("  before make_boresch.py, with no setup.done and no reference frame",
          not [l for l in sb.stubs() if l.startswith("make_boresch")]
          and not os.path.exists(os.path.join(s7, "setup.done"))
          and not os.path.exists(os.path.join(s7, "npt_probe_cluster.gro")),
          [l.split()[0] for l in sb.stubs()])

    # -----------------------------------------------------------------------
    print("\n[8] a make_boresch.py abort is reported by its reason")
    s8 = crafted(sb, "S8", v15=True, proto=None, setup_done=False, cycle1=False,
                 cycle2_nptc=False, restraints=False)
    r = sb.job(s8, ("--setup",), MB_ABORT="NPT_C_MDP_MISSING")
    check("the log names NPT_C_MDP_MISSING, not an anchor failure",
          "make_boresch.py stopped with NPT_C_MDP_MISSING" in r.stdout
          and "anchor selection failed" not in r.stdout, r.stdout[-300:])
    check("  status BORESCH_FAIL, exit 0, no setup.done",
          r.returncode == 0 and sb.status("S8")[-1:] == ["BORESCH_FAIL"]
          and not os.path.exists(os.path.join(s8, "setup.done")), sb.status("S8"))

    # -----------------------------------------------------------------------
    print("\n[9] merge_ligand.py is told why ligands are or are not parametrised")
    for sid, ff, lp, mode in (("L1", "amber19sb_opc3", 1, "on"),
                              ("L2", "amber19sb_opc3", 0, "off-user"),
                              ("L3", "charmm36", 1, "off-ff"),
                              ("L4", "gromos54a8_rf", 1, "off-ff")):
        s9 = sb.struct(sid, "A 2 1 %s %d" % (ff, lp))
        r = sb.job(s9, ("--setup",), MERGE_FAIL="1", FAKE_LIGANDS="SO4_A")
        merges = [l for l in sb.stubs() if l.startswith("merge_ligand.py")]
        params = [l for l in sb.stubs() if l.startswith("parametrize_ligand.py")]
        check("%s, ligand field %d: --ligand-param %s%s" % (ff, lp, mode,
              ", ligands parametrised" if mode == "on" else ", none parametrised"),
              len(merges) == 1 and merges[0].endswith("--ligand-param %s" % mode)
              and (len(params) == 1) == (mode == "on"), (merges, params))
        fuse = [l for l in sb.stubs() if l.startswith("fuse_glycans.py")]
        ren = [l for l in sb.stubs() if l.startswith("renumber_pdb.py")]
        check("  glycans fused only when on: fuse_glycans.py %s, renumber_pdb.py -f %s"
              % (("ran", "fused.pdb") if mode == "on" else ("did not run", "input.pdb")),
              len(fuse) == (1 if mode == "on" else 0) and len(ren) == 1
              and ren[0].startswith("renumber_pdb.py -f %s " % ("fused.pdb" if mode == "on" else "input.pdb")),
              (fuse, ren))
        check("  and its ligand_failed.gs line stops the structure (LIGAND_FAIL)",
              sb.status(sid)[-1:] == ["LIGAND_FAIL"] and r.returncode == 0,
              (r.returncode, sb.status(sid)))

    # The exit status of parametrize_ligand.py is not ignored, as in job.run: one that
    # dies before it records anything gets its line from the runner, and one that
    # recorded its own reason gets no second line.
    for sid, extra, want in (
            ("L5", dict(PL_RC="1"), ["SO4 A 401 parametrize_ligand.py exited 1 without recording a reason"]),
            ("L6", dict(PL_RC="1", PL_REASON="its own reason"), ["SO4 A 401 its own reason"])):
        s9 = sb.struct(sid, "A 2 1 amber19sb_opc3 1")
        r = sb.job(s9, ("--setup",), FAKE_LIGANDS="SO4_A", **extra)
        lf = os.path.join(s9, "ligand_failed.gs")
        got = [l.rstrip("\n") for l in open(lf) if l.strip()] if os.path.isfile(lf) else []
        check("parametrize_ligand.py exits 1%s: ligand_failed.gs is %s" % (
              " after recording its reason" if "PL_REASON" in extra else " without a reason",
              "its own line, once" if "PL_REASON" in extra else "the runner's line"),
              got == want and "failed to parametrise (exit 1)" in r.stdout, (got, r.stdout[-300:]))
        check("  and the structure stops as LIGAND_FAIL, exit 0",
              sb.status(sid)[-1:] == ["LIGAND_FAIL"] and r.returncode == 0, (r.returncode, sb.status(sid)))

    # -----------------------------------------------------------------------
    print("\n[10] a re-run after PROBE_FAIL resumes the probe, as its messages say")
    # The restart recovery (conf.gro but no emin_solv.gro) fired after every
    # PROBE_FAIL and deleted every probe replica, up to five of 20 ns, although the
    # failure messages promise a re-measurement and a resume from the checkpoint
    # (review S4).
    def replica_runs():
        return sorted({l.split("-deffnm ")[1].split()[0] for l in sb.gmx()
                       if l.startswith("mdrun") and "-deffnm npt_probe" in l})

    mb_path = os.path.join(sb.co, "utils", "measure_box.py")
    mb_orig = open(mb_path).read()
    with open(mb_path, "w") as f:          # the first answer is not a number
        f.write(STUB_HEAD + "m = 'mb_called'\nprint('nan' if not os.path.exists(m) else '1.500')\n"
                "open(m, 'a').close()\n")
    sa = sb.struct("PA")
    r = sb.job(sa, ("--setup",))
    check("an unreadable box padding: PROBE_FAIL, 'removed for re-measurement'",
          sb.status("PA")[-1:] == ["PROBE_FAIL"] and "removed for re-measurement" in r.stdout,
          (sb.status("PA"), r.stdout[-300:]))
    reps = sorted(n for n in os.listdir(sa) if re.match(r"npt_probe\d\.(gro|cpt)$", n))
    for n in reps:
        touch(sa, n, text="REPLICA OF RUN 1\n")
    r = sb.job(sa, ("--setup",))
    check("  the re-run keeps the setup: no restart recovery, no renumber_pdb.py, no pdb2gmx",
          "Restart recovery" not in r.stdout and not [l for l in sb.stubs() if l.startswith("renumber_pdb")]
          and not [l for l in sb.gmx() if l.startswith("pdb2gmx")], r.stdout[-300:])
    check("  measures again on the %d finished replica files, simulating none of them" % len(reps),
          len(reps) == 10 and replica_runs() == [] and all("RUN 1" in open(os.path.join(sa, n)).read() for n in reps)
          and any(l.startswith("measure_box.py") for l in sb.stubs()), (len(reps), replica_runs()))
    check("  and completes the setup", "Setup complete for PA" in r.stdout and r.returncode == 0,
          (r.returncode, r.stdout[-300:]))
    with open(mb_path, "w") as f:
        f.write(mb_orig)
    gmx_path = os.path.join(sb.bin, "gmx")
    gmx_orig = open(gmx_path).read()
    with open(gmx_path, "w") as f:         # replica 3 dies twice, leaving its checkpoint
        f.write(gmx_orig.replace("  mdrun)\n", "  mdrun)\n"
                                 '    if [[ "$deffnm" == npt_probe3 && ! -f .p3_ok ]]; then\n'
                                 '      echo "fake cpt" > npt_probe3.cpt; exit 1\n'
                                 '    fi\n', 1))
    sbb = sb.struct("PB")
    r = sb.job(sbb, ("--setup",))
    check("a probe replica that dies twice: PROBE_FAIL, 'runs npt_probe3 again from step 0'",
          sb.status("PB")[-1:] == ["PROBE_FAIL"] and "runs npt_probe3 again from step 0" in r.stdout,
          (sb.status("PB"), r.stdout[-300:]))
    touch(sbb, ".p3_ok", text="")
    r = sb.job(sbb, ("--setup",))
    p3 = [l for l in sb.gmx() if l.startswith("mdrun") and "-deffnm npt_probe3" in l]
    check("  the re-run runs replica 3 again from step 0 (no -cpi), re-runs neither 1 nor 2",
          "Restart recovery" not in r.stdout and replica_runs() == ["npt_probe3", "npt_probe4", "npt_probe5"]
          and p3 and not [l for l in p3 if "-cpi" in l], (replica_runs(), p3))
    check("  and completes the setup", "Setup complete for PB" in r.stdout and r.returncode == 0,
          (r.returncode, r.stdout[-300:]))
    with open(gmx_path, "w") as f:
        f.write(gmx_orig)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all runner-guard checks passed")
