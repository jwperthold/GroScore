#!/usr/bin/env python3
"""Glycans in the setup chain of both runners: fused before every renumbering when
ligands are parametrised, and back to ligands, in one ordered pass, when their
fused residue fails to parametrise.

Decision of 2026-10-04: a glycan covalently bonded to ASN ND2, SER OG or THR OG1
(1MAH: NAG A 544 on ASN A 350; 2AJF: NAGs on ASN A 53, A 322, A 546 and E 330)
is fused with its amino acid into one NCAA by utils/fuse_glycans.py, where it used
to stop the structure as LIGAND_FAIL. The real job.run and job_fe.run run here in a
scratch checkout whose utilities are stubs that log their arguments and do what the
runner relies on: the fusion stub renames ASN A 350 to NG1 (and ASN A 53 to NG2)
and moves NAG A 544 (NAG A 701) behind it as ATOM records, renumber_pdb.py extracts
every HETATM residue to ligand_*.pdb under the real file names, parametrize_ncaa.py
writes the failure lists a scenario asks for, and merge_ligand.py ends every run
with a ligand_failed.gs line, before any minimisation. Pinned, in both runners:

  [1] LIGAND_PARAM_MODE on (AMBER19SB/OPC3): fuse_glycans.py runs right before
      every renumber_pdb.py call, which reads fused.pdb; fix_pdb.py keeps NG1; the
      fused NAG A 544 has no ligand file (NAG A 701, not fused in this run, has
      one). off-ff (CHARMM36, GROMOS 54A8_rf) and off-user
      (--no-ligand-param): no fusion anywhere, input.pdb is renumbered and both
      NAGs are extracted (ligand_NAG_A_544.pdb, ligand_NAG_A_701.pdb).
  [2] Ligand files of an earlier pass (ligand_NAG_A.pdb, ligand_NAG_A_703.pdb,
      ligand_info.gs) are gone whenever renumber_pdb.py starts, in the first pass
      and in the fallback (critic F3: two passes over 2AJF left one NAG in two
      files, and an old 1MAH directory kept its ligand_NAG_A.pdb).
  [3] NG1 named in glycan_failed.gs: after parametrize_ncaa.py, in one pass and in
      this order, fuse_glycans.py --exclude-file glycan_failed.gs, renumber_pdb.py
      on fused.pdb, fix_pdb.py on a file without NG1 and without keeping it, and
      cap_termini.py (critic F4: the old block ran fix_pdb.py first, on the
      renumbered.pdb that still held the failed residue). NG2, which did
      parametrise, stays kept, and the NCAA list the runner prints afterwards
      is NG2 and MSE. NAG A 544 is extracted and parametrised as a ligand,
      and its glycan_unfused.gs line carries the reason once, whether the fusion
      wrote a line without it, no line, or the reason already (fuse_glycans.py's
      own wording, spaces normalised), and also when glycans.json labels the sugar
      'NAG A 544A' while its line is keyed 'NAG A 544', as merge_ligand.py looks it
      up. An NCAA (MSE) and a glycan failing together take the same single pass.
  [4] parametrize_ncaa.py exits 1 with NG1 and NG2 fused and no failure file: both
      count as failed ('NAME parametrize_ncaa.py exited 1') and both NAGs become
      ligands, also when glycans.json cannot be read (one reason for that exit):
      the fused names are those fused.pdb has and input.pdb has not. Without a
      fused residue a crash starts no fallback, as before.
  [5] An NCAA failure alone (MSE): the fallback does not fuse again, it renumbers
      the first pass's fused.pdb, which is what parametrize_ncaa.py parametrised,
      so NG1 stays fused and kept; MSE is not kept (runner review R2: a fusion
      redone from scratch can fuse what the first pass did not, or the other way
      round, when a CCD entry answers in one pass only).
  [6] fuse_glycans.py exits 1: LIGAND_FAIL, 'fuse_glycans.py exited 1', nothing
      renumbered. GROSCORE_GLYCAN_MAX_SUGARS=2 and =1 reach it as --max-sugars 2
      and 1, unset as no --max-sugars at all; '0' and 'two', which fuse_glycans.py
      would refuse, stop the structure as FAILED before any fusion.
  [7] A fresh setup starts without an earlier attempt's failure lists: stale 'NG1'
      and 'MSE' lines (no conf.gro) start no fallback. Nor with its ligand
      parametrisation outputs (ligand_manifest.gs, ligand_*.itp/.gro/.sdf,
      ligand_failed.gs): an attempt whose pdb2gmx wrote no conf.gro left a manifest
      line of a NAG that the next attempt fused, and merge_ligand.py merged it on top
      of its own sugar (runner review R1). A pdb2gmx that writes no conf.gro stops
      the structure as FAILED at once, before any ligand is parametrised. The
      restart cleanup of each runner removes fused.pdb, glycans.json,
      glycan_unfused.gs, glycan_failed.gs, ligand_*.pdb and ligand_info.gs.
  [8] Both runners carry the same text for the shared setup chain (the functions
      fuse_and_renumber, ncaa_names and ncaa_fallback) and the same call sites;
      each calls renumber_pdb.py and fuse_glycans.py once, the renumbering right
      after the ligand files are removed; no grep of renumbered.pdb.log is left;
      both parse (bash -n).

Standalone, no pytest: python3 tests/test_glycan_runner.py
"""
import json, os, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNERS = ("job.run", "job_fe.run")
failures = []


def check(name, ok, detail=""):
    print("  %-74s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:400]))
    if not ok:
        failures.append(name)


def pdb_line(rec, serial, name, res, chain, num, xyz, el):
    nm = name if len(name) >= 4 else " " + name
    return "%-6s%5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s" % (
        rec, serial, nm, res, chain, num, xyz[0], xyz[1], xyz[2], el)


def input_pdb():
    """Chain A: GLY 52, ASN 53, ASN 350, MSE 351 (an NCAA, as ATOM records); chain B:
    GLY 9; HETATM NAG A 544 (on ASN 350), NAG A 701 (on ASN 53), SO4 A 401 (a plain
    ligand) and one water. Coordinates are arbitrary: no stub reads them."""
    out, k = [], 0
    for res, num, names in (("GLY", 52, "N CA C O"), ("ASN", 53, "N CA C O CB CG OD1 ND2"),
                            ("ASN", 350, "N CA C O CB CG OD1 ND2"), ("MSE", 351, "N CA C O CB CG SE CE")):
        for a in names.split():
            k += 1
            out.append(pdb_line("ATOM", k, a, res, "A", num, (k, 0, 0), "SE" if a == "SE" else a[0]))
    out.append("TER")
    for a in ("N", "CA", "C", "O"):
        k += 1
        out.append(pdb_line("ATOM", k, a, "GLY", "B", 9, (k, 5, 0), a[0]))
    out.append("TER")
    for res, num, names in (("NAG", 544, "C1 C2 O5 N2"), ("NAG", 701, "C1 C2 O5 N2"), ("SO4", 401, "S O1"),
                            ("HOH", 901, "O")):
        for a in names.split():
            k += 1
            out.append(pdb_line("HETATM", k, a, res, "A", num, (k, 9, 0), a[0]))
    return "\n".join(out) + "\nEND\n"


FAKE_GMX = r"""#!/bin/bash
echo "$*" >> "$GMXLOG"
sub=$1; shift
out=""
while [[ $# -gt 0 ]]; do
  case "$1" in -o) out=$2; shift ;; esac
  shift
done
case "$sub" in
  pdb2gmx) cat > /dev/null; [[ -n "$FAKE_PDB2GMX_FAIL" ]] && exit 1
           echo "fake gro" > "$out"; echo "fake top" > topol.top ;;
  *) [[ -n "$out" ]] && echo "fake $sub" > "$out" ;;
esac
exit 0
"""

STUB_HEAD = r'''import collections, json, os, shutil, sys
def arg(k):
    a = sys.argv[1:]
    return a[a.index(k) + 1] if k in a else None
def log(**extra):
    rec = {"util": os.path.basename(sys.argv[0]), "argv": sys.argv[1:]}
    rec.update(extra)
    with open(os.environ["UTILLOG"], "a") as f:
        f.write(json.dumps(rec) + "\n")
def ligand_files():
    return sorted(f for f in os.listdir(".")
                  if (f.startswith("ligand_") and f.endswith(".pdb")) or f == "ligand_info.gs")
def resnames(path):
    if not os.path.isfile(path):
        return []
    return sorted({l[17:20].strip() for l in open(path) if l.startswith(("ATOM  ", "HETATM"))})
'''

# FAKE_SITES: 'NAME CHAIN NUM SUGAR SCHAIN SNUM' entries, ';'-separated: the ASN CHAIN
# NUM becomes NAME and the HETATM sugar moves behind it as ATOM records of NAME, unless
# NAME is in the --exclude-file. FAKE_UNFUSED: what is written to the --unfused file
# for an excluded sugar: 'plain' (a line without the reason), 'none' (no line) or
# 'copy' (utils/fuse_glycans.py's own line, 'NAG A 544 NG1 could not be parametrised:
# <reason>', the reason whitespace-normalised). FAKE_ICODE: an insertion code that
# glycans.json appends to the sugar's label ('NAG A 544A', fuse_glycans.py's label()),
# while the --unfused line keys the sugar without it, as renumber_pdb.py names it.
FUSE = r'''
log(cap=os.environ.get("GROSCORE_GLYCAN_MAX_SUGARS"))
if int(os.environ.get("FAKE_FUSE_RC", "0")):
    sys.exit(int(os.environ["FAKE_FUSE_RC"]))
reasons = {}
if arg("--exclude-file") and os.path.isfile(arg("--exclude-file")):
    for l in open(arg("--exclude-file")):
        if l.split():
            reasons[l.split()[0]] = " ".join(l.split()[1:]) or "no reason recorded"
sites = [s.split() for s in os.environ.get("FAKE_SITES", "").split(";") if s.strip()]
style = os.environ.get("FAKE_UNFUSED", "plain")
icode = os.environ.get("FAKE_ICODE", "")
fused, unfused, residues = [], [], collections.OrderedDict()
for name, ch, num, sres, sch, snum in sites:
    if name in reasons:
        if style == "plain":
            unfused.append("%s %s %s excluded: %s is listed in glycan_failed.gs" % (sres, sch, snum, name))
        elif style == "copy":
            unfused.append("%s %s %s %s could not be parametrised: %s" % (sres, sch, snum, name, reasons[name]))
        continue
    fused.append((name, ch, num, sres, sch, snum))
    residues.setdefault(name, {"resname": name, "parent": "ASN", "instances": []})["instances"].append(
        {"chain": ch, "resnum": num, "icode": "", "sugars": ["%s %s %s%s" % (sres, sch, snum, icode)]})
lines = open(arg("-f")).read().splitlines()
key = lambda l: (l[17:20].strip(), l[21], l[22:26].strip()) if l.startswith(("ATOM  ", "HETATM")) else None
out = []
for l in lines:
    if any(key(l) == (f[3], f[4], f[5]) for f in fused):
        continue
    site = [f for f in fused if key(l) == ("ASN", f[1], f[2])]
    out.append(l[:17] + "%3s" % site[0][0] + l[20:] if site else l)
    if site and not any(key(n) == ("ASN", site[0][1], site[0][2]) for n in lines[lines.index(l) + 1:]):
        name, ch, num = site[0][:3]
        for s in lines:
            if key(s) == site[0][3:]:
                out.append("ATOM  " + s[6:12] + "%-4s" % (" " + s[12:16].strip() + "A")[:4] + " "
                           + "%3s" % name + " " + ch + "%4s" % num + s[26:])
open(arg("-o"), "w").write("\n".join(out) + "\n")
json.dump({"residues": list(residues.values()),
           "report": {"fused": [f[0] for f in fused], "not_fused": unfused}}, open(arg("-d"), "w"))
if os.environ.get("FAKE_BAD_JSON"):           # FAKE_BAD_JSON: an unreadable glycans.json
    open(arg("-d"), "w").write("{not json\n")
if arg("--unfused"):
    open(arg("--unfused"), "w").write("".join(u + "\n" for u in unfused))
'''

RENUMBER = r'''
log(at_entry=ligand_files())
src = open(arg("-f")).read().splitlines()
open(arg("-o"), "w").write("".join(l + "\n" for l in src if l.startswith("ATOM  ")) + "END\n")
open("chain_map.gs", "w").write("# Protein B chain(s): B\n5\n")
open("ion_residues.gs", "w").close()
het = collections.OrderedDict()
for l in src:
    if l.startswith("HETATM") and l[17:20].strip() != "HOH":
        het.setdefault((l[17:20].strip(), l[21], l[22:26].strip()), []).append(l)
count = collections.Counter((r, c) for r, c, n in het)
for (r, c, n), ls in het.items():
    name = "ligand_%s_%s_%s.pdb" % (r, c, n) if count[(r, c)] > 1 else "ligand_%s_%s.pdb" % (r, c)
    open(name, "w").write("".join(x + "\n" for x in ls))
if het:
    open("ligand_info.gs", "w").write("".join("%s %s %d\n" % (r, c, len(ls)) for (r, c, n), ls in het.items()))
'''

PARAM_NCAA = r'''
log()
if arg("--gmx-ff") == "amber19sb":
    why = os.environ.get("FAKE_GLYREASON", "AM1-BCC did not finish within 600 s")
    for n in filter(None, os.environ.get("FAKE_GLYFAIL", "").split(",")):
        open("glycan_failed.gs", "a").write("%s %s\n" % (n, why))
    for n in filter(None, os.environ.get("FAKE_NCAAFAIL", "").split(",")):
        open("ncaa_failed.gs", "a").write("%s\n" % n)
    sys.exit(int(os.environ.get("FAKE_PN_RC", "0")))
'''

MERGE = r'''
log(unfused=open("glycan_unfused.gs").read() if os.path.isfile("glycan_unfused.gs") else None,
    ligands=ligand_files(),
    params=sorted(f for f in os.listdir(".") if f.startswith("ligand_") and f not in ligand_files()))
open("ligand_failed.gs", "w").write("STOP stub: the test ends the setup here\n")
'''

STUBS = {
    "fuse_glycans.py": FUSE,
    "renumber_pdb.py": RENUMBER,
    "parametrize_ncaa.py": PARAM_NCAA,
    "fix_pdb.py": 'log(resnames=resnames(arg("-f")))\nshutil.copy(arg("-f"), arg("-o"))\n',
    "cap_termini.py": 'log()\nshutil.copy(arg("-f"), arg("-o"))\n',
    "make_cutout.py": 'log()\nshutil.copy("fixed_capped.pdb", "cutout.pdb")\n',
    "check_brokenloop.py": 'print("0")\n',
    "merge_ligand.py": MERGE,
}
for _n in ("fix_ion_protonation.py", "parametrize_ligand.py", "merge_crystal_waters.py",
           "make_ion_restraints.py"):
    STUBS[_n] = "log()\n"
STUBS["retype_ions.py"] = "pass\n"      # runs after pdb2gmx; tests/test_retype_ions.py has it


class Sandbox(object):
    """A scratch checkout: co/utils (stubs and the real fe_protocol.py), co/settings
    (the repository's), co/run/<id>/ per structure, bin/gmx."""

    def __init__(self, base):
        self.base = base
        self.co = os.path.join(base, "co")
        self.run = os.path.join(self.co, "run")
        utils = os.path.join(self.co, "utils")
        os.makedirs(utils)
        os.makedirs(self.run)
        for name, body in STUBS.items():
            with open(os.path.join(utils, name), "w") as f:
                f.write(STUB_HEAD + body)
        shutil.copy(os.path.join(ROOT, "utils", "fe_protocol.py"), utils)
        os.symlink(os.path.join(ROOT, "settings"), os.path.join(self.co, "settings"))
        self.bin = os.path.join(base, "bin")
        os.makedirs(self.bin)
        with open(os.path.join(self.bin, "gmx"), "w") as f:
            f.write(FAKE_GMX)
        os.chmod(os.path.join(self.bin, "gmx"), 0o755)
        # job_fe.run reports its GPUs at start and end; not this host's real ones.
        with open(os.path.join(self.bin, "nvidia-smi"), "w") as f:
            f.write('#!/bin/sh\necho "0, stub GPU, 00000000:01:00.0, 1 MiB, 2 MiB, 0 %"\n')
        os.chmod(os.path.join(self.bin, "nvidia-smi"), 0o755)
        self.n = 0

    def job(self, runner, runs="B 2 1 amber19sb_opc3 1", plant=None, **fake):
        """One run of `runner` in a fresh structure directory -> Result. plant: files
        {name: text} put there first; fake: environment for the stubs and runner."""
        self.n += 1
        sid = "S%d" % self.n
        s = os.path.join(self.run, sid)
        os.makedirs(s)
        shutil.copy(os.path.join(ROOT, runner), os.path.join(s, "job.run"))
        os.chmod(os.path.join(s, "job.run"), 0o755)
        open(os.path.join(s, "run.gs"), "w").write(runs + "\n")
        open(os.path.join(s, "input.pdb"), "w").write(input_pdb())
        for name, text in (plant or {}).items():
            open(os.path.join(s, name), "w").write(text)
        ulog, glog = os.path.join(self.base, sid + ".util"), os.path.join(self.base, sid + ".gmx")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GROSCORE_", "FAKE_", "SLURM_"))}
        env.update(PATH=self.bin + ":" + os.environ["PATH"], UTILLOG=ulog, GMXLOG=glog,
                   GROSCORE_NT="1", PYTHONDONTWRITEBYTECODE="1")
        env.update({k: str(v) for k, v in fake.items()})
        args = ["bash", "./job.run"] + (["--setup"] if runner == "job_fe.run" else [])
        p = subprocess.run(args, cwd=s, env=env, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=600)
        calls = [json.loads(l) for l in open(ulog)] if os.path.isfile(ulog) else []
        res0 = os.path.join(self.run, "results_0.gs")
        status = [l.split()[1] for l in open(res0) if l.split()[:1] == [sid]] if os.path.isfile(res0) else []
        return Result(s, p, calls, status)


class Result(object):
    def __init__(self, d, p, calls, status):
        self.d, self.p, self.calls, self.status = d, p, calls, status
        self.out = p.stdout + p.stderr
        # The setup chain: every call before fix_ion_protonation.py, the first call
        # after the NCAA block in both runners.
        k = [i for i, c in enumerate(calls) if c["util"] == "fix_ion_protonation.py"]
        self.chain = calls[:k[0]] if k else calls
        k = [i for i, c in enumerate(self.chain) if c["util"] == "parametrize_ncaa.py"]
        self.after_pn = self.chain[k[0] + 1:] if k else []

    def names(self, calls=None):
        return [c["util"] for c in (self.chain if calls is None else calls)]

    def of(self, util, calls=None):
        return [c for c in (self.chain if calls is None else calls) if c["util"] == util]

    def file(self, name):
        p = os.path.join(self.d, name)
        return open(p).read() if os.path.isfile(p) else None

    def merge(self):
        m = [c for c in self.calls if c["util"] == "merge_ligand.py"]
        return m[0] if m else {}


def opt(call, flag):
    a = call["argv"]
    return a[a.index(flag) + 1] if flag in a else None


def keep(call):
    """The --keep-ncaa names of a fix_pdb.py call, as a set."""
    v = opt(call, "--keep-ncaa")
    return set(v.split(",")) if v else set()


def list_after(r):
    """The NCAA list the fallback ends with, as the runner prints it (a set), or None."""
    m = re.findall(r"NCAA residues after the fallback: (\S+)", r.out)
    return None if not m else (set() if m[-1] == "none" else set(m[-1].split(",")))


def fusion_before_every_renumber(r):
    """Every renumber_pdb.py call comes right after a fuse_glycans.py call."""
    n = r.names()
    idx = [i for i, u in enumerate(n) if u == "renumber_pdb.py"]
    return bool(idx) and all(i > 0 and n[i - 1] == "fuse_glycans.py" for i in idx)


def block(src, start, end):
    """The lines from the first matching `start` to the next matching `end`, inclusive."""
    lines = src.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if re.search(start, ln):
            for j in range(i + 1, len(lines)):
                if re.search(end, lines[j]):
                    return "".join(lines[i:j + 1])
            return None
    return None


ONE = "NG1 A 350 NAG A 544"
TWO = "NG1 A 350 NAG A 544;NG2 A 53 NAG A 701"
REASON = "AM1-BCC did not finish within 600 s"

with tempfile.TemporaryDirectory(prefix="glycan_runner_") as tmp:
    sb = Sandbox(tmp)
    for runner in RUNNERS:
        print("\n===== %s =====" % runner)

        print("[1] fused before every renumbering when ligands are parametrised, and only then")
        stale = {"ligand_NAG_A.pdb": "HETATM    1  C1  NAG A 544  stale\n",
                 "ligand_NAG_A_703.pdb": "HETATM    1  C1  NAG A 703  stale\n",
                 "ligand_info.gs": "NAG A 14\n"}
        r = sb.job(runner, plant=stale, FAKE_SITES=ONE)
        rn = r.of("renumber_pdb.py")
        fx = r.of("fix_pdb.py")
        check("%s on: fuse_glycans.py -f input.pdb -o fused.pdb -d glycans.json --unfused glycan_unfused.gs"
              % runner, [c["argv"] for c in r.of("fuse_glycans.py")]
              == [["-f", "input.pdb", "-o", "fused.pdb", "-d", "glycans.json", "--unfused", "glycan_unfused.gs"]],
              [c["argv"] for c in r.of("fuse_glycans.py")])
        check("  then renumber_pdb.py -f fused.pdb, once, right after the fusion",
              len(rn) == 1 and opt(rn[0], "-f") == "fused.pdb" and fusion_before_every_renumber(r), r.names())
        check("  fix_pdb.py keeps NG1 and MSE, no fallback",
              len(fx) == 1 and keep(fx[0]) == {"NG1", "MSE"} and not r.after_pn, (r.names(), [keep(c) for c in fx]))
        check("  and the NCAA list names NG1", re.search(r"NCAA residues detected: \S*NG1", r.out) is not None,
              r.out[-300:])
        m = r.merge()
        check("  merge_ligand.py finds NAG A 701 and SO4, no file of NAG A 544",
              m.get("ligands") == ["ligand_NAG_A.pdb", "ligand_SO4_A.pdb", "ligand_info.gs"]
              and "NAG A 701" in (r.file("ligand_NAG_A.pdb") or "")
              and "NAG A 544" not in (r.file("ligand_NAG_A.pdb") or ""), m.get("ligands"))
        print("[2] the ligand files of an earlier pass are gone before renumber_pdb.py starts")
        check("  %s: no ligand file at renumber_pdb.py's start (3 planted)" % runner,
              [c["at_entry"] for c in rn] == [[]], [c["at_entry"] for c in rn])
        for runs, mode in (("B 2 1 charmm36 1", "off-ff (CHARMM36)"), ("B 2 1 gromos54a8_rf 1", "off-ff (GROMOS)"),
                           ("B 2 1 amber19sb_opc3 0", "off-user")):
            r = sb.job(runner, runs=runs, plant=stale, FAKE_SITES=ONE)
            rn = r.of("renumber_pdb.py", r.calls)
            check("%s %s: no fusion, renumber_pdb.py -f input.pdb" % (runner, mode),
                  not r.of("fuse_glycans.py", r.calls) and len(rn) == 1 and opt(rn[0], "-f") == "input.pdb",
                  r.names(r.calls))
            check("  both NAGs extracted as ligands, the stale files gone first",
                  r.merge().get("ligands") == ["ligand_NAG_A_544.pdb", "ligand_NAG_A_701.pdb", "ligand_SO4_A.pdb",
                                               "ligand_info.gs"] and rn[0]["at_entry"] == [],
                  (r.merge().get("ligands"), rn[0]["at_entry"]))

        print("[3] a fused residue that fails goes back to ligands, in one ordered pass")
        for style, extra, what in (("plain", {}, "a line without the reason"), ("none", {}, "no line"),
                                   ("copy", {"FAKE_GLYREASON": "AM1-BCC  did not finish  within 600 s"},
                                    "the reason, spaces normalised"),
                                   ("none", {"FAKE_ICODE": "A"}, "no line; label NAG A 544A"),
                                   ("plain", {"FAKE_ICODE": "A"}, "a line; label NAG A 544A")):
            r = sb.job(runner, plant=stale, FAKE_SITES=TWO, FAKE_GLYFAIL="NG1", FAKE_UNFUSED=style, **extra)
            a = r.after_pn
            check("%s, NG1 failed (fusion writes %s): the ordered fallback" % (runner, what),
                  r.names(a) == ["fuse_glycans.py", "renumber_pdb.py", "fix_pdb.py", "cap_termini.py"]
                  and opt(a[0], "--exclude-file") == "glycan_failed.gs" and opt(a[1], "-f") == "fused.pdb",
                  [(c["util"], c["argv"][:2]) for c in a])
            if style == "plain" and not extra:
                check("  every renumbering right after a fusion, no ligand file at its start",
                      fusion_before_every_renumber(r)
                      and [c["at_entry"] for c in r.of("renumber_pdb.py")] == [[], []],
                      [c["at_entry"] for c in r.of("renumber_pdb.py")])
                fx = r.of("fix_pdb.py", a)
                check("  fix_pdb.py: NG1 neither in its input nor kept; NG2 and MSE kept",
                      fx and "NG1" not in fx[0]["resnames"] and keep(fx[0]) == {"NG2", "MSE"},
                      fx and (fx[0]["resnames"], keep(fx[0])))
                check("  the NCAA list after the fallback is NG2 and MSE, without NG1",
                      list_after(r) == {"NG2", "MSE"}, list_after(r))
                check("  NAG A 544 is a ligand again and parametrised as one",
                      r.merge().get("ligands") == ["ligand_NAG_A.pdb", "ligand_SO4_A.pdb", "ligand_info.gs"]
                      and "NAG A 544" in (r.file("ligand_NAG_A.pdb") or "")
                      and any(opt(c, "-f") == "ligand_NAG_A.pdb" for c in r.of("parametrize_ligand.py", r.calls)),
                      r.merge().get("ligands"))
                check("  the log says which fused residue went back",
                      "The fused glycans that failed to parametrise go back to ligands: NG1" in r.out, r.out[-300:])
            seen = (r.merge().get("unfused") or "").splitlines()
            line = [l for l in seen if l.split()[:3] == ["NAG", "A", "544"]]
            check("  merge_ligand.py reads one NAG A 544 line holding the reason once",
                  len(line) == 1 and " ".join(line[0].split()).count(REASON) == 1 and "NG1" in line[0]
                  and not [l for l in seen if l.split()[:3] == ["NAG", "A", "544A"]], seen)
        r = sb.job(runner, plant=stale, FAKE_SITES=TWO, FAKE_GLYFAIL="NG1", FAKE_NCAAFAIL="MSE")
        a = r.after_pn
        fx = r.of("fix_pdb.py", a)
        check("%s, NG1 and MSE failed together: one pass, fix_pdb.py keeps NG2 only" % runner,
              r.names(a) == ["fuse_glycans.py", "renumber_pdb.py", "fix_pdb.py", "cap_termini.py"]
              and len(r.of("renumber_pdb.py")) == 2 and fx and keep(fx[0]) == {"NG2"}
              and list_after(r) == {"NG2"}, (r.names(), fx and keep(fx[0]), list_after(r)))

        print("[4] a parametrize_ncaa.py that exits 1 fails every fused residue")
        r = sb.job(runner, FAKE_SITES=TWO, FAKE_PN_RC=1)
        a = r.after_pn
        fx = r.of("fix_pdb.py", a)
        gf = sorted((r.file("glycan_failed.gs") or "").splitlines())
        check("%s: glycan_failed.gs 'NG1/NG2 parametrize_ncaa.py exited 1'" % runner,
              gf == ["NG1 parametrize_ncaa.py exited 1", "NG2 parametrize_ncaa.py exited 1"], gf)
        check("  both excluded in one pass: fix_pdb.py keeps MSE only",
              r.names(a) == ["fuse_glycans.py", "renumber_pdb.py", "fix_pdb.py", "cap_termini.py"]
              and opt(a[0], "--exclude-file") == "glycan_failed.gs" and fx and keep(fx[0]) == {"MSE"}
              and list_after(r) == {"MSE"}, (r.names(a), fx and keep(fx[0]), list_after(r)))
        check("  both NAGs become ligands",
              r.merge().get("ligands") == ["ligand_NAG_A_544.pdb", "ligand_NAG_A_701.pdb", "ligand_SO4_A.pdb",
                                           "ligand_info.gs"], r.merge().get("ligands"))
        r = sb.job(runner, FAKE_SITES=TWO, FAKE_PN_RC=1, FAKE_BAD_JSON=1)
        gf = sorted((r.file("glycan_failed.gs") or "").splitlines())
        check("%s: the same with an unreadable glycans.json (names from fused.pdb)" % runner,
              gf == ["NG1 parametrize_ncaa.py exited 1", "NG2 parametrize_ncaa.py exited 1"]
              and r.names(r.after_pn) == ["fuse_glycans.py", "renumber_pdb.py", "fix_pdb.py", "cap_termini.py"]
              and list_after(r) == {"MSE"}, (gf, r.names(r.after_pn), list_after(r)))
        r = sb.job(runner, FAKE_SITES="", FAKE_PN_RC=1)
        check("%s: exit 1 without a fused residue: no fallback, no glycan_failed.gs" % runner,
              r.after_pn == [] and len(r.of("renumber_pdb.py")) == 1 and r.file("glycan_failed.gs") is None,
              r.names())

        print("[5] an NCAA failure alone keeps the glycans fused")
        r = sb.job(runner, FAKE_SITES=ONE, FAKE_NCAAFAIL="MSE")
        a = r.after_pn
        fx = r.of("fix_pdb.py", a)
        check("%s, MSE failed: no fusion again, renumber the first fused.pdb, fix_pdb.py keeps NG1 only" % runner,
              r.names(a) == ["renumber_pdb.py", "fix_pdb.py", "cap_termini.py"]
              and opt(a[0], "-f") == "fused.pdb" and a[0]["at_entry"] == []
              and fx and keep(fx[0]) == {"NG1"} and list_after(r) == {"NG1"},
              ([(c["util"], c["argv"]) for c in a[:2]], fx and keep(fx[0]), list_after(r)))
        check("  and NAG A 544 is no ligand", r.merge().get("ligands") == ["ligand_NAG_A.pdb", "ligand_SO4_A.pdb",
                                                                         "ligand_info.gs"], r.merge().get("ligands"))

        print("[6] a failed fusion stops the structure; the cap comes from the environment")
        r = sb.job(runner, FAKE_SITES=ONE, FAKE_FUSE_RC=1)
        check("%s, fuse_glycans.py exits 1: LIGAND_FAIL, said so, nothing renumbered" % runner,
              r.status[-1:] == ["LIGAND_FAIL"] and "fuse_glycans.py exited 1" in r.out
              and not r.of("renumber_pdb.py", r.calls) and r.p.returncode == 0, (r.status, r.out[-300:]))
        for cap in ("2", "1"):
            r = sb.job(runner, FAKE_SITES=ONE, GROSCORE_GLYCAN_MAX_SUGARS=cap)
            fz = r.of("fuse_glycans.py")
            check("%s, GROSCORE_GLYCAN_MAX_SUGARS=%s: --max-sugars %s, and the log says so" % (runner, cap, cap),
                  len(fz) == 1 and opt(fz[0], "--max-sugars") == cap
                  and ("Glycans of up to %s sugars" % cap) in r.out, ([c["argv"] for c in fz], r.out[-200:]))
        r = sb.job(runner, FAKE_SITES=ONE)
        check("%s, unset: no --max-sugars (fuse_glycans.py's own default)" % runner,
              [c for c in r.of("fuse_glycans.py") if "--max-sugars" in c["argv"]] == []
              and len(r.of("fuse_glycans.py")) == 1, [c["argv"] for c in r.of("fuse_glycans.py")])
        # fuse_glycans.py takes a whole number of at least 1; anything else is a setting
        # to fix, not a ligand that failed.
        for bad in ("0", "two"):
            r = sb.job(runner, FAKE_SITES=ONE, GROSCORE_GLYCAN_MAX_SUGARS=bad)
            check("%s, GROSCORE_GLYCAN_MAX_SUGARS=%s: FAILED before any fusion or renumbering" % (runner, bad),
                  r.status[-1:] == ["FAILED"] and ("GROSCORE_GLYCAN_MAX_SUGARS is '%s'" % bad) in r.out
                  and not r.of("fuse_glycans.py", r.calls) and not r.of("renumber_pdb.py", r.calls),
                  (r.status, r.out[-300:]))

        print("[7] a fresh setup starts without an earlier attempt's failure lists")
        r = sb.job(runner, plant={"glycan_failed.gs": "NG1 stale reason\n", "ncaa_failed.gs": "MSE\n"},
                   FAKE_SITES=ONE)
        fx = r.of("fix_pdb.py")
        check("%s: stale NG1 and MSE lines start no fallback; fix_pdb.py keeps both" % runner,
              r.after_pn == [] and len(r.of("renumber_pdb.py")) == 1 and len(fx) == 1
              and keep(fx[0]) == {"NG1", "MSE"}, (r.names(), [keep(c) for c in fx]))
        params = {"ligand_manifest.gs": "NAG\tligand_NAG.itp\tligand_NAG_atomtypes.itp\tligand_NAG_A.gro\t29\tA\t544\t"
                                        "ligand_NAG_A.pdb\n",
                  "ligand_NAG.itp": "x\n", "ligand_NAG_atomtypes.itp": "x\n", "ligand_NAG_A.gro": "x\n",
                  "ligand_NAG.sdf": "x\n", "ligand_failed.gs": "NAG A 544 an earlier attempt\n"}
        r = sb.job(runner, plant=params, FAKE_SITES=ONE)
        check("%s: an earlier attempt's ligand parametrisation outputs are gone when merge_ligand.py runs" % runner,
              r.merge().get("params") == [] and r.merge().get("ligands") is not None, r.merge().get("params"))
        r = sb.job(runner, FAKE_SITES=ONE, FAKE_PDB2GMX_FAIL=1)
        check("%s: pdb2gmx writes no conf.gro: FAILED at once, no ligand parametrised, no merge" % runner,
              r.status[-1:] == ["FAILED"] and "pdb2gmx produced no conf.gro" in r.out
              and not r.of("parametrize_ligand.py", r.calls) and not r.of("merge_ligand.py", r.calls),
              (r.status, [c["util"] for c in r.calls][-4:], r.out[-200:]))

        src = open(os.path.join(ROOT, runner)).read()
        if runner == "job.run":
            clean = block(src, r"^if \[\[ -f conf\.gro && ! -f emin_solv\.gro \]\]; then$", r"^fi$")
        else:
            clean = block(src, r'^if \[\[ "\$MODE" != "cycle" && -f conf\.gro && ! -f emin_solv\.gro \]\] \\$',
                          r"^fi$")
        gone = ["fused.pdb", "glycans.json", "glycan_unfused.gs", "glycan_failed.gs", "ligand_NAG_A.pdb",
                "ligand_NAG_A_703.pdb", "ligand_info.gs", "ncaa_failed.gs"]
        if clean:
            d = os.path.join(tmp, "clean_" + runner)
            os.makedirs(d)
            for f in gone + ["conf.gro", "input.pdb", "run.gs"]:
                open(os.path.join(d, f), "w").write("x\n")
            subprocess.run(["bash", "-c", "MODE=setup\n" + clean], cwd=d, capture_output=True, text=True)
            left = sorted(os.listdir(d))
        check("%s: the restart cleanup removes the fusion's files and every ligand_*.pdb" % runner,
              clean is not None and left == ["input.pdb", "run.gs"], left if clean else "no cleanup block")

    print("\n[8] one text for the shared setup chain, one renumbering per runner")
    srcs = {r: open(os.path.join(ROOT, r)).read() for r in RUNNERS}
    blocks = {r: block(s, r"^# ---- GLYCANS AND NCAAs IN THE SETUP CHAIN",
                       r"^# ---- end of the setup chain both runners share ----$") for r, s in srcs.items()}
    check("both runners carry the block, with fuse_and_renumber, ncaa_names, ncaa_fallback",
          all(b and all("\n%s() {\n" % f in b for f in ("fuse_and_renumber", "ncaa_names", "ncaa_fallback"))
              for b in blocks.values()), {r: bool(b) for r, b in blocks.items()})
    check("  and its text is identical in job.run and job_fe.run",
          blocks["job.run"] is not None and blocks["job.run"] == blocks["job_fe.run"])
    # The call sites: the start of the setup block, the NCAA list, the fallback.
    starts = {r: block(s, r"^  # What an earlier attempt left of the fusion", r"^  fuse_and_renumber$")
              for r, s in srcs.items()}
    check("the setup block starts with the same lines in both (rm of the old lists, then the fusion)",
          starts["job.run"] is not None and starts["job.run"] == starts["job_fe.run"]
          and "rm -f ncaa_failed.gs glycan_failed.gs fused.pdb glycans.json glycan_unfused.gs\n" in starts["job.run"],
          starts["job.run"])
    sites = {r: [l.strip() for l in s.splitlines() if l.strip() in ("NCAA_LIST=$(ncaa_names renumbered.pdb)",
                                                                   "ncaa_fallback $?")] for r, s in srcs.items()}
    check("  each runner lists the NCAAs with ncaa_names and calls ncaa_fallback once",
          sites["job.run"] == sites["job_fe.run"] == ["NCAA_LIST=$(ncaa_names renumbered.pdb)",
                                                       "NCAA_LIST=$(ncaa_names renumbered.pdb)", "ncaa_fallback $?"],
          sites)
    for runner, s in srcs.items():
        lines = s.splitlines()
        ren = [i for i, l in enumerate(lines) if "utils/renumber_pdb.py" in l and not l.lstrip().startswith("#")]
        check("%s: one renumber_pdb.py call, right after 'rm -f ligand_*.pdb ligand_info.gs'" % runner,
              len(ren) == 1 and lines[ren[0] - 1].strip() == "rm -f ligand_*.pdb ligand_info.gs",
              [lines[i - 1:i + 1] for i in ren])
        fz = [l for l in lines if "utils/fuse_glycans.py" in l and not l.lstrip().startswith("#")]
        check("  one fuse_glycans.py call", len(fz) == 1, fz)
        check("  no grep of renumbered.pdb.log", re.search(r"grep[^\n]*renumbered\.pdb\.log", s) is None)
        p = subprocess.run(["bash", "-n", os.path.join(ROOT, runner)], capture_output=True, text=True)
        check("  bash -n %s" % runner, p.returncode == 0, p.stderr[:200])

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all glycan runner checks passed")
