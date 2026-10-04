#!/usr/bin/env python3
"""A GROMOS 54A8 residue name of four characters reaches pdb2gmx.

parametrize_ncaa.py --gmx-ff gromos54a8 maps the CCD's D-alanine DAL to the GROMOS
residue DALA, and both runners renamed it in renumbered.pdb with '%-3s' into columns
18-20 followed by line[20:], so every later column moved one to the right: the chain
landed in column 23, fix_pdb.py stopped on 'could not convert string to float' and the
structure ended as FAILED 'pdb2gmx produced no conf.gro' (ncaa review F2; every GROMOS
54A8 structure with a D-alanine). A 4-character name belongs in columns 18-21 (column
21 is the fourth residue-name character GROMACS and OpenMM read), and since OpenMM
writes at most 3 characters of a residue name, fix_pdb.py and cap_termini.py hand back
DAL, which pdb2gmx calls 'Other'; the H-stripping step before pdb2gmx puts DALA back.
Pinned on a 1GRM segment (tests/data/gromos, PDB data CC0) with ALA 1 renamed DAL:

  [1] job.run and job_fe.run carry the same rename block and the same two
      H-stripping blocks (full model and cutout);
  [2] parametrize_ncaa.py --gmx-ff gromos54a8 maps DAL to DALA (ncaa_rename.gs) and
      keeps it (DLE and DVA, which the force field lacks, go to their parents); the
      rename writes 'DALA' into columns 18-21, the chain stays in column 22 and every
      coordinate in its columns;
  [3] fix_pdb.py --keep-ncaa and cap_termini.py --ace-only --rename-ace-carbon run;
      the full-model H-stripping block writes DALA back into columns 18-21, and
      pdb2gmx -ff gromos54a8 builds the chain with a DALA residue; the cutout block
      does the same on a file cap_termini.py wrote.

Run:  python3 tests/test_gromos_rename.py   (about 15 s)
"""
import os, re, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(REPO, "utils")
DATA = os.path.join(REPO, "tests", "data", "gromos", "1GRM_segment.pdb")
failures = []


def check(name, ok, detail=""):
    print("  %-72s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:400]))
    if not ok:
        failures.append(name)


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


SRC = {r: open(os.path.join(REPO, r)).read() for r in ("job.run", "job_fe.run")}
RENAME = {r: block(s, r"^      python3 - << 'RENAME_PY'$", r"^RENAME_PY$") for r, s in SRC.items()}
FULL = {r: block(s, r"^  # Strip H from NCAA residues", r"^  fi$") for r, s in SRC.items()}
CUT = {r: block(s, r"^      # Strip H from NCAA residues in cutout", r"^      fi$") for r, s in SRC.items()}
print("[1] one text in both runners")
check("the rename block", RENAME["job.run"] is not None and RENAME["job.run"] == RENAME["job_fe.run"])
check("the H-stripping blocks, full model and cutout", FULL["job.run"] is not None and CUT["job.run"] is not None
      and FULL["job.run"] == FULL["job_fe.run"] and CUT["job.run"] == CUT["job_fe.run"])


def run(cmd, cwd, env=None, stdin=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, input=stdin, capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def res_cols(path, num):
    """(columns 18-21, column 22, columns 23-26) of the first ATOM record of residue `num`."""
    for l in (open(path) if os.path.isfile(path) else ()):
        if l.startswith("ATOM") and l[22:26].strip() == str(num):
            return l[17:21], l[21], l[22:26]
    return None


tmp = tempfile.mkdtemp(prefix="gromos_rename_")
try:
    d = os.path.join(tmp, "s")
    os.makedirs(d)
    seg = [l for l in open(DATA) if l.startswith(("ATOM", "END"))]
    seg = [(l[:17] + "DAL" + l[20:]) if l.startswith("ATOM") and l[22:26].strip() == "1" else l for l in seg]
    open(os.path.join(d, "renumbered.pdb"), "w").write("".join(seg))
    open(os.path.join(d, "chain_map.gs"), "w").write("# Protein B chain(s): B\n")
    print("\n[2] the rename")
    gmx_env = dict(os.environ, GMXLIB=os.path.join(REPO, "forcefield"))     # as the runners set it for GROMOS
    rc, log = run([sys.executable, os.path.join(UTILS, "parametrize_ncaa.py"), "-f", "renumbered.pdb",
                   "--gmx-ff", "gromos54a8"], d, env=gmx_env)
    rename = open(os.path.join(d, "ncaa_rename.gs")).read() if os.path.isfile(os.path.join(d, "ncaa_rename.gs")) else ""
    keep = [l.strip() for l in open(os.path.join(d, "ncaa_residues.gs"))] if os.path.isfile(
        os.path.join(d, "ncaa_residues.gs")) else []
    check("parametrize_ncaa.py maps DAL to DALA, keeps DALA (DLE, DVA go to their parents)",
          rc == 0 and rename.split("\t")[:2] == ["DAL", "DALA"] and keep == ["DALA"], (rc, rename, keep, log[-300:]))
    before = open(os.path.join(d, "renumbered.pdb")).read().splitlines()
    rc, log = run(["bash", "-c", RENAME["job.run"].strip()], d)
    after = open(os.path.join(d, "renumbered.pdb")).read().splitlines()
    check("the rename: 'DALA' in columns 18-21, chain A in 22, residue 1 in 23-26",
          rc == 0 and res_cols(os.path.join(d, "renumbered.pdb"), 1) == ("DALA", "A", "   1"),
          (rc, res_cols(os.path.join(d, "renumbered.pdb"), 1), log[-200:]))
    check("  columns 1-12 and 22 on of every record as they were (chain, number, coordinates)",
          len(before) == len(after) and all(a[:12] + a[21:] == b[:12] + b[21:] for a, b in zip(after, before)),
          [a for a, b in zip(after, before) if a[:12] + a[21:] != b[:12] + b[21:]][:2])
    print("\n[3] fix_pdb.py, cap_termini.py, the H-stripping block, pdb2gmx")
    rc, log = run([sys.executable, os.path.join(UTILS, "fix_pdb.py"), "-f", "renumbered.pdb", "-o", "fixed.pdb",
                   "--keep-ncaa", ",".join(keep)], d)
    check("fix_pdb.py --keep-ncaa DALA runs", rc == 0 and os.path.isfile(os.path.join(d, "fixed.pdb")),
          (rc, log[-300:]))
    rc, log = run([sys.executable, os.path.join(UTILS, "cap_termini.py"), "-f", "fixed.pdb", "-o", "fixed_capped.pdb",
                   "-m", "chain_map.gs", "--ace-only", "--rename-ace-carbon"], d)
    check("cap_termini.py --ace-only --rename-ace-carbon runs",
          rc == 0 and os.path.isfile(os.path.join(d, "fixed_capped.pdb")), (rc, log[-300:]))
    capped = os.path.join(d, "fixed_capped.pdb")
    if os.path.isfile(capped):
        text = open(capped).read()
        open(capped, "w").write("".join(l for l in text.splitlines(keepends=True) if " OXT " not in l))
        shutil.copy(capped, os.path.join(d, "cutout_capped.pdb"))
    env = dict(os.environ, NCAA_FLAG="--keep-ncaa " + ",".join(keep), NCAA_LIST=",".join(keep),
               PDB2GMX_INPUT="fixed_capped.pdb", CUTOUT_PDB="cutout_capped.pdb")
    rc, log = run(["bash", "-c", FULL["job.run"]], d, env=env)
    got = res_cols(capped, 2) if os.path.isfile(capped) else None   # ACE 1 comes first
    check("the full-model block puts DALA back into columns 18-21 (after ACE 1)", rc == 0 and got == ("DALA", "A", "   2"),
          (rc, got, log[-200:]))
    rc, log = run(["gmx", "pdb2gmx", "-f", "fixed_capped.pdb", "-o", "conf.gro", "-p", "topol.top", "-water", "spc",
                   "-ff", "gromos54a8", "-ignh", "-missing", "-ter"], d, env=gmx_env, stdin="8\n1\n")
    gro = open(os.path.join(d, "conf.gro")).read().splitlines()[2:-1] if rc == 0 else []
    check("pdb2gmx -ff gromos54a8 builds the chain with a DALA residue",
          rc == 0 and any(l[5:10].strip() == "DALA" for l in gro), (rc, log[-400:]))
    rc, log = run(["bash", "-c", CUT["job.run"]], d, env=env)
    got = res_cols(os.path.join(d, "cutout_capped.pdb"), 2)
    check("the cutout block does the same", rc == 0 and got == ("DALA", "A", "   2"), (rc, got, log[-200:]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all GROMOS rename checks passed")
