#!/usr/bin/env python3
"""capri/prepare_round.py splits a live CAPRI scoring set into GroScore structures.

On a synthetic scoring set of four models (REMARK 9 MD5 lines; model 2 lacks
chain B):
  1. every complete model gets <T>_<NNNN>/input.pdb with its records unchanged,
     a TER after each chain and END; model 2 is skipped and named; sp.gs lists
     the written models with protein B; md5_map.gs maps them to their MD5; exit 1
     because a model was skipped.
  2. the same file gzipped and cut inside model 3 (as Target383.pdb.gz of round
     66 was) gives models 0 and 1, says TRUNCATED and exits 1; no partial
     input.pdb of model 3 is left.
  3. an existing run directory is refused without --resume; with --resume and
     the whole file, only model 3 is written, model 0's input.pdb is not
     rewritten, and sp.gs / md5_map.gs list 0, 1, 3.

Run:  python3 tests/test_capri_round.py
"""
import gzip, os, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP = os.path.join(REPO, "capri", "prepare_round.py")
failures = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def atom(serial, name, resname, chain, resnum, x):
    return "ATOM  %5d  %-3s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s" % (
        serial, name, resname, chain, resnum, x, 0.0, 0.0, name[0])


def model_lines(chains):
    out, s = [], 1
    for ch in chains:
        for r in (1, 2):
            for nm in ("N", "CA", "C", "O"):
                out.append(atom(s, nm, "GLY", ch, r, float(s)))
                s += 1
        out.append("TER")
    return out


MD5 = {m: ("%x" % (m + 1)) * 32 for m in range(4)}
lines = ["HEADER    CAPRI ROUND 00 TARGET 999"]
lines += ["REMARK   9 MODEL %8d MD5 %s" % (m, MD5[m]) for m in range(4)]
for m in range(4):
    lines += ["MODEL %8d" % m] + model_lines("A" if m == 2 else "AB") + ["ENDMDL"]
lines += ["END"]
text = "\n".join(lines) + "\n"


def run(*args):
    p = subprocess.run([sys.executable, "-I", PREP] + list(args), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def listed(path):
    return [l.split() for l in open(path) if not l.startswith("#")]


with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, "Target999.pdb")
    open(src, "w").write(text)

    # 1. whole file
    out1 = os.path.join(tmp, "full")
    rc, log = run(src, "-c", "B", "-o", out1)
    check(rc == 1 and "SKIPPED T999_0002: no chain B" in log, "1: model without chain B skipped, exit 1")
    want = "\n".join(model_lines("AB")[:-1] + ["TER", "END"]) + "\n"
    got = open(os.path.join(out1, "T999_0000", "input.pdb")).read()
    check(got == want, "1: input.pdb holds the model's records, TER per chain, END")
    check(listed(os.path.join(out1, "sp.gs")) == [["T999_0000", "B"], ["T999_0001", "B"], ["T999_0003", "B"]],
          "1: sp.gs lists the written models with protein B")
    check(listed(os.path.join(out1, "md5_map.gs"))[2] == ["T999_0003", "3", MD5[3]], "1: md5_map.gs maps model 3")

    # 2. gzip cut inside model 3
    cut = os.path.join(tmp, "Target999.pdb.gz")
    raw = text.encode()
    # cut the compressed stream so that it ends inside model 3: compress a prefix that
    # stops in model 3 and drop the gzip trailer, as an aborted upload would
    prefix = raw[:raw.index(b"\nMODEL        3") + 200]
    open(cut, "wb").write(gzip.compress(prefix)[:-8])
    out2 = os.path.join(tmp, "cut")
    rc, log = run(cut, "-c", "B", "-o", out2)
    dirs = sorted(d for d in os.listdir(out2) if d.startswith("T999_"))
    check(rc == 1 and "TRUNCATED" in log and "model 3 is incomplete" in log, "2: truncation reported, exit 1")
    check(dirs == ["T999_0000", "T999_0001"], "2: only the complete models written")
    check("(truncated)" in open(os.path.join(out2, "md5_map.gs")).readline(), "2: md5_map.gs header says truncated")

    # 3. resume with the whole (gzipped) file
    open(cut, "wb").write(gzip.compress(text.encode()))
    rc, log = run(cut, "-c", "B", "-o", out2)
    check(rc != 0 and "pass --resume" in log, "3: existing run directory refused without --resume")
    m0 = os.path.join(out2, "T999_0000", "input.pdb")
    open(m0, "a").write("REMARK marker\n")
    rc, log = run(cut, "-c", "B", "-o", out2, "--resume")
    check("written 1, already present 2" in log, "3: --resume writes only the missing model")
    check(open(m0).read().endswith("REMARK marker\n"), "3: an existing input.pdb is not rewritten")
    check([r[0] for r in listed(os.path.join(out2, "sp.gs"))] == ["T999_0000", "T999_0001", "T999_0003"],
          "3: sp.gs lists every model directory present")

print("\n%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
