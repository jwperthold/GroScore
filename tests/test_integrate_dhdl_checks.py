#!/usr/bin/env python3
"""integrate_dhdl.py integrates only a whole record, and job_fe.run says why not.

On 2026-10-13 a full /gpfs (bm_fe_amber_3a) dropped 8 KiB stdio buffers from dhdl
files while mdrun carried on, and what was left still parsed: 1I4D bindrevH_17 began
with a one-number fragment, 1IBR bindrevB_12 lost 780 ps mid-file around an orphan
number, and 1OPH bindrevI_19 carried the fragment '15 -299.31186' between rows near
2000 ps, which integrated to 169.30 instead of about 190. The integrator skipped
unparsable lines, cut rows to the narrowest one and had no time check.

  [1] A whole record integrates as before: the value, and the note for a leg that
      ends short of its last step (nsteps not a multiple of nstdhdl).
  [2] Each damage pattern of 2026-10-13, and its variants on a line boundary, prints
      NaN, exits 1 and names the line: a fragment at the head, an orphan mid-file,
      a stray row that goes back in time, a gap, a missing head or tail (with
      --leg-ps), a record past the end of the leg, an unparsable line, a repeated
      time. Without --leg-ps the interior checks still hold.
  [2b] The last row, which a disk that fills at the end of a leg takes (mdrun
      flushes at its final checkpoint, before it prints that row): with --last-ps
      (job_fe.run passes it) a record without it is refused, and one that ends
      there passes also when nsteps is not a multiple of nstdhdl; a last line cut
      without its line end, and a value that is not finite, are refused; an old
      _Wdhdl.dat is not left beside a refused record.
  [3] Times written with four decimals (a step of 1/3 ps) pass; record_problem is
      importable, as main() runs only under __main__.
  [4] job_fe.run's integrate_or_nan passes a failing integrator's stderr into the
      job log, with the cycle, and keeps a succeeding one's out of it.

Standalone, no pytest: python3 tests/test_integrate_dhdl_checks.py
"""
import importlib.util, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "utils", "integrate_dhdl.py")
JOB = os.path.join(ROOT, "job_fe.run")
fails = 0


def check(name, ok, detail=""):
    global fails
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:300]))
    if not ok:
        fails += 1


HEAD = ('# This file was created by a test\n# gmx mdrun is part of G R O M A C S:\n'
        '@    title "dH/d\\xl\\f{}"\n@    xaxis  label "Time (ps)"\n@TYPE xy\n'
        '@ subtitle "T = 310 (K) "\n')


def rows(t_from, t_to, step=2.0, leg=100.0):
    """Rows of dH/dlambda = 10 + 20 lambda, lambda = t / leg (a forward 0 -> 1 ramp)."""
    out, t = [], t_from
    while t <= t_to + 1e-9:
        out.append("%.4f %.6f" % (t, 10.0 + 20.0 * t / leg))
        t += step
    return out


def run(tmp, name, lines, leg_ps=100.0, head=True, last_ps=None, eol=True):
    path = os.path.join(tmp, name + ".xvg")
    with open(path, "w") as f:
        f.write((HEAD if head else "") + "\n".join(lines) + ("\n" if lines and eol else ""))
    argv = [sys.executable, TOOL, "-f", path, "--direction", "fwd"]
    if leg_ps:
        argv += ["--leg-ps=%g" % leg_ps]
    if last_ps is not None:
        argv += ["--last-ps=%g" % last_ps]
    r = subprocess.run(argv, cwd=tmp, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def refused(r, *needles):
    rc, out, err = r
    return rc == 1 and out == "NaN" and all(n in err for n in needles)


with tempfile.TemporaryDirectory() as tmp:
    print("\n[1] a whole record")
    r = run(tmp, "whole", rows(0, 100))
    check("integrates to 20 (dH/dl = 10 + 20 lambda over 0 -> 1), exit 0",
          r[0] == 0 and abs(float(r[1]) - 20.0) < 1e-9 and r[2] == "", r)
    r = run(tmp, "short", rows(0, 100), leg_ps=101.0)
    check("a leg 1 ps past its last row (step 2 ps): accepted, with the 'stops at lambda' note",
          r[0] == 0 and "stops at lambda" in r[2] and "no work is computed" not in r[2], r)
    # lambda = t/101 there, so dH/dl = 10 + 20.2 lambda, integrated up to 100/101
    check("  and integrated over the recorded span only",
          abs(float(r[1]) - 20.0 * 100 / 101.0) < 1e-9, r)

    print("\n[2] damaged records")
    body = rows(0, 100)
    # 1I4D bindrevH_17: the head never written, the file starts mid-row
    r = run(tmp, "frag_head", ["-68.060959"] + rows(40, 100), head=False)
    check("a one-number fragment at the head (1I4D bindrevH_17)",
          refused(r, "line 1 has 1 field where the other rows have 2"), r)
    r = run(tmp, "lost_head", rows(40, 100))
    check("the head lost on a line boundary (with --leg-ps)",
          refused(r, "starts at 40 ps, not at 0: its head is missing"), r)
    # 1IBR bindrevB_12: a buffer lost mid-file, one orphan number left
    lines = rows(0, 30) + [" -72.081558"] + rows(52, 100)
    r = run(tmp, "orphan", lines)
    n_orphan = len(HEAD.splitlines()) + len(rows(0, 30)) + 1
    check("an orphan number mid-file (1IBR bindrevB_12) names its line",
          refused(r, "line %d has 1 field" % n_orphan, "broken or spliced row"), r)
    r = run(tmp, "gap", rows(0, 30) + rows(52, 100))
    check("rows lost on a line boundary: the gap, with the rows missing",
          refused(r, "time goes from 30 to 52 ps", "record step is 2 ps",
                  "a gap of about 10 rows"), r)
    r = run(tmp, "gap_noleg", rows(0, 30) + rows(52, 100), leg_ps=None)
    check("  also without --leg-ps", refused(r, "time goes from 30 to 52 ps"), r)
    r = run(tmp, "gap1", rows(0, 30) + rows(34, 100))
    check("  one row missing: 'a gap of about 1 row'", refused(r, "a gap of about 1 row)"), r)
    # 1OPH bindrevI_19: a fragment that reads as a row at t = 15 ps
    lines = rows(0, 60) + ["15 -299.31186"] + rows(62, 100)
    r = run(tmp, "stray", lines)
    check("a stray row back in time (1OPH bindrevI_19)",
          refused(r, "time 15 ps does not follow 60 ps"), r)
    r = run(tmp, "dup", rows(0, 60) + rows(60, 100))
    check("a repeated time", refused(r, "time 60 ps does not follow 60 ps"), r)
    r = run(tmp, "tail", rows(0, 80))
    check("the tail lost on a line boundary (with --leg-ps)",
          refused(r, "ends at 80 ps, more than one record step (2 ps) before the end "
                     "of the leg at 100 ps: its tail is missing"), r)
    r = run(tmp, "tail_noleg", rows(0, 80), leg_ps=None)
    check("  without --leg-ps a shorter record is not judged (no leg length)",
          r[0] == 0, r)
    r = run(tmp, "past", rows(0, 104))
    check("a record past the end of the leg", refused(r, "after the end of the leg at 100 ps"), r)
    r = run(tmp, "junk", rows(0, 40) + ["42.0000 -4x.1"] + rows(44, 100))
    check("an unparsable line", refused(r, "is not a row of numbers: '42.0000 -4x.1'"), r)
    r = run(tmp, "wide", rows(0, 40) + ["42.0000 1.0 2.0"] + rows(44, 100))
    check("a row of three fields among rows of two", refused(r, "has 3 fields where the other rows have 2"), r)
    r = run(tmp, "empty", [])
    check("an empty record keeps its old message",
          r[0] == 1 and r[1] == "NaN" and "not enough data in" in r[2], r)

    print("\n[2b] the last row")
    r = run(tmp, "last_ok", rows(0, 100), last_ps=100.0)
    check("--last-ps: a record that ends there passes", r[0] == 0 and abs(float(r[1]) - 20.0) < 1e-9, r)
    r = run(tmp, "last_short_leg", rows(0, 100), leg_ps=101.0, last_ps=100.0)
    check("  also for a leg 1 ps longer (nsteps not a multiple of nstdhdl)", r[0] == 0, r)
    r = run(tmp, "last_lost", rows(0, 98), last_ps=100.0)
    check("--last-ps: the last row lost on a line boundary is refused",
          refused(r, "ends at 98 ps, before its last row at 100 ps: its tail is missing"), r)
    r = run(tmp, "last_lost_noflag", rows(0, 98))
    check("  (without --last-ps one lost row cannot be told from a short leg: accepted)",
          r[0] == 0, r)
    r = run(tmp, "last_past", rows(0, 100), leg_ps=101.0, last_ps=98.0)
    check("--last-ps: a record past its last row is refused",
          refused(r, "ends at 100 ps, after its last row at 98 ps"), r)
    lines = rows(0, 100)
    lines[-1] = lines[-1][:-4]
    r = run(tmp, "cut", lines, eol=False)
    n_last = len(HEAD.splitlines()) + len(lines)
    check("a last line cut without its line end is refused",
          refused(r, "line %d, the last, has no line end: the file was cut there" % n_last), r)
    lines = rows(0, 100)
    lines[10] = "20.0000 inf"
    r = run(tmp, "inf", lines)
    check("an infinite value is refused",
          refused(r, "line %d holds a value that is not finite" % (len(HEAD.splitlines()) + 11)), r)
    lines = rows(0, 100)
    lines[10] = "nan 14.0"
    r = run(tmp, "nantime", lines)
    check("a nan time is refused", refused(r, "holds a value that is not finite"), r)
    stale = os.path.join(tmp, "stale_Wdhdl.dat")
    with open(stale, "w") as f:
        f.write("0.5\t9.9\n")
    r = run(tmp, "stale", rows(0, 30) + rows(52, 100))
    check("a refused record leaves no _Wdhdl.dat of an earlier integration",
          refused(r, "time goes from 30 to 52 ps") and not os.path.exists(stale), r)

    print("\n[3] rounding and import")
    third = ["%.4f %.6f" % (k / 3.0, 1.0) for k in range(0, 301)]
    r = run(tmp, "third", third, leg_ps=100.0)
    check("a step of 1/3 ps written with four decimals passes", r[0] == 0, r)
    spec = importlib.util.spec_from_file_location("integrate_dhdl", TOOL)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["integrate_dhdl.py"]
    try:
        spec.loader.exec_module(mod)
        ok = True
    except SystemExit:
        ok = False
    sys.argv = argv
    check("importing the module parses no argv and runs nothing", ok)
    if ok:
        good = [(i + 1, [2.0 * i, 1.0]) for i in range(51)]
        check("record_problem(whole rows) is None", mod.record_problem(good, None, 100.0) is None)
        check("record_problem names a gap", "a gap of about 3 rows"
              in (mod.record_problem(good[:10] + good[13:], None) or ""))

    print("\n[4] job_fe.run's integrate_or_nan")
    src = open(JOB).read()
    m = re.search(r"^      integrate_or_nan\(\) \{\n.*?^      \}\n", src, re.S | re.M)
    check("integrate_or_nan is extractable", m is not None)
    if m:
        fake = os.path.join(tmp, "fake_tool.py")
        with open(fake, "w") as f:
            f.write("import sys\nif sys.argv[1] == 'bad':\n"
                    "    sys.stderr.write('integrate_dhdl: x.xvg: line 7 has 1 field\\n')\n"
                    "    print('NaN'); sys.exit(1)\n"
                    "sys.stderr.write('a note on success\\n'); print('2.5')\n")
        script = (m.group(0) + "cycle=3\n"
                  "v=$(integrate_or_nan %s %s bad -f x.xvg); echo \"BAD=$v\"\n"
                  "v=$(integrate_or_nan %s %s good); echo \"GOOD=$v\"\n"
                  % (sys.executable, fake, sys.executable, fake))
        tdir = os.path.join(tmp, "tmpdir")
        os.makedirs(tdir)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp,
                           env=dict(os.environ, TMPDIR=tdir))
        check("a failure gives NaN", "BAD=NaN" in r.stdout, r.stdout)
        check("  and its stderr reaches the job's stderr, with the cycle and the arguments",
              "fake_tool.py failed for cycle 3 (exit 1): bad -f x.xvg" in r.stderr
              and "    integrate_dhdl: x.xvg: line 7 has 1 field" in r.stderr, r.stderr)
        check("a success gives its value and leaves stderr empty",
              "GOOD=2.5" in r.stdout and "a note on success" not in r.stderr, (r.stdout, r.stderr))
        check("no temporary file is left behind", os.listdir(tdir) == [], os.listdir(tdir))

print()
sys.exit(1 if fails else 0)
