#!/usr/bin/env python3
"""integrate.py prints a work only for a finished leg; anything else is NaN.

Through v1.4 integrate.py integrated whatever rows parsed. A missing, empty,
header-only or one-row force file printed -0.0 with exit 0, a leg that died part
way printed the work of the part that ran, and a row cut inside the integrated
columns raised IndexError with empty stdout, which job.run wrote as an empty field
and groscore.py then read the rebinding RMSD as the push work (v15_work
digest_audit: runner R2, scorer SCORER-1 and SCORER-4, shared SH7; a real mdrun
killed 4-8 s into a 1IJK pull leg left one row and printed -0.0).

Pinned here, on files written in GROMACS's own layout (tab-separated rows, one
'@ sN legend' line per pull coordinate):
  [1] a finished leg integrates exactly as before: 3 interface columns at
      -10 kJ/mol/nm for 5000 ps at 0.0002 nm/ps is 30.0, a linear ramp 45.0, and
      the elastic-network columns after the first -nr are never summed.
  [2] missing, empty, header-only and one-row files are NaN with a non-zero exit;
      two rows are the minimum that integrates.
  [3] the row the writer was cut in is dropped, never integrated and never fatal:
      cut inside the integrated columns, inside the network tail, or complete but
      without its newline. The rows before it integrate (15.0 for 2500 ps).
  [4] --tend: the last row must reach the leg's end to within one output
      interval (2 ps here): just below, at and just above the threshold, and a leg
      that stopped at 3000 of 5000 ps.
  [5] a file from another restraint set is refused: -nr beyond the columns, -nr 0,
      --ncols against the legends, a short row before the last.
  [6] time running backwards and a non-finite force are refused.
Every refusal prints exactly NaN on stdout, so a caller that reads only stdout
still writes a field no score accepts.

Standalone, no pytest: python3 tests/test_integrate_completeness.py
"""
import os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEGRATE = os.path.join(ROOT, "utils", "integrate.py")
failures = []


def check(name, ok, detail=""):
    print("  %-70s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:200]))
    if not ok:
        failures.append(name)


def xvg(path, rows, ncoord, tail=None, legends=True):
    """A pull-force file as GROMACS writes it. rows: [(t, [forces])]; tail: raw text
    appended after the last complete row (the cut row), no newline added."""
    with open(path, "w") as f:
        f.write("# This file was created by a test\n#\n")
        f.write('@    title "Pull force"\n@    xaxis  label "Time (ps)"\n@TYPE xy\n')
        f.write("@ legend on\n@ legend box on\n")
        if legends:
            for k in range(ncoord):
                f.write('@ s%d legend "%d"\n' % (k, k + 1))
        for t, fs in rows:
            f.write("%.4f" % t + "".join("\t%g" % x for x in fs) + "\n")
        if tail is not None:
            f.write(tail)
    return path


def run(path, *extra):
    r = subprocess.run([sys.executable, INTEGRATE, "-f", path] + list(extra),
                       capture_output=True, text=True, cwd=os.path.dirname(path))
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def is_work(res, want):
    rc, out, _ = res
    try:
        return rc == 0 and abs(float(out) - want) < 1e-9
    except ValueError:
        return False


def is_nan(res):
    rc, out, _ = res
    return rc != 0 and out == "NaN"


# 3 interface columns at -10 and 2 elastic-network columns at +7, every 2 ps.
FULL = [(2.0 * k, [-10.0, -10.0, -10.0, 7.0, 7.0]) for k in range(2501)]          # 0..5000 ps
HALF = FULL[:1251]                                                                # 0..2500 ps

with tempfile.TemporaryDirectory() as d:
    p = lambda name: os.path.join(d, name)

    print("\n[1] a finished leg integrates exactly as before")
    f_full = xvg(p("full.xvg"), FULL, 5)
    res = run(f_full, "-nr", "3")
    check("3 columns at -10 kJ/mol/nm, 5000 ps, rate 0.0002 -> 30.0", is_work(res, 30.0), res)
    check("  the same with --tend 5000 and --ncols 5",
          is_work(run(f_full, "-nr", "3", "--tend", "5000", "--ncols", "5"), 30.0))
    check("  -nr 5 sums the network columns too (-0.0002*(-16)*5000 = 16.0)",
          is_work(run(f_full, "-nr", "5"), 16.0))
    ramp = [(2.0 * k, [-10.0 - 0.002 * 2.0 * k] * 3 + [7.0, 7.0]) for k in range(2501)]
    check("a linear ramp -10 - 0.002 t per column integrates to 45.0 (trapezoid exact)",
          is_work(run(xvg(p("ramp.xvg"), ramp, 5), "-nr", "3"), 45.0))
    check("  and the rate scales it: -r 0.0001 -> 22.5",
          is_work(run(p("ramp.xvg"), "-nr", "3", "-r", "0.0001"), 22.5))

    print("\n[2] nothing short of two complete rows is a work")
    check("a missing file -> NaN (v1.4: -0.0, exit 0)", is_nan(run(p("nope.xvg"), "-nr", "3")))
    open(p("empty.xvg"), "w").close()
    check("an empty file -> NaN", is_nan(run(p("empty.xvg"), "-nr", "3")))
    check("a header-only file -> NaN", is_nan(run(xvg(p("hdr.xvg"), [], 5), "-nr", "3")))
    check("one row -> NaN", is_nan(run(xvg(p("one.xvg"), FULL[:1], 5), "-nr", "3")))
    check("two rows integrate: -0.0002*(-30)*2 = 0.012",
          is_work(run(xvg(p("two.xvg"), FULL[:2], 5), "-nr", "3"), 0.012))

    print("\n[3] the row the writer was cut in is dropped, wherever it was cut")
    cuts = (("inside the integrated columns (v1.4: IndexError, empty stdout)", "2502.0000\t-10\t-1"),
            ("inside the network tail (v1.4: integrated it, 15.012)", "2502.0000\t-10\t-10\t-10\t7"),
            ("complete but without its newline", "2502.0000\t-10\t-10\t-10\t7\t7"))
    for i, (what, tail) in enumerate(cuts):
        f_cut = xvg(p("cut%d.xvg" % i), HALF, 5, tail=tail)
        check("cut %s -> 15.0" % what, is_work(run(f_cut, "-nr", "3"), 15.0),
              run(f_cut, "-nr", "3"))
        check("  and NaN with --tend 5000", is_nan(run(f_cut, "-nr", "3", "--tend", "5000")))
    check("a header plus one cut row -> NaN",
          is_nan(run(xvg(p("onecut.xvg"), [], 5, tail="0.0000\t-10\t-10"), "-nr", "3")))

    print("\n[4] --tend: within one output interval (2 ps) of the end")
    for tend, ok in (("5001.9", True), ("5002.0", True), ("5002.1", False)):
        res = run(f_full, "-nr", "3", "--tend", tend)
        check("last row 5000 ps, --tend %s -> %s" % (tend, "30.0" if ok else "NaN"),
              is_work(res, 30.0) if ok else is_nan(res), res)
    res = run(xvg(p("dead.xvg"), FULL[:1501], 5), "-nr", "3", "--tend", "5000")
    check("a leg that stopped at 3000 of 5000 ps -> NaN", is_nan(res), res)
    check("  saying where it stopped", "3000" in res[2], res[2])
    check("  but without --tend its rows still integrate (18.0)",
          is_work(run(p("dead.xvg"), "-nr", "3"), 18.0))

    print("\n[5] a file from another restraint set is refused")
    check("-nr 6 on 5 force columns -> NaN (v1.4: crash, empty stdout)",
          is_nan(run(f_full, "-nr", "6")))
    check("-nr 0 -> NaN (v1.4: -0.0)", is_nan(run(f_full, "-nr", "0")))
    check("--ncols 4 against 5 legends -> NaN", is_nan(run(f_full, "-nr", "3", "--ncols", "4")))
    check("--ncols 6 against 5 legends -> NaN", is_nan(run(f_full, "-nr", "3", "--ncols", "6")))
    short = FULL[:5] + [(10.0, [-10.0, -10.0, -10.0, 7.0])] + FULL[6:]
    check("a short row before the last -> NaN (v1.4: integrated it)",
          is_nan(run(xvg(p("short.xvg"), short, 5), "-nr", "3")))
    check("  also without legends, by the first row's width",
          is_nan(run(xvg(p("shortnl.xvg"), short, 5, legends=False), "-nr", "3")))
    check("without legends a consistent file integrates (30.0)",
          is_work(run(xvg(p("nolegend.xvg"), FULL, 5, legends=False), "-nr", "3"), 30.0))
    check("  and --ncols then sets the width: 5 passes",
          is_work(run(p("nolegend.xvg"), "-nr", "3", "--ncols", "5"), 30.0))
    check("  and 4 refuses every row", is_nan(run(p("nolegend.xvg"), "-nr", "3", "--ncols", "4")))

    print("\n[6] time running backwards, a non-finite force")
    back = FULL[:10] + [(4.0, [-10.0, -10.0, -10.0, 7.0, 7.0])] + FULL[10:]
    check("a row whose time goes back -> NaN", is_nan(run(xvg(p("back.xvg"), back, 5), "-nr", "3")))
    nanrow = FULL[:10] + [(20.0, [-10.0, float("nan"), -10.0, 7.0, 7.0])] + FULL[11:]
    check("a nan force -> NaN (v1.4: printed nan with exit 0)",
          is_nan(run(xvg(p("nanf.xvg"), nanrow, 5), "-nr", "3")))
    dup = FULL[:10] + [FULL[9]] + FULL[10:]
    check("a repeated time stamp adds nothing (30.0)",
          is_work(run(xvg(p("dup.xvg"), dup, 5), "-nr", "3"), 30.0))

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all integrate.py completeness checks passed")
