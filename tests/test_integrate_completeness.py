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
  [7] --last-ps: the last complete row must be exactly where the leg writes its
      last row. A record that lost only that row passes --tend (one interval) and
      is refused here, also when the row is there but cut; one that ends a row
      late is refused; four-decimal times pass.
  [8] the runners hand it over: job_fe.run's pull_span and job.run's integ_opts,
      taken from the scripts and run on every shipped pull mdp of every settings
      tree, give floor(nsteps / pull-nstfout) * pull-nstfout * dt (bindfwdD:
      2290 of 2291 ps), and integrate.py accepts what they print.
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


def rc_ok(res):
    rc, out, _ = res
    return rc == 0 and out not in ("", "NaN")


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

    print("\n[7] --last-ps: the last complete row is where the leg writes its last row")
    res = run(f_full, "-nr", "3", "--tend", "5000", "--last-ps", "5000")
    check("last row at 5000, --last-ps 5000 -> 30.0", is_work(res, 30.0), res)
    lost = xvg(p("lost.xvg"), FULL[:-1], 5)                                  # 0..4998 ps
    res = run(lost, "-nr", "3", "--tend", "5000")
    check("the last row lost: --tend alone lets it through (29.988)", is_work(res, 29.988), res)
    res = run(lost, "-nr", "3", "--tend", "5000", "--last-ps", "5000")
    check("  --last-ps 5000 refuses it", is_nan(res), res)
    check("  saying where the record ends and where it should",
          "4998" in res[2] and "5000" in res[2] and "lost its tail" in res[2], res[2])
    res = run(xvg(p("lostcut.xvg"), FULL[:-1], 5, tail="5000.0000\t-10\t-10\t-10\t7"),
              "-nr", "3", "--last-ps", "5000")
    check("the last row there but cut: refused, the cut line named", is_nan(res)
          and "cut, was dropped" in res[2], res)
    # bindfwdD: nsteps 572750, pull-nstfout 500, dt 0.004: rows 0..2290 ps, leg 2291.
    D = FULL[:1146]
    res = run(xvg(p("legD.xvg"), D, 5), "-nr", "3", "--tend", "2291", "--last-ps", "2290")
    check("a leg whose nsteps is no multiple of pull-nstfout (2290 of 2291 ps) passes",
          is_work(res, 0.0002 * 30 * 2290), res)
    res = run(xvg(p("legDlost.xvg"), D[:-1], 5), "-nr", "3", "--last-ps", "2290")
    check("  its record without the 2290 row: --last-ps 2290 refuses it", is_nan(res), res)
    # bindfwdA: nsteps 1260000 is a multiple of 500, rows 0..5040 ps. Losing the
    # 5040 row leaves 5038, one interval short, which --tend 5040 accepts.
    A = [(2.0 * k, [-10.0, -10.0, -10.0, 7.0, 7.0]) for k in range(2521)]
    res = run(xvg(p("legAlost.xvg"), A[:-1], 5), "-nr", "3", "--tend", "5040")
    check("a multiple of pull-nstfout (bindfwdA) without its last row passes --tend 5040",
          rc_ok(res), res)
    res = run(p("legAlost.xvg"), "-nr", "3", "--tend", "5040", "--last-ps", "5040")
    check("  and --last-ps 5040 refuses it", is_nan(res), res)
    res = run(xvg(p("late.xvg"), FULL + [(5002.0, [-10.0] * 3 + [7.0, 7.0])], 5),
              "-nr", "3", "--last-ps", "5000")
    check("a record one row past its last row -> NaN (ran longer than its mdp)",
          is_nan(res) and "longer" in res[2], res)
    # dt 0.00125, pull-nstfout 3: rows every 0.00375 ps, written with four decimals,
    # so the last row, 1333 * 3 * 0.00125 = 4.99875 ps, reads 4.9988.
    four = [(k * 3 * 0.00125, [-10.0, -10.0, -10.0, 7.0, 7.0]) for k in range(1334)]
    res = run(xvg(p("four.xvg"), four, 5), "-nr", "3", "--tend", "5",
              "--last-ps", "%.10g" % (1333 * 3 * 0.00125))
    check("times rounded to four decimals (4.9988) match --last-ps 4.99875", rc_ok(res), res)

print("\n[8] what the runners pass, on every shipped pull mdp")
import glob, re
SETTINGS = os.path.join(ROOT, "settings")


def mdp_value(path, key):
    v = None
    for line in open(path):
        k, _, rest = line.partition("=")
        if k.strip().replace("_", "-") == key:
            v = rest.split(";")[0].strip()
    return v


# The legs that write pull output (pull-nstfout 0, the holds, write none).
mdps = sorted(m for m in glob.glob(os.path.join(SETTINGS, "*", "*.mdp"))
              if (mdp_value(m, "pull") or "").lower() == "yes"
              and int(mdp_value(m, "pull-nstfout") or 0) > 0)


def expected(path):
    n, dt, s = int(mdp_value(path, "nsteps")), float(mdp_value(path, "dt")), int(mdp_value(path, "pull-nstfout"))
    return n * dt, (n // s) * s * dt


src_fe = open(os.path.join(ROOT, "job_fe.run")).read()
src_cl = open(os.path.join(ROOT, "job.run")).read()
m_fe = re.search(r"\n(\s*)pull_span\(\) \{\n.*?\n\1\}\n", src_fe, re.S)
m_cl = re.search(r"\n(\s*)integ_opts\(\) \{\n.*?\n\1\}\n", src_cl, re.S)
check("job_fe.run defines pull_span(), job.run integ_opts()", bool(m_fe and m_cl))
check("job_fe.run passes it to both integrate.py calls",
      len(re.findall(r"utils/integrate\.py[^\n]*\\\n[^\n]*-r \$_rate[^\n]*\$_span\)", src_fe)) == 2)
check("job.run passes the pull leg bind.mdp's and the push leg bindrev.mdp's",
      "PULL_OPTS=$(integ_opts bind.mdp)" in src_cl and "PUSH_OPTS=$(integ_opts bindrev.mdp)" in src_cl
      and "-f $PULL.xvg -nr $NUMPERTRES $PULL_OPTS)" in src_cl
      and "-f $PUSH.xvg -nr $NUMPERTRES $PUSH_OPTS)" in src_cl)
check("found the shipped pull mdps (%d)" % len(mdps), len(mdps) >= 50, len(mdps))
bad = []
if m_fe and m_cl:
    for m in mdps:
        tend, last = expected(m)
        fe = subprocess.run(["bash", "-c", m_fe.group(0) + '\npull_span "$1"', "_", m],
                            capture_output=True, text=True).stdout.split()
        cl = subprocess.run(["bash", "-c", m_cl.group(0) + '\ninteg_opts "$1"', "_", m],
                            capture_output=True, text=True).stdout.split()
        want_fe = ["--tend=%.10g" % tend, "--last-ps=%.10g" % last]
        if fe != want_fe:
            bad.append("pull_span %s: %s, want %s" % (os.path.relpath(m, SETTINGS), fe, want_fe))
        if cl[:4] != ["--tend", "%.10g" % tend, "--last-ps", "%.10g" % last]:
            bad.append("integ_opts %s: %s" % (os.path.relpath(m, SETTINGS), cl))
check("both give --tend nsteps*dt and --last-ps floor(nsteps/nstfout)*nstfout*dt", not bad,
      "; ".join(bad[:3]))
dmdp = os.path.join(SETTINGS, "amber19sb_opc3", "bindfwdD_fe.mdp")
if m_fe and os.path.isfile(dmdp):
    span = subprocess.run(["bash", "-c", m_fe.group(0) + '\npull_span "$1"', "_", dmdp],
                          capture_output=True, text=True).stdout.split()
    check("bindfwdD_fe.mdp: --tend=2291 --last-ps=2290", span == ["--tend=2291", "--last-ps=2290"], span)
    with tempfile.TemporaryDirectory() as d:
        f_d = xvg(os.path.join(d, "bindfwdD_1_pullf.xvg"), FULL[:1146], 5)
        check("  integrate.py takes them: the whole record passes",
              is_work(run(f_d, "-nr", "3", *span), 0.0002 * 30 * 2290))
        f_l = xvg(os.path.join(d, "bindfwdD_2_pullf.xvg"), FULL[:1145], 5)
        check("  and refuses the record without its last row", is_nan(run(f_l, "-nr", "3", *span)))
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, "x.mdp"), "w").write("nsteps = 1000\n")
    fe = subprocess.run(["bash", "-c", m_fe.group(0) + '\npull_span "$1"', "_", os.path.join(d, "x.mdp")],
                        capture_output=True, text=True) if m_fe else None
    check("pull_span on an mdp without dt prints nothing and fails (the work stays NaN)",
          fe is not None and fe.returncode != 0 and not fe.stdout.strip())
    open(os.path.join(d, "y.mdp"), "w").write("nsteps = 1000\ndt = 0.004\n")
    fe = subprocess.run(["bash", "-c", m_fe.group(0) + '\npull_span "$1"', "_", os.path.join(d, "y.mdp")],
                        capture_output=True, text=True) if m_fe else None
    check("  without pull-nstfout: --tend only", fe is not None and fe.stdout.split() == ["--tend=4"],
          fe.stdout if fe else None)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all integrate.py completeness checks passed")
