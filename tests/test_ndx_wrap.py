#!/usr/bin/env python3
"""The ensemble pass in make_boresch.py must not be able to fail silently again.

On the first HADDOCK FE pass (2026-09-09) six of 24 setups ran with 20,000-40,000
interface springs at k = 0.3-0.6 instead of 400-1500 at k = 8-50, and every number
they produced is void. Three things had to line up:

  1. every trjconv selection was written as ONE index line. GROMACS 2026.0 takes a
     one-line group of 3800 atoms and aborts on 4000 with a glibc malloc assertion;
     the same 4000 wrapped 15 per line is fine. A big interface selection crosses
     that line, the backbone selection never did.
  2. the reader swallowed the abort (`continue`) and reported "only 0 frames".
  3. finalise_interface then returned the candidates untouched, on a comment that
     said the cutoff had already accepted them. That was true when candidates were
     enumerated at 0.6 nm and false since CANDIDATE_CUT widened the pass to 1.2.

This checks each link, on the source and on the writer's actual output.

Standalone, no pytest: python3 tests/test_ndx_wrap.py
"""
import os, re, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB = os.path.join(ROOT, "utils", "make_boresch.py")
src = open(MB).read()
fails = 0

def check(name, ok, detail=""):
    global fails
    print("  %-70s %s" % (name, "pass" if ok else "FAIL  %s" % detail))
    if not ok: fails += 1

print("\n[1] the index writer wraps")
m = re.search(r"^def write_ndx_group\(path, name, atoms\):.*?(?=^\S)", src, re.S | re.M)
check("write_ndx_group exists at module level", m is not None)
if m:
    ns = {}
    exec(compile(m.group(0), "write_ndx_group", "exec"), ns)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ndx")
        ns["write_ndx_group"](p, "sel", range(1, 4001))
        lines = open(p).read().splitlines()
        check("header line is the group name", lines[0] == "[ sel ]")
        body = lines[1:]
        check("no body line holds more than 15 numbers",
              max(len(ln.split()) for ln in body) <= 15, str(max(len(ln.split()) for ln in body)))
        check("no body line is longer than the gmx reader survives (<= 100 chars)",
              max(len(ln) for ln in body) <= 100, str(max(len(ln) for ln in body)))
        check("every atom is there exactly once, in order",
              [int(x) for ln in body for x in ln.split()] == list(range(1, 4001)))
        ns["write_ndx_group"](p, "e", [])
        check("an empty selection still writes a legal header", open(p).read() == "[ e ]\n")
check("no one-line index group is written anywhere any more",
      re.search(r'f\.write\("\[ \w+ \]\\n" \+ " "\.join', src) is None)
check("all four selections go through the writer",
      len(re.findall(r'write_ndx_group\(ndx, "(bbsel|ifsel|xrsel|borsel)"', src)) == 4,
      str(re.findall(r'write_ndx_group\(ndx, "(\w+)"', src)))

print("\n[2] a trjconv failure is never silent")
# the backbone pass builds its argv as `cmd = [...]` first, the others inline it
calls = [m.start() for m in re.finditer(r'\["gmx", "trjconv"', src)]
check("there are four trjconv passes to check", len(calls) == 4, str(len(calls)))
for i, pos in enumerate(calls):
    window = src[pos:pos + 600]
    check("trjconv call %d is followed by trjconv_failed(...)" % (i + 1),
          "trjconv_failed(r, traj, gro," in window)
check("nothing still does the bare returncode-and-continue",
      re.search(r"if r\.returncode != 0 or not os\.path\.isfile\(gro\):\n\s+continue", src) is None)
m = re.search(r"^def trjconv_failed\(.*?(?=^\S)", src, re.S | re.M)
check("trjconv_failed exists", m is not None)
if m:
    check("and it writes the last stderr line, not just a return code",
          "stderr" in m.group(0) and "sys.stderr.write" in m.group(0))

print("\n[3] the fallback in finalise_interface applies the contact cutoff")
fin = src[src.index("def finalise_interface("):]
fin = fin[:fin.index("\ninterdis, _ref_note")]
fb = fin[fin.index('if not note.startswith("mean over"):'):]
fb = fb[:fb.index("return")]
check("the no-trajectory branch re-cuts at interfacecutoff",
      "keep = [p for p in pairs if p[2] <= interfacecutoff]" in fb)
check("and returns the cut set, not the raw candidates",
      re.search(r'return keep, "single reference frame"', fin) is not None)
check("and aborts on an empty set rather than proceeding",
      "NO_INTERFACE_CONTACTS" in fb)
check("the old 'nothing to re-apply' comment is gone",
      "nothing to re-apply and nothing to drop" not in src)

print()
sys.exit(1 if fails else 0)
