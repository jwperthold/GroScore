#!/usr/bin/env python3
"""The SLURM templates, and the one way they fail silently.

sbatch reads #SBATCH directives only from the leading block of a script: comments
and blank lines between them are fine, but the FIRST executable line ends the block
and every directive after it is ignored without a word. groscore_fe.py appends
--job-name and --array to the template and then writes shell code, so a template
that ends in shell, or a directive added below the code, disables itself.

That is worth pinning because the failure is invisible: the job still runs, just
without the option. The --exclude line added in this repo sits below a long comment
block, which is legal, and this test is what keeps it legal.

Standalone, no pytest: python3 tests/test_slurm_templates.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM = os.path.join(ROOT, "slurm")
failures = []


def check(name, ok, detail=""):
    print("  %-64s %s" % (name, "pass" if ok else "FAIL"))
    if not ok:
        failures.append("%s %s" % (name, detail))


def first_code_line(text):
    """1-based index of the first line sbatch treats as an executable command."""
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return i
    return None


def directives(text):
    return [(i, l.strip()) for i, l in enumerate(text.splitlines(), 1)
            if l.strip().startswith("#SBATCH")]


print("\n[1] every template's directives precede its first executable line")
for name in sorted(os.listdir(SLURM)):
    if not name.endswith(".sh"):
        continue
    text = open(os.path.join(SLURM, name)).read()
    code = first_code_line(text)
    ds = directives(text)
    if code is None:
        check("%s is all directives and comments" % name, True)
    else:
        bad = [i for i, _ in ds if i > code]
        check("%s: no #SBATCH below the first command (line %s)" % (name, code),
              not bad, "directives at lines %s" % bad)

print("\n[2] the generated scripts, assembled the way groscore_fe.py assembles them")
tmpl = open(os.path.join(SLURM, "vsc5.sh")).read()
setup = tmpl + "\n#SBATCH --job-name=fesetup_X\n\n" + "cd X || exit 1\n./job.run --setup\n"
cycles = (tmpl + "\n#SBATCH --job-name=fecyc_X\n#SBATCH --array=1-50\n\n"
          "export GROSCORE_NUMCYCLES=50\ncd X || exit 1\n"
          "./job.run --cycle $SLURM_ARRAY_TASK_ID\n")
for label, txt in (("setup", setup), ("cycles", cycles)):
    code = first_code_line(txt)
    ds = directives(txt)
    bad = [i for i, _ in ds if i > code]
    check("%s script keeps all %d directives above the code" % (label, len(ds)),
          not bad, "stranded at %s" % bad)
    check("%s script still carries --exclude" % label,
          any("--exclude" in d for _, d in ds))

print("\n[3] the exclude list itself")
for name in ("vsc5.sh", "vsc5_fast.sh"):
    text = open(os.path.join(SLURM, name)).read()
    m = re.search(r"^#SBATCH --exclude=(\S+)\s*$", text, re.M)
    check("%s has exactly one --exclude directive" % name,
          m is not None and len(re.findall(r"^#SBATCH --exclude=", text, re.M)) == 1)
    if m:
        nodes = m.group(1).split(",")
        check("%s: node names are well formed" % name,
              all(re.fullmatch(r"n\d{4}-\d{3}", n) for n in nodes),
              str([n for n in nodes if not re.fullmatch(r"n\d{4}-\d{3}", n)]))
        check("%s: no duplicates" % name, len(nodes) == len(set(nodes)))
        check("%s: sorted, so a diff of the list is readable" % name,
              nodes == sorted(nodes), str(nodes))

check("both vsc5 templates exclude the same set",
      re.search(r"--exclude=(\S+)", open(os.path.join(SLURM, "vsc5.sh")).read()).group(1)
      == re.search(r"--exclude=(\S+)", open(os.path.join(SLURM, "vsc5_fast.sh")).read()).group(1))

print("\n[4] the list is regenerable rather than folklore")
text = open(os.path.join(SLURM, "vsc5.sh")).read()
check("the template names the tool that regenerates it",
      "fe_node_report.py" in text)
check("and that tool exists",
      os.path.isfile(os.path.join(ROOT, "utils", "fe_node_report.py")))
check("workstation.sh is left alone (it is not SLURM)",
      "--exclude" not in open(os.path.join(SLURM, "workstation.sh")).read())

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all slurm-template checks passed")
