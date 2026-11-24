#!/usr/bin/env python3
"""The SLURM settings of a submission, so that a --restart uses them again.

groscore.py and groscore_fe.py build every SLURM script from a template in slurm/
(--slurm, default workstation), and groscore_fe.py throttles its cycle arrays with
--array-throttle (SLURM %N, default none). A --restart without those options fell
back to the defaults: a run submitted to VSC-5 with --slurm vsc5 went out again with
the workstation template (no partition, no QOS, no GPU), and a throttled FE run
unthrottled.

So every generated script names its template in a comment line right after the
template's own lines (a comment does not end sbatch's directive block, see
tests/test_slurm_templates.py):

    # GroScore SLURM template: vsc5

A --restart without --slurm takes the template of the newest previous script, and
groscore_fe.py without --array-throttle the %N of the newest previous cycle array
(none there: no throttle). A script written before the comment existed is recognised
by its header: it starts with the text of a template in slurm/ (the longest one when
several do). When a previous script names no template and starts with none (the
template was edited or removed since), the restart stops and asks for --slurm rather
than guess. An explicit option always wins; the log says when it differs from the
previous submission. --run-local uses no template and is not affected.

Standard library only; imported by both orchestrators.
"""
import os, re

TEMPLATE_MARK = "# GroScore SLURM template: "
DEFAULT_TEMPLATE = "workstation"


def template_names(script_dir):
    d = os.path.join(script_dir, "slurm")
    return sorted(n[:-3] for n in os.listdir(d) if n.endswith(".sh")) if os.path.isdir(d) else []


def read_template(script_dir, name):
    """The template's text without trailing newlines, as the scripts embed it."""
    with open(os.path.join(script_dir, "slurm", name + ".sh")) as f:
        return f.read().rstrip("\n")


def marker(name):
    """The comment line that names a script's template (written after the template)."""
    return TEMPLATE_MARK + name + "\n"


def _newest(paths):
    existing = [p for p in paths if os.path.isfile(p)]
    return max(existing, key=lambda p: (os.path.getmtime(p), p)) if existing else None


def template_of(path, script_dir):
    """The template a generated script was built from: its marker line, else the
    template in slurm/ whose text the script starts with; None when neither."""
    with open(path) as f:
        text = f.read()
    for line in text.splitlines():
        if line.startswith(TEMPLATE_MARK):
            return line[len(TEMPLATE_MARK):].strip() or None
    hits = []
    for name in template_names(script_dir):
        try:
            t = read_template(script_dir, name)
        except OSError:
            continue
        if text.startswith(t + "\n"):
            hits.append((len(t), name))
    return max(hits)[1] if hits else None


def throttle_of(path):
    """The %N of a script's '#SBATCH --array=' line: N, 0 without one, None without
    an --array line."""
    with open(path) as f:
        for line in f:
            m = re.match(r"#SBATCH --array=(\S+)\s*$", line)
            if m:
                t = re.search(r"%(\d+)$", m.group(1))
                return int(t.group(1)) if t else 0
    return None


def choose_template(given, restart, paths, script_dir):
    """(template name, note to print or None, error or None) for a submission.

    given: --slurm as typed (None when not given); paths: the scripts a previous
    submission of this run directory may have written."""
    prev = _newest(paths) if restart else None
    prev_name = template_of(prev, script_dir) if prev else None
    if given:
        if prev_name and prev_name != given:
            return given, ("SLURM template %s, as asked (the previous submission, %s, used %s)."
                           % (given, prev, prev_name)), None
        return given, None, None
    if prev is None:
        return DEFAULT_TEMPLATE, None, None
    if prev_name:
        return prev_name, ("SLURM template %s, as the previous submission (%s); --slurm overrides."
                           % (prev_name, prev)), None
    return None, None, ("ERROR: --restart without --slurm, and the previous submission script %s "
                        "names no template and starts with none in slurm/ (edited or removed "
                        "since?). Nothing was submitted. Pass --slurm <name>; templates: %s."
                        % (prev, ", ".join(template_names(script_dir)) or "none"))


def choose_throttle(given, restart, paths):
    """(array throttle, note or None): --array-throttle as typed wins; a --restart
    without it takes the newest previous cycle array's %N; else 0 (none)."""
    if given is not None:
        return given, None
    prev = _newest(paths) if restart else None
    t = throttle_of(prev) if prev else None
    if t:
        return t, "Array throttle %%%d, as the previous submission (%s); --array-throttle overrides." % (t, prev)
    return 0, None
