#!/usr/bin/env python3
"""Generate the six FE .mdp files for each force field from that force field's own
base mdps, so force-field-specific physics (cutoffs, vdw treatment, DispCorr) is
inherited and only the FE-specific settings are applied."""
import os, re, sys

SET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings")
FFS = ["amber19sb_opc3", "amber19sb_opc", "charmm36", "gromos54a8"]

# leg -> (base mdp, ps of simulated time, init-lambda, lambda direction, pull force out)
#   unbinding/rebinding : 10000 ps (0.0001 nm/ps -> 1.0 nm separation). Lengthened
#                         from 5 ns because this leg dominates the CGI uncertainty.
#   bound restraint legs: 1000 ps  (converges quickly, small dG_intro error)
#   hold / equilibration: 1000 ps
# NOTE: the pull rate must satisfy rate * unbinding_time = pull distance (1.0 nm).
# It lives in make_boresch.py (--pull-rate) and is consumed by integrate.py (-r),
# so all three must be changed together.
LEGS = {
    "bind_fe.mdp":    ("bind.mdp",   10000.0, 0, +1, 500),
    "bindrev_fe.mdp": ("bindrev.mdp", 10000.0, 1, -1, 500),
    "boundfwd.mdp":   ("bind.mdp",    1000.0, 0, +1, 0),
    "boundrev.mdp":   ("bind.mdp",    1000.0, 1, -1, 0),
    "nptrev_fe.mdp":  ("nptrev.mdp",  1000.0, 1,  0, 0),
    "npt_fe.mdp":     ("npt.mdp",     1000.0, None, None, None),
}

HEADERS = {
    "bind_fe.mdp": ("; bind_fe.mdp - FE unbinding leg (forward). Interface restraints fade out\n"
                    "; (k -> kB=0) while Boresch restraints fade in (k=0 -> kB); the interface and\n"
                    "; Boresch-r references move outward at pull-rate to +1.0 nm. The full pull block\n"
                    "; (groups + coords) is appended by make_boresch.py.\n"),
    "bindrev_fe.mdp": ("; bindrev_fe.mdp - FE rebinding leg (reverse of bind_fe). lambda ramps 1 -> 0:\n"
                       "; interface restraints fade back in, Boresch fade out, references move inward\n"
                       "; from +1.0 nm back to the bound geometry.\n"),
    "boundfwd.mdp": ("; boundfwd.mdp - Bound-state restraint introduction (dhdl only, no pulling).\n"
                     "; Interface restraints switched ON in the bound state (k=0 -> kB=full).\n"),
    "boundrev.mdp": ("; boundrev.mdp - Bound-state restraint removal (reverse of boundfwd, lambda 1 -> 0).\n"),
    "nptrev_fe.mdp": ("; nptrev_fe.mdp - Hold the unbound, Boresch-restrained state (lambda = 1)\n"
                      "; between the unbinding and rebinding legs. No switching (delta-lambda = 0).\n"),
    "npt_fe.mdp": ("; npt_fe.mdp - 1 ns NPT equilibration of the unrestrained bound state before\n"
                   "; each FE cycle (extended from the 100 ps npt.mdp of the classic protocol).\n"),
}


def read(path):
    with open(path) as f:
        return f.read()


def get_param(text, key):
    m = re.search(r"^%s\s*=\s*(\S+)" % re.escape(key), text, re.M)
    return m.group(1) if m else None


def set_param(text, key, value):
    """Replace 'key = ...' keeping the original column alignment; append if absent."""
    pat = re.compile(r"^(%s\s*=\s*).*$" % re.escape(key), re.M)
    if pat.search(text):
        return pat.sub(lambda m: m.group(1) + str(value), text, count=1)
    return text.rstrip("\n") + "\n%-24s = %s\n" % (key, value)


def strip_pull_block(text):
    """Remove the trailing 'pull = yes' print block so it can be re-emitted."""
    i = text.find("pull                     = yes")
    return text[:i].rstrip("\n") + "\n" if i != -1 else text.rstrip("\n") + "\n"


for ff in FFS:
    d = os.path.join(SET, ff)
    if not os.path.isdir(d):
        print("skip missing %s" % ff); continue
    for out_name, (base_name, ps, init_lam, lam_dir, pull_fout) in LEGS.items():
        base_path = os.path.join(d, base_name)
        if not os.path.isfile(base_path):
            print("  %s: missing base %s, skipped" % (ff, base_name)); continue
        text = read(base_path)

        dt = float(get_param(text, "dt"))
        nsteps = int(round(ps / dt))

        text = strip_pull_block(text)
        text = set_param(text, "nsteps", nsteps)
        # Uniform trajectory output across all FE legs (~12 frames per leg),
        # independent of which base mdp the leg was derived from.
        text = set_param(text, "nstxout-compressed", 100000)

        if init_lam is not None:                      # a free-energy leg
            text = set_param(text, "nstcalcenergy", 100)   # nstdhdl must be a multiple
            dl = 0.0 if lam_dir == 0 else lam_dir * (1.0 / nsteps)
            fe = ("\n; ---- Free-energy: force-constant switching k -> kB, driven by lambda ----\n"
                  "free-energy              = yes\n"
                  "init-lambda              = %d\n"
                  "delta-lambda             = %s\n"
                  "dhdl-derivatives         = yes\n"
                  "nstdhdl                  = 500\n"
                  "separate-dhdl-file       = yes\n"
                  "calc-lambda-neighbors    = 0\n"
                  "sc-alpha                 = 0\n"
                  % (init_lam, "0" if dl == 0 else ("%.8e" % dl)))
            pull = ("\n; ---- COM pulling (pull block appended by make_boresch.py) ----\n"
                    "pull                     = yes\n"
                    "pull-print-com1          = no\n"
                    "pull-print-com2          = no\n"
                    "pull-print-ref-value     = no\n"
                    "pull-print-components    = no\n"
                    "pull-nstxout             = 0\n"
                    "pull-nstfout             = %d\n" % pull_fout)
            text = text.rstrip("\n") + "\n" + fe + pull
        else:
            text = text.rstrip("\n") + "\n"

        header = HEADERS[out_name] + ";\n; nsteps %d * dt %g = %g ps.\n\n" % (nsteps, dt, ps)
        with open(os.path.join(d, out_name), "w") as f:
            f.write(header + text)
    print("generated 6 FE mdps for %s" % ff)
