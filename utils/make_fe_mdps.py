#!/usr/bin/env python3
"""Generate the six FE .mdp files for each force field from that force field's own
base mdps, so force-field-specific physics (cutoffs, vdw treatment, DispCorr) is
inherited and only the FE-specific settings are applied."""
import os, re, sys

SET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings")
FFS = ["amber19sb_opc3", "amber19sb_opc", "charmm36", "gromos54a8"]

# leg -> (base mdp, ps of simulated time, init-lambda, lambda direction,
#         pull force out, trajectory frame interval in ps)
#   unbinding/rebinding : 20000 ps (0.00005 nm/ps -> 1.0 nm separation). This leg
#                         dominates the CGI uncertainty: at 10 ns the forward and
#                         reverse work distributions still had zero overlap, and
#                         at 20 ns they still do -- 17.5 RT of empty space, with
#                         no work of either direction inside the other's range.
#   bound restraint legs: 2000 ps  (these converge: their distributions overlap)
#   unbound hold        : 5000 ps  (see below)
#   bound equilibration : 1000 ps
#
# The hold went 1000 -> 5000 ps on 2026-08-11, and the bound legs 1500 -> 2000,
# for the next test. The reasoning is worth recording, because raising the hold
# is the cheaper of the two things that could explain the missing overlap:
# measured over 40 cycles of 2KTF the rebinding works are 2.4x WIDER than the
# unbinding works (sigma 51.5 vs 21.6). The reverse leg is the one that starts
# from the Boresch-restrained separated state, and it got only 1 ns to settle
# after a 1.0 nm separation -- short for interfacial water and side chains to
# relax. An under-equilibrated starting ensemble inflates that width and breaks
# the forward/reverse pairing Crooks needs, and no amount of switching time or
# extra cycles repairs it. 4 ns more hold costs ~9% per cycle against ~44% for
# doubling a 20 ns leg, so it is worth ruling out first.
#
# NOTE: the pull rate must satisfy rate * unbinding_time = pull distance (1.0 nm).
# It lives in make_boresch.py (--pull-rate) and is consumed by integrate.py (-r),
# so all three must be changed together. Only the UNBINDING time is coupled to
# the rate; the hold and the bound legs can move freely.
#
# npt_init_fe.mdp is the SETUP equilibration and exists separately from
# npt_fe.mdp, which every cycle runs. The anchor selection in make_boresch.py
# measures backbone flexibility from this trajectory, so it needs length (11 ns,
# of which the first 1 ns is discarded as equilibration) and frames (every 10 ps,
# giving 1000 usable ones). Both are paid once per structure. Sharing one mdp
# would have added 10 ns to every cycle instead.
LEGS = {
    "bind_fe.mdp":     ("bind.mdp",   20000.0, 0, +1, 500, 400.0),
    "bindrev_fe.mdp":  ("bindrev.mdp", 20000.0, 1, -1, 500, 400.0),
    "boundfwd.mdp":    ("bind.mdp",    2000.0, 0, +1, 0, 400.0),
    "boundrev.mdp":    ("bind.mdp",    2000.0, 1, -1, 0, 400.0),
    "nptrev_fe.mdp":   ("nptrev.mdp",  5000.0, 1,  0, 0, 400.0),
    "npt_relax_fe.mdp": ("npt.mdp",     100.0, None, None, None, 50.0),
    "npt_fe.mdp":      ("npt.mdp",     1000.0, None, None, None, 400.0),
    "npt_init_fe.mdp": ("npt.mdp",    11000.0, None, None, None, 10.0),
}

# Barostat per leg, overriding whatever the base mdp carries.
#
# Berendsen does not generate any well-defined ensemble: it damps the volume
# toward the target without the fluctuation term, so it gives too-narrow volume
# fluctuations and no rigorous NPT distribution. It is fine for pushing a fresh
# box to roughly the right density and nothing else, so it survives here in
# exactly one place, the 100 ps relaxation.
#
# Everything that produces configurations the estimator consumes runs
# Parrinello-Rahman, which does sample the isothermal-isobaric ensemble. That
# includes BOTH equilibrations, and npt_fe is the one that matters most: it
# defines the bound-state ensemble every cycle's boundfwd starts from, and the
# Crooks/Jarzynski analysis assumes those initial states are drawn from the true
# equilibrium distribution. Equilibrating them under Berendsen broke that
# assumption on every cycle. npt_init_fe additionally supplies the trajectory
# make_boresch.py measures backbone RMSF from, so its volume fluctuations feed
# the anchor selection.
#
# Parrinello-Rahman is second-order and oscillates when started far from
# equilibrium, which is why it is not used for the relaxation. The 100 ps
# Berendsen step exists to hand it a box that is already close.
BAROSTAT = {
    "npt_relax_fe.mdp": ("Berendsen",         0.5),
    "npt_fe.mdp":       ("Parrinello-Rahman", 2.0),
    "npt_init_fe.mdp":  ("Parrinello-Rahman", 2.0),
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
    "npt_relax_fe.mdp": ("; npt_relax_fe.mdp - 100 ps weak-coupling NPT run before every long NPT\n"
                         "; equilibration, in setup and in each cycle. Berendsen ONLY here: it relaxes\n"
                         "; the box to roughly the right density without the oscillation\n"
                         "; Parrinello-Rahman shows when started far from equilibrium. It generates no\n"
                         "; well-defined ensemble, so nothing downstream may consume its output as a\n"
                         "; sample; it exists to hand the next leg a box that is already close.\n"),
    "npt_fe.mdp": ("; npt_fe.mdp - 1 ns NPT equilibration of the unrestrained bound state before\n"
                   "; each FE cycle (extended from the 100 ps npt.mdp of the classic protocol).\n"
                   "; Parrinello-Rahman, so the configurations handed to boundfwd are drawn from the\n"
                   "; true NPT ensemble the Crooks analysis assumes. Preceded by npt_relax_fe.\n"),
    "npt_init_fe.mdp": ("; npt_init_fe.mdp - 11 ns NPT equilibration run ONCE during setup. Longer\n"
                        "; than the per-cycle npt_fe.mdp because make_boresch.py measures backbone\n"
                        "; flexibility from this trajectory to choose the Boresch anchor groups:\n"
                        "; the first 1 ns is discarded and the remaining 10 ns, sampled every 10 ps,\n"
                        "; gives the 1000 frames the RMSF is taken from. Parrinello-Rahman, preceded\n"
                        "; by npt_relax_fe.\n"),
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
    for out_name, (base_name, ps, init_lam, lam_dir, pull_fout, xout_ps) in LEGS.items():
        base_path = os.path.join(d, base_name)
        if not os.path.isfile(base_path):
            print("  %s: missing base %s, skipped" % (ff, base_name)); continue
        text = read(base_path)

        dt = float(get_param(text, "dt"))
        nsteps = int(round(ps / dt))

        text = strip_pull_block(text)
        text = set_param(text, "nsteps", nsteps)
        if out_name in BAROSTAT:
            pcoupl, tau_p = BAROSTAT[out_name]
            text = set_param(text, "pcoupl", pcoupl)
            text = set_param(text, "tau_p", tau_p)
        # Frame interval is per leg: ~12 frames is plenty for a switching leg,
        # but the setup equilibration is the input to the anchor selection and
        # needs enough frames to measure a backbone RMSF from.
        text = set_param(text, "nstxout-compressed", int(round(xout_ps / dt)))

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
    print("generated %d FE mdps for %s" % (len(LEGS), ff))
