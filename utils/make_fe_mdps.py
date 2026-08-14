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
# The unbinding leg is SPLIT at u = 0.3, where 75% of the hysteresis is spent.
#
# Measured on 50 cycles of 2KTF: the first 0.3 nm carries 75.5% of the total
# dissipation and the remaining 0.7 nm carries 10.6%, so a uniform ramp spends
# 70% of its time where almost nothing dissipates. Stage A now takes 15 ns over
# 0.3 nm and stage B 5 ns over 0.7 nm, which is 3.5x slower through the rupture
# and 2.8x faster through the tail, at the same 20 ns per direction.
#
# lam_from -> lam_to tracks the same split: lambda runs 0 -> 0.3 over stage A and
# 0.3 -> 1 over stage B, so the force-constant switching stays proportional to the
# distance travelled and the two stages join continuously.
#
# The 1 ns holds at u = 0.3 are what make the split worth having. Summing the two
# stages' works per cycle would reproduce the unstaged distribution exactly and
# improve nothing; estimating each stage separately and adding the estimates is
# what shrinks the dissipation per estimate, and that is only legitimate because
# the hold equilibrates between them. Both holds have delta-lambda = 0 and pull
# rate 0, so neither contributes work.
#
# The unbound hold drops 5 ns -> 3 ns: measured drift across the old 5 ns was
# 0.031 +- 1.87 kJ/mol (p = 0.91), so it was equilibrated within 1 ns, and the
# interface kept drifting away from the rebinding reference while it ran.
#
# Per cycle: 2 + 15 + 1 + 5 + 3 + 5 + 1 + 15 + 2 = 49 ns, unchanged.
LEGS = {
    "bindfwdA_fe.mdp":  ("bind.mdp",   15000.0, 0.0, 0.3, 500, 400.0),
    "holdmid_fe.mdp":   ("nptrev.mdp",  1000.0, 0.3, 0.3, 0, 400.0),
    "bindfwdB_fe.mdp":  ("bind.mdp",    5000.0, 0.3, 1.0, 500, 400.0),
    "nptrev_fe.mdp":    ("nptrev.mdp",  3000.0, 1.0, 1.0, 0, 400.0),
    "bindrevB_fe.mdp":  ("bindrev.mdp", 5000.0, 1.0, 0.3, 500, 400.0),
    "holdmidrev_fe.mdp":("nptrev.mdp",  1000.0, 0.3, 0.3, 0, 400.0),
    "bindrevA_fe.mdp":  ("bindrev.mdp",15000.0, 0.3, 0.0, 500, 400.0),
    "boundfwd.mdp":     ("bind.mdp",    2000.0, 0.0, 1.0, 0, 400.0),
    "boundrev.mdp":     ("bind.mdp",    2000.0, 1.0, 0.0, 0, 400.0),
    "npt_fe.mdp":       ("npt.mdp",     1000.0, None, None, None, 400.0),
    "npt_init_fe.mdp":  ("npt.mdp",    11000.0, None, None, None, 10.0),
}

# Distance the pull reference has already travelled when each leg STARTS, in nm,
# as a fraction of --pull-dist. make_boresch.py adds this to pull-coordN-init for
# the moving coordinates, because pull-coordN-start = no makes init absolute and
# stage B has to begin where stage A stopped.
LEG_U0 = {
    "bindfwdA_fe.mdp": 0.0, "holdmid_fe.mdp": 0.3, "bindfwdB_fe.mdp": 0.3,
    "nptrev_fe.mdp":   1.0, "bindrevB_fe.mdp": 1.0, "holdmidrev_fe.mdp": 0.3,
    "bindrevA_fe.mdp": 0.3, "boundfwd.mdp":    0.0, "boundrev.mdp":     0.0,
}

# One barostat for every pressure-coupled leg, overriding whatever the base mdp
# carries so it cannot drift back by inheritance.
#
# C-rescale (Bernetti and Bussi, J. Chem. Phys. 153, 114107, 2020) is stochastic
# cell rescaling: Berendsen's first-order volume relaxation plus a correctly sized
# noise term. It is to the Berendsen barostat what V-rescale is to the Berendsen
# thermostat, and this pipeline already uses V-rescale for temperature.
#
# It replaces two different things at once.
#
# Berendsen, which the equilibrations used, samples no ensemble at all. GROMACS
# 2026 says so itself: "The Berendsen barostat does not generate any strictly
# correct ensemble, and should not be used for new production simulations (in our
# opinion). We recommend using the C-rescale barostat instead." That warning was
# being swallowed by the -maxwarn on every grompp. It mattered most for npt_fe,
# whose output is the bound-state configuration each cycle's boundfwd starts
# from: the Crooks/Jarzynski analysis assumes those initial states are drawn from
# the true equilibrium distribution.
#
# Parrinello-Rahman, which the legs used, does sample the right ensemble but is
# second order, so it oscillates when started from a box that has never been
# pressure-coupled. That is why an equilibration protocol built on it needs a
# separate weak-coupling stage in front. C-rescale is first order and relaxes
# monotonically, so it is stable from the NVT ladder directly and that stage is
# not needed. One barostat, every leg, no staging.
#
# tau_p 2.0 ps is unchanged and sits inside the 1-5 ps range C-rescale is normally
# run at.
BAROSTAT_ALL = ("C-rescale", 2.0)

HEADERS = {
    "bindfwdA_fe.mdp": ("; bindfwdA_fe.mdp - FE unbinding, STAGE A: u 0 -> 0.3 nm over 15 ns,\n"
                        "; lambda 0 -> 0.3. This is the rupture, which carries 75% of the leg's\n"
                        "; dissipation in 30% of its distance, so it gets 75% of the time. Interface\n"
                        "; restraints fade out (k -> kB=0) while Boresch fade in (k=0 -> kB). The full\n"
                        "; pull block (groups + coords) is appended by make_boresch.py.\n"),
    "holdmid_fe.mdp": ("; holdmid_fe.mdp - 1 ns equilibrium hold at u = 0.3, lambda = 0.3, between the\n"
                       "; two forward stages. delta-lambda = 0 and pull rate 0, so it contributes no\n"
                       "; work. It exists so the two stages can be estimated SEPARATELY and their\n"
                       "; estimates added: summing their per-cycle works instead would reproduce the\n"
                       "; unstaged distribution exactly and gain nothing.\n"),
    "bindfwdB_fe.mdp": ("; bindfwdB_fe.mdp - FE unbinding, STAGE B: u 0.3 -> 1.0 nm over 5 ns,\n"
                        "; lambda 0.3 -> 1. The tail, which carries about 11% of the dissipation and\n"
                        "; previously received 70% of the time.\n"),
    "bindrevB_fe.mdp": ("; bindrevB_fe.mdp - FE rebinding, STAGE B reversed: u 1.0 -> 0.3 over 5 ns,\n"
                        "; lambda 1 -> 0.3. Pairs with bindfwdB for the stage-B work distributions.\n"),
    "holdmidrev_fe.mdp": ("; holdmidrev_fe.mdp - 1 ns equilibrium hold at u = 0.3 on the way back, so\n"
                          "; stage A reversed starts from an equilibrated lambda = 0.3 ensemble rather\n"
                          "; than from wherever stage B reversed happened to end.\n"),
    "bindrevA_fe.mdp": ("; bindrevA_fe.mdp - FE rebinding, STAGE A reversed: u 0.3 -> 0 over 15 ns,\n"
                        "; lambda 0.3 -> 0. Pairs with bindfwdA. This is the re-contact, where 78% of\n"
                        "; the reverse leg's work variance was measured to be generated.\n"),
    "boundfwd.mdp": ("; boundfwd.mdp - Bound-state restraint introduction (dhdl only, no pulling).\n"
                     "; Interface restraints switched ON in the bound state (k=0 -> kB=full).\n"),
    "boundrev.mdp": ("; boundrev.mdp - Bound-state restraint removal (reverse of boundfwd, lambda 1 -> 0).\n"),
    "nptrev_fe.mdp": ("; nptrev_fe.mdp - Hold the unbound, Boresch-restrained state (lambda = 1)\n"
                      "; between the unbinding and rebinding legs. No switching (delta-lambda = 0).\n"
                      "; 3 ns, down from 5: the Boresch channel equilibrates inside 1 ns (drift across\n"
                      "; the old 5 ns was 0.031 +- 1.87 kJ/mol, p = 0.91) and the interface kept\n"
                      "; drifting away from the rebinding reference for as long as the hold ran.\n"),
    "npt_fe.mdp": ("; npt_fe.mdp - 1 ns NPT equilibration of the unrestrained bound state before\n"
                   "; each FE cycle (extended from the 100 ps npt.mdp of the classic protocol).\n"
                   "; C-rescale, so the configurations handed to boundfwd are drawn from the true\n"
                   "; NPT ensemble the Crooks analysis assumes. Runs straight off the NVT ladder:\n"
                   "; C-rescale is first order and needs no weak-coupling stage in front of it.\n"),
    "npt_init_fe.mdp": ("; npt_init_fe.mdp - 11 ns NPT equilibration run ONCE during setup. Longer\n"
                        "; than the per-cycle npt_fe.mdp because make_boresch.py measures backbone\n"
                        "; flexibility from this trajectory to choose the Boresch anchor groups:\n"
                        "; the first 1 ns is discarded and the remaining 10 ns, sampled every 10 ps,\n"
                        "; gives the 1000 frames the RMSF is taken from. The discarded first 1 ns\n"
                        "; also absorbs the C-rescale box relaxation off the NVT ladder.\n"),
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
    for out_name, (base_name, ps, lam_from, lam_to, pull_fout, xout_ps) in LEGS.items():
        base_path = os.path.join(d, base_name)
        if not os.path.isfile(base_path):
            print("  %s: missing base %s, skipped" % (ff, base_name)); continue
        text = read(base_path)

        dt = float(get_param(text, "dt"))
        nsteps = int(round(ps / dt))

        text = strip_pull_block(text)
        text = set_param(text, "nsteps", nsteps)
        if get_param(text, "pcoupl") not in (None, "no"):
            text = set_param(text, "pcoupl", BAROSTAT_ALL[0])
            text = set_param(text, "tau_p", BAROSTAT_ALL[1])
        # Frame interval is per leg: ~12 frames is plenty for a switching leg,
        # but the setup equilibration is the input to the anchor selection and
        # needs enough frames to measure a backbone RMSF from.
        text = set_param(text, "nstxout-compressed", int(round(xout_ps / dt)))

        if lam_from is not None:                      # a free-energy leg
            text = set_param(text, "nstcalcenergy", 100)   # nstdhdl must be a multiple
            # lambda is PER STEP, and the endpoints are now fractional: a stage
            # that covers 0 -> 0.3 must land exactly on 0.3 so the next stage
            # starts where this one stopped. A hold has lam_from == lam_to and so
            # gets delta-lambda = 0 for free, which is also what makes it
            # contribute no work.
            dl = (lam_to - lam_from) / nsteps if lam_to != lam_from else 0.0
            fe = ("\n; ---- Free-energy: force-constant switching k -> kB, driven by lambda ----\n"
                  "free-energy              = yes\n"
                  "init-lambda              = %s\n"
                  "delta-lambda             = %s\n"
                  "dhdl-derivatives         = yes\n"
                  "nstdhdl                  = 500\n"
                  "separate-dhdl-file       = yes\n"
                  "calc-lambda-neighbors    = 0\n"
                  "sc-alpha                 = 0\n"
                  % (("%g" % lam_from), "0" if dl == 0 else ("%.8e" % dl)))
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
