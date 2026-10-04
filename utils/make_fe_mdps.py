#!/usr/bin/env python3
"""Generate the FE .mdp files for each force field from that force field's own base
mdps, so force-field-specific physics (cutoffs, vdw treatment, DispCorr) is
inherited and only the FE-specific settings are applied.

WHICH legs exist, how long they are and what lambda each one spans is NOT decided
here: it comes from utils/fe_protocol.py, which is the single definition of the
cycle. This file only knows how to turn one leg description into one mdp. Moving a
stage boundary or adding a hold is an edit to fe_protocol.RAMP and nothing else.

    python3 utils/make_fe_mdps.py
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_protocol as P

SET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings")
FFS = ["amber19sb_opc3", "amber19sb_opc", "charmm36", "gromos54a8",
       "gromos54a8_rf"]

# Which base mdp each kind of leg inherits its physics from.
BASE = {"stage_fwd": "bind.mdp", "stage_rev": "bindrev.mdp",
        "bound": "bind.mdp", "hold": "nptrev.mdp", "npt": "npt.mdp"}

# One barostat for every pressure-coupled leg, overriding whatever the base mdp
# carries so it cannot drift back by inheritance.
#
# C-rescale (Bernetti and Bussi, J. Chem. Phys. 153, 114107, 2020) is stochastic
# cell rescaling: Berendsen's first-order volume relaxation plus a correctly sized
# noise term. It is to the Berendsen barostat what V-rescale is to the Berendsen
# thermostat, and this pipeline already uses V-rescale for temperature.
#
# Berendsen, which the equilibrations used, samples no ensemble at all, and
# GROMACS 2026 says so itself in a warning that -maxwarn was swallowing. It
# mattered most for the per-cycle npt_c (npt_c_fe.mdp since v1.5, npt_fe.mdp
# before), whose output each cycle's boundfwd1 starts from: the Crooks analysis
# assumes those initial states are drawn from the true equilibrium distribution.
# Parrinello-Rahman does sample
# the right ensemble but is second order, so it oscillates when started from a box
# that has never been pressure-coupled, which is why a protocol built on it needs
# a weak-coupling stage in front. C-rescale is first order and relaxes
# monotonically, so it runs straight off the NVT ladder. One barostat, every leg.
BAROSTAT_ALL = ("C-rescale", 2.0)


def header_for(leg):
    """The comment block at the top of a generated mdp: what this leg is for."""
    n, ps = leg["name"], leg["ps"]
    if leg["kind"] == "stage":
        return (
            "; %s - FE %s, stage %s: u %g -> %g nm over %g ns, lambda %g -> %g.\n"
            "; Interface restraints fade out (k -> kB=0) as the Boresch restraints fade\n"
            "; in (k=0 -> kB), in proportion to the distance travelled. The full pull\n"
            "; block (groups + coords) is appended per structure by make_boresch.py,\n"
            "; which reads this file for the stage's length and derives its pull rate\n"
            "; from it -- so changing nsteps here changes the rate to match.\n"
            % (leg["mdp"], "unbinding" if leg["dirn"] == "fwd" else "rebinding",
               leg["stage"], leg["u_from"], leg["u_to"], ps / 1000.0,
               leg["lam_from"], leg["lam_to"]))
    if leg["kind"] == "hold":
        # Four kinds of hold, told apart by where they sit in the table rather than
        # by name: inside the bound lambda switch, inside the ramp, the bound -> ramp
        # handoff at u = 0, and the unbound hold at u = PULL_DIST. The staging
        # argument (separate estimates added) holds for the first two only.
        # mdrun -deffnm writes a hold's dH/dlambda to <leg>_<cycle>.xvg; job_fe.run
        # keeps it, since that is what a hold length is measured from.
        later = [l["name"] for l in P.legs() if l["start_from"] == n]
        if leg["block"] == "bound":
            return (
                "; %s - %g ns equilibrium hold inside the bound restraint switch, u = 0,\n"
                "; lambda = %g. delta-lambda = 0 and pull rate 0 under the bound block of\n"
                "; the sub-legs either side, so it contributes NO work and the bound sub-leg\n"
                "; works still sum exactly to the work of the whole bound switch. It exists\n"
                "; so the sub-legs either side of it can be estimated SEPARATELY and their\n"
                "; estimates added, which is only legitimate if the ensemble here is\n"
                "; equilibrated: summing their per-cycle works instead would reproduce the\n"
                "; unstaged distribution exactly and gain nothing.\n"
                % (leg["mdp"], ps / 1000.0, leg["lam_from"]))
        if leg["u_from"] >= P.PULL_DIST:
            return (
                "; %s - %g ns equilibrium hold at the unbound state, u = %g nm, lambda = %g:\n"
                "; the Boresch restraints, and the elastic network if the structure has one,\n"
                "; with no interface springs. delta-lambda = 0 and pull rate 0, so it\n"
                "; contributes NO work. It joins the two directions: the forward ramp ends\n"
                "; here and the reverse ramp starts from its end.\n"
                % (leg["mdp"], ps / 1000.0, leg["u_from"], leg["lam_from"]))
        if leg["u_from"] == 0.0:
            return (
                "; %s - %g ns equilibrium hold at u = 0, lambda = %g: the handoff from the\n"
                "; bound legs to the ramp, under the Hamiltonian both run there (full\n"
                "; interface springs, Boresch at zero strength). delta-lambda = 0 and pull\n"
                "; rate 0, so it contributes NO work. The forward ramp starts from its end%s.\n"
                % (leg["mdp"], ps / 1000.0, leg["lam_from"],
                   (",\n; and so does %s, the first reverse bound leg (fe_protocol.legs())"
                    % " and ".join(later)) if later else ""))
        return (
            "; %s - %g ns equilibrium hold at u = %g, lambda = %g.\n"
            "; delta-lambda = 0 and pull rate 0, so it contributes NO work and the stage\n"
            "; works still sum exactly to the work of the whole ramp. It exists so the\n"
            "; stages either side of it can be estimated SEPARATELY and their estimates\n"
            "; added, which is only legitimate if the ensemble here is equilibrated:\n"
            "; summing their per-cycle works instead would reproduce the unstaged\n"
            "; distribution exactly and gain nothing.\n"
            % (leg["mdp"], ps / 1000.0, leg["u_from"], leg["lam_from"]))
    if leg["kind"] == "bound":
        return (
            "; %s - Bound-state restraint %s (dhdl only, no pulling).\n"
            "; Interface restraints switched %s with the partners in place, lambda %g -> %g.\n"
            % (leg["mdp"], "introduction" if leg["dirn"] == "fwd" else "removal",
               "ON (k=0 -> kB=full)" if leg["dirn"] == "fwd" else "OFF",
               leg["lam_from"], leg["lam_to"]))
    return "; %s\n" % leg["mdp"]


NPT_HEADER = (
    "; npt_fe.mdp - %g ns NPT of the bound state with no pull code: the physics of the\n"
    "; per-cycle npt_c, which since v1.5 runs npt_c_fe.mdp, i.e. this file plus the\n"
    "; elastic network if the structure has one. Kept as the pull-free reference that\n"
    "; file is checked against.\n"
    "; The Crooks analysis assumes the states boundfwd1 starts from are drawn from the\n"
    "; true NPT ensemble, so it uses C-rescale and runs straight off the NVT ladder\n"
    "; (C-rescale is first order and needs no weak-coupling stage in front of it).\n"
    % (P.NPT_PS / 1000.0))

# WHY npt_c CARRIES THE ELASTIC NETWORK. boundfwd1 starts from the end of npt_c, and
# its Crooks pair assumes that start is an equilibrium sample of H(lambda = 0) of the
# bound block: interface springs off, no Boresch, and the elastic network (EN, the
# intra-partner pull coordinates at k = kB, independent of lambda) ON. Through v1.4
# npt_c ran npt_fe.mdp with no pull code, so the EN came on at full strength at t = 0
# of boundfwd1: a median 299 kJ/mol (up to 33,549) never recorded, and in 20 of 920
# pass-2 cycles a cutout fragment held only by the EN had drifted so far during the
# 20 ns that its pair passed 0.49 times the box and the boundfwd1 grompp refused it
# (bm_fe_amber_2/_ana/digest/fu3.txt F10, qc.txt npt_c_elastic_network_off).
#
# The EN is per structure, so this template carries only the pull print header the
# leg mdps carry; make_boresch.py appends the EN-only block (the same groups, pairs
# and references the legs use, rate 0), or sets pull = no for a structure without
# EN pairs, which then runs exactly the npt_fe physics. npt_init_fe.mdp stays
# pull-free: the EN is built from the probe it drives.
NPT_C_HEADER = (
    "; npt_c_fe.mdp - %g ns NPT of the bound state at the start of every FE cycle,\n"
    "; with the elastic network on if the structure has one, i.e. H(lambda = 0) of the\n"
    "; bound legs, so that boundfwd1 starts in equilibrium with the Hamiltonian it runs.\n"
    "; The physics is npt_fe.mdp's. make_boresch.py appends the network's pull block to\n"
    "; the structure's copy, or sets pull = no there for a structure without network\n"
    "; pairs (2KTF), which then runs npt_fe.mdp's physics unchanged; job_fe.run runs\n"
    "; that copy, ./npt_c_fe.mdp.\n"
    % (P.NPT_PS / 1000.0))

NPT_INIT_HEADER = (
    "; npt_init_fe.mdp - %g ns NPT equilibration of one probe replica, in the small\n"
    "; probe box; the setup runs N_PROBES = %d of them, once per structure rather than\n"
    "; once per cycle. make_boresch.py discards the first %g ns of each\n"
    "; (PROBE_SKIP_FRAC), which also absorbs the C-rescale box relaxation off the NVT\n"
    "; ladder, and takes the backbone RMSF behind the Boresch anchor choice and the\n"
    "; restraint references from the rest, sampled every %g ps. measure_box.py sizes\n"
    "; the production box from every frame.\n"
    % (P.NPT_INIT_PS / 1000.0, P.N_PROBES, P.NPT_INIT_PS * P.PROBE_SKIP_FRAC / 1000.0,
       P.XOUT_PS_INIT))


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

def pull_header(pull_fout):
    """The pull print header every mdp make_boresch.py appends a block to carries.
    One definition, so the legs and npt_c_fe.mdp cannot drift apart."""
    return ("\n; ---- COM pulling (pull block appended by make_boresch.py) ----\n"
            "pull                     = yes\n"
            "pull-print-com1          = no\n"
            "pull-print-com2          = no\n"
            "pull-print-ref-value     = no\n"
            "pull-print-components    = no\n"
            "pull-nstxout             = 0\n"
            "pull-nstfout             = %d\n" % pull_fout)


def build(text, ps, lam_from, lam_to, pull_fout, xout_ps):
    """One leg's mdp body, given its base text."""
    dt = float(get_param(text, "dt"))
    nsteps = int(round(ps / dt))

    text = strip_pull_block(text)
    text = set_param(text, "nsteps", nsteps)
    if get_param(text, "pcoupl") not in (None, "no"):
        text = set_param(text, "pcoupl", BAROSTAT_ALL[0])
        text = set_param(text, "tau_p", BAROSTAT_ALL[1])
    text = set_param(text, "nstxout-compressed", int(round(xout_ps / dt)))

    if lam_from is None:                              # not a free-energy leg
        return text.rstrip("\n") + "\n", nsteps, dt

    text = set_param(text, "nstcalcenergy", 100)      # nstdhdl must be a multiple
    # delta-lambda is PER STEP and the endpoints are fractional, so a stage that
    # covers 0 -> 0.3 must land exactly on 0.3 and the next stage start there. A
    # hold has lam_from == lam_to and gets 0 for free, which is also what makes it
    # contribute no work.
    #
    # WHY %.12e AND NOT %.8e. The drift at the endpoint is nsteps times whatever
    # the last digit rounds away, so the longer the leg the worse it gets: at
    # 8 significant digits and 1.3M steps stage E landed on 1.0 to within 5e-10,
    # and doubling the leg to 2.6M steps took that to 1e-9 and tripped the span
    # check in tests/test_staged.py. Nothing was wrong with the leg -- the format
    # was simply too narrow to survive a longer one, which is the same failure
    # pull-coordN-rate had at %.8f, where each stage overshot its end reference by
    # 4e-5 nm. Written wide enough that leg length cannot reach it.
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
          % (("%g" % lam_from), "0" if dl == 0 else ("%.12e" % dl)))
    return text.rstrip("\n") + "\n" + fe + pull_header(pull_fout), nsteps, dt


def main():
    legs = P.legs()
    for ff in FFS:
        d = os.path.join(SET, ff)
        if not os.path.isdir(d):
            print("skip missing %s" % ff); continue

        written = 0
        for leg in legs:
            key = ("stage_%s" % leg["dirn"]) if leg["kind"] == "stage" else leg["kind"]
            base_path = os.path.join(d, BASE[key])
            if not os.path.isfile(base_path):
                print("  %s: missing base %s, skipped" % (ff, BASE[key])); continue
            body, nsteps, dt = build(read(base_path), leg["ps"],
                                     leg["lam_from"], leg["lam_to"],
                                     P.PULL_NSTFOUT if leg["pull"] else 0,
                                     P.XOUT_PS)
            head = header_for(leg) + ";\n; nsteps %d * dt %g = %g ps.\n\n" % (nsteps, dt, leg["ps"])
            with open(os.path.join(d, leg["mdp"]), "w") as f:
                f.write(head + body)
            written += 1

        # The equilibrations carry no lambda. npt_c_fe.mdp is npt_fe.mdp plus the
        # pull print header, for the elastic network (see NPT_C_HEADER); the other
        # two carry no pull at all.
        for name, ps, xout, head, pull in (
                ("npt_fe.mdp", P.NPT_PS, P.XOUT_PS, NPT_HEADER, False),
                ("npt_c_fe.mdp", P.NPT_PS, P.XOUT_PS, NPT_C_HEADER, True),
                ("npt_init_fe.mdp", P.NPT_INIT_PS, P.XOUT_PS_INIT, NPT_INIT_HEADER, False)):
            base_path = os.path.join(d, BASE["npt"])
            if not os.path.isfile(base_path):
                print("  %s: missing base %s, skipped" % (ff, BASE["npt"])); continue
            body, nsteps, dt = build(read(base_path), ps, None, None, 0, xout)
            if pull:
                body += pull_header(0)
            with open(os.path.join(d, name), "w") as f:
                f.write(head + ";\n; nsteps %d * dt %g = %g ps.\n\n" % (nsteps, dt, ps) + body)
            written += 1

        print("generated %d FE mdps for %s" % (written, ff))

    print("")
    # cycle_ps() already counts npt_c, so the legs are the difference.
    print("cycle: %g ns over %d legs plus %g ns of npt_c, %g ns in all"
          % ((P.cycle_ps() - P.NPT_PS) / 1000.0, len(legs), P.NPT_PS / 1000.0,
             P.cycle_ps() / 1000.0))
    print("stale mdps from a previous protocol are NOT removed; run")
    print("  git status settings/  to see anything left behind by a reshaped ramp.")


if __name__ == "__main__":
    main()
