#!/usr/bin/env python3
"""The FE cycle, defined once.

Everything downstream derives from the table at the top of this file: the leg
mdps (utils/make_fe_mdps.py), the pull blocks and per-stage rates
(utils/make_boresch.py), the leg sequence and work extraction (job_fe.run, via
--shell), the result-row layout and the staged estimator (groscore_fe.py), and
the leg diagnostic (utils/fe_leg_efficiency.py).

It exists because the ramp has been reshaped three times in two days, and each
time the shape lived as four parallel hardcoded lists that had to be kept in
step by hand. Every one of those lists is now computed from RAMP.

    python3 utils/fe_protocol.py            human-readable summary of the cycle
    python3 utils/fe_protocol.py --shell    the same as shell arrays, for job_fe.run

THE RAMP. The unbinding pull runs from u = 0 to u = PULL_DIST in stages, with an
equilibrium hold at every internal boundary in both directions. Each stage is its
own Crooks process; the holds do zero work, so the stage works still sum exactly
to the work of the whole ramp, and the summed value is scored alongside the
staged one as the assumption-free cross-check.

WHY THESE BOUNDARIES AND TIMES (measured on 50 cycles of 2KTF, test3):

  u 0.0-0.3   40.07 kJ/mol in 15 ns   local rate response p = 0.69
  u 0.3-0.5   19.35 kJ/mol in 1.43 ns                     p = 1.05
  u 0.5-1.0   19.16 kJ/mol in 3.57 ns                     p = 0.07

The middle zone is the only part of the pull where time converts to reduced
dissipation at the full near-equilibrium 1/t rate, and the two-stage split put it
in the FAST stage, where it received 1.43 ns and paid 19.35 kJ/mol. The outer
half barely responds to rate at all: running it 2.8x faster than test2 cost
1.26 kJ/mol across five windows. So the boundary at u = 0.5 separates the part
worth slowing from the part that is not.

Minimising sum(c_i * t_i^-p_i) at fixed total 20 ns, with the tail held at the
fastest rate ever actually run (0.14 nm/ns, hence 3.5 ns for its 0.5 nm), gives
13 / 3.5 / 3.5 ns. Predicted 70.9 kJ/mol against the 78.6 measured, i.e. about
7.7 kJ/mol for free. The gain REQUIRES taking time from stage A: hold A at 15 ns
and the remaining 5 ns cannot beat what stage B already gets, so the saving is
exactly the transfer from a p = 0.69 zone to a p = 1.05 one.

Those exponents are two-point estimates across runs that differ on six axes, so
treat the allocation as the best available guess and not as an optimum.
"""
import sys

# ---------------------------------------------------------------- the definition

PULL_DIST = 1.0          # nm of COM-COM separation added over the whole ramp

# (stage letter, u_from, u_to, ps). Must be contiguous, start at 0 and end at
# PULL_DIST. Adding or moving a boundary is an edit to THIS LIST and nothing else.
RAMP = [
    ("A", 0.0, 0.3, 13000.0),
    ("B", 0.3, 0.5,  3500.0),
    ("C", 0.5, 1.0,  3500.0),
]

HOLD_PS         = 1000.0   # every equilibrium hold on the ramp, both directions
UNBOUND_HOLD_PS = 4000.0   # the hold at u = PULL_DIST, between the two directions
BOUND_PS        = 2000.0   # boundfwd / boundrev, the restraint switch (no pull)
NPT_PS          = 10000.0  # per-cycle equilibration of the bound state
NPT_INIT_PS     = 11000.0  # setup equilibration, once per structure

# Trajectory frame interval, ps. The setup equilibration is the input to the
# anchor selection and needs frames to measure a backbone RMSF from; nothing else
# needs more than a handful.
XOUT_PS         = 400.0
XOUT_PS_INIT    = 10.0
PULL_NSTFOUT    = 500      # on the switching stages only

# ---------------------------------------------------------------- derived

def boundaries():
    """[0.0, ...internal..., PULL_DIST] in nm."""
    return [RAMP[0][1]] + [s[2] for s in RAMP]

def _check():
    if RAMP[0][1] != 0.0:
        raise ValueError("the ramp must start at u = 0")
    if abs(RAMP[-1][2] - PULL_DIST) > 1e-12:
        raise ValueError("the ramp must end at u = PULL_DIST")
    for a, b in zip(RAMP, RAMP[1:]):
        if abs(a[2] - b[1]) > 1e-12:
            raise ValueError("ramp stages must be contiguous: %s ends at %g, %s starts at %g"
                             % (a[0], a[2], b[0], b[1]))
    if len({s[0] for s in RAMP}) != len(RAMP):
        raise ValueError("stage letters must be unique")
_check()

RAMP_PS = sum(s[3] for s in RAMP)

def legs():
    """The whole cycle in run order.

    Each entry is a dict with:
      name      leg name, also the file prefix every output of that leg carries
      mdp       the mdp it runs with
      kind      npt | bound | stage | hold
      ps        simulated time
      u_from/u_to, lam_from/lam_to   lambda tracks u/PULL_DIST, so the force-constant
                switching stays proportional to the distance travelled and the
                stages join continuously
      pull      whether the leg writes a pull-force xvg (switching stages only)
      dhdl      whether the leg writes a dhdl xvg
      stage     the stage letter for a switching stage, else None
      dirn      'fwd' | 'rev' | None
    """
    out = []
    def add(name, mdp, kind, ps, u0, u1, pull, dhdl, stage=None, dirn=None, lam=None):
        # On the ramp, lambda tracks u so the force-constant switching stays
        # proportional to the distance travelled. The bound legs are the exception
        # and carry their own lambda, since they switch the restraints on and off
        # at u = 0 and their pull references never move.
        l0, l1 = lam if lam else (u0 / PULL_DIST, u1 / PULL_DIST)
        out.append(dict(name=name, mdp=mdp, kind=kind, ps=ps, u_from=u0, u_to=u1,
                        lam_from=l0, lam_to=l1,
                        pull=pull, dhdl=dhdl, stage=stage, dirn=dirn))

    # Bound state: the interface restraints are switched on with the partners in
    # place. u is 0 throughout, and lambda runs the full 0 -> 1.
    add("boundfwd", "boundfwd.mdp", "bound", BOUND_PS, 0.0, 0.0, False, True,
        dirn="fwd", lam=(0.0, 1.0))

    # Forward: hold at every boundary the stage is about to leave, then the stage.
    for i, (letter, u0, u1, ps) in enumerate(RAMP):
        add("holdfwd%d" % i, "holdfwd%d_fe.mdp" % i, "hold", HOLD_PS, u0, u0, False, False)
        add("bindfwd%s" % letter, "bindfwd%s_fe.mdp" % letter, "stage", ps, u0, u1,
            True, True, stage=letter, dirn="fwd")

    # The unbound hold, at the far end.
    add("nptrev", "nptrev_fe.mdp", "hold", UNBOUND_HOLD_PS, PULL_DIST, PULL_DIST, False, False)

    # Reverse: mirror image, stages in the opposite order.
    for i in range(len(RAMP) - 1, -1, -1):
        letter, u0, u1, ps = RAMP[i]
        if i < len(RAMP) - 1:
            add("holdrev%d" % (i + 1), "holdrev%d_fe.mdp" % (i + 1), "hold",
                HOLD_PS, u1, u1, False, False)
        add("bindrev%s" % letter, "bindrev%s_fe.mdp" % letter, "stage", ps, u1, u0,
            True, True, stage=letter, dirn="rev")
    add("holdrev0", "holdrev0_fe.mdp", "hold", HOLD_PS, 0.0, 0.0, False, False)

    add("boundrev", "boundrev.mdp", "bound", BOUND_PS, 0.0, 0.0, False, True,
        dirn="rev", lam=(1.0, 0.0))
    return out

def cycle_ps():
    return NPT_PS + sum(l["ps"] for l in legs())

def works():
    """The work fields of a result row, in order.

    (field, leg, channel, direction). The pull channel of a stage is integrated
    at that stage's own rate and the dhdl channel over that stage's own lambda
    span; both come from the leg's mdp, which is why the leg name is carried
    here rather than the numbers.
    """
    out = [("W_intro", "boundfwd", "dhdl", "fwd")]
    for l in legs():
        if l["kind"] != "stage":
            continue
        tag = "u" if l["dirn"] == "fwd" else "r"
        out.append(("W%s%s_pull" % (tag, l["stage"]), l["name"], "pull", l["dirn"]))
        out.append(("W%s%s_dhdl" % (tag, l["stage"]), l["name"], "dhdl", l["dirn"]))
    out.append(("W_remove", "boundrev", "dhdl", "rev"))
    return out

def result_nf():
    """Fields in one results_fe.d row: id, cycle, the works, the RMSD."""
    return 2 + len(works()) + 1

def stage_letters():
    return [s[0] for s in RAMP]

# ---------------------------------------------------------------- reporting

def _summary():
    b = boundaries()
    print("FE cycle: %d legs, %g ns per cycle (%g ns of ramp per direction)"
          % (len(legs()), cycle_ps() / 1000.0, RAMP_PS / 1000.0))
    print("boundaries at u = %s nm" % ", ".join("%g" % x for x in b))
    print("")
    print("  %-12s %-20s %-7s %8s %14s %14s %8s"
          % ("leg", "mdp", "kind", "ps", "u", "lambda", "rate nm/ps"))
    print("  " + "-" * 92)
    print("  %-12s %-20s %-7s %8g %14s %14s %8s"
          % ("npt_c", "npt_fe.mdp", "npt", NPT_PS, "0", "-", "-"))
    for l in legs():
        rate = ((l["u_to"] - l["u_from"]) / l["ps"]) if l["ps"] else 0.0
        print("  %-12s %-20s %-7s %8g %14s %14s %8s"
              % (l["name"], l["mdp"], l["kind"], l["ps"],
                 "%g -> %g" % (l["u_from"], l["u_to"]) if l["kind"] != "bound" else "0 (no pull)",
                 "%g -> %g" % (l["lam_from"], l["lam_to"]),
                 ("%.3e" % rate) if l["kind"] == "stage" else "0"))
    print("")
    print("  result row: %d fields" % result_nf())
    print("  " + " ".join(f for f, _, _, _ in works()))

def _shell():
    """Shell arrays for job_fe.run. Emitted rather than duplicated, so the runner
    cannot fall out of step with the protocol it is running."""
    L = legs()
    print("FE_RESULT_NF=%d" % result_nf())
    print("FE_RAMP_STAGES=(%s)" % " ".join(stage_letters()))
    print("FE_LEGS=(%s)" % " ".join(l["name"] for l in L))
    print("FE_LEG_MDP=(%s)" % " ".join(l["mdp"] for l in L))
    print("FE_LEG_PULL=(%s)" % " ".join("1" if l["pull"] else "0" for l in L))
    print("FE_LEG_DHDL=(%s)" % " ".join("1" if l["dhdl"] else "0" for l in L))
    # Every mdp a structure directory must hold, i.e. every one make_boresch.py
    # writes a pull block into.
    print("FE_LEG_MDPS=(%s)" % " ".join(sorted({l["mdp"] for l in L})))
    # field:leg:channel:direction, in row order.
    print("FE_WORKS=(%s)" % " ".join("%s:%s:%s:%s" % w for w in works()))
    print("FE_WORK_FIELDS=(%s)" % " ".join(f for f, _, _, _ in works()))

if __name__ == "__main__":
    if "--shell" in sys.argv:
        _shell()
    else:
        _summary()
