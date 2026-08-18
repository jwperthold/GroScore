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

THE CYCLE IS STAGED IN BOTH HALVES. The bound-state restraint switch runs in BOUND
sub-legs and the unbinding pull in RAMP stages, with an equilibrium hold at every
internal boundary in both directions. Each sub-leg is its own Crooks process; the
holds do zero work, so the sub-leg works still sum exactly to the work of the whole
switch, and each sum is scored alongside the staged one as the assumption-free
cross-check.

WHY STAGE AT ALL. A leg whose dissipation exceeds its own work width has forward and
reverse histograms that do not meet, and then BAR returns nothing whatever the
sampling. Splitting partitions the dissipation into pieces each estimator can
handle, and -- because each stage runs at CONSTANT rate -- it also lowers the total,
since a slow region can then be crossed slowly and a fast one quickly.

WHY THESE BOUNDARIES, AND WHY FIVE. The friction profile is recovered as zeta = g/v,
because the stages ran at rates differing 8x and the raw dissipation density is not
zeta. It is measured on ALL THREE repeats and pooled, not on one of them: the
boundary is a property of the setup draw, and a cycle bootstrap inside a single run
understates its uncertainty by 4-9x.

              b1      b2      b3      b4      b5
  test10   0.164   0.218   0.295   0.419   0.630
  test11   0.094   0.160   0.241   0.355   0.535
  test12   0.088   0.156   0.230   0.330   0.518
  sd_within 0.005  0.005   0.008   0.009   0.016     <- cycle bootstrap
  sd_betwn  0.042  0.035   0.035   0.046   0.061     <- between runs

Boundaries sit at equal dissipation per stage, which for constant-rate stages is
equal sqrt(du * int zeta du), and the times follow the same quantity. Equalising the
dissipation makes the times come out equal too, which is why every stage is 5200 ps.

FIVE AND NOT SIX. Every boundary costs a hold in each direction out of a fixed 80 ns
of legs, so the ramp shrinks by 1 ns per boundary added. Priced at that fixed budget
on the pooled friction:

  N   ramp ns  D total  W/stage  n_sig  sd_sum  P(fail)  LOO penalty
  3      27.0     63.3     21.1   1.26    4.12       0%        13.6%
  4      26.5     61.2     15.3   1.07    3.75       0%        20.0%
  5      26.0     60.3     12.1   0.95    3.50       0%        23.6%
  6      25.5     62.0     10.3   0.88    3.54       0%        27.4%
  7      25.0     63.0      9.0   0.82    3.58       0%        28.6%

Total dissipation and summed BAR variance both bottom out at N = 5 and get worse
after, because past that the holds cost more ramp time than the finer partition
saves. The LOO column is the price of transfer: boundaries fitted on two runs,
scored on the worst stage of the third, against boundaries fitted on the third
itself. It climbs monotonically, so each added boundary generalises less well than
the last. Six stages cost 1 ns of ramp, 2.8% more dissipation and one more boundary
to overfit, and returned nothing. Widths are calibrated rather than assumed:
kappa = sigma^2/(2 RT W) measures 2.57 over the nine run/stage pairs, so the works
are wider than linear response and overlap is easier than it would otherwise be.

WHY NOT JUST RETIME. Because that is exhausted. The previous 17/3.5/3.5 split was
already within 5% of optimal against the measured friction, and reallocating between
the bound leg and the ramp is flat from 5 to 13 ns per direction against a Monte
Carlo noise of 0.1, so the shipped 7.5 sits inside the flat region. Crooks also
forbids the obvious response to noisy rebinding -- more time in reverse than
forward -- since the reverse protocol must be the exact time-reverse of the
forward one.

HOLD LENGTHS ARE MEASURED. See RAMP_HOLD_PS below: dH/dlambda is written during a
hold, so its autocorrelation time and its residual drift are both observable, and
1000 ps was 60-90 tau everywhere except the two bound<->ramp handoffs.

THE PROBE IS REPLICATED. N_PROBES independent equilibrations, everything measured on
the pooled second half of all of them. This is the only part of the design that
touches the between-run scatter; see N_PROBES.

100 ns per cycle, plus the 0.1 ns NVT ladder job_fe.run runs ahead of it.
"""
import sys

# ---------------------------------------------------------------- the definition

PULL_DIST = 1.0          # nm of COM-COM separation added over the whole ramp

# (stage letter, u_from, u_to, ps). Must be contiguous, start at 0 and end at
# PULL_DIST. Adding or moving a boundary is an edit to THIS LIST and nothing else.
RAMP = [
    ("A", 0.0,   0.118, 5200.0),
    ("B", 0.118, 0.199, 5200.0),
    ("C", 0.199, 0.306, 5200.0),
    ("D", 0.306, 0.494, 5200.0),
    ("E", 0.494, 1.000, 5200.0),
]

# THE BOUND LEG IS STAGED TOO, but NOT because it needs the overlap. Pooled over the
# three repeats it dissipates 10.8 kJ/mol unsplit against a work width of 23, i.e.
# 0.46 sigma, nowhere near the cliff the ramp was at. What the split buys is
# variance: about 10% off the summed BAR error (2.89 -> 2.61), and a third sub-leg
# buys nothing further (2.61) while costing 2 ns. So two, for variance alone.
#
# The boundary stays at lambda = 0.25. The pooled optimum is 0.277, but the per-run
# estimates are 0.186 / 0.226 / 0.388, sd 0.107, which is three times noisier than
# any ramp boundary and not resolvable; and the difference costs nothing measurable,
# 0.90/0.94 sigma at 0.25 against 0.93/0.92 at the optimum.
#
# WHAT THE BOUND LEG'S WIDTH ACTUALLY IS. At the same 5 ns leg and the same sum_k,
# the three repeats give dissipation 15.7 / 17.8 / 21.4 against work widths of
# 19.9 / 42.4 / 58.9. A 3x range in width across a 1.4x range in dissipation is not
# a rate effect, so no leg time, boundary or sub-leg count reaches it. N_PROBES is
# the only part of the design that does, and this is where it should show first,
# since the interface reference geometry is exactly what the probe measures.
#
# (sub-leg name, lambda_from, lambda_to, ps) for the FORWARD direction; the reverse
# runs them backwards, exactly as the ramp does.
BOUND = [
    ("1", 0.0,  0.25, 3750.0),
    ("2", 0.25, 1.0,  3750.0),
]

# HOLD LENGTHS ARE MEASURED, NOT ASSUMED. dH/dlambda is written during a hold, so
# how long one needs is a question the data answers: the autocorrelation time is
# 11-16 ps on the ramp and 39 ps at u = 1, the relaxation profiles are flat after the
# first tenth, and the end-of-hold state does not predict the next stage's work
# (r = +0.06 over 49 cycles). The 1000 ps holds were 60-90 tau.
#
# The exception is the bound <-> ramp handoff. holdfwd0 and holdrev0 settle at 500
# and 700 ps, about 40 tau, where the purely mid-ramp holds settle in 0-300. They
# cross a change of restraint regime rather than a lambda step, so they keep 1 ns.
# The bound leg's own internal hold is a handoff of the same kind and keeps 1 ns too.
RAMP_HOLD_PS    = 500.0    # holds between ramp stages, both directions
HANDOFF_HOLD_PS = 1000.0   # holdfwd0 / holdrev0, and the bound leg's internal hold
UNBOUND_HOLD_PS = 5000.0   # the hold at u = PULL_DIST, between the two directions
NPT_PS          = 20000.0  # per-cycle equilibration of the bound state
NPT_INIT_PS     = 20000.0  # ONE probe replica; N_PROBES of them run per structure

# Independent probe equilibrations per structure. Every restraint and every anchor
# is measured on the POOLED last half of all of them, because the alternative is
# what three repeats of one protocol measured: dG_bind scattering 14.4 kJ/mol
# between runs against a within-run bootstrap of 4.4, since all 50 cycles of a run
# share one probe, one triad and one spring set and resampling cycles cannot see it.
N_PROBES        = 5
PROBE_SKIP_FRAC = 0.5      # discard this much of each replica before measuring

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
    # The bound leg is checked the same way, in lambda instead of u.
    if BOUND[0][1] != 0.0 or abs(BOUND[-1][2] - 1.0) > 1e-12:
        raise ValueError("the bound leg must run lambda 0 -> 1")
    for a, b in zip(BOUND, BOUND[1:]):
        if abs(a[2] - b[1]) > 1e-12:
            raise ValueError("bound sub-legs must be contiguous: %s ends at %g, %s starts at %g"
                             % (a[0], a[2], b[0], b[1]))
    if len({s[0] for s in BOUND}) != len(BOUND):
        raise ValueError("bound sub-leg names must be unique")
_check()

RAMP_PS = sum(s[3] for s in RAMP)
BOUND_PS = sum(s[3] for s in BOUND)

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
    # place. u is 0 throughout, and lambda runs the full 0 -> 1 in BOUND sub-legs,
    # with an equilibrium hold at each internal boundary exactly as on the ramp.
    for i, (name, l0, l1, ps) in enumerate(BOUND):
        if i:
            add("holdbfwd%d" % i, "holdbfwd%d_fe.mdp" % i, "hold", HANDOFF_HOLD_PS,
                0.0, 0.0, False, False, lam=(l0, l0))
        add("boundfwd%s" % name, "boundfwd%s.mdp" % name, "bound", ps, 0.0, 0.0,
            False, True, stage=name, dirn="fwd", lam=(l0, l1))

    # Forward: hold at every boundary the stage is about to leave, then the stage.
    # The first one is the bound -> ramp handoff and is longer; see HANDOFF_HOLD_PS.
    for i, (letter, u0, u1, ps) in enumerate(RAMP):
        add("holdfwd%d" % i, "holdfwd%d_fe.mdp" % i, "hold",
            HANDOFF_HOLD_PS if i == 0 else RAMP_HOLD_PS, u0, u0, False, False)
        add("bindfwd%s" % letter, "bindfwd%s_fe.mdp" % letter, "stage", ps, u0, u1,
            True, True, stage=letter, dirn="fwd")

    # The unbound hold, at the far end.
    add("nptrev", "nptrev_fe.mdp", "hold", UNBOUND_HOLD_PS, PULL_DIST, PULL_DIST, False, False)

    # Reverse: mirror image, stages in the opposite order.
    for i in range(len(RAMP) - 1, -1, -1):
        letter, u0, u1, ps = RAMP[i]
        if i < len(RAMP) - 1:
            add("holdrev%d" % (i + 1), "holdrev%d_fe.mdp" % (i + 1), "hold",
                RAMP_HOLD_PS, u1, u1, False, False)
        add("bindrev%s" % letter, "bindrev%s_fe.mdp" % letter, "stage", ps, u1, u0,
            True, True, stage=letter, dirn="rev")
    add("holdrev0", "holdrev0_fe.mdp", "hold", HANDOFF_HOLD_PS, 0.0, 0.0, False, False)

    # Reverse bound leg: the sub-legs run backwards, each retracing its own lambda
    # span, so the whole cycle is the exact time-reverse Crooks requires.
    for i in range(len(BOUND) - 1, -1, -1):
        name, l0, l1, ps = BOUND[i]
        if i < len(BOUND) - 1:
            add("holdbrev%d" % (i + 1), "holdbrev%d_fe.mdp" % (i + 1), "hold",
                HANDOFF_HOLD_PS, 0.0, 0.0, False, False, lam=(l1, l1))
        add("boundrev%s" % name, "boundrev%s.mdp" % name, "bound", ps, 0.0, 0.0,
            False, True, stage=name, dirn="rev", lam=(l1, l0))
    return out

def cycle_ps():
    return NPT_PS + sum(l["ps"] for l in legs())

def works():
    """The work fields of a result row, in order.

    (field, leg, channel, direction). The pull channel of a stage is integrated
    at that stage's own rate and the dhdl channel over that stage's own lambda
    span; both come from the leg's mdp, which is why the leg name is carried
    here rather than the numbers.

    The row is laid out so that a sub-leg's forward field and its reverse field are
    mirror images about the centre: bound-forward fields, then the ramp stages
    forward then reverse, then bound-reverse. Sub-leg j of either group therefore
    pairs with the j-th field counted from its group's other end, which is the rule
    read_works uses and the only reason the reverse ordering is not arbitrary.
    """
    L = legs()
    out = [("W_intro%s" % l["stage"], l["name"], "dhdl", "fwd")
           for l in L if l["kind"] == "bound" and l["dirn"] == "fwd"]
    for l in L:
        if l["kind"] != "stage":
            continue
        tag = "u" if l["dirn"] == "fwd" else "r"
        out.append(("W%s%s_pull" % (tag, l["stage"]), l["name"], "pull", l["dirn"]))
        out.append(("W%s%s_dhdl" % (tag, l["stage"]), l["name"], "dhdl", l["dirn"]))
    out += [("W_remove%s" % l["stage"], l["name"], "dhdl", "rev")
            for l in L if l["kind"] == "bound" and l["dirn"] == "rev"]
    return out

def n_bound():
    """Bound sub-legs per direction."""
    return len(BOUND)

def n_stages():
    """Ramp stages per direction."""
    return len(RAMP)

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
    # Setup shape, so job_fe.run cannot disagree with the protocol about how
    # many probe replicas to run or how much of each to discard.
    print("N_PROBES=%d" % N_PROBES)
    print("FE_PROBE_PS=%g" % NPT_INIT_PS)
    print("FE_PROBE_SKIP_PS=%g" % (NPT_INIT_PS * PROBE_SKIP_FRAC))
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
