#!/usr/bin/env python3
"""Free-energy estimators shared by groscore.py and groscore_fe.py.

Currently the Bennett Acceptance Ratio (BAR) and the overlap statistic that
gates it. The Crooks average and CGI stay where they are; see tests/test_cgi.py
for why those three copies are pinned by test rather than shared.

CONVENTIONS, stated once, because getting either wrong is silent.

  Sign. `fwd` is the forward work of a leg and `rev` is the reverse work IN ITS
  OWN SIGN, NOT negated. That is the convention the Crooks relation is written
  in, P_f(W) / P_r(-W) = exp((W - dG) / RT), and it is the convention
  groscore_fe.py already stores (its `rev` arrays are raw; _stream_avg negates
  internally). It is also pymbar's, verified empirically rather than from its
  docstring, which is silent, and against bar_zero's header formula, which is
  written for the negated quantity and contradicts its own code.

  groscore.py is the exception: its `pushes` are already sign-ALIGNED, because
  job.run calls integrate.py without -r for both directions so the reversed
  force integral flips the sign for free. That call site must pass -pushes. The
  negation is done there rather than here so this module has exactly one
  convention and the asymmetry stays visible where it originates.

  Units. Works are kJ/mol and `rt` is RT in kJ/mol. The solver reduces to kT
  internally and converts back, so callers never see kT.

Nothing here raises, and nothing here returns a number with a caveat attached.
Every entry point returns either a value and an empty reason, or NaN and a
reason token. There is deliberately no third state: a BAR value carrying a flag
still gets plotted and regressed by everything that does not read the flag.
"""

import numpy as np
from scipy.special import logsumexp

# Three cycles, matching N_MIN_OVL and the ncyc >= 3 gate CGI already uses in
# groscore_fe.py, so the two estimators agree about when there is enough data.
BAR_MIN_N = 3

# Bisection halves the bracket each pass, so 45 passes take a 1000 kT bracket to
# 3e-11 kT. Fixed count with no early exit: an early exit needs per-row
# branching, which would cost the vectorisation that makes the bootstrap
# affordable at all.
_BISECT_ITERS = 45


def overlap_count(fwd, rev):
    """How many of the 2n works lie inside the OTHER direction's observed range.

    Order statistics only, no distributional assumption, which is the point:
    this gates BAR, and BAR assumes no distribution either. Returns 0 when the
    forward and sign-aligned reverse works are cleanly separated, which is the
    condition under which every bidirectional estimator is extrapolating rather
    than measuring.
    """
    f = np.asarray(fwd, dtype=float).ravel()
    r = -np.asarray(rev, dtype=float).ravel()          # sign-aligned reverse
    if f.size == 0 or r.size == 0:
        return 0
    if not (np.isfinite(f).all() and np.isfinite(r).all()):
        return 0
    return int(((f >= r.min()) & (f <= r.max())).sum()
               + ((r >= f.min()) & (r <= f.max())).sum())


def _fzero(dF, WF, WR, M):
    """Bennett's implicit function, vectorised over rows. Zero at the BAR root.

    This is pymbar's bar_zero. It is MONOTONICALLY INCREASING in dF, which is
    what lets the bisection below be branch-free: fzero > 0 means dF is too
    large. WF/WR are (n_rows, n_samples) in kT; dF is (n_rows,) in kT.
    """
    d = np.atleast_1d(dF)[:, None]
    lf = -np.logaddexp(0.0, M + WF - d)
    lr = -np.logaddexp(0.0, -(M - WR - d))
    return logsumexp(lf, axis=1) - logsumexp(lr, axis=1)


def _bar_rows_kt(WF, WR):
    """BAR for every row of WF/WR, in kT. NaN where the root is not bracketed.

    WF, WR: (n_rows, n_samples) arrays of forward and reverse work in kT, the
    reverse in its own sign.
    """
    WF = np.atleast_2d(WF)
    WR = np.atleast_2d(WR)
    nF, nR = WF.shape[1], WR.shape[1]
    M = np.log(float(nF) / float(nR))

    lo_seed = np.minimum(WF.min(1), -WR.max(1))
    hi_seed = np.maximum(WF.max(1), -WR.min(1))
    span = hi_seed - lo_seed
    lo = lo_seed - 10.0 - span
    hi = hi_seed + 10.0 + span

    # A bracket that does not straddle the root would give a confident wrong
    # answer, which is the one outcome this module exists to prevent. Two extra
    # evaluations out of 47.
    with np.errstate(invalid="ignore"):
        bad = ~((_fzero(lo, WF, WR, M) < 0) & (_fzero(hi, WF, WR, M) > 0))
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            z = _fzero(mid, WF, WR, M)
            hi = np.where(z > 0, mid, hi)
            lo = np.where(z > 0, lo, mid)

    out = 0.5 * (lo + hi)
    out[bad] = np.nan
    return out


def _validate(f, r, min_n, require_overlap):
    """Shared precondition check. Returns a reason token, or "" if the inputs
    are usable.

    Ordered cheapest and most decisive first, and run BEFORE any solve, so a
    zero-overlap leg costs nothing. That ordering is what makes the classic
    engine, where essentially every leg is rejected here, effectively free.
    """
    if f.size != r.size:
        return "BAR_UNPAIRED"
    if not (np.isfinite(f).all() and np.isfinite(r).all()):
        return "BAR_NONFINITE"
    if f.size < min_n:
        return "BAR_N_LT_%d" % min_n
    if require_overlap and overlap_count(f, r) == 0:
        return "BAR_NO_OVERLAP"
    return ""


def bar(fwd, rev, rt, min_n=BAR_MIN_N, require_overlap=True):
    """BAR for one leg. Returns (value_kJ_per_mol, reason).

    reason is "" on success, and on failure the value is NaN and reason is one
    of BAR_UNPAIRED, BAR_NONFINITE, BAR_N_LT_<n>, BAR_NO_OVERLAP,
    BAR_UNBRACKETED.

    require_overlap defaults True and should stay True for anything written to a
    score file. None of the catastrophic inputs is detectable from the solver's
    own output: identical works with no overlap return a confident finite value,
    a single cycle returns a finite value, and real data at 132 RT dissipation
    returns a finite value with the wrong sign. The solver brackets cleanly in
    every one of those cases. overlap_count == 0 is the only check that catches
    them, which is why it is not optional in practice.
    """
    f = np.asarray(fwd, dtype=float).ravel()
    r = np.asarray(rev, dtype=float).ravel()
    reason = _validate(f, r, min_n, require_overlap)
    if reason:
        return float("nan"), reason
    v = _bar_rows_kt(f[None, :] / rt, r[None, :] / rt)[0] * rt
    if not np.isfinite(v):
        return float("nan"), "BAR_UNBRACKETED"
    return float(v), ""


def bar_bootstrap(fwd, rev, rt, idx, require_overlap=True):
    """One BAR per bootstrap row, in kJ/mol. NaN where a row is unusable.

    idx is an (n_rows, n_cycles) index array. Pass a PREFIX of the caller's
    existing shared index rather than drawing a new one: in groscore_fe.py the
    same index resamples every leg, so slicing it keeps the intro/unbind pairing
    that makes the combined dG_bind interval carry their covariance.

    A resample can lose overlap even when the full sample has it, so the overlap
    test is applied per row and the losers become NaN. Callers use np.nanstd,
    exactly as the CGI block already does.
    """
    f = np.asarray(fwd, dtype=float).ravel()
    r = np.asarray(rev, dtype=float).ravel()
    if _validate(f, r, 1, False):
        return np.full(np.shape(idx)[0], np.nan)

    WF = f[idx] / rt
    WR = r[idx] / rt
    out = _bar_rows_kt(WF, WR) * rt

    if require_overlap:
        # Same order statistic as overlap_count, vectorised: two comparisons
        # over the array, negligible beside 45 logaddexp passes.
        A = f[idx]
        B = -r[idx]
        amin, amax = A.min(1)[:, None], A.max(1)[:, None]
        bmin, bmax = B.min(1)[:, None], B.max(1)[:, None]
        n_in = (((A >= bmin) & (A <= bmax)).sum(1)
                + ((B >= amin) & (B <= amax)).sum(1))
        out = np.where(n_in == 0, np.nan, out)
    return out
