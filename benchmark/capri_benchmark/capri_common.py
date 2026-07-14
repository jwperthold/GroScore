#!/usr/bin/env python3
"""Shared loaders for CAPRI Score_set scoring analysis.

Joins GroScore output (per-target scores_avg.gs / scores_cgi.gs, at the repo
root) with the CAPRI benchmark quality/RMSD annotations (CAPRI/database/*.csv).

Conventions (matching the GroScore paper / thesis chapter 3):
  * GroScore: more negative = more favourable binding (best = rank 1).
  * Failed / un-scored simulations carry no score: they are ranked *last* (worse
    than every scored pose) in a random order among themselves (see
    _assign_rank_scores), so their arbitrary input order can't bias the metrics.
  * CAPRI stars: 0 incorrect, 1 acceptable (*), 2 medium (**), 3 high (***);
    "near-native" = acceptable or better (stars >= 1).
  * CSV L-/I-RMSD are in Angstrom; converted here to nm (/10) to match the
    thesis figures.
"""
import os
import csv
import numpy as np
from scipy.stats import rankdata

REPO_ROOT = "/home/jwperthold/GroScore"
DB_DIR = os.path.join(REPO_ROOT, "CAPRI", "database")
FAILED_SCORE = float("nan")   # failed / un-scored poses carry no score (ranked last)
RANK_SEED = 0                 # fixed seed -> reproducible random order among failed poses

# target directory name -> CAPRI database CSV basename (matches prepare_capri.py)
TARGET_CSV = {
    "T29": "U-T029.1", "T30": "U-T030.1", "T32": "U-T032.1", "T35": "U-T035.1",
    "T36": "U-T036.1", "T37": "U-T037.1", "T38": "U-T038.1", "T39": "U-T039.1",
    "T40_1": "U-T040.1", "T40_2": "U-T040.2", "T41": "U-T041.1", "T46": "U-T046.1",
    "T47": "U-T047.1", "T50": "U-T050.1", "T53": "U-T053.1", "T54": "U-T054.1",
    "T89": "U-T089.1", "T96": "U-T096.1", "T97": "U-T097.1",
}
# display / iteration order. "T40" is the merged original target (both trypsin
# interfaces of the double-headed API-A inhibitor — see join_target); the two split
# interfaces remain available individually as "T40_1" / "T40_2".
TARGETS = ["T29", "T30", "T32", "T35", "T36", "T37", "T38", "T39", "T40",
           "T41", "T46", "T47", "T50", "T53", "T54", "T89", "T96", "T97"]

STAR_LABEL = {0: "incorrect", 1: "acceptable", 2: "medium", 3: "high"}


def load_scores(path):
    """Read a GroScore scores_*.gs file.

    Returns (scores, numeric_ids):
      scores      : {pose_id: float}  — non-numeric statuses (BROKEN/nan/FAILED/
                    ENTANGLED/NODIR) map to FAILED_SCORE.
      numeric_ids : set of pose ids that had an actual numeric score.
    """
    scores, numeric = {}, set()
    if not os.path.isfile(path):
        return scores, numeric
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                try:
                    v = float(p[1])
                    if np.isnan(v):
                        scores[p[0]] = FAILED_SCORE
                    else:
                        scores[p[0]] = v
                        numeric.add(p[0])
                except ValueError:
                    scores[p[0]] = FAILED_SCORE
    return scores, numeric


def load_labels(csv_path):
    """Read a CAPRI database CSV. Returns {pose_id: dict(stars, irms_nm, lrms_nm, cls)}."""
    labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row["identification"].strip()

            def _f(k):
                try:
                    return float(row[k])
                except (ValueError, KeyError, TypeError):
                    return float("nan")

            try:
                stars = int(row["stars"])
            except (ValueError, KeyError, TypeError):
                stars = 0
            labels[pid] = dict(stars=stars,
                               irms_nm=_f("irms") / 10.0,
                               lrms_nm=_f("lrms") / 10.0,
                               cls=row.get("classification", "").strip())
    return labels


def _assign_rank_scores(rows, seed=RANK_SEED):
    """Give every row a `rank_score` used for *all* ranking (Top-k / enrichment / ROC).

    Scored poses rank by their GroScore (more negative = better). Failed / un-scored
    simulations carry no score and must rank *last* — worse than every scored pose,
    including scored poses with an unfavourable (positive) GroScore. A plain score of
    0.0 would not do that (it would outrank any pose scoring > 0). Instead each failed
    pose gets a rank_score strictly beyond the worst real score, in a random order
    among themselves (fixed seed -> reproducible), so their arbitrary input order can
    never bias the enrichment / ROC results. Mutates and returns `rows`.
    """
    scored = [r for r in rows if r.get("scored")]
    failed = [r for r in rows if not r.get("scored")]
    for r in scored:
        r["rank_score"] = r["score"]
    if failed:
        worst = max((r["score"] for r in scored), default=0.0)
        shuffled = np.random.default_rng(seed).permutation(len(failed))
        for r, k in zip(failed, shuffled):
            r["rank_score"] = worst + 1.0 + float(k)   # distinct, all > any real score
    return rows


# Targets whose CAPRI reference has TWO interfaces (a pose is near-native if it
# matches EITHER) -> (interface CSV basenames, GroScore score dir(s)). The same
# poses are assessed against two reference binding modes; their near-native sets
# are disjoint, so "quality = the better interface" reproduces the papers' union.
#   T37: JIP4 homodimer, two binding modes; one extracted structure -> one score dir.
#   T40: double-headed API-A inhibitor, two trypsin sites; two rigid-placed copies.
MULTI_INTERFACE = {
    "T37": (["U-T037.1", "U-T037.2"], ["T37"]),
    "T40": (["U-T040.1", "U-T040.2"], ["T40_1", "T40_2"]),
}


def _join_multi(target, targets_root, db_dir, score_file):
    """Merge a two-interface target (see MULTI_INTERFACE).

    Per pose: quality = the better interface assessment (higher stars; tie -> lower
    I-RMSD). Score = the mean of whichever score dir(s) were run -- for T40 the two
    dirs are rigid replicas of one complex (~0.0007 A RMSD) so they are averaged;
    T37 has a single score dir.
    """
    csvs, score_dirs = MULTI_INTERFACE[target]
    labs = [load_labels(os.path.join(db_dir, c + ".csv")) for c in csvs]
    score_paths = [os.path.join(targets_root, d, score_file) for d in score_dirs]
    loaded = [load_scores(p) for p in score_paths]

    ids = set()
    for lab in labs:
        ids |= set(lab)

    rows, numeric = [], set()
    for pid in ids:
        cands = [lab[pid] for lab in labs if pid in lab]
        best = max(cands, key=lambda d: (d["stars"],
                   -(d["irms_nm"] if np.isfinite(d["irms_nm"]) else float("inf"))))
        cand_scores = [sc[pid] for sc, num in loaded if pid in num]
        if cand_scores:
            score = float(np.mean(cand_scores))   # replicas / single run -> mean
            numeric.add(pid)
            has_score = True
        else:
            score = FAILED_SCORE
            has_score = False
        rows.append(dict(id=pid, score=score, scored=has_score, **best))
    scored = any(os.path.isfile(p) for p in score_paths)
    return _assign_rank_scores(rows), len(numeric), scored


def join_target(target, targets_root=REPO_ROOT, db_dir=DB_DIR, score_file="scores_avg.gs"):
    """Join scores with labels over the full benchmark set for one target.

    "T37" and "T40" are merged two-interface targets (see MULTI_INTERFACE); T40's
    split interfaces "T40_1" / "T40_2" can still be requested individually.

    The CSV is the universe of poses; each pose gets its numeric GroScore (or NaN if
    not scored) plus a `rank_score` that ranks failed poses last. Returns
    (rows, n_numeric, scored):
      rows      : list of dict(id, score, rank_score, scored, stars, irms_nm, lrms_nm, cls)
      n_numeric : number of poses that had a real numeric score
      scored    : True if the target's score file exists (i.e. it has been run)
    """
    if target in MULTI_INTERFACE:
        return _join_multi(target, targets_root, db_dir, score_file)
    csv_path = os.path.join(db_dir, TARGET_CSV[target] + ".csv")
    score_path = os.path.join(targets_root, target, score_file)
    labels = load_labels(csv_path)
    scores, numeric = load_scores(score_path)
    rows = [dict(id=pid, score=scores.get(pid, FAILED_SCORE), scored=(pid in numeric), **lab)
            for pid, lab in labels.items()]
    n_numeric = len(numeric & set(labels))
    return _assign_rank_scores(rows), n_numeric, os.path.isfile(score_path)


def _positive_mask(rows, pmin):
    return np.array([1 if r["stars"] >= pmin else 0 for r in rows])


def load_native_scores(score_file="scores_avg.gs", natives_dir=None):
    """{target: native GroScore} for natives that have a numeric score in natives/.

    The native/experimental reference of each target is scored in natives/<target>/
    (keyed by target name, incl. the merged T37/T40).
    """
    natives_dir = natives_dir or os.path.join(REPO_ROOT, "natives")
    scores, numeric = load_scores(os.path.join(natives_dir, score_file))
    return {t: scores[t] for t in numeric}


def roc_auc(rows, positive_min_stars=1, native_score=None):
    """Standard ROC AUC (rank-based Mann-Whitney) for near-native detection.

    Positive class = stars >= positive_min_stars; "confidence" = -rank_score, i.e.
    more negative GroScore = more favourable, with failed poses ranked last (see
    _assign_rank_scores). Equals P(a near-native pose ranks better than a non-native
    one), ties = 0.5. NaN if either class is empty.

    If native_score is given, the native/experimental structure (I-RMSD 0, always
    near-native) is added as one extra positive at that GroScore -- the figure
    caption states the native is part of the ROC-AUC. For a target with no near-native
    decoys in the set, the native is then the sole positive, so the ROC reduces to the
    native's own percentile rank (e.g. T36); this is intentional and still reported.
    """
    y = _positive_mask(rows, positive_min_stars)
    scores = np.array([r["rank_score"] for r in rows], dtype=float)
    if native_score is not None and np.isfinite(native_score):
        y = np.append(y, 1)                    # native = experimental reference = positive
        scores = np.append(scores, float(native_score))
    P = int(y.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    r = rankdata(-scores)                      # -score so the best binder ranks highest
    return float((r[y == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def enrichment_auc(rows, positive_min_stars=1):
    """Area under the enrichment curve (thesis chapter 3 / paper Fig 6).

    Fraction of near-native poses recovered vs fraction of poses considered when
    ranked by GroScore (best first, failed poses last); random = 0.5. NaN if all/none
    are positive.
    """
    order = sorted(rows, key=lambda d: d["rank_score"])
    y = _positive_mask(order, positive_min_stars).astype(float)
    n = len(y)
    P = y.sum()
    if P == 0 or P == n:
        return float("nan")
    cum = np.cumsum(y) / P                      # fraction of positives recovered
    x = np.arange(1, n + 1) / n                 # fraction of poses considered
    x = np.concatenate(([0.0], x))
    yv = np.concatenate(([0.0], cum))
    return float(np.sum((x[1:] - x[:-1]) * (yv[1:] + yv[:-1]) / 2.0))
