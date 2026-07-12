#!/usr/bin/env python3
"""Shared loaders for CAPRI Score_set scoring analysis.

Joins GroScore output (per-target scores_avg.gs / scores_cgi.gs, at the repo
root) with the CAPRI benchmark quality/RMSD annotations (CAPRI/database/*.csv).

Conventions (matching the GroScore paper / thesis chapter 3):
  * GroScore: more negative = more favourable binding (best = rank 1).
  * Failed or un-scored poses are assigned score 0.0 kJ/mol (rank last).
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
FAILED_SCORE = 0.0   # paper: structures that fail stage 0 get a GroScore of 0

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


def _pick_best_label(l1, l2):
    """Best of a pose's two T40 interface assessments: higher stars, tie -> lower I-RMSD."""
    cands = [x for x in (l1, l2) if x is not None]

    def key(d):
        irms = d["irms_nm"] if np.isfinite(d["irms_nm"]) else float("inf")
        return (d["stars"], -irms)

    return max(cands, key=key)


def _join_t40(targets_root, db_dir, score_file):
    """Merged original T40 (matches the 2014/2019 papers).

    The two files hold the SAME 2180 poses; for a given pose, .1 and .2 are the
    identical receptor+inhibitor complex (verified: ~0.0007 A RMSD after rigid
    superposition), differing only by a rigid-body placement and the receptor
    chain label (A vs B) — both pull ligand chain C. They differ only in which
    reference binding mode the CAPRI quality is measured against. So per pose:
    quality = the better of the two interface assessments (their near-native sets
    are disjoint, reproducing the papers' union). The GroScore is invariant to the
    rigid placement, so either file gives the same score; if both T40_1 and T40_2
    happen to be scored, they are independent replicas of one structure and are
    averaged.
    """
    lab1 = load_labels(os.path.join(db_dir, TARGET_CSV["T40_1"] + ".csv"))
    lab2 = load_labels(os.path.join(db_dir, TARGET_CSV["T40_2"] + ".csv"))
    p1 = os.path.join(targets_root, "T40_1", score_file)
    p2 = os.path.join(targets_root, "T40_2", score_file)
    sc1, num1 = load_scores(p1)
    sc2, num2 = load_scores(p2)

    rows, numeric = [], set()
    for pid in set(lab1) | set(lab2):
        best = _pick_best_label(lab1.get(pid), lab2.get(pid))
        cand = [s for s, is_num in ((sc1.get(pid), pid in num1),
                                    (sc2.get(pid), pid in num2)) if is_num]
        if cand:
            score = float(np.mean(cand))   # .1/.2 are replicas of one complex -> average
            numeric.add(pid)
            has_score = True
        else:
            score = FAILED_SCORE
            has_score = False
        rows.append(dict(id=pid, score=score, scored=has_score, **best))
    return rows, len(numeric), (os.path.isfile(p1) or os.path.isfile(p2))


def join_target(target, targets_root=REPO_ROOT, db_dir=DB_DIR, score_file="scores_avg.gs"):
    """Join scores with labels over the full benchmark set for one target.

    "T40" is the merged original target (both trypsin interfaces); the split
    interfaces "T40_1" / "T40_2" can still be requested individually.

    The CSV is the universe of poses; each pose gets its numeric GroScore or
    FAILED_SCORE if not scored. Returns (rows, n_numeric, scored):
      rows      : list of dict(id, score, stars, irms_nm, lrms_nm, cls)
      n_numeric : number of poses that had a real numeric score
      scored    : True if the target's score file exists (i.e. it has been run)
    """
    if target == "T40":
        return _join_t40(targets_root, db_dir, score_file)
    csv_path = os.path.join(db_dir, TARGET_CSV[target] + ".csv")
    score_path = os.path.join(targets_root, target, score_file)
    labels = load_labels(csv_path)
    scores, numeric = load_scores(score_path)
    rows = [dict(id=pid, score=scores.get(pid, FAILED_SCORE), scored=(pid in numeric), **lab)
            for pid, lab in labels.items()]
    n_numeric = len(numeric & set(labels))
    return rows, n_numeric, os.path.isfile(score_path)


def _positive_mask(rows, pmin):
    return np.array([1 if r["stars"] >= pmin else 0 for r in rows])


def roc_auc(rows, positive_min_stars=1):
    """Standard ROC AUC (rank-based Mann-Whitney) for near-native detection.

    Positive class = stars >= positive_min_stars; "confidence" = -GroScore (more
    negative = more favourable). Equals P(a near-native pose ranks better than a
    non-native one), ties = 0.5. NaN if either class is empty.
    """
    y = _positive_mask(rows, positive_min_stars)
    scores = np.array([r["score"] for r in rows], dtype=float)
    P = int(y.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    r = rankdata(-scores)                      # -score so the best binder ranks highest
    return float((r[y == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def enrichment_auc(rows, positive_min_stars=1):
    """Area under the enrichment curve (thesis chapter 3 / paper Fig 6).

    Fraction of near-native poses recovered vs fraction of poses considered when
    ranked by GroScore (best first); random = 0.5. NaN if all/none are positive.
    """
    order = sorted(rows, key=lambda d: d["score"])
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
