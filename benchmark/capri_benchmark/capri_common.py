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
# display / iteration order
TARGETS = ["T29", "T30", "T32", "T35", "T36", "T37", "T38", "T39", "T40_1", "T40_2",
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


def join_target(target, targets_root=REPO_ROOT, db_dir=DB_DIR, score_file="scores_avg.gs"):
    """Join scores with labels over the full benchmark set for one target.

    The CSV is the universe of poses; each pose gets its numeric GroScore or
    FAILED_SCORE if not scored. Returns (rows, n_numeric, scored):
      rows      : list of dict(id, score, stars, irms_nm, lrms_nm, cls)
      n_numeric : number of poses that had a real numeric score
      scored    : True if the target's score file exists (i.e. it has been run)
    """
    csv_path = os.path.join(db_dir, TARGET_CSV[target] + ".csv")
    score_path = os.path.join(targets_root, target, score_file)
    labels = load_labels(csv_path)
    scores, numeric = load_scores(score_path)
    rows = [dict(id=pid, score=scores.get(pid, FAILED_SCORE), **lab)
            for pid, lab in labels.items()]
    n_numeric = len(numeric & set(labels))
    return rows, n_numeric, os.path.isfile(score_path)
