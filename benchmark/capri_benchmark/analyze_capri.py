#!/usr/bin/env python3
"""CAPRI Score_set scoring performance (thesis chapter 3, Table 3-1).

For each target the poses are ranked by GroScore (most negative = best) and:

  * Top-10 : number of acceptable-or-better / medium (**) / high (***) quality
             poses among the 10 best-ranked, formatted like the paper
             (e.g. "10/6***/4**": 10 near-native, 6 high, 4 medium).
  * AUC    : area under the enrichment curve — fraction of near-native poses
             recovered vs fraction of poses considered (ranked by score);
             random ranking = 0.5, perfect = 1.0. This is the metric shown in
             chapter 3 Table 3-1 / paper Figure 6.
  * ROC    : standard ROC AUC (rank-based Mann-Whitney), same positive class.

Positive class ("near-native") defaults to acceptable-or-better (stars >= 1).
Failed / un-scored poses get score 0 (paper convention) and rank last.

Usage:
  python3 analyze_capri.py                     # all targets, scores_avg.gs
  python3 analyze_capri.py -s scores_cgi.gs    # use CGI scores
  python3 analyze_capri.py --positive medium   # positive class = medium+ for AUC
  python3 analyze_capri.py -o ../results/capri_scores_table.tsv
"""
import os
import argparse
import numpy as np
from scipy.stats import rankdata

import capri_common as cc

STARS_MIN = {"acceptable": 1, "medium": 2, "high": 3}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("-s", "--score-file", default="scores_avg.gs",
                help="per-target scores file (default: scores_avg.gs)")
ap.add_argument("--targets-root", default=cc.REPO_ROOT,
                help="directory holding the per-target folders (default: repo root)")
ap.add_argument("--db", default=cc.DB_DIR, help="CAPRI database dir with the CSVs")
ap.add_argument("--positive", choices=list(STARS_MIN), default="acceptable",
                help="quality threshold counted as near-native for the AUCs (default: acceptable)")
ap.add_argument("-k", "--topk", type=int, default=10, help="Top-K to count (default: 10)")
ap.add_argument("-o", "--out", default=None, help="optional TSV output path")
args = ap.parse_args()

PMIN = STARS_MIN[args.positive]


def enrichment_auc(stars_ranked, pmin):
    """Area under the fraction-recovered vs fraction-considered curve (random=0.5)."""
    y = (stars_ranked >= pmin).astype(float)
    n = len(y)
    P = y.sum()
    if P == 0 or P == n:
        return float("nan")
    cum = np.cumsum(y) / P                    # fraction of positives recovered
    x = np.arange(1, n + 1) / n               # fraction of poses considered
    x = np.concatenate(([0.0], x))
    yv = np.concatenate(([0.0], cum))
    return float(np.sum((x[1:] - x[:-1]) * (yv[1:] + yv[:-1]) / 2.0))


def roc_auc(stars, scores, pmin):
    """Standard ROC AUC via average ranks (higher confidence = more negative score)."""
    y = (np.asarray(stars) >= pmin).astype(int)
    P = int(y.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    r = rankdata(-np.asarray(scores))         # -score so best binder ranks highest
    return float((r[y == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def fmt_top10(n_nn, n_high, n_med):
    s = str(n_nn)
    if n_high:
        s += "/%d***" % n_high
    if n_med:
        s += "/%d**" % n_med
    return s


hdr = "%-6s %7s %7s   %-16s %5s %5s %5s   %8s %8s" % (
    "Target", "N", "scored", "Top-%d" % args.topk, "***", "**", "*", "AUC", "ROC")
print(hdr)
print("-" * len(hdr))

rows_out = []
tot = dict(N=0, scored=0, nn=0, high=0, med=0, acc=0)
aucs, rocs = [], []

for t in cc.TARGETS:
    rows, n_numeric, scored = cc.join_target(t, args.targets_root, args.db, args.score_file)
    N = len(rows)
    tot["N"] += N                       # full benchmark size — count every target
    if not scored or n_numeric == 0:
        print("%-6s %7d %7s   %-16s" % (t, N, "-", "(not scored)"))
        rows_out.append((t, N, 0, "", 0, 0, 0, float("nan"), float("nan")))
        continue

    order = sorted(rows, key=lambda d: d["score"])   # best (most negative) first
    top = order[:args.topk]
    n_high = sum(1 for d in top if d["stars"] == 3)
    n_med = sum(1 for d in top if d["stars"] == 2)
    n_acc = sum(1 for d in top if d["stars"] == 1)
    n_nn = n_high + n_med + n_acc

    stars_ranked = np.array([d["stars"] for d in order])
    scores_ranked = np.array([d["score"] for d in order])
    e_auc = enrichment_auc(stars_ranked, PMIN)
    r_auc = roc_auc(stars_ranked, scores_ranked, PMIN)

    print("%-6s %7d %7d   %-16s %5d %5d %5d   %8.3f %8.3f" % (
        t, N, n_numeric, fmt_top10(n_nn, n_high, n_med), n_high, n_med, n_acc, e_auc, r_auc))
    rows_out.append((t, N, n_numeric, fmt_top10(n_nn, n_high, n_med),
                     n_high, n_med, n_acc, e_auc, r_auc))

    tot["scored"] += n_numeric; tot["nn"] += n_nn
    tot["high"] += n_high; tot["med"] += n_med; tot["acc"] += n_acc
    if not np.isnan(e_auc):
        aucs.append(e_auc)
    if not np.isnan(r_auc):
        rocs.append(r_auc)

print("-" * len(hdr))
mean_auc = np.mean(aucs) if aucs else float("nan")
mean_roc = np.mean(rocs) if rocs else float("nan")
print("%-6s %7d %7d   %-16s %5d %5d %5d   %8.3f %8.3f" % (
    "TOTAL", tot["N"], tot["scored"],
    fmt_top10(tot["nn"], tot["high"], tot["med"]),
    tot["high"], tot["med"], tot["acc"], mean_auc, mean_roc))
print("\n(near-native = %s or better; AUC = enrichment-curve area [chapter 3], "
      "ROC = standard ROC AUC; both random = 0.5. TOTAL: N = full benchmark (all "
      "targets); scored / Top-%d / ***/**/* sum over scored targets; AUC/ROC are "
      "means over scored targets.)" % (args.positive, args.topk))

if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("target\tN\tn_scored\ttop%d\thigh\tmedium\tacceptable\tAUC_enrich\tROC_AUC\n" % args.topk)
        for r in rows_out:
            f.write("%s\t%d\t%d\t%s\t%d\t%d\t%d\t%.4f\t%.4f\n" % r)
    print("Wrote %s" % args.out)
