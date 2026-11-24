#!/usr/bin/env python3
"""Analyse the GroScore scores of a live CAPRI round (a run directory made by
capri/prepare_round.py): i-RMSD to the best-scoring model against predicted
pKd, and FCC clustering with the scores per cluster.

Scores are read from <rundir>/scores_avg.gs, the score the pKd calibration was
fitted on (benchmark/haddock_benchmark/analyze_correlation.py), and always
reported as pKd = slope * score + intercept with the fit of the force field the
run used (-ff, default amber19sb_opc3: pKd = -0.0176 * score + 3.4513; the four
fits are the ones in README.md, Benchmark Data). Higher pKd = tighter binding.
A structure without a finite score (not yet run, nan, or a stage-0 status such
as BROKEN) has no pKd; it is still clustered and its i-RMSD measured.

Structures: <ID>/input.pdb, or <ID>/input.pdb inside <ID>.tar.gz once job.run
has archived the structure. Protein B is taken from sp.gs, the receptor is every
other chain. Residues are matched between models by chain and residue number
(CAPRI models of one target share the numbering; trimmed models just lack
residues). A residue whose name differs from the reference's is left out of
that comparison, and the number of such residues is reported.

i-RMSD (CAPRI definition, reference = the best-scoring model, or --ref):
interface residues are the reference residues with a heavy atom within 10 A of
the other partner; the N, CA, C and O atoms of those residues that the model
also has are superposed (Kabsch) and the RMSD taken.

FCC clustering (Rodrigues et al., Proteins 2012; the haddocking/fcc algorithm):
a model's contacts are the receptor/protein B residue pairs with heavy atoms
within 5 A. FCC(i, j) = |C_i & C_j| / |C_i|. j is a neighbour of i when
FCC(i, j) >= cutoff and FCC(j, i) >= cutoff * strictness. The model with the
most unclustered neighbours becomes a cluster centre with them, until no model
has min_size - 1 unclustered neighbours; the rest are unclustered. Defaults are
HADDOCK's: cutoff 0.60, strictness 0.75, min size 4. Clusters are ranked by the
mean pKd of their best min(4, n) scored members (HADDOCK ranks by its top 4),
among clusters with at least --min-scored scored members; the others are listed
after, unranked.

Writes into <rundir>/analysis/ (or -o):
  models.gs           one row per structure: score, pKd, i-RMSD, FCC to the
                      reference, cluster, contacts, MD5
  clusters.gs         one row per cluster: rank, size, scored, top-4 / mean /
                      best pKd, centre and its i-RMSD
  irmsd_vs_pkd.png    i-RMSD to the best-scoring model (x) vs pKd (y, reversed: 0 at the top, best at the bottom)
  clusters_pkd.png    pKd of the scored members of each cluster
Run with the groscore environment (numpy, scipy, matplotlib):
  python capri/analyze_round.py capri66/T381
"""
import argparse, math, os, sys, tarfile

import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse

# pKd = slope * score + intercept, fitted on scores_avg.gs of the HADDOCK benchmark (README.md)
CALIBRATION = {
    "amber19sb_opc3": (-0.0176, 3.4513),
    "amber19sb_opc": (-0.0155, 4.2704),
    "charmm36": (-0.0209, 2.9054),
    "gromos54a8": (-0.0178, 3.7434),
}
BACKBONE = ("N", "CA", "C", "O")
CONTACT_A = 5.0      # FCC contact distance
INTERFACE_A = 10.0   # CAPRI i-RMSD interface distance

# validated categorical slots 1-3 (all-pairs safe for a scatter), plus muted ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
GRAY = "#a3a29c"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"


# ---------------------------------------------------------------- inputs

def read_gs_rows(path):
    with open(path) as fh:
        return [l.split() for l in fh if l.strip() and not l.strip().startswith("#")]


def read_scores(path):
    """{ID: (score or nan, status, cycles, rmsd_flag)}; status is 'OK', 'NO_SCORE'
    or a stage-0 status that scores_avg.gs writes into the Score column."""
    out = {}
    for p in read_gs_rows(path):
        sid = p[0]
        try:
            s = float(p[1])
            status = "OK" if math.isfinite(s) else "NO_SCORE"
        except ValueError:
            s, status = float("nan"), p[1]
        cyc = p[3] if len(p) > 3 else "0"
        flag = p[6] if len(p) > 6 else "nan"
        out[sid] = (s, status, cyc, flag)
    return out


def read_pdb_text(rundir, sid):
    d = os.path.join(rundir, sid, "input.pdb")
    if os.path.isfile(d):
        with open(d) as fh:
            return fh.read()
    tgz = os.path.join(rundir, sid + ".tar.gz")
    if os.path.isfile(tgz):
        with tarfile.open(tgz) as tf:
            m = tf.extractfile("%s/input.pdb" % sid)
            if m is not None:
                return m.read().decode("ascii", "replace")
    return None


def is_hydrogen(line):
    el = line[76:78].strip().upper() if len(line) >= 78 else ""
    if el:
        return el in ("H", "D")
    name = line[12:16].strip()
    first = name.lstrip("0123456789")[:1].upper()
    return first in ("H", "D")


class Model:
    """Heavy atoms of one model, split into receptor (A side) and protein B."""

    def __init__(self, text, protb):
        keys, names, xyz, side = [], [], [], []
        resname = {}
        for line in text.splitlines():
            if not line.startswith(("ATOM  ", "HETATM")) or is_hydrogen(line):
                continue
            if line[16] not in (" ", "A"):          # keep the first alternate location only
                continue
            key = (line[21], line[22:27].strip())
            keys.append(key)
            names.append(line[12:16].strip())
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            side.append(line[21] in protb)
            resname.setdefault(key, line[17:20].strip())
        self.keys = keys
        self.names = names
        self.xyz = np.array(xyz, dtype=float).reshape(-1, 3)
        self.side = np.array(side, dtype=bool)
        self.resname = resname
        self.bb = {}
        for i, (k, n) in enumerate(zip(keys, names)):
            if n in BACKBONE:
                self.bb.setdefault((k, n), i)

    def residue_pairs(self, cutoff):
        """Set of (receptor residue, protein B residue) with heavy atoms within cutoff."""
        ia = np.where(~self.side)[0]
        ib = np.where(self.side)[0]
        if len(ia) == 0 or len(ib) == 0:
            return set()
        hits = cKDTree(self.xyz[ib]).query_ball_point(self.xyz[ia], cutoff)
        pairs = set()
        for a, js in zip(ia, hits):
            ka = self.keys[a]
            for j in js:
                pairs.add((ka, self.keys[ib[j]]))
        return pairs


def kabsch_rmsd(p, q):
    """RMSD of q onto p after optimal superposition (both (n, 3))."""
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, s, vt = np.linalg.svd(qc.T @ pc)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    e = (pc ** 2).sum() + (qc ** 2).sum() - 2 * s.sum()
    return math.sqrt(max(e, 0.0) / len(p))


# ---------------------------------------------------------------- clustering

def fcc_matrix(contacts):
    """FCC[i, j] = |C_i & C_j| / |C_i| (0 for a model without contacts)."""
    vocab = {}
    rows, cols = [], []
    for i, c in enumerate(contacts):
        for pair in c:
            rows.append(i)
            cols.append(vocab.setdefault(pair, len(vocab)))
    n = len(contacts)
    m = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, max(len(vocab), 1)))
    inter = (m @ m.T).toarray().astype(float)
    size = np.array([len(c) for c in contacts], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fcc = np.where(size[:, None] > 0, inter / size[:, None], 0.0)
    return fcc


def cluster_fcc(fcc, cutoff, strictness, min_size):
    """Clusters as lists of model indices, centre first, in order of formation."""
    n = len(fcc)
    nb = (fcc >= cutoff) & (fcc.T >= cutoff * strictness)
    np.fill_diagonal(nb, False)
    free = np.ones(n, dtype=bool)
    clusters = []
    while True:
        counts = (nb & free[None, :]).sum(1)
        counts[~free] = -1
        centre = int(np.argmax(counts))          # ties: lowest index
        if counts[centre] < min_size - 1:
            break
        members = [centre] + [int(j) for j in np.where(nb[centre] & free)[0]]
        free[members] = False
        clusters.append(members)
    return clusters


# ---------------------------------------------------------------- plots

def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_irmsd(path, rows, ref_id, top3, title, calib_txt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    style(ax)
    sc = [r for r in rows if math.isfinite(r["pkd"]) and math.isfinite(r["irmsd"])]
    rest = [r for r in sc if r["cluster"] not in top3]
    ax.scatter([r["irmsd"] for r in rest], [r["pkd"] for r in rest], s=22, color=GRAY,
               edgecolors="white", linewidths=0.8, zorder=2, label="other clusters / unclustered")
    for k, cid in enumerate(top3):
        pts = [r for r in sc if r["cluster"] == cid]
        ax.scatter([r["irmsd"] for r in pts], [r["pkd"] for r in pts], s=30, color=SERIES[k],
                   marker=MARKERS[k], edgecolors="white", linewidths=0.8, zorder=3,
                   label="cluster %d (rank %d)" % (cid, k + 1))
    for x in (2.0, 4.0):
        ax.axvline(x, color=INK2, lw=0.6, ls=(0, (3, 3)), zorder=1)
        ax.text(x, 1.0, " %g \u00c5" % x, transform=ax.get_xaxis_transform(), va="top", ha="left",
                fontsize=7, color=INK2)
    ref = next(r for r in rows if r["id"] == ref_id)
    if math.isfinite(ref["pkd"]):
        ax.annotate(ref_id, (ref["irmsd"], ref["pkd"]), xytext=(8, -2), textcoords="offset points",
                    fontsize=7.5, color=INK, va="center",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=5)
    ax.set_xlabel("i-RMSD to best-scoring model %s (\u00c5)" % ref_id, fontsize=9)
    ax.set_ylabel("predicted pKd (axis reversed: best at the bottom)", fontsize=9)
    ax.set_xlim(left=-1.0)
    # pKd reversed, as an energy is drawn (as if it were -pKd): 0 at the top, the
    # highest pKd at the bottom, so the best models sit low as in a docking funnel
    y = [r["pkd"] for r in sc]
    if y:
        lo, hi = min(0.0, min(y)), max(y)
        pad = 0.04 * max(hi - lo, 1.0)
        ax.set_ylim(hi + pad, lo - pad if lo < 0 else 0.0)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.text(0.01, 0.0, calib_txt, fontsize=7, color=INK2, va="top")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", labelcolor=INK2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_clusters(path, cl_rows, rows_by_cluster, unclustered_pkd, title, calib_txt, max_rows, n_ranked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = cl_rows[:max_rows]
    labels = ["%s cl %d  (%d/%d)" % ("#%d" % c["rank"] if c["rank"] else "", c["cluster"], c["n_scored"],
                                     c["size"]) for c in show]
    data = [rows_by_cluster[c["cluster"]] for c in show]
    if unclustered_pkd:
        labels.append("unclustered (%d)" % len(unclustered_pkd))
        data.append(unclustered_pkd)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.28 * max(n, 4)))
    style(ax)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        y = n - 1 - i
        is_un = unclustered_pkd and i == n - 1
        col = GRAY if is_un else SERIES[0]
        jit = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(vals, y + jit, s=14, color=col, edgecolors="white", linewidths=0.6, zorder=3)
        if not is_un and len(vals) > 1:
            top = np.mean(sorted(vals, reverse=True)[:4])
            ax.plot([top, top], [y - 0.32, y + 0.32], color=INK, lw=1.5, solid_capstyle="round", zorder=4)
    if 0 < n_ranked < min(len(show), max_rows):      # ranked clusters above the line, unranked below
        ax.axhline(n - n_ranked - 0.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=7.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("predicted pKd (dots: scored members; bar: mean of the best 4)", fontsize=9)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.text(0.01, 0.0, calib_txt + "\nlabels: #rank cluster (scored/size); clusters with fewer scored "
             "members are below the dashed line, unranked", fontsize=7, color=INK2, va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def fmt(x, nd=2):
    return "nan" if x is None or not math.isfinite(x) else ("%.*f" % (nd, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rundir", help="run directory (sp.gs, scores_avg.gs, structure dirs or archives)")
    ap.add_argument("-ff", "--forcefield", default="amber19sb_opc3", choices=sorted(CALIBRATION),
                    help="force field of the run, selects the pKd calibration (default amber19sb_opc3)")
    ap.add_argument("-s", "--scores", default="scores_avg.gs", help="score file in the run directory")
    ap.add_argument("--ref", help="reference model for the i-RMSD (default: the best-scoring one)")
    ap.add_argument("--cutoff", type=float, default=0.60, help="FCC cutoff (default 0.60)")
    ap.add_argument("--strictness", type=float, default=0.75, help="FCC strictness (default 0.75)")
    ap.add_argument("--min-size", type=int, default=4, help="minimum cluster size (default 4)")
    ap.add_argument("--min-scored", type=int, default=3,
                    help="scored members a cluster needs to be ranked (default 3)")
    ap.add_argument("--max-rows", type=int, default=25, help="clusters shown in clusters_pkd.png")
    ap.add_argument("-o", "--outdir", help="output directory (default <rundir>/analysis)")
    ap.add_argument("--submit", action="store_true",
                    help="also write <target>_scoring.pdb, the scorer submission (see the header)")
    ap.add_argument("--pick", choices=("top", "cluster"), default="top",
                    help="submission models: 'top' = best pKd; 'cluster' = the best scored model of each "
                         "cluster (ranked clusters first), then best pKd (default top)")
    ap.add_argument("--nsubmit", type=int, default=10, help="models in the submission (default 10)")
    ap.add_argument("--group", help="group name for a REMARK line of the submission, e.g. 'HUMAN Team 6'")
    args = ap.parse_args()

    rundir = args.rundir.rstrip("/")
    target = os.path.basename(os.path.abspath(rundir))
    outdir = args.outdir or os.path.join(rundir, "analysis")
    slope, icpt = CALIBRATION[args.forcefield]
    calib_txt = "pKd = %.4f x score + %.4f (%s, %s)" % (slope, icpt, args.forcefield, args.scores)

    sp = read_gs_rows(os.path.join(rundir, "sp.gs"))
    ids = [p[0] for p in sp]
    protb = {sid: set(p[1].split(",")) for sid, p in zip(ids, (p for p in sp))}
    md5 = {}
    if os.path.isfile(os.path.join(rundir, "md5_map.gs")):
        md5 = {p[0]: (p[1], p[2]) for p in read_gs_rows(os.path.join(rundir, "md5_map.gs"))}
    scores = read_scores(os.path.join(rundir, args.scores))

    # structures
    models, missing = {}, []
    for k, sid in enumerate(ids):
        text = read_pdb_text(rundir, sid)
        if text is None:
            missing.append(sid)
            continue
        models[sid] = Model(text, protb[sid])
        if (k + 1) % 250 == 0:
            print("  read %d / %d structures" % (k + 1, len(ids)), file=sys.stderr)
    if missing:
        print("WARNING: %d structure(s) without input.pdb or archive, left out: %s%s"
              % (len(missing), " ".join(missing[:5]), " ..." if len(missing) > 5 else ""))
    ids = [s for s in ids if s in models]

    pkd = {}
    for sid in ids:
        s = scores.get(sid, (float("nan"), "NOT_IN_SCORES", "0", "nan"))[0]
        pkd[sid] = slope * s + icpt if math.isfinite(s) else float("nan")
    scored = [s for s in ids if math.isfinite(pkd[s])]
    if not scored:
        sys.exit("analyze_round: no structure of %s has a finite score in %s" % (rundir, args.scores))
    ref_id = args.ref or max(scored, key=lambda s: pkd[s])
    if ref_id not in models:
        sys.exit("analyze_round: reference %s has no structure" % ref_id)
    ref = models[ref_id]

    # i-RMSD to the reference
    iface = set()
    for a, b in ref.residue_pairs(INTERFACE_A):
        iface.add(a)
        iface.add(b)
    ref_bb = [(k, n) for (k, n) in ref.bb if k in iface]
    irmsd, mism = {}, {}
    for sid in ids:
        m = models[sid]
        bad = {k for k in iface if k in m.resname and m.resname[k] != ref.resname[k]}
        common = [kn for kn in ref_bb if kn in m.bb and kn[0] not in bad]
        mism[sid] = len(bad)
        if len(common) < 12:
            irmsd[sid] = float("nan")
            continue
        p = ref.xyz[[ref.bb[kn] for kn in common]]
        q = m.xyz[[m.bb[kn] for kn in common]]
        irmsd[sid] = kabsch_rmsd(p, q)

    # FCC clustering
    contacts = [models[s].residue_pairs(CONTACT_A) for s in ids]
    fcc = fcc_matrix(contacts)
    clusters = cluster_fcc(fcc, args.cutoff, args.strictness, args.min_size)
    cluster_of = {}
    for cid, members in enumerate(clusters, start=1):
        for i in members:
            cluster_of[ids[i]] = cid
    iref = ids.index(ref_id)

    rows = []
    for i, sid in enumerate(ids):
        s, status, cyc, flag = scores.get(sid, (float("nan"), "NOT_IN_SCORES", "0", "nan"))
        rows.append(dict(id=sid, model=md5.get(sid, ("?", "?"))[0], md5=md5.get(sid, ("?", "?"))[1],
                         score=s, status=status, cycles=cyc, flag=flag, pkd=pkd[sid], irmsd=irmsd[sid],
                         fcc_ref=fcc[i, iref], cluster=cluster_of.get(sid, 0), ncont=len(contacts[i]),
                         mism=mism[sid]))
    by_id = {r["id"]: r for r in rows}

    # cluster statistics and ranking
    cl_rows = []
    for cid, members in enumerate(clusters, start=1):
        vals = sorted((pkd[ids[i]] for i in members if math.isfinite(pkd[ids[i]])), reverse=True)
        best_id = max((ids[i] for i in members if math.isfinite(pkd[ids[i]])), key=lambda s: pkd[s], default="-")
        cl_rows.append(dict(cluster=cid, size=len(members), n_scored=len(vals),
                            top4=float(np.mean(vals[:4])) if vals else float("nan"),
                            mean=float(np.mean(vals)) if vals else float("nan"),
                            sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                            best=vals[0] if vals else float("nan"), best_id=best_id,
                            centre=ids[members[0]], centre_irmsd=irmsd[ids[members[0]]]))
    ranked = sorted((c for c in cl_rows if c["n_scored"] >= args.min_scored), key=lambda c: -c["top4"])
    unranked = sorted((c for c in cl_rows if c["n_scored"] < args.min_scored),
                      key=lambda c: (-c["n_scored"], -(c["top4"] if math.isfinite(c["top4"]) else -1e9)))
    for k, c in enumerate(ranked, start=1):
        c["rank"] = k
    for c in unranked:
        c["rank"] = 0

    # outputs
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "models.gs"), "w") as o:
        o.write("# %s: %s\n# reference for i-RMSD / FCC_ref: %s (%s)\n"
                % (target, calib_txt, ref_id, "best-scoring" if not args.ref else "--ref"))
        o.write("# FCC clustering: cutoff %.2f, strictness %.2f, min size %d; cluster 0 = unclustered\n"
                % (args.cutoff, args.strictness, args.min_size))
        o.write("# Structure_ID  CAPRI_model  Score  Status  Cycles  RMSD_flag  pKd  iRMSD_A  FCC_ref  "
                "Cluster  Contacts  ResnameMismatch  MD5\n")
        for r in sorted(rows, key=lambda r: (-(r["pkd"]) if math.isfinite(r["pkd"]) else 1e9, r["id"])):
            o.write("%s  %s  %s  %s  %s  %s  %s  %s  %s  %d  %d  %d  %s\n"
                    % (r["id"], r["model"], fmt(r["score"], 1), r["status"], r["cycles"], r["flag"],
                       fmt(r["pkd"]), fmt(r["irmsd"]), fmt(r["fcc_ref"]), r["cluster"], r["ncont"],
                       r["mism"], r["md5"]))
    with open(os.path.join(outdir, "clusters.gs"), "w") as o:
        o.write("# %s: %s\n# ranked by the mean pKd of the best min(4, n) scored members, among clusters "
                "with >= %d scored members; rank 0 = too few scored members\n" % (target, calib_txt, args.min_scored))
        o.write("# Rank  Cluster  Size  Scored  Top4_pKd  Mean_pKd  SD_pKd  Best_pKd  Best_ID  Centre  "
                "Centre_iRMSD_A\n")
        for c in ranked + unranked:
            o.write("%d  %d  %d  %d  %s  %s  %s  %s  %s  %s  %s\n"
                    % (c["rank"], c["cluster"], c["size"], c["n_scored"], fmt(c["top4"]), fmt(c["mean"]),
                       fmt(c["sd"]), fmt(c["best"]), c["best_id"], c["centre"], fmt(c["centre_irmsd"])))

    top3 = [c["cluster"] for c in ranked[:3]]
    title = "%s: %d of %d models scored" % (target, len(scored), len(ids))
    plot_irmsd(os.path.join(outdir, "irmsd_vs_pkd.png"), rows, ref_id, top3, title, calib_txt)
    by_cluster = {c["cluster"]: [pkd[ids[i]] for i in clusters[c["cluster"] - 1] if math.isfinite(pkd[ids[i]])]
                  for c in cl_rows}
    unclustered = [r["pkd"] for r in rows if r["cluster"] == 0 and math.isfinite(r["pkd"])]
    show = ranked + [c for c in unranked if c["n_scored"] > 0]
    plot_clusters(os.path.join(outdir, "clusters_pkd.png"), show, by_cluster, unclustered,
                  title + ", %d FCC clusters" % len(clusters), calib_txt, args.max_rows, len(ranked))

    # report
    n_un = sum(1 for r in rows if r["cluster"] == 0)
    print("%s: %d structures, %d scored, %d without contacts" % (target, len(ids), len(scored),
                                                                  sum(1 for c in contacts if not c)))
    print("  %s" % calib_txt)
    print("  reference %s: pKd %s, %d interface residues, %d backbone atoms"
          % (ref_id, fmt(pkd[ref_id]), len(iface), len(ref_bb)))
    mm = [r for r in rows if r["mism"]]
    if mm:
        print("  WARNING: %d model(s) with interface residues named differently from the reference "
              "(left out of their i-RMSD)" % len(mm))
    print("  FCC: %d clusters (largest %d), %d unclustered; %d ranked (>= %d scored members)"
          % (len(clusters), max((len(c) for c in clusters), default=0), n_un, len(ranked), args.min_scored))
    print("\n  Top models by pKd:")
    print("    %-10s %6s %7s %6s %7s" % ("ID", "pKd", "iRMSD", "FCC", "cluster"))
    for r in sorted((r for r in rows if math.isfinite(r["pkd"])), key=lambda r: -r["pkd"])[:10]:
        print("    %-10s %6s %7s %6s %7d" % (r["id"], fmt(r["pkd"]), fmt(r["irmsd"]), fmt(r["fcc_ref"]), r["cluster"]))
    if ranked:
        print("\n  Top clusters (mean pKd of best 4 scored):")
        print("    %4s %7s %5s %6s %6s %6s  %-10s" % ("rank", "cluster", "size", "scored", "top4", "best", "best_ID"))
        for c in ranked[:10]:
            print("    %4d %7d %5d %6d %6s %6s  %-10s" % (c["rank"], c["cluster"], c["size"], c["n_scored"],
                                                         fmt(c["top4"]), fmt(c["best"]), c["best_id"]))
    print("\n  written: %s/{models.gs, clusters.gs, irmsd_vs_pkd.png, clusters_pkd.png}" % outdir)
    if args.submit:
        return write_submission(args, rundir, target, outdir, rows, ranked, unranked, calib_txt)
    return 0


def pick_models(rows, ranked, unranked, how, n):
    """IDs of the submission in ranking order."""
    scored = sorted((r for r in rows if math.isfinite(r["pkd"])), key=lambda r: (-r["pkd"], r["id"]))
    if how == "top":
        return [r["id"] for r in scored[:n]]
    by_id = {r["id"]: r for r in rows}
    picks = [c["best_id"] for c in ranked]                       # best member of each ranked cluster
    rest = [c["best_id"] for c in unranked if c["best_id"] != "-"]
    rest += [r["id"] for r in scored if r["cluster"] == 0]       # an unclustered model is its own cluster
    picks += sorted(rest, key=lambda s: (-by_id[s]["pkd"], s))
    for r in scored:                                             # fill up with the best remaining models
        if r["id"] not in picks:
            picks.append(r["id"])
    return picks[:n]


def write_submission(args, rundir, target, outdir, rows, ranked, unranked, calib_txt):
    """<target>_scoring.pdb as CAPRI asks for a scorer submission (answer of the organisers,
    round 66): the models in decreasing order, numbered MODEL 1..n; the REMARK 9 MD5 line of
    each model in the header, in the same order; HEADER, COMPND and AUTHOR are not required.
    The models are written with the scoring-set coordinates, unchanged."""
    by_id = {r["id"]: r for r in rows}
    picks = pick_models(rows, ranked, unranked, args.pick, args.nsubmit)
    if len(picks) < args.nsubmit:
        print("  WARNING: only %d scored model(s); the submission has %d instead of %d"
              % (len(picks), len(picks), args.nsubmit))
    bad = [s for s in picks if not by_id[s]["md5"] or by_id[s]["md5"] in ("?", "MISSING")]
    if bad:
        print("  ERROR: no MD5 code for %s (md5_map.gs); no submission written" % " ".join(bad))
        return 1
    path = os.path.join(outdir, "%s_scoring.pdb" % target)
    out = []
    if args.group:
        out.append("REMARK   1 %s SCORER SUBMISSION, %s" % (target, args.group))
    out.append("REMARK   1 GROSCORE SMD SCORES AS PKD, %s, MODELS PICKED BY %s"
               % (calib_txt.split(" (")[0].replace(" x ", " * "), "PKD" if args.pick == "top" else "CLUSTER, THEN PKD"))
    for s in picks:
        out.append("REMARK   9 MODEL %8d MD5 %s" % (int(by_id[s]["model"]), by_id[s]["md5"]))
    for k, s in enumerate(picks, start=1):
        text = read_pdb_text(rundir, s)
        out.append("MODEL     %4d" % k)
        out += [l.rstrip("\r\n") for l in text.splitlines() if l.startswith(("ATOM  ", "HETATM", "TER"))]
        out.append("ENDMDL")
    out.append("END")
    with open(path, "w") as o:
        o.write("\n".join(out) + "\n")
    with open(os.path.join(outdir, "submission.gs"), "w") as o:
        o.write("# %s scorer submission (%s): %s\n" % (target, os.path.basename(path), calib_txt))
        o.write("# Rank  Structure_ID  CAPRI_model  pKd  Cycles  RMSD_flag  Cluster  iRMSD_A  MD5\n")
        for k, s in enumerate(picks, start=1):
            r = by_id[s]
            o.write("%d  %s  %s  %s  %s  %s  %d  %s  %s\n" % (k, s, r["model"], fmt(r["pkd"]), r["cycles"],
                                                         r["flag"], r["cluster"], fmt(r["irmsd"]), r["md5"]))
    print("\n  Submission (%s, pick %s):" % (os.path.basename(path), args.pick))
    print("    %4s %-10s %6s %6s %7s %7s" % ("rank", "ID", "model", "pKd", "cluster", "cycles"))
    for k, s in enumerate(picks, start=1):
        r = by_id[s]
        print("    %4d %-10s %6s %6s %7d %7s" % (k, s, r["model"], fmt(r["pkd"]), r["cluster"], r["cycles"]))
    print("  written: %s, %s" % (path, os.path.join(outdir, "submission.gs")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
