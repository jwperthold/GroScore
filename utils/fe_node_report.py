#!/usr/bin/env python3
"""Which VSC5 nodes are slow, from the GROMACS logs a finished run already has.

The --exclude list in slurm/vsc5*.sh is a measurement, and a measurement made once
rots: nodes get repaired, replaced and re-tuned. This regenerates it, so the list in
those templates never has to be believed on faith.

    python3 utils/fe_node_report.py test13 test14 test15
    python3 utils/fe_node_report.py test14 --exclude-line

Every switching leg writes its own ns/day, which is already normalised for leg
length, so legs of different lengths compare directly. A node is counted slow on a
leg if it managed less than SLOW_FRAC of the fleet median.

WHAT THE DISTINCTION IS FOR. A node slow on nearly every leg it ever ran is slow
hardware or a permanently oversubscribed one, and excluding it is a straight win. A
node that is sometimes fast and sometimes slow is contended, and excluding it only
moves the contention elsewhere while shrinking the pool -- so those are reported
separately and NOT put in the exclude line.
"""
import argparse, collections, glob, os, re, statistics as st, sys

SLOW_FRAC = 0.6          # of the fleet median
ALWAYS = 0.8             # fraction of a node's legs that must be slow to exclude
MIN_LEGS = 40            # below this a node has not been sampled enough to judge

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("runs", nargs="+", help="run directories holding <sid>/*.log")
ap.add_argument("--exclude-line", action="store_true",
                help="print only the SBATCH line, for pasting into slurm/vsc5*.sh")
ap.add_argument("--slow-frac", type=float, default=SLOW_FRAC)
ap.add_argument("--min-legs", type=int, default=MIN_LEGS)
args = ap.parse_args()

per = collections.defaultdict(lambda: collections.defaultdict(list))
for run in args.runs:
    tag = os.path.basename(run.rstrip("/"))
    pats = [os.path.join(run, "*", "*.log"), os.path.join(run, "*.log")]
    for pat in pats:
        for f in glob.glob(pat):
            b = os.path.basename(f)[:-4]
            if b.startswith(("nvt", "emin", "ions")):
                continue
            try:
                txt = open(f, errors="ignore").read()
            except OSError:
                continue
            h = re.search(r"Hardware detected on host (\S+?):", txt)
            p = re.search(r"Performance:\s+([\d.]+)", txt)
            if h and p:
                per[h.group(1)][tag].append(float(p.group(1)))

if not per:
    sys.stderr.write("no GROMACS logs with a host and a Performance line under %s\n"
                     % ", ".join(args.runs))
    sys.exit(1)

allv = [x for n in per for r in per[n] for x in per[n][r]]
fleet = st.median(allv)
cut = args.slow_frac * fleet

rows = []
for node in per:
    v = [x for r in per[node] for x in per[node][r]]
    frac = sum(1 for x in v if x < cut) / len(v)
    rows.append((st.median(v), node, len(v), frac, len(per[node])))
rows.sort()

drop = [r for r in rows if r[3] > ALWAYS and r[2] >= args.min_legs]
mixed = [r for r in rows if 0.2 < r[3] <= ALWAYS]
thin = [r for r in rows if r[3] > ALWAYS and r[2] < args.min_legs]

if args.exclude_line:
    print("#SBATCH --exclude=%s" % ",".join(sorted(r[1] for r in drop)))
    sys.exit(0)

print("fleet median %.0f ns/day over %d legs on %d nodes; slow means below %.0f"
      % (fleet, len(allv), len(rows), cut))
print("\n  %-13s %7s %7s %10s %9s" % ("node", "legs", "runs", "median", "slow"))
for med, node, n, frac, nr in rows:
    mark = "  <- exclude" if (frac > ALWAYS and n >= args.min_legs) else (
        "  (mixed)" if 0.2 < frac <= ALWAYS else "")
    print("  %-13s %7d %7d %10.0f %8.0f%%%s" % (node, n, nr, med, 100 * frac, mark))

print("\nSLOW ALMOST ALWAYS (>%.0f%% of legs, >=%d legs seen): exclude these"
      % (100 * ALWAYS, args.min_legs))
for med, node, n, frac, nr in drop:
    print("  %-13s %4.0f ns/day  %3.0f%% of %d legs, in %d run(s)"
          % (node, med, 100 * frac, n, nr))
if thin:
    print("\nSLOW BUT BARELY SAMPLED (<%d legs) -- not excluded, watch them:" % args.min_legs)
    for med, node, n, frac, nr in thin:
        print("  %-13s %4.0f ns/day  %d legs" % (node, med, n))
print("\nMIXED, i.e. contended rather than bad -- NOT excluded:")
for med, node, n, frac, nr in mixed:
    print("  %-13s %4.0f ns/day  %3.0f%% of %d legs" % (node, med, 100 * frac, n))

if drop:
    kept = [r for r in rows if r not in drop]
    print("\nexcluding %d of %d nodes; they ran %.0f%% of the legs, and %.1fx slower "
          "than the rest (%.0f against %.0f ns/day)"
          % (len(drop), len(rows),
             100.0 * sum(r[2] for r in drop) / len(allv),
             st.median([r[0] for r in kept]) / st.median([r[0] for r in drop]),
             st.median([r[0] for r in drop]), st.median([r[0] for r in kept])))
    print("\n#SBATCH --exclude=%s" % ",".join(sorted(r[1] for r in drop)))
