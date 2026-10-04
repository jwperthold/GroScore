#!/usr/bin/env python3
#
# measure_box.py - Size the production box from the equilibration, instead of
# from a constant.
#
# The GROMACS pull code aborts when any checked pair of pull groups is further
# apart than 0.49 * the shortest box VECTOR. The padding that satisfies that has
# always been a hard-coded number in job_fe.run (1.5, then 1.8), derived from one
# structure and applied to every complex. This measures it per structure.
#
# The rule: the production box vector is
#
#     L = D + 2 * pad
#
# where D is the DIAMETER OF THE PROTEIN COMPLEX, the largest distance between two
# of its atoms, at the 99th percentile over every frame of every probe replica, and
# pad is the clearance wanted on each side (default 1.5 nm).
#
# What this gets right that a naive version does not:
#
#   * the complex is REASSEMBLED in every frame, not clustered. A cutout partner is
#     one moleculetype made of several capped, unbonded fragments. gmx trjconv
#     -pbc cluster makes each moleculetype whole by chaining every atom to the
#     image nearest the atom before it (GROMACS 2026 pbcmethods.cpp 153-158) and
#     then moves it as one unit, so wherever two sequence-consecutive fragments sit
#     more than half a box apart every later fragment stays a box vector off, and
#     the "diameter" becomes a box length. That is what pass 2 (bm_fe_amber_2)
#     measured: D_max exceeded the probe box it was measured in for 18 of 46
#     structures (2AJF 20.07 nm in a 9.75 nm box, 1IJK above its box even at the
#     median frame), 11 of them then asked for more than job_fe.run's 4.0 nm cap
#     and fell back to a 1.8 nm pad, and the other 7 kept boxes of 1.7-3.6 times
#     the volume a box on their median diameter would have. Here the first frame
#     of each replica is assembled in space (utils/pbc_tools.assemble: a spanning
#     tree over one anchor per residue, each placed at the image nearest its
#     neighbour), and every later frame follows the previous one atom by atom, as
#     -pbc nojump does but from an assembled start. The longest tree link is
#     printed; on the five 2KTF probe replicas of test29 it is 0.41-0.42 nm in a
#     6.71 nm box, and there, where nothing is split, the per-frame diameters
#     reproduce -pbc cluster's to 0.0003 nm (mean 4.8099 both, max 5.4252 against
#     5.4249).
#   * only the PROTEIN is measured. Protein_Struct also carries the residues of
#     ligand_residues.gs and ion_residues.gs, and that list includes
#     crystallisation additives: sulfate built as neutral H2SO4 in 6 pass-2
#     structures, glycol and PEG fragments, N-glycans cut loose from their
#     asparagine. Detached, they diffuse through the solvent and size the box by
#     their random walk, not by the complex. Leaving out everything that is not
#     polymer costs the box nothing it is there for. It is there for the pull code,
#     whose groups are the elastic network and the Boresch anchors, both protein
#     atoms, and the interface springs, whose ends are protein atoms or ligand and
#     ion atoms within the contact cutoff of the OTHER partner (make_boresch.py
#     takes spring candidates from every non-hydrogen solute atom), i.e. between
#     the two proteins and inside the complex's envelope, never at its edge. A
#     ligand or ion bound anywhere else sits in a pocket or on side chains, within
#     or a few Angstrom outside the protein's envelope, which the 2 x 1.5 nm of
#     padding absorbs, and the 0.49-box check never sees it. Polymer means a
#     residue with atoms N, CA and C (any amino acid, NCAAs included) or a cap
#     residue; see pbc_tools.polymer_atoms. What was left out is listed on stderr.
#   * the whole equilibration at a high PERCENTILE, neither the frame being
#     re-solvated nor the running maximum. The last frame is one draw: on 2KTF it
#     once read 4.6078 nm against 5.3441 over the run, a 0.74 nm difference that
#     is the whole margin. The maximum of n frames grows with n (as sqrt(2 ln n)
#     standard deviations for a Gaussian), so the running max made the box a
#     function of how many frames the probe happened to write, and one bad frame,
#     a PBC artefact or a single transient excursion, set it outright.
#     Measured on the 2KTF probe of test29 (5 x 2001 frames 10 ps apart,
#     diameter mean 4.810, sd 0.154 nm): the running max reads 5.212 after the
#     first 1000 frames and 5.425 after 5000, and the five replica maxima alone
#     span 5.01-5.43, while the 99th percentile of the first k frames stays within
#     5.048-5.167 for every k from 1000 up. It is a property of the ensemble far
#     more than of the sample size, but the frames are not independent: per
#     replica the diameter's autocorrelation is 0.40-0.59 at 100 ps and 0.16-0.37
#     at 500 ps, its integrated autocorrelation time 0.5-2.4 ns, so the 10,005
#     frames hold about 125 independent ones and the 100 set aside, 1 ns of the
#     100 ns, cover 0.4-2 decorrelation times. A single bad frame cannot move p99;
#     one excursion lasting about a nanosecond can. Its sampling noise is small
#     against the pad all the same: sd 0.014 nm by a 2 ns block bootstrap within
#     the replicas, 0.035 nm (95% range 5.07-5.21) by resampling whole replicas.
#     The price on a clean trajectory: on that probe p99 is 5.164 nm against a
#     maximum of 5.425, 0.26 nm, which builds the 2KTF production box at 8.22 nm
#     instead of 8.48 and puts the pull limit (0.49L after shrinkage) at 4.00 nm
#     instead of 4.13. What that limit has to hold reaches 2.36 nm on the current
#     protocol: over the 1,200 frames of bindfwdI, bindrevI and bindrevH in all 50
#     test29 cycles (the 2KTF spring set v1.5 builds) the longest spring is
#     2.361 nm, and at the start of the 50 x 43 legs no pull coordinate exceeds
#     2.256 nm (Boresch r 2.147 nm): 1.64 nm of margin, d/limit 0.59. The 3.77 nm
#     pair that aborted the old fixed -d 1.5 box (d5e792f, 2026-08-12) and the
#     3.56 nm pair of ee45e12 came before the Boresch dihedral sign fix (b4c267f,
#     2026-08-13); its mirrored references made the partners reorient, so those
#     pairs say nothing about the margin now. Should the pilot show 0.49-box
#     refusals after all, the remedy is a higher PERCENTILE, not the maximum
#     again: p99.9 is 5.281 on the same frames. It is computed as the largest
#     diameter left once the widest floor(n * (100 - P) / 100) frames are set
#     aside, so the cut is a whole number of frames, and with fewer than 100
#     frames at P = 99 it is the maximum.
#   * NPT shrinkage is pre-paid (--compression), and the -c structure's own
#     diameter is measured so the returned editconf -d produces the requested L on
#     THAT file. That diameter is the whole file's, ligands included, because that
#     is what editconf pads.
#
# The equilibration is run in a SMALL box, since the bound complex never needs the
# production box: on 2KTF the largest checked pull pair in the bound reference is
# 1.18 nm against a limit of 3.29 nm even at -d 1.00. The measurement is what the
# small box is for.
#
# Usage:
#   python3 measure_box.py -f npt_init.xtc -s npt_init.tpr -n index.ndx \
#           -g Protein_Struct -c npt_init.gro --pad 1.5
#
# Prints the editconf -d value to use for the production box on stdout, and the
# measurement on stderr.

import os, sys, argparse, subprocess, shutil, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_tools as pbc

# The box is sized from this percentile of the per-frame diameters; see the header.
PERCENTILE = 99


def build_parser():
  p = argparse.ArgumentParser(description="Size the production box from an equilibration trajectory.")
  p.add_argument('-f', '--traj', nargs='+', default=["npt_init.xtc"],
                 help="Equilibration trajectory, or several. The percentile is "
                      "taken over the frames of ALL of them pooled: with independent "
                      "probe replicas the production box has to hold whichever one "
                      "swung widest.")
  p.add_argument('-s', '--tpr', nargs='+', default=["npt_init.tpr"],
                 help="Run input matching --traj. One, broadcast to every "
                      "trajectory, or one per trajectory.")
  p.add_argument('-n', '--index', default="index.ndx", help="Index file.")
  p.add_argument('-g', '--group', default="Protein_Struct",
                 help="Group extracted from the trajectory. Only its polymer "
                      "residues are measured; !Water would also do, but costs a "
                      "much larger extraction (default: Protein_Struct).")
  p.add_argument('-c', '--struct', default=None,
                 help="Structure the production box will actually be built from. Its "
                      "own diameter is measured so the -d that comes back produces "
                      "the requested L on THAT file.")
  p.add_argument('--pad', type=float, default=1.5,
                 help="Clearance per side, in nm. L = D + 2*pad (default: 1.5).")
  p.add_argument('--compression', type=float, default=0.9932,
                 help="Expected NPT box shrinkage, pre-paid so the equilibrated box "
                      "still meets the target (default: 0.9932, measured on 2KTF).")
  p.add_argument('--min-d', type=float, default=0.0,
                 help="Floor on the returned padding, e.g. the solvation floor.")
  p.add_argument('-b', '--begin', type=float, default=0.0, help="Skip to this time, ps.")
  p.add_argument('--percentile', type=float, default=PERCENTILE,
                 help="Percentile of the per-frame diameters the box is sized from "
                      "(default: %g)." % PERCENTILE)
  return p


def fail(msg):
  sys.stderr.write("measure_box: ERROR - %s\n" % msg)
  sys.exit(1)


def max_pairwise(xyz):
  """Exact largest distance between any two points.

  The convex hull carries the diameter, so hull vertices are enough and are far
  cheaper than the full N^2. Falls back to brute force if scipy is unavailable or
  the hull is degenerate (collinear/coplanar input)."""
  pts = xyz
  try:
    from scipy.spatial import ConvexHull
    if len(xyz) > 8:
      pts = xyz[ConvexHull(xyz).vertices]
  except Exception:
    pts = xyz
  d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
  return float(np.sqrt(d2.max()))


def gro_diameter(path):
  """Diameter of a single-frame .gro, i.e. its largest atom-atom distance.

  This is exactly the quantity editconf calls "diameter" and pads with -d
  (verified: editconf reports 4.713 on 2KTF's emin_vac.gro and brute force gives
  4.71254). Computed here rather than parsed from editconf, which only prints the
  diameter line for -bt dodecahedron and would make this depend on the box type
  of a probe run."""
  for fr in pbc.iter_gro(path):
    return max_pairwise(fr.xyz)
  return None


def frames_set_aside(n, pct=PERCENTILE):
  """floor(n * (100 - pct) / 100), the widest frames the percentile ignores. The
  1e-9 keeps 200 frames at 99 from losing one to binary rounding."""
  return int(np.floor(n * (100.0 - pct) / 100.0 + 1e-9))


def percentile_value(values, pct=PERCENTILE):
  """The largest value left once the frames_set_aside() widest are dropped: a
  nearest-rank percentile whose cut is a whole number of frames. With fewer than
  100 frames at pct 99 nothing is set aside and this is the maximum."""
  v = np.sort(np.asarray(values, dtype=float))
  if not len(v):
    return float("nan")
  return float(v[len(v) - 1 - frames_set_aside(len(v), pct)])


def measure_frames(frames):
  """Per-frame diameters of the polymer atoms of one trajectory, reassembled.

  `frames` yields pbc.Frame objects of the extracted group, the first one with its
  atom heads. The first frame is assembled in space and every later one follows
  its predecessor, so each frame is measured as one piece. Returns
  (diameters, info) with info describing what was measured."""
  it = iter(frames)
  first = next(it, None)
  if first is None:
    return [], {}
  blocks = pbc.residue_blocks(first.head)
  sel = pbc.polymer_atoms(first.head, blocks)
  measured_all = not len(sel)
  if measured_all:                  # no polymer at all: measure the group as given
    sel = np.arange(len(first))
  head = [first.head[i] for i in sel]
  sub = pbc.residue_blocks(head)
  cur, longest = pbc.assemble(first.xyz[sel], first.box, sub,
                              pbc.anchor_atoms(head, sub))
  out = [max_pairwise(cur)]
  for fr in it:
    cur = pbc.follow(cur, fr.xyz[sel], fr.box)
    out.append(max_pairwise(cur))
  # What was left out, by residue name, so a reader sees an additive leave the
  # measurement rather than having to trust that it did.
  keep = np.zeros(len(first), dtype=bool)
  keep[sel] = True
  excluded = {}
  for s, e in blocks:
    if not keep[s]:
      name = first.head[s][5:10].strip()
      excluded[name] = excluded.get(name, 0) + 1
  return out, dict(n_group=len(first), n_measured=len(sel), longest=longest,
                   excluded=excluded, measured_all=measured_all,
                   shortest_box=float(min(np.linalg.norm(first.box, axis=1))))


def main():
  args = build_parser().parse_args()
  if not 50 <= args.percentile <= 100:
    fail("--percentile %g is outside 50..100" % args.percentile)
  if len(args.tpr) == 1 and len(args.traj) > 1:
    args.tpr = args.tpr * len(args.traj)
  if len(args.tpr) != len(args.traj):
    fail("%d --traj against %d --tpr; give one tpr, or one per trajectory"
         % (len(args.traj), len(args.tpr)))
  for f in list(args.traj) + list(args.tpr) + [args.index]:
    if not os.path.isfile(f):
      fail("%s not found" % f)
  if not shutil.which("gmx"):
    fail("gmx not on PATH")

  # The group is extracted as mdrun wrote it, with no -pbc treatment at all: the
  # reassembly below does the PBC work, and -pbc cluster is what failed.
  work = tempfile.mkdtemp(prefix="measure_box.")
  sub = os.path.join(work, "solute.gro")
  per_frame, per_rep = [], []
  try:
    for traj, tpr in zip(args.traj, args.tpr):
      if os.path.isfile(sub):
        os.remove(sub)
      cmd = ["gmx", "trjconv", "-s", tpr, "-f", traj, "-n", args.index,
             "-o", sub, "-b", "%g" % args.begin]
      r = subprocess.run(cmd, input="%s\n" % args.group,
                         capture_output=True, text=True)
      if r.returncode != 0 or not os.path.isfile(sub):
        fail("trjconv failed for group %s on %s:\n%s"
             % (args.group, traj, r.stderr[-1500:]))
      d, info = measure_frames(pbc.iter_gro(sub, names=False))
      if not d:
        continue
      per_frame.extend(d)
      per_rep.append((os.path.basename(traj), d, info))
  finally:
    shutil.rmtree(work, ignore_errors=True)

  if not per_frame:
    fail("no usable frames in %s" % ", ".join(args.traj))

  info = per_rep[0][2]
  if info["measured_all"]:
    sys.stderr.write("measure_box: WARNING - group %s has no polymer residue; "
                     "measuring all %d of its atoms\n" % (args.group, info["n_group"]))
  else:
    gone = ", ".join("%d x %s" % (n, k) for k, n in sorted(info["excluded"].items()))
    sys.stderr.write("measure_box: group %s, %d atoms; measuring the %d polymer atoms%s\n"
                     % (args.group, info["n_group"], info["n_measured"],
                        (" (left out: %s)" % gone) if gone else ""))
  for name, d, inf in per_rep:
    sys.stderr.write("measure_box: %s %d frames, median %.4f  p%g %.4f  max %.4f nm; "
                     "longest assembly link %.2f nm in a %.2f nm box\n"
                     % (name, len(d), np.median(d), args.percentile,
                        percentile_value(d, args.percentile), max(d),
                        inf["longest"], inf["shortest_box"]))

  a = np.array(per_frame)
  D = percentile_value(a, args.percentile)
  drop = frames_set_aside(len(a), args.percentile)
  sys.stderr.write("measure_box: %d frames pooled, protein diameter: mean %.4f median %.4f "
                   "min %.4f std %.4f, max %.4f nm\n"
                   % (len(a), a.mean(), np.median(a), a.min(), a.std(), a.max()))
  sys.stderr.write("measure_box: D = p%g = %.4f nm, the widest of the %d frames left "
                   "once the %d widest are set aside (the running max would add %.4f)\n"
                   % (args.percentile, D, len(a) - drop, drop, a.max() - D))

  # Target box vector, with the NPT shrinkage pre-paid so the EQUILIBRATED box
  # still meets it rather than the freshly built one.
  L_target = (D + 2.0 * args.pad) / args.compression
  sys.stderr.write("measure_box: L = D %.4f + 2 * pad %.2f = %.4f nm, "
                   "/ compression %.4f -> build at %.4f nm (pull limit 0.49L = "
                   "%.4f after shrinkage)\n"
                   % (D, args.pad, D + 2.0 * args.pad, args.compression,
                      L_target, 0.49 * L_target * args.compression))

  if args.struct:
    if not os.path.isfile(args.struct):
      fail("%s not found" % args.struct)
    d_struct = gro_diameter(args.struct)
    if d_struct is None:
      fail("could not read any frame from %s" % args.struct)
    pad_d = (L_target - d_struct) / 2.0
    sys.stderr.write("measure_box: %s has diameter %.4f nm, so editconf -d %.3f "
                     "gives L = %.4f\n" % (args.struct, d_struct, pad_d, L_target))
  else:
    pad_d = args.pad

  if pad_d < args.min_d:
    sys.stderr.write("measure_box: raising -d %.3f to the floor %.3f\n"
                     % (pad_d, args.min_d))
    pad_d = args.min_d

  print("%.3f" % pad_d)


if __name__ == "__main__":
  main()
