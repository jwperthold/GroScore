#!/usr/bin/env python3
#
# groscore_fe.py - Free-energy (absolute-binding) variant of GroScore.
#
# Unlike the classic engine (groscore.py), which reports a relative pull-work
# "score", this variant estimates an absolute binding free energy from a
# thermodynamic cycle built out of bidirectional non-equilibrium switching:
#
#   unrestrained-bound
#        | +dG_intro     (bound-state restraint introduction; dhdl)
#   interface-restrained-bound
#        | +dG_unbind     (interface -> Boresch handoff + separation to 1.5 nm;
#        |                 pull-force work + dhdl work)
#   Boresch-restrained-unbound
#        | +dG_release    (analytical Boresch standard-state term, eq. 32)
#   unrestrained-unbound (1 mol/L standard state)
#
#   dG_bind = -(dG_intro + dG_unbind + dG_release)
#
# Each simulated leg is run forward and reverse per cycle; the free energies are
# estimated with the Crooks-Gaussian-Intersection (CGI) and the simple average
# estimator, reusing the machinery of the classic engine.
#
# job_fe.run writes, per completed cycle, a line to results_fe.gs:
#   STRUCT_ID cycle W_intro Wunbind_pull Wunbind_dhdl Wrebind_pull Wrebind_dhdl W_remove
# and, per structure, a line to results_analytical.gs:
#   STRUCT_ID dG_release_kJ_mol
# All works are in kJ/mol.
#
# SIGN CONVENTIONS (physical work, all kJ/mol):
#   * dhdl work (integrate_dhdl.py) is the physical switching work along the ramp:
#       forward  ramp lambda 0->1 : W_fwd = int_0^1 <dH/dl> dl
#       reverse  ramp lambda 1->0 : W_rev = int_1^0 <dH/dl> dl  (opposite sign)
#   * pull work (integrate.py) returns -rate*int F dt with rate hard-coded +0.0002.
#     Mapping to physical work W = rate_actual * int F dt:
#       forward leg (rate +0.0002): W_pull_fwd = -(integrate.py output)
#       reverse leg (rate -0.0002): W_pull_rev = +(integrate.py output)
#   * A forward/reverse pair is combined with Crooks by feeding
#       pulls  = W_forward
#       pushes = -W_reverse
#     so that (mean(pulls)+mean(pushes))/2 -> dG and the CGI intersection -> dG.
#
# NOTE: the relative sign of the pull-work and dhdl-work terms, and the pull-force
# sign convention of this GROMACS build, MUST be validated on the first real
# bidirectional run using the standard CGI diagnostic (forward and reverse work
# histograms should overlap around dG). See the SIGN_* constants below.

import math, os, sys, argparse, shutil
import numpy as np

#------------------------------------------------------

parser = argparse.ArgumentParser(description="GroScore FE: absolute binding free energy via Boresch restraints.")
parser.add_argument('-n', '--numruns', type=int, default=5, help="Number of bidirectional cycles per structure (default: 5).")
parser.add_argument('-s', '--structparams', type=str, default="sp.gs", help="Structure parameter file (default: sp.gs).")
parser.add_argument('-ff', '--forcefield', type=str, default="amber19sb_opc3",
                    choices=["gromos54a8", "charmm36", "amber19sb_opc", "amber19sb_opc3"],
                    help="Force field (default: amber19sb_opc3).")
parser.add_argument('--no-cutout', dest='cutout', action='store_false', help="Disable interface cutout.")
parser.add_argument('--no-ligand-param', dest='ligand_param', action='store_false', help="Disable OpenFF small-molecule parametrization.")
parser.add_argument('--slurm', type=str, default="workstation", help="SLURM template name from slurm/ (default: workstation).")
parser.add_argument('--restart', action='store_true', help="Resubmit jobs even if run.gs exists.")
parser.add_argument('--inject-job-run', action='store_true', help="Inject fresh job_fe.run into archived (.tar.gz) structures.")
parser.add_argument('--temp', type=float, default=310.0, help="Temperature in K (default: 310).")
parser.set_defaults(cutout=True, ligand_param=True)
args = parser.parse_args()

RT = 0.00831446261815324 * args.temp  # kJ/mol

# Sign toggles to flip during first-run validation if the diagnostic requires it.
SIGN_PULL_FWD = -1.0   # W_pull_fwd = SIGN_PULL_FWD * integrate.py(fwd)
SIGN_PULL_REV = +1.0   # W_pull_rev = SIGN_PULL_REV * integrate.py(rev)

#------------------------------------------------------

def readstructparams(filepath):
  ids, chains = [], []
  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        if not line.strip().startswith("#"):
          tmp = line.split()
          try:
            ids.append(tmp[0]); chains.append(tmp[1])
          except (IndexError, AttributeError):
            pass
  return (ids, chains) if len(ids) == len(chains) and ids else ([], [])

#------------------------------------------------------

def bootstrap_stream(pulls, pushes, n_bootstrap=50000, method='avg'):
  """Bootstrap standard error of the CGI/avg estimate for one stream.
  `pushes` are already sign-aligned (i.e. -W_reverse)."""
  if len(pulls) == 0 or len(pushes) == 0:
    return float('nan')
  pa, pb = np.array(pulls), np.array(pushes)
  bi = pa[np.random.randint(0, len(pa), size=(n_bootstrap, len(pa)))]
  bj = pb[np.random.randint(0, len(pb), size=(n_bootstrap, len(pb)))]
  if method == 'avg':
    scores = (bi.mean(axis=1) + bj.mean(axis=1)) / 2.0
    return float(np.std(scores))
  if method == 'cgi' and len(pulls) > 2 and len(pushes) > 2:
    ap, vp = bi.mean(axis=1), bi.var(axis=1)
    aq, vq = bj.mean(axis=1), bj.var(axis=1)
    mask = (vp > 0) & (vq > 0) & (vp != vq)
    if not np.any(mask):
      return float('nan')
    ap, vp, aq, vq = ap[mask], vp[mask], aq[mask], vq[mask]
    dinv = 1.0 / vp - 1.0 / vq
    t1 = ap / vp - aq / vq
    t2 = np.sqrt((ap - aq)**2 / (vp * vq) + 2.0 * dinv * np.log(vq / vp))
    s1 = (t1 + t2) / dinv
    s2 = (t1 - t2) / dinv
    mid = (ap + aq) / 2.0
    scores = np.where(np.abs(mid - s1) > np.abs(mid - s2), s2, s1)
    return float(np.std(scores))
  return float('nan')

def cgi_point(pulls, pushes):
  """CGI intersection for one stream (pushes already sign-aligned)."""
  if len(pulls) < 3 or len(pushes) < 3:
    return float('nan')
  ap, vp = np.mean(pulls), np.var(pulls)
  aq, vq = np.mean(pushes), np.var(pushes)
  if vp <= 0 or vq <= 0 or vp == vq:
    return float('nan')
  dinv = 1.0 / vp - 1.0 / vq
  t1 = ap / vp - aq / vq
  t2 = math.sqrt((ap - aq)**2 / (vp * vq) + 2.0 * dinv * math.log(vq / vp))
  s1 = (t1 + t2) / dinv
  s2 = (t1 - t2) / dinv
  mid = (ap + aq) / 2.0
  return s2 if math.fabs(mid - s1) > math.fabs(mid - s2) else s1

def estimate_stream(forward, reverse):
  """Return dict with avg/cgi free energy (kJ/mol) and 95% CIs for one stream.
  forward = physical forward works; reverse = physical reverse works."""
  pulls = list(forward)
  pushes = [-w for w in reverse]   # sign-align the reverse process
  out = dict(n=min(len(pulls), len(pushes)),
             avg=float('nan'), avg_ci=float('nan'),
             cgi=float('nan'), cgi_ci=float('nan'))
  if pulls and pushes:
    out['avg'] = (np.mean(pulls) + np.mean(pushes)) / 2.0
    se = bootstrap_stream(pulls, pushes, method='avg')
    out['avg_ci'] = 1.96 * se if not math.isnan(se) else float('nan')
  if len(pulls) > 2 and len(pushes) > 2:
    out['cgi'] = cgi_point(pulls, pushes)
    se = bootstrap_stream(pulls, pushes, method='cgi')
    out['cgi_ci'] = 1.96 * se if not math.isnan(se) else float('nan')
  return out

#------------------------------------------------------

def setup_and_submit(structids, structchains):
  """Write per-structure run.gs, copy job_fe.run, build and submit the array."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  job_src = os.path.join(script_dir, "job_fe.run")
  if not os.path.isfile(job_src):
    print("Error: job_fe.run not found in %s" % script_dir); sys.exit(1)

  with open("struct_map.gs", "w") as f:
    f.write("# Array_Index Structure_ID\n")
    for i, sid in enumerate(structids):
      f.write("%d %s\n" % (i, sid))

  for i, sid in enumerate(structids):
    if os.path.isdir("./%s" % sid):
      print("Setting up %s." % sid)
      if not args.restart:
        with open("./%s/run.gs" % sid, "w") as f:
          f.write("%s %d %d %s %d\n" % (structchains[i], args.numruns * 2,
                                        1 if args.cutout else 0, args.forcefield,
                                        1 if args.ligand_param else 0))
      # Install atomically: copy to a temp file, then rename over job.run. A
      # rename gives the new file a fresh inode, so if a previous SLURM job is
      # still executing the old job.run its bash keeps reading the old inode and
      # finishes cleanly. An in-place copy (shutil.copy onto the live path)
      # truncates and rewrites the same inode, corrupting a running script
      # mid-execution (observed: interleaved cycle-based and PUSH_IDX naming).
      dst = "./%s/job.run" % sid
      tmp = "./%s/.job.run.%d.tmp" % (sid, os.getpid())
      shutil.copy(job_src, tmp)
      os.chmod(tmp, 0o755)
      os.replace(tmp, dst)
    elif os.path.isfile("./%s.tar.gz" % sid):
      if args.inject_job_run:
        import tarfile
        tmpdir = "./%s" % sid
        os.makedirs(tmpdir, exist_ok=True)
        shutil.copy(job_src, os.path.join(tmpdir, "job.run"))
        os.chmod(os.path.join(tmpdir, "job.run"), 0o755)
        arc = "./%s.tar.gz" % sid
        new = "./%s_new.tar.gz" % sid
        with tarfile.open(arc, "r:gz") as old_tar, tarfile.open(new, "w:gz") as new_tar:
          for m in old_tar:
            if m.name.endswith("/job.run"):
              continue
            new_tar.addfile(m, old_tar.extractfile(m) if m.isreg() else None)
          new_tar.add(os.path.join(tmpdir, "job.run"), arcname="./%s/job.run" % sid)
        os.replace(new, arc)
        shutil.rmtree(tmpdir)
        print("Injected job_fe.run into %s.tar.gz" % sid)
      else:
        print("Skipping %s.tar.gz (archived, use --inject-job-run)." % sid)
    else:
      print("Structure %s: directory doesn't exist." % sid)
      with open("results_0.gs", "a") as f:
        f.write("%s NODIR\n" % sid)

  slurm_template = os.path.join(script_dir, "slurm", args.slurm + ".sh")
  if not os.path.isfile(slurm_template):
    print("Error: SLURM template not found: %s" % slurm_template); sys.exit(1)
  with open(slurm_template) as t:
    template = t.read()
  with open("array_submit.run", "w") as f:
    f.write(template.rstrip("\n") + "\n")
    f.write("#SBATCH --array=0-%d\n\n" % (len(structids) - 1))
    f.write("STRUCT_ID=$(awk -v idx=\"$SLURM_ARRAY_TASK_ID\" '$1 == idx {print $2}' struct_map.gs)\n")
    f.write("if [[ ! -d \"$STRUCT_ID\" && -f \"${STRUCT_ID}.tar.gz\" ]]; then\n")
    f.write("  tar -xzf \"${STRUCT_ID}.tar.gz\"\n  rm \"${STRUCT_ID}.tar.gz\"\nfi\n")
    f.write("cd $STRUCT_ID\n./job.run\n")
  os.system("sbatch array_submit.run")
  print("Submitted all FE simulation jobs.\n")

#------------------------------------------------------

def read_status(filepath, structids):
  """Read results_0.gs -> {struct_id: status} for non-OK stage-0 outcomes."""
  status = {}
  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        if line.strip().startswith("#"):
          continue
        tmp = line.split()
        if len(tmp) >= 2 and tmp[0] in structids and tmp[1] != "OK":
          status[tmp[0]] = tmp[1]
  return status

def read_analytical(filepath):
  """Read results_analytical.gs -> {struct_id: dG_release_kJ_mol}."""
  vals = {}
  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        if line.strip().startswith("#"):
          continue
        tmp = line.split()
        if len(tmp) >= 2:
          try:
            vals[tmp[0]] = float(tmp[1])
          except ValueError:
            pass
  return vals

def read_works(filepath):
  """Read results_fe.gs -> {struct_id: [ (cycle, W_intro, Wu_pull, Wu_dhdl,
  Wr_pull, Wr_dhdl, W_remove), ... ]} keeping only rows with all values numeric."""
  works = {}
  if os.path.isfile(filepath):
    with open(filepath) as f:
      for line in f:
        if line.strip().startswith("#"):
          continue
        tmp = line.split()
        if len(tmp) < 8:
          continue
        sid = tmp[0]
        try:
          cyc = int(tmp[1])
          vals = [float(x) for x in tmp[2:8]]
        except ValueError:
          continue
        if any(math.isnan(v) for v in vals):
          continue
        works.setdefault(sid, []).append((cyc, *vals))
  return works

#------------------------------------------------------

def score(structids):
  status = read_status("results_0.gs", structids)
  analytical = read_analytical("results_analytical.gs")
  works = read_works("results_fe.gs")

  rows = []  # (sid, dGbind_avg, ci_avg, dGbind_cgi, ci_cgi, dGintro, dGunbind, dGrelease, ncyc, note)
  for sid in structids:
    if sid in status:
      rows.append((sid, None, None, None, None, None, None, None, 0, status[sid]))
      continue
    cycles = works.get(sid, [])
    # deduplicate by cycle index (restart safety), keep last occurrence
    by_cycle = {}
    for row in cycles:
      by_cycle[row[0]] = row
    cycles = [by_cycle[c] for c in sorted(by_cycle)]
    if not cycles or sid not in analytical:
      rows.append((sid, None, None, None, None, None, None, None, len(cycles), "PENDING"))
      continue

    W_intro   = [c[1] for c in cycles]
    Wu_pull   = [c[2] for c in cycles]
    Wu_dhdl   = [c[3] for c in cycles]
    Wr_pull   = [c[4] for c in cycles]
    Wr_dhdl   = [c[5] for c in cycles]
    W_remove  = [c[6] for c in cycles]

    # Physical total works for the unbinding/rebinding stream.
    Wtot_f = [SIGN_PULL_FWD * up + ud for up, ud in zip(Wu_pull, Wu_dhdl)]
    Wtot_r = [SIGN_PULL_REV * rp + rd for rp, rd in zip(Wr_pull, Wr_dhdl)]

    bound  = estimate_stream(W_intro, W_remove)   # dG_intro
    unbind = estimate_stream(Wtot_f, Wtot_r)      # dG_unbind
    dG_release = analytical[sid]

    def combine(intro, unb):
      if math.isnan(intro) or math.isnan(unb):
        return float('nan')
      return -(intro + unb + dG_release)

    dGbind_avg = combine(bound['avg'], unbind['avg'])
    dGbind_cgi = combine(bound['cgi'], unbind['cgi'])
    ci_avg = math.sqrt((bound['avg_ci'] or 0)**2 + (unbind['avg_ci'] or 0)**2) \
             if not (math.isnan(bound['avg_ci']) or math.isnan(unbind['avg_ci'])) else float('nan')
    ci_cgi = math.sqrt((bound['cgi_ci'] or 0)**2 + (unbind['cgi_ci'] or 0)**2) \
             if not (math.isnan(bound['cgi_ci']) or math.isnan(unbind['cgi_ci'])) else float('nan')

    rows.append((sid, dGbind_avg, ci_avg, dGbind_cgi, ci_cgi,
                 bound['avg'], unbind['avg'], dG_release, len(cycles), ""))

  # Report binding free energies in kJ/mol and as pKD (never kcal/mol).
  # dG_bind = -RT ln(Ka) = RT ln(KD)  =>  pKD = -log10(KD) = -dG_bind / (RT ln 10).
  RTLN10 = RT * math.log(10.0)
  def pkd(x):
    return -x / RTLN10 if (x is not None and not (isinstance(x, float) and math.isnan(x))) else float('nan')

  rows_valid = [r for r in rows if r[1] is not None and not math.isnan(r[1])]
  rows_valid.sort(key=lambda r: r[1])

  with open("scores_fe.gs", "w") as f:
    f.write("# GroScore-FE absolute binding free energies (kJ/mol; pKD dimensionless, T=%.1f K)\n" % args.temp)
    f.write("# Structure_ID  dGbind_avg_kJ  CI95_kJ  pKD_avg  dGbind_cgi_kJ  CI95_kJ  pKD_cgi  dG_intro_kJ  dG_unbind_kJ  dG_release_kJ  Ncycles  Note\n")
    for sid, a, aci, c, cci, gi, gu, gr, n, note in rows:
      if a is None:
        f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n"
                % (sid, note or "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", n, note or ""))
      else:
        aci_s = ("%.2f" % aci) if not math.isnan(aci) else "nan"
        cgi_s = ("%.2f" % c) if not math.isnan(c) else "nan"
        cgi_ci_s = ("%.2f" % cci) if not math.isnan(cci) else "nan"
        cgi_pkd_s = ("%.2f" % pkd(c)) if not math.isnan(c) else "nan"
        f.write("%s\t%.2f\t%s\t%.2f\t%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%d\t%s\n"
                % (sid, a, aci_s, pkd(a), cgi_s, cgi_ci_s, cgi_pkd_s, gi, gu, gr, n, note))

  done = len(rows_valid)
  print("Scored %d/%d structures with complete cycles. Wrote scores_fe.gs." % (done, len(structids)))

#------------------------------------------------------

def main():
  print("")
  print("#################################")
  print("#                               #")
  print("#       GroScore-FE 0.1         #")
  print("#                               #")
  print("#################################")
  print("")

  structids, structchains = readstructparams(args.structparams)
  if not structids:
    print("Error: No valid structures found in %s" % args.structparams); sys.exit(1)
  print("GroScore-FE: %d structures, %d bidirectional cycles each.\n" % (len(structids), args.numruns))

  # Submit jobs on first invocation (or on --restart).
  if args.restart or not os.path.isfile("struct_map.gs"):
    if not os.path.isfile("results_0.gs"):
      with open("results_0.gs", "w") as f:
        f.write("# Stage-0 setup status\n")
    setup_and_submit(structids, structchains)

  # Score whatever results are currently available.
  score(structids)

if __name__ == "__main__":
  main()
