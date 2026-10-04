#!/usr/bin/env python3
"""The v1.5 cycle: where each leg starts, which pull block it runs, and the runner
pieces that depend on them.

The defects behind each section are in bm_fe_amber_2/_ana/digest/ (pass 2 of the
HADDOCK benchmark, 46 complexes, v1.3 physics).

  [1] THE TABLE. The whole cycle in run order is written out below rather than
      derived: every leg's name, mdp, pull block ("block") and start
      ("start_from"). There is no holdrev0 any more, and boundrev2 starts from
      holdfwd0, because the rebound interface at the end of the reverse ramp sat a
      median 14 kJ/mol of spring energy above the forward state and the 1 ns
      holdrev0 removed a median 8% of that (fu3.txt F6, rlx_handoff.py). The two
      bound holds run the bound block like the sub-legs either side, where v1.3
      switched them to the unbind block and back unrecorded (fu3.txt F1).
      make_boresch.py writes every pull block from "block" alone, so a wrong entry
      is a wrong Hamiltonian in every cycle.
  [2] THE EXPORT. fe_protocol.py --shell carries FE_LEG_FROM, one quoted element
      per leg, job_fe.run refuses a protocol without it, and the result row did not
      move: 43 fields, the same 40 works in the same order. The pairing is checked
      the other way too (review CYC-2): the export's first line exits 1 with FATAL,
      defining nothing, unless FE_RUNNER_HONOURS_START_FROM=1, which a job.run from
      before v1.5 never sets and job_fe.run sets, unexported, right before its eval.
  [3] THE SETTINGS. npt_c_fe.mdp exists in all five trees: npt_fe.mdp's physics
      line for line plus the pull print header, with no pull coordinates and no
      free-energy block, because make_boresch.py appends the elastic network per
      structure (fu3.txt F10: the network switched on unrecorded at boundfwd1,
      20 of 920 cycles lost). No tree ships holdrev0_fe.mdp or any other FE mdp the
      protocol does not define. Every hold runs free-energy = yes, which is why
      mdrun writes its dH/dlambda, and the generated headers say what each file is
      (review CYC-9 and the critic's false-docs list).
  [4] THE LEG LOOP of job_fe.run, extracted verbatim and driven by a fake gmx that
      records its arguments: boundrev2 is grompp'd from holdfwd0_<c>.tpr and .cpt
      and every other leg from the leg before it, also on a restart that skips the
      finished legs, and a missing branch source stops the cycle. The rebinding
      QC still compares npt_c with the end of bindrevA. fe_xvgs names every xvg a
      cycle writes, the holds' <hold>_<c>.xvg included, so they are gzipped and kept.
  [5] npt_c, extracted the same way from the cycle header to its require_gro: it is
      grompp'd from ./npt_c_fe.mdp, the setup copies that file and checks that
      make_boresch.py completed it, and without it the cycle stops with a message
      instead of falling back to npt_fe.mdp. The cycle also stops unless
      protocol_id.gs names the protocol the code runs. Both stops hold for EVERY
      leg, not just npt_c: a cycle whose npt_c, or all but one leg, finished under
      an older setup stops too (review CYC-5: a guard on npt_c alone passed every
      check here before), and only a cycle with every leg done goes on, to
      integration, with no gmx call.
  [6] THE BOX. The measured padding is clamped into [FLOOR, 4.0], never discarded:
      v1.3 replaced eleven requests of 4.16-7.74 nm by the 1.8 fallback (setup.txt
      S5). 3.99 and 4.00 pass unchanged, 4.01 and 2AJF's 7.737 become 4.0, the
      floor side the same way, and the log says when it clamped; the fallback only
      without boxpad.gs, and a boxpad.gs holding no number, an empty file
      included, is a PROBE_FAIL. A value with DOS line ends is read as the number
      it is (review CYC-10).
  [7] LIGAND_FAIL. A non-empty ligand_failed.gs stops the setup with
      'ID LIGAND_FAIL' in results_0.gs and exit 0; empty or absent goes on
      (setup.txt S3, S4: dropped AlF3, GNP, ATP and iodides went unreported). The
      check sits after both merge_ligand.py calls, and the restart cleanup removes
      the file. merge_ligand.py is told why ligands are or are not parametrised:
      on for AMBER19SB, off-user for --no-ligand-param, off-ff for CHARMM36 and
      GROMOS 54A8 (contract F4).

Expected values are fixed numbers and names, never read back from the constants
under test, and each threshold is checked just below, at and just above.

Standalone, no pytest: python3 tests/test_cycle_v15.py
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))
import fe_protocol as P

JOB = os.path.join(ROOT, "job_fe.run")
SRC = open(JOB).read()
failures = []


def check(name, ok, detail=""):
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % str(detail)[:300]))
    if not ok:
        failures.append(name)


# The v1.5 cycle, in run order: leg, mdp, pull block, start ('-' = the leg before).
CYCLE = """
boundfwd1   boundfwd1.mdp      bound    -
holdbfwd1   holdbfwd1_fe.mdp   bound    -
boundfwd2   boundfwd2.mdp      bound    -
holdfwd0    holdfwd0_fe.mdp    unbind   -
bindfwdA    bindfwdA_fe.mdp    unbind   -
holdfwd1    holdfwd1_fe.mdp    unbind   -
bindfwdB    bindfwdB_fe.mdp    unbind   -
holdfwd2    holdfwd2_fe.mdp    unbind   -
bindfwdC    bindfwdC_fe.mdp    unbind   -
holdfwd3    holdfwd3_fe.mdp    unbind   -
bindfwdD    bindfwdD_fe.mdp    unbind   -
holdfwd4    holdfwd4_fe.mdp    unbind   -
bindfwdE    bindfwdE_fe.mdp    unbind   -
holdfwd5    holdfwd5_fe.mdp    unbind   -
bindfwdF    bindfwdF_fe.mdp    unbind   -
holdfwd6    holdfwd6_fe.mdp    unbind   -
bindfwdG    bindfwdG_fe.mdp    unbind   -
holdfwd7    holdfwd7_fe.mdp    unbind   -
bindfwdH    bindfwdH_fe.mdp    unbind   -
holdfwd8    holdfwd8_fe.mdp    unbind   -
bindfwdI    bindfwdI_fe.mdp    unbind   -
nptrev      nptrev_fe.mdp      unbind   -
bindrevI    bindrevI_fe.mdp    unbind   -
holdrev8    holdrev8_fe.mdp    unbind   -
bindrevH    bindrevH_fe.mdp    unbind   -
holdrev7    holdrev7_fe.mdp    unbind   -
bindrevG    bindrevG_fe.mdp    unbind   -
holdrev6    holdrev6_fe.mdp    unbind   -
bindrevF    bindrevF_fe.mdp    unbind   -
holdrev5    holdrev5_fe.mdp    unbind   -
bindrevE    bindrevE_fe.mdp    unbind   -
holdrev4    holdrev4_fe.mdp    unbind   -
bindrevD    bindrevD_fe.mdp    unbind   -
holdrev3    holdrev3_fe.mdp    unbind   -
bindrevC    bindrevC_fe.mdp    unbind   -
holdrev2    holdrev2_fe.mdp    unbind   -
bindrevB    bindrevB_fe.mdp    unbind   -
holdrev1    holdrev1_fe.mdp    unbind   -
bindrevA    bindrevA_fe.mdp    unbind   -
boundrev2   boundrev2.mdp      bound    holdfwd0
holdbrev1   holdbrev1_fe.mdp   bound    -
boundrev1   boundrev1.mdp      bound    -
"""
TABLE = [l.split() for l in CYCLE.strip().splitlines()]
NAMES = [t[0] for t in TABLE]
# Where each leg's grompp must read from, straight off the table.
SOURCE = {}
for i, (name, _, _, frm) in enumerate(TABLE):
    SOURCE[name] = frm if frm != "-" else (TABLE[i - 1][0] if i else "npt_c")

WORK_FIELDS = ("W_intro1 W_intro2 "
               "WuA_pull WuA_dhdl WuB_pull WuB_dhdl WuC_pull WuC_dhdl WuD_pull WuD_dhdl "
               "WuE_pull WuE_dhdl WuF_pull WuF_dhdl WuG_pull WuG_dhdl WuH_pull WuH_dhdl "
               "WuI_pull WuI_dhdl "
               "WrI_pull WrI_dhdl WrH_pull WrH_dhdl WrG_pull WrG_dhdl WrF_pull WrF_dhdl "
               "WrE_pull WrE_dhdl WrD_pull WrD_dhdl WrC_pull WrC_dhdl WrB_pull WrB_dhdl "
               "WrA_pull WrA_dhdl W_remove2 W_remove1").split()

TREES = ["amber19sb_opc3", "amber19sb_opc", "charmm36", "gromos54a8", "gromos54a8_rf"]

# The protocol id of this table, written out (tests/test_protocol_id.py pins how it
# is made). A setup records it in protocol_id.gs; a cycle runs only under it.
WANT_ID = "fe-1.5-f78ecb6a"
# What job_fe.run declares before it evaluates the export.
DECLARE = "FE_RUNNER_HONOURS_START_FROM=1\n"

# A gmx that does nothing but record the call and leave behind the file the call
# would have written, so the runner's control flow can be driven without GROMACS.
FAKE_GMX = r"""#!/bin/bash
echo "$*" >> "$GMXLOG"
sub=$1; shift
out=""; deffnm=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) out=$2; shift ;;
    -deffnm) deffnm=$2; shift ;;
  esac
  shift
done
case "$sub" in
  grompp)   [[ -n "$out" ]] && touch "${out%.tpr}.tpr" ;;
  mdrun)    [[ -n "$deffnm" ]] && touch "$deffnm.gro" "$deffnm.cpt" ;;
  editconf) [[ -n "$out" ]] && touch "$out" ;;
esac
exit 0
"""


def block(start, end, src=SRC):
    """The lines from the first one matching `start` to the next one matching `end`,
    inclusive, verbatim; None if either is missing."""
    lines = src.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if re.search(start, ln):
            for j in range(i + 1, len(lines)):
                if re.search(end, lines[j]):
                    return "".join(lines[i:j + 1])
            return None
    return None


def line_no(pat, src=SRC):
    """1-based numbers of every line matching pat."""
    return [i for i, ln in enumerate(src.splitlines(), 1) if re.search(pat, ln)]


def run_bash(script, cwd, gmxlog):
    fake_dir = os.path.join(cwd, ".fakebin")
    os.makedirs(fake_dir, exist_ok=True)
    fake = os.path.join(fake_dir, "gmx")
    with open(fake, "w") as f:
        f.write(FAKE_GMX)
    os.chmod(fake, 0o755)
    env = dict(os.environ, PATH=fake_dir + ":" + os.environ["PATH"], GMXLOG=gmxlog)
    return subprocess.run(["bash", "-c", script], cwd=cwd, capture_output=True,
                          text=True, env=env)


def grompp_calls(gmxlog):
    """{output name: {'-c': ..., '-t': ..., '-f': ...}} for every grompp call, and
    the list of outputs in call order."""
    calls, order = {}, []
    if not os.path.isfile(gmxlog):
        return calls, order
    for ln in open(gmxlog):
        tok = ln.split()
        if not tok or tok[0] != "grompp":
            continue
        opt = {k: tok[i + 1] for i, k in enumerate(tok[:-1]) if k in ("-c", "-t", "-f", "-o")}
        calls[opt.get("-o")] = opt
        order.append(opt.get("-o"))
    return calls, order


def shell_export():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "utils", "fe_protocol.py"),
                        "--shell"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


RUN_LEG_MDRUN = block(r"^run_leg_mdrun\(\) \{$", r"^\}$")
# Every mdrun goes through run_bg, which records the running leg in the locks
# (review S3); run_bg and its proc_start come with the leg runner.
_RUN_BG = [block(r"^%s\(\) \{$" % n, r"^\}$") for n in ("proc_start", "run_bg")]
if RUN_LEG_MDRUN is not None:
    RUN_LEG_MDRUN = None if None in _RUN_BG else "".join(_RUN_BG) + RUN_LEG_MDRUN
REQUIRE_GRO = block(r"^  require_gro\(\) \{$", r"^  \}$")
EXPORT = shell_export()

# ---------------------------------------------------------------------------
print("\n[1] the cycle table: no holdrev0, the branch, every leg's pull block")
L = P.legs()
names = [l["name"] for l in L]
check("no leg is named holdrev0", "holdrev0" not in names)
check("the legs are the 42 of the table, in run order", names == NAMES,
      "got %d legs: %s" % (len(names), " ".join(names[37:])))
check("each leg runs the mdp of the table",
      [l["mdp"] for l in L] == [t[1] for t in TABLE],
      [l["mdp"] for l in L if l["mdp"] not in [t[1] for t in TABLE]])
check("every leg carries the keys 'block' and 'start_from'",
      all("block" in l and "start_from" in l for l in L),
      [l["name"] for l in L if "block" not in l or "start_from" not in l][:4])
want_block = {t[0]: t[2] for t in TABLE}
bad = ["%s=%s" % (l["name"], l.get("block")) for l in L
       if l.get("block") != want_block.get(l["name"])]
check("each leg's block is the table's (bound legs and bound holds 'bound')",
      not bad and len(L) == 42, bad[:6])
check("  holdbfwd1 and holdbrev1 run the bound block, not the unbind one",
      [l.get("block") for l in L if l["name"] in ("holdbfwd1", "holdbrev1")]
      == ["bound", "bound"])
check("  nptrev and holdfwd0 run the unbind block",
      [l.get("block") for l in L if l["name"] in ("holdfwd0", "nptrev")]
      == ["unbind", "unbind"])
check("boundrev2 starts from holdfwd0",
      [l.get("start_from") for l in L if l["name"] == "boundrev2"] == ["holdfwd0"])
check("every other leg starts from the leg before it (start_from None)",
      all(l.get("start_from") is None for l in L if l["name"] != "boundrev2")
      and len(L) == 42,
      [(l["name"], l.get("start_from")) for l in L
       if l["name"] != "boundrev2" and l.get("start_from") is not None])

# ---------------------------------------------------------------------------
print("\n[2] the shell export and the result row")
r = subprocess.run(["bash", "-c", DECLARE + 'eval "$1"; printf "%s|" "${FE_LEG_FROM[@]}"; echo;'
                    ' echo "${#FE_LEG_FROM[@]} ${#FE_LEGS[@]}"; echo "${FE_LEGS[*]}";'
                    ' echo "$FE_RESULT_NF"; echo "${FE_WORK_FIELDS[*]}"', "_", EXPORT],
                   capture_output=True, text=True)
out = r.stdout.splitlines() + [""] * 5
froms = out[0].split("|")[:-1]
check("FE_LEG_FROM is exported", "FE_LEG_FROM=(" in EXPORT)
check("  one element per leg, 42 of each", out[1].split() == ["42", "42"], out[1])
check("  'holdfwd0' for boundrev2 (index 39), empty for the other 41",
      len(froms) == 42 and froms[39] == "holdfwd0"
      and all(f == "" for i, f in enumerate(froms) if i != 39), out[0])
check("FE_LEGS is the table", out[2].split() == NAMES, out[2])
check("the result row is still 43 fields", out[3] == "43", out[3])
check("and its 40 works are the same, in the same order", out[4].split() == WORK_FIELDS,
      out[4])
guard = block(r"^if \[\[ \$\{#FE_LEG_FROM\[@\]\} -ne \$\{#FE_LEGS\[@\]\} \]\]; then$", r"^fi$")
check("job_fe.run refuses a protocol without FE_LEG_FROM (guard exists)", guard is not None)
if guard:
    with tempfile.TemporaryDirectory() as d:
        r_old = run_bash("FE_LEGS=(a b c)\n" + guard + "echo GO\n", d, os.path.join(d, "g"))
        r_new = run_bash('FE_LEGS=(a b c); FE_LEG_FROM=("" "a" "")\n' + guard + "echo GO\n",
                         d, os.path.join(d, "g"))
    check("  a pre-v1.5 export (no FE_LEG_FROM) exits 1 with FATAL",
          r_old.returncode == 1 and "FATAL" in r_old.stderr and "GO" not in r_old.stdout,
          (r_old.returncode, r_old.stderr))
    check("  a complete export goes on", r_new.returncode == 0 and "GO" in r_new.stdout,
          (r_new.returncode, r_new.stderr))
else:
    check("  a pre-v1.5 export (no FE_LEG_FROM) exits 1 with FATAL", False, "no guard")
    check("  a complete export goes on", False, "no guard")

# The other direction (review CYC-2): a job.run from before v1.5 evaluates the export
# without declaring anything, ignores FE_LEG_FROM and would run the table as a plain
# chain, boundrev2 straight after bindrevA. The export's FIRST line stops it.
first = EXPORT.splitlines()[0] if EXPORT else ""
check("the export's first line is the runner check",
      first.startswith('[[ "${FE_RUNNER_HONOURS_START_FROM:-0}" == 1 ]] ||'), first[:80])
r_old = subprocess.run(["bash", "-c", 'eval "$1"; echo "AFTER ${#FE_LEGS[@]}"', "_", EXPORT],
                       capture_output=True, text=True)
check("  an old runner's eval exits 1 with FATAL and defines no leg",
      r_old.returncode == 1 and "FATAL" in r_old.stderr and "AFTER" not in r_old.stdout,
      (r_old.returncode, r_old.stdout[-80:], r_old.stderr[-120:]))
check("  and the message sends the structure to a fresh setup",
      "fresh run directory" in r_old.stderr, r_old.stderr[-200:])
r_zero = subprocess.run(["bash", "-c", 'FE_RUNNER_HONOURS_START_FROM=0; eval "$1"; echo AFTER',
                         "_", EXPORT], capture_output=True, text=True)
check("  a declaration other than 1 is refused too", r_zero.returncode == 1
      and "AFTER" not in r_zero.stdout, (r_zero.returncode, r_zero.stdout[-80:]))
r_new = subprocess.run(["bash", "-c", DECLARE + 'eval "$1"; echo "ID=$FE_PROTOCOL_ID"', "_",
                        EXPORT], capture_output=True, text=True)
check("  a declaring runner gets the protocol, its id included",
      r_new.returncode == 0 and ("ID=%s" % WANT_ID) in r_new.stdout,
      (r_new.returncode, r_new.stdout[-80:], r_new.stderr[-120:]))
decl = line_no(r"^FE_RUNNER_HONOURS_START_FROM=1$")
ev = line_no(r'^eval "\$FE_PROTO"$')
check("job_fe.run declares itself on the line before its eval of the export",
      len(decl) == 1 and len(ev) == 1 and decl[0] + 1 == ev[0], (decl, ev))
check("  without exporting it to the processes it starts",
      not line_no(r"export\s+FE_RUNNER_HONOURS_START_FROM"))
idg = block(r'^if \[\[ -z "\$\{FE_PROTOCOL_ID:-\}" \]\]; then$', r"^fi$")
check("job_fe.run refuses a protocol that names no FE_PROTOCOL_ID", idg is not None
      and "exit 1" in (idg or ""))
if idg:
    with tempfile.TemporaryDirectory() as d:
        r1 = run_bash(idg + "echo GO\n", d, os.path.join(d, "g"))
        r2 = run_bash('FE_PROTOCOL_ID=x\n' + idg + "echo GO\n", d, os.path.join(d, "g"))
    check("  and goes on with one", r1.returncode == 1 and "GO" not in r1.stdout
          and r2.returncode == 0 and "GO" in r2.stdout, (r1.returncode, r2.returncode))
else:
    check("  and goes on with one", False, "no FE_PROTOCOL_ID check")

# ---------------------------------------------------------------------------
print("\n[3] npt_c_fe.mdp in every tree, and nothing stale")


def settings(path):
    """[(key, value)] in file order, comments and blank lines dropped."""
    out = []
    for ln in open(path):
        ln = ln.split(";")[0].strip()
        if "=" in ln:
            k, v = ln.split("=", 1)
            out.append((k.strip(), v.strip()))
    return out


HEADER = [("pull", "yes"), ("pull-print-com1", "no"), ("pull-print-com2", "no"),
          ("pull-print-ref-value", "no"), ("pull-print-components", "no"),
          ("pull-nstxout", "0"), ("pull-nstfout", "0")]
FE_KEYS = {"free-energy", "init-lambda", "init-lambda-state", "delta-lambda",
           "dhdl-derivatives", "nstdhdl", "separate-dhdl-file", "calc-lambda-neighbors",
           "sc-alpha", "couple-moltype"}
WANT_FE_MDPS = {t[1] for t in TABLE} | {"npt_fe.mdp", "npt_c_fe.mdp", "npt_init_fe.mdp"}
for tree in TREES:
    d = os.path.join(ROOT, "settings", tree)
    f_c, f_n = os.path.join(d, "npt_c_fe.mdp"), os.path.join(d, "npt_fe.mdp")
    have = os.path.isfile(f_c)
    check("[%s] npt_c_fe.mdp exists" % tree, have)
    if have and os.path.isfile(f_n):
        sc, sn = settings(f_c), settings(f_n)
        pull = [kv for kv in sc if kv[0].lower().startswith("pull")]
        rest = [kv for kv in sc if not kv[0].lower().startswith("pull")]
        check("  carries exactly the pull print header the legs carry", pull == HEADER, pull)
        check("  and no pull groups or coordinates",
              not [k for k, _ in sc if re.match(r"pull-(ngroups|ncoords|group|coord)", k)])
        check("  and no free-energy block",
              not [k for k, _ in sc if k.lower().replace("_", "-") in FE_KEYS])
        check("  every other line is npt_fe.mdp's physics, in order", rest == sn,
              sorted(set(rest) ^ set(sn))[:4])
    else:
        for what in ("carries exactly the pull print header the legs carry",
                     "and no pull groups or coordinates", "and no free-energy block",
                     "every other line is npt_fe.mdp's physics, in order"):
            check("  " + what, False, "no npt_c_fe.mdp")
    check("  holdrev0_fe.mdp is gone", not os.path.exists(os.path.join(d, "holdrev0_fe.mdp")))
    fe = {f for f in os.listdir(d) if re.search(r"(_fe\.mdp|^bound(fwd|rev)\d+\.mdp)$", f)}
    check("  the FE mdps are exactly the protocol's 42 plus the three NPTs",
          fe == WANT_FE_MDPS, sorted(fe ^ WANT_FE_MDPS))
    # Holds and nptrev get no -dhdl, yet the runner keeps <hold>_<c>.xvg: mdrun
    # writes it because every hold runs free-energy = yes.
    holds = [t[1] for t in TABLE if not t[0].startswith(("bind", "bound"))]
    no_fe = [m for m in holds if os.path.isfile(os.path.join(d, m))
             and ("free-energy", "yes") not in settings(os.path.join(d, m))]
    check("  every hold and nptrev runs free-energy = yes (20 files)",
          len(holds) == 20 and not no_fe, no_fe[:4])

# The generated headers say what each file is (review CYC-9; the critic's list).
def header(tree, name):
    p = os.path.join(ROOT, "settings", tree, name)
    if not os.path.isfile(p):
        return ""
    out = []
    for ln in open(p):
        if not ln.startswith(";"):
            break
        out.append(ln[1:].strip())
    return " ".join(out)

for tree in TREES:
    hb = header(tree, "holdbfwd1_fe.mdp") + " " + header(tree, "holdbrev1_fe.mdp")
    check("[%s] the bound holds sum to the bound switch, not to the ramp" % tree,
          hb.count("the work of the whole bound switch") == 2 and "whole ramp" not in hb,
          hb[:160])
    hc = header(tree, "npt_c_fe.mdp")
    check("  npt_c_fe.mdp: the network only if the structure has one, else pull = no",
          "elastic network on if the structure has one" in hc and "pull = no" in hc, hc[:160])
    hi = header(tree, "npt_init_fe.mdp")
    check("  npt_init_fe.mdp: one of 5 probe replicas, the first 10 ns discarded",
          "N_PROBES = 5" in hi and "first 10 ns" in hi and "1 ns" not in hi
          and "Longer than" not in hi, hi[:160])
    hn = header(tree, "nptrev_fe.mdp") + " " + header(tree, "holdfwd0_fe.mdp")
    check("  nptrev and holdfwd0 do not claim to sit inside a staged channel",
          "either side of it can be estimated" not in hn
          and "and so does boundrev2" in hn, hn[:160])


# ---------------------------------------------------------------------------
print("\n[4] the leg loop: boundrev2 grompps from holdfwd0, everything else in turn")
LOOP = block(r"^    LEG_FAILED=0$", r"^    done$")
check("the leg loop, run_leg_mdrun and require_gro are extractable",
      None not in (LOOP, RUN_LEG_MDRUN, REQUIRE_GRO))


def drive_loop(pre_done=(), drop=()):
    """Run the verbatim loop for cycle 3 with a fake gmx. pre_done: legs whose
    .gro/.tpr/.cpt already exist (a restart); drop: files to delete before it runs."""
    with tempfile.TemporaryDirectory() as d:
        for leg in ("npt_c",) + tuple(pre_done):
            for ext in ("gro", "tpr", "cpt"):
                open(os.path.join(d, "%s_3.%s" % (leg, ext) if leg != "npt_c"
                                  else "npt_c3.%s" % ext), "w").close()
        for f in drop:
            os.remove(os.path.join(d, f))
        log = os.path.join(d, "gmx.log")
        script = ("set -u\nnt=1; GPUOPT=''\n" + RUN_LEG_MDRUN + REQUIRE_GRO + DECLARE +
                  'eval "$FEPROTO"\ncycle=3\n' + LOOP +
                  'echo "LEG_FAILED=$LEG_FAILED"\n')
        os.environ["FEPROTO"] = EXPORT
        r = run_bash(script, d, log)
        calls, order = grompp_calls(log)
        return r, calls, order


if None not in (LOOP, RUN_LEG_MDRUN, REQUIRE_GRO):
    r, calls, order = drive_loop()
    check("a fresh cycle runs to the end", "LEG_FAILED=0" in r.stdout and r.returncode == 0,
          (r.stdout[-200:], r.stderr[-300:]))
    check("every leg is grompp'd once, in the table's order",
          order == ["%s_3" % n for n in NAMES], order[36:])
    b = calls.get("boundrev2_3", {})
    check("boundrev2 gets -c holdfwd0_3.tpr -t holdfwd0_3.cpt",
          b.get("-c") == "holdfwd0_3.tpr" and b.get("-t") == "holdfwd0_3.cpt", b)
    wrong = []
    for n in NAMES:
        src = "npt_c3" if SOURCE[n] == "npt_c" else SOURCE[n] + "_3"
        c = calls.get(n + "_3", {})
        if c.get("-c") != src + ".tpr" or c.get("-t") != src + ".cpt":
            wrong.append("%s<-%s" % (n, c.get("-c")))
    check("every other leg gets the leg before it (boundfwd1 gets npt_c3)", not wrong, wrong[:5])
    check("each leg runs its own mdp",
          all(calls.get(n + "_3", {}).get("-f") == t for n, t in [x[:2] for x in TABLE]))

    # A restart after the reverse ramp: everything up to bindrevA is on disk.
    done = NAMES[:NAMES.index("boundrev2")]
    r, calls, order = drive_loop(pre_done=done)
    check("on a restart only the three reverse bound legs run",
          order == ["boundrev2_3", "holdbrev1_3", "boundrev1_3"], order)
    b = calls.get("boundrev2_3", {})
    check("  and boundrev2 still branches from holdfwd0_3, not from bindrevA_3",
          b.get("-c") == "holdfwd0_3.tpr" and b.get("-t") == "holdfwd0_3.cpt", b)
    check("  and holdbrev1 continues from boundrev2_3",
          calls.get("holdbrev1_3", {}).get("-c") == "boundrev2_3.tpr")

    # The branch source has lost its checkpoint: the cycle stops, nothing is guessed.
    r, calls, order = drive_loop(pre_done=done, drop=("holdfwd0_3.cpt",))
    check("a missing holdfwd0_3.cpt stops the cycle before any grompp",
          "LEG_FAILED=1" in r.stdout and order == [] and "holdfwd0_3" in r.stdout,
          (order, r.stdout[-300:]))
else:
    for what in ("a fresh cycle runs to the end", "every leg is grompp'd once, in order",
                 "boundrev2 gets -c holdfwd0_3.tpr -t holdfwd0_3.cpt",
                 "every other leg gets the leg before it", "each leg runs its own mdp",
                 "on a restart only the three reverse bound legs run",
                 "  and boundrev2 still branches from holdfwd0_3",
                 "  and holdbrev1 continues from boundrev2_3",
                 "a missing holdfwd0_3.cpt stops the cycle"):
        check(what, False, "loop not extractable")

# The rebinding RMSD and the interface QC compare npt_c with the end of the reverse
# ramp. That is still bindrevA: the branch moved the reverse BOUND legs, not the
# ramp, and the QC must not drift onto a bound leg.
LAST = block(r'^      LAST_STAGE=""$', r"^      done$")
if LAST:
    with tempfile.TemporaryDirectory() as d:
        os.environ["FEPROTO"] = EXPORT
        r = run_bash(DECLARE + 'eval "$FEPROTO"\n' + LAST + 'echo "LAST=$LAST_STAGE"\n', d,
                     os.path.join(d, "g"))
    check("the rebinding QC still compares against the end of bindrevA",
          "LAST=bindrevA" in r.stdout, r.stdout[-100:])
else:
    check("the rebinding QC still compares against the end of bindrevA", False,
          "no LAST_STAGE loop")

# Every xvg a cycle writes, by name, so a restart can gunzip them and the end of the
# cycle gzip them. Holds and nptrev have no -dhdl, and mdrun -deffnm writes their
# dH/dlambda to <hold>_<c>.xvg (the integrator's 2KTF run: holdfwd0_1.xvg,
# holdbfwd1_1.xvg, nptrev_1.xvg); the pilot measures hold lengths from them, so they
# are kept too. Expected list from the table above, not from the protocol module.
XVGS = block(r"^fe_xvgs\(\) \{$", r"^\}$")
want_x = []
for name, _, _, _ in TABLE:
    if name.startswith("bind"):
        want_x += ["%s_3_pullf" % name, "%s_3_dhdl" % name]
    elif name.startswith("bound"):
        want_x += ["%s_3_dhdl" % name]
    else:
        want_x += ["%s_3" % name]
if XVGS:
    with tempfile.TemporaryDirectory() as d:
        os.environ["FEPROTO"] = EXPORT
        r = run_bash(DECLARE + 'eval "$FEPROTO"\ncycle=3\n' + XVGS + "fe_xvgs\n", d,
                     os.path.join(d, "g"))
    got_x = r.stdout.split()
    check("fe_xvgs names all 60 xvgs of a cycle, in leg order", got_x == want_x,
          sorted(set(got_x) ^ set(want_x))[:6])
    check("  the 20 holds' <hold>_<c>.xvg among them (holdfwd0_3, nptrev_3, holdbfwd1_3)",
          {"holdfwd0_3", "nptrev_3", "holdbfwd1_3", "holdbrev1_3"} <= set(got_x)
          and len([x for x in got_x if not x.endswith(("_pullf", "_dhdl"))]) == 20,
          [x for x in got_x if not x.endswith(("_pullf", "_dhdl"))][:5])
else:
    check("fe_xvgs names all 60 xvgs of a cycle, in leg order", False, "no fe_xvgs")
    check("  the 20 holds' <hold>_<c>.xvg among them", False, "no fe_xvgs")

# ---------------------------------------------------------------------------
print("\n[5] npt_c runs ./npt_c_fe.mdp, which the setup provides and checks")
HEAD = block(r'echo "=== FE cycle \$cycle of \$NUMCYCLES ==="',
             r"^    require_gro npt_c\$\{cycle\} \|\|")
check("the cycle header through npt_c's require_gro is extractable",
      None not in (HEAD, RUN_LEG_MDRUN, REQUIRE_GRO))


ALL_DONE = ["npt_c1"] + ["%s_1" % x for x in NAMES]


def drive_nptc(with_file, done=(), proto=WANT_ID):
    """The verbatim cycle header for cycle 1. with_file: ./npt_c_fe.mdp exists;
    done: outputs (npt_c1, <leg>_1) already finished, with .gro, .tpr and .cpt;
    proto: the text of protocol_id.gs, None for no file."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "mdp"))
        open(os.path.join(d, "mdp", "npt_fe.mdp"), "w").close()   # the old target
        if with_file:
            open(os.path.join(d, "npt_c_fe.mdp"), "w").close()
        if proto is not None:
            with open(os.path.join(d, "protocol_id.gs"), "w") as f:
                f.write(proto)
        for n in done:
            for ext in ("gro", "tpr", "cpt"):
                open(os.path.join(d, "%s.%s" % (n, ext)), "w").close()
        log = os.path.join(d, "gmx.log")
        script = ("set -u\nnt=1; GPUOPT=''; MDP_DIR=%s/mdp; NUMCYCLES=1; CYCLE_FAILED=0\n"
                  "release_cycle_lock() { :; }\n" % d + RUN_LEG_MDRUN + REQUIRE_GRO +
                  DECLARE + 'eval "$FEPROTO"\ncycle=1; CYCLE_END=1\n'
                  "while [[ $cycle -le $CYCLE_END ]]; do\n" + HEAD +
                  'echo "REACHED_LEGS"\ncycle=$((cycle+1))\ndone\n'
                  'echo "CYCLE_FAILED=$CYCLE_FAILED"\n')
        os.environ["FEPROTO"] = EXPORT
        r = run_bash(script, d, log)
        calls, order = grompp_calls(log)
        nlog = len(open(log).readlines()) if os.path.isfile(log) else 0
        return r, calls, nlog


def stopped(r, n):
    return "REACHED_LEGS" not in r.stdout and "CYCLE_FAILED=1" in r.stdout and n == 0


if None not in (HEAD, RUN_LEG_MDRUN, REQUIRE_GRO):
    r, calls, _ = drive_nptc(True)
    check("npt_c is grompp'd with -f ./npt_c_fe.mdp",
          calls.get("npt_c1", {}).get("-f") == "./npt_c_fe.mdp", calls.get("npt_c1"))
    check("  and the cycle goes on to the legs", "REACHED_LEGS" in r.stdout
          and "CYCLE_FAILED=0" in r.stdout, r.stdout[-200:])
    r, calls, n = drive_nptc(False)
    check("without npt_c_fe.mdp the cycle stops before any gmx call", stopped(r, n),
          (n, r.stdout[-200:]))
    check("  saying which file is missing", "npt_c_fe.mdp" in r.stdout, r.stdout[-300:])
    check("  and never falls back to npt_fe.mdp",
          not any(c.get("-f", "").endswith("/npt_fe.mdp") for c in calls.values()))
    r, calls, n = drive_nptc(False, done=ALL_DONE)
    check("a cycle whose legs are all done still reaches integration without it",
          "REACHED_LEGS" in r.stdout and n == 0, r.stdout[-200:])
    # Review CYC-5: the stop covers every leg. A guard on npt_c alone (or without
    # the per-leg loop) passed every check above.
    r, calls, n = drive_nptc(False, done=["npt_c1"])
    check("npt_c finished under an old setup, legs not: still stops, no gmx call",
          stopped(r, n), (n, r.stdout[-200:]))
    r, calls, n = drive_nptc(False, done=[x for x in ALL_DONE if x != "boundrev1_1"])
    check("every leg but the last finished: still stops, no gmx call", stopped(r, n),
          (n, r.stdout[-200:]))
    r, calls, n = drive_nptc(False, done=[x for x in ALL_DONE if x != "npt_c1"])
    check("every leg finished but npt_c's .gro gone: still stops", stopped(r, n),
          (n, r.stdout[-200:]))

    # The protocol a setup was built for (protocol_id.gs) must be the one this code
    # runs, with the same every-leg reach (review CYC-1: v1.4 cycles went on from
    # their checkpoints into v1.5 legs and wrote rows indistinguishable from v1.5).
    r, calls, n = drive_nptc(True, proto=None)
    check("no protocol_id.gs (a pre-v1.5 setup): stops before any gmx call",
          stopped(r, n), (n, r.stdout[-200:]))
    check("  saying so, and sending it to a fresh directory",
          "protocol_id.gs" in r.stdout and "fresh directory" in r.stdout, r.stdout[-300:])
    r, calls, n = drive_nptc(True, proto="fe-1.5-00000000\n")
    check("protocol_id.gs of another protocol: stops before any gmx call",
          stopped(r, n), (n, r.stdout[-200:]))
    check("  naming both protocols", "fe-1.5-00000000" in r.stdout and WANT_ID in r.stdout,
          r.stdout[-300:])
    r, calls, n = drive_nptc(True, proto=None, done=["npt_c1"])
    check("  no protocol_id.gs, npt_c finished, legs not: stops", stopped(r, n),
          (n, r.stdout[-200:]))
    r, calls, n = drive_nptc(True, proto=None,
                             done=[x for x in ALL_DONE if x != "boundrev1_1"])
    check("  no protocol_id.gs, all but the last leg finished: stops", stopped(r, n),
          (n, r.stdout[-200:]))
    r, calls, n = drive_nptc(True, proto=None, done=ALL_DONE)
    check("  no protocol_id.gs, every leg finished: integration only, no gmx call",
          "REACHED_LEGS" in r.stdout and "CYCLE_FAILED=0" in r.stdout and n == 0,
          r.stdout[-200:])
    r, calls, n = drive_nptc(True, proto="  %s  \n\n" % WANT_ID)
    check("  the right id with surrounding blanks runs", "REACHED_LEGS" in r.stdout
          and calls.get("npt_c1", {}).get("-f") == "./npt_c_fe.mdp", r.stdout[-200:])
else:
    for what in ("npt_c is grompp'd with -f ./npt_c_fe.mdp",
                 "  and the cycle goes on to the legs",
                 "without npt_c_fe.mdp the cycle stops before any gmx call",
                 "  saying which file is missing", "  and never falls back to npt_fe.mdp",
                 "a cycle whose legs are all done still reaches integration without it",
                 "npt_c finished under an old setup, legs not",
                 "every leg but the last finished", "every leg finished but npt_c",
                 "no protocol_id.gs", "  saying so", "protocol_id.gs of another protocol",
                 "  naming both protocols", "  no protocol_id.gs, npt_c finished",
                 "  no protocol_id.gs, all but the last", "  no protocol_id.gs, every leg",
                 "  the right id with surrounding blanks runs"):
        check(what, False, "block not extractable")
check("no grompp in job_fe.run reads $MDP_DIR/npt_fe.mdp any more",
      not line_no(r"gmx grompp.*npt_fe\.mdp"), line_no(r"gmx grompp.*npt_fe\.mdp"))

COPY = [ln for ln in SRC.splitlines()
        if re.match(r'^  for m in "\$\{FE_LEG_MDPS\[@\]\}".*cp "\$MDP_DIR/\$m"', ln)]
check("the setup copies the leg mdps in one line", len(COPY) == 1, COPY)
if len(COPY) == 1:
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "mdp")); os.makedirs(os.path.join(d, "s"))
        for f in ("a_fe.mdp", "npt_c_fe.mdp"):
            open(os.path.join(d, "mdp", f), "w").close()
        r = run_bash("MDP_DIR=%s/mdp; FE_LEG_MDPS=(a_fe.mdp)\n%s\n" % (d, COPY[0]),
                     os.path.join(d, "s"), os.path.join(d, "g"))
        copied = sorted(os.listdir(os.path.join(d, "s")))
    check("  and copies npt_c_fe.mdp with them", "npt_c_fe.mdp" in copied, copied)
else:
    check("  and copies npt_c_fe.mdp with them", False, "no copy line")

POST = block(r"^  if ! grep -q '\^pull-ngroups' npt_c_fe\.mdp", r"^  fi$")
check("the setup post-condition checks npt_c_fe.mdp", POST is not None)
if POST:
    def post(text):
        with tempfile.TemporaryDirectory() as d:
            if text is not None:
                with open(os.path.join(d, "npt_c_fe.mdp"), "w") as f:
                    f.write(text)
            r = run_bash('SETUP_MISSING=""\n' + POST + 'echo "MISSING=[$SETUP_MISSING]"\n',
                         d, os.path.join(d, "g"))
            return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    tmpl = open(os.path.join(ROOT, "settings", "amber19sb_opc3", "npt_c_fe.mdp")).read() \
        if os.path.isfile(os.path.join(ROOT, "settings", "amber19sb_opc3", "npt_c_fe.mdp")) else ""
    en = tmpl + "\npull-ngroups            = 2\npull-ncoords            = 1\n"
    off = re.sub(r"(?m)^pull(\s+)= yes", r"pull\1= no", tmpl)
    check("  an appended elastic network passes", post(en) == "MISSING=[]", post(en))
    check("  'pull = no' (no network pairs) passes", "pull" in off and "= no" in off
          and post(off) == "MISSING=[]", post(off))
    check("  the untouched template fails the setup",
          "npt_c_fe.mdp" in post(tmpl), post(tmpl))
    check("  a missing file fails the setup", "npt_c_fe.mdp" in post(None), post(None))
else:
    for what in ("an appended elastic network passes", "'pull = no' (no network pairs) passes",
                 "the untouched template fails the setup", "a missing file fails the setup"):
        check("  " + what, False, "no post-condition")

# ---------------------------------------------------------------------------
print("\n[6] the box padding is clamped into [FLOOR, 4.0], never discarded")
BOX = block(r"^      if \[\[ ! -f box\.gro \]\]; then$", r"^      fi$")
check("the box block is extractable", BOX is not None)
check("boxpad_used.gs is written exactly once",
      SRC.count('echo "$BOXPAD" > boxpad_used.gs') == 1,
      SRC.count('echo "$BOXPAD" > boxpad_used.gs'))


def drive_box(content, raw=False):
    """Run the box block with boxpad.gs holding `content` and a newline, or exactly
    `content` if raw (None: no file).
    -> (-d handed to editconf or None, boxpad_used.gs, stdout, results_0.gs, boxpad.gs left)"""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "mdp")); s = os.path.join(d, "S1"); os.makedirs(s)
        with open(os.path.join(d, "mdp", "leg_fe.mdp"), "w") as f:
            f.write("rvdw                     = 1.0\n")     # FLOOR = 1.0/2 + 1.0/2 = 1.00
        if content is not None:
            with open(os.path.join(s, "boxpad.gs"), "w", newline="") as f:
                f.write(content if raw else content + "\n")
        log = os.path.join(d, "gmx.log")
        r = run_bash("set -u\nSTRUCT_ID=S1; MDP_DIR=%s/mdp; FE_LEG_MDPS=(leg_fe.mdp); "
                     "PULL_DIST=1.0\n%s\necho END_OF_BLOCK\n" % (d, BOX), s, log)
        dval = None
        if os.path.isfile(log):
            for ln in open(log):
                tok = ln.split()
                if tok and tok[0] == "editconf" and "-d" in tok:
                    dval = float(tok[tok.index("-d") + 1])
        used = open(os.path.join(s, "boxpad_used.gs")).read().strip() \
            if os.path.isfile(os.path.join(s, "boxpad_used.gs")) else None
        res0 = open(os.path.join(d, "results_0.gs")).read() \
            if os.path.isfile(os.path.join(d, "results_0.gs")) else ""
        return dval, used, r.stdout, res0, os.path.isfile(os.path.join(s, "boxpad.gs"))


if BOX:
    # (boxpad.gs, -d expected, clamped?)
    for val, want, clamped in (("3.99", 3.99, False), ("4.00", 4.00, False),
                               ("4.01", 4.0, True), ("7.737", 4.0, True),
                               ("1.01", 1.01, False), ("1.00", 1.00, False),
                               ("0.99", 1.00, True), ("0.30", 1.00, True)):
        dval, used, out, _, _ = drive_box(val)
        check("boxpad.gs %-6s -> -d %.2f%s" % (val, want, ", clamped and said so" if clamped
                                               else ", unchanged"),
              dval is not None and abs(dval - want) < 1e-9
              and used is not None and abs(float(used) - want) < 1e-9
              and (("clamp" in out.lower()) == clamped),
              (dval, used, out.strip()[-160:]))
    dval, used, out, _, _ = drive_box(None)
    check("no boxpad.gs -> the 1.8 fallback, marked NOT measured",
          dval is not None and abs(dval - 1.8) < 1e-9 and "NOT measured" in out,
          (dval, out.strip()[-160:]))
    dval, used, out, res0, left = drive_box("nan")
    check("a boxpad.gs holding no number -> PROBE_FAIL, no box, file removed",
          dval is None and "S1 PROBE_FAIL" in res0 and not left
          and "END_OF_BLOCK" not in out, (dval, res0, left, out.strip()[-160:]))
    # Review CYC-10: a 0-byte boxpad.gs is a failed measurement, not an absent one
    # (it took the 1.8 fallback, logged as "no boxpad.gs"), and so is a blank first
    # line; a value written with DOS line ends is that value (it was a PROBE_FAIL,
    # and the file was deleted).
    for what, content, raw in (("an empty (0-byte) boxpad.gs", "", True),
                               ("a boxpad.gs with a blank first line", "\n1.754\n", True)):
        dval, used, out, res0, left = drive_box(content, raw=raw)
        check("%s -> PROBE_FAIL, no box, no fallback" % what,
              dval is None and "S1 PROBE_FAIL" in res0 and not left
              and "NOT measured" not in out, (dval, res0, left, out.strip()[-160:]))
    dval, used, out, res0, left = drive_box("1.754\r\n", raw=True)
    check("boxpad.gs '1.754' with a CRLF line end -> -d 1.754, unchanged, kept",
          dval is not None and abs(dval - 1.754) < 1e-9 and used == "1.754"
          and "clamp" not in out.lower() and res0 == "" and left,
          (dval, used, res0, left, out.strip()[-160:]))
    dval, used, out, res0, left = drive_box("4.01\r\n", raw=True)
    check("boxpad.gs '4.01' with a CRLF line end -> clamped DOWN to 4.0",
          dval is not None and abs(dval - 4.0) < 1e-9 and "clamped DOWN" in out,
          (dval, out.strip()[-160:]))
else:
    for what in ("3.99", "4.00", "4.01", "7.737", "1.01", "1.00", "0.99", "0.30",
                 "no boxpad.gs", "garbage", "empty", "blank first line", "CRLF",
                 "CRLF 4.01"):
        check("boxpad.gs %s" % what, False, "box block not extractable")

# ---------------------------------------------------------------------------
print("\n[7] LIGAND_FAIL")
LIG = block(r"^  if \[\[ -f ligand_failed\.gs \]\]", r"^  fi$")
check("the ligand_failed.gs check exists", LIG is not None)


def drive_lig(content):
    with tempfile.TemporaryDirectory() as d:
        s = os.path.join(d, "S1"); os.makedirs(s)
        open(os.path.join(d, "results_0.gs"), "w").close()
        if content is not None:
            with open(os.path.join(s, "ligand_failed.gs"), "w") as f:
                f.write(content)
        r = run_bash("set -u\nSTRUCT_ID=S1\n%s\necho CONTINUED\n" % LIG, s,
                     os.path.join(d, "g"))
        return r, open(os.path.join(d, "results_0.gs")).read()


if LIG:
    two = ("GNP B 1167 AM1-BCC failed (sqm); not in topol.top\n"
           "SO4 A 401 built as neutral H2SO4\n")
    r, res0 = drive_lig(two)
    check("two failed ligands: exit 0 and no further step",
          r.returncode == 0 and "CONTINUED" not in r.stdout, (r.returncode, r.stdout[-200:]))
    check("  'S1 LIGAND_FAIL' is the structure's last status line",
          res0.strip().splitlines()[-1:] == ["S1 LIGAND_FAIL"], res0)
    check("  and both lines are printed to the log",
          "GNP B 1167" in r.stdout and "SO4 A 401" in r.stdout, r.stdout[-300:])
    for what, content in (("an empty file", ""), ("a blank-lines-only file", "\n  \n"),
                          ("no file", None)):
        r, res0 = drive_lig(content)
        check("%s goes on, writing no status" % what,
              "CONTINUED" in r.stdout and res0 == "", (r.stdout[-200:], res0))
else:
    for what in ("two failed ligands: exit 0 and no further step",
                 "  'S1 LIGAND_FAIL' is the structure's last status line",
                 "  and both lines are printed to the log", "an empty file goes on",
                 "a blank-lines-only file goes on", "no file goes on"):
        check(what, False, "no ligand check")

# Contract F4: merge_ligand.py is told why ligands are or are not parametrised, so
# that a force field without ligand parametrisation stops every in-region ligand as
# LIGAND_FAIL instead of running without it, while --no-ligand-param stays a choice.
MODE_BLK = block(r'^if \[\[ "\$LIGAND_PARAM" != "1" \]\]; then$', r"^fi$")
check("the ligand-parametrisation mode is defined once", MODE_BLK is not None)
if MODE_BLK:
    for lp, ffb, want in (("1", "amber19sb_opc3", "on"), ("1", "amber19sb_opc", "on"),
                          ("0", "amber19sb_opc3", "off-user"), ("1", "charmm36", "off-ff"),
                          ("1", "gromos54a8", "off-ff"), ("0", "gromos54a8", "off-user"),
                          ("0", "charmm36", "off-user")):
        with tempfile.TemporaryDirectory() as d:
            r = run_bash("LIGAND_PARAM=%s; FF_BASE=%s\n%s\necho MODE=$LIGAND_PARAM_MODE\n"
                         % (lp, ffb, MODE_BLK), d, os.path.join(d, "g"))
        check("  run.gs ligand field %s, %s -> %s" % (lp, ffb, want),
              ("MODE=%s" % want) in r.stdout, r.stdout[-80:])
else:
    check("  run.gs ligand field / force field -> mode", False, "no mode block")
mcalls = [i for i, ln in enumerate(SRC.splitlines())
          if "python3 ../../utils/merge_ligand.py" in ln]
margs = [" ".join(SRC.splitlines()[i:i + 2]) for i in mcalls]
check("both merge_ligand.py calls pass --ligand-param \"$LIGAND_PARAM_MODE\"",
      len(margs) == 2 and all('--ligand-param "$LIGAND_PARAM_MODE"' in a for a in margs),
      margs)
gate = line_no(r'^  if \[\[ "\$LIGAND_PARAM_MODE" == "on" \]\]; then$')
plig = line_no(r"python3 \.\./\.\./utils/parametrize_ligand\.py")
check("  and the parametrisation loop runs exactly when the mode is on",
      len(gate) == 1 and len(plig) == 1 and 0 < plig[0] - gate[0] < 15, (gate, plig))

lig = line_no(r"^  if \[\[ -f ligand_failed\.gs \]\]")
merge = line_no(r"python3 \.\./\.\./utils/merge_ligand\.py")
param = line_no(r"python3 \.\./\.\./utils/parametrize_ligand\.py")
evac = line_no(r"gmx grompp -f \$MDP_DIR/emin_vac\.mdp")
setup = line_no(r"^# ===== SETUP PHASE")
check("it runs after both merge_ligand.py calls and the parametrisation",
      len(lig) == 1 and len(merge) == 2 and param and lig[0] > max(merge + param),
      (lig, merge, param))
check("  and before the first grompp of the setup, inside the setup phase",
      len(lig) == 1 and evac and setup and setup[0] < lig[0] < evac[0], (setup, lig, evac))

# The condition continues on a second line since the recovery spares an interrupted
# probe pass (review S4); this case (no emin_vac.gro) is one it still cleans.
CLEAN = block(r'^if \[\[ "\$MODE" != "cycle" && -f conf\.gro && ! -f emin_solv\.gro \]\]',
              r"^fi$")
check("the restart cleanup is extractable", CLEAN is not None)
if CLEAN:
    with tempfile.TemporaryDirectory() as d:
        for f in ("conf.gro", "ligand_failed.gs", "ligand_GNP.itp", "setup.done",
                  "protocol_id.gs"):
            open(os.path.join(d, f), "w").close()
        run_bash("MODE=setup\n" + CLEAN, d, os.path.join(d, "g"))
        left = sorted(os.listdir(d))
    check("  and removes ligand_failed.gs with the other ligand files",
          "ligand_failed.gs" not in left and "ligand_GNP.itp" not in left, left)
    # A setup wiped for a fresh start no longer vouches for anything: its completion
    # marker and protocol record go with it, and the new setup writes both again.
    check("  and the completion record of the setup it wipes (setup.done, protocol_id.gs)",
          "setup.done" not in left and "protocol_id.gs" not in left, left)
else:
    check("  and removes ligand_failed.gs with the other ligand files", False, "no cleanup")
    check("  and the completion record of the setup it wipes", False, "no cleanup")

r = subprocess.run(["bash", "-n", JOB], capture_output=True, text=True)
check("bash -n job_fe.run", r.returncode == 0, r.stderr.strip()[:200])

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all v1.5 cycle checks passed")
