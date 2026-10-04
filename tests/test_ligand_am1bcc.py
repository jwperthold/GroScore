#!/usr/bin/env python3
"""AM1-BCC charges: the deposited conformer first, then seeded retries under a cap.

v1.4 ran AM1-BCC once, on the conformer the OpenFF AmberTools wrapper embeds with
RDKit seed 1. sqm failed there for 1KXP ATP and 1HE8 GNP (both dropped) and for 2BTF
ATP on one machine only; the first round retried on seeds 2-6, which made the
charges depend on which conformer happened to converge on which machine (up to
0.375 e on one 1KXP ATP oxygen) and let six FAD-sized attempts run for about 2 h
(review L11). The decision for v1.5: the deposited conformer first, then seeded
RDKit conformers, with a wall-time cap, and the itp names what was used. This
pins, with the sqm run itself replaced where a test needs a failure:

  [flow]  the deposited conformer is the first attempt and is what am1bcc_attempt
          receives; on charge failures the retries use seeds 1, 2, 3, 4, 5 in that
          order; when every attempt fails there are exactly 6 attempts and a
          LigandFailure that lists them; a failure that is not about charges is
          raised at once (1 attempt); the note names the conformer that worked and
          the attempts before it, an antechamber failure as 'antechamber/sqm exit
          status 1' instead of OpenFF's 3.8 kB wrapper message.
  [cap]   the first attempt runs without a timeout; retries share
          AM1BCC_RETRY_WALL_S = 3600 s: with every attempt taking 1000 s the
          timeouts are None, 3600, 2600, 1600, 600 and seed 5 is not tried; an
          attempt that times out ends the retries.
  [kill]  am1bcc_attempt() stops a run at its timeout together with what it started
          (a 'sleep 60' child is gone within seconds), leaves no work directory,
          and returns real AM1-BCC charges for methanol on its deposited conformer
          (6 charges summing to 0).
  [term]  SIGTERM of the parametrisation ends it with 143, the attempt's 'sleep 60'
          is gone and no work directory is left; after SIGKILL the 'sleep 60' is
          gone as well (v15_work review L-3: both left orphans running).
  [itp]   make_interchange() hands the charges of the attempt that worked to the
          topology unchanged (fixed charges in, the same charges in the exported
          [ atoms ]), and checks the valence parameters before any sqm run: AlF3
          fails with UnassignedBondError and no attempt.

In-process; one real sqm run on methanol (about a second) and three OpenFF parameter
assignments; a few seconds.

Standalone, no pytest: python3 tests/test_ligand_am1bcc.py
"""
import importlib.util
import io
import os
import contextlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL = os.path.join(ROOT, "utils", "parametrize_ligand.py")
fails = 0


def check(name, ok, detail=""):
    global fails
    print("  %-72s %s" % (name, "pass" if ok else "FAIL  %s" % detail))
    if not ok:
        fails += 1


def load():
    spec = importlib.util.spec_from_file_location("parametrize_ligand_under_test", PL)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [PL]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        return None
    finally:
        sys.argv = argv
    return mod


mod = load()
needed = ("am1bcc_charges", "am1bcc_attempt", "seeded_conformer", "AM1BCC_RETRY_WALL_S")
if mod is None or not all(hasattr(mod, n) for n in needed):
    print("parametrize_ligand.py has no deposited-first AM1-BCC (%s)" % ", ".join(n for n in needed if not hasattr(mod, n)))
    sys.exit(1)
from openff.toolkit import Molecule, ForceField  # noqa: E402
from openff.units import unit  # noqa: E402

work = tempfile.mkdtemp(prefix="test_ligand_am1bcc_")
os.chdir(work)


def deposited_molecule(smi="CCO", seed=17):
    """An OpenFF molecule whose one conformer stands in for the deposited one."""
    rd = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(rd, randomSeed=seed)
    return Molecule.from_rdkit(rd, allow_undefined_stereo=True)


SQM_FAIL = "ValueError: Cannot properly run sqm"
calls = []


def fake(script):
    """am1bcc_attempt stand-in: script[k] is None (succeed) or an exception for call k."""
    def attempt(offmol, conformer, timeout=None):
        calls.append((conformer, timeout))
        k = len(calls) - 1
        what = script[k] if k < len(script) else None
        if what is None:
            return [0.0] * offmol.n_atoms
        raise what
    return attempt


real_attempt = mod.am1bcc_attempt
off = deposited_molecule()
dep = off.conformers[0].m_as(unit.angstrom)

print("\n[flow] deposited conformer first, then seeds 1-5")
calls.clear()
mod.am1bcc_attempt = fake([None])
q, note = mod.am1bcc_charges(off)
check("success on the deposited conformer: one attempt", len(calls) == 1, str(len(calls)))
check("and it received the deposited coordinates", np.allclose(calls[0][0].m_as(unit.angstrom), dep), "")
check("note: 'AM1-BCC charges from the deposited conformer (attempt 1 of 6)'",
      note == "AM1-BCC charges from the deposited conformer (attempt 1 of 6)", note)
calls.clear()
mod.am1bcc_attempt = fake([RuntimeError(SQM_FAIL), RuntimeError(SQM_FAIL), None])
q, note = mod.am1bcc_charges(off)
check("deposited and seed 1 fail, seed 2 works: three attempts", len(calls) == 3, str(len(calls)))
seeds_ok = all(np.allclose(calls[k][0].m_as(unit.angstrom), mod.seeded_conformer(off, s).m_as(unit.angstrom))
               for k, s in ((1, 1), (2, 2)))
check("the retries are the RDKit conformers of seeds 1 and 2", seeds_ok)
check("the note names seed 2 and both failures before it",
      note.startswith("AM1-BCC charges from an RDKit conformer embedded with seed 2 (attempt 3 of 6); before it: "
                      "the deposited conformer failed") and "seed 1 failed" in note, note)
calls.clear()
mod.am1bcc_attempt = fake([RuntimeError(SQM_FAIL)] * 10)
try:
    mod.am1bcc_charges(off)
    raised = None
except mod.LigandFailure as e:
    raised = str(e)
check("every attempt fails: exactly 1 + 5 attempts, then LigandFailure", len(calls) == 6 and raised is not None,
      "%d %s" % (len(calls), raised))
order = [s for s in (1, 2, 3, 4, 5)]
check("retried in seed order 1, 2, 3, 4, 5",
      all(np.allclose(calls[k + 1][0].m_as(unit.angstrom), mod.seeded_conformer(off, s).m_as(unit.angstrom))
          for k, s in enumerate(order)))
check("the failure lists all six", raised is not None and raised.count("failed (ValueError: Cannot properly run sqm)") == 6,
      str(raised))
calls.clear()
mod.am1bcc_attempt = fake([ValueError("molecule has no conformers")] + [None] * 5)
try:
    mod.am1bcc_charges(off)
    raised = None
except Exception as e:
    raised = "%s: %s" % (type(e).__name__, e)
check("a failure that is not about charges is raised at once (one attempt)",
      len(calls) == 1 and raised is not None and "no conformers" in raised, "%d %s" % (len(calls), raised))

# OpenFF's message when antechamber fails (1H1V ATP, seed 1; 3806 characters, the
# molecule and its coordinates elided here): the note keeps the AmberTools entry only
OPENFF_FAIL = ("No registered toolkits can provide the capability \"assign_partial_charges\" for args \"()\" and kwargs "
               "\"{'molecule': Molecule with name '' and SMILES '[H][O][C@@]1([H])...', 'partial_charge_method': 'am1bcc', "
               "'use_conformers': [<Quantity([[...]], 'angstrom')>]}\"\nAvailable toolkits are: [ToolkitWrapper around "
               "The RDKit version 2025.09.5, ToolkitWrapper around AmberTools version 24.8, ToolkitWrapper around Built-in "
               "Toolkit version None]\n ToolkitWrapper around The RDKit version 2025.09.5 <class 'openff.toolkit.utils."
               "exceptions.ChargeMethodUnavailableError'> : partial_charge_method 'am1bcc' is not available from "
               "RDKitToolkitWrapper. Available charge methods are {'mmff94': {}, 'gasteiger': {}}\n ToolkitWrapper around "
               "AmberTools version 24.8 <class 'subprocess.CalledProcessError'> : Command '['antechamber', '-i', "
               "'molecule.sdf', '-fi', 'sdf', '-o', 'charged.mol2', '-fo', 'mol2', '-pf', 'yes', '-dr', 'n', '-c', 'bcc', "
               "'-nc', '-4.0']' returned non-zero exit status 1.\n ToolkitWrapper around Built-in Toolkit version None "
               "<class 'openff.toolkit.utils.exceptions.ChargeMethodUnavailableError'> : Partial charge method \"am1bcc\"\" "
               "is not supported by the Built-in toolkit.")
if hasattr(mod, "charge_failure_text"):
    brief = mod.charge_failure_text("ValueError", OPENFF_FAIL)
    check("an antechamber failure is reported as 'antechamber/sqm exit status 1'",
          brief == "assign_partial_charges via AmberTools: antechamber/sqm exit status 1", brief)
    check("and still counts as a charge failure (retried)", mod.is_charge_failure(brief))
    other = mod.charge_failure_text("ValueError", "molecule has no conformers")
    check("anything else keeps its type and text", other == "ValueError: molecule has no conformers", other)
else:
    check("charge_failure_text() exists", False)

print("\n[cap] retries share AM1BCC_RETRY_WALL_S")
check("AM1BCC_RETRY_WALL_S is 3600 s and the seeds are 1-5",
      mod.AM1BCC_RETRY_WALL_S == 3600.0 and tuple(mod.AM1BCC_RETRY_SEEDS) == (1, 2, 3, 4, 5),
      "%s %s" % (mod.AM1BCC_RETRY_WALL_S, mod.AM1BCC_RETRY_SEEDS))
clock_t = [0.0]


def clock():
    return clock_t[0]


def slow_fail(offmol, conformer, timeout=None):
    calls.append((conformer, timeout))
    clock_t[0] += 1000.0
    raise RuntimeError(SQM_FAIL)


calls.clear()
mod.am1bcc_attempt = slow_fail
try:
    mod.am1bcc_charges(off, clock=clock)
    raised = None
except mod.LigandFailure as e:
    raised = str(e)
check("1000 s per attempt: timeouts None, 3600, 2600, 1600, 600",
      [c[1] for c in calls] == [None, 3600.0, 2600.0, 1600.0, 600.0], str([c[1] for c in calls]))
check("and seed 5 is not tried (cap spent)", raised is not None and "seed 5 not tried" in raised, str(raised))


def times_out(offmol, conformer, timeout=None):
    calls.append((conformer, timeout))
    if timeout is None:
        raise RuntimeError(SQM_FAIL)
    raise TimeoutError("AM1-BCC did not finish within %.0f s" % timeout)


calls.clear()
mod.am1bcc_attempt = times_out
try:
    mod.am1bcc_charges(off)
    raised = None
except mod.LigandFailure as e:
    raised = str(e)
check("a retry that times out ends the retries (2 attempts)", len(calls) == 2 and "retry cap" in (raised or ""),
      "%d %s" % (len(calls), raised))
mod.am1bcc_attempt = real_attempt

print("\n[kill] am1bcc_attempt stops a run at its timeout")
pidfile = os.path.join(work, "sleep.pid")
real_apc = Molecule.assign_partial_charges


def hanging(self, *a, **k):
    p = subprocess.Popen(["sleep", "60"])
    open(pidfile, "w").write(str(p.pid))
    p.wait()


Molecule.assign_partial_charges = hanging
t0 = time.time()
try:
    mod.am1bcc_attempt(off, off.conformers[0], timeout=2.0)
    raised = None
except TimeoutError as e:
    raised = str(e)
finally:
    Molecule.assign_partial_charges = real_apc
took = time.time() - t0
check("TimeoutError after about 2 s", raised is not None and 1.5 < took < 15, "%s after %.1f s" % (raised, took))
gone = False
pid = int(open(pidfile).read()) if os.path.isfile(pidfile) else None
for _ in range(50):
    try:
        st = open("/proc/%d/stat" % pid).read().split()[2] if pid else "X"
        if st == "Z":
            gone = True
            break
    except (FileNotFoundError, ProcessLookupError):
        gone = True
        break
    time.sleep(0.1)
check("the 'sleep 60' it started is gone too", gone, "pid %s" % pid)
check("no am1bcc_* work directory is left", not [f for f in os.listdir(work) if f.startswith("am1bcc_")],
      str(os.listdir(work)))
meoh = deposited_molecule("CO", seed=3)
try:
    q = mod.am1bcc_attempt(meoh, meoh.conformers[0], timeout=300)
except Exception as e:
    q = "%s: %s" % (type(e).__name__, e)
check("real sqm on methanol: 6 charges summing to 0.000", isinstance(q, list) and len(q) == 6 and abs(sum(q)) < 1e-6,
      str(q))

print("\n[term] a kill of the parametrisation ends the attempt's group too")
# The attempt runs in its own process group, which outlived SIGTERM and SIGKILL of
# the parametrisation (GNU timeout and SLURM signal that process, not the group):
# python, antechamber and sqm ran on as orphans (review L-3). Driven here in a
# separate process whose attempt starts a 'sleep 60' and waits on it.
DRIVER = r'''
import importlib.util, subprocess, sys
PL, pidfile = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("pl_driver", PL)
mod = importlib.util.module_from_spec(spec)
sys.argv = [PL]
spec.loader.exec_module(mod)
from openff.toolkit import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
rd = Chem.AddHs(Chem.MolFromSmiles("CO"))
AllChem.EmbedMolecule(rd, randomSeed=3)
off = Molecule.from_rdkit(rd, allow_undefined_stereo=True)
def hanging(self, *a, **k):
    p = subprocess.Popen(["sleep", "60"])
    open(pidfile, "w").write(str(p.pid))
    p.wait()
Molecule.assign_partial_charges = hanging
mod.am1bcc_attempt(off, off.conformers[0], timeout=None)
'''


def proc_gone(pid, wait_s=10.0):
    """pid has exited (or is a zombie awaiting its reaper) within wait_s."""
    t_end = time.time() + wait_s
    while time.time() < t_end:
        try:
            if open("/proc/%d/stat" % pid).read().split()[2] == "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            return True
        time.sleep(0.1)
    return False


for signame, sig, want_rc in (("SIGTERM", signal.SIGTERM, 143), ("SIGKILL", signal.SIGKILL, -9)):
    d = tempfile.mkdtemp(prefix="term_", dir=work)
    pidf = os.path.join(d, "sleep.pid")
    open(os.path.join(d, "driver.py"), "w").write(DRIVER)
    drv = subprocess.Popen([sys.executable, os.path.join(d, "driver.py"), PL, pidf], cwd=d,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t0 = time.time()
    while not os.path.isfile(pidf) and time.time() - t0 < 120 and drv.poll() is None:
        time.sleep(0.1)
    time.sleep(0.5)
    if not os.path.isfile(pidf):
        drv.kill()
        check("%s: the driver's attempt started" % signame, False, drv.communicate()[1][-300:])
        continue
    spid = int(open(pidf).read())
    drv.send_signal(sig)
    try:
        rc = drv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        drv.kill()
        rc = "hung"
    check("%s of the parametrisation: it ends (rc %s)" % (signame, rc), rc == want_rc, rc)
    check("  and the 'sleep 60' its attempt started is gone", proc_gone(spid), "pid %d" % spid)
    if sig == signal.SIGTERM:
        check("  and no am1bcc_* work directory is left", not [f for f in os.listdir(d) if f.startswith("am1bcc_")],
              str(os.listdir(d)))
    try:
        os.kill(spid, signal.SIGKILL)
    except OSError:
        pass

print("\n[itp] the charges that worked reach the topology")
ff = ForceField("openff-2.2.1.offxml")
fixed = [0.111, -0.222, 0.333, -0.444, 0.055, 0.055, 0.056, 0.056]   # CCO: C C O H H H H H H ... per atom
fixed = (fixed + [0.0] * off.n_atoms)[:off.n_atoms]
fixed[-1] -= sum(fixed)
calls.clear()


def fixed_attempt(offmol, conformer, timeout=None):
    calls.append((conformer, timeout))
    return list(fixed)


mod.am1bcc_attempt = fixed_attempt
try:
    with contextlib.redirect_stdout(io.StringIO()):
        inter, note = mod.make_interchange(ff, off)
    inter.to_gromacs(os.path.join(work, "lig"))
    got, sec = [], None
    for line in open(os.path.join(work, "lig.top")):
        s = line.split(";")[0].strip()
        if s.startswith("["):
            sec = s.strip("[] ")
            continue
        if sec == "atoms" and s:
            got.append(float(s.split()[6]))
    check("the exported [ atoms ] carry the attempt's charges", len(got) == len(fixed)
          and max(abs(a - b) for a, b in zip(got, fixed)) < 1e-6, "%s vs %s" % (got, fixed))
    check("from one attempt on the deposited conformer", len(calls) == 1 and calls[0][1] is None, str(len(calls)))
except Exception as e:
    check("make_interchange with fixed charges", False, "%s: %s" % (type(e).__name__, e))
alf3 = Chem.RWMol()
for el in ("Al", "F", "F", "F"):
    alf3.AddAtom(Chem.Atom(el))
for k in (1, 2, 3):
    alf3.AddBond(0, k, Chem.BondType.SINGLE)
alf3 = alf3.GetMol()
Chem.SanitizeMol(alf3)
AllChem.EmbedMolecule(alf3, randomSeed=1)
calls.clear()
try:
    mod.make_interchange(ff, Molecule.from_rdkit(alf3, allow_undefined_stereo=True))
    raised = None
except Exception as e:
    raised = type(e).__name__
check("AlF3: UnassignedBondError before any AM1-BCC attempt", raised == "UnassignedBondError" and not calls,
      "%s, %d attempts" % (raised, len(calls)))
mod.am1bcc_attempt = real_attempt

os.chdir(ROOT)
shutil.rmtree(work, ignore_errors=True)
print("\n%s" % ("ALL PASS" if fails == 0 else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
