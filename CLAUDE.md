# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GroScore is a computational chemistry toolkit for protein-protein affinity scoring using short steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

**Contact:** jan@ackergarten.at

## Tech Stack

- Python 3.10 with NumPy 2.2, SciPy 1.15, OpenMM 8.5, and PDBFixer 1.12
- GROMACS 2026.0 (external MD engine)
- SLURM 23.11 job scheduler for HPC execution
- Force fields, one settings tree each: `amber19sb_opc3` (AMBER19SB with OPC3 water, default), `amber19sb_opc` (AMBER19SB with OPC water), `charmm36` (all-atom), `gromos54a8` (united-atom, PME) and `gromos54a8_rf` (GROMOS 54A8 with its reaction field)
- Water models: OPC3 or OPC (AMBER19SB), TIP3P (CHARMM36) or SPC (GROMOS 54A8)

## Running GroScore

```bash
# Run with default settings (5 cycles, AMBER19SB/OPC3)
python groscore.py

# Run with 10 simulation cycles
python groscore.py -n 10

# With custom structure parameter file (default: sp.gs)
python groscore.py -s myparams.gs

# Disable interface cutout (use full protein structure)
python groscore.py --no-cutout

# Run on a single 8-GPU workstation instead of SLURM
python groscore.py --run-local --ngpus 8
```

**Command-line options:**
- `-n, --numruns` - Number of pull/push cycles (default: 5)
- `-s, --structparams` - Structure parameter file (default: `sp.gs`)
- `-ff, --forcefield` - Force field: `amber19sb_opc3` (default), `amber19sb_opc`, `gromos54a8`, `gromos54a8_rf` (GROMOS 54A8 with its reaction field), or `charmm36`
- `--no-cutout` - Use full protein structure instead of interface cutout (slower, cutout is default)
- `--no-ligand-param` - Run without small-molecule ligands (none parametrized; each is left out and named in the job log). Needed to run a structure with a ligand in the simulated region under CHARMM36 or GROMOS, which otherwise stops it as `LIGAND_FAIL`
- `--restart` - Resubmit jobs. With a larger `-n` it adds cycles: `job.run` and `job_fe.run` take the total from `GROSCORE_NUMCYCLES`, which the orchestrators set from `-n`. `groscore_fe.py` holds back a structure whose last stage-0 status is `LIGAND_FAIL`, `BROKEN` or `ENTANGLED` (`STAGE0_FINAL`); a `LIGAND_FAIL` only when its `ligand_failed.gs` gives a verdict, not a crash or a kill of the parametrisation. Appending `<ID> SETUP_RUNNING` to `results_0.gs` sets one up again on purpose. On the array path a structure directory with every cycle's `results_fe_c<N>.done` and no `<ID>.tar.gz` gets a `job_fe.run --archive` job (`slurm_fe/archive_<ID>.run`, `--time=08:00:00`; locally `archive_<ID>`, logging to `../job_local_archive_<ID>.out`); under `--sequential` its whole legacy job runs and archives at its end. A directory whose rows are all complete but some `.done` missing (in a directory with `setup.done` and at least one `.done`) gets those cycles, which write it. None is submitted while `<ID>.tar.gz.part` or `<ID>/.archive.lock` exists (an archive under way or killed; reported), and a directory with an `<ID>.tar.gz` beside it is reported
- `--inject-job-run` - Install the current `job.run` into archived (.tar.gz) structures and submit them; an archive is submitted only with this flag
- `--rmsd-warn` - Rebinding sanity-check threshold in Å (default: 10.0); only flags, never changes a score
- `--run-local` - Run locally instead of via SLURM; requires `--ngpus N`. Optional: `--jobs-per-gpu N` (default 8, 0 = start everything), `--threads-per-job N` (default 1)

After initial run, jobs are submitted via auto-generated `array_submit.run`; with `--run-local` they are handed to a detached `utils/local_runner.py` process instead.

### Execution Backends

Both orchestrators submit to SLURM by default and run locally with `--run-local --ngpus N`.

| | SLURM (default) | `--run-local` |
|---|---|---|
| Dispatch | `sbatch array_submit.run` (job array) | detached `utils/local_runner.py` |
| Threads | `SLURM_CPUS_PER_TASK` | `GROSCORE_NT` (`--threads-per-job`, default 1) |
| GPU | scheduler allocation | `GROSCORE_GPU_ID` → `mdrun -gpu_id`, round-robin over `--ngpus` |
| Concurrency | queue / `--array-throttle` | `--ngpus × --jobs-per-gpu` worker slots |
| Monitoring | `squeue` | `local_status.gs`, `local_runner.log`, `<struct>/job_local.out` |

`job.run` / `job_fe.run` read both env vars and build `GPUOPT="-gpu_id N -pin off"`, empty under SLURM. `local_runner.py` runs a bounded worker pool (one thread per slot, slot k pinned to GPU k % ngpus), pulls jobs from a shared queue, honours "after any" dependencies (GroScore-FE: cycles wait for their structure's setup job), unpacks archived structures, and writes `local_runner.pid` so a second submission into an active run is refused.

Other variables the runners read: `GROSCORE_NUMCYCLES` (total cycles; wins over `run.gs`), `GROSCORE_LEG_RETRIES` (attempts per leg, default 2; both runners start every attempt from step 0 and never continue an unfinished run from its checkpoint), `GROSCORE_LEG_TIMEOUT` (`job.run` only: the 5 ns pull and push legs and the minimizations, default 24h; `slurm/vsc5*.sh` set 180m), `GROSCORE_SETUP_WAIT` / `GROSCORE_SETUP_POLL` (`job_fe.run --cycle` waiting for its setup), `GROSCORE_CCD_CACHE` (extra CCD cache directories, colon-separated) and `GROSCORE_GLYCAN_MAX_SUGARS` (an optional cap on the glycans `fuse_glycans.py` fuses; unset or empty means no cap; with ligand parametrisation on, any other value must be a whole number of at least 1 in digits only, without a sign, leading zero or space, and anything else stops the structure as `FAILED`; handed on as `--max-sugars`, so the log shows it). `parametrize_ncaa.py` reads `GROSCORE_GLYCAN_AM1BCC_WALL_S` from the job's environment (seconds allowed for the AM1-BCC of a fused glycan residue, all attempts together; unset means no limit, since the ligand path's 3600 s cap would send every large glycan back to the ligands).

## Architecture

### Main Components

- **groscore.py** - Main orchestrator that reads structure parameters, dispatches jobs (SLURM submission scripts, or a local GPU pool with `--run-local`), monitors job completion, and performs final statistical analysis
- **job.run** - Bash script executed per structure, as a SLURM array task or a local pool job; runs the complete MD workflow. One job per structure directory: it holds `job_run.lock` and exits 1 when another job holds it. Every mdrun runs as a background job (`run_bg`) whose pid is recorded in the lock, so the lock outlives a job killed hard while its leg runs, and INT/TERM stop the running leg before the job exits. A lock is taken over only from a job shown to be gone (same host: pid and leg; another host: a definite `squeue` answer)
- **groscore_fe.py** / **job_fe.run** - GroScore-FE, the absolute binding free energy engine, with the same inputs and directory layout. `job_fe.run --setup` runs stage 0, the probe replicas and `make_boresch.py` once per structure; `job_fe.run --cycle N` runs one cycle as an array task; `job_fe.run --archive` archives a structure whose cycles are all done (it holds `job_run.lock`, takes the total from `GROSCORE_NUMCYCLES`, else `numruns` in `../run_config.gs`, else `run.gs`, and refuses, exit 1, while a live task holds a cycle lock, while `<ID>.tar.gz` exists or another task archives it); `groscore_fe.py` scores into `scores_fe.gs`. The cycle (legs, rates, result-row layout, `PROTOCOL_ID`) is defined once in `utils/fe_protocol.py`. Archiving (`archive_structure`, also run by the task that finishes the last cycle) writes `<ID>.tar.gz.part` without `job_run.lock` and `.archive.lock`, tests it with `gzip -t`, renames it and only then removes the directory, printing `Archived <ID> to <ID>.tar.gz (N bytes, gzip test passed); directory removed.` and appending each outcome with time and host to `archive.log` in the run directory; INT or TERM says how far it got and removes the `.part` and `.archive.lock` unless the tested archive is already in place. A cycle task does not start while `.archive.lock` exists, and one that finds its row complete without its `.done` writes the `.done`. Every `job_fe.run` task prints a `Task start:` and a `Task end:` line (time, host, mode or exit status and duration, SLURM job as `<array>_<task>` or the job id, `CUDA_VISIBLE_DEVICES`, `GROSCORE_GPU_ID`, and one `GPU` line per GPU `nvidia-smi` lists, `nvidia-smi failed (exit N): ...`, or `did not answer within 30 s`, after which it is left behind)

### Utility Scripts (utils/)

| Script | Purpose |
|--------|---------|
| `fuse_glycans.py` | Fuses each glycan covalently bonded to ASN ND2, SER OG or THR OG1 with its amino acid into one residue (`NG<k>`, `SG<k>`, `TG<k>`) for the NCAA path, before `renumber_pdb.py` and only with ligand parametrisation on. It writes `fused.pdb` (a byte copy when nothing is fused and no glycan arrived as ATOM records), `glycans.json` (the chemistry) and `glycan_unfused.gs` (why a glycan stayed ligands; written only when one did). A glycosidic or site link deposited longer than the covalent rule, up to 2.5 A, is joined to the closest acceptor and logged (`joined stretched link ...`) unless it is ambiguous. `--max-sugars N` sets an optional size cap (default none, or `$GROSCORE_GLYCAN_MAX_SUGARS`) and `--exclude-file glycan_failed.gs` leaves out the glycans whose residue failed to parametrise and, being the fallback, fuses no chemistry the existing `glycans.json` lacks. Standard library only; exit 0 unless the input cannot be read |
| `renumber_pdb.py` | Assigns sequential residue numbers across chains, detects chain breaks. A new insertion code, a residue-name change or a repeated atom name starts a new residue |
| `fix_pdb.py` | Fixes missing atoms and non-standard residues using PDBFixer. Exits 1 and writes no output when two residues share chain, number and insertion code (PDBFixer would merge them). PDBFixer adds the missing atoms with a fixed seed (`ADD_ATOMS_SEED`) on the CPU platform with one thread, as `cap_termini.py` runs, so every run gives the same bytes |
| `cap_termini.py` | Adds ACE/NME caps (AMBER19SB) or ACE-only caps with `--ace-only` (CHARMM36; GROMOS 54A8 with `--rename-ace-carbon` as well). Deterministic (seeded, CPU, one thread), and only the cap atoms move: every other atom keeps its input coordinates. A residue that matches no amber14 template (an NCAA, a fused glycan residue, a CYM without hydrogens in a cutout) gets a generic template from its own atoms and bonds. The bonds OpenMM leaves out are made first; backbone N/CA/C/O take amber14 types, every other atom is an uncharged Lennard-Jones sphere of its element, and all of it is massless. The log says `made N bond(s) OpenMM left out` and `generic templates for <names>`. A minimisation that still fails warns, names the residue and keeps PDBFixer's caps (exit 0, output written). Inputs without such residues give the same bytes as before |
| `parametrize_ncaa.py` | Non-standard amino acids. AMBER19SB: OpenFF sidechain parameters with the AMBER19SB backbone, injected into a local force field copy. A residue defined in `glycans.json` (a fused glycan) is built from that definition and never looked up by name in the CCD (NG1 is itself a CCD code). It gets AM1-BCC on the deposited conformer with seeded retries, without a time limit unless `GROSCORE_GLYCAN_AM1BCC_WALL_S` sets one, and hydrogens named after protonation. One that fails is written to `glycan_failed.gs` (`RESNAME reason`), never to `ncaa_failed.gs`, and its glycan goes back to the ligands. For every NCAA on this path: Strategy 2's template is the residue's CCD entry from the cache (files.rcsb.org second); the ligand pH 7.4 rules apply (`ob_protonate`, then `ionize_acids`), so SEP, TPO and PTR are -2, ORN, DAB and DPP +1; the written rtp charges sum exactly to the formal charge (the rounding residual goes on CA); CB hydrogens are typed HC, H1, H2 or H3 by CB's N, O, S and halogen neighbours, as amber19sb types SER, THR and CYS, and HP next to an sp3 amine N, as in LYS (with the XC-CT-HP angle amber19sb lacks); the side-chain impropers are written sorted; an hdb entry of the same name is replaced; a stale `ncaa_failed.gs` or `glycan_failed.gs` is removed at start. GROMOS 54A8 (`--gmx-ff gromos54a8`): sorts the NCAAs into those the tree has native entries for and the rest, which are replaced by their parent residues. The runners rename the native ones (`ncaa_rename.gs`); a 4-character name such as DALA goes into columns 18-21 and is written back after `fix_pdb.py` and `cap_termini.py`, which keep 3 characters of a residue name |
| `fix_ion_protonation.py` | Sets the protonation of ion-coordinating CYS and HIS before pdb2gmx |
| `make_ion_restraints.py` | Topology-level coordination restraints for structural ions. Decides which partner a ligand belongs to from the merged residue number in the ligand manifest, not from line position |
| `merge_crystal_waters.py` | Merges crystal waters as SOL. Honours `#ifdef` / `#ifndef` / `#else` / `#endif` / `#define` in the water itp, which it finds through GMXLIB and `$GMXDATA/top` as grompp does |
| `fix_topol_intermolecular.py` | Moves the `[ intermolecular_interactions ]` block back to the end of the topology after `gmx solvate` / `genion` append their entries inside it |
| `retype_ions.py` | Right after every `pdb2gmx` of both runners (`retype_ions`, the same text in both): in `topol.top` and the `topol_*.itp` it includes, every `[ atoms ]` type ending in `_opc` becomes its twin for the run's water model (`Mg2+_opc3` under OPC3), since `aminoacids.rtp` hard-codes the OPC ion types. Nothing under OPC water or another force field. The force field is found as grompp finds it (topology directory, GMXLIB, `$GMXDATA/top`, the gmx on PATH); a twin it lacks stops the structure as `FAILED` with no file changed. Standard library only |
| `check_brokenloop.py` | Validates protein loop connectivity before simulation. Prints 0 (pass), 1 (broken loop) or 2 (the check could not be made; exit 1) |
| `check_entangledloops.py` | Detects topological knots that would invalidate results. Same 0, 1, 2 output |
| `make_cutout.py` | Extracts interface region. Minimum fragment is 5 residues (`MIN_FRAGMENT_LEN`); extension follows contiguous protein residues and stops at caps, ions and TER records, and gaps are filled again after extension |
| `make_disres_en.py` | Generates distance restraints and elastic network (classic engine). Measures every distance with the triclinic minimum image, and exits 1 on an empty interface or a frame split across the periodic boundary |
| `integrate.py` | Integrates force curves from pulling simulations. Prints NaN and exits 1 for a file that is not a finished leg |
| `rebound_rmsd.py` | Rebinding sanity check: backbone RMSD of the re-bound structure vs. the bound reference |
| `parametrize_ligand.py` | OpenFF parametrization of one ligand instance: bond orders from the CCD entry when its heavy atoms match, else OpenBabel with checks; pH 7.4 acid rules; AM1-BCC on the deposited conformer, then seeded retries. Records every failure in `ligand_failed.gs`. A ligand with an element the force field types no atom of (Al in openff-2.2.1) stops before any build, naming the element; a heavy-atom count that differs from the CCD entry's names the atoms not deposited (when the deposited names are distinct and one set holds the other); a missing OpenFF bond, angle or torsion term is reported by element (e.g. `F-Xe (2 terms)`), the full OpenFF text going to the job log; a reason longer than 400 characters is cut at a word, with the cut said |
| `merge_ligand.py` | Merges the parametrized ligands into topology and coordinates. `--ligand-param on` / `off-user` / `off-ff` says whether parametrization ran; rewrites `ligand_failed.gs`. A sugar of a glycan that `fuse_glycans.py` left as ligands fails in the region with its `glycan_unfused.gs` reason (`RES CHAIN NUM reason`) as `glycan not fused: <reason>`, after any other reason of its line, also without a covalent bond (a deposited link longer than the rule). The deposited protein is read from `fused.pdb`, where a fused glycan's sugars are protein |
| `covalent.py` | The covalent-contact rule (radii, slack, clash limits) that `merge_ligand.py` and `fuse_glycans.py` share. Standard library only |
| `ccd_cache.py` | wwPDB Chemical Component Dictionary entries: read from `forcefield/ccd_cache` (then `$GROSCORE_CCD_CACHE`), else files.rcsb.org. Run on a login node with PDB files as arguments to prefetch. The prefetch also fetches ASN, SER and THR (the parent entries of the glycan fusion) whenever an input has a HETATM residue other than water and ions |
| `assemble_frame.py` | Assembles `Protein_Struct` of a frame by contact (whole residues moved by lattice vectors only) for `npt_init_cluster.gro` / `npt_probe_cluster.gro`; exits 1 on any failure |
| `pbc_tools.py` | Exact triclinic minimum image and the residue spanning tree (`assemble`). Imported by `assemble_frame`, `measure_box`, `interface_qc`, `rebound_rmsd`, `make_disres_en`, `make_boresch` and the two loop checks |
| `make_cluster_group.py` | Adds the `Protein_Struct` index group (protein, structural ions, ligands), the group `assemble_frame.py` assembles |
| `local_runner.py` | `--run-local` backend: bounded job pool with one GPU pinned per worker slot (also importable: `launch_local`, `print_local_status`) |
| `estimators.py` | BAR and the overlap statistic that gates it. **Imported as a module** by both orchestrators, unlike every other entry here, which are run as subprocesses. Numpy/scipy only, no side effects, so `tests/test_bar.py` can import it directly |
| `fe_protocol.py` | GroScore-FE: the cycle defined once (legs, rates, result-row layout, `PROTOCOL_ID`). `--shell` exports it as shell arrays for `job_fe.run`; the exported text exits 1 when evaluated without `FE_RUNNER_HONOURS_START_FROM=1` |
| `make_fe_mdps.py` | GroScore-FE: generates every settings tree's leg mdps (and `npt_c_fe.mdp`) from `fe_protocol.py` |
| `make_boresch.py` | GroScore-FE: Boresch, interface and elastic-network restraints for the legs, from the assembled probe frame. Stops with a code in `boresch_failed.gs` (e.g. `INPUT_FRAME_SPLIT`) |
| `measure_box.py` | GroScore-FE: production box padding from the probe replicas (99th percentile of the reassembled polymer diameter) |
| `integrate_dhdl.py` | GroScore-FE: dH/dlambda switching work. Prints NaN and exits 1, naming the line on stderr, unless the record is whole: every data line a row of finite numbers, all rows one width, the last one ending in a line end, time rising strictly by one record step throughout, and with `--leg-ps` starting at 0 and ending at `--last-ps` (floor(nsteps/nstdhdl)·nstdhdl·dt, which `job_fe.run`'s `lam_span` passes; without it, within one step of the leg's end, never after). Removes its old `<base>_Wdhdl.dat` first. `job_fe.run`'s `integrate_or_nan` copies a failing integrator's stderr into the job log |
| `interface_qc.py` | GroScore-FE: interface contact recovery after rebinding (`recovered`, `formed`, `rms_dev`, `sc_rmsd`). The sugar atoms of a fused glycan are non-backbone atoms of their residue, so they count in `sc_rmsd`; it is a QC column only |
| `fe_leg_efficiency.py` | GroScore-FE diagnostic: where a leg's hysteresis comes from, and what a longer leg or more cycles would buy |
| `fe_node_report.py` | Slow-node report from finished GROMACS logs (the `--exclude` list of the VSC-5 templates) |

### Simulation Pipeline

1. **Stage 0**: Structure validation, PDB conversion, solvation, energy minimization (creates `emin_solv.gro`)
2. **Initial Equilibration**: One full equilibration to generate distance restraints. `trjconv -pbc whole` and then `assemble_frame.py` (not `-pbc cluster`) make `npt_init_cluster.gro`, the frame `make_disres_en.py` reads
3. **Independent Cycles**: N cycles, each with:
   - Fresh full equilibration (NVT 1-5 + NPT) from `emin_solv.gro` with new random velocities
   - Pull simulation (unbinding)
   - Short NPT re-equilibration + Push simulation (binding)
   - Rebinding sanity check (`rebound_rmsd.py`), stored as the last field of the cycle's results row

   In `job.run` each cycle is integrated once and marked done. A re-walk never re-integrates a finished cycle, and once a pull leg exists the restraint set (`numpertres.gs`, the pull blocks, `index.ndx`) is kept as it is.
4. **Final**: Statistical analysis producing two ranking methods, in two files:
   - `scores_avg.gs` - Simple average of pulls/pushes
   - `scores_cgi.gs` - Crooks Gaussian Intersection

   Both files additionally carry appended `BAR`, `BAR_CI95`, `BAR_note` and
   `Score_note` columns, 11 in all. `Score_note` gives `CI95_N_LT_2`,
   `CGI_N_LT_20`, `CGI_NO_OVERLAP` or `CGI_DEGENERATE` when Score or CI95 is nan.
   BAR is a comparison column in the classic engine, NOT its score, and
   reads `nan BAR_NO_OVERLAP` on essentially every structure because the forward
   and reverse works do not overlap. In `groscore_fe.py` the staged BAR sum
   (`dGbind_bar`) is the headline. It assumes the holds between stages equilibrate
   the interface, and `fe_protocol.py` (HOLD LENGTHS, WHAT THE TAU ARGUMENT DOES NOT
   COVER) records that they do not: they settle dH/dlambda but close a median -6%
   to 49% of the forward/reverse interface gap at the internal boundaries.
   `dG_unbind_1s_bar` is the whole-ramp cross-check. Which estimator the pilot
   verdict reads is still open. In `scores_avg.gs` `CI95` is the half-width of the
   paired Student-t interval of the per-cycle scores (pull + push)/2; in
   `scores_cgi.gs` it is a bootstrap over whole cycles scaled to the same t quantile.

This architecture ensures statistically independent samples by starting each pull/push cycle from a fresh equilibration.

### Results Layout

Both engines write into the run directory (the one that holds `sp.gs`), next to one directory per structure. `results_0.gs` holds the stage-0 status lines `ID STATUS`, append-only, and only the last line per structure counts.

Classic (`job.run`, `groscore.py`):
- `results.d/<ID>_c<N>.gs` - one row per cycle, `ID N pull push RMSD`, written with `>`. `NaN` where a value is missing, never an empty field. A row with both works finite is also kept as `results_c<N>.done` in the structure directory; its presence marks the cycle done, and it is never integrated again
- `results.d/<ID>_setup.gs` - `ID numpertres <count>`, written once the restraint set exists; it puts the structure under this layout, so older `results_<n>.gs` rows of it are never read
- `results_<n>.gs` - legacy rows (`ID work [RMSD]`, one work per line, append-only) of runs before v1.5; `groscore.py` still scores such run directories
- `scores_avg.gs`, `scores_cgi.gs` and the per-threshold `scores_avg_c<N>.gs`, `scores_cgi_c<N>.gs` (scores from the first N cycles)
- in each structure directory: `ligand_failed.gs`, `ligand_manifest.gs`, `job_run.lock` and, under AMBER19SB with ligand parametrisation, `fused.pdb`, `glycans.json`, `glycan_unfused.gs` and `glycan_failed.gs` (the last two only when a glycan stayed ligands or a fused residue failed)

FE (`job_fe.run`, `groscore_fe.py`):
- `results_fe.d/<id>_c<N>.gs` - one row of 43 fields per cycle: id, cycle, the 40 works in the order of `fe_protocol.works()`, RMSD
- `results_qc.d/<id>_c<N>.gs` - `ID N recovered .. formed .. rms_dev .. sc_rmsd .. npairs ..` from `interface_qc.py`
- `results_analytical.d/<id>.gs` - `ID dG_release_kJ_mol`
- `run_config.gs` - settings the directory remembers (`numruns`, `sum_k`, `sd_max`, `forcefield`, `temp`, `cutout`, `ligand_param`, `protocol_id`)
- `scores_fe.gs`
- `<ID>.tar.gz` - a structure whose cycles are all done, archived in place of its directory, without `job_run.lock` and `.archive.lock` (older archives carry `.archive.lock`; every unpack removes it)
- `archive.log` - one line per archive outcome (time, host, the `Archived ...` or failure line)
- in each structure directory: `setup.done`, `protocol_id.gs` (written when a setup completes; a cycle with legs left to run refuses to run when it is missing or differs from `fe_protocol.PROTOCOL_ID`), `results_fe_c<N>.done` and `results_fe_c<N>.lock`, `boresch_failed.gs`, `ligand_failed.gs`, `job_run.lock` and, under AMBER19SB with ligand parametrisation, `fused.pdb`, `glycans.json`, `glycan_unfused.gs` and `glycan_failed.gs` (the last two only when a glycan stayed ligands or a fused residue failed)

### Ligands

`parametrize_ligand.py` builds each ligand instance and `merge_ligand.py` decides what reaches `topol.top`. `parametrize_ligand.py` writes a line `RESNAME CHAIN RESNUM reason` for every instance it cannot build or builds as another molecule, and `merge_ligand.py` rewrites `ligand_failed.gs` to hold exactly the in-region instances that are missing from the topology, built as another molecule or covalently bonded. A ligand more than 0.5 nm (0.61 nm on heavy atoms when it has no parameters) from every atom of the simulated protein is outside the region and is only named in the log. Both runners stop the structure as `LIGAND_FAIL` when the file is non-empty, and also when `merge_ligand.py` crashes. `--ligand-param` tells `merge_ligand.py` what happened (`on`, `off-user` for `--no-ligand-param`, `off-ff` for CHARMM36 and GROMOS 54A8, where every in-region ligand gets the line `not parametrised: force field <ff> has no ligand parametrisation; pass --no-ligand-param to run without ligands`); without the option it uses `on` only when a manifest or a line from `parametrize_ligand.py` shows that parametrization ran, else `off-ff`. By decision AlF3 (openff-2.2.1 has no aluminium parameters of any kind; `parametrize_ligand.py` names the element before any build) and the REA of 1RLB (deposited without the O2 of its CCD entry, most likely retinol filed under the code of retinoic acid; built as another molecule, and the reason names the atom not deposited) stay `LIGAND_FAIL`; reconfirmed 2026-10-25, against rebuilding GDP + AlF3 as GTP or relabelling the REA as retinol. Crystallisation additives that are part of the deposited structure (SO4, PO4, EDO, PEG fragments such as PG4 and PGE, ...) are ligands like any other: parametrised, and simulated when they are in the region (decision of 2026-10-25).

A glycan covalently bonded to ASN ND2, SER OG or THR OG1 is fused with its amino acid into one residue when ligands are parametrised (AMBER19SB with OpenFF; `utils/fuse_glycans.py`, decision of 2026-10-04): `NG<k>` on Asn, `SG<k>` on Ser, `TG<k>` on Thr, defined in `glycans.json` in the structure directory and parametrised on the NCAA path, so 1MAH (NG1 = ASN A 350 + NAG A 544) and 2AJF (four NG1) run. There is one name per distinct chemistry (k is 1 to 9, then A to Z, in order of first appearance; names the input uses or the force fields and tools take, such as SGB and NGL, are skipped), and no step may look these names up in the CCD, where nearly every 3-character code is an entry. The amino acid keeps its atom names and a sugar atom is its CCD name plus a tag letter (A for the sugar on the amino acid, then breadth first), so the link of an N-glycan is ND2-C1A. The job log has one line per glycan (`fused NG1: ASN A 350 + NAG A 544 (ND2-C1 1.45 A)` or `NOT fused ...: <reason>`). The sugar atoms of a fused glycan are protein atoms in both engines and can carry interface springs, so they enter the score like any interface atom. A glycan that is not fused stays ligands; in the simulated region each of its sugars gets a `LIGAND_FAIL` line ending in `glycan not fused: <reason>` from `glycan_unfused.gs`. That happens, for example, when it has more sugars than a cap set with `GROSCORE_GLYCAN_MAX_SUGARS` (none by default), when a member is no saccharide, when its deposited atoms differ from the CCD entry or their names do not fit its bonds, when a further sugar is linked to it by a deposited bond longer than the covalent rule that could not be joined (ambiguous: its leaving O deposited as an atom of its own, two acceptors about equally close, or one acceptor closest to two sugars; that sugar gets the line too), or when its fused residue failed to parametrise (the setup chain then runs again without that fusion); the header of `fuse_glycans.py` lists every condition. A link deposited longer than the covalent rule, up to 2.5 A, is joined to the closest acceptor (1FZC NAG K 2 C1 to NAG K 1 O4 at 1.85 A, 2FD6 NAG C 1 C1 to ASN U 172 ND2 at 1.99 A) and gets the normal bond parameters; minimisation pulls it to bond length. A deposited anomeric O1 makes no link and is dropped, a sugar on the OD1 of an Asn whose ND2 is free sits on a flipped amide (OD1 and ND2 are swapped and the glycan fused), and a glycan written as ATOM records is a candidate as a HETATM one is (left unfused, its records become HETATM). Under CHARMM36, GROMOS 54A8 and `--no-ligand-param` glycans are not fused: they are ligands like any other there (`LIGAND_FAIL` under the first two, left out and named under `--no-ligand-param`). A deposited anomer or ring centre that contradicts the sugar's CCD code is logged, not refused.

Both runners run `fuse_glycans.py` in `fuse_and_renumber` (the same text in `job.run` and `job_fe.run`) before `renumber_pdb.py`, and only when `LIGAND_PARAM_MODE` is `on`; `renumber_pdb.py` then reads `fused.pdb` (`input.pdb` otherwise). A non-zero exit of `fuse_glycans.py`, which its contract allows only when the input cannot be read, stops the structure as `LIGAND_FAIL`. `ligand_*.pdb` and `ligand_info.gs` are deleted immediately before every `renumber_pdb.py` call, since it never removes them and the set of extracted ligands changes between the passes. After `parametrize_ncaa.py`, `ncaa_fallback` handles `ncaa_failed.gs` and `glycan_failed.gs` in one pass, in this order: `fuse_glycans.py --exclude-file glycan_failed.gs` when that file is non-empty (it then fuses no chemistry the first pass did not), else nothing, as the first pass's `fused.pdb` is what `parametrize_ncaa.py` parametrised; `renumber_pdb.py` on `fused.pdb`, the NCAA list less the names in `ncaa_failed.gs`, `fix_pdb.py --keep-ncaa`, `cap_termini.py`. A non-zero exit of `parametrize_ncaa.py` while fused names exist counts as a failure of every fused name, and each `glycan_failed.gs` reason is copied into the `glycan_unfused.gs` line of every excluded sugar. `ncaa_failed.gs` and `glycan_failed.gs` are deleted at the start of a fresh setup, before `fuse_glycans.py`, with the ligand parametrisation outputs (`ligand_*.itp`, `ligand_*.gro`, `ligand_*.sdf`, `ligand_manifest.gs`, `ligand_failed.gs`), and a `pdb2gmx` that writes no `conf.gro` stops the structure as `FAILED` before any ligand is parametrised. The restart cleanups remove `fused.pdb`, `glycans.json`, `glycan_unfused.gs`, `glycan_failed.gs`, `ligand_*.pdb` and `ligand_info.gs` as well.

Chemistry, in the order `build_molecule()` tries it: the CCD entry of the residue name of the records (not `-r`) when its heavy atoms match exactly, by atom name if the named atoms also fit every bond of the entry (at most 1.3 times the covalent radius sum), else by bond graph (bond orders, formal charges and hydrogens; guanine nucleotides get the canonical N1-H tautomer; only composition and acid states are checked; a cached entry in which an atom has no bond, or that builds as a radical (a file cut before a ring-closure bond row, such as a GDP.cif without its last N3-C4 row), is checked against the one at files.rcsb.org and replaced by it when that copy is complete and closed-shell; the itp header then names the damaged cache file; the ligand stops as `LIGAND_FAIL` when the entry is a radical at files.rcsb.org as well (nitric oxide), when files.rcsb.org serves it incomplete as well, or when a cached copy builds as a radical and files.rcsb.org cannot be asked (the reason names the cached file); OpenFF cannot parametrise a radical, and OpenBabel would build another molecule), else OpenBabel's perception, which must pass the composition, hydrogen, acid-state and geometry checks (an OpenBabel guanine built as the 3H tautomer fails the hydrogen check). When nothing passes, the OpenBabel molecule is built and recorded in `ligand_failed.gs` as "built, but not the deposited molecule". Then OpenBabel's pH 7.4 hydrogens and the explicit acid rules. AM1-BCC runs on the deposited conformer first, then on RDKit seeds 1 to 5 within a 3600 s retry cap, and the itp header records the conformer used. CCD entries come from `forcefield/ccd_cache`, then `$GROSCORE_CCD_CACHE`, then files.rcsb.org.

### Rebinding Sanity Check

Each cycle's work values only describe the intended binding event if the push leg actually restored the original complex. `utils/rebound_rmsd.py` measures the backbone RMSD between the bound state the cycle was equilibrated in and the state the push/rebinding leg ended in:

| Engine | Reference | Query | Written to |
|--------|-----------|-------|------------|
| `job.run` (classic) | `npt_c<N>.gro` | `bindrev_<2N>.gro` | 5th field of `results.d/<ID>_c<N>.gs` (`ID N pull push RMSD`); legacy runs: 3rd column of `results_<2N>.gs` |
| `job_fe.run` (FE) | `npt_c<N>.gro` | `bindrevA_<N>.gro` | last column of `results_fe.d/<id>_c<N>.gs` |

The Protein atoms of both frames are reassembled in space (`pbc_tools.assemble`, a minimum spanning tree over one anchor atom per residue, each residue placed at the periodic image nearest the residue it hangs from) and passed to a single `gmx rms` call; `gmx trjconv` is not used, so unbonded cutout fragments are never left in different images. `-v` prints the longest link of each assembly. Values are always computed and never abort a run. `groscore.py` / `groscore_fe.py` summarize them, add `RMSD_mean_A` / `RMSD_max_A` columns to the score files, and flag structures above `--rmsd-warn` (default 10 Å) as `HIGH_RMSD`. A missing or failed measurement is recorded as `nan` and changes nothing. `groscore_fe.py` adds a second per-structure flag, `LOW_RECOVERY`: the median over the scored cycles of the contact recovery from `interface_qc.py` (fraction of restrained pairs back within 0.15 nm of their reference distance, in `results_qc.d/`) is below 0.85. Neither flag changes a score. In the FE cycle no leg starts from the rebound state (`boundrev2` branches from the end of `holdfwd0`), so a high RMSD marks a dissipative reverse ramp, not an open cycle.

### Cutout Mode

By default (cutout on; `--no-cutout` turns it off, and `groscore_fe.py` also accepts `--cutout`), GroScore extracts only interface-proximal residues for faster simulation:
```
conf.gro → make_cutout.py → cutout.pdb → pdb2gmx → conf_cutout.gro → editconf → conf_vacbox.gro
```

With `--no-cutout`, the full protein structure is used:
```
conf.gro → editconf → conf_vacbox.gro
```

### Fragment Handling

GroScore handles complex protein structures with chain breaks:

1. **Chain Break Detection** - `renumber_pdb.py` detects gaps in residue numbering and adds TER records
2. **Minimum Fragment Size** - `make_cutout.py` extends fragments of fewer than 5 residues (`MIN_FRAGMENT_LEN`) by adding neighboring protein residues. Extension follows contiguous protein residues and stops at caps, ions and TER records; gaps are filled again after extension
3. **Fragment Merging** - Same-chain fragments are merged into single moleculetypes via `-merge interactive`
4. **Terminal Capping**:
   - **AMBER19SB**: ACE/NME caps added as explicit residues via `cap_termini.py`
   - **CHARMM36/GROMOS 54A8**: ACE caps via `cap_termini.py --ace-only`, COOH patches for C-termini

The merge input is generated by comparing chain IDs: `y` if same chain as previous fragment, `n` if different.

## Force Fields

Settings are organized by force field in `settings/<forcefield>/`:

| Setting | AMBER19SB | AMBER19SB OPC3 (default) | GROMOS 54A8 | GROMOS 54A8 RF | CHARMM36 |
|---------|---------------------|----------------|-------------|-------------|----------|
| Settings tree | `amber19sb_opc` | `amber19sb_opc3` | `gromos54a8` | `gromos54a8_rf` | `charmm36` |
| Water model | OPC (4-point) | OPC3 (3-point) | SPC | SPC | TIP3P |
| Constraints | all-bonds | all-bonds | all-bonds | all-bonds | all-bonds |
| Coulomb | PME | PME | PME | reaction field (`epsilon_rf` 61) | PME |
| VdW modifier | potential-shift | potential-shift | none | none | force-switch |
| Cutoffs | 1.0 nm | 1.0 nm | 1.4 nm | 1.4 nm | 1.2 nm |

All force fields use:
- All-bonds constraints for stable 4 fs timesteps
- Heavy hydrogen masses (`mass-repartition-factor = 3` in every MD mdp; the three minimization mdps carry none) for increased timestep
- Fragment merging (`-merge interactive`) for same-chain fragments
- Chain separation via `-chainsep id_or_ter`

### Terminal Capping Details

**AMBER19SB** (OPC water):
- ACE/NME caps via `cap_termini.py` (without --rename-nme-carbon)
- NME's methyl carbon keeps PDBFixer name "C" (AMBER19SB naming convention)
- No terminal selections in pdb2gmx (force field has no termini defined)
- OPC water model (4-point, uses `tip4p.gro` for solvation box coordinates)
- Included in standard GROMACS installations

**AMBER19SB OPC3** (default, OPC3 water):
- Same as AMBER19SB but with OPC3 water model (3-point, uses `spc216.gro` for solvation)
- Crystal ions get the OPC3 Li/Merz types: GROMACS 2026's `amber19sb.ff/aminoacids.rtp` types every ion residue `X_opc` (MG `Mg2+_opc`, ...) whatever `-water` says, so `retype_ions.py` gives them their `_opc3` twins right after each `pdb2gmx` (genion's NA/CL come from `ions_opc3.itp` already). Setups made before 2026-10-25 (bm_fe_amber_3a included) ran their crystal ions with the OPC types, and keep them on `--restart` (a structure's cycles all run on one topology); `job_fe.run` retypes a setup resumed in its probe pass, before its production box is built
- All other settings identical to AMBER19SB
- Select with `-ff amber19sb_opc3`

**GROMOS 54A8** (`gromos54a8`, PME):
- ACE caps via `cap_termini.py --ace-only` (explicit N-terminal ACE residues)
- Renames ACE's methyl carbon from "CH3" to "CA" (GROMOS RTP naming convention)
- COOH patches for C-termini via pdb2gmx
- Terminal selection "8" (none) for N-term (ACE is explicit), "1" (COOH) for C-term
- Parameters bundled in `forcefield/gromos54a8.ff/`

**GROMOS 54A8 RF** (`gromos54a8_rf`):
- The same force field, capping and parameters as `gromos54a8`, under the reaction field it was parametrized with (`coulombtype = Reaction-Field`, `epsilon_rf = 61`)
- In `job.run` / `job_fe.run` only `MDP_DIR` takes the full name. Every test of which force field this is (pdb2gmx `-ff`, GMXLIB, the capping options, the NCAA rename table, the ion protonation map) uses `FF_BASE="${FORCEFIELD%_rf}"`, because each is an `== "gromos54a8"` test that would otherwise fall through to the AMBER branch

**CHARMM36**:
- ACE caps via `cap_termini.py --ace-only` (explicit N-terminal ACE residues)
- ACE's methyl carbon keeps PDBFixer name "CH3" (CHARMM36 naming convention)
- COOH patches for C-termini via pdb2gmx
- Terminal selection "8" (none) for N-term (ACE is explicit), "1" (COOH) for C-term
- Parameters bundled in `forcefield/charmm36-jul2022.ff/` (MacKerell lab)

For all force fields, `chain_map.gs` is updated after capping since residue numbers shift due to inserted ACE (or ACE/NME) residues.

## File Formats

- `.gs` - GroScore data files (whitespace-separated, `#` for comments). Classic per-cycle rows have 5 fields (`ID N pull push RMSD`); FE rows have 43
- `.mdp` - GROMACS parameter files (in `settings/<forcefield>/`)
- `.gro` - GROMACS coordinate files
- `.xvg` - GROMACS output data (force curves)
- `.json` - `glycans.json` in a structure directory, the definitions of the fused glycan residues: `residues` (per `resname`: `family`, `parent`, `link`, `atoms` with `name`, `element`, `charge`, `h` and `src`, `bonds` as `[a, b, order]`, `hnames`, `sugars`, `instances`), `excluded`, `max_sugars` (null without a cap) and `report`; an instance with a joined stretched link lists it in `stretched_links`. `fuse_glycans.py` writes it (also when nothing is fused) and `parametrize_ncaa.py` reads it

## Key Parameters

- Timestep: 4 fs (heavy hydrogen masses)
- Interface cutoff: 0.6 nm
- Elastic network range: 0.4-0.9 nm
- Keep cutoff for interface extraction: 2.0 nm
- Minimum fragment size: 5 residues
- Ion concentration: 0.15 M NaCl
- LINCS order: 6 (for improved constraint stability)

## Structure Parameter File (sp.gs)

The `sp.gs` file specifies which structures to analyze and which PDB chain(s) constitute "protein B" (the protein to be pulled away):

```
# Structure_ID  Chains_for_Protein_B
1               B
2               B,C
3               D
```

- **Structure_ID**: Directory name containing `input.pdb`
- **Chains_for_Protein_B**: Comma-separated PDB chain identifiers to pull away

The `renumber_pdb.py` utility reads the PDB file, assigns sequential residue numbers, and generates `chain_map.gs` containing residue numbers for protein B, which other utilities use for protein separation.

## Code Patterns

- File parsing filters comments with `if not line.strip().startswith("#")`
- Results are collected into arrays pre-filled with NaN (works as (structures, 2 x cycles), pull and push interleaved; RMSDs as (structures, cycles)), so a missing value stays NaN and its cycle is not scored
- Distances across the periodic boundary use the exact triclinic minimum image of `utils/pbc_tools.py`. The scripts built on it read `.gro` files by atom position and fixed columns, not by the atom-number field, which wraps at 100000
- Exit codes: 0 (success), 1 (failure); stage-0 status strings (results_0.gs): "OK", "BROKEN" (broken interface loop), "ENTANGLED" (entangled loops), "FAILED" (setup/EM failure, e.g. emin_vac.gro not produced), "NODIR" (missing structure dir). Anything != "OK" excludes the structure from scoring, except "SETUP_RUNNING" in `groscore_fe.py`. Further statuses:
  - "SETUP_RUNNING": not a failure. `groscore.py` does not score the structure until OK; `groscore_fe.py` (read_status) ignores it and scores the cycles that exist, else PENDING
  - "SETUP_FAIL", "TRJCONV_FAIL", "ASSEMBLE_FAIL" (`assemble_frame.py` could not assemble the restraint frame), "DISRES_FAIL" (classic: no interface, a split frame, or an unusable frozen restraint set), "LIGAND_FAIL" (`ligand_failed.gs` names the instance and reason)
  - "PROBE_FAIL" and "BORESCH_FAIL" (FE; for BORESCH_FAIL the job log prints the first line of `boresch_failed.gs`, e.g. `INPUT_FRAME_SPLIT`)

  Only the last line per structure counts. Classic `job.run` writes OK only once the restraint set exists. `job_fe.run` writes OK after the loop checks, and `setup.done` marks a complete FE setup. A stage-0 failure exits 0; `job.run` exits 1 when a cycle has no complete row, or when another job holds `job_run.lock`.
- Protein separation uses chain map file (`-m/--chainmap` parameter) containing residue numbers for protein B
