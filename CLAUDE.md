# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GroScore is a computational chemistry toolkit for protein-protein affinity scoring using short steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

**Contact:** jan@ackergarten.at

## Tech Stack

- Python 3.10 with NumPy 2.2, SciPy 1.15, OpenMM 8.5, and PDBFixer 1.12
- GROMACS 2026.0 (external MD engine)
- SLURM 23.11 job scheduler for HPC execution
- Force fields: AMBER19SB with OPC3 water (default), AMBER19SB with OPC water, GROMOS 54A7 (united-atom), GROMOS 54A8 (united-atom), or CHARMM36 (all-atom)
- Water models: SPC (GROMOS) or TIP3P (CHARMM)

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
- `-ff, --forcefield` - Force field: `amber19sb_opc3` (default), `amber19sb_opc`, `gromos54a8`, or `charmm36`
- `--no-cutout` - Use full protein structure instead of interface cutout (slower, cutout is default)
- `--rmsd-warn` - Rebinding sanity-check threshold in Å (default: 10.0); only flags, never changes a score
- `--run-local` - Run locally instead of via SLURM; requires `--ngpus N`. Optional: `--jobs-per-gpu N` (default 8, 0 = start everything), `--threads-per-job N` (default 1)

After initial run, jobs are submitted via auto-generated `array_submit.run` — or, with `--run-local`, handed to a detached `utils/local_runner.py` process.

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

## Architecture

### Main Components

- **groscore.py** - Main orchestrator that reads structure parameters, dispatches jobs (SLURM submission scripts, or a local GPU pool with `--run-local`), monitors job completion, and performs final statistical analysis
- **job.run** - Bash script executed per structure, as a SLURM array task or a local pool job; runs the complete MD workflow

### Utility Scripts (utils/)

| Script | Purpose |
|--------|---------|
| `renumber_pdb.py` | Assigns sequential residue numbers across chains, detects chain breaks |
| `fix_pdb.py` | Fixes missing atoms and non-standard residues using PDBFixer |
| `cap_termini.py` | Adds ACE/NME terminal caps (CHARMM36/AMBER19SB only) |
| `check_brokenloop.py` | Validates protein loop connectivity before simulation |
| `check_entangledloops.py` | Detects topological knots that would invalidate results |
| `make_cutout.py` | Extracts interface region, enforces minimum 3-residue fragments |
| `make_disres_en.py` | Generates distance restraints and elastic network |
| `integrate.py` | Integrates force curves from pulling simulations |
| `rebound_rmsd.py` | Rebinding sanity check: backbone RMSD of the re-bound structure vs. the bound reference |
| `local_runner.py` | `--run-local` backend: bounded job pool with one GPU pinned per worker slot (also importable: `launch_local`, `print_local_status`) |
| `estimators.py` | BAR and the overlap statistic that gates it. **Imported as a module** by both orchestrators, unlike every other entry here, which are run as subprocesses. Numpy/scipy only, no side effects, so `tests/test_bar.py` can import it directly |

### Simulation Pipeline

1. **Stage 0**: Structure validation, PDB conversion, solvation, energy minimization (creates `emin_solv.gro`)
2. **Initial Equilibration**: One full equilibration to generate distance restraints (creates `npt_init_cluster.gro` for `make_disres_en.py`)
3. **Independent Cycles**: N cycles, each with:
   - Fresh full equilibration (NVT 1-5 + NPT) from `emin_solv.gro` with new random velocities
   - Pull simulation (unbinding)
   - Short NPT re-equilibration + Push simulation (binding)
   - Rebinding sanity check (`rebound_rmsd.py`), appended to the cycle's results file
4. **Final**: Statistical analysis producing two ranking methods, in two files:
   - `scores_avg.gs` - Simple average of pulls/pushes
   - `scores_cgi.gs` - Crooks Gaussian Intersection

   Both files additionally carry appended `BAR`, `BAR_CI95` and `BAR_note`
   columns. BAR is a comparison column in the classic engine, NOT its score, and
   reads `nan BAR_NO_OVERLAP` on essentially every structure because the forward
   and reverse works do not overlap. In `groscore_fe.py` BAR *is* the headline.

This architecture ensures statistically independent samples by starting each pull/push cycle from a fresh equilibration.

### Rebinding Sanity Check

Each cycle's work values only describe the intended binding event if the push leg actually restored the original complex. `utils/rebound_rmsd.py` measures the backbone RMSD between the bound state the cycle was equilibrated in and the state the push/rebinding leg ended in:

| Engine | Reference | Query | Written to |
|--------|-----------|-------|------------|
| `job.run` (classic) | `npt_c<N>.gro` | `bindrev_<2N>.gro` | 3rd column of `results_<2N>.gs` |
| `job_fe.run` (FE) | `npt_c<N>.gro` | `bindrevA_<N>.gro` | last column of `results_fe.d/<id>_c<N>.gs` |

Both frames are made whole, reduced to the `Protein` group and image-corrected three ways each (self-fix, `pbc cluster`, combined); the minimum of the 3×3 RMSD grid is taken, so a chain sitting in the wrong periodic image cannot fake a large value. Values are always computed and never abort a run — `groscore.py` / `groscore_fe.py` summarise them, add `RMSD_mean_A` / `RMSD_max_A` columns to the score files, and flag structures above `--rmsd-warn` (default 10 Å) as `HIGH_RMSD`. A missing or failed measurement is recorded as `nan` and changes nothing.

### Cutout Mode

By default (`--cutout`), GroScore extracts only interface-proximal residues for faster simulation:
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
2. **Minimum Fragment Size** - `make_cutout.py` extends fragments < 3 residues by adding neighbors
3. **Fragment Merging** - Same-chain fragments are merged into single moleculetypes via `-merge interactive`
4. **Terminal Capping**:
   - **AMBER19SB**: ACE/NME caps added as explicit residues via `cap_termini.py`
   - **CHARMM36/GROMOS 54A7/GROMOS 54A8**: ACE caps via `cap_termini.py --ace-only`, COOH patches for C-termini

The merge input is generated by comparing chain IDs: `y` if same chain as previous fragment, `n` if different.

## Force Fields

Settings are organized by force field in `settings/<forcefield>/`:

| Setting | AMBER19SB | AMBER19SB OPC3 (default) | GROMOS 54A7 | GROMOS 54A8 | CHARMM36 |
|---------|---------------------|----------------|-------------|-------------|----------|
| Water model | OPC (4-point) | OPC3 (3-point) | SPC | SPC | TIP3P |
| Constraints | all-bonds | all-bonds | all-bonds | all-bonds | all-bonds |
| Coulomb | PME | PME | PME | PME | PME |
| VdW modifier | none | none | none | none | force-switch |
| Cutoffs | 1.0 nm | 1.0 nm | 1.4 nm | 1.4 nm | 1.2 nm |

All force fields use:
- All-bonds constraints for stable 4 fs timesteps
- Heavy hydrogen masses (`-heavyh`) for increased timestep
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
- All other settings identical to AMBER19SB
- Select with `-ff amber19sb_opc3`

**GROMOS 54A7**:
- ACE caps via `cap_termini.py --ace-only` (explicit N-terminal ACE residues)
- Renames ACE's methyl carbon from "CH3" to "CA" (GROMOS RTP naming convention)
- COOH patches for C-termini via pdb2gmx
- Terminal selection "2" (none) for N-term (ACE is explicit), "1" (COOH) for C-term

**GROMOS 54A8**:
- ACE caps via `cap_termini.py --ace-only` (explicit N-terminal ACE residues)
- Renames ACE's methyl carbon from "CH3" to "CA" (GROMOS RTP naming convention)
- COOH patches for C-termini via pdb2gmx
- Terminal selection "8" (none) for N-term (ACE is explicit), "1" (COOH) for C-term
- Parameters bundled in `forcefield/gromos54a8.ff/`

**CHARMM36**:
- ACE caps via `cap_termini.py --ace-only` (explicit N-terminal ACE residues)
- ACE's methyl carbon keeps PDBFixer name "CH3" (CHARMM36 naming convention)
- COOH patches for C-termini via pdb2gmx
- Terminal selection "8" (none) for N-term (ACE is explicit), "1" (COOH) for C-term
- Parameters bundled in `forcefield/charmm36-jul2022.ff/` (MacKerell lab)

For all force fields, `chain_map.gs` is updated after capping since residue numbers shift due to inserted ACE (or ACE/NME) residues.

## File Formats

- `.gs` - GroScore data files (two/three column, `#` for comments)
- `.mdp` - GROMACS parameter files (in `settings/<forcefield>/`)
- `.gro` - GROMACS coordinate files
- `.xvg` - GROMACS output data (force curves)

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
- Large arrays pre-allocated (1,000,000 elements) then sliced to actual size
- Distance calculations use explicit 3D Euclidean formula
- Exit codes: 0 (success), 1 (failure); stage-0 status strings (results_0.gs): "OK", "BROKEN" (broken interface loop), "ENTANGLED" (entangled loops), "FAILED" (setup/EM failure, e.g. emin_vac.gro not produced), "NODIR" (missing structure dir). Anything != "OK" excludes the structure from scoring.
- Protein separation uses chain map file (`-m/--chainmap` parameter) containing residue numbers for protein B
