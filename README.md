# GroScore

<p align="center">
  <img src="logo.png" alt="GroScore Logo" width="200">
</p>

**Computational Chemistry Toolkit for Protein-Protein Affinity Scoring with MD**

[![Python](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2026-orange.svg)](https://www.gromacs.org/)

GroScore estimates binding affinities between protein pairs using short steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

---

## Features

- **Automated MD Pipeline** - Complete workflow from structure preparation to final scoring
- **SLURM Integration** - Efficient HPC execution via job arrays
- **Multiple Force Fields** - Support for AMBER19SB (all-atom), CHARMM36m (all-atom), and GROMOS 54A8 (united-atom)
- **Structural Ion Support** - Automatic handling of 21 ion types (ZN, CA, MG, CU, FE, MN, CO, NI, K, CD, SR, BA, etc.) with coordination restraints
- **Small Molecule Support** - OpenFF-based parametrization of ligands and cofactors (AMBER19SB only), with OpenBabel bond perception and RCSB template fallback
- **Crystal Water Preservation** - Crystal waters from PDB structures are retained and included in simulations
- **Cutout Mode** - Choose between interface-only (faster, default) or full-protein simulations
- **Elastic Network Restraints** - Maintains protein stability when simulating only interface-proximal atoms (within a distance cutoff) for faster computation
- **Smart Fragment Handling** - Chain break detection, small gap filling (< 4 residues), minimum fragment size enforcement, isolated cap removal, and chain boundary protection
- **Structure Validation** - Built-in checks for broken loops and topological knots
- **Rebinding QC** - Every cycle is checked for whether the push leg actually restored the original complex; structures that did not re-bind are flagged in the score files

## Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.10 | Runtime |
| NumPy | 2.2 | Numerical operations |
| SciPy | 1.15 | Distance calculations |
| OpenMM | 8.5 | PDBFixer backend |
| PDBFixer | 1.12 | Missing atom/residue repair |
| RDKit | 2025.09 | Molecular graph handling |
| OpenBabel | 3.1 | Bond order perception from 3D coordinates |
| OpenFF Toolkit | 0.16 | Small molecule parametrization |
| Interchange | 0.4 | OpenFF → GROMACS topology export |
| GROMACS | 2026 | Molecular dynamics engine |
| SLURM | 23.11 | HPC job scheduler |

## Installation

```bash
git clone https://github.com/jwperthold/GroScore.git
cd GroScore
```

### Python environment

The easiest way is to use the provided environment file (installs all Python dependencies):

```bash
conda env create -f GroScore_env.yml
conda activate GroScore
```

Or manually:

```bash
conda create -n GroScore -c conda-forge python=3.10
conda activate GroScore
conda install -c conda-forge numpy scipy openmm pdbfixer openbabel rdkit openff-toolkit openff-interchange
```

### GROMACS

`openbabel=3.1.1` requires `libxml2 <2.14`, while `gromacs=2026` requires `libhwloc ≥2.12.2` which pulls in `libxml2 ≥2.14`. These are mutually exclusive in conda. GROMACS must be installed separately:

- **HPC cluster:** load the system module (`module load gromacs/2026` or similar)
- **Workstation, separate conda env:**
  ```bash
  conda create -n gmx2026 -c conda-forge gromacs=2026   # GPU build selected automatically on CUDA 12.9+
  export PATH="$(conda info --base)/envs/gmx2026/bin:$PATH"
  ```
- **From source:** follow the [official install guide](https://manual.gromacs.org/current/install-guide/index.html)

### SLURM

**Optional.** On a single multi-GPU machine you can skip SLURM entirely and use `--run-local` instead; see [Running Without SLURM](#running-without-slurm-single-multi-gpu-workstation).

On HPC clusters SLURM is managed by the system administrators; verify with `squeue --version`. On a local workstation (Ubuntu/Debian):

```bash
sudo apt-get install slurm-wm munge
sudo systemctl enable --now munge slurmd slurmctld
```

A minimal `slurm.conf` is required; refer to the [SLURM quick-start guide](https://slurm.schedmd.com/quickstart_admin.html). GroScore ships with a `slurm/workstation.sh` template tuned for single-node execution.

## Quick Start

### 1. Set Up Project Directory

GroScore runs from a **project subdirectory** that contains your structures. Create a project folder and structure directories:

```bash
cd GroScore
mkdir -p myproject/6UD7
mkdir -p myproject/1ABC
```

### 2. Prepare Input Files

Place an `input.pdb` file in each structure directory:

```
myproject/
├── sp.gs              # Structure parameter file
├── 6UD7/
│   └── input.pdb      # Protein complex PDB
└── 1ABC/
    └── input.pdb
```

Create `sp.gs` specifying which PDB chain(s) to pull away as "protein B":

```
# Structure_ID    Chains_for_Protein_B
6UD7              B
1ABC              A,B
```

The file is whitespace-separated with two columns. Lines starting with `#` are
comments. Structure IDs can be alphanumeric (e.g., PDB IDs) and must match the
directory names. Multi-chain protein-B groups are comma-separated with no spaces
(`A,B`); the remaining chains in the input PDB form protein A.

### 3. Run GroScore

Run from within your project directory:

```bash
cd myproject
python ../groscore.py
```

**Options:**
- `-n, --numruns` - Number of independent pull/push cycles (default: 5)
- `-s, --structparams` - Structure parameter file (default: `sp.gs`)
- `-ff, --forcefield` - Force field: `amber19sb_opc3` (default), `amber19sb_opc`, `gromos54a8`, or `charmm36`
- `--no-cutout` - Use full protein structure instead of interface cutout (slower, cutout is default)
- `--no-ligand-param` - Skip OpenFF small molecule parametrization (AMBER forcefields).
- `--slurm` - SLURM template name from `slurm/` directory (default: `workstation`). Templates are plain `#SBATCH`-prefixed shell scripts; ship with `slurm/workstation.sh` (single workstation) and `slurm/vsc5.sh` (VSC-5 cluster). To target a different system, drop a new `<name>.sh` template into `slurm/` and pass `--slurm <name>`.
- `--run-local` - Run on this machine instead of submitting to SLURM, spreading jobs over the local GPUs. Requires `--ngpus`. See [Running Without SLURM](#running-without-slurm-single-multi-gpu-workstation)
- `--ngpus N` - Number of GPUs to distribute local jobs over (mandatory with `--run-local`)
- `--jobs-per-gpu N` - Concurrent jobs per GPU in local mode (default: 8; `0` starts every job at once)
- `--threads-per-job N` - CPU threads per local job, i.e. `gmx mdrun -nt` (default: 1)
- `--restart` - Resubmit jobs (useful for continuing interrupted runs)
- `--inject-job-run` - Inject fresh job.run into archived (.tar.gz) structures (skipped by default)
- `--rmsd-warn` - Threshold in Å for the [rebinding QC](#rebinding-sanity-check-qc) (default: 10.0). Only flags structures, never changes a score

This will:
- Generate `struct_map.gs` (maps SLURM array indices to structure IDs)
- Copy `job.run` to each structure directory
- Create `run.gs` in each structure directory with chain and run parameters
- Generate and submit `array_submit.run` (SLURM job array script)

### 4. Monitor Progress

GroScore uses SLURM job arrays to run simulations in parallel. Monitor with:

```bash
squeue -u $USER
```

With `--run-local` there is no queue to inspect; use `local_status.gs` and `local_runner.log` instead (see [Running Without SLURM](#running-without-slurm-single-multi-gpu-workstation)).

Re-run `python ../groscore.py` periodically to check progress and collect results.

### 5. Collect Results

Results are written to two output files ranked by binding affinity:

| Output File | Method | Required Cycles |
|-------------|--------|-----------------|
| `scores_avg.gs` | Simple average of pull/push works | any (≥ 1) |
| `scores_cgi.gs` | Crooks Gaussian Intersection of forward/reverse work distributions | ≥ 20 |

Note that CGI requires at least 20 cycles to fit forward and reverse work distributions; with the default `--numruns 5` only `scores_avg.gs` is produced. Increase `-n` if you want CGI estimates.

Both files also carry `BAR`, `BAR_CI95` and `BAR_note` as **appended** columns. BAR is a comparison column here and not the classic score, which stays the average. On the classic protocol it will read `nan ... BAR_NO_OVERLAP` on essentially every structure, and that is the informative part: forward and reverse works are separated by roughly 132 RT at the current pull rate, so no sampled work lies in the region a bidirectional estimator reads ΔG from. See [why BAR leads the FE scores but not these](#why-bar-is-the-fe-headline-and-not-the-classic-one).

#### Interpreting the score

- **Sign convention**: more negative score ⇄ tighter binding. Predicted pKd is monotonically *decreasing* in the score.
- **Units**: kJ·mol⁻¹. The score is the integrated pulling work along the unbinding/rebinding coordinate, averaged over cycles.
- **Convert to pKd**: use the linear fits provided per force field in the [Benchmark Data](#benchmark-data-haddocking-protein-protein-affinity-benchmark) section, e.g. for AMBER19SB/OPC3: `pKd ≈ -0.0176 × score + 3.4513`. These coefficients are calibrated on the HADDOCKING benchmark.
- **Uncertainty**: the `CI95` column is a 95 % confidence interval on the score from the between-cycle scatter.
- **Trustworthiness**: check the `RMSD_flag` column before using a score; see [Rebinding Sanity Check (QC)](#rebinding-sanity-check-qc) below.

## Running Without SLURM (Single Multi-GPU Workstation)

GPU cloud providers typically rent out one fat machine (4 or 8 GPUs in a single box) with no scheduler installed. `--run-local` replaces the SLURM job array with a background runner that starts the jobs itself and pins each of them to one GPU:

```bash
python ../groscore.py -n 5 --run-local --ngpus 8
```

Every job gets `gmx mdrun -nt 1 -gpu_id <n> -pin off`: one CPU thread, one GPU. That is deliberate: GroScore systems are small enough to run essentially GPU-resident (nonbonded, PME and the update/constraints all offloaded), so a job's CPU thread mostly feeds the device, and 8 single-threaded jobs on 8 GPUs beat one 8-threaded job on one GPU by close to the full factor.

**How work is distributed.** `--ngpus` is mandatory and defines the round-robin: job 1 → GPU 0, job 2 → GPU 1, …, job 9 → GPU 0 again. By default 8 jobs share each GPU (`--jobs-per-gpu 8`), as a single cutout-sized system leaves the GPU idle during CPU-side work, so stacking jobs raises *aggregate* throughput well past what one job per device achieves. Anything beyond `--ngpus × --jobs-per-gpu` waits in a queue and starts as slots free up, so a 500-structure screen does not try to open 500 GROMACS processes at once. Jobs are handed out dynamically rather than pre-assigned, so a slow structure cannot leave its GPU idle at the end of the run.

| Option | Effect |
|---|---|
| `--ngpus N` | GPUs to distribute over (mandatory with `--run-local`) |
| `--jobs-per-gpu N` | Concurrent jobs per GPU (default `8`), trading GPU memory and host RAM for throughput. Lower it for `--no-cutout` runs or on GPUs with little memory. `0` starts every job immediately, no queue |
| `--threads-per-job N` | `gmx mdrun -nt` per job (default `1`). Raise it only if you have far more cores than concurrent jobs |

The defaults assume a cloud-style node: `--ngpus 8 --jobs-per-gpu 8` is 64 concurrent single-threaded jobs, which wants ~64 CPU cores and enough GPU memory for 8 solvated systems per device. On a workstation with fewer cores than that, either lower `--jobs-per-gpu` or accept that the jobs share cores.

**Monitoring.** The runner is detached from the launching terminal, so simulations survive closing the shell or losing the SSH connection. Two files in the project directory track it:

```bash
cat local_status.gs      # counts: total / done / failed / running / pending
tail -f local_runner.log # per-job start and exit lines, with the GPU used
```

Re-running `python ../groscore.py --run-local --ngpus 8` while a runner is active is safe: it refuses to start a second one and just re-scores whatever has finished, printing a one-line progress summary. Each structure's own output goes to `<structure>/job_local.out`, the local equivalent of a SLURM job log.

To stop a run, `kill` the pid in `local_runner.pid`; the runner terminates its running jobs and exits. Since every stage of `job.run` is restart-safe, re-launching later resumes from the last completed step.

**Caveats.** `--ngpus` must match the machine: a job assigned to a non-existent device fails in `mdrun` (GroScore warns if `nvidia-smi` reports fewer GPUs than requested). The runner uses `mdrun -gpu_id`, so device numbering follows `nvidia-smi`; it does not set `CUDA_VISIBLE_DEVICES`. And unlike SLURM there is no memory accounting: with a high `--jobs-per-gpu` on large no-cutout systems the machine can run out of host RAM or GPU memory.

## Rebinding Sanity Check (QC)

A cycle's work only describes the intended binding event if the push leg actually put the complex back together. If the partners re-associate in a different pose, or drift apart and never return, the integrated force curve is still a number, just not the number you wanted. Every cycle of both engines is therefore checked automatically; no extra command is needed.

`utils/rebound_rmsd.py` measures the **backbone RMSD between the bound state the cycle was equilibrated in and the state the rebinding leg ended in**:

| Engine | Reference (bound) | Query (re-bound) | Stored in |
|--------|-------------------|------------------|-----------|
| `groscore.py` (classic) | `npt_c<N>.gro` | `bindrev_<2N>.gro` | 3rd column of `results_<2N>.gs` |
| `groscore_fe.py` (FE) | `npt_c<N>.gro` | `bindrevA_<N>.gro` | last column of `results_fe.d/<id>_c<N>.gs` |

### Reading the result

`groscore.py` adds three columns to `scores_avg.gs` / `scores_cgi.gs` (and to the per-cycle `scores_*_c<N>.gs`), summarising exactly the cycles that entered the score:

```
# Structure_ID  Score  CI95  Cycles_Used  RMSD_mean_A  RMSD_max_A  RMSD_flag
1A2K   -117.9   2.5   5   1.75   1.98   OK
2OOB    -92.4   2.1   5  11.45  18.70   HIGH_RMSD
```

and closes the run with a summary of what went wrong, if anything:

```
Rebinding sanity check (backbone RMSD of the re-bound structure, warn > 10.0 A):
  25 simulations analyzed: mean 2.14 A, median 1.88 A, max 18.70 A

  WARNING: 1 structure did not re-bind properly in at least one cycle.
  Their scores are reported but should be treated with caution
  (flagged HIGH_RMSD in scores_avg.gs / scores_cgi.gs):
    2OOB                 c1=18.7, c4=12.3
```

The console stays short whatever the screen size: one aggregate line, then the ten worst structures (five bad cycles each) and a count of the rest. `grep HIGH_RMSD scores_avg.gs` gives the complete list. Individual per-cycle values are printed once each into the structure's own SLURM log, not into the summary.

`groscore_fe.py` behaves the same way, with `RMSD_mean_A` / `RMSD_max_A` columns and a `HIGH_RMSD` entry in the `Note` column of `scores_fe.gs`. There the check has a second meaning: the thermodynamic cycle is only closed if the rebinding leg returned the system to the state the bound leg started from.

### Interpretation

| RMSD | Meaning |
|------|---------|
| 1–5 Å | Normal. Thermal fluctuation plus whatever the interface relaxed into during the cycle. |
| 5–10 Å | Worth a look. Partial rebinding, a shifted interface, or a flexible loop that did not recover. |
| > 10 Å | `HIGH_RMSD`. The partners did not return to the original pose. |
| `nan` | Not measured: a run that predates the check, or a failed measurement. Never an error. |

The threshold is `--rmsd-warn` (default 10.0 Å) on both engines. **Scores and free energies are always computed and reported**: the check never aborts a simulation, never drops a cycle and never changes a number. It tells you which results to distrust: with a handful of flagged cycles, the usual response is to add cycles (`-n` plus `--restart`) and see whether the structure's score is dominated by them.

A single measurement can also be taken by hand from inside a structure directory (`<project>/<structure_id>/`):

```bash
python3 ../../utils/rebound_rmsd.py --ref npt_c3.gro --query bindrev_6.gro -v
# 1.9750
# rebound_rmsd: [ref=sf, query=sf]
```

## Benchmark Data (HADDOCKING Protein-Protein Affinity Benchmark)

| **AMBER19SB/OPC** | |
|--------------------|-------------|
| **Fit:** pKd = -0.0155 × GroScore + 4.2704 | Convergence |
| <img src="/benchmark/results/correlation_plot_amber_opc.png" alt="Correlation Plot AMBER19SB/OPC" height="290"> | <img src="/benchmark/results/convergence_plot_amber_opc.png" alt="Convergence Plot AMBER19SB/OPC" height="290"> |

| **AMBER19SB/OPC3** | |
|--------------------|-------------|
| **Fit:** pKd = -0.0176 × GroScore + 3.4513 | Convergence |
| <img src="/benchmark/results/correlation_plot_amber_opc3.png" alt="Correlation Plot AMBER19SB/OPC3" height="290"> | <img src="/benchmark/results/convergence_plot_amber_opc3.png" alt="Convergence Plot AMBER19SB/OPC3" height="290"> |

| **CHARMM36m/TIP3P** | |
|--------------------|-------------|
| **Fit:** pKd = -0.0209 × GroScore + 2.9054 | Convergence |
| <img src="/benchmark/results/correlation_plot_charmm.png" alt="Correlation Plot CHARMM36m/TIP3P" height="290"> | <img src="/benchmark/results/convergence_plot_charmm.png" alt="Convergence Plot CHARMM36m/TIP3P" height="290"> |

| **GROMOS 54A8/SPC** | |
|---------------------|-------------|
| **Fit:** pKd = -0.0178 × GroScore + 3.7434 | Convergence |
| <img src="/benchmark/results/correlation_plot_gromos_54a8.png" alt="Correlation Plot GROMOS 54A8/SPC" height="290"> | <img src="/benchmark/results/convergence_plot_gromos_54a8.png" alt="Convergence Plot GROMOS 54A8/SPC" height="290"> |

To reproduce the benchmark, see the [Benchmark](#benchmark) section below.

## Force Fields

GroScore supports multiple force fields, selectable via the `-ff` option:

| Force Field | Type | Water Model | Constraints | Cutoffs | Terminal Capping |
|-------------|------|-------------|-------------|---------|------------------|
| **AMBER19SB OPC3** (default) | All-atom | OPC3 (3-point) | h-bonds | 1.0 nm | ACE/NME  |
| **AMBER19SB OPC** | All-atom | OPC (4-point) | h-bonds | 1.0 nm | ACE/NME |
| **CHARMM36m** | All-atom | TIP3P | h-bonds | 1.2 nm | ACE/COOH |
| **GROMOS 54A8** | United-atom | SPC | all-bonds | 1.4 nm | ACE/COOH |

All force fields use:
- **Electrostatics**: PME (Particle Mesh Ewald) for long-range electrostatic interactions
- **Constraints**: h-bonds, which with `mass-repartition-factor = 3` is what justifies the 4 fs timestep. Constraining heavy-atom bonds as well builds coupled constraint chains longer than GPU LINCS supports, which disables the GPU-resident update and cost 1.29× in throughput on the real 2KTF system. GROMOS 54A8 keeps **all-bonds**: it is united-atom, so hydrogen mass repartitioning has almost nothing to act on, and all bonds constrained is its native validated protocol rather than a workaround
- **Heavy hydrogens**: `mass-repartition-factor = 3` for stable 4 fs timesteps
- **Timestep**: 4 fs (`dt = 0.004` ps) for all production stages
- **SMD pulling per leg**: 1.25 × 10⁶ steps × 4 fs = 5 ns; one cycle = pull + push = 10 ns of SMD plus ~120 ps NVT/NPT equilibration

**Terminal Capping Details:**
- **AMBER19SB**: Uses ACE (N-acetyl) and NME (N-methylamide) caps added as explicit residues via PDBFixer before pdb2gmx processing. This provides proper neutral termini for fragment ends.
- **CHARMM36m**: Uses ACE (N-acetyl) caps at N-termini (explicit residues) and COOH patches at C-termini for improved stability.
- **GROMOS 54A8**: Uses ACE caps at N-termini (explicit residues) and COOH patches at C-termini.

The CHARMM36m force field parameters (from [MacKerell lab](https://mackerell.umaryland.edu/charmm_ff.shtml)) are included in `forcefield/charmm36-jul2022.ff/`. The GROMOS 54A8 force field parameters (from [Oostenbrink group](https://boku.ac.at/en/nwnr/mmsi/research/force-field-development)) are included in `forcefield/gromos54a8.ff/`.

## Simulation Pipeline

```
Stage 0: Preparation
├── PDB fixing (missing atoms, non-standard residues)
├── NCAA parametrization (OpenFF sidechain + AMBER backbone)
├── Ligand extraction & OpenFF parametrization (AMBER19SB)
├── Ion coordination protonation (CYS/HIS)
├── Crystal water extraction
├── Structure validation
├── PDB conversion (pdb2gmx)
├── Ligand/water/ion merging into topology
├── Ion coordination restraints (topology-level)
├── Solvation (water + 0.15 M NaCl)
└── Energy minimization → emin_solv.gro

Initial Equilibration (for distance restraints)
└── 5-phase NVT + NPT → npt_init_cluster.gro   (classic engine)

Independent Cycles (N cycles, default 5)
├── Cycle 1:
│   ├── Fresh full equilibration (NVT 1-5 + NPT)
│   ├── Pull (unbinding SMD)
│   ├── Short NPT + Push (binding SMD)
│   └── Rebinding QC (backbone RMSD, re-bound vs. bound)
├── Cycle 2:
│   ├── Fresh full equilibration (NVT 1-5 + NPT)
│   ├── Pull (unbinding SMD)
│   ├── Short NPT + Push (binding SMD)
│   └── Rebinding QC (backbone RMSD, re-bound vs. bound)
└── ... (each cycle independent, new random velocities)

Final: Analysis
├── Statistical scoring (2 methods)
└── Rebinding QC summary (flags HIGH_RMSD structures)
```

Each cycle starts fresh from `emin_solv.gro` with independent equilibration, providing statistically independent samples for robust scoring.

### Reproducibility

Each cycle draws fresh velocities from a Maxwell-Boltzmann distribution at `gen_temp = 62` K and heats to `ref_t = 310` K over the five-stage NVT ladder. The initial-velocity seed in every NVT/NPT/SMD `.mdp` is set to `gen_seed = -1`, i.e. GROMACS picks a fresh seed from the wall clock at submission time. This is deliberate, as independent cycles must sample independent trajectories, but it does mean that scores from a re-submitted run will not be bitwise-identical to the original. The CI95 column in `scores_avg.gs` quantifies the resulting between-cycle variance.

### Throughput

On a single GPU (consumer-grade RTX-class), expect roughly **8 GPU-hours per structure for 5 cycles** (including all equilibration legs and 5 × 10 ns of SMD). Cost scales linearly with cycle count and roughly linearly with system size; the interface-cutout mode (default) keeps system size near-constant across most complexes, so per-structure walltimes are tightly clustered. The benchmark directory contains `compute_walltime.py`, which extracts realised walltimes and PFLOP counts from completed SLURM logs.

## Absolute Binding Free Energies (GroScore-FE)

> **Experimental / in development.** A sign error in the Boresch dihedral references was found and fixed on 2026-08-14; every free-energy result produced before that is void. Convergence of the unbinding leg is still unresolved. See [Caveats](#caveats) at the end of this section.

The classic pipeline already produces an *absolute* free-energy estimate from the pulling work, but it is **biased**: the empirical interface distance restraints are present throughout and their free-energy contribution is never removed, hence the scores are not directly comparable to experiment. `groscore_fe.py` (with `job_fe.run`) removes that bias by accounting for the restraint free energies explicitly, i.e. by replacing the many empirical interface restraints with a rigorous, analytically correctable restraint scheme, so that the result approaches an experimentally comparable absolute binding free energy.

During unbinding, the atom-atom interface restraints are gradually switched off while a set of **Boresch orientational restraints** (one distance, two angles and three dihedrals, defined on backbone center-of-mass anchor groups) is switched on. As the Boresch restraint has a closed-form standard-state free energy (Boresch et al. 2003), its contribution in the separated state is computed analytically rather than simulated. Every leg is run forward and reverse, and the resulting works are combined with the Crooks-Gaussian-Intersection estimator as in the classic engine.

The absolute binding free energy is assembled from a thermodynamic cycle:

```
dG_bind = -( dG_intro + dG_unbind + dG_release )
   dG_intro    interface restraints introduced in the bound state   (dhdl)
   dG_unbind    interface -> Boresch handoff + separation to 1.0 nm  (pull work + dhdl)
   dG_release   analytical Boresch standard-state term               (closed form)
```

As GROMACS provides no lambda-dependent pull reference, the switching work is captured in two channels which add without double-counting: the **pull force** (mechanical separation via the moving reference) and **dH/dλ** (force-constant switching). Relative to the classic protocol, each cycle uses 20 ns of bound-state equilibration and a 1.0 nm separation, and adds the bound-state restraint legs (see [Compute cost](#compute-cost) below).

Run it like the classic engine (same inputs and working-directory layout), substituting the script name:

```bash
python3 ../groscore_fe.py -s sp.gs -n 5 -ff amber19sb_opc3 --slurm workstation
```

All four force fields are supported (`amber19sb_opc3`, `amber19sb_opc`, `charmm36`, `gromos54a8`). Each one's FE `.mdp` files are derived from its own `bind.mdp` / `npt.mdp` / `nptrev.mdp`, so its nonbonded treatment (cutoffs, vdw modifier, dispersion correction) carries over unchanged and only the free-energy settings are added. If you edit a force field's base mdps, regenerate the FE set with:

```bash
python3 utils/make_fe_mdps.py
```

Results are written to `scores_fe.gs` (absolute dG_bind in kJ/mol and as pKD, together with the three cycle components), fed by the per-cycle works in `results_fe.d/`. **`dG_bind` and `pKD` are BAR**; the average and CGI values are retained beside them, so the file leads with `dGbind_bar` and then repeats the block for `_avg` and `_cgi`. Three estimators over the same works are the cheapest convergence check there is: where they agree the leg has converged, and where they disagree it has not, whichever number you would rather believe. `dG_unbind_*` is the [staged](#why-these-boundaries) sum over the ramp stages, and the columns after it are its audit trail: one `dG_unb<L>_bar` pair per stage, then `dG_unbind_1s_bar`, the same free energy taken from the whole ramp in one shot. The stage columns follow the protocol, so the current five-stage ramp writes `dG_unbA_bar` through `dG_unbE_bar`. `dG_intro_*` is staged the same way, with one `dG_intro<N>_bar` pair per bound sub-leg and `dG_intro_1s_bar` as its one-shot cross-check. Each cycle also carries the [rebinding sanity check](#rebinding-sanity-check-qc): the thermodynamic cycle only closes if the rebinding leg returned the complex to the pose the bound leg started from, hence `RMSD_mean_A` / `RMSD_max_A` and a `HIGH_RMSD` note flag the structures whose numbers should not be trusted.

Every scoring pass also writes **`fe_works.png`**, the work distributions of every leg with one row per structure: bound-state legs on the left, then one panel per ramp stage. No second command is needed; the figure appears alongside `scores_fe.gs` whenever any cycle has finished, including part-way through a run. The undivided ramp is not drawn, since its stages already are, but it does appear in the consistency table below, as `unbind A+B+C+D+E`, the baseline the split is measured against. That row is arithmetic, not a leg: no simulation of that name runs.

Every structure is plotted, in `sp.gs` order, **16 rows to a file**. Beyond that the figure is paginated into `fe_works_01.png`, `fe_works_02.png` and so on (a run needing only one page keeps the plain `fe_works.png`, and pages left over from an earlier, longer run are removed). Sixteen rows correspond to 1980 × 9792 px and roughly 2 MB. Matplotlib will write a considerably taller image, but GPU textures and most viewers cap a dimension near 16384 px and PIL rejects anything above 89 Mpx as a decompression bomb, hence a single 200-structure sheet would be unopenable as well as costing about 80 s and 1.8 GB to render.

The reverse distribution is drawn sign-aligned (`−W`), so that the forward and reverse histograms should **overlap and cross at ΔG**. Well-separated histograms indicate that the leg is being driven too fast: the estimate is then dominated by dissipated work, and both the average and the CGI value fall in a region where neither distribution has samples. A third consequence is that such a leg gets **no BAR value at all**, so its rule is absent from the panel and `scores_fe.gs` carries `nan` with the reason in `Note`. The per-panel `dissipation` annotation is half the gap between the two means, i.e. `(⟨W_f⟩ + ⟨W_r⟩)/2`; the free energy cancels from that sum, hence it measures hysteresis without assuming any ΔG estimate.

Scoring then prints a **Gaussian consistency table**, one row per leg, which decides whether the CGI number is a measured crossing or an extrapolation:

```
  structure  leg              n      diss   ratio   sf/sr   p_fwd   p_rev      ovl    sep  sep_max  diss/RT   verdict
  T30        restraints      60      4.12    1.15    1.05   0.618   0.412  104/120    1.9      4.3      1.6   OK
  T30        unbind/rebind   60    148.49    8.89    0.94 3.3e-05   0.089    0/120   32.0      4.3     57.6   FD+OVL+SEP
```

CGI fits **one Gaussian to each distribution** and reads ΔG from the crossing of the two curves, hence three conditions must hold. Each is tested where the assumption is actually made:

- **FD**: a **Shapiro-Wilk test of each work distribution alone**, reported as `p_fwd` and `p_rev`. The leg is flagged if either falls below `α/2`, the factor of two being a Bonferroni correction for the two tests per leg, so that a leg of genuinely Gaussian works misfires only 5 % of the time. A leg switched too fast breaks normality from the tail inward: the rare low-dissipation cycles which carry ΔG are exponentially unlikely and `n` cycles never reach that far, hence the sampled distribution is skewed and the fitted Gaussian comes out too narrow and too far out.
- **OVL**: of the `2n` sampled works, the number which land inside the **other direction's observed min…max range** (`ovl` column). Flagged at exactly zero, which means that no estimator has data in the crossing region and that whatever number is reported is the fitted model extrapolated into empty space. The gap is given in RT and states how far into an exponentially unlikely tail a cycle would have to reach, which is why additional cycles are usually futile once it is large.
- **SEP**: the crossing lies `sep/2` σ into either tail, and the most extreme of `n` samples reaches `z_n = Φ⁻¹(1 − 1/n)`, hence the limit is `sep_max = 2·z_n`, i.e. 2.3 at n=8, 3.1 at n=16, 3.7 at n=32 and 4.7 at n=100. More cycles reach further and more separation is therefore tolerable. The limit follows `n` without a ceiling, as a run long enough to place samples at 5 σ has genuinely measured a 5 σ crossing.

**OVL and SEP address the same question, but only OVL survives an outlier.** `sep` divides by a pooled σ which is quadratic in deviations, hence a single extreme work inflates the denominator and drags `sep` down; a leg can pass purely because one trajectory went badly, with the distributions exactly as far apart as before. A count moves by at most one per outlier out of `2n`. `OVL` should therefore be read first, and `sep` treated as the graded measure once `OVL` is non-zero. Only exact zero is flagged, as that is the statement requiring no assumption about shape, but a handful of works is little better than none, so the printed count should be read rather than the verdict alone.

`ratio = diss / ((σ_f² + σ_r²)/4RT)` is the linear-response consistency, as `W_diss = σ²/2RT` holds near equilibrium and 1 is therefore the ideal value. It is reported for information and **is not a flag**: it moves for reasons other than the shape of either distribution, and testing it conflated the Gaussian assumption with Crooks and with linear response, so that a failure never indicated which of the three had broken. `sf/sr` is likewise descriptive.

A check which the cycle count cannot answer reports **n/a** rather than failing. `FD` requires `n ≥ 3` (the Shapiro-Wilk minimum) and both distributions non-degenerate, `OVL` requires `n ≥ 3` for a min…max range, and `SEP` requires `n ≥ 4` for a tail. Whichever flags fired are followed by an explanation and a per-leg breakdown.

> **A passing FD is weak evidence.** At these sample sizes Shapiro-Wilk has little power; at n = 16 only gross departures are detected, hence `OK` means *not detectably non-Gaussian at n cycles* rather than *Gaussian*. `OVL` and the figure carry more information at low cycle counts.

Pass `--temp` to match a run performed at a non-default temperature. Both the figure and the table are diagnostic and change no result.

When a leg **fails**, the table reports that it failed but not why. `utils/fe_leg_efficiency.py` answers this from files which a finished cycle already leaves behind, without any new simulation:

```bash
python3 ../utils/fe_leg_efficiency.py -s 2KTF          # from the project dir
```

It reports whether the forward and reverse works overlap *at all* (a blunter question than `sep`, as an empty gap means that every estimator is reporting its fitted model rather than the run), where along the pull the hysteresis is generated, whether the dissipation obeys the near-equilibrium `1/t` law (measured by comparing the friction along the existing trajectories against the observed hysteresis), and what lengthening the leg or adding cycles would each cost. At a fixed budget the average estimator has `SE ∝ t^((1−p)/2)` for `diss ∝ t^−p`; as `p = 1` is the ceiling, doubling the leg can at best break even on variance, whereas the bias contribution points the other way. Both options are therefore priced rather than one being recommended.

### Job layout (parallel cycles)

Convergence needs many cycles, and cycles are independent, as each restarts from `emin_solv.gro` with fresh velocities. So each structure is submitted as **two jobs**:

1. a one-off **setup** job (stage 0 + initial equilibration + restraint definition), and
2. a **cycle job array** (`--array=1-N`), submitted with `--dependency=afterany` on the setup job.

Because all cycles share the restraints (elastic network, interface, Boresch) that only the setup defines, a cycle task that SLURM starts early does **not** terminate; `job_fe.run --cycle N` waits for the setup's `setup.done` marker (polling every 30 s, 6 h cap; override with `GROSCORE_SETUP_WAIT`/`GROSCORE_SETUP_POLL`). If the setup reports a failure, the waiting tasks exit cleanly rather than hanging. Each cycle writes its own `results_fe.d/<id>_c<n>.gs` (no concurrent appends to a shared file), and whichever task finishes last archives the structure, elected atomically via `mkdir`, so two tasks can never tar/delete the same directory.

Useful options:

- `--array-throttle N`: cap concurrent cycle tasks per structure (SLURM `%N`). Rarely needed: several GROMACS processes sharing one GPU give higher *aggregate* throughput than one at a time, because a single small-system run leaves the GPU idle during CPU-side work. Leave it unset unless you run out of GPU memory.
- `--sequential`: legacy layout: one job per structure running all cycles in sequence.
- `--run-local --ngpus N`: run on this machine instead of SLURM, see [Running Without SLURM](#running-without-slurm-single-multi-gpu-workstation). The same two-stage layout applies: the setup job runs first and its cycles only start once it has finished, on whichever GPU frees up. `--array-throttle` has no effect locally; `--jobs-per-gpu` is the concurrency limit there.

Re-running the command later re-submits only what is missing (a structure whose `setup.done` exists gets no new setup job) and re-scores whatever has completed.

### Adding cycles later

`-n` is the **total** number of cycles wanted, so convergence can be extended at any time by asking for more:

```bash
python3 ../groscore_fe.py -s sp.gs -n 200 --restart      # was -n 50
```

Only cycles without a complete result are queued: the array is submitted as an explicit index list (e.g. `--array=51-200`), so topping up costs one task per missing cycle instead of re-walking the finished ones. A cycle whose stored result contains `NaN` counts as missing and is recomputed; if its simulation legs are still present this is just re-integration (seconds, no MD).

This works for **archived** structures too: the setup job unpacks the tarball, the new cycles run, and the structure is re-archived once the new total is reached. The requested total is passed to the jobs via `GROSCORE_NUMCYCLES`, since `run.gs` inside a tarball cannot be rewritten. Lower `-n` values are not destructive; they simply queue nothing new.

### Compute cost

Each cycle runs twenty-six switching/hold legs plus one equilibration. **Both
halves of the cycle are staged**: the bound-state restraint switch runs as two
sub-legs and the unbinding ramp as five, with an equilibrium hold at every internal
boundary in both directions.

**The five stages are equal in time and unequal in span.** An attempt to place and
time them against a fitted dissipation density was reverted: the per-stage
dissipation varies 17-26% between setup draws, which is the same size as the gain
the fit was chasing, so it fitted sampling error and cost two of four runs their
BAR. See [Why these boundaries](#why-these-boundaries-and-why-five).

| Leg | Purpose | lambda | Length |
|---|---|---|---|
| `npt_c` | equilibrate the bound state | - | 20 ns |
| `boundfwd1` | bound restraints on (dhdl) | 0 -> 0.25 | 3.75 ns |
| `holdbfwd1` | hold | 0.25 | 1 ns |
| `boundfwd2` | bound restraints on (dhdl) | 0.25 -> 1 | 3.75 ns |
| `holdfwd0` | hold, bound | 0 | 1 ns |
| `bindfwdA` | unbind | 0 -> 0.12 | 5.2 ns |
| `holdfwd1` | hold | 0.12 | 0.5 ns |
| `bindfwdB` | unbind | 0.12 -> 0.2 | 5.2 ns |
| `holdfwd2` | hold | 0.2 | 0.5 ns |
| `bindfwdC` | unbind | 0.2 -> 0.31 | 5.2 ns |
| `holdfwd3` | hold | 0.31 | 0.5 ns |
| `bindfwdD` | unbind | 0.31 -> 0.49 | 5.2 ns |
| `holdfwd4` | hold | 0.49 | 0.5 ns |
| `bindfwdE` | unbind | 0.49 -> 1 | 5.2 ns |
| `nptrev_fe` | hold unbound | 1 | 5 ns |
| `bindrevE` .. `bindrevA` | rebind, the exact mirror | 1 -> 0 | the same, reversed |
| `holdrev4` .. `holdrev1` | holds | | 4 × 0.5 ns |
| `holdrev0` | hold, bound | 0 | 1 ns |
| `boundrev2` | bound restraints off (dhdl) | 1 -> 0.25 | 3.75 ns |
| `holdbrev1` | hold | 0.25 | 1 ns |
| `boundrev1` | bound restraints off (dhdl) | 0.25 -> 0 | 3.75 ns |
| **per cycle** | | | **100 ns** (+0.1 NVT ladder) |

At the default 50 cycles this is **~5.0 µs/structure**, plus a setup pass of
5 × 20 ns of probe equilibration. The default is 50 rather than the classic
engine's 5 because BAR returns nothing without forward/reverse work overlap, and on
a real run no shorter prefix has ever produced it: five, ten, twenty and forty
cycles all came back `BAR_NO_OVERLAP` where the full 47 scored.

#### The protocol is defined in one place

`utils/fe_protocol.py` holds the whole cycle as a table, and everything else
derives from it: the mdps (`make_fe_mdps.py`), the pull blocks and per-stage
rates (`make_boresch.py`), the leg sequence and work extraction (`job_fe.run`,
via `--shell`), the result-row width and the staged estimator (`groscore_fe.py`),
and the leg diagnostic. Moving a boundary or adding a hold is an edit to
`fe_protocol.RAMP` followed by a re-run of `make_fe_mdps.py`; nothing else
changes, and nothing else *can* disagree with it.

```bash
python3 utils/fe_protocol.py        # the cycle, leg by leg
```

#### Why these boundaries, and why five

As one 20 ns process the ramp had **zero forward/reverse overlap**, so BAR was
refused on every structure. A leg whose dissipation exceeds its own work width has
histograms that do not meet, and no amount of sampling repairs that. Splitting
partitions the dissipation into pieces each estimator can handle, and because each
stage runs at **constant rate** it also lowers the total, since a slow region can
then be crossed slowly and a fast one quickly.

The boundaries come from the friction profile, recovered as `zeta = g/v` from the
excess mean force, per stage, because the stages ran at rates differing 8x and the
raw dissipation density is not `zeta`. It is measured on **all three repeats and
pooled**, not on one of them, because the profile turned out to be a property of the
setup draw rather than of the sampling:

| | b1 | b2 | b3 | b4 | b5 |
|---|---|---|---|---|---|
| test10 | 0.164 | 0.218 | 0.295 | 0.419 | 0.630 |
| test11 | 0.094 | 0.160 | 0.241 | 0.355 | 0.535 |
| test12 | 0.088 | 0.156 | 0.230 | 0.330 | 0.518 |
| sd, cycle bootstrap **within** a run | 0.005 | 0.005 | 0.008 | 0.009 | 0.016 |
| sd **between** runs | 0.042 | 0.035 | 0.035 | 0.046 | 0.061 |

A bootstrap inside one run understates the boundary uncertainty by four to nine
times, so a single run can neither locate a boundary nor reveal that it cannot.
Boundaries then sit at **equal dissipation per stage**, which for constant-rate
stages means equal `sqrt(du × int zeta du)`; equalising the dissipation makes the
Sivak-Crooks times equal too, which is why every stage is 5.2 ns.

**Five and not six.** Each boundary costs an equilibrium hold in both directions out
of a fixed 80 ns of legs, so the ramp loses 1 ns per boundary added. Priced at that
fixed budget on the pooled friction:

| N | ramp ns | total dissipation | per stage | n_sigma | summed BAR sd | transfer penalty |
|---|---|---|---|---|---|---|
| 3 | 27.0 | 63.3 | 21.1 | 1.26 | 4.12 | 13.6% |
| 4 | 26.5 | 61.2 | 15.3 | 1.07 | 3.75 | 20.0% |
| **5** | **26.0** | **60.3** | **12.1** | **0.95** | **3.50** | **23.6%** |
| 6 | 25.5 | 62.0 | 10.3 | 0.88 | 3.54 | 27.4% |
| 7 | 25.0 | 63.0 | 9.0 | 0.82 | 3.58 | 28.6% |

Both total dissipation and summed BAR variance bottom out at five and get worse
after, because past that the holds cost more ramp time than the finer partition
saves. The last column prices **transfer**: boundaries fitted on two runs, scored on
the worst stage of the third, against boundaries fitted on that third run itself. It
climbs monotonically, so each added boundary generalises less well than the last. It
never inverts, though, and fitted boundaries beat naive equal-u spacing roughly two
to one throughout, so the fitting is worth doing.

Work widths are calibrated rather than assumed: `kappa = sigma^2/(2 RT W)` measures
**2.57** across the nine run/stage pairs, so the works are wider than linear response
and overlap is easier than the textbook value. The stage count was checked at
`kappa = 1` as well, the pessimistic case, where five still holds.

Two boundaries land on 0.306 and 0.494, i.e. on the u = 0.3 and u = 0.5 holds the
earlier three-stage ramp already had. That coincidence survives pooling; it simply
belongs at five stages rather than six.

#### Why the bound legs are 3.75 ns each, and split

The interface restraints are harmonic umbrellas switched `k_A = 0 -> k_B`, so the
free-energy integrand is exactly

    dH/dlambda = 0.5 * k_B * S,   S = sum over pairs of (d - d_ref)^2

i.e. the strain against the reference IS the work. Raising `npt_c` to 10 ns
doubled the strain handed to `boundfwd` (S 12.7 -> 27.7 nm^2, rms deviation per
restrained pair 1.65 -> 2.45 A) without rescaling the 2 ns switch, and the leg
stopped finishing what it started: at the end of `boundfwd`, `dH/dlambda` sat
**71% above** the restrained-equilibrium value, where the same quantity was 6%
*below* it when `npt_c` was 1 ns. That leg carries 12.7% of the dG_bind variance,
and its BAR overlap rested on exactly two cycles out of 47.

The fix is not to shorten `npt_c`: a long bound equilibration is what makes each
cycle an independent draw from the ensemble the Crooks analysis assumes. It is to
let the switch keep up with it. `npt_c` has since gone to 20 ns, which sharpens the
same argument.

The leg is also **split at lambda = 0.25**, but not for overlap. Pooled over the
three repeats it dissipates 10.8 kJ/mol unsplit against a work width of 23, i.e.
0.46 sigma, nowhere near the cliff the ramp was at. What the split buys is variance:
about 10% off the summed BAR error, and a third sub-leg buys nothing further while
costing 2 ns. The boundary stays at 0.25 rather than the pooled optimum of 0.277
because the per-run estimates are 0.186 / 0.226 / 0.388, three times noisier than
any ramp boundary and not resolvable.

Its **width is not a rate effect**. At the same leg length and the same interface
stiffness, the three repeats give dissipation 15.7 / 17.8 / 21.4 against work widths
of 19.9 / 42.4 / 58.9: a threefold range in width across a 1.4-fold range in
dissipation. No leg time, boundary or sub-leg count reaches that. The replicated
probe below is the only part of the design that does, and this is where it should
show up first, since the interface reference geometry is exactly what the probe
measures. Splitting the leg budget between the bound leg and the ramp is likewise
flat anywhere from 5 to 13 ns per direction, so it is not a knob worth turning.

#### The holds, and what they are for

There is now an equilibrium hold at every boundary the ramp crosses, including
the two where the ramp meets the bound legs, plus the unbound hold. All carry
`delta-lambda = 0` and pull rate 0, so they do **no work**: the stage works still
sum exactly to the work of the whole ramp, and `dG_unbind_1s` remains an
assumption-free cross-check on the staged sum.

They exist because the staged estimate is only unbiased if the system is at
equilibrium where the stages meet. The per-cycle bound equilibration went
1 ns -> 10 ns -> 20 ns for the same reason: it is what the Crooks analysis assumes
the cycle starts from.

**Hold lengths are measured, not assumed.** `dH/dlambda` is written during a hold,
so how long one needs is a question the data answers directly: its autocorrelation
time is 11-16 ps on the ramp and 39 ps at u = 1, the relaxation profiles are flat
after the first tenth, and the state at the end of a hold does not predict the next
stage's work (r = +0.06 over 49 cycles). The 1 ns holds were therefore 60-90 tau and
the mid-ramp ones are now **0.5 ns**.

The exception is the **bound/ramp handoff**. `holdfwd0` and `holdrev0` settle at 500
and 700 ps, about 40 tau, where the purely mid-ramp holds settle within 0-300 ps.
They cross a change of restraint regime rather than a lambda step, so they keep 1 ns,
and the bound leg's own internal hold is a handoff of the same kind and keeps 1 ns
too.

### Setup: the box is measured, not assumed

The FE legs need a bigger box than the classic protocol, because GROMACS checks every pull coordinate against `0.49 ×` the shortest box **vector** and the interface restraints separate as the partners come apart. That demand cannot be predicted from the minimised structure, so it used to be a constant (`-d 1.5`, then `-d 1.8`) measured once on 2KTF and applied to everything.

Setup now measures it per structure:

```
emin_vac.gro
  ├── probe pass    small box (-d = rvdw/2 + 0.5), N_PROBES = 5 INDEPENDENT
  │                 replicas, each a full NVT ladder + 20 ns NPT
  │                 the bound complex never needs the production box: its largest
  │                 checked pull pair is 1.18 nm against a 3.29 nm limit at -d 1.00
  ├── measure_box.py  running max solute atom-atom distance over ALL replicas
  │                 -> boxpad.gs  (L = D_max + 2 × 1.5 nm)
  └── production    box from boxpad.gs, solvate, minimise, ladder, 1 ns NPT
                    -> emin_solv.gro, which every cycle starts from
  every restraint and every anchor is measured on the POOLED last half of all
  five replicas: the production system is deliberately NOT equilibrated here,
  because each cycle runs its own ladder from emin_solv.gro with fresh
  velocities and that is what generates the bound-state ensemble
```

**Why five replicas.** Three repeats of one protocol scattered `dG_bind` by
14.4 kJ/mol between runs against a within-run bootstrap of 4.4. The reason is
structural rather than statistical: all 50 cycles of a run shared one probe, one
Boresch triad and one spring set, so resampling cycles holds every one of those
fixed and the reported interval is conditional on a setup draw that is itself
random. Replicating the probe is the only part of the design that attacks that
term. The ladder is re-run per replica rather than branching five copies off one
equilibrated state, since branching late would leave them sharing the history this
is meant to break.

Pooling the references needs two things that are each silently wrong if skipped: the
ligand triad is minimum-imaged before any angle is taken, because `trjconv -pbc
whole` can leave the two proteins in different images; and the three dihedrals are
averaged **circularly**, since +179 and -179 are two degrees apart and average to
180, where an arithmetic mean gives 0 and a reference pointing the other way.

#### Which interface pairs get a spring

A pair is a contact if it is inside 0.6 nm, and with the references now averaged
over the pooled replicas that test is applied to the **mean** distance. A mean
cannot see how much of the time a pair is actually in contact, and pooling five
replicas is what makes the difference visible: measured across three runs, 27-40%
of the springs that survive the mean cutoff sit outside it in more than a quarter
of the frames, and `r(mean, sd)` is only +0.27 to +0.32, so filtering on the mean
does not remove them. They hide at short mean distance and swing.

They are not free. A harmonic spring referenced to `r0 = <d>` costs `var(d)`, so

    <W_intro> = 0.5 * sum_k * mean_i(sd_i^2)

which does **not** depend on the number of springs: dropping the widest lowers the
bound leg's work and its fluctuation even though `k = sum_k/N` rises to compensate.
The widest 10% of pairs carry 33-48% of `sum(sd^2)`.

So a second criterion applies, `--sd-max`, defaulting to **0.15 nm**. On the three
runs it takes the final spring count down only 5-8% (322/272/239 to 295/257/227),
because most wide pairs also sit far out and the two filters overlap, while
removing 26-40% of the `sum(sd^2)` of the springs the cutoff would have kept
anyway. Pass `--sd-max 0` to keep every pair; the spread is reported either way,
and the value is recorded in `boresch_analytical.gs` beside `sum_k`, since it
selects which springs exist and two directories built at different values do not
have comparable bound legs.

It refuses to strip the interface bare: if the ceiling would leave fewer than 60
springs it is not applied at all and says so, because an interface pinned at a
handful of points is a worse failure than one pinned at some mobile ones.

`make_boresch.py` then takes its geometry from the production reference (`-f`) and its backbone RMSF from the probe trajectories (`--traj`, which takes the whole list of replicas); solute atom numbering is identical between the two systems, only the solvent count differs. On 2KTF the measurement returns `-d 1.844`, within 0.05 nm of the constant it replaces, but it will move on the next complex instead of staying put.

All pressure coupling is **C-rescale** (stochastic cell rescaling, Bernetti and Bussi 2020) at `tau_p` 2.0 ps, with V-rescale for temperature. Berendsen generates no strictly correct ensemble, which matters most for the per-cycle `npt_fe`: its output is the bound-state configuration `boundfwd` starts from, and Crooks assumes those initial states are drawn from the true equilibrium distribution.

The effort is deliberately concentrated where the uncertainty is. The unbinding/rebinding legs dominate the error and are given 52 ns per direction against the bound switch's 15. **Each stage's pull rate is tied to its own length**: `rate x stage_time = stage_span`, hence 2.31e-5, 1.54e-5, 2.12e-5, 3.46e-5 and 9.81e-5 nm/ps for stages A to E, which is simply each stage's own span over the same 5.2 ns. `delta-lambda` is written at `%.12e` for the same reason the pull rate is written at `%.10g`: the endpoint drifts by `nsteps` times whatever the last digit rounds away, so a format wide enough for a short leg silently stops being wide enough for a long one. Nothing sets a rate by hand: `make_boresch.py` reads each stage's own mdp and derives it, then records all of them in `boresch_analytical.gs` as `stage_rate_nm_ps`, because `integrate.py` turns the time integral of the pull force into work by multiplying by the rate and a stage integrated at another stage's rate is silently rescaled. The lambda spans are read back from the mdps for the same reason on the dhdl side.

The **5 ns unbound hold** (1 ns before 2026-08-11, then 5, 3, 6 and back to 5) exists because the rebinding leg is the one which starts from the Boresch-restrained separated state, and an under-equilibrated starting ensemble inflates the reverse width and breaks the forward/reverse pairing Crooks requires, which neither longer switching legs nor additional cycles can repair. Over 40 cycles of 2KTF the rebinding works were 2.4× wider than the unbinding works (σ 51.5 vs 21.6). Raising the hold to 5 ns did not fix that: the next run came back with `sf/sr` 0.32 against 0.42, i.e. moved the wrong way. That run also carried the dihedral sign fix, so the two changes are confounded and the hold is not convicted on it, but nothing supported 5 ns either. It is now 4 ns, and the measurement that matters is a different one: on the two-stage run it was fully plateaued after 800 ps, where the earlier 5 ns hold never settled at all. Whether a hold is long enough can be checked with `utils/fe_leg_efficiency.py`: the width ratio `sf/sr` should move toward 1.

`fe_leg_efficiency.py` takes `--leg unbind<L>` for each stage and `--leg unbind` for the ramp as a whole; the choices are generated from `fe_protocol.py`, so they follow the ramp. The stages are the real switching processes and are what the friction and cost model apply to; the whole ramp has no trace files of its own, so for it the tool reports the work distributions only.

### Why BAR is the FE headline and not the classic one

BAR (Bennett Acceptance Ratio) is the maximum-likelihood estimator for bidirectional work data and, unlike CGI, assumes nothing about the shape of either distribution. It is therefore the headline for `scores_fe.gs`. It is deliberately *not* the classic engine's score.

Every bidirectional estimator, BAR included, reads ΔG off the region where the forward and sign-aligned reverse works both have samples. The classic protocol has no such region: forward and reverse are separated by a median of 132 RT, with 0 of 46 structures in `bm_amber` overlapping at all and 1 of 2431 in `bm_ppb_amber`. With nothing in the crossing region BAR collapses toward the Jarzynski limit, dominated by the single most extreme work, and a synthetic benchmark at that dissipation puts its error at 3.97 kT against 1.56 for the average. The classic score is also a relative pull work calibrated against experiment downstream by regression rather than an absolute ΔG, so swapping in a tail-dominated estimator would trade a calibrated number for a worse one.

So the classic engine keeps the average and carries BAR as an appended column that mostly reads `nan BAR_NO_OVERLAP`. That suppression is the point: it puts the reason the classic score is not a free energy into the output rather than leaving it in prose.

**A `nan` in a BAR column never means the solver failed.** It means the leg had no overlap and the estimate was refused, with the reason in `Note`. BAR is suppressed rather than flagged because a flagged number still gets plotted and regressed by anything that does not read the flag, and on separated data the solver returns a confident finite value, with the wrong sign at the dissipation the unbinding leg currently runs at. The average and CGI are still reported there, because they are model extrapolations by construction and never claimed otherwise.

The solver lives in `utils/estimators.py` and is shared by both engines. It is implemented in-repo rather than calling pymbar because pymbar's `bar()` cannot be vectorised over bootstrap rows, which would have cost about 73 s per structure against 0.66 s; `tests/test_bar.py` cross-checks against pymbar when it is importable and agrees to ten decimal places.

### Choosing the Boresch anchors

The six Boresch coordinates fix the relative placement of two anchor triads but say nothing about the shape of either triad, and those edges are intramolecular. If the anchor groups are not themselves rigid the frame wanders, and eq.32, which assumes anchor points fixed in rigid bodies, stops describing what is actually restrained.

Anchors are therefore selected from **measured** backbone rigidity rather than from a static heuristic. Each protein is fitted onto itself frame by frame over the probe trajectory, which removes rigid-body motion and leaves flexibility; candidate groups are compact backbone clusters (0.70 nm about a seed CA, at least 18 atoms) kept only if their measured COM RMSF is below a ceiling, thinned to 0.5 nm COM separation, and ranked on `eps / (arm × conditioning)`. Selecting on measured COM RMSF rather than on size is the point: averaging suppresses only uncorrelated motion, as `1/√N`, so a loop that moves collectively averages to nothing however many atoms it has.

Two refinements matter enough to state:

**The chosen triad is rebuilt with disjoint membership.** Candidate groups are seed neighbourhoods, so a residue near two seeds belongs to both, and the 0.5 nm thinning constrains COM separation only. Shared atoms pull the two COMs together, which shortens the lever arm just as the frame error `eps/L` wants it long, and it leaves the six coordinates correlated in a way the factorised eq.32 does not model. Each residue is reassigned to its nearest of the three seeds and the arms, conditioning and RMSF are all re-tested on the split geometry.

**The rigidity ceiling escalates serially**, `EPS_LADDER = (0.045, 0.060, 0.080)` nm, first success wins. The ceiling is a physical requirement rather than a tuning knob, so the tightest one that yields an anchor set is the right one and later entries are concessions to a structure that cannot supply groups that quiet. The burial heuristic is reached only when every ceiling fails. Note the split starves the candidate pool, which is why the ladder exists: on 2KTF the ligand drops from 3 ranked triads to 1 surviving at 0.045.

The selection is then **validated**: the relative rotation of the two frames is recomputed over the trajectory and reported, with a warning above 8°. The predecessor heuristic, which chose single-residue N/CA/C triads by burial, let the partners rotate 22-52° with every Boresch coordinate satisfied; the measured path reaches 4.5-6.6° on 2KTF.

### Caveats

- **All free-energy results predating 2026-08-14 are void.** `dihedral_deg` returned the negated dihedral, so all three φ references were written to every leg mdp as mirror images. On 2KTF this put 3145 kJ/mol into `dH/dλ` at t=0 and drove θ_B toward the pull-frame singularity, where the `1/sin(θ_B)` lever tore anchor residues apart. `make_boresch.py` now reads its own pull block back through a zero-step grompp and aborts if GROMACS does not reproduce the references it was given.
- **Convergence of the unbinding leg is unresolved.** The forward and reverse work distributions of the whole ramp had zero overlap at 20 ns both before and after the dihedral sign fix, so the mirror-orientation strain was not the explanation. Splitting the ramp at u = 0.3 resolved the first stage on a real run (`dG_A` = 76.17 +- 3.89 kJ/mol, the first BAR value this channel has produced) but left the second blocked by 2.14 RT, so `dG_unbind` is still refused. The ramp is now split again at u = 0.5, where the measurement says the time is worth most. **Whether three stages resolve has not been measured.** The thing to watch is every `dG_unb<L>_bar` column, `dG_unbind_bar` against `dG_unbind_1s_bar` where the latter resolves at all, and the stage panels of `fe_works.png`.
- **The holds are an assumption, not a measurement.** The staged sum is only unbiased if the system is at equilibrium where the stages meet. On the two-stage run the arrival mismatch decayed with a time constant of about 1.19 ns, so a 1 ns hold was 0.84 tau and two thirds of the cycles were still drifting when it ended; two routes then sized the resulting bias anywhere between 0.9 and 24 kJ/mol. The holds are still 1 ns. Lengthening them and watching the answer stand still is the outstanding test.
- **The classic and FE protocols changed barostat on 2026-08-14.** Results from before are not strictly comparable with results from after.

## Heteroatom Support

### Crystal Waters

Crystal water molecules (HOH) from PDB structures are preserved and included as SOL in the simulation. They are placed at crystallographic positions before bulk solvation.

### Structural Ions

21 ion types are supported: ZN, CA, MG, CU, CU1, FE, FE2, NA, CL, MN, CO, NI, K, CD, SR, BA, CS, LI, HG, PB, and SD (sulfide from FeS clusters). Ions are automatically detected from PDB HETATM records and carried through the full pipeline. Ion-protein coordination is maintained via topology-level harmonic restraints using optimal distances from force field parameters and literature (e.g., Zn-S 0.232 nm, Zn-N 0.207 nm). Ions participate in the pulling restraints and are assigned to their respective protein chain. Metal clusters ([2Fe-2S], [4Fe-4S]) are modeled as individual ion atoms with intra-cluster distance restraints.

**Ion coordination protonation**: Residues coordinating metal ions are automatically assigned correct protonation states before topology generation. Cysteine thiolates (CYS → CYM) are deprotonated to expose the lone pair on sulfur. Histidine residues are set so the coordinating nitrogen is deprotonated (ND1 coordinates → HIE; NE2 coordinates → HID). Detection uses a 3.0 Å distance cutoff. This is supported for all force fields (AMBER19SB, CHARMM36m, GROMOS 54A8) with the appropriate residue naming conventions.

### Small Molecules (AMBER19SB only)

Ligands and cofactors are automatically extracted from PDB HETATM records and parametrized using the [Open Force Field](https://openforcefield.org/) (Sage 2.2.1):

1. **Bond order perception**: OpenBabel reads 3D coordinates and assigns bond orders. If kekulization fails (common for fused ring systems without explicit H), the RCSB Chemical Component Dictionary is used as fallback.
2. **Protonation**: Assigned at physiological pH (7.4) by OpenBabel
3. **Parametrization**: OpenFF Sage force field via Interchange → GROMACS topology
4. **Merging**: Ligand topology and coordinates are merged into the protein system

For best results with novel (non-PDB) ligands, provide input structures with explicit hydrogen coordinates. To skip ligand parametrization entirely, use `--no-ligand-param`.

### Non-Standard Amino Acids

Modified amino acids (e.g., TPO, SEP, PTR, TYS, MSE, HYP, MLY, CSO, TRQ) are automatically detected from HETATM records that contain backbone atoms (N, CA, C, O). Treatment depends on the force field.

#### AMBER19SB

GroScore parametrizes the NCAA with OpenFF while retaining AMBER19SB backbone parameters:

1. **Detection**: HETATM residues with backbone atoms are identified as modified amino acids
2. **Capped tripeptide**: An ACE-NCAA-NME fragment is built from the PDB coordinates for charge consistency
3. **Bond orders**: OpenBabel 3D perception with RCSB Chemical Component Dictionary fallback for complex ring systems
4. **Parametrization**: OpenFF Sage assigns charges and bonded parameters for the sidechain; backbone atoms retain AMBER19SB types and charges
5. **Force field injection**: Custom RTP, HDB, atom types, bonded parameters, and CMAP (from parent residue) are injected into a local force field copy

This is active with AMBER19SB force fields and ligand parametrization enabled (default). Use `--no-ligand-param` to disable.

#### GROMOS 54A8

GROMOS 54A8 ships with ~80 PTM NCAAs pre-parametrized in its residue topology files, so no OpenFF is required. pdb2gmx handles them natively after renaming:

| PDB CCD | GROMOS name | Notes |
|---------|-------------|-------|
| TPO     | T1P         | Phosphothreonine; P→PD, O1P→OE1, O2P→OE2, O3P→OE3 |
| SEP     | S1P         | Phosphoserine; same phosphate atom renames |
| PTR     | Y1P         | Phosphotyrosine; P→PT, O1P→OI1, O2P→OI2, O3P→OI3 |
| TYS     | YSU         | Sulfotyrosine; S→ST, O1S→OI1, O2S→OI2, O3S→OI3 |
| NLE     | LNO         | Norleucine |
| DAL     | DALA        | D-alanine |
| OCS     | CSE         | Cysteinesulfinic acid |
| CSO     | CSA         | S-hydroxycysteine |

For NCAAs not in the GROMOS RTP (no native parameters), GroScore falls back to parent residue replacement (same as the `--no-ligand-param` behavior for AMBER19SB).

## Fragment Handling

GroScore automatically handles complex protein structures with multiple chains and chain breaks:

- **Chain Break Detection** - Gaps in residue numbering within a chain are detected and marked with TER records
- **Small Gap Filling** - Gaps < 4 residues introduced by interface filtering are automatically filled to avoid introducing artificial chain breaks, while respecting TER positions (never merges different chains)
- **Minimum Fragment Size** - Fragments smaller than 5 residues are automatically extended by adding neighboring residues for improved stability
- **Isolated Cap Removal** - ACE/NME caps that lost their partners during interface filtering are removed to prevent orphaned caps
- **Fragment Merging** - Fragments from the same original PDB chain are merged into a single moleculetype for GROMACS
- **Terminal Capping** - Fragment termini are capped to provide neutral ends:
  - **AMBER19SB**: ACE/NME residues added explicitly via `cap_termini.py` before pdb2gmx
  - **CHARMM36m/GROMOS 54A8**: ACE residues (N-termini) added via `cap_termini.py`, COOH patches (C-termini) applied during pdb2gmx

This ensures proper topology generation even for structures with missing loops or multi-chain complexes, while maintaining chain boundaries and avoiding artificial chain breaks.

## Key SMD Pulling Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pull distance| 1 nm | Max protein-protein separation |
| Pull speed | 0.0002 nm·ps<sup>–1</sup> | Rate of distance increase, 5 ns per leg |
| Sum of pull force constants | 25000 kJ·mol<sup>–1</sup>·nm<sup>–2</sup> | Sum of pull force constants is the same for all complexes |
| Interface cutoff | 0.6 nm | Defines protein-protein interface |
| Elastic network range | 0.4-0.9 nm | Restraint distance bounds |
| Keep cutoff | 2.0 nm | Interface extraction radius |
| Minimum fragment size | 5 residues | Ensures stable fragments in cutout mode |
| Gap filling threshold | < 4 residues | Fills small gaps to avoid artificial chain breaks |
| Ion coordination cutoff | 0.3 nm | Detection radius for ion-ligand coordination |
| Ion coordination k | 10000 kJ·mol<sup>–1</sup>·nm<sup>–2</sup> | Harmonic restraint force constant for coordination bonds |

## File Formats

- `.gs` - GroScore data files (tab-separated, `#` for comments). Per-cycle results carry the work value plus the [rebinding QC](#rebinding-sanity-check-qc) RMSD; score files carry `Score`, `CI95`, `Cycles_Used` and the QC columns
- `.mdp` - GROMACS molecular dynamics parameter files
- `.gro` - GROMACS coordinate files
- `.xvg` - GROMACS output data (force curves)
- `.itp` - GROMACS topology include files (ligand parameters)
- `.sdf` - Structure-data files (ligand bond orders, for debugging)

## Project Structure

```
GroScore/
├── groscore.py          # Main orchestrator
├── job.run              # SLURM job template
├── forcefield/
│   ├── charmm36-jul2022.ff/  # CHARMM36m force field parameters
│   └── gromos54a8.ff/        # GROMOS 54A8 force field parameters
├── settings/
│   ├── amber19sb_opc/   # AMBER19SB/OPC parameter files
│   ├── amber19sb_opc3/  # AMBER19SB/OPC3 parameter files
│   ├── gromos54a8/      # GROMOS 54A8 parameter files
│   │   ├── emin_*.mdp   # Energy minimization
│   │   ├── nvt_*.mdp    # NVT equilibration phases
│   │   ├── npt*.mdp     # NPT equilibration
│   │   └── bind*.mdp    # SMD pulling parameters
│   └── charmm36/        # CHARMM36m parameter files
│       └── (same files)
└── utils/
    ├── renumber_pdb.py              # Assign sequential residue numbers, extract ligands/waters
    ├── fix_pdb.py                   # Fix missing atoms with PDBFixer
    ├── cap_termini.py               # Add ACE/NME terminal caps
    ├── parametrize_ncaa.py          # NCAA parametrization (OpenFF sidechain + AMBER backbone)
    ├── parametrize_ligand.py        # OpenFF small molecule parametrization
    ├── fix_ion_protonation.py       # Ion-coordinating CYS/HIS protonation states
    ├── merge_ligand.py              # Merge ligand topology into protein system
    ├── merge_crystal_waters.py      # Merge crystal waters as SOL
    ├── make_ion_restraints.py       # Ion coordination restraints
    ├── make_cluster_group.py        # PBC clustering index group
    ├── fix_topol_intermolecular.py  # Fix topology after solvation/genion
    ├── check_brokenloop.py          # Loop connectivity validation
    ├── check_entangledloops.py      # Topological knot detection
    ├── make_cutout.py               # Interface region extraction
    ├── make_disres_en.py            # Distance restraints & elastic network
    ├── integrate.py                 # Force curve integration
    ├── rebound_rmsd.py              # Rebinding QC (re-bound vs. bound backbone RMSD)
    └── local_runner.py              # --run-local: background job pool, one GPU per slot
```

## Troubleshooting

### Common Issues

**BROKEN status**: Protein loop connectivity failed validation. Check your input structure for missing residues or chain breaks.

**ENTANGLED status**: Topological knots detected. The protein structure may have threading artifacts that would invalidate pulling simulations.

**FAILED status**: Stage-0 setup or energy minimization did not complete, e.g. `emin_vac.gro` was not produced (grompp hit a topology/coordinate mismatch) or the entanglement check returned no result. Check the structure's SLURM output for the first GROMACS error. Any status other than `OK` excludes the structure from scoring.

**HIGH_RMSD flag**: The [rebinding QC](#rebinding-sanity-check-qc) found at least one cycle whose complex did not return to the bound pose. The score is still reported; inspect the flagged cycles (`RMSD_max_A` in the score files, per-cycle values in the third column of `results_<even>.gs`), add cycles with `-n <more> --restart` and check whether the score moves.

**RMSD reported as `nan`**: The measurement was not made: either the run predates the check, or `gmx trjconv`/`gmx rms` failed for that cycle. It never affects the score. Reproduce the failure with `python3 ../../utils/rebound_rmsd.py --ref npt_c<N>.gro --query bindrev_<2N>.gro -v` inside the structure directory; the reason is printed to stderr.

**Job failures**: Ensure GROMACS modules are loaded and paths are correctly set in your SLURM environment.

**Ligand parametrization fails**: If OpenBabel kekulization fails and RCSB download is unavailable, provide input PDB with explicit hydrogen coordinates for the ligand.

## Benchmark

The `benchmark/` directory is organized into one subdirectory per benchmark set, with shared plots in `benchmark/results/`:

- `haddock_benchmark/`: [HADDOCKING Protein-Protein Affinity Benchmark](https://github.com/haddocking/binding-affinity-benchmark) (46 structures), described below
- `ppb_benchmark/`: PPB-Affinity benchmark
- `hetatom_benchmark/`: heteroatom (molecular glue / PROTAC) benchmark
- `capri_benchmark/`: CAPRI Score_set benchmark

To run the HADDOCKING benchmark:

```bash
cd benchmark/haddock_benchmark
python setup_benchmark.py  # Downloads PDBs and creates sp.gs
python ../../groscore.py
```

## Citation

If you use GroScore in your research, please cite:

> Perthold, J. W.; Oostenbrink, C. GroScore: Accurate Scoring of Protein–Protein Binding Poses Using Explicit-Solvent Free-Energy Calculations. *J. Chem. Inf. Model.* **2019**, *59* (12), 5074–5085. https://doi.org/10.1021/acs.jcim.9b00687

For the improved method (Chapter 3 included as [theory/thesis_chapter_3.pdf](theory/thesis_chapter_3.pdf)), see:

> Perthold, J. W. New developments and critical views on binding free-energy calculations using molecular mechanics. *Doctoral Dissertation*, University of Natural Resources and Life Sciences, Vienna (BOKU), **2023**. [Library catalog](https://litsearch.boku.ac.at/primo-explore/fulldisplay?docid=BOK_alma2198734100003345&vid=BOK)

## Acknowledgements

J.W.P. has been a recipient of a DOC Fellowship of the Austrian Academy of Sciences (ÖAW) at the Institute for Molecular Modeling and Simulation at the University of Natural Resources and Life Sciences, Vienna (Grant No. 24987).

The computational results have been achieved using the Austrian Scientific Computing (ASC) infrastructure.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Author:** Jan Walther Perthold
**Email:** jan@ackergarten.at
