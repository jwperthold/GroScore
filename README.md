# GroScore

<p align="center">
  <img src="logo.png" alt="GroScore Logo" width="200">
</p>

**Computational Chemistry Toolkit for Protein-Protein Affinity Scoring with MD**

[![Python](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2026-orange.svg)](https://www.gromacs.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22686986.svg)](https://doi.org/10.5281/zenodo.22686986)

GroScore estimates binding affinities between protein pairs using short steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

---

## Features

- **Automated MD Pipeline** - Complete workflow from structure preparation to final scoring
- **SLURM Integration** - Efficient HPC execution via job arrays
- **Multiple Force Fields** - Support for AMBER19SB (all-atom), CHARMM36m (all-atom), and GROMOS 54A8 (united-atom)
- **Structural Ion Support** - Automatic handling of 21 ion types (ZN, CA, MG, CU, FE, MN, CO, NI, K, CD, SR, BA, etc.) with coordination restraints
- **Small Molecule Support** - OpenFF parametrization of ligands and cofactors (AMBER19SB only). Bond orders, charges and hydrogens come from the exact PDB Chemical Component Dictionary entry when its heavy atoms match the deposited ligand (local cache `forcefield/ccd_cache` first, then files.rcsb.org); otherwise OpenBabel perceives them and the result must pass composition, hydrogen, acid-state and geometry checks. A ligand in the simulated region that cannot be built, is built wrong or is covalently bonded stops the structure as `LIGAND_FAIL` in both engines; a glycan covalently bonded to an Asn, Ser or Thr is instead fused with its amino acid into one residue and simulated as part of the protein (up to 3 sugars by default)
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
- `-ff, --forcefield` - Force field: `amber19sb_opc3` (default), `amber19sb_opc`, `gromos54a8`, `gromos54a8_rf` (GROMOS 54A8 with its reaction field), or `charmm36`
- `--no-cutout` - Simulate the full protein structure (slower; the interface cutout is the default)
- `--no-ligand-param` - Run without small-molecule ligands: none is parametrized, and each is left out of the topology and named in the job log. With AMBER19SB, modified amino acids are then replaced by their parent residues, and covalently bonded glycans are not fused: they are left out and named like the other ligands. Without this option, a CHARMM36 or GROMOS 54A8 run (no ligand parametrization) stops every structure with a ligand in the simulated region as `LIGAND_FAIL`, with one line per ligand in `ligand_failed.gs` (`... not parametrised: force field <ff> has no ligand parametrisation; pass --no-ligand-param to run without ligands`)
- `--slurm` - SLURM template name from `slurm/` directory (default: `workstation`). Templates are plain `#SBATCH`-prefixed shell scripts; ship with `slurm/workstation.sh` (single workstation) and `slurm/vsc5.sh` (VSC-5 cluster). To target a different system, drop a new `<name>.sh` template into `slurm/` and pass `--slurm <name>`.
- `--run-local` - Run the jobs on this machine, spread over the local GPUs. Requires `--ngpus`. See [Running Without SLURM](#running-without-slurm-single-multi-gpu-workstation)
- `--ngpus N` - Number of GPUs to distribute local jobs over (mandatory with `--run-local`)
- `--jobs-per-gpu N` - Concurrent jobs per GPU in local mode (default: 8; `0` starts every job at once)
- `--threads-per-job N` - CPU threads per local job, i.e. `gmx mdrun -nt` (default: 1)
- `--restart` - Resubmit jobs (useful for continuing interrupted runs). With a larger `-n` it adds cycles. A restart continues each structure where it stopped: a finished cycle is never run or integrated again, and once a structure's first pull leg has run its restraint set is kept as it is. `run.gs` is not rewritten, so `-ff`, `--no-cutout` and `--no-ligand-param` keep the values of the first submission. To rebuild a structure with other settings or code, set it up again in a fresh run directory. In GroScore-FE it also submits an archive job (`job_fe.run --archive`) for a structure whose cycles are all done but which is still a directory, as after an archive that failed
- `--inject-job-run` - Inject fresh job.run into archived (.tar.gz) structures and submit them. An archive is submitted only with this flag
- `--rmsd-warn` - Threshold in Å for the [rebinding QC](#rebinding-sanity-check-qc) (default: 10.0). Flags structures; scores are unaffected

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

Re-run `python ../groscore.py -n <N>`, with the `-n` of the submission, periodically to check progress and collect results. `groscore.py` does not remember `-n`: it scores at most `-n` cycles per structure, so a re-run with the default 5 scores only the first 5.

### 5. Collect Results

Results are written to two output files, each listing every structure of `sp.gs` in file order:

| Output File | Method | Required Cycles |
|-------------|--------|-----------------|
| `scores_avg.gs` | Simple average of pull/push works | any (≥ 1) |
| `scores_cgi.gs` | Crooks Gaussian Intersection of forward/reverse work distributions | ≥ 20 |

Both files are always written. Where the CGI Score cannot be computed it is `nan`, with the reason in the last column, `Score_note`:

- `CGI_N_LT_20`: fewer than 20 cycles. The default `--numruns 5` always gives this; increase `-n` if you want CGI estimates.
- `CGI_NO_OVERLAP`: the pull and push work distributions do not overlap, the usual case on the classic protocol.
- `CGI_DEGENERATE`: one distribution has zero width.

Both files also carry `BAR`, `BAR_CI95`, `BAR_note` and `Score_note` as **appended** columns, 11 columns in all. BAR is a comparison column here; the classic Score is the average. On the classic protocol BAR reads `nan ... BAR_NO_OVERLAP` on essentially every structure, and that is the informative part: at the classic pull rate the forward and reverse works lie so far apart that no sampled work falls in the region a bidirectional estimator reads ΔG from. On the FE protocol the ramp is split into stages so that each stage's works can overlap, and the staged BAR sum is the headline (see [Caveats](#caveats) of the FE section for what it assumes). `Overlap_min_pct` in `scores_fe.gs` reads 0 on the rows where some BAR channel has no overlap.

#### Interpreting the score

- **Sign convention**: more negative score ⇄ tighter binding. Predicted pKd is monotonically *decreasing* in the score.
- **Units**: kJ·mol⁻¹. The score is the integrated pulling work along the unbinding/rebinding coordinate, averaged over cycles.
- **Convert to pKd**: use the linear fits provided per force field in the [Benchmark Data](#benchmark-data-haddocking-protein-protein-affinity-benchmark) section, e.g. for AMBER19SB/OPC3: `pKd ≈ -0.0176 × score + 3.4513`. These coefficients are calibrated on the HADDOCKING benchmark.
- **Uncertainty**: in `scores_avg.gs`, `CI95` is the half-width of the 95 % paired Student-t interval of the per-cycle scores (pull + push)/2, with n - 1 degrees of freedom; it is `nan` for a single cycle (`Score_note` `CI95_N_LT_2`). In `scores_cgi.gs` it is a bootstrap over whole cycles, scaled to the same t quantile.
- **Trustworthiness**: check the `RMSD_flag` column before using a score; see [Rebinding Sanity Check (QC)](#rebinding-sanity-check-qc) below.

## Running Without SLURM (Single Multi-GPU Workstation)

GPU cloud providers typically rent out one fat machine (4 or 8 GPUs in a single box) with no scheduler installed. `--run-local` replaces the SLURM job array with a background runner that starts the jobs itself and pins each of them to one GPU:

```bash
python ../groscore.py -n 5 --run-local --ngpus 8
```

Every job gets `gmx mdrun -nt 1 -gpu_id <n> -pin off`: one CPU thread, one GPU. That is deliberate: GroScore systems are small enough to run essentially GPU-resident (nonbonded and PME offloaded), so a job's CPU thread mostly feeds the device, and 8 single-threaded jobs on 8 GPUs beat one 8-threaded job on one GPU. All bonds are constrained, which puts the update and LINCS on the CPU thread, so `--threads-per-job 2` may pay on a workstation. Measure before assuming.

**How work is distributed.** `--ngpus` is mandatory and defines the round-robin: job 1 → GPU 0, job 2 → GPU 1, …, job 9 → GPU 0 again. By default 8 jobs share each GPU (`--jobs-per-gpu 8`), as a single cutout-sized system leaves the GPU idle during CPU-side work, so stacking jobs raises *aggregate* throughput well past what one job per device achieves. Anything beyond `--ngpus × --jobs-per-gpu` waits in a queue and starts as slots free up, so a 500-structure screen does not try to open 500 GROMACS processes at once. Jobs are handed out dynamically, so a slow structure cannot leave its GPU idle at the end of the run.

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

A cycle's work only describes the intended binding event if the push leg actually put the complex back together. If the partners re-associate in a different pose, or drift apart and stay apart, the integrated force curve still yields a number, and that number describes a different event. Every cycle of both engines is therefore checked automatically; no extra command is needed.

`utils/rebound_rmsd.py` measures the **backbone RMSD between the bound state the cycle was equilibrated in and the state the rebinding leg ended in**:

| Engine | Reference (bound) | Query (re-bound) | Stored in |
|--------|-------------------|------------------|-----------|
| `groscore.py` (classic) | `npt_c<N>.gro` | `bindrev_<2N>.gro` | 5th field of `results.d/<ID>_c<N>.gs` (row `ID N pull push RMSD`, `NaN` where missing); legacy run directories: 3rd column of `results_<2N>.gs` |
| `groscore_fe.py` (FE) | `npt_c<N>.gro` | `bindrevA_<N>.gro` | last column of `results_fe.d/<id>_c<N>.gs` |

### Reading the result

`groscore.py` adds three columns, `RMSD_mean_A`, `RMSD_max_A` and `RMSD_flag`, to `scores_avg.gs` / `scores_cgi.gs` (and to `scores_*_c<N>.gs`, which score each structure's first N cycles), summarizing exactly the cycles that entered the score. The BAR and `Score_note` columns follow them:

```
# Structure_ID  Score  CI95  Cycles_Used  RMSD_mean_A  RMSD_max_A  RMSD_flag  BAR  BAR_CI95  BAR_note  Score_note
1A2K   -117.9   2.5   5   1.75   1.98   OK         nan   nan   BAR_NO_OVERLAP   nan
2OOB    -92.4   2.1   5  11.45  18.70   HIGH_RMSD  nan   nan   BAR_NO_OVERLAP   nan
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

The console stays short whatever the screen size: one aggregate line, then the ten worst structures (five bad cycles each) and a count of the rest. `grep HIGH_RMSD scores_avg.gs` gives the complete list. Individual per-cycle values are printed once each into the structure's own SLURM log, whereas the summary keeps the aggregate.

`groscore_fe.py` behaves the same way, with `RMSD_mean_A` / `RMSD_max_A` columns and a `HIGH_RMSD` entry in the `Note` column of `scores_fe.gs`. There the query is the end of the reverse ramp. No leg starts from the rebound state: the reverse bound legs branch from the end of `holdfwd0`. A high RMSD marks a reverse ramp that ended away from the bound pose, i.e. a dissipative reverse path. It does not set the start state of any later leg. A second per-structure flag, `LOW_RECOVERY`, marks a structure whose median contact recovery over its scored cycles is below 0.85, where recovery is the fraction of restrained interface pairs back within 0.15 nm of their reference distance at the end of the reverse ramp (`utils/interface_qc.py`). Neither flag changes a free energy.

### Interpretation

| RMSD | Meaning |
|------|---------|
| 1–5 Å | Normal. Thermal fluctuation plus whatever the interface relaxed into during the cycle. |
| 5–10 Å | Worth a look. Partial rebinding, a shifted interface, or a flexible loop that did not recover. |
| > 10 Å | `HIGH_RMSD`. The partners did not return to the original pose. |
| `nan` | Not measured: a legacy run directory without the check, a failed measurement, or (classic) a cycle whose legs did not finish. That cycle has no work values and is not scored. A missing RMSD never changes a score. |

The threshold is `--rmsd-warn` (default 10.0 Å) on both engines. **Scores and free energies are always computed and reported**: the check runs alongside the pipeline and leaves every simulation, every cycle and every number in place. It marks which results warrant distrust: with a handful of flagged cycles, add cycles with `-n <more> --restart` and see whether the structure's score is dominated by them. `job.run` and `job_fe.run` take the new total from `GROSCORE_NUMCYCLES`, which the orchestrators set from `-n`. Pass the same `-n` to every later `groscore.py` invocation; `groscore_fe.py` remembers it in `run_config.gs`. Add `--inject-job-run` for structures whose cycles are all done: such a structure is archived, and an archive is submitted only with that flag.

A single measurement can also be taken by hand from inside a structure directory (`<project>/<structure_id>/`). Both frames are reassembled in space first, and `-v` reports the longest link of that assembly; a link far below half the box means no residue was placed at a wrong periodic image:

```bash
python3 ../../utils/rebound_rmsd.py --ref npt_c3.gro --query bindrev_6.gro -v
# 1.9750
# rebound_rmsd: [assembled; longest link ref 0.45 nm, query 0.47 nm]
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
| **AMBER19SB OPC3** (default) | All-atom | OPC3 (3-point) | all-bonds | 1.0 nm | ACE/NME  |
| **AMBER19SB OPC** | All-atom | OPC (4-point) | all-bonds | 1.0 nm | ACE/NME |
| **CHARMM36m** | All-atom | TIP3P | all-bonds | 1.2 nm | ACE/COOH |
| **GROMOS 54A8** | United-atom | SPC | all-bonds | 1.4 nm | ACE/COOH |
| **GROMOS 54A8 RF** | United-atom | SPC | all-bonds | 1.4 nm | ACE/COOH |

Under AMBER19SB, ions kept from the crystal take the Li/Merz parameters of the run's water model (the FeS sulfide excepted, which the runners type as a sulfur); CHARMM36m and GROMOS 54A8 use their own ion parameters. Under AMBER19SB OPC3 that needs a correction: GROMACS 2026's `amber19sb.ff/aminoacids.rtp` types every ion residue with its OPC set (`Mg2+_opc`, `Zn2+_opc`, ...) whatever water model `pdb2gmx` is given, so both runners give these ions their OPC3 twins (`Mg2+_opc3`, ...) right after `pdb2gmx` (`utils/retype_ions.py`). Setups made before this correction (2026-10-25) ran their crystal ions with the OPC parameters, and keep them when a run directory is restarted: a structure's cycles all run on one topology, and a setup resumed in its probe pass is retyped before its production box is built.

PTM parameters for GROMOS 54A8 are taken from the Vienna-PTM parameter set, https://github.com/viennaptm/ptm_parameters (Petrov et al., *PLoS Comput. Biol.* **2013**, *9*, e1003154, https://doi.org/10.1371/journal.pcbi.1003154; Margreitter et al., *J. Comput. Chem.* **2017**, *38*, 714, https://doi.org/10.1002/jcc.24733).

All force fields use:
- **Electrostatics**: PME (Particle Mesh Ewald) for long-range electrostatic interactions, except GROMOS 54A8 RF, which applies the reaction field GROMOS 54A8 was parametrized with (`epsilon_rf` 61, 1.4 nm cutoff sphere)
- **Constraints**: all-bonds, with `mass-repartition-factor = 3`, in every force field: the recipe for 4 fs with heavy hydrogens. The update and the constraints run on the CPU, and a blow-up shows up as a LINCS warning that names the atom
- **Heavy hydrogens**: `mass-repartition-factor = 3` for stable 4 fs timesteps
- **Timestep**: 4 fs (`dt = 0.004` ps) for all production stages
- **SMD pulling per leg**: 1.25 × 10⁶ steps × 4 fs = 5 ns; one cycle = pull + push = 10 ns of SMD plus 300 ps of NVT/NPT equilibration (5 × 20 ps NVT and 100 ps NPT before the pull, 100 ps NPT before the push)

**Terminal Capping Details:**
- **AMBER19SB**: Uses ACE (N-acetyl) and NME (N-methylamide) caps added as explicit residues via PDBFixer before pdb2gmx processing. This provides proper neutral termini for fragment ends. The caps are then minimized in vacuum with every other atom held fixed. A residue that amber14 has no template for (a non-standard amino acid, a fused glycan residue, an ion-binding Cys without hydrogens in a cutout) gets a generic template from its own atoms and bonds, so such structures are minimized like any other; if a minimization still fails, `cap_termini.py` warns, names the residue and keeps the caps where PDBFixer placed them.
- **CHARMM36m**: Uses ACE (N-acetyl) caps at N-termini (explicit residues) and COOH patches at C-termini for improved stability.
- **GROMOS 54A8**: Uses ACE caps at N-termini (explicit residues) and COOH patches at C-termini.

The CHARMM36m force field parameters (from [MacKerell lab](https://mackerell.umaryland.edu/charmm_ff.shtml)) are included in `forcefield/charmm36-jul2022.ff/`. The GROMOS 54A8 force field parameters (from [Oostenbrink group](https://boku.ac.at/en/nwnr/mmsi/research/force-field-development)) are included in `forcefield/gromos54a8.ff/`, which both GROMOS 54A8 settings trees use.

## Simulation Pipeline

```
Stage 0: Preparation
├── Glycan fusion (AMBER19SB with ligand parametrization: each covalently bonded glycan + its Asn/Ser/Thr -> one NCAA residue)
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
└── 5-phase NVT + NPT → npt_init_cluster.gro   (classic engine; frame assembled by assemble_frame.py)

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

Each cycle draws fresh velocities from a Maxwell-Boltzmann distribution at `gen_temp = 62` K in `nvt_1.mdp`, with `gen_seed = -1`, i.e. a fresh seed from the wall clock each time a cycle starts, and heats to `ref_t = 310` K over the five-stage NVT ladder; every later leg continues from those velocities. This is deliberate, as independent cycles must sample independent trajectories, but it does mean that scores from a re-submitted run will not be bitwise-identical to the original. The CI95 column in `scores_avg.gs` quantifies the resulting between-cycle variance.

### Throughput

On a single GPU (consumer-grade RTX-class), expect roughly **10 GPU-hours per structure for 5 cycles** (including all equilibration legs and 5 × 10 ns of SMD). Cost scales linearly with cycle count and roughly linearly with system size; the interface-cutout mode (default) keeps system size near-constant across most complexes, so per-structure walltimes are tightly clustered. The benchmark directory contains `compute_walltime.py`, which extracts realised walltimes and PFLOP counts from completed SLURM logs.

## Absolute Binding Free Energies (GroScore-FE)

> **Output of one FE protocol is not comparable with output of another.** A run
> directory holds one protocol (see [Caveats](#caveats)).

`groscore_fe.py`, together with `job_fe.run`, computes an absolute binding free
energy. The classic engine's score carries the free-energy contribution of the
empirical interface restraints, which biases it; here that contribution is
accounted for explicitly.

Over the unbinding leg the atom-atom interface restraints are switched off while a
set of Boresch orientational restraints (one distance, two angles and three
dihedrals, defined on backbone center-of-mass anchor groups) is switched on. The
standard-state free energy of the Boresch restraint has a closed form (Boresch et
al. 2003), hence its contribution in the separated state is obtained analytically.
Every leg is run in both directions, and the works are combined with BAR, with the
average and CGI estimators reported alongside as a convergence check.

```
dG_bind = -( dG_intro + dG_unbind + dG_release )
   dG_intro     interface restraints introduced in the bound state   (dhdl)
   dG_unbind    interface to Boresch handoff, separation to 1.0 nm   (pull + dhdl)
   dG_release   analytical Boresch standard-state term               (closed form)
```

Since GROMACS provides no lambda-dependent pull reference, the switching work is
captured in two channels which add without double-counting: the pull force, i.e.
mechanical separation via the moving reference, and dH/dlambda, i.e. force-constant
switching.

Invocation follows the classic engine, with the same inputs and directory layout:

```bash
python3 /path/to/GroScore/groscore_fe.py -ff amber19sb_opc3
```

`-n` is the total number of cycles per structure, 50 by default. At the first
invocation `groscore_fe.py` records `-n`, `-ff`, `--sum-k`, `--sd-max`, `--temp`,
`--cutout`/`--no-cutout` and `--ligand-param`/`--no-ligand-param` in
`run_config.gs`, and every later invocation in the directory takes them from there;
a value given that differs from the recorded one is used, recorded and warned about,
except a new `-n`, which is recorded without a warning. Setup and cycles are separate jobs: one setup job per
structure (`job_fe.run --setup`) and an array of its missing cycles
(`job_fe.run --cycle N`) that waits for it. `--array-throttle N` caps the concurrent
cycle tasks per structure, `--sequential` runs setup and all cycles in one job per
structure, and `--n-boot-bar` sets the BAR bootstrap rows (default 5000).

The task that finishes a structure's last cycle archives the structure directory as
`<ID>.tar.gz`. The archive is tested with `gzip -t` before the directory is removed,
and the log says `Archived <ID> to <ID>.tar.gz (N bytes, gzip test passed);
directory removed.` The same line, with time and host, goes to `archive.log` in the
run directory, the place to look under `--run-local`, whose job logs live inside the
archived directory. A structure whose cycles are all done but which is still a
directory, because its archive failed (a full disk, say), gets an archive-only job
from `--restart` (with `--sequential`, its whole job, which archives at its end).
By hand, `GROSCORE_NUMCYCLES=<-n> ./job.run --archive` inside the structure
directory does the same; without the variable it takes the run's `-n` from
`run_config.gs`. Every task prints a `Task start:` and a `Task end:` line with the
time, the host, its SLURM job and the GPUs `nvidia-smi` lists, so a slow node or a
lost GPU can be found from the log alone.

Results are written to `scores_fe.gs`: dG_bind in kJ/mol and as pKD under each of
the three estimators, the three cycle components, one column pair per leg, and two
per-row diagnostics. `Overlap_mean_pct` and `Overlap_min_pct` give the forward and
reverse work overlap across every BAR channel, as a percentage of 2n; sorting on
the minimum identifies the rows whose BAR values warrant caution, since a single
channel without overlap withholds the whole estimate. `RMSD_mean_A` and
`RMSD_max_A` carry the [rebinding sanity check](#rebinding-sanity-check-qc).
The `Note` column carries `HIGH_RMSD`, `LOW_RECOVERY`, `NO_SCORE` (rows, but no
finite score) and the BAR notes of withheld channels. A structure without a score
shows its stage-0 status there (`LIGAND_FAIL`, `PROBE_FAIL`, ...), or `PENDING`
(no cycle or no analytical term yet), `PROTOCOL_MISMATCH` (its directory was set
up under another protocol) or `MIXED_LAYOUT` (its cycles carry different row
layouts). With fewer than 3 cycles the overlap columns are `nan`. The report also
prints, per bound sub-leg, the mean dissipation `(<W_f> + <W_r>)/2` paired by cycle,
flagged `NEG_DISS` from 10 cycles on where it lies below zero by more than the
two-sided 95 % Student-t quantile of its standard error. It is printed only and
changes no number.

The protocol is defined in one place, `utils/fe_protocol.py`. The leg mdps, the
pull blocks and per-stage rates, the leg sequence, the result-row layout and the
leg diagnostic are all derived from the table at the top of that file, which is
therefore the only place to edit when the cycle changes:

```bash
python3 utils/fe_protocol.py        # the current cycle, leg by leg
python3 utils/make_fe_mdps.py       # regenerate every force field's mdps from it
```

The reasoning behind the cycle, i.e. the measurements it rests on, is kept in the
comments of `fe_protocol.py` and `make_boresch.py`, alongside the code it
constrains, and in the commit history.

### Caveats

- **`make_boresch.py` checks the restraints it writes.** It reads its own pull
  block back through a zero-step grompp and stops the structure with
  `BORESCH_FAIL` on any discrepancy.
- **The electrostatics must match the force field.** `settings/gromos54a8` uses
  PME, whereas `settings/gromos54a8_rf` applies the reaction field GROMOS 54A8 was
  parametrized with (`epsilon_rf` 61, 1.4 nm cutoff sphere). The tree is selected
  with `-ff`.
- **The reported interval is a bootstrap over the cycles of one run.** It covers
  the cycle-to-cycle spread within a single setup, whose probe equilibration,
  anchor triad and interface restraint set are drawn once and then held fixed for
  every cycle of that run.
- **The staged BAR sum assumes the holds between stages equilibrate the
  interface.** Each stage is its own Crooks pair only if both of its runs start
  from equilibrium, which at an internal boundary means the hold must have
  equilibrated the interface. The holds settle dH/dlambda but do not equilibrate
  the interface (`fe_protocol.py`, HOLD LENGTHS), so the staged BAR and CGI sums
  can carry a bias their intervals do not show. `dG_unbind_1s`, the whole ramp in
  one shot, does not depend on the holds, and the gap between it and the staged
  value shows the size of that bias. The `dGbind_bar` column of `scores_fe.gs` is
  the staged sum.
- **A run directory holds one protocol.** `groscore_fe.py` records the protocol id
  of `utils/fe_protocol.py` (`PROTOCOL_ID`) in `run_config.gs` at the first
  submission into a directory, and `job_fe.run` writes it to each structure's
  `protocol_id.gs` when its setup completes. A cycle never runs on a setup of
  another protocol. `groscore_fe.py` refuses to submit into a directory that
  records another id, or records none although it already holds work (a run
  directory set up by an older FE protocol): start a fresh run directory for the
  new protocol, holding only `sp.gs` and the structure directories with their
  `input.pdb`. A refused invocation still scores what is there, with a warning,
  except structures whose setup belongs to another protocol (`PROTOCOL_MISMATCH`),
  and then exits 1. A structure held back from submission (another
  protocol, result rows but no setup, an archive without `--inject-job-run`, a
  setup whose last stage-0 status is `LIGAND_FAIL`, `BROKEN` or `ENTANGLED`) is
  left untouched and listed in a `NOT SUBMITTED` block at the end of the
  submission, with what to do. Such a setup would only reach the same verdict
  again (a `LIGAND_FAIL` counts only when `ligand_failed.gs` gives a verdict, not a
  crashed or killed parametrisation, which is submitted again); to set one up again
  on purpose, append the line `<ID> SETUP_RUNNING` to `results_0.gs`.

## Heteroatom Support

### Crystal Waters

Crystal water molecules (HOH) from PDB structures are preserved and included as SOL in the simulation. They are placed at crystallographic positions before bulk solvation.

### Structural Ions

21 ion types are supported: ZN, CA, MG, CU, CU1, FE, FE2, NA, CL, MN, CO, NI, K, CD, SR, BA, CS, LI, HG, PB, and SD (sulfide from FeS clusters). Ions are automatically detected from PDB HETATM records and carried through the full pipeline. Ion-protein coordination is maintained via topology-level harmonic restraints using optimal distances from force field parameters and literature (e.g., Zn-S 0.232 nm, Zn-N 0.207 nm). Ions participate in the pulling restraints and are assigned to their respective protein chain. Metal clusters ([2Fe-2S], [4Fe-4S]) are modeled as individual ion atoms with intra-cluster distance restraints.

**Ion coordination protonation**: Residues coordinating metal ions are automatically assigned correct protonation states before topology generation. Cysteine thiolates (CYS → CYM) are deprotonated to expose the lone pair on sulfur. Histidine residues are set so the coordinating nitrogen is deprotonated (ND1 coordinates → HIE; NE2 coordinates → HID). Detection uses a 3.0 Å distance cutoff. This is supported for all force fields (AMBER19SB, CHARMM36m, GROMOS 54A8) with the appropriate residue naming conventions.

### Small Molecules (AMBER19SB only)

Ligands and cofactors are automatically extracted from PDB HETATM records and parametrized using the [Open Force Field](https://openforcefield.org/) (Sage 2.2.1):

1. **Bond orders, formal charges and hydrogens** come from the PDB Chemical Component Dictionary (CCD) entry of the residue name when its heavy atoms match the deposited ligand, by atom name or by bond graph. The entry is read from `forcefield/ccd_cache` first, then from files.rcsb.org, and the itp header records which one was used. A CCD build is checked for composition and acid states only. Otherwise OpenBabel perceives the bond orders from the 3D coordinates, and the molecule must pass composition, hydrogen, acid-state and geometry checks. A lone halogen is simulated as the force field's own halide ion.
2. **Protonation**: OpenBabel's pH 7.4 model, plus explicit acid rules for carboxylic acids (-1), phosphates and phosphonates, sulfates and sulfonates
3. **Parametrization**: OpenFF Sage force field via Interchange → GROMACS topology. Partial charges are AM1-BCC on the deposited conformer first, then on RDKit conformers with seeds 1 to 5 within a 3600 s retry cap; the itp header records the conformer used
4. **Merging**: Ligand topology and coordinates are merged into the protein system

A ligand more than 0.5 nm from every atom of the simulated protein (0.61 nm, measured on its heavy atoms, for a ligand without parameters) lies outside the simulated region: it is not merged and only named in the job log. A ligand inside the region that cannot be built, is built as another molecule, or is covalently bonded to the protein or to another ligand stops the structure as `LIGAND_FAIL` in both engines, and `ligand_failed.gs` in the structure directory names the instance (`RESNAME CHAIN RESNUM`) and the reason. Examples are AlF3, which has no parameters (openff-2.2.1 has no aluminium term, and the reason says so), and the REA of 1RLB, deposited without the O2 of its CCD entry (most likely retinol filed under the code of retinoic acid), which OpenBabel then builds as another molecule. Crystallisation additives that are part of the deposited structure (sulfate, phosphate, glycols, PEG fragments) are ligands like any other. Under CHARMM36 and GROMOS 54A8, which have no ligand parametrization, every ligand in the simulated region stops the structure the same way. `--no-ligand-param` is the deliberate way to run without ligands.

**Glycans.** With ligand parametrization on (AMBER19SB), `utils/fuse_glycans.py` fuses a glycan covalently bonded to an Asn (ND2), Ser (OG) or Thr (OG1) with its amino acid into one residue before the structure is renumbered: `NG<k>` on Asn, `SG<k>` on Ser and `TG<k>` on Thr, one name per distinct chemistry. The residue is parametrized as a non-standard amino acid (see [Non-Standard Amino Acids](#non-standard-amino-acids)) and simulated as part of the protein. `fused.pdb` in the structure directory is the input with the fusion applied, and `glycans.json` holds the chemistry of each fused residue. The job log has one line per glycan, `fused NG1: ASN A 350 + NAG A 544 (ND2-C1 1.45 A)` or `NOT fused ...` with the reason. A glycan is fused when every member is a saccharide by its CCD entry, its sugars form a tree that hangs by one bond on an amino acid with a peptide neighbor on both sides, each sugar bonds its parent through its anomeric carbon to an oxygen, its deposited heavy atoms are those of its CCD entry (less the anomeric leaving oxygen, which is dropped when it is deposited) and their names fit the entry's bonds, and no further sugar is linked to it by a deposited bond that could not be joined. A deposited glycosidic link longer than the covalent rule, up to 2.5 A, is joined: the free anomeric carbon of a sugar is bonded to the closest hydroxyl oxygen of another sugar or to the Asn ND2, Ser OG or Thr OG1 in reach, and the log says so (`joined stretched link NAG K 1 O4-NAG K 2 C1 1.85 A`). The bond gets the normal parameters, and the energy minimization pulls it to bond length. Such a link stays unjoined, and the glycan a ligand, when it is ambiguous: the sugar's leaving oxygen is deposited as an atom of its own, two acceptors are about equally close, or two sugars reach for the same acceptor. A sugar on the OD1 of an Asn sits on a flipped amide: OD1 and ND2 are swapped and the glycan is fused. A glycan written as ATOM records is handled as a HETATM one. There is no limit on the number of sugars by default (26 at most fit the atom names of one residue). Set `GROSCORE_GLYCAN_MAX_SUGARS` in the shell that submits the jobs to cap it: unset or empty means no cap, any other value must be a whole number of at least 1 written in digits only (no sign, leading zero or space), and anything else stops the structure as `FAILED`.

A glycan that is not fused stays a ligand. In the simulated region it then stops the structure as `LIGAND_FAIL`, and the line of each of its sugars in `ligand_failed.gs` ends in `glycan not fused: <reason>`, taken from `glycan_unfused.gs` in the structure directory. This happens when the glycan does not meet the conditions above, for example when it has more sugars than a cap you set, when a member is no saccharide or when its deposited atoms differ from the CCD entry, and when its fused residue fails to parametrize (the setup then runs again without that fusion). Glycans are not fused under CHARMM36, GROMOS 54A8 or `--no-ligand-param`; there they are ligands like any other (`LIGAND_FAIL` under CHARMM36 and GROMOS 54A8, left out and named under `--no-ligand-param`). The sugar atoms of a fused glycan are protein atoms in both engines, so they count as interface atoms: they can carry interface restraints and enter the score. A deposited anomer or ring center that contradicts the sugar's CCD code is logged, not refused.

For nodes without network, prefetch the CCD entries of a project on a login node with `python3 utils/ccd_cache.py myproject/*/input.pdb`, run from the GroScore directory. It writes into `forcefield/ccd_cache`; with `-o <dir>` it writes elsewhere, and that directory is then listed in `GROSCORE_CCD_CACHE` (colon-separated). The prefetch also fetches ASN, SER and THR, the parents of a fused glycan, whenever an input has a HETATM residue other than water and ions. `forcefield/ccd_cache` carries the entries of the HADDOCK benchmark ligands and of ASN, SER and THR. A cached entry that is not complete (an error page, a file cut short), or that builds as a radical, is checked against the one at files.rcsb.org and replaced by it when that copy is complete and closed-shell; the itp header names the damaged file. For best results with novel (non-PDB) ligands, provide input structures with explicit hydrogen coordinates.

### Non-Standard Amino Acids

Modified amino acids (e.g., TPO, SEP, PTR, TYS, MSE, HYP, MLY, CSO, TRQ) are automatically detected from HETATM records that contain backbone atoms (N, CA, C, O). Treatment depends on the force field.

#### AMBER19SB

GroScore parametrizes the NCAA with OpenFF while retaining AMBER19SB backbone parameters:

1. **Detection**: HETATM residues with backbone atoms are identified as modified amino acids
2. **Capped tripeptide**: An ACE-NCAA-NME fragment is built from the PDB coordinates for charge consistency
3. **Bond orders**: OpenBabel 3D perception, with the residue's Chemical Component Dictionary entry as template where that fails (read from `forcefield/ccd_cache` first, then from files.rcsb.org, so a node without network builds it too)
4. **Protonation**: the ligand pH 7.4 rules, OpenBabel's pH model and the explicit acid rules, so a phosphoserine, phosphothreonine or phosphotyrosine (SEP, TPO, PTR) is -2, an amine side chain (ORN, DAB, DPP) +1 and a carboxylate -1
5. **Parametrization**: OpenFF Sage assigns charges and bonded parameters for the sidechain; backbone atoms retain AMBER19SB types and charges, and the CB hydrogens take AMBER19SB's types by CB's neighbours (H1 next to O, N or S, HP next to an amine N as in LYS). The charges written to the RTP sum exactly to the formal charge of the residue (the rounding residual goes on CA)
6. **Force field injection**: Custom RTP, HDB, atom types, bonded parameters, and CMAP (from the parent residue its CCD entry names, read from `forcefield/ccd_cache` first, or ALA's when the entry names none, as for D- and beta-amino acids) are injected into a local force field copy. An NCAA whose CCD entry cannot be read is handled as one that cannot be parametrized: PDBFixer puts in its parent residue when PDBFixer knows the NCAA or files.rcsb.org names the parent, and otherwise pdb2gmx stops the structure

This is active with AMBER19SB force fields and ligand parametrization enabled (default). Use `--no-ligand-param` to disable.

**Glycans.** A fused glycan residue (`NG<k>`, `SG<k>` or `TG<k>`, see [Small Molecules](#small-molecules-amber19sb-only)) takes the same path, with these differences:

- Its bond orders, formal charges and hydrogen counts come from `glycans.json`, never from perception or from its name, which is not looked up in the CCD (NG1 is itself the CCD code of another molecule). Its hydrogens are named after protonation.
- The ligand pH 7.4 rules apply, so a sialic acid carboxylate is -1. Partial charges are AM1-BCC on the deposited conformer with seeded retries, without a time limit by default. One GlcNAc takes about 1 to 2.5 minutes, and the cost grows about with the cube of the residue size: a 9-sugar Fc glycan takes about 2 hours of the setup. `GROSCORE_GLYCAN_AM1BCC_WALL_S` (in seconds, for all attempts together) sets a limit, after which the glycan goes back to the ligands (see [Troubleshooting](#troubleshooting)).
- The CMAP is the parent amino acid's, and one charge set serves every instance of one chemistry.
- A fused residue that fails to parametrize sends its glycan back to the ligands: `glycan_failed.gs` in the structure directory names the residue and the reason.

#### GROMOS 54A8

The GROMOS 54A8 tree in `forcefield/` carries ~80 PTM NCAAs from the [Vienna-PTM parameter set](https://github.com/viennaptm/ptm_parameters) (references under [Force Fields](#force-fields)) pre-parametrized in its residue topology files, so no OpenFF is required. pdb2gmx handles them natively after renaming:

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
- **Small Gap Filling** - Gaps < 4 residues introduced by interface filtering are automatically filled to avoid introducing artificial chain breaks, while respecting TER positions (chains are kept separate)
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

- `.gs` - GroScore data files (whitespace-separated, `#` for comments)
  - Classic per-cycle results: one file per cycle, `results.d/<ID>_c<N>.gs`, holding `ID N pull push RMSD` (`NaN` where missing; space-separated). The RMSD is the [rebinding QC](#rebinding-sanity-check-qc) value. A legacy run directory holds `results_<n>.gs` instead, one work per line, and is still scored
  - FE per-cycle rows: `results_fe.d/<id>_c<N>.gs`, 43 fields
  - Classic score files carry `Score`, `CI95`, `Cycles_Used`, the QC columns, `BAR`, `BAR_CI95`, `BAR_note` and `Score_note`
- `.mdp` - GROMACS molecular dynamics parameter files
- `.gro` - GROMACS coordinate files
- `.xvg` - GROMACS output data (force curves)
- `.json` - `glycans.json`, the chemistry of each fused glycan residue (heavy atoms, formal charges, hydrogen counts, bond orders), written into the structure directory by `fuse_glycans.py`
- `.itp` - GROMACS topology include files (ligand parameters)
- `.sdf` - Structure-data files (ligand bond orders, for debugging)

## Project Structure

```
GroScore/
├── groscore.py          # Main orchestrator
├── job.run              # SLURM job template
├── groscore_fe.py       # GroScore-FE orchestrator (absolute binding free energies)
├── job_fe.run           # GroScore-FE job template
├── forcefield/
│   ├── ccd_cache/            # wwPDB Chemical Component Dictionary entries (ligand bond orders)
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
│   ├── gromos54a8_rf/   # GROMOS 54A8 with the reaction field
│   └── charmm36/        # CHARMM36m parameter files
│       └── (same files)
└── utils/
    ├── fuse_glycans.py              # Fuse covalently bonded glycans with their Asn/Ser/Thr into one NCAA residue
    ├── renumber_pdb.py              # Assign sequential residue numbers, extract ligands/waters
    ├── fix_pdb.py                   # Fix missing atoms with PDBFixer
    ├── cap_termini.py               # Add ACE/NME terminal caps (generic templates for residues amber14 lacks)
    ├── parametrize_ncaa.py          # NCAA parametrization (OpenFF sidechain + AMBER backbone)
    ├── parametrize_ligand.py        # OpenFF small molecule parametrization
    ├── ccd_cache.py                 # Local CCD cache and prefetch for ligand bond orders
    ├── fix_ion_protonation.py       # Ion-coordinating CYS/HIS protonation states
    ├── merge_ligand.py              # Merge ligand topology into protein system
    ├── covalent.py                  # Covalent-contact rule shared by merge_ligand.py and fuse_glycans.py
    ├── merge_crystal_waters.py      # Merge crystal waters as SOL
    ├── make_ion_restraints.py       # Ion coordination restraints
    ├── make_cluster_group.py        # Protein_Struct index group, the group assemble_frame.py assembles
    ├── assemble_frame.py            # Assemble a frame's group by contact (lattice-vector moves only) for the restraint reference
    ├── pbc_tools.py                 # Periodic-boundary helpers (exact minimum image, residue spanning tree)
    ├── fix_topol_intermolecular.py  # Fix topology after solvation/genion
    ├── retype_ions.py               # Crystal ions get the water model's Li/Merz types (Mg2+_opc3 under OPC3)
    ├── check_brokenloop.py          # Loop connectivity validation
    ├── check_entangledloops.py      # Topological knot detection
    ├── make_cutout.py               # Interface region extraction
    ├── make_disres_en.py            # Distance restraints & elastic network
    ├── integrate.py                 # Force curve integration
    ├── rebound_rmsd.py              # Rebinding QC (re-bound vs. bound backbone RMSD)
    ├── estimators.py                # BAR and its overlap statistic, used by both orchestrators
    ├── local_runner.py              # --run-local: background job pool, one GPU per slot
    ├── fe_protocol.py               # GroScore-FE: the cycle defined once (legs, rates, row layout, protocol id)
    ├── make_fe_mdps.py              # GroScore-FE: every force field's leg mdps, generated from fe_protocol.py
    ├── make_boresch.py              # GroScore-FE: Boresch, interface and elastic-network restraints for the legs
    ├── measure_box.py               # GroScore-FE: production box size from the probe equilibration
    ├── integrate_dhdl.py            # GroScore-FE: dH/dlambda switching work
    ├── interface_qc.py              # GroScore-FE: interface contact recovery after rebinding
    ├── fe_leg_efficiency.py         # GroScore-FE: where a leg's hysteresis comes from, and what a longer leg or more cycles would buy
    └── fe_node_report.py            # Slow-node report from finished GROMACS logs (the --exclude list of the VSC-5 templates)
```

## Troubleshooting

### Common Issues

Only the last status line of a structure in `results_0.gs` counts. Any other status than `OK` keeps the structure out of scoring, and the score files name it in its place (classic: the Score column; FE: the `Note` column). The one exception is `SETUP_RUNNING` in `groscore_fe.py` (see below). Classic `OK` is written once the restraint set exists. An FE setup is complete once `setup.done` exists in the structure directory. Its `OK` is written earlier, after the loop checks, and a later line can still report a failed probe or restraint set.

**NODIR status**: `sp.gs` names a structure that has neither a directory nor a `.tar.gz` archive in the run directory.

**BROKEN status**: Protein loop connectivity failed validation. Check your input structure for missing residues or chain breaks.

**ENTANGLED status**: Topological knots detected. The protein structure may have threading artifacts that would invalidate pulling simulations.

**SETUP_RUNNING status**: A setup is in progress, or was interrupted (resubmit with `--restart`). It is not a failure. `groscore.py` does not score the structure until the setup ends in `OK` and writes `SETUP_RUNNING` in place of its score; `groscore_fe.py` scores the cycles the structure already has, or lists it as `PENDING`.

**FAILED status**: `pdb2gmx` left no `conf.gro` (e.g. `fix_pdb.py` refused residues that would merge), the vacuum minimization did not produce `emin_vac.gro` (grompp hit a topology/coordinate mismatch), the broken-loop or entanglement check crashed, or `GROSCORE_GLYCAN_MAX_SUGARS` is set but is not a whole number of at least 1 written in digits only (with ligand parametrization on). Check the structure's SLURM output for the first error.

**SETUP_FAIL status**: Classic: solvation, the solvated minimization or the initial equilibration did not finish. FE: the setup produced no usable restraints, or it refused to run beside cycles it did not start (use a fresh run directory).

**TRJCONV_FAIL status**: `gmx trjconv -pbc whole` produced no frame to build the restraints on.

**ASSEMBLE_FAIL status**: `assemble_frame.py` could not assemble the restraint frame; its reason is on stderr in the job log, just above the `Stage 0 failed` line.

**DISRES_FAIL status** (classic): `make_disres_en.py` found no interface or a frame split across the periodic boundary, or the restraint set frozen by earlier pull legs is unusable. In the last case, set the structure up again in a fresh run directory.

**LIGAND_FAIL status**: see *Ligand parametrization fails* below. The same status stops a structure when `fuse_glycans.py` exits with an error, since which glycans belong to the protein is then unknown; the job log names its exit status.

**PROBE_FAIL status** (FE): The probe equilibration, the box measurement or the box padding failed; the job log says which. Running the structure again keeps the finished probe replicas and stages, runs an unfinished stage again from step 0 and measures the box again.

**BORESCH_FAIL status** (FE): `make_boresch.py` stopped. The job log names its reason, the first line of `boresch_failed.gs` (e.g. `INPUT_FRAME_SPLIT`).

**HIGH_RMSD flag**: The [rebinding QC](#rebinding-sanity-check-qc) found at least one cycle whose complex did not return to the bound pose. The score is still reported; inspect the flagged cycles (`RMSD_max_A` in the score files, per-cycle values in the 5th field of `results.d/<ID>_c<N>.gs`, or, in a legacy run directory, in the 3rd column of `results_<even>.gs`), add cycles with `-n <more> --restart` (plus `--inject-job-run` for archived structures) and check whether the score moves.

**RMSD reported as `nan`**: The measurement was not made. Possible reasons: a legacy run directory without the check; a frame could not be read or assembled; the index has no `[ Protein ]` group; `gmx rms` failed; or (classic) the cycle's legs did not finish. A missing RMSD never changes a score. Reproduce the failure with `python3 ../../utils/rebound_rmsd.py --ref npt_c<N>.gro --query bindrev_<2N>.gro -v` inside the structure directory; the reason is printed to stderr.

**Job failures**: Ensure GROMACS modules are loaded and paths are correctly set in your SLURM environment. `job.run` and `job_fe.run` exit 1 when a cycle ends without a complete result. `job.run` (and `job_fe.run` outside `--cycle` mode) also exits 1 when another job holds `job_run.lock` in the structure directory, as after a second submission while the first still runs, or while an MD leg started by a job that was killed still runs; the message starts with "Safety check". A lock is taken over only from a job shown to be gone: on the same host by its pid and its running leg, on another host by `squeue`, which must answer (a controller error keeps the lock). If that job is gone and `squeue` cannot say so, remove the lock by hand and resubmit. An FE cycle task skips a cycle that another job is running (`results_fe_c<N>.lock` in the structure directory; remove it by hand if that job is gone and `squeue` cannot say so). A TERM (scancel, the walltime, stopping the local runner) stops the running leg and ends the job; the next run starts that leg again from step 0. An FE work that comes out NaN is explained in the job log under `... failed for cycle N`: `integrate_dhdl.py` refuses a dH/dλ record that is not whole (a broken, unparsable or cut row, a time that goes back or skips rows, or a missing head or last row) and names the line, and `integrate.py` names the row it cannot read. A damaged file stays damaged, and every rerun integrates it again: move aside the files of that leg and of every leg that goes on from it (each leg starts from the one before; `boundrev2`, `holdbrev1` and `boundrev1` from `holdfwd0`), or of the whole cycle, then resubmit with `--restart` to run them again.

Every MD leg that dies is started again from step 0, never from its own checkpoint: in `job.run` every NVT, NPT, pull and push leg, in `job_fe.run` every stage of the probe and of a cycle (the NVT stages, the probe NPT replicas, `npt_c` and the FE legs, each from a newly built tpr). An unfinished run is never continued, and a leg reads the checkpoint of the leg before only once that one has written its `.gro`. Both runners make up to `GROSCORE_LEG_RETRIES` attempts in all (default 2). In `job.run`, `GROSCORE_LEG_TIMEOUT` (default `24h`, sized for a CPU run; the VSC-5 GPU templates set `180m`) ends a 5 ns SMD leg or a minimization that hangs; its NVT/NPT legs have 60 min each. `job_fe.run` uses fixed limits: 60 min per NVT leg, 600 min for `npt_c` and each probe replica, 180 min per FE leg. A `job_fe.run --cycle` task waits up to `GROSCORE_SETUP_WAIT` seconds (default 21600) for its setup, checking every `GROSCORE_SETUP_POLL` seconds (default 30).

**Ligand parametrization fails**: A ligand that cannot be parametrized or merged stops the structure as `LIGAND_FAIL`; `ligand_failed.gs` in the structure directory names the instance and the reason. A glycan that was not fused carries the reason after `glycan not fused:` in that line. Glycans of any size are fused by default; a glycan stopped by more sugars than a cap you set with `GROSCORE_GLYCAN_MAX_SUGARS` in the shell that submits the jobs is fused once you raise or unset it (unset or empty means no cap; otherwise a whole number of at least 1 in digits only, and anything else stops the structure as `FAILED`). AM1-BCC of a fused residue takes about 1 to 2.5 minutes with one sugar, about 3 with two and about 9 with three, and grows about with the cube of the residue size, so a large glycan lengthens the setup by hours. It has no time limit by default; `GROSCORE_GLYCAN_AM1BCC_WALL_S`, set the same way, gives the seconds all its attempts together may take, and a glycan whose residue exceeds it goes back to the ligands. CCD entries are read from `forcefield/ccd_cache` first. On nodes without network, prefetch them with `python3 utils/ccd_cache.py myproject/*/input.pdb` on a login node. To go on, fix the deposited ligand, or run without ligands with `--no-ligand-param`.

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

The setup scripts write the chains of the complex and the oligosaccharides covalently linked to them, which the PDB keeps in chains of their own (`benchmark/glycan_chains.py`).

## Citation

If you use GroScore in your research, please cite:

> Perthold, J. W.; Petrov, D.; Oostenbrink, C. GroScore: Automated Scoring of Protein–Protein Interactions Using Bidirectional Steered Molecular Dynamics. *J. Chem. Inf. Model.* **2026**, Application Note. https://doi.org/10.1021/acs.jcim.6c01771

The software itself is archived on Zenodo with one DOI per release from v1.4 on; the concept DOI https://doi.org/10.5281/zenodo.22686986 resolves to the latest release.

The original, unidirectional method:

> Perthold, J. W.; Oostenbrink, C. GroScore: Accurate Scoring of Protein–Protein Binding Poses Using Explicit-Solvent Free-Energy Calculations. *J. Chem. Inf. Model.* **2019**, *59* (12), 5074–5085. https://doi.org/10.1021/acs.jcim.9b00687

The bidirectional scheme is derived in Chapter 3 of the thesis (included as [theory/thesis_chapter_3.pdf](theory/thesis_chapter_3.pdf)):

> Perthold, J. W. New developments and critical views on binding free-energy calculations using molecular mechanics. *Doctoral Dissertation*, University of Natural Resources and Life Sciences, Vienna (BOKU), **2023**. [Library catalog](https://litsearch.boku.ac.at/primo-explore/fulldisplay?docid=BOK_alma2198734100003345&vid=BOK)

## Acknowledgements

J.W.P. has been a recipient of a DOC Fellowship of the Austrian Academy of Sciences (ÖAW) at the Institute for Molecular Modeling and Simulation at the University of Natural Resources and Life Sciences, Vienna (Grant No. 24987).

The computational results have been achieved using the Austrian Scientific Computing (ASC) infrastructure.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Author:** Jan Walther Perthold
**Email:** jan@ackergarten.at
