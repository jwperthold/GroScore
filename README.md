# GroScore

<p align="center">
  <img src="logo.png" alt="GroScore Logo" width="200">
</p>

**Computational Chemistry Toolkit for Protein-Protein Affinity Scoring with MD**

[![Version](https://img.shields.io/badge/version-0.99-blue.svg)](https://github.com/jwperthold/GroScore)
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

`openbabel=3.1.1` requires `libxml2 <2.14`, while `gromacs=2026` requires `libhwloc ≥2.12.2` which pulls in `libxml2 ≥2.14` — these are mutually exclusive in conda. GROMACS must be installed separately:

- **HPC cluster:** load the system module (`module load gromacs/2026` or similar)
- **Workstation — separate conda env:**
  ```bash
  conda create -n gmx2026 -c conda-forge gromacs=2026   # GPU build selected automatically on CUDA 12.9+
  export PATH="$(conda info --base)/envs/gmx2026/bin:$PATH"
  ```
- **From source:** follow the [official install guide](https://manual.gromacs.org/current/install-guide/index.html)

### SLURM

On HPC clusters SLURM is managed by the system administrators — verify with `squeue --version`. On a local workstation (Ubuntu/Debian):

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
- `--restart` - Resubmit jobs (useful for continuing interrupted runs)
- `--inject-job-run` - Inject fresh job.run into archived (.tar.gz) structures (skipped by default)

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

Re-run `python ../groscore.py` periodically to check progress and collect results.

### 5. Collect Results

Results are written to two output files ranked by binding affinity:

| Output File | Method | Required Cycles |
|-------------|--------|-----------------|
| `scores_avg.gs` | Simple average of pull/push works | any (≥ 1) |
| `scores_cgi.gs` | Crooks Gaussian Intersection of forward/reverse work distributions | ≥ 20 |

Note that CGI requires at least 20 cycles to fit forward and reverse work distributions; with the default `--numruns 5` only `scores_avg.gs` is produced. Increase `-n` if you want CGI estimates.

#### Interpreting the score

- **Sign convention**: more negative score ⇄ tighter binding. Predicted pKd is monotonically *decreasing* in the score.
- **Units**: kJ·mol⁻¹. The score is the integrated pulling work along the unbinding/rebinding coordinate, averaged over cycles.
- **Convert to pKd**: use the linear fits provided per force field in the [Benchmark Data](#benchmark-data-haddocking-protein-protein-affinity-benchmark) section, e.g. for AMBER19SB/OPC3: `pKd ≈ -0.0176 × score + 3.4513`. These coefficients are calibrated on the HADDOCKING benchmark.
- **Uncertainty**: the `CI95` column is a 95 % confidence interval on the score from the between-cycle scatter.

## Benchmark Data (HADDOCKING Protein-Protein Affinity Benchmark)

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

All force fields use:
- **Electrostatics**: PME (Particle Mesh Ewald) for long-range electrostatic interactions
- **Constraints**: all-bonds for maximum stability
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
└── 5-phase NVT + NPT → npt_init_cluster.gro

Independent Cycles (N cycles, default 5)
├── Cycle 1:
│   ├── Fresh full equilibration (NVT 1-5 + NPT)
│   ├── Pull (unbinding SMD)
│   └── Short NPT + Push (binding SMD)
├── Cycle 2:
│   ├── Fresh full equilibration (NVT 1-5 + NPT)
│   ├── Pull (unbinding SMD)
│   └── Short NPT + Push (binding SMD)
└── ... (each cycle independent, new random velocities)

Final: Analysis
└── Statistical scoring (2 methods)
```

Each cycle starts fresh from `emin_solv.gro` with independent equilibration, providing statistically independent samples for robust scoring.

### Reproducibility

Each cycle draws fresh velocities from a Maxwell-Boltzmann distribution at 300 K. The initial-velocity seed in every NVT/NPT/SMD `.mdp` is set to `gen_seed = -1`, i.e. GROMACS picks a fresh seed from the wall clock at submission time. This is deliberate — independent cycles must sample independent trajectories — but it does mean that scores from a re-submitted run will not be bitwise-identical to the original. The CI95 column in `scores_avg.gs` quantifies the resulting between-cycle variance.

### Throughput

On a single GPU (consumer-grade RTX-class), expect roughly **8 GPU-hours per structure for 5 cycles** (including all equilibration legs and 5 × 10 ns of SMD). Cost scales linearly with cycle count and roughly linearly with system size; the interface-cutout mode (default) keeps system size near-constant across most complexes, so per-structure walltimes are tightly clustered. The benchmark directory contains `compute_walltime.py`, which extracts realised walltimes and PFLOP counts from completed SLURM logs.

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

- `.gs` - GroScore data files (tab-separated, `#` for comments)
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
    └── integrate.py                 # Force curve integration
```

## Troubleshooting

### Common Issues

**BROKEN status**: Protein loop connectivity failed validation. Check your input structure for missing residues or chain breaks.

**ENTANGLED status**: Topological knots detected. The protein structure may have threading artifacts that would invalidate pulling simulations.

**FAILED status**: Stage-0 setup or energy minimization did not complete — e.g. `emin_vac.gro` was not produced (grompp hit a topology/coordinate mismatch) or the entanglement check returned no result. Check the structure's SLURM output for the first GROMACS error. Any status other than `OK` excludes the structure from scoring.

**Job failures**: Ensure GROMACS modules are loaded and paths are correctly set in your SLURM environment.

**Ligand parametrization fails**: If OpenBabel kekulization fails and RCSB download is unavailable, provide input PDB with explicit hydrogen coordinates for the ligand.

## Benchmark

The `benchmark/` directory contains a setup script for the [HADDOCKING Protein-Protein Affinity Benchmark](https://github.com/haddocking/binding-affinity-benchmark) (46 structures). To run the benchmark:

```bash
cd benchmark
python setup_benchmark.py  # Downloads PDBs and creates sp.gs
python ../groscore.py
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
