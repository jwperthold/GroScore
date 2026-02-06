# GroScore

<p align="center">
  <img src="logo.png" alt="GroScore Logo" width="200">
</p>

**Computational Chemistry Toolkit for Protein-Protein Affinity Scoring**

[![Version](https://img.shields.io/badge/version-0.96-blue.svg)](https://github.com/jwperthold/GroScore)
[![Python](https://img.shields.io/badge/python-3.13-green.svg)](https://www.python.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2026.0-orange.svg)](https://www.gromacs.org/)

GroScore estimates binding affinities between protein pairs using short steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

---

## Features

- **Automated MD Pipeline** - Complete workflow from structure preparation to final scoring
- **SLURM Integration** - Efficient HPC execution via job arrays
- **Multiple Scoring Methods** - Two different ranking approaches for robust results
- **Structure Validation** - Built-in checks for broken loops and topological knots
- **Elastic Network Restraints** - Maintains protein stability when simulating only interface-proximal atoms (within a distance cutoff) for faster computation
- **Optional Cutout Mode** - Choose between interface-only (faster) or full-protein simulations
- **Multiple Force Fields** - Support for GROMOS 54A7 (united-atom), CHARMM36 (all-atom), and AMBER19SB (all-atom)
- **Automatic Fragment Handling** - Chain break detection, minimum fragment size enforcement, and same-chain fragment merging

## Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.13 |
| NumPy | 2.3 |
| SciPy | 1.16 |
| OpenMM | 8.2 |
| PDBFixer | 1.9 |
| GROMACS | 2026.0 |
| SLURM | 23.11 |

## Force Fields

GroScore supports three force fields, selectable via the `-ff` option:

| Force Field | Type | Water Model | Constraints | Cutoffs | Terminal Capping |
|-------------|------|-------------|-------------|---------|------------------|
| **CHARMM36** (default) | All-atom | TIP3P | all-bonds | 1.2 nm | ACE/NME (explicit residues) |
| **GROMOS 54A7** | United-atom | SPC | all-bonds | 1.4 nm | NH2/COOH (patches) |
| **AMBER19SB** | All-atom | OPC | all-bonds | 1.0 nm | ACE/NME (explicit residues) |

All use all-bonds constraints and heavy hydrogen masses (`-heavyh` flag) for stable 4 fs timesteps.

**Terminal Capping Details:**
- **CHARMM36/AMBER19SB**: Use ACE (N-acetyl) and NME (N-methylamide) caps added as explicit residues via PDBFixer before pdb2gmx processing. This provides proper neutral termini for fragment ends.
- **GROMOS 54A7**: Uses traditional NH2/COOH terminal patches applied during pdb2gmx processing.

The CHARMM36 force field parameters (from [MacKerell lab](https://mackerell.umaryland.edu/charmm_ff.shtml)) are included in `forcefield/charmm36-jul2022.ff/`.

## Installation

```bash
git clone https://github.com/jwperthold/GroScore.git
cd groscore
```

Ensure GROMACS and SLURM are available in your environment.

## Quick Start

### 1. Set Up Project Directory

GroScore runs from a **project subdirectory** that contains your structures. Create a project folder and structure directories:

```bash
cd groscore
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

Structure IDs can be alphanumeric (e.g., PDB IDs) and must match the directory names.

### 3. Run GroScore

Run from within your project directory:

```bash
cd myproject
python ../groscore.py
```

**Options:**
- `-n, --numruns` - Number of pull/push cycles (default: 5)
- `-s, --structparams` - Structure parameter file (default: `sp.gs`)
- `-ff, --forcefield` - Force field: `charmm36` (default), `gromos54a7`, or `amber19sb`
- `--cutout` - Extract interface region only (default, faster)
- `--no-cutout` - Use full protein structure (slower, more accurate)

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

| Output File | Method |
|-------------|--------|
| `scores_avg.gs` | Simple average of pulls/pushes |
| `scores_cgi.gs` | Crooks Gaussian Intersection |

## Simulation Pipeline

```
Stage 0: Preparation
├── PDB fixing (missing atoms, non-standard residues)
├── Structure validation
├── PDB conversion (heavy hydrogen)
├── Solvation (SPC water, 0.15 M NaCl)
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

## Project Structure

```
groscore/
├── groscore.py          # Main orchestrator
├── job.run              # SLURM job template
├── forcefield/
│   └── charmm36-jul2022.ff/  # CHARMM36 force field parameters
├── settings/
│   ├── gromos54a7/      # GROMOS 54A7 parameter files
│   │   ├── emin_*.mdp   # Energy minimization
│   │   ├── nvt_*.mdp    # NVT equilibration phases
│   │   ├── npt*.mdp     # NPT equilibration
│   │   └── bind*.mdp    # SMD pulling parameters
│   └── charmm36/        # CHARMM36 parameter files
│       └── (same files)
└── utils/
    ├── renumber_pdb.py          # Assign sequential residue numbers
    ├── fix_pdb.py               # Fix missing atoms with PDBFixer
    ├── cap_termini.py           # Add ACE/NME caps (CHARMM36/AMBER19SB)
    ├── check_brokenloop.py      # Loop connectivity validation
    ├── check_entangledloops.py  # Topological knot detection
    ├── make_cutout.py           # Interface region extraction
    ├── make_disres_en.py        # Distance restraints & elastic network
    └── integrate.py             # Force curve integration
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Timestep | 4 fs | Integration timestep (heavy hydrogen) |
| Interface cutoff | 0.6 nm | Defines protein-protein interface |
| Elastic network range | 0.4-0.9 nm | Restraint distance bounds |
| Keep cutoff | 2.0 nm | Interface extraction radius |
| Ion concentration | 0.15 M | NaCl for physiological conditions |
| Minimum fragment size | 3 residues | Ensures stable fragments in cutout mode |

## Fragment Handling

GroScore automatically handles complex protein structures with multiple chains and chain breaks:

- **Chain Break Detection** - Gaps in residue numbering within a chain are detected and marked with TER records
- **Minimum Fragment Size** - Fragments smaller than 3 residues are automatically extended by adding neighboring residues
- **Fragment Merging** - Fragments from the same original PDB chain are merged into a single moleculetype for GROMACS
- **Terminal Capping** - Fragment termini are capped to provide neutral ends:
  - **CHARMM36/AMBER19SB**: ACE/NME residues added explicitly via `cap_termini.py` before pdb2gmx
  - **GROMOS 54A7**: NH2/COOH patches applied during pdb2gmx processing

This ensures proper topology generation even for structures with missing loops or multi-chain complexes.

## File Formats

- `.gs` - GroScore data files (tab-separated, `#` for comments)
- `.mdp` - GROMACS molecular dynamics parameter files
- `.gro` - GROMACS coordinate files
- `.xvg` - GROMACS output data (force curves)

## Troubleshooting

### Common Issues

**BROKEN status**: Protein loop connectivity failed validation. Check your input structure for missing residues or chain breaks.

**ENTANGLED status**: Topological knots detected. The protein structure may have threading artifacts that would invalidate pulling simulations.

**Job failures**: Ensure GROMACS modules are loaded and paths are correctly set in your SLURM environment.

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

For the improved method (Chapter 3 included in `theory/`), see:

> Perthold, J. W. New developments and critical views on binding free-energy calculations using molecular mechanics. *Doctoral Dissertation*, University of Natural Resources and Life Sciences, Vienna (BOKU), **2023**. [Library catalog](https://litsearch.boku.ac.at/primo-explore/fulldisplay?docid=BOK_alma2198734100003345&vid=BOK)

## Acknowledgements

J.W.P. has been a recipient of a DOC Fellowship of the Austrian Academy of Sciences (ÖAW) at the Institute for Molecular Modeling and Simulation at the University of Natural Resources and Life Sciences, Vienna (Grant No. 24987).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Author:** Jan Walther Perthold
**Email:** jan@ackergarten.at
