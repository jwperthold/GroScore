# GroScore

<p align="center">
  <img src="logo.png" alt="GroScore Logo" width="200">
</p>

**Computational Chemistry Toolkit for Protein-Protein Affinity Scoring with MD**

[![Version](https://img.shields.io/badge/version-0.98-blue.svg)](https://github.com/jwperthold/GroScore)
[![Python](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2026.0-orange.svg)](https://www.gromacs.org/)

GroScore estimates binding affinities between protein pairs using short steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

---

## Features

- **Automated MD Pipeline** - Complete workflow from structure preparation to final scoring
- **SLURM Integration** - Efficient HPC execution via job arrays
- **Structure Validation** - Built-in checks for broken loops and topological knots
- **Cutout Mode** - Choose between interface-only (faster, default) or full-protein simulations
- **Elastic Network Restraints** - Maintains protein stability when simulating only interface-proximal atoms (within a distance cutoff) for faster computation
- **Smart Fragment Handling** - Chain break detection, small gap filling (< 4 residues), minimum fragment size enforcement, isolated cap removal, and chain boundary protection
- **Structural Ion Support** - Automatic handling of metal ions (ZN, CA, MG, CU, CU1, NA, CL) with coordination restraints
- **Small Molecule Support** - OpenFF-based parametrization of ligands and cofactors (AMBER19SB only), with OpenBabel bond perception and RCSB template fallback
- **Crystal Water Preservation** - Crystal waters from PDB structures are retained and included in simulations
- **Multiple Force Fields** - Support for AMBER19SB (all-atom), GROMOS 54A8 (united-atom), and CHARMM36 (all-atom)

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
| GROMACS | 2026.0 | Molecular dynamics engine |
| SLURM | 23.11 | HPC job scheduler |

**Conda environment:** All Python dependencies are available in the `groscore` conda environment.

## Force Fields

GroScore supports multiple force fields, selectable via the `-ff` option:

| Force Field | Type | Water Model | Constraints | Cutoffs | Terminal Capping |
|-------------|------|-------------|-------------|---------|------------------|
| **AMBER19SB OPC** | All-atom | OPC (4-point) | all-bonds | 1.0 nm | ACE/NME |
| **AMBER19SB OPC3** (default) | All-atom | OPC3 (3-point) | all-bonds | 1.0 nm | ACE/NME  |
| **CHARMM36** | All-atom | TIP3P | all-bonds | 1.2 nm | ACE/COOH |
| **GROMOS 54A8** | United-atom | SPC | all-bonds | 1.4 nm | ACE/COOH |

All force fields use:
- **Electrostatics**: PME (Particle Mesh Ewald) for long-range electrostatic interactions
- **Constraints**: all-bonds for maximum stability
- **Heavy hydrogens**: `mass-repartition-factor = 3` for stable 4 fs timesteps

**Terminal Capping Details:**
- **AMBER19SB**: Uses ACE (N-acetyl) and NME (N-methylamide) caps added as explicit residues via PDBFixer before pdb2gmx processing. This provides proper neutral termini for fragment ends.
- **CHARMM36**: Uses ACE (N-acetyl) caps at N-termini (explicit residues) and COOH patches at C-termini for improved stability.
- **GROMOS 54A8**: Uses ACE caps at N-termini (explicit residues) and COOH patches at C-termini.

The CHARMM36 force field parameters (from [MacKerell lab](https://mackerell.umaryland.edu/charmm_ff.shtml)) are included in `forcefield/charmm36-jul2022.ff/`. The GROMOS 54A8 force field parameters are included in `forcefield/gromos54a8.ff/`.

## Heteroatom Support

### Structural Ions

Metal ions (ZN, CA, MG, CU, CU1, NA, CL) are automatically detected from PDB HETATM records and carried through the full pipeline. Ion-protein coordination is maintained via topology-level harmonic restraints using optimal distances from force field parameters and literature (e.g., Zn-S 0.232 nm, Zn-N 0.207 nm). Ions participate in the pulling restraints and are assigned to their respective protein chain.

### Small Molecules (AMBER19SB only)

Ligands and cofactors are automatically extracted from PDB HETATM records and parametrized using the [Open Force Field](https://openforcefield.org/) (Sage 2.2.1):

1. **Bond order perception**: OpenBabel reads 3D coordinates and assigns bond orders. If kekulization fails (common for fused ring systems without H), the RCSB Chemical Component Dictionary is used as fallback.
2. **Protonation**: Assigned at physiological pH (7.4) by OpenBabel
3. **Parametrization**: OpenFF Sage force field via Interchange → GROMACS topology
4. **Merging**: Ligand topology and coordinates are merged into the protein system

For best results with novel (non-PDB) ligands, provide input structures with explicit hydrogen coordinates.

### Crystal Waters

Crystal water molecules (HOH) from PDB structures are preserved and included as SOL in the simulation. They are placed at crystallographic positions before bulk solvation.

## Benchmark Data (HADDOCKING Protein-Protein Affinity Benchmark)

| **AMBER19SB/OPC** | |
|-------------------|-------------|
| **Fit:** pKd = -0.0139 × GroScore + 4.5175 | Convergence |
| <img src="/benchmark/results/correlation_plot_amber.png" alt="Correlation Plot AMBER19SB/OPC" height="290"> | <img src="/benchmark/results/convergence_plot_amber.png" alt="Convergence Plot AMBER19SB/OPC" height="290"> |

| **AMBER19SB/OPC3** | |
|--------------------|-------------|
| **Fit:** pKd = -0.0168 × GroScore + 3.5896 | Convergence |
| <img src="/benchmark/results/correlation_plot_amber_opc3.png" alt="Correlation Plot AMBER19SB/OPC3" height="290"> | <img src="/benchmark/results/convergence_plot_amber_opc3.png" alt="Convergence Plot AMBER19SB/OPC3" height="290"> |

| **CHARMM36/TIP3P** | |
|--------------------|-------------|
| **Fit:** pKd = -0.0178 × GroScore + 3.4113 | Convergence |
| <img src="/benchmark/results/correlation_plot_charmm.png" alt="Correlation Plot CHARMM36/TIP3P" height="290"> | <img src="/benchmark/results/convergence_plot_charmm.png" alt="Convergence Plot CHARMM36/TIP3P" height="290"> |

| **GROMOS 54A8/SPC** | |
|---------------------|-------------|
| **Fit:** pKd = -0.0178 × GroScore + 3.7654 | Convergence |
| <img src="/benchmark/results/correlation_plot_gromos_54a8.png" alt="Correlation Plot GROMOS 54A8/SPC" height="290"> | <img src="/benchmark/results/convergence_plot_gromos_54a8.png" alt="Convergence Plot GROMOS 54A8/SPC" height="290"> |


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
- `-n, --numruns` - Number of independent pull/push cycles (default: 3)
- `-s, --structparams` - Structure parameter file (default: `sp.gs`)
- `-ff, --forcefield` - Force field: `amber19sb_opc3` (default), `amber19sb_opc`, `gromos54a8`, or `charmm36`
- `--cutout` - Extract interface region only (default, faster)
- `--no-cutout` - Use full protein structure (slower)
- `--restart` - Resubmit jobs (useful for continuing interrupted runs)

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
| `scores_cgi.gs` | Crooks Gaussian Intersection (for $\ge$ 20 cycles) |

## Simulation Pipeline

```
Stage 0: Preparation
├── PDB fixing (missing atoms, non-standard residues)
├── Ligand extraction & OpenFF parametrization (AMBER19SB)
├── Crystal water extraction
├── Structure validation
├── PDB conversion (pdb2gmx)
├── Ligand/water/ion merging into topology
├── Ion coordination restraints (topology-level)
├── Solvation (water + 0.15 M NaCl)
└── Energy minimization → emin_solv.gro

Initial Equilibration (for distance restraints)
└── 5-phase NVT + NPT → npt_init_cluster.gro

Independent Cycles (N cycles, default 3)
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
│   ├── charmm36-jul2022.ff/  # CHARMM36 force field parameters
│   └── gromos54a8.ff/        # GROMOS 54A8 force field parameters
├── settings/
│   ├── amber19sb_opc/   # AMBER19SB/OPC parameter files
│   ├── amber19sb_opc3/  # AMBER19SB/OPC3 parameter files
│   ├── gromos54a8/      # GROMOS 54A8 parameter files
│   │   ├── emin_*.mdp   # Energy minimization
│   │   ├── nvt_*.mdp    # NVT equilibration phases
│   │   ├── npt*.mdp     # NPT equilibration
│   │   └── bind*.mdp    # SMD pulling parameters
│   └── charmm36/        # CHARMM36 parameter files
│       └── (same files)
└── utils/
    ├── renumber_pdb.py              # Assign sequential residue numbers, extract ligands/waters
    ├── fix_pdb.py                   # Fix missing atoms with PDBFixer
    ├── cap_termini.py               # Add ACE/NME terminal caps
    ├── parametrize_ligand.py        # OpenFF small molecule parametrization
    ├── merge_ligand.py              # Merge ligand topology into protein system
    ├── merge_crystal_waters.py      # Merge crystal waters as SOL
    ├── make_ion_restraints.py       # Ion coordination restraints
    ├── fix_topol_intermolecular.py  # Fix topology after solvation/genion
    ├── check_brokenloop.py          # Loop connectivity validation
    ├── check_entangledloops.py      # Topological knot detection
    ├── make_cutout.py               # Interface region extraction
    ├── make_disres_en.py            # Distance restraints & elastic network
    └── integrate.py                 # Force curve integration
```

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

## Fragment Handling

GroScore automatically handles complex protein structures with multiple chains and chain breaks:

- **Chain Break Detection** - Gaps in residue numbering within a chain are detected and marked with TER records
- **Small Gap Filling** - Gaps < 4 residues introduced by interface filtering are automatically filled to avoid introducing artificial chain breaks, while respecting TER positions (never merges different chains)
- **Minimum Fragment Size** - Fragments smaller than 5 residues are automatically extended by adding neighboring residues for improved stability
- **Isolated Cap Removal** - ACE/NME caps that lost their partners during interface filtering are removed to prevent orphaned caps
- **Fragment Merging** - Fragments from the same original PDB chain are merged into a single moleculetype for GROMACS
- **Terminal Capping** - Fragment termini are capped to provide neutral ends:
  - **AMBER19SB**: ACE/NME residues added explicitly via `cap_termini.py` before pdb2gmx
  - **CHARMM36/GROMOS 54A8**: ACE residues (N-termini) added via `cap_termini.py`, COOH patches (C-termini) applied during pdb2gmx

This ensures proper topology generation even for structures with missing loops or multi-chain complexes, while maintaining chain boundaries and avoiding artificial chain breaks.

## File Formats

- `.gs` - GroScore data files (tab-separated, `#` for comments)
- `.mdp` - GROMACS molecular dynamics parameter files
- `.gro` - GROMACS coordinate files
- `.xvg` - GROMACS output data (force curves)
- `.itp` - GROMACS topology include files (ligand parameters)
- `.sdf` - Structure-data files (ligand bond orders, for debugging)

## Troubleshooting

### Common Issues

**BROKEN status**: Protein loop connectivity failed validation. Check your input structure for missing residues or chain breaks.

**ENTANGLED status**: Topological knots detected. The protein structure may have threading artifacts that would invalidate pulling simulations.

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

For the improved method (Chapter 3 included in `theory/`), see:

> Perthold, J. W. New developments and critical views on binding free-energy calculations using molecular mechanics. *Doctoral Dissertation*, University of Natural Resources and Life Sciences, Vienna (BOKU), **2023**. [Library catalog](https://litsearch.boku.ac.at/primo-explore/fulldisplay?docid=BOK_alma2198734100003345&vid=BOK)

## Acknowledgements

J.W.P. has been a recipient of a DOC Fellowship of the Austrian Academy of Sciences (ÖAW) at the Institute for Molecular Modeling and Simulation at the University of Natural Resources and Life Sciences, Vienna (Grant No. 24987).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Author:** Jan Walther Perthold
**Email:** jan@ackergarten.at
