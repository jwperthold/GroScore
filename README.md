# GroScore

**Computational Chemistry Toolkit for Protein-Protein Binding Free Energy Estimation**

[![Version](https://img.shields.io/badge/version-0.82-blue.svg)](https://github.com/your-repo/groscore)
[![Python](https://img.shields.io/badge/python-3.x-green.svg)](https://www.python.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2019.5-orange.svg)](https://www.gromacs.org/)

GroScore estimates binding free energies between protein pairs using steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

---

## Features

- **Automated MD Pipeline** - Complete workflow from structure preparation to final scoring
- **SLURM Integration** - Efficient HPC execution via job arrays
- **Multiple Scoring Methods** - Two different ranking approaches for robust results
- **Structure Validation** - Built-in checks for broken loops and topological knots
- **Elastic Network Restraints** - Maintains protein stability when simulating only interface-proximal atoms (within a distance cutoff) for faster computation

## Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.x |
| NumPy | - |
| SciPy | - |
| GROMACS | 2019.5 |
| SLURM | - |

## Installation

```bash
git clone https://github.com/your-repo/groscore.git
cd groscore
```

Ensure GROMACS and SLURM are available in your environment.

## Quick Start

### 1. Prepare Input Files

Create a structure parameter file (`sp.gs`) with your protein structures:

```
# Structure ID    Chain B Start Residue
1                 150
2                 142
3                 165
```

### 2. Run GroScore

```bash
# Run with 10 simulation cycles
python groscore.py -n 10

# With custom parameter file
python groscore.py -s myparams.gs -n 10
```

### 3. Submit Jobs

After initialization, submit the generated SLURM script:

```bash
sbatch array_submit.run
```

### 4. Collect Results

Results are written to two output files, each using a different statistical method:

| Output File | Method |
|-------------|--------|
| `scores_avg.gs` | Simple average of pulls/pushes |
| `scores_cgi.gs` | Gaussian intersection |

## Simulation Pipeline

```
Stage 0: Preparation
├── Structure validation
├── PDB conversion
├── Solvation (SPC water, 0.15 M NaCl)
└── Equilibration (5-phase NVT + NPT)

Stages 1-N: Production
├── Unbinding (pulling SMD)
└── Binding (pushing SMD)

Final: Analysis
└── Statistical scoring (2 methods)
```

## Project Structure

```
groscore/
├── groscore.py          # Main orchestrator
├── job.run              # SLURM job template
├── settings/            # GROMACS parameter files
│   ├── emin_*.mdp       # Energy minimization
│   ├── nvt_*.mdp        # NVT equilibration phases
│   ├── npt*.mdp         # NPT equilibration
│   └── bind*.mdp        # SMD pulling parameters
└── utils/
    ├── check_brokenloop.py      # Loop connectivity validation
    ├── check_entangledloops.py  # Topological knot detection
    ├── make_cutout.py           # Interface region extraction
    ├── make_disres_en.py        # Distance restraints & elastic network
    └── integrate.py             # Force curve integration
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Interface cutoff | 0.6 nm | Defines protein-protein interface |
| Elastic network range | 0.4-0.9 nm | Restraint distance bounds |
| Keep cutoff | 2.0 nm | Interface extraction radius |
| Ion concentration | 0.15 M | NaCl for physiological conditions |

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

## Citation

If you use GroScore in your research, please cite:

> Perthold, J. W.; Oostenbrink, C. GroScore: Accurate Scoring of Protein–Protein Binding Poses Using Explicit-Solvent Free-Energy Calculations. *J. Chem. Inf. Model.* **2019**, *59* (12), 5074–5085. https://doi.org/10.1021/acs.jcim.9b00687

## License

[License information to be added]

## Contact

**Author:** Jan Walther Perthold
**Email:** jan@ackergarten.at
