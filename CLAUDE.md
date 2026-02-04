# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GroScore is a computational chemistry toolkit for estimating binding free energies between protein pairs using steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

**Version:** 0.82
**Contact:** jan@ackergarten.at

## Tech Stack

- Python 3.13 with NumPy 2.3
- GROMACS 2019.5 (external MD engine)
- SLURM 23.11 job scheduler for HPC execution
- SPC water model for solvation

## Running GroScore

```bash
# Run with N simulation cycles (required parameter)
python groscore.py -n 10

# With custom structure parameter file (default: sp.gs)
python groscore.py -s myparams.gs -n 10
```

After initial run, jobs are submitted via auto-generated `array_submit.run`.

## Architecture

### Main Components

- **groscore.py** - Main orchestrator that reads structure parameters, generates SLURM submission scripts, monitors job completion, and performs final statistical analysis
- **job.run** - Bash script executed per structure as SLURM array task; runs the complete MD workflow

### Utility Scripts (utils/)

| Script | Purpose |
|--------|---------|
| `check_brokenloop.py` | Validates protein loop connectivity before simulation |
| `check_entangledloops.py` | Detects topological knots that would invalidate results |
| `make_cutout.py` | Extracts interface region from full protein structures |
| `make_disres_en.py` | Generates distance restraints and elastic network |
| `integrate.py` | Integrates force curves from pulling simulations |

### Simulation Pipeline

1. **Stage 0**: Structure validation, PDB conversion, solvation, 5-phase NVT + NPT equilibration
2. **Stages 1-N**: Alternating unbinding (pulling) and binding (pushing) SMD simulations
3. **Final**: Statistical analysis producing two ranking methods:
   - `scores_avg.gs` - Simple average of pulls/pushes
   - `scores_cgi.gs` - Gaussian intersection

## File Formats

- `.gs` - GroScore data files (two/three column, `#` for comments)
- `.mdp` - GROMACS parameter files (in `settings/`)
- `.gro` - GROMACS coordinate files
- `.xvg` - GROMACS output data (force curves)

## Key Parameters

- Interface cutoff: 0.6 nm
- Elastic network range: 0.4-0.9 nm
- Keep cutoff for interface extraction: 2.0 nm
- Ion concentration: 0.15 M NaCl

## Code Patterns

- File parsing filters comments with `if not line.strip().startswith("#")`
- Large arrays pre-allocated (1,000,000 elements) then sliced to actual size
- Distance calculations use explicit 3D Euclidean formula
- Exit codes: 0 (success), 1 (failure); status strings: "OK", "BROKEN", "ENTANGLED"
- Protein chains divided at configurable residue boundary (`-s/--startb` parameter)
