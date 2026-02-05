# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GroScore is a computational chemistry toolkit for protein-protein affinity scoring using short steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

**Version:** 0.85
**Contact:** jan@ackergarten.at

## Tech Stack

- Python 3.13 with NumPy 2.3 and SciPy 1.16
- GROMACS 2026.0 (external MD engine)
- SLURM 23.11 job scheduler for HPC execution
- GROMOS 54A7 united-atom force field for protein parametrization
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
| `extract_chains.py` | Extracts chain info from PDB and generates residue map for protein B |
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
   - `scores_cgi.gs` - Crooks Gaussian Intersection

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

The `extract_chains.py` utility reads the PDB file and generates `chain_map.gs` containing residue numbers for protein B, which other utilities use for protein separation.

## Code Patterns

- File parsing filters comments with `if not line.strip().startswith("#")`
- Large arrays pre-allocated (1,000,000 elements) then sliced to actual size
- Distance calculations use explicit 3D Euclidean formula
- Exit codes: 0 (success), 1 (failure); status strings: "OK", "BROKEN", "ENTANGLED"
- Protein separation uses chain map file (`-m/--chainmap` parameter) containing residue numbers for protein B
