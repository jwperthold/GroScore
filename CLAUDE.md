# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GroScore is a computational chemistry toolkit for protein-protein affinity scoring using short steered molecular dynamics (SMD) simulations. It orchestrates GROMACS simulations via SLURM job arrays to perform repeated pulling/pushing cycles and calculates binding affinity scores using multiple statistical methods.

**Version:** 0.85
**Contact:** jan@ackergarten.at

## Tech Stack

- Python 3.13 with NumPy 2.3, SciPy 1.16, OpenMM, and PDBFixer
- GROMACS 2026.0 (external MD engine)
- SLURM 23.11 job scheduler for HPC execution
- Force fields: GROMOS 54A7 (united-atom) or CHARMM36 (all-atom)
- Water models: SPC (GROMOS) or TIP3P (CHARMM)

## Running GroScore

```bash
# Run with N simulation cycles (required parameter)
python groscore.py -n 10

# With custom structure parameter file (default: sp.gs)
python groscore.py -s myparams.gs -n 10

# Use CHARMM36 force field instead of GROMOS 54A7
python groscore.py -n 10 -ff charmm36

# Disable interface cutout (use full protein structure)
python groscore.py -n 10 --no-cutout
```

**Command-line options:**
- `-n, --numruns` - Number of pull/push cycles (required)
- `-s, --structparams` - Structure parameter file (default: `sp.gs`)
- `-ff, --forcefield` - Force field: `gromos54a7` (default) or `charmm36`
- `--cutout` - Extract interface region only (default, faster)
- `--no-cutout` - Use full protein structure (slower, more accurate)

After initial run, jobs are submitted via auto-generated `array_submit.run`.

## Architecture

### Main Components

- **groscore.py** - Main orchestrator that reads structure parameters, generates SLURM submission scripts, monitors job completion, and performs final statistical analysis
- **job.run** - Bash script executed per structure as SLURM array task; runs the complete MD workflow

### Utility Scripts (utils/)

| Script | Purpose |
|--------|---------|
| `fix_pdb.py` | Fixes missing atoms and non-standard residues using PDBFixer |
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

### Cutout Mode

By default (`--cutout`), GroScore extracts only interface-proximal residues for faster simulation:
```
conf.gro → make_cutout.py → cutout.pdb → pdb2gmx → conf_cutout.gro → editconf → conf_vacbox.gro
```

With `--no-cutout`, the full protein structure is used:
```
conf.gro → editconf → conf_vacbox.gro
```

## Force Fields

Settings are organized by force field in `settings/<forcefield>/`:

| Setting | GROMOS 54A7 | CHARMM36 |
|---------|-------------|----------|
| Water model | SPC | TIP3P |
| Constraints | all-bonds | h-bonds |
| Coulomb | P3M-AD | PME |
| VdW modifier | none | force-switch |
| Cutoffs | 1.4 nm | 1.2 nm |
| pdb2gmx | interactive (16) | -ff charmm36-jul2022 |

CHARMM36 parameters are bundled in `forcefield/charmm36-jul2022.ff/` (from MacKerell lab).

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
