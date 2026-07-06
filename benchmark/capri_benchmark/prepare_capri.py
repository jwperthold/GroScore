#!/usr/bin/env python3
"""Prepare CAPRI Score_set poses for GroScore scoring.

Each database/U-Txxx.N.pdb is a multi-model PDB. Every model carries its own
  REMARK 3 RECEPTOR <chains> LIGAND <chains> THETA ...   (chain assignment)
  REMARK 4 FROM <posename> SEQID ...                     (unique pose id)
and protein-only ATOM records with TER between chains (no H/HETATM).

For each target we create a parent folder directly under the GroScore repo root
containing:
  - sp.gs                          : one line per pose  "<posename>\t<ligand chains>"
  - <posename>/input.pdb           : that model's ATOM+TER records (+END)

Protein B (the entity GroScore pulls away) = the LIGAND chains from REMARK 3.

Usage:
  python3 prepare_capri.py            # all targets
  python3 prepare_capri.py T36 T40_1  # only the named targets (validation)
"""
import os, sys, time

REPO_ROOT = "/home/jwperthold/GroScore"
DB = os.path.join(REPO_ROOT, "CAPRI", "database")

# parent_dir_name -> source basename (without extension)
TARGETS = [
    ("T29",   "U-T029.1"),
    ("T30",   "U-T030.1"),
    ("T32",   "U-T032.1"),
    ("T35",   "U-T035.1"),
    ("T36",   "U-T036.1"),
    ("T37",   "U-T037.1"),   # .2 is the SAME structure (C/D homodimer relabel) -> extract once
    ("T38",   "U-T038.1"),
    ("T39",   "U-T039.1"),
    ("T40_1", "U-T040.1"),   # receptor A + ligand C
    ("T40_2", "U-T040.2"),   # receptor B + ligand C  (distinct interface, different coords)
    ("T41",   "U-T041.1"),
    ("T46",   "U-T046.1"),
    ("T47",   "U-T047.1"),
    ("T50",   "U-T050.1"),
    ("T53",   "U-T053.1"),
    ("T54",   "U-T054.1"),
    ("T89",   "U-T089.1"),
    ("T96",   "U-T096.1"),
    ("T97",   "U-T097.1"),
]


def ligand_chains_from_remark3(toks):
    """['REMARK','3','RECEPTOR','A','LIGAND','C','D','THETA',...] -> 'C,D'."""
    li = toks.index("LIGAND")
    ti = toks.index("THETA")
    chains = toks[li + 1:ti]
    return ",".join(chains) if chains else None


def process(parent, src):
    pdb = os.path.join(DB, src + ".pdb")
    parent_path = os.path.join(REPO_ROOT, parent)
    os.makedirs(parent_path, exist_ok=True)

    sp = []          # (posename, ligchains) in file order
    seen = set()     # posename dup guard
    buf = []
    posename = None
    ligchains = None
    nmodel = 0
    warns = 0

    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("TER"):
                buf.append(line)
            elif line.startswith("MODEL"):
                buf = []; posename = None; ligchains = None
            elif line.startswith("REMARK"):
                toks = line.split()
                if len(toks) >= 2 and toks[1] == "3":
                    try:
                        ligchains = ligand_chains_from_remark3(toks)
                    except ValueError:
                        ligchains = None
                elif len(toks) >= 4 and toks[1] == "4" and toks[2] == "FROM":
                    posename = toks[3]
            elif line.startswith("ENDMDL"):
                nmodel += 1
                if not posename or not ligchains:
                    sys.stderr.write("  [WARN] %s model#%d: missing name/ligand "
                                     "(name=%s lig=%s)\n" % (parent, nmodel, posename, ligchains))
                    warns += 1
                elif posename in seen:
                    sys.stderr.write("  [WARN] %s: duplicate pose id %s -- skipped\n"
                                     % (parent, posename))
                    warns += 1
                else:
                    seen.add(posename)
                    d = os.path.join(parent_path, posename)
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, "input.pdb"), "w") as g:
                        g.writelines(buf)
                        g.write("END\n")
                    sp.append((posename, ligchains))
                buf = []; posename = None; ligchains = None

    with open(os.path.join(parent_path, "sp.gs"), "w") as g:
        g.write("# Structure_ID\tChains_for_Protein_B\n")
        for name, ch in sp:
            g.write("%s\t%s\n" % (name, ch))

    distinct = sorted(set(ch for _, ch in sp))
    txt = os.path.join(DB, src + ".txt")
    txt_n = sum(1 for _ in open(txt)) if os.path.isfile(txt) else -1
    flag = "" if (txt_n == -1 or txt_n == len(sp)) else "  <-- COUNT MISMATCH"
    print("%-6s src=%s  models=%d  written=%d  txt=%d  ligandB=%s  warns=%d%s"
          % (parent, src, nmodel, len(sp), txt_n, distinct, warns, flag))
    return len(sp)


def main():
    sel = set(sys.argv[1:])
    t0 = time.time()
    total = 0
    n = 0
    for parent, src in TARGETS:
        if sel and parent not in sel:
            continue
        total += process(parent, src)
        n += 1
    print("\nDONE: %d target(s), %d pose dirs, %.1fs" % (n, total, time.time() - t0))


if __name__ == "__main__":
    main()
