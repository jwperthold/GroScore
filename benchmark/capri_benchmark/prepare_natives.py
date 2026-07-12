#!/usr/bin/env python3
"""Prepare the CAPRI native/reference target structures for GroScore scoring.

For each of our 18 benchmark targets, take the scoreset reference complex from
targets/T0NN.M.pdb (protein-only ATOM records, standardized chain IDs) and create
    natives/<target>/input.pdb
plus a target-level natives/sp.gs listing the Protein-B (ligand) pull chains --
the SAME chain assignment used for that target's docked poses (verified against
each pose dir's sp.gs). Scoring these gives the "native" reference GroScore for
each target (the correct complex should score very favourably).

Run from natives/ afterwards:  cd ../natives && python ../groscore.py
"""
import os

REPO = "/home/jwperthold/GroScore"
SRC = os.path.join(REPO, "targets")
OUT = os.path.join(REPO, "natives")

# target -> (reference PDB basename, Protein-B / ligand chains)
# Protein-B verified identical to each target's pose sp.gs; T37/T40 use interface .1
# (the .2 references are the same complex up to a rigid transform).
TARGETS = [
    ("T29", "T029.1", "F"),
    ("T30", "T030.1", "A"),
    ("T32", "T032.1", "C"),
    ("T35", "T035.1", "A"),
    ("T36", "T036.1", "A"),
    ("T37", "T037.1", "C,D"),
    ("T38", "T038.1", "A"),
    ("T39", "T039.1", "A"),
    ("T40", "T040.1", "C"),
    ("T41", "T041.1", "A"),
    ("T46", "T046.1", "A"),
    ("T47", "T047.1", "B"),
    ("T50", "T050.1", "C"),
    ("T53", "T053.1", "A"),
    ("T54", "T054.1", "C"),
    ("T89", "T089.1", "B"),
    ("T96", "T096.1", "C"),
    ("T97", "T097.1", "F"),
]


def prepare(target, pdb):
    src = os.path.join(SRC, pdb + ".pdb")
    d = os.path.join(OUT, target)
    os.makedirs(d, exist_ok=True)
    atoms = [ln for ln in open(src) if ln.startswith(("ATOM", "HETATM", "TER"))]
    with open(os.path.join(d, "input.pdb"), "w") as g:
        g.writelines(atoms)
        g.write("END\n")
    chs = []
    for ln in atoms:
        if ln.startswith("ATOM"):
            c = ln[21]
            if not chs or chs[-1] != c:
                chs.append(c)
    return len([ln for ln in atoms if ln.startswith("ATOM")]), "".join(dict.fromkeys(chs))


os.makedirs(OUT, exist_ok=True)
sp = []
for target, pdb, chains in TARGETS:
    natoms, chs = prepare(target, pdb)
    # sanity: every Protein-B chain must exist in the structure
    missing = [c for c in chains.split(",") if c not in chs]
    flag = "  <-- MISSING %s" % ",".join(missing) if missing else ""
    sp.append((target, chains))
    print("%-5s <- %-9s  atoms=%-5d chains=[%s]  proteinB=%s%s" % (
        target, pdb, natoms, chs, chains, flag))

with open(os.path.join(OUT, "sp.gs"), "w") as g:
    g.write("# Structure_ID\tChains_for_Protein_B\n")
    for target, chains in sp:
        g.write("%s\t%s\n" % (target, chains))

print("\nDONE: %d native targets prepared in %s (+ sp.gs)" % (len(sp), OUT))
