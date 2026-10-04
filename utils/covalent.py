#!/usr/bin/env python3
#
# covalent.py - When two heavy atoms of a deposited model are covalently bonded
#
# One rule for the two places that ask: merge_ligand.py (a ligand bonded to the
# protein or to another ligand is a link the pipeline does not model, so it stops the
# structure) and fuse_glycans.py (a glycan bonded to ASN ND2, SER OG or THR OG1 is
# fused with that amino acid into one residue). Both import it, so 'bonded' means the
# same in both: the links fuse_glycans.py fuses a glycan through are the ones
# merge_ligand.py would call covalent bonds, and the sugars of a glycan it leaves
# unfused are found bonded there. fuse_glycans.py also joins glycosidic links
# deposited longer than this rule, up to 2.5 A (its stretched_links()); the sugars it
# fuses that way are protein records of fused.pdb, which merge_ligand.py reads as
# protein, so they never reach it as ligands.
#
# Standard library only: merge_ligand.py and fuse_glycans.py import it before any
# heavy import.
#

# Covalent radii (A). Two heavy atoms closer than r1 + r2 + COVALENT_SLACK are bonded:
# 1.82 A for C-N, so the 1.45 A N-glycosidic C1-ND2 link of a glycan to its Asn
# counts and a 2.6 A hydrogen bond or a 2.2 A crystal clash does not. Measured on the
# DEPOSITED model (input.pdb), never on conf.gro: cap_termini.py minimises the whole
# protein in vacuum without the ligands, which moves atoms within 4 A of a ligand by
# up to 1.7 A (1H1V ATP pocket) and put a Thr OG1 1.62 A from the gamma-phosphate
# oxygen of the 1Z0K GTP. On the deposited models exactly five benchmark ligands are
# bonded: the N-glycan NAGs of 1MAH (1) and 2AJF (4), C1-ND2 1.45-1.46 A. With ligand
# parametrisation on, fuse_glycans.py fuses these with their Asn.
COVALENT_RADII = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "SE": 1.20}
COVALENT_SLACK = 0.35
# The same rule also caught atoms that overlap without being bonded: in 5HYS five
# sulfates each have an O 1.16-1.30 A from a Ser OG, a Gly N or a Tyr OH, with no LINK
# record and occupancies of 0.55-1.00, and all five were called covalently bonded
# (review L9). A contact shorter than the shortest bond the element pair forms is a
# clash in the deposited model: merge_ligand.py prints it and merges the ligand as
# v1.4 did (minimisation separates the atoms). A link through O-O is a peroxide bond
# (1.45-1.49 A) and one through N-O a single bond (1.40 A), so contacts below 1.40 and
# 1.35 A are clashes; for the other pairs a link is at least 0.85 of the single-bond
# radius sum (C=O 1.20 A, C=N 1.27 A as in a PLP aldimine), so below 0.80 of it.
BOND_MIN = {("O", "O"): 1.40, ("N", "O"): 1.35}
BOND_MIN_FRACTION = 0.80


def contact_kind(e1, e2, d):
    """'bond', 'clash' or None for two heavy atoms of elements e1, e2 d A apart.
    Both elements must be keys of COVALENT_RADII."""
    rsum = COVALENT_RADII[e1] + COVALENT_RADII[e2]
    if d >= rsum + COVALENT_SLACK:
        return None
    return "clash" if d < BOND_MIN.get(tuple(sorted((e1, e2))), BOND_MIN_FRACTION * rsum) else "bond"
