#!/usr/bin/env python3
#
# cap_termini.py - Add ACE/NME capping residues to all fragment termini
#
# Usage: python cap_termini.py -f input.pdb -o output.pdb [-m chain_map.gs]
#

import os
import sys
import random
import argparse
import tempfile
import openmm
from pdbfixer import PDBFixer
from openmm import unit

# Deterministic caps (audit SH4). Three runs of the v1.4 script on the identical
# 2FD6 fixed.pdb gave three different fixed_capped.pdb: 663-892 of 5163 non-cap
# heavy atoms differed by more than 0.1 A, up to 2.2-2.8 A, and make_cutout then
# cut 348, 348 and 355 residues. The cap atoms start from random positions
# (Modeller.addHydrogens draws from Python's random), PDBFixer's integrator was
# unseeded, and both ran on whatever platform OpenMM picked (CUDA if present) with
# every thread. Output became byte-identical only with all four pinned: Python
# random seeded, addMissingAtoms(seed), the CPU platform and one thread. One
# thread costs about 30 s per call on the 2FD6 full model and 18-23 s on its
# cutout (v1.4: 9 s on 24 threads); repeated runs are byte-identical.
CAP_SEED = 1
random.seed(CAP_SEED)
CPU = openmm.Platform.getPlatformByName('CPU')
CPU.setPropertyDefaultValue('Threads', '1')

# Ion residue names supported by all GroScore force fields
ION_RESIDUES = {"BA", "CA", "CD", "CL", "CO", "CS", "CU", "CU1", "FE", "FE2", "HG", "K", "LI", "MG", "MN", "NA", "NI", "PB", "SD", "SR", "ZN"}

parser = argparse.ArgumentParser(description="Add ACE/NME caps to fragment termini using PDBFixer")
parser.add_argument('-f', '--file', type=str, required=True, help="Input PDB file")
parser.add_argument('-o', '--output', type=str, required=True, help="Output PDB file")
parser.add_argument('-m', '--chainmap', type=str, default=None, help="Chain map file to update (chain_map.gs)")
parser.add_argument('--rename-nme-carbon', action='store_true', help="Rename NME carbon from C to CH3 (for CHARMM36)")
parser.add_argument('--ace-only', action='store_true', help="Only add ACE caps (skip NME), for use with COOH C-term patches")
parser.add_argument('--rename-ace-carbon', action='store_true', help="Rename ACE methyl carbon from CH3 to CA (for GROMOS)")
args = parser.parse_args()

if not os.path.isfile(args.file):
    print(f"Error: PDB file '{args.file}' not found.", file=sys.stderr)
    sys.exit(1)

# Read original chain_map.gs before modifying anything
original_b_resnums = set()
original_chain_header = ""  # preserve "chain(s): B,C" line for merge_ligand.py
if args.chainmap and os.path.isfile(args.chainmap):
    with open(args.chainmap, "r") as f:
        for line in f:
            if "chain(s):" in line:
                original_chain_header = line.rstrip()
            elif not line.strip().startswith("#"):
                try:
                    original_b_resnums.add(int(line.strip()))
                except (ValueError, IndexError):
                    pass

# Read original ion_residues.gs (will be updated with new residue numbers)
original_ion_resnums = set()
ion_residues_path = os.path.join(os.path.dirname(args.chainmap) if args.chainmap else ".", "ion_residues.gs")
if os.path.isfile(ion_residues_path):
    with open(ion_residues_path, "r") as f:
        for line in f:
            if not line.strip().startswith("#"):
                try:
                    original_ion_resnums.add(int(line.strip()))
                except (ValueError, IndexError):
                    pass

# Read original PDB to map residue numbers to chain IDs
# We need this to determine protein A/B membership for ACE/NME caps
# Also separate ion lines (PDBFixer cannot handle single-atom ion residues)
orig_resnum_to_chain = {}
ion_lines = []
with open(args.file, "r") as f:
    for line in f:
        if line.startswith("ATOM"):
            chain_id = line[21]
            resname = line[17:20].strip()
            try:
                resnum = int(line[22:26].strip())
                if resnum not in orig_resnum_to_chain:
                    orig_resnum_to_chain[resnum] = chain_id
                if resname in ION_RESIDUES:
                    ion_lines.append(line)
            except (ValueError, IndexError):
                pass

# Write protein-only PDB for PDBFixer (strip ions and their surrounding TER records)
# Removing TER records adjacent to ions prevents PDBFixer from seeing a false chain break
protein_only_path = None
if ion_lines:
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
    protein_only_path = tmp.name
    # Read all lines to identify which TER records are adjacent to ion ATOM lines
    all_lines = open(args.file).readlines()
    skip_lines = set()
    for i, line in enumerate(all_lines):
        if line.startswith("ATOM"):
            resname = line[17:20].strip()
            if resname in ION_RESIDUES:
                skip_lines.add(i)
                # Also skip TER records immediately before and after the ion
                if i > 0 and all_lines[i - 1].startswith("TER"):
                    skip_lines.add(i - 1)
                if i + 1 < len(all_lines) and all_lines[i + 1].startswith("TER"):
                    skip_lines.add(i + 1)
    for i, line in enumerate(all_lines):
        if i not in skip_lines:
            tmp.write(line)
    tmp.close()
    pdbfixer_input = protein_only_path
else:
    pdbfixer_input = args.file

# Load PDB with PDBFixer (CPU, one thread: see CAP_SEED)
fixer = PDBFixer(filename=pdbfixer_input, platform=CPU)
fixer.findMissingResidues()
fixer.missingResidues = {}  # Clear auto-detected missing residues

# Add ACE/NME caps to each chain's termini
for chain in fixer.topology.chains():
    residues = list(chain.residues())
    if len(residues) == 0:
        continue
    # Skip if already capped (e.g. from a previous capping step)
    if residues[0].name != 'ACE':
        fixer.missingResidues[(chain.index, 0)] = ['ACE']
    if not args.ace_only:
        if residues[-1].name != 'NME':
            fixer.missingResidues[(chain.index, len(residues))] = ['NME']

fixer.findMissingAtoms()
# PDBFixer holds every existing atom massless while it places the new ones (the
# caps), so this step leaves the input coordinates exactly as they were.
fixer.addMissingAtoms(seed=CAP_SEED)

# Strip extra terminal atoms that conflict with caps
# PDBFixer adds ACE/NME but doesn't remove N-terminal H2/H3 or C-terminal OXT
# from the adjacent residue, causing template mismatches in OpenMM
from openmm.app import Modeller as _Modeller
atoms_to_delete = []
for chain in fixer.topology.chains():
    residues = list(chain.residues())
    if len(residues) < 2:
        continue
    # If first residue is ACE, strip extra N-terminal H from second residue
    if residues[0].name == 'ACE':
        next_res = residues[1]
        for atom in next_res.atoms():
            if atom.name in ('H2', 'H3'):
                atoms_to_delete.append(atom)
    # If last residue is NME/NHE, strip OXT from second-to-last residue
    if residues[-1].name in ('NME', 'NHE'):
        prev_res = residues[-2]
        for atom in prev_res.atoms():
            if atom.name == 'OXT':
                atoms_to_delete.append(atom)
if atoms_to_delete:
    modeller_cleanup = _Modeller(fixer.topology, fixer.positions)
    modeller_cleanup.delete(atoms_to_delete)
    fixer.topology = modeller_cleanup.topology
    fixer.positions = modeller_cleanup.positions
    print(f"Removed {len(atoms_to_delete)} extra terminal atom(s) conflicting with caps")

# Brief OpenMM energy minimization of cap atoms to resolve steric clashes
# PDBFixer places ACE/NME atoms from templates without checking for clashes,
# which can result in cap hydrogens overlapping backbone atoms (e.g., ACE H1 on CA)
from io import StringIO
import numpy as np
from openmm import app as mmapp

cap_residue_names = {'ACE', 'NME', 'NHE'}
cap_atom_indices = set()
for atom in fixer.topology.atoms():
    if atom.residue.name in cap_residue_names:
        cap_atom_indices.add(atom.index)

# Residues without an amber14 template (v1.5). OpenMM bonds only the residues it
# knows, and amber14-all has templates only for those. An NCAA (2MTA TRQ, 2BTF HIC),
# a fused glycan residue (NG1, Asn with its NAG) or a CYM that fix_ion_protonation.py
# made from an ion-binding Cys and make_cutout.py hands back without hydrogens
# (addHydrogens has no CYM entry; 1E6E, 2MTA cutouts) got neither internal bonds nor
# the peptide bond from the residue before, so createSystem failed on that neighbour
# ("No template found for residue 34 (THR) ... missing 1 C atom"), the exception
# below caught it, and EVERY cap of the structure kept PDBFixer's raw position: 7 cap
# heavy-atom contacts below 1.8 A down to 1.11 A on the fused 2AJF full model, 16
# down to 0.85 A on its cutout, 6-20 on the 2BTF, 2MTA and 1E6E files. Now the bonds
# OpenMM left out are made first, and every residue that still matches no template
# gets one built from its own atoms and bonds (GenericTemplates; instances of one
# name with different atom counts get one each). Its N, CA, C and O take the amber14
# backbone types, so a cap bonded to it keeps the amber14 bond, angle and torsion
# terms across the junction; every other atom is an uncharged Lennard-Jones sphere of
# its element. All of it is massless like every non-cap atom: it never moves and only
# shapes the space the caps relax in. A structure without such residues makes no
# bond and never calls the generator, so its output is byte for byte what it was.
# Measured on the 92 fixed.pdb and cutout.pdb files of the 46 HADDOCK structures: the
# 5 that failed (1E6E cutout, 2BTF and 2MTA both) now minimise, contacts below 1.8 A
# 6-20 -> 0-3, and the other 87 give the same bytes as before. On the fused 2AJF the
# full model goes from 7 to 0 such contacts and its cutout from 16 to 1.
#
# Covalent radii (A; Cordero et al. 2008, merge_ligand.py's values for C, N, O, S, P
# and Se) for bonding a residue that OpenMM left without any bond: two atoms closer
# than r1 + r2 + BOND_SLACK_A are bonded, a hydrogen only to its nearest such atom.
# Any other element counts as 0.80 A.
BOND_RADII_A = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07,
                "S": 1.05, "Cl": 1.02, "Se": 1.20, "Br": 1.20, "I": 1.39}
BOND_RADIUS_OTHER_A = 0.80
BOND_SLACK_A = 0.35
# A peptide bond C(i)-N(i+1) that OpenMM did not make is made below 0.2 nm (the bond
# is 0.133 nm), and between a cap and its neighbour wherever PDBFixer put the cap.
PEPTIDE_BOND_MAX_NM = 0.2
# amber14 types of a generic residue's backbone, by atom name and element.
BACKBONE_TYPES = {("N", "N"): "protein-N", ("CA", "C"): "protein-CX",
                  ("C", "C"): "protein-C", ("O", "O"): "protein-O"}
# Lennard-Jones sigma (nm) and epsilon (kJ/mol) of every other generic atom, by
# element: amber14 HC, CT, N and S, parm10 OS, F and P, GAFF cl, br and i, Se as S.
# Any other element, or an atom without one, is a carbon-sized sphere.
GENERIC_LJ = {"H": (0.2649532787749369, 0.0656888), "C": (0.3399669508423535, 0.4577296),
              "N": (0.3249998523775958, 0.71128), "O": (0.3000012343465, 0.71128),
              "F": (0.3118145513, 0.255224), "P": (0.3741774616, 0.8368),
              "S": (0.35635948725613575, 1.046), "Cl": (0.3470941, 1.10876),
              "Se": (0.35635948725613575, 1.046), "Br": (0.3955590, 1.33888),
              "I": (0.4187224, 1.6736)}
GENERIC_LJ_OTHER = GENERIC_LJ["C"]


def make_missing_bonds(topology, positions):
    """Make the bonds OpenMM left out, so that every residue reaches the template
    generator with its real graph: inside each residue that has no bond at all, by
    covalent distance, and the peptide bond C(i)-N(i+1) of a chain where there is
    none, when the two atoms lie within PEPTIDE_BOND_MAX_NM or one of the residues is
    a cap. Returns the number of bonds made."""
    xyz = np.array([p.value_in_unit(unit.nanometer) for p in positions])
    have = {tuple(sorted((a.index, b.index))) for a, b in topology.bonds()}
    bonded_inside = {a.residue.index for a, b in topology.bonds() if a.residue is b.residue}
    new = []

    def bond(a, b):
        key = tuple(sorted((a.index, b.index)))
        if key not in have:
            have.add(key)
            new.append((a, b))

    def radius(atom):
        return BOND_RADII_A.get(atom.element.symbol if atom.element else "", BOND_RADIUS_OTHER_A)

    for res in topology.residues():
        atoms = list(res.atoms())
        if len(atoms) < 2 or res.index in bonded_inside:
            continue
        heavy = [a for a in atoms if a.element is None or a.element.symbol != "H"]
        for i, a in enumerate(heavy):
            for b in heavy[i + 1:]:
                d = 10.0 * np.linalg.norm(xyz[a.index] - xyz[b.index])
                if d < radius(a) + radius(b) + BOND_SLACK_A:
                    bond(a, b)
        for h in atoms:
            if h.element is None or h.element.symbol != "H" or not heavy:
                continue
            d, k = min((10.0 * np.linalg.norm(xyz[h.index] - xyz[a.index]), k)
                       for k, a in enumerate(heavy))
            if d < radius(h) + radius(heavy[k]) + BOND_SLACK_A:
                bond(h, heavy[k])
    for chain in topology.chains():
        residues = list(chain.residues())
        for r1, r2 in zip(residues, residues[1:]):
            c = [a for a in r1.atoms() if a.name == "C"]
            n = [a for a in r2.atoms() if a.name == "N"]
            if len(c) != 1 or len(n) != 1:
                continue
            if (np.linalg.norm(xyz[c[0].index] - xyz[n[0].index]) < PEPTIDE_BOND_MAX_NM
                    or r1.name in cap_residue_names or r2.name in cap_residue_names):
                bond(c[0], n[0])
    for a, b in new:
        topology.addBond(a, b)
    return len(new)


class GenericTemplates:
    """ForceField template generator: a residue that matches no amber14 template
    gets one built from its own atoms and bonds, typed as described above. OpenMM
    calls it once per residue graph it cannot match, so the four NG1 of 2AJF share
    one template and a truncated instance of an NCAA gets its own. A residue it
    cannot template is named in self.failed, and createSystem then raises."""

    def __init__(self):
        self.names = []        # residue name of each template made, in order
        self.failed = []       # 'NAME CHAIN NUM: reason' of each residue it could not template
        self.types = set()     # generic atom types registered so far
        self.count = 0

    def __call__(self, forcefield, residue):
        self.count += 1
        try:
            xml, new_types = self.ffxml(forcefield, residue)
            forcefield.loadFile(StringIO(xml))
        except Exception as e:
            self.failed.append("%s %s %s%s: %s" % (residue.name, residue.chain.id, residue.id,
                                                   residue.insertionCode.strip(), e))
            return False
        self.types.update(new_types)
        self.names.append(residue.name)
        return True

    def ffxml(self, forcefield, residue):
        """The template as force field XML, with any generic atom type not yet
        registered first, and the names of those types. Atoms are named by index:
        OpenMM matches templates by element and bond graph, so the residue's own
        names (which may repeat) need not be valid here."""
        atoms = list(residue.atoms())
        index = {a.index: i for i, a in enumerate(atoms)}
        types, new_types = [], []
        for a in atoms:
            symbol = a.element.symbol if a.element is not None else None
            t = BACKBONE_TYPES.get((a.name, symbol))
            if t is None:
                t = "gscap-%s" % (symbol or "X")
                if t not in self.types and t not in [n for n, _ in new_types]:
                    new_types.append((t, a.element))
            types.append(t)
        xml = ["<ForceField>"]
        if new_types:
            xml.append("<AtomTypes>")
            for t, el in new_types:
                xml.append('<Type name="%s" class="%s"%s mass="%r"/>'
                           % (t, t, ' element="%s"' % el.symbol if el is not None else "",
                              el.mass.value_in_unit(unit.dalton) if el is not None else 12.0))
            xml.append("</AtomTypes>")
        xml.append('<Residues><Residue name="GSCAP%d_%s">' % (self.count, residue.name))
        for i, t in enumerate(types):
            xml.append('<Atom name="a%d" type="%s" charge="0"/>' % (i, t))
        inside, outside = set(), set()
        for a, b in residue.chain.topology.bonds():
            ia, ib = index.get(a.index), index.get(b.index)
            if ia is not None and ib is not None:
                inside.add(tuple(sorted((ia, ib))))
            elif ia is not None or ib is not None:
                outside.add((ia, b.index) if ia is not None else (ib, a.index))
        xml += ['<Bond from="%d" to="%d"/>' % p for p in sorted(inside)]
        xml += ['<ExternalBond from="%d"/>' % p[0] for p in sorted(outside)]
        xml.append("</Residue></Residues>")
        if new_types:
            nb = [g for g in forcefield.getGenerators() if isinstance(g, mmapp.forcefield.NonbondedGenerator)][0]
            xml.append('<NonbondedForce coulomb14scale="%r" lj14scale="%r">' % (nb.coulomb14scale, nb.lj14scale))
            for t, el in new_types:
                sigma, epsilon = GENERIC_LJ.get(el.symbol if el is not None else "", GENERIC_LJ_OTHER)
                xml.append('<Atom type="%s" charge="0" sigma="%r" epsilon="%r"/>' % (t, sigma, epsilon))
            xml.append("</NonbondedForce>")
        xml.append("</ForceField>")
        return "\n".join(xml), [t for t, _ in new_types]

    def summary(self):
        """'NG1' or 'TRQ (2 templates), CYM': the templated residue names in order."""
        order = list(dict.fromkeys(self.names))
        return ", ".join(n if self.names.count(n) == 1 else "%s (%d templates)" % (n, self.names.count(n))
                         for n in order)


if cap_atom_indices:
    # Build modeller with hydrogens for AMBER14 compatibility. A residue amber14 has
    # no template for gets a generic one (see above); a failure that remains, such
    # as a residue the generator cannot template, falls back to the raw caps and says so.
    generic = GenericTemplates()
    try:
        ff_mm = mmapp.ForceField('amber14-all.xml')
        ff_mm.registerTemplateGenerator(generic)
        n_made = make_missing_bonds(fixer.topology, fixer.positions)
        if n_made:
            print(f"Cap minimization: made {n_made} bond(s) OpenMM left out")
        modeller = mmapp.Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(ff_mm, platform=CPU)

        # Build mapping: fixer atom index -> modeller atom index (by residue index + atom name)
        # Modeller.addHydrogens() inserts H atoms between existing atoms, so indices don't match
        mm_atom_lookup = {}
        for atom in modeller.topology.atoms():
            key = (atom.residue.index, atom.name)
            mm_atom_lookup[key] = atom.index

        fixer_to_mm = {}
        for atom in fixer.topology.atoms():
            key = (atom.residue.index, atom.name)
            if key in mm_atom_lookup:
                fixer_to_mm[atom.index] = mm_atom_lookup[key]

        # A 1 nm cutoff: only cap atoms move, and an atom beyond 1 nm of a cap
        # cannot clash with it. With all pairs (v1.4's NoCutoff) one thread took
        # 56-66 s per call on the 2FD6 full model, with the cutoff about 30 s; the
        # cap atoms differ between the two by a median 0.03 A, at most 0.27 A.
        system = ff_mm.createSystem(modeller.topology, nonbondedMethod=mmapp.CutoffNonPeriodic,
                                    nonbondedCutoff=1.0*unit.nanometer,
                                    constraints=None, rigidWater=False)
        if generic.names:
            print(f"Cap minimization: generic templates for {generic.summary()}")

        # Only cap atoms move. Every atom of a non-cap residue is massless, which
        # the minimizer never moves, and only cap positions are copied back below.
        # v1.4 held the protein with a 1000 kJ/mol/nm^2 restraint in vacuum and
        # copied every minimized position back: on 2FD6 non-cap heavy atoms moved
        # by a median 0.15 A, 99th percentile 0.9 A, at most 2.6 A, and those
        # coordinates fed pdb2gmx's His assignment and make_cutout's 0.6/2.0 nm
        # cut-offs. The hydrogens Modeller adds to non-cap residues are only
        # stand-ins for clash relief (pdb2gmx -ignh rebuilds all hydrogens), so
        # they stay where Modeller put them too.
        for atom in modeller.topology.atoms():
            if atom.residue.name not in cap_residue_names:
                system.setParticleMass(atom.index, 0.0)

        integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        context = openmm.Context(system, integrator, CPU, {'Threads': '1'})
        context.setPositions(modeller.positions)
        openmm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=200)

        # Map minimized cap positions back to fixer topology using residue+name mapping
        min_positions = context.getState(getPositions=True).getPositions()
        new_positions = list(fixer.positions)
        for fixer_idx, mm_idx in fixer_to_mm.items():
            if fixer_idx in cap_atom_indices:
                new_positions[fixer_idx] = min_positions[mm_idx]
        fixer.positions = new_positions

        n_caps = len([r for r in fixer.topology.residues() if r.name in cap_residue_names])
        print(f"Minimized {n_caps} cap residue(s) to resolve steric clashes")
    except Exception as e:
        why = str(e) + "".join("; no generic template for %s" % f for f in generic.failed)
        print(f"Warning: Cap minimization failed ({why}), using PDBFixer positions as-is")

# Build mapping: for each PDBFixer chain, determine if it belongs to protein B
# All original residues in a PDBFixer chain share the same protein membership
# (TER records create separate chains, and chains don't mix proteins)
chain_is_b = {}
for chain in fixer.topology.chains():
    is_b = None
    for residue in chain.residues():
        if residue.name in ('ACE', 'NME'):
            continue
        # Try to match this residue to an original residue number
        try:
            resnum = int(residue.id)
        except (ValueError, TypeError):
            continue
        if resnum in original_b_resnums:
            is_b = True
            break
        elif resnum in orig_resnum_to_chain:
            is_b = False
            break
    chain_is_b[chain.index] = is_b if is_b is not None else False

# Write custom PDB output with proper formatting
# We need ATOM records (not HETATM), fixed-width format, and TER records
positions = fixer.positions
new_b_resnums = set()
new_ion_resnums = set()
resnum_counter = 0
atom_num = 0
prev_chain_idx = None

with open(args.output, "w") as out:
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if len(residues) == 0:
            continue

        # Write TER between chains (not before first chain)
        if prev_chain_idx is not None:
            out.write("TER\n")

        for residue in residues:
            resnum_counter += 1
            resname = residue.name

            # Determine protein A/B membership
            if chain_is_b.get(chain.index, False):
                new_b_resnums.add(resnum_counter)

            # Determine chain ID from original residue or inherited from chain
            if resname in ('ACE', 'NME'):
                # Inherit chain ID from the chain's original residues
                chain_id = None
                for r in residues:
                    if r.name not in ('ACE', 'NME'):
                        try:
                            orig_resnum = int(r.id)
                            chain_id = orig_resnum_to_chain.get(orig_resnum)
                            if chain_id:
                                break
                        except (ValueError, TypeError):
                            continue
                if chain_id is None:
                    chain_id = 'A'
            else:
                try:
                    orig_resnum = int(residue.id)
                    chain_id = orig_resnum_to_chain.get(orig_resnum, 'A')
                except (ValueError, TypeError):
                    chain_id = 'A'

            for atom in residue.atoms():
                atom_num += 1
                idx = atom.index
                x = positions[idx].value_in_unit(unit.angstroms)
                atomname = atom.name
                # PDBFixer names NME's methyl carbon "C", but CHARMM36 expects "CH3"
                # AMBER19SB expects "C", so only rename if --rename-nme-carbon flag is set
                if args.rename_nme_carbon and resname == 'NME' and atomname == 'C':
                    atomname = 'CH3'
                # PDBFixer names ACE's methyl carbon "CH3", but GROMOS expects "CA"
                # (GROMOS RTP defines atom NAME "CA" with atom TYPE "CH3")
                # CHARMM36 expects "CH3" so this rename is GROMOS-specific
                if args.rename_ace_carbon and resname == 'ACE' and atomname == 'CH3':
                    atomname = 'CA'
                element = atom.element.symbol if atom.element else atomname[0]

                # PDB fixed-width format
                # Columns: 1-6 record, 7-11 serial, 12 space, 13-16 name,
                #          17 altloc, 18-20 resname, 21 space, 22 chainID,
                #          23-26 resSeq, 27 iCode, 28-30 spaces,
                #          31-38 x, 39-46 y, 47-54 z, 55-60 occupancy,
                #          61-66 tempFactor, 77-78 element
                # Atom name: left-align in col 13-16 with leading space for
                # 1-3 char names (standard PDB convention)
                if len(atomname) < 4:
                    atomname_fmt = f" {atomname:<3s}"
                else:
                    atomname_fmt = f"{atomname:<4s}"

                out.write(f"ATOM  {atom_num:5d} {atomname_fmt} {resname:>3s} {chain_id}{resnum_counter:4d}    "
                          f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}"
                          f"  1.00  0.00          {element:>2s}\n")

        prev_chain_idx = chain.index

    # Re-insert structural ion lines
    if ion_lines:
        for ion_line in ion_lines:
            resname = ion_line[17:20].strip()
            resnum = int(ion_line[22:26].strip())
            chain_id = ion_line[21]
            atomname = ion_line[12:16].strip()
            x_coord = float(ion_line[30:38])
            y_coord = float(ion_line[38:46])
            z_coord = float(ion_line[46:54])

            resnum_counter += 1
            atom_num += 1

            # Determine protein B membership
            if resnum in original_b_resnums:
                new_b_resnums.add(resnum_counter)

            # Track ion residue number mapping
            if resnum in original_ion_resnums:
                new_ion_resnums.add(resnum_counter)

            if len(atomname) < 4:
                atomname_fmt = f" {atomname:<3s}"
            else:
                atomname_fmt = f"{atomname:<4s}"

            out.write("TER\n")
            out.write(f"ATOM  {atom_num:5d} {atomname_fmt} {resname:>3s} {chain_id}{resnum_counter:4d}    "
                      f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
                      f"  1.00  0.00          {atomname[0]:>2s}\n")

    out.write("END\n")

# Clean up temp file
if protein_only_path:
    os.unlink(protein_only_path)

# Update chain_map.gs if requested
if args.chainmap:
    with open(args.chainmap, "w") as f:
        f.write("# Sequential residue numbers belonging to protein B\n")
        if original_chain_header:
            f.write(f"{original_chain_header}\n")
        cap_label = "ACE caps" if args.ace_only else "ACE/NME caps"
        f.write(f"# Updated by cap_termini.py with {cap_label}\n")
        for resnum in sorted(new_b_resnums):
            f.write(f"{resnum}\n")

# Update ion_residues.gs with new residue numbers
if new_ion_resnums and os.path.isfile(ion_residues_path):
    with open(ion_residues_path, "w") as f:
        f.write("# Sequential residue numbers of structural ions\n")
        f.write("# Updated by cap_termini.py\n")
        for resnum in sorted(new_ion_resnums):
            f.write(f"{resnum}\n")

cap_type = "ACE caps (no NME)" if args.ace_only else "ACE/NME caps"
print(f"Added {cap_type}, wrote {args.output} ({resnum_counter} residues, {atom_num} atoms)")
if args.chainmap:
    print(f"Updated {args.chainmap} with {len(new_b_resnums)} residues for protein B")
if new_ion_resnums:
    print(f"Updated {ion_residues_path} with {len(new_ion_resnums)} ion(s)")
