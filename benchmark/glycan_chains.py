#!/usr/bin/env python3
#
# glycan_chains.py - The oligosaccharides of an mmCIF entry linked to the kept chains
#
# Since the wwPDB carbohydrate remediation every glycan of two or more sugars is a
# branched entity with a chain of its own, while a single sugar keeps the chain of its
# protein. The benchmark setup scripts keep the atoms of the chains a benchmark names,
# so they dropped every oligosaccharide and kept every single sugar: in 2AJF the
# NAG-NAG-BMA on ASN A 90 (chain C), whose BMA lies 4.1 A from the RBD, and the
# glycans on ASN U 52 and U 172 of 2FD6 (fusion review F2). The setup scripts keep
# the instances linked_glycan_chains() names as well; fuse_glycans.py then decides
# whether a glycan is fused with its amino acid.
#
# Standard library only (utils/ccd_cache.py reads the mmCIF).
#

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from ccd_cache import parse_cif  # noqa: E402


def linked_glycan_chains(cif_text, kept):
    """The label_asym_ids of the branched entity instances (_entity.type 'branched',
    the oligosaccharides) that a covalent _struct_conn row links to an atom of a
    chain whose auth_asym_id is in `kept`, directly or through another such instance."""
    d = parse_cif(cif_text)
    branched = {e for e, t in zip(d.get("_entity.id", []), d.get("_entity.type", [])) if t.lower() == "branched"}
    auth, sugar = {}, set()
    for asym, ent, ch in zip(d.get("_atom_site.label_asym_id", []), d.get("_atom_site.label_entity_id", []),
                             d.get("_atom_site.auth_asym_id", [])):
        auth.setdefault(asym, ch)
        if ent in branched:
            sugar.add(asym)
    links = [(a, b) for t, a, b in zip(d.get("_struct_conn.conn_type_id", []),
                                       d.get("_struct_conn.ptnr1_label_asym_id", []),
                                       d.get("_struct_conn.ptnr2_label_asym_id", [])) if t.lower() == "covale"]
    out, grown = set(), True
    while grown:
        grown = False
        for a, b in links:
            for x, y in ((a, b), (b, a)):
                if y in sugar and y not in out and auth.get(y) not in kept and (auth.get(x) in kept or x in out):
                    out.add(y)
                    grown = True
    return out
