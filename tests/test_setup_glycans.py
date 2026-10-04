#!/usr/bin/env python3
"""The benchmark setup keeps the oligosaccharides linked to the chains it keeps.

Since the wwPDB carbohydrate remediation every glycan of two or more sugars is a
branched entity with a chain of its own, while a single sugar keeps the chain of its
protein. The setup scripts keep the atoms of the chains a benchmark names, so every
oligosaccharide was dropped and every single sugar kept: 2AJF lost the NAG-NAG-BMA on
ASN A 90 (chain C, its BMA 4.1 A from the RBD) and 2FD6 the glycans on ASN U 52 and
U 172, and the glycan fusion was tested on inputs without them (fusion review F2).
Pinned on a synthetic mmCIF (protein chains A and E kept, X not; branched glycans C
on ASN A 90, G on C, and D on chain X; a single NAG A 501; a metal contact between A
and D that is no covalent link):

  [1] benchmark/glycan_chains.py: linked_glycan_chains() names C and G, not D;
  [2] haddock_benchmark/setup_benchmark.py, offline: input.pdb holds NAG C 1, NAG C 2
      and NAG G 1 as HETATM records besides chains A and E and NAG A 501, and nothing
      of D or X;
  [3] ppb_benchmark and hetatom_benchmark/setup_benchmark.py keep the same records:
      each keeps an atom whose label_asym_id linked_glycan_chains() names.

Run:  python3 tests/test_setup_glycans.py
"""
import io, os, runpy, shutil, sys, tempfile, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "benchmark")
failures = []


def check(name, ok, detail=""):
    print("  %-72s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:400]))
    if not ok:
        failures.append(name)


# (group, auth chain, label asym, entity, residue, number, atoms)
RESIDUES = [("ATOM", "A", "A", "1", "GLY", 89, ("N", "CA", "C", "O")),
            ("ATOM", "A", "A", "1", "ASN", 90, ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2")),
            ("ATOM", "A", "A", "1", "GLY", 91, ("N", "CA", "C", "O")),
            ("ATOM", "E", "B", "2", "GLY", 1, ("N", "CA", "C", "O")),
            ("ATOM", "X", "E", "1", "GLY", 5, ("N", "CA", "C", "O")),
            ("HETATM", "A", "F", "5", "NAG", 501, ("C1", "O5")),
            ("HETATM", "C", "C", "3", "NAG", 1, ("C1", "O4")),
            ("HETATM", "C", "C", "3", "NAG", 2, ("C1", "O4")),
            ("HETATM", "D", "D", "3", "NAG", 1, ("C1", "O4")),
            ("HETATM", "G", "G", "6", "NAG", 1, ("C1", "O5"))]
CONN = [("covale1", "covale", "A", "ASN", "ND2", "A", "C", "NAG", "C1", "C"),
        ("covale2", "covale", "C", "NAG", "O4", "C", "C", "NAG", "C1", "C"),
        ("covale3", "covale", "C", "NAG", "O4", "C", "G", "NAG", "C1", "G"),
        ("covale4", "covale", "E", "GLY", "N", "X", "D", "NAG", "C1", "D"),
        ("metalc1", "metalc", "A", "ASN", "OD1", "A", "D", "NAG", "O4", "D")]


def mmcif():
    cols = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
            "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
            "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num"]
    rows, k = [], 0
    for grp, ch, asym, ent, res, num, atoms in RESIDUES:
        for a in atoms:
            k += 1
            rows.append("%s %d %s %s . %s %s %s %s ? %.3f %.3f %.3f 1.00 20.00 %d %s %s %s 1" % (
                grp, k, a[0], a, res, asym, ent, "." if grp == "HETATM" else str(num), 1.5 * k, 0.0, 0.0, num, res, ch, a))
    ent = ["1 polymer", "2 polymer", "3 branched", "5 non-polymer", "6 branched"]
    conn_cols = ["id", "conn_type_id", "ptnr1_label_asym_id", "ptnr1_label_comp_id", "ptnr1_label_atom_id",
                 "ptnr1_auth_asym_id", "ptnr2_label_asym_id", "ptnr2_label_comp_id", "ptnr2_label_atom_id",
                 "ptnr2_auth_asym_id"]
    return ("data_TST2\n#\nloop_\n_entity.id\n_entity.type\n" + "\n".join(ent) + "\n#\nloop_\n"
            + "".join("_struct_conn.%s\n" % c for c in conn_cols) + "\n".join(" ".join(r) for r in CONN) + "\n#\nloop_\n"
            + "".join("_atom_site.%s\n" % c for c in cols) + "\n".join(rows) + "\n#\n")


def residues_of(path):
    return sorted({(l[:6].strip(), l[17:20], l[21], l[22:26].strip()) for l in open(path)
                   if l.startswith(("ATOM", "HETATM"))}) if os.path.isfile(path) else []


print("[1] linked_glycan_chains()")
sys.path.insert(0, BENCH)
try:
    from glycan_chains import linked_glycan_chains
    got = linked_glycan_chains(mmcif(), {"A", "E"})
except Exception as e:
    got = "%s: %s" % (type(e).__name__, e)
check("C (on ASN A 90) and G (on C), not D (on X, and a metal contact to A)", got == {"C", "G"}, got)

print("\n[2] haddock_benchmark/setup_benchmark.py, offline")
tmp = tempfile.mkdtemp(prefix="setup_glycans_")
try:
    d = os.path.join(tmp, "haddock")
    os.makedirs(d)
    open(os.path.join(d, "benchmark.csv"), "w").write("pdb_id,chain_id_1,chain_id_2,chain_A,chain_E\nTST2,A,E,GNG,G\n")
    cif = mmcif().encode()
    real, cwd, out = urllib.request.urlopen, os.getcwd(), sys.stdout
    urllib.request.urlopen = lambda url, timeout=None: io.BytesIO(cif)
    try:
        os.chdir(d)
        sys.stdout = io.StringIO()
        runpy.run_path(os.path.join(BENCH, "haddock_benchmark", "setup_benchmark.py"), run_name="__main__")
    except Exception as e:
        sys.stdout = out
        print("  setup_benchmark.py raised %s: %s" % (type(e).__name__, e))
    finally:
        sys.stdout = out
        os.chdir(cwd)
        urllib.request.urlopen = real
    res = residues_of(os.path.join(d, "TST2", "input.pdb"))
    het = [r for r in res if r[0] == "HETATM"]
    check("NAG C 1, NAG C 2, NAG G 1 and NAG A 501 as HETATM records",
          het == [("HETATM", "NAG", "A", "501"), ("HETATM", "NAG", "C", "1"), ("HETATM", "NAG", "C", "2"),
                  ("HETATM", "NAG", "G", "1")], het)
    check("  chains A and E kept, nothing of D or X", {r[2] for r in res} == {"A", "C", "E", "G"}, res)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n[3] the ppb and hetatom setup scripts")
for name in ("ppb_benchmark", "hetatom_benchmark"):
    src = open(os.path.join(BENCH, name, "setup_benchmark.py")).read()
    check("%s keeps the atoms of the instances linked_glycan_chains() names" % name,
          "linked_glycan_chains(cif_content, chains_to_keep)" in src
          and "fields[label_chain_col] not in glycan_asyms" in src, name)

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all setup glycan checks passed")
