#!/usr/bin/env python3
"""Insertion-code residues survive the setup: setup_benchmark.py writes the code,
renumber_pdb.py gives every residue its own number, fix_pdb.py refuses to merge.

Audit (digest_audit/audit_shared.txt, verifier extra finding): the HADDOCK inputs
carried no insertion-code column, so Kabat 82/82A/82B/82C arrived as four residues
numbered 82; renumber_pdb.py keyed residues by (chain, number) and gave them one
new number; OpenMM's reader in fix_pdb.py fused the consecutive Ser and kept the
second copy's atoms, with 'duplicate atom' warnings only. 2FD6 chain H went from
212 to 211 CA, 1IQD chain B from 200 to 199, both 16-21 A from the antigen.
Pinned here on a synthetic heavy chain H 81, 82, 82A, 82B, 82C, 83 (LEU SER SER
SER LEU THR) and a partner chain U 1, 1A, 2 (GLY ALA GLY):

  1. setup_benchmark.py, run offline on a synthetic mmCIF, writes the codes A, B,
     C, A in column 27 of input.pdb and blank elsewhere; sp.gs gets 'TST1 U'.
  2. renumber_pdb.py numbers the 9 residues 1-9, with or without the code column:
     a new residue starts when the code changes, when the residue name changes
     (82B SER -> 82C LEU) or when an atom name repeats (82A SER -> 82B SER). An
     alternate location B of an atom is not a repeat, even under another residue
     name (B THR beside A SER). Column 27 is cleared, chain_map.gs lists 7, 8, 9.
  3. fix_pdb.py stops (exit 1, no output, a stale output removed) on two residues
     that share chain and number, as the v1.4 renumbering produced; on the new
     renumbering it keeps all 6 chain-H CA atoms (v1.4: 5).

Run:  python3 tests/test_setup_inscode.py
"""

import io, os, runpy, shutil, subprocess, sys, tempfile, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENUM = os.path.join(REPO, "utils", "renumber_pdb.py")
FIX = os.path.join(REPO, "utils", "fix_pdb.py")
SETUP = os.path.join(REPO, "benchmark", "haddock_benchmark", "setup_benchmark.py")
failures = []

# (chain, number, insertion code, residue name), in file order
RESIDUES = [("H", 81, "", "LEU"), ("H", 82, "", "SER"), ("H", 82, "A", "SER"), ("H", 82, "B", "SER"),
            ("H", 82, "C", "LEU"), ("H", 83, "", "THR"), ("U", 1, "", "GLY"), ("U", 1, "A", "ALA"),
            ("U", 2, "", "GLY")]


def check(name, ok, detail=""):
    detail = " | ".join(l.strip() for l in str(detail).splitlines() if l.strip())
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        failures.append(name)


def backbone(k, chain):
    """N, CA, C, O of residue k along x, C(k)-N(k+1) 1.65 A (bonded)."""
    y0 = 0.0 if chain == "H" else 12.0
    x = 3.8 * k
    return [("N", x - 1.0, y0 + 0.8, 0.0), ("CA", x, y0, 0.0), ("C", x + 1.2, y0 + 0.4, 0.0),
            ("O", x + 1.2, y0 + 1.6, 0.0)]


def pdb_lines(icodes=True, altloc=False):
    out, serial = [], 0
    prev = None
    for k, (ch, num, ic, rn) in enumerate(RESIDUES):
        if prev is not None and ch != prev:
            out.append("TER")
        prev = ch
        for name, x, y, z in backbone(k, ch):
            serial += 1
            out.append("ATOM  %5d  %-3s%s%3s %s%4d%s   %8.3f%8.3f%8.3f  1.00  0.00           %s" % (
                serial, name, " ", rn, ch, num, ic if icodes else " ", x, y, z, name[0]))
            if altloc and k == 1 and name == "CA":
                # the same atom again as alternate location B, once more under a
                # second residue name (microheterogeneity): never a new residue
                out[-1] = out[-1][:16] + "A" + out[-1][17:]
                for alt_rn in (rn, "THR"):
                    serial += 1
                    out.append("ATOM  %5d  %-3s%s%3s %s%4d%s   %8.3f%8.3f%8.3f  0.50  0.00           %s" % (
                        serial, name, "B", alt_rn, ch, num, ic if icodes else " ", x + 0.3, y, z, name[0]))
    out.append("END")
    return "\n".join(out) + "\n"


def residues_of(path):
    """[(chain, number, code column, name)] per residue in file order: a residue is
    a run of lines with the same chain, number and code; its name is the one of its
    first line (alternate locations B, C, ... never name a residue)."""
    out, key = [], None
    for l in open(path):
        if l.startswith("ATOM"):
            k = (l[21], l[22:27])
            if k != key and l[16] in " A":
                key = k
                out.append((l[21], int(l[22:26]), l[26], l[17:20].strip()))
    return out


def ca_count(path, chain):
    return sum(1 for l in open(path) if l.startswith("ATOM") and l[12:16] == " CA " and l[21] == chain)


def run(args, cwd):
    p = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def mmcif():
    cols = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
            "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
            "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num"]
    rows, serial = [], 0
    for k, (ch, num, ic, rn) in enumerate(RESIDUES):
        for name, x, y, z in backbone(k, ch):
            serial += 1
            rows.append("ATOM %d %s %s . %s %s 1 %d %s %.3f %.3f %.3f 1.00 20.00 %d %s %s %s 1" % (
                serial, name[0], name, rn, ch, k + 1, ic or "?", x, y, z, num, rn, ch, name))
    return "data_TST1\n#\nloop_\n" + "".join("_atom_site.%s\n" % c for c in cols) + "\n".join(rows) + "\n#\n"


def main():
    for f in (RENUM, FIX, SETUP):
        if not os.path.isfile(f):
            print("missing %s" % f)
            sys.exit(1)
    tmp = tempfile.mkdtemp(prefix="inscode_")
    try:
        # ------------------------------------------------------------ 1
        print("[1] setup_benchmark.py on a synthetic mmCIF (offline)")
        d1 = os.path.join(tmp, "setup")
        os.makedirs(d1)
        with open(os.path.join(d1, "benchmark.csv"), "w") as f:
            f.write("pdb_id,chain_id_1,chain_id_2,chain_H,chain_U\nTST1,H,U,LSSSLT,GAG\n")
        cif = mmcif().encode()
        real_urlopen = urllib.request.urlopen
        urllib.request.urlopen = lambda url, timeout=None: io.BytesIO(cif)
        cwd, out = os.getcwd(), sys.stdout
        try:
            os.chdir(d1)
            sys.stdout = io.StringIO()
            runpy.run_path(SETUP, run_name="__main__")
        finally:
            sys.stdout = out
            os.chdir(cwd)
            urllib.request.urlopen = real_urlopen
        inp = os.path.join(d1, "TST1", "input.pdb")
        res = residues_of(inp) if os.path.isfile(inp) else []
        codes = [r[2] for r in res]
        check("[1] input.pdb codes in column 27: ' ',' ','A','B','C',' ',' ','A',' '",
              codes == [" ", " ", "A", "B", "C", " ", " ", "A", " "], str(codes))
        sp = open(os.path.join(d1, "sp.gs")).read() if os.path.isfile(os.path.join(d1, "sp.gs")) else ""
        check("[1] sp.gs entry TST1 U", "TST1\tU" in sp, repr(sp[-40:]))

        # ------------------------------------------------------------ 2
        print("[2] renumber_pdb.py")
        want = [("H", 1, " ", "LEU"), ("H", 2, " ", "SER"), ("H", 3, " ", "SER"), ("H", 4, " ", "SER"),
                ("H", 5, " ", "LEU"), ("H", 6, " ", "THR"), ("U", 7, " ", "GLY"), ("U", 8, " ", "ALA"),
                ("U", 9, " ", "GLY")]
        renum_dirs = {}
        for tag, text in (("codes", pdb_lines(True)), ("no codes", pdb_lines(False)),
                          ("altloc B", pdb_lines(False, altloc=True)),
                          ("setup output", open(inp).read() if os.path.isfile(inp) else "")):
            d = os.path.join(tmp, "renum_" + tag.replace(" ", "_"))
            os.makedirs(d)
            open(os.path.join(d, "input.pdb"), "w").write(text)
            rc, log = run([RENUM, "-f", "input.pdb", "-o", "renumbered.pdb", "-c", "U"], d)
            got = residues_of(os.path.join(d, "renumbered.pdb")) if rc == 0 else []
            check("[2] %s: 9 residues numbered 1-9, column 27 blank" % tag, got == want, str(got) if got != want else "")
            cm = [l.strip() for l in open(os.path.join(d, "chain_map.gs")) if l.strip() and not l.startswith("#")] \
                if rc == 0 else []
            check("[2] %s: chain_map.gs = 7 8 9" % tag, cm == ["7", "8", "9"], str(cm))
            renum_dirs[tag] = d

        # ------------------------------------------------------------ 3
        print("[3] fix_pdb.py")
        d = os.path.join(tmp, "fix_merged")
        os.makedirs(d)
        # two Ser sharing chain H and number 2, as the v1.4 renumbering wrote them
        merged = []
        for l in open(os.path.join(renum_dirs["no codes"], "renumbered.pdb")):
            if l.startswith("ATOM") and l[21] == "H" and int(l[22:26]) == 3:
                l = l[:22] + "%4d" % 2 + l[26:]
            merged.append(l)
        open(os.path.join(d, "renumbered.pdb"), "w").write("".join(merged))
        open(os.path.join(d, "fixed.pdb"), "w").write("stale output of an earlier run\n")
        rc, log = run([FIX, "-f", "renumbered.pdb", "-o", "fixed.pdb"], d)
        check("[3] merged Ser: exit 1 naming the duplicate atoms", rc == 1 and "duplicate atom" in log, "rc %d %s" % (rc, log[-200:]))
        check("[3] merged Ser: no fixed.pdb left (the stale one removed)", not os.path.exists(os.path.join(d, "fixed.pdb")))
        for tag in ("no codes", "setup output"):
            d = renum_dirs[tag]
            rc, log = run([FIX, "-f", "renumbered.pdb", "-o", "fixed.pdb"], d)
            n = ca_count(os.path.join(d, "fixed.pdb"), "H") if rc == 0 else -1
            check("[3] %s: fixed.pdb keeps all 6 chain-H CA" % tag, rc == 0 and n == 6, "rc %d CA %d %s" % (rc, n, log[-200:] if rc else ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("%d check(s) FAILED:" % len(failures))
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
