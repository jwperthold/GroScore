#!/usr/bin/env python3
"""parametrize_ncaa.py reads an NCAA's parent residue from the CCD cache first; an
NCAA whose CCD entry cannot be read fails instead of getting ALA's CMAP, and one
whose entry names no parent keeps ALA's CMAP, with a note.

The AMBER19SB CMAP of an NCAA is copied from its parent residue. The parent came
from files.rcsb.org only, after the parametrisation, with ALA as the silent fallback:
on a node without network the TRQ of 2MTA and the HIC of 2BTF got ALA's CMAP,
although forcefield/ccd_cache holds their entries with parents TRP and HIS
(v15_work review L-4). The fix then failed every NCAA whose entry names no parent,
the D- and beta-amino acids among them, which v1.4 parametrized with ALA's CMAP
(review V2). Pinned, with the network refused:

  [cache]  ccd_parent() gives HIS for HIC, MET for MSE and TRP for TRQ from the
           repository cache, without a network request; an uncached NCAA gives no
           parent and says why.
  [none]   D-alanine (the real CCD entry DAL, which names no parent) in a cache
           directory: ccd_parent() gives ALA and says why, and parametrize_ncaa.py
           parametrizes it with ALA's CMAP and a note in the log, not ncaa_failed.gs.
           An ncaa_failed.gs naming DAL that an earlier run left in the directory is
           gone afterwards: the script only wrote that file, so a stale one sent a
           residue that parametrises now down the fallback (critic F3).
  [fail]   parametrize_ncaa.py on a peptide with an NCAA whose CCD entry cannot be
           read (no cache entry, no network) writes it to ncaa_failed.gs and says
           why, without ALA's CMAP. ncaa_failed.gs is the path of an NCAA that cannot
           be parametrized: fix_pdb.py puts in the parent residue when PDBFixer knows
           the NCAA or files.rcsb.org names its parent, and otherwise pdb2gmx stops
           the structure.
  [offline] the deposited 2BTF HIC A 73 between GLU 72 and GLY 74, network refused:
           Strategy 2 takes its template from the CCD entry in the repository cache
           and HIC is parametrized with HIS's CMAP. Strategy 2 downloaded
           files.rcsb.org's HIC_ideal.sdf and nothing else, so on a node without
           network HIC (and the TRQ of 2MTA) went to ncaa_failed.gs and became its
           parent residue without stopping anything (ncaa review P2).

Run:  python3 tests/test_ncaa_parent.py
"""
import ast, os, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(REPO, "utils")
SCRIPT = os.path.join(UTILS, "parametrize_ncaa.py")
failures = []


def check(name, ok, detail=""):
    print("  %-70s %s%s" % (name, "PASS" if ok else "FAIL", "" if ok else "  " + str(detail)[:300]))
    if not ok:
        failures.append(name)


SRC = open(SCRIPT).read()
sys.path.insert(0, UTILS)
import ccd_cache  # noqa: E402

calls = []


def no_network(*a, **k):
    calls.append(a[0] if a else k)
    raise OSError("network disabled in this test")


ccd_cache.urllib.request.urlopen = no_network
os.environ.pop("GROSCORE_CCD_CACHE", None)

print("\n[cache] the parent from the CCD cache")
fn = next((n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "ccd_parent"), None)
check("parametrize_ncaa.py defines ccd_parent()", fn is not None)
if fn is not None:
    ns = {"ccd_cache": ccd_cache}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SCRIPT, "exec"), ns)
    for res, want in (("HIC", "HIS"), ("MSE", "MET"), ("TRQ", "TRP")):
        got = ns["ccd_parent"](res)
        check("%s: parent %s from the cache" % (res, want), got == (want, None), got)
    check("  with no network request", not calls, calls)
    got = ns["ccd_parent"]("ZQ9")
    check("an uncached NCAA without network: no parent, and why",
          got[0] is None and "not in the CCD cache" in (got[1] or ""), got)


def run_peptide(resname, cache=None, stale=False, pdb=None):
    """parametrize_ncaa.py --gmx-ff amber19sb on ALA-<resname>-ALA (or on the PDB text
    `pdb`), network refused, with GROSCORE_CCD_CACHE=cache, and with a stale
    ncaa_failed.gs naming <resname> in the directory when stale: (process,
    ncaa_failed.gs names, ncaa_residues.gs names)."""
    d = tempfile.mkdtemp(prefix="ncaa_parent_")
    try:
        rows = []
        k = 0
        for rn, res, dx in ((1, "ALA", 0.0), (2, resname, 3.8), (3, "ALA", 7.6)):
            for name, x, y, z in (("N", 0.0, 0.0, 0.0), ("CA", 1.46, 0.0, 0.0), ("C", 2.0, 1.4, 0.0),
                                  ("O", 1.3, 2.4, 0.0), ("CB", 2.0, -0.8, 1.2)):
                k += 1
                rows.append("ATOM  %5d  %-3s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s"
                            % (k, name, res, rn, x + dx, y, z, name[0]))
        open(os.path.join(d, "fixed.pdb"), "w").write(pdb or ("\n".join(rows) + "\nEND\n"))
        if stale:
            open(os.path.join(d, "ncaa_failed.gs"), "w").write(resname + "\n")
        env = {k_: v for k_, v in os.environ.items() if k_ != "GROSCORE_CCD_CACHE"}
        if cache:
            env["GROSCORE_CCD_CACHE"] = cache
        # No network: every request goes to a closed local port and is refused at once.
        env.update(https_proxy="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                   http_proxy="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9", no_proxy="", NO_PROXY="")
        p = subprocess.run([sys.executable, SCRIPT, "-f", "fixed.pdb", "--gmx-ff", "amber19sb"], cwd=d, env=env,
                           capture_output=True, text=True, timeout=900)
        names = []
        for f in ("ncaa_failed.gs", "ncaa_residues.gs"):
            path = os.path.join(d, f)
            names.append([l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
                         if os.path.isfile(path) else [])
        return p, names[0], names[1]
    finally:
        shutil.rmtree(d, ignore_errors=True)


print("\n[none] an entry that names no parent keeps ALA's CMAP, with a note")
# The real DAL entry from files.rcsb.org (2026-10-04) up to its bond loop, with the
# trailing blanks stripped; the records after it are not read.
DAL_CIF = (
    'data_DAL\n'
    '#\n'
    '_chem_comp.id                                    DAL\n'
    '_chem_comp.name                                  D-ALANINE\n'
    '_chem_comp.type                                  "D-PEPTIDE LINKING"\n'
    '_chem_comp.pdbx_type                             ATOMP\n'
    '_chem_comp.formula                               "C3 H7 N O2"\n'
    '_chem_comp.mon_nstd_parent_comp_id               ?\n'
    '_chem_comp.pdbx_synonyms                         ?\n'
    '_chem_comp.pdbx_formal_charge                    0\n'
    '_chem_comp.pdbx_initial_date                     1999-07-08\n'
    '_chem_comp.pdbx_modified_date                    2024-09-27\n'
    '_chem_comp.pdbx_ambiguous_flag                   N\n'
    '_chem_comp.pdbx_release_status                   REL\n'
    '_chem_comp.pdbx_replaced_by                      ?\n'
    '_chem_comp.pdbx_replaces                         ?\n'
    '_chem_comp.formula_weight                        89.093\n'
    '_chem_comp.one_letter_code                       A\n'
    '_chem_comp.three_letter_code                     DAL\n'
    '_chem_comp.pdbx_model_coordinates_details        ?\n'
    '_chem_comp.pdbx_model_coordinates_missing_flag   N\n'
    '_chem_comp.pdbx_ideal_coordinates_details        ?\n'
    '_chem_comp.pdbx_ideal_coordinates_missing_flag   N\n'
    '_chem_comp.pdbx_model_coordinates_db_code        ?\n'
    '_chem_comp.pdbx_subcomponent_list                ?\n'
    '_chem_comp.pdbx_processing_site                  RCSB\n'
    '_chem_comp.pdbx_pcm                              Y\n'
    '#\n'
    'loop_\n'
    '_chem_comp_atom.comp_id\n'
    '_chem_comp_atom.atom_id\n'
    '_chem_comp_atom.alt_atom_id\n'
    '_chem_comp_atom.type_symbol\n'
    '_chem_comp_atom.charge\n'
    '_chem_comp_atom.pdbx_align\n'
    '_chem_comp_atom.pdbx_aromatic_flag\n'
    '_chem_comp_atom.pdbx_leaving_atom_flag\n'
    '_chem_comp_atom.pdbx_stereo_config\n'
    '_chem_comp_atom.pdbx_backbone_atom_flag\n'
    '_chem_comp_atom.pdbx_n_terminal_atom_flag\n'
    '_chem_comp_atom.pdbx_c_terminal_atom_flag\n'
    '_chem_comp_atom.model_Cartn_x\n'
    '_chem_comp_atom.model_Cartn_y\n'
    '_chem_comp_atom.model_Cartn_z\n'
    '_chem_comp_atom.pdbx_model_Cartn_x_ideal\n'
    '_chem_comp_atom.pdbx_model_Cartn_y_ideal\n'
    '_chem_comp_atom.pdbx_model_Cartn_z_ideal\n'
    '_chem_comp_atom.pdbx_component_atom_id\n'
    '_chem_comp_atom.pdbx_component_comp_id\n'
    '_chem_comp_atom.pdbx_ordinal\n'
    'DAL N   N   N 0 1 N N N Y Y N 17.005 31.325 27.552 -1.564 -0.992 0.101  N   DAL 1\n'
    'DAL CA  CA  C 0 1 N N R Y N N 17.495 30.650 26.346 -0.724 0.176  0.402  CA  DAL 2\n'
    'DAL CB  CB  C 0 1 N N N N N N 16.859 31.287 25.124 -1.205 1.374  -0.420 CB  DAL 3\n'
    'DAL C   C   C 0 1 N N N Y N Y 17.165 29.151 26.377 0.709  -0.132 0.051  C   DAL 4\n'
    'DAL O   O   O 0 1 N N N Y N Y 16.244 28.758 27.139 1.001  -1.213 -0.403 O   DAL 5\n'
    'DAL OXT OXT O 0 1 N Y N Y N Y 17.840 28.394 25.631 1.660  0.795  0.243  OXT DAL 6\n'
    'DAL H   H   H 0 1 N N N Y Y N 16.893 30.656 28.287 -1.281 -1.723 0.736  H   DAL 7\n'
    'DAL H2  HN2 H 0 1 N Y N Y Y N 17.663 32.023 27.834 -2.509 -0.741 0.351  H2  DAL 8\n'
    'DAL HA  HA  H 0 1 N N N Y N N 18.589 30.759 26.304 -0.796 0.411  1.464  HA  DAL 9\n'
    'DAL HB1 1HB H 0 1 N N N N N N 16.705 32.361 25.308 -1.133 1.139  -1.481 HB1 DAL 10\n'
    'DAL HB2 2HB H 0 1 N N N N N N 17.521 31.155 24.255 -2.241 1.597  -0.166 HB2 DAL 11\n'
    'DAL HB3 3HB H 0 1 N N N N N N 15.890 30.807 24.923 -0.582 2.240  -0.197 HB3 DAL 12\n'
    'DAL HXT HXT H 0 1 N Y N Y N Y 17.544 27.498 25.738 2.580  0.598  0.018  HXT DAL 13\n'
    '#\n'
    'loop_\n'
    '_chem_comp_bond.comp_id\n'
    '_chem_comp_bond.atom_id_1\n'
    '_chem_comp_bond.atom_id_2\n'
    '_chem_comp_bond.value_order\n'
    '_chem_comp_bond.pdbx_aromatic_flag\n'
    '_chem_comp_bond.pdbx_stereo_config\n'
    '_chem_comp_bond.pdbx_ordinal\n'
    'DAL N   CA  SING N N 1\n'
    'DAL N   H   SING N N 2\n'
    'DAL N   H2  SING N N 3\n'
    'DAL CA  CB  SING N N 4\n'
    'DAL CA  C   SING N N 5\n'
    'DAL CA  HA  SING N N 6\n'
    'DAL CB  HB1 SING N N 7\n'
    'DAL CB  HB2 SING N N 8\n'
    'DAL CB  HB3 SING N N 9\n'
    'DAL C   O   DOUB N N 10\n'
    'DAL C   OXT SING N N 11\n'
    'DAL OXT HXT SING N N 12\n'
    '#\n'
)
dal_cache = tempfile.mkdtemp(prefix="ncaa_dal_")
try:
    open(os.path.join(dal_cache, "DAL.cif"), "w").write(DAL_CIF)
    if fn is not None:
        os.environ["GROSCORE_CCD_CACHE"] = dal_cache
        try:
            got = ns["ccd_parent"]("DAL")
        finally:
            os.environ.pop("GROSCORE_CCD_CACHE", None)
        check("DAL from a cache directory: ALA, saying that the entry names no parent",
              got == ("ALA", "the CCD entry DAL names no parent residue, so it gets ALA's CMAP"), got)
    p, failed, residues = run_peptide("DAL", dal_cache, stale=True)
    check("parametrize_ncaa.py parametrizes DAL: ncaa_residues.gs, no ncaa_failed.gs",
          p.returncode == 0 and residues == ["DAL"] and failed == [],
          (p.returncode, residues, failed, p.stdout[-300:], p.stderr[-300:]))
    check("  with ALA's CMAP and the note in its log",
          "NOTE: DAL: the CCD entry DAL names no parent residue, so it gets ALA's CMAP" in p.stdout
          and "NCAA DAL: parent residue = ALA" in p.stdout and "Added CMAP for DAL (copied from ALA)" in p.stdout,
          p.stdout[-600:])
finally:
    shutil.rmtree(dal_cache, ignore_errors=True)

print("\n[fail] an NCAA whose CCD entry cannot be read fails")
p, failed, residues = run_peptide("ZQ9")
check("ZQ9 is written to ncaa_failed.gs", failed == ["ZQ9"], (failed, p.stdout[-300:], p.stderr[-300:]))
check("  saying that its parent residue could not be read",
      "ZQ9: no parent residue for its CMAP" in p.stdout, p.stdout[-400:])
check("  and without ALA's CMAP", "NCAA ZQ9: parent residue" not in p.stdout and "Added CMAP for ZQ9" not in p.stdout,
      p.stdout[-400:])

print("\n[offline] Strategy 2 takes its template from the CCD cache")
BTF = os.path.join(REPO, "tests", "data", "glycans", "2BTF_caps.pdb")
seg = "".join("ATOM  " + l[6:] for l in open(BTF) if l.startswith(("ATOM", "HETATM")) and l[21] == "A"
              and l[22:26].strip() in ("72", "73", "74")) + "END\n"
p, failed, residues = run_peptide("HIC", pdb=seg)
check("2BTF HIC A 73, no network: parametrized, no ncaa_failed.gs", p.returncode == 0 and residues == ["HIC"]
      and failed == [], (p.returncode, residues, failed, p.stdout[-400:], p.stderr[-300:]))
check("  Strategy 2's template from the CCD entry in forcefield/ccd_cache, HIS's CMAP",
      "Strategy 2" in p.stdout and "template from the CCD entry HIC (local cache forcefield/ccd_cache/HIC.cif)" in p.stdout
      and "Added CMAP for HIC (copied from HID (AMBER name for HIS))" in p.stdout, p.stdout[-800:])

print()
if failures:
    print("FAILED (%d):" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all NCAA parent checks passed")
