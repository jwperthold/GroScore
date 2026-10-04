#!/usr/bin/env python3
#
# ccd_cache.py - Chemical Component Dictionary entries for ligand bond orders
#
# parametrize_ligand.py takes a ligand's bond orders, formal charges and hydrogens
# from the wwPDB Chemical Component Dictionary (CCD) entry named by its residue
# name whenever that entry describes exactly the deposited heavy atoms. This module
# finds the entry: first in a local cache (forcefield/ccd_cache/ in the repository,
# then every directory in $GROSCORE_CCD_CACHE, colon-separated), and only then at
# files.rcsb.org. Compute nodes often have no network, and without the entry the
# 2026-09 template repairs (1E6E FAD, 1IBR GNP, 2FD6 ETX and PGE) became failures or
# flags; the cache makes the setup independent of the network and pins the CCD
# version a benchmark was built with. Every build says in its itp header which
# entry it used, where it came from and the entry's modification date.
#
# The CCD is released by the wwPDB under CC0 1.0, so the entries can be shipped.
#
# Prefetch (run on a login node, or anywhere with network):
#   python3 utils/ccd_cache.py benchmark/haddock_benchmark/*/input.pdb
#   python3 utils/ccd_cache.py -o /path/to/cache --refresh input1.pdb input2.pdb
# It fetches the entry of every HETATM residue name in the given PDB files, except
# water and the structural ions renumber_pdb.py keeps as ions, into the cache
# (default forcefield/ccd_cache), skipping entries that are already there. Whenever
# there is such a residue it also fetches GLYCAN_PARENTS, the amino acids
# fuse_glycans.py fuses a glycan with: it reads their entries (in-chain hydrogens,
# bond orders) for every glycan it fuses, and a compute node may have no network.
#
# Standard library only: parametrize_ligand.py imports this before any heavy import.
#

import os
import sys
import argparse
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.normpath(os.path.join(HERE, "..", "forcefield", "ccd_cache"))
CACHE_ENV = "GROSCORE_CCD_CACHE"
CCD_URL = "https://files.rcsb.org/ligands/download/%s.cif"
NETWORK_TIMEOUT = 15   # s, as the v1.4 template download

# Residues the prefetch skips: water, and the ions renumber_pdb.py keeps as ions
# (they are never parametrised as ligands).
WATERS = {"HOH", "WAT", "DOD", "H2O", "SOL"}
ION_RESIDUES = {"BA", "CA", "CD", "CL", "CO", "CS", "CU", "CU1", "FE", "FE2", "HG", "K", "LI", "MG",
                "MN", "NA", "NI", "PB", "SD", "SR", "ZN"}
# The amino acids a glycan is fused with (fuse_glycans.py SITES: Asn ND2, Ser OG, Thr
# OG1). Their entries ship in forcefield/ccd_cache.
GLYCAN_PARENTS = ("ASN", "SER", "THR")


def cache_dirs():
    """The cache directories, in lookup order: the repository's, then $GROSCORE_CCD_CACHE."""
    dirs = [DEFAULT_CACHE] + [d for d in os.environ.get(CACHE_ENV, "").split(":") if d]
    out = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def valid_entry(text, ccd_id):
    """Is `text` the CCD component file of `ccd_id`: not an error page, and with its
    bond loop when it has more than one atom? A cache file cut off before the bond
    loop passed as the entry and was built as one radical fragment per atom (review
    L-2); an entry of one atom (IOD) legitimately has no bonds."""
    head = text.lstrip()[:200].splitlines()
    if not (bool(head) and head[0].strip().upper() == "DATA_" + ccd_id.upper() and "_chem_comp_atom." in text):
        return False
    try:
        comp = component(text)
    except Exception:
        return False
    return len(comp["atoms"]) <= 1 or bool(comp["bonds"])


def fetch(ccd_id, network=True, cache=True):
    """(cif text, source) of the CCD entry `ccd_id`, or (None, why not).

    `source` is the cache file's path, or 'files.rcsb.org'. Entries fetched from the
    network are not written into the cache: the cache only changes through the
    prefetch, so a run never alters what the repository ships. cache=False skips the
    cache (for an entry whose cached copy turned out incomplete). A cached file that
    is not a valid entry, or cannot be read, is skipped, and the reason for a miss
    names it: an unreadable file (another user's, mode 0600) used to end the ligand
    in an uncaught PermissionError (review V4)."""
    ccd_id = ccd_id.strip().upper()
    if not ccd_id or not all(c.isalnum() for c in ccd_id):
        return None, "no valid CCD identifier (%r)" % ccd_id
    bad, unread = [], []
    for d in (cache_dirs() if cache else []):
        path = os.path.join(d, ccd_id + ".cif")
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    text = f.read()
            except (OSError, ValueError) as e:
                unread.append("; the cached %s could not be read (%s)" % (path, getattr(e, "strerror", None) or e))
                continue
            if valid_entry(text, ccd_id):
                return text, path
            bad.append(path)
    skipped = ("; the cached %s %s not a complete entry" % (", ".join(bad), "is" if len(bad) == 1 else "are")
               if bad else "") + "".join(unread)
    if not network:
        return None, "%s is not in the CCD cache (%s)%s" % (ccd_id, ", ".join(cache_dirs()), skipped)
    try:
        text = urllib.request.urlopen(CCD_URL % ccd_id, timeout=NETWORK_TIMEOUT).read().decode()
    except Exception as e:
        err = " ".join(str(e).split())[:120]
        if not cache:
            return None, "files.rcsb.org did not answer for %s (%s)" % (ccd_id, err)
        return None, "%s is not in the CCD cache and files.rcsb.org did not answer (%s)%s" % (
            ccd_id, err, skipped)
    if not valid_entry(text, ccd_id):
        return None, "files.rcsb.org has no CCD entry %s%s" % (ccd_id, skipped)
    return text, "files.rcsb.org"


# ---------------------------------------------------------------------------
# A minimal mmCIF reader: enough for the CCD component files (data items, loops,
# quoted values such as "O5'" and "H5''", and ;-delimited text fields).
# ---------------------------------------------------------------------------

def _tokens(text):
    """(value, quoted) for every token of a CIF file, in order."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            yield "\n".join(buf).strip(), True
            continue
        j, n = 0, len(line)
        while j < n:
            c = line[j]
            if c.isspace():
                j += 1
                continue
            if c == "#":
                break
            if c in "'\"":
                # a quote ends a value only when whitespace or the line end follows it
                k = j + 1
                while k < n and not (line[k] == c and (k + 1 == n or line[k + 1].isspace())):
                    k += 1
                yield line[j + 1:k], True
                j = k + 1
                continue
            k = j
            while k < n and not line[k].isspace():
                k += 1
            yield line[j:k], False
            j = k
        i += 1


def parse_cif(text):
    """{'_category.item': [values]} of the first data block."""
    data = {}
    toks = list(_tokens(text))
    i = 0

    def is_keyword(tok):
        v, quoted = tok
        return not quoted and (v.startswith("_") or v.lower() == "loop_" or v.lower().startswith("data_"))

    while i < len(toks):
        v, quoted = toks[i]
        if not quoted and v.lower() == "loop_":
            i += 1
            tags = []
            while i < len(toks) and not toks[i][1] and toks[i][0].startswith("_"):
                tags.append(toks[i][0])
                i += 1
            vals = []
            while i < len(toks) and not is_keyword(toks[i]):
                vals.append(toks[i][0])
                i += 1
            for k, tag in enumerate(tags):
                data[tag] = vals[k::len(tags)]
        elif not quoted and v.startswith("_"):
            data[v] = [toks[i + 1][0]] if i + 1 < len(toks) else [""]
            i += 2
        else:
            i += 1
    return data


def component(text):
    """The parts of a CCD entry the ligand builder needs:
    {'id', 'name', 'modified', 'formal_charge',
     'atoms': [{'name', 'alt', 'element', 'charge', 'leaving'}],
     'bonds': [(name1, name2, 'SING'|'DOUB'|'TRIP'|..., aromatic Y/N)]}"""
    d = parse_cif(text)

    def item(tag, default=""):
        v = d.get(tag, [default])[0]
        return default if v in ("?", ".") else v

    def col(tag, n):
        v = d.get(tag)
        return v if v is not None and len(v) == n else ["?"] * n

    names = d.get("_chem_comp_atom.atom_id", [])
    n = len(names)
    alt = col("_chem_comp_atom.alt_atom_id", n)
    els = col("_chem_comp_atom.type_symbol", n)
    chg = col("_chem_comp_atom.charge", n)
    leave = col("_chem_comp_atom.pdbx_leaving_atom_flag", n)
    atoms = []
    for k in range(n):
        try:
            q = int(chg[k]) if chg[k] not in ("?", ".") else 0
        except ValueError:
            q = 0
        atoms.append({"name": names[k], "alt": alt[k] if alt[k] not in ("?", ".") else names[k],
                      "element": els[k].upper(), "charge": q, "leaving": leave[k].upper() == "Y"})
    a1 = d.get("_chem_comp_bond.atom_id_1", [])
    nb = len(a1)
    a2 = col("_chem_comp_bond.atom_id_2", nb)
    order = col("_chem_comp_bond.value_order", nb)
    arom = col("_chem_comp_bond.pdbx_aromatic_flag", nb)
    bonds = [(a1[k], a2[k], order[k].upper(), arom[k].upper()) for k in range(nb)]
    try:
        fq = int(item("_chem_comp.pdbx_formal_charge", "0"))
    except ValueError:
        fq = 0
    return {"id": item("_chem_comp.id"), "name": item("_chem_comp.name"),
            "modified": item("_chem_comp.pdbx_modified_date"), "formal_charge": fq,
            "atoms": atoms, "bonds": bonds}


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------

def ligand_ids(paths):
    """Sorted HETATM residue names of the given PDB files, water and ions left out."""
    ids = set()
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.startswith("HETATM"):
                    r = line[17:20].strip().upper()
                    if r and r not in WATERS and r not in ION_RESIDUES:
                        ids.add(r)
    return sorted(ids)


def prefetch_ids(paths):
    """The entries the prefetch fetches for these PDB files: ligand_ids(), and with
    them GLYCAN_PARENTS whenever there is a HETATM residue other than water and ions
    (any of them could be a glycan for fuse_glycans.py to fuse)."""
    ids = ligand_ids(paths)
    return sorted(set(ids) | set(GLYCAN_PARENTS)) if ids else ids


def prefetch(ids, outdir, refresh=False):
    """Fetch each entry from files.rcsb.org into outdir; returns (fetched, kept, failed).

    A cached file that cannot be read is fetched again, as fetch() passes over it: the
    unguarded read ended the prefetch in a PermissionError traceback on a file written
    with the 0600 mode of the L-10 code (review N3). os.replace() needs write access to
    the directory only. A file that cannot be written is a failure with the reason."""
    os.makedirs(outdir, exist_ok=True)
    fetched, kept, failed = [], [], []
    for cid in ids:
        path = os.path.join(outdir, cid + ".cif")
        unread = ""
        if os.path.isfile(path) and not refresh:
            try:
                with open(path) as f:
                    text = f.read()
            except (OSError, ValueError) as e:
                unread = "; the cached %s could not be read (%s)" % (path, getattr(e, "strerror", None) or e)
                print("prefetch: %s, fetching %s again" % (unread[2:], cid), file=sys.stderr)
            else:
                if valid_entry(text, cid):
                    kept.append(cid)
                    continue
        try:
            text = urllib.request.urlopen(CCD_URL % cid, timeout=NETWORK_TIMEOUT).read().decode()
        except Exception as e:
            failed.append((cid, " ".join(str(e).split())[:120] + unread))
            continue
        if not valid_entry(text, cid):
            failed.append((cid, "no CCD entry" + unread))
            continue
        # A private temp name: two prefetches into one directory shared '<ID>.cif.tmp'
        # and could overwrite each other's file before the rename (review L-10).
        # mkstemp makes it 0600, so it gets the mode open() would give it: in a cache
        # shared with other users they could not read the entries (review V4).
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(prefix=cid + ".", suffix=".tmp", dir=outdir)
            with os.fdopen(fd, "w") as f:
                f.write(text)
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
            os.replace(tmp, path)
        except OSError as e:
            if tmp is not None and os.path.exists(tmp):
                os.remove(tmp)
            failed.append((cid, "could not write %s (%s)%s" % (path, getattr(e, "strerror", None) or e, unread)))
            continue
        fetched.append(cid)
    return fetched, kept, failed


def main():
    ap = argparse.ArgumentParser(description="Prefetch the CCD entries of every ligand in a set of PDB files "
                                             "into the local cache parametrize_ligand.py reads first.")
    ap.add_argument("pdb", nargs="+", help="input PDB files (e.g. benchmark/haddock_benchmark/*/input.pdb)")
    ap.add_argument("-o", "--outdir", default=DEFAULT_CACHE, help="cache directory (default: %(default)s)")
    ap.add_argument("--refresh", action="store_true",
                    help="fetch again even if an entry is cached; with the default directory "
                         "this rewrites entries the repository ships")
    args = ap.parse_args()
    ids = prefetch_ids(args.pdb)
    print("%d CCD entries for %d file(s) (the HETATM residue names, and %s when there is one): %s"
          % (len(ids), len(args.pdb), " ".join(GLYCAN_PARENTS), " ".join(ids)))
    fetched, kept, failed = prefetch(ids, args.outdir, args.refresh)
    print("cache %s: %d fetched, %d already there, %d failed" % (args.outdir, len(fetched), len(kept), len(failed)))
    for cid, why in failed:
        print("  %s: %s" % (cid, why), file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
