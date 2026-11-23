#!/usr/bin/env python3
"""Prepare the scoring set of a live CAPRI round for GroScore.

A CAPRI scoring set (e.g. Target383.pdb.gz from the round's download page) is one
multi-model PDB: a REMARK 9 line per model with the MD5 code the scorer
submission must quote ('REMARK   9 MODEL   12 MD5 <hash>'), then MODEL/ENDMDL
blocks of ATOM records. Unlike the Score_set files of the CAPRI benchmark
(benchmark/capri_benchmark/prepare_capri.py), it says nothing about which chains
are receptor and ligand, so protein B (the chains GroScore pulls away) is given
with -c.

For every complete model (closed by ENDMDL) it writes, into the run directory:
  <ID>/input.pdb  the model's ATOM/HETATM records as submitted, TER after each
                  chain, END. <ID> is <target>_<model number, 4 digits>
  sp.gs           '<ID>  <protein B chains>' for every model directory present
  md5_map.gs      '<ID>  <CAPRI model>  <MD5>', the REMARK 9 code of each model

A model is skipped, and named, when it lacks one of the protein B chains or has
no other chain. A gzip file that ends early (a download cut short, or a file cut
short on the server, as Target383.pdb.gz of round 66 was) is read up to where it
ends; the models completed before that are written and the run exits 1.

input.pdb is written to a temporary name and renamed, so an interrupted run never
leaves a partial one. An existing run directory is refused unless --resume is
given; then only models without a directory are written (an existing directory
may already hold simulation output and is never touched) and sp.gs / md5_map.gs
are rewritten over every model directory present. That is how a fixed copy of a
truncated file is added to the models already set up.

Usage:
  python3 capri/prepare_round.py Target381.pdb -c B
  python3 capri/prepare_round.py Target383.pdb.gz -c B -o capri66/T383 --resume
Standard library only, Python 3.7 or later. Exit 0 when every model listed in REMARK 9 was complete
and written (or already present), 1 otherwise.
"""
import argparse, collections, os, re, sys, zlib


def default_target(path):
    """'Target383.pdb.gz' -> 'T383'; anything else -> the name without extensions."""
    base = os.path.basename(path)
    m = re.match(r"Target0*(\d+)", base)
    if m:
        return "T%s" % m.group(1)
    return base.split(".")[0]


def read_lines(path, problems):
    """The text lines of a .pdb or .pdb.gz. A gzip file is decompressed piece by
    piece, so every line before a cut is returned (gzip.open would raise on the
    whole last block it reads and lose up to 8 KB before the cut); a cut, a
    damaged stream or trailing garbage is appended to problems."""
    if not path.endswith(".gz"):
        with open(path) as fh:
            for line in fh:
                yield line.rstrip("\r\n")
        return
    with open(path, "rb") as fh:   # a missing or unreadable file raises here
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        rest = b""
        while True:
            chunk = fh.read(1 << 20)
            try:
                data = d.decompress(chunk) if chunk else d.flush()
            except zlib.error as e:
                problems.append("damaged gzip stream (zlib: %s)" % e)
                data = b""
                chunk = b""
            if d.eof and d.unused_data:          # a further gzip member (concatenated files)
                tail = d.unused_data
                d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                data += d.decompress(tail)
            lines = (rest + data).split(b"\n")
            rest = lines.pop()
            for l in lines:
                yield l.decode("ascii", "replace").rstrip("\r")
            if not chunk:
                break
        if not d.eof:
            problems.append("gzip stream ends early (%d compressed bytes; the file was cut short)"
                            % os.path.getsize(path))
        if rest:
            yield rest.decode("ascii", "replace").rstrip("\r")


def read_models(path):
    """Yield ('md5', model, hash), ('model', number, [record lines]) and, when the
    file ends early or is damaged, ('truncated', reason, last model number or None)."""
    problems = []
    cur = None
    buf = []
    for line in read_lines(path, problems):
        if line.startswith("REMARK   9 MODEL"):
            p = line.split()
            if len(p) >= 6 and p[4] == "MD5":
                yield ("md5", int(p[3]), p[5])
        elif line.startswith("MODEL "):
            cur = int(line.split()[1])
            buf = []
        elif line.startswith(("ATOM  ", "HETATM")) and cur is not None:
            buf.append(line)
        elif line.startswith("ENDMDL") and cur is not None:
            yield ("model", cur, buf)
            cur = None
    if problems or cur is not None:
        reason = "; ".join(problems) or "file ends inside a model (no ENDMDL)"
        yield ("truncated", reason, cur)


def with_ter(records):
    """The records with a TER after each chain, and END."""
    out = []
    last = None
    for r in records:
        if last is not None and r[21] != last:
            out.append("TER")
        out.append(r)
        last = r[21]
    return "\n".join(out + ["TER", "END"]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="CAPRI scoring-set PDB (.pdb or .pdb.gz)")
    ap.add_argument("-c", "--chains", required=True,
                    help="protein B, the chains GroScore pulls away (comma-separated, e.g. B or B,C)")
    ap.add_argument("-t", "--target", help="structure ID prefix (default: T<n> from Target<n>.pdb)")
    ap.add_argument("-o", "--outdir", help="run directory (default: ./<target>)")
    ap.add_argument("--resume", action="store_true",
                    help="add the models that have no directory yet to an existing run directory")
    args = ap.parse_args()

    target = args.target or default_target(args.source)
    outdir = args.outdir or target
    protb = [c for c in args.chains.split(",") if c]
    if not protb or any(len(c) != 1 for c in protb):
        sys.exit("prepare_round: -c takes single-character chain IDs, comma-separated (got %r)" % args.chains)
    if os.path.exists(outdir) and not args.resume:
        sys.exit("prepare_round: %s exists; pass --resume to add the models it lacks" % outdir)
    os.makedirs(outdir, exist_ok=True)

    md5 = {}
    written, present, skipped = [], [], []
    compositions = collections.Counter()
    truncated = None
    for kind, a, b in read_models(args.source):
        if kind == "md5":
            md5[a] = b
            continue
        if kind == "truncated":
            truncated = (a, b)
            break
        model, records = a, b
        sid = "%s_%04d" % (target, model)
        chains = collections.OrderedDict()
        for r in records:
            chains.setdefault(r[21], set()).add((r[22:27], r[17:20]))
        comp = " ".join("%s:%d" % (c, len(v)) for c, v in chains.items())
        compositions[comp] += 1
        missing = [c for c in protb if c not in chains]
        if missing or not set(chains) - set(protb):
            why = ("no chain %s" % ",".join(missing)) if missing else "no chain besides protein B"
            skipped.append((sid, why))
            continue
        d = os.path.join(outdir, sid)
        if os.path.isdir(d):
            present.append(sid)
            continue
        os.makedirs(d)
        tmp = os.path.join(d, ".input.pdb.part")
        with open(tmp, "w") as o:
            o.write(with_ter(records))
        os.replace(tmp, os.path.join(d, "input.pdb"))
        written.append(sid)

    # sp.gs and md5_map.gs list every model directory now in the run directory
    pat = re.compile(r"^%s_(\d+)$" % re.escape(target))
    dirs = []
    for n in os.listdir(outdir):
        m = pat.match(n)
        if m and os.path.isfile(os.path.join(outdir, n, "input.pdb")):
            dirs.append((int(m.group(1)), n))
    dirs.sort()
    src = os.path.basename(args.source) + (" (truncated)" if truncated else "")
    with open(os.path.join(outdir, "sp.gs"), "w") as o:
        o.write("# CAPRI %s from %s: one structure per model; protein B = chain(s) %s\n"
                % (target, src, ",".join(protb)))
        o.write("# Structure_ID  Chains_for_Protein_B\n")
        for _, sid in dirs:
            o.write("%s  %s\n" % (sid, ",".join(protb)))
    with open(os.path.join(outdir, "md5_map.gs"), "w") as o:
        o.write("# Structure_ID  CAPRI_model  MD5 (REMARK 9 of %s; the scorer submission quotes these)\n" % src)
        for n, sid in dirs:
            o.write("%s  %d  %s\n" % (sid, n, md5.get(n, "MISSING")))

    print("%s -> %s" % (args.source, outdir))
    print("  REMARK 9 MD5 lines: %d" % len(md5))
    print("  chain composition (chain:residues), models:")
    for comp, n in compositions.most_common():
        print("    %-30s %d" % (comp, n))
    print("  written %d, already present %d, skipped %d; sp.gs lists %d"
          % (len(written), len(present), len(skipped), len(dirs)))
    for sid, why in skipped:
        print("  SKIPPED %s: %s" % (sid, why))
    have = {n for n, _ in dirs}
    absent = sorted(set(md5) - have - {int(s.rsplit("_", 1)[1]) for s, _ in skipped})
    no_md5 = sorted(have - set(md5))
    if no_md5:
        print("  WARNING: %d model(s) without a REMARK 9 MD5 line (MISSING in md5_map.gs): %s"
              % (len(no_md5), " ".join(map(str, no_md5[:10]))))
    if truncated:
        reason, last = truncated
        print("  TRUNCATED: %s%s" % (reason, "" if last is None else "; model %d is incomplete and was not written" % last))
    if absent:
        print("  INCOMPLETE: %d of the %d models in REMARK 9 have no directory (first: %d)"
              % (len(absent), len(md5), absent[0]))
    return 1 if (truncated or absent or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
