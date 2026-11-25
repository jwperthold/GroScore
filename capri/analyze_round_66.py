#!/usr/bin/env python3
"""CAPRI round 66: capri/analyze_round.py plus what the three targets have in
common, an antibody interface.

T381, T382 and T383 are the same receptor, the HER2 ectodomain (chain A, mature
numbering, four subdomains), with a different camelid VHH each (chain B, protein
B in sp.gs). This script adds the two questions a general scorer cannot ask:

  where on the receptor does a model bind, and does the VHH bind with the part
  of itself that an immune system selects, its CDRs?

A nanobody's paratope is CDR1, CDR2 and CDR3, almost always dominated by CDR3,
plus two framework pieces that belong to it: FR2 (the face that in a
conventional antibody carries the light chain; the VHH hallmark positions sit
there) and the first part of FR3, the loop next to CDR2 sometimes called CDR4.
The opposite end of the domain - the N-terminal strand of FR1, the back of FR3,
the C-terminal strand after the FR4 tryptophan, and any purification tag -
points away from the antigen and carries the hinge in the real molecule. A model
that buries that end in the receptor is not a binding mode an immune system
could have selected, however well it scores.

The regions are read from the chain's own sequence, from the anchors every
immunoglobulin V domain has (the FR1 cysteine, the FR2 tryptophan of W-x-R-Q,
the FR3 cysteine of Y-Y-C and the FR4 tryptophan of W-G-x-G), so nothing here is
hard-coded per target and a chain that is no V domain is simply reported as such.

The filter (--max-back, --min-cdr3) rejects a model when too large a share of
its contacts falls on the back of the VHH, or when CDR3 makes too little of the
interface. It says nothing about where on the receptor the model binds: on a
blind target that would be assuming the answer. --require-domain does assume it,
and is off by default.

Defaults were set on T381 (1322 models): the 499-member consensus cluster never
puts more than 14% of its contacts on the back and never less than 19% on CDR3,
so 0.25 and 0.10 pass every model of it with room to spare while rejecting the
6% of the set that binds backwards. The rejected models are not deleted: they
keep their score, cluster and place in both plots, they are only not submitted.

Writes <rundir>/analysis/paratope.gs next to the files of analyze_round.py, and
passes every other option through to it:

  python capri/analyze_round_66.py capri66/T381 --submit --group 'HUMAN Team 6'
  python capri/analyze_round_66.py capri66/T381 --max-back 0.15 --no-filter
"""
import argparse, importlib.util, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("analyze_round", os.path.join(HERE, "analyze_round.py"))
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)

AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# HER2 ectodomain, mature numbering (the receptor of every round-66 target)
HER2_START = "TQVCTGTDMKLRLPASPETHLDML"
HER2_DOMAINS = [("I", 1, 195), ("II", 196, 319), ("III", 320, 488), ("IV", 489, 602)]

# the VHH paratope: CDR1, CDR2, CDR3, FR2 and the CDR4 loop at the start of FR3
FACE = ("CDR1", "CDR2", "CDR3", "FR2", "CDR4")
BACK = ("FR1", "FR3", "FR4", "tag")
REGIONS = ("FR1", "CDR1", "FR2", "CDR2", "CDR4", "FR3", "CDR3", "FR4", "tag")


# ------------------------------------------------------------------ chains

def chain_sequence(model, chain):
    """[(residue number, one-letter code)] of a chain, in file order."""
    seen, out = set(), []
    for key, name in zip(model.keys, model.names):
        if key[0] != chain or name != "CA" or key in seen:
            continue
        seen.add(key)
        num = re.match(r"-?\d+", key[1])
        if num:
            out.append((int(num.group()), AA3TO1.get(model.resname[key], "X")))
    return out


def vh_regions(seq):
    """{region: (first, last)} of an immunoglobulin V domain in its own residue
    numbers, or None when the chain carries no such domain.

    seq is [(residue number, one-letter code)]. The anchors are the ones every V
    domain has: the FR1 cysteine (~22), the tryptophan of FR2's W-x-R-Q (~36),
    the cysteine of FR3's Y-Y-C (~96) and the tryptophan of FR4's W-G-x-G (~109).
    CDR2 has no second anchor, so it runs from 14 after the FR2 tryptophan to two
    before FR3's R-[FLVIA]-[TAS]-[IVLM]-S, as Kabat's H2 does. The first 12
    residues of FR3 are the CDR4 loop; the FR4 tryptophan packs against CDR3 and
    counts to the paratope, everything after it (a tag included) does not.
    """
    s = "".join(c for _, c in seq)
    num = [n for n, _ in seq]
    cys1 = s.find("C", 14, 32)
    if cys1 < 0:
        return None
    m = re.compile(r"W.RQ").search(s, cys1 + 8, cys1 + 30)
    trp1 = m.start() if m else -1
    m = None
    for m in re.compile(r"WG.G").finditer(s, max(trp1, cys1) + 30):
        pass                                               # the last W-G-x-G is FR4's
    trp2 = m.start() if m else -1
    if trp1 < 0 or trp2 < 0 or trp2 - trp1 < 40:
        return None
    cys2 = s.rfind("C", trp1, trp2)
    if cys2 < 0 or trp2 - cys2 > 40:
        return None
    cdr2 = trp1 + 14
    m = re.compile(r"R[FLVIA][TAS][IVLM]S").search(s, cdr2 + 5, cys2)
    if not m:
        return None
    cdr2_end = m.start() - 2
    if not cys1 + 4 <= trp1 - 1 or not cdr2 <= cdr2_end or not cys2 + 1 <= trp2 - 1:
        return None
    bounds = [("FR1", 0, cys1 + 3), ("CDR1", cys1 + 4, trp1 - 1), ("FR2", trp1, cdr2 - 1),
              ("CDR2", cdr2, cdr2_end), ("CDR4", cdr2_end + 1, min(cdr2_end + 12, cys2 - 1)),
              ("FR3", min(cdr2_end + 13, cys2), cys2), ("CDR3", cys2 + 1, trp2 - 1),
              ("FR4", trp2, trp2)]
    out = {}
    for name, lo, hi in bounds:
        if lo <= hi:
            out[name] = (num[lo], num[hi])
    tag = len(s)                                           # a trailing His tag, reported apart
    while tag > trp2 + 1 and s[tag - 1] == "H":
        tag -= 1
    if trp2 + 1 <= tag - 1:
        out["FR4"] = (num[trp2], num[tag - 1])             # FR4's W is paratope, the strand is not
    if tag < len(s):
        out["tag"] = (num[tag], num[-1])
    return out


def region_of(regions, n, default="FR4"):
    for name in REGIONS:
        if name in regions and regions[name][0] <= n <= regions[name][1]:
            return name
    return default


def her2_domain(n, seq_ok):
    if not seq_ok:
        return "?"
    for name, lo, hi in HER2_DOMAINS:
        if lo <= n <= hi:
            return name
    return "tag"


# ------------------------------------------------------------------ analysis

def describe(model, protb, regions, her2):
    """Contact fractions per receptor domain and per VHH region."""
    pairs = model.residue_pairs(ar.CONTACT_A)
    if not pairs:
        return None
    dom, reg = {}, {}
    for ka, kb in pairs:
        na, nb = int(re.match(r"-?\d+", ka[1]).group()), int(re.match(r"-?\d+", kb[1]).group())
        d = her2_domain(na, her2)
        dom[d] = dom.get(d, 0) + 1
        r = region_of(regions, nb)
        reg[r] = reg.get(r, 0) + 1
    n = float(len(pairs))
    return dict(npairs=len(pairs), dom={k: v / n for k, v in dom.items()},
                reg={k: v / n for k, v in reg.items()},
                back=sum(reg.get(k, 0) for k in BACK) / n,
                face=sum(reg.get(k, 0) for k in FACE) / n,
                cdr3=reg.get("CDR3", 0) / n,
                top_domain=max(dom, key=lambda k: dom[k]) if dom else "?")


def verdict(d, max_back, min_cdr3, domain):
    why = []
    if d["back"] > max_back:
        why.append("back %.2f" % d["back"])
    if d["cdr3"] < min_cdr3:
        why.append("CDR3 %.2f" % d["cdr3"])
    if domain and d["top_domain"] != domain:
        why.append("domain %s" % d["top_domain"])
    return ("-".join(why) if why else "ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir", help="run directory (as capri/analyze_round.py takes it)")
    ap.add_argument("--max-back", type=float, default=0.25,
                    help="reject a model with more than this fraction of its contacts on the back of "
                         "the VHH, FR1 / FR3 / the strand after FR4's W / a tag (default 0.25)")
    ap.add_argument("--min-cdr3", type=float, default=0.10,
                    help="reject a model whose CDR3 makes less than this fraction of the contacts "
                         "(default 0.10)")
    ap.add_argument("--require-domain", choices=[d[0] for d in HER2_DOMAINS],
                    help="also reject a model whose epitope is not mainly in this HER2 subdomain. Off "
                         "by default: on a blind target it assumes the answer")
    ap.add_argument("--no-filter", action="store_true",
                    help="describe every model but submit as capri/analyze_round.py would")
    args, rest = ap.parse_known_args()

    rundir = args.rundir.rstrip("/")
    outdir = os.path.join(rundir, "analysis")
    for k, opt in enumerate(rest):                        # honour an -o of analyze_round.py
        if opt in ("-o", "--outdir") and k + 1 < len(rest):
            outdir = rest[k + 1]
    sp = ar.read_gs_rows(os.path.join(rundir, "sp.gs"))
    if not sp:
        sys.exit("analyze_round_66: no structures in %s/sp.gs" % rundir)

    # the V domain and the receptor, from the first readable model
    regions, her2, chain_b = None, False, None
    for sid, chains in ((p[0], set(p[1].split(","))) for p in sp):
        text = ar.read_pdb_text(rundir, sid)
        if text is None:
            continue
        m = ar.Model(text, chains)
        chain_b = sorted(chains)[0]
        regions = vh_regions(chain_sequence(m, chain_b))
        rec = [c for c in sorted({k[0] for k in m.keys}) if c not in chains]
        her2 = bool(rec) and "".join(c for _, c in chain_sequence(m, rec[0])).startswith(HER2_START[:16])
        print("reference chains from %s: receptor %s, protein B %s"
              % (sid, ",".join(rec) or "-", chain_b))
        break
    if regions is None:
        sys.exit("analyze_round_66: chain %s of %s carries no immunoglobulin V domain (anchors not "
                 "found). Use capri/analyze_round.py." % (chain_b, rundir))
    print("  chain %s, V domain regions: %s" % (chain_b, "  ".join(
        "%s %d-%d" % (k, regions[k][0], regions[k][1]) for k in REGIONS if k in regions)))
    print("  receptor: %s" % ("HER2 ectodomain, subdomains I 1-195, II 196-319, III 320-488, IV 489-602"
                              if her2 else "not HER2 by sequence; domains reported as '?'"))

    # describe every model
    desc = {}
    for k, p in enumerate(sp):
        sid, chains = p[0], set(p[1].split(","))
        text = ar.read_pdb_text(rundir, sid)
        if text is None:
            continue
        d = describe(ar.Model(text, chains), chains, regions, her2)
        if d is not None:
            desc[sid] = d
        if (k + 1) % 250 == 0:
            print("  described %d / %d structures" % (k + 1, len(sp)), file=sys.stderr)

    domain = args.require_domain
    for sid, d in desc.items():
        d["verdict"] = verdict(d, args.max_back, args.min_cdr3, domain)
    rejected = sorted(s for s, d in desc.items() if d["verdict"] != "ok")

    os.makedirs(outdir, exist_ok=True)
    excl = os.path.join(outdir, "paratope_rejected.gs")
    with open(excl, "w") as o:
        o.write("# models that fail the paratope filter (back > %.2f, CDR3 < %.2f%s)%s\n"
                % (args.max_back, args.min_cdr3, ", domain != %s" % domain if domain else "",
                   "; --no-filter, so they were submitted all the same" if args.no_filter else ""))
        o.write("# Structure_ID  reason\n")
        for s in rejected:
            o.write("%s  %s\n" % (s, desc[s]["verdict"]))
    with open(os.path.join(outdir, "paratope.gs"), "w") as o:
        o.write("# %s: contacts (%.0f A) per HER2 subdomain and per VHH region\n"
                % (os.path.basename(os.path.abspath(rundir)), ar.CONTACT_A))
        o.write("# V domain of chain %s: %s\n" % (chain_b, " ".join(
            "%s=%d-%d" % (k, regions[k][0], regions[k][1]) for k in REGIONS if k in regions)))
        o.write("# paratope = %s; back = %s\n" % ("+".join(FACE), "+".join(BACK)))
        o.write("# Structure_ID  Pairs  Epitope  %s  Face  Back  CDR3  Verdict\n"
                % "  ".join(REGIONS))
        for sid in sorted(desc):
            d = desc[sid]
            o.write("%s  %d  %s  %s  %.3f  %.3f  %.3f  %s\n"
                    % (sid, d["npairs"], d["top_domain"],
                       "  ".join("%.3f" % d["reg"].get(k, 0.0) for k in REGIONS),
                       d["face"], d["back"], d["cdr3"], d["verdict"]))

    nd = len(desc)
    print("\n  %d model(s) with an interface; median CDR3 %.2f, median back %.2f"
          % (nd, _median([d["cdr3"] for d in desc.values()]), _median([d["back"] for d in desc.values()])))
    if her2:
        print("  epitope: " + ", ".join(
            "%s %d%%" % (dm, round(100.0 * sum(1 for d in desc.values() if d["top_domain"] == dm) / nd))
            for dm, _, _ in HER2_DOMAINS))
    print("  paratope filter (back <= %.2f, CDR3 >= %.2f%s): %d of %d model(s) fail it (%.0f%%)%s"
          % (args.max_back, args.min_cdr3, ", domain %s" % domain if domain else "",
             len(rejected), nd, 100.0 * len(rejected) / nd,
             "; --no-filter, submitted all the same" if args.no_filter else ", not submitted"))
    print("  written: %s, %s" % (os.path.join(outdir, "paratope.gs"), excl))
    print()

    argv = [rundir] + rest
    if not args.no_filter and rejected:
        argv += ["--exclude", excl]
    return ar.main(argv)


def _median(v):
    v = sorted(v)
    return float("nan") if not v else (v[len(v) // 2] if len(v) % 2 else
                                       0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2]))


if __name__ == "__main__":
    sys.exit(main())
