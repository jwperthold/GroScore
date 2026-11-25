#!/usr/bin/env python3
"""capri/analyze_round_66.py: the V-domain regions, the paratope filter, and
capri/analyze_round.py's --exclude.

  1. vh_regions reads the regions of an immunoglobulin V domain from its own
     anchors: the three round-66 VHHs (T381, T382 and T383, whose CDR3 is 20
     residues and whose FR3 is one shorter in T382) and a conventional VH give
     the right CDR sequences, a trailing His tag is split off FR4, a domain
     without one ends FR4 at its last residue, and a non-Ig chain gives None.
  2. describe/region_of count contacts per region: a model touching only CDR3
     has cdr3 = face = 1, one touching only FR1 has back = 1, and FR2 and the
     CDR4 loop count to the paratope, not to the back.
  3. verdict rejects on back, on CDR3 and, only when asked, on the subdomain.
  4. end to end: paratope.gs has a row per model, paratope_rejected.gs names the
     back-binder and the model without CDR3 contacts, and with --submit neither
     is in the submission although both keep their row in models.gs; --no-filter
     submits them and says so.
  5. analyze_round.py --exclude: the excluded models keep score, cluster and
     i-RMSD but are not submitted, and with --pick cluster a cluster whose best
     member is excluded contributes its best remaining one instead.

Run with the groscore environment:  python tests/test_capri_round66.py
"""
import importlib.util, os, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "capri", "analyze_round_66.py")
GENERAL = os.path.join(REPO, "capri", "analyze_round.py")
spec = importlib.util.spec_from_file_location("analyze_round_66", SCRIPT)
a66 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a66)
failures = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


AA1TO3 = {v: k for k, v in a66.AA3TO1.items()}
VHH = {
    "T381": "QVQLQESGGGLVQAGGSLRLSCTASERTFSIDYMGWYRQAPGKQREFVGTIEWNSGSTGYGDSVKGRFTISRDNAKNT"
            "GYLQMNSLKPEDTAVYYCAAKYGSSRLWDYWGQGTQVTVSSHHHHHH",
    "T382": "QVQLQESGGGSVQAGGSLRLSCAASGRTLSGWVMGWFRQAPGKDRDFVAHLSGDTTAYAHSVKGRFTISRDNAKDT"
            "VFLQMNSLKPEDTGVYYCAARPLGSTWTEHPYWGQGTQVTVSSHHHHHH",
    "T383": "QVQLQESGGGLVQAGGSLRLSCAASGRTLSGAAMGWFRQAPGKEREFVAGISWSTSRTQYADSVKGRFTISRDNAKNT"
            "VFLQMNSLNAEDTAVYYCAADGRFYSDYVWSSPDEYAYWGQGTQVTVSSHHHHHH",
}
VH_CONV = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
           "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS")
LYSOZYME = ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNI"
            "PCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")


def seq_of(s):
    return list(enumerate(s, 1))


def sub(s, r):
    return s[r[0] - 1:r[1]]


# 1. region detection
r381 = a66.vh_regions(seq_of(VHH["T381"]))
r382 = a66.vh_regions(seq_of(VHH["T382"]))
r383 = a66.vh_regions(seq_of(VHH["T383"]))
check(r381 and sub(VHH["T381"], r381["CDR1"]) == "ERTFSIDYMG"
      and sub(VHH["T381"], r381["CDR2"]) == "TIEWNSGSTGYGDSVK"
      and sub(VHH["T381"], r381["CDR3"]) == "AAKYGSSRLWDY", "1: T381 VHH CDRs")
check(r382 and sub(VHH["T382"], r382["CDR3"]) == "AARPLGSTWTEHPY" and r382["CDR2"] == (50, 63),
      "1: T382 VHH, one residue shorter in FR3, keeps its frame")
check(r383 and sub(VHH["T383"], r383["CDR3"]) == "AADGRFYSDYVWSSPDEYAY",
      "1: T383 VHH, 20-residue CDR3")
check(all(r and r["tag"] == (len(s) - 5, len(s)) for r, s in
          ((r381, VHH["T381"]), (r382, VHH["T382"]), (r383, VHH["T383"]))),
      "1: the trailing His6 is a tag, not FR4")
check(all(r and r["FR4"][1] == r["tag"][0] - 1 and r["FR4"][0] == r["CDR3"][1] + 1
          for r in (r381, r382, r383)), "1: FR4 runs from after CDR3 to the tag")
rc = a66.vh_regions(seq_of(VH_CONV))
check(rc and sub(VH_CONV, rc["CDR3"]) == "SRWGGDGFYAMDY" and "tag" not in rc
      and rc["FR4"][1] == len(VH_CONV), "1: a conventional VH without a tag: FR4 to the last residue")
check(a66.vh_regions(seq_of(LYSOZYME)) is None, "1: a chain that is no V domain gives None")
check(a66.vh_regions(seq_of(VHH["T381"][:60])) is None, "1: a truncated V domain gives None")

# 2. contacts per region
R = r381


def at(x, y, z, chain, num, resname, name, serial):
    return ("ATOM  %5d  %-3s %s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s\n"
            % (serial, name, resname, chain, num, x, y, z, name[0]))


def model_text(touch, seq=VHH["T381"]):
    """Chain B = the VHH, residue i at x = 6i; chain A one residue above each
    chain-B residue in touch (4 A away, so only that residue is in contact)."""
    out, serial = [], 0
    for i, c in enumerate(seq, 1):
        for k, name in enumerate(("N", "CA", "C", "O")):
            serial += 1
            out.append(at(6.0 * i, 0.3 * k, 0.0, "B", i, AA1TO3[c], name, serial))
    out.append("TER\n")
    for j, b in enumerate(touch, 1):
        for k, name in enumerate(("N", "CA", "C", "O")):
            serial += 1
            out.append(at(6.0 * b, 4.0 + 0.3 * k, 0.0, "A", j, "ALA", name, serial))
    out.append("TER\nEND\n")
    return "".join(out)


def desc(touch):
    return a66.describe(a66.ar.Model(model_text(touch), {"B"}), {"B"}, R, False)


d_cdr3 = desc(list(range(99, 106)))
check(abs(d_cdr3["cdr3"] - 1.0) < 1e-9 and abs(d_cdr3["face"] - 1.0) < 1e-9 and d_cdr3["back"] == 0,
      "2: contacts only in CDR3: cdr3 = face = 1, back = 0")
d_fr1 = desc(list(range(3, 12)))
check(abs(d_fr1["back"] - 1.0) < 1e-9 and d_fr1["cdr3"] == 0 and d_fr1["face"] == 0,
      "2: contacts only in FR1: back = 1")
d_face = desc([37, 44, 47, 72, 74, 75])
check(abs(d_face["face"] - 1.0) < 1e-9 and d_face["back"] == 0,
      "2: FR2 and the CDR4 loop are paratope, not back")
d_tail = desc(list(range(110, 126)))
check(abs(d_tail["back"] - 1.0) < 1e-9, "2: the strand after FR4's W and the His tag are back")
d_mix = desc([100, 101, 102, 103, 5, 6])
check(abs(d_mix["cdr3"] - 4.0 / 6) < 1e-6 and abs(d_mix["back"] - 2.0 / 6) < 1e-6,
      "2: the fractions are per contact pair")
check(a66.describe(a66.ar.Model(model_text([]), {"B"}), {"B"}, R, False) is None,
      "2: a model without contacts is not described")

# 3. the verdict
v = dict(back=0.30, cdr3=0.40, top_domain="IV")
check(a66.verdict(v, 0.25, 0.10, None).startswith("back"), "3: too much back is rejected")
check(a66.verdict(dict(back=0.0, cdr3=0.05, top_domain="IV"), 0.25, 0.10, None).startswith("CDR3"),
      "3: too little CDR3 is rejected")
check(a66.verdict(dict(back=0.14, cdr3=0.19, top_domain="I"), 0.25, 0.10, None) == "ok",
      "3: the consensus extremes of T381 pass the defaults")
check(a66.verdict(dict(back=0.0, cdr3=0.4, top_domain="I"), 0.25, 0.10, None) == "ok"
      and "domain I" in a66.verdict(dict(back=0.0, cdr3=0.4, top_domain="I"), 0.25, 0.10, "IV"),
      "3: the subdomain is only judged with --require-domain")

# 4. end to end
TOUCH = {0: list(range(99, 106)),           # CDR3 + CDR2, a proper paratope
         1: list(range(52, 60)) + list(range(100, 105)),
         2: list(range(2, 12)) + [100],     # mostly FR1: back-binder
         3: list(range(50, 62)),            # no CDR3 at all
         4: list(range(99, 106)) + [52]}
SCORES = ["-260.0", "-250.0", "-300.0", "-290.0", "-240.0"]   # the two bad ones score best


def build(run):
    os.makedirs(run)
    with open(os.path.join(run, "sp.gs"), "w") as sp, \
            open(os.path.join(run, "scores_avg.gs"), "w") as sc, \
            open(os.path.join(run, "md5_map.gs"), "w") as md:
        sc.write("# Structure_ID  Score  CI95  Cycles_Used  RMSD_mean_A  RMSD_max_A  RMSD_flag\n")
        for i, s in enumerate(SCORES):
            sid = "T381_%04d" % i
            os.makedirs(os.path.join(run, sid))
            with open(os.path.join(run, sid, "input.pdb"), "w") as o:
                o.write(model_text(TOUCH[i]))
            sp.write("%s  B\n" % sid)
            sc.write("%s\t%s\tnan\t1\t2.0\t2.0\tOK\n" % (sid, s))
            md.write("%s  %d  %032x\n" % (sid, i, i + 1))


def read_rows(path):
    return {l.split()[0]: l.split() for l in open(path) if not l.startswith("#")}


with tempfile.TemporaryDirectory() as tmp:
    run = os.path.join(tmp, "T381")
    build(run)
    common = ["--min-size", "2", "--min-scored", "2"]
    p = subprocess.run([sys.executable, SCRIPT, run, "--submit"] + common,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    ok = p.returncode == 0
    check(ok, "4: runs (exit %d)%s" % (p.returncode, "" if ok else "\n" + p.stdout))
    if ok:
        ana = os.path.join(run, "analysis")
        par = read_rows(os.path.join(ana, "paratope.gs"))
        check(len(par) == 5, "4: paratope.gs has a row per model")
        rej = read_rows(os.path.join(ana, "paratope_rejected.gs"))
        check(set(rej) == {"T381_0002", "T381_0003"},
              "4: the back-binder and the model without CDR3 are rejected, the others are not")
        check(rej.get("T381_0002", ["", ""])[1].startswith("back")
              and rej.get("T381_0003", ["", ""])[1].startswith("CDR3"), "4: each with its reason")
        models = read_rows(os.path.join(ana, "models.gs"))
        check(all(s in models and models[s][6] != "nan" for s in ("T381_0002", "T381_0003")),
              "4: a rejected model keeps its row and its pKd in models.gs")
        subm = [l.split()[1] for l in open(os.path.join(ana, "submission.gs")) if not l.startswith("#")]
        check(subm and "T381_0002" not in subm and "T381_0003" not in subm,
              "4: neither is submitted")
        check(subm[0] == "T381_0000", "4: the best-scoring model that passes leads the submission")
        check("2 of 5 model(s) fail it" in p.stdout, "4: the count is reported")
    p = subprocess.run([sys.executable, SCRIPT, run, "--submit", "--no-filter"] + common,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode == 0:
        subm = [l.split()[1] for l in open(os.path.join(run, "analysis", "submission.gs"))
                if not l.startswith("#")]
        check(subm[0] == "T381_0002" and "T381_0003" in subm,
              "4: --no-filter submits them again, best score first")
        check("--no-filter" in p.stdout and "fail it" in p.stdout,
              "4: --no-filter still says how many would have been rejected")

# 5. analyze_round.py --exclude
with tempfile.TemporaryDirectory() as tmp:
    run = os.path.join(tmp, "T381")
    build(run)
    excl = os.path.join(tmp, "skip.gs")
    with open(excl, "w") as o:
        o.write("# a comment\nT381_0002  reason\nT381_0000\n")
    common = ["--min-size", "2", "--min-scored", "2", "--submit", "--exclude", excl]
    p = subprocess.run([sys.executable, GENERAL, run] + common,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    ok = p.returncode == 0
    check(ok, "5: analyze_round.py --exclude runs (exit %d)%s" % (p.returncode, "" if ok else "\n" + p.stdout))
    if ok:
        ana = os.path.join(run, "analysis")
        models = read_rows(os.path.join(ana, "models.gs"))
        check(models["T381_0000"][6] != "nan" and models["T381_0002"][6] != "nan",
              "5: an excluded model keeps score, cluster and i-RMSD")
        subm = [l.split()[1] for l in open(os.path.join(ana, "submission.gs")) if not l.startswith("#")]
        check("T381_0000" not in subm and "T381_0002" not in subm, "5: neither is submitted")
        check("2 ID(s), 2 of them scored, not submitted" in p.stdout, "5: the exclusion is reported")
        cl = {l.split()[1]: l.split() for l in open(os.path.join(ana, "clusters.gs"))
              if not l.startswith("#")}
        check(any(c[8] == "T381_0000" for c in cl.values()),
              "5: clusters.gs still names the excluded model as its cluster's best")
        p2 = subprocess.run([sys.executable, GENERAL, run] + common + ["--pick", "cluster"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if p2.returncode == 0:
            subm = [l.split()[1] for l in open(os.path.join(ana, "submission.gs")) if not l.startswith("#")]
            check("T381_0000" not in subm and "T381_0002" not in subm and subm,
                  "5: --pick cluster takes a cluster's best remaining member instead")

print("\n%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
