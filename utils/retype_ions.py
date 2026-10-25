#!/usr/bin/env python3
#
# retype_ions.py - Give the ions pdb2gmx writes the atom types of the water model
#
# AMBER19SB in GROMACS 2026 ships the 12-6 Li/Merz ions for OPC and for OPC3 in
# ffnonbonded.itp (Mg2+_opc and Mg2+_opc3, Zn2+_opc and Zn2+_opc3, ...: 60 ion
# types per set, the AmberTools frcmod.ionslm_126_opc and _opc3, plus the OPC
# water's OW_opc/HW_opc; every _opc type has an _opc3 twin), and genion's NA/CL
# come from the ions_<water>.itp the topology includes. But aminoacids.rtp, which
# pdb2gmx types every residue from, hard-codes the _opc types for every ion residue
# (MG Mg2+_opc, ZN Zn2+_opc, ...), and -water does not reach the .rtp. So under
# amber19sb_opc3 every ion kept from the crystal got the parameters fitted for OPC
# water in a box of OPC3 water: in bm_fe_amber_3a 21 of the 43 structures that
# passed stage 0 (Mg in 1IBR, 1KXP, 1Z0K, ..., Ca in 1H1V and 1R0R, Mn in 1T6B,
# crystal Cl next to genion's Cl-_opc3 in 1BVN). The OPC-fitted divalent cations
# are up to 0.06 A smaller in Rmin/2 than their OPC3 twins (Zn2+ 0.06, Mg2+ 0.03,
# Ca2+ 0.02), with a shallower well.
#
# This script runs right after every pdb2gmx call of both runners. In the
# topology files pdb2gmx wrote (topol.top and the topol_*.itp it includes from
# the topology's own directory) it changes every [ atoms ] type ending in _opc
# into the same name ending in _<water>, the twin, when the water model is not
# opc itself. Nothing else in the files changes. A twin the force field does not
# define stops it before any file is written (exit 1; the runners stop the
# structure as FAILED), since an ion typed for another water model is exactly
# what this exists to prevent. The force field is found the way grompp finds it,
# from the topology's '#include "<ff>.ff/forcefield.itp"': the topology's
# directory (a local amber19sb.ff copy made for NCAAs or FeS clusters), then
# every GMXLIB entry, then $GMXDATA/top (then the share/gromacs/top of the gmx
# on PATH).
#
# Usage: python retype_ions.py -t topol.top --water opc3
#
# Standard library only. Exit 0 also when there is nothing to retype (another
# force field, OPC water, no ion).

import argparse
import os
import re
import shutil
import sys


def resolve_include(name, topdir):
    """The file an #include "name" in a topology in topdir refers to, searched the way
    grompp does: the topology's directory, every GMXLIB entry, then $GMXDATA/top
    (as merge_crystal_waters.py does), and, for a GROMACS whose GMXRC was not
    sourced, the share/gromacs/top beside the gmx on PATH (as fuse_glycans.py
    does). None if not found."""
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    dirs = [topdir] + [d for d in os.environ.get("GMXLIB", "").split(":") if d]
    if os.environ.get("GMXDATA"):
        dirs.append(os.path.join(os.environ["GMXDATA"], "top"))
    gmx = shutil.which("gmx")
    if gmx:
        dirs.append(os.path.join(os.path.dirname(os.path.realpath(gmx)), "..", "share", "gromacs", "top"))
    for d in dirs:
        p = os.path.normpath(os.path.join(d, name))
        if os.path.isfile(p):
            return p
    return None


def includes(path):
    """The #include names of a topology file, in order."""
    out = []
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*#include\s+"([^"]+)"', line)
            if m:
                out.append(m.group(1))
    return out


def ff_atomtypes(forcefield_itp):
    """Every [ atomtypes ] name of the force field whose forcefield.itp is given,
    following its #includes inside the force field directory."""
    names, seen = set(), set()
    ffdir = os.path.dirname(forcefield_itp)

    def read(path):
        if path in seen or not os.path.isfile(path):
            return
        seen.add(path)
        section = None
        with open(path) as f:
            for line in f:
                s = line.split(";")[0].strip()
                m = re.match(r'#include\s+"([^"]+)"', s)
                if m:
                    read(os.path.normpath(os.path.join(ffdir, m.group(1))))
                    continue
                if not s or s.startswith("#"):
                    continue
                if s.startswith("["):
                    section = s.strip("[] ").lower()
                    continue
                if section == "atomtypes":
                    names.add(s.split()[0])
    read(forcefield_itp)
    return names


def retyped(text, suffix_from, suffix_to):
    """(new text, {old type: count}): every [ atoms ] type (and B type) ending in
    suffix_from given suffix_to instead. Only the type field changes; the spacing of
    the line is kept, widened by the longer name."""
    out, changed, section = [], {}, None
    for line in text.splitlines(keepends=True):
        s = line.split(";")[0].strip()
        if s.startswith("["):
            section = s.strip("[] ").lower()
            out.append(line)
            continue
        if section != "atoms" or not s or s.startswith("#"):
            out.append(line)
            continue
        body, sep, comment = line.partition(";")
        fields = re.split(r"(\s+)", body)          # words and the spaces between them
        words = [k for k, w in enumerate(fields) if w and not w.isspace()]
        for pos in (1, 8):                         # type, and typeB of a perturbed atom
            if pos < len(words):
                k = words[pos]
                if fields[k].endswith(suffix_from):
                    changed[fields[k]] = changed.get(fields[k], 0) + 1
                    fields[k] = fields[k][: -len(suffix_from)] + suffix_to
        out.append("".join(fields) + sep + comment)
    return "".join(out), changed


def main():
    p = argparse.ArgumentParser(description="Give pdb2gmx's ion atom types the twins of the water model.")
    p.add_argument("-t", "--topol", default="topol.top", help="topology pdb2gmx wrote (default topol.top)")
    p.add_argument("--water", required=True, help="the water model pdb2gmx was given (-water), e.g. opc3")
    p.add_argument("--from-suffix", default="_opc",
                   help="the suffix of the types the .rtp hard-codes (default _opc)")
    args = p.parse_args()

    to_suffix = "_" + args.water
    if to_suffix == args.from_suffix:
        return 0
    topdir = os.path.dirname(os.path.abspath(args.topol))
    files = [os.path.abspath(args.topol)]
    for name in includes(args.topol):
        local = os.path.join(topdir, name)
        if not os.path.isabs(name) and os.path.isfile(local) and os.path.basename(name).startswith("topol"):
            files.append(os.path.normpath(local))

    new = {}
    for path in files:
        with open(path) as f:
            text = f.read()
        out, changed = retyped(text, args.from_suffix, to_suffix)
        if changed:
            new[path] = (out, changed)
    if not new:
        return 0

    ff = next((n for n in includes(args.topol) if n.endswith("forcefield.itp")), None)
    ff_path = resolve_include(ff, topdir) if ff else None
    if ff_path is None:
        print("retype_ions: the force field of %s was not found (topology directory, GMXLIB, GMXDATA/top, "
              "the gmx on PATH); nothing changed" % args.topol, file=sys.stderr)
        return 1
    known = ff_atomtypes(ff_path)
    missing = sorted({t[: -len(args.from_suffix)] + to_suffix for _, ch in new.values() for t in ch}
                     - known)
    if missing:
        print("retype_ions: %s defines no %s, so these ions cannot be given the %s types; nothing changed"
              % (ff_path, ", ".join(missing), args.water), file=sys.stderr)
        return 1
    for path, (out, changed) in new.items():
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(out)
        os.replace(tmp, path)
        print("retype_ions: %s: %s" % (os.path.basename(path), ", ".join(
            "%s -> %s%s (%d atom%s)" % (t, t[: -len(args.from_suffix)], to_suffix, n, "" if n == 1 else "s")
            for t, n in sorted(changed.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
