#!/usr/bin/env python3
"""ADR-046: the step's lip, read as the ADR's first and second revisions state.

Reads the marches' logs and Cp files, CFL3D's committed results, TMR's grids
and the files fetched under ADR-046, and prints each step's reading. It never
runs a march and writes no file. It never prints PASS, and prints FAIL only in
outcome's statement that gate 4 stays FAIL as ADR-045 recorded it.

Modes:
  reading      step 1 (R): the reading bias b and C40, reported, not judged
  headers      a downloaded file's header lines, with every value masked
  reference    step 2 (F): FUN3D against CFL3D on level 1
  finest       step 3 (T): CFL3D on the finest grid against its level 1
  s0           step 4 (S0): the chain again, against ADR-045
  commands     the exact commands of steps 5 and 6
  grid         step 5 (G): ours on TMR's finest grid
  ablations    step 6 (A): the six level-2 marches
  consistency  a march's lip print against its .cp
  outcome      step 8: what the steps do to ADR-045's decision

Run:  backstep_lip.py <mode> [--s0 DIR] [--g DIR] [--a ROOT] [--log LOG]
      [--labels JSON] [--fun3d F] [--cfl3d-finest F] [--fun3d-finest F]
      [--cfl3d-l1 F] [--grid0 P2DFMT] [--l0 HEX] [--l1 HEX]
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1] / "cases" / "backstep"))
from backstep_gate import (TMR, UC_CFL3D, final, floor_values, parse_final,  # noqa: E402
                           split_floors, two_col)
from make_mesh import merge, read_p2d  # noqa: E402

CFL3D_CP = TMR / "backstep_cfl3d_cp_sst.dat"
LABELS = TMR / "labels.json"
GRID_L = {1: TMR / "backstep5_1levdn.p2dfmt.gz", 2: TMR / "backstep5_2levdn.p2dfmt.gz"}
BOUND = 0.015              # gate 4's
NFLOOR = 0.0015            # n: the spread of CFL3D's own two values at x = 8
X_LIP = -0.0208296         # "-0.021", CFL3D's point
LIP_TOL = 1e-5
POINTS = ("corner", "-0.021")
READINGS = ("G", "F")      # the gate reading and the face reading
# ADR-045's printed d at the two lip points, and the excess E = d - 0.015.
D045 = {1: (0.0205, 0.0174), 2: (0.0221, 0.0181)}
E045 = {1: (0.0055, 0.0024), 2: (0.0071, 0.0031)}
# Its other printed upstream points on level 1, for step 2 (three digits).
OTHER3 = ((-4.0, 0.005), (-0.125, 0.011), (-0.063, 0.014))
ROUND3 = 0.0005
# Step 4: ADR-045's printed values, taken as exact, with the unit of each
# one's last printed digit; and the bands, half of its change from level 2 to 1.
A045 = {("xr", 2): (6.5567, 1e-4), ("xr", 1): (6.5210, 1e-4),
        ("cf4", 2): (2.8793e-3, 1e-7), ("cf4", 1): (2.92834e-3, 1e-8),
        ("uc4", 2): (0.99929, 1e-5), ("uc4", 1): (0.99891, 1e-5)}
U_IN_045 = (0.944775, 1e-6)
BAND = {"xr": 0.018, "cf4": 2.4e-5, "uc4": 0.0002, "u_in": 0.0002, "corner": 0.0008, "-0.021": 0.0004}
S0_MARCHES = ("L2_up", "L2_cal", "L2", "L1")
CELLS_L0 = 1277952
SECONDS_G = 604800.0       # 168 hours
DT_G = "0.04"
ABLATIONS = ("0", "a", "b", "c", "d", "e")
CHANGE = {"0": [], "a": ["VIBEFLOW_CORRECTORS=12"], "b": [], "c": ["VIBEFLOW_NO_NONORTH=1"],
          "d": ["VIBEFLOW_BS_NO_TRANSPOSE=1"], "e": ["VIBEFLOW_CONVECTION=upwind"]}
HEADER = "backward-facing step, SST-1994,"
HEADER_RE = re.compile(r"backward-facing step, SST-1994, (.*): (\d+) cells, volume \S+ \(area \S+\), "
                       r"dt (\S+), U_in (\S+)$")
CLASSED = ("PISO", "initial", "resumed", "LIPX", "LIP", "step", "FINAL", "REFUSED")


class Stop(Exception):
    """Missing inputs, or the committed CFL3D file not as expected: the script stops."""


class NotJudged(Exception):
    """A downloaded file the contract reports and does not judge."""


def f4(v):
    return "not judged" if v is None else f"{v:.4f}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cp(path):
    """A march's .cp or .cpw, as two_col reads it, stopping on a line that does
    not parse by its number alone: the line may hold a value."""
    rows = []
    for no, ln in enumerate(open(path), 1):
        if not ln.strip() or ln.lstrip().startswith(("#", "v", "V")):
            continue
        v = _floats(ln.split())
        if v is None or len(v) != 2:
            raise Stop(f"{Path(path).name}: line {no} does not parse")
        rows.append(v)
    if not rows:
        raise Stop(f"{Path(path).name}: no rows")
    return np.array(rows, dtype=float)


# --- references ---------------------------------------------------------------

def lin(fx, fv, q):
    """A reference on one floor, linear between its own points in file order
    (x non-decreasing). At a duplicated x a query to its left uses the first
    entry and a query to its right the second; a query at a point gives every
    entry there. Outside the floor's span the quantity is not made: None."""
    q = float(q)
    if len(fx) == 0 or q < fx[0] or q > fx[-1]:
        return None
    hit = np.flatnonzero(fx == q)
    if len(hit):
        return [float(fv[i]) for i in hit]
    k = int(np.searchsorted(fx, q, side="right")) - 1
    t = (q - fx[k]) / (fx[k + 1] - fx[k])
    return [float((1.0 - t) * fv[k] + t * fv[k + 1])]


def lin1(fx, fv, q, what):
    v = lin(fx, fv, q)
    if v is None:
        raise NotJudged(f"{what}: x = {q:.6g} lies outside the floor")
    if len(v) != 1:
        raise NotJudged(f"{what}: x = {q:.6g} is a duplicated point")
    return v[0]


class Reference:
    """A reference's Cp on the bottom wall: its two floors in file order, in the
    frame (CFL3D's level-1 file as published; any other shifted by a constant so
    that its value at x = 40 is C40), and its lip points at its own x."""

    def __init__(self, name, rows, c40=None, committed=False):
        self.name, self.committed, self.issues = name, committed, []
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 2 or len(rows) < 4:
            self._fail("no rows of x and Cp")
        if not np.all(np.isfinite(rows)):
            self._fail("an x or Cp that is not finite")
        # The x = 0 count before split_floors, whose assert -O strips.
        nzero = int(np.count_nonzero(rows[:, 0] == 0.0))
        if nzero != 2:
            self._fail(f"{nzero} points at x = 0, not two")
        up, low = split_floors(rows)
        for label, fl in (("upstream", up), ("lower", low)):
            if np.any(np.diff(fl[:, 0]) < 0.0):
                self._fail(f"x decreases on its {label} floor")
        self.raw_up, self.raw_low = up.copy(), low.copy()
        r40 = lin(low[:, 0], low[:, 1], 40.0)
        self.r40 = r40[0] if r40 is not None and len(r40) == 1 else None
        if committed:
            self.shift = 0.0
        elif self.r40 is None:
            self.shift = None
            self.issues.append("its lower floor does not reach both x <= 40 and x >= 40" if r40 is None
                               else "x = 40 is a duplicated point of its lower floor")
        else:
            self.shift = c40 - self.r40
        s = self.shift or 0.0
        self.up = np.column_stack([up[:, 0], up[:, 1] + s])
        self.low = np.column_stack([low[:, 0], low[:, 1] + s])
        if not (up[0, 0] <= -4.0 and up[-1, 0] >= 0.0):
            self.issues.append("its upstream floor does not span -4 <= x <= 0")
        self.spans30 = low[0, 0] <= 0.0 and low[-1, 0] >= 30.0
        near = np.flatnonzero(np.abs(up[:, 0] - X_LIP) <= LIP_TOL)
        if len(near) != 1:
            self._fail(f"{len(near)} upstream points within 1e-5 of -0.0208296, not one", judged_only=True)
            self.lip = None
        else:
            self.lip = {"corner": (0.0, float(self.up[-1, 1])),
                        "-0.021": (float(self.up[near[0], 0]), float(self.up[near[0], 1]))}
        if committed and self.issues:
            raise Stop(f"{name}: {'; '.join(self.issues)}")

    def _fail(self, why, judged_only=False):
        if self.committed:
            raise Stop(f"{self.name}: {why}")
        if judged_only:
            self.issues.append(why)
            return
        raise NotJudged(why)

    @property
    def judged(self):
        return not self.issues

    def c_up(self, x):
        return lin1(self.up[:, 0], self.up[:, 1], x, f"{self.name}, upstream floor")

    def c_low(self, x):
        return lin1(self.low[:, 0], self.low[:, 1], x, f"{self.name}, lower floor")


def cfl3d():
    """CFL3D's level-1 file as gate 4 reads it, and C40."""
    ref = Reference("CFL3D level 1", two_col(CFL3D_CP), committed=True)
    try:
        return ref, ref.c_low(40.0)
    except NotJudged as e:
        raise Stop(str(e)) from None


# --- downloaded files ---------------------------------------------------------

def read_text(path):
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="replace")


def tokens(line):
    return [t for t in re.split(r"[\s,]+", line.strip()) if t]


def is_num(tok):
    try:
        float(tok.replace("D", "E").replace("d", "E"))
        return True
    except ValueError:
        return False


def num(tok):
    return float(tok.replace("D", "E").replace("d", "E"))


def is_row(tk):
    return bool(tk) and all(is_num(t) for t in tk)


def layout(text):
    """The parser of the first revision: (names or None, x column, Cp column)
    or NotJudged, without reading a value."""
    lines = text.splitlines()
    named, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not is_row(tokens(s)) and s.lower().startswith("variables"):
            rest = s.split("=", 1)[1] if "=" in s else ""
            if '"' in rest:
                names = re.findall(r'"([^"]*)"', rest)
                while i + 1 < len(lines) and re.fullmatch(r'\s*("[^"]*"[\s,]*)+', lines[i + 1]):
                    i += 1
                    names += re.findall(r'"([^"]*)"', lines[i])
            else:
                names = tokens(rest)
            named.append(names)
        i += 1
    widths = Counter(len(tk) for tk in map(tokens, lines) if is_row(tk))
    if not widths:
        raise NotJudged("no numeric rows")
    if not named:
        if set(widths) != {2}:
            raise NotJudged("no variables line, and not every numeric row has two columns")
        return None, 0, 1
    if any(n != named[0] for n in named):
        raise NotJudged("its variables lines differ")
    names = named[0]
    norm = [re.sub(r"[^a-z0-9]", "", n.lower()) for n in names]
    xs = [k for k, n in enumerate(norm) if n in ("x", "xh")]
    cps = [k for k, n in enumerate(norm) if n == "cp"]
    if len(xs) != 1 or len(cps) != 1:
        raise NotJudged(f"not one x column and one Cp column among {masked(names)}")
    if "-" in names[cps[0]] or "−" in names[cps[0]]:     # -Cp, (-Cp), a Unicode minus
        raise NotJudged(f"its Cp column is named {masked(names[cps[0]])}")
    if set(widths) != {len(names)}:
        raise NotJudged(f"numeric rows of {sorted(widths)} columns against {len(names)} names")
    return names, xs[0], cps[0]


def parse_download(path):
    """A downloaded file's (x, Cp) rows in file order across zones."""
    text = read_text(path)
    _, ix, icp = layout(text)
    rows = [[num(tk[ix]), num(tk[icp])] for tk in map(tokens, text.splitlines()) if is_row(tk)]
    return np.array(rows, dtype=float)


MASK = re.compile(r"[-+]?(\d+\.\d*|\.\d+)([EeDd][-+]?\d+)?|[-+]?\d+[EeDd][-+]?\d+")


def masked(v):
    """Text from a downloaded file, its real numbers masked as headers masks them."""
    return MASK.sub("<num>", repr(v))


def mode_headers(args):
    files = [(o, getattr(args, o.replace("-", "_"))) for o in ("fun3d", "cfl3d-finest", "fun3d-finest", "cfl3d-l1")]
    files = [(o, Path(p)) for o, p in files if p]
    if not files:
        raise Stop("missing: at least one of --fun3d, --cfl3d-finest, --fun3d-finest, --cfl3d-l1")
    for opt, path in files:
        need(path, f"--{opt}")
        text = read_text(path)
        print(f"--{opt} {path}")
        print(f"  sha256 {sha256(path)}")
        widths = Counter()
        for ln in text.splitlines():
            tk = tokens(ln)
            if is_row(tk):
                widths[len(tk)] += 1
                continue
            s = ln.strip().lstrip("#!%*/ \t")
            first = tokens(s)[0] if tokens(s) else ""
            if first and re.search("[A-Za-z]", first) and not is_num(first):
                print(f"  header: {MASK.sub('<num>', ln.rstrip())}")
            else:
                print(f"  header: {len(tk)} tokens, {sum(is_num(t) for t in tk)} numbers")
        print("  numeric rows: " + (", ".join(f"{n} of {w} columns" for w, n in sorted(widths.items())) or "none"))
        try:
            names, ix, icp = layout(text)
            print(f"  layout: {'columns ' + masked(names) if names else 'two columns, no variables line'}; "
                  f"x is column {ix + 1}, Cp column {icp + 1}")
        except NotJudged as e:
            print(f"  layout: fits neither layout ({e}): reported, not judged")
    return 0


# --- labels -------------------------------------------------------------------

def grid_name(s):
    s = s.strip().lower()
    if s.endswith(".gz"):
        s = s[:-3]
    if s.endswith(".p2dfmt"):
        s = s[:-7]
    return s


def zone_dims(path):
    return [(x.shape[1], x.shape[0]) for x, _ in read_p2d(path)]


def grids(args):
    """The grids a label may name: TMR's level 1 and, once committed, the finest."""
    g = {"level 1": (grid_name(GRID_L[1].name), zone_dims(GRID_L[1]))}
    if args.grid0:
        need(Path(args.grid0), "--grid0")
        g["finest"] = (grid_name(Path(args.grid0).name), zone_dims(args.grid0))
    return g


def dims_match(given, dims):
    try:
        if len(given) != len(dims):
            return False
        for z, (ni, nj) in zip(given, dims):
            if len(z) != 2 or sorted(float(v) for v in z) != sorted((float(ni), float(nj))):
                return False
    except (TypeError, ValueError):
        return False
    return True


def place_variant(place):
    v = place.get("variant")
    if not v or v == "SST":
        return None
    return "SSTm" if v == "SSTm" else "other"


def place_grid(place, gr):
    by_name = by_dims = None
    if place.get("grid_name"):
        n = grid_name(place["grid_name"])
        by_name = next((g for g, (gn, _) in gr.items() if gn == n), "other")
    if place.get("grid_dims"):
        by_dims = next((g for g, (_, gd) in gr.items() if dims_match(place["grid_dims"], gd)), "other")
    if by_name and by_dims and by_name != by_dims:
        return "other"
    return by_name or by_dims


def check_labels(path, labels_path, want, gr):
    """The contract on a file's labels: one place gives both a variant and a
    grid, every place that gives either agrees, and they are SSTm and `want`."""
    need(labels_path, "--labels")
    layout_stop = Stop(f"{labels_path} is not labels.json's layout")
    try:
        entries = json.loads(Path(labels_path).read_text())["files"]
    except (ValueError, KeyError, TypeError):
        raise layout_stop from None
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise layout_stop
    entry = next((e for e in entries if e.get("name") == Path(path).name), None)
    if entry is None:
        raise NotJudged("no entry in labels.json")
    places = entry.get("places")
    text_fields = ("place", "text", "variant", "grid_name")
    if not isinstance(places, list) or not all(
            isinstance(p, dict) and all(p.get(k) is None or isinstance(p[k], str) for k in text_fields)
            for p in places):
        raise layout_stop
    if entry.get("sha256") != sha256(path):
        raise NotJudged("its sha256 is not labels.json's")
    for p in places:
        # Each place is one of the three, and its tokens are taken from its text verbatim.
        text = p.get("text") or ""
        if p.get("place") not in ("file", "link", "page") or any(
                p.get(k) and p[k] not in text for k in ("variant", "grid_name")):
            raise NotJudged("a place in labels.json is not file, link or page, or its tokens are not in its text")
    got = [(p.get("place"), place_variant(p), place_grid(p, gr)) for p in places]
    for where, v, g in got:
        print(f"  label ({where}): variant {v or 'none'}, grid {g or 'none'}")
    if not any(v and g for _, v, g in got):
        raise NotJudged("no one place gives both a variant and a grid")
    vs, gs = {v for _, v, _ in got if v}, {g for _, _, g in got if g}
    if len(vs) > 1 or len(gs) > 1:
        raise NotJudged("its places disagree")
    v, g = vs.pop(), gs.pop()
    if v != "SSTm" or g != want:
        raise NotJudged(f"labelled {v} on {g}, not SSTm on {want}")


def downloaded(path, opt, c40, labels_path, want, gr):
    """A fetched file read under the contract: (Reference or None, reasons it is
    reported, not judged)."""
    need(Path(path), opt)
    why = []
    try:
        check_labels(path, labels_path, want, gr)
    except NotJudged as e:
        why.append(str(e))
    try:
        ref = Reference(f"{opt} {Path(path).name}", parse_download(path), c40)
    except NotJudged as e:
        return None, why + [str(e)]
    return ref, why + ref.issues


# --- the logs -----------------------------------------------------------------

Sample = namedtuple("Sample", "step p40 c4")
Fifty = namedtuple("Fifty", "step dt seconds")
Ramp = namedtuple("Ramp", "step kind")


class Attempt:
    def __init__(self, header):
        m = HEADER_RE.match(header)
        self.mesh, self.cells, self.dt, self.uin = (m.group(1), int(m.group(2)), m.group(3),
                                                    m.group(4)) if m else (None,) * 4
        self.correctors = self.initial = self.resumed = self.notfinite = None
        self.lipx, self.lip, self.fifty, self.ramps, self.finals = [], [], [], [], []
        self.refused = False
        self.kind = self.last = None


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.attempts, self.truncated, self.errors, self.resumes = [], [], [], []
        self.lip, self.fifty, self.ramps, self.all_fifty = [], [], [], []
        self.before = self.ignored = 0
        self.final = None
        self.cls = None
        self.lipx = None

    @property
    def fresh(self):
        return [i for i, a in enumerate(self.attempts) if a.kind == "fresh"]

    @property
    def refused(self):
        return [i for i, a in enumerate(self.attempts) if a.kind == "refused"]

    @property
    def usable(self):
        return not self.errors and not self.fresh

    def structure(self):
        kinds = ", ".join(a.kind for a in self.attempts) or "none"
        return (f"{len(self.attempts)} attempt(s) ({kinds}); resumes at {self.resumes or 'none'}; "
                f"{len(self.truncated)} truncated line(s); {len(self.lip)} kept LIP lines; "
                f"class {self.cls}")


def first_attempt(L):
    """A log's first attempt that is not a refused attempt, or None."""
    return next((a for a in L.attempts if a.kind != "refused"), None)


def _floats(tk):
    """The fields as finite floats, or None: a field that is not finite does not parse."""
    try:
        v = [float(t) for t in tk]
    except ValueError:
        return None
    return v if all(np.isfinite(v)) else None


def _final_ok(d):
    try:
        for k in ("xr", "cf4", "uc4", "dt", "time", "seconds"):
            float(d[k])
        for k in ("steps", "full", "bounded"):
            int(d[k])
        return len(d["bubble"]) == 2 and d["settled"] in ("steady", "quasi", "no")
    except (KeyError, ValueError, TypeError):
        return False


def _ran(a):
    return bool(a.lip or a.fifty or a.ramps or a.notfinite is not None or a.finals)


def parse_log(path):
    """ADR-046's reading of a march's log: its attempts, the LIP, 50-step and
    ramp lines kept after each resume, its class and its errors. Errors and
    truncated lines are reported by their place in the log, never by content."""
    L = Log(path)
    lines = []
    for no, ln in enumerate(Path(path).read_text(errors="replace").splitlines(), 1):
        # A header inside a line: the text before it is a truncated line.
        start = 0
        for m in re.finditer(re.escape(HEADER), ln):
            if m.start() > start and ln[start:m.start()].strip():
                L.truncated.append((no, m.start() - start))
                start = m.start()
        lines.append((no, ln[start:]))
    bad = []                      # classed lines that do not parse: (attempt, line, what)
    for no, ln in lines:
        s = ln.strip()
        tk = s.split()
        if not tk:
            continue
        if tk[0] == "backward-facing":
            L.attempts.append(Attempt(s))
            L.attempts[-1].last = no
            continue
        if tk[0].startswith("REFUSED"):      # ADR-047: a line beginning REFUSED
            tk[0] = "REFUSED"
            # A refusal comes before any step: one printed before any header,
            # after a run's lines or after another refusal is an attempt of its own.
            if not L.attempts or _ran(L.attempts[-1]) or L.attempts[-1].refused:
                L.attempts.append(Attempt(""))
        if tk[0] not in CLASSED:
            continue
        if not L.attempts:
            L.before += 1
            continue
        a, ai = L.attempts[-1], len(L.attempts) - 1
        a.last = no                          # the attempt's last classed line so far
        if tk[0] == "PISO":
            m = re.match(r"PISO correctors (\d+),", s)
            a.correctors = int(m.group(1)) if m else None
        elif tk[0] == "initial":
            m = re.match(r"initial state: (.*) \(\d+ coarse cells\)$", s)
            a.initial = m.group(1) if m else None
        elif tk[0] == "resumed":
            m = re.match(r"resumed from (.*?): step (\d+),", s)
            if m:
                a.resumed = int(m.group(2))
        elif tk[0] == "LIPX":
            v = _floats(tk[1:]) if len(tk) == 5 else None
            if v is None:
                bad.append((ai, no, "LIPX"))
            else:
                a.lipx.append(tuple(tk[1:]))
        elif tk[0] == "LIP":
            v = _floats(tk[2:]) if len(tk) == 7 and tk[1].isdigit() else None
            if v is None:
                bad.append((ai, no, "LIP"))
            else:
                a.lip.append(Sample(int(tk[1]), v[0], tuple(v[1:])))
        elif tk[0] == "step":
            if s.endswith("not finite"):
                m = re.match(r"step (\d+): not finite$", s)
                if m:
                    a.notfinite = int(m.group(1))
                else:
                    L.ignored += 1
            elif len(tk) > 2 and tk[2] == "t":
                ok = (len(tk) == 27 and tk[1].isdigit() and tk[4] == "dt" and tk[25].startswith("(")
                      and tk[26] == "s)" and _floats([tk[3], tk[5], tk[25][1:]]) is not None)
                if not ok:
                    bad.append((ai, no, "50-step"))
                else:
                    a.fifty.append(Fifty(int(tk[1]), tk[5], float(tk[25][1:])))
            elif len(tk) > 2 and tk[2] == "ramp:" and tk[1].isdigit() and tk[-1] in ("holds", "doubles", "halves"):
                a.ramps.append(Ramp(int(tk[1]), tk[-1]))
            else:
                L.ignored += 1
        elif tk[0] == "FINAL":
            try:
                d = parse_final(s)
            except (IndexError, ValueError):
                d = None
            if d is None or not _final_ok(d):
                bad.append((ai, no, "FINAL"))
            else:
                a.finals.append(d)
        elif tk[0] == "REFUSED":
            a.refused = True
    # A line that does not parse and is its attempt's last classed line is what
    # an interruption leaves (whatever unclassed lines the next process printed
    # before its header follow it): a truncated line, not used. Any other is an error.
    for ai, no, what in bad:
        if no == L.attempts[ai].last:
            L.truncated.append((no, None))
        else:
            L.errors.append(f"log line {no}: a {what} line does not parse")

    # The attempts: refused, resumed, the first, or a silent fresh start.
    first = False
    for a in L.attempts:
        a.kind = ("refused" if a.refused and not _ran(a) else "resumed" if a.resumed is not None
                  else "fresh" if first else "first")
        first |= a.kind != "refused"
    for a in L.attempts:
        if a.kind in ("refused", "fresh"):   # a silent fresh start: its structure only
            continue
        if a.kind == "resumed":
            n = a.resumed
            L.lip = [x for x in L.lip if x.step <= n]
            L.fifty = [x for x in L.fifty if x.step <= n]
            L.ramps = [x for x in L.ramps if x.step <= n]
            L.resumes.append(n)
        L.lip += a.lip
        L.fifty += a.fifty
        L.ramps += a.ramps
        L.all_fifty += a.fifty

    # The march's last attempt: a refused attempt keeps and drops nothing.
    ran = [a for a in L.attempts if a.kind != "refused"]
    last = ran[-1] if ran else None
    nfinal = sum(len(a.finals) for a in L.attempts)
    if nfinal > 1:
        L.errors.append("more than one FINAL")
    elif nfinal == 1 and not last.finals:
        L.errors.append("a FINAL outside its last attempt")
    lipx = {x for a in L.attempts for x in a.lipx}
    if len(lipx) > 1:
        L.errors.append("its LIPX lines differ")
    L.lipx = lipx.pop() if len(lipx) == 1 else None
    steps = [x.step for x in L.lip]
    if len(steps) != len(set(steps)):
        L.errors.append("two kept LIP lines share a step")

    L.final = last.finals[0] if last and len(last.finals) == 1 else None
    # A log with a silent fresh start keeps its structure only: no step checks.
    check = not L.fresh
    if last is not None and last.notfinite is not None:
        L.cls = "diverged"
        nf = last.notfinite
        if check and (L.final is None or int(L.final["steps"]) != nf or any(x.step >= nf for x in L.lip)):
            L.errors.append("diverged, and the not-finite step is not FINAL's steps or a kept LIP line is at or above it")
    elif L.final is None or L.final["settled"] == "no":
        L.cls = "unsettled"
    else:
        L.cls = L.final["settled"]
        if check and (not L.lip or L.lip[-1].step != int(L.final["steps"])):
            L.errors.append("settled, and its last kept LIP line's step is not FINAL's steps")
    return L


def window(L):
    """A settled march's window: its kept LIP samples with S - 1,000 < step <= S,
    S being FINAL's steps. Returns (samples, contains a resume, errors)."""
    if L.cls not in ("steady", "quasi"):
        return [], False, ["not settled: no window"]
    S, errs = int(L.final["steps"]), []
    if int(L.final["full"]) < 1000:
        errs.append(f"FINAL's full is {L.final['full']}, below 1,000")
    want = "%.2e" % float(L.final["dt"])
    for f in L.fifty:
        if S - 1000 < f.step <= S and f.dt != want:
            errs.append(f"the 50-step line at step {f.step} prints dt {f.dt}, not {want}")
    for r in L.ramps:
        if (r.kind == "doubles" and r.step >= S - 998) or (r.kind == "halves" and r.step >= S - 999):
            errs.append(f"a kept '{r.kind}' line at step {r.step}")
    samples = [x for x in L.lip if S - 1000 < x.step <= S]
    if not samples:
        errs.append("no kept LIP sample in the window")
    return samples, any(n >= S - 1000 for n in L.resumes), errs


def lip_values(lipx, sample, ref, point):
    """A sample's d at a lip point of R: (gate reading, face reading)."""
    x4 = np.array([float(v) for v in lipx])
    c4 = np.array(sample.c4)
    xl, rl = ref.lip[point]
    d_g = float(floor_values([xl], True, x4, c4)[0]) - rl
    e = c4 - np.array([ref.c_up(x) for x in x4])
    d_f = float(floor_values([xl], True, x4, e)[0])
    return d_g, d_f


def sample_values(lipx, sample, ref):
    out = {}
    for p in POINTS:
        out[(p, "G")], out[(p, "F")] = lip_values(lipx, sample, ref, p)
    return out


def window_stats(L, samples, ref):
    """B (the range), m (numpy.mean) and the last sample's value of each lip value."""
    vals = [sample_values(L.lipx, s, ref) for s in samples]
    st = {}
    for k in vals[0]:
        a = np.array([v[k] for v in vals])
        st[k] = {"B": float(a.max() - a.min()), "m": float(np.mean(a)), "last": float(a[-1])}
    return vals, st


def single_march_rule(vals, pred):
    """A rule about one march: holds at every sample under both readings,
    fails at every one under both, otherwise inconclusive."""
    got = [pred(v, r) for v in vals for r in READINGS]
    return "holds" if all(got) else "fails" if not any(got) else "inconclusive"


# --- final states ---------------------------------------------------------------

def floor_readings(cp, ref, upstream):
    """d at R's points on one floor of a .cp (or .cpw): (x, gate, face)."""
    fl = ref.up if upstream else ref.low
    xr, vr = fl[:, 0], fl[:, 1]
    m = cp[:, 0] < 0.0 if upstream else cp[:, 0] > 0.0
    if np.count_nonzero(m) < 2:
        raise NotJudged(f"fewer than two of our faces on the {'upstream' if upstream else 'lower'} floor")
    d_g = floor_values(xr, upstream, cp[:, 0], cp[:, 1]) - vr
    fx, fv = cp[m, 0], cp[m, 1]
    keep = (fx >= xr[0]) & (fx <= xr[-1])
    fx, fv = fx[keep], fv[keep]
    if len(fx) < 2:
        raise NotJudged(f"fewer than two faces inside {ref.name}'s span on a floor")
    c = np.array([lin1(xr, vr, x, ref.name) for x in fx])
    d_f = floor_values(xr, upstream, fx, fv - c)
    return xr, d_g, d_f


def final_lip(cp, ref):
    """d at R's two lip points from a final state's whole upstream floor."""
    xr, d_g, d_f = floor_readings(cp, ref, True)
    out = {}
    for p in POINTS:
        k = len(xr) - 1 if p == "corner" else int(np.flatnonzero(xr == ref.lip[p][0])[0])
        out[(p, "G")], out[(p, "F")] = float(d_g[k]), float(d_f[k])
    return out


def report_cp(label, path, ref):
    cp = read_cp(path)
    print(f"  {label} against {ref.name} ({path.name})")
    try:
        xr, d_g, d_f = floor_readings(cp, ref, True)
        for x, g, f in zip(xr, d_g, d_f):
            if -4.0 <= x <= 0.0:
                print(f"    x {x:+.7f}  d gate {g:+.6f}  face {f:+.6f}")
    except NotJudged as e:               # each floor on its own
        print(f"    upstream floor: not computed ({e})")
    xr, d_g, d_f = floor_readings(cp, ref, False)
    m = (xr > 0.0) & (xr <= 30.0)
    for name, d in (("gate", d_g), ("face", d_f)):
        if not m.any():
            print(f"    behind the step, 0 < x <= 30: no point of {ref.name}")
            break
        k = np.argmax(np.abs(d[m]))
        print(f"    behind the step, 0 < x <= 30: largest |d| {abs(d[m][k]):.6f} at x = {xr[m][k]:.4f} ({name})")


def report_march(name, L, ref, extra_refs=(), cp=None, cpw=None):
    """Code's list, for one march."""
    print(f"\n{name}: {L.path}")
    print(f"  {L.structure()}")
    for no, n in L.truncated:
        print(f"  truncated line at log line {no}{f' ({n} characters)' if n else ''}, not used")
    for e in L.errors:
        print(f"  error: {e}")
    if L.fresh:
        print(f"  attempts {L.fresh}: silent fresh starts; reported with their structure only")
        return
    if L.final:
        d = L.final
        print(f"  FINAL: x_r {d['xr']}  Cf(-4) {d['cf4']}  u_c(-4) {d['uc4']}  bubble "
              f"{d['bubble'][0]:.5f}-{d['bubble'][1]:.5f}  dt {d['dt']}  steps {d['steps']}  full {d['full']}  "
              f"settled {d['settled']}  seconds {d['seconds']}")
    s = {f.step: f.seconds for f in L.fifty}
    if 100 in s and 50 in s:
        print(f"  time per step: {s[100] / 100:.3f} s over the first 100 steps; (s100 - s50)/50 "
              f"{(s[100] - s[50]) / 50:.3f} s")
    # A reference without its lip points (a file the contract does not judge) gives
    # no lip series. A Cp file that does not parse, or a reading that cannot be
    # answered, is reported here as not computed: the step decides what it means.
    extra_refs = tuple(r for r in extra_refs if r is not None and r.shift is not None)
    for r in (ref,) + extra_refs:
        for lab, p in (("Cp", cp), ("Cp from .cpw", cpw)):
            if p is not None:
                try:
                    report_cp(lab, p, r)
                except (Stop, NotJudged) as e:
                    print(f"  {lab} against {r.name}: not computed ({e})")
    if L.errors or L.lipx is None:
        return
    for r in (ref,) + tuple(x for x in extra_refs if x.lip is not None):
        try:
            vals = [(smp.step, sample_values(L.lipx, smp, r)) for smp in L.lip]
        except NotJudged as e:           # an extra reference that does not reach the lip faces
            if r is ref:
                raise
            print(f"  the lip series against {r.name}: not computed ({e})")
            continue
        print(f"  the lip series against {r.name} (step: corner gate, face; -0.021 gate, face)")
        for step, v in vals:
            print(f"    {step:6d}  {v[('corner', 'G')]:+.6f} {v[('corner', 'F')]:+.6f}  "
                  f"{v[('-0.021', 'G')]:+.6f} {v[('-0.021', 'F')]:+.6f}")


YET = "not judged yet (no FINAL line: running, or awaiting its resume)"


def no_final_yet(L):
    """A march with no FINAL line has not ended: it runs, or waits for its resume
    (step 7 resumes every interrupted march and stops none by hand)."""
    return not L.errors and not L.fresh and L.cls == "unsettled" and L.final is None


def class_line(L, win_errors=()):
    if L.errors or L.fresh or win_errors:
        return "reported, not judged"
    if no_final_yet(L):
        return YET
    return {"steady": "settled steady", "quasi": "settled quasi", "diverged": "diverged",
            "unsettled": "not settled"}[L.cls]


def check_command(L, mesh, dt, uin, correctors, init):
    """The log against its command: the first attempt shows the initial state,
    every later one a resume with the same mesh, dt, U_in and correctors."""
    errs = []
    real = os.path.realpath
    for i, a in enumerate(x for x in L.attempts if x.kind != "refused"):
        if a.mesh is None or real(a.mesh) != real(mesh):
            errs.append(f"attempt {i}: mesh {a.mesh}, not {mesh}")
        if a.dt != "%g" % float(dt):
            errs.append(f"attempt {i}: dt {a.dt}, not {'%g' % float(dt)}")
        if a.uin != "%.6f" % float(uin):
            errs.append(f"attempt {i}: U_in {a.uin}, not {'%.6f' % float(uin)}")
        if a.correctors != correctors:
            errs.append(f"attempt {i}: {a.correctors} correctors, not {correctors}")
        if i == 0 and (a.initial is None or real(a.initial) != real(init)):
            errs.append(f"attempt 0: initial state {a.initial}, not {init}")
        if i > 0 and a.resumed is None:
            errs.append(f"attempt {i}: no 'resumed from' line")
    return errs


# --- inputs -------------------------------------------------------------------

def need(path, opt):
    if path is None or not Path(path).exists():
        raise Stop(f"missing: {opt} ({path})" if path else f"missing: {opt}")
    return Path(path)


def s0_dir(args):
    d = need(args.s0, "--s0")
    for n in S0_MARCHES:
        need(d / f"{n}.log", f"S0's {n}.log")
    return d


def s0_u_in(d):
    """U_in as backstep_gate.py computes it, and s, the string it passes."""
    cal = parse_log(d / "L2_cal.log")
    if not cal.usable or cal.cls not in ("steady", "quasi"):
        raise Stop("S0's L2_cal has not settled, or its log is an error: no U_in")
    u = UC_CFL3D / final(d / "L2_cal.log")["uc4"]
    return u, "%.12g" % u


def judged_points(d_s0):
    """The lip points judged on a base level: S0's d there above 0.015."""
    return [p for p in POINTS if d_s0[(p, "G")] > BOUND]


# --- step 1 -------------------------------------------------------------------

def tmr_faces(path):
    """Face centres on the upstream floor from TMR's grid alone."""
    pts = merge(read_p2d(path))[0]
    w = np.sort(pts[(np.abs(pts[:, 1] - 1.0) < 1e-9) & (pts[:, 0] <= 1e-12), 0])
    m = 0.5 * (w[1:] + w[:-1])
    return m[m >= -110.0]


def step1(args, cfl, c40, quiet=False):
    say = (lambda *a: None) if quiet else print
    levels = [(1, GRID_L[1]), (2, GRID_L[2])] + ([(0, Path(args.grid0))] if args.grid0 else [])
    up = cfl.up
    say(f"C40 (CFL3D's lower floor at x = 40): {c40:+.6f}")
    findings, lines = [], []
    for lv, path in levels:
        m = tmr_faces(need(path, "--grid0" if lv == 0 else str(path)))
        c = np.array([cfl.c_up(x) for x in m])
        b = floor_values(up[:, 0], True, m, c) - up[:, 1]
        say(f"\nlevel {lv}: {len(m)} upstream wall faces from {path.name}; b at CFL3D's upstream points "
            f"(* -4 <= x <= 0)")
        for x, v in zip(up[:, 0], b):
            say(f"  {'*' if -4.0 <= x <= 0.0 else ' '} x {x:+.7f}  b {v:+.6f}")
        if args.s0 and lv in (1, 2) and not quiet:   # outcome never prints it
            cp = read_cp(need(Path(args.s0) / f"L{lv}.cp", f"S0's L{lv}.cp"))
            xo = cp[cp[:, 0] < 0.0, 0]
            say(f"  S0's L{lv}.cp: {len(xo)} upstream faces, TMR's {len(m)}" +
                (f"; largest |m - x| {np.abs(m - xo).max():.3e}" if len(xo) == len(m) else ""))
        if lv in (1, 2):
            for i, p in enumerate(POINTS):
                k = len(up) - 1 if p == "corner" else int(np.flatnonzero(up[:, 0] == cfl.lip[p][0])[0])
                d = D045[lv][i]
                db, dc = d - b[k], d + c40
                lines.append(f"step 1 (R) level {lv} {p}: d-b {db:.4f} within 0.015 {'yes' if db <= BOUND else 'no'}, "
                             f"d+C40 {dc:.4f} within 0.015 {'yes' if dc <= BOUND else 'no'}")
                if db <= BOUND:
                    findings.append(f"step 1: on level {lv} at {p}, b alone brings gate 4's reading within 0.015")
                if dc <= BOUND:
                    findings.append(f"step 1: on level {lv} at {p}, C40 alone brings gate 4's reading within 0.015")
    say("")
    for ln in lines:
        say(ln)
    say("step 1 (R): reported, not judged")
    return findings


def mode_reading(args):
    cfl, c40 = cfl3d()
    step1(args, cfl, c40)
    return 0


# --- step 2 -------------------------------------------------------------------

def three_points(cfl):
    """Step 2's three other points, from CFL3D's x alone: checked before any
    value of FUN3D is read, since the script stops if one is not unique."""
    three = []
    for x0, d in OTHER3:
        k = np.flatnonzero(np.abs(cfl.up[:, 0] - x0) <= 1e-3)
        if len(k) != (2 if x0 == -4.0 else 1):
            raise Stop(f"CFL3D has {len(k)} upstream points within 0.001 of {x0:g}; a revision states the point")
        three.append((x0, d, k))
    return three


def step2(args, cfl, c40, gr, s0=None):
    """FUN3D against CFL3D on level 1: (line, result)."""
    if not args.fun3d:
        return "step 2 (F): not run", "not run"
    three = three_points(cfl)
    fun, why = downloaded(args.fun3d, "--fun3d", c40, args.labels, "level 1", gr)
    if fun is None:
        return f"step 2 (F): reported, not judged ({'; '.join(why)})", "reported"
    if fun.shift is None:
        print("  D unshifted only: FUN3D's lower floor gives no value at x = 40")
        for x, c in cfl.up:
            v = lin(fun.raw_up[:, 0], fun.raw_up[:, 1], x)
            if v is not None:
                print(f"    x {x:+.7f}  D unshifted " + " ".join(f"{u - c:+.6f}" for u in v))
        if s0 is not None:
            print("  S0's level-1 .cp against FUN3D: not computed (FUN3D has no frame)")
        return f"step 2 (F): reported, not judged ({'; '.join(why)})", "reported"
    print(f"  FUN3D at x = 40 (its own): {fun.r40:+.6f}; shift into the frame {fun.shift:+.6f}")
    print("  D = FUN3D - CFL3D at CFL3D's upstream points inside FUN3D's span (x, D, D unshifted)")
    for x, c in cfl.up:
        v = lin(fun.up[:, 0], fun.up[:, 1], x)
        if v is not None:
            print(f"    x {x:+.7f}  D " + " ".join(f"{u - c:+.6f}" for u in v) + "  unshifted " +
                  " ".join(f"{u - fun.shift - c:+.6f}" for u in v))
    beh = [(x, [u - c for u in v]) for x, c in cfl.low if 0.0 <= x <= 30.0
           for v in [lin(fun.low[:, 0], fun.low[:, 1], x)] if v is not None]
    if beh:
        x, dd = max(beh, key=lambda t: max(abs(u) for u in t[1]))
        print(f"  behind the step, 0 <= x <= 30: {len(beh)} points, largest |D| "
              f"{max(abs(u) for u in dd):.6f} at x = {x:.4f}")
    if s0 is not None:
        cp = s0 / "L1.cp"
        print("  S0's level-1 .cp against FUN3D, at FUN3D's points")
        for upstream in (True, False):       # each floor on its own
            fl = "upstream" if upstream else "lower"
            try:
                xr, d_g, d_f = floor_readings(read_cp(need(cp, "S0's L1.cp")), fun, upstream)
            except (NotJudged, Stop) as e:
                print(f"    {fl} floor: not computed ({e})")
                continue
            for name, d in (("gate", d_g), ("face", d_f)):
                k = int(np.argmax(np.abs(d)))
                print(f"    {fl} floor: largest |d| {abs(d[k]):.6f} at x = {xr[k]:.4f} ({name}), "
                      f"over {len(xr)} points")
            if upstream:
                for x, g, f in zip(xr, d_g, d_f):
                    if -4.0 <= x <= 0.0:
                        print(f"      x {x:+.7f}  d gate {g:+.6f}  face {f:+.6f}")
    if why:
        return f"step 2 (F): reported, not judged ({'; '.join(why)})", "reported"

    # The lip: D at CFL3D's two lip points.
    D = {p: lin1(fun.up[:, 0], fun.up[:, 1], cfl.lip[p][0], fun.name) - cfl.lip[p][1] for p in POINTS}
    # The three other points.
    other = []
    for x0, d, k in three:
        Ds = []
        for i in k:
            xi = cfl.up[i, 0]
            # At -4, each FUN3D entry there if it has two (within 1e-5 of CFL3D's x).
            fk = np.flatnonzero(np.abs(fun.up[:, 0] - xi) <= LIP_TOL) if x0 == -4.0 else []
            fv = [float(fun.up[j, 1]) for j in fk] if len(fk) == 2 else lin(fun.up[:, 0], fun.up[:, 1], xi)
            if fv is None:
                raise NotJudged(f"x = {xi:g} lies outside FUN3D's upstream floor")
            Ds += [u - cfl.up[i, 1] for u in fv]
        print(f"  {x0:g}: CFL3D x " + " ".join(f"{cfl.up[i, 0]:+.7f}" for i in k) +
              f", ADR-045's d {d:+.3f}, D " + " ".join(f"{u:+.6f}" for u in Ds))
        other.append(all(abs(d - u) + ROUND3 <= BOUND for u in Ds))
    E = dict(zip(POINTS, E045[1]))
    print("  lip: " + "  ".join(f"{p} D {D[p]:+.6f} E {E[p]:.4f}" for p in POINTS))
    if all(D[p] >= E[p] for p in POINTS) and all(other):
        return "step 2 (F): belongs to the reference", "reference"
    if all(D[p] <= NFLOOR for p in POINTS):
        return "step 2 (F): ruled out", "ruled out"
    return f"step 2 (F): D/E {D['corner'] / E['corner']:.4f} {D['-0.021'] / E['-0.021']:.4f}", "D/E"


def mode_reference(args):
    need(args.fun3d, "--fun3d")
    cfl, c40 = cfl3d()
    s0 = Path(args.s0) if args.s0 else None
    try:
        line, _ = step2(args, cfl, c40, grids(args), s0)
    except NotJudged as e:
        line = f"step 2 (F): reported, not judged ({e})"
    print(line)
    return 0


# --- step 3 -------------------------------------------------------------------

def compare_l1(args):
    """A downloaded CFL3D level-1 file against the committed one (the contract)."""
    p = need(Path(args.cfl3d_l1), "--cfl3d-l1")
    if p.read_bytes() == CFL3D_CP.read_bytes():
        print("  the downloaded CFL3D level-1 file equals the committed one byte for byte")
        return
    try:
        r = parse_download(p)
        c = two_col(CFL3D_CP)
        if r.shape == c.shape:
            print(f"  the downloaded CFL3D level-1 file differs: largest |dx| {np.abs(r[:, 0] - c[:, 0]).max():.3e}, "
                  f"largest |dCp| {np.abs(r[:, 1] - c[:, 1]).max():.3e}")
        else:
            print(f"  the downloaded CFL3D level-1 file differs: {len(r)} rows against {len(c)}")
    except NotJudged as e:
        print(f"  the downloaded CFL3D level-1 file differs and does not parse ({e})")


def step3(args, cfl, c40, gr):
    """CFL3D on the finest grid against its level-1 file: (line, T, finest ref)."""
    if args.cfl3d_l1:
        compare_l1(args)
    if not args.cfl3d_finest:
        return "step 3 (T): not run", "not run", None
    if "finest" not in gr:
        why = ["no committed finest grid (--grid0) to class its label against"]
        try:
            fin = Reference("CFL3D finest", parse_download(need(Path(args.cfl3d_finest), "--cfl3d-finest")), c40)
            why += fin.issues
        except NotJudged as e:
            fin = None
            why.append(str(e))
    else:
        fin, why = downloaded(args.cfl3d_finest, "--cfl3d-finest", c40, args.labels, "finest", gr)
    if fin is not None and not fin.spans30:
        why.append("its lower floor does not span 0 <= x <= 30")
    if fin is None or fin.shift is None:
        return f"step 3 (T): reported, not judged ({'; '.join(why)})", "reported", None
    try:
        dref = {p: lin1(fin.up[:, 0], fin.up[:, 1], cfl.lip[p][0], fin.name) - cfl.lip[p][1] for p in POINTS}
    except NotJudged as e:
        return f"step 3 (T): reported, not judged ({'; '.join(why + [str(e)])})", "reported", fin
    print("  Delta_ref = CFL3D finest - CFL3D level 1: " + "  ".join(f"{p} {dref[p]:+.6f}" for p in POINTS))
    fun_ok = bool(args.fun3d and args.fun3d_finest)
    if fun_ok:
        try:
            three_points(cfl)    # step 2's stop comes before any value of FUN3D
        except Stop as e:
            print(f"  reported: FUN3D's change not computed ({e})")
            fun_ok = False
    if fun_ok:
        try:
            f1 = Reference("FUN3D level 1", parse_download(need(Path(args.fun3d), "--fun3d")), c40)
            f0 = Reference("FUN3D finest", parse_download(need(Path(args.fun3d_finest), "--fun3d-finest")), c40)
            if f1.shift is None or f0.shift is None:
                raise NotJudged("a FUN3D file has no frame")
            print("  reported: FUN3D finest - FUN3D level 1: " + "  ".join(
                f"{p} {lin1(f0.up[:, 0], f0.up[:, 1], cfl.lip[p][0], f0.name) - lin1(f1.up[:, 0], f1.up[:, 1], cfl.lip[p][0], f1.name):+.6f}"
                for p in POINTS))
        except NotJudged as e:
            print(f"  reported: FUN3D's change not computed ({e})")
    if why:
        return f"step 3 (T): reported, not judged ({'; '.join(why)})", "reported", fin
    if all(dref[p] >= NFLOOR for p in POINTS):
        t = "towards ours"
    elif all(dref[p] <= -NFLOOR for p in POINTS):
        t = "away"
    elif all(abs(dref[p]) < NFLOOR for p in POINTS):
        t = "unchanged"
    else:
        t = "mixed"
    return f"step 3 (T): {t}", t, fin


def mode_finest(args):
    if not args.cfl3d_finest and not args.cfl3d_l1:
        raise Stop("missing: --cfl3d-finest (or --cfl3d-l1 alone)")
    cfl, c40 = cfl3d()
    line, _, _ = step3(args, cfl, c40, grids(args))
    print(line)
    return 0


# --- step 4 -------------------------------------------------------------------

def flag(value, ref, unit, band):
    """Would the verdict change under half a unit of ADR-045's last digit?"""
    lo, hi = max(0.0, abs(value - ref) - unit / 2), abs(value - ref) + unit / 2
    return (lo <= band) != (hi <= band)


def step4(args, cfl, quiet=False):
    """S0: (final line, d_S0 per level, logs, U_in string, findings)."""
    say = (lambda *a: None) if quiet else print
    d = s0_dir(args)
    logs = {n: parse_log(d / f"{n}.log") for n in S0_MARCHES}
    wins = {}
    for n in ("L2", "L1"):
        if logs[n].usable and logs[n].cls in ("steady", "quasi"):
            wins[n] = window(logs[n])
    for n in S0_MARCHES:
        cp = cpw = None
        if n in ("L2", "L1"):
            cp = d / f"{n}.cp" if (d / f"{n}.cp").exists() else None
            cpw = d / f"{n}.cpw" if (d / f"{n}.cpw").exists() else None
        if not quiet:
            report_march(f"S0 {n}", logs[n], cfl, cp=cp, cpw=cpw)
        if n in wins:
            samples, res, errs = wins[n]
            for e in errs:
                say(f"  window error: {e}")
            if not errs and samples:
                _, st = window_stats(logs[n], samples, cfl)
                say(f"  window: {len(samples)} samples, steps {samples[0].step}-{samples[-1].step}"
                    f"{', contains a resume' if res else ''}")
                for (p, r), v in st.items():
                    say(f"    {p} {r}: m {v['m']:+.6f}  B {v['B']:.6f}")
    status = {n: class_line(logs[n], wins.get(n, (None, None, []))[2]) for n in S0_MARCHES}
    lines = [f"step 4 (S0) {n}: {status[n]}" for n in S0_MARCHES]
    findings = []
    if any(status[n] in ("diverged", "not settled") for n in S0_MARCHES):
        lines.append("step 4 (S0): failed chain")
        return lines, None, logs, None, ["step 4: S0's chain failed; a revision states one other start "
                                         "(steps 5 and 6 are void only if that chain fails too)"]
    if any(status[n] == "reported, not judged" for n in S0_MARCHES):
        lines.append("step 4 (S0): not judged (a log is an error, was started over, or fails its window checks)")
        return lines, None, logs, None, ["step 4: S0 not judged"]
    if any(status[n] == YET for n in S0_MARCHES):
        lines.append("step 4 (S0): not judged yet (a march has no FINAL line)")
        return lines, None, logs, None, ["step 4: S0 not judged yet"]
    for n in ("L2", "L1"):
        need(d / f"{n}.cp", f"S0's {n}.cp")

    u_in, s = s0_u_in(d)
    for n in ("L2", "L1"):
        for a in logs[n].attempts:
            if a.kind != "refused" and a.uin != "%.6f" % float(s):
                lines.append(f"step 4 (S0): not judged ({n}'s header prints U_in {a.uin}, not {'%.6f' % float(s)})")
                return lines, None, logs, s, ["step 4: S0 not judged"]
    dlv = {lv: final_lip(read_cp(d / f"L{lv}.cp"), cfl) for lv in (1, 2)}
    say("\nreproduction (S0, ADR-045, |difference|, band, margin)")
    ok = True
    rows = [("U_in", u_in, U_IN_045, BAND["u_in"])]
    for lv in (2, 1):
        fin = logs[f"L{lv}"].final
        for q in ("xr", "cf4", "uc4"):
            rows.append((f"level {lv} {q}", float(fin[q]), A045[(q, lv)], BAND[q]))
        for i, p in enumerate(POINTS):
            rows.append((f"level {lv} lip d {p}", dlv[lv][(p, "G")], (D045[lv][i], 1e-4), BAND[p]))
    for name, v, (ref, unit), band in rows:
        diff = abs(v - ref)
        ok &= diff <= band
        say(f"  {name:20s} {v:.8g}  {ref:.8g}  {diff:.3e}  {band:.1e}  margin {band - diff:+.3e}"
            f"{'  flagged: half a unit of the last digit changes it' if flag(v, ref, unit, band) else ''}")
    say(f"  U_in string s: {s}")
    for lv in (1, 2):
        for p in POINTS:
            v = dlv[lv][(p, "G")]
            lines.append(f"step 4 (S0) level {lv} {p}: {'judged' if v > BOUND else 'not judged'} (d_S0 {v:.4f})")
            if v <= BOUND:
                findings.append(f"step 4: S0's lip d on level {lv} at {p} is at most 0.015: "
                                f"the lip's sensitivity to the start or the toolchain")
    lines.append(f"step 4 (S0): {'reproduces' if ok else 'does not reproduce'}")
    return lines, dlv, logs, s, findings


def mode_s0(args):
    cfl, _ = cfl3d()
    lines, *_ = step4(args, cfl)
    print()
    for ln in lines:
        print(ln)
    return 0


# --- commands -----------------------------------------------------------------

def s0_inputs(args):
    """From S0's outputs: the binary, s, dt_S0 and S0's L2.hex."""
    d = s0_dir(args).resolve()
    _, s = s0_u_in(d)
    l2 = parse_log(d / "L2.log")
    if L2_bad(l2):
        raise Stop(f"S0's L2.log: {l2.errors or l2.cls}")
    # backstep_gate.py's layout: S0's directory is <clone>/build/backstep, its
    # meshes <clone>/cases/backstep, the binary <clone>/build/tests/backstep.
    clone, b = d.parent.parent, d.parent / "tests" / "backstep"
    if os.path.realpath(first_attempt(l2).mesh) != os.path.realpath(clone / "cases" / "backstep" / "L2.hex"):
        raise Stop(f"S0's L2 read {first_attempt(l2).mesh}, not {clone}/cases/backstep/L2.hex: "
                   "S0's directory is not <clone>/build/backstep")
    need(b, "S0's binary <clone>/build/tests/backstep")
    return {"dir": d, "bin": b, "s": s, "dt": l2.final["dt"], "l2hex": first_attempt(l2).mesh, "l2": l2}


def L2_bad(l2):
    return not l2.usable or l2.cls not in ("steady", "quasi") or first_attempt(l2) is None or first_attempt(l2).mesh is None


def step6_dt(dt_s0, k):
    return "%.12g" % (float(dt_s0) / 2) if k == "b" else dt_s0


def mode_commands(args):
    inp = s0_inputs(args)
    if not (args.g and args.l0) and not args.a:
        raise Stop("missing: --g and --l0 (step 5), or --a (step 6)")
    b = inp["bin"]
    print(f"binary {b}  sha256 {sha256(b) if b.exists() else 'missing'}")
    print(f"U_in string s {inp['s']}; dt_S0 {inp['dt']}; (b) {step6_dt(inp['dt'], 'b')}")
    env = "OMP_NUM_THREADS=2 OMP_PROC_BIND=false VIBEFLOW_BS_SAVE=1"
    print("No other VIBEFLOW_ variable is set: check `env | grep VIBEFLOW_` prints nothing.")
    if args.g and args.l0:
        g, l0 = Path(args.g).resolve(), Path(args.l0).resolve()
        cmd = f"{env} VIBEFLOW_BS_INIT={inp['dir'] / 'L1.state'} {b} {l0} {g / 'L0'} {DT_G} 5000 {inp['s']}"
        print("\nstep 5, two threads")
        print(f"  sha256sum {b}")
        print(f"  mkdir {g} && env {cmd} > {g / 'L0.log'} 2>&1")
        print(f"  resume: env VIBEFLOW_BS_RESUME=1 {cmd} >> {g / 'L0.log'} 2>&1")
    if args.a:
        a = Path(args.a).resolve()
        print(f"\nstep 6, two threads each\n  sha256sum {b}\n  mkdir -p {a}")
        for k in ABLATIONS:
            out = a / k / f"L2_{k}"
            cmd = " ".join([env, f"VIBEFLOW_BS_INIT={inp['dir'] / 'L2.state'}"] + CHANGE[k] +
                           [str(b), inp["l2hex"], str(out), step6_dt(inp["dt"], k), "5000", inp["s"]])
            print(f"  ({k}) mkdir {a / k} && env {cmd} > {out}.log 2>&1")
            print(f"      resume: env VIBEFLOW_BS_RESUME=1 {cmd} >> {out}.log 2>&1")
    return 0


# --- step 5 -------------------------------------------------------------------

def read_hex_points(path):
    """PolyMesh::fromHexFile's layout: points and cells, then the points; the
    first half lies on z = 0."""
    with open(path) as fh:
        npts, ncells = (int(v) for v in fh.readline().split()[:2])
        pts = np.loadtxt(fh, max_rows=npts)
    return ncells, pts[: npts // 2]


def nested(l0, l1):
    from scipy.spatial import cKDTree
    cells0, p0 = read_hex_points(l0)
    _, p1 = read_hex_points(l1)
    dist = cKDTree(p0).query(p1)[0]
    print(f"  nesting: L0 has {cells0} cells (nested: {CELLS_L0}); the largest distance from an L1 z = 0 "
          f"point to L0's is {dist.max():.3e} (nested: within 1e-9)")
    return cells0 == CELLS_L0 and dist.max() <= 1e-9


def step5(args, cfl, c40, gr, t3, fin, d_s0):
    """Ours on TMR's finest grid: (line, result, findings)."""
    # "If neither point qualifies for a step, that step is void": S0 alone decides
    # it, before step 5's march is asked for.
    jp = judged_points(d_s0[1])
    if not jp:
        return "step 5 (G): void (no lip point judged on level 1)", "void", ["step 5: void, no lip point judged"]
    if not args.g:
        return "step 5 (G): not run", "not run", []
    g = need(Path(args.g), "--g")
    log_p = need(g / "L0.log", "step 5's L0.log")
    l0, l1 = need(args.l0, "--l0"), need(args.l1, "--l1")
    inp = s0_inputs(args)
    L = parse_log(log_p)
    errs = check_command(L, l0, DT_G, inp["s"], 8, inp["dir"] / "L1.state")
    is_nested = nested(l0, l1)
    for e in errs:
        L.errors.append(f"command: {e}")
    cp_p = g / "L0.cp"
    report_march("step 5 L0", L, cfl, (fin,), cp=cp_p if cp_p.exists() else None,
                 cpw=g / "L0.cpw" if (g / "L0.cpw").exists() else None)
    if L.fresh or L.errors:
        return "step 5 (G): reported, not judged (its log is an error or was started over)", "error", []
    if no_final_yet(L):
        if any(f.seconds > SECONDS_G for f in L.all_fifty):
            return "step 5 (G): void (its printed seconds pass 604,800 with no FINAL)", "void", ["step 5: void"]
        return f"step 5 (G): {YET}", "yet", []
    if L.cls not in ("steady", "quasi"):
        return f"step 5 (G): void ({L.cls})", "void", ["step 5: void"]
    S = int(L.final["steps"])
    over = [f.step for f in L.all_fifty if f.step < S and f.seconds > SECONDS_G]
    at = [f.seconds for f in L.fifty if f.step == S]
    if not at:
        return "step 5 (G): reported, not judged (no 50-step line at FINAL's steps)", "error", []
    if over or at[0] > SECONDS_G:
        return "step 5 (G): void (its printed seconds pass 604,800 before it settled)", "void", ["step 5: void"]
    print(f"  reported: FINAL's seconds {L.final['seconds']} (including the outputs' writing)")
    cp = read_cp(need(cp_p, "step 5's L0.cp"))
    dl0 = final_lip(cp, cfl)
    delta = {p: d_s0[1][(p, "G")] - dl0[(p, "G")] for p in POINTS}
    e_s0 = {p: d_s0[1][(p, "G")] - BOUND for p in POINTS}
    frac = {p: delta[p] / e_s0[p] if p in jp else None for p in POINTS}
    print("  Delta = d_L1(S0) - d_L0, final states: " + "  ".join(
        f"{p} gate {delta[p]:+.6f} face {d_s0[1][(p, 'F')] - dl0[(p, 'F')]:+.6f}" for p in POINTS))
    fraction = f"step 5 (G): fraction {f4(frac['corner'])} {f4(frac['-0.021'])}"
    # "With only one, Delta/E_S0 is recorded there and nothing else" (final states, so nesting is moot).
    if len(jp) < 2:
        return fraction, "fraction", [f"step 5: fraction {f4(frac['corner'])} {f4(frac['-0.021'])}"]
    if not is_nested:
        print("  not nested: the lip values from the final .cp; " + "  ".join(
            f"{p} {r} {v:+.6f}" for (p, r), v in dl0.items()))
        return "step 5 (G): inconclusive (the grid is not nested)", "inconclusive", ["step 5: inconclusive"]
    samples, res, werrs = window(L)
    if werrs:
        for e in werrs:
            print(f"  window error: {e}")
        return "step 5 (G): reported, not judged (window errors)", "error", []
    print(f"  window: {len(samples)} samples, steps {samples[0].step}-{samples[-1].step}"
          f"{', flagged: it contains a resume' if res else ''}")
    vals, st = window_stats(L, samples, cfl)
    l1log = parse_log(inp["dir"] / "L1.log")
    s1 = window_stats(l1log, window(l1log)[0], cfl)[1]
    print("  reported: Delta from the window means: " + "  ".join(
        f"{p} {r} {s1[(p, r)]['m'] - st[(p, r)]['m']:+.6f}" for p in POINTS for r in READINGS))
    if t3 in ("towards ours", "unchanged", "away", "mixed") and fin is not None:
        rest = {}
        for r in READINGS:
            ok = True
            for upstream in (True, False):
                xr, d_g, d_f = floor_readings(cp, fin, upstream)
                d = d_g if r == "G" else d_f
                m = (xr >= -4.0) & (xr <= 0.0) if upstream else (xr >= 0.0) & (xr <= 30.0)
                if upstream:
                    m &= (xr != 0.0) & (xr != fin.lip["-0.021"][0])
                ok &= bool(np.all(np.abs(d[m]) <= BOUND))
            rest[r] = ok
        fvals = [sample_values(L.lipx, smp, fin) for smp in samples]
        rule = single_march_rule(fvals, lambda v, r: rest[r] and all(abs(v[(p, r)]) <= BOUND for p in POINTS))
        print(f"  the rule against CFL3D finest: {rule}")
        if rule == "inconclusive":
            return "step 5 (G): inconclusive", "inconclusive", ["step 5: inconclusive"]
        if rule == "holds":
            if t3 == "towards ours":
                return "step 5 (G): belongs to the grid", "grid", []
            return "step 5 (G): this solver needs a finer grid", "finer", ["step 5: this solver needs a finer grid"]
        return fraction, "fraction", [f"step 5: fraction {f4(frac['corner'])} {f4(frac['-0.021'])}"]
    rule = single_march_rule(vals, lambda v, r: all(abs(v[(p, r)]) <= BOUND for p in POINTS))
    print(f"  T not judged; the lip against CFL3D's level-1 file: {rule}")
    if rule == "inconclusive":
        return "step 5 (G): inconclusive", "inconclusive", ["step 5: inconclusive"]
    if rule == "holds":
        return "step 5 (G): this solver needs a finer grid", "finer", ["step 5: this solver needs a finer grid"]
    return fraction, "fraction", [f"step 5: fraction {f4(frac['corner'])} {f4(frac['-0.021'])}"]


def mode_grid(args):
    cfl, c40 = cfl3d()
    _, d_s0, *_ = step4(args, cfl, quiet=True)
    if d_s0 is None:
        raise Stop("S0 is not judged, so step 5 cannot be read (step 4 says what follows)")
    if judged_points(d_s0[1]):           # otherwise step 5 is void and needs no march
        for o in ("g", "l0", "l1"):
            need(getattr(args, o), f"--{o}")
    gr = grids(args)
    t3_line, t3, fin = step3(args, cfl, c40, gr)
    print(t3_line)
    line, *_ = step5(args, cfl, c40, gr, t3, fin, d_s0)
    print(line)
    return 0


# --- step 6 -------------------------------------------------------------------

def step6(args, cfl, d_s0):
    """The ablations: (lines, findings)."""
    # "If neither point qualifies for a step, that step is void": S0 alone decides
    # it, before the ablations are asked for.
    jp = judged_points(d_s0[2])
    if not jp:
        return ["step 6 (A): void (no lip point judged on level 2)"], ["step 6: void"]
    if not args.a:
        return ["step 6 (A): not run"], []
    root = need(Path(args.a), "--a")
    for k in ABLATIONS:                  # all six before any is read
        need(root / k / f"L2_{k}.log", f"ablation ({k})'s log")
    inp = s0_inputs(args)
    l2 = inp["l2"]
    s_samples, _, s_errs = window(l2)
    if s_errs:
        raise Stop(f"S0's L2 window: {s_errs}")
    _, s_st = window_stats(l2, s_samples, cfl)
    fin_s0 = l2.final
    e_s0 = {p: d_s0[2][(p, "G")] - BOUND for p in POINTS}
    out, conf, stats, logs = {}, {}, {}, {}
    for k in ABLATIONS:
        p = root / k / f"L2_{k}.log"
        L = parse_log(p)
        dt = step6_dt(inp["dt"], k)
        for e in check_command(L, inp["l2hex"], dt, inp["s"], 12 if k == "a" else 8, inp["dir"] / "L2.state"):
            L.errors.append(f"command: {e}")
        cp = root / k / f"L2_{k}.cp"
        report_march(f"step 6 ({k})", L, cfl, cp=cp if cp.exists() else None,
                     cpw=root / k / f"L2_{k}.cpw" if (root / k / f"L2_{k}.cpw").exists() else None)
        logs[k] = L
        if L.fresh or L.errors:
            conf[k] = "error"
            continue
        if no_final_yet(L):
            conf[k] = "yet"
            continue
        if L.cls not in ("steady", "quasi") or L.final["dt"] != "%.6g" % float(dt):
            conf[k] = "confounded"
            continue
        samples, res, werrs = window(L)
        if werrs:
            for e in werrs:
                print(f"  window error: {e}")
            conf[k] = "error"
            continue
        print(f"  window: {len(samples)} samples, steps {samples[0].step}-{samples[-1].step}"
              f"{', flagged: it contains a resume' if res else ''}")
        stats[k] = window_stats(L, samples, cfl)
    # Each march's line when step 6 is not judged as a whole.
    form = {None: "reported", "yet": YET, "confounded": "confounded",
            "error": "reported, not judged (its log or window is an error)"}
    lines = []
    if "yet" in conf.values():
        return ([f"step 6 (A) {k}: {form[conf.get(k)]}" for k in ABLATIONS] +
                ["step 6 (A): not judged yet (a march has no FINAL line)"], [])
    if conf.get("0"):
        lines += [f"step 6 (A) {k}: {form[conf.get(k)]}" for k in ABLATIONS]
        why = "(0) confounded" if conf["0"] == "confounded" else "(0)'s log or window is an error"
        lines.append(f"step 6 (A): not judged, {why}")
        return lines, [f"step 6: not judged, {why}"]
    findings = []
    st0 = stats["0"][1]
    for k in ABLATIONS:
        if conf.get(k) == "confounded":
            lines.append(f"step 6 (A) {k}: confounded")
            continue
        if conf.get(k):
            lines.append(f"step 6 (A) {k}: reported, not judged (its log or window is an error)")
            continue
        vals, st = stats[k]
        moves = {}
        for r in READINGS:
            ok = True
            for p in jp:
                d0 = s_st[(p, r)]["m"] - st0[(p, r)]["m"]
                dd = s_st[(p, r)]["m"] - st[(p, r)]["m"]
                last = s_st[(p, r)]["last"] - st[(p, r)]["last"]
                ok &= abs(dd) > max(abs(d0), (s_st[(p, r)]["B"] + st[(p, r)]["B"]) / 2)
                ok &= np.sign(last) == np.sign(dd) and last != 0.0
            moves[r] = bool(ok)
        mv = "holds" if all(moves.values()) else "no" if not any(moves.values()) else "inconclusive"
        part = single_march_rule(vals, lambda v, r: all(v[(p, r)] <= BOUND for p in jp))
        f = logs[k].final
        two = all(abs(float(f[q]) - float(fin_s0[q])) <= 0.02 * abs(float(fin_s0[q])) for q in ("xr", "cf4"))
        closes = "inconclusive" if part == "inconclusive" else "holds" if part == "holds" and two else "fails"
        rat = {p: (s_st[(p, "G")]["m"] - st[(p, "G")]["m"]) / e_s0[p] if p in jp else None for p in POINTS}
        ratf = {p: (s_st[(p, "F")]["m"] - st[(p, "F")]["m"]) / e_s0[p] if p in jp else None for p in POINTS}
        print(f"  ({k}) reported: delta/E_S0 by the face reading {f4(ratf['corner'])} {f4(ratf['-0.021'])}")
        lines.append(f"step 6 (A) {k}: moves the lip {mv}, closes level 2's excess {closes}, "
                     f"delta/E_S0 {f4(rat['corner'])} {f4(rat['-0.021'])}")
        if mv == "holds" or closes == "holds":
            findings.append(f"step 6 ({k}): a sensitivity of the lip (moves {mv}, closes {closes}), delta/E_S0 "
                            f"{f4(rat['corner'])} {f4(rat['-0.021'])}")
    lines.append("step 6 (A): judged")
    return lines, findings


def mode_ablations(args):
    cfl, _ = cfl3d()
    _, d_s0, *_ = step4(args, cfl, quiet=True)
    if d_s0 is None:
        raise Stop("S0 is not judged, so step 6 cannot be read (step 4 says what follows)")
    if judged_points(d_s0[2]):           # otherwise step 6 is void and needs no march
        need(args.a, "--a")
    lines, _ = step6(args, cfl, d_s0)
    print()
    for ln in lines:
        print(ln)
    return 0


# --- consistency ----------------------------------------------------------------

def mode_consistency(args):
    log_p = need(args.log, "--log")
    cp_p = need(Path(log_p).with_suffix(".cp"), "the log's .cp")
    cfl, _ = cfl3d()
    L = parse_log(log_p)
    print(f"{log_p}: {L.structure()}; LIPX lines {sum(len(a.lipx) for a in L.attempts)}")
    for e in L.errors:
        print(f"  error: {e}")
    if L.fresh or L.errors:
        print("  not computed: the log is an error or was started over (structure only)")
        return 1
    if L.lipx is None or not L.lip:
        raise Stop("no LIPX or no kept LIP line")
    cp = read_cp(cp_p)                   # every line parses, or a stop by line number
    up = [ln.split() for ln in open(cp_p) if ln.strip() and not ln.lstrip().startswith(("#", "v", "V"))]
    up = [t for t in up if float(t[0]) < 0.0]
    if len(up) < 4 or np.count_nonzero(cp[:, 0] > 0.0) < 2:
        raise Stop(f"{cp_p.name}: fewer than four upstream or two lower faces")
    xs = tuple(t[0] for t in up[-4:])
    print(f"  LIPX against the .cp x strings of the lip faces: {'equal' if xs == L.lipx else 'DIFFERENT (an error)'}")
    last = L.lip[-1]
    c4cp = np.array([float(t[1]) for t in up[-4:]])
    print(f"  the last LIP line (step {last.step}) against the .cp at the lip faces: largest difference "
          f"{np.abs(np.array(last.c4) - c4cp).max():.3e}")
    whole = final_lip(cp, cfl)
    four = sample_values(L.lipx, last, cfl)
    print(f"  d_G from the four faces against the whole .cp floor: largest difference "
          f"{max(abs(four[(p, 'G')] - whole[(p, 'G')]) for p in POINTS):.3e}")
    x4 = np.array([float(v) for v in L.lipx])
    worst = 0.0
    for p in POINTS:
        xl, rl = cfl.lip[p]
        b = float(floor_values([xl], True, x4, np.array([cfl.c_up(x) for x in x4]))[0]) - rl
        worst = max(worst, abs(four[(p, "F")] - (four[(p, "G")] - b)))
    print(f"  |d_F - (d_G - b)|, b at the four faces: largest {worst:.3e}")
    return 0 if xs == L.lipx else 1


# --- step 8 -------------------------------------------------------------------

def mode_outcome(args):
    cfl, c40 = cfl3d()
    gr = grids(args)
    findings = step1(args, cfl, c40, quiet=True)
    lines = []
    try:
        l2, r2 = step2(args, cfl, c40, gr, Path(args.s0) if args.s0 else None)
    except NotJudged as e:
        l2, r2 = f"step 2 (F): reported, not judged ({e})", "reported"
    lines.append(l2)
    if r2 in ("D/E", "reported"):
        findings.append(l2)
    l3, t3, fin = step3(args, cfl, c40, gr)
    lines.append(l3)
    d_s0 = None
    if args.s0:
        l4, d_s0, _, _, f4s = step4(args, cfl, quiet=True)
        lines += l4
        findings += f4s
    else:
        lines.append("step 4 (S0): not run (no --s0)")
    attrib = None
    if r2 == "reference":
        attrib = "belongs to the reference"
        lines += ["step 5 (G): not run (step 2 attributed)", "step 6 (A): not run (step 2 attributed)"]
    elif d_s0 is None:
        lines += ["step 5 (G): not run (no judged S0)", "step 6 (A): not run (no judged S0)"]
    else:
        l5, r5, f5 = step5(args, cfl, c40, gr, t3, fin, d_s0)
        lines.append(l5)
        findings += f5
        if r5 == "grid":
            attrib = "belongs to the grid"
        l6, f6 = step6(args, cfl, d_s0)
        lines += l6
        findings += f6
    print()
    for ln in lines:
        print(ln)
    if any("not judged yet" in ln for ln in lines):
        # A march still running or awaiting its resume: no mapping yet.
        print("step 8: not judged yet (a march has no FINAL line)")
        print("step 8: gate 4 stays FAIL as ADR-045 recorded it, never re-scored")
        print("step 8: closing v2b and starting v2c is left to a later ADR")
        return 0
    if attrib:
        print(f"step 8: {attrib}")
        print("step 8: ADR-045's decision that v2b's second benchmark does not pass is withdrawn")
    else:
        print("step 8: ADR-045's decision stands")
    for f in findings:
        print(f"  for the known limits: {f}")
    if not attrib:
        print("step 8: cause not found")
    print("step 8: gate 4 stays FAIL as ADR-045 recorded it, never re-scored")
    print("step 8: closing v2b and starting v2c is left to a later ADR")
    return 0


# --- main -----------------------------------------------------------------------

MODES = {"reading": mode_reading, "headers": mode_headers, "reference": mode_reference,
         "finest": mode_finest, "s0": mode_s0, "commands": mode_commands, "grid": mode_grid,
         "ablations": mode_ablations, "consistency": mode_consistency, "outcome": mode_outcome}


def main(argv):
    ap = argparse.ArgumentParser(description="ADR-046's reading script")
    ap.add_argument("mode", choices=sorted(MODES))
    for o in ("s0", "g", "a", "log", "fun3d", "cfl3d-finest", "fun3d-finest", "cfl3d-l1", "grid0", "l0", "l1"):
        ap.add_argument(f"--{o}")
    ap.add_argument("--labels", default=str(LABELS))
    args = ap.parse_args(argv)
    try:
        return MODES[args.mode](args)
    except (Stop, NotJudged) as e:
        print(f"stopped: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
