#!/usr/bin/env python3
"""ADR-047: the backward-facing step under MPI, gates 0 to 5.

    python3 tests/benchmark/backstep_mpi_gate.py <build> <reference build> <out>
            [S0's directory] [--gates 0,1,2,3,4,5] [--ranks 1,2,3,4] [--cores 2]
            [--jobs 2] [--baseline <out> --fault F1..F6]

<build> is ADR-047's build: its tests/backstep, tests/test_backends_mpi and
tests/backstep_owners. <reference build> is the lip-print build that S0 runs;
its clone's cases/backstep holds the L2.hex and L1.hex that every gate reads
(ADR-047's first revision). Nothing is written in either clone. S0's directory
(its build/backstep) is gate 6's: S0's L2.state, L2_cal.log, L1.log and L1.cp,
read only. Gate 6 runs S0's L1 command on four ranks, resumes it on four
ranks if interrupted, and then the control, S0's L1 command on two threads
with the lip-print binary, reported only.

<out> holds one new directory per run, kept, and record.txt. A run whose
directory has its <name>.json beside it is reused. A directory without one is
an interrupted run: the script stops, and nothing is removed. A run that needs
more cores (ranks x threads) than --cores, or than six less ADR-046's marches
(two threads each), waits; its gate is reported as not yet run, never as
passed.

Each run sets OMP_NUM_THREADS, OMP_PROC_BIND=false and the variables its gate
names, and no others: every VIBEFLOW_ and OMP_ variable of the caller is
dropped. The P = 1 runs are the serial harness without mpirun, on one thread,
unless a gate says otherwise. MPI runs are

    mpirun --oversubscribe --allow-run-as-root --bind-to none --report-bindings
           -n N -x <each variable> ...

one at a time, and in gates 1 to 5 each is killed at ten times its P = 1 run's
time, which counts as a failure (a hang). If a P > 1 run starts with more of
ADR-046's marches running than its P = 1 run had, the P = 1 run is repeated
under that load first (the first revision).

Logs are read as bytes: mpirun appends its own text, and a NUL byte, after a
non-zero exit (the first revision). Only lines with a known first token count.

--baseline and --fault run a deliberate fault: the P = 1 runs, gate 2's bounds
and gate 3's state are taken from the baseline <out>, read only, and only the
P > 1 runs are made, with <build> the fault's scratch build. For F5 and F6 the
exception is decided, before any run, from backstep_owners and P = 1's EQ lines.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATIONS = (-4.0, 1.0, 4.0, 6.0, 10.0)   # backstep.cpp's
X_CF4 = -4.0
C_UP, C_DOWN = 3.0, 6.0                  # the ramp's thresholds
EXTS = ("cf", "cp", "cpw", "prof")
FIELDS = ("u", "v", "w", "p", "k", "omega")   # .state columns 2 to 7
EQ_NAMES = ("step", "dt", "C", "Cgid", "xr", "cf4", "uc", "c0", "c1", "dxr", "dcf", "change",
            "rel_u", "rel_k", "rel_w", "gid_u", "gid_k", "gid_w", "differ", "smin", "smax")
EQ_JUDGED = ("C", "xr", "cf4", "uc", "change", "rel_u", "rel_k", "rel_w")
LIP_NAMES = ("p40", "lip1", "lip2", "lip3", "lip4")
FINAL_SAME = ("steps", "full", "dt", "time", "settled")
B_MIN, FLOOR_MAX = 1e-10, 1e-7           # gate 2
TOL3 = 1e-5                              # gate 3
LIMIT_FLOOR = 120.0                      # seconds: mpirun's own start-up (the first revision)
TAU6 = 0.00048                           # gate 6: ADR-046's cap on the wall's Cp
SECONDS6 = 54000.0                       # gate 6: printed seconds, three times ADR-045's L1
QUIET6 = 3600.0                          # gate 6: wall seconds without a 50-step line
REL6, UC6 = 0.002, 0.0002                # gate 6 (ii)
TOL4 = 1e-10                             # gate 4 (iii)
XY4 = 1e-12                              # gate 4 (i)
MARGIN4 = 1e-12                          # gate 4 (v) and the station margins
SECONDS_AT = 64                          # VFBSCK01's seconds (gate 0(c))
CK01_BOOK = (24, 64)                     # step .. ceiling in VFBSCK01
CK02_BOOK = (40, 80)                     # step .. ceiling in VFBSCK02
L2_ARGS = ("0.16", "5000", "1")
GATE2_ENV = {"VIBEFLOW_PRESSURE": "cg+jacobi", "VIBEFLOW_MOMENTUM": "bicgstab+jacobi",
             "VIBEFLOW_BS_SWEEPS": "2", "VIBEFLOW_CORRECTORS": "2", "VIBEFLOW_BS_STOP": "150",
             "VIBEFLOW_BS_SAVE_AT": "50,100", "VIBEFLOW_BS_SAVE": "1"}
FAULT_NAMED = {"F5": ("EQ:C",), "F6": ("EQ:change", "EQ:rel_u", "EQ:rel_k", "EQ:rel_w")}
FAULT_GATE = {"F1": (2, (4,)), "F2": (3, (4,)), "F3": (1, (2,)), "F4": (4, (4,)),
              "F5": (2, (2, 3, 4)), "F6": (2, (2, 3, 4))}


class Stop(Exception):
    """Something the script will not decide by itself."""


class G:
    """The invocation."""
    build = ref = out = s0 = baseline = None
    cores = 2
    jobs = 1
    ranks = (1, 2, 3, 4)
    fault = None
    untestable = set()
    per_p = {}                       # (gate, P) -> True if that P's runs passed
    named = {}                       # P -> True if F5's or F6's named quantity failed
    lock = threading.Lock()
    mpi = threading.Lock()           # one of this ADR's MPI runs at a time
    kill_reason = None
    commit = None
    build_sha = {}                   # the build's binaries: what a gate's result belongs to


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(msg):
    with G.lock:
        line = f"{now()} {msg}"
        print(line, flush=True)
        with open(G.out / "record.txt", "a") as fh:
            fh.write(line + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mesh(level):
    return G.ref.parent / "cases" / "backstep" / f"L{level}.hex"


def marches_running():
    """ADR-046's marches now running: processes of the lip-print binary whose
    working directory is not under <out> (gate 0 runs that binary too)."""
    exe, own, n = str(binary("ref")), str(G.out), 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv0 = (d / "cmdline").read_bytes().split(b"\0")[0].decode()
            if argv0 != exe:
                continue
            cwd = os.readlink(d / "cwd")
        except OSError:
            continue
        if not (cwd == own or cwd.startswith(own + "/")):
            n += 1
    return n


def loadavg():
    try:
        return Path("/proc/loadavg").read_text().split()[0]
    except OSError:
        return "?"


_active = {"cores": 0}


def budget():
    return min(G.cores, 6 - 2 * marches_running())


def clean_env(threads, extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VIBEFLOW_", "OMP_"))}
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "false"
    env.update(extra)
    return env


# ---------------------------------------------------------------- runs

class Run:
    """One run in a new directory <out>/g<gate>/<name>.

    procs 0 is the serial harness without mpirun; procs >= 1 is mpirun -n
    procs. kind 'backstep' runs <binary> <mesh> <prefix> 0.16 5000 1 with
    cwd the run's directory; kind 'raw' runs argv as given."""

    def __init__(self, gate, name, procs=0, threads=1, binary="new", env=None, level=2,
                 kind="backstep", argv=None, limit_from=None, copy_in=(), log_beside=False,
                 watch=None, base=None, args=L2_ARGS, resumable=False):
        self.args, self.resumable = tuple(args), resumable
        self.gate, self.name, self.procs, self.threads = gate, name, procs, threads
        self.binary, self.env, self.level, self.kind = binary, dict(env or {}), level, kind
        self.argv, self.limit_from, self.copy_in = argv, limit_from, list(copy_in)
        self.log_beside, self.watch = log_beside, watch
        root = base if base is not None else G.out
        self.dir = root / f"g{gate}" / name
        self.meta = root / f"g{gate}" / f"{name}.json"

    @property
    def prefix(self):
        return f"L{self.level}"

    @property
    def log(self):
        if self.log_beside or self.kind == "raw":
            return self.dir.parent / f"{self.name}.log"
        return self.dir / f"{self.prefix}.log"

    def path(self, ext):
        return self.dir / f"{self.prefix}.{ext}"

    def cores(self):
        return max(1, self.procs) * self.threads

    def done(self):
        return self.meta.exists()

    def info(self):
        return json.loads(self.meta.read_text()) if self.meta.exists() else None


def binary(which):
    return (G.ref if which == "ref" else G.build) / "tests" / "backstep"


def command(run):
    if run.kind == "raw":
        argv = [str(a) for a in run.argv]
    else:
        argv = [str(binary(run.binary)), str(mesh(run.level)), run.prefix, *run.args]
    if run.procs == 0:
        return argv
    names = ["OMP_NUM_THREADS", "OMP_PROC_BIND", *sorted(run.env)]
    cmd = ["mpirun", "--oversubscribe", "--allow-run-as-root", "--bind-to", "none",
           "--report-bindings", "-n", str(run.procs)]
    for n in names:
        cmd += ["-x", n]
    return cmd + argv


def spawn(cmd, env, cwd, log, limit, watch):
    """Runs cmd. A watch that returns a reason kills the run, as the limit does."""
    t0 = time.monotonic()
    killed = False
    G.kill_reason = None
    with open(log, "ab") as fh:
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc, killed = _wait(p, t0, limit, watch)
        except BaseException:                # the script interrupted: its run goes with it
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                os.killpg(p.pid, signal.SIGKILL)
            raise
    if watch:
        why = watch(final=True)
        if why and not killed:
            G.kill_reason, killed = why, True    # found on the last read: the limit was passed
    return rc, time.monotonic() - t0, killed


def _wait(p, t0, limit, watch):
    """Waits for p, polling its watch; kills it at its limit or when the watch says."""
    killed = False
    while True:
        try:
            rc = p.wait(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            why = watch() if watch else None
            if why:
                G.kill_reason = why
            if why or (limit is not None and time.monotonic() - t0 > limit):
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                rc, killed = None, True
                break
    return rc, killed


def limit_for(run):
    """Ten times the P = 1 run's time, under a load no lighter than now."""
    if run.limit_from is None:
        return None
    ref = run.limit_from
    info = ref.info()
    if info is None:
        raise Stop(f"{run.name}: its P = 1 run {ref.name} has not run")
    busy = marches_running()
    if busy > info["marches"]:
        again = Run(ref.gate, f"{ref.name}.load{busy}", ref.procs, ref.threads, ref.binary, ref.env,
                    ref.level, ref.kind, ref.argv, None, ref.copy_in, ref.log_beside, None,
                    base=G.out)
        if again.kind == "raw":
            again.argv = [str(a).replace(str(ref.dir), str(again.dir)) for a in ref.argv]
        say(f"  {run.name}: {busy} of ADR-046's marches running, {info['marches']} when "
            f"{ref.name} ran: {ref.name} again under this load first")
        got = execute(again, wait_for_cores=True)
        return max(LIMIT_FLOOR, 10.0 * got["seconds"])
    return max(LIMIT_FLOOR, 10.0 * info["seconds"])


def running_in(d):
    """Processes whose working directory is d or under it."""
    d, n = str(d), 0
    for q in Path("/proc").iterdir():
        if q.name.isdigit():
            try:
                cwd = os.readlink(q / "cwd")
            except OSError:
                continue
            n += cwd == d or cwd.startswith(d + "/")
    return n


def resume_refused(run):
    """Why an interrupted resumable run may not resume, or None. It resumes only
    from a complete per-rank set, before its FINAL, with nothing running there."""
    if running_in(run.dir):
        return "a process is still running in its directory"
    if run.log.exists() and finals(run.log) is not None:
        return "its log already holds a FINAL line"
    P = max(1, run.procs)
    have = sorted(f.name for f in run.dir.glob(f"{run.prefix}.ckpt.*of{P}") if not f.name.endswith(".tmp"))
    want = [f"{run.prefix}.ckpt.{r}of{P}" for r in range(P)]
    if have != want:
        return f"no complete checkpoint set ({len(have)} of {P} files)"
    return None


def execute(run, wait_for_cores=False):
    """Runs it unless it has run. None if the cores are not free."""
    if run.done():
        return run.info()
    resume = False
    if run.dir.exists() or run.log.exists():
        if not (run.resumable and run.dir.exists()):
            raise Stop(f"{run.dir} or {run.log.name} exists without {run.meta.name}: an interrupted run. "
                       f"Look at it, and remove both to run it again")
        why = resume_refused(run)
        if why:
            raise Stop(f"{run.dir}: interrupted, and not resumed: {why}")
        resume = True
    with G.lock:
        free = budget() - _active["cores"]
        if run.cores() > free:
            if not wait_for_cores:
                return None
        _active["cores"] += run.cores()
    try:
        limit = limit_for(run)
        if not resume:
            run.dir.mkdir(parents=True)
            for src, dst in run.copy_in:
                shutil.copy2(src, run.dir / dst)
        before = {p.name: sha256(p) for p in sorted(run.dir.iterdir())}
        if resume:                           # ADR-047 gate 6: on the same ranks, appending to the log
            run.env["VIBEFLOW_BS_RESUME"] = "1"
            say(f"  g{run.gate}/{run.name}: interrupted before; resumed in its own directory")
        env = clean_env(run.threads, run.env)
        cmd = command(run)
        busy, load, started = marches_running(), loadavg(), now()
        say(f"  run g{run.gate}/{run.name}: {' '.join(cmd)}  [{', '.join(f'{k}={v}' for k, v in sorted(run.env.items()))}]"
            f"  threads {run.threads}, ADR-046 marches running {busy}, load {load}"
            + (f", limit {limit:.0f} s" if limit else ""))
        if run.procs > 0:
            with G.mpi:
                rc, secs, killed = spawn(cmd, env, run.dir, run.log, limit, run.watch)
        else:
            rc, secs, killed = spawn(cmd, env, run.dir, run.log, limit, run.watch)
        after = {p.name: sha256(p) for p in sorted(run.dir.iterdir())}
        info = {"name": run.name, "gate": run.gate, "cmd": cmd, "env": run.env,
                "threads": run.threads, "procs": run.procs, "exit": rc, "seconds": secs,
                "killed": killed, "kill_reason": G.kill_reason if killed else None,
                "limit": limit, "marches": busy, "load": load, "resumed": resume,
                "load_end": loadavg(), "files_before": before, "files_after": after,
                "started": started, "ended": now(), "build": G.build_sha}
        run.meta.write_text(json.dumps(info, indent=1))
        say(f"  done g{run.gate}/{run.name}: exit {rc}, {secs:.0f} s"
            + (f", KILLED ({G.kill_reason or 'at its limit'})" if killed else ""))
        return info
    finally:
        with G.lock:
            _active["cores"] -= run.cores()


def execute_all(runs):
    """Runs up to --jobs at a time, within the cores; MPI runs one at a time."""
    if G.jobs <= 1:
        return [execute(r) for r in runs]
    with ThreadPoolExecutor(max_workers=G.jobs) as ex:
        return list(ex.map(execute, runs))


# ---------------------------------------------------------------- reading

def log_lines(path):
    return Path(path).read_bytes().decode("latin-1").splitlines()


def first(line):
    tk = line.split()
    return tk[0] if tk else ""


def finals(path):
    ls = [ln for ln in log_lines(path) if first(ln) == "FINAL"]
    if not ls:
        return None
    tk = ls[-1].split()
    d, i = {}, 1
    while i + 1 < len(tk):
        if tk[i] == "bubble":
            d["bubble"] = (tk[i + 1], tk[i + 2] if i + 2 < len(tk) else "")
            i += 3
        else:
            d[tk[i]] = tk[i + 1]
            i += 2
    d["line"] = ls[-1]
    d["noseconds"] = re.sub(r" seconds \S+\s*$", "", ls[-1])
    return d


def eq_lines(path):
    out = {}
    for ln in log_lines(path):
        tk = ln.split()
        if tk[:1] == ["EQ"] and len(tk) == len(EQ_NAMES) + 1:
            out[int(tk[1])] = dict(zip(EQ_NAMES, tk[1:]))
    return out


def lip_lines(path):
    """The lip print: LIPX lines in order, and LIP lines by step."""
    lipx, lip = [], {}
    for ln in log_lines(path):
        tk = ln.split()
        if tk[:1] == ["LIPX"]:
            lipx.append(ln.strip())
        elif tk[:1] == ["LIP"] and len(tk) == 7:
            lip[int(tk[1])] = dict(zip(LIP_NAMES, tk[2:]))
    return lipx, lip


def lip_text(path):
    return [ln.strip() for ln in log_lines(path) if first(ln) in ("LIPX", "LIP")]


RAMP = re.compile(r"^\s*step\s+(\d+)\s+ramp: (?:C (\S+) at dt (\S+), at \([^)]*\): (\w+)"
                  r"|dt (\S+) was halved from before: (\w+))")


def ramp(path):
    out = []
    for ln in log_lines(path):
        m = RAMP.match(ln)
        if m:
            if m.group(4):
                out.append((int(m.group(1)), m.group(4), float(m.group(2)), m.group(3)))
            else:
                out.append((int(m.group(1)), m.group(6), None, m.group(5)))
    return out


def solves(path):
    ls = [ln.strip() for ln in log_lines(path) if first(ln) == "SOLVES"]
    return ls[-1] if ls else "(no SOLVES line)"


def rows(path):
    out = []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            out.append(s.split())
    return out


def state_rows(path):
    r = rows(path)
    return r[1:] if r and len(r[0]) == 1 else r


def floats(rws):
    return np.array([[float(t) for t in r] for r in rws], dtype=float)


def ratio(d, s):
    if d == 0.0:
        return 0.0
    if not np.isfinite(d):
        return math.inf
    return d / s if s > 0.0 else math.inf


def state_metric(pa, pb):
    """Each field's largest |delta| over its scale (gate 2's scales), and x, y's
    largest |delta|. None when the files' rows differ in number."""
    if not (Path(pa).exists() and Path(pb).exists()):
        return None
    A, B = floats(state_rows(pa)), floats(state_rows(pb))
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 8:
        return None
    su = float(np.max(np.linalg.norm(A[:, 2:5], axis=1)))
    out = {}
    for i, n in enumerate(FIELDS, start=2):
        s = su if n in ("u", "v", "w") else float(np.max(np.abs(A[:, i])))
        out[n] = ratio(float(np.max(np.abs(A[:, i] - B[:, i]))), s)
    out["xy"] = float(np.max(np.abs(A[:, :2] - B[:, :2])))
    return out


def file_metric(pa, pb, skip=()):
    """A %.10e file: the x column byte-identical with the same rows, and each
    value column's max(|delta| - 1e-10 |v1|, 0) over its largest |v1|. Rows
    whose first column is in skip are left out of both."""
    ra = [r for r in rows(pa) if r[0] not in skip]
    rb = [r for r in rows(pb) if r[0] not in skip]
    if len(ra) != len(rb) or any(a[0] != b[0] or len(a) != len(b) for a, b in zip(ra, rb)):
        return None
    if not ra:
        return {}
    A, B = floats([r[1:] for r in ra]), floats([r[1:] for r in rb])
    out = {}
    for j in range(A.shape[1]):
        v1 = A[:, j]
        excess = np.maximum(np.abs(v1 - B[:, j]) - 1e-10 * np.abs(v1), 0.0)
        out[j + 1] = ratio(float(np.max(excess)) if np.all(np.isfinite(excess)) else math.inf,
                           float(np.max(np.abs(v1))))
    return out


def scalar(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return ratio(abs(a - b), abs(a))


def same_bytes(pa, pb):
    try:
        return Path(pa).read_bytes() == Path(pb).read_bytes()
    except OSError:
        return False


def compare(base, other, states, eq_names, with_lip):
    """Every judged quantity's metric between two backstep runs. A quantity
    whose structure differs (rows, steps) gets inf and a note."""
    q, notes = {}, []
    for st in states:
        m = state_metric(base / st, other / st)
        if m is None:
            notes.append(f"{st}: missing, or its rows differ")
            for n in FIELDS:
                q[f"{st}:{n}"] = math.inf
        else:
            for n in FIELDS:
                q[f"{st}:{n}"] = m[n]
    for ext in EXTS:
        pa, pb = base / f"L2.{ext}", other / f"L2.{ext}"
        m = file_metric(pa, pb) if pa.exists() and pb.exists() else None
        cols = ("y", "u") if ext == "prof" else ("value",)
        if m is None:
            notes.append(f"L2.{ext}: rows or x column differ, or missing")
            for c in cols:
                q[f"L2.{ext}:{c}"] = math.inf
        else:
            for j, c in enumerate(cols, start=1):
                q[f"L2.{ext}:{c}"] = m.get(j, math.inf)
    ea, eb = eq_lines(base / "L2.log"), eq_lines(other / "L2.log")
    if sorted(ea) != sorted(eb):
        notes.append(f"EQ steps differ: {sorted(ea)} and {sorted(eb)}")
    for n in eq_names:
        worst = 0.0 if ea and sorted(ea) == sorted(eb) else math.inf
        for s in ea:
            if s not in eb:
                continue
            a, b = ea[s][n], eb[s][n]
            if n == "xr" and float(a) <= 0.0:
                worst = max(worst, 0.0 if b == a else math.inf)      # -1 must stay -1
            else:
                worst = max(worst, scalar(a, b))
        q[f"EQ:{n}"] = worst
    if with_lip:
        la, lb = lip_lines(base / "L2.log")[1], lip_lines(other / "L2.log")[1]
        if sorted(la) != sorted(lb):
            notes.append(f"LIP steps differ: {sorted(la)} and {sorted(lb)}")
        for n in LIP_NAMES:
            worst = 0.0 if la and sorted(la) == sorted(lb) else math.inf
            for s in la:
                if s in lb:
                    worst = max(worst, scalar(la[s][n], lb[s][n]))
            q[f"LIP:{n}"] = worst
    return q, notes


def finals_same(a, b, names=FINAL_SAME):
    fa, fb = finals(a), finals(b)
    if fa is None or fb is None:
        return False, "no FINAL line"
    diff = [n for n in names if fa.get(n) != fb.get(n)]
    return not diff, ", ".join(f"{n} {fa.get(n)} against {fb.get(n)}" for n in diff) or "identical"


def fmt(v):
    return "inf" if v == math.inf else f"{v:.3e}"


def verdict(ok):
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------- checkpoints

def ck_vectors(data, at):
    """The eight vectors after a checkpoint's header: u, p, k, omega, F as
    doubles, the step history as int32, then x_r's and Cf(-4)'s as doubles."""
    out = []
    for kind in ("d", "d", "d", "d", "d", "i", "d", "d"):
        n = int(np.frombuffer(data, dtype="<i8", count=1, offset=at)[0])
        at += 8
        size = 4 if kind == "i" else 8
        out.append(data[at:at + n * size])
        at += n * size
    if at != len(data):
        raise Stop("a checkpoint's length does not match its counts")
    return out


def ck_book(path):
    """(bookkeeping bytes without seconds, step history, x_r's, Cf(-4)'s)."""
    data = Path(path).read_bytes()
    magic = data[:8]
    if magic == b"VFBSCK01":
        lo, hi, vec = CK01_BOOK[0], CK01_BOOK[1], 72
    elif magic == b"VFBSCK02":
        lo, hi, vec = CK02_BOOK[0], CK02_BOOK[1], 88
    else:
        raise Stop(f"{path}: not a checkpoint")
    v = ck_vectors(data, vec)
    return data[lo:hi], v[5], np.frombuffer(v[6], dtype="<f8"), np.frombuffer(v[7], dtype="<f8")


def ck_step(path):
    data = Path(path).read_bytes()
    at = 24 if data[:8] == b"VFBSCK01" else 40
    return int(np.frombuffer(data, dtype="<i4", count=1, offset=at)[0])


# ---------------------------------------------------------------- gate 0

def function_text(src, head):
    text = Path(src).read_text()
    i = text.find(head)
    if i < 0:
        return None
    j = text.find("\n}\n", i)
    return text[i:j + 3]


def gate0():
    say("gate 0: the serial path is unchanged (level 2, one thread)")
    A = {"VIBEFLOW_BS_SAVE": "1", "VIBEFLOW_BS_STOP": "200", "VIBEFLOW_CONVECTION": "upwind"}
    a = [Run(0, "a_ref", binary="ref", env=A), Run(0, "a_new", env=A), Run(0, "a_mpi1", procs=1, env=A)]
    execute_all(a)
    if not all(r.done() for r in a):
        return "NOT YET", ["(a) waits for cores"]
    B = {"VIBEFLOW_BS_INIT": str(a[0].path("state")), "VIBEFLOW_BS_SAVE": "1",
         "VIBEFLOW_BS_STOP": "100", "VIBEFLOW_BS_CKPT_EVERY": "50"}
    b = [Run(0, "b_ref", binary="ref", env=B), Run(0, "b_new", env=B), Run(0, "b_mpi1", procs=1, env=B)]
    C = dict(B, VIBEFLOW_BS_STOP="75")
    kept = G.out / "g0" / "c_ref.step50.ckpt"
    c_ref = Run(0, "c_ref", binary="ref", env=C)

    def grab(final=False):
        src = c_ref.path("ckpt")
        if not kept.exists() and src.exists():
            shutil.copy2(src, kept.with_suffix(".tmp"))
            os.replace(kept.with_suffix(".tmp"), kept)
    c_ref.watch = grab
    c_new = Run(0, "c_new", env=dict(C, VIBEFLOW_BS_KEEP_CKPT="1"))
    execute_all(b + [c_ref, c_new])
    if not all(r.done() for r in b + [c_ref, c_new]):
        return "NOT YET", ["(b) or (c) waits for cores"]
    if not kept.exists():
        return "FAIL", ["(c) the reference's step-50 checkpoint was never seen"]
    D = dict(B, VIBEFLOW_BS_RESUME="1")
    d = [Run(0, "d_ref", binary="ref", env=D, copy_in=[(kept, "L2.ckpt")]),
         Run(0, "d_new", env=D, copy_in=[(kept, "L2.ckpt")])]
    execute_all(d)
    if not all(r.done() for r in d):
        return "NOT YET", ["(d) waits for cores"]

    lines, ok = [], True
    for tag, trio in (("a", a), ("b", b)):
        base = trio[0]
        for r in trio[1:]:
            fa, fb = finals(base.log), finals(r.log)
            same_final = fa is not None and fb is not None and fa["noseconds"] == fb["noseconds"]
            same_files = {e: same_bytes(base.path(e), r.path(e)) for e in (*EXTS, "state")}
            same_lip = lip_text(base.log) == lip_text(r.log) and bool(lip_text(base.log))
            good = same_final and all(same_files.values()) and same_lip
            ok &= good
            lines.append(f"({tag}) {r.name} against {base.name}: FINAL without seconds "
                         f"{'identical' if same_final else 'DIFFERS'}; "
                         + ", ".join(f".{e} {'identical' if v else 'DIFFERS'}" for e, v in same_files.items())
                         + f"; lip print {'identical' if same_lip else 'DIFFERS'}  {verdict(good)}")
    ka, kb = kept.read_bytes(), c_new.path("ckpt").read_bytes() if c_new.path("ckpt").exists() else b""
    good = (len(ka) == len(kb) and ka[:SECONDS_AT] == kb[:SECONDS_AT]
            and ka[SECONDS_AT + 8:] == kb[SECONDS_AT + 8:] and ck_step(kept) == 50
            and bool(kb) and ck_step(c_new.path("ckpt")) == 50)
    ok &= good
    lines.append(f"(c) the two step-50 checkpoints: {len(ka)} and {len(kb)} bytes, identical but the "
                 f"8 bytes at {SECONDS_AT}: {verdict(good)}")
    same = {e: same_bytes(d[0].path(e), d[1].path(e)) for e in (*EXTS, "state")}
    fa, fb = finals(d[0].log), finals(d[1].log)
    same_final = fa is not None and fb is not None and fa["noseconds"] == fb["noseconds"]
    same_lip = lip_text(d[0].log) == lip_text(d[1].log)
    good = all(same.values()) and same_final and same_lip
    ok &= good
    lines.append("(d) the two resumes: " + ", ".join(f".{e} {'identical' if v else 'DIFFERS'}"
                                                      for e, v in same.items())
                 + f"; FINAL without seconds {'identical' if same_final else 'DIFFERS'}; lip print "
                 f"{'identical' if same_lip else 'DIFFERS'}  {verdict(good)}")
    src_new = G.build.parent / "tests" / "benchmark" / "backstep.cpp"
    src_ref = G.ref.parent / "tests" / "benchmark" / "backstep.cpp"
    for head in ("void writeCheckpoint(", "bool readCheckpoint("):
        tn, tr = function_text(src_new, head), function_text(src_ref, head)
        good = tn is not None and tn == tr
        ok &= good
        lines.append(f"the serial checkpoint's {head.split()[1][:-1]} unedited: {verdict(good)}")
    return verdict(ok), lines


# ---------------------------------------------------------------- gate 1

def gate1():
    say("gate 1: PETSc on more than one rank (test_backends_mpi)")
    exe = G.build / "tests" / "test_backends_mpi"
    fix = G.build.parent / "tests" / "fixtures"
    p1 = Run(1, "p1", kind="raw", base=G.baseline or G.out)
    p1.argv = [exe, fix, p1.dir]
    if G.baseline is None and 1 in G.ranks:
        execute(p1)
    if not p1.done():
        return "NOT YET", ["P = 1 has not run"]
    lines, ok, pending = [], True, False
    i1 = p1.info()
    ok &= i1["exit"] == 0
    lines.append(f"P = 1: exit {i1['exit']}  {verdict(i1['exit'] == 0)}")
    for ln in log_lines(p1.log):
        if "PASS" in ln or "FAIL" in ln or "iterations" in ln or "options" in ln or "describe" in ln:
            lines.append("    " + ln.strip())
    for P in (2, 3, 4):
        if P not in G.ranks:
            pending = True
            continue
        r = Run(1, f"p{P}", procs=P, kind="raw", argv=[exe, fix, p1.dir], limit_from=p1)
        info = execute(r)
        if info is None:
            pending = True
            lines.append(f"P = {P}: waits for cores")
            continue
        good = info["exit"] == 0 and not info["killed"]
        G.per_p[(1, P)] = good
        ok &= good
        lines.append(f"P = {P}: exit {info['exit']}{', killed at its limit' if info['killed'] else ''}"
                     f"  {verdict(good)}")
        for ln in log_lines(r.log):
            if "PASS" in ln or "FAIL" in ln or "max |du|" in ln or "ERROR: Could not" in ln:
                lines.append("    " + ln.strip())
    if pending:
        return ("FAIL" if not ok else "NOT YET"), lines
    return verdict(ok), lines


# ---------------------------------------------------------------- gate 2

def gate2_runs(base):
    p1 = Run(2, "p1", env=GATE2_ENV, base=base)
    t2 = Run(2, "p1_t2", threads=2, env=GATE2_ENV, base=base)
    return p1, t2


def gate2_bounds():
    """Gate 2's floors and bounds, from its calibration. None until it has run."""
    path = (G.baseline or G.out) / "g2" / "bounds.json"
    return json.loads(path.read_text()) if path.exists() else None


def gate2():
    say("gate 2: round-off, calibrated before any MPI run")
    p1, t2 = gate2_runs(G.baseline or G.out)
    if G.baseline is None and 1 in G.ranks:
        execute(p1)
        if p1.done():
            execute(t2)
    if not (p1.done() and t2.done()):
        return "NOT YET", ["the calibration (P = 1 on one and on two threads) has not run"]
    bounds = gate2_bounds()
    if bounds is None and G.baseline is not None:
        return "NOT YET", ["the baseline has no bounds.json"]
    if bounds is None:
        q, notes = compare(p1.dir, t2.dir, ("L2.state.50", "L2.state.100", "L2.state"), EQ_JUDGED, True)
        rs, rt = ramp(p1.log), ramp(t2.log)
        bounds = {"floors": q, "bounds": {k: max(B_MIN, 10.0 * v) for k, v in q.items()},
                  "notes": notes, "ramp_same": [(s, o) for s, o, _, _ in rs] == [(s, o) for s, o, _, _ in rt],
                  "final_same": finals_same(p1.log, t2.log)[1], "recorded": now()}
        (G.out / "g2" / "bounds.json").write_text(json.dumps(bounds, indent=1))
        say("  gate 2's floors and bounds, recorded before any P > 1 run:")
        for k in q:
            say(f"    {k:22s} floor {fmt(q[k])}  bound {fmt(bounds['bounds'][k])}")
        for n in notes:
            say(f"    note: {n}")
    lines = []
    over = {k: v for k, v in bounds["floors"].items() if not v <= FLOOR_MAX}
    if over:
        return "STOPPED", [f"floors above {FLOOR_MAX:g}: " + ", ".join(f"{k} {fmt(v)}" for k, v in over.items()),
                           "a revision may shorten the march (never below 101 steps), before any P > 1 run"]
    margins = [f"step {s} {o}: C {c:.2f}, margin "
               f"{abs(c - (C_DOWN if o == 'halves' else C_UP)) / (C_DOWN if o == 'halves' else C_UP):.4f}"
               + (" (within the print's precision)" if abs(c - (C_DOWN if o == "halves" else C_UP)) < 0.005 else "")
               for s, o, c, _ in ramp(p1.log) if c is not None]
    lines.append("P = 1's ramp decisions: " + ("; ".join(margins) or "none with a printed C"))
    lines.append(f"P = 1: {solves(p1.log)}")
    if G.fault in ("F5", "F6"):
        lines += fault_exception(p1)
    ok, pending = True, False
    for P in (2, 3, 4):
        if P not in G.ranks:
            pending = True
            continue
        r = Run(2, f"p{P}", procs=P, env=GATE2_ENV, limit_from=p1)
        info = execute(r)
        if info is None:
            pending = True
            lines.append(f"P = {P}: waits for cores")
            continue
        if info["killed"]:
            ok = False
            G.per_p[(2, P)] = False
            G.named[P] = True                        # a hang counts as failing
            lines.append(f"P = {P}: killed at its limit (a hang)  FAIL")
            continue
        q, notes = compare(p1.dir, r.dir, ("L2.state.50", "L2.state.100", "L2.state"), EQ_JUDGED, True)
        bad = {k: v for k, v in q.items() if not v <= bounds["bounds"][k]}
        same_ramp = [(s, o) for s, o, _, _ in ramp(p1.log)] == [(s, o) for s, o, _, _ in ramp(r.log)]
        fin_ok, fin = finals_same(p1.log, r.log)
        good = not bad and same_ramp and fin_ok and not notes
        G.per_p[(2, P)] = good
        if G.fault in FAULT_NAMED:
            aborted = finals(r.log) is None
            G.named[P] = aborted or any(k in bad for k in FAULT_NAMED[G.fault])
        ok &= good
        worst = max(q, key=lambda k: q[k] / bounds["bounds"][k]) if q else None
        lines.append(f"P = {P}: {len(q)} quantities within their bounds: {'yes' if not bad else 'NO'}"
                     + (f"; closest {worst} {fmt(q[worst])} of {fmt(bounds['bounds'][worst])}" if worst else "")
                     + f"; ramp decisions {'identical' if same_ramp else 'DIFFER'}; FINAL {fin}"
                     + f"{uc_note(p1.path('state'))}  {verdict(good)}")
        for k, v in bad.items():
            lines.append(f"    {k}: {fmt(v)} > {fmt(bounds['bounds'][k])}")
        for n in notes:
            lines.append(f"    {n}")
        ea, eb = eq_lines(p1.log), eq_lines(r.log)
        for s in sorted(set(ea) & set(eb)):
            lines.append(f"    step {s}: bubble {ea[s]['c0']} {ea[s]['c1']} against {eb[s]['c0']} {eb[s]['c1']}, "
                         f"drifts {ea[s]['dxr']} {ea[s]['dcf']} against {eb[s]['dxr']} {eb[s]['dcf']}")
        lines.append(f"    {solves(r.log)}")
    if pending:
        return ("FAIL" if not ok else "NOT YET"), lines
    return verdict(ok), lines


def fault_exception(p1):
    """F5 and F6: at each P, does rank 0 own the cells that set C (F5) or the
    three relChange scales (F6) at every 50-step line of P = 1? Decided from
    backstep_owners before any fault run."""
    eq = eq_lines(p1.log)
    names = ("Cgid",) if G.fault == "F5" else ("gid_u", "gid_k", "gid_w")
    lines = []
    G.untestable = set()
    for P in (2, 3, 4):
        r = subprocess.run([str(G.build / "tests" / "backstep_owners"), str(mesh(2)), str(P)],
                           capture_output=True, text=True, check=True)
        owner = {}
        for ln in r.stdout.splitlines():
            g, k = ln.split()
            owner[int(g)] = int(k)
        all0 = all(owner[int(eq[s][n])] == 0 for s in eq for n in names)
        if all0:
            G.untestable.add(P)
        lines.append(f"{G.fault} at P = {P}: rank 0 owns {', '.join(names)} at every 50-step line: "
                     f"{'yes, untestable at this P' if all0 else 'no, the fault must make gate 2 fail'}")
    for ln in lines:
        say("  " + ln)
    return lines


# ---------------------------------------------------------------- gate 3

def gate3():
    say("gate 3: the production solvers, 50 steps")
    E = {"VIBEFLOW_BS_STOP": "50", "VIBEFLOW_BS_SAVE": "1"}
    base = G.baseline or G.out
    p1 = Run(3, "p1", env=E, base=base)
    t2 = Run(3, "p1_t2", threads=2, env=E, base=base)
    if G.baseline is None and 1 in G.ranks:
        execute(p1)
        if p1.done():
            execute(t2)
    if not p1.done():
        return "NOT YET", ["P = 1 has not run"]
    lines, ok, pending = [], True, False
    if t2.done():
        q, _ = compare(p1.dir, t2.dir, ("L2.state",), ("C", "cf4", "uc"), False)
        lines.append("reported, P = 1 on two threads against one: "
                     + ", ".join(f"{k} {fmt(v)}" for k, v in q.items()))
    lines.append(f"P = 1: {solves(p1.log)}")
    for P in (2, 3, 4):
        if P not in G.ranks:
            pending = True
            continue
        r = Run(3, f"p{P}", procs=P, env=E, limit_from=p1)
        info = execute(r)
        if info is None:
            pending = True
            lines.append(f"P = {P}: waits for cores")
            continue
        if info["killed"]:
            ok = False
            G.per_p[(3, P)] = False
            lines.append(f"P = {P}: killed at its limit (a hang)  FAIL")
            continue
        q, notes = compare(p1.dir, r.dir, ("L2.state",), (), False)
        ea, eb = eq_lines(p1.log), eq_lines(r.log)
        if 50 in ea and 50 in eb:
            for n in ("C", "cf4", "uc"):
                q[f"EQ:{n}"] = scalar(ea[50][n], eb[50][n])
        else:
            notes.append("no step-50 EQ line")
        bad = {k: v for k, v in q.items() if not v <= TOL3}
        fin_ok, fin = finals_same(p1.log, r.log)
        differ = [s for s, e in eb.items() if e["differ"] != "0"]
        good = not bad and not notes and fin_ok and not differ and bool(eb)
        G.per_p[(3, P)] = good
        ok &= good
        lines.append(f"P = {P}: within {TOL3:g}: {'yes' if not bad else 'NO'}; FINAL {fin}; "
                     f"EQ lines with differing sweep counts: {differ or 'none'}"
                     f"{uc_note(p1.path('state'))}  {verdict(good)}")
        for k, v in sorted(q.items(), key=lambda kv: -kv[1])[:6]:
            lines.append(f"    {k} {fmt(v)}")
        for n in notes:
            lines.append(f"    {n}")
        if 50 in ea and 50 in eb:
            lines.append(f"    reported: x_r {ea[50]['xr']} against {eb[50]['xr']}, bubble {eb[50]['c0']} "
                         f"{eb[50]['c1']}, drifts {eb[50]['dxr']} {eb[50]['dcf']}")
        lines.append(f"    {solves(r.log)}")
    if pending:
        return ("FAIL" if not ok else "NOT YET"), lines
    return verdict(ok), lines


# ---------------------------------------------------------------- gate 4

def nearest_two(coarse, fine):
    """Each fine cell's (nearest, second nearest) squared distance in (x, y)
    to the coarse cells' centres."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(coarse).query(fine, k=2)
    return d[:, 0] ** 2, d[:, 1] ** 2


def margins_report(state):
    """P = 1's smallest |x_c - x_station| per station, and the margin between
    the two nearest |y - 5| in each column around x = -4 (the harness's
    columns: cells within 1e-9 of the nearest centroid x on each side)."""
    A = floats(state_rows(state))
    x, y = A[:, 0], A[:, 1]
    out, small = [], {}
    for s in STATIONS:
        m = float(np.min(np.abs(x - s)))
        small[f"{s:.4f}"] = m
        out.append(f"station x = {s:g}: smallest |x_c - x| {m:.3e}")
    cols = {}
    for tag, xc in (("left", x[x < X_CF4].max()), ("right", x[x > X_CF4].min())):
        dy = np.sort(np.abs(y[np.abs(x - xc) < 1e-9] - 5.0))
        cols[tag] = float(dy[1] - dy[0]) if dy.size > 1 else math.inf
        out.append(f"x = -4's {tag} column (x {xc:.6f}): the two nearest |y - 5| differ by {cols[tag]:.3e}")
    return out, {k for k, v in small.items() if v < MARGIN4}, min(cols.values())


def uc_note(state):
    """Beside every u_c(-4) comparison: the centre column's tie margin, when it
    is below 1e-12 (ADR-047, gate 4)."""
    try:
        m = margins_report(state)[2]
    except (OSError, ValueError):
        return " (u_c(-4)'s centre-column margin could not be read)"
    return f" (u_c(-4)'s centre-column margin {m:.3e} is below {MARGIN4:g})" if m < MARGIN4 else ""


def gate4():
    say("gate 4: extraction and the state across rank counts")
    base = G.baseline or G.out
    init = base / "g3" / "p1" / "L2.state"
    if not init.exists():
        return "NOT YET", ["gate 3's P = 1 state does not exist"]
    E = {"VIBEFLOW_BS_STOP": "0", "VIBEFLOW_BS_SAVE": "1", "VIBEFLOW_BS_INIT": str(init)}
    p1 = Run(4, "p1", env=E, base=base)
    l1p1 = Run(4, "L1_p1", env=E, level=1, base=base)
    if G.baseline is None and 1 in G.ranks:
        execute(p1)
    if not p1.done():
        return "NOT YET", ["P = 1 has not run"]
    lines, ok, pending = [], True, False
    mr, skip, umargin = margins_report(p1.path("state"))
    lines += mr
    if skip:
        lines.append(f"stations with a margin below {MARGIN4:g}, their .prof rows reported, not judged: "
                     + ", ".join(sorted(skip)))
    lines.append(f"u_c(-4)'s centre-column margin: {umargin:.3e}")
    s1 = state_rows(p1.path("state"))
    for P in (2, 3, 4):
        if P not in G.ranks:
            pending = True
            continue
        r = Run(4, f"p{P}", procs=P, env=E, limit_from=p1)
        info = execute(r)
        if info is None:
            pending = True
            lines.append(f"P = {P}: waits for cores")
            continue
        if info["killed"] or not r.path("state").exists():
            ok = False
            G.per_p[(4, P)] = False
            lines.append(f"P = {P}: {'killed at its limit (a hang)' if info['killed'] else 'no .state'}  FAIL")
            continue
        sP = state_rows(r.path("state"))
        same_n = len(s1) == len(sP)
        cols = same_n and all(a[2:] == b[2:] for a, b in zip(s1, sP))
        xy = max((abs(float(a[i]) - float(b[i])) for a, b in zip(s1, sP) for i in (0, 1)),
                 default=0.0) if same_n else math.inf
        cp = same_bytes(p1.path("cp"), r.path("cp"))
        third = {}
        for ext in ("cf", "cpw", "prof"):
            m = file_metric(p1.path(ext), r.path(ext), skip if ext == "prof" else ()) \
                if r.path(ext).exists() else None
            third[ext] = math.inf if m is None else max(m.values(), default=0.0)
        good = cols and xy <= XY4 and cp and all(v <= TOL4 for v in third.values())
        G.per_p[(4, P)] = good
        ok &= good
        lines.append(f"P = {P}: (i) u..omega byte-identical {cols}, x and y within {fmt(xy)}; "
                     f"(ii) .cp byte-identical {cp}; (iii) "
                     + ", ".join(f".{k} {fmt(v)}" for k, v in third.items()) + f"  {verdict(good)}")
        if skip:
            m = file_metric(p1.path("prof"), r.path("prof"))
            lines.append(f"    .prof with every station's rows (reported): {m}")
        if r.path("cf").exists():
            lines.append(f"    reported: {xr_cf4(r.path('cf'))} (P = 1: {xr_cf4(p1.path('cf'))})")
    if G.fault:
        return ("FAIL" if not ok else "NOT YET"), lines
    p4 = Run(4, "p4", procs=4, env=E)
    back = Run(4, "p1_from_p4", env=dict(E, VIBEFLOW_BS_INIT=str(p4.path("state"))))
    if p4.done() and p4.path("state").exists() and 1 in G.ranks:
        execute(back)
    if back.done():
        same = {e: same_bytes(p1.path(e), back.path(e)) for e in (*EXTS, "state")}
        good = all(same.values())
        ok &= good
        lines.append("(iv) P = 1 from P = 4's state against P = 1 from serial's: " + ", ".join(
            f".{e} {'identical' if v else 'DIFFERS'}" for e, v in same.items()) + f"  {verdict(good)}")
    else:
        pending = True
        lines.append("(iv) waits for P = 4")
    if 1 in G.ranks:
        execute(l1p1)
    if l1p1.done():
        coarse = floats(state_rows(init))[:, :2]
        fine = floats(state_rows(l1p1.path("state")))[:, :2]
        d1, d2 = nearest_two(coarse, fine)
        margin = np.where(d1 > 0.0, (d2 - d1) / np.where(d1 > 0.0, d1, 1.0), np.inf)
        safe = margin > MARGIN4
        lines.append(f"(v) L1 from L2: {int(np.sum(~safe))} fine cells with a margin of at most {MARGIN4:g}"
                     + (": " + ", ".join(f"({fine[i, 0]:.6f}, {fine[i, 1]:.6e})"
                                         for i in np.flatnonzero(~safe)[:20]) if np.any(~safe) else ""))
        f1 = state_rows(l1p1.path("state"))
        for P in (2, 3, 4):
            if P not in G.ranks:
                pending = True
                continue
            r = Run(4, f"L1_p{P}", procs=P, env=E, level=1, limit_from=l1p1)
            info = execute(r)
            if info is None:
                pending = True
                lines.append(f"(v) P = {P}: waits for cores")
                continue
            fP = state_rows(r.path("state")) if r.path("state").exists() else []
            if len(fP) != len(f1) or info["killed"]:
                ok = False
                lines.append(f"(v) P = {P}: no state of the right size"
                             f"{', killed at its limit' if info['killed'] else ''}  FAIL")
                continue
            bad = [i for i in np.flatnonzero(safe) if f1[i][2:] != fP[i][2:]]
            other = sum(1 for i in np.flatnonzero(~safe) if f1[i][2:] != fP[i][2:])
            good = not bad
            ok &= good
            lines.append(f"(v) P = {P}: u..omega byte-identical at every cell with a margin above "
                         f"{MARGIN4:g}: {'yes' if good else f'NO, {len(bad)} cells'}; "
                         f"of the others, {other} differ  {verdict(good)}")
    else:
        pending = True
        lines.append("(v) waits for L1 at P = 1")
    if pending:
        return ("FAIL" if not ok else "NOT YET"), lines
    return verdict(ok), lines


def xr_cf4(path):
    A = floats(rows(path))
    x, cf = A[:, 0], A[:, 1]
    c4 = float(np.interp(X_CF4, x, cf))
    xr = -1.0
    down = np.flatnonzero(x > 0.0)
    for i in down[:-1]:
        if cf[i] < 0.0 <= cf[i + 1]:
            xr = float(x[i] - cf[i] * (x[i + 1] - x[i]) / (cf[i + 1] - cf[i]))
    return f"x_r {xr:.6f}, Cf(-4) {c4:.8e}"


# ---------------------------------------------------------------- gate 5

def ck_copies(run, P):
    """The checkpoint files a run kept: <prefix>.ckpt at P = 1, the per-rank
    set otherwise, as (source, name) pairs for copy_in."""
    if P == 1:
        return [(run.path("ckpt"), run.path("ckpt").name)]
    return [(f, f.name) for f in sorted(run.dir.glob(f"L2.ckpt.*of{P}")) if not f.name.endswith(".tmp")]


def gate5():
    say("gate 5: checkpoint and resume under MPI (gate 2's settings)")
    bounds = gate2_bounds()
    if bounds is None:
        return "NOT YET", ["gate 2's calibration has not run"]
    over = {k: v for k, v in bounds["floors"].items() if not v <= FLOOR_MAX}
    if over:
        return "STOPPED", [f"gate 2's floors above {FLOOR_MAX:g}: " + ", ".join(over)]
    b = bounds["bounds"]
    E = dict(GATE2_ENV, VIBEFLOW_BS_STOP="120", VIBEFLOW_BS_CKPT_EVERY="10", VIBEFLOW_BS_KEEP_CKPT="1")
    Eb = dict(GATE2_ENV, VIBEFLOW_BS_RESUME="1", VIBEFLOW_BS_STOP="0", VIBEFLOW_BS_KEEP_CKPT="1")
    Ec = dict(GATE2_ENV, VIBEFLOW_BS_RESUME="1", VIBEFLOW_BS_STOP="150")
    Ed = dict(GATE2_ENV, VIBEFLOW_BS_RESUME="1", VIBEFLOW_BS_STOP="150", VIBEFLOW_BS_CKPT_EVERY="10")
    a1 = Run(5, "a_p1", env=E)
    # The P = 1 runs first: (a), (b) and (d).
    if 1 in G.ranks:
        execute(a1)
    if not a1.done():
        return "NOT YET", ["(a) at P = 1 has not run"]
    try:
        step1 = ck_step(a1.path("ckpt"))
    except (OSError, ValueError, IndexError) as e:
        return "FAIL", [f"(a) P = 1's checkpoint cannot be read: {e}"]
    if step1 != 120:
        return "FAIL", [f"(a) P = 1's checkpoint is at step {step1}, not 120"]
    b1 = Run(5, "b_p1", env=Eb, copy_in=ck_copies(a1, 1))
    d1 = Run(5, "d_p1", env=Ed, copy_in=ck_copies(a1, 1))
    if 1 in G.ranks:
        execute(b1)
        execute(d1)
    a4 = Run(5, "a_p4", procs=4, env=E, limit_from=a1)
    if 4 in G.ranks:
        execute(a4)
    lines, ok, pending = [], True, False
    if b1.done():
        same = {e: same_bytes(a1.path(e), b1.path(e)) for e in (*EXTS, "state")}
        good = all(same.values())
        ok &= good
        lines.append("(b) P = 1 resumed with no step, against (a): " + ", ".join(
            f".{e} {'identical' if v else 'DIFFERS'}" for e, v in same.items()) + f"  {verdict(good)}")
    else:
        pending = True
    if not a4.done():
        return ("FAIL" if not ok else "NOT YET"), lines + ["(a) at P = 4 waits for cores"]
    if a4.info()["killed"]:
        return "FAIL", lines + ["(a) at P = 4 killed at its limit (a hang)"]
    files4 = [f for f, _ in ck_copies(a4, 4)]
    try:
        book1, hs1, xr1, cf1 = ck_book(a1.path("ckpt"))
        books = [ck_book(f) for f in files4]
        good = len(files4) == 4
    except (Stop, OSError, ValueError, IndexError) as e:
        books, good = [], False
        book1 = hs1 = b""
        xr1 = cf1 = np.zeros(0)
        lines.append(f"(a) a checkpoint cannot be read: {e}")
    for bk, hs, xr, cf in books:
        good &= bk[:16] == book1[:16]                # step, sinceChange, holdUntil, refused
        good &= bk[16:40] == book1[16:40]            # time, dtNow, ceiling
        good &= hs == hs1                            # the step history
        good &= len(xr) == len(xr1) and all(
            (x2 == x1) if x1 <= 0.0 else scalar(x1, x2) <= b["EQ:xr"] for x1, x2 in zip(xr1, xr))
        good &= len(cf) == len(cf1) and all(scalar(c1, c2) <= b["EQ:cf4"] for c1, c2 in zip(cf1, cf))
    good &= bool(books) and all(bk == books[0][0] and hs == books[0][1] and xr.tobytes() == books[0][2].tobytes()
                                and cf.tobytes() == books[0][3].tobytes() for bk, hs, xr, cf in books)
    ok &= good
    lines.append(f"(a) {len(files4)} VFBSCK02 files at P = 4 against P = 1's VFBSCK01 at step "
                 f"{step1}, and alike but seconds: {verdict(good)}")
    b4 = Run(5, "b_p4", procs=4, env=Eb, copy_in=ck_copies(a4, 4), limit_from=b1)
    if 4 in G.ranks and b1.done():
        execute(b4)
    if b4.done():
        same = {e: same_bytes(a4.path(e), b4.path(e)) for e in (*EXTS, "state")}
        good = all(same.values()) and not b4.info()["killed"]
        ok &= good
        lines.append("(b) P = 4 resumed with no step, against (a): " + ", ".join(
            f".{e} {'identical' if v else 'DIFFERS'}" for e, v in same.items()) + f"  {verdict(good)}")
    else:
        pending = True
        lines.append("(b) P = 4 waits")
    cross = (("c_p4_at_p3", 3, ck_copies(a4, 4)), ("c_p4_at_p1", 0, ck_copies(a4, 4)),
             ("c_p1_at_p4", 4, ck_copies(a1, 1)))
    for name, procs, files in cross:
        if max(1, procs) not in G.ranks:
            pending = True
            lines.append(f"(c) {name}: not asked for")
            continue
        r = Run(5, name, procs=procs, env=Ec, copy_in=files, log_beside=True,
                limit_from=b1 if procs else None)
        if procs and not b1.done():
            pending = True
            lines.append(f"(c) {name}: waits for (b) at P = 1, which sets its time limit")
            continue
        info = execute(r)
        if info is None:
            pending = True
            lines.append(f"(c) {name}: waits for cores")
            continue
        refused = any(first(ln).startswith("REFUSED") for ln in log_lines(r.log))
        same = info["files_before"] == info["files_after"]
        good = refused and info["exit"] == 3 and same and not info["killed"]
        ok &= good
        lines.append(f"(c) {name}: REFUSED {refused}, exit {info['exit']}, every file's hash unchanged and "
                     f"no new file {same}{', killed at its limit' if info['killed'] else ''}  {verdict(good)}")
    d4 = Run(5, "d_p4", procs=4, env=Ed, copy_in=ck_copies(a4, 4), limit_from=d1)
    if 4 in G.ranks and d1.done():
        execute(d4)
    if d4.done() and d4.info()["killed"]:
        ok = False
        lines.append("(d) P = 4 killed at its limit (a hang)  FAIL")
    elif d1.done() and d4.done():
        q, notes = compare(d1.dir, d4.dir, ("L2.state",), EQ_JUDGED, True)
        bad = {k: v for k, v in q.items() if not v <= b[k]}
        fin_ok, fin = finals_same(d1.log, d4.log, ("steps", "full", "dt", "time"))
        good = not bad and not notes and fin_ok and not d4.info()["killed"]
        ok &= good
        lines.append(f"(d) P = 4 against P = 1, both resumed at step 120 and run to 150, within gate 2's "
                     f"bounds: {'yes' if not bad else 'NO'}; FINAL {fin}{uc_note(a1.path('state'))}  {verdict(good)}")
        for k, v in bad.items():
            lines.append(f"    {k}: {fmt(v)} > {fmt(b[k])}")
        for n in notes:
            lines.append(f"    {n}")
    else:
        pending = True
        lines.append("(d) waits")
    if pending:
        return ("FAIL" if not ok else "NOT YET"), lines
    return verdict(ok), lines


# ---------------------------------------------------------------- gate 6

def watch6(log):
    """Gate 6's limits: printed seconds past 54,000, or an hour of wall time
    with no new 50-step line. Read every 30 s."""
    state = {"t": time.monotonic(), "seen": None, "read": 0.0}
    pat = re.compile(r"^\s*step\s+(\d+)\s+t\s.*\((\d+(?:\.\d*)?) s\)\s*$")

    def watch(final=False):
        now_ = time.monotonic()
        if now_ - state["read"] < 30.0 and not final:
            return None
        state["read"] = now_
        last = None
        try:
            for ln in log_lines(log):
                m = pat.match(ln)
                if m:
                    last = (int(m.group(1)), float(m.group(2)))
        except OSError:
            return None
        if last is not None and last != state["seen"]:
            state["seen"], state["t"] = last, now_
        if last is not None and last[1] > SECONDS6:
            return f"printed seconds {last[1]:.0f} past {SECONDS6:.0f}"
        if now_ - state["t"] > QUIET6 and not final:
            return f"no new 50-step line for {QUIET6:.0f} s"
        return None
    return watch


def lip_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import backstep_lip
    import backstep_gate
    return backstep_lip, backstep_gate


def read6(BL, BG, ref, log, cp):
    """A march's quantities for gate 6: its parsed log, its .cp, the gate
    reading at the three points, the window means of the two lip values."""
    L = BL.parse_log(log)
    c = BL.read_cp(cp)
    pts = {"corner": float(BG.floor_values([ref.lip["corner"][0]], True, c[:, 0], c[:, 1])[0]),
           "-0.021": float(BG.floor_values([ref.lip["-0.021"][0]], True, c[:, 0], c[:, 1])[0]),
           "x = 0, lower floor": float(BG.floor_values([0.0], False, c[:, 0], c[:, 1])[0])}
    samples, _, errs = BL.window(L)
    means = {}
    if not errs and L.lipx is not None:
        for p in BL.POINTS:
            means[p] = float(np.mean([BL.lip_values(L.lipx, x, ref, p)[0] for x in samples]))
    return L, c, pts, means, errs


def compare6(BL, BG, ref, s0, ours, name):
    """Gate 6's judged quantities between S0's L1 and a march: (ok, lines)."""
    L0, c0, p0, m0, e0 = s0
    L1, c1, p1, m1, e1 = ours
    lines, ok = [], True
    if not L1.usable or L1.cls not in ("steady", "quasi"):
        return False, [f"{name}: not a settled, usable march (class {L1.cls}, errors {L1.errors})"]
    w0 = (c0[:, 0] >= -4.0) & (c0[:, 0] <= 30.0)
    w1 = (c1[:, 0] >= -4.0) & (c1[:, 0] <= 30.0)
    same_x = np.array_equal(c0[w0, 0], c1[w1, 0])
    dmax = float(np.max(np.abs(c1[w1, 1] - c0[w0, 1]))) if same_x else math.inf
    good = same_x and dmax <= TAU6
    ok &= good
    k = int(np.argmax(np.abs(c1[w1, 1] - c0[w0, 1]))) if same_x else 0
    lines.append(f"{name} (i) .cp at every bottom-wall face with -4 <= x <= 30: faces paired {same_x}, "
                 f"largest |dCp| {dmax:.6f}" + (f" at x = {c0[w0, 0][k]:+.6f}" if same_x else "")
                 + f" against {TAU6}  {verdict(good)}")
    for pt in p0:
        d = p1[pt] - p0[pt]
        good = abs(d) <= TAU6
        ok &= good
        lines.append(f"{name} (i) the gate reading at {pt}: dCp {d:+.6f}  {verdict(good)}")
    if e1 or e0 or len(m0) != 2 or len(m1) != 2:
        ok = False
        lines.append(f"{name} (i) the window means: not computed (S0: {e0 or 'ok'}; {name}: {e1 or 'ok'})  FAIL")
    else:
        for pt in BL.POINTS:
            d = m1[pt] - m0[pt]
            good = abs(d) <= TAU6
            ok &= good
            lines.append(f"{name} (i) the window mean at {pt}: dm {d:+.6f}  {verdict(good)}")
    f0, f1 = L0.final, L1.final
    for key, tol, rel in (("xr", REL6, True), ("cf4", REL6, True), ("uc4", UC6, False)):
        a, b = float(f0[key]), float(f1[key])
        d = abs(b - a)
        good = d <= (tol * abs(a) if rel else tol)
        ok &= good
        lines.append(f"{name} (ii) {key} {f1[key]} against S0's {f0[key]}: "
                     + (f"{d / abs(a):.5f} of S0's" if rel else f"|d| {d:.6f}") + f"  {verdict(good)}")
    good = f1["settled"] in ("steady", "quasi") and float(f1["dt"]) == float(f0["dt"])
    ok &= good
    lines.append(f"{name} (iii) settled {f1['settled']} at dt {f1['dt']} (S0's L1: {f0['settled']} at dt "
                 f"{f0['dt']})  {verdict(good)}")
    return ok, lines


def printed_seconds(log):
    """The largest printed seconds of a log's 50-step and FINAL lines."""
    pat = re.compile(r"^\s*step\s+\d+\s+t\s.*\((\d+(?:\.\d*)?) s\)\s*$")
    worst = 0.0
    for ln in log_lines(log):
        m = pat.match(ln)
        if m:
            worst = max(worst, float(m.group(1)))
    f = finals(log)
    if f and "seconds" in f:
        try:
            worst = max(worst, float(f["seconds"]))
        except ValueError:
            pass
    return worst


def judge6(BL, BG, ref, s0, run, name):
    """(ok, lines) for a gate 6 march; a .cp or log that cannot be read fails."""
    try:
        return compare6(BL, BG, ref, s0, read6(BL, BG, ref, run.log, run.path("cp")), name)
    except (BL.Stop, BL.NotJudged, OSError, ValueError, IndexError) as e:
        return False, [f"{name}: not computed ({e})  FAIL"]


def gate6():
    say("gate 6: the production march on four ranks against S0's L1")
    if G.s0 is None:
        return "NOT YET", ["S0's directory not given"]
    saved = G.out / "results.json"
    got = json.loads(saved.read_text()) if saved.exists() else {}
    before = [g for g in range(6) if got.get(str(g), {}).get("status") != "PASS"
              or got.get(str(g), {}).get("build") != G.build_sha]
    r = Run(6, "p4", procs=4, level=1, resumable=True)
    if before and not r.dir.exists():
        return "NOT YET", [f"gates {before} have not passed with this build's binaries (ADR-047, Order 5)"]
    BL, BG = lip_module()
    d = G.s0
    try:
        S = BL.parse_log(d / "L1.log")
        if not S.usable or S.cls not in ("steady", "quasi"):
            return "STOPPED", [f"S0's L1 is not a settled, usable march (class {S.cls})"]
        a = BL.first_attempt(S)
        if a is None or a.mesh is None or Path(a.mesh).resolve() != mesh(1).resolve():
            return "STOPPED", [f"S0's L1 read {a.mesh if a else None}, not {mesh(1)}"]
        u, ustr = BL.s0_u_in(d)
        ref, _ = BL.cfl3d()
        s0 = read6(BL, BG, ref, d / "L1.log", d / "L1.cp")
    except (BL.Stop, BL.NotJudged, OSError, ValueError) as e:
        return "STOPPED", [f"S0's L1 cannot be read: {e}"]
    init = d / "L2.state"
    say(f"  S0's L1: {d / 'L1.log'}, mesh {a.mesh} sha256 {sha256(mesh(1))}; U_in {ustr} from "
        f"S0's L2_cal.log; init {init} sha256 {sha256(init)}")
    E = {"VIBEFLOW_BS_INIT": str(init), "VIBEFLOW_BS_SAVE": "1"}
    args = ("0.08", "5000", ustr)
    r = Run(6, "p4", procs=4, env=E, level=1, args=args, resumable=True)
    r.watch = watch6(r.log)
    info = execute(r)
    if info is None:
        return "NOT YET", ["P = 4 waits for four free cores"]
    lines, ok = [], True
    secs = printed_seconds(r.log)
    if info["killed"]:
        ok = False
        lines.append(f"P = 4 killed: {info.get('kill_reason') or 'at its limit'}  FAIL")
    elif secs > SECONDS6:
        ok = False
        lines.append(f"P = 4 printed {secs:.0f} s, past {SECONDS6:.0f}  FAIL")
    elif not r.path("cp").exists():
        ok = False
        lines.append("P = 4 wrote no .cp  FAIL")
    else:
        good, ls = judge6(BL, BG, ref, s0, r, "P = 4")
        ok &= good
        lines += ls
        eqs = [ln.split() for ln in log_lines(r.log) if first(ln) == "EQ"]
        differ = [int(t[1]) for t in eqs if len(t) == len(EQ_NAMES) + 1 and t[19] != "0"]
        good = bool(eqs) and not differ
        ok &= good
        lines.append(f"P = 4 (iv) EQ lines with differing sweep counts: {differ or 'none'} of {len(eqs)}  "
                     f"{verdict(good)}")
        f = finals(r.log)
        lines.append(f"reported: P = 4 {f['steps'] if f else '?'} steps in {f['seconds'] if f else '?'} s; "
                     f"S0's L1 {S.final['steps']} steps in {S.final['seconds']} s on two threads; {solves(r.log)}")
    # P = 4's verdict is recorded before the control starts; the control changes none.
    for ln in lines:
        say(f"  gate 6: {ln}")
    say(f"gate 6, P = 4 against S0's L1: {verdict(ok)} (the control follows, reported only)")
    c = Run(6, "control", threads=2, binary="ref", env=E, level=1, args=args)
    c.watch = watch6(c.log)
    try:
        cinfo = execute(c)
    except Stop as e:
        cinfo, lines = None, lines + [f"the control: {e} (reported)"]
    if cinfo is None:
        lines.append("the control (S0's L1 command, two threads, the lip-print binary): not run yet (reported)")
    elif cinfo["killed"] or not c.path("cp").exists():
        lines.append(f"the control: {cinfo.get('kill_reason') or 'no .cp'} (reported)")
    else:
        _, ls = judge6(BL, BG, ref, s0, c, "the control")
        lines += [ln.replace("  PASS", "  (reported)").replace("  FAIL", "  (reported)") for ln in ls]
    return verdict(ok), lines


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build")
    ap.add_argument("reference")
    ap.add_argument("out")
    ap.add_argument("s0", nargs="?")
    ap.add_argument("--gates", default="0,1,2,3,4,5")   # gate 6 only when asked for
    ap.add_argument("--ranks", default="1,2,3,4")
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--baseline")
    ap.add_argument("--fault", choices=sorted(FAULT_GATE))
    a = ap.parse_args()
    G.build, G.ref, G.out = Path(a.build).resolve(), Path(a.reference).resolve(), Path(a.out).resolve()
    G.s0 = Path(a.s0).resolve() if a.s0 else None
    G.cores, G.jobs = a.cores, a.jobs
    G.ranks = tuple(int(s) for s in a.ranks.split(","))
    G.fault = a.fault
    if bool(a.baseline) != bool(a.fault):
        raise Stop("--baseline and --fault go together")
    if a.baseline:
        G.baseline = Path(a.baseline).resolve()
        G.ranks = tuple(P for P in G.ranks if P > 1)
    gates = [int(s) for s in a.gates.split(",")]
    if a.fault:
        gates = [FAULT_GATE[a.fault][0]]
        G.ranks = tuple(P for P in G.ranks if P in FAULT_GATE[a.fault][1])
    G.out.mkdir(parents=True, exist_ok=True)
    for g in range(7):
        (G.out / f"g{g}").mkdir(exist_ok=True)
    say(f"backstep_mpi_gate.py: build {G.build}, reference {G.ref}, gates {gates}, ranks {G.ranks}, "
        f"cores {G.cores}, jobs {G.jobs}" + (f", fault {G.fault} against {G.baseline}" if G.fault else ""))
    for clone in (G.build.parent, G.ref.parent):
        r = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True)
        st = subprocess.run(["git", "-C", str(clone), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True)
        say(f"  {clone}: commit {r.stdout.strip()}{' (modified)' if st.stdout.strip() else ''}")
        if clone == G.build.parent:
            G.commit = r.stdout.strip() + (" (modified)" if st.stdout.strip() else "")
    for lv in (2, 1):
        say(f"  L{lv}.hex {mesh(lv)} sha256 {sha256(mesh(lv))}")
    for exe in ("backstep", "test_backends_mpi", "backstep_owners"):
        p = G.build / "tests" / exe
        if p.exists():
            G.build_sha[exe] = sha256(p)
            say(f"  {exe} sha256 {G.build_sha[exe]}")
    say(f"  reference backstep sha256 {sha256(binary('ref'))}")
    table = {0: gate0, 1: gate1, 2: gate2, 3: gate3, 4: gate4, 5: gate5, 6: gate6}
    results = {}
    saved = G.out / "results.json"
    for g in gates:
        status, lines = table[g]()
        results[g] = status
        for ln in lines:
            say(f"  gate {g}: {ln}")
        say(f"gate {g}: {status}")
        if not G.fault:
            got = json.loads(saved.read_text()) if saved.exists() else {}
            got[str(g)] = {"status": status, "commit": G.commit, "build": G.build_sha, "at": now()}
            saved.write_text(json.dumps(got, indent=1))
    say("summary: " + "  ".join(f"gate {g} {s}" for g, s in results.items()))
    if G.fault:
        gate, wanted = FAULT_GATE[G.fault]
        caught = True
        for P in wanted:
            if P in G.untestable:
                say(f"{G.fault} at P = {P}: untestable (rank 0 owns the cells at every line)")
                continue
            got = G.per_p.get((gate, P))
            if got is None:
                caught = False
                say(f"{G.fault} at P = {P}: not run")
            elif G.fault in FAULT_NAMED:
                hit = G.named.get(P, False)
                caught &= hit
                say(f"{G.fault} at P = {P}: gate {gate} {'failed' if not got else 'passed'}; on "
                    f"{' or '.join(FAULT_NAMED[G.fault])} (or a hang or an abort): "
                    f"{'yes, caught' if hit else 'NO, NOT CAUGHT'}")
            else:
                caught &= not got
                say(f"{G.fault} at P = {P}: gate {gate} {'failed: caught' if not got else 'PASSED: NOT CAUGHT'}")
        say(f"{G.fault}: {'its gate failed wherever testable' if caught else 'NOT shown to make its gate fail'}")
        return 0 if caught else 1
    if any(s in ("FAIL", "STOPPED") for s in results.values()):
        return 1
    return 0 if all(s == "PASS" for s in results.values()) else 2


def _signalled(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signalled)
    signal.signal(signal.SIGHUP, _signalled)
    try:
        sys.exit(main())
    except Stop as e:
        print(f"STOP: {e}", flush=True)
        sys.exit(3)
