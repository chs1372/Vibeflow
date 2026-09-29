#!/usr/bin/env python3
"""ADR-043: a linear model of the C++ PISO step for the odd-even mode.

The flat plate's two-corrector PISO grew a streamwise odd-even mode of the
pressure and the velocity on TMR's 545x385 grid and not on 273x193. This
model asks the C++ step, term for term, what it does to that mode at one
station of a steady state.

The column. The C++ steady state's cells at the station -- its rows and
faces, its width dx, u(y) and nu_t(y) -- made periodic in x over nx cells
(two by default: the mean and the odd-even mode, nothing else). The base is
the parallel flow U(y), V = 0, p = 0, held steady to round-off by a body
force equal to its discrete residual. Upwinding is fixed at the base's: from
the owner on the vertical faces (U > 0), from below on the horizontal ones,
as a vanishing V > 0 would give.

The step is PisoSolver::advance with one outer iteration, on a Cartesian
mesh: BDF2 (BDF1 on request); upwind convection and diffusion (nu + nu_t on
the faces) implicit; the linear-upwind correction and nu_t (grad u)^T
explicit; least-squares gradients with 1/d^2 weights and the boundary values
the C++ gives each field, the wall pressure extrapolated by three sweeps;
the predictor solved exactly; n correctors, each with H/aP, the exact
Rhie-Chow flux and its old-flux term, the pressure solved exactly, the flux
and the velocity corrected; the step's last flux in the next step's matrix
and correction; the top at p = 0 with zero-gradient velocity; the wall
no-slip. k, omega and nu_t are frozen, as the frozen C++ run says they may
be. Non-orthogonal and skewness corrections vanish on this mesh.

The growth factor is the largest |eigenvalue| of the step's Jacobian over
the odd-even subspace (the step commutes with a shift by one cell, so that
subspace is invariant). The Jacobian comes from central differences.

Switches (Model(..., opts)): each removes or replaces one term, for the
ablations ADR-043's rules ask for; see OPTS below.

Run:
  python3 piso_mode.py <cells dump> <mesh level> x0 dt ncorr [opt=value ...]
  python3 piso_mode.py --scan <cells dump> <mesh level> dt ncorr [opt=value ...]
      the growth factor at every station of STATIONS, by Arnoldi, and the
      largest
The cells dump is flat_plate's VIBEFLOW_FP_DUMP file (x y u v k omega
nu_t/nu p per cell); the level names the TMR grid it was run on (545x385,
273x193, ...; cases/flatplate/make_mesh.py).
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

NU = 2e-7
_DUMPS = {}          # cells dumps read once

# Stations along the plate for a scan: from the leading edge to the outlet.
STATIONS = (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
            0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95, 1.99)

# The switches, with their defaults (the C++ step).
OPTS = {
    "bdf1": False,          # BDF1 on every step
    "deferred": True,       # the linear-upwind deferred correction
    "transpose": True,      # the explicit nu_t (grad u)^T term
    "choi": True,           # the old-flux (Choi) term
    "lagflux": True,        # the step's last flux in the next step's matrix;
                            # False: the base flux, the flux perturbation
                            # dropped from convection
    "upwind_y": "below",    # upwinding on the horizontal faces: below/above
    "extrap_sweeps": 3,     # the wall pressure's extrapolation sweeps
    "d_scale": 1.0,         # Rhie-Chow's D and the pressure matrix scaled
    "aP_rc": "full",        # the aP Rhie-Chow's D reads: full, or "time"
                            # (V/dt' only), or "noy" (no vertical diffusion)
    "nut_scale": 1.0,       # nu_t multiplied
    "vd_scale": 1.0,        # the vertical diffusion inside the aP that D
                            # reads, multiplied (0 is aP_rc="noy")
}


class Column:
    """One station: y faces, the column width, and the base profiles."""

    def __init__(self, yf, dx, U, nut):
        self.yf = np.asarray(yf, float)
        self.dx = float(dx)
        self.U = np.asarray(U, float)
        self.nut = np.asarray(nut, float)
        assert len(self.U) == len(self.yf) - 1 == len(self.nut)

    @staticmethod
    def from_dump(cells_path, level, x0):
        """The column of cells whose centre is nearest x0, from a dump on the
        TMR grid `level`."""
        here = Path(__file__).resolve().parents[2] / "cases" / "flatplate"
        sys.path.insert(0, str(here))
        import make_mesh  # noqa: E402
        if level in make_mesh.HYBRIDS:
            sx, sy = make_mesh.HYBRIDS[level]
            x, y = make_mesh.read_p2d(here / "tmr" / make_mesh.LEVELS["545x385"])
            x, y = x[::sy, ::sx], y[::sy, ::sx]
        else:
            x, y = make_mesh.read_p2d(here / "tmr" / make_mesh.LEVELS[level])
        xl = x[0]
        yf = y[:, 0]
        xc = 0.5 * (xl[1:] + xl[:-1])
        i = int(np.argmin(np.abs(xc - x0)))
        dx = xl[i + 1] - xl[i]
        d = _DUMPS.get(str(cells_path))
        if d is None:
            d = _DUMPS[str(cells_path)] = np.loadtxt(cells_path)
        sel = np.abs(d[:, 0] - xc[i]) < 1e-9 * max(1.0, abs(xc[i]))
        col = d[sel]
        col = col[np.argsort(col[:, 1])]
        assert len(col) == len(yf) - 1, (len(col), len(yf))
        ym = 0.5 * (yf[1:] + yf[:-1])
        assert np.abs(col[:, 1] - ym).max() < 1e-9
        return Column(yf, dx, col[:, 2], col[:, 6] * NU), xc[i]


class Model:
    """The strip: nx cells periodic in x, the column's rows in y."""

    def __init__(self, col, dt, ncorr, nx=2, opts=None):
        self.o = dict(OPTS)
        if opts:
            for k, v in opts.items():
                if k not in OPTS:
                    raise KeyError(k)
                self.o[k] = v
        self.dt, self.ncorr, self.nx = dt, ncorr, nx
        yf, dx = col.yf, col.dx
        ny = len(yf) - 1
        self.ny, self.nc = ny, nx * ny
        dy = np.diff(yf)
        ym = 0.5 * (yf[1:] + yf[:-1])
        self.dy, self.ym = dy, ym
        cid = lambda i, j: j * nx + i  # noqa: E731
        self.vol = np.repeat(dx * dy, nx)
        self.cc = np.zeros((self.nc, 2))
        for j in range(ny):
            for i in range(nx):
                self.cc[cid(i, j)] = ((i + 0.5) * dx, ym[j])
        own, nei, S, d, w, rO, rN, kind = [], [], [], [], [], [], [], []
        for j in range(ny):                      # vertical faces, periodic
            for i in range(nx):
                own.append(cid(i, j)); nei.append(cid((i + 1) % nx, j))
                S.append((dy[j], 0.0)); d.append((dx, 0.0)); w.append(0.5)
                rO.append((0.5 * dx, 0.0)); rN.append((-0.5 * dx, 0.0)); kind.append(0)
        for j in range(ny - 1):                  # horizontal faces
            for i in range(nx):
                own.append(cid(i, j)); nei.append(cid(i, j + 1))
                S.append((0.0, dx)); d.append((0.0, ym[j + 1] - ym[j]))
                w.append(dy[j + 1] / (dy[j] + dy[j + 1]))
                rO.append((0.0, 0.5 * dy[j])); rN.append((0.0, -0.5 * dy[j + 1])); kind.append(1)
        self.own, self.nei = np.array(own), np.array(nei)
        self.S, self.d = np.array(S), np.array(d)
        self.w, self.rO, self.rN = np.array(w), np.array(rO), np.array(rN)
        self.kind = np.array(kind)
        self.nf = len(own)
        self.a = (self.S ** 2).sum(1) / (self.S * self.d).sum(1)
        self.wl = 1.0 / (self.d ** 2).sum(1)
        # Boundary faces: nx wall faces, then nx top faces.
        bc, Sb, db, btype = [], [], [], []
        for i in range(nx):
            bc.append(cid(i, 0)); Sb.append((0.0, -dx)); db.append((0.0, -0.5 * dy[0])); btype.append(0)
        for i in range(nx):
            bc.append(cid(i, ny - 1)); Sb.append((0.0, dx)); db.append((0.0, 0.5 * dy[-1])); btype.append(1)
        self.bc, self.Sb, self.db = np.array(bc), np.array(Sb), np.array(db)
        self.btype = np.array(btype)             # 0 wall, 1 top
        self.nb = len(bc)
        self.ab = (self.Sb ** 2).sum(1) / (self.Sb * self.db).sum(1)
        self.wb = 1.0 / (self.db ** 2).sum(1)
        self.wall = self.btype == 0
        self.top = self.btype == 1
        # The least-squares normal matrices, inverted.
        A = np.zeros((self.nc, 2, 2))
        dd = self.wl[:, None, None] * self.d[:, :, None] * self.d[:, None, :]
        np.add.at(A, self.own, dd); np.add.at(A, self.nei, dd)
        np.add.at(A, self.bc, self.wb[:, None, None] * self.db[:, :, None] * self.db[:, None, :])
        self.Ainv = np.linalg.inv(A)
        # nu_t on cells and faces; zero on the wall (k = 0 there).
        nut = np.repeat(col.nut * self.o["nut_scale"], nx)
        self.nut = nut
        self.nutf = self.w * nut[self.own] + (1.0 - self.w) * nut[self.nei]
        self.nutB = np.where(self.wall, 0.0, nut[self.bc])
        # Frozen upwinding: True where the owner is upwind.
        self.selO = np.where(self.kind == 0, True, self.o["upwind_y"] == "below")
        # The base.
        U = np.repeat(col.U, nx)
        self.u0 = np.zeros((self.nc, 2)); self.u0[:, 0] = U
        F0 = np.where(self.kind == 0, (self.w * U[self.own] + (1 - self.w) * U[self.nei]) * self.S[:, 0], 0.0)
        self.src = np.zeros((self.nc, 2))
        self.step_count = 1                      # BDF2, as a steady state marches
        base = self.pack(self.u0, self.u0, np.zeros(self.nc), F0, np.zeros(self.nb))
        self.base = base
        self.src = self.body_force(base)
        # Typical sizes per state entry, for the finite differences.
        self.scale = self.pack(np.ones((self.nc, 2)), np.ones((self.nc, 2)), np.ones(self.nc),
                               np.where(self.kind == 0, self.S[:, 0], self.S[:, 1]),
                               np.abs(self.Sb[:, 1]))

    # --- state packing -----------------------------------------------------
    def pack(self, u, uo, p, F, Fb):
        return np.concatenate([u.ravel(), uo.ravel(), p, F, Fb])

    def unpack(self, s):
        nc, nf = self.nc, self.nf
        u = s[:2 * nc].reshape(nc, 2)
        uo = s[2 * nc:4 * nc].reshape(nc, 2)
        p = s[4 * nc:5 * nc]
        F = s[5 * nc:5 * nc + nf]
        Fb = s[5 * nc + nf:]
        return u.copy(), uo.copy(), p.copy(), F.copy(), Fb.copy()

    # --- operators ---------------------------------------------------------
    def bdf(self):
        dt = self.dt
        if self.step_count == 0 or self.o["bdf1"]:
            return 1.0 / dt, -1.0 / dt, 0.0
        return 1.5 / dt, -2.0 / dt, 0.5 / dt

    def grad(self, phi, phiB):
        rhs = np.zeros((self.nc, 2))
        s = (self.wl * (phi[self.nei] - phi[self.own]))[:, None] * self.d
        np.add.at(rhs, self.own, s); np.add.at(rhs, self.nei, s)
        sb = (self.wb * (phiB - phi[self.bc]))[:, None] * self.db
        np.add.at(rhs, self.bc, sb)
        return np.einsum("cab,cb->ca", self.Ainv, rhs)

    def grad_p(self, p):
        # Wall: extrapolated from the cell gradient, iterated from the cell
        # value; top: prescribed, 0.
        v = np.where(self.top, 0.0, p[self.bc])
        for _ in range(self.o["extrap_sweeps"]):
            g = self.grad(p, v)
            ext = p[self.bc] + (g[self.bc] * self.db).sum(1)
            v = np.where(self.top, 0.0, ext)
        return self.grad(p, v)

    def assemble(self, u, uo, uo2, F, Fb, src):
        aPt, a1, a2 = self.bdf()
        own, nei, w = self.own, self.nei, self.w
        diag = aPt * self.vol
        nuf = NU + self.nutf
        Fp = np.where(self.selO, F, 0.0)
        Fn = np.where(self.selO, 0.0, -F)
        np.add.at(diag, own, nuf * self.a + Fp)
        np.add.at(diag, nei, nuf * self.a + Fn)
        up = -nuf * self.a - Fn
        lo = -nuf * self.a - Fp
        # Wall: Dirichlet; top: zero gradient, its flux implicit (outflow).
        np.add.at(diag, self.bc, np.where(self.wall, (NU + self.nutB) * self.ab, Fb))
        # Velocity gradients: the wall's value 0, the top's the cell's.
        G = []
        for dcomp in range(2):
            uB = np.where(self.wall, 0.0, u[self.bc, dcomp])
            G.append(self.grad(u[:, dcomp], uB))
        b = src * self.vol[:, None] - (a1 * uo + a2 * uo2) * self.vol[:, None]
        upc = np.where(self.selO, own, nei)
        rU = np.where(self.selO[:, None], self.rO, self.rN)
        for dcomp in range(2):
            ud = u[upc, dcomp]
            ho = ud + (G[dcomp][upc] * rU).sum(1)            # linear upwind
            dc = F * (ho - ud) if self.o["deferred"] else 0.0
            tflux = 0.0
            if self.o["transpose"]:
                tf = np.zeros(self.nf)
                for j in range(2):
                    gj = w * G[j][own, dcomp] + (1 - w) * G[j][nei, dcomp]
                    tf += gj * self.S[:, j]
                tflux = self.nutf * tf
            val = -dc + tflux
            np.add.at(b[:, dcomp], own, val)
            np.add.at(b[:, dcomp], nei, -val)
            # Boundary faces add nothing: u = 0 and nu_t = 0 on the wall, no
            # backflow term with the top's flux taken implicit.
        n = self.nc
        A = sp.csr_matrix((np.concatenate([diag, up, lo]),
                           (np.concatenate([np.arange(n), own, nei]),
                            np.concatenate([np.arange(n), nei, own]))), shape=(n, n))
        return A, diag, up, lo, b

    def vertical_diffusion(self):
        """Each cell's share of aP from the diffusion through its horizontal
        faces and the wall."""
        vd = np.zeros(self.nc)
        ky = self.kind == 1
        cy = ((NU + self.nutf) * self.a)[ky]
        np.add.at(vd, self.own[ky], cy); np.add.at(vd, self.nei[ky], cy)
        np.add.at(vd, self.bc, np.where(self.wall, (NU + self.nutB) * self.ab, 0.0))
        return vd

    def step(self, s):
        u, uold, p, F, Fb = self.unpack(s)
        own, nei, w, vol = self.own, self.nei, self.w, self.vol
        aPt, _, _ = self.bdf()
        # Time levels.
        uo2 = uold
        uo = u.copy()
        FOld = F.copy()
        Fb = np.where(self.top, Fb, 0.0)
        # The old residual, R = F_old - I[u_old].S (exact form).
        uf = ((w[:, None] * uo[own] + (1 - w[:, None]) * uo[nei]) * self.S).sum(1)
        rOld = FOld - uf
        rb = np.where(self.top, Fb - (uo[self.bc] * self.Sb).sum(1), 0.0)
        Fmat = F if self.o["lagflux"] else self.unpack(self.base)[3]
        Fbmat = Fb if self.o["lagflux"] else np.zeros(self.nb)
        A, aP, up, lo, b = self.assemble(u, uo, uo2, Fmat, Fbmat, self.src)
        gp = self.grad_p(p)
        lu = spla.splu(A.tocsc())
        for dcomp in range(2):
            u[:, dcomp] = lu.solve(b[:, dcomp] - gp[:, dcomp] * vol)
        # Rhie-Chow's aP and the pressure matrix.
        if self.o["aP_rc"] == "full" and self.o["vd_scale"] != 1.0:
            aRC = aP + (self.o["vd_scale"] - 1.0) * self.vertical_diffusion()
        elif self.o["aP_rc"] == "full":
            aRC = aP
        elif self.o["aP_rc"] == "time":
            aRC = aPt * vol
        elif self.o["aP_rc"] == "noy":
            # aP without the diffusion through the horizontal faces and the
            # wall: what a mode uniform in y does not feel.
            aRC = aP - self.vertical_diffusion()
        else:
            raise ValueError(self.o["aP_rc"])
        D = self.o["d_scale"] * (w * vol[own] + (1 - w) * vol[nei]) / (w * aRC[own] + (1 - w) * aRC[nei])
        Dfb = self.o["d_scale"] * vol[self.bc] / aRC[self.bc]
        n = self.nc
        ap = self.a * D
        pdiag = np.zeros(n)
        np.add.at(pdiag, own, ap); np.add.at(pdiag, nei, ap)
        np.add.at(pdiag, self.bc, np.where(self.top, self.ab * Dfb, 0.0))
        P = sp.csr_matrix((np.concatenate([pdiag, -ap, -ap]),
                           (np.concatenate([np.arange(n), own, nei]),
                            np.concatenate([np.arange(n), nei, own]))), shape=(n, n))
        plu = spla.splu(P.tocsc())
        for _ in range(self.ncorr):
            H = b.copy()
            np.add.at(H, own, -up[:, None] * u[nei])
            np.add.at(H, nei, -lo[:, None] * u[own])
            H /= aP[:, None]
            q = H - gp * (vol / aP)[:, None]
            Fs = ((w[:, None] * q[own] + (1 - w[:, None]) * q[nei]) * self.S).sum(1)
            Fs += D * ((w[:, None] * gp[own] + (1 - w[:, None]) * gp[nei]) * self.S).sum(1)
            if self.o["choi"]:
                Fs += D * aPt * rOld
            fbs = np.where(self.top, (H[self.bc] * self.Sb).sum(1)
                           + (Dfb * aPt * rb if self.o["choi"] else 0.0), 0.0)
            srcp = np.zeros(n)
            np.add.at(srcp, own, -Fs); np.add.at(srcp, nei, Fs)
            np.add.at(srcp, self.bc, -fbs)
            p = plu.solve(srcp)
            gp = self.grad_p(p)
            F = Fs - ap * (p[nei] - p[own])
            Fb = np.where(self.top, fbs - self.ab * Dfb * (0.0 - p[self.bc]), 0.0)
            u = H - gp * (vol / aP)[:, None]
        return self.pack(u, uo, p, F, Fb)

    def body_force(self, s):
        """The source that makes s a fixed point of the step: the residual of
        the momentum equation at s, per unit volume."""
        u, uold, p, F, Fb = self.unpack(s)
        A, aP, up, lo, b = self.assemble(u, u, u, F, Fb, np.zeros((self.nc, 2)))
        r = np.column_stack([A @ u[:, 0] - b[:, 0], A @ u[:, 1] - b[:, 1]])
        return r / self.vol[:, None]

    # --- the odd-even subspace ----------------------------------------------
    def parity_basis(self, sign=-1):
        """Columns spanning the states odd (sign=-1) or even (+1) under a shift
        by one cell in x; nx must be 2."""
        assert self.nx == 2
        cols = []
        nc, nf = self.nc, self.nf

        def cellvec(block, j, comp=None):
            v = np.zeros(self.base.size)
            for i, sgn in ((0, 1.0), (1, sign)):
                c = j * 2 + i
                if block in (0, 1):
                    v[block * 2 * nc + 2 * c + comp] = sgn
                else:
                    v[4 * nc + c] = sgn
            return v
        for j in range(self.ny):
            for block, comp in ((0, 0), (0, 1), (1, 0), (1, 1), (2, None)):
                cols.append(cellvec(block, j, comp))
        # Faces: the pair (i = 0, 1) at each place, the second with the sign.
        for f in range(0, nf, 2):
            v = np.zeros(self.base.size)
            v[5 * nc + f] = 1.0; v[5 * nc + f + 1] = sign
            cols.append(v)
        v = np.zeros(self.base.size)
        v[5 * nc + nf + 2] = 1.0; v[5 * nc + nf + 3] = sign     # the top pair
        cols.append(v)
        # The wall faces carry no flux: nothing to perturb there.
        return np.array(cols).T

    def jacobian(self, basis, h=1e-6):
        """The step's Jacobian restricted to span(basis), in that basis."""
        m = basis.shape[1]
        J = np.zeros((m, m))
        B = basis * self.scale[:, None]
        # The basis columns are orthogonal; project back with their norms.
        norm2 = (B * B).sum(0)
        for k in range(m):
            e = B[:, k]
            dp = self.step(self.base + h * e)
            dm = self.step(self.base - h * e)
            ds = (dp - dm) / (2 * h)
            J[:, k] = (B.T @ ds) / norm2
        return J

    def growth(self, sign=-1, h=1e-6, full=False):
        J = self.jacobian(self.parity_basis(sign), h)
        ev = np.linalg.eigvals(J)
        k = np.argmax(np.abs(ev))
        return (ev[k], ev) if full else ev[k]

    def growth_arnoldi(self, sign=-1, h=1e-6, nev=2):
        """The same largest eigenvalue by Arnoldi iteration on Jacobian-vector
        products (central differences of the step), without forming the
        Jacobian: a few dozen steps instead of two per basis vector."""
        B = sp.csc_matrix(self.parity_basis(sign) * self.scale[:, None])
        norm2 = np.asarray(B.multiply(B).sum(0)).ravel()
        m = B.shape[1]

        def mv(c):
            e = B @ np.asarray(c).ravel()
            ds = (self.step(self.base + h * e) - self.step(self.base - h * e)) / (2 * h)
            return (B.T @ ds) / norm2
        op = spla.LinearOperator((m, m), matvec=mv, dtype=float)
        try:
            ev = spla.eigs(op, k=nev, which="LM", return_eigenvectors=False, tol=1e-5,
                           ncv=max(2 * nev + 1, 40), maxiter=10)
        except spla.ArpackNoConvergence:
            # A crowded top of the spectrum: the growth of a random
            # perturbation over 200 steps, as a march would see it (the
            # modulus only; the sign of a real eigenvalue from the last ratio).
            rng = np.random.default_rng(0)
            c = rng.standard_normal(m)
            c /= np.linalg.norm(c)
            logs = []
            for _ in range(200):
                c2 = mv(c)
                nrm = np.linalg.norm(c2)
                logs.append(np.log(nrm))
                sgn = np.sign(np.dot(c, c2))
                c = c2 / nrm
            return sgn * np.exp(np.mean(logs[-50:]))
        return ev[np.argmax(np.abs(ev))]

    def check_base(self):
        """How far the base moves in one step, relative to its size."""
        s1 = self.step(self.base)
        return np.abs((s1 - self.base) / self.scale).max()


def scan(dump, level, dt, ncorr, opts=None, stations=STATIONS):
    """The odd-even growth factor at each station, by Arnoldi: a list of
    (x, lambda)."""
    out = []
    for x0 in stations:
        col, xs = Column.from_dump(dump, level, x0)
        out.append((xs, Model(col, dt, ncorr, opts=opts).growth_arnoldi()))
    return out


def main(argv):
    if len(argv) >= 2 and argv[1] == "--scan":
        # piso_mode.py --scan <dump> <level> dt ncorr [opt=value ...]
        dump, level, dt, ncorr = argv[2], argv[3], float(argv[4]), int(argv[5])
        opts = {}
        for a in argv[6:]:
            k, v = a.split("=")
            dv = OPTS[k]
            opts[k] = (v.lower() in ("1", "true", "yes")) if isinstance(dv, bool) else type(dv)(v)
        res = scan(dump, level, dt, ncorr, opts)
        best = max(res, key=lambda r: abs(r[1]))
        print(f"{level} dt {dt:g} correctors {ncorr} {opts or ''}: largest |lambda| "
              f"{abs(best[1]):.5f} at x = {best[0]:.4f}")
        print("  " + "  ".join(f"{x:.3f}:{abs(l):.4f}" for x, l in res))
        return 0
    if len(argv) < 6:
        print(__doc__)
        return 2
    dump, level, x0, dt, ncorr = argv[1], argv[2], float(argv[3]), float(argv[4]), int(argv[5])
    opts = {}
    for a in argv[6:]:
        k, v = a.split("=")
        dv = OPTS[k]
        opts[k] = (v.lower() in ("1", "true", "yes")) if isinstance(dv, bool) else type(dv)(v)
    col, xs = Column.from_dump(dump, level, x0)
    m = Model(col, dt, ncorr, opts=opts)
    print(f"{level} x = {xs:.5f} dx = {col.dx:.4e} rows {m.ny}  dt {dt:g}  correctors {ncorr}  "
          f"{opts or ''}")
    print(f"  base steady to {m.check_base():.1e}")
    lam, ev = m.growth(full=True)
    order = np.argsort(-np.abs(ev))
    print(f"  odd-even: |lambda| {abs(lam):.6f}  lambda {lam:.6f}")
    print("  next:", " ".join(f"{abs(ev[i]):.5f}" for i in order[1:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
