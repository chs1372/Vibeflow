"""
Menter's k-omega SST model, integrated to the wall (ADR-042).

    dk/dt + div(u k) = P~ - beta* w k + div[(nu + sigma_k nu_t) grad k]
    dw/dt + div(u w) = gamma P~(or P)/nu_t - beta w^2 + div[(nu + sigma_w nu_t) grad w]
                       + 2 (1 - F1) sigma_w2 grad k . grad w / w

in two variants (SST-2003, the default, and SST-1994, in which the TMR
benchmarks were run); ADR-042 has the table of what differs. P = nu_t S^2
exactly, the flow being incompressible.

k and omega are transported as the temperature is (ADR-038): BDF2, upwind in
the matrix plus the deferred correction to the skew-corrected face value,
diffusion with the non-orthogonal correction, the face diffusivity
interpolated linearly from the cells. Destruction is implicit, production
explicit, the cross diffusion explicit where positive and implicit where
negative. PisoSolver calls solve() once per outer iteration, after the
pressure correctors; advance_frozen() marches the pair alone in a given flow.

Boundary kinds, one per boundary face:
  DIRICHLET       k and omega prescribed (an inlet, or a manufactured value)
  ZERO_GRADIENT   an outlet, a slip face, a symmetry plane
  WALL            k = 0, omega = 10 * 6 nu / (beta_1 d_1^2), d_1 the wall
                  distance of the adjacent cell centre (Menter 1994)
The wall distance is measured from the faces flagged `wall` at construction,
which need not be the WALL faces: a manufactured solution measures d from
y = 0 and prescribes the exact values there.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fvm import DiffusionOperator
from walldist import wall_distance

DIRICHLET, ZERO_GRADIENT, WALL = 0, 1, 2

SIGMA_K1, SIGMA_W1, BETA1 = 0.85, 0.5, 0.075
SIGMA_K2, SIGMA_W2, BETA2 = 1.0, 0.856, 0.0828
BETA_STAR, KAPPA, A1 = 0.09, 0.41, 0.31


class SstModel:
    def __init__(self, mesh, nu, variant="2003", wall=None, k_floor=1e-20, w_floor=1e-20):
        if variant == "2003":
            self.g1, self.g2 = 5.0 / 9.0, 0.44
            self.cd_floor, self.plim = 1e-10, 10.0
            self.limit_w_production, self.use_vorticity = True, False
        elif variant == "1994":
            rb = np.sqrt(BETA_STAR)
            self.g1 = BETA1 / BETA_STAR - SIGMA_W1 * KAPPA ** 2 / rb
            self.g2 = BETA2 / BETA_STAR - SIGMA_W2 * KAPPA ** 2 / rb
            self.cd_floor, self.plim = 1e-20, 20.0
            self.limit_w_production, self.use_vorticity = False, True
        else:
            raise ValueError(variant)
        self.variant = variant
        m = mesh
        self.m = m
        self.nu = nu
        self.ops = DiffusionOperator(mesh, 1.0)          # geometry of the diffusion
        self.grad = self.ops.grad
        dof = np.linalg.norm(m.face_centre - m.cell_centre[m.owner], axis=1)
        dnf = np.linalg.norm(m.face_centre - m.cell_centre[m.neigh], axis=1)
        self.fw = dnf / (dof + dnf)
        x_lin = (self.fw[:, None] * m.cell_centre[m.owner]
                 + (1.0 - self.fw)[:, None] * m.cell_centre[m.neigh])
        self.skew_vec = m.face_centre - x_lin

        nb = len(m.b_cell)
        self.wall = np.zeros(nb, dtype=bool) if wall is None else np.asarray(wall, dtype=bool)
        self.d = wall_distance(m, self.wall)
        self.d_b = wall_distance(m, self.wall, m.b_centre)
        self.d_b[self.wall] = 0.0

        self.k_floor, self.w_floor = k_floor, w_floor
        self.k = np.zeros(m.nc)
        self.w = np.zeros(m.nc)
        self.k_old, self.k_old2 = self.k.copy(), self.k.copy()
        self.w_old, self.w_old2 = self.w.copy(), self.w.copy()
        self.nut = np.zeros(m.nc)
        self.nut_b = np.zeros(nb)
        self.kind = np.full(nb, ZERO_GRADIENT)
        self.k_b = np.zeros(nb)
        self.w_b = np.zeros(nb)
        self.k_src = np.zeros(m.nc)
        self.w_src = np.zeros(m.nc)
        self.bounded = 0            # cells bounded in the last solve
        self.bounded_total = 0
        self.step_index = 0         # for advance_frozen's BDF

    # ------------------------------------------------------------ set-up
    def set_boundary(self, k_b, w_b, kind):
        self.k_b = np.asarray(k_b, float).copy()
        self.w_b = np.asarray(w_b, float).copy()
        self.kind = np.asarray(kind).copy()

    def set_source(self, k_src, w_src):
        self.k_src = np.asarray(k_src, float).copy()
        self.w_src = np.asarray(w_src, float).copy()

    def set_state(self, k, w):
        self.k = np.asarray(k, float).copy()
        self.w = np.asarray(w, float).copy()
        self.k_old, self.k_old2 = self.k.copy(), self.k.copy()
        self.w_old, self.w_old2 = self.w.copy(), self.w.copy()

    def shift(self):
        """Time levels move once per step."""
        self.k_old2, self.k_old = self.k_old, self.k.copy()
        self.w_old2, self.w_old = self.w_old, self.w.copy()

    # ------------------------------------------------------------ model
    def omega_wall(self):
        return 60.0 * self.nu / (BETA1 * self.d[self.m.b_cell] ** 2)

    def boundary_values(self):
        """Face values the gradients and the Dirichlet faces use."""
        bc = self.m.b_cell
        wall = self.kind == WALL
        dirichlet = self.kind == DIRICHLET
        kb = np.where(dirichlet, self.k_b, np.where(wall, 0.0, self.k[bc]))
        wb = np.where(dirichlet, self.w_b, np.where(wall, self.omega_wall(), self.w[bc]))
        return kb, wb

    def invariants(self, u, u_b):
        """Cell velocity gradient G[c, i, :] = grad u_i, S and Omega."""
        G = np.stack([self.grad(u[:, i], u_b[:, i]) for i in range(3)], axis=1)
        sym = G + np.transpose(G, (0, 2, 1))
        skw = G - np.transpose(G, (0, 2, 1))
        S = np.sqrt(0.5 * np.einsum("cij,cij->c", sym, sym))
        Om = np.sqrt(0.5 * np.einsum("cij,cij->c", skw, skw))
        return G, S, Om

    def blending(self, k, w, kw, d, V):
        """F1, F2 and nu_t from k, omega, grad k . grad omega, the wall
        distance and the rate the limiter uses. F1 = F2 = 1 on the wall."""
        with np.errstate(divide="ignore", invalid="ignore"):
            rk = np.sqrt(np.maximum(k, 0.0))
            brA = rk / (BETA_STAR * w * d)
            brB = 500.0 * self.nu / (d * d * w)
            CD = np.maximum(2.0 * SIGMA_W2 * kw / w, self.cd_floor)
            brC = 4.0 * SIGMA_W2 * k / (CD * d * d)
            arg1 = np.minimum(np.maximum(brA, brB), brC)
            arg2 = np.maximum(2.0 * rk / (BETA_STAR * w * d), brB)
            F1 = np.where(d > 0.0, np.tanh(arg1 ** 4), 1.0)
            F2 = np.where(d > 0.0, np.tanh(arg2 ** 2), 1.0)
        nut = A1 * k / np.maximum(A1 * w, V * F2)
        return F1, F2, nut

    def fields(self, u, u_b):
        """Everything the two equations need, from the current k and omega."""
        m = self.m
        kb, wb = self.boundary_values()
        gk = self.grad(self.k, kb)
        gw = self.grad(self.w, wb)
        kw = np.einsum("ij,ij->i", gk, gw)
        G, S, Om = self.invariants(u, u_b)
        V = Om if self.use_vorticity else S
        F1, F2, nut = self.blending(self.k, self.w, kw, self.d, V)
        bc = m.b_cell
        F1b, F2b, nutb = self.blending(kb, wb, kw[bc], self.d_b, V[bc])
        zg = self.kind == ZERO_GRADIENT
        nutb = np.where(zg, nut[bc], nutb)
        F1b = np.where(zg, F1[bc], F1b)
        return dict(kb=kb, wb=wb, gk=gk, gw=gw, kw=kw, S=S, V=V, F1=F1, F2=F2,
                    nut=nut, F1b=F1b, nutb=nutb)

    def update_nut(self, u, u_b):
        f = self.fields(u, u_b)
        self.nut, self.nut_b = f["nut"], f["nutb"]
        return f

    # ------------------------------------------------------------ transport
    def _transport(self, phi, old, old2, phib, gphi, Gc, Gb, F, Fb, bdf, diag_v, rhs_v):
        """One implicit solve of d(phi)/dt + div(F phi) = div(G grad phi) + ...,
        with diag_v and rhs_v the implicit and explicit sources per unit
        volume; as PisoSolver.solve_energy, the diffusivity a field."""
        m, ops = self.m, self.ops
        aP_t, a1, a2 = bdf
        V = m.cell_volume
        fixed = self.kind != ZERO_GRADIENT
        Gf = self.fw * Gc[m.owner] + (1.0 - self.fw) * Gc[m.neigh]
        a = Gf * ops.a_int
        Fp, Fn = np.maximum(F, 0.0), np.maximum(-F, 0.0)
        rows = [m.owner, m.owner, m.neigh, m.neigh, m.b_cell,
                m.owner, m.owner, m.neigh, m.neigh, np.arange(m.nc), m.b_cell]
        cols = [m.owner, m.neigh, m.neigh, m.owner, m.b_cell,
                m.owner, m.neigh, m.neigh, m.owner, np.arange(m.nc), m.b_cell]
        vals = [+a, -a, +a, -a, np.where(fixed, Gb * ops.a_bnd, 0.0),
                Fp, -Fn, Fn, -Fp, (aP_t + diag_v) * V,
                np.where(fixed, 0.0, np.maximum(Fb, 0.0))]   # outflow, zero gradient
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m.nc, m.nc)).tocsr()

        rhs = rhs_v * V - (a1 * old + a2 * old2) * V
        np.add.at(rhs, m.b_cell, np.where(fixed, Gb * ops.a_bnd * phib, 0.0))
        gf = (ops.w_owner[:, None] * gphi[m.owner] + (1.0 - ops.w_owner)[:, None] * gphi[m.neigh])
        corr = Gf * np.einsum("ij,ij->i", ops.k_int, gf)
        np.add.at(rhs, m.owner, corr)
        np.add.at(rhs, m.neigh, -corr)
        np.add.at(rhs, m.b_cell, np.where(
            fixed, Gb * np.einsum("ij,ij->i", ops.k_bnd, gphi[m.b_cell]), 0.0))
        lin = self.fw * phi[m.owner] + (1.0 - self.fw) * phi[m.neigh]
        gfs = self.fw[:, None] * gphi[m.owner] + (1.0 - self.fw)[:, None] * gphi[m.neigh]
        ho = lin + np.einsum("ij,ij->i", gfs, self.skew_vec)
        ud = np.where(F > 0.0, phi[m.owner], phi[m.neigh])
        dc = F * (ho - ud)
        np.add.at(rhs, m.owner, -dc)
        np.add.at(rhs, m.neigh, +dc)
        conv_b = np.where(fixed, Fb * phib, np.minimum(Fb, 0.0) * phib)
        np.add.at(rhs, m.b_cell, -conv_b)
        return spla.spsolve(A.tocsc(), rhs)

    def solve(self, F, Fb, u, u_b, bdf):
        """k, then omega, with the coefficients of the current iterate; then
        nu_t. One call per outer iteration."""
        f = self.fields(u, u_b)
        k, w = self.k, self.w
        F1, nut, S2 = f["F1"], f["nut"], f["S"] ** 2
        F1b = f["F1b"]
        sk = F1 * SIGMA_K1 + (1 - F1) * SIGMA_K2
        sw = F1 * SIGMA_W1 + (1 - F1) * SIGMA_W2
        beta = F1 * BETA1 + (1 - F1) * BETA2
        gamma = F1 * self.g1 + (1 - F1) * self.g2
        skb = F1b * SIGMA_K1 + (1 - F1b) * SIGMA_K2
        swb = F1b * SIGMA_W1 + (1 - F1b) * SIGMA_W2

        P = nut * S2
        Pk = np.minimum(P, self.plim * BETA_STAR * w * k)
        if self.limit_w_production:
            prod_w = gamma * np.minimum(S2, self.plim * BETA_STAR * w * k / nut)
        else:
            prod_w = gamma * S2                      # gamma P / nu_t, P = nu_t S^2
        cross = 2.0 * (1.0 - F1) * SIGMA_W2 * f["kw"] / w

        # An imposed source (a manufactured one) enters as Patankar's rule
        # has it: its positive part explicit, its negative part on the
        # diagonal, -s/phi of the current iterate, so a sink cannot drive the
        # quantity negative. The steady state is the same.
        ks_pos, ks_neg = np.maximum(self.k_src, 0.0), np.maximum(-self.k_src, 0.0)
        ws_pos, ws_neg = np.maximum(self.w_src, 0.0), np.maximum(-self.w_src, 0.0)
        k_new = self._transport(k, self.k_old, self.k_old2, f["kb"], f["gk"],
                                self.nu + sk * nut, self.nu + skb * f["nutb"], F, Fb, bdf,
                                BETA_STAR * w + ks_neg / k, Pk + ks_pos)
        w_new = self._transport(w, self.w_old, self.w_old2, f["wb"], f["gw"],
                                self.nu + sw * nut, self.nu + swb * f["nutb"], F, Fb, bdf,
                                beta * w + np.where(cross < 0.0, -cross / w, 0.0) + ws_neg / w,
                                prod_w + np.where(cross > 0.0, cross, 0.0) + ws_pos)
        low = (k_new < self.k_floor) | (w_new < self.w_floor)
        self.bounded = int(low.sum())
        self.bounded_total += self.bounded
        self.k = np.maximum(k_new, self.k_floor)
        self.w = np.maximum(w_new, self.w_floor)
        self.update_nut(u, u_b)

    def advance_frozen(self, u, u_b, F, Fb, dt):
        """One BDF step of k and omega alone in the flow (u, F)."""
        self.shift()
        bdf = ((1.0 / dt, -1.0 / dt, 0.0) if self.step_index == 0
               else (1.5 / dt, -2.0 / dt, 0.5 / dt))
        self.solve(F, Fb, u, u_b, bdf)
        self.step_index += 1
