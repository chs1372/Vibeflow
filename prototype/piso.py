"""
Incompressible Navier-Stokes: SIMPLE/PISO with Rhie-Chow, BDF2 in time.

    du/dt + div(u u) = -grad(p) + nu laplacian(u) + s,    div(u) = 0

Collocated variables (ADR-010). The face mass flux comes from Rhie-Chow
interpolation, not from interpolating u, because interpolating u leaves the
pressure decoupled on the odd-even mode.

Two things here are easy to get subtly wrong and both are covered by gates:

1. THE SECOND CORRECTOR. PISO's extra corrector is only worth anything if H is
   re-evaluated from the updated velocity. Writing HbyA = u + grad(p) V/aP
   after having just set u = HbyA - grad(p) V/aP is an identity, and the second
   corrector then does nothing while appearing to run. H is recomputed here
   from the assembled matrix: H = b - (A - diag) u.

2. THE TIME-STEP TRAP (ADR-010). aP contains the transient term V/dt, so a
   naive Rhie-Chow flux loses its pressure damping as dt shrinks, and a steady
   state reached with one dt differs from the same state reached with another.
   The old-time flux term below cancels the transient part of aP, making the
   flux dt-independent. gate_dt_independence measures exactly that.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fvm import DiffusionOperator


class PisoSolver:
    def __init__(self, mesh, nu, dt, n_correctors=2, n_nonorth=40,
                 n_outer=1, outer_tol=1e-10, consistent_rhie_chow=True):
        self.m = mesh
        self.nu = nu
        self.dt = dt
        self.nCorr = n_correctors
        # Outer (PIMPLE) iterations. With n_outer = 1 the convecting mass flux
        # and the deferred convection correction are those of the PREVIOUS time
        # step, which is an O(dt) error however good the time scheme is: BDF2
        # then measures about 1.5 order instead of 2. Iterating the outer loop
        # to convergence makes them new-time values and restores second order.
        self.nOuter = n_outer
        self.outerTol = outer_tol
        self.outer_used = 0
        # The non-orthogonal pressure corrector is iterated to convergence,
        # not to a fixed count: on a skewed mesh a fixed three sweeps leaves a
        # continuity residual around 1e-7, which then limits the whole solver.
        self.nNonOrth = n_nonorth
        self.nonorth_sweeps = 0
        self.consistent = consistent_rhie_chow

        m = mesh
        self.diff = DiffusionOperator(mesh, nu)      # momentum viscous term
        self.pdiff = DiffusionOperator(mesh, 1.0)    # pressure Laplacian geometry
        self.grad = self.diff.grad

        dof = np.linalg.norm(m.face_centre - m.cell_centre[m.owner], axis=1)
        dnf = np.linalg.norm(m.face_centre - m.cell_centre[m.neigh], axis=1)
        self.w = dnf / (dof + dnf)
        x_lin = (self.w[:, None] * m.cell_centre[m.owner]
                 + (1.0 - self.w)[:, None] * m.cell_centre[m.neigh])
        self.skew_vec = m.face_centre - x_lin

        self.u = np.zeros((m.nc, 3))
        self.u_old = np.zeros((m.nc, 3))
        self.u_old2 = np.zeros((m.nc, 3))
        self.p = np.zeros(m.nc)
        self.F = np.zeros(len(m.owner))
        self.F_old = np.zeros(len(m.owner))
        self.F_old2 = np.zeros(len(m.owner))
        self.Fb = np.zeros(len(m.b_cell))
        self.step_index = 0
        self._pb = np.zeros(len(m.b_cell))

    # ------------------------------------------------------------ time scheme
    def bdf(self):
        """(aP, a_old, a_old2). BDF1 on the first step, BDF2 after."""
        if self.step_index == 0:
            return 1.0 / self.dt, -1.0 / self.dt, 0.0
        return 1.5 / self.dt, -2.0 / self.dt, 0.5 / self.dt

    def p_boundary(self, p, n_iter=3):
        """Boundary face pressure by consistent extrapolation.

        Using the cell value (zero normal gradient) is WRONG here: the wall
        normal pressure gradient is not zero, it is whatever the momentum
        equation requires. A zero-gradient value makes the gradient in the
        boundary layer of cells first-order accurate, and in 3D a first-order
        boundary layer drags the global L2 order down to about 1.5 -- which is
        exactly what this solver measured before the fix.

        Extrapolating from the interior gradient keeps it second order:
            p_b = p_P + grad(p)_P . (x_b - x_P),   iterated a few times.
        """
        m = self.m
        d = m.b_centre - m.cell_centre[m.b_cell]
        v = p[m.b_cell]
        for _ in range(n_iter):
            g = self.grad(p, v)
            v = p[m.b_cell] + np.einsum("ij,ij->i", g[m.b_cell], d)
        self._pb = v
        return v

    def grad_gauss(self, q, q_b, n_iter=2):
        """Green-Gauss gradient: (1/V) sum over faces of q_f S_f.

        The pressure gradient must be reconstructed the same way the pressure
        equation measures divergence, or the projection is not a discrete
        Helmholtz decomposition and the velocity it produces is not the one the
        flux correction assumes. A least-squares gradient here is accurate in
        isolation but inconsistent with the face-based continuity equation, and
        on a skewed mesh that inconsistency costs a full order.

        Face values are skewness-corrected, iterating once on the gradient.
        """
        m = self.m
        g = np.zeros((m.nc, 3))
        for _ in range(n_iter):
            qf = self.w * q[m.owner] + (1 - self.w) * q[m.neigh]
            gf = self.w[:, None] * g[m.owner] + (1 - self.w)[:, None] * g[m.neigh]
            qf = qf + np.einsum("ij,ij->i", gf, self.skew_vec)
            acc = np.zeros((m.nc, 3))
            np.add.at(acc, m.owner, qf[:, None] * m.face_area)
            np.add.at(acc, m.neigh, -qf[:, None] * m.face_area)
            np.add.at(acc, m.b_cell, q_b[:, None] * m.b_area)
            g = acc / m.cell_volume[:, None]
        return g

    def grad_p(self, p):
        # MEASURED on the exact Ethier-Steinman pressure over a skewed mesh
        # (max non-orthogonality 30 deg), L2 error of the reconstructed
        # gradient against the analytic one:
        #
        #   n     Green-Gauss    order      LSQ linear    order
        #   6       1.64e-01        -         1.19e-01        -
        #  12       1.47e-01     0.16         5.49e-02     1.11
        #  24       1.62e-01    -0.14         2.66e-02     1.05
        #
        # Green-Gauss does not converge at all on a randomly perturbed mesh;
        # least squares is first order. Neither is second order, and since the
        # velocity correction u = H/aP - grad(p) V/aP uses this gradient
        # directly, the velocity inherits the gradient's order: the solver
        # measures 1.17 for velocity while the PRESSURE it is built from
        # converges at 2.06. A quadratic least-squares fit over a wider
        # stencil is the known fix and is the next task. LSQ is used until
        # then because first order beats zeroth.
        return self.grad(p, self.p_boundary(p))

    def face_interp(self, q, grad_q):
        """Linear interpolation with skewness correction, componentwise."""
        m = self.m
        lin = self.w[:, None] * q[m.owner] + (1 - self.w)[:, None] * q[m.neigh]
        gf = (self.w[:, None, None] * grad_q[m.owner]
              + (1 - self.w)[:, None, None] * grad_q[m.neigh])
        return lin + np.einsum("ijk,ik->ij", gf, self.skew_vec)

    # -------------------------------------------------------------- momentum
    def assemble_momentum(self, u_b, src):
        """Matrix and pressure-free right-hand sides for the three components."""
        m = self.m
        aP_t, a1, a2 = self.bdf()
        Fp = np.maximum(self.F, 0.0)
        Fn = np.maximum(-self.F, 0.0)
        a = self.diff.a_int
        nu = self.nu

        rows = [m.owner, m.owner, m.neigh, m.neigh, m.b_cell,
                m.owner, m.owner, m.neigh, m.neigh, np.arange(m.nc)]
        cols = [m.owner, m.neigh, m.neigh, m.owner, m.b_cell,
                m.owner, m.neigh, m.neigh, m.owner, np.arange(m.nc)]
        vals = [+nu * a, -nu * a, +nu * a, -nu * a, +nu * self.diff.a_bnd,
                Fp, -Fn, Fn, -Fp, aP_t * m.cell_volume]
        A = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m.nc, m.nc)).tocsr()

        grads = np.stack([self.grad(self.u[:, d], u_b[:, d]) for d in range(3)], axis=1)
        ho = self.face_interp(self.u, grads)

        b = np.empty((m.nc, 3))
        for d in range(3):
            g = grads[:, d, :]
            rhs = src[:, d] * m.cell_volume
            rhs -= (a1 * self.u_old[:, d] + a2 * self.u_old2[:, d]) * m.cell_volume

            np.add.at(rhs, m.b_cell, nu * self.diff.a_bnd * u_b[:, d])
            gf = (self.diff.w_owner[:, None] * g[m.owner]
                  + (1 - self.diff.w_owner)[:, None] * g[m.neigh])
            corr = nu * np.einsum("ij,ij->i", self.diff.k_int, gf)
            np.add.at(rhs, m.owner, corr)
            np.add.at(rhs, m.neigh, -corr)
            np.add.at(rhs, m.b_cell,
                      nu * np.einsum("ij,ij->i", self.diff.k_bnd, g[m.b_cell]))

            ud = np.where(self.F > 0.0, self.u[m.owner, d], self.u[m.neigh, d])
            dc = self.F * (ho[:, d] - ud)
            np.add.at(rhs, m.owner, -dc)
            np.add.at(rhs, m.neigh, +dc)
            np.add.at(rhs, m.b_cell, -self.Fb * u_b[:, d])
            b[:, d] = rhs
        return A, b

    def HbyA(self, A, b, aP, u):
        """H/aP with H = b - (A - diag) u, recomputed from the current u.

        Recomputing this is the whole point of a second PISO corrector.
        """
        H = np.empty_like(u)
        for d in range(3):
            H[:, d] = b[:, d] - (A @ u[:, d]) + aP * u[:, d]
        return H / aP[:, None]

    # -------------------------------------------------- Rhie-Chow face flux
    def rhie_chow(self, HbyA, aP, u_b=None):
        m = self.m
        Df = (self.w * (m.cell_volume / aP)[m.owner]
              + (1 - self.w) * (m.cell_volume / aP)[m.neigh])

        # H/aP must be interpolated to the face WITH the skewness correction.
        # Plain linear interpolation is first order once the face centre is off
        # the line joining the cell centres, and since this sets the mass flux
        # it drags the whole solution down with it: the scheme measures second
        # order on an orthogonal mesh and about first order on a skewed one.
        gH = np.stack([self.grad(HbyA[:, d], HbyA[m.b_cell, d]) for d in range(3)],
                      axis=1)
        Hf = self.face_interp(HbyA, gH)
        F = np.einsum("ij,ij->i", Hf, m.face_area)

        gp = self.grad_p(self.p)
        gpf = self.w[:, None] * gp[m.owner] + (1 - self.w)[:, None] * gp[m.neigh]
        snGrad = self.pdiff.a_int * (self.p[m.neigh] - self.p[m.owner])
        F += Df * (np.einsum("ij,ij->i", gpf, m.face_area) - snGrad)

        if self.consistent:
            # Old-flux term (Choi 1999). R = F - u_f_bar . S is the Rhie-Chow
            # residual; carrying it forward with the coefficient Df * aP_t
            # cancels the transient part of aP, so the pressure damping -- and
            # therefore the steady state -- does not depend on dt.
            #
            # Weighting this term with the full BDF2 coefficients instead
            # (-Df (a1 R_old + a2 R_old2)) looks more consistent and gives the
            # same steady value, but its recursion has a root at exactly 1:
            # the initial R never decays, so the converged state depends on
            # the path taken to reach it. MEASURED on the steady MMS, skewed
            # mesh, dt of 0.05 vs 2.0, both run to |du| < 1e-13:
            #   BDF2 weights: L2 2.055e-02 vs 2.002e-02  (2.6% apart)
            # The form below has a strictly decaying mode and one fixed point.
            aP_t, _, _ = self.bdf()
            uf_old = (self.w[:, None] * self.u_old[m.owner]
                      + (1 - self.w)[:, None] * self.u_old[m.neigh])
            R_old = self.F_old - np.einsum("ij,ij->i", uf_old, m.face_area)
            F += Df * aP_t * R_old
        return F, Df

    # --------------------------------------------------------- pressure solve
    def solve_pressure(self, Fstar, Df, pref=0):
        m = self.m
        a = self.pdiff.a_int * Df
        rows = [m.owner, m.owner, m.neigh, m.neigh]
        cols = [m.owner, m.neigh, m.neigh, m.owner]
        vals = [+a, -a, +a, -a]
        A = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m.nc, m.nc)).tolil()
        # Velocity is Dirichlet everywhere, so pressure is fixed only up to a
        # constant: pin one cell or the matrix is singular.
        A[pref, :] = 0.0
        A[pref, pref] = 1.0
        lu = spla.splu(A.tocsr().tocsc())

        # Discretely, div(F* + a(p_P - p_N)) = 0 is  A p = -div(F*).
        # Solving A p = +div flips the correction and doubles the divergence
        # instead of removing it -- it still runs and still looks like a
        # pressure field, which is why the continuity residual is reported.
        p = np.zeros(m.nc)
        nonorth = np.zeros(len(m.owner))

        # Non-orthogonal correctors: the skewed part of the pressure Laplacian
        # is deferred, so it needs its own inner loop. The loop must converge,
        # or the flux correction and the equation just solved disagree and the
        # continuity residual never reaches zero.
        scale = max(np.abs(Fstar).max(), 1e-300)
        for k in range(self.nNonOrth):
            div = np.zeros(m.nc)
            np.add.at(div, m.owner, Fstar - nonorth)
            np.add.at(div, m.neigh, -(Fstar - nonorth))
            np.add.at(div, m.b_cell, self.Fb)
            div[pref] = 0.0

            p = lu.solve(-div)
            p -= p.mean()

            gp = self.grad_p(p)
            gpf = self.w[:, None] * gp[m.owner] + (1 - self.w)[:, None] * gp[m.neigh]
            new = Df * np.einsum("ij,ij->i", self.pdiff.k_int, gpf)
            delta = np.abs(new - nonorth).max() / scale
            nonorth = new
            if delta < 1e-14:
                break
        self.nonorth_sweeps = k + 1

        F = Fstar - a * (p[m.neigh] - p[m.owner]) - nonorth
        return p, F

    def continuity_error(self, F):
        m = self.m
        div = np.zeros(m.nc)
        np.add.at(div, m.owner, F)
        np.add.at(div, m.neigh, -F)
        np.add.at(div, m.b_cell, self.Fb)
        return np.abs(div).max()

    # ------------------------------------------------------------------ step
    def advance(self, u_b, Fb, src):
        m = self.m
        self.Fb = Fb
        # Time levels shift ONCE per step, not once per outer iteration.
        self.u_old2 = self.u_old.copy()
        self.u_old = self.u.copy()
        self.F_old2 = self.F_old.copy()
        self.F_old = self.F.copy()

        for outer in range(self.nOuter):
            u_prev = self.u.copy()

            A, b = self.assemble_momentum(u_b, src)
            aP = A.diagonal()
            lu = spla.splu(A.tocsc())

            gp = self.grad_p(self.p)
            for d in range(3):
                self.u[:, d] = lu.solve(b[:, d] - gp[:, d] * m.cell_volume)

            for _ in range(self.nCorr):
                hba = self.HbyA(A, b, aP, self.u)
                Fstar, Df = self.rhie_chow(hba, aP)
                self.p, self.F = self.solve_pressure(Fstar, Df)
                gp = self.grad_p(self.p)
                self.u = hba - gp * (m.cell_volume / aP)[:, None]

            self.outer_used = outer + 1
            scale = max(np.abs(self.u).max(), 1e-300)
            if np.abs(self.u - u_prev).max() < self.outerTol * scale:
                break

        self.step_index += 1
        return self.continuity_error(self.F)
