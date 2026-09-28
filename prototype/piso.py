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
   The old-time flux term below cancels the transient part of aP -- but only
   exactly if the residual it carries is built with the same interpolation as
   the predicted flux, and the damping coefficient is V_f/aP_f (ADR-037).
   The first version subtracted a plain linear interpolation of the old
   velocity from a skew-corrected prediction, and the difference, divided by
   a factor that falls like dt, moved the steady state by 69% of its
   discretisation error across dt = 0.02 .. 2.0. gate_dt_independence
   measures exactly that, and now demands 1e-6.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fvm import DiffusionOperator
from gradient import QuadraticLSQGradient


class PisoSolver:
    def __init__(self, mesh, nu, dt, n_correctors=2, n_nonorth=40,
                 n_outer=1, outer_tol=1e-10, consistent_rhie_chow=True,
                 gradient="linear", rhie_chow_form="standard", old_flux="exact",
                 kappa=None, beta_g=None, t_ref=0.0, t_ref_grad=None,
                 buoyancy_form="cell", turbulence=None):
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
        # How the old-flux term and the damping coefficient are built (ADR-037).
        # "exact": the steady equations contain no dt at all --
        #   * the old residual R = F - I[u].S uses the SAME skew-corrected
        #     interpolation I as the predicted flux,
        #   * the predicted flux interpolates q = H/aP - (V/aP) grad p and adds
        #     D_f L[grad p].S back, so the product (V/aP) grad p is never
        #     interpolated as a product,
        #   * D_f = V_f / aP_f, so 1/D_f - a0 = aPs_f / V_f exactly.
        #   A steady state then has F = I[u].S + (V_f/aPs_f)(L[grad p].Delta -
        #   a_f (p_N - p_P)), with no dt in it anywhere.
        # "v1": R = F - L[u].S with plain linear L, D_f = interp(V/aP). On a
        #   skewed mesh the steady residual carries (I - L)[u].S divided by
        #   1 - D_f a0 ~ (2/3) Co, which grows as dt shrinks: the steady velocity
        #   moved by 69% of its discretisation error between dt = 0.02 and 2.0,
        #   and the cylinder's far wake grew a spurious velocity at dt = 0.025
        #   (ADR-031). Kept so the gate can be shown to fail on it.
        if old_flux not in ("exact", "v1"):
            raise ValueError(old_flux)
        self.old_flux = old_flux
        self._R_old = None
        self._R_old_step = -1
        # "standard": F* = (H/aP)_f . S, the only pressure term in the flux
        # being the compact face gradient applied by the pressure solve.
        # "interpolated": F* also carries D (grad(p)_f . S - snGrad p_old).
        # This file and the C++ solver were both written in the interpolated
        # form, and it is WRONG for a solver that re-solves the full pressure
        # each corrector: the checkerboard part of the pressure equation then
        # reads p_new = -p_old + forcing, an eigenvalue near -1, so any
        # decoupled mode flips sign on every solve and grows once the velocity
        # coupling tips it past one. ADR-026. Kept only to demonstrate it.
        if rhie_chow_form not in ("standard", "interpolated"):
            raise ValueError(rhie_chow_form)
        self.rhie_chow_form = rhie_chow_form

        m = mesh
        self.diff = DiffusionOperator(mesh, nu)      # momentum viscous term
        self.pdiff = DiffusionOperator(mesh, 1.0)    # pressure Laplacian geometry
        self.grad = self.diff.grad                    # weighted linear LSQ
        # A quadratic fit IS a second-order gradient operator (1.99 against
        # the analytic gradient, where linear LSQ is 1.05 and Green-Gauss
        # does not converge) but it does NOT make the solver more accurate,
        # and it destabilises the corrector. Kept selectable so the
        # measurement can be reproduced; see ADR-012.
        self.gradient_kind = gradient
        self.gradq = QuadraticLSQGradient(mesh) if gradient == "quadratic" else None

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

        # Energy equation and Boussinesq buoyancy (ADR-038). Off unless kappa
        # is given. Temperature is transported like a velocity component --
        # BDF2, upwind plus the deferred correction to the skew-corrected face
        # value, diffusion with the non-orthogonal correction -- and solved
        # inside every outer iteration after the pressure correctors, so a
        # converged outer loop carries no coupling lag. The body force per
        # unit mass is f = -(T - T_ref(x)) beta_g, with beta_g = beta * g and
        # T_ref(x) = t_ref + t_ref_grad . x a reference stratification along
        # g: its buoyancy is a gradient, absorbed into the pressure, so a fluid
        # resting in exactly that stratification is an exact fixed point.
        self.energy = kappa is not None
        self.kappa = kappa
        self.beta_g = np.zeros(3) if beta_g is None else np.asarray(beta_g, float)
        self.t_ref = t_ref
        self.t_ref_grad = np.zeros(3) if t_ref_grad is None else np.asarray(t_ref_grad, float)
        self.T = np.zeros(m.nc)
        self.T_old = np.zeros(m.nc)
        self.T_old2 = np.zeros(m.nc)

        # A turbulence model (ADR-042): an sst.SstModel, solved once per
        # outer iteration after the pressure correctors (and the energy
        # equation). Its eddy viscosity enters the momentum equation through
        # the face viscosity and the explicit part div(nu_t grad(u)^T) of the
        # stress. None: every operation is v2a's.
        self.turb = turbulence
        self._turb_ready = False

        # How the buoyancy enters (ADR-041).
        # "cell": ADR-038's cell force f = -(T - T_ref) beta_g in the momentum
        #   source. A resting fluid in a stratification T_ref does not match
        #   drifts: its quadratic pressure is carried by the least-squares
        #   gradient and the boundary extrapolation, neither exact for it.
        # "balanced": the force enters through the faces. Each outer iteration
        #   a hydrostatic pressure p_h absorbs the gradient part of the face
        #   force B_f = f(T_f).S_f -- the residual r_f = B_f - [a_f dp_h +
        #   k_f . grad p_h] is made divergence-free, with r = 0 on boundary
        #   faces -- and the cell force is reconstructed from r alone. On a
        #   layered mesh a resting fluid in any T(z) then has r = 0 exactly.
        if buoyancy_form not in ("cell", "balanced"):
            raise ValueError(buoyancy_form)
        self.buoyancy_form = buoyancy_form
        self.p_h = np.zeros(m.nc)
        self._r_face = np.zeros(len(m.owner))
        self._g_cell = np.zeros((m.nc, 3))
        if buoyancy_form == "balanced":
            S, Sb = m.face_area, m.b_area
            Sa, Sba = np.linalg.norm(S, axis=1), np.linalg.norm(Sb, axis=1)
            M = np.zeros((m.nc, 3, 3))
            outer = np.einsum("ij,ik->ijk", S, S) / Sa[:, None, None]
            np.add.at(M, m.owner, outer)
            np.add.at(M, m.neigh, outer)
            np.add.at(M, m.b_cell, np.einsum("ij,ik->ijk", Sb, Sb) / Sba[:, None, None])
            self._recon_Minv = np.linalg.inv(M)
            self._recon_w = S / Sa[:, None]
            a = self.pdiff.a_int
            A = sp.coo_matrix((np.concatenate([a, -a, a, -a]),
                               (np.concatenate([m.owner, m.owner, m.neigh, m.neigh]),
                                np.concatenate([m.owner, m.neigh, m.neigh, m.owner]))),
                              shape=(m.nc, m.nc)).tolil()
            A[0, :] = 0.0                     # Neumann: pin one cell
            A[0, 0] = 1.0
            self._ph_lu = spla.splu(A.tocsc())

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
        # Extrapolation, not interpolation: the quadratic fit is MORE
        # accurate inside its stencil and LESS accurate outside it, because
        # the quadratic terms grow fastest where there is no data to
        # constrain them. Measured on the skewed mesh, using the quadratic
        # gradient here dropped the velocity order from 1.44 to 1.29.
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
        # Reconstruction error against the analytic gradient of the exact
        # pressure, on a distorted mesh (ADR-012):
        #
        #   n    Green-Gauss  order   LSQ linear  order   LSQ quadratic  order
        #   6       1.64e-01      -     1.19e-01      -        1.78e-01      -
        #  12       1.47e-01   0.16     5.49e-02   1.11        4.86e-02   1.87
        #  24       1.62e-01  -0.14     2.66e-02   1.05        1.22e-02   1.99
        #
        # The quadratic fit is the only second-order operator of the three,
        # and it still does not help the SOLVER. Velocity order on the smooth
        # mesh family: 1.816 with linear, 1.752 with quadratic. At nu = 1 the
        # quadratic version diverges outright, because a two-ring gradient in
        # the velocity correction is inconsistent with the compact pressure
        # Laplacian that produced the correction. Linear is the default.
        if self.gradq is not None:
            return self.gradq(p, self.p_boundary(p))
        return self.grad(p, self.p_boundary(p))

    def face_interp(self, q, grad_q):
        """Linear interpolation with skewness correction, componentwise."""
        m = self.m
        lin = self.w[:, None] * q[m.owner] + (1 - self.w)[:, None] * q[m.neigh]
        gf = (self.w[:, None, None] * grad_q[m.owner]
              + (1 - self.w)[:, None, None] * grad_q[m.neigh])
        return lin + np.einsum("ijk,ik->ij", gf, self.skew_vec)

    # -------------------------------------------------------------- momentum
    def slip_values(self, u_b, slip):
        """Boundary velocity with slip faces replaced by the tangential part
        of the adjacent cell's velocity (ADR-039)."""
        if slip is None or not slip.any():
            return u_b
        m = self.m
        n = m.b_area / np.linalg.norm(m.b_area, axis=1)[:, None]
        uc = self.u[m.b_cell]
        tang = uc - np.einsum("ij,ij->i", uc, n)[:, None] * n
        return np.where(slip[:, None], tang, u_b)

    def assemble_momentum(self, u_b, src, slip=None):
        """Matrix and pressure-free right-hand sides for the three components.

        A slip face (ADR-039) is treated as a Dirichlet face whose value is the
        tangential part of the cell velocity, taken from the latest iterate:
        the implicit diagonal nu*a_b balances the lagged cell value, so at
        convergence the viscous flux through the face is -nu a_b (u.n) n --
        the normal component driven to zero, the tangential stress zero. Its
        non-orthogonal correction acts on the normal component only.
        """
        m = self.m
        aP_t, a1, a2 = self.bdf()
        Fp = np.maximum(self.F, 0.0)
        Fn = np.maximum(-self.F, 0.0)
        a = self.diff.a_int
        nu = self.nu            # the face viscosity; nub on boundary faces
        nub = self.nu
        if self.turb is not None:
            nt = self.turb.nut
            nt_f = self.w * nt[m.owner] + (1 - self.w) * nt[m.neigh]
            nu = self.nu + nt_f
            nub = self.nu + self.turb.nut_b

        rows = [m.owner, m.owner, m.neigh, m.neigh, m.b_cell,
                m.owner, m.owner, m.neigh, m.neigh, np.arange(m.nc)]
        cols = [m.owner, m.neigh, m.neigh, m.owner, m.b_cell,
                m.owner, m.neigh, m.neigh, m.owner, np.arange(m.nc)]
        vals = [+nu * a, -nu * a, +nu * a, -nu * a, +nub * self.diff.a_bnd,
                Fp, -Fn, Fn, -Fp, aP_t * m.cell_volume]
        A = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m.nc, m.nc)).tocsr()

        grads = np.stack([self.grad(self.u[:, d], u_b[:, d]) for d in range(3)], axis=1)
        ho = self.face_interp(self.u, grads)
        if self.turb is not None:
            # div(nu_t grad(u)^T): component i of the face flux is
            # nu_t,f sum_j (du_j/dx_i)_f S_j, explicit from the latest u.
            gfu = (self.w[:, None, None] * grads[m.owner]
                   + (1 - self.w)[:, None, None] * grads[m.neigh])
            tflux = nt_f[:, None] * np.einsum("fji,fj->fi", gfu, m.face_area)
            tflux_b = self.turb.nut_b[:, None] * np.einsum("fji,fj->fi", grads[m.b_cell],
                                                           m.b_area)
            if slip is not None:
                # A slip face carries no tangential stress, so none there.
                tflux_b[slip] = 0.0

        b = np.empty((m.nc, 3))
        for d in range(3):
            g = grads[:, d, :]
            rhs = src[:, d] * m.cell_volume
            rhs -= (a1 * self.u_old[:, d] + a2 * self.u_old2[:, d]) * m.cell_volume

            np.add.at(rhs, m.b_cell, nub * self.diff.a_bnd * u_b[:, d])
            gf = (self.diff.w_owner[:, None] * g[m.owner]
                  + (1 - self.diff.w_owner)[:, None] * g[m.neigh])
            corr = nu * np.einsum("ij,ij->i", self.diff.k_int, gf)
            np.add.at(rhs, m.owner, corr)
            np.add.at(rhs, m.neigh, -corr)
            nonorth_b = nub * np.einsum("ij,ij->i", self.diff.k_bnd, g[m.b_cell])
            if slip is not None and slip.any():
                n_hat = m.b_area / np.linalg.norm(m.b_area, axis=1)[:, None]
                g_n = np.einsum("fe,fek->fk", n_hat, grads[m.b_cell])   # grad(u.n)
                slip_corr = nub * n_hat[:, d] * np.einsum("ij,ij->i", self.diff.k_bnd, g_n)
                nonorth_b = np.where(slip, slip_corr, nonorth_b)
            np.add.at(rhs, m.b_cell, nonorth_b)
            if self.turb is not None:
                np.add.at(rhs, m.owner, tflux[:, d])
                np.add.at(rhs, m.neigh, -tflux[:, d])
                np.add.at(rhs, m.b_cell, tflux_b[:, d])

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
    def _old_residual(self):
        """R = F_old - I[u_old].S with the skew-corrected interpolation, once
        per step (ADR-037). The gradient of u_old takes the cell values as its
        boundary values, as the gradient of H/aP does, so that I is the same
        linear operator on both."""
        if self._R_old_step != self.step_index:
            m = self.m
            gu = np.stack([self.grad(self.u_old[:, d], self.u_old[m.b_cell, d])
                           for d in range(3)], axis=1)
            uf = self.face_interp(self.u_old, gu)
            self._R_old = self.F_old - np.einsum("ij,ij->i", uf, m.face_area)
            self._R_old_step = self.step_index
        return self._R_old

    def rhie_chow(self, HbyA, aP, gp=None):
        m = self.m
        V = m.cell_volume
        exact = self.old_flux == "exact" and self.rhie_chow_form == "standard"
        if exact:
            # Volume and aP interpolated separately, so the transient part
            # cancels exactly: 1/D_f - a0 = aPs_f / V_f.
            Vf = self.w * V[m.owner] + (1 - self.w) * V[m.neigh]
            aPf = self.w * aP[m.owner] + (1 - self.w) * aP[m.neigh]
            Df = Vf / aPf
        else:
            Df = (self.w * (V / aP)[m.owner]
                  + (1 - self.w) * (V / aP)[m.neigh])

        if exact:
            # q is the velocity the last pressure implies. Interpolating it,
            # rather than H/aP, keeps the product (V/aP) grad p out of the
            # interpolation; the pressure comes back with the face coefficient.
            # Both pressure terms are interpolated cell gradients, blind to a
            # checkerboard -- not the compact old-pressure term of ADR-026.
            if gp is None:
                gp = self.grad_p(self.p)
            q = HbyA - gp * (V / aP)[:, None]
            gq = np.stack([self.grad(q[:, d], q[m.b_cell, d]) for d in range(3)],
                          axis=1)
            F = np.einsum("ij,ij->i", self.face_interp(q, gq), m.face_area)
            gpf = self.w[:, None] * gp[m.owner] + (1 - self.w)[:, None] * gp[m.neigh]
            F += Df * np.einsum("ij,ij->i", gpf, m.face_area)
        else:
            # H/aP must be interpolated to the face WITH the skewness correction.
            # Plain linear interpolation is first order once the face centre is off
            # the line joining the cell centres, and since this sets the mass flux
            # it drags the whole solution down with it: the scheme measures second
            # order on an orthogonal mesh and about first order on a skewed one.
            gH = np.stack([self.grad(HbyA[:, d], HbyA[m.b_cell, d]) for d in range(3)],
                          axis=1)
            Hf = self.face_interp(HbyA, gH)
            F = np.einsum("ij,ij->i", Hf, m.face_area)

            if self.rhie_chow_form == "interpolated":
                gp = self.grad_p(self.p)
                gpf = self.w[:, None] * gp[m.owner] + (1 - self.w)[:, None] * gp[m.neigh]
                snGrad = self.pdiff.a_int * (self.p[m.neigh] - self.p[m.owner])
                F += Df * (np.einsum("ij,ij->i", gpf, m.face_area) - snGrad)

        if self.energy and self.buoyancy_form == "balanced":
            # The compact face residual in place of the interpolated cell
            # force, as the pressure's compact gradient replaces its
            # interpolated one (ADR-041).
            gf = (self.w[:, None] * self._g_cell[m.owner]
                  + (1 - self.w)[:, None] * self._g_cell[m.neigh])
            F += Df * (self._r_face - np.einsum("ij,ij->i", gf, m.face_area))

        if self.consistent:
            # Old-flux term (Choi 1999). R = F - u_f . S is the Rhie-Chow
            # residual; carrying it forward with the coefficient Df * aP_t
            # cancels the transient part of aP, so the pressure damping -- and
            # therefore the steady state -- does not depend on dt. That holds
            # exactly only in the "exact" form above (ADR-037).
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
            if exact:
                R_old = self._old_residual()
            else:
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
    # --------------------------------------------------------- temperature
    def buoyancy(self):
        """Body force per unit mass, -(T - T_ref(x)) beta g, per cell."""
        t_ref = self.t_ref + self.m.cell_centre @ self.t_ref_grad
        return -(self.T - t_ref)[:, None] * self.beta_g[None, :]

    def balanced_buoyancy(self, tb):
        """Cell force and face residual of the balanced form (ADR-041).

        tb: the boundary temperatures the energy equation's gradient uses.
        """
        m = self.m
        pd = self.pdiff
        gT = self.grad(self.T, tb)
        Tf = self.face_interp(self.T[:, None], gT[:, None, :])[:, 0]
        B = np.einsum("ij,ij->i",
                      -(Tf - (self.t_ref + m.face_centre @ self.t_ref_grad))[:, None]
                      * self.beta_g[None, :], m.face_area)
        Bb = np.einsum("ij,ij->i",
                       -(tb - (self.t_ref + m.b_centre @ self.t_ref_grad))[:, None]
                       * self.beta_g[None, :], m.b_area)

        def grad_ph(ph):
            # Boundary values that leave no residual on a boundary face:
            # a_b (p_b - p_P) + k_b . grad p_P = B_b, iterated with the gradient.
            v = ph[m.b_cell] + Bb / pd.a_bnd
            for _ in range(3):
                g = self.grad(ph, v)
                v = ph[m.b_cell] + (Bb - np.einsum("ij,ij->i", pd.k_bnd, g[m.b_cell])) / pd.a_bnd
            return self.grad(ph, v)

        ph = self.p_h
        kg = np.zeros(len(m.owner))
        scale = max(np.abs(B).max(), np.abs(Bb).max(), 1e-300)
        for _ in range(self.nNonOrth):
            c = B - kg
            div = np.zeros(m.nc)
            np.add.at(div, m.owner, c)
            np.add.at(div, m.neigh, -c)
            div[0] = 0.0
            ph = self._ph_lu.solve(-div)
            g = grad_ph(ph)
            gfo = (pd.w_owner[:, None] * g[m.owner]
                   + (1.0 - pd.w_owner)[:, None] * g[m.neigh])
            kg_new = np.einsum("ij,ij->i", pd.k_int, gfo)
            delta = np.abs(kg_new - kg).max() / scale
            kg = kg_new
            if delta < 1e-14:
                break
        self.p_h = ph - ph.mean()
        r = B - pd.a_int * (ph[m.neigh] - ph[m.owner]) - kg
        rhs = np.zeros((m.nc, 3))
        np.add.at(rhs, m.owner, self._recon_w * r[:, None])
        np.add.at(rhs, m.neigh, self._recon_w * r[:, None])
        g_cell = np.einsum("ijk,ik->ij", self._recon_Minv, rhs)
        return g_cell, r

    def solve_energy(self, T_b, T_src, T_bc):
        """One implicit solve for T with the current face flux (ADR-038).

        T_bc per boundary face: 0 = fixed temperature T_b, 1 = fixed heat
        flux T_b (kappa dT/dn into the domain, per unit area; 0 is adiabatic).
        """
        m = self.m
        k = self.kappa
        aP_t, a1, a2 = self.bdf()
        Fp = np.maximum(self.F, 0.0)
        Fn = np.maximum(-self.F, 0.0)
        a = self.diff.a_int
        dirichlet = T_bc == 0
        area_b = np.linalg.norm(m.b_area, axis=1)

        rows = [m.owner, m.owner, m.neigh, m.neigh, m.b_cell,
                m.owner, m.owner, m.neigh, m.neigh, np.arange(m.nc)]
        cols = [m.owner, m.neigh, m.neigh, m.owner, m.b_cell,
                m.owner, m.neigh, m.neigh, m.owner, np.arange(m.nc)]
        vals = [+k * a, -k * a, +k * a, -k * a,
                np.where(dirichlet, k * self.diff.a_bnd, 0.0),
                Fp, -Fn, Fn, -Fp, aP_t * m.cell_volume]
        # Outflow through a fixed-flux face carries the cell's own value.
        out_b = np.where(dirichlet, 0.0, np.maximum(self.Fb, 0.0))
        rows.append(m.b_cell); cols.append(m.b_cell); vals.append(out_b)
        A = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m.nc, m.nc)).tocsr()

        # Boundary values for the gradient: the prescribed temperature, or
        # the cell value on a fixed-flux face.
        tb = np.where(dirichlet, T_b, self.T[m.b_cell])
        g = self.grad(self.T, tb)
        ho = self.face_interp(self.T[:, None], g[:, None, :])[:, 0]

        rhs = T_src * m.cell_volume - (a1 * self.T_old + a2 * self.T_old2) * m.cell_volume
        np.add.at(rhs, m.b_cell, np.where(dirichlet, k * self.diff.a_bnd * T_b,
                                          T_b * area_b))
        gf = (self.diff.w_owner[:, None] * g[m.owner]
              + (1 - self.diff.w_owner)[:, None] * g[m.neigh])
        corr = k * np.einsum("ij,ij->i", self.diff.k_int, gf)
        np.add.at(rhs, m.owner, corr)
        np.add.at(rhs, m.neigh, -corr)
        np.add.at(rhs, m.b_cell, np.where(
            dirichlet, k * np.einsum("ij,ij->i", self.diff.k_bnd, g[m.b_cell]), 0.0))
        ud = np.where(self.F > 0.0, self.T[m.owner], self.T[m.neigh])
        dc = self.F * (ho - ud)
        np.add.at(rhs, m.owner, -dc)
        np.add.at(rhs, m.neigh, +dc)
        # Inflow through a fixed-flux face, and every fixed-temperature face,
        # carry the boundary value; outflow on a fixed-flux face is implicit.
        conv_b = np.where(dirichlet, self.Fb * T_b, np.minimum(self.Fb, 0.0) * tb)
        np.add.at(rhs, m.b_cell, -conv_b)
        return spla.spsolve(A.tocsc(), rhs)

    def advance(self, u_b, Fb, src, T_b=None, T_src=None, T_bc=None, u_bc=None):
        m = self.m
        self.Fb = Fb
        # Velocity boundary codes per face, as in the C++ solver: 0 Dirichlet,
        # 2 slip (ADR-039). Zero gradient (1) exists only on the C++ side.
        slip = None
        if u_bc is not None:
            u_bc = np.asarray(u_bc)
            if np.any(u_bc == 1):
                raise NotImplementedError("zero-gradient velocity faces")
            slip = u_bc == 2
        u_b_given = u_b
        # Time levels shift ONCE per step, not once per outer iteration.
        self.u_old2 = self.u_old.copy()
        self.u_old = self.u.copy()
        self.F_old2 = self.F_old.copy()
        self.F_old = self.F.copy()
        if self.turb is not None:
            self.turb.shift()
        if self.energy:
            self.T_old2 = self.T_old.copy()
            self.T_old = self.T.copy()
            nb = len(m.b_cell)
            T_b = np.zeros(nb) if T_b is None else np.asarray(T_b, float)
            T_src = np.zeros(m.nc) if T_src is None else np.asarray(T_src, float)
            T_bc = np.zeros(nb, dtype=int) if T_bc is None else np.asarray(T_bc)

        for outer in range(self.nOuter):
            u_prev = self.u.copy()
            if self.energy:
                T_prev = self.T.copy()
                if self.buoyancy_form == "balanced":
                    tb = np.where(T_bc == 0, T_b, self.T[m.b_cell])
                    self._g_cell, self._r_face = self.balanced_buoyancy(tb)
                    src_total = src + self._g_cell
                else:
                    src_total = src + self.buoyancy()
            else:
                src_total = src

            u_b = self.slip_values(u_b_given, slip)
            if self.turb is not None and not self._turb_ready:
                self.turb.update_nut(self.u, u_b)      # nu_t of the initial state
                self._turb_ready = True
            if self.turb is not None:
                k_prev, om_prev = self.turb.k.copy(), self.turb.w.copy()
            A, b = self.assemble_momentum(u_b, src_total, slip)
            aP = A.diagonal()
            lu = spla.splu(A.tocsc())

            gp = self.grad_p(self.p)
            for d in range(3):
                self.u[:, d] = lu.solve(b[:, d] - gp[:, d] * m.cell_volume)

            for _ in range(self.nCorr):
                hba = self.HbyA(A, b, aP, self.u)
                Fstar, Df = self.rhie_chow(hba, aP, gp)
                self.p, self.F = self.solve_pressure(Fstar, Df)
                gp = self.grad_p(self.p)
                self.u = hba - gp * (m.cell_volume / aP)[:, None]

            if self.energy:
                self.T = self.solve_energy(T_b, T_src, T_bc)
            if self.turb is not None:
                self.turb.solve(self.F, self.Fb, self.u, u_b, self.bdf())

            self.outer_used = outer + 1
            scale = max(np.abs(self.u).max(), 1e-300)
            change = np.abs(self.u - u_prev).max() / scale
            if self.energy:
                change = max(change, np.abs(self.T - T_prev).max()
                             / max(np.abs(self.T).max(), 1e-300))
            if self.turb is not None:
                for new_, old_ in ((self.turb.k, k_prev), (self.turb.w, om_prev)):
                    change = max(change, np.abs(new_ - old_).max()
                                 / max(np.abs(new_).max(), 1e-300))
            if change < self.outerTol:
                break

        self.step_index += 1
        return self.continuity_error(self.F)
