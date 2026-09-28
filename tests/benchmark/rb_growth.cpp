// The transient-coupling gate (ADR-040): the Rayleigh-Benard growth rate
// against linear theory.
//
// The onset gate (rayleigh_benard) judges where the growth rate crosses zero,
// which does not depend on how the solver marches. This one judges the rate
// itself, at Ra = 1800, against sigma* = 0.693973025 from a Chebyshev
// solution of the linearised equations (tests/benchmark/rb_linear.py) -- so a
// scheme whose steady states are right and whose dynamics are not fails here.
// The setup is the onset gate's (rb_common.hpp).
//
// Every step's outer loop is iterated to convergence: outerTol 1e-10, a cap of
// 2000 iterations, and a step that reaches the cap fails the gate. Pressure
// solves go to 1e-12. (The cap was 400 when the gate was written; 32 cells
// needs 648 a step, and ADR-040 records the revision.)
//
//   1. Space: 16, 24 and 32 cells across the layer at dt = 0.01. Observed
//      order in [1.5, 2.6]; Richardson extrapolation within 1% of sigma*.
//   2. Time: 16 cells at dt = 0.04, 0.02 and 0.01 against dt = 0.00125. The
//      order of the last pair in [1.8, 2.6].
//   3. Each fit's two halves agree within 1e-3.
//
// Beside each converged rate, the onset gate's four fixed iterations give
// theirs -- reported, not gated: it is the finding this gate keeps in view.
//
// Runs in the balanced buoyancy form (ADR-041); VIBEFLOW_BUOYANCY=cell selects
// ADR-038's cell force.
//
// Run:  rb_growth

#include "rb_common.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

using namespace vibeflow;
using namespace vibeflow::rb;

namespace {

constexpr Real RA = 1800.0;
constexpr Real SIGMA_REF = 0.693973025;
constexpr int OUTER_CAP = 2000;

Marching converged(Real dt) {
  Marching mk;
  mk.dt = dt;
  mk.outer = OUTER_CAP;
  mk.outerTol = 1e-10;
  mk.pressureSolveTol = 1e-12;
  return mk;
}

// The fit settled, and no step reached the outer cap.
bool sound(const Growth& g) {
  bool ok = true;
  if (std::abs(g.sigmaEarly - g.sigmaLate) > 1e-3) {
    std::printf("  FAIL: growth rate not settled over the fit window\n");
    ok = false;
  }
  if (g.maxOuterUsed >= OUTER_CAP) {
    std::printf("  FAIL: a step's outer loop reached the cap of %d\n", OUTER_CAP);
    ok = false;
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  // Exploration only: VIBEFLOW_PROBE_N=<N> marches 20 steps of dt 0.01 on N
  // cells at the gate's settings and prints the most outer iterations a step
  // needed to converge -- the cost the cap has to allow for.
  if (const char* e = std::getenv("VIBEFLOW_PROBE_N")) {
    const Growth g = growthRate(std::atoi(e), RA, converged(0.01), 0.1, 0.2);
    std::printf("probe: N=%s, 20 steps of dt 0.01: at most %d outer iterations a step\n",
                e, g.maxOuterUsed);
    Kokkos::finalize();
    return 0;
  }
  {
    bool ok = true;
    std::printf("Rayleigh-Benard growth rate at Ra = %.0f against linear theory, "
                "sigma* = %.9f\n", RA, SIGMA_REF);
    try {
      // 1. Space.
      const Index grids[3] = {16, 24, 32};
      Real sig[3], hs[3];
      for (int i = 0; i < 3; ++i) {
        const Growth g = growthRate(grids[i], RA, converged(0.01));
        const Growth f = growthRate(grids[i], RA, Marching{});
        sig[i] = g.sigma;
        hs[i] = 1.0 / grids[i];
        std::printf("  N=%-3d dt 0.01    growth rate %.9f  (%+.3f%%)   halves %.9f %.9f   "
                    "outer <= %d   [four fixed iterations: %.6f]\n",
                    grids[i], g.sigma, 100.0 * (g.sigma - SIGMA_REF) / SIGMA_REF,
                    g.sigmaEarly, g.sigmaLate, g.maxOuterUsed, f.sigma);
        std::fflush(stdout);
        ok &= sound(g);
      }
      const Real p = observedOrder(hs, sig);
      const bool orderOk = std::isfinite(p) && p >= 1.5 && p <= 2.6;
      Real ext = std::nan("");
      if (std::isfinite(p)) ext = sig[2] + (sig[2] - sig[1]) / (std::pow(hs[1] / hs[2], p) - 1.0);
      const bool extOk = std::isfinite(ext) && std::abs(ext - SIGMA_REF) / SIGMA_REF <= 0.01;
      std::printf("  -> spatial order %.3f (in [1.5, 2.6]): %s\n", p, orderOk ? "PASS" : "FAIL");
      std::printf("  -> Richardson extrapolation %.6f, %+.3f%% (within 1%%): %s\n", ext,
                  100.0 * (ext - SIGMA_REF) / SIGMA_REF, extOk ? "PASS" : "FAIL");
      ok &= orderOk && extOk;

      // 2. Time, on 16 cells.
      const Growth ref = growthRate(16, RA, converged(0.00125));
      std::printf("  N=16  dt 0.00125 growth rate %.9f   (reference)   outer <= %d\n",
                  ref.sigma, ref.maxOuterUsed);
      ok &= sound(ref);
      const Real dts[3] = {0.04, 0.02, 0.01};
      Real err[3];
      for (int i = 0; i < 3; ++i) {
        const Growth g = growthRate(16, RA, converged(dts[i]));
        err[i] = std::abs(g.sigma - ref.sigma);
        std::printf("  N=16  dt %-7g growth rate %.9f   error %.3e   outer <= %d\n",
                    dts[i], g.sigma, err[i], g.maxOuterUsed);
        std::fflush(stdout);
        ok &= sound(g);
      }
      const Real q = std::log(err[1] / err[2]) / std::log(2.0);
      const bool tOk = std::isfinite(q) && q >= 1.8 && q <= 2.6;
      std::printf("  -> temporal order %.3f then %.3f (last in [1.8, 2.6]): %s\n",
                  std::log(err[0] / err[1]) / std::log(2.0), q, tOk ? "PASS" : "FAIL");
      ok &= tOk;
    } catch (const std::exception& e) {
      std::printf("  FAIL: %s\n", e.what());
      ok = false;
    }
    std::printf("\nv2a transient-coupling GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
