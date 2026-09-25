#pragma once
// Face mass fluxes from a vector potential, and face quadrature.
//
// For verification the prescribed velocity must be divergence-free
// DISCRETELY, not just analytically. Evaluating u at face centres and dotting
// with the area vector leaves an O(h) specific divergence that multiplies phi
// in the convective term -- it corrupts the measured order and looks like a
// bug in the scheme. Stokes' theorem removes it: with u = curl(A),
//
//     F_f = integral over f of (curl A) . dS = contour integral of A . dl
//
// Each edge of a cell is shared by exactly two of its faces, traversed in
// opposite directions, so the fluxes out of a closed cell cancel exactly --
// whatever quadrature is used, as long as both faces use the same points.

#include "core/Types.hpp"
#include <functional>
#include <vector>

namespace vibeflow {

class HexMesh;

using VectorFn = std::function<Vec3(const Vec3&)>;

// Contour integral of A around every face. Sign follows each face's own area
// vector. Fills internal and boundary arrays.
void fluxFromPotential(const HexMesh& mesh, const VectorFn& A,
                       ScalarField& fInternal, ScalarField& fBoundary);

// Integral of u . dS over each boundary face, exact for quadratics.
// A single face-centre evaluation is only exact for linear fields; the
// resulting O(h) mass source is carried into the interior by the elliptic
// pressure, so it costs a full order everywhere, not just at the wall.
void integrateBoundaryFlux(const HexMesh& mesh, const VectorFn& u, ScalarField& fb);

// Area-weighted average of u over each boundary face, exact for quadratics.
// A finite-volume Dirichlet condition constrains the face AVERAGE.
void averageBoundaryValue(const HexMesh& mesh, const VectorFn& u, VectorField& ub);

// Sum of outward fluxes per cell; machine zero for a potential-derived flux.
Real maxDiscreteDivergence(const HexMesh& mesh, const ScalarField& fInternal,
                           const ScalarField& fBoundary);

}  // namespace vibeflow
