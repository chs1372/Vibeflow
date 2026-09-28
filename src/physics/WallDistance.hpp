#pragma once
// Wall distance (ADR-042): the exact distance from each point to the nearest
// boundary face flagged in wallMask. Each face is split into the four
// triangles of its fan from the vertex average -- the triangles its geometry
// uses -- and the distance to a triangle is the distance to its closest point
// (Ericson, Real-Time Collision Detection, 5.1.5). Brute force over every wall
// triangle of every rank, which each rank gathers: once per mesh.

#include "core/Parallel.hpp"
#include "core/Types.hpp"

namespace vibeflow {

class Mesh;

// points: (n, 3). Returns (n).
ScalarField wallDistance(const Mesh& mesh, const View1<int>& wallMask,
                         const VectorField& points, Index n, const Comm& comm = Comm());

}  // namespace vibeflow
