#!/bin/sh
# The two cylinder meshes the v1 gates run on. They are generated, not stored.
#
# With gmsh 4.15.2 both come out byte for byte identical to the meshes behind
# the numbers in docs/DECISIONS.md:
#
#   debug.hex      6,763 cells, first cell D/17   md5 0c23099ccd64709811b7c97dbd877d5d
#   cylinder.hex  21,811 cells, first cell D/33   md5 08d942dd9a6e38ea7600c13b8b532e1f
#
# Another gmsh version may place nodes differently. That gives a different but
# equally valid mesh, and the gates judge the flow, not the file, so a
# checksum mismatch alone is not a failure.
set -e
cd "$(dirname "$0")"
CYL_SIZEMIN=0.06 CYL_SIZEMAX=1.0 python3 make_mesh.py debug.hex
python3 make_mesh.py cylinder.hex
