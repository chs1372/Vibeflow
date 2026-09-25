cmake_minimum_required(VERSION 3.23)
project(Vibeflow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VIBEFLOW_BUILD_TESTS "Build the verification suite" ON)

# Kokkos is the only hard dependency. MPI, CGNS and PETSc are optional and are
# detected where they are used (src/core, src/mesh, src/linalg): each one found
# turns on its code path and its gate, and each one missing only skips them.
find_package(Kokkos REQUIRED)

# Layer targets. These dependencies ENFORCE the architecture: a layer may only
# link layers below it. Adding an upward link is a design change, not a build fix.
add_subdirectory(src/core)             # -> Kokkos      (+ MPI when found)
add_subdirectory(src/mesh)             # -> core        (+ CGNS when found)
add_subdirectory(src/linalg)           # -> core, mesh  (+ PETSc when found)
add_subdirectory(src/discretization)   # -> core, mesh, linalg
# add_subdirectory(src/field)          # v1
add_subdirectory(src/physics)          # -> discretization, linalg
add_subdirectory(src/io)               # -> core, mesh
# add_subdirectory(src/app)            # v1

add_subdirectory(tools)

if(VIBEFLOW_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
