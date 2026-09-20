cmake_minimum_required(VERSION 3.23)
project(nsflow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NSFLOW_BUILD_TESTS "Build the verification suite" ON)
option(NSFLOW_WITH_PETSC  "Build the PETSc linear-solver backend" OFF)

find_package(Kokkos REQUIRED)
if(NSFLOW_WITH_PETSC)
  find_package(PETSc REQUIRED)
endif()

# Layer targets. These dependencies ENFORCE the architecture: a layer may only
# link layers below it. Adding an upward link is a design change, not a build fix.
add_subdirectory(src/core)             # -> Kokkos
add_subdirectory(src/mesh)             # -> core
add_subdirectory(src/linalg)           # -> core, mesh  (+ PETSc when enabled)
add_subdirectory(src/discretization)   # -> core, mesh, linalg
# add_subdirectory(src/field)          # v1
# add_subdirectory(src/physics)        # v1
# add_subdirectory(src/io)             # v0 remaining: CGNS
# add_subdirectory(src/app)            # v1

if(NSFLOW_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
