cmake_minimum_required(VERSION 3.23)
project(nsflow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NSFLOW_ENABLE_GPU   "Build Kokkos with a GPU backend" OFF)
option(NSFLOW_BUILD_TESTS  "Build the verification suite"    ON)

find_package(Kokkos REQUIRED)
find_package(MPI    REQUIRED)
find_package(PETSc  REQUIRED)

# Layer targets. Dependencies here ENFORCE the architecture: a layer may only
# link layers below it. Adding an upward link is a design change, not a build fix.
add_subdirectory(src/core)             # -> Kokkos, MPI
add_subdirectory(src/mesh)             # -> core
add_subdirectory(src/field)            # -> core, mesh
add_subdirectory(src/linalg)           # -> core, mesh, PETSc
add_subdirectory(src/discretization)   # -> core, mesh, field, linalg
add_subdirectory(src/physics)          # -> discretization
add_subdirectory(src/io)               # -> core, mesh, field
add_subdirectory(src/app)              # -> everything

if(NSFLOW_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
