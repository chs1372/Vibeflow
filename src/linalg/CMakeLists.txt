add_library(nsflow_linalg LinearSystem.cpp NativeCG.cpp)
target_link_libraries(nsflow_linalg PUBLIC nsflow_core nsflow_mesh)

# PETSc is optional. Without it NativeCG is the only backend -- fine for the
# gates, not for production, where BoomerAMG is what makes the pressure solve
# scale.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PETSC IMPORTED_TARGET PETSc)
endif()
if(PETSC_FOUND)
  target_sources(nsflow_linalg PRIVATE PetscSolver.cpp)
  target_link_libraries(nsflow_linalg PUBLIC PkgConfig::PETSC)
  target_compile_definitions(nsflow_linalg PUBLIC NSFLOW_HAVE_PETSC)
  message(STATUS "PETSc found: ${PETSC_VERSION}")
else()
  message(STATUS "PETSc not found -- only the native CG backend is available")
endif()
