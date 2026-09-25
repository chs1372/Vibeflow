add_library(vibeflow_core Parallel.cpp)
target_include_directories(vibeflow_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(vibeflow_core PUBLIC Kokkos::kokkos)

# MPI is optional: without it the solver is serial and the parallel gate is
# skipped, but nothing else changes.
find_package(MPI COMPONENTS CXX)
if(MPI_CXX_FOUND)
  target_link_libraries(vibeflow_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(vibeflow_core PUBLIC VIBEFLOW_HAVE_MPI)
  message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
else()
  message(STATUS "MPI not found -- serial build")
endif()
