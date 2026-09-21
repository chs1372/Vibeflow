add_library(nsflow_core Parallel.cpp)
target_include_directories(nsflow_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nsflow_core PUBLIC Kokkos::kokkos)

# MPI is optional: without it the solver is serial and the parallel gate is
# skipped, but nothing else changes.
find_package(MPI COMPONENTS CXX)
if(MPI_CXX_FOUND)
  target_link_libraries(nsflow_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(nsflow_core PUBLIC NSFLOW_HAVE_MPI)
  message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
else()
  message(STATUS "MPI not found -- serial build")
endif()
