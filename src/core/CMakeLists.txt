add_library(nsflow_core INTERFACE)
target_include_directories(nsflow_core INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nsflow_core INTERFACE Kokkos::kokkos)
