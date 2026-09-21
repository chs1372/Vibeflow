find_package(PkgConfig QUIET)
add_library(nsflow_mesh Geometry.cpp HexMesh.cpp DistributedMesh.cpp)
target_link_libraries(nsflow_mesh PUBLIC nsflow_core)

# CGNS is optional: without it the solver still runs on generated meshes, it
# just cannot read an external one.
find_path(CGNS_INCLUDE_DIR cgnslib.h)
find_library(CGNS_LIBRARY NAMES cgns)
if(CGNS_INCLUDE_DIR AND CGNS_LIBRARY)
  target_sources(nsflow_mesh PRIVATE CgnsMesh.cpp)
  target_include_directories(nsflow_mesh PUBLIC ${CGNS_INCLUDE_DIR})
  target_link_libraries(nsflow_mesh PUBLIC ${CGNS_LIBRARY})
  target_compile_definitions(nsflow_mesh PUBLIC NSFLOW_HAVE_CGNS)
  message(STATUS "CGNS found: ${CGNS_LIBRARY}")
else()
  message(STATUS "CGNS not found -- external mesh reading disabled")
endif()
