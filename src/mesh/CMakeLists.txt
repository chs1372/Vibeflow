find_package(PkgConfig QUIET)
add_library(vibeflow_mesh Geometry.cpp HexMesh.cpp DistributedMesh.cpp PolyMesh.cpp RawMesh.cpp)
target_link_libraries(vibeflow_mesh PUBLIC vibeflow_core)

# CGNS is optional: without it the solver still runs on generated meshes, it
# just cannot read an external one.
find_path(CGNS_INCLUDE_DIR cgnslib.h)
find_library(CGNS_LIBRARY NAMES cgns)
if(CGNS_INCLUDE_DIR AND CGNS_LIBRARY)
  target_sources(vibeflow_mesh PRIVATE CgnsMesh.cpp)
  target_include_directories(vibeflow_mesh PUBLIC ${CGNS_INCLUDE_DIR})
  target_link_libraries(vibeflow_mesh PUBLIC ${CGNS_LIBRARY})
  target_compile_definitions(vibeflow_mesh PUBLIC VIBEFLOW_HAVE_CGNS)
  message(STATUS "CGNS found: ${CGNS_LIBRARY}")
else()
  message(STATUS "CGNS not found -- external mesh reading disabled")
endif()
