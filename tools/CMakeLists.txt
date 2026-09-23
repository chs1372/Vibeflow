if(CGNS_INCLUDE_DIR AND CGNS_LIBRARY)
  add_executable(write_cgns_fixture write_cgns_fixture.cpp)
  target_include_directories(write_cgns_fixture PRIVATE ${CGNS_INCLUDE_DIR})
  target_link_libraries(write_cgns_fixture PRIVATE ${CGNS_LIBRARY})
endif()

add_executable(dump_mesh_vertices dump_mesh_vertices.cpp)
target_link_libraries(dump_mesh_vertices PRIVATE nsflow_mesh)

add_executable(mesh_quality mesh_quality.cpp)
target_link_libraries(mesh_quality PRIVATE nsflow_mesh)
