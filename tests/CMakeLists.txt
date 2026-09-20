add_executable(test_geometry unit/test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE nsflow_mesh)

add_executable(mms_diffusion mms/mms_diffusion.cpp)
target_link_libraries(mms_diffusion PRIVATE nsflow_discretization)

set(FIXTURES ${CMAKE_SOURCE_DIR}/tests/fixtures)
add_test(NAME geometry       COMMAND test_geometry  ${FIXTURES})
add_test(NAME mms_diffusion  COMMAND mms_diffusion  ${FIXTURES})
