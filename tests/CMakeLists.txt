add_executable(test_geometry unit/test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE vibeflow_mesh)

add_executable(mms_diffusion mms/mms_diffusion.cpp)
target_link_libraries(mms_diffusion PRIVATE vibeflow_discretization)

set(FIXTURES ${CMAKE_SOURCE_DIR}/tests/fixtures)
add_test(NAME geometry       COMMAND test_geometry  ${FIXTURES})
add_test(NAME mms_diffusion  COMMAND mms_diffusion  ${FIXTURES})

add_executable(test_vtu unit/test_vtu.cpp)
target_link_libraries(test_vtu PRIVATE vibeflow_mesh vibeflow_io)
add_test(NAME vtu COMMAND test_vtu ${FIXTURES} ${CMAKE_BINARY_DIR}/vtu_check)

add_executable(test_cgns unit/test_cgns.cpp)
target_link_libraries(test_cgns PRIVATE vibeflow_mesh)
add_test(NAME cgns COMMAND test_cgns ${FIXTURES})

add_executable(mms_parallel mms/mms_parallel.cpp)
target_link_libraries(mms_parallel PRIVATE vibeflow_discretization)
add_test(NAME mms_parallel COMMAND mms_parallel ${FIXTURES})

add_executable(test_backends unit/test_backends.cpp)
target_link_libraries(test_backends PRIVATE vibeflow_discretization)
add_test(NAME backends COMMAND test_backends ${FIXTURES})

add_executable(mms_convection mms/mms_convection.cpp)
target_link_libraries(mms_convection PRIVATE vibeflow_discretization)
add_test(NAME mms_convection COMMAND mms_convection)

add_executable(ethier_steinman mms/ethier_steinman.cpp)
target_link_libraries(ethier_steinman PRIVATE vibeflow_physics)
add_test(NAME ethier_steinman COMMAND ethier_steinman)

add_executable(mms_parallel_mem mms/mms_parallel_mem.cpp)
target_link_libraries(mms_parallel_mem PRIVATE vibeflow_mesh)
add_test(NAME mms_parallel_mem COMMAND mms_parallel_mem)

add_executable(mms_parallel_ns mms/mms_parallel_ns.cpp)
target_link_libraries(mms_parallel_ns PRIVATE vibeflow_physics)
add_test(NAME mms_parallel_ns COMMAND mms_parallel_ns)

add_executable(open_domain mms/open_domain.cpp)
target_link_libraries(open_domain PRIVATE vibeflow_physics)
add_test(NAME open_domain COMMAND open_domain 16)

add_executable(cavity benchmark/cavity.cpp)
target_link_libraries(cavity PRIVATE vibeflow_physics)
add_test(NAME cavity COMMAND cavity 64)

add_executable(cylinder benchmark/cylinder.cpp)
target_link_libraries(cylinder PRIVATE vibeflow_physics vibeflow_io)

add_executable(mms_parallel_read mms/mms_parallel_read.cpp)
target_link_libraries(mms_parallel_read PRIVATE vibeflow_discretization)
add_test(NAME mms_parallel_read COMMAND mms_parallel_read ${FIXTURES} 16 read)
