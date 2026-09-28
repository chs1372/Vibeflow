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

# ADR-010 / ADR-037: a steady state must not depend on the time step.
add_executable(steady_dt mms/steady_dt.cpp)
target_link_libraries(steady_dt PRIVATE vibeflow_physics)
add_test(NAME steady_dt COMMAND steady_dt ${CMAKE_SOURCE_DIR}/tests/fixtures)

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

# v2a (ADR-038, ADR-039): the energy equation, Boussinesq buoyancy, slip walls.
add_executable(heat_transfer mms/heat_transfer.cpp)
target_link_libraries(heat_transfer PRIVATE vibeflow_physics)
add_test(NAME heat_transfer COMMAND heat_transfer ${FIXTURES})

add_executable(taylor_green mms/taylor_green.cpp)
target_link_libraries(taylor_green PRIVATE vibeflow_physics)
add_test(NAME taylor_green COMMAND taylor_green)

add_executable(mms_parallel_heat mms/mms_parallel_heat.cpp)
target_link_libraries(mms_parallel_heat PRIVATE vibeflow_physics)
add_test(NAME mms_parallel_heat COMMAND mms_parallel_heat)

add_executable(rayleigh_benard benchmark/rayleigh_benard.cpp)
target_link_libraries(rayleigh_benard PRIVATE vibeflow_physics)
add_test(NAME rayleigh_benard COMMAND rayleigh_benard)

# ADR-040: the growth rate itself, against linear theory.
add_executable(rb_growth benchmark/rb_growth.cpp)
target_link_libraries(rb_growth PRIVATE vibeflow_physics)
add_test(NAME rb_growth COMMAND rb_growth)
# v2b: the k-omega SST model (ADR-042).
add_executable(sst_mms mms/sst_mms.cpp)
target_link_libraries(sst_mms PRIVATE vibeflow_physics)
add_test(NAME sst_mms COMMAND sst_mms ${FIXTURES})

# Hours on two cores at 128^2 and Ra = 1e6: a suite gate, not a ctest.
add_executable(heated_cavity benchmark/heated_cavity.cpp)
target_link_libraries(heated_cavity PRIVATE vibeflow_physics)
