add_library(nsflow_io VtuWriter.cpp)
target_link_libraries(nsflow_io PUBLIC nsflow_core nsflow_mesh)
