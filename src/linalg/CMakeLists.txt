add_library(nsflow_linalg LinearSystem.cpp NativeCG.cpp)
target_link_libraries(nsflow_linalg PUBLIC nsflow_core nsflow_mesh)
