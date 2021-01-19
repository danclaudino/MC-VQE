configure_file(mc_vqe_example.in.cpp ${CMAKE_BINARY_DIR}/examples/mc_vqe_example.cpp)
add_executable(mc_vqe_example ${CMAKE_BINARY_DIR}/examples/mc_vqe_example.cpp)
target_link_libraries(mc_vqe_example PRIVATE xacc::xacc)