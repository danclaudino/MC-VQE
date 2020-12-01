cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(MC-VQE LANGUAGES CXX)

set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

option(MCVQE_BUILD_TESTS "Build test programs" OFF)

if (NOT XACC_DIR)
  set(XACC_DIR "$ENV{HOME}/.xacc")
endif()

# Find Xacc
find_package(XACC REQUIRED)

add_subdirectory(examples)
add_subdirectory(src)