cmake_minimum_required(VERSION 3.10..3.20)

project(
    EigenTTB
    VERSION 0.0
    DESCRIPTION "An implementation of the MATLAB tensor toolbox with Eigen3 Tensors"
    LANGUAGES CXX)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

add_subdirectory(tensors)
add_subdirectory(utils)
