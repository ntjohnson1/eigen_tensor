set(TTB_TENSORS ${CMAKE_PROJECT_NAME}_tensors)
project(${TTB_TENSORS})

add_library(${TTB_TENSORS} INTERFACE)
target_include_directories(${TTB_TENSORS}
INTERFACE
    ${CMAKE_SOURCE_DIR}/tensors/include/
    ${CMAKE_SOURCE_DIR}/thirdparty/eigen
)
target_link_libraries(${TTB_TENSORS} INTERFACE
    ${CMAKE_PROJECT_NAME}_utils
)
target_compile_features(${TTB_TENSORS} INTERFACE cxx_std_17)

add_executable(${TTB_TENSORS}_tests tests/dense.cpp)
target_link_libraries(${TTB_TENSORS}_tests PRIVATE Catch2::Catch2WithMain ${CMAKE_PROJECT_NAME}_tensors)

catch_discover_tests(${TTB_TENSORS}_tests)
