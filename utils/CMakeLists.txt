set(TTB_UTILS ${CMAKE_PROJECT_NAME}_utils)
project(${TTB_UTILS})

add_library(${TTB_UTILS} INTERFACE)
target_include_directories(${TTB_UTILS}
INTERFACE
    ${CMAKE_SOURCE_DIR}/utils/include/
    ${CMAKE_SOURCE_DIR}/thirdparty/eigen
)
target_compile_features(${TTB_UTILS} INTERFACE cxx_std_17)

add_executable(${TTB_UTILS}_tests tests/khatrirao.cpp)
target_link_libraries(${TTB_UTILS}_tests PRIVATE Catch2::Catch2WithMain ${TTB_UTILS})

catch_discover_tests(${TTB_UTILS}_tests)
