add_library(plascan_placamera_runtime STATIC
    ProjectCameraStore.cpp
    ProjectCameraStore.h
)
add_library(plascan::placamera_runtime ALIAS plascan_placamera_runtime)

target_compile_features(plascan_placamera_runtime PUBLIC cxx_std_20)
target_include_directories(plascan_placamera_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/src/core
)
target_link_libraries(plascan_placamera_runtime PUBLIC
    placamera::placamera
    placamera::state
    Qt6::Core
)

if(BUILD_TESTS)
    include(GoogleTest)
    add_executable(test_placamera_project_store
        tests/test_project_camera_store.cpp
    )
    target_link_libraries(test_placamera_project_store PRIVATE
        plascan_placamera_runtime
        GTest::gtest_main
    )
    gtest_discover_tests(test_placamera_project_store TEST_PREFIX "PlaCameraProjectStore.")
endif()
