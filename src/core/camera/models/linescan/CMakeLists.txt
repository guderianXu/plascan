add_library(camera_models_linescan STATIC
    LineScanDefinition.cpp
    LineScanDefinition.h
    LineScanTrajectory.cpp
    LineScanTrajectory.h
    LineScanInstance.cpp
    LineScanInstance.h
    LineScanOptics.cpp
    LineScanOptics.h
    LineScanProjection.cpp
    LineScanProjection.h
    LineScanOptimization.cpp
    LineScanOptimization.h
)

target_include_directories(camera_models_linescan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
target_compile_features(camera_models_linescan PUBLIC cxx_std_20)
target_link_libraries(camera_models_linescan PUBLIC camera_core)

if(BUILD_TESTS)
    include(GoogleTest)
    add_executable(test_linescan_camera_model
        tests/test_linescan_camera_model.cpp
    )
    target_link_libraries(test_linescan_camera_model PRIVATE
        camera_models_linescan
        GTest::gtest_main
    )
    target_include_directories(test_linescan_camera_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    )
    gtest_discover_tests(test_linescan_camera_model)
endif()
