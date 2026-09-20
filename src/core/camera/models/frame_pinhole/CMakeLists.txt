add_library(camera_models_frame_pinhole STATIC
    FramePinholeDefinition.cpp
    FramePinholeDefinition.h
    FramePinholeInstance.cpp
    FramePinholeInstance.h
    FramePinholeProjection.cpp
    FramePinholeProjection.h
    FramePinholeOptimization.cpp
    FramePinholeOptimization.h
    FramePinholeNumericState.cpp
    FramePinholeNumericState.h
)

target_include_directories(camera_models_frame_pinhole PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

target_compile_features(camera_models_frame_pinhole PUBLIC cxx_std_20)
target_link_libraries(camera_models_frame_pinhole PUBLIC camera_core)

if(BUILD_TESTS)
    include(GoogleTest)

    add_executable(test_frame_pinhole_model
        tests/test_frame_pinhole_model.cpp
    )
    target_link_libraries(test_frame_pinhole_model PRIVATE
        camera_models_frame_pinhole
        GTest::gtest_main
    )
    target_include_directories(test_frame_pinhole_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    )
    gtest_discover_tests(test_frame_pinhole_model)
endif()
