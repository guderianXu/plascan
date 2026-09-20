add_library(camera_core STATIC
    CameraCore.cpp
    CameraCore.h
    capabilities/CameraCapabilities.cpp
    capabilities/CameraCapabilities.h
    capabilities/CameraOperationPlan.cpp
    capabilities/CameraOperationPlan.h
    capabilities/CapabilityRequirements.cpp
    capabilities/CapabilityRequirements.h
    capabilities/StaticPoseProvider.h
    model/CameraDefinition.cpp
    model/CameraDefinition.h
    model/CameraInstance.cpp
    model/CameraInstance.h
    model/CameraInstanceSet.cpp
    model/CameraInstanceSet.h
    model/CameraModelRegistry.cpp
    model/CameraModelRegistry.h
    types/CameraErrors.h
    types/CameraIds.h
    types/CameraPose.h
    types/CameraUncertainty.h
)

target_include_directories(camera_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_compile_features(camera_core PUBLIC cxx_std_20)

target_link_libraries(camera_core PUBLIC
    coordinate_system_types
)

if(BUILD_TESTS)
    include(GoogleTest)

    add_executable(test_camera_operation_plan
        tests/test_camera_operation_plan.cpp
    )
    target_link_libraries(test_camera_operation_plan PRIVATE
        camera_core
        GTest::gtest_main
    )
    target_include_directories(test_camera_operation_plan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
    )
    gtest_discover_tests(test_camera_operation_plan)
endif()
