add_library(camera_project STATIC
    CameraInstanceUpdate.h
    CameraProjectRuntime.cpp
    CameraProjectRuntime.h
    CameraProjectStore.cpp
    CameraProjectStore.h
    CameraProjectValidation.cpp
    CameraProjectValidation.h
    CameraProjectRecords.cpp
    CameraProjectRecords.h
)

target_include_directories(camera_project PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_compile_features(camera_project PUBLIC cxx_std_20)
target_link_libraries(camera_project PUBLIC camera_core camera_models_registry Qt6::Core)

if(BUILD_TESTS)
    include(GoogleTest)

    add_executable(test_camera_project_store
        tests/test_camera_project_store.cpp
    )
    target_link_libraries(test_camera_project_store PRIVATE
        camera_project
        camera_models_registry
        Qt6::Core
        GTest::gtest_main
    )
    target_include_directories(test_camera_project_store PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
    )
    gtest_discover_tests(test_camera_project_store)
endif()
