add_library(camera_reference STATIC
    model/CameraReferenceSet.cpp
    model/CameraReferenceSet.h
    model/CameraReferenceTypes.h
    io/CameraReferenceSetJson.cpp
    io/CameraReferenceSetJsonDecode.cpp
    io/CameraReferenceSetJson.h
    io/CameraReferenceSetStore.cpp
    io/CameraReferenceSetStore.h
)

target_include_directories(camera_reference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(camera_reference PUBLIC
    Qt6::Core
)

target_compile_features(camera_reference PUBLIC cxx_std_20)

if(BUILD_TESTS)
    include(GoogleTest)

    add_executable(test_camera_reference_set_store
        tests/test_camera_reference_set_store.cpp
    )

    target_link_libraries(test_camera_reference_set_store PRIVATE
        camera_reference
        GTest::gtest_main
    )

    gtest_discover_tests(test_camera_reference_set_store DISCOVERY_MODE PRE_TEST)
endif()
