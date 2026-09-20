add_library(camera_reference_core STATIC
    model/CameraReferenceObservation.cpp
    model/CameraReferenceObservation.h
    resolve/CameraReferenceResolver.cpp
    resolve/CameraReferenceResolver.h
    resolve/CameraReferencePosePrior.cpp
    resolve/CameraReferencePosePrior.h
)

target_include_directories(camera_reference_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(camera_reference_core PUBLIC
    camera_core
    coordinate_system_transform
)

target_compile_features(camera_reference_core PUBLIC cxx_std_20)

# Projection-ready reference geometry is a concrete frame-pinhole adapter.  It
# deliberately lives in a separate target so the generic reference resolver
# remains independent from every camera model; RPC and line-scan references can
# add their own adapters without changing camera_reference_core.
add_library(camera_reference_geometry STATIC
    geometry/ReferenceCameraGeometry.cpp
    geometry/ReferenceCameraGeometry.h
)

target_include_directories(camera_reference_geometry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(camera_reference_geometry PUBLIC
    camera_reference_core
    camera_models_frame_pinhole
)

target_compile_features(camera_reference_geometry PUBLIC cxx_std_20)

add_library(camera_reference STATIC
    model/CameraReferenceSet.cpp
    model/CameraReferenceSet.h
    model/CameraReferenceTypes.h
    compare/CameraReferenceComparator.cpp
    compare/CameraReferenceComparator.h
    io/CameraReferenceSetJson.cpp
    io/CameraReferenceSetJsonDecode.cpp
    io/CameraReferenceSetJson.h
    io/CameraReferenceSetStore.cpp
    io/CameraReferenceSetStore.h
)

target_include_directories(camera_reference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(camera_reference PUBLIC
    camera_reference_core
    camera_core
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
