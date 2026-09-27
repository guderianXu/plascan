include(GoogleTest)

add_executable(test_placoordinate_core
    test_coordinate_system_types.cpp
    test_coordinate_transform_service.cpp
)
target_link_libraries(test_placoordinate_core PRIVATE
    placoordinate::transform
    GTest::gtest_main
)
set_target_properties(test_placoordinate_core PROPERTIES CXX_EXTENSIONS OFF)
gtest_discover_tests(test_placoordinate_core TEST_PREFIX "PlaCoordinate.")

add_test(
    NAME PlaCoordinate.CoreDependencyBoundary
    COMMAND ${CMAKE_COMMAND}
        -DPLACOORDINATE_PUBLIC_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../include/placoordinate
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_core_dependencies.cmake
)

if(PLACOORDINATE_BUILD_GDAL)
    add_executable(test_placoordinate_gdal
        test_gdal_coordinate_transform.cpp
    )
    target_link_libraries(test_placoordinate_gdal PRIVATE
        placoordinate::gdal
        GTest::gtest_main
    )
    set_target_properties(test_placoordinate_gdal PROPERTIES CXX_EXTENSIONS OFF)
    gtest_discover_tests(test_placoordinate_gdal TEST_PREFIX "PlaCoordinate.")
endif()

if(PLACOORDINATE_BUILD_PRESETS)
    add_executable(test_placoordinate_presets
        test_projection_presets.cpp
    )
    target_link_libraries(test_placoordinate_presets PRIVATE
        placoordinate::presets
        GTest::gtest_main
    )
    set_target_properties(test_placoordinate_presets PROPERTIES CXX_EXTENSIONS OFF)
    gtest_discover_tests(test_placoordinate_presets TEST_PREFIX "PlaCoordinate.")
endif()

if(PLACOORDINATE_BUILD_GDAL AND PLACOORDINATE_BUILD_STATE)
    add_executable(test_placoordinate_state
        test_coordinate_context_json.cpp
    )
    target_link_libraries(test_placoordinate_state PRIVATE
        placoordinate::gdal
        placoordinate::state
        GTest::gtest_main
    )
    set_target_properties(test_placoordinate_state PROPERTIES CXX_EXTENSIONS OFF)
    gtest_discover_tests(test_placoordinate_state TEST_PREFIX "PlaCoordinate.")
endif()
