add_library(coordinate_system_types STATIC
    context/CoordinateContext.cpp
    context/CoordinateContext.h
    context/CoordinateHash.cpp
    context/CoordinateHash.h
    context/SolverFrameDefinition.cpp
    context/SolverFrameDefinition.h
    context/SpatialReferenceDefinition.cpp
    context/SpatialReferenceDefinition.h
    types/CoordinateErrors.h
    types/CoordinateFrames.cpp
    types/CoordinateFrames.h
    types/CoordinateIds.cpp
    types/CoordinateIds.h
    types/TimeReference.cpp
    types/TimeReference.h
)

target_include_directories(coordinate_system_types PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(coordinate_system_types PUBLIC cxx_std_20)

add_library(coordinate_system_transform STATIC
    transform/CoordinateTransformService.cpp
    transform/CoordinateTransformService.h
)

target_include_directories(coordinate_system_transform PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(coordinate_system_transform PUBLIC
    coordinate_system_types
)

target_compile_features(coordinate_system_transform PUBLIC cxx_std_20)

add_library(coordinate_system_gdal STATIC
    gdal/GdalCoordinateTransform.cpp
    gdal/GdalCoordinateTransform.h
)

target_include_directories(coordinate_system_gdal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(GDAL_INCLUDE_DIRS)
    target_include_directories(coordinate_system_gdal PRIVATE ${GDAL_INCLUDE_DIRS})
endif()

target_link_libraries(coordinate_system_gdal PUBLIC
    coordinate_system_transform
    ${PLASCAN_GDAL_TARGET}
)

target_compile_features(coordinate_system_gdal PUBLIC cxx_std_20)

find_package(nlohmann_json CONFIG REQUIRED)

add_library(coordinate_system_json STATIC
    serialization/CoordinateContextJson.cpp
    serialization/CoordinateContextJson.h
    serialization/CoordinateContextJsonStrings.cpp
    serialization/CoordinateContextJsonStrings.h
)

target_include_directories(coordinate_system_json PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(coordinate_system_json PUBLIC
    coordinate_system_types
    nlohmann_json::nlohmann_json
)

target_compile_features(coordinate_system_json PUBLIC cxx_std_20)

if(BUILD_TESTS)
    include(GoogleTest)

    add_executable(test_coordinate_system_types
        tests/test_coordinate_system_types.cpp
    )
    target_link_libraries(test_coordinate_system_types PRIVATE
        coordinate_system_types
        GTest::gtest_main
    )
    gtest_discover_tests(test_coordinate_system_types)

    add_executable(test_coordinate_transform_service
        tests/test_coordinate_transform_service.cpp
    )
    target_link_libraries(test_coordinate_transform_service PRIVATE
        coordinate_system_transform
        GTest::gtest_main
    )
    gtest_discover_tests(test_coordinate_transform_service)

    add_executable(test_gdal_coordinate_transform
        tests/test_gdal_coordinate_transform.cpp
    )
    target_link_libraries(test_gdal_coordinate_transform PRIVATE
        coordinate_system_gdal
        GTest::gtest_main
    )
    gtest_discover_tests(test_gdal_coordinate_transform)

    add_executable(test_coordinate_context_json
        tests/test_coordinate_context_json.cpp
    )
    target_link_libraries(test_coordinate_context_json PRIVATE
        coordinate_system_json
        coordinate_system_gdal
        GTest::gtest_main
    )
    gtest_discover_tests(test_coordinate_context_json)
endif()
