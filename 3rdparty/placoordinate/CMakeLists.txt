cmake_minimum_required(VERSION 3.21)

project(placoordinate
    VERSION 0.1.0
    DESCRIPTION "Typed coordinate contexts and transforms for planetary photogrammetry"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(PLACOORDINATE_BUILD_GDAL "Build the GDAL-backed CRS transform component" ON)
option(PLACOORDINATE_BUILD_PRESETS "Build the GDAL-backed projection presets component" ${PLACOORDINATE_BUILD_GDAL})
option(PLACOORDINATE_BUILD_STATE "Build the JSON coordinate-context component" ON)
option(PLACOORDINATE_BUILD_TESTS "Build PlaCoordinate unit tests" ${PROJECT_IS_TOP_LEVEL})
option(PLACOORDINATE_BUILD_EXAMPLES "Build PlaCoordinate examples" OFF)
option(PLACOORDINATE_INSTALL "Generate PlaCoordinate install rules" ${PROJECT_IS_TOP_LEVEL})

configure_file(
    include/placoordinate/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/placoordinate/version.h
    @ONLY
)

add_library(placoordinate_types STATIC
    src/context/CoordinateContext.cpp
    src/context/CoordinateHash.cpp
    src/context/SolverFrameDefinition.cpp
    src/context/SpatialReferenceDefinition.cpp
    src/types/CoordinateFrames.cpp
    src/types/CoordinateIds.cpp
    src/types/TimeReference.cpp
)
add_library(placoordinate::types ALIAS placoordinate_types)
set_target_properties(placoordinate_types PROPERTIES
    CXX_EXTENSIONS OFF
    EXPORT_NAME types
    OUTPUT_NAME placoordinate_types
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_compile_features(placoordinate_types PUBLIC cxx_std_20)
target_include_directories(placoordinate_types
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(placoordinate_transform STATIC
    src/transform/CoordinateTransformService.cpp
)
add_library(placoordinate::transform ALIAS placoordinate_transform)
set_target_properties(placoordinate_transform PROPERTIES
    CXX_EXTENSIONS OFF
    EXPORT_NAME transform
    OUTPUT_NAME placoordinate_transform
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_compile_features(placoordinate_transform PUBLIC cxx_std_20)
target_link_libraries(placoordinate_transform PUBLIC placoordinate::types)

set(_placoordinate_install_targets
    placoordinate_types
    placoordinate_transform
)

if(PLACOORDINATE_BUILD_PRESETS AND NOT PLACOORDINATE_BUILD_GDAL)
    message(FATAL_ERROR "PLACOORDINATE_BUILD_PRESETS requires PLACOORDINATE_BUILD_GDAL")
endif()

if(PLACOORDINATE_BUILD_GDAL)
    find_package(GDAL REQUIRED)
    if(NOT TARGET GDAL::GDAL)
        message(FATAL_ERROR "PlaCoordinate requires the imported GDAL::GDAL target")
    endif()

    add_library(placoordinate_gdal STATIC
        src/gdal/GdalCoordinateTransform.cpp
    )
    add_library(placoordinate::gdal ALIAS placoordinate_gdal)
    set_target_properties(placoordinate_gdal PROPERTIES
        CXX_EXTENSIONS OFF
        EXPORT_NAME gdal
        OUTPUT_NAME placoordinate_gdal
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    target_compile_features(placoordinate_gdal PUBLIC cxx_std_20)
    target_link_libraries(placoordinate_gdal
        PUBLIC placoordinate::types
        PRIVATE GDAL::GDAL
    )
    list(APPEND _placoordinate_install_targets placoordinate_gdal)
endif()

if(PLACOORDINATE_BUILD_PRESETS)
    add_library(placoordinate_presets STATIC
        src/presets/ProjectionPresets.cpp
    )
    add_library(placoordinate::presets ALIAS placoordinate_presets)
    set_target_properties(placoordinate_presets PROPERTIES
        CXX_EXTENSIONS OFF
        EXPORT_NAME presets
        OUTPUT_NAME placoordinate_presets
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    target_compile_features(placoordinate_presets PUBLIC cxx_std_20)
    target_link_libraries(placoordinate_presets
        PUBLIC placoordinate::gdal
        PRIVATE GDAL::GDAL
    )
    list(APPEND _placoordinate_install_targets placoordinate_presets)
endif()

if(PLACOORDINATE_BUILD_STATE)
    find_package(nlohmann_json CONFIG REQUIRED)

    add_library(placoordinate_state STATIC
        src/serialization/CoordinateContextJson.cpp
        src/serialization/CoordinateContextJsonStrings.cpp
    )
    add_library(placoordinate::state ALIAS placoordinate_state)
    set_target_properties(placoordinate_state PROPERTIES
        CXX_EXTENSIONS OFF
        EXPORT_NAME state
        OUTPUT_NAME placoordinate_state
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    target_compile_features(placoordinate_state PUBLIC cxx_std_20)
    target_link_libraries(placoordinate_state
        PUBLIC
            placoordinate::types
            nlohmann_json::nlohmann_json
    )
    list(APPEND _placoordinate_install_targets placoordinate_state)
endif()

if(PLACOORDINATE_BUILD_TESTS)
    include(CTest)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

if(PLACOORDINATE_BUILD_EXAMPLES)
    add_subdirectory(examples/consumer)
    if(PLACOORDINATE_BUILD_PRESETS)
        add_subdirectory(examples/presets_consumer)
    endif()
endif()

if(PLACOORDINATE_INSTALL)
    install(TARGETS ${_placoordinate_install_targets}
        EXPORT placoordinateTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY
        include/placoordinate/context
        include/placoordinate/transform
        include/placoordinate/types
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/placoordinate
        FILES_MATCHING PATTERN "*.h"
    )
    if(PLACOORDINATE_BUILD_GDAL)
        install(DIRECTORY include/placoordinate/gdal
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/placoordinate
            FILES_MATCHING PATTERN "*.h"
        )
    endif()
    if(PLACOORDINATE_BUILD_PRESETS)
        install(DIRECTORY include/placoordinate/presets
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/placoordinate
            FILES_MATCHING PATTERN "*.h"
        )
    endif()
    if(PLACOORDINATE_BUILD_STATE)
        install(DIRECTORY include/placoordinate/serialization
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/placoordinate
            FILES_MATCHING PATTERN "*.h"
        )
    endif()
    install(FILES
        include/placoordinate/placoordinate.h
        ${CMAKE_CURRENT_BINARY_DIR}/generated/placoordinate/version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/placoordinate
    )
    install(EXPORT placoordinateTargets
        NAMESPACE placoordinate::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placoordinate
    )

    configure_package_config_file(
        cmake/placoordinateConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/placoordinateConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placoordinate
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/placoordinateConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/placoordinateConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/placoordinateConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placoordinate
    )
    install(FILES LICENSE README.md CHANGELOG.md
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/placoordinate
    )
endif()
