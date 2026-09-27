cmake_minimum_required(VERSION 3.21)

project(placamera
    VERSION 0.2.0
    DESCRIPTION "Dependency-light C++ camera model library for planetary photogrammetry"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(PLACAMERA_BUILD_TESTS "Build PlaCamera unit tests" ${PROJECT_IS_TOP_LEVEL})
option(PLACAMERA_BUILD_EXAMPLES "Build PlaCamera examples" OFF)
option(PLACAMERA_BUILD_STATE "Build the canonical JSON camera-state component" ON)
option(PLACAMERA_BUILD_REFERENCE "Build external camera-reference support" OFF)
option(PLACAMERA_BUILD_GDAL "Build raster RPC file support" OFF)
option(PLACAMERA_INSTALL "Generate PlaCamera install rules" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET placoordinate::types)
    find_package(placoordinate 0.1 CONFIG REQUIRED COMPONENTS types)
endif()

configure_file(
    include/placamera/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/placamera/version.h
    @ONLY
)

add_library(placamera STATIC
    src/capabilities.cpp
    src/camera_baseline.cpp
    src/camera_topology.cpp
    src/colmap.cpp
    src/dataset_formats.cpp
    src/frame_camera_definition.cpp
    src/frame_camera_model.cpp
    src/frame_numeric_state.cpp
    src/instance_set.cpp
    src/internal/frame_camera_math.cpp
    src/internal/linescan_optics.cpp
    src/internal/linescan_numeric_projection.cpp
    src/internal/linescan_projection.cpp
    src/internal/rpc_geodesy.cpp
    src/internal/rpc_math.cpp
    src/linescan_camera_definition.cpp
    src/linescan_camera_model.cpp
    src/linescan_numeric_state.cpp
    src/linescan_trajectory.cpp
    src/model.cpp
    src/optimization.cpp
    src/project_import.cpp
    src/metashape_reference_import.cpp
    src/registry.cpp
    src/rpc_adjustment.cpp
    src/rpc_intersection.cpp
    src/rpc_metadata.cpp
    src/rpc_camera_definition.cpp
    src/rpc_camera_model.cpp
    src/types.cpp
    src/tsai.cpp
)
add_library(placamera::placamera ALIAS placamera)

target_compile_features(placamera PUBLIC cxx_std_20)
target_link_libraries(placamera PUBLIC placoordinate::types)
target_include_directories(placamera
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(placamera PROPERTIES
    CXX_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(PLACAMERA_BUILD_REFERENCE)
    if(NOT TARGET placoordinate::transform)
        find_package(placoordinate 0.1 CONFIG REQUIRED COMPONENTS transform)
    endif()
    add_library(placamera_reference STATIC
        src/reference/CameraReferenceObservation.cpp
        src/reference/CameraReferencePosePrior.cpp
        src/reference/CameraReferenceResolver.cpp
        src/reference/CameraReferenceUncertainty.cpp
        src/reference/MetashapeCameraReferenceAdapter.cpp
        src/reference/CameraReferenceComparator.cpp
        src/reference/ReferenceCameraGeometry.cpp
    )
    add_library(placamera::reference ALIAS placamera_reference)
    target_compile_features(placamera_reference PUBLIC cxx_std_20)
    target_link_libraries(placamera_reference PUBLIC placamera::placamera placoordinate::transform)
    target_include_directories(placamera_reference PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    set_target_properties(placamera_reference PROPERTIES CXX_EXTENSIONS OFF EXPORT_NAME reference)
endif()

if(PLACAMERA_BUILD_GDAL)
    if(NOT TARGET GDAL::GDAL)
        find_package(GDAL REQUIRED)
    endif()
    if(NOT TARGET GDAL::GDAL AND GDAL_FOUND)
        add_library(GDAL::GDAL INTERFACE IMPORTED GLOBAL)
        set(_placamera_gdal_include_dirs "${GDAL_INCLUDE_DIRS}")
        if(NOT _placamera_gdal_include_dirs)
            set(_placamera_gdal_include_dirs "${GDAL_INCLUDE_DIR}")
        endif()
        set_target_properties(GDAL::GDAL PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_placamera_gdal_include_dirs}"
            INTERFACE_LINK_LIBRARIES "${GDAL_LIBRARIES}"
        )
        unset(_placamera_gdal_include_dirs)
    endif()
    if(NOT TARGET GDAL::GDAL)
        message(FATAL_ERROR "PLACAMERA_BUILD_GDAL requires the GDAL::GDAL CMake target")
    endif()
    add_library(placamera_gdal STATIC src/rpc_raster.cpp)
    add_library(placamera::gdal ALIAS placamera_gdal)
    target_compile_features(placamera_gdal PUBLIC cxx_std_20)
    target_link_libraries(placamera_gdal PUBLIC placamera::placamera PRIVATE GDAL::GDAL)
    set_target_properties(placamera_gdal PROPERTIES
        CXX_EXTENSIONS OFF
        EXPORT_NAME gdal
        OUTPUT_NAME placamera_gdal
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

if(PLACAMERA_BUILD_STATE)
    find_package(nlohmann_json CONFIG REQUIRED)

    add_library(placamera_state STATIC
        src/isd.cpp
        src/internal/state_json_common.cpp
        src/state_codec.cpp
        src/state_frame_codec.cpp
        src/state_linescan_definition_codec.cpp
        src/state_linescan_instance_codec.cpp
        src/state_rpc_codec.cpp
    )
    add_library(placamera::state ALIAS placamera_state)
    target_compile_features(placamera_state PUBLIC cxx_std_20)
    target_link_libraries(placamera_state
        PUBLIC placamera::placamera
        PRIVATE nlohmann_json::nlohmann_json
    )
    target_include_directories(placamera_state PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(placamera_state PROPERTIES
        CXX_EXTENSIONS OFF
        EXPORT_NAME state
        OUTPUT_NAME placamera_state
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

if(PLACAMERA_BUILD_TESTS)
    include(CTest)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

if(PLACAMERA_BUILD_EXAMPLES)
    add_executable(placamera_tsai_projector examples/tsai_projector.cpp)
    target_compile_features(placamera_tsai_projector PRIVATE cxx_std_20)
    target_link_libraries(placamera_tsai_projector PRIVATE placamera::placamera)
    set_target_properties(placamera_tsai_projector PROPERTIES CXX_EXTENSIONS OFF)

    add_executable(placamera_project_import examples/project_import.cpp)
    target_compile_features(placamera_project_import PRIVATE cxx_std_20)
    target_link_libraries(placamera_project_import PRIVATE placamera::placamera)
    set_target_properties(placamera_project_import PROPERTIES CXX_EXTENSIONS OFF)

    add_executable(placamera_linescan_solver examples/linescan_solver.cpp)
    target_compile_features(placamera_linescan_solver PRIVATE cxx_std_20)
    target_link_libraries(placamera_linescan_solver PRIVATE placamera::placamera)
    set_target_properties(placamera_linescan_solver PROPERTIES CXX_EXTENSIONS OFF)

    add_executable(placamera_rpc_correction examples/rpc_correction.cpp)
    target_compile_features(placamera_rpc_correction PRIVATE cxx_std_20)
    target_link_libraries(placamera_rpc_correction PRIVATE placamera::placamera)
    set_target_properties(placamera_rpc_correction PROPERTIES CXX_EXTENSIONS OFF)

    if(TARGET placamera::reference)
        add_executable(placamera_reference_covariance examples/reference_covariance.cpp)
        target_compile_features(placamera_reference_covariance PRIVATE cxx_std_20)
        target_link_libraries(placamera_reference_covariance PRIVATE placamera::reference)
        set_target_properties(placamera_reference_covariance PROPERTIES CXX_EXTENSIONS OFF)
    endif()

    if(TARGET placamera::gdal)
        add_executable(placamera_rpc_raster_import examples/rpc_raster_import.cpp)
        target_compile_features(placamera_rpc_raster_import PRIVATE cxx_std_20)
        target_link_libraries(placamera_rpc_raster_import PRIVATE placamera::gdal)
        set_target_properties(placamera_rpc_raster_import PROPERTIES CXX_EXTENSIONS OFF)
    endif()

    if(TARGET placamera::state)
        add_executable(placamera_isd_import examples/isd_import.cpp)
        target_compile_features(placamera_isd_import PRIVATE cxx_std_20)
        target_link_libraries(placamera_isd_import PRIVATE placamera::state)
        set_target_properties(placamera_isd_import PROPERTIES CXX_EXTENSIONS OFF)
    endif()

    add_subdirectory(examples/consumer)
endif()

if(PLACAMERA_INSTALL)
    set(PLACAMERA_CONFIG_HAS_STATE ${PLACAMERA_BUILD_STATE})
    set(PLACAMERA_CONFIG_HAS_REFERENCE ${PLACAMERA_BUILD_REFERENCE})
    set(PLACAMERA_CONFIG_HAS_GDAL ${PLACAMERA_BUILD_GDAL})
    install(TARGETS placamera
        EXPORT placameraTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    if(PLACAMERA_BUILD_STATE)
        install(TARGETS placamera_state
            EXPORT placameraStateTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
    endif()
    if(PLACAMERA_BUILD_REFERENCE)
        install(TARGETS placamera_reference EXPORT placameraReferenceTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif()
    if(PLACAMERA_BUILD_GDAL)
        install(TARGETS placamera_gdal
            EXPORT placameraGdalTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
    endif()
    install(DIRECTORY include/placamera
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/generated/placamera/version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/placamera
    )
    install(EXPORT placameraTargets
        NAMESPACE placamera::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placamera
    )
    if(PLACAMERA_BUILD_STATE)
        install(EXPORT placameraStateTargets
            FILE placameraStateTargets.cmake
            NAMESPACE placamera::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placamera
        )
    endif()
    if(PLACAMERA_BUILD_REFERENCE)
        install(EXPORT placameraReferenceTargets
            FILE placameraReferenceTargets.cmake
            NAMESPACE placamera::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placamera)
    endif()
    if(PLACAMERA_BUILD_GDAL)
        install(EXPORT placameraGdalTargets
            FILE placameraGdalTargets.cmake
            NAMESPACE placamera::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placamera
        )
    endif()

    configure_package_config_file(
        cmake/placameraConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/placameraConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placamera
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/placameraConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/placameraConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/placameraConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/placamera
    )
    install(FILES LICENSE README.md CHANGELOG.md
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/placamera
    )
    install(FILES
        docs/DEPENDENCIES.md
        docs/METALIGN_CAMERA_MAPPING.md
        docs/MIGRATION.md
        docs/REFERENCE.md
        docs/USAGE.md
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/placamera/docs
    )
endif()
