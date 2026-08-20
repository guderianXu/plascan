add_library(lidar STATIC
    IsisControlNetworkPvl.cpp
    LaserConstraintAssociation.cpp
    LaserConstraintMap.cpp
    PlanetaryLaserBaAdapter.cpp
    PlanetaryLaserIsisJson.cpp
    PlanetaryLaserJson.cpp
    PlanetaryLaserShot.cpp
    PlanetaryLineScanBundleAdjust.cpp
    PlanetaryLineScanBundleAdjustCeres.cpp
    PlanetaryLineScanBundleAdjustModel.cpp
    PlanetaryLineScanBundleAdjustPlaMatrix.cpp
    PlanetaryLineScanBundleAdjustPlaMatrixAssembly.cpp
)

target_include_directories(lidar PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../overlap
    ${CMAKE_CURRENT_SOURCE_DIR}/../bundle_adjust
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

target_link_libraries(lidar PUBLIC
    bundle_adjust
    camera
    plascan_common_io
    plapoint::plapoint
)

target_link_libraries(lidar PRIVATE Qt6::Core plamatrix::plamatrix)
target_compile_features(lidar PUBLIC cxx_std_20)
set_property(TARGET lidar PROPERTY PLASCAN_HAS_PLANETARY_LINE_SCAN_BA TRUE)

if(PLASCAN_ENABLE_CERES_REFERENCE)
    target_link_libraries(lidar PRIVATE Ceres::ceres)
    target_compile_definitions(lidar PRIVATE PLASCAN_LIDAR_HAS_CERES=1)
    message(STATUS "lidar: optional planetary line-scan Ceres reference enabled (${CERES_VERSION})")
else()
    message(STATUS "lidar: planetary line-scan BA uses PlaMatrix without Ceres")
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
