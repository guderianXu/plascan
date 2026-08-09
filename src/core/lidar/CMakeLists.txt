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

target_link_libraries(lidar PRIVATE Qt6::Core)
target_compile_features(lidar PUBLIC cxx_std_20)

find_package(Ceres CONFIG QUIET)
if(Ceres_FOUND)
    target_link_libraries(lidar PRIVATE Ceres::ceres)
    target_compile_definitions(lidar PRIVATE PLASCAN_LIDAR_HAS_CERES=1)
    set_property(TARGET lidar PROPERTY PLASCAN_HAS_PLANETARY_LINE_SCAN_BA TRUE)
    message(STATUS "lidar: planetary line-scan Ceres BA enabled (${CERES_VERSION})")
else()
    message(STATUS "lidar: Ceres not found, planetary line-scan BA unavailable")
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
