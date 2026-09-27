add_library(lidar STATIC
    IsisControlNetworkPvl.cpp
    LaserConstraintAssociation.cpp
    LaserConstraintMap.cpp
    PlanetaryLaserBaAdapter.cpp
    PlanetaryLaserIsisJson.cpp
    PlanetaryLaserJson.cpp
    PlanetaryLaserShot.cpp
    PlanetaryLineScanBundleAdjust.cpp
)

target_include_directories(lidar PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../overlap
)

target_link_libraries(lidar PUBLIC
    plabundle::plabundle
    placamera::placamera
    plascan_common_io
    plapoint::plapoint
)

target_link_libraries(lidar PRIVATE Qt6::Core)
target_compile_features(lidar PUBLIC cxx_std_20)
set_property(TARGET lidar PROPERTY PLASCAN_HAS_PLANETARY_LINE_SCAN_BA TRUE)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
