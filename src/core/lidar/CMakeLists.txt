add_library(lidar STATIC
    LaserConstraintAssociation.cpp
    LaserConstraintMap.cpp
    PlanetaryLaserBaAdapter.cpp
    PlanetaryLaserIsisJson.cpp
    PlanetaryLaserJson.cpp
    PlanetaryLaserShot.cpp
)

target_include_directories(lidar PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../overlap
    ${CMAKE_CURRENT_SOURCE_DIR}/../bundle_adjust
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

target_link_libraries(lidar PUBLIC
    bundle_adjust
    plascan_common_io
    plapoint::plapoint
)

target_link_libraries(lidar PRIVATE Qt6::Core)
target_compile_features(lidar PUBLIC cxx_std_20)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
