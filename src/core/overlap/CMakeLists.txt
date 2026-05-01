add_library(overlap STATIC
    GroundBackProjector.cpp
    OverlapAnalyzer.cpp
)

target_include_directories(overlap PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

target_link_libraries(overlap PUBLIC
    camera
    plascan_common_spatial
)
