
add_library(intersection STATIC Intersection.cpp)

target_include_directories(intersection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
    ${CMAKE_SOURCE_DIR}/src/core
)

target_link_libraries(intersection
    PUBLIC camera camera_models_frame_pinhole
    PRIVATE plascan_common_concurrency
)
