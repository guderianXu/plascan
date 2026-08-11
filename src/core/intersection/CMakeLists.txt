
add_library(intersection STATIC Intersection.cpp)

target_include_directories(intersection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

target_link_libraries(intersection
    PUBLIC camera
    PRIVATE plascan_common_concurrency
)
