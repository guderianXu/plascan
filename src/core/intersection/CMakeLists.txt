
add_library(intersection STATIC Intersection.cpp)

target_include_directories(intersection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

target_link_libraries(intersection PUBLIC camera)

# Build a demo executable inside the intersection subproject so this CMake
# can be used independently to produce the example binary.
add_executable(intersection_demo intersection_demo.cpp)
target_include_directories(intersection_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../camera)
target_link_libraries(intersection_demo PRIVATE intersection camera)
