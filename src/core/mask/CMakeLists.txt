add_library(mask
    MaskGenerator.cpp
    u2net/U2NetMaskGenerator.cpp
)

target_include_directories(mask PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mask PUBLIC
    ${OpenCV_LIBS}
)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
