cmake_minimum_required(VERSION 3.21)
project(placamera_consumer LANGUAGES CXX)

if(NOT TARGET placamera::placamera)
    find_package(placamera CONFIG REQUIRED)
endif()

add_executable(placamera_consumer main.cpp)
target_compile_features(placamera_consumer PRIVATE cxx_std_20)
target_link_libraries(placamera_consumer PRIVATE placamera::placamera)
set_target_properties(placamera_consumer PROPERTIES CXX_EXTENSIONS OFF)
