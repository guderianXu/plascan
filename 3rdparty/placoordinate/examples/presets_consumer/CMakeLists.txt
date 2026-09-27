cmake_minimum_required(VERSION 3.21)
project(placoordinate_presets_consumer LANGUAGES CXX)

if(NOT TARGET placoordinate::presets)
    find_package(placoordinate CONFIG REQUIRED COMPONENTS presets)
endif()

add_executable(placoordinate_presets_consumer main.cpp)
target_compile_features(placoordinate_presets_consumer PRIVATE cxx_std_20)
target_link_libraries(placoordinate_presets_consumer PRIVATE placoordinate::presets)
set_target_properties(placoordinate_presets_consumer PROPERTIES CXX_EXTENSIONS OFF)
