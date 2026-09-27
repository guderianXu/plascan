cmake_minimum_required(VERSION 3.21)
project(placoordinate_consumer LANGUAGES CXX)

if(NOT TARGET placoordinate::transform)
    find_package(placoordinate CONFIG REQUIRED COMPONENTS types transform)
endif()

add_executable(placoordinate_consumer main.cpp)
target_compile_features(placoordinate_consumer PRIVATE cxx_std_20)
target_link_libraries(placoordinate_consumer PRIVATE placoordinate::transform)
set_target_properties(placoordinate_consumer PROPERTIES CXX_EXTENSIONS OFF)
