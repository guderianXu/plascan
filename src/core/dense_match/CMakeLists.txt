cmake_minimum_required(VERSION 3.16)
project(dense_match LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(DM_HAS_CUDA TRUE)
    message(STATUS "dense_match: CUDA enabled (${CMAKE_CUDA_COMPILER})")
else()
    set(DM_HAS_CUDA FALSE)
    message(STATUS "dense_match: CUDA unavailable, CPU-only")
endif()

set(DM_CXX_SOURCES
    CostFunctions.cpp
    BlockMatcher.cpp
    SgmMatcher.cpp
    SubpixelRefiner.cpp
    DisparityValidator.cpp
    DenseMatchService.cpp
    opencv/OpenCVSgbmWrapper.cpp
)

set(DM_HEADERS
    DenseMatchTypes.h
    DenseMatchConfig.h
    CostFunctions.h
    BlockMatcher.h
    SgmMatcher.h
    SubpixelRefiner.h
    DisparityValidator.h
    DenseMatchService.h
    opencv/OpenCVSgbmWrapper.h
)

if(DM_HAS_CUDA)
    set(DM_CUDA_SOURCES
        CostFunctions.cu
    )
else()
    set(DM_CUDA_SOURCES)
endif()

add_library(dense_match STATIC
    ${DM_CXX_SOURCES}
    ${DM_CUDA_SOURCES}
    ${DM_HEADERS}
)

if(DM_HAS_CUDA)
    set_target_properties(dense_match PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(dense_match PUBLIC DM_ENABLE_CUDA=1)
    target_compile_options(dense_match PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    )
endif()

target_include_directories(dense_match PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(dense_match
    PUBLIC
        plascan_common_io
        ${OpenCV_LIBS}
    PRIVATE
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(dense_match PRIVATE HAS_OPENMP)
endif()

if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        include(GoogleTest)
        add_executable(test_dense_match_unit
            tests/CostFunctionTest.cpp
            tests/BlockMatcherTest.cpp
            tests/SgmMatcherTest.cpp
            tests/SubpixelRefinerTest.cpp
            tests/DisparityValidatorTest.cpp
            tests/DenseMatchIntegrationTest.cpp
        )
        target_link_libraries(test_dense_match_unit PRIVATE
            dense_match
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_dense_match_unit PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/dense_match
        )
        gtest_discover_tests(test_dense_match_unit)
    endif()
endif()
