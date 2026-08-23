cmake_minimum_required(VERSION 3.16)
project(plascan_image_matching LANGUAGES CXX)

if(NOT PLASCAN_ENABLE_TENSORRT AND CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
endif()

# 统一影像匹配目标。SIFT、LightGlue、几何验证和 `.pimatch` 存储最终都会在
# 本目标内注册，调用方只依赖 image_matching，不再分别依赖提取器和匹配器目录。
add_library(image_matching STATIC
    FeatureSet.cpp
    FeatureSet.h
    ImageMatchFile.cpp
    ImageMatchFile.h
    ImageMatchIndexFile.cpp
    ImageMatchIndexFile.h
    ImageMatchRepository.cpp
    ImageMatchRepository.h
    ImageMatchTypes.cpp
    ImageMatchTypes.h
    ImageMatchingAlgorithm.cpp
    ImageMatchingAlgorithm.h
    ImageMatchingRegistry.cpp
    ImageMatchingRegistry.h
    MatchResult.cpp
    MatchResult.h
    geometry/MatchGeometryVerifier.cpp
    geometry/MatchGeometryVerifier.h
    sift/AutoSiftAlgorithm.cpp
    sift/AutoSiftAlgorithm.h
    sift/SiftComputeBackend.cpp
    sift/SiftComputeBackend.h
    sift/SiftBackendType.h
    sift/SiftFeatureExtractor.cpp
    sift/SiftFeatureExtractor.h
    sift/SiftGuidedMatcher.cpp
    sift/SiftGuidedMatcher.h
    sift/SiftLowTextureRecovery.cpp
    sift/SiftLowTextureRecovery.h
    sift/SiftMatchFilter.cpp
    sift/SiftMatchFilter.h
    lightglue/LightGlueFeatureBudget.h
    lightglue/LightGluePostprocessor.cpp
    lightglue/LightGluePostprocessor.h
)

if(APPLE)
    enable_language(OBJCXX)
    target_sources(image_matching PRIVATE
        sift/SiftMetalBackend.mm
        sift/SiftMetalKernels.h
    )
    target_compile_definitions(image_matching PUBLIC PLASCAN_HAS_METAL_SIFT=1)
    target_compile_options(image_matching PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )
    target_link_libraries(image_matching PRIVATE
        "-framework Foundation"
        "-framework Metal"
    )
endif()

if(PLASCAN_ENABLE_OPENCL AND TARGET OpenCL::OpenCL)
    target_sources(image_matching PRIVATE
        sift/SiftOpenClBackend.cpp
        sift/SiftOpenClKernels.h
    )
    target_compile_definitions(image_matching PRIVATE PLASCAN_HAS_OPENCL_SIFT=1)
    target_link_libraries(image_matching PRIVATE OpenCL::OpenCL)
endif()

if(PLASCAN_ENABLE_TENSORRT)
    target_sources(image_matching PRIVATE
        sift_lightglue/SiftLightGlueAlgorithm.cpp
        sift_lightglue/SiftLightGlueAlgorithm.h
        loma_r/LoMaRAlgorithm.cpp
        loma_r/LoMaRAlgorithm.h
        loma_r/LoMaRTensorRtBackend.cpp
        loma_r/LoMaRTensorRtBackend.h
        lightglue/TensorRtLightGlueMatcher.cpp
        lightglue/TensorRtLightGlueMatcher.h
    )
endif()

if(PLASCAN_ENABLE_TENSORRT AND CMAKE_CUDA_COMPILER)
    target_sources(image_matching PRIVATE
        lightglue/TensorRtLightGluePostprocessor.cu
        lightglue/TensorRtLightGluePostprocessor.h
    )
    target_compile_definitions(image_matching PRIVATE
        PLASCAN_TENSORRT_CUDA_POSTPROCESS=1
    )
    set_target_properties(image_matching PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    )
    target_compile_options(image_matching PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif()

if(NOT TARGET plascan_cudasift AND CMAKE_CUDA_COMPILER AND
   EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/cudasift/CMakeLists.txt")
    add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/cudasift"
                     "${CMAKE_BINARY_DIR}/3rdparty/cudasift")
endif()

target_include_directories(image_matching PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/lightglue
    ${CMAKE_CURRENT_SOURCE_DIR}/geometry
    ${CMAKE_CURRENT_SOURCE_DIR}/sift
    ${CMAKE_CURRENT_SOURCE_DIR}/sift_lightglue
    ${CMAKE_CURRENT_SOURCE_DIR}/loma_r
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/3rdparty/cudasift
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(image_matching PUBLIC
    Qt6::Core
    ${OpenCV_LIBS}
    plascan_common_io
    plascan_inference
)

if(TARGET plascan_cudasift)
    target_sources(image_matching PRIVATE sift/SiftCudaBackend.cpp)
    target_link_libraries(image_matching PUBLIC
        plascan_cudasift
        CUDA::cudart
    )
    target_compile_definitions(image_matching PUBLIC PLASCAN_HAS_CUDA_SIFT=1)
endif()

target_compile_features(image_matching PUBLIC cxx_std_20)

if(BUILD_TESTS)
    add_executable(test_image_match_file
        tests/test_image_match_file.cpp
    )
    target_link_libraries(test_image_match_file PRIVATE
        GTest::gtest
        GTest::gtest_main
        image_matching
        Qt6::Core
    )
    add_test(NAME ImageMatchFileTest COMMAND test_image_match_file)

    add_executable(test_image_matching_registry
        tests/test_image_matching_registry.cpp
    )
    target_link_libraries(test_image_matching_registry PRIVATE
        GTest::gtest
        GTest::gtest_main
        image_matching
        Qt6::Core
    )
    add_test(NAME ImageMatchingRegistryTest COMMAND test_image_matching_registry)

    add_executable(test_match_geometry_verifier
        tests/test_match_geometry_verifier.cpp
    )
    target_link_libraries(test_match_geometry_verifier PRIVATE
        GTest::gtest
        GTest::gtest_main
        image_matching
        ${OpenCV_LIBS}
    )
    add_test(NAME MatchGeometryVerifierTest COMMAND test_match_geometry_verifier)

    add_executable(test_auto_sift
        tests/test_auto_sift.cpp
    )
    target_link_libraries(test_auto_sift PRIVATE
        GTest::gtest
        GTest::gtest_main
        image_matching
    )
    target_compile_definitions(test_auto_sift PRIVATE
        TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
    )
    add_test(NAME AutoSiftTest COMMAND test_auto_sift)
endif()
