cmake_minimum_required(VERSION 3.16)
project(plascan_inference LANGUAGES CXX)

add_library(plascan_inference STATIC
    tensorrt/TensorRtCapabilities.h
    tensorrt/TensorRtEngineCache.h
    tensorrt/TensorRtEngineBuilder.h
    tensorrt/TensorRtTensorInfo.cpp
    tensorrt/TensorRtTensorInfo.h
)

if(PLASCAN_ENABLE_TENSORRT)
    target_sources(plascan_inference PRIVATE
        tensorrt/TensorRtCapabilities.cpp
        tensorrt/TensorRtBuildSupport.h
        tensorrt/TensorRtEngineCache.cpp
        tensorrt/TensorRtEngineBuilder.cpp
        tensorrt/TensorRtNetworkBuilder.cpp
        tensorrt/TensorRtNetworkBuilder.h
        tensorrt/TensorRtSession.cpp
        tensorrt/TensorRtSession.h
        tensorrt/TensorRtSessionExecution.cpp
        tensorrt/TensorRtSessionImpl.cpp
        tensorrt/TensorRtSessionImpl.h
        tensorrt/TensorRtSessionResources.h
        tensorrt/TensorRtTypeConversions.h
    )
    target_link_libraries(plascan_inference PUBLIC
        TensorRT::nvinfer
        TensorRT::nvonnxparser
        CUDA::cudart
    )
    target_link_libraries(plascan_inference PRIVATE
        plascan_common_io
    )
    target_compile_definitions(plascan_inference PUBLIC PLASCAN_HAS_TENSORRT=1)
else()
    target_sources(plascan_inference PRIVATE
        tensorrt/TensorRtCapabilitiesStub.cpp
        tensorrt/TensorRtEngineBuilderStub.cpp
    )
endif()

target_include_directories(plascan_inference PUBLIC
    ${CMAKE_SOURCE_DIR}/src/core
)

target_link_libraries(plascan_inference PUBLIC
    Qt6::Core
)

target_compile_features(plascan_inference PUBLIC cxx_std_20)

if(BUILD_TESTS)
    add_executable(test_tensorrt_tensor_info
        tests/test_tensorrt_tensor_info.cpp
    )
    target_link_libraries(test_tensorrt_tensor_info PRIVATE
        GTest::gtest
        GTest::gtest_main
        plascan_inference
    )
    add_test(NAME TensorRtTensorInfoTest COMMAND test_tensorrt_tensor_info)
endif()
