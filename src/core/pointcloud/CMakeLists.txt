add_library(pointcloud STATIC
    data/PointCloudPoint.cpp
    data/PointCloud.cpp
    processing/PointCloudProcessor.cpp
    processing/PointCloudProcessorMT.cpp
    processing/PointCloudProcessorCuda.cpp
    io/PointCloudIO.cpp
    io/ObjMtlLoader.cpp
)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

target_include_directories(pointcloud PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pointcloud
    PUBLIC
        plascan_common_log
        plascan_common_math
        plascan_common_spatial
    PRIVATE
        Qt6::Core
        Qt6::Gui
        ${OpenCV_LIBS}
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_sources(pointcloud PRIVATE processing/PointCloudProcessorCuda.cu)
    target_compile_definitions(pointcloud PUBLIC POINTCLOUD_ENABLE_CUDA=1)
    set_target_properties(pointcloud PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    )
else()
    message(WARNING "pointcloud: CUDA compiler not found, fallback to CPU path")
endif()

if(BUILD_TESTS)
    include(GoogleTest)

    function(add_pointcloud_gtest testTarget testSource)
        add_executable(${testTarget} ${testSource})
        target_link_libraries(${testTarget} PRIVATE
            pointcloud
            Qt6::Core
            Qt6::Gui
            GTest::gtest_main
        )
        target_include_directories(${testTarget} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_target_properties(${testTarget} PROPERTIES
            CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
        )
        gtest_discover_tests(${testTarget})
    endfunction()

    add_pointcloud_gtest(test_pointcloud_io tests/test_pointcloud_io.cpp)
    add_pointcloud_gtest(test_pointcloud_processor tests/test_pointcloud_processor.cpp)

    add_executable(pointcloud_obj_info
        tests/pointcloud_obj_info.cpp
    )

    set_target_properties(pointcloud_obj_info PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    )

    target_link_libraries(pointcloud_obj_info PRIVATE
        pointcloud
    )

    target_include_directories(pointcloud_obj_info PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
