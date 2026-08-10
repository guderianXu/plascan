add_library(mask
    MaskGenerator.cpp
    birefnet/BiRefNetImageProcessing.cpp
    birefnet/BiRefNetMaskGenerator.cpp
    birefnet/BiRefNetTensorRtBackend.cpp
    u2net/U2NetImageProcessing.cpp
    u2net/U2NetMaskGenerator.cpp
    u2net/U2NetOpenCvCpuBackend.cpp
    u2net/U2NetTensorRtBackend.cpp
)

target_include_directories(mask PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mask PUBLIC
    plascan_inference
    ${OpenCV_LIBS}
)

target_compile_features(mask PUBLIC cxx_std_20)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
