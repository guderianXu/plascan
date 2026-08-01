include(FindPackageHandleStandardArgs)

set(TensorRT_ROOT "" CACHE PATH "TensorRT SDK root containing include/ and lib/")

set(_TensorRT_HINTS
    ${TensorRT_ROOT}
    $ENV{TENSORRT_ROOT}
    $ENV{TensorRT_ROOT}
)

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInferRuntime.h
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES include
)

find_library(TensorRT_NVINFER_LIBRARY
    NAMES nvinfer_11 nvinfer_10 nvinfer
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES lib lib/x64
)

if(WIN32)
    find_file(TensorRT_NVINFER_RUNTIME_LIBRARY
        NAMES nvinfer_11.dll nvinfer_10.dll nvinfer.dll
        HINTS ${_TensorRT_HINTS}
        PATH_SUFFIXES bin lib lib/x64
    )
endif()

set(_TensorRT_REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY)
if(WIN32)
    list(APPEND _TensorRT_REQUIRED_VARS TensorRT_NVINFER_RUNTIME_LIBRARY)
endif()

find_package_handle_standard_args(TensorRT
    REQUIRED_VARS ${_TensorRT_REQUIRED_VARS}
)
unset(_TensorRT_REQUIRED_VARS)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    if(WIN32 AND TensorRT_NVINFER_RUNTIME_LIBRARY)
        add_library(TensorRT::nvinfer SHARED IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_IMPLIB "${TensorRT_NVINFER_LIBRARY}"
            IMPORTED_LOCATION "${TensorRT_NVINFER_RUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    else()
        add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIBRARY
    TensorRT_NVINFER_RUNTIME_LIBRARY
)
