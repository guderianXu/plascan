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

find_library(TensorRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser_11 nvonnxparser_10 nvonnxparser
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES lib lib/x64
)

if(WIN32)
    find_file(TensorRT_NVINFER_RUNTIME_LIBRARY
        NAMES nvinfer_11.dll nvinfer_10.dll nvinfer.dll
        HINTS ${_TensorRT_HINTS}
        PATH_SUFFIXES bin lib lib/x64
    )
    find_file(TensorRT_NVONNXPARSER_RUNTIME_LIBRARY
        NAMES nvonnxparser_11.dll nvonnxparser_10.dll nvonnxparser.dll
        HINTS ${_TensorRT_HINTS}
        PATH_SUFFIXES bin lib lib/x64
    )
    if(TensorRT_NVINFER_RUNTIME_LIBRARY)
        get_filename_component(_TensorRT_RUNTIME_DIRECTORY
            "${TensorRT_NVINFER_RUNTIME_LIBRARY}" DIRECTORY)
        file(GLOB TensorRT_DYNAMIC_RUNTIME_LIBRARIES
            LIST_DIRECTORIES false
            "${_TensorRT_RUNTIME_DIRECTORY}/nvinfer_builder_resource_*.dll"
            "${_TensorRT_RUNTIME_DIRECTORY}/nvinfer_plugin_*.dll")
        set(TensorRT_RUNTIME_LIBRARIES
            "${TensorRT_NVINFER_RUNTIME_LIBRARY}"
            "${TensorRT_NVONNXPARSER_RUNTIME_LIBRARY}"
            ${TensorRT_DYNAMIC_RUNTIME_LIBRARIES})
        list(REMOVE_DUPLICATES TensorRT_RUNTIME_LIBRARIES)
        # find_package 在 image_matching 子目录执行，而 GUI/CLI 位于同级目录。
        # 缓存该列表，确保所有最终可执行目标都能部署同一组构建/推理运行库。
        set(TensorRT_RUNTIME_LIBRARIES "${TensorRT_RUNTIME_LIBRARIES}"
            CACHE INTERNAL "TensorRT runtime DLLs required by PlaScan" FORCE)
    endif()
endif()

set(_TensorRT_REQUIRED_VARS
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIBRARY
    TensorRT_NVONNXPARSER_LIBRARY)
if(WIN32)
    list(APPEND _TensorRT_REQUIRED_VARS
        TensorRT_NVINFER_RUNTIME_LIBRARY
        TensorRT_NVONNXPARSER_RUNTIME_LIBRARY)
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

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvonnxparser)
    if(WIN32 AND TensorRT_NVONNXPARSER_RUNTIME_LIBRARY)
        add_library(TensorRT::nvonnxparser SHARED IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_IMPLIB "${TensorRT_NVONNXPARSER_LIBRARY}"
            IMPORTED_LOCATION "${TensorRT_NVONNXPARSER_RUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    else()
        add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_NVONNXPARSER_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIBRARY
    TensorRT_NVINFER_RUNTIME_LIBRARY
    TensorRT_NVONNXPARSER_LIBRARY
    TensorRT_NVONNXPARSER_RUNTIME_LIBRARY
    TensorRT_DYNAMIC_RUNTIME_LIBRARIES
    TensorRT_RUNTIME_LIBRARIES
)
