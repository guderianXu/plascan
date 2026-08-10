function(plascan_deploy_tensorrt_runtime target_name)
  if(NOT WIN32 OR NOT PLASCAN_ENABLE_TENSORRT OR NOT TARGET "${target_name}" OR
     NOT TensorRT_RUNTIME_LIBRARIES)
    return()
  endif()

  set(_plascan_tensorrt_runtime_dlls ${TensorRT_RUNTIME_LIBRARIES})
  set(_plascan_tensorrt_runtime_dirs)
  foreach(_plascan_tensorrt_dll IN LISTS _plascan_tensorrt_runtime_dlls)
    if(EXISTS "${_plascan_tensorrt_dll}")
      get_filename_component(_plascan_tensorrt_runtime_dir
        "${_plascan_tensorrt_dll}" DIRECTORY)
      list(APPEND _plascan_tensorrt_runtime_dirs
        "${_plascan_tensorrt_runtime_dir}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _plascan_tensorrt_runtime_dirs)
  foreach(_plascan_tensorrt_runtime_dir IN LISTS _plascan_tensorrt_runtime_dirs)
    file(GLOB _plascan_tensorrt_dynamic_dlls
      LIST_DIRECTORIES false
      "${_plascan_tensorrt_runtime_dir}/nvinfer.dll"
      "${_plascan_tensorrt_runtime_dir}/nvinfer_[0-9]*.dll"
      "${_plascan_tensorrt_runtime_dir}/nvinfer_plugin*.dll"
      "${_plascan_tensorrt_runtime_dir}/nvinfer_builder_resource_*.dll"
      "${_plascan_tensorrt_runtime_dir}/nvonnxparser*.dll")
    list(APPEND _plascan_tensorrt_runtime_dlls
      ${_plascan_tensorrt_dynamic_dlls})
  endforeach()

  set(_plascan_cuda_runtime_dirs
    "${CUDAToolkit_BIN_DIR}"
    "${CUDAToolkit_BIN_DIR}/x64"
    "${CUDAToolkit_ROOT}/bin"
    "${CUDAToolkit_ROOT}/bin/x64"
    "$ENV{CUDA_PATH}/bin"
    "$ENV{CUDA_PATH}/bin/x64")
  list(REMOVE_DUPLICATES _plascan_cuda_runtime_dirs)
  foreach(_plascan_cuda_runtime_dir IN LISTS _plascan_cuda_runtime_dirs)
    if(NOT IS_DIRECTORY "${_plascan_cuda_runtime_dir}")
      continue()
    endif()
    file(GLOB _plascan_cuda_dynamic_dlls
      LIST_DIRECTORIES false
      "${_plascan_cuda_runtime_dir}/cudart64_*.dll"
      "${_plascan_cuda_runtime_dir}/cublas64_*.dll"
      "${_plascan_cuda_runtime_dir}/cublasLt64_*.dll"
      "${_plascan_cuda_runtime_dir}/nvfatbin_*.dll"
      "${_plascan_cuda_runtime_dir}/nvrtc64_*.dll"
      "${_plascan_cuda_runtime_dir}/nvrtc-builtins64_*.dll")
    list(APPEND _plascan_tensorrt_runtime_dlls
      ${_plascan_cuda_dynamic_dlls})
  endforeach()
  list(REMOVE_DUPLICATES _plascan_tensorrt_runtime_dlls)

  set(_plascan_has_nvinfer FALSE)
  set(_plascan_has_onnx_parser FALSE)
  set(_plascan_has_plugin FALSE)
  set(_plascan_has_builder_resource FALSE)
  set(_plascan_has_cudart FALSE)
  set(_plascan_has_cublas FALSE)
  set(_plascan_has_cublas_lt FALSE)
  set(_plascan_has_nvfatbin FALSE)
  set(_plascan_has_nvrtc FALSE)
  set(_plascan_has_nvrtc_builtins FALSE)
  foreach(_plascan_tensorrt_dll IN LISTS _plascan_tensorrt_runtime_dlls)
    get_filename_component(_plascan_tensorrt_name
      "${_plascan_tensorrt_dll}" NAME)
    if(_plascan_tensorrt_name MATCHES "^cudnn.*\\.dll$")
      message(FATAL_ERROR
        "TensorRT deployment must not include cuDNN: ${_plascan_tensorrt_dll}")
    elseif(_plascan_tensorrt_name MATCHES "^nvinfer(_[0-9]+)?\\.dll$")
      set(_plascan_has_nvinfer TRUE)
    elseif(_plascan_tensorrt_name MATCHES "^nvonnxparser(_[0-9]+)?\\.dll$")
      set(_plascan_has_onnx_parser TRUE)
    elseif(_plascan_tensorrt_name MATCHES "^nvinfer_plugin(_[0-9]+)?\\.dll$")
      set(_plascan_has_plugin TRUE)
    elseif(_plascan_tensorrt_name MATCHES
           "^nvinfer_builder_resource_.+\\.dll$")
      set(_plascan_has_builder_resource TRUE)
    elseif(_plascan_tensorrt_name MATCHES "^cudart64_.+\\.dll$")
      set(_plascan_has_cudart TRUE)
    elseif(_plascan_tensorrt_name MATCHES "^cublas64_.+\\.dll$")
      set(_plascan_has_cublas TRUE)
    elseif(_plascan_tensorrt_name MATCHES "^cublasLt64_.+\\.dll$")
      set(_plascan_has_cublas_lt TRUE)
    elseif(_plascan_tensorrt_name MATCHES "^nvfatbin_.+\\.dll$")
      set(_plascan_has_nvfatbin TRUE)
    elseif(_plascan_tensorrt_name MATCHES "^nvrtc64_.+\\.dll$")
      set(_plascan_has_nvrtc TRUE)
    elseif(_plascan_tensorrt_name MATCHES "^nvrtc-builtins64_.+\\.dll$")
      set(_plascan_has_nvrtc_builtins TRUE)
    endif()
  endforeach()

  if(NOT _plascan_has_nvinfer OR NOT _plascan_has_onnx_parser OR
     NOT _plascan_has_plugin OR NOT _plascan_has_builder_resource)
    message(FATAL_ERROR
      "TensorRT deployment for ${target_name} is incomplete. Required: "
      "nvinfer, nvonnxparser, nvinfer_plugin, and builder resource DLLs.")
  endif()
  if(NOT _plascan_has_cudart OR NOT _plascan_has_cublas OR
     NOT _plascan_has_cublas_lt OR NOT _plascan_has_nvfatbin OR
     NOT _plascan_has_nvrtc OR NOT _plascan_has_nvrtc_builtins)
    message(FATAL_ERROR
      "CUDA deployment for ${target_name} is incomplete. Required: cudart, "
      "cuBLAS, cuBLAS Lt, nvFatbin, NVRTC, and NVRTC builtins DLLs.")
  endif()

  foreach(_plascan_tensorrt_dll IN LISTS _plascan_tensorrt_runtime_dlls)
    if(EXISTS "${_plascan_tensorrt_dll}")
      add_custom_command(TARGET "${target_name}" POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_plascan_tensorrt_dll}"
          "$<TARGET_FILE_DIR:${target_name}>/"
        VERBATIM)
    endif()
  endforeach()
endfunction()
