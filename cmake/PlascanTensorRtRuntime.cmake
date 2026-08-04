function(plascan_deploy_tensorrt_runtime target_name)
  if(NOT WIN32 OR NOT TARGET "${target_name}" OR NOT TensorRT_RUNTIME_LIBRARIES)
    return()
  endif()

  foreach(_plascan_tensorrt_dll IN LISTS TensorRT_RUNTIME_LIBRARIES)
    if(EXISTS "${_plascan_tensorrt_dll}")
      add_custom_command(TARGET "${target_name}" POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_plascan_tensorrt_dll}"
          "$<TARGET_FILE_DIR:${target_name}>/"
        VERBATIM)
    endif()
  endforeach()
endfunction()
