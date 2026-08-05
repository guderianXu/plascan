function(plascan_deploy_vcpkg_runtime target_name)
  if(NOT WIN32)
    return()
  endif()

  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR
      "plascan_deploy_vcpkg_runtime: target does not exist: ${target_name}")
  endif()

  set(_plascan_vcpkg_triplet "${VCPKG_TARGET_TRIPLET}")
  if(NOT _plascan_vcpkg_triplet)
    set(_plascan_vcpkg_triplet "x64-windows")
  endif()

  set(_plascan_vcpkg_roots)
  if(VCPKG_INSTALLED_DIR)
    list(APPEND _plascan_vcpkg_roots "${VCPKG_INSTALLED_DIR}")
  endif()
  if(DEFINED ENV{VCPKG_INSTALLED_DIR} AND NOT "$ENV{VCPKG_INSTALLED_DIR}" STREQUAL "")
    list(APPEND _plascan_vcpkg_roots "$ENV{VCPKG_INSTALLED_DIR}")
  endif()
  list(APPEND _plascan_vcpkg_roots "${CMAKE_BINARY_DIR}/vcpkg_installed")
  list(REMOVE_DUPLICATES _plascan_vcpkg_roots)

  set(_plascan_vcpkg_runtime_dir "")
  foreach(_plascan_vcpkg_root IN LISTS _plascan_vcpkg_roots)
    foreach(_plascan_vcpkg_candidate IN ITEMS
        "${_plascan_vcpkg_root}/${_plascan_vcpkg_triplet}/bin"
        "${_plascan_vcpkg_root}/bin")
      if(IS_DIRECTORY "${_plascan_vcpkg_candidate}")
        set(_plascan_vcpkg_runtime_dir "${_plascan_vcpkg_candidate}")
        break()
      endif()
    endforeach()
    if(_plascan_vcpkg_runtime_dir)
      break()
    endif()
  endforeach()

  if(NOT _plascan_vcpkg_runtime_dir)
    message(STATUS
      "plascan: no vcpkg runtime directory found for ${target_name}; skipping DLL deployment")
    return()
  endif()

  file(GLOB _plascan_vcpkg_runtime_dlls
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    "${_plascan_vcpkg_runtime_dir}/*.dll")
  if(NOT _plascan_vcpkg_runtime_dlls)
    message(STATUS
      "plascan: no vcpkg runtime DLLs found in ${_plascan_vcpkg_runtime_dir}")
    return()
  endif()

  list(LENGTH _plascan_vcpkg_runtime_dlls _plascan_vcpkg_runtime_dll_count)
  add_custom_command(TARGET "${target_name}" POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      ${_plascan_vcpkg_runtime_dlls}
      "$<TARGET_FILE_DIR:${target_name}>"
    COMMAND_EXPAND_LISTS
    VERBATIM
    COMMENT
      "Deploying ${_plascan_vcpkg_runtime_dll_count} vcpkg runtime DLLs for ${target_name}")
endfunction()
