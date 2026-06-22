function(plascan_get_windows_test_runtime_path out_var)
    set(runtime_dirs "")

    if(WIN32)
        if(DEFINED Torch_DIR)
            get_filename_component(_plascan_torch_root "${Torch_DIR}/../../.." ABSOLUTE)
            foreach(_plascan_torch_dir
                    "${_plascan_torch_root}/bin"
                    "${_plascan_torch_root}/lib")
                if(EXISTS "${_plascan_torch_dir}")
                    list(APPEND runtime_dirs "${_plascan_torch_dir}")
                endif()
            endforeach()
        endif()

        if(DEFINED CUDAToolkit_BIN_DIR AND EXISTS "${CUDAToolkit_BIN_DIR}")
            list(APPEND runtime_dirs "${CUDAToolkit_BIN_DIR}")
        elseif(DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin")
            list(APPEND runtime_dirs "${CUDAToolkit_ROOT}/bin")
        elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin")
            list(APPEND runtime_dirs "$ENV{CUDA_PATH}/bin")
        endif()

        if(DEFINED ENV{VCPKG_INSTALLED_DIR}
                AND EXISTS "$ENV{VCPKG_INSTALLED_DIR}/x64-windows/bin")
            list(APPEND runtime_dirs "$ENV{VCPKG_INSTALLED_DIR}/x64-windows/bin")
        endif()
    endif()

    list(REMOVE_DUPLICATES runtime_dirs)
    set(${out_var} "${runtime_dirs}" PARENT_SCOPE)
endfunction()

function(plascan_gtest_discover_tests target_name)
    include(GoogleTest)

    plascan_get_windows_test_runtime_path(_plascan_runtime_path)
    if(WIN32 AND _plascan_runtime_path)
        set(_plascan_environment_modifications "")
        foreach(_plascan_runtime_dir IN LISTS _plascan_runtime_path)
            list(APPEND _plascan_environment_modifications
                "PATH=path_list_prepend:${_plascan_runtime_dir}")
        endforeach()

        gtest_discover_tests(${target_name}
            PROPERTIES
                ENVIRONMENT_MODIFICATION "${_plascan_environment_modifications}")
    else()
        gtest_discover_tests(${target_name})
    endif()
endfunction()
