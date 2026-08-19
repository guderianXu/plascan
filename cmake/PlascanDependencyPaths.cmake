include_guard(GLOBAL)

# PlaScan 的 C++ 依赖统一由 manifest 模式的 vcpkg toolchain 提供。
# CUDA 与 TensorRT 是可选的外部 SDK，由各自的标准 CMake 变量显式指定；
# 此处不再读取或拼接任何环境管理器前缀。
function(plascan_configure_dependency_paths)
    if(NOT CMAKE_TOOLCHAIN_FILE)
        message(FATAL_ERROR
            "PlaScan requires the vcpkg toolchain. Configure with a vcpkg preset "
            "and set VCPKG_ROOT, for example: cmake --preset linux-vcpkg-release")
    endif()

    file(TO_CMAKE_PATH "${CMAKE_TOOLCHAIN_FILE}" _toolchainFile)
    if(NOT EXISTS "${_toolchainFile}")
        message(FATAL_ERROR
            "PlaScan vcpkg toolchain does not exist: ${_toolchainFile}")
    endif()

    if(NOT VCPKG_MANIFEST_MODE)
        message(FATAL_ERROR
            "PlaScan requires vcpkg manifest mode. Use the repository presets "
            "or pass the vcpkg toolchain before the first configure.")
    endif()

    if(NOT VCPKG_TARGET_TRIPLET)
        message(FATAL_ERROR
            "PlaScan requires VCPKG_TARGET_TRIPLET to be set by the selected preset.")
    endif()

    set(PLASCAN_DEPENDENCY_PROVIDER_SUMMARY
        "vcpkg(manifest,${VCPKG_TARGET_TRIPLET})"
        PARENT_SCOPE)
endfunction()
