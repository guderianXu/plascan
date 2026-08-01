if(NOT DEFINED ENV{VCPKG_ROOT} OR NOT EXISTS "$ENV{VCPKG_ROOT}/ports/pixman")
    message(FATAL_ERROR "The pixman overlay requires VCPKG_ROOT to point to the vcpkg installation.")
endif()
set(UPSTREAM_PORT_DIR "$ENV{VCPKG_ROOT}/ports/pixman")

# GitLab archive downloads are blocked by Anubis on this network. Fetch the
# dereferenced 0.46.4 release tag through Git and retain the upstream patch.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://gitlab.freedesktop.org/pixman/pixman.git"
    REF 9cc163c9da0fb4da430641715313d95a6ec466d9
    HEAD_REF master
    PATCHES "${UPSTREAM_PORT_DIR}/no-host-cpu-checks.patch"
)

set(x86_architectures x86 x64)
if(VCPKG_TARGET_ARCHITECTURE IN_LIST x86_architectures AND NOT VCPKG_TARGET_IS_UWP)
    list(APPEND OPTIONS -Dmmx=enabled -Dsse2=enabled -Dssse3=enabled)
else()
    list(APPEND OPTIONS -Dmmx=disabled -Dsse2=disabled -Dssse3=disabled)
    if(VCPKG_TARGET_IS_ANDROID)
        vcpkg_cmake_get_vars(cmake_vars_file)
        include("${cmake_vars_file}")
        find_path(cpu_features_dir
            NAMES cpu-features.c
            PATHS "${VCPKG_DETECTED_CMAKE_ANDROID_NDK}"
            PATH_SUFFIXES "sources/android/cpufeatures"
            NO_DEFAULT_PATH
        )
        if(VCPKG_DETECTED_CMAKE_ANDROID_ARM_NEON AND cpu_features_dir)
            list(APPEND OPTIONS "-Dcpu-features-path=${cpu_features_dir}")
        endif()
    endif()
    if(VCPKG_TARGET_IS_WINDOWS)
        list(APPEND OPTIONS -Da64-neon=disabled -Darm-simd=disabled -Dneon=disabled)
    endif()
endif()

vcpkg_configure_meson(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS ${OPTIONS} -Ddemos=disabled -Dgtk=disabled -Dlibpng=enabled -Dtests=disabled
)
vcpkg_install_meson()
vcpkg_fixup_pkgconfig()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")

set(licenses "${SOURCE_PATH}/COPYING")
if(VCPKG_DETECTED_CMAKE_ANDROID_ARM_NEON AND cpu_features_dir)
    file(READ "${cpu_features_dir}/cpu-features.c" cpu_features_c)
    string(REGEX REPLACE "[*]/.*" "*/\n" cpu_features_license "${cpu_features_c}")
    file(WRITE "${CURRENT_PACKAGES_DIR}/${TARGET_TRIPLET}-rel/cpu-features (BSD-2-Clause)" "${cpu_features_license}")
    list(APPEND licenses "${CURRENT_PACKAGES_DIR}/${TARGET_TRIPLET}-rel/cpu-features (BSD-2-Clause)")
endif()
vcpkg_install_copyright(FILE_LIST ${licenses})
