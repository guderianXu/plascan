if(NOT DEFINED ENV{VCPKG_ROOT} OR NOT EXISTS "$ENV{VCPKG_ROOT}/ports/fontconfig")
    message(FATAL_ERROR "The fontconfig overlay requires VCPKG_ROOT to point to the vcpkg installation.")
endif()
set(UPSTREAM_PORT_DIR "$ENV{VCPKG_ROOT}/ports/fontconfig")

# GitLab archive downloads are blocked by Anubis on this network.  Fetch the
# dereferenced 2.17.1 release tag through Git and retain the upstream patches.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://gitlab.freedesktop.org/fontconfig/fontconfig.git"
    REF 6d0a98982ec351c165c9224c8b7dbdfca3010e47
    HEAD_REF master
    PATCHES
        "${UPSTREAM_PORT_DIR}/dllexport.diff"
        "${UPSTREAM_PORT_DIR}/no-etc-symlinks.patch"
        "${UPSTREAM_PORT_DIR}/libgetopt.patch"
        "${UPSTREAM_PORT_DIR}/libintl.diff"
        "${UPSTREAM_PORT_DIR}/fix-wasm-shared-memory-atomics.patch"
)

set(options "")
if("iconv" IN_LIST FEATURES)
    list(APPEND options "-Diconv=enabled")
else()
    list(APPEND options "-Diconv=disabled")
endif()
if("nls" IN_LIST FEATURES)
    list(APPEND options "-Dnls=enabled")
else()
    list(APPEND options "-Dnls=disabled")
endif()
if("tools" IN_LIST FEATURES)
    list(APPEND options "-Dtools=enabled")
else()
    list(APPEND options "-Dtools=disabled")
endif()

vcpkg_configure_meson(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS ${options} -Ddoc=disabled -Dcache-build=disabled -Dxml-backend=expat -Dtests=disabled
    ADDITIONAL_BINARIES "gperf = ['${CURRENT_HOST_INSTALLED_DIR}/tools/gperf/gperf${VCPKG_HOST_EXECUTABLE_SUFFIX}']"
)

set(replacement "")
if(VCPKG_TARGET_IS_WINDOWS)
    set(replacement "**invalid-fontconfig-dir-do-not-use**")
endif()
set(configfile "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/meson-config.h")
vcpkg_replace_string("${configfile}" "${CURRENT_PACKAGES_DIR}" "${replacement}")
if(NOT VCPKG_BUILD_TYPE)
    set(configfile "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-dbg/meson-config.h")
    vcpkg_replace_string("${configfile}" "${CURRENT_PACKAGES_DIR}/debug" "${replacement}")
endif()

vcpkg_install_meson(ADD_BIN_TO_PATH)
vcpkg_copy_pdbs()
if("nls" IN_LIST FEATURES AND VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW)
    if(NOT VCPKG_BUILD_TYPE)
        vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig/fontconfig.pc" "-liconv" "-liconv -lintl" IGNORE_UNCHANGED)
    endif()
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/lib/pkgconfig/fontconfig.pc" "-liconv" "-liconv -lintl" IGNORE_UNCHANGED)
endif()
vcpkg_fixup_pkgconfig()
if(NOT VCPKG_BUILD_TYPE)
    set(fontconfig_pc_debug "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig/fontconfig.pc")
    vcpkg_replace_string("${fontconfig_pc_debug}" "/etc" "/../etc" REGEX)
    vcpkg_replace_string("${fontconfig_pc_debug}" "/var" "/../var" REGEX)
endif()
set(_file "${CURRENT_PACKAGES_DIR}/etc/fonts/fonts.conf")
if(EXISTS "${_file}")
    vcpkg_replace_string("${_file}" "${CURRENT_PACKAGES_DIR}/var/cache/fontconfig" "./../../var/cache/fontconfig" IGNORE_UNCHANGED)
endif()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/var" "${CURRENT_PACKAGES_DIR}/debug/share" "${CURRENT_PACKAGES_DIR}/debug/etc" "${CURRENT_PACKAGES_DIR}/var")
if("tools" IN_LIST FEATURES)
    vcpkg_copy_tools(TOOL_NAMES fc-match fc-cat fc-list fc-pattern fc-query fc-scan fc-cache fc-validate fc-conflist AUTO_CLEAN)
endif()
configure_file("${UPSTREAM_PORT_DIR}/vcpkg-cmake-wrapper.cmake.in" "${CURRENT_PACKAGES_DIR}/share/${PORT}/vcpkg-cmake-wrapper.cmake" @ONLY)
file(INSTALL "${UPSTREAM_PORT_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
