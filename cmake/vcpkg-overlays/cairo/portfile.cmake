if(NOT DEFINED ENV{VCPKG_ROOT} OR NOT EXISTS "$ENV{VCPKG_ROOT}/ports/cairo")
    message(FATAL_ERROR "The cairo overlay requires VCPKG_ROOT to point to the vcpkg installation.")
endif()
set(UPSTREAM_PORT_DIR "$ENV{VCPKG_ROOT}/ports/cairo")

set(EXTRA_PATCHES "")
if(VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW)
    list(APPEND EXTRA_PATCHES "${UPSTREAM_PORT_DIR}/fix_clang-cl_build.patch")
endif()

# GitLab archive downloads are blocked by Anubis on this network. Fetch the
# dereferenced 1.18.4 release tag through Git and retain the upstream patches.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://gitlab.freedesktop.org/cairo/cairo.git"
    REF 4541e0cd3a751b85e52e2a83d02ac6145a5efa85
    HEAD_REF master
    PATCHES "${UPSTREAM_PORT_DIR}/msvc-convenience.diff" ${EXTRA_PATCHES}
)

if("fontconfig" IN_LIST FEATURES)
    list(APPEND OPTIONS -Dfontconfig=enabled)
else()
    list(APPEND OPTIONS -Dfontconfig=disabled)
endif()
if("freetype" IN_LIST FEATURES)
    list(APPEND OPTIONS -Dfreetype=enabled)
else()
    list(APPEND OPTIONS -Dfreetype=disabled)
endif()
if("x11" IN_LIST FEATURES)
    message(WARNING "X11 support requires system packages: libx11-dev libxft-dev libxext-dev")
    list(APPEND OPTIONS -Dxlib=enabled)
else()
    list(APPEND OPTIONS -Dxlib=disabled)
endif()
list(APPEND OPTIONS -Dxcb=disabled -Dxlib-xcb=disabled)
if("gobject" IN_LIST FEATURES)
    list(APPEND OPTIONS -Dglib=enabled)
else()
    list(APPEND OPTIONS -Dglib=disabled)
endif()
if("lzo" IN_LIST FEATURES)
    list(APPEND OPTIONS -Dlzo=enabled)
else()
    list(APPEND OPTIONS -Dlzo=disabled)
endif()

vcpkg_configure_meson(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS ${OPTIONS} -Dtests=disabled -Dzlib=enabled -Dpng=enabled -Dspectre=auto -Dgtk2-utils=disabled -Dsymbol-lookup=disabled
)
vcpkg_install_meson()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/include/cairo/cairo.h" "defined(CAIRO_WIN32_STATIC_BUILD)" "1")
endif()
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()
if(VCPKG_LIBRARY_LINKAGE STREQUAL "static" OR NOT VCPKG_TARGET_IS_WINDOWS)
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin" "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING" "${SOURCE_PATH}/COPYING-LGPL-2.1" "${SOURCE_PATH}/COPYING-MPL-1.1")
