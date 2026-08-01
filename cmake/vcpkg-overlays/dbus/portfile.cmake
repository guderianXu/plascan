vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

if(NOT DEFINED ENV{VCPKG_ROOT} OR NOT EXISTS "$ENV{VCPKG_ROOT}/ports/dbus")
    message(FATAL_ERROR "The dbus overlay requires VCPKG_ROOT to point to the vcpkg installation.")
endif()
set(UPSTREAM_PORT_DIR "$ENV{VCPKG_ROOT}/ports/dbus")

# GitLab archive downloads are blocked by Anubis on this network. Fetch the
# dereferenced dbus-1.16.2 release tag through Git and retain upstream patches.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://gitlab.freedesktop.org/dbus/dbus.git"
    REF 958bf9db2100553bcd2fe2a854e1ebb42e886054
    HEAD_REF master
    PATCHES
        "${UPSTREAM_PORT_DIR}/cmake.dep.patch"
        "${UPSTREAM_PORT_DIR}/pkgconfig.patch"
        "${UPSTREAM_PORT_DIR}/getpeereid.patch"
        "${UPSTREAM_PORT_DIR}/libsystemd.patch"
        "${UPSTREAM_PORT_DIR}/remove-path.patch"
        "${UPSTREAM_PORT_DIR}/remove-var-lib-dbus-creation.patch"
        "${UPSTREAM_PORT_DIR}/session-socket-dir.diff"
)

vcpkg_check_features(OUT_FEATURE_OPTIONS options
    FEATURES
        systemd ENABLE_SYSTEMD
        x11     DBUS_BUILD_X11
        x11     CMAKE_REQUIRE_FIND_PACKAGE_X11
)

if(NOT VCPKG_TARGET_IS_WINDOWS)
    list(APPEND options "-DDBUS_RUNSTATEDIR=/run")
endif()

unset(ENV{DBUSDIR})

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DDBUS_BUILD_TESTS=OFF
        -DDBUS_ENABLE_DOXYGEN_DOCS=OFF
        -DDBUS_ENABLE_XML_DOCS=OFF
        -DDBUS_INSTALL_SYSTEM_LIBS=OFF
        -DDBUS_WITH_GLIB=OFF
        -DTHREADS_PREFER_PTHREAD_FLAG=ON
        -DXSLTPROC_EXECUTABLE=FALSE
        "-DCMAKE_INSTALL_SYSCONFDIR=${CURRENT_PACKAGES_DIR}/etc/${PORT}"
        "-DWITH_SYSTEMD_SYSTEMUNITDIR=lib/systemd/system"
        "-DWITH_SYSTEMD_USERUNITDIR=lib/systemd/user"
        ${options}
    OPTIONS_RELEASE
        -DDBUS_DISABLE_ASSERT=OFF
        -DDBUS_ENABLE_STATS=OFF
        -DDBUS_ENABLE_VERBOSE_MODE=OFF
    MAYBE_UNUSED_VARIABLES
        DBUS_BUILD_X11
        DBUS_WITH_GLIB
        ENABLE_SYSTEMD
        THREADS_PREFER_PTHREAD_FLAG
        WITH_SYSTEMD_SYSTEMUNITDIR
        WITH_SYSTEMD_USERUNITDIR
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(PACKAGE_NAME "DBus1" CONFIG_PATH "lib/cmake/DBus1")
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
    "${CURRENT_PACKAGES_DIR}/debug/var/"
    "${CURRENT_PACKAGES_DIR}/etc"
    "${CURRENT_PACKAGES_DIR}/share/dbus-1/services"
    "${CURRENT_PACKAGES_DIR}/share/dbus-1/session.d"
    "${CURRENT_PACKAGES_DIR}/share/dbus-1/system-services"
    "${CURRENT_PACKAGES_DIR}/share/dbus-1/system.d"
    "${CURRENT_PACKAGES_DIR}/share/dbus-1/system.conf"
    "${CURRENT_PACKAGES_DIR}/share/doc"
    "${CURRENT_PACKAGES_DIR}/var"
)

vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/dbus-1/session.conf" "<include ignore_missing=\"yes\">${CURRENT_PACKAGES_DIR}/etc/dbus/dbus-1/session.conf</include>" "")
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/dbus-1/session.conf" "<includedir>${CURRENT_PACKAGES_DIR}/etc/dbus/dbus-1/session.d</includedir>" "")
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/dbus-1/session.conf" "<include ignore_missing=\"yes\">${CURRENT_PACKAGES_DIR}/etc/dbus/dbus-1/session-local.conf</include>" "")

set(TOOLS daemon launch monitor run-session send test-tool update-activation-environment)
if(VCPKG_TARGET_IS_WINDOWS)
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/tools/${PORT}")
    file(RENAME "${CURRENT_PACKAGES_DIR}/bin/dbus-env.bat" "${CURRENT_PACKAGES_DIR}/tools/${PORT}/dbus-env.bat")
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/tools/${PORT}/dbus-env.bat" "${CURRENT_PACKAGES_DIR}" "%~dp0/../..")
else()
    list(APPEND TOOLS cleanup-sockets uuidgen)
endif()
list(TRANSFORM TOOLS PREPEND "dbus-")
if(NOT DEFINED VCPKG_BUILD_TYPE OR VCPKG_BUILD_TYPE STREQUAL "debug")
    vcpkg_copy_tools(TOOL_NAMES ${TOOLS} SEARCH_DIR ${CURRENT_PACKAGES_DIR}/debug/bin DESTINATION "${CURRENT_PACKAGES_DIR}/debug/tools/${PORT}")
endif()
vcpkg_copy_tools(TOOL_NAMES ${TOOLS} AUTO_CLEAN)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
