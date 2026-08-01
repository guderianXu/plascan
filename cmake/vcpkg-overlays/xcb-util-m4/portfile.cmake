set(VCPKG_POLICY_EMPTY_INCLUDE_FOLDER enabled)

# GitLab's archive endpoint is protected by Anubis in some network environments.
# Fetch the same pinned upstream commit through Git instead of weakening source
# verification or accepting the HTML denial response as an archive.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://gitlab.freedesktop.org/xorg/util/xcb-util-m4.git"
    REF c617eee22ae5c285e79e81ec39ce96862fd3262f
    HEAD_REF master
)

file(GLOB_RECURSE M4_FILES "${SOURCE_PATH}/*.m4")
file(INSTALL ${M4_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/share/xorg/aclocal")
file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(TOUCH "${CURRENT_PACKAGES_DIR}/share/xcb-util-m4/copyright")
