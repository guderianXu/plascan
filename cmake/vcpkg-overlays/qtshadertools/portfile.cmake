set(SCRIPT_PATH "${CURRENT_INSTALLED_DIR}/share/qtbase")
include("${SCRIPT_PATH}/qt_install_submodule.cmake")

set(${PORT}_PATCHES)

set(TOOL_NAMES qsb)

# qtbase itself builds successfully with the macOS Command Line Tools SDK, but
# Qt's submodule helper otherwise requires a full Xcode version query before it
# will configure ShaderTools. Keep the SDK and real compile/link checks and
# disable only that redundant Xcode minimum-version gate.
qt_install_submodule(
    PATCHES ${${PORT}_PATCHES}
    TOOL_NAMES ${TOOL_NAMES}
    CONFIGURE_OPTIONS
        -DQT_NO_XCODE_MIN_VERSION_CHECK=ON
    CONFIGURE_OPTIONS_RELEASE
    CONFIGURE_OPTIONS_DEBUG
)
