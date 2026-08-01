set(VCPKG_POLICY_EMPTY_INCLUDE_FOLDER enabled)

if(NOT EXISTS "/usr/bin/gperf")
    message(FATAL_ERROR "The gperf overlay requires the WSL system package: apt-get install gperf")
endif()

# GNU mirror TLS negotiation fails through the configured proxy.  The Ubuntu
# package provides the same host build tool required by dependent vcpkg ports.
file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/tools/${PORT}")
file(INSTALL "/usr/bin/gperf" DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")
file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(INSTALL "/usr/share/doc/gperf/copyright" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
