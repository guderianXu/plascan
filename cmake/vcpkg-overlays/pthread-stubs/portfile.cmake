set(VCPKG_POLICY_EMPTY_INCLUDE_FOLDER enabled)

# GitLab archive downloads are blocked by Anubis on this network. Fetch the
# dereferenced 0.5 release tag through Git instead.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://gitlab.freedesktop.org/xorg/lib/pthread-stubs.git"
    REF 9c8d612698c78f9775b37c5d901a0748b9e4b4b6
    HEAD_REF master
)

vcpkg_make_configure(SOURCE_PATH "${SOURCE_PATH}" AUTORECONF)
vcpkg_make_install()
vcpkg_fixup_pkgconfig(SYSTEM_LIBRARIES pthread)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")

function(fix_pthread_stubs_pc pc_file library_dir)
    file(READ "${pc_file}" contents)
    string(REPLACE "Cflags: -pthread" "Cflags: " contents "${contents}")
    if(EXISTS "${library_dir}/pthreadVC3.lib")
        string(REPLACE "Libs: -pthread" "Libs: -lpthreadVC3" contents "${contents}")
    endif()
    if(EXISTS "${library_dir}/pthreadGC3.lib")
        string(REPLACE "Libs: -pthread" "Libs: -lpthreadGC3" contents "${contents}")
    endif()
    if(EXISTS "${library_dir}/pthreadVC3d.lib")
        string(REPLACE "Libs: -pthread" "Libs: -lpthreadVC3d" contents "${contents}")
    endif()
    if(EXISTS "${library_dir}/pthreadGC3d.lib")
        string(REPLACE "Libs: -pthread" "Libs: -lpthreadGC3d" contents "${contents}")
    endif()
    file(WRITE "${pc_file}" "${contents}")
endfunction()

fix_pthread_stubs_pc(
    "${CURRENT_PACKAGES_DIR}/lib/pkgconfig/pthread-stubs.pc"
    "${CURRENT_INSTALLED_DIR}/lib"
)
if(NOT VCPKG_BUILD_TYPE)
    fix_pthread_stubs_pc(
        "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig/pthread-stubs.pc"
        "${CURRENT_INSTALLED_DIR}/debug/lib"
    )
endif()
