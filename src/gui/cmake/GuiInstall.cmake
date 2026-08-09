# 安装与打包

option(PLASCAN_BUNDLE_RUNTIME "Bundle runtime shared libraries (Qt/OpenCV etc.) into install/package" ON)
option(PLASCAN_BUNDLE_ONNX_MODELS
  "Bundle verified U2Net and portable LightGlue ONNX models into install/package" ON)
option(PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME
  "Fail Linux install/package staging when the bundled runtime is not relocatable" OFF)
option(PLASCAN_LINUX_REQUIRE_XCB_PLUGIN
  "Require the Qt XCB platform plugin in Linux install/package staging" OFF)

set(PLASCAN_U2NET_ONNX_PATH
  "${CMAKE_SOURCE_DIR}/resources/models/U2Net_v1.onnx"
  CACHE FILEPATH "U2Net ONNX model bundled into PlaScan packages")
set(PLASCAN_LIGHTGLUE_ONNX_PATH
  "${CMAKE_SOURCE_DIR}/resources/models/lightglue_tensorrt/lightglue_sift_bucket4096.onnx"
  CACHE FILEPATH "Portable LightGlue ONNX model bundled into PlaScan packages")

install(TARGETS plascan_gui
  RUNTIME DESTINATION bin
  COMPONENT Runtime
)

install(FILES
  "${CMAKE_SOURCE_DIR}/scripts/env/bootstrap_python_runtime.ps1"
  "${CMAKE_SOURCE_DIR}/scripts/env/setup_python_runtime.py"
  "${CMAKE_SOURCE_DIR}/scripts/env/env_common.py"
  DESTINATION share/plascan/scripts/env
  COMPONENT Runtime
)

if(WIN32)
  install(FILES
    "${CMAKE_SOURCE_DIR}/resources/plascan.png"
    "${CMAKE_SOURCE_DIR}/resources/plascan.svg"
    DESTINATION share/plascan
    COMPONENT Runtime
  )
else()
  # 应用图标 — PNG (256x256, 所有桌面环境通用)
  install(FILES "${CMAKE_SOURCE_DIR}/resources/plascan.png"
    DESTINATION /usr/share/icons/hicolor/256x256/apps
    RENAME plascan.png
    COMPONENT Runtime
  )
  # SVG 备选 (供支持矢量图标的桌面环境)
  install(FILES "${CMAKE_SOURCE_DIR}/resources/plascan.svg"
    DESTINATION /usr/share/icons/hicolor/scalable/apps
    RENAME plascan.svg
    COMPONENT Runtime
  )

  # 桌面启动器 (StartupWMClass 必须与 setDesktopFileName 一致)
  install(FILES "${CMAKE_SOURCE_DIR}/resources/plascan.desktop"
    DESTINATION /usr/share/applications
    COMPONENT Runtime
  )
  install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
    DESTINATION /usr/share/doc/plascan
    RENAME copyright
    COMPONENT Runtime
  )
endif()

set(PLASCAN_QT_CONF_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/qt.conf.in")
configure_file("${PLASCAN_QT_CONF_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/qt.conf" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  DESTINATION bin
  COMPONENT Runtime)

if(PLASCAN_BUNDLE_ONNX_MODELS)
  get_filename_component(PLASCAN_BUNDLED_U2NET_ONNX
    "${PLASCAN_U2NET_ONNX_PATH}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  get_filename_component(PLASCAN_BUNDLED_LIGHTGLUE_ONNX
    "${PLASCAN_LIGHTGLUE_ONNX_PATH}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  file(TO_CMAKE_PATH "${PLASCAN_BUNDLED_U2NET_ONNX}" PLASCAN_BUNDLED_U2NET_ONNX)
  file(TO_CMAKE_PATH "${PLASCAN_BUNDLED_LIGHTGLUE_ONNX}" PLASCAN_BUNDLED_LIGHTGLUE_ONNX)

  set(PLASCAN_U2NET_ONNX_SIZE "175997641")
  set(PLASCAN_U2NET_ONNX_SHA256
    "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491")
  set(PLASCAN_LIGHTGLUE_ONNX_SIZE "51072656")
  set(PLASCAN_LIGHTGLUE_ONNX_SHA256
    "773d3de316c37e8d408312d39139352b45e2a93ba055e59cfa2806c5d54ede69")

  foreach(_model_prefix IN ITEMS U2NET LIGHTGLUE)
    if(EXISTS "${PLASCAN_BUNDLED_${_model_prefix}_ONNX}")
      message(STATUS
        "CPack will bundle ${_model_prefix} ONNX: "
        "${PLASCAN_BUNDLED_${_model_prefix}_ONNX}")
    else()
      message(STATUS
        "Bundled ${_model_prefix} ONNX is not present during configure; "
        "install/CPack will require: ${PLASCAN_BUNDLED_${_model_prefix}_ONNX}")
    endif()
  endforeach()

  set(_plascan_verify_models_script
    "${CMAKE_CURRENT_BINARY_DIR}/VerifyBundledModels.cmake")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyBundledModels.cmake.in"
    "${_plascan_verify_models_script}"
    @ONLY)
  install(SCRIPT "${_plascan_verify_models_script}"
    COMPONENT Runtime)

  install(FILES "${PLASCAN_BUNDLED_U2NET_ONNX}"
    DESTINATION resources/models
    RENAME U2Net_v1.onnx
    COMPONENT Runtime)
  install(FILES "${PLASCAN_BUNDLED_LIGHTGLUE_ONNX}"
    DESTINATION resources/models/lightglue_tensorrt
    RENAME lightglue_sift_bucket4096.onnx
    COMPONENT Runtime)
  install(FILES
    "${CMAKE_SOURCE_DIR}/docs/models/Apache-2.0.txt"
    "${CMAKE_SOURCE_DIR}/docs/models/U2Net_NOTICE.md"
    "${CMAKE_SOURCE_DIR}/docs/models/LightGlue_NOTICE.md"
    "${CMAKE_SOURCE_DIR}/docs/models/models-v1.1.0.sha256"
    DESTINATION share/plascan/models
    COMPONENT Runtime)
endif()

if(NOT WIN32)
  set(PLASCAN_LAUNCHER_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/plascan_gui_launcher.sh.in")
  set(PLASCAN_PATH_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/plascan_path.sh.in")

  configure_file("${PLASCAN_LAUNCHER_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan" @ONLY)
  configure_file("${PLASCAN_LAUNCHER_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan_gui" @ONLY)
  configure_file("${PLASCAN_PATH_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan_path.sh" @ONLY)

  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan"
    DESTINATION bin
    COMPONENT Runtime)
  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan_gui"
    DESTINATION bin
    COMPONENT Runtime)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/plascan_path.sh"
    DESTINATION /etc/profile.d
    RENAME plascan.sh
    COMPONENT Runtime)
  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan"
    DESTINATION /usr/bin
    COMPONENT Runtime)
endif()

if(PLASCAN_BUNDLE_RUNTIME AND WIN32)
  set(PLASCAN_WINDOWS_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  if(NOT PLASCAN_WINDOWS_RUNTIME_DIR)
    set(PLASCAN_WINDOWS_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")
  endif()

  set(PLASCAN_WINDOWS_INSTALL_BUNDLE_SCRIPT
    "${CMAKE_CURRENT_BINARY_DIR}/InstallBundledRuntimeWindows.cmake")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/packaging/InstallBundledRuntimeWindows.cmake.in"
    "${PLASCAN_WINDOWS_INSTALL_BUNDLE_SCRIPT}"
    @ONLY
  )
  install(SCRIPT "${PLASCAN_WINDOWS_INSTALL_BUNDLE_SCRIPT}"
    COMPONENT Runtime)
endif()

if(PLASCAN_BUNDLE_RUNTIME AND NOT WIN32)
  set(_plascan_qt_core_lib "")
  foreach(_qt_cfg RELEASE RELWITHDEBINFO MINSIZEREL DEBUG)
    get_target_property(_qt_core_candidate Qt6::Core IMPORTED_LOCATION_${_qt_cfg})
    if(_qt_core_candidate)
      set(_plascan_qt_core_lib "${_qt_core_candidate}")
      break()
    endif()
  endforeach()
  if(NOT _plascan_qt_core_lib)
    get_target_property(_plascan_qt_core_lib Qt6::Core IMPORTED_LOCATION)
  endif()

  if(_plascan_qt_core_lib)
    get_filename_component(_plascan_qt_lib_dir "${_plascan_qt_core_lib}" DIRECTORY)
    get_filename_component(_plascan_qt_prefix_dir "${_plascan_qt_lib_dir}" DIRECTORY)
    set(_plascan_qt_plugin_candidates
      "${_plascan_qt_prefix_dir}/Qt6/plugins"
      "${_plascan_qt_prefix_dir}/plugins"
      "${_plascan_qt_prefix_dir}/lib/qt6/plugins"
      "${_plascan_qt_lib_dir}/qt6/plugins"
    )
    set(PLASCAN_QT_PLUGINS_DIR "")
    foreach(_plugin_candidate IN LISTS _plascan_qt_plugin_candidates)
      if(EXISTS "${_plugin_candidate}")
        set(PLASCAN_QT_PLUGINS_DIR "${_plugin_candidate}")
        break()
      endif()
    endforeach()
  else()
    set(PLASCAN_QT_PLUGINS_DIR "")
  endif()

  set(PLASCAN_LINUX_RUNTIME_PREFIXES "")
  foreach(_runtime_prefix IN ITEMS
      "${PLASCAN_CONDA_PREFIX}"
      "${_plascan_qt_prefix_dir}")
    if(_runtime_prefix AND EXISTS "${_runtime_prefix}")
      get_filename_component(_runtime_prefix "${_runtime_prefix}" REALPATH)
      list(APPEND PLASCAN_LINUX_RUNTIME_PREFIXES "${_runtime_prefix}")
    endif()
  endforeach()
  if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
    set(_vcpkg_runtime_prefix
      "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
    if(EXISTS "${_vcpkg_runtime_prefix}")
      get_filename_component(_vcpkg_runtime_prefix
        "${_vcpkg_runtime_prefix}" REALPATH)
      list(APPEND PLASCAN_LINUX_RUNTIME_PREFIXES
        "${_vcpkg_runtime_prefix}")
    endif()
  endif()
  list(REMOVE_DUPLICATES PLASCAN_LINUX_RUNTIME_PREFIXES)

  set(PLASCAN_LINUX_RUNTIME_LIBRARY_DIRS "")
  foreach(_runtime_prefix IN LISTS PLASCAN_LINUX_RUNTIME_PREFIXES)
    foreach(_library_suffix IN ITEMS lib lib64 bin)
      if(EXISTS "${_runtime_prefix}/${_library_suffix}")
        list(APPEND PLASCAN_LINUX_RUNTIME_LIBRARY_DIRS
          "${_runtime_prefix}/${_library_suffix}")
      endif()
    endforeach()
  endforeach()
  foreach(_system_runtime_dir IN ITEMS
      "${CUDAToolkit_LIBRARY_DIR}"
      "${CUDAToolkit_LIBRARY_ROOT}/lib64"
      "${TensorRT_ROOT}/lib"
      "${TensorRT_ROOT}/lib64")
    if(_system_runtime_dir AND EXISTS "${_system_runtime_dir}")
      list(APPEND PLASCAN_LINUX_RUNTIME_LIBRARY_DIRS
        "${_system_runtime_dir}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES PLASCAN_LINUX_RUNTIME_LIBRARY_DIRS)

  foreach(_plugin_group IN ITEMS
      platforms imageformats tls generic platformthemes)
    if(PLASCAN_QT_PLUGINS_DIR AND
       EXISTS "${PLASCAN_QT_PLUGINS_DIR}/${_plugin_group}")
      install(
        DIRECTORY "${PLASCAN_QT_PLUGINS_DIR}/${_plugin_group}/"
        DESTINATION "plugins/${_plugin_group}"
        COMPONENT Runtime
        FILES_MATCHING
        PATTERN "libq*.so*"
      )
    endif()
  endforeach()

  if(PLASCAN_LINUX_REQUIRE_XCB_PLUGIN AND
     (NOT PLASCAN_QT_PLUGINS_DIR OR
      NOT EXISTS "${PLASCAN_QT_PLUGINS_DIR}/platforms/libqxcb.so"))
    message(FATAL_ERROR
      "Linux package requires Qt's XCB platform plugin. Rebuild qtbase with "
      "the xcb, xkbcommon-x11, fontconfig and dbus features.")
  endif()

  set(PLASCAN_GDAL_DATA_SOURCE "")
  set(PLASCAN_PROJ_DATA_SOURCE "")
  foreach(_runtime_prefix IN LISTS PLASCAN_LINUX_RUNTIME_PREFIXES)
    if(NOT PLASCAN_GDAL_DATA_SOURCE AND
       EXISTS "${_runtime_prefix}/share/gdal")
      set(PLASCAN_GDAL_DATA_SOURCE "${_runtime_prefix}/share/gdal")
    endif()
    if(NOT PLASCAN_PROJ_DATA_SOURCE AND
       EXISTS "${_runtime_prefix}/share/proj/proj.db")
      set(PLASCAN_PROJ_DATA_SOURCE "${_runtime_prefix}/share/proj")
    endif()
  endforeach()

  if(PLASCAN_GDAL_DATA_SOURCE)
    install(DIRECTORY "${PLASCAN_GDAL_DATA_SOURCE}/"
      DESTINATION share/gdal
      COMPONENT Runtime
      PATTERN "*.cmake" EXCLUDE
      PATTERN "vcpkg.*" EXCLUDE)
  elseif(PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME)
    message(FATAL_ERROR "Linux package requires the GDAL data directory")
  endif()
  if(PLASCAN_PROJ_DATA_SOURCE)
    install(DIRECTORY "${PLASCAN_PROJ_DATA_SOURCE}/"
      DESTINATION share/proj
      COMPONENT Runtime
      PATTERN "*.cmake" EXCLUDE
      PATTERN "vcpkg.*" EXCLUDE)
  elseif(PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME)
    message(FATAL_ERROR "Linux package requires the PROJ data directory and proj.db")
  endif()

  set(PLASCAN_INSTALL_BUNDLE_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallBundledRuntime.cmake")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/packaging/InstallBundledRuntime.cmake.in"
    "${PLASCAN_INSTALL_BUNDLE_SCRIPT}"
    @ONLY
  )
  install(SCRIPT "${PLASCAN_INSTALL_BUNDLE_SCRIPT}"
    COMPONENT Runtime)

  if(PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME)
    set(PLASCAN_VERIFY_LINUX_RUNTIME_SCRIPT
      "${CMAKE_CURRENT_BINARY_DIR}/VerifyLinuxPackageRuntime.cmake")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/packaging/VerifyLinuxPackageRuntime.cmake.in"
      "${PLASCAN_VERIFY_LINUX_RUNTIME_SCRIPT}"
      @ONLY)
    install(SCRIPT "${PLASCAN_VERIFY_LINUX_RUNTIME_SCRIPT}"
      COMPONENT Runtime)
  endif()
endif()

message(STATUS "plascan_gui configuration complete")
