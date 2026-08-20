# 安装与打包

if(WIN32)
  set(_plascan_bundle_runtime_default ON)
else()
  set(_plascan_bundle_runtime_default OFF)
endif()
option(PLASCAN_BUNDLE_RUNTIME
  "Bundle runtime shared libraries (Qt/OpenCV etc.) into install/package"
  ${_plascan_bundle_runtime_default})
unset(_plascan_bundle_runtime_default)
option(PLASCAN_BUNDLE_ONNX_MODELS
  "Bundle verified U2Net and portable LightGlue ONNX models into install/package" ON)
option(PLASCAN_BUNDLE_LOMA_R_MODELS
  "Bundle verified portable LoMa-R ONNX models and manifests into install/package" OFF)
option(PLASCAN_BUNDLE_BIREFNET_DYNAMIC
  "Bundle verified BiRefNet Dynamic ONNX and provenance into TensorRT packages" OFF)
option(PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME
  "Fail Linux install/package staging when the bundled runtime is not relocatable" OFF)
option(PLASCAN_LINUX_REQUIRE_XCB_PLUGIN
  "Require the Qt XCB platform plugin in Linux install/package staging" OFF)
option(PLASCAN_LINUX_REQUIRE_TLS_PLUGIN
  "Require the Qt OpenSSL TLS backend in Linux install/package staging" OFF)

if(PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME)
  if(WIN32 OR APPLE)
    message(FATAL_ERROR
      "PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME is only valid on Linux")
  endif()
  if(NOT PLASCAN_BUNDLE_RUNTIME OR NOT PLASCAN_BUNDLE_ONNX_MODELS)
    message(FATAL_ERROR
      "Verified Linux packages require PLASCAN_BUNDLE_RUNTIME=ON and "
      "PLASCAN_BUNDLE_ONNX_MODELS=ON")
  endif()
  if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/opt/plascan")
    message(FATAL_ERROR
      "Verified Linux DEB builds require CMAKE_INSTALL_PREFIX=/opt/plascan: "
      "${CMAKE_INSTALL_PREFIX}")
  endif()
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR
      "The verified Linux DEB presets currently support x86_64 only: "
      "${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  if(NOT EXISTS "/etc/os-release")
    message(FATAL_ERROR "Cannot verify the Linux distribution without /etc/os-release")
  endif()
  file(READ "/etc/os-release" _plascan_os_release)
  if(NOT _plascan_os_release MATCHES "(^|\n)ID=ubuntu(\n|$)" OR
     NOT _plascan_os_release MATCHES "(^|\n)VERSION_ID=\"?24[.]04\"?(\n|$)")
    message(FATAL_ERROR
      "Verified Linux DEB builds must run on Ubuntu 24.04 x86_64. "
      "Detected /etc/os-release:\n${_plascan_os_release}")
  endif()
endif()

set(PLASCAN_U2NET_ONNX_PATH
  "${CMAKE_SOURCE_DIR}/resources/models/U2Net_v1.onnx"
  CACHE FILEPATH "U2Net ONNX model bundled into PlaScan packages")
set(PLASCAN_LIGHTGLUE_ONNX_PATH
  "${CMAKE_SOURCE_DIR}/resources/models/lightglue_tensorrt/lightglue_sift_bucket4096.onnx"
  CACHE FILEPATH "Portable LightGlue ONNX model bundled into PlaScan packages")
set(PLASCAN_LOMA_R_MODEL_DIR
  "${CMAKE_SOURCE_DIR}/resources/models/loma_r_tensorrt"
  CACHE PATH "Directory containing portable LoMa-R ONNX models and manifests")
set(PLASCAN_BIREFNET_DYNAMIC_MODEL_DIR
  "${CMAKE_SOURCE_DIR}/resources/models/birefnet_dynamic"
  CACHE PATH "Directory containing BiRefNet Dynamic ONNX and provenance assets")

if(PLASCAN_BUNDLE_LOMA_R_MODELS AND NOT PLASCAN_BUNDLE_ONNX_MODELS)
  message(FATAL_ERROR
    "PLASCAN_BUNDLE_LOMA_R_MODELS requires PLASCAN_BUNDLE_ONNX_MODELS=ON")
endif()

if(PLASCAN_BUNDLE_BIREFNET_DYNAMIC)
  if(NOT PLASCAN_BUNDLE_ONNX_MODELS)
    message(FATAL_ERROR
      "PLASCAN_BUNDLE_BIREFNET_DYNAMIC requires PLASCAN_BUNDLE_ONNX_MODELS=ON")
  endif()
  if(NOT PLASCAN_ENABLE_TENSORRT)
    message(FATAL_ERROR
      "PLASCAN_BUNDLE_BIREFNET_DYNAMIC requires PLASCAN_ENABLE_TENSORRT=ON")
  endif()
endif()

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
  if(PLASCAN_DEBIAN_PACKAGE_VARIANT STREQUAL "cuda")
    set(_plascan_debian_doc_dir /usr/share/doc/plascan-cuda)
  else()
    set(_plascan_debian_doc_dir /usr/share/doc/plascan)
  endif()
  install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
    DESTINATION "${_plascan_debian_doc_dir}"
    RENAME copyright
    COMPONENT Runtime
  )
  unset(_plascan_debian_doc_dir)
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

  if(PLASCAN_BUNDLE_LOMA_R_MODELS)
    get_filename_component(PLASCAN_BUNDLED_LOMA_R_MODEL_DIR
      "${PLASCAN_LOMA_R_MODEL_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    file(TO_CMAKE_PATH
      "${PLASCAN_BUNDLED_LOMA_R_MODEL_DIR}"
      PLASCAN_BUNDLED_LOMA_R_MODEL_DIR)
    set(PLASCAN_BUNDLED_LOMA_R_FEATURE_ONNX
      "${PLASCAN_BUNDLED_LOMA_R_MODEL_DIR}/loma_r_features_k3840_fp16.onnx")
    set(PLASCAN_BUNDLED_LOMA_R_MATCHER_ONNX
      "${PLASCAN_BUNDLED_LOMA_R_MODEL_DIR}/loma_r_matcher_dynamic_fp16.onnx")
    set(PLASCAN_BUNDLED_LOMA_R_MANIFEST_K1024
      "${PLASCAN_BUNDLED_LOMA_R_MODEL_DIR}/loma_r_k1024_fp16.json")
    set(PLASCAN_BUNDLED_LOMA_R_MANIFEST_K2048
      "${PLASCAN_BUNDLED_LOMA_R_MODEL_DIR}/loma_r_k2048_fp16.json")
    set(PLASCAN_BUNDLED_LOMA_R_MANIFEST_K3840
      "${PLASCAN_BUNDLED_LOMA_R_MODEL_DIR}/loma_r_k3840_fp16.json")
  endif()

  if(PLASCAN_BUNDLE_BIREFNET_DYNAMIC)
    get_filename_component(PLASCAN_BUNDLED_BIREFNET_DYNAMIC_MODEL_DIR
      "${PLASCAN_BIREFNET_DYNAMIC_MODEL_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    file(TO_CMAKE_PATH
      "${PLASCAN_BUNDLED_BIREFNET_DYNAMIC_MODEL_DIR}"
      PLASCAN_BUNDLED_BIREFNET_DYNAMIC_MODEL_DIR)
    set(PLASCAN_BUNDLED_BIREFNET_DYNAMIC_ONNX
      "${PLASCAN_BUNDLED_BIREFNET_DYNAMIC_MODEL_DIR}/BiRefNet_dynamic_1024.onnx")
    set(PLASCAN_BUNDLED_BIREFNET_DYNAMIC_PROVENANCE
      "${PLASCAN_BUNDLED_BIREFNET_DYNAMIC_MODEL_DIR}/BiRefNet_dynamic_1024.provenance.json")
  endif()

  set(PLASCAN_U2NET_ONNX_SIZE "175997641")
  set(PLASCAN_U2NET_ONNX_SHA256
    "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491")
  set(PLASCAN_LIGHTGLUE_ONNX_SIZE "51072656")
  set(PLASCAN_LIGHTGLUE_ONNX_SHA256
    "773d3de316c37e8d408312d39139352b45e2a93ba055e59cfa2806c5d54ede69")
  set(PLASCAN_LOMA_R_FEATURE_ONNX_SIZE "1318960639")
  set(PLASCAN_LOMA_R_FEATURE_ONNX_SHA256
    "2b2671850f6a79f071a171eb9b523a8807474bcde19b5ded0191b9593ed97e19")
  set(PLASCAN_LOMA_R_MATCHER_ONNX_SIZE "45501499")
  set(PLASCAN_LOMA_R_MATCHER_ONNX_SHA256
    "5c91444393c2245e66553e8f493e5b35dc39e8a099b9988a684391fdcdf90195")
  set(PLASCAN_LOMA_R_MANIFEST_SIZE "644")
  set(PLASCAN_LOMA_R_MANIFEST_K1024_SHA256
    "db3b242ed7cda10e16fd7c304844c1f809a3b37bc40af10a81c7248ca9e51aea")
  set(PLASCAN_LOMA_R_MANIFEST_K2048_SHA256
    "68ae6a68bb184375285d486384344b7a6500195d5f372e96c8b429bd8787c91e")
  set(PLASCAN_LOMA_R_MANIFEST_K3840_SHA256
    "5d55026fe3e0bc59bb93bc997d928ec46940905e22c7836b831a859e1dae2715")
  set(PLASCAN_BIREFNET_DYNAMIC_ONNX_SIZE "972558911")
  set(PLASCAN_BIREFNET_DYNAMIC_ONNX_SHA256
    "3af7fe29f80be80e12595671293c877af6767cae71566a8765face68965f0742")
  set(PLASCAN_BIREFNET_DYNAMIC_PROVENANCE_SIZE "1688")
  set(PLASCAN_BIREFNET_DYNAMIC_PROVENANCE_SHA256
    "9e100509b59aedfeabd0aabc7277009b0d620803b27f482abb2e28220de8d4ff")

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

  if(PLASCAN_BUNDLE_LOMA_R_MODELS)
    foreach(_loma_r_asset IN ITEMS
        PLASCAN_BUNDLED_LOMA_R_FEATURE_ONNX
        PLASCAN_BUNDLED_LOMA_R_MATCHER_ONNX
        PLASCAN_BUNDLED_LOMA_R_MANIFEST_K1024
        PLASCAN_BUNDLED_LOMA_R_MANIFEST_K2048
        PLASCAN_BUNDLED_LOMA_R_MANIFEST_K3840)
      if(EXISTS "${${_loma_r_asset}}")
        message(STATUS "CPack will bundle LoMa-R asset: ${${_loma_r_asset}}")
      else()
        message(STATUS
          "Bundled LoMa-R asset is not present during configure; "
          "install/CPack will require: ${${_loma_r_asset}}")
      endif()
    endforeach()
  endif()

  if(PLASCAN_BUNDLE_BIREFNET_DYNAMIC)
    foreach(_birefnet_asset IN ITEMS
        PLASCAN_BUNDLED_BIREFNET_DYNAMIC_ONNX
        PLASCAN_BUNDLED_BIREFNET_DYNAMIC_PROVENANCE)
      if(EXISTS "${${_birefnet_asset}}")
        message(STATUS "CPack will bundle BiRefNet Dynamic asset: ${${_birefnet_asset}}")
      else()
        message(STATUS
          "Bundled BiRefNet Dynamic asset is not present during configure; "
          "install/CPack will require: ${${_birefnet_asset}}")
      endif()
    endforeach()
  endif()

  set(_plascan_verify_models_script
    "${CMAKE_CURRENT_BINARY_DIR}/VerifyBundledModels.cmake")
  set(PLASCAN_VERIFY_BUNDLE_ONNX_MODELS ON)
  set(PLASCAN_VERIFY_BUNDLE_LOMA_R_MODELS OFF)
  set(PLASCAN_VERIFY_BUNDLE_BIREFNET_DYNAMIC OFF)
  set(PLASCAN_VERIFY_INSTALLED_BIREFNET_DYNAMIC OFF)
  if(PLASCAN_BUNDLE_LOMA_R_MODELS)
    set(PLASCAN_VERIFY_BUNDLE_LOMA_R_MODELS ON)
  endif()
  if(PLASCAN_BUNDLE_BIREFNET_DYNAMIC)
    set(PLASCAN_VERIFY_BUNDLE_BIREFNET_DYNAMIC ON)
  endif()
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
  if(PLASCAN_BUNDLE_LOMA_R_MODELS)
    install(FILES
      "${PLASCAN_BUNDLED_LOMA_R_FEATURE_ONNX}"
      "${PLASCAN_BUNDLED_LOMA_R_MATCHER_ONNX}"
      "${PLASCAN_BUNDLED_LOMA_R_MANIFEST_K1024}"
      "${PLASCAN_BUNDLED_LOMA_R_MANIFEST_K2048}"
      "${PLASCAN_BUNDLED_LOMA_R_MANIFEST_K3840}"
      DESTINATION resources/models/loma_r_tensorrt
      COMPONENT Runtime)
  endif()
  if(PLASCAN_BUNDLE_BIREFNET_DYNAMIC)
    install(FILES
      "${PLASCAN_BUNDLED_BIREFNET_DYNAMIC_ONNX}"
      "${PLASCAN_BUNDLED_BIREFNET_DYNAMIC_PROVENANCE}"
      DESTINATION resources/models/birefnet_dynamic
      COMPONENT Runtime)
  endif()

  set(_plascan_model_notices
    "${CMAKE_SOURCE_DIR}/docs/models/Apache-2.0.txt"
    "${CMAKE_SOURCE_DIR}/docs/models/U2Net_NOTICE.md"
    "${CMAKE_SOURCE_DIR}/docs/models/LightGlue_NOTICE.md"
    "${CMAKE_SOURCE_DIR}/docs/models/models-v1.1.0.sha256")
  if(PLASCAN_BUNDLE_LOMA_R_MODELS)
    list(APPEND _plascan_model_notices
      "${CMAKE_SOURCE_DIR}/docs/models/LoMa-R_NOTICE.md")
  endif()
  if(PLASCAN_BUNDLE_BIREFNET_DYNAMIC)
    list(APPEND _plascan_model_notices
      "${CMAKE_SOURCE_DIR}/docs/models/BiRefNet_NOTICE.md"
      "${CMAKE_SOURCE_DIR}/docs/models/BiRefNet-MIT.txt"
      "${CMAKE_SOURCE_DIR}/docs/models/models-v1.2.0.sha256")
  endif()
  install(FILES ${_plascan_model_notices}
    DESTINATION share/plascan/models
    COMPONENT Runtime)
  unset(_plascan_model_notices)

  if(PLASCAN_BUNDLE_BIREFNET_DYNAMIC)
    set(_plascan_verify_installed_models_script
      "${CMAKE_CURRENT_BINARY_DIR}/VerifyInstalledBundledModels.cmake")
    set(PLASCAN_VERIFY_BUNDLE_ONNX_MODELS OFF)
    set(PLASCAN_VERIFY_BUNDLE_LOMA_R_MODELS OFF)
    set(PLASCAN_VERIFY_BUNDLE_BIREFNET_DYNAMIC OFF)
    set(PLASCAN_VERIFY_INSTALLED_BIREFNET_DYNAMIC ON)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyBundledModels.cmake.in"
      "${_plascan_verify_installed_models_script}"
      @ONLY)
    install(SCRIPT "${_plascan_verify_installed_models_script}"
      COMPONENT Runtime)
  endif()

  install(CODE [=[
    set(_plascan_physical_prefix "${CMAKE_INSTALL_PREFIX}")
    if(DEFINED ENV{DESTDIR} AND NOT "$ENV{DESTDIR}" STREQUAL "")
      set(_plascan_destdir "$ENV{DESTDIR}")
      string(REGEX REPLACE "[/\\\\]$" "" _plascan_destdir "${_plascan_destdir}")
      if(CMAKE_INSTALL_PREFIX MATCHES "^[/\\\\]")
        set(_plascan_physical_prefix
          "${_plascan_destdir}${CMAKE_INSTALL_PREFIX}")
      else()
        set(_plascan_physical_prefix
          "${_plascan_destdir}/${CMAKE_INSTALL_PREFIX}")
      endif()
    endif()
    file(GLOB_RECURSE _plascan_bundled_engines
      LIST_DIRECTORIES FALSE
      "${_plascan_physical_prefix}/resources/models/*.engine")
    if(_plascan_bundled_engines)
      list(JOIN _plascan_bundled_engines ", " _plascan_bundled_engine_text)
      message(FATAL_ERROR
        "PlaScan packages must not contain machine-specific TensorRT engines: "
        "${_plascan_bundled_engine_text}")
    endif()
    unset(_plascan_physical_prefix)
    unset(_plascan_destdir)
  ]=] COMPONENT Runtime)
endif()

if(UNIX AND NOT APPLE)
  set(PLASCAN_LAUNCHER_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/plascan_gui_launcher.sh.in")
  set(PLASCAN_PATH_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/plascan_path.sh.in")

  configure_file("${PLASCAN_LAUNCHER_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan-launcher" @ONLY)
  configure_file("${PLASCAN_PATH_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan_path.sh" @ONLY)

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/plascan_path.sh"
    DESTINATION /etc/profile.d
    RENAME plascan.sh
    COMPONENT Runtime)
  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan-launcher"
    DESTINATION /usr/bin
    RENAME plascan
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
  foreach(_runtime_library IN ITEMS
      "${TensorRT_NVINFER_LIBRARY}"
      "${TensorRT_NVONNXPARSER_LIBRARY}")
    if(_runtime_library AND EXISTS "${_runtime_library}")
      get_filename_component(_runtime_library_dir
        "${_runtime_library}" DIRECTORY)
      list(APPEND PLASCAN_LINUX_RUNTIME_LIBRARY_DIRS
        "${_runtime_library_dir}")
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
  if(PLASCAN_LINUX_REQUIRE_TLS_PLUGIN AND
     (NOT PLASCAN_QT_PLUGINS_DIR OR
      NOT EXISTS "${PLASCAN_QT_PLUGINS_DIR}/tls/libqopensslbackend.so"))
    message(FATAL_ERROR
      "Linux package requires Qt's OpenSSL TLS backend. Rebuild qtbase with "
      "the network and openssl features.")
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
