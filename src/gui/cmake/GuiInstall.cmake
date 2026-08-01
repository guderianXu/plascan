# 安装与打包

option(PLASCAN_BUNDLE_RUNTIME "Bundle runtime shared libraries (Qt/OpenCV/Torch etc.) into install/package" ON)

install(TARGETS plascan_gui
  RUNTIME DESTINATION bin
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
    DESTINATION share/icons/hicolor/256x256/apps
    RENAME plascan.png
  )
  # SVG 备选 (供支持矢量图标的桌面环境)
  install(FILES "${CMAKE_SOURCE_DIR}/resources/plascan.svg"
    DESTINATION share/icons/hicolor/scalable/apps
    RENAME plascan.svg
  )

  # 桌面启动器 (StartupWMClass 必须与 setDesktopFileName 一致)
  install(FILES "${CMAKE_SOURCE_DIR}/resources/plascan.desktop"
    DESTINATION share/applications
  )
endif()

set(PLASCAN_QT_CONF_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/qt.conf.in")
configure_file("${PLASCAN_QT_CONF_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/qt.conf" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
  DESTINATION bin
  COMPONENT Runtime)

if(WIN32)
  set(_plascan_u2net_model "${CMAKE_SOURCE_DIR}/resources/models/U2Net_v1.onnx")
  if(EXISTS "${_plascan_u2net_model}")
    install(FILES "${_plascan_u2net_model}"
      DESTINATION resources/models
      COMPONENT Runtime)
  else()
    message(WARNING "PlaScan package will not include U2Net_v1.onnx: ${_plascan_u2net_model}")
  endif()
endif()

if(NOT WIN32)
  set(PLASCAN_LAUNCHER_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/plascan_gui_launcher.sh.in")
  set(PLASCAN_PATH_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/plascan_path.sh.in")

  configure_file("${PLASCAN_LAUNCHER_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan" @ONLY)
  configure_file("${PLASCAN_LAUNCHER_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan_gui" @ONLY)
  configure_file("${PLASCAN_PATH_TEMPLATE}" "${CMAKE_CURRENT_BINARY_DIR}/plascan_path.sh" @ONLY)

  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan" DESTINATION bin)
  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan_gui" DESTINATION bin)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/plascan_path.sh" DESTINATION /etc/profile.d RENAME plascan.sh)
  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/plascan" DESTINATION /usr/bin)
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

  set(PLASCAN_TORCH_LIB_DIR "")
  foreach(_torch_lib IN LISTS TORCH_LIBRARIES)
    if(EXISTS "${_torch_lib}")
      get_filename_component(_torch_lib_dir_candidate "${_torch_lib}" DIRECTORY)
      if(EXISTS "${_torch_lib_dir_candidate}")
        set(PLASCAN_TORCH_LIB_DIR "${_torch_lib_dir_candidate}")
        break()
      endif()
    endif()
  endforeach()

  if(PLASCAN_QT_PLUGINS_DIR AND EXISTS "${PLASCAN_QT_PLUGINS_DIR}/platforms")
    install(
      DIRECTORY "${PLASCAN_QT_PLUGINS_DIR}/platforms/"
      DESTINATION plugins/platforms
      FILES_MATCHING
      PATTERN "libq*.so*"
    )
  endif()

  if(PLASCAN_QT_PLUGINS_DIR AND EXISTS "${PLASCAN_QT_PLUGINS_DIR}/imageformats")
    install(
      DIRECTORY "${PLASCAN_QT_PLUGINS_DIR}/imageformats/"
      DESTINATION plugins/imageformats
      FILES_MATCHING
      PATTERN "libq*.so*"
    )
  endif()

  set(PLASCAN_INSTALL_BUNDLE_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallBundledRuntime.cmake")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/packaging/InstallBundledRuntime.cmake.in"
    "${PLASCAN_INSTALL_BUNDLE_SCRIPT}"
    @ONLY
  )
  install(SCRIPT "${PLASCAN_INSTALL_BUNDLE_SCRIPT}")
endif()

if(NOT WIN32)
  # 安装后更新图标缓存 (GNOME/KDE 需要)
  install(CODE "
    find_program(GTK_UPDATE_EXECUTABLE gtk-update-icon-cache)
    if(GTK_UPDATE_EXECUTABLE)
      set(_icon_dir \"\${CMAKE_INSTALL_PREFIX}/share/icons/hicolor\")
      if(EXISTS \"\${_icon_dir}\")
        execute_process(COMMAND \"\${GTK_UPDATE_EXECUTABLE}\" -f -t \"\${_icon_dir}\"
          ERROR_QUIET)
        message(STATUS \"Updated GTK icon cache: \${_icon_dir}\")
      endif()
    endif()
  ")
endif()

message(STATUS "plascan_gui configuration complete")
