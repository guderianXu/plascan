cmake_minimum_required(VERSION 3.16)

#==============================================================================
# PlaScan GUI 模块
#==============================================================================

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 所有依赖已由根 CMakeLists.txt 通过 PlascanPackages.cmake 统一查找

#==============================================================================
# 源文件
#==============================================================================

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GuiSources.cmake)

add_library(gui_project STATIC ${GUI_PROJECT_SOURCES})

set(PLASCAN_GUI_RESOURCES ${CMAKE_SOURCE_DIR}/resources/resources.qrc)
set_source_files_properties(${PLASCAN_GUI_RESOURCES} PROPERTIES
  AUTORCC_OPTIONS "--no-compress"
)

#==============================================================================
# plascan_gui 可执行文件
#==============================================================================

if(PLASCAN_BUILD_GUI)
  if(Qt6_VERSION VERSION_LESS "6.7")
    message(FATAL_ERROR
      "PlaScan GUI requires Qt 6.7 or newer for QRhiWidget. "
      "Configure with -DPLASCAN_BUILD_GUI=OFF to build core libraries, CLI tools, and tests.")
  endif()

  add_executable(plascan_gui
    ${GUI_SOURCES}
    ${GUI_DIALOG_SOURCES}
    ${PLASCAN_GUI_RESOURCES}
  )

  qt_add_shaders(plascan_gui camera_scene_shaders
    PREFIX
      "/shaders"
    BASE
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    FILES
      shaders/camera_scene_color.vert
      shaders/camera_scene_color.frag
      shaders/camera_scene_point.vert
      shaders/camera_scene_point.frag
      shaders/camera_scene_mesh.vert
      shaders/camera_scene_mesh.frag
      shaders/camera_scene_textured_mesh.vert
      shaders/camera_scene_textured_mesh.frag
      shaders/camera_scene_image.vert
      shaders/camera_scene_image.frag
      shaders/camera_scene_camera.vert
      shaders/camera_scene_camera_leader.vert
  )
endif()

# 包含目录
set(PLASCAN_GUI_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/config
  ${CMAKE_CURRENT_SOURCE_DIR}/config/settings
  ${CMAKE_CURRENT_SOURCE_DIR}/main_window
  ${CMAKE_CURRENT_SOURCE_DIR}/menu
  ${CMAKE_CURRENT_SOURCE_DIR}/platform
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/markers
  ${CMAKE_CURRENT_SOURCE_DIR}/panels
  ${CMAKE_CURRENT_SOURCE_DIR}/project
  ${CMAKE_CURRENT_SOURCE_DIR}/project/data
  ${CMAKE_CURRENT_SOURCE_DIR}/project/manager
  ${CMAKE_CURRENT_SOURCE_DIR}/project/services
  ${CMAKE_CURRENT_SOURCE_DIR}/project/support
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks
  ${CMAKE_CURRENT_SOURCE_DIR}/views
  ${CMAKE_CURRENT_SOURCE_DIR}/widgets
  ${CMAKE_CURRENT_SOURCE_DIR}/dialogs
  ${CMAKE_SOURCE_DIR}/src/core/intersection
  ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
  ${CMAKE_SOURCE_DIR}/src/core/camera
  ${CMAKE_SOURCE_DIR}/src/core/lidar
  ${CMAKE_SOURCE_DIR}/src/core/mask
  ${CMAKE_SOURCE_DIR}/src/core/qc
  ${CMAKE_SOURCE_DIR}/src/core/control_points
  ${CMAKE_SOURCE_DIR}/src/core/overlap
  ${CMAKE_SOURCE_DIR}/src/core/terrain
  ${CMAKE_SOURCE_DIR}/src/core/sfm
  ${CMAKE_SOURCE_DIR}/src/core/matchphototask
  ${CMAKE_SOURCE_DIR}/src/core/matchphototask/algorithm
  ${CMAKE_SOURCE_DIR}/src/core/matchphototask/pair_selection
  ${CMAKE_SOURCE_DIR}/src/core/matchphototask/runtime
  ${CMAKE_SOURCE_DIR}/src/core/matchphototask/task
  ${CMAKE_SOURCE_DIR}/src/core/image_matching
  ${CMAKE_SOURCE_DIR}/src/core/dense_match
  ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
  ${CMAKE_SOURCE_DIR}/src/common
  ${OpenCV_INCLUDE_DIRS}
)

if(PLASCAN_BUILD_GUI)
  target_include_directories(plascan_gui PRIVATE ${PLASCAN_GUI_INCLUDE_DIRECTORIES})
  target_compile_definitions(plascan_gui PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    PLASCAN_VERSION="${PROJECT_VERSION}"
  )
endif()
target_include_directories(gui_project PRIVATE ${PLASCAN_GUI_INCLUDE_DIRECTORIES})
target_compile_definitions(gui_project PRIVATE
  PLASCAN_VERSION="${PROJECT_VERSION}"
)

set_target_properties(gui_project PROPERTIES
  AUTOMOC ON
  AUTOUIC OFF
  AUTORCC OFF
)

target_link_libraries(gui_project PRIVATE
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  Qt6::Network
  Qt6::Concurrent
  plascan_common_project
  project_workflows
)

# GUI project services own these workflow calls, so the library declares the
# corresponding core dependencies instead of relying on final executable link order.
if(TARGET lidar)
  target_link_libraries(gui_project PRIVATE lidar)
endif()

if(TARGET mask)
  target_link_libraries(gui_project PRIVATE mask)
endif()

if(TARGET mvs)
  target_link_libraries(gui_project PRIVATE mvs)
endif()

if(PLASCAN_BUILD_GUI)
  if(GDAL_INCLUDE_DIRS)
    target_include_directories(plascan_gui PRIVATE ${GDAL_INCLUDE_DIRS})
  endif()

  if(TIFF_INCLUDE_DIRS)
    target_include_directories(plascan_gui PRIVATE ${TIFF_INCLUDE_DIRS})
  endif()

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GuiCoreLinking.cmake)

# 链接Qt和其他库
target_link_libraries(plascan_gui PRIVATE
  gui_project
  Qt6::Core
  Qt6::Gui
  Qt6::GuiPrivate
  Qt6::Widgets
  Qt6::Network
  Qt6::Concurrent
  Qt6::ShaderTools
  plascan_common_log
  plascan_common_io
  plascan_common_model
  project_workflows
  ${OpenCV_LIBS}
  ${PLASCAN_LIBZIP_TARGET}
  ${PLASCAN_GDAL_TARGET}
  ${PLASCAN_TIFF_TARGET}
)

if(WIN32)
  target_link_libraries(plascan_gui PRIVATE advapi32 shell32)
endif()

#==============================================================================
# 设置RPATH
#==============================================================================

if(WIN32)
  set(PLASCAN_GUI_OUTPUT_NAME "plascan")
else()
  set(PLASCAN_GUI_OUTPUT_NAME "plascan_gui.bin")
endif()

set_target_properties(plascan_gui PROPERTIES
  OUTPUT_NAME "${PLASCAN_GUI_OUTPUT_NAME}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Qt Network 的 HTTPS 能力由 TLS 插件提供。显式部署插件，保证直接从 build/bin 启动
# 与安装包启动时都能下载 GitHub Release 中的模型资源。
plascan_deploy_qt_tls_runtime(plascan_gui)
plascan_deploy_tensorrt_runtime(plascan_gui)
plascan_deploy_vcpkg_runtime(plascan_gui)

if(PLASCAN_EXTRA_NVRTC_RPATHS)
  set_property(TARGET plascan_gui APPEND PROPERTY BUILD_RPATH "${PLASCAN_EXTRA_NVRTC_RPATHS}")
endif()

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GuiInstall.cmake)
endif()
