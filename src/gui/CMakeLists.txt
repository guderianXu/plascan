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

list(REMOVE_ITEM GUI_SOURCES ${GUI_PROJECT_SOURCES})
add_library(gui_project STATIC ${GUI_PROJECT_SOURCES})

#==============================================================================
# plascan_gui 可执行文件
#==============================================================================

add_executable(plascan_gui
  ${GUI_SOURCES}
  ${GUI_DIALOG_SOURCES}
  ${CMAKE_SOURCE_DIR}/resources/resources.qrc
)

qt_add_shaders(plascan_gui camera_scene_shaders
  PREFIX
    "/shaders"
  BASE
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  FILES
    shaders/camera_scene_color.vert
    shaders/camera_scene_color.frag
    shaders/camera_scene_mesh.vert
    shaders/camera_scene_mesh.frag
    shaders/camera_scene_textured_mesh.vert
    shaders/camera_scene_textured_mesh.frag
    shaders/camera_scene_image.vert
    shaders/camera_scene_image.frag
)

# 包含目录
target_include_directories(plascan_gui PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/config
  ${CMAKE_CURRENT_SOURCE_DIR}/config/settings
  ${CMAKE_CURRENT_SOURCE_DIR}/main_window
  ${CMAKE_CURRENT_SOURCE_DIR}/menu
  ${CMAKE_CURRENT_SOURCE_DIR}/platform
  ${CMAKE_CURRENT_SOURCE_DIR}/markers
  ${CMAKE_CURRENT_SOURCE_DIR}/panels
  ${CMAKE_CURRENT_SOURCE_DIR}/project
  ${CMAKE_CURRENT_SOURCE_DIR}/project/archive
  ${CMAKE_CURRENT_SOURCE_DIR}/project/data
  ${CMAKE_CURRENT_SOURCE_DIR}/project/io
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
  ${TORCH_INCLUDE_DIRS}
)

target_include_directories(gui_project PRIVATE
  $<TARGET_PROPERTY:plascan_gui,INCLUDE_DIRECTORIES>
  ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
)
target_compile_definitions(plascan_gui PRIVATE PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

set_target_properties(gui_project PROPERTIES
  AUTOMOC ON
  AUTOUIC OFF
  AUTORCC OFF
)

target_link_libraries(gui_project PRIVATE
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  Qt6::Concurrent
  plascan_common_project
  project_workflows
)

# ProjectDenseReconstructionManager 位于 GUI 协调层，但其算法调用只通过
# mvs 公共接口完成。显式声明依赖，确保 gui_project 被测试或其它目标单独链接时
# 不依赖 plascan_gui 偶然补齐未解析符号。
if(TARGET mvs)
  target_link_libraries(gui_project PRIVATE mvs)
endif()

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
  Qt6::Concurrent
  Qt6::ShaderTools
  plascan_common_log
  plascan_common_io
  plascan_common_model
  project_workflows
  ${TORCH_LIBRARIES}
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

if(PLASCAN_EXTRA_NVRTC_RPATHS)
  set_property(TARGET plascan_gui APPEND PROPERTY BUILD_RPATH "${PLASCAN_EXTRA_NVRTC_RPATHS}")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GuiInstall.cmake)
