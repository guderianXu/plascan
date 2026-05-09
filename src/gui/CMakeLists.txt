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
list(REMOVE_ITEM GUI_SOURCES ${GUI_TASK_SOURCES})
add_library(gui_project OBJECT ${GUI_PROJECT_SOURCES})
add_library(gui_tasks OBJECT ${GUI_TASK_SOURCES})

#==============================================================================
# plascan_gui 可执行文件
#==============================================================================

add_executable(plascan_gui
  ${GUI_SOURCES}
  ${GUI_DIALOG_SOURCES}
  $<TARGET_OBJECTS:gui_project>
  $<TARGET_OBJECTS:gui_tasks>
  ${CMAKE_SOURCE_DIR}/resources/resources.qrc
)

# 包含目录
target_include_directories(plascan_gui PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/config
  ${CMAKE_CURRENT_SOURCE_DIR}/config/settings
  ${CMAKE_CURRENT_SOURCE_DIR}/main_window
  ${CMAKE_CURRENT_SOURCE_DIR}/menu
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
  ${CMAKE_SOURCE_DIR}/src/core/overlap
  ${CMAKE_SOURCE_DIR}/src/core/terrain
  ${CMAKE_SOURCE_DIR}/src/core/sfm
  ${CMAKE_SOURCE_DIR}/src/core/dense_match
  ${CMAKE_SOURCE_DIR}/src/core/pipeline
  ${CMAKE_SOURCE_DIR}/src/common
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

target_include_directories(gui_project PRIVATE $<TARGET_PROPERTY:plascan_gui,INCLUDE_DIRECTORIES>)
target_include_directories(gui_tasks PRIVATE $<TARGET_PROPERTY:plascan_gui,INCLUDE_DIRECTORIES>)
target_compile_definitions(gui_tasks PRIVATE PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

set_target_properties(gui_project gui_tasks PROPERTIES
  AUTOMOC ON
  AUTOUIC OFF
  AUTORCC OFF
)

if(GDAL_INCLUDE_DIRS)
  target_include_directories(plascan_gui PRIVATE ${GDAL_INCLUDE_DIRS})
endif()

if(TIFF_INCLUDE_DIRS)
  target_include_directories(plascan_gui PRIVATE ${TIFF_INCLUDE_DIRS})
endif()

# 添加SuperPoint头文件目录（superpoint库始终可用）
target_include_directories(plascan_gui PRIVATE ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/superpoint)
target_include_directories(plascan_gui PRIVATE ${CMAKE_SOURCE_DIR}/src/core/feature_extractors)

# 添加特征匹配头文件目录
target_include_directories(plascan_gui PRIVATE ${CMAKE_SOURCE_DIR}/src/core/feature_match/superglue)
target_include_directories(plascan_gui PRIVATE ${CMAKE_SOURCE_DIR}/src/core/feature_match/lightglue)
target_include_directories(plascan_gui PRIVATE ${CMAKE_SOURCE_DIR}/src/core/feature_match/tradition)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GuiCoreLinking.cmake)

# 链接Qt和其他库
target_link_libraries(plascan_gui PRIVATE
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  Qt6::Concurrent
  Qt6::OpenGL
  Qt6::OpenGLWidgets
  plascan_common_log
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  ${PLASCAN_LIBZIP_TARGET}
  ${PLASCAN_GDAL_TARGET}
  ${PLASCAN_TIFF_TARGET}
)

#==============================================================================
# 设置RPATH
#==============================================================================

set_target_properties(plascan_gui PROPERTIES
  OUTPUT_NAME "plascan_gui.bin"
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
