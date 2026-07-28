# 生产静态库：Camera/Tsai I/O、坐标归一化和外部格式转换器。
add_library(camera STATIC Camera.cpp CameraBaseline.cpp CameraFormatConverter.cpp)
target_include_directories(camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(camera
	PUBLIC plascan_common_io
	PRIVATE plascan_common_string_utils ${PLASCAN_LIBZIP_TARGET}
)

# 始终构建的手工投影诊断工具；它读取用户指定的 Tsai 和世界点，不属于 CTest。
add_executable(camera_test test/test.cpp)
target_link_libraries(camera_test PRIVATE camera)

# 自动化单元测试仅在显式启用测试且依赖可用时注册，避免生产构建强制引入 GTest。
if(BUILD_TESTS)
	find_package(GTest QUIET)
	if(GTest_FOUND)
		include(GoogleTest)
		# 纯合成内参与投影测试；include 指向模块根以保持 #include "Camera.h" 稳定。
		add_executable(test_camera_unit test/Camera_tests.cpp)
		target_link_libraries(test_camera_unit PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_unit PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		gtest_discover_tests(test_camera_unit)

		# 相机中心距离、三角交会角与深度/基线比的统一摄影测量几何测试。
		add_executable(test_camera_baseline test/CameraBaseline_tests.cpp)
		target_link_libraries(test_camera_baseline PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_baseline PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		gtest_discover_tests(test_camera_baseline)

		# 仓库测试数据回归：TEST_DATA_DIR 使用绝对源码路径，不依赖 CTest 工作目录。
		add_executable(test_camera_tsai test/test_tsai_loader.cpp)
		target_link_libraries(test_camera_tsai PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_tsai PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		target_compile_definitions(test_camera_tsai PRIVATE
			TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
		)
		gtest_discover_tests(test_camera_tsai)

		# 四种外部格式的合成数据集转换与输出契约测试。
		add_executable(test_camera_format_converter test/CameraFormatConverter_tests.cpp)
		target_link_libraries(test_camera_format_converter PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_format_converter PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		gtest_discover_tests(test_camera_format_converter)
	endif()
endif()
