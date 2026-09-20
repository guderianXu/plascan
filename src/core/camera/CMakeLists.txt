add_subdirectory(core)
add_subdirectory(models/frame_pinhole)
add_subdirectory(models/rpc)
add_subdirectory(models/linescan)

add_library(camera_models_registry STATIC
    models/CameraModelFactories.cpp
    models/CameraModelFactories.h
    models/LineScanModelJsonDecode.cpp
    models/LineScanModelJsonEncode.cpp
    models/LineScanModelJson.h
)
target_include_directories(camera_models_registry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(camera_models_registry PUBLIC cxx_std_20)
target_link_libraries(camera_models_registry PUBLIC
    camera_core
    camera_models_frame_pinhole
    camera_models_rpc
    camera_models_linescan
    Qt6::Core
)

add_subdirectory(project)
add_subdirectory(reference)

# 生产静态库：相机外部格式导入、转换和通用摄影测量辅助能力。
add_library(camera STATIC
	FramePinholeTsaiIO.cpp
	PlanetaryLineScanIsdIO.cpp
	PlanetaryLineScanIsdIO.h
	RpcRasterIO.cpp
	CameraBaseline.cpp
	CameraIntrinsicPrior.cpp
	CameraFormatConverter.cpp
	ColmapImageUndistorter.cpp
	ProjectCameraIO.cpp
	ProjectLineScanCameraIO.cpp
	ProjectRpcCameraIO.cpp
)
target_include_directories(camera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/core
)
target_compile_features(camera PUBLIC cxx_std_20)
target_link_libraries(camera
	PUBLIC plascan_common_io
	        camera_models_frame_pinhole
		        camera_models_linescan
		        camera_models_registry
		        camera_models_rpc
	PRIVATE plascan_common_string_utils ${PLASCAN_LIBZIP_TARGET} ${OpenCV_LIBS} ${PLASCAN_GDAL_TARGET}
)

# 始终构建的手工投影诊断工具；它读取用户指定的 Tsai 和世界点，不属于 CTest。
add_executable(camera_test test/test.cpp)
target_link_libraries(camera_test PRIVATE camera)

# 自动化单元测试仅在显式启用测试且依赖可用时注册，避免生产构建强制引入 GTest。
if(BUILD_TESTS)
	find_package(GTest QUIET)
	if(GTest_FOUND)
		include(GoogleTest)
		# 纯合成内参与投影测试；include 指向模块根以保持公共相机头路径稳定。
		add_executable(test_camera_unit
			 test/RpcRasterIO_tests.cpp
		)
		target_link_libraries(test_camera_unit PRIVATE
			camera
			${PLASCAN_GDAL_TARGET}
			GTest::gtest_main
		)
		target_include_directories(test_camera_unit PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		gtest_discover_tests(test_camera_unit)

		# USGSCSM ISD 推扫相机的时间、姿轨、焦平面和投影回归测试。
		add_executable(test_planetary_line_scan_camera
			test/PlanetaryLineScanIsdIO_tests.cpp
		)
		target_link_libraries(test_planetary_line_scan_camera PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_planetary_line_scan_camera PRIVATE
			${CMAKE_SOURCE_DIR}/src/core/camera
		)
		target_compile_definitions(test_planetary_line_scan_camera PRIVATE
			PLANETARY_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
		)
		gtest_discover_tests(test_planetary_line_scan_camera)

		# 相机中心距离、三角交会角与深度/基线比的统一摄影测量几何测试。
		add_executable(test_camera_baseline test/CameraBaseline_tests.cpp)
		target_link_libraries(test_camera_baseline PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_baseline PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		gtest_discover_tests(test_camera_baseline)

		# EXIF 焦距与固定镜头型号目录的像素焦距换算。
		add_executable(test_camera_intrinsic_prior test/CameraIntrinsicPrior_tests.cpp)
		target_link_libraries(test_camera_intrinsic_prior PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_intrinsic_prior PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		gtest_discover_tests(test_camera_intrinsic_prior)

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
		add_executable(test_camera_format_converter
			test/CameraFormatConverter_tests.cpp
			test/ColmapImageUndistorter_tests.cpp
		)
		target_link_libraries(test_camera_format_converter PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_format_converter PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		gtest_discover_tests(test_camera_format_converter)

		# 覆盖输出祖先/别名保护及目录事务的回滚与残留清理。
		add_executable(test_camera_format_converter_safety
			test/CameraFormatConverterSafety_tests.cpp
		)
		target_link_libraries(test_camera_format_converter_safety PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_format_converter_safety PRIVATE
			${CMAKE_SOURCE_DIR}/src/core/camera
		)
		gtest_discover_tests(test_camera_format_converter_safety)
	endif()
endif()
