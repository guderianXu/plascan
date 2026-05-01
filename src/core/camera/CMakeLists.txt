add_library(camera STATIC Camera.cpp PositiveDepthCameraModel.cpp)
target_include_directories(camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# no extra dependencies

add_executable(camera_test test.cpp)
target_link_libraries(camera_test PRIVATE camera)

# 单元测试：将测试目标放在模块目录下，且仅在启用测试并找到 GTest 时添加
if(BUILD_TESTS)
	find_package(GTest QUIET)
	if(GTest_FOUND)
		include(GoogleTest)
		add_executable(test_camera_unit ${CMAKE_SOURCE_DIR}/src/core/camera/Camera_tests.cpp)
		target_link_libraries(test_camera_unit PRIVATE
			camera
			GTest::gtest_main
		)
		target_include_directories(test_camera_unit PRIVATE ${CMAKE_SOURCE_DIR}/src/core/camera)
		gtest_discover_tests(test_camera_unit)
	endif()
endif()
