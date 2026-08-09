include(GoogleTest)

add_executable(test_planetary_laser_json
    test_planetary_laser_ba_adapter.cpp
    test_planetary_laser_json.cpp
)

target_link_libraries(test_planetary_laser_json PRIVATE
    lidar
    GTest::gtest_main
)

gtest_discover_tests(test_planetary_laser_json DISCOVERY_MODE PRE_TEST)
