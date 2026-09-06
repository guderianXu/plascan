cmake_minimum_required(VERSION 3.16)
project(task_runtime LANGUAGES CXX)

add_library(task_runtime STATIC
    TaskControl.cpp
    TaskControl.h
    TaskJournal.cpp
    TaskJournal.h
    TaskScheduler.cpp
    TaskScheduler.h
    TaskTypes.cpp
    TaskTypes.h
)

target_include_directories(task_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(task_runtime PUBLIC cxx_std_20)

if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        include(GoogleTest)
        add_executable(test_task_runtime
            tests/test_task_runtime.cpp
        )
        target_link_libraries(test_task_runtime PRIVATE
            task_runtime
            GTest::gtest_main
        )
        gtest_discover_tests(test_task_runtime)
    endif()
endif()
