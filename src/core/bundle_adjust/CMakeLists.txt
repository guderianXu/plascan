add_library(bundle_adjust STATIC BundleAdjust.cpp)

target_include_directories(bundle_adjust PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

target_link_libraries(bundle_adjust
    PUBLIC
        camera
        plascan_common_log
    PRIVATE
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(bundle_adjust PRIVATE HAS_OPENMP)
    message(STATUS "bundle_adjust: OpenMP 并行已启用 (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "bundle_adjust: OpenMP 未找到，将使用单线程模式")
endif()
