add_library(camera_models_rpc STATIC
    RpcDefinition.cpp
    RpcDefinition.h
    RpcInstance.cpp
    RpcInstance.h
    RpcProjection.cpp
    RpcProjection.h
    RpcIntersectionService.cpp
    RpcIntersectionService.h
    RpcBiasAdjustment.cpp
    RpcBiasAdjustment.h
)

target_include_directories(camera_models_rpc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
target_compile_features(camera_models_rpc PUBLIC cxx_std_20)
target_link_libraries(camera_models_rpc PUBLIC camera_core)

if(BUILD_TESTS)
    include(GoogleTest)
    add_executable(test_rpc_camera_model
        tests/test_rpc_camera_model.cpp
    )
    target_link_libraries(test_rpc_camera_model PRIVATE
        camera_models_rpc
        GTest::gtest_main
    )
    target_include_directories(test_rpc_camera_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    )
    gtest_discover_tests(test_rpc_camera_model)
endif()
