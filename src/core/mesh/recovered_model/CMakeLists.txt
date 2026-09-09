# Compile the supplied GLSL sources and embed SPIR-V in the core library, so
# installed CLI/GUI programs never depend on the source tree or external files.
find_package(Vulkan REQUIRED)
find_program(PLASCAN_GLSLANG_VALIDATOR NAMES glslangValidator
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    REQUIRED)

set(recovered_shader_outputs)
foreach(shader_spec IN ITEMS
        "000;comp;depth_range.comp"
        "001;vert;depth.vert"
        "002;frag;depth.frag"
        "003;comp;project.comp"
        "004;comp;clear.comp"
        "005;comp;geometry.comp"
        "006;comp;color.comp")
    list(GET shader_spec 0 shader_number)
    list(GET shader_spec 1 shader_stage)
    list(GET shader_spec 2 shader_source)
    set(shader_output "${CMAKE_CURRENT_BINARY_DIR}/shader_${shader_number}_recovered.spv")
    add_custom_command(
        OUTPUT "${shader_output}"
        COMMAND "${PLASCAN_GLSLANG_VALIDATOR}" -V
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -S "${shader_stage}" -o "${shader_output}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader_source}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader_source}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/recovered_camera.glsl"
        VERBATIM)
    list(APPEND recovered_shader_outputs "${shader_output}")
endforeach()
qt_add_resources(meshing recovered_vertex_color
    PREFIX "/plascan/recovered_vertex_color"
    BASE "${CMAKE_CURRENT_BINARY_DIR}"
    FILES ${recovered_shader_outputs})
target_sources(meshing PRIVATE vertex_color_vulkan.cpp RecoveredModelColorizer.cpp)
target_link_libraries(meshing PRIVATE Vulkan::Vulkan CUDA::cudart)
target_compile_definitions(meshing PRIVATE PLASCAN_RECOVERED_VERTEX_COLOR_VULKAN=1)
if(MSVC)
    set_property(SOURCE vertex_color_vulkan.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:strict)
else()
    set_property(SOURCE vertex_color_vulkan.cpp APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
endif()
