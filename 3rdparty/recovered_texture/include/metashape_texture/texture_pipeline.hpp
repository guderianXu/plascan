#pragma once

#include <filesystem>

namespace metashape_texture
{

    struct BuildTextureParameters
    {
        int texture_size = 8192;
        int downscale = 2;
        float sharpening = 1.0F;
        bool fill_holes = true;
        bool ghosting_filter = true;
        bool out_of_focus_filter = false;
        bool color_enhancement = false;
        int anti_aliasing = 1;
    };

    struct PipelineInput
    {
        std::filesystem::path model_obj;
        std::filesystem::path scene_json;
        std::filesystem::path output_directory;
        // Five consecutive texture_size x texture_size RGBA32F atlas bands from
        // the recovered Vulkan enblend pipeline (band 0 through band 4).
        std::filesystem::path atlas_attachments;
        bool keep_input_uv = false;
        // Preserve the in-memory UV/position values when an OBJ is used as an
        // internal hand-off between atlas generation and page reconstruction.
        // Public/final OBJ output keeps Metashape's six-decimal formatting.
        bool precise_output_uv = false;
        // Validation-only oracle used to isolate blending from winner estimation.
        std::filesystem::path winner_labels;
        // Project-derived scale-4 resolution, scale-2 face weight, nadirness and
        // optional edge-count arrays used by NaturalBlending winner selection.
        std::filesystem::path winner_quality_input_directory;
        // Exact float32 mesh/camera resources extracted from the project.  This
        // geometry is required by winner selection, Natural UV, and blending,
        // regardless of whether out-of-focus filtering is enabled.
        std::filesystem::path project_input_directory;
        // Recovered project-derived inputs needed to generate optional focus maps
        // inside this C++ process when scene.json does not provide fixture paths.
        std::filesystem::path focus_project_input_directory;
        std::filesystem::path focus_zrange_directory;
        std::filesystem::path focus_shader_directory;
        std::filesystem::path focus_work_directory;
        std::filesystem::path focus_quality_graphics_directory;
        std::filesystem::path focus_quality_resolution_directory;
        std::filesystem::path focus_face_weight_path;
        std::filesystem::path focus_nadirness_shader_path;
        std::filesystem::path focus_resolution_shader_path;
        BuildTextureParameters parameters{};
    };

    void build_texture_compat(const PipelineInput& input);

} // namespace metashape_texture
