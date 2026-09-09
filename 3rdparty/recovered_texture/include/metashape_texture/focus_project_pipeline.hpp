#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace metashape_texture::recovered {

struct FocusQualityImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
    // Optional exact outputs of the original 7104/38008-byte Vulkan face
    // reducers, indexed by face rank.
    std::vector<float> face_sum;
    std::vector<float> face_count;
    std::vector<float> face_nadirness;
    // Exact scale=4 per-camera producer output and the shared scale=2
    // second-greatest threshold used by candidate/winner estimation.
    std::vector<float> face_resolution;
    std::vector<float> face_weight;
    // Exact per-face/per-local-edge count emitted by the original
    // 8016+38636-byte edge-quality chain (face-major, 3 slots per face).
    std::vector<float> edge_count;
};

struct FocusProjectPipelineInput {
    std::filesystem::path scene_json;
    std::filesystem::path project_input_directory;
    std::filesystem::path zrange_directory;
    std::filesystem::path shader_directory;
    std::filesystem::path quality_graphics_directory;
    std::filesystem::path quality_resolution_directory;
    std::filesystem::path face_weight_path;
    std::filesystem::path nadirness_shader_path;
    // Optional original 36376-byte producer.  When supplied, per-camera
    // resolution and the second-greatest face weight are built in-process,
    // replacing quality_resolution_directory and face_weight_path.
    std::filesystem::path resolution_shader_path;
    bool ghosting_filter = false;
    // If empty, a private temporary directory is removed after the images are
    // loaded.  Supplying a directory retains all stage files for validation.
    std::filesystem::path work_directory;
};

// One-process C++20/Vulkan composition of the already validated stage tools:
// project camera/mesh -> metric depth, source photo -> gray JPEG/upload,
// Sobel/NMS/components/multiscale, host score/factor/U8, exact PNG encoding.
std::vector<FocusQualityImage> build_focus_quality_images(
    const FocusProjectPipelineInput &input);

}  // namespace metashape_texture::recovered
