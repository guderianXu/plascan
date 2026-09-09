#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>

namespace metmodel {

enum class FilterMode { None, Mild, Moderate, Aggressive };
enum class SurfaceType { Arbitrary, HeightField };
enum class Interpolation { Disabled, Enabled, Extrapolated };
enum class FaceCount { Low, Medium, High, Custom };
enum class DataSource { DepthMaps, PointCloud, TiePoints };
enum class MappingMode { Generic, Orthophoto, AdaptiveOrthophoto, Spherical, Camera };
enum class BlendingMode { Average, Mosaic, Natural, Min, Max, Disabled };
enum class ModelFormat { Auto, OBJ, PLY, STL, GLTF };
enum class ComputeBackend { Auto, CPU, CUDA, OpenCL, Vulkan };

struct GPUOptions {
    ComputeBackend backend = ComputeBackend::Auto;
    std::uint64_t gpu_mask = std::numeric_limits<std::uint64_t>::max();
    bool cpu_enable = true;
    std::size_t memory_guard_mb = 512;
    bool deterministic = false;
};

struct DepthOptions {
    int downscale = 4;
    FilterMode filter_mode = FilterMode::Mild;
    bool reuse_depth = false;
    std::size_t max_neighbors = 16;
    bool subdivide_task = true;
    std::size_t workitem_size_cameras = 20;
    std::size_t max_workgroup_size = 100;
    std::size_t hypotheses = 48;
    double min_confidence = 0.08;
};

struct ModelOptions {
    SurfaceType surface_type = SurfaceType::Arbitrary;
    Interpolation interpolation = Interpolation::Enabled;
    FaceCount face_count = FaceCount::High;
    std::size_t face_count_custom = 200000;
    DataSource source_data = DataSource::DepthMaps;
    bool vertex_colors = true;
    bool vertex_confidence = true;
    bool volumetric_masks = false;
    bool keep_depth = true;
    bool replace_asset = false;
    bool split_in_blocks = false;
    double blocks_size = 250.0;
    bool clip_to_boundary = false;
    bool export_blocks = false;
    int trimming_radius = 10;
    std::optional<std::filesystem::path> point_cloud;
    double voxel_size = 0.0;
};

struct PostprocessOptions {
    std::optional<std::size_t> decimate_face_count;
    double smooth_strength = 0.0;
    bool smooth_fix_borders = true;
    bool smooth_preserve_edges = false;
    bool refine_model = false;
    int refine_downscale = 4;
    std::size_t refine_iterations = 10;
    double refine_smoothness = 0.5;
};

struct UVOptions {
    bool enabled = true;
    MappingMode mapping_mode = MappingMode::Generic;
    std::size_t page_count = 1;
    std::size_t texture_size = 8192;
    double pixel_size = 0.0;
    std::optional<std::size_t> camera;
};

struct TextureOptions {
    bool enabled = true;
    BlendingMode blending_mode = BlendingMode::Natural;
    std::size_t texture_size = 8192;
    int downscale = 2;
    double sharpening = 1.0;
    bool use_assigned_images = false;
    bool fill_holes = true;
    bool ghosting_filter = true;
    bool out_of_focus_filter = false;
    bool color_enhancement = false;
    std::size_t anti_aliasing = 1;
    int jpeg_quality = 90;
};

struct ExportOptions {
    ModelFormat format = ModelFormat::Auto;
    bool binary = true;
    int precision = 6;
    bool save_texture = true;
    bool save_uv = true;
    bool save_normals = true;
    bool save_colors = true;
    bool save_confidence = false;
    bool save_cameras = true;
    bool save_markers = true;
    bool save_alpha = false;
    bool embed_texture = false;
    std::string comment;
};

struct ProgramOptions {
    std::filesystem::path image_directory;
    std::filesystem::path alignment_directory;
    std::filesystem::path output_model;
    std::filesystem::path working_directory;
    std::size_t threads = 0;
    unsigned int random_seed = 0x4D4F444CU;
    GPUOptions gpu;
    DepthOptions depth;
    ModelOptions model;
    PostprocessOptions postprocess;
    UVOptions uv;
    TextureOptions texture;
    ExportOptions export_options;
};

std::string to_string(FilterMode value);
std::string to_string(SurfaceType value);
std::string to_string(Interpolation value);
std::string to_string(FaceCount value);
std::string to_string(DataSource value);
std::string to_string(MappingMode value);
std::string to_string(BlendingMode value);
std::string to_string(ModelFormat value);
std::string to_string(ComputeBackend value);
ProgramOptions parse_arguments(int argc, char** argv);
std::string command_help();

}  // namespace metmodel
