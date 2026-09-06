#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace metalign {

enum class ReferencePreselectionMode { Source, Estimated, Sequential };

struct MatchPhotosOptions {
    int downscale = 1;
    int downscale_3d = 1;
    bool generic_preselection = true;
    bool reference_preselection = true;
    ReferencePreselectionMode reference_preselection_mode =
        ReferencePreselectionMode::Source;
    bool filter_mask = false;
    bool mask_tiepoints = true;
    bool filter_stationary_points = true;
    std::size_t keypoint_limit = 40000;
    std::size_t keypoint_limit_3d = 100000;
    std::size_t keypoint_limit_depth_maps = 10000;
    std::size_t keypoint_limit_per_mpx = 1000;
    std::size_t tiepoint_limit = 4000;
    bool keep_keypoints = false;
    bool guided_matching = false;
    bool reset_matches = false;
    bool subdivide_task = true;
    std::size_t workitem_size_cameras = 20;
    std::size_t workitem_size_pairs = 80;
    std::size_t max_workgroup_size = 100;
    int laser_scans_vertical_axis = 0;
    bool laser_scans_use_initial_orientation = false;
    bool match_laser_scans = false;
    bool match_depth_maps = false;
    std::size_t reference_preselection_neighbors = 10;
    bool filter_weak_points = false;
};

struct AlignCamerasOptions {
    std::size_t min_image = 2;
    bool adaptive_fitting = false;
    bool reset_alignment = false;
    bool subdivide_task = true;
    bool align_laser_scans = false;
};

struct ProgramOptions {
    std::filesystem::path image_directory;
    std::filesystem::path output_directory;
    std::optional<std::filesystem::path> masks_directory;
    std::optional<std::filesystem::path> reference_csv;
    std::optional<std::filesystem::path> sensor_csv;
    std::optional<std::filesystem::path> pairs_file;
    std::optional<std::filesystem::path> detector_schedule;
    std::optional<std::filesystem::path> coarse_orientation_schedule;
    std::optional<std::filesystem::path> full_orientation_schedule;
    std::optional<std::filesystem::path> capture_gomp_schedules;
    std::optional<std::filesystem::path> feature_cache;
    std::optional<double> focal_length_pixels;
    double assumed_horizontal_fov_degrees = 60.0;
    std::size_t threads = 0;
    unsigned int random_seed = 0x4D455441U;
    std::string gpu_backend = "auto";
    int gpu_device = -1;
    std::uint64_t gpu_mask = ~std::uint64_t{0};
    bool cpu_enable = true;
    bool gpu_feature_extraction = false;
    bool target_gomp_runtime = false;
    bool refresh_feature_cache = false;
    MatchPhotosOptions match;
    AlignCamerasOptions align;
};

std::string to_string(ReferencePreselectionMode mode);
inline float alignment_accuracy_source_scale(int downscale)
{
    if (downscale == 0)
    {
        return 0.5F;
    }
    if (downscale == 1 || downscale == 2 || downscale == 4 || downscale == 8)
    {
        return static_cast<float>(downscale);
    }
    throw std::invalid_argument("invalid alignment downscale");
}
ProgramOptions parse_arguments(int argc, char** argv);
std::string command_help();

}  // namespace metalign
