#pragma once

#include "metalign/geometry.hpp"
#include "metalign/image.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace metmodel {

struct Camera {
    std::size_t index = 0;
    std::string name;
    std::filesystem::path path;
    bool aligned = false;
    metalign::CameraModel model;
    metalign::Pose pose;
    metalign::Vec3 center;
    metalign::Image image;
    metalign::CameraModel working_model;
    metalign::Image working_image;
    std::vector<std::uint32_t> track_ids;
};

struct SparsePoint {
    metalign::Vec3 position;
    std::array<std::uint8_t, 3> color{255, 255, 255};
    std::uint32_t track_id = std::numeric_limits<std::uint32_t>::max();
    float homogeneous_w = 1.0F;
};

// Chunk reconstruction region in the exact representation consumed by the
// recovered Metashape point-selection path: a column-axis 3x3 rotation,
// center, and full box size.  It is part of the computation input, not display
// metadata, because PatchMatch depth bounds discard sparse points outside it.
struct ReconstructionRegion {
    std::array<double, 9> rotation{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    metalign::Vec3 center;
    metalign::Vec3 size;
    bool specified = false;
};

struct Scene {
    std::vector<Camera> cameras;
    std::vector<SparsePoint> sparse_points;
    ReconstructionRegion region;
    std::size_t neighbor_common_threshold = 50;
};

struct DepthMap {
    std::size_t camera_index = 0;
    std::size_t width = 0;
    std::size_t height = 0;
    double scale_x = 1.0;
    double scale_y = 1.0;
    std::vector<float> depth;
    std::vector<float> confidence;
    std::vector<std::uint8_t> valid;

    float& depth_at(std::size_t x, std::size_t y) { return depth[y * width + x]; }
    float depth_at(std::size_t x, std::size_t y) const { return depth[y * width + x]; }
    float& confidence_at(std::size_t x, std::size_t y) { return confidence[y * width + x]; }
    float confidence_at(std::size_t x, std::size_t y) const { return confidence[y * width + x]; }
    std::uint8_t& valid_at(std::size_t x, std::size_t y) { return valid[y * width + x]; }
    std::uint8_t valid_at(std::size_t x, std::size_t y) const { return valid[y * width + x]; }
};

struct Vertex {
    metalign::Vec3 position;
    metalign::Vec3 normal;
    metalign::Vec2 uv;
    std::array<std::uint8_t, 3> color{255, 255, 255};
    float confidence = 0.0F;
};

struct Face {
    std::array<std::size_t, 3> vertices{};
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
};

struct TextureImage {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<std::uint8_t> rgb;
    std::vector<std::uint8_t> coverage;
};

}  // namespace metmodel
