#pragma once

#include <array>
#include <optional>

namespace metashape_texture {

struct Vec2d {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3d {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct FrameCalibration {
    int width = 0;
    int height = 0;
    double f = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double k1 = 0.0;
    double k2 = 0.0;
    double k3 = 0.0;
    double k4 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    double p3 = 0.0;
    double p4 = 0.0;
    // The GPU camera block contains an r^2 validity limit. Infinite is the
    // observable behaviour for ordinary Frame calibrations without a fitted
    // valid-radius crop.
    double maximum_radius_squared = 1.0e30;
};

struct CameraPose {
    // Camera-to-world rotation, row-major, followed by the camera centre.
    std::array<double, 9> rotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    Vec3d center{};
    double scale = 1.0;
};

[[nodiscard]] Vec3d world_to_camera(const CameraPose &pose, Vec3d world);

[[nodiscard]] std::optional<Vec2d> project_frame(const CameraPose &pose,
                                                  const FrameCalibration &calibration,
                                                  Vec3d world,
                                                  bool require_inside = true);

[[nodiscard]] double face_view_dot(Vec3d camera_a, Vec3d camera_b, Vec3d camera_c);

[[nodiscard]] double recovered_depth_tolerance(double view_dot);

}  // namespace metashape_texture
