#include "metashape_texture/camera_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace metashape_texture {
namespace {

Vec3d subtract(Vec3d a, Vec3d b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3d add(Vec3d a, Vec3d b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3d multiply(Vec3d value, double scale) { return {value.x * scale, value.y * scale, value.z * scale}; }
Vec3d cross(Vec3d a, Vec3d b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double dot(Vec3d a, Vec3d b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3d normalize(Vec3d value) {
    const double length = std::sqrt(dot(value, value));
    if (length <= std::numeric_limits<double>::epsilon()) return {};
    return multiply(value, 1.0 / length);
}

}  // namespace

Vec3d world_to_camera(const CameraPose &pose, Vec3d world) {
    if (pose.scale == 0.0) throw std::invalid_argument("camera pose has zero scale");
    const Vec3d delta = subtract(world, pose.center);
    // Exact shader operation: transpose(camera_to_world_rotation) * (X - C) / scale.
    return {
        (pose.rotation[0] * delta.x + pose.rotation[3] * delta.y + pose.rotation[6] * delta.z) /
            pose.scale,
        (pose.rotation[1] * delta.x + pose.rotation[4] * delta.y + pose.rotation[7] * delta.z) /
            pose.scale,
        (pose.rotation[2] * delta.x + pose.rotation[5] * delta.y + pose.rotation[8] * delta.z) /
            pose.scale,
    };
}

std::optional<Vec2d> project_frame(const CameraPose &pose,
                                   const FrameCalibration &calibration,
                                   Vec3d world,
                                   bool require_inside) {
    const Vec3d point = world_to_camera(pose, world);
    if (point.z <= 0.0) return std::nullopt;

    const double x = point.x / point.z;
    const double y = point.y / point.z;
    const double raw_radius_squared = x * x + y * y;
    const double radius_squared = std::min(raw_radius_squared, calibration.maximum_radius_squared);
    const double radius_fourth = radius_squared * radius_squared;
    const double radial = calibration.k1 * radius_squared + calibration.k2 * radius_fourth +
                          calibration.k3 * radius_squared * radius_fourth +
                          calibration.k4 * radius_fourth * radius_fourth;
    const double tangential_scale = 1.0 + calibration.p3 * radius_squared +
                                    calibration.p4 * radius_fourth;
    const double radius_clamp = raw_radius_squared > calibration.maximum_radius_squared
                                    ? calibration.maximum_radius_squared / raw_radius_squared
                                    : 1.0;
    const double distorted_x = x + radius_clamp *
        (x * radial + (calibration.p1 * (y * y + 3.0 * x * x) +
                       2.0 * calibration.p2 * x * y) * tangential_scale);
    const double distorted_y = y + radius_clamp *
        (y * radial + (calibration.p2 * (x * x + 3.0 * y * y) +
                       2.0 * calibration.p1 * x * y) * tangential_scale);

    Vec2d pixel{
        (calibration.f + calibration.b1) * distorted_x + calibration.b2 * distorted_y +
            0.5 * static_cast<double>(calibration.width) + calibration.cx,
        calibration.f * distorted_y + 0.5 * static_cast<double>(calibration.height) + calibration.cy,
    };
    if (require_inside && (pixel.x < 0.0 || pixel.y < 0.0 ||
                           pixel.x >= static_cast<double>(calibration.width) ||
                           pixel.y >= static_cast<double>(calibration.height))) {
        return std::nullopt;
    }
    return pixel;
}

double face_view_dot(Vec3d camera_a, Vec3d camera_b, Vec3d camera_c) {
    const Vec3d normal = normalize(cross(subtract(camera_b, camera_a), subtract(camera_c, camera_a)));
    const Vec3d centroid = multiply(add(add(camera_a, camera_b), camera_c), 1.0 / 3.0);
    return dot(normal, normalize(centroid));
}

double recovered_depth_tolerance(double view_dot) {
    const double quality = std::clamp(2.0 * std::max(0.0, -0.5 - view_dot), 0.0, 1.0);
    return 0.999 + 0.0009 * quality;
}

}  // namespace metashape_texture
