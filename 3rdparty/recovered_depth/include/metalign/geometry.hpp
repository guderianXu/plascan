#pragma once

#include "metalign/math.hpp"

namespace metalign
{
struct CameraModel
{
    double f = 1.0;
    double cx = 0.0;
    double cy = 0.0;
    double cx_offset = 0.0;
    double cy_offset = 0.0;
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
    double resection_threshold_pixels = 16.384000778198242;
};

Vec2 project_local(const CameraModel &camera, Vec3 local);
Vec2 project(const CameraModel &camera, const Pose &pose, Vec3 point);
Vec3 bearing(const CameraModel &camera, Vec2 pixel);
} // namespace metalign
