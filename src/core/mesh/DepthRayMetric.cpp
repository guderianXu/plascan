#include "DepthRayMetric.h"

#include "Camera.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::mesh
{
namespace
{

constexpr double kMinimumPositiveDepth = 1.0e-12;
constexpr double kMinimumRayScale = 1.0e-12;
constexpr double kHalfPixel = 0.5;

bool isFinite(const std::array<double, 3> &value)
{
    return std::isfinite(value[0]) &&
           std::isfinite(value[1]) &&
           std::isfinite(value[2]);
}

std::array<double, 3> subtract(
    const std::array<double, 3> &left,
    const std::array<double, 3> &right)
{
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2]};
}

double norm(const std::array<double, 3> &value)
{
    return std::hypot(value[0], value[1], value[2]);
}

std::array<double, 3> cross(
    const std::array<double, 3> &left,
    const std::array<double, 3> &right)
{
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]};
}

bool unproject(
    const Camera &camera,
    const std::array<double, 2> &pixel,
    double depth,
    std::array<double, 3> *world)
{
    if (!world)
    {
        return false;
    }
    const double raw_pixel[2] = {pixel[0], pixel[1]};
    double raw_world[3]{};
    if (!camera.unprojectPixel(raw_pixel, depth, raw_world))
    {
        return false;
    }
    const std::array<double, 3> candidate{
        raw_world[0], raw_world[1], raw_world[2]};
    if (!isFinite(candidate))
    {
        return false;
    }
    *world = candidate;
    return true;
}

bool makeUnitRay(
    const Camera &camera,
    const std::array<double, 2> &pixel,
    const std::array<double, 3> &center,
    std::array<double, 3> *unit_ray)
{
    std::array<double, 3> world{};
    if (!unit_ray || !unproject(camera, pixel, 1.0, &world))
    {
        return false;
    }
    const std::array<double, 3> ray = subtract(world, center);
    const double ray_length = norm(ray);
    if (!(ray_length > kMinimumRayScale) || !std::isfinite(ray_length))
    {
        return false;
    }
    *unit_ray = {
        ray[0] / ray_length,
        ray[1] / ray_length,
        ray[2] / ray_length};
    return isFinite(*unit_ray);
}

double pointToRayDistance(
    const std::array<double, 3> &point,
    const std::array<double, 3> &ray_origin,
    const std::array<double, 3> &unit_ray)
{
    return norm(cross(subtract(point, ray_origin), unit_ray));
}

bool footprintForAxis(
    const Camera &camera,
    const std::array<double, 2> &pixel,
    const std::array<double, 3> &center,
    const std::array<double, 3> &world_point,
    int axis,
    double *footprint)
{
    if (!footprint || axis < 0 || axis > 1)
    {
        return false;
    }
    std::array<double, 2> negative_pixel = pixel;
    std::array<double, 2> positive_pixel = pixel;
    negative_pixel[axis] -= kHalfPixel;
    positive_pixel[axis] += kHalfPixel;

    std::array<double, 3> negative_ray{};
    std::array<double, 3> positive_ray{};
    if (!makeUnitRay(camera, negative_pixel, center, &negative_ray) ||
        !makeUnitRay(camera, positive_pixel, center, &positive_ray))
    {
        return false;
    }
    const double negative_half = pointToRayDistance(
        world_point, center, negative_ray);
    const double positive_half = pointToRayDistance(
        world_point, center, positive_ray);
    const double candidate = negative_half + positive_half;
    if (!(candidate > 0.0) || !std::isfinite(candidate))
    {
        return false;
    }
    *footprint = candidate;
    return true;
}

bool validSample(const DepthRayMetricSample &sample)
{
    return sample.valid &&
           sample.cameraZDepth > kMinimumPositiveDepth &&
           sample.rayDistance > kMinimumPositiveDepth &&
           sample.rayDistancePerCameraZ > kMinimumRayScale &&
           std::isfinite(sample.cameraZDepth) &&
           std::isfinite(sample.rayDistance) &&
           std::isfinite(sample.rayDistancePerCameraZ) &&
           isFinite(sample.cameraCenter) &&
           isFinite(sample.unitWorldRay) &&
           isFinite(sample.worldPoint);
}

bool pointAtDistances(
    const DepthRayMetricSample &sample,
    double camera_z_depth,
    double ray_distance,
    std::array<double, 3> *world_point)
{
    if (!world_point ||
        !(camera_z_depth > kMinimumPositiveDepth) ||
        !(ray_distance > kMinimumPositiveDepth) ||
        !std::isfinite(camera_z_depth) ||
        !std::isfinite(ray_distance))
    {
        return false;
    }
    const std::array<double, 3> candidate{
        sample.cameraCenter[0] + sample.unitWorldRay[0] * ray_distance,
        sample.cameraCenter[1] + sample.unitWorldRay[1] * ray_distance,
        sample.cameraCenter[2] + sample.unitWorldRay[2] * ray_distance};
    if (!isFinite(candidate))
    {
        return false;
    }
    *world_point = candidate;
    return true;
}

} // namespace

DepthRayMetricSample DepthRayMetric::evaluate(
    const Camera &camera,
    const std::array<double, 2> &pixel,
    double positive_camera_z_depth)
{
    DepthRayMetricSample result;
    if (!camera.isValid() ||
        !(positive_camera_z_depth > kMinimumPositiveDepth) ||
        !std::isfinite(positive_camera_z_depth) ||
        !std::isfinite(pixel[0]) ||
        !std::isfinite(pixel[1]))
    {
        return result;
    }

    result.cameraCenter = camera.cameraCenter();
    result.cameraZDepth = positive_camera_z_depth;
    if (!isFinite(result.cameraCenter) ||
        !unproject(camera, pixel, positive_camera_z_depth, &result.worldPoint))
    {
        return {};
    }

    const std::array<double, 3> ray =
        subtract(result.worldPoint, result.cameraCenter);
    result.rayDistance = norm(ray);
    if (!(result.rayDistance > kMinimumPositiveDepth) ||
        !std::isfinite(result.rayDistance))
    {
        return {};
    }
    result.rayDistancePerCameraZ =
        result.rayDistance / positive_camera_z_depth;
    if (!(result.rayDistancePerCameraZ > kMinimumRayScale) ||
        !std::isfinite(result.rayDistancePerCameraZ))
    {
        return {};
    }
    result.unitWorldRay = {
        ray[0] / result.rayDistance,
        ray[1] / result.rayDistance,
        ray[2] / result.rayDistance};

    if (!footprintForAxis(
            camera,
            pixel,
            result.cameraCenter,
            result.worldPoint,
            0,
            &result.worldPixelFootprintX) ||
        !footprintForAxis(
            camera,
            pixel,
            result.cameraCenter,
            result.worldPoint,
            1,
            &result.worldPixelFootprintY))
    {
        return {};
    }
    result.worldPixelFootprint = std::sqrt(
        result.worldPixelFootprintX * result.worldPixelFootprintY);
    if (!(result.worldPixelFootprint > 0.0) ||
        !std::isfinite(result.worldPixelFootprint))
    {
        return {};
    }
    result.valid = true;
    return result;
}

bool DepthRayMetric::pointAtCameraZOffset(
    const DepthRayMetricSample &sample,
    double signed_camera_z_offset,
    std::array<double, 3> *world_point,
    double *ray_distance)
{
    if (!validSample(sample) || !std::isfinite(signed_camera_z_offset))
    {
        return false;
    }
    const double candidate_camera_z =
        sample.cameraZDepth + signed_camera_z_offset;
    const double candidate_ray_distance =
        candidate_camera_z * sample.rayDistancePerCameraZ;
    if (!pointAtDistances(
            sample,
            candidate_camera_z,
            candidate_ray_distance,
            world_point))
    {
        return false;
    }
    if (ray_distance)
    {
        *ray_distance = candidate_ray_distance;
    }
    return true;
}

bool DepthRayMetric::pointAtRayDistanceOffset(
    const DepthRayMetricSample &sample,
    double signed_ray_distance_offset,
    std::array<double, 3> *world_point,
    double *positive_camera_z_depth)
{
    if (!validSample(sample) || !std::isfinite(signed_ray_distance_offset))
    {
        return false;
    }
    const double candidate_ray_distance =
        sample.rayDistance + signed_ray_distance_offset;
    const double candidate_camera_z =
        candidate_ray_distance / sample.rayDistancePerCameraZ;
    if (!pointAtDistances(
            sample,
            candidate_camera_z,
            candidate_ray_distance,
            world_point))
    {
        return false;
    }
    if (positive_camera_z_depth)
    {
        *positive_camera_z_depth = candidate_camera_z;
    }
    return true;
}

} // namespace xjw::mesh
