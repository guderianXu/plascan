#include "CameraBaseline.h"

#include <algorithm>
#include <cmath>

namespace xjw
{
namespace
{

constexpr double kGeometryEpsilon = 1e-12;
constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

bool isFinitePoint(const std::array<double, 3> &point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

double squaredDistance(const std::array<double, 3> &left,
                       const std::array<double, 3> &right)
{
    const double dx = left[0] - right[0];
    const double dy = left[1] - right[1];
    const double dz = left[2] - right[2];
    return dx * dx + dy * dy + dz * dz;
}

std::array<double, 3> vectorToPoint(const std::array<double, 3> &center,
                                    const std::array<double, 3> &worldPoint)
{
    return {{worldPoint[0] - center[0], worldPoint[1] - center[1], worldPoint[2] - center[2]}};
}

double vectorLength(const std::array<double, 3> &vector)
{
    return std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
}

double physicalDepth(const Camera &camera, const std::array<double, 3> &worldPoint)
{
    const std::array<double, 3> center = camera.cameraCenter();
    const std::array<double, 9> rotation = camera.cameraToWorldRotation();
    const std::array<double, 3> offset = vectorToPoint(center, worldPoint);
    const double cameraZ = rotation[2] * offset[0]
        + rotation[5] * offset[1]
        + rotation[8] * offset[2];
    return camera.depthAxisFlipped() ? -cameraZ : cameraZ;
}

} // namespace

CameraBaseline CameraBaseline::evaluate(const Camera &first, const Camera &second)
{
    CameraBaseline result;
    const std::array<double, 3> firstCenter = first.cameraCenter();
    const std::array<double, 3> secondCenter = second.cameraCenter();
    if (!isFinitePoint(firstCenter) || !isFinitePoint(secondCenter))
    {
        return result;
    }

    const double squaredLength = squaredDistance(firstCenter, secondCenter);
    if (!(squaredLength > kGeometryEpsilon * kGeometryEpsilon)
        || !std::isfinite(squaredLength))
    {
        return result;
    }

    result._length = std::sqrt(squaredLength);
    result._valid = std::isfinite(result._length);
    return result;
}

CameraBaseline CameraBaseline::evaluate(const Camera &first,
                                        const Camera &second,
                                        const std::array<double, 3> &worldPoint)
{
    CameraBaseline result = evaluate(first, second);
    if (!result._valid || !isFinitePoint(worldPoint))
    {
        return result;
    }

    const std::array<double, 3> firstRay = vectorToPoint(first.cameraCenter(), worldPoint);
    const std::array<double, 3> secondRay = vectorToPoint(second.cameraCenter(), worldPoint);
    const double firstRayLength = vectorLength(firstRay);
    const double secondRayLength = vectorLength(secondRay);
    if (!(firstRayLength > kGeometryEpsilon) || !(secondRayLength > kGeometryEpsilon)
        || !std::isfinite(firstRayLength) || !std::isfinite(secondRayLength))
    {
        return result;
    }

    const double dot = firstRay[0] * secondRay[0]
        + firstRay[1] * secondRay[1]
        + firstRay[2] * secondRay[2];
    const double cosine = std::clamp(dot / (firstRayLength * secondRayLength), -1.0, 1.0);
    result._triangulationAngleDeg = std::acos(cosine) * kRadiansToDegrees;
    result._hasPointGeometry = std::isfinite(*result._triangulationAngleDeg);

    const double firstDepth = physicalDepth(first, worldPoint);
    const double secondDepth = physicalDepth(second, worldPoint);
    result._pointInFrontOfBothCameras = firstDepth > kGeometryEpsilon
        && secondDepth > kGeometryEpsilon
        && std::isfinite(firstDepth)
        && std::isfinite(secondDepth);
    if (result._pointInFrontOfBothCameras)
    {
        result._meanDepthToBaselineRatio = (firstDepth + secondDepth) / (2.0 * result._length);
    }
    return result;
}

} // namespace xjw
