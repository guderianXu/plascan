#include "PlanetaryLineScanBundleAdjustInternal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{
namespace lidar
{
namespace detail
{
namespace
{

using Vector3 = std::array<double, 3>;

double vectorNorm(const Vector3 &value)
{
    return std::sqrt(value[0] * value[0] +
                     value[1] * value[1] +
                     value[2] * value[2]);
}

} // namespace

camera_models::linescan::LineScanTrajectoryBias lineScanPoseBias(
    const std::array<double, 6> &parameters)
{
    camera_models::linescan::LineScanTrajectoryBias bias;
    std::copy_n(parameters.begin(), 3, bias.translationMeters.begin());
    std::copy_n(parameters.begin() + 3, 3, bias.rotationVectorRadians.begin());
    return bias;
}

bool evaluateLineScanImageObservation(
    const camera_models::linescan::LineScanInstance& camera,
    const LineScanImageObservation &observation,
    const double *cameraParameters,
    const double *point,
    double imageSigmaPixels,
    double *residuals)
{
    std::array<double, 6> parameters{};
    std::copy_n(cameraParameters, parameters.size(), parameters.begin());
    const Vector3 ground{{point[0], point[1], point[2]}};
    camera_models::linescan::LineScanProjectionResult projection;
    if (!camera_models::linescan::LineScanProjection::projectAtLine(
            camera, ground, observation.linePixels, lineScanPoseBias(parameters), &projection))
    {
        return false;
    }
    residuals[0] = (projection.sample - observation.samplePixels) /
                   imageSigmaPixels;
    residuals[1] = projection.lineResidualPixels / imageSigmaPixels;
    return std::isfinite(residuals[0]) && std::isfinite(residuals[1]);
}

double lineScanImageRms(const PlanetaryLineScanBaWorkingSet &workingSet)
{
    if (workingSet.imageObservations.empty())
    {
        return 0.0;
    }
    double squared = 0.0;
    for (const LineScanImageObservation &observation : workingSet.imageObservations)
    {
        double residuals[2]{};
        if (!evaluateLineScanImageObservation(
                *workingSet.cameraInstances[observation.cameraIndex],
                observation,
                workingSet.cameraParameters[observation.cameraIndex].data(),
                workingSet.tiePoints[observation.pointIndex].data(),
                1.0,
                residuals))
        {
            return std::numeric_limits<double>::infinity();
        }
        squared += residuals[0] * residuals[0] + residuals[1] * residuals[1];
    }
    return std::sqrt(squared / (2.0 * workingSet.imageObservations.size()));
}

double lineScanLaserRangeRms(const PlanetaryLineScanBaWorkingSet &workingSet)
{
    if (workingSet.laserObservations.empty())
    {
        return 0.0;
    }
    double squared = 0.0;
    for (const LineScanLaserObservation &observation : workingSet.laserObservations)
    {
        const auto &camera = workingSet.cameraParameters[observation.cameraIndex];
        const auto &point = workingSet.laserPoints[observation.laserPointIndex]
                                .refinedBodyFixedMeters;
        const Vector3 delta{{
            point[0] - observation.nominalSensorCenterMeters[0] - camera[0],
            point[1] - observation.nominalSensorCenterMeters[1] - camera[1],
            point[2] - observation.nominalSensorCenterMeters[2] - camera[2]}};
        const double residual = vectorNorm(delta) - observation.observedRangeMeters;
        squared += residual * residual;
    }
    return std::sqrt(squared / workingSet.laserObservations.size());
}

} // namespace detail
} // namespace lidar
} // namespace xjw
