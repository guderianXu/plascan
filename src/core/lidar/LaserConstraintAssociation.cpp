#include "LaserConstraintAssociation.h"

#include <algorithm>
#include <cmath>

namespace xjw
{
namespace lidar
{
namespace
{

bool isFinitePoint(const std::array<double, 3> &point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

double signedDistanceToPlane(const std::array<double, 3> &point,
                             const LaserPlaneSample &plane)
{
    return (point[0] - plane.point[0]) * plane.normal[0]
        + (point[1] - plane.point[1]) * plane.normal[1]
        + (point[2] - plane.point[2]) * plane.normal[2];
}

double qualityWeightFactor(const LaserPlaneSample &sample,
                           double distanceMeters,
                           const LaserAssociationOptions &options)
{
    if (!options.enableQualityWeighting)
    {
        return 1.0;
    }

    double factor = 1.0;
    if (options.maxDistanceMeters > 0.0 && std::isfinite(distanceMeters))
    {
        factor *= std::clamp(1.0 - distanceMeters / options.maxDistanceMeters, 0.0, 1.0);
    }
    if (options.maxCurvatureForWeighting > 0.0 && std::isfinite(sample.curvature))
    {
        factor *= std::clamp(1.0 - sample.curvature / options.maxCurvatureForWeighting, 0.0, 1.0);
    }

    const double minFactor = std::clamp(options.minQualityWeight, 0.0, 1.0);
    return std::clamp(std::max(factor, minFactor), 0.0, 1.0);
}

} // namespace

LaserAssociationSummary attachLaserPlaneConstraints(const LaserConstraintMap &map,
                                                    std::vector<xjw::BATrack> *tracks,
                                                    const LaserAssociationOptions &options)
{
    LaserAssociationSummary summary;
    if (!tracks)
    {
        return summary;
    }

    summary.totalTracks = static_cast<int>(tracks->size());
    for (xjw::BATrack &track : *tracks)
    {
        track.laserPlaneConstraints.clear();

        if (!isFinitePoint(track.initialPoint))
        {
            ++summary.rejectedInvalidTrack;
            continue;
        }

        LaserPlaneSample nearest;
        double distance = 0.0;
        if (!map.nearestPlane(track.initialPoint, &nearest, &distance))
        {
            ++summary.rejectedInvalidTrack;
            continue;
        }

        if (options.maxDistanceMeters > 0.0 && distance > options.maxDistanceMeters)
        {
            ++summary.rejectedByDistance;
            continue;
        }

        xjw::BALaserPlaneConstraint constraint;
        constraint.point = nearest.point;
        constraint.normal = nearest.normal;
        constraint.weight = options.weight * qualityWeightFactor(nearest, distance, options);
        constraint.initialSignedDistance = signedDistanceToPlane(track.initialPoint, nearest);
        constraint.sourceFrameIndex = nearest.sourceFrameIndex;
        track.laserPlaneConstraints.push_back(constraint);
        ++summary.associatedTracks;
    }

    return summary;
}

} // namespace lidar
} // namespace xjw
