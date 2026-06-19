#pragma once

#include "BundleAdjust.h"
#include "LaserConstraintMap.h"

#include <vector>

namespace xjw
{
namespace lidar
{

struct LaserAssociationOptions
{
    double maxDistanceMeters = 1.0;
    double weight = 1.0;
    bool enableQualityWeighting = false;
    double maxCurvatureForWeighting = 0.2;
    double minQualityWeight = 0.05;
};

struct LaserAssociationSummary
{
    int totalTracks = 0;
    int associatedTracks = 0;
    int rejectedByDistance = 0;
    int rejectedInvalidTrack = 0;
};

LaserAssociationSummary attachLaserPlaneConstraints(const LaserConstraintMap &map,
                                                    std::vector<xjw::BATrack> *tracks,
                                                    const LaserAssociationOptions &options);

} // namespace lidar
} // namespace xjw
