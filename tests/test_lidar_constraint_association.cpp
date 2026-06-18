#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "LaserConstraintAssociation.h"
#include "LaserConstraintMap.h"

#include <string>
#include <vector>

namespace
{

xjw::lidar::LaserConstraintMap makePlaneMap()
{
    std::vector<xjw::lidar::LaserPlaneSample> samples;

    xjw::lidar::LaserPlaneSample origin;
    origin.point = {{0.0, 0.0, 10.0}};
    origin.normal = {{0.0, 0.0, 1.0}};
    origin.curvature = 0.02;
    samples.push_back(origin);

    xjw::lidar::LaserPlaneSample far;
    far.point = {{5.0, 0.0, 10.0}};
    far.normal = {{0.0, 0.0, 1.0}};
    far.curvature = 0.02;
    samples.push_back(far);

    xjw::lidar::LaserConstraintMapOptions mapOptions;
    mapOptions.maxSamples = 100;

    xjw::lidar::LaserConstraintMap map;
    std::string error;
    EXPECT_TRUE(map.build(samples, mapOptions, &error)) << error;
    return map;
}

} // namespace

TEST(LaserConstraintAssociationTest, AttachesNearestPlaneConstraintToNearbyTrack)
{
    xjw::BATrack nearTrack;
    nearTrack.initialPoint = {{0.1, 0.0, 10.2}};

    xjw::BATrack farTrack;
    farTrack.initialPoint = {{0.0, 0.0, 12.0}};

    std::vector<xjw::BATrack> tracks{nearTrack, farTrack};

    xjw::lidar::LaserAssociationOptions options;
    options.maxDistanceMeters = 0.6;
    options.weight = 2.5;

    const xjw::lidar::LaserAssociationSummary summary =
        xjw::lidar::attachLaserPlaneConstraints(makePlaneMap(), &tracks, options);

    EXPECT_EQ(summary.totalTracks, 2);
    EXPECT_EQ(summary.associatedTracks, 1);
    EXPECT_EQ(summary.rejectedByDistance, 1);

    ASSERT_EQ(tracks[0].laserPlaneConstraints.size(), 1u);
    const xjw::BALaserPlaneConstraint &constraint = tracks[0].laserPlaneConstraints.front();
    EXPECT_NEAR(constraint.point[2], 10.0, 1e-12);
    EXPECT_NEAR(constraint.normal[2], 1.0, 1e-12);
    EXPECT_NEAR(constraint.initialSignedDistance, 0.2, 1e-12);
    EXPECT_DOUBLE_EQ(constraint.weight, 2.5);

    EXPECT_TRUE(tracks[1].laserPlaneConstraints.empty());
}

TEST(LaserConstraintAssociationTest, ClearsStaleConstraintsBeforeAssociating)
{
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 10.1}};
    xjw::BALaserPlaneConstraint stale;
    stale.point = {{99.0, 99.0, 99.0}};
    track.laserPlaneConstraints.push_back(stale);

    std::vector<xjw::BATrack> tracks{track};

    xjw::lidar::LaserAssociationOptions options;
    options.maxDistanceMeters = 0.5;

    const xjw::lidar::LaserAssociationSummary summary =
        xjw::lidar::attachLaserPlaneConstraints(makePlaneMap(), &tracks, options);

    EXPECT_EQ(summary.associatedTracks, 1);
    ASSERT_EQ(tracks[0].laserPlaneConstraints.size(), 1u);
    EXPECT_NEAR(tracks[0].laserPlaneConstraints.front().point[0], 0.0, 1e-12);
}
