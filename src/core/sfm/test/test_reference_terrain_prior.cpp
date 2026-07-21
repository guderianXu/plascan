#include "ReferenceTerrainPrior.h"

#include <gtest/gtest.h>

using xjw::BATrack;
using xjw::ReferenceTerrainGrid;
using xjw::ReferenceTerrainPrior;
using xjw::ReferenceTerrainPriorOptions;

TEST(ReferenceTerrainPrior, BilinearSamplesReferenceDemHeight)
{
    ReferenceTerrainGrid grid;
    grid.width = 2;
    grid.height = 2;
    grid.originX = 0.0;
    grid.originY = 0.0;
    grid.pixelSizeX = 1.0;
    grid.pixelSizeY = 1.0;
    grid.nodata = -9999.0;
    grid.heights = {10.0, 12.0,
                    14.0, 16.0};

    bool ok = false;
    const double height = ReferenceTerrainPrior::sampleHeight(grid, 0.5, 0.5, &ok);

    EXPECT_TRUE(ok);
    EXPECT_NEAR(height, 13.0, 1e-9);
}

TEST(ReferenceTerrainPrior, AttachesHeightPlaneConstraintsToNearbyTracks)
{
    ReferenceTerrainGrid grid;
    grid.width = 3;
    grid.height = 3;
    grid.originX = 0.0;
    grid.originY = 0.0;
    grid.pixelSizeX = 1.0;
    grid.pixelSizeY = 1.0;
    grid.nodata = -9999.0;
    grid.heights.assign(9, 5.0);

    std::vector<BATrack> tracks(2);
    tracks[0].initialPoint = {{1.0, 1.0, 5.15}};
    tracks[1].initialPoint = {{1.0, 1.0, 8.0}};

    ReferenceTerrainPriorOptions options;
    options.sigmaMeters = 0.25;
    options.maxAssociationDistanceMeters = 0.5;
    options.huberDeltaMeters = 0.3;

    const auto stats = ReferenceTerrainPrior::attachHeightPlaneConstraints(grid, &tracks, options);

    EXPECT_EQ(stats.inputTrackCount, 2);
    EXPECT_EQ(stats.associatedTrackCount, 1);
    EXPECT_EQ(stats.rejectedByDistanceCount, 1);
    ASSERT_EQ(tracks[0].laserPlaneConstraints.size(), 1);
    EXPECT_NEAR(tracks[0].laserPlaneConstraints.front().point[2], 5.0, 1e-9);
    EXPECT_NEAR(tracks[0].laserPlaneConstraints.front().normal[2], 1.0, 1e-9);
    EXPECT_NEAR(tracks[0].laserPlaneConstraints.front().weight, 4.0, 1e-9);
    EXPECT_TRUE(tracks[1].laserPlaneConstraints.empty());
}

TEST(ReferenceTerrainPrior, ConfiguresBundleAdjustOptionsForSoftHeightPrior)
{
    ReferenceTerrainPriorOptions priorOptions;
    priorOptions.sigmaMeters = 0.5;
    priorOptions.huberDeltaMeters = 0.25;

    const auto baOptions = ReferenceTerrainPrior::makeBundleAdjustOptions(priorOptions);

    EXPECT_TRUE(baOptions.enableLaserPlaneConstraints);
    EXPECT_NEAR(baOptions.laserPlaneWeight, 2.0, 1e-9);
    EXPECT_NEAR(baOptions.laserHuberDeltaMeters, 0.25, 1e-9);
    EXPECT_TRUE(baOptions.refineCameraPose);
}
