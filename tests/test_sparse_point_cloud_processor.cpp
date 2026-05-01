#include <gtest/gtest.h>

#include "filtering/SparsePointCloudProcessor.h"

#include <vector>

using namespace xjw;

namespace {

SparsePointCloudPoint makePoint(double x,
                                double y,
                                double z,
                                double reproj,
                                double triAngle,
                                int trackLen)
{
    SparsePointCloudPoint point;
    point.x = x;
    point.y = y;
    point.z = z;
    point.rmsReprojPx = reproj;
    point.minTriAngleDeg = triAngle;
    point.trackLen = trackLen;
    return point;
}

} // namespace

TEST(SparsePointCloudProcessorTest, OptimizeCanAppendSpatialCleanup)
{
    std::vector<SparsePointCloudPoint> points = {
        makePoint(0.00, 0.00, 0.00, 0.5, 3.0, 4),
        makePoint(0.02, 0.00, 0.00, 0.6, 3.0, 4),
        makePoint(0.04, 0.00, 0.00, 0.4, 3.0, 4),
        makePoint(5.00, 5.00, 5.00, 0.5, 3.0, 4)
    };

    SparsePointCloudOptimizeOptions options;
    options.filterOptions.filterByReprojError = false;
    options.filterOptions.filterByTrackLen = false;
    options.filterOptions.filterByTriAngle = false;
    options.filterOptions.filterByStatistical = false;
    options.filterOptions.filterByDensity = false;
    options.enableSpatialCleanup = true;
    options.spatialCleanupOptions.voxelSize = 0.2;
    options.spatialCleanupOptions.minVoxelPoints = 2;
    options.spatialCleanupOptions.localReprojFilter = false;

    const SparsePointCloudOptimizeResult result = SparsePointCloudProcessor::optimize(points, options);

    EXPECT_EQ(result.inputPoints, 4);
    EXPECT_EQ(result.outputPoints, 3);
    ASSERT_EQ(result.rounds.size(), 1u);
    EXPECT_EQ(result.rounds.front().filterStats.outputPoints, 4);
}

TEST(SparsePointCloudProcessorTest, RefineMatchesOptimizeWrapper)
{
    std::vector<SparsePointCloudPoint> points = {
        makePoint(0.0, 0.0, 0.0, 0.4, 4.0, 4),
        makePoint(0.1, 0.0, 0.0, 0.5, 3.5, 4),
        makePoint(0.2, 0.0, 0.0, 1.8, 3.0, 3),
        makePoint(10.0, 10.0, 10.0, 0.5, 4.0, 4),
        makePoint(10.2, 10.0, 10.0, 0.6, 4.0, 4),
        makePoint(50.0, 50.0, 50.0, 8.0, 0.3, 1)
    };

    SparsePointCloudRefineOptions refineOptions;
    refineOptions.knnNeighbors = 3;
    refineOptions.stdDevMultiplier = 1.8;
    refineOptions.maxReprojError = 4.0;
    refineOptions.minTriAngleDeg = 1.0;
    refineOptions.minTrackLen = 2;
    refineOptions.iterRounds = 2;
    refineOptions.retriangulate = true;
    refineOptions.normalConsistency = false;

    SparsePointCloudOptimizeOptions optimizeOptions;
    optimizeOptions.filterOptions.maxReprojError = refineOptions.maxReprojError;
    optimizeOptions.filterOptions.minTrackLen = refineOptions.minTrackLen;
    optimizeOptions.filterOptions.minTriAngleDeg = refineOptions.minTriAngleDeg;
    optimizeOptions.filterOptions.statK = refineOptions.knnNeighbors;
    optimizeOptions.filterOptions.statStdDevMul = refineOptions.stdDevMultiplier;
    optimizeOptions.filterOptions.filterByNormalConsistency = refineOptions.normalConsistency;
    optimizeOptions.iterative = true;
    optimizeOptions.iterRounds = refineOptions.iterRounds;
    optimizeOptions.restartFromInputEachRound = refineOptions.retriangulate;
    optimizeOptions.tightenThresholds = true;

    const SparsePointCloudRefineResult refineResult =
        SparsePointCloudProcessor::refine(points, refineOptions);
    const SparsePointCloudOptimizeResult optimizeResult =
        SparsePointCloudProcessor::optimize(points, optimizeOptions);

    EXPECT_EQ(refineResult.inputPoints, optimizeResult.inputPoints);
    EXPECT_EQ(refineResult.outputPoints, optimizeResult.outputPoints);
    EXPECT_EQ(refineResult.removedTotal, optimizeResult.removedTotal);
    ASSERT_EQ(refineResult.rounds.size(), optimizeResult.rounds.size());
}

TEST(SparsePointCloudProcessorTest, LocalOptimWrapperMatchesSpatialCleanup)
{
    std::vector<SparsePointCloudPoint> pointsA = {
        makePoint(0.00, 0.00, 0.00, 0.1, 2.0, 5),
        makePoint(0.02, 0.00, 0.00, 0.2, 2.0, 4),
        makePoint(0.04, 0.00, 0.00, 10.0, 2.0, 3),
        makePoint(2.00, 2.00, 2.00, 0.2, 2.0, 2)
    };
    std::vector<SparsePointCloudPoint> pointsB = pointsA;

    SparsePointCloudSpatialCleanupOptions options;
    options.voxelSize = 0.5;
    options.minVoxelPoints = 2;
    options.localReprojFilter = true;
    options.localReprojStdMul = 0.5;
    options.deduplicationRadius = 0.03;

    const SparsePointCloudSpatialCleanupResult spatialResult =
        SparsePointCloudProcessor::spatialCleanup(&pointsA, options);
    const SparseCloudLocalOptimResult compatResult =
        SparsePointCloudProcessor::localOptim(&pointsB, options);

    EXPECT_EQ(spatialResult.outputPoints, compatResult.outputPoints);
    EXPECT_EQ(spatialResult.removedByVoxelIsolation, compatResult.removedByVoxelIsolation);
    EXPECT_EQ(spatialResult.removedByLocalReproj, compatResult.removedByLocalReproj);
    EXPECT_EQ(spatialResult.removedByDeduplication, compatResult.removedByDeduplication);
    ASSERT_EQ(pointsA.size(), pointsB.size());
    ASSERT_EQ(pointsA.size(), 1u);
    EXPECT_EQ(pointsA.front().trackLen, 5);
    EXPECT_EQ(pointsB.front().trackLen, 5);
}