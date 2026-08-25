#include <gtest/gtest.h>

#include "filtering/SparsePointCloudProcessor.h"

#include <plapoint/filters/preprocessing.h>

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

TEST(SparsePointCloudProcessorTest, StatisticalFilterUsesRobustThresholdWhenExtremeOutlierInflatesStddev)
{
    std::vector<SparsePointCloudPoint> points = {
        makePoint(0.0,   0.0, 0.0, 0.0, 0.0, 0),
        makePoint(1.0,   0.0, 0.0, 0.0, 0.0, 0),
        makePoint(2.0,   0.0, 0.0, 0.0, 0.0, 0),
        makePoint(3.0,   0.0, 0.0, 0.0, 0.0, 0),
        makePoint(4.0,   0.0, 0.0, 0.0, 0.0, 0),
        makePoint(100.0, 0.0, 0.0, 0.0, 0.0, 0)
    };

    SparsePointCloudFilterOptions options;
    options.filterByReprojError = false;
    options.filterByTrackLen = false;
    options.filterByTriAngle = false;
    options.filterByStatistical = true;
    options.statK = 2;
    options.statStdDevMul = 2.5;
    options.filterByDensity = false;
    options.processingDevice = plapoint::ProcessingDevice::CPU;

    const SparsePointCloudFilterStats stats = SparsePointCloudProcessor::filter(&points, options);

    EXPECT_EQ(stats.inputPoints, 6);
    EXPECT_EQ(stats.removedByStatistical, 1);
    ASSERT_EQ(points.size(), 5u);
    for (const SparsePointCloudPoint &point : points)
    {
        EXPECT_LT(point.x, 50.0);
    }
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
    refineOptions.processingDevice = plapoint::ProcessingDevice::CPU;

    SparsePointCloudOptimizeOptions optimizeOptions;
    optimizeOptions.filterOptions.maxReprojError = refineOptions.maxReprojError;
    optimizeOptions.filterOptions.minTrackLen = refineOptions.minTrackLen;
    optimizeOptions.filterOptions.minTriAngleDeg = refineOptions.minTriAngleDeg;
    optimizeOptions.filterOptions.statK = refineOptions.knnNeighbors;
    optimizeOptions.filterOptions.statStdDevMul = refineOptions.stdDevMultiplier;
    optimizeOptions.filterOptions.filterByNormalConsistency = refineOptions.normalConsistency;
    optimizeOptions.filterOptions.processingDevice = plapoint::ProcessingDevice::CPU;
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

TEST(SparsePointCloudProcessorTest, FilteringPreservesSourceIndices)
{
    std::vector<SparsePointCloudPoint> points = {
        makePoint(0.0, 0.0, 0.0, 0.5, 3.0, 1),
        makePoint(1.0, 0.0, 0.0, 0.5, 3.0, 3),
        makePoint(2.0, 0.0, 0.0, 0.5, 3.0, 4)
    };
    points[0].sourceIndex = 7;
    points[1].sourceIndex = 11;
    points[2].sourceIndex = 19;

    SparsePointCloudFilterOptions options;
    options.filterByReprojError = false;
    options.filterByTrackLen = true;
    options.minTrackLen = 3;
    options.filterByTriAngle = false;
    options.filterByStatistical = false;
    options.filterByDensity = false;

    const SparsePointCloudFilterStats stats =
        SparsePointCloudProcessor::filter(&points, options);

    EXPECT_EQ(stats.removedByTrackLen, 1);
    ASSERT_EQ(points.size(), 2u);
    EXPECT_EQ(points[0].sourceIndex, 11u);
    EXPECT_EQ(points[1].sourceIndex, 19u);
}

TEST(SparsePointCloudProcessorTest, ZeroReprojectionThresholdIsAppliedWithoutClearingAllPoints)
{
    std::vector<SparsePointCloudPoint> points = {
        makePoint(0.0, 0.0, 0.0, 0.0, 3.0, 3),
        makePoint(1.0, 0.0, 0.0, 0.1, 3.0, 3),
        makePoint(2.0, 0.0, 0.0, 0.2, 3.0, 3)
    };

    SparsePointCloudFilterOptions options;
    options.filterByReprojError = true;
    options.maxReprojError = 0.0;
    options.filterByTrackLen = false;
    options.filterByTriAngle = false;
    options.filterByStatistical = false;
    options.filterByDensity = false;

    SparsePointCloudFilterStats stats =
        SparsePointCloudProcessor::filter(&points, options);

    EXPECT_EQ(stats.removedByReprojError, 2);
    ASSERT_EQ(points.size(), 1u);
    EXPECT_DOUBLE_EQ(points.front().rmsReprojPx, 0.0);

    points = {
        makePoint(0.0, 0.0, 0.0, 0.1, 3.0, 3),
        makePoint(1.0, 0.0, 0.0, 0.2, 3.0, 3)
    };
    stats = SparsePointCloudProcessor::filter(&points, options);
    EXPECT_EQ(stats.removedByReprojError, 0);
    EXPECT_EQ(points.size(), 2u);
}

TEST(SparsePointCloudProcessorTest, FiltersUncertaintyAndProjectionAccuracyAboveThreshold)
{
    std::vector<SparsePointCloudPoint> points = {
        makePoint(0.0, 0.0, 0.0, 0.2, 3.0, 4),
        makePoint(1.0, 0.0, 0.0, 0.2, 3.0, 4),
        makePoint(2.0, 0.0, 0.0, 0.2, 3.0, 4),
        makePoint(3.0, 0.0, 0.0, 0.2, 3.0, 4)
    };
    points[0].reconstructionUncertainty = 5.0;
    points[0].projectionAccuracy = 1.0;
    points[1].reconstructionUncertainty = 15.0;
    points[1].projectionAccuracy = 1.0;
    points[2].reconstructionUncertainty = 6.0;
    points[2].projectionAccuracy = 3.0;

    SparsePointCloudFilterOptions options;
    options.filterByReprojError = false;
    options.filterByTrackLen = false;
    options.filterByTriAngle = false;
    options.filterByReconstructionUncertainty = true;
    options.maxReconstructionUncertainty = 10.0;
    options.filterByProjectionAccuracy = true;
    options.maxProjectionAccuracy = 2.0;
    options.filterByStatistical = false;
    options.filterByDensity = false;

    const SparsePointCloudFilterStats stats =
        SparsePointCloudProcessor::filter(&points, options);

    EXPECT_EQ(stats.removedByReconstructionUncertainty, 1);
    EXPECT_EQ(stats.removedByProjectionAccuracy, 1);
    ASSERT_EQ(points.size(), 2u);
    EXPECT_DOUBLE_EQ(points[0].x, 0.0);
    EXPECT_DOUBLE_EQ(points[1].x, 3.0);
}
