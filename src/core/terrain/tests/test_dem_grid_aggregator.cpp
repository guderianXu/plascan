#include "DemGridAggregator.h"

#include <gtest/gtest.h>

using xjw::DemGenerationOptions;
using xjw::DemGridAggregator;
using xjw::DemGridData;
using xjw::DemGridSample;

namespace
{

DemGridSample sample(int row,
                     int col,
                     float z,
                     float confidence = 1.0f,
                     float triangulationError = 0.0f)
{
    DemGridSample s;
    s.row = row;
    s.col = col;
    s.elevation = z;
    s.confidence = confidence;
    s.triangulationError = triangulationError;
    return s;
}

float valueAt(const cv::Mat &mat, int row, int col)
{
    return mat.at<float>(row, col);
}

} // namespace

TEST(DemGridAggregator, ComputesMeanMedianCountAndStdDev)
{
    DemGridData meanGrid;
    ASSERT_TRUE(DemGridAggregator::aggregateSamples(
        2,
        2,
        {sample(0, 0, 10.0f), sample(0, 0, 14.0f), sample(1, 1, 3.0f)},
        DemGenerationOptions::ElevationAggregation::Mean,
        &meanGrid));
    EXPECT_FLOAT_EQ(valueAt(meanGrid.elevation, 0, 0), 12.0f);
    EXPECT_EQ(meanGrid.pointCount.at<int>(0, 0), 2);
    EXPECT_EQ(meanGrid.validMask.at<uchar>(1, 1), 255);

    DemGridData medianGrid;
    ASSERT_TRUE(DemGridAggregator::aggregateSamples(
        1,
        1,
        {sample(0, 0, 1.0f), sample(0, 0, 100.0f), sample(0, 0, 3.0f)},
        DemGenerationOptions::ElevationAggregation::Median,
        &medianGrid));
    EXPECT_FLOAT_EQ(valueAt(medianGrid.elevation, 0, 0), 3.0f);

    DemGridData countGrid;
    ASSERT_TRUE(DemGridAggregator::aggregateSamples(
        1,
        1,
        {sample(0, 0, 1.0f), sample(0, 0, 100.0f), sample(0, 0, 3.0f)},
        DemGenerationOptions::ElevationAggregation::Count,
        &countGrid));
    EXPECT_FLOAT_EQ(valueAt(countGrid.elevation, 0, 0), 3.0f);

    DemGridData stdGrid;
    ASSERT_TRUE(DemGridAggregator::aggregateSamples(
        1,
        1,
        {sample(0, 0, 2.0f), sample(0, 0, 4.0f), sample(0, 0, 4.0f), sample(0, 0, 4.0f)},
        DemGenerationOptions::ElevationAggregation::StdDev,
        &stdGrid));
    EXPECT_NEAR(valueAt(stdGrid.elevation, 0, 0), 0.8660254f, 1e-5f);
}

TEST(DemGridAggregator, UsesConfidenceAndErrorForWeightedAverage)
{
    DemGridData grid;
    ASSERT_TRUE(DemGridAggregator::aggregateSamples(
        1,
        1,
        {
            sample(0, 0, 10.0f, 0.9f, 0.1f),
            sample(0, 0, 100.0f, 0.1f, 10.0f),
        },
        DemGenerationOptions::ElevationAggregation::WeightedAverage,
        &grid));

    EXPECT_NEAR(valueAt(grid.elevation, 0, 0), 10.1f, 0.2f);
    EXPECT_NEAR(valueAt(grid.confidence, 0, 0), 0.5f, 1e-5f);
    EXPECT_NEAR(valueAt(grid.triangulationError, 0, 0), 5.05f, 1e-5f);
}

TEST(DemGridAggregator, ComputesNmadAndPercentile80)
{
    DemGridData nmadGrid;
    ASSERT_TRUE(DemGridAggregator::aggregateSamples(
        1,
        1,
        {sample(0, 0, 10.0f), sample(0, 0, 11.0f), sample(0, 0, 12.0f), sample(0, 0, 50.0f)},
        DemGenerationOptions::ElevationAggregation::Nmad,
        &nmadGrid));
    EXPECT_NEAR(valueAt(nmadGrid.elevation, 0, 0), 1.4826f, 1e-4f);

    DemGridData p80Grid;
    ASSERT_TRUE(DemGridAggregator::aggregateSamples(
        1,
        1,
        {sample(0, 0, 1.0f), sample(0, 0, 2.0f), sample(0, 0, 3.0f), sample(0, 0, 4.0f), sample(0, 0, 5.0f)},
        DemGenerationOptions::ElevationAggregation::Percentile80,
        &p80Grid));
    EXPECT_FLOAT_EQ(valueAt(p80Grid.elevation, 0, 0), 5.0f);
}

TEST(DemGridAggregator, RejectsInvalidGridAndIgnoresOutOfRangeSamples)
{
    QString error;
    DemGridData grid;
    EXPECT_FALSE(DemGridAggregator::aggregateSamples(
        0,
        2,
        {sample(0, 0, 1.0f)},
        DemGenerationOptions::ElevationAggregation::Mean,
        &grid,
        &error));
    EXPECT_FALSE(error.isEmpty());

    ASSERT_TRUE(DemGridAggregator::aggregateSamples(
        1,
        1,
        {sample(3, 0, 1.0f), sample(0, 0, 7.0f)},
        DemGenerationOptions::ElevationAggregation::Mean,
        &grid));
    EXPECT_FLOAT_EQ(valueAt(grid.elevation, 0, 0), 7.0f);
    EXPECT_EQ(grid.pointCount.at<int>(0, 0), 1);
}
