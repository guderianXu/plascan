#include "DemMosaic.h"

#include <gtest/gtest.h>

using xjw::DemGridData;
using xjw::DemMosaic;
using xjw::DemMosaicBlendMode;

namespace
{

DemGridData makeGrid(float z00, float z01, float z10, float z11)
{
    DemGridData grid;
    grid.width = 2;
    grid.height = 2;
    grid.stepX = 1.0;
    grid.stepY = 1.0;
    grid.elevation = (cv::Mat_<float>(2, 2) << z00, z01, z10, z11);
    grid.validMask = cv::Mat(2, 2, CV_8UC1, cv::Scalar(255));
    return grid;
}

float z(const DemGridData &grid, int row, int col)
{
    return grid.elevation.at<float>(row, col);
}

} // namespace

TEST(DemMosaic, SupportsFirstLastMeanMedianMinMax)
{
    const DemGridData a = makeGrid(1.0f, 2.0f, 3.0f, 4.0f);
    const DemGridData b = makeGrid(5.0f, 6.0f, 7.0f, 8.0f);
    const DemGridData c = makeGrid(9.0f, 10.0f, 11.0f, 12.0f);

    DemGridData out;
    ASSERT_TRUE(DemMosaic::mosaicSameGrid({a, b, c}, DemMosaicBlendMode::First, &out));
    EXPECT_FLOAT_EQ(z(out, 0, 0), 1.0f);

    ASSERT_TRUE(DemMosaic::mosaicSameGrid({a, b, c}, DemMosaicBlendMode::Last, &out));
    EXPECT_FLOAT_EQ(z(out, 0, 0), 9.0f);

    ASSERT_TRUE(DemMosaic::mosaicSameGrid({a, b, c}, DemMosaicBlendMode::Mean, &out));
    EXPECT_FLOAT_EQ(z(out, 0, 0), 5.0f);

    ASSERT_TRUE(DemMosaic::mosaicSameGrid({a, b, c}, DemMosaicBlendMode::Median, &out));
    EXPECT_FLOAT_EQ(z(out, 0, 0), 5.0f);

    ASSERT_TRUE(DemMosaic::mosaicSameGrid({a, b, c}, DemMosaicBlendMode::Min, &out));
    EXPECT_FLOAT_EQ(z(out, 0, 0), 1.0f);

    ASSERT_TRUE(DemMosaic::mosaicSameGrid({a, b, c}, DemMosaicBlendMode::Max, &out));
    EXPECT_FLOAT_EQ(z(out, 0, 0), 9.0f);
}

TEST(DemMosaic, WeightsByConfidenceAndInverseError)
{
    DemGridData low = makeGrid(100.0f, 100.0f, 100.0f, 100.0f);
    low.confidence = cv::Mat(2, 2, CV_32FC1, cv::Scalar(0.1f));
    low.triangulationError = cv::Mat(2, 2, CV_32FC1, cv::Scalar(10.0f));

    DemGridData high = makeGrid(10.0f, 10.0f, 10.0f, 10.0f);
    high.confidence = cv::Mat(2, 2, CV_32FC1, cv::Scalar(0.9f));
    high.triangulationError = cv::Mat(2, 2, CV_32FC1, cv::Scalar(0.1f));

    DemGridData out;
    ASSERT_TRUE(DemMosaic::mosaicSameGrid({low, high}, DemMosaicBlendMode::ConfidenceWeighted, &out));
    EXPECT_NEAR(z(out, 0, 0), 19.0f, 0.1f);

    ASSERT_TRUE(DemMosaic::mosaicSameGrid({low, high}, DemMosaicBlendMode::InverseErrorWeighted, &out));
    EXPECT_NEAR(z(out, 0, 0), 10.9f, 0.1f);
}

TEST(DemMosaic, PropagatesValidMaskAndRejectsGridMismatch)
{
    DemGridData a = makeGrid(1.0f, 2.0f, 3.0f, 4.0f);
    DemGridData b = makeGrid(5.0f, 6.0f, 7.0f, 8.0f);
    a.validMask.at<uchar>(0, 0) = 0;
    b.validMask.at<uchar>(0, 1) = 0;

    DemGridData out;
    ASSERT_TRUE(DemMosaic::mosaicSameGrid({a, b}, DemMosaicBlendMode::Mean, &out));
    EXPECT_FLOAT_EQ(z(out, 0, 0), 5.0f);
    EXPECT_FLOAT_EQ(z(out, 0, 1), 2.0f);
    EXPECT_EQ(out.validMask.at<uchar>(1, 1), 255);

    b.width = 3;
    QString error;
    EXPECT_FALSE(DemMosaic::mosaicSameGrid({a, b}, DemMosaicBlendMode::Mean, &out, &error));
    EXPECT_FALSE(error.isEmpty());
}
