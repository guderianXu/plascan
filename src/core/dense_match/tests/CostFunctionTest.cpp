// =============================================================================
// 文件: CostFunctionTest.cpp
// 功能: 代价函数单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "CostFunctions.h"
#include <opencv2/core.hpp>

using namespace xjw::dense_match;

class CostFunctionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        leftWhite  = cv::Mat(5, 5, CV_8UC1, cv::Scalar(255));
        rightWhite = cv::Mat(5, 5, CV_8UC1, cv::Scalar(255));
        leftBlack  = cv::Mat(5, 5, CV_8UC1, cv::Scalar(0));
        rightBlack = cv::Mat(5, 5, CV_8UC1, cv::Scalar(0));
    }

    cv::Mat leftWhite, rightWhite, leftBlack, rightBlack;
};

TEST_F(CostFunctionTest, AD_IdenticalImages_ZeroCost)
{
    auto volume = computeCostVolume(leftWhite, rightWhite, 0, 5, 3, 3,
                                    CostFunction::AbsoluteDifference);
    ASSERT_EQ(volume.size(), 5u);
    for (int d = 0; d < 5; ++d)
    {
        double minVal, maxVal;
        cv::minMaxLoc(volume[d], &minVal, &maxVal);
        EXPECT_NEAR(minVal, 0.0, 1e-5);
        EXPECT_NEAR(maxVal, 0.0, 1e-5);
    }
}

TEST_F(CostFunctionTest, AD_MaxDifference_Produces255)
{
    auto volume = computeCostVolume(leftWhite, rightBlack, 0, 1, 1, 1,
                                    CostFunction::AbsoluteDifference);
    ASSERT_EQ(volume.size(), 1u);
    double minVal, maxVal;
    cv::minMaxLoc(volume[0], &minVal, &maxVal);
    EXPECT_NEAR(maxVal, 255.0, 1e-5);
}

TEST_F(CostFunctionTest, SD_IdenticalImages_ZeroCost)
{
    auto volume = computeCostVolume(leftWhite, rightWhite, 0, 5, 3, 3,
                                    CostFunction::SquaredDifference);
    ASSERT_EQ(volume.size(), 5u);
    for (int d = 0; d < 5; ++d)
    {
        double minVal, maxVal;
        cv::minMaxLoc(volume[d], &minVal, &maxVal);
        EXPECT_NEAR(minVal, 0.0, 1e-5);
    }
}

TEST_F(CostFunctionTest, NCC_IdenticalImages_ZeroCost)
{
    auto volume = computeCostVolume(leftWhite, rightWhite, 0, 1, 3, 3,
                                    CostFunction::NormalizedCrossCorr);
    ASSERT_EQ(volume.size(), 1u);
    double minVal, maxVal;
    cv::minMaxLoc(volume[0], &minVal, &maxVal);
    EXPECT_NEAR(minVal, 0.0, 0.01);
}

TEST_F(CostFunctionTest, NCC_InverseImages_ProducesCostTwo)
{
    auto volume = computeCostVolume(leftWhite, rightBlack, 0, 1, 3, 3,
                                    CostFunction::NormalizedCrossCorr);
    ASSERT_EQ(volume.size(), 1u);
    double minVal, maxVal;
    cv::minMaxLoc(volume[0], &minVal, &maxVal);
    EXPECT_NEAR(maxVal, 2.0, 0.01);
}

TEST_F(CostFunctionTest, Census_Identical_ZeroCost)
{
    auto volume = computeCostVolume(leftWhite, rightWhite, 0, 1, 3, 3,
                                    CostFunction::CensusTransform);
    ASSERT_EQ(volume.size(), 1u);
    double minVal, maxVal;
    cv::minMaxLoc(volume[0], &minVal, &maxVal);
    EXPECT_NEAR(minVal, 0.0, 1e-5);
}

TEST_F(CostFunctionTest, VolumeCorrectDimensions)
{
    int minDisp = 0, maxDisp = 16;
    auto volume = computeCostVolume(leftWhite, rightWhite, minDisp, maxDisp,
                                    5, 5, CostFunction::AbsoluteDifference);
    ASSERT_EQ(volume.size(), static_cast<size_t>(maxDisp - minDisp));
    for (const auto &slice : volume)
    {
        EXPECT_EQ(slice.rows, leftWhite.rows);
        EXPECT_EQ(slice.cols, leftWhite.cols);
    }
}
