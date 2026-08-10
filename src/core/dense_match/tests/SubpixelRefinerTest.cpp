// =============================================================================
// 文件: SubpixelRefinerTest.cpp
// 功能: SubpixelRefiner 单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "SubpixelRefiner.h"
#include <opencv2/core.hpp>

using namespace xjw::dense_match;

TEST(SubpixelRefinerTest, None_Mode_ReturnsOriginal)
{
    cv::Mat disp(2, 2, CV_32FC1, cv::Scalar(3.0f));
    CostVolume empty;
    DenseMatchConfig cfg;
    cfg.subpixel = SubpixelMode::None;
    SubpixelRefiner refiner(cfg);
    auto refined = refiner.refine(disp, empty, 0, 7);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            EXPECT_FLOAT_EQ(refined.at<float>(y, x), 3.0f);
        }
    }
}

TEST(SubpixelRefinerTest, Parabola_Symmetrical_NoChange)
{
    cv::Mat disp(3, 9, CV_32FC1, cv::Scalar(5.0f));
    cv::Mat validMask(disp.size(), CV_8UC1, cv::Scalar(0));
    validMask.colRange(6, 9).setTo(1);
    CostVolume costVol(0, 7, disp.size());
    for (int d = 0; d < 7; ++d)
    {
        for (int y = 0; y < disp.rows; ++y)
        {
            for (int x = 0; x < disp.cols; ++x)
            {
                if (costVol.isValid(static_cast<std::size_t>(d), y, x))
                {
                    costVol[static_cast<std::size_t>(d)].at<float>(y, x) =
                        static_cast<float>((d - 5) * (d - 5));
                }
            }
        }
    }
    DenseMatchConfig cfg;
    cfg.subpixel = SubpixelMode::Parabola;
    SubpixelRefiner refiner(cfg);
    auto refined = refiner.refine(disp, costVol, 0, 7, validMask);
    for (int y = 0; y < disp.rows; ++y)
    {
        for (int x = 6; x < disp.cols; ++x)
        {
            EXPECT_NEAR(refined.at<float>(y, x), 5.0f, 0.01f);
        }
    }
}

TEST(SubpixelRefinerTest, Parabola_SubpixelShift)
{
    cv::Mat disp(1, 8, CV_32FC1, cv::Scalar(4.0f));
    cv::Mat validMask(disp.size(), CV_8UC1, cv::Scalar(0));
    validMask.at<uchar>(0, 7) = 1;
    CostVolume costVol(0, 7, disp.size());
    for (int d = 0; d < 7; ++d)
    {
        costVol[static_cast<std::size_t>(d)].at<float>(0, 7) =
            static_cast<float>((d - 4.3) * (d - 4.3));
    }
    DenseMatchConfig cfg;
    cfg.subpixel = SubpixelMode::Parabola;
    SubpixelRefiner refiner(cfg);
    auto refined = refiner.refine(disp, costVol, 0, 7, validMask);
    EXPECT_NEAR(refined.at<float>(0, 7), 4.3, 0.01);
}

TEST(SubpixelRefinerTest, Parabola_DoesNotReadInvalidNeighborHypothesis)
{
    cv::Mat disparity(1, 6, CV_32FC1, cv::Scalar(5.0f));
    cv::Mat validMask(disparity.size(), CV_8UC1, cv::Scalar(0));
    validMask.at<uchar>(0, 5) = 1;
    CostVolume costVolume(0, 7, disparity.size());
    costVolume[4].at<float>(0, 5) = 4.0f;
    costVolume[5].at<float>(0, 5) = 1.0f;
    ASSERT_FALSE(costVolume.isValid(6, 0, 5));

    DenseMatchConfig config;
    config.subpixel = SubpixelMode::Parabola;
    SubpixelRefiner refiner(config);
    const cv::Mat refined = refiner.refine(
        disparity,
        costVolume,
        0,
        7,
        validMask);

    EXPECT_FLOAT_EQ(refined.at<float>(0, 5), 5.0f);
}
