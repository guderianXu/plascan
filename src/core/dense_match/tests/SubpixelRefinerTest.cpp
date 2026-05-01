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
    cv::Mat disp(3, 3, CV_32FC1, cv::Scalar(5.0f));
    CostVolume costVol(7);
    for (int d = 0; d < 7; ++d)
    {
        costVol[d] = cv::Mat(3, 3, CV_32FC1);
        for (int y = 0; y < 3; ++y)
        {
            for (int x = 0; x < 3; ++x)
            {
                costVol[d].at<float>(y, x) = static_cast<float>((d - 5) * (d - 5));
            }
        }
    }
    DenseMatchConfig cfg;
    cfg.subpixel = SubpixelMode::Parabola;
    SubpixelRefiner refiner(cfg);
    auto refined = refiner.refine(disp, costVol, 0, 7);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            EXPECT_NEAR(refined.at<float>(y, x), 5.0f, 0.01f);
        }
    }
}

TEST(SubpixelRefinerTest, Parabola_SubpixelShift)
{
    cv::Mat disp(1, 1, CV_32FC1, cv::Scalar(4.0f));
    CostVolume costVol(7);
    for (int d = 0; d < 7; ++d)
    {
        costVol[d] = cv::Mat(1, 1, CV_32FC1);
        costVol[d].at<float>(0, 0) = static_cast<float>((d - 4.3) * (d - 4.3));
    }
    DenseMatchConfig cfg;
    cfg.subpixel = SubpixelMode::Parabola;
    SubpixelRefiner refiner(cfg);
    auto refined = refiner.refine(disp, costVol, 0, 7);
    EXPECT_NEAR(refined.at<float>(0, 0), 4.3, 0.5);
}
