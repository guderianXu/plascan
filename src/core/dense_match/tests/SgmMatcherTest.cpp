// =============================================================================
// 文件: SgmMatcherTest.cpp
// 功能: SGM匹配器单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "SgmMatcher.h"
#include <opencv2/imgproc.hpp>

using namespace xjw::dense_match;

TEST(SgmMatcherTest, IdenticalImages_ZeroDisparity)
{
    cv::Mat left(64, 64, CV_8UC1, cv::Scalar(128));
    cv::Mat right = left.clone();
    DenseMatchConfig cfg;
    cfg.algorithm   = StereoAlgorithm::SemiGlobalMatch;
    cfg.costFunc    = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 16;
    cfg.corrKernelW = 3;
    cfg.corrKernelH = 3;
    cfg.p1 = 8;
    cfg.p2 = 32;
    SgmMatcher sgm(cfg);
    auto result = sgm.compute(left, right);
    ASSERT_FALSE(result.disparity.empty());
    cv::Rect interior(8, 8, 48, 48);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 0.0, 1.0);
}

TEST(SgmMatcherTest, OutputHasValidMask)
{
    cv::Mat left(64, 64, CV_8UC1);
    cv::Mat right(64, 64, CV_8UC1);
    cv::randu(left, 0, 256);
    cv::randu(right, 0, 256);
    DenseMatchConfig cfg;
    cfg.corrKernelW = 3;
    cfg.corrKernelH = 3;
    SgmMatcher sgm(cfg);
    auto result = sgm.compute(left, right);
    EXPECT_EQ(result.disparity.size(), left.size());
    EXPECT_EQ(result.confidence.size(), left.size());
    EXPECT_EQ(result.validMask.size(), left.size());
    EXPECT_EQ(result.validMask.type(), CV_8UC1);
}
