// =============================================================================
// 文件: BlockMatcherTest.cpp
// 功能: WTA块匹配器单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "BlockMatcher.h"
#include <opencv2/imgproc.hpp>

using namespace xjw::dense_match;

static std::pair<cv::Mat, cv::Mat> makeShiftedPair(int w, int h, int shift)
{
    cv::Mat left(h, w, CV_8UC1);
    cv::randu(left, 0, 256);
    cv::Mat right = cv::Mat::zeros(h, w, CV_8UC1);
    left(cv::Rect(shift, 0, w - shift, h)).copyTo(right(cv::Rect(0, 0, w - shift, h)));
    left(cv::Rect(0, 0, shift, h)).copyTo(right(cv::Rect(w - shift, 0, shift, h)));
    return {left, right};
}

TEST(BlockMatcherTest, ZeroShift_ProducesZeroDisparity)
{
    cv::Mat left(64, 64, CV_8UC1, cv::Scalar(128));
    cv::Mat right = left.clone();
    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 16;
    cfg.corrKernelW = 3;
    cfg.corrKernelH = 3;
    BlockMatcher bm(cfg);
    auto result = bm.compute(left, right);
    ASSERT_FALSE(result.disparity.empty());
    cv::Rect interior(8, 8, 48, 48);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 0.0, 1.0);
}

TEST(BlockMatcherTest, KnownShift_ProducesCorrectDisparity)
{
    int w = 128, h = 128, shift = 8;
    auto [left, right] = makeShiftedPair(w, h, shift);
    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW = 11;
    cfg.corrKernelH = 11;
    BlockMatcher bm(cfg);
    auto result = bm.compute(left, right);
    cv::Rect interior(32, 32, 64, 64);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], static_cast<double>(shift), 0.5);
}

TEST(BlockMatcherTest, OutputHasCorrectDimensions)
{
    cv::Mat left(100, 80, CV_8UC1);
    cv::Mat right(100, 80, CV_8UC1);
    cv::randu(left, 0, 256);
    cv::randu(right, 0, 256);
    DenseMatchConfig cfg;
    cfg.corrKernelW = 3;
    cfg.corrKernelH = 3;
    BlockMatcher bm(cfg);
    auto result = bm.compute(left, right);
    EXPECT_EQ(result.disparity.rows, 100);
    EXPECT_EQ(result.disparity.cols, 80);
    EXPECT_FALSE(result.confidence.empty());
    EXPECT_FALSE(result.validMask.empty());
}
