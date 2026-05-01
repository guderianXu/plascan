#include <gtest/gtest.h>
#include "DenseMatchService.h"
#include "BlockMatcher.h"
#include "SgmMatcher.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>

using namespace xjw::dense_match;

class DenseMatchIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        left = cv::Mat(128, 128, CV_8UC1);
        cv::randu(left, 0, 256);
        right = cv::Mat::zeros(128, 128, CV_8UC1);
        left(cv::Rect(10, 0, 118, 128)).copyTo(right(cv::Rect(0, 0, 118, 128)));
        left(cv::Rect(0, 0, 10, 128)).copyTo(right(cv::Rect(118, 0, 10, 128)));
    }
    cv::Mat left, right;
};

TEST_F(DenseMatchIntegrationTest, BM_EndToEnd)
{
    DenseMatchConfig cfg;
    cfg.algorithm    = StereoAlgorithm::BlockMatch;
    cfg.costFunc     = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW  = 9;
    cfg.corrKernelH  = 9;
    DenseMatchService service(cfg);
    auto result = service.process(left, right);
    ASSERT_FALSE(result.disparity.empty());
    cv::Rect interior(30, 30, 68, 68);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 10.0, 0.5);
}

TEST_F(DenseMatchIntegrationTest, SGM_EndToEnd)
{
    DenseMatchConfig cfg;
    cfg.algorithm    = StereoAlgorithm::SemiGlobalMatch;
    cfg.costFunc     = CostFunction::CensusTransform;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW  = 5;
    cfg.corrKernelH  = 5;
    cfg.p1 = 8;
    cfg.p2 = 32;
    DenseMatchService service(cfg);
    auto result = service.process(left, right);
    ASSERT_FALSE(result.disparity.empty());
    cv::Rect interior(30, 30, 68, 68);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 10.0, 2.0);
}

TEST_F(DenseMatchIntegrationTest, SaveAndReloadDisparity)
{
    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc  = CostFunction::AbsoluteDifference;
    DenseMatchService service(cfg);
    auto result = service.process(left, right);

    const char *tmpFile = "/tmp/test_disparity.tif";
    ASSERT_TRUE(DenseMatchService::saveDisparity(result, tmpFile));

    cv::Mat reloaded = cv::imread(tmpFile, cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(reloaded.empty());

    double maxDiff = 0;
    for (int y = 0; y < left.rows; ++y)
    {
        for (int x = 0; x < left.cols; ++x)
        {
            maxDiff = std::max(maxDiff,
                (double)std::abs(result.disparity.at<float>(y, x) - reloaded.at<float>(y, x)));
        }
    }
    EXPECT_LT(maxDiff, 0.5);

    std::remove(tmpFile);
}
