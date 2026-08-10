#include <gtest/gtest.h>
#include "DenseMatchService.h"
#include "BlockMatcher.h"
#include "SgmMatcher.h"
#include "opencv/OpenCVSgbmWrapper.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>

using namespace xjw::dense_match;

class DenseMatchIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        left = cv::Mat(128, 128, CV_8UC1);
        cv::RNG rng(0x8142u);
        rng.fill(left, cv::RNG::UNIFORM, 0, 256);
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
    cfg.subpixel = SubpixelMode::None;
    cfg.medianFilterSize = 0;
    cfg.supportIntensityThreshold = 0;
    cfg.useCuda = false;
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
    cfg.subpixel = SubpixelMode::None;
    cfg.medianFilterSize = 0;
    cfg.supportIntensityThreshold = 0;
    cfg.useCuda = false;
    DenseMatchService service(cfg);
    auto result = service.process(left, right);
    ASSERT_FALSE(result.disparity.empty());
    cv::Rect interior(30, 30, 68, 68);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 10.0, 2.0);
}

TEST_F(DenseMatchIntegrationTest, LeftRightCheckUsesReverseRangeAndOppositeSign)
{
    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc = CostFunction::AbsoluteDifference;
    cfg.subpixel = SubpixelMode::None;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW = 7;
    cfg.corrKernelH = 7;
    cfg.enableLRCheck = true;
    cfg.lrCheckThreshold = 0.1f;
    cfg.medianFilterSize = 0;
    cfg.supportIntensityThreshold = 0;
    cfg.useCuda = false;
    DenseMatchService service(cfg);

    const DisparityResult result = service.process(left, right);

    const cv::Rect interior(30, 20, 68, 88);
    int validCount = 0;
    int accurateCount = 0;
    for (int y = interior.y; y < interior.y + interior.height; ++y)
    {
        for (int x = interior.x; x < interior.x + interior.width; ++x)
        {
            if (result.validMask.at<uchar>(y, x) == 0)
            {
                continue;
            }
            ++validCount;
            if (std::abs(result.disparity.at<float>(y, x) - 10.0f) <= 0.5f)
            {
                ++accurateCount;
            }
        }
    }
    EXPECT_GE(validCount, static_cast<int>(interior.area() * 0.99));
    EXPECT_GE(accurateCount, static_cast<int>(validCount * 0.99));
}

TEST_F(DenseMatchIntegrationTest, NegativeDisparitySurvivesValidation)
{
    constexpr int width = 96;
    constexpr int height = 64;
    constexpr int disparity = -6;
    cv::Mat left(height, width, CV_8UC1);
    cv::Mat right(height, width, CV_8UC1);
    cv::RNG rng(0x59acu);
    rng.fill(left, cv::RNG::UNIFORM, 1, 256);
    rng.fill(right, cv::RNG::UNIFORM, 1, 256);
    left(cv::Rect(0, 0, width + disparity, height))
        .copyTo(right(cv::Rect(-disparity, 0, width + disparity, height)));

    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc = CostFunction::AbsoluteDifference;
    cfg.subpixel = SubpixelMode::None;
    cfg.minDisparity = -16;
    cfg.maxDisparity = 1;
    cfg.corrKernelW = 7;
    cfg.corrKernelH = 7;
    cfg.medianFilterSize = 0;
    cfg.supportIntensityThreshold = 0;
    cfg.useCuda = false;
    DenseMatchService service(cfg);

    const DisparityResult result = service.process(left, right);
    const cv::Rect interior(16, 12, 56, 40);
    ASSERT_GE(cv::countNonZero(result.validMask(interior)),
              static_cast<int>(interior.area() * 0.99));
    double maximumError = 0.0;
    for (int y = interior.y; y < interior.y + interior.height; ++y)
    {
        for (int x = interior.x; x < interior.x + interior.width; ++x)
        {
            if (result.validMask.at<uchar>(y, x) != 0)
            {
                maximumError = std::max(
                    maximumError,
                    static_cast<double>(
                        std::abs(result.disparity.at<float>(y, x) - disparity)));
            }
        }
    }
    EXPECT_LE(maximumError, 0.5);
}

TEST_F(DenseMatchIntegrationTest, OpenCVSgbmUsesWrapperPath)
{
    DenseMatchConfig cfg;
    cfg.algorithm    = StereoAlgorithm::OpenCV_SGBM;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW  = 5;
    cfg.corrKernelH  = 5;
    cfg.supportIntensityThreshold = 0;
    DenseMatchService service(cfg);

    auto result = service.process(left, right);

    ASSERT_FALSE(result.disparity.empty());
    ASSERT_FALSE(result.confidence.empty());
    cv::Rect interior(30, 30, 68, 68);
    ASSERT_GT(cv::countNonZero(result.validMask(interior)), 0);
    cv::Mat validConfidence;
    result.confidence(interior).copyTo(validConfidence, result.validMask(interior));
    double minConfidence = 0.0;
    double maxConfidence = 0.0;
    cv::minMaxLoc(validConfidence, &minConfidence, &maxConfidence, nullptr, nullptr, result.validMask(interior));
    EXPECT_NEAR(minConfidence, 1.0, 1e-6);
    EXPECT_NEAR(maxConfidence, 1.0, 1e-6);
}

TEST_F(DenseMatchIntegrationTest, OpenCVSgbmRejectsNonMultipleOf16RangeWithoutExpandingIt)
{
    DenseMatchConfig config;
    config.algorithm = StereoAlgorithm::OpenCV_SGBM;
    config.minDisparity = -3;
    config.maxDisparity = 14;
    config.supportIntensityThreshold = 0;
    DenseMatchService service(config);

    const DisparityResult result = service.process(left, right);

    EXPECT_TRUE(result.disparity.empty());
    EXPECT_TRUE(result.confidence.empty());
    EXPECT_TRUE(result.validMask.empty());
}

TEST(DenseMatchFailureTest, ReportsSizeAndExtremeDisparityRangeWithoutAllocating)
{
    DenseMatchConfig config;
    config.algorithm = StereoAlgorithm::BlockMatch;
    config.minDisparity = std::numeric_limits<int>::min();
    config.maxDisparity = std::numeric_limits<int>::max();
    config.useCuda = true;
    config.enableLRCheck = false;
    config.medianFilterSize = 0;
    DenseMatchService service(config);
    const cv::Mat image(1, 1, CV_8UC1, cv::Scalar(0));

    const DisparityResult result = service.process(image, image);

    EXPECT_TRUE(result.disparity.empty());
    const std::string &error = service.lastError();
    EXPECT_NE(error.find("size=1x1"), std::string::npos);
    EXPECT_NE(
        error.find("disparity=[-2147483648,2147483647)"),
        std::string::npos);
    EXPECT_NE(error.find("disparity count"), std::string::npos);
}

TEST(DenseMatchFailureTest, RejectsUnrepresentableReverseRange)
{
    DenseMatchConfig config;
    config.algorithm = StereoAlgorithm::BlockMatch;
    config.costFunc = CostFunction::AbsoluteDifference;
    config.subpixel = SubpixelMode::None;
    config.minDisparity = std::numeric_limits<int>::min();
    config.maxDisparity = std::numeric_limits<int>::min() + 1;
    config.corrKernelW = 1;
    config.corrKernelH = 1;
    config.enableLRCheck = true;
    config.lrCheckThreshold = 1.0f;
    config.medianFilterSize = 0;
    config.supportIntensityThreshold = 0;
    config.useCuda = false;
    DenseMatchService service(config);
    const cv::Mat image(1, 1, CV_8UC1, cv::Scalar(1));

    const DisparityResult result = service.process(image, image);

    EXPECT_TRUE(result.disparity.empty());
    EXPECT_NE(service.lastError().find("size=1x1"), std::string::npos);
    EXPECT_NE(service.lastError().find("reverse disparity range"), std::string::npos);
}

TEST(DenseMatchFailureTest, OpenCvWrapperRejectsExtremeDisparitySpan)
{
    DenseMatchConfig config;
    config.algorithm = StereoAlgorithm::OpenCV_SGBM;
    config.minDisparity = std::numeric_limits<int>::min();
    config.maxDisparity = std::numeric_limits<int>::max();
    OpenCVSgbmWrapper wrapper(config);
    const cv::Mat image(1, 1, CV_8UC1, cv::Scalar(0));

    EXPECT_THROW(
        static_cast<void>(wrapper.compute(image, image)),
        cv::Exception);
}

TEST_F(DenseMatchIntegrationTest, SaveAndReloadDisparity)
{
    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc  = CostFunction::AbsoluteDifference;
    cfg.subpixel = SubpixelMode::None;
    cfg.maxDisparity = 32;
    cfg.useCuda = false;
    DenseMatchService service(cfg);
    auto result = service.process(left, right);

    const std::filesystem::path tmpFile =
        std::filesystem::temp_directory_path() / "plascan_test_disparity.tif";
    ASSERT_TRUE(DenseMatchService::saveDisparity(result, tmpFile.string()));

    cv::Mat reloaded = cv::imread(tmpFile.string(), cv::IMREAD_UNCHANGED);
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

    std::filesystem::remove(tmpFile);
}
