#include "DepthPyramidPolicy.h"
#include "DepthPyramidEstimator.h"

#include <gtest/gtest.h>

namespace
{

class CapturingPatchMatchBackend final : public xjw::mvs::IPatchMatchBackend
{
public:
    bool estimate(const xjw::mvs::PatchMatchBackendRequest &request,
                  xjw::mvs::DepthLevelResult &result,
                  std::string *) override
    {
        capturedReferenceMask = request.referenceValidMask.clone();
        capturedSourceMasks = request.sourceValidMasks;
        result.level = request.levelConfig.level;
        result.downsampleFactor = request.levelConfig.patchMatch.downsampleFactor;
        result.depth = cv::Mat(request.referenceValidMask.size(), CV_32F, cv::Scalar(2.0f));
        result.confidence = cv::Mat(result.depth.size(), CV_32F, cv::Scalar(1.0f));
        return true;
    }

    cv::Mat capturedReferenceMask;
    std::vector<cv::Mat> capturedSourceMasks;
};

TEST(MvsDepthPyramidPolicyTest, SmallImagesKeepAFullResolutionFinalLevel)
{
    xjw::mvs::PatchMatchConfig base_config;
    base_config.downsampleFactor = 4;

    const xjw::mvs::DepthPyramidConfig config =
        xjw::mvs::makeDepthPyramidConfig(base_config, 640, 480);

    ASSERT_EQ(config.activeLevelCount, 2);
    EXPECT_EQ(config.levels[0].level, 2);
    EXPECT_EQ(config.levels[0].patchMatch.downsampleFactor, 2);
    EXPECT_EQ(config.levels[1].level, 1);
    EXPECT_EQ(config.levels[1].patchMatch.downsampleFactor, 1);

    const cv::Size final_size = xjw::mvs::depthPyramidWorkingSize(
        640,
        480,
        config.levels[config.activeLevelCount - 1].patchMatch.downsampleFactor);
    EXPECT_GE(std::min(final_size.width, final_size.height), 320);
}

TEST(MvsDepthPyramidPolicyTest, OneKilopixelHighQualityImagesKeepNativeFinalLevel)
{
    xjw::mvs::PatchMatchConfig base_config;
    base_config.downsampleFactor = 2;

    const xjw::mvs::DepthPyramidConfig config =
        xjw::mvs::makeDepthPyramidConfig(base_config, 1024, 1024);

    ASSERT_EQ(config.activeLevelCount, 3);
    EXPECT_EQ(config.levels[0].patchMatch.downsampleFactor, 4);
    EXPECT_EQ(config.levels[1].patchMatch.downsampleFactor, 2);
    EXPECT_EQ(config.levels[2].patchMatch.downsampleFactor, 1);
}

TEST(MvsDepthPyramidPolicyTest, LargeAerialImagesKeepRequestedFinalDownsample)
{
    xjw::mvs::PatchMatchConfig base_config;
    base_config.downsampleFactor = 4;

    const xjw::mvs::DepthPyramidConfig config =
        xjw::mvs::makeDepthPyramidConfig(base_config, 6000, 4000);

    ASSERT_EQ(config.activeLevelCount, 3);
    EXPECT_EQ(config.levels[0].patchMatch.downsampleFactor, 16);
    EXPECT_EQ(config.levels[1].patchMatch.downsampleFactor, 8);
    EXPECT_EQ(config.levels[2].patchMatch.downsampleFactor, 4);
}

TEST(MvsDepthPyramidEstimatorTest, ResizesReferenceAndSourceMasksAtEveryLevel)
{
    CapturingPatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);

    xjw::mvs::DepthPyramidRequest request;
    request.referenceImage = cv::Mat(6, 8, CV_8U, cv::Scalar(100));
    request.sourceImages = {cv::Mat(6, 8, CV_8U, cv::Scalar(100))};
    request.referenceValidMask = cv::Mat(6, 8, CV_8U, cv::Scalar(0));
    request.referenceValidMask(cv::Rect(0, 0, 4, 6)).setTo(cv::Scalar(255));
    cv::Mat source_mask(6, 8, CV_8U, cv::Scalar(0));
    source_mask(cv::Rect(4, 0, 4, 6)).setTo(cv::Scalar(255));
    request.sourceValidMasks = {source_mask};
    request.referenceCamera = xjw::Camera();
    request.sourceCameras = {xjw::Camera()};
    request.zNear = 1.0f;
    request.zFar = 3.0f;
    request.pyramidConfig.activeLevelCount = 1;
    request.pyramidConfig.levels[0].level = 1;
    request.pyramidConfig.levels[0].patchMatch.downsampleFactor = 2;

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(request);
    ASSERT_TRUE(result.success) << result.errorMessage;
    ASSERT_EQ(backend.capturedReferenceMask.size(), cv::Size(4, 3));
    ASSERT_EQ(backend.capturedSourceMasks.size(), 1u);
    ASSERT_EQ(backend.capturedSourceMasks.front().size(), cv::Size(4, 3));
    EXPECT_EQ(cv::countNonZero(backend.capturedReferenceMask), 6);
    EXPECT_EQ(cv::countNonZero(backend.capturedSourceMasks.front()), 6);
    EXPECT_EQ(backend.capturedReferenceMask.at<std::uint8_t>(1, 0), 255);
    EXPECT_EQ(backend.capturedReferenceMask.at<std::uint8_t>(1, 3), 0);
    EXPECT_EQ(backend.capturedSourceMasks.front().at<std::uint8_t>(1, 0), 0);
    EXPECT_EQ(backend.capturedSourceMasks.front().at<std::uint8_t>(1, 3), 255);
}

} // namespace
