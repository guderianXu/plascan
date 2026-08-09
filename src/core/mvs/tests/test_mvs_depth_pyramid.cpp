#include "DepthPyramidPolicy.h"
#include "DepthPyramidEstimator.h"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

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

class OddRoundTripPatchMatchBackend final : public xjw::mvs::IPatchMatchBackend
{
public:
    bool estimate(const xjw::mvs::PatchMatchBackendRequest &request,
                  xjw::mvs::DepthLevelResult &result,
                  std::string *) override
    {
        result.level = request.levelConfig.level;
        result.downsampleFactor = request.levelConfig.patchMatch.downsampleFactor;
        if (request.levelConfig.patchMatch.returnNativeResolution)
        {
            const cv::Size native_size = xjw::mvs::depthPyramidWorkingSize(
                request.referenceImage.cols,
                request.referenceImage.rows,
                result.downsampleFactor);
            result.depth = cv::Mat(native_size, CV_32F, cv::Scalar(0.0f));
            result.confidence = cv::Mat(native_size, CV_32F, cv::Scalar(0.0f));
            result.supportCount = cv::Mat(native_size, CV_16U, cv::Scalar(0));
            result.depth.at<float>(0, 0) = 10.0f;
            result.depth.at<float>(0, 1) = 11.0f;
            result.confidence.at<float>(0, 0) = 0.2f;
            result.confidence.at<float>(0, 1) = 0.8f;
            result.supportCount.at<std::uint16_t>(0, 0) = 2;
            result.supportCount.at<std::uint16_t>(0, 1) = 5;
        }
        else
        {
            result.depth = cv::Mat(request.referenceImage.size(), CV_32F, cv::Scalar(0.0f));
            result.confidence = cv::Mat(result.depth.size(), CV_32F, cv::Scalar(0.8f));
            result.supportCount = cv::Mat(result.depth.size(), CV_16U, cv::Scalar(4));
            for (int index = 0; index < 17; ++index)
            {
                result.depth.ptr<float>()[index] = 10.0f;
            }
        }
        result.validMask = result.depth > 0.0f;
        return true;
    }
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

TEST(MvsDepthPyramidEstimatorTest,
     OddImageSummaryPreservesLegacyNearestRoundTripAndFallbackGate)
{
    OddRoundTripPatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);

    xjw::mvs::DepthPyramidRequest request;
    request.referenceImage = cv::Mat(5, 7, CV_8U, cv::Scalar(100));
    request.sourceImages = {request.referenceImage};
    request.sourceCameras = {xjw::Camera()};
    request.zNear = 1.0f;
    request.zFar = 20.0f;
    request.pyramidConfig.activeLevelCount = 2;
    request.pyramidConfig.levels[0].level = 2;
    request.pyramidConfig.levels[0].patchMatch.downsampleFactor = 2;
    request.pyramidConfig.levels[1].level = 1;
    request.pyramidConfig.levels[1].patchMatch.downsampleFactor = 1;

    const xjw::mvs::DepthPyramidResult result = estimator.estimate(request);

    ASSERT_TRUE(result.success) << result.errorMessage;
    ASSERT_EQ(result.levelSummaries.size(), 2u);
    const xjw::mvs::DepthLevelSummary &coarse = result.levelSummaries.front();
    EXPECT_EQ(coarse.validPixelCount, 6);
    EXPECT_FLOAT_EQ(coarse.validCoverage, 1.0f);
    EXPECT_NEAR(coarse.meanConfidence, 0.4f, 1.0e-6f);
    EXPECT_NEAR(coarse.meanSupportViews, 3.0f, 1.0e-6f);

    EXPECT_FALSE(result.levelSummaries.back().success);
    EXPECT_NE(result.errorMessage.find("coverage regression"), std::string::npos);
    EXPECT_EQ(result.finalLevel.level, 2);
    EXPECT_EQ(result.finalLevel.depth.size(), request.referenceImage.size());
}

TEST(MvsDepthPyramidPropagationTest,
     VirtualNearestNeighborParentMatchesMaterializedFullResolutionParent)
{
    xjw::mvs::DepthLevelResult native_parent;
    native_parent.depth = (cv::Mat_<float>(2, 3) <<
        2.0f, 5.0f, 11.0f,
        3.0f, 9.0f, 17.0f);
    native_parent.confidence = (cv::Mat_<float>(2, 3) <<
        0.95f, 0.70f, 0.35f,
        0.85f, 0.55f, 0.20f);
    native_parent.uncertainty = (cv::Mat_<float>(2, 3) <<
        0.02f, 0.04f, 0.08f,
        0.03f, 0.06f, 0.12f);
    native_parent.validMask = (cv::Mat_<std::uint8_t>(2, 3) <<
        255, 0, 255,
        255, 255, 255);
    native_parent.normalMap = cv::Mat(2, 3, CV_32FC3);
    native_parent.normalMap.at<cv::Vec3f>(0, 0) = cv::Vec3f(1.0f, 0.0f, 1.0f);
    native_parent.normalMap.at<cv::Vec3f>(0, 1) = cv::Vec3f(0.5f, 0.0f, 1.0f);
    native_parent.normalMap.at<cv::Vec3f>(0, 2) = cv::Vec3f(0.0f, 1.0f, 1.0f);
    native_parent.normalMap.at<cv::Vec3f>(1, 0) = cv::Vec3f(-1.0f, 0.0f, 1.0f);
    native_parent.normalMap.at<cv::Vec3f>(1, 1) = cv::Vec3f(0.0f, -1.0f, 1.0f);
    native_parent.normalMap.at<cv::Vec3f>(1, 2) = cv::Vec3f(1.0f, 1.0f, 1.0f);

    const cv::Size logical_parent_size(8, 5);
    const cv::Size target_size(7, 4);
    xjw::mvs::DepthLevelResult materialized_parent = native_parent;
    auto materialize_nearest = [&logical_parent_size](cv::Mat &artifact)
    {
        cv::Mat materialized;
        cv::resize(artifact,
                   materialized,
                   logical_parent_size,
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);
        artifact = std::move(materialized);
    };
    materialize_nearest(materialized_parent.depth);
    materialize_nearest(materialized_parent.confidence);
    materialize_nearest(materialized_parent.uncertainty);
    materialize_nearest(materialized_parent.validMask);
    materialize_nearest(materialized_parent.normalMap);

    cv::Mat guide(logical_parent_size, CV_8U);
    for (int row = 0; row < guide.rows; ++row)
    {
        for (int column = 0; column < guide.cols; ++column)
        {
            guide.at<std::uint8_t>(row, column) = static_cast<std::uint8_t>(
                17 * row + 23 * column + ((row + column) % 3) * 11);
        }
    }

    const xjw::mvs::DepthSearchPrior materialized_prior =
        xjw::mvs::propagateDepthPrior(materialized_parent, guide, target_size);
    const xjw::mvs::DepthSearchPrior virtual_prior =
        xjw::mvs::propagateDepthPrior(
            native_parent,
            guide,
            target_size,
            logical_parent_size);
    const xjw::mvs::DepthSearchPrior direct_native_prior =
        xjw::mvs::propagateDepthPrior(native_parent, guide, target_size);

    ASSERT_FALSE(materialized_prior.center.empty());
    ASSERT_FALSE(virtual_prior.center.empty());
    // Keep the fixture sensitive to the old bug: sampling the native grid directly changes the
    // logical four-neighbor gradient and therefore the propagated search radius.
    EXPECT_GT(cv::norm(materialized_prior.radius,
                       direct_native_prior.radius,
                       cv::NORM_INF),
              1.0e-3);
    EXPECT_EQ(cv::countNonZero(materialized_prior.validMask != virtual_prior.validMask), 0);
    EXPECT_LE(cv::norm(materialized_prior.center, virtual_prior.center, cv::NORM_INF), 1.0e-6);
    EXPECT_LE(cv::norm(materialized_prior.radius, virtual_prior.radius, cv::NORM_INF), 1.0e-6);
    EXPECT_LE(cv::norm(materialized_prior.normalMap,
                       virtual_prior.normalMap,
                       cv::NORM_INF),
              1.0e-6);
}

} // namespace
