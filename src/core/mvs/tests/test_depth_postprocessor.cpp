#include <cstdint>

#include <gtest/gtest.h>

#include "DepthMissingReason.h"
#include "depth_processing/DepthPostprocessor.h"

using namespace xjw::mvs;

TEST(DepthPostprocessorContract, EmptyAndWrongTypeInputsAreUnchanged)
{
    cv::Mat confidence;
    cv::Mat empty;
    EXPECT_EQ(DepthPostprocessor::postprocessFusionDepthMap(empty, confidence, {}, 0, 4).validAfterPostprocess, 0);
    cv::Mat wrong(4, 4, CV_16UC1, cv::Scalar(10));
    EXPECT_EQ(DepthPostprocessor::removeLocalDepthOutliers(wrong, confidence, 3, 0.25f, 0.2f, 0), 0);
    EXPECT_EQ(DepthPostprocessor::removeSmallDepthComponents(wrong, confidence, 8, 0.2f, 0), 0);
    EXPECT_EQ(wrong.at<std::uint16_t>(0, 0), 10);
}

TEST(DepthPostprocessorContract, SparseSupportIsOnlyASoftConfidencePrior)
{
    cv::Mat depth(4, 4, CV_32FC1, cv::Scalar(10));
    cv::Mat confidence(4, 4, CV_32FC1, cv::Scalar(1));
    cv::Mat support(4, 4, CV_8UC1, cv::Scalar(0));
    support.at<std::uint8_t>(1, 1) = 255;
    DepthPostprocessor::applySparseSupportPrior(depth, confidence, support, 0);
    EXPECT_EQ(cv::countNonZero(depth > 0), 16);
    EXPECT_FLOAT_EQ(confidence.at<float>(0, 0), 0.75f);
    EXPECT_FLOAT_EQ(confidence.at<float>(1, 1), 1.0f);
}

TEST(DepthPostprocessorContract, ExcessiveSpeckleRemovalRollsBack)
{
    cv::Mat depth(8, 8, CV_32FC1, cv::Scalar(0));
    cv::Mat confidence(8, 8, CV_32FC1, cv::Scalar(1));
    depth.at<float>(1, 1) = 10;
    depth.at<float>(6, 6) = 20;
    EXPECT_EQ(DepthPostprocessor::removeSmallDepthComponents(depth, confidence, 4, 0.2f, 0), 0);
    EXPECT_EQ(cv::countNonZero(depth > 0), 2);
    EXPECT_FLOAT_EQ(confidence.at<float>(1, 1), 1);
}

TEST(DepthPostprocessorContract, ConfidenceFilteringRequiresRealGeometryForRetention)
{
    cv::Mat depth(8, 8, CV_32FC1, cv::Scalar(10));
    cv::Mat confidence(8, 8, CV_32FC1, cv::Scalar(0.9f));
    confidence.at<float>(2, 2) = 0.4f;
    confidence.at<float>(5, 5) = 0.4f;
    cv::Mat reasons(8, 8, CV_8UC1, cv::Scalar(0));
    DepthPostProcessEvidence evidence;
    evidence.geometrySupportCount = cv::Mat(8, 8, CV_16UC1, cv::Scalar(0));
    evidence.inverseDepthRelativeSpread = cv::Mat(8, 8, CV_32FC1, cv::Scalar(0.01f));
    evidence.geometrySupportCount.at<std::uint16_t>(2, 2) = 3;
    FusionConfig config;
    config.confidenceThresh = 0.6f;
    config.enableAdaptiveConfidenceFilter = false;
    config.enableLocalDepthOutlierFilter = false;
    config.enableSpeckleFilter = false;
    config.enableBoundaryAwareRetention = false;
    config.enableGeometrySupportedLowConfidenceRetention = true;
    config.geometrySupportedMinimumConfidence = 0.3f;
    config.geometrySupportedMinimumObservationCount = 3;
    config.geometrySupportedMaximumInverseDepthSpread = 0.02f;

    const auto stats =
        DepthPostprocessor::postprocessFusionDepthMap(depth, confidence, config, 0, 4, &reasons, &evidence);
    EXPECT_EQ(stats.validBeforePostprocess, 64);
    EXPECT_EQ(stats.lowConfidenceCandidateCount, 2);
    EXPECT_EQ(stats.geometrySupportedLowConfidenceRetained, 1);
    EXPECT_EQ(stats.confidenceRemoved, 1);
    EXPECT_EQ(stats.validAfterPostprocess, 63);
    EXPECT_FLOAT_EQ(depth.at<float>(2, 2), 10);
    EXPECT_FLOAT_EQ(depth.at<float>(5, 5), 0);
    EXPECT_EQ(reasons.at<std::uint8_t>(5, 5), static_cast<std::uint8_t>(DepthMissingReason::LowConfidence));
}

TEST(DepthPostprocessorContract, LocalSpikeRemovalPreservesReasonAndConfidenceAlignment)
{
    cv::Mat depth(9, 9, CV_32FC1, cv::Scalar(10));
    cv::Mat confidence(9, 9, CV_32FC1, cv::Scalar(1));
    cv::Mat reasons(9, 9, CV_8UC1, cv::Scalar(0));
    depth.at<float>(4, 4) = 100;
    FusionConfig config;
    config.confidenceThresh = 0;
    config.enableSpeckleFilter = false;
    const auto stats = DepthPostprocessor::postprocessFusionDepthMap(depth, confidence, config, 0, 4, &reasons);
    EXPECT_EQ(stats.localDepthOutlierRemoved, 1);
    EXPECT_EQ(stats.validAfterPostprocess, 80);
    EXPECT_FLOAT_EQ(confidence.at<float>(4, 4), 0);
    EXPECT_EQ(reasons.at<std::uint8_t>(4, 4), static_cast<std::uint8_t>(DepthMissingReason::LocalDepthOutlier));
}
