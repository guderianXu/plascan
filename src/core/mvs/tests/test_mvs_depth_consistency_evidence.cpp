#include "DepthConsistencyEvidencePolicy.h"
#include "DepthConsensusBiasPolicy.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>

namespace
{

TEST(DepthConsistencyEvidencePolicyTest,
     RetainsSingleConfirmationWithoutContradictionAtReducedConfidence)
{
    cv::Mat original_depth(1, 3, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat original_confidence(1, 3, CV_32FC1, cv::Scalar(0.8f));
    cv::Mat support_mask(1, 3, CV_8UC1, cv::Scalar(255));
    cv::Mat consistent_votes(1, 3, CV_16UC1, cv::Scalar(0));
    cv::Mat contradicted_votes(1, 3, CV_16UC1, cv::Scalar(0));
    cv::Mat filtered_depth(1, 3, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat filtered_confidence = original_confidence.clone();
    consistent_votes.at<std::uint16_t>(0, 0) = 1;
    consistent_votes.at<std::uint16_t>(0, 1) = 1;
    contradicted_votes.at<std::uint16_t>(0, 1) = 1;

    xjw::mvs::WeakNativeDepthRetentionOptions options;
    options.confidenceMultiplier = 0.5f;
    options.minimumRetainedConfidence = 0.0f;
    const xjw::mvs::WeakNativeDepthRetentionStats stats =
        xjw::mvs::retainWeaklyVerifiedNativeDepth(
            original_depth,
            original_confidence,
            support_mask,
            consistent_votes,
            contradicted_votes,
            options,
            &filtered_depth,
            &filtered_confidence);

    EXPECT_EQ(stats.consideredPixelCount, 3U);
    EXPECT_EQ(stats.retainedPixelCount, 1U);
    EXPECT_EQ(stats.rejectedContradictionPixelCount, 1U);
    EXPECT_EQ(stats.rejectedNoConfirmationPixelCount, 1U);
    EXPECT_FLOAT_EQ(filtered_depth.at<float>(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(filtered_confidence.at<float>(0, 0), 0.4f);
    EXPECT_FLOAT_EQ(filtered_depth.at<float>(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(filtered_depth.at<float>(0, 2), 0.0f);
}

TEST(DepthConsistencyEvidencePolicyTest, DoesNotReplaceAlreadyAcceptedDepth)
{
    cv::Mat original_depth(1, 1, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat original_confidence(1, 1, CV_32FC1, cv::Scalar(0.8f));
    cv::Mat support_mask(1, 1, CV_8UC1, cv::Scalar(255));
    cv::Mat consistent_votes(1, 1, CV_16UC1, cv::Scalar(1));
    cv::Mat contradicted_votes(1, 1, CV_16UC1, cv::Scalar(0));
    cv::Mat filtered_depth(1, 1, CV_32FC1, cv::Scalar(1.9f));
    cv::Mat filtered_confidence(1, 1, CV_32FC1, cv::Scalar(0.7f));

    const xjw::mvs::WeakNativeDepthRetentionStats stats =
        xjw::mvs::retainWeaklyVerifiedNativeDepth(
            original_depth,
            original_confidence,
            support_mask,
            consistent_votes,
            contradicted_votes,
            {},
            &filtered_depth,
            &filtered_confidence);

    EXPECT_EQ(stats.consideredPixelCount, 0U);
    EXPECT_FLOAT_EQ(filtered_depth.at<float>(0, 0), 1.9f);
    EXPECT_FLOAT_EQ(filtered_confidence.at<float>(0, 0), 0.7f);
}

TEST(DepthConsistencyEvidencePolicyTest,
     CanRetainUnconfirmedNativeDepthButNeverContradictedDepth)
{
    cv::Mat original_depth(1, 2, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat original_confidence(1, 2, CV_32FC1, cv::Scalar(0.85f));
    cv::Mat support_mask(1, 2, CV_8UC1, cv::Scalar(255));
    cv::Mat consistent_votes(1, 2, CV_16UC1, cv::Scalar(0));
    cv::Mat contradicted_votes(1, 2, CV_16UC1, cv::Scalar(0));
    contradicted_votes.at<std::uint16_t>(0, 1) = 1;
    cv::Mat filtered_depth(1, 2, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat filtered_confidence = original_confidence.clone();
    xjw::mvs::WeakNativeDepthRetentionOptions options;
    options.retainUnconfirmedWithoutContradiction = true;

    const xjw::mvs::WeakNativeDepthRetentionStats stats =
        xjw::mvs::retainWeaklyVerifiedNativeDepth(
            original_depth,
            original_confidence,
            support_mask,
            consistent_votes,
            contradicted_votes,
            options,
            &filtered_depth,
            &filtered_confidence);

    EXPECT_EQ(stats.retainedPixelCount, 1U);
    EXPECT_EQ(stats.retainedUnconfirmedPixelCount, 1U);
    EXPECT_EQ(stats.rejectedContradictionPixelCount, 1U);
    EXPECT_FLOAT_EQ(filtered_depth.at<float>(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(filtered_depth.at<float>(0, 1), 0.0f);
}

TEST(ReferenceAnchoredDepthConsensusTest,
     RemovesUniformConsensusBiasWithoutMovingReferenceDepth)
{
    const cv::Size size(16, 16);
    cv::Mat raw_depth(size, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat inverse_depth_mean(size, CV_32FC1, cv::Scalar(1.0f / 2.02f));
    cv::Mat geometry_support(size, CV_16UC1, cv::Scalar(4));
    cv::Mat inverse_depth_spread(size, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat confidence(size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat repaired_mask(size, CV_8UC1, cv::Scalar(0));
    cv::Mat support_mask(size, CV_8UC1, cv::Scalar(255));
    cv::Mat depth_valid_mask(size, CV_8UC1, cv::Scalar(255));
    xjw::mvs::ReferenceAnchoredDepthConsensusOptions options;
    options.contourExclusionPixels = 0;
    options.minimumBiasSampleCount = 1;
    options.blendWeight = 1.0f;
    options.maximumCandidateRelativeDifference = 0.02f;
    options.maximumAppliedRelativeCorrection = 0.02f;

    const auto result = xjw::mvs::makeReferenceAnchoredDepthConsensus(
        raw_depth,
        inverse_depth_mean,
        geometry_support,
        inverse_depth_spread,
        confidence,
        repaired_mask,
        support_mask,
        depth_valid_mask,
        options);

    ASSERT_TRUE(result.calibration.valid);
    EXPECT_EQ(result.calibration.sampleCount, 256);
    EXPECT_NEAR(result.calibration.additiveDepthBias, 0.02f, 1.0e-5f);
    EXPECT_LT(cv::norm(result.depth, raw_depth, cv::NORM_INF), 1.0e-6);
    EXPECT_EQ(result.appliedPixelCount, 0);
}

TEST(ReferenceAnchoredDepthConsensusTest,
     ReducesZeroMedianUnbiasedNoiseAfterBiasCalibration)
{
    const cv::Size size(33, 1);
    cv::Mat raw_depth(size, CV_32FC1);
    for (int column = 0; column < raw_depth.cols; ++column)
    {
        raw_depth.at<float>(0, column) =
            2.0f + static_cast<float>((column % 3) - 1) * 0.02f;
    }
    cv::Mat inverse_depth_mean(size, CV_32FC1, cv::Scalar(0.5f));
    cv::Mat geometry_support(size, CV_16UC1, cv::Scalar(4));
    cv::Mat inverse_depth_spread(size, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat confidence(size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat repaired_mask(size, CV_8UC1, cv::Scalar(0));
    cv::Mat support_mask(size, CV_8UC1, cv::Scalar(255));
    cv::Mat depth_valid_mask(size, CV_8UC1, cv::Scalar(255));
    xjw::mvs::ReferenceAnchoredDepthConsensusOptions options;
    options.contourExclusionPixels = 0;
    options.minimumBiasSampleCount = 1;
    options.blendWeight = 1.0f;
    options.maximumCandidateRelativeDifference = 0.02f;
    options.maximumAppliedRelativeCorrection = 0.02f;

    const auto result = xjw::mvs::makeReferenceAnchoredDepthConsensus(
        raw_depth,
        inverse_depth_mean,
        geometry_support,
        inverse_depth_spread,
        confidence,
        repaired_mask,
        support_mask,
        depth_valid_mask,
        options);

    ASSERT_TRUE(result.calibration.valid);
    EXPECT_NEAR(result.calibration.additiveDepthBias, 0.0f, 1.0e-6f);
    EXPECT_EQ(cv::countNonZero(result.eligibleMask), 33);
    EXPECT_EQ(result.appliedPixelCount, 22);
    double before_error = 0.0;
    double after_error = 0.0;
    for (int column = 0; column < raw_depth.cols; ++column)
    {
        before_error += std::fabs(raw_depth.at<float>(0, column) - 2.0f);
        after_error += std::fabs(result.depth.at<float>(0, column) - 2.0f);
    }
    EXPECT_LT(after_error, before_error * 0.1);
}

TEST(ReferenceAnchoredDepthConsensusTest,
     LeavesContourAndRepairedPixelsAtReferenceDepth)
{
    const cv::Size size(15, 15);
    cv::Mat raw_depth(size, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat inverse_depth_mean(size, CV_32FC1, cv::Scalar(0.5f));
    cv::Mat geometry_support(size, CV_16UC1, cv::Scalar(4));
    cv::Mat inverse_depth_spread(size, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat confidence(size, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat repaired_mask(size, CV_8UC1, cv::Scalar(0));
    cv::Mat support_mask(size, CV_8UC1, cv::Scalar(0));
    cv::rectangle(support_mask, cv::Rect(1, 1, 13, 13), cv::Scalar(255), cv::FILLED);
    cv::Mat depth_valid_mask = support_mask.clone();
    repaired_mask.at<std::uint8_t>(7, 7) = 255;
    inverse_depth_mean.at<float>(1, 7) = 1.0f / 2.01f;
    inverse_depth_mean.at<float>(7, 7) = 1.0f / 2.01f;
    inverse_depth_mean.at<float>(7, 8) = 1.0f / 2.01f;
    xjw::mvs::ReferenceAnchoredDepthConsensusOptions options;
    options.contourExclusionPixels = 2;
    options.minimumBiasSampleCount = 1;
    options.blendWeight = 1.0f;
    options.maximumCandidateRelativeDifference = 0.02f;
    options.maximumAppliedRelativeCorrection = 0.02f;

    const auto result = xjw::mvs::makeReferenceAnchoredDepthConsensus(
        raw_depth,
        inverse_depth_mean,
        geometry_support,
        inverse_depth_spread,
        confidence,
        repaired_mask,
        support_mask,
        depth_valid_mask,
        options);

    ASSERT_TRUE(result.calibration.valid);
    EXPECT_EQ(result.eligibleMask.at<std::uint8_t>(1, 7), 0);
    EXPECT_EQ(result.eligibleMask.at<std::uint8_t>(7, 7), 0);
    EXPECT_EQ(result.eligibleMask.at<std::uint8_t>(7, 8), 255);
    EXPECT_FLOAT_EQ(result.depth.at<float>(1, 7), 2.0f);
    EXPECT_FLOAT_EQ(result.depth.at<float>(7, 7), 2.0f);
    EXPECT_NEAR(result.depth.at<float>(7, 8), 2.01f, 1.0e-5f);
    EXPECT_EQ(result.appliedPixelCount, 1);
}

} // namespace
