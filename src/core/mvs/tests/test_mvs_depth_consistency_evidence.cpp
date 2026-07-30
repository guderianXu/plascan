#include "DepthConsistencyEvidencePolicy.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

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

} // namespace
