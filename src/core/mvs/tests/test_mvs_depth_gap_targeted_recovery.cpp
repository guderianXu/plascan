#include "DepthGapTargetedRecovery.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

namespace
{

TEST(DepthGapTargetedRecoveryTest, BuildsBoundedTargetFromMissingRegion)
{
    cv::Mat depth(24, 24, CV_32FC1, cv::Scalar(4.0f));
    depth(cv::Rect(8, 8, 8, 8)).setTo(0.0f);
    const cv::Mat support(24, 24, CV_8UC1, cv::Scalar(255));
    xjw::mvs::DepthGapTargetedRecoveryOptions options;
    options.minimumGapPixelCount = 16;

    const auto target = xjw::mvs::buildDepthGapTarget(
        depth, support, options);

    EXPECT_TRUE(target.valid);
    EXPECT_EQ(target.requestedGapPixelCount, 64);
    EXPECT_EQ(target.priorCoveredGapPixelCount, 64);
    EXPECT_EQ(cv::countNonZero(target.gapMask), 64);
    EXPECT_GT(cv::countNonZero(target.estimationMask), 64);
    EXPECT_FLOAT_EQ(target.hintDepth.at<float>(12, 12), 4.0f);
    EXPECT_GT(target.hintRadius.at<float>(12, 12), 0.0f);
}

TEST(DepthGapTargetedRecoveryTest, SkipsUnsafeNearFullFrameGap)
{
    cv::Mat depth(20, 20, CV_32FC1, cv::Scalar(0.0f));
    depth(cv::Rect(0, 0, 4, 20)).setTo(3.0f);
    const cv::Mat support(20, 20, CV_8UC1, cv::Scalar(255));
    xjw::mvs::DepthGapTargetedRecoveryOptions options;
    options.minimumGapPixelCount = 1;
    options.maximumGapRatio = 0.50f;

    const auto target = xjw::mvs::buildDepthGapTarget(
        depth, support, options);

    EXPECT_FALSE(target.valid);
    EXPECT_EQ(target.skippedReason,
              QStringLiteral("gap_above_safety_limit"));
}

TEST(DepthGapTargetedRecoveryTest, MergesOnlyConfidentPriorConsistentCandidates)
{
    cv::Mat depth(12, 12, CV_32FC1, cv::Scalar(5.0f));
    depth(cv::Rect(4, 4, 4, 4)).setTo(0.0f);
    const cv::Mat support(12, 12, CV_8UC1, cv::Scalar(255));
    xjw::mvs::DepthGapTargetedRecoveryOptions options;
    options.minimumGapPixelCount = 1;
    const auto target = xjw::mvs::buildDepthGapTarget(
        depth, support, options);
    ASSERT_TRUE(target.valid);

    cv::Mat candidates(12, 12, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat candidate_confidence(12, 12, CV_32FC1, cv::Scalar(0.0f));
    candidates.at<float>(5, 5) = 5.1f;
    candidate_confidence.at<float>(5, 5) = 0.8f;
    candidates.at<float>(5, 6) = 5.0f;
    candidate_confidence.at<float>(5, 6) = 0.1f;
    candidates.at<float>(6, 5) = 8.0f;
    candidate_confidence.at<float>(6, 5) = 0.9f;
    cv::Mat confidence;
    cv::Mat recovered;

    const auto stats = xjw::mvs::mergeTargetedDepthGapCandidates(
        depth,
        confidence,
        candidates,
        candidate_confidence,
        target,
        &recovered,
        options);

    EXPECT_TRUE(stats.attempted);
    EXPECT_EQ(stats.candidatePixelCount, 3);
    EXPECT_EQ(stats.recoveredPixelCount, 1);
    EXPECT_EQ(stats.rejectedConfidencePixelCount, 1);
    EXPECT_EQ(stats.rejectedPriorPixelCount, 1);
    EXPECT_FLOAT_EQ(depth.at<float>(5, 5), 5.1f);
    EXPECT_EQ(cv::countNonZero(recovered), 1);
}

} // namespace
