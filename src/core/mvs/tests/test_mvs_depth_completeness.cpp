#include "DepthCompletenessMetrics.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

namespace
{

TEST(DepthCompletenessMetricsTest, SeparatesSmallHolesLargeOpeningsAndBoundaryLoss)
{
    cv::Mat mask = cv::Mat::zeros(48, 48, CV_8UC1);
    mask(cv::Rect(2, 2, 44, 44)).setTo(255);
    cv::Mat depth = cv::Mat::zeros(mask.size(), CV_32FC1);
    depth.setTo(2.0f, mask);

    depth(cv::Rect(8, 8, 2, 2)).setTo(0.0f);
    depth(cv::Rect(20, 20, 10, 10)).setTo(0.0f);
    depth(cv::Rect(2, 36, 3, 3)).setTo(0.0f);

    const auto metrics = xjw::mvs::analyzeDepthCompleteness(depth, mask);

    ASSERT_TRUE(metrics.validInputs);
    EXPECT_EQ(metrics.maskPixelCount, 44 * 44);
    EXPECT_EQ(metrics.smallHoleAreaLimit, 64);
    EXPECT_EQ(metrics.smallInteriorHoleCount, 1);
    EXPECT_EQ(metrics.smallInteriorHolePixelCount, 4);
    EXPECT_EQ(metrics.largeInteriorOpeningCount, 1);
    EXPECT_EQ(metrics.largeInteriorOpeningPixelCount, 100);
    EXPECT_EQ(metrics.boundaryConnectedInvalidCount, 1);
    EXPECT_EQ(metrics.boundaryConnectedInvalidPixelCount, 9);
    EXPECT_EQ(metrics.invalidWithinMaskCount, 113);
}

TEST(DepthCompletenessMetricsTest, NeverCountsPixelsOutsideMaskAsMissing)
{
    cv::Mat mask = cv::Mat::zeros(20, 24, CV_8UC1);
    mask(cv::Rect(4, 3, 12, 10)).setTo(17);
    cv::Mat depth = cv::Mat::zeros(mask.size(), CV_32FC1);
    depth.setTo(1.0f, mask);

    const auto metrics = xjw::mvs::analyzeDepthCompleteness(depth, mask);

    ASSERT_TRUE(metrics.validInputs);
    EXPECT_EQ(metrics.maskPixelCount, 120);
    EXPECT_EQ(metrics.validWithinMaskCount, 120);
    EXPECT_EQ(metrics.invalidWithinMaskCount, 0);
    EXPECT_FLOAT_EQ(metrics.validWithinMaskRatio, 1.0f);
}

TEST(DepthCompletenessMetricsTest, RejectsIncompatibleInputs)
{
    const cv::Mat depth(8, 8, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat wrong_type(8, 8, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat wrong_size(7, 8, CV_8UC1, cv::Scalar(255));

    EXPECT_FALSE(xjw::mvs::analyzeDepthCompleteness(depth, wrong_type).validInputs);
    EXPECT_FALSE(xjw::mvs::analyzeDepthCompleteness(depth, wrong_size).validInputs);
}

TEST(DepthCompletenessMetricsTest, SerializesUnavailableInputsWithoutFakeZeros)
{
    xjw::mvs::DepthCompletenessDiagnostics diagnostics;
    const QJsonObject json = xjw::mvs::depthCompletenessDiagnosticsToJson(diagnostics);

    EXPECT_FALSE(json.value(QStringLiteral("available")).toBool(true));
    EXPECT_FALSE(json.contains(QStringLiteral("mask_pixel_count")));
    EXPECT_FALSE(json.contains(QStringLiteral("output_filter_retention_ratio")));
    EXPECT_FALSE(json.contains(QStringLiteral("published_post_consistency_valid_count")));
    EXPECT_FALSE(json.contains(QStringLiteral("published_consistency_retention_ratio")));
    EXPECT_FALSE(json.value(QStringLiteral("consistency_publication_fallback_applied")).toBool());
    EXPECT_EQ(json.value(QStringLiteral("restored_from_prefilter_count")).toInt(-1), 0);
}

TEST(DepthCompletenessMetricsTest, SerializesCrossViewGeometryEvidence)
{
    xjw::mvs::DepthCompletenessDiagnostics diagnostics;
    diagnostics.preConsistencyValidCount = 1000;
    diagnostics.postConsistencyValidCount = 80;
    diagnostics.consistencyRetentionRatio = 0.08f;
    diagnostics.publishedPostConsistencyValidCount = 1000;
    diagnostics.publishedConsistencyRetentionRatio = 1.0f;
    diagnostics.consistencyPublicationFallbackApplied = true;
    diagnostics.consistencyConfirmedObservationCount = 120;
    diagnostics.consistencyOccludedObservationCount = 30;
    diagnostics.consistencyContradictedObservationCount = 12;
    diagnostics.consistencyUnverifiableObservationCount = 7;
    diagnostics.consistencyRejectedPixelCount = 5;
    diagnostics.crossViewRepairedCount = 19;

    const QJsonObject json = xjw::mvs::depthCompletenessDiagnosticsToJson(diagnostics);

    EXPECT_EQ(json.value(QStringLiteral("post_consistency_valid_count")).toInt(), 80);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("consistency_retention_ratio")).toDouble(),
                     static_cast<double>(diagnostics.consistencyRetentionRatio));
    EXPECT_EQ(json.value(QStringLiteral("published_post_consistency_valid_count")).toInt(), 1000);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("published_consistency_retention_ratio")).toDouble(),
                     static_cast<double>(diagnostics.publishedConsistencyRetentionRatio));
    EXPECT_TRUE(json.value(QStringLiteral("consistency_publication_fallback_applied")).toBool());

    EXPECT_EQ(json.value(QStringLiteral("consistency_confirmed_observation_count")).toInt(), 120);
    EXPECT_EQ(json.value(QStringLiteral("consistency_occluded_observation_count")).toInt(), 30);
    EXPECT_EQ(json.value(QStringLiteral("consistency_contradicted_observation_count")).toInt(), 12);
    EXPECT_EQ(json.value(QStringLiteral("consistency_unverifiable_observation_count")).toInt(), 7);
    EXPECT_EQ(json.value(QStringLiteral("consistency_rejected_pixel_count")).toInt(), 5);
    EXPECT_EQ(json.value(QStringLiteral("cross_view_repaired_count")).toInt(), 19);
}

TEST(DepthCompletenessMetricsTest, RestoresOnlyLocallySupportedSmallInteriorHoles)
{
    cv::Mat mask(48, 48, CV_8UC1, cv::Scalar(255));
    cv::Mat depth(48, 48, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat candidate = depth.clone();
    cv::Mat confidence(48, 48, CV_32FC1, cv::Scalar(0.95f));
    depth(cv::Rect(10, 10, 3, 3)).setTo(0.0f);
    candidate(cv::Rect(10, 10, 3, 3)).setTo(2.05f);

    const int restored = xjw::mvs::restoreSmallInteriorDepthHoles(depth, candidate, confidence, mask);

    EXPECT_EQ(restored, 9);
    EXPECT_EQ(cv::countNonZero(depth(cv::Rect(10, 10, 3, 3)) > 0.0f), 9);
}

TEST(DepthCompletenessMetricsTest, PreservesLargeOpeningsAndDepthDiscontinuities)
{
    cv::Mat mask(48, 48, CV_8UC1, cv::Scalar(255));
    cv::Mat depth(48, 48, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat candidate = depth.clone();
    cv::Mat confidence(48, 48, CV_32FC1, cv::Scalar(0.95f));
    depth(cv::Rect(8, 8, 9, 9)).setTo(0.0f);
    depth(cv::Rect(28, 28, 3, 3)).setTo(0.0f);
    candidate(cv::Rect(28, 28, 3, 3)).setTo(4.0f);

    const int restored = xjw::mvs::restoreSmallInteriorDepthHoles(
        depth, candidate, confidence, mask);

    EXPECT_EQ(restored, 0);
    EXPECT_EQ(cv::countNonZero(depth(cv::Rect(8, 8, 9, 9)) > 0.0f), 0);
    EXPECT_EQ(cv::countNonZero(depth(cv::Rect(28, 28, 3, 3)) > 0.0f), 0);
}

} // namespace
