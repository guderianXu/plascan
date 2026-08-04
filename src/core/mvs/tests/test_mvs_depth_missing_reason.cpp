#include "DepthMissingReason.h"

#include <gtest/gtest.h>

namespace
{

using xjw::mvs::DepthMissingReason;

std::uint8_t code(DepthMissingReason reason)
{
    return static_cast<std::uint8_t>(reason);
}

TEST(DepthMissingReasonTest, InitializesSupportAndPatchMatchFailures)
{
    cv::Mat depth(2, 3, CV_32FC1, cv::Scalar(0.0f));
    depth.at<float>(0, 0) = 2.0f;
    depth.at<float>(1, 2) = 3.0f;
    cv::Mat support(2, 3, CV_8UC1, cv::Scalar(255));
    support.at<std::uint8_t>(1, 2) = 0;

    const cv::Mat reasons = xjw::mvs::initializeDepthMissingReasonMap(
        depth, support);

    ASSERT_EQ(reasons.type(), CV_8UC1);
    EXPECT_EQ(reasons.at<std::uint8_t>(0, 0), code(DepthMissingReason::Valid));
    EXPECT_EQ(reasons.at<std::uint8_t>(0, 1),
              code(DepthMissingReason::PatchMatchUnresolved));
    EXPECT_EQ(reasons.at<std::uint8_t>(1, 2),
              code(DepthMissingReason::OutsideSupport));
}

TEST(DepthMissingReasonTest, TracksFilterLossAndRestoredPixels)
{
    cv::Mat before(1, 3, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat after = before.clone();
    after.at<float>(0, 1) = 0.0f;
    const cv::Mat support(1, 3, CV_8UC1, cv::Scalar(255));
    cv::Mat reasons = xjw::mvs::initializeDepthMissingReasonMap(
        before, support);

    xjw::mvs::markDepthLossReason(
        reasons, before, after, DepthMissingReason::LowConfidence);
    EXPECT_EQ(reasons.at<std::uint8_t>(0, 1),
              code(DepthMissingReason::LowConfidence));

    after.at<float>(0, 1) = 2.1f;
    xjw::mvs::finalizeDepthMissingReasonMap(reasons, after, support);
    EXPECT_EQ(reasons.at<std::uint8_t>(0, 1), code(DepthMissingReason::Valid));
}

TEST(DepthMissingReasonTest, RefinesUnresolvedPixelsFromGeometryEvidence)
{
    const cv::Mat depth(1, 3, CV_32FC1, cv::Scalar(0.0f));
    const cv::Mat support(1, 3, CV_8UC1, cv::Scalar(255));
    cv::Mat geometry_support(1, 3, CV_16UC1, cv::Scalar(0));
    cv::Mat contradiction(1, 3, CV_16UC1, cv::Scalar(0));
    geometry_support.at<std::uint16_t>(0, 1) = 2;
    contradiction.at<std::uint16_t>(0, 2) = 1;
    cv::Mat reasons = xjw::mvs::initializeDepthMissingReasonMap(
        depth, support);

    xjw::mvs::finalizeDepthMissingReasonMap(
        reasons, depth, support, geometry_support, contradiction);

    EXPECT_EQ(reasons.at<std::uint8_t>(0, 0),
              code(DepthMissingReason::InsufficientGeometrySupport));
    EXPECT_EQ(reasons.at<std::uint8_t>(0, 1),
              code(DepthMissingReason::PatchMatchUnresolved));
    EXPECT_EQ(reasons.at<std::uint8_t>(0, 2),
              code(DepthMissingReason::GeometryContradiction));
}

TEST(DepthMissingReasonTest, SummarizesEveryMissingCategory)
{
    cv::Mat reasons(1, 9, CV_8UC1);
    for (int column = 0; column < reasons.cols; ++column)
    {
        reasons.at<std::uint8_t>(0, column) =
            static_cast<std::uint8_t>(column);
    }
    cv::Mat support(1, 9, CV_8UC1, cv::Scalar(255));
    support.at<std::uint8_t>(0, 1) = 0;

    const auto summary = xjw::mvs::summarizeDepthMissingReasons(
        reasons, support);
    const QJsonObject json = xjw::mvs::depthMissingReasonSummaryToJson(summary);

    EXPECT_TRUE(summary.validInputs);
    EXPECT_EQ(summary.supportPixelCount, 8);
    EXPECT_EQ(summary.validPixelCount, 1);
    EXPECT_EQ(summary.missingPixelCount, 7);
    EXPECT_FLOAT_EQ(summary.missingWithinSupportRatio, 7.0f / 8.0f);
    EXPECT_EQ(json.value(QStringLiteral("schema_version")).toInt(), 1);
}

TEST(DepthMissingReasonTest, PreviewKeepsValidAndOutsidePixelsTransparent)
{
    cv::Mat reasons(1, 3, CV_8UC1);
    reasons.at<std::uint8_t>(0, 0) = code(DepthMissingReason::Valid);
    reasons.at<std::uint8_t>(0, 1) = code(DepthMissingReason::OutsideSupport);
    reasons.at<std::uint8_t>(0, 2) =
        code(DepthMissingReason::GeometryContradiction);

    const cv::Mat preview = xjw::mvs::makeDepthMissingReasonPreview(reasons);

    ASSERT_EQ(preview.type(), CV_8UC4);
    EXPECT_EQ(preview.at<cv::Vec4b>(0, 0)[3], 0);
    EXPECT_EQ(preview.at<cv::Vec4b>(0, 1)[3], 0);
    EXPECT_GT(preview.at<cv::Vec4b>(0, 2)[3], 0);
    EXPECT_GT(preview.at<cv::Vec4b>(0, 2)[2], 0);
}

} // namespace
