#include "DepthProvenance.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

TEST(DepthProvenanceTest, TracksMeasuredAndInterpolatedSourcesSeparately)
{
    cv::Mat depth(3, 4, CV_32FC1, cv::Scalar(2.0f));
    depth.at<float>(0, 0) = 0.0f;
    cv::Mat targeted(depth.size(), CV_8UC1, cv::Scalar(0));
    targeted.at<std::uint8_t>(1, 1) = 255;
    cv::Mat provenance = xjw::mvs::initializeDepthProvenance(
        depth, targeted);
    cv::Mat cross_view(depth.size(), CV_8UC1, cv::Scalar(0));
    cross_view.at<std::uint8_t>(1, 2) = 255;
    cv::Mat interpolated(depth.size(), CV_8UC1, cv::Scalar(0));
    interpolated.at<std::uint8_t>(1, 3) = 255;

    xjw::mvs::updateDepthProvenance(
        provenance, depth, targeted, cross_view, interpolated);
    const auto summary = xjw::mvs::summarizeDepthProvenance(
        provenance, depth);

    EXPECT_TRUE(summary.available);
    EXPECT_EQ(summary.validPixelCount, 11);
    EXPECT_EQ(summary.nativePatchMatchPixelCount, 8);
    EXPECT_EQ(summary.targetedPatchMatchPixelCount, 1);
    EXPECT_EQ(summary.crossViewMeasuredPixelCount, 1);
    EXPECT_EQ(summary.anchoredInterpolationPixelCount, 1);
    EXPECT_EQ(summary.unclassifiedValidPixelCount, 0);
    EXPECT_TRUE(xjw::mvs::isInterpolatedDepthProvenance(
        provenance.at<std::uint8_t>(1, 3)));
    EXPECT_FALSE(xjw::mvs::isInterpolatedDepthProvenance(
        provenance.at<std::uint8_t>(1, 2)));
}

TEST(DepthProvenanceTest, ClearsProvenanceWhenDepthIsRemoved)
{
    cv::Mat depth(2, 2, CV_32FC1, cv::Scalar(3.0f));
    cv::Mat provenance = xjw::mvs::initializeDepthProvenance(depth);
    depth.at<float>(0, 1) = 0.0f;

    xjw::mvs::updateDepthProvenance(provenance, depth);

    EXPECT_EQ(provenance.at<std::uint8_t>(0, 1), 0);
    EXPECT_EQ(provenance.at<std::uint8_t>(1, 1),
              static_cast<std::uint8_t>(
                  xjw::mvs::DepthProvenance::NativePatchMatch));
}
