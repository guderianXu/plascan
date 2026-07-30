#include "DepthPoseAlignmentRefiner.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

cv::Matx33d rotationZ(double angle)
{
    return cv::Matx33d(
        std::cos(angle), -std::sin(angle), 0.0,
        std::sin(angle), std::cos(angle), 0.0,
        0.0, 0.0, 1.0);
}

std::vector<xjw::mvs::DepthPoseAlignmentSample> makeSamples()
{
    std::vector<xjw::mvs::DepthPoseAlignmentSample> samples;
    const cv::Matx33d drift_rotation = rotationZ(1.2 * kPi / 180.0);
    const cv::Vec3d drift_translation(0.006, -0.004, 0.003);
    for (int latitude = -4; latitude <= 4; ++latitude)
    {
        for (int longitude = -6; longitude <= 6; ++longitude)
        {
            const double theta =
                static_cast<double>(longitude) * 0.10;
            const double phi =
                static_cast<double>(latitude) * 0.09;
            const cv::Vec3d target(
                std::cos(phi) * std::cos(theta),
                std::cos(phi) * std::sin(theta),
                std::sin(phi));
            xjw::mvs::DepthPoseAlignmentSample sample;
            sample.cameraIndex = 1;
            sample.targetPointWorld = target;
            sample.targetNormalWorld = target;
            sample.sourcePointWorld =
                drift_rotation * target + drift_translation;
            sample.confidence = 0.9;
            samples.push_back(sample);
        }
    }
    return samples;
}

TEST(DepthPoseAlignmentRefinerTest, ReducesRobustPointToPlaneResidual)
{
    const std::vector<xjw::mvs::DepthPoseAlignmentSample> samples =
        makeSamples();
    xjw::mvs::DepthPoseAlignmentOptions options;
    options.enabled = true;
    options.maximumTranslation = 0.02;
    options.maximumRotationDegrees = 2.0;
    options.huberDelta = 0.002;

    const xjw::mvs::DepthPoseAlignmentResult result =
        xjw::mvs::DepthPoseAlignmentRefiner::refine(samples, options);

    ASSERT_EQ(result.corrections.size(), 1);
    const auto &correction = result.corrections.front();
    EXPECT_TRUE(result.acceptedAny);
    EXPECT_TRUE(correction.accepted) << correction.reason;
    EXPECT_LT(correction.residualP90After,
              correction.residualP90Before * 0.2);
    EXPECT_LE(cv::norm(correction.translation),
              options.maximumTranslation);
}

TEST(DepthPoseAlignmentRefinerTest, KeepsAnchorCameraFixed)
{
    std::vector<xjw::mvs::DepthPoseAlignmentSample> samples =
        makeSamples();
    for (auto &sample : samples)
    {
        sample.cameraIndex = 3;
    }
    xjw::mvs::DepthPoseAlignmentOptions options;
    options.enabled = true;
    options.anchorCameraIndex = 3;

    const xjw::mvs::DepthPoseAlignmentResult result =
        xjw::mvs::DepthPoseAlignmentRefiner::refine(samples, options);

    ASSERT_EQ(result.corrections.size(), 1);
    EXPECT_FALSE(result.corrections.front().accepted);
    EXPECT_EQ(result.corrections.front().reason, "anchor_camera");
    EXPECT_LE(cv::norm(result.corrections.front().rotation -
                       cv::Matx33d::eye()),
              1.0e-12);
}

TEST(DepthPoseAlignmentRefinerTest, IgnoresOccludedAndInvalidSamples)
{
    std::vector<xjw::mvs::DepthPoseAlignmentSample> samples =
        makeSamples();
    for (int index = 0; index < 10; ++index)
    {
        xjw::mvs::DepthPoseAlignmentSample outlier = samples[index];
        outlier.targetPointWorld += cv::Vec3d(10.0, -8.0, 6.0);
        outlier.occluded = true;
        samples.push_back(outlier);
    }
    xjw::mvs::DepthPoseAlignmentOptions options;
    options.enabled = true;

    const auto result =
        xjw::mvs::DepthPoseAlignmentRefiner::refine(samples, options);

    ASSERT_EQ(result.corrections.size(), 1);
    EXPECT_TRUE(result.corrections.front().accepted);
    EXPECT_EQ(result.corrections.front().correspondenceCount, 117);
}

TEST(DepthPoseAlignmentRefinerTest, IsDisabledByDefault)
{
    const auto result =
        xjw::mvs::DepthPoseAlignmentRefiner::refine(makeSamples());

    EXPECT_FALSE(result.enabled);
    EXPECT_FALSE(result.acceptedAny);
    EXPECT_TRUE(result.corrections.empty());
}

TEST(DepthPoseAlignmentRefinerTest, RejectsCandidateWithoutResidualImprovement)
{
    std::vector<xjw::mvs::DepthPoseAlignmentSample> samples =
        makeSamples();
    for (auto &sample : samples)
    {
        sample.sourcePointWorld = sample.targetPointWorld;
    }
    xjw::mvs::DepthPoseAlignmentOptions options;
    options.enabled = true;

    const auto result =
        xjw::mvs::DepthPoseAlignmentRefiner::refine(samples, options);

    ASSERT_EQ(result.corrections.size(), 1);
    EXPECT_FALSE(result.corrections.front().accepted);
    EXPECT_EQ(result.corrections.front().reason,
              "insufficient_residual_improvement");
}

} // namespace
