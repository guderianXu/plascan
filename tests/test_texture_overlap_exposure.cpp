#include <gtest/gtest.h>

#include "TextureOverlapExposure.h"
#include "TextureMappingV4Internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{

using xjw::mesh::texture_v4::ExposureObservation;
using xjw::mesh::texture_v4::ExposureSolveOptions;
using xjw::mesh::texture_v4::ExposureSolveResult;

std::vector<ExposureObservation> twoViewObservations(
    int count,
    double first_scale,
    double second_scale)
{
    std::vector<ExposureObservation> observations;
    observations.reserve(static_cast<std::size_t>(count * 2));
    for (int point = 0; point < count; ++point)
    {
        const double reflectance = 0.10 + 0.01 * point;
        observations.push_back({
            0,
            static_cast<std::uint64_t>(point),
            reflectance * first_scale});
        observations.push_back({
            1,
            static_cast<std::uint64_t>(point),
            reflectance * second_scale});
    }
    return observations;
}

ExposureSolveOptions permissiveOptions()
{
    ExposureSolveOptions options;
    options.minimumPairSamples = 6;
    options.maximumPairLogMad = 0.05;
    return options;
}

} // namespace

TEST(TextureOverlapExposureTest, SolvesGainFromCommonThreeDimensionalPoints)
{
    const ExposureSolveResult result =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            2,
            twoViewObservations(12, 1.05, 0.95),
            permissiveOptions());

    ASSERT_TRUE(result.graphConnected);
    ASSERT_TRUE(result.applied);
    ASSERT_EQ(result.gains.size(), 2U);
    EXPECT_EQ(result.acceptedPairCount, 1U);
    EXPECT_LT(result.gains[0], 1.0f);
    EXPECT_GT(result.gains[1], 1.0f);
    EXPECT_NEAR(result.gains[0] * 1.05, result.gains[1] * 0.95, 1.0e-6);
}

TEST(TextureOverlapExposureTest, PairMedianRejectsSparseColorOutliers)
{
    std::vector<ExposureObservation> observations =
        twoViewObservations(12, 1.04, 0.96);
    observations[1].linearLuminance *= 0.25;
    observations[7].linearLuminance *= 3.0;

    const ExposureSolveResult result =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            2, observations, permissiveOptions());

    ASSERT_TRUE(result.graphConnected);
    EXPECT_EQ(result.rejectedHighMadPairCount, 0U);
    EXPECT_NEAR(result.gains[0] * 1.04, result.gains[1] * 0.96, 1.0e-6);
}

TEST(TextureOverlapExposureTest, HighPairMadFailsClosedToUnitGain)
{
    std::vector<ExposureObservation> observations;
    const std::vector<double> log_ratios{
        -0.40, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40};
    for (std::size_t point = 0; point < log_ratios.size(); ++point)
    {
        observations.push_back({0, point, 0.30});
        observations.push_back({
            1,
            point,
            0.30 / std::exp(log_ratios[point])});
    }

    const ExposureSolveResult result =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            2, observations, permissiveOptions());

    EXPECT_FALSE(result.graphConnected);
    EXPECT_FALSE(result.applied);
    ASSERT_EQ(result.gains.size(), 2U);
    EXPECT_FLOAT_EQ(result.gains[0], 1.0f);
    EXPECT_FLOAT_EQ(result.gains[1], 1.0f);
    EXPECT_EQ(result.rejectedHighMadPairCount, 1U);
    EXPECT_EQ(result.status, "high_pair_mad");
}

TEST(TextureOverlapExposureTest, NoOverlapFailsClosedToUnitGain)
{
    std::vector<ExposureObservation> observations;
    for (std::uint64_t point = 0; point < 8; ++point)
    {
        observations.push_back({0, point, 0.20});
        observations.push_back({1, point + 100, 0.30});
    }

    const ExposureSolveResult result =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            2, observations, permissiveOptions());

    EXPECT_FALSE(result.graphConnected);
    EXPECT_EQ(result.status, "no_overlap");
    EXPECT_EQ(result.gains, std::vector<float>({1.0f, 1.0f}));
}

TEST(TextureOverlapExposureTest, DisconnectedViewGraphSolvesSupportedComponent)
{
    std::vector<ExposureObservation> observations =
        twoViewObservations(8, 1.02, 0.98);
    for (std::uint64_t point = 100; point < 108; ++point)
    {
        observations.push_back({2, point, 0.25});
    }

    const ExposureSolveResult result =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            3, observations, permissiveOptions());

    EXPECT_FALSE(result.graphConnected);
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.acceptedPairCount, 1U);
    EXPECT_EQ(result.status, "applied_partial_components");
    EXPECT_EQ(result.connectedComponentCount, 2);
    EXPECT_EQ(result.correctedViewCount, 2);
    EXPECT_LT(result.gains[0], 1.0f);
    EXPECT_GT(result.gains[1], 1.0f);
    EXPECT_FLOAT_EQ(result.gains[2], 1.0f);
}

TEST(TextureOverlapExposureTest, InsufficientCommonSamplesFailClosed)
{
    const ExposureSolveResult result =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            2,
            twoViewObservations(5, 1.02, 0.98),
            permissiveOptions());

    EXPECT_FALSE(result.graphConnected);
    EXPECT_EQ(result.rejectedInsufficientPairCount, 1U);
    EXPECT_EQ(result.status, "insufficient_overlap_samples");
    EXPECT_EQ(result.gains, std::vector<float>({1.0f, 1.0f}));
}

TEST(TextureOverlapExposureTest, EveryViewGainIsHardLimitedToTenPercent)
{
    const ExposureSolveResult result =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            2,
            twoViewObservations(12, 1.80, 0.45),
            permissiveOptions());

    ASSERT_TRUE(result.graphConnected);
    ASSERT_EQ(result.gains.size(), 2U);
    EXPECT_FLOAT_EQ(result.gains[0], 0.90f);
    EXPECT_FLOAT_EQ(result.gains[1], 1.10f);
}

TEST(TextureOverlapExposureTest, SolutionIsDeterministicAcrossInputOrdering)
{
    std::vector<ExposureObservation> observations =
        twoViewObservations(24, 1.03, 0.97);
    const ExposureSolveResult ordered =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            2, observations, permissiveOptions());
    std::mt19937 generator(42);
    std::shuffle(observations.begin(), observations.end(), generator);
    const ExposureSolveResult shuffled =
        xjw::mesh::texture_v4::solveRobustOverlapExposure(
            2, observations, permissiveOptions());

    EXPECT_EQ(ordered.status, shuffled.status);
    EXPECT_EQ(ordered.gains, shuffled.gains);
    EXPECT_EQ(ordered.observationCount, shuffled.observationCount);
    EXPECT_EQ(ordered.acceptedPairCount, shuffled.acceptedPairCount);
    EXPECT_DOUBLE_EQ(
        ordered.maximumAcceptedLogMad,
        shuffled.maximumAcceptedLogMad);
}

TEST(TextureOverlapExposureTest, AppliesExposureGainInLinearSrgb)
{
    const cv::Vec3f encoded_gray(188.0f, 188.0f, 188.0f);
    const cv::Vec3f linear =
        xjw::mesh::texture_v4::srgb8ToLinear(encoded_gray);
    const cv::Vec3f half_exposure =
        xjw::mesh::texture_v4::applyLinearSrgbExposureGain(
            encoded_gray, 0.5f);

    EXPECT_NEAR(linear[0], 0.503f, 0.003f);
    EXPECT_NEAR(half_exposure[0], 137.0f, 1.0f);
    EXPECT_GT(half_exposure[0], encoded_gray[0] * 0.5f + 30.0f);
    const cv::Vec3f identity =
        xjw::mesh::texture_v4::applyLinearSrgbExposureGain(
            encoded_gray, 1.0f);
    EXPECT_EQ(identity, encoded_gray);
}

TEST(TextureOverlapExposureTest, PipelineCollectsOnlySharedDepthVisiblePoints)
{
    xjw::mesh::texture_v4::PipelineData data;
    std::array<cv::Mat, 2> depths;
    std::array<cv::Mat, 2> confidences;
    std::array<cv::Mat, 2> valid_masks;
    std::array<cv::Mat, 2> support_masks;
    for (int view_index = 0; view_index < 2; ++view_index)
    {
        xjw::mesh::texture_v4::PreparedView view;
        view.sourceIndex = view_index;
        view.evidenceCamera.setIntrinsics(40.0, 40.0, 24.0, 18.0);
        view.evidenceCamera.setPose(
            std::array<double, 9>{1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0},
            std::array<double, 3>{0.0, 0.0, 0.0});
        view.colorCamera = view.evidenceCamera;
        const int encoded_gray = view_index == 0 ? 180 : 150;
        view.colorBgr = cv::Mat(
            36, 48, CV_8UC3, cv::Scalar::all(encoded_gray));
        view.supportDistance = cv::Mat(
            36, 48, CV_32FC1, cv::Scalar(4.0f));
        depths[view_index] = cv::Mat(
            36, 48, CV_32FC1, cv::Scalar(2.0f));
        confidences[view_index] = cv::Mat(
            36, 48, CV_32FC1, cv::Scalar(0.9f));
        valid_masks[view_index] = cv::Mat(
            36, 48, CV_8UC1, cv::Scalar(255));
        support_masks[view_index] = cv::Mat(
            36, 48, CV_8UC1, cv::Scalar(255));
        view.depth = &depths[view_index];
        view.confidence = &confidences[view_index];
        view.depthValidMask = &valid_masks[view_index];
        view.supportMask = &support_masks[view_index];
        data.views.push_back(std::move(view));
    }
    data.medianEdgeLength = 0.10;
    for (int point = 0; point < 32; ++point)
    {
        xjw::mesh::texture_v4::FaceGeometry face;
        face.centroid = {
            -0.40 + 0.16 * (point % 6),
            -0.30 + 0.12 * (point / 6),
            2.0};
        data.geometry.push_back(face);
    }

    xjw::mesh::TextureMappingConfig config;
    config.enableColorCorrection = true;
    config.enableFinalMeshVisibility = false;
    xjw::mesh::TextureMappingResult result;
    std::string error;
    ASSERT_TRUE(xjw::mesh::texture_v4::estimateOverlapExposureGains(
        config, &data, &result, &error)) << error;

    EXPECT_TRUE(result.exposureCorrectionApplied);
    EXPECT_TRUE(result.exposureCorrectionGraphConnected);
    EXPECT_EQ(result.exposureCorrectionStatus, "applied");
    EXPECT_EQ(result.exposureCorrectionObservationCount, 64U);
    EXPECT_EQ(result.exposureCorrectionAcceptedPairCount, 1U);
    EXPECT_EQ(result.exposureCorrectionConnectedComponentCount, 1);
    EXPECT_EQ(result.exposureCorrectionCorrectedViewCount, 2);
    EXPECT_FLOAT_EQ(result.exposureCorrectionMinimumGain, 0.90f);
    EXPECT_FLOAT_EQ(result.exposureCorrectionMaximumGain, 1.10f);
    EXPECT_FLOAT_EQ(data.views[0].exposureGain, 0.90f);
    EXPECT_FLOAT_EQ(data.views[1].exposureGain, 1.10f);
}
