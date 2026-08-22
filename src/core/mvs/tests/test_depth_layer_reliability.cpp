#include "DepthLayerReliability.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <cstdint>
#include <chrono>

namespace
{

struct ReliabilityFixture
{
    cv::Mat depth = cv::Mat(96, 96, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat guide = cv::Mat(96, 96, CV_8UC1, cv::Scalar(128));
    cv::Mat effective = cv::Mat(96, 96, CV_32FC1, cv::Scalar(3.0f));
    cv::Mat conflict = cv::Mat(96, 96, CV_32FC1, cv::Scalar(0.10f));
    cv::Mat spread = cv::Mat(96, 96, CV_32FC1, cv::Scalar(0.001f));
};

int classCount(const cv::Mat &map,
               xjw::mvs::DepthLayerReliabilityClass reliability_class)
{
    return cv::countNonZero(
        map == static_cast<std::uint8_t>(reliability_class));
}

TEST(DepthLayerReliabilityTest, RejectsIncompatibleEvidenceWithoutOutput)
{
    ReliabilityFixture fixture;
    fixture.conflict.release();
    const auto result = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth,
        fixture.guide,
        fixture.effective,
        fixture.conflict,
        fixture.spread);

    EXPECT_FALSE(result.validInputs);
    EXPECT_TRUE(result.classMap.empty());
    EXPECT_EQ(result.errorMessage,
              "missing_or_incompatible_depth_geometry_evidence");
}

TEST(DepthLayerReliabilityTest, StrongGeometryKeepsSmoothPlaneReliable)
{
    ReliabilityFixture fixture;
    const auto result = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth,
        fixture.guide,
        fixture.effective,
        fixture.conflict,
        fixture.spread);

    ASSERT_TRUE(result.validInputs);
    EXPECT_EQ(result.candidatePixelCount, 0);
    EXPECT_EQ(result.rejectedLayerPixelCount, 0);
    EXPECT_EQ(classCount(result.classMap,
                         xjw::mvs::DepthLayerReliabilityClass::Reliable),
              fixture.depth.rows * fixture.depth.cols);
}

TEST(DepthLayerReliabilityTest, MarksCoherentWeakWrongLayer)
{
    ReliabilityFixture fixture;
    const cv::Rect island(32, 36, 30, 22);
    fixture.depth(island).setTo(2.06f);
    fixture.effective(island).setTo(1.0f);
    fixture.conflict(island).setTo(0.75f);
    fixture.spread(island).setTo(0.006f);

    const auto result = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth,
        fixture.guide,
        fixture.effective,
        fixture.conflict,
        fixture.spread);

    ASSERT_TRUE(result.validInputs);
    EXPECT_EQ(result.rejectedLayerComponentCount, 1);
    EXPECT_EQ(result.rejectedLayerPixelCount, island.area());
    EXPECT_EQ(classCount(
                  result.classMap(island),
                  xjw::mvs::DepthLayerReliabilityClass::RejectedLayer),
              island.area());
    ASSERT_FALSE(result.components.empty());
    EXPECT_GE(result.components.front().absoluteRelativeResidualMedian, 0.02f);
    EXPECT_LE(result.components.front().boundarySurfaceFitP90, 1.0e-5f);
}

void fillCurvedSurface(cv::Mat &depth)
{
    for (int y = 0; y < depth.rows; ++y)
    {
        float *row = depth.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x)
        {
            const float u = (x + 0.5f - 48.0f) / 48.0f;
            const float v = (y + 0.5f - 48.0f) / 48.0f;
            const float inverse_depth = 0.5f + 0.025f * u +
                0.018f * v + 0.035f * u * u - 0.020f * u * v +
                0.028f * v * v;
            row[x] = 1.0f / inverse_depth;
        }
    }
}

TEST(DepthLayerReliabilityTest, RejectsWrongLayerOnSmoothCurvedSurface)
{
    ReliabilityFixture fixture;
    fillCurvedSurface(fixture.depth);
    const cv::Rect island(30, 33, 36, 30);
    fixture.depth(island) *= 1.03f;
    fixture.effective(island).setTo(1.0f);
    fixture.conflict(island).setTo(0.75f);
    fixture.spread(island).setTo(0.006f);

    const auto result = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth,
        fixture.guide,
        fixture.effective,
        fixture.conflict,
        fixture.spread);

    ASSERT_TRUE(result.validInputs);
    ASSERT_FALSE(result.components.empty());
    EXPECT_EQ(result.rejectedLayerPixelCount, island.area());
    EXPECT_EQ(classCount(
                  result.classMap(island),
                  xjw::mvs::DepthLayerReliabilityClass::RejectedLayer),
              island.area());
    EXPECT_LE(result.components.front().boundarySurfaceFitP90, 1.0e-5f);
    EXPECT_GE(result.components.front().absoluteRelativeResidualMedian, 0.02f);
}

TEST(DepthLayerReliabilityTest, KeepsCorrectPatchOnSmoothCurvedSurface)
{
    ReliabilityFixture fixture;
    fillCurvedSurface(fixture.depth);
    const cv::Rect patch(30, 33, 36, 30);
    fixture.effective(patch).setTo(1.0f);
    fixture.conflict(patch).setTo(0.75f);
    fixture.spread(patch).setTo(0.006f);

    const auto result = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth,
        fixture.guide,
        fixture.effective,
        fixture.conflict,
        fixture.spread);

    ASSERT_TRUE(result.validInputs);
    EXPECT_EQ(result.rejectedLayerPixelCount, 0);
    EXPECT_EQ(classCount(
                  result.classMap(patch),
                  xjw::mvs::DepthLayerReliabilityClass::AmbiguousLowTexture),
              patch.area());
}

TEST(DepthLayerReliabilityTest, WeakButCoplanarPatchRemainsAmbiguous)
{
    ReliabilityFixture fixture;
    const cv::Rect patch(30, 34, 34, 26);
    fixture.effective(patch).setTo(1.0f);
    fixture.conflict(patch).setTo(0.70f);
    fixture.spread(patch).setTo(0.005f);

    const auto result = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth,
        fixture.guide,
        fixture.effective,
        fixture.conflict,
        fixture.spread);

    ASSERT_TRUE(result.validInputs);
    EXPECT_EQ(result.rejectedLayerPixelCount, 0);
    EXPECT_EQ(classCount(
                  result.classMap(patch),
                  xjw::mvs::DepthLayerReliabilityClass::AmbiguousLowTexture),
              patch.area());
}

TEST(DepthLayerReliabilityTest, MixedBoundaryProtectsRealDepthStep)
{
    ReliabilityFixture fixture;
    fixture.depth.colRange(48, 96).setTo(3.0f);
    const cv::Rect weak_band(40, 30, 16, 36);
    fixture.effective(weak_band).setTo(1.0f);
    fixture.conflict(weak_band).setTo(0.80f);
    fixture.spread(weak_band).setTo(0.008f);

    const auto result = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth,
        fixture.guide,
        fixture.effective,
        fixture.conflict,
        fixture.spread);

    ASSERT_TRUE(result.validInputs);
    EXPECT_EQ(result.rejectedLayerPixelCount, 0);
    EXPECT_EQ(classCount(
                  result.classMap(weak_band),
                  xjw::mvs::DepthLayerReliabilityClass::AmbiguousLowTexture),
              weak_band.area());
}

TEST(DepthLayerReliabilityTest, ProducesDeterministicClassAndDiagnostics)
{
    ReliabilityFixture fixture;
    const cv::Rect island(35, 35, 24, 24);
    fixture.depth(island).setTo(2.05f);
    fixture.effective(island).setTo(1.0f);
    fixture.conflict(island).setTo(0.72f);
    fixture.spread(island).setTo(0.005f);

    const auto first = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth, fixture.guide, fixture.effective,
        fixture.conflict, fixture.spread);
    const auto second = xjw::mvs::analyzeDepthLayerReliability(
        fixture.depth, fixture.guide, fixture.effective,
        fixture.conflict, fixture.spread);

    ASSERT_TRUE(first.validInputs);
    ASSERT_TRUE(second.validInputs);
    EXPECT_EQ(cv::countNonZero(first.classMap != second.classMap), 0);
    EXPECT_EQ(
        xjw::mvs::depthLayerReliabilityDiagnosticsToJson(first),
        xjw::mvs::depthLayerReliabilityDiagnosticsToJson(second));
}

TEST(DepthLayerReliabilityTest, ManyComponentsStayBoundedOnProductionGrid)
{
    constexpr int width = 1552;
    constexpr int height = 1033;
    cv::Mat depth(height, width, CV_32FC1, cv::Scalar(2.0f));
    cv::Mat guide(height, width, CV_8UC1, cv::Scalar(128));
    cv::Mat effective(height, width, CV_32FC1, cv::Scalar(3.0f));
    cv::Mat conflict(height, width, CV_32FC1, cv::Scalar(0.10f));
    cv::Mat spread(height, width, CV_32FC1, cv::Scalar(0.001f));
    int expected_rejected_pixels = 0;
    for (int row = 0; row < 8; ++row)
    {
        for (int column = 0; column < 12; ++column)
        {
            const cv::Rect island(40 + column * 120, 40 + row * 115, 8, 8);
            depth(island).setTo(2.06f);
            effective(island).setTo(1.0f);
            conflict(island).setTo(0.75f);
            spread(island).setTo(0.006f);
            expected_rejected_pixels += island.area();
        }
    }

    xjw::mvs::DepthLayerReliabilityOptions options;
    options.maximumReportedComponents = 128;
    const auto start = std::chrono::steady_clock::now();
    const auto result = xjw::mvs::analyzeDepthLayerReliability(
        depth, guide, effective, conflict, spread, options);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    ASSERT_TRUE(result.validInputs);
    EXPECT_EQ(result.rejectedLayerComponentCount, 96);
    EXPECT_EQ(result.rejectedLayerPixelCount, expected_rejected_pixels);
    EXPECT_LT(elapsed_ms, 5000.0);
}

} // namespace
