#include "geometry/TriangulationQuality.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

namespace
{
    placamera::FramePinholeModel makeQualityCamera(int index, double focal, double centerX)
    {
        placamera::FrameIntrinsics intrinsics;
        intrinsics.focalX = focal;
        intrinsics.focalY = focal;
        const placamera::FrameId frame("quality-world");
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId("quality-definition-" + std::to_string(index)),
            intrinsics,
            {},
            placamera::PixelConvention::PixelCenter,
            frame);
        return placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("quality-instance-" + std::to_string(index)),
            placamera::ImageId("quality-image-" + std::to_string(index)),
            definition,
            {4000, 4000},
            placamera::Pose::create(frame, {centerX, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    }
} // namespace

TEST(TriangulationQualityTest, ReconstructionUncertaintyRespondsToBaseline)
{
    const auto left = makeQualityCamera(0, 800.0, -1.0);
    const auto wideRight = makeQualityCamera(1, 800.0, 1.0);
    const auto narrowRight = makeQualityCamera(2, 800.0, -0.8);

    const std::array<double, 3> point{0.0, 0.0, 10.0};
    const double wide = xjw::reconstructionUncertainty({{&left, 2.0}, {&wideRight, 2.0}}, point);
    const double narrow = xjw::reconstructionUncertainty({{&left, 2.0}, {&narrowRight, 2.0}}, point);

    EXPECT_TRUE(std::isfinite(wide));
    EXPECT_TRUE(std::isfinite(narrow));
    EXPECT_GE(wide, 1.0);
    EXPECT_GT(narrow, wide);
}

TEST(TriangulationQualityTest, ProjectionAccuracyAveragesEveryObservationScale)
{
    const auto camera = makeQualityCamera(0, 800.0, 0.0);
    const std::vector<xjw::TiePointQualityObservation> observations{{&camera, 1.0}, {&camera, 2.0}, {&camera, 3.0}};
    EXPECT_DOUBLE_EQ(xjw::projectionAccuracy(observations), 2.0);

    std::vector<xjw::TiePointQualityObservation> incomplete = observations;
    incomplete.back().measurementScale = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(xjw::projectionAccuracy(incomplete)));
}

TEST(TriangulationQualityTest, CleanTiePointQualityMatchesReferenceContract)
{
    std::vector<placamera::FramePinholeModel> cameras;
    cameras.reserve(3);
    const std::array<double, 3> camera_x{-1.0, 0.0, 1.0};
    const std::array<double, 3> scales{1.0, 2.0, 3.0};
    const std::array<double, 3> point{0.0, 0.0, 5.0};
    std::vector<xjw::TiePointQualityObservation> observations;
    for (std::size_t index = 0; index < camera_x.size(); ++index)
    {
        cameras.push_back(makeQualityCamera(static_cast<int>(index), 1000.0, camera_x[index]));
        const auto projection = cameras.back().groundToImage({cameras.back().groundFrame(), point});
        ASSERT_TRUE(projection);
        observations.push_back(
            {&cameras.back(),
             scales[index],
             {projection.value().image.sample + (index == 0 ? 2.0 : 0.0), projection.value().image.line}});
    }

    const xjw::CleanTiePointQuality quality = xjw::evaluateCleanTiePointQuality(observations, point);

    EXPECT_DOUBLE_EQ(quality.reprojectionError, 2.0);
    EXPECT_EQ(quality.imageCount, 3U);
    EXPECT_DOUBLE_EQ(quality.projectionAccuracy, 2.0);
    ASSERT_TRUE(quality.hasProjectionGeometry);
    EXPECT_NEAR(quality.reconstructionUncertainty, std::sqrt(37.5), 1.0e-9);

    observations.front().measurementScale = 0.0;
    const xjw::CleanTiePointQuality zeroScale = xjw::evaluateCleanTiePointQuality(observations, point);
    EXPECT_DOUBLE_EQ(zeroScale.reprojectionError, 2.0);
    EXPECT_DOUBLE_EQ(zeroScale.projectionAccuracy, 5.0 / 3.0);
    EXPECT_NEAR(zeroScale.reconstructionUncertainty, quality.reconstructionUncertainty, 1.0e-12);
}
