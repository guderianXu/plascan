#include <gtest/gtest.h>

#include "geometry/SimilarityGaugeNormalizer.h"

#include <array>
#include <cmath>
#include <vector>

namespace
{

    placamera::FramePinholeNumericState makeCamera(double x, double y, double z)
    {
        const placamera::FrameId frame("gauge-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("gauge-definition"),
                                                      {1000.0, 1000.0, 512.0, 384.0, 1.0, 1, 1},
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        return placamera::FramePinholeNumericState::fromModel(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("gauge-instance"),
            placamera::ImageId("gauge-image"),
            definition,
            {1024, 768},
            placamera::Pose::create(frame, {x, y, z}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
}

double cameraDistance(const placamera::FramePinholeNumericState& left, const placamera::FramePinholeNumericState& right)
{
    const auto a = left.pose().center;
    const auto b = right.pose().center;
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

TEST(SimilarityGaugeNormalizerTest, RestoresAnchorAndBaselineWithoutChangingShape)
{
    const std::vector<placamera::FramePinholeNumericState> reference{
        makeCamera(10.0, 20.0, 30.0),
        makeCamera(12.0, 20.0, 30.0),
        makeCamera(10.0, 24.0, 30.0),
    };
    std::vector<placamera::FramePinholeNumericState> refined{
        makeCamera(-1.0, -2.0, -3.0),
        makeCamera(3.0, -2.0, -3.0),
        makeCamera(-1.0, 6.0, -3.0),
    };
    std::vector<plabundle::RefinedPoint> points(1);
    points[0].point = {{-1.0, -2.0, 5.0}};
    points[0].valid = true;

    const xjw::SimilarityGaugeNormalizationResult result =
        xjw::normalizeSimilarityGauge(reference, 0, 1, &refined, &points);

    ASSERT_TRUE(result.applied);
    EXPECT_NEAR(result.scale, 0.5, 1.0e-12);
    EXPECT_EQ(refined[0].pose().center, reference[0].pose().center);
    EXPECT_NEAR(cameraDistance(refined[0], refined[1]), 2.0, 1.0e-12);
    EXPECT_NEAR(cameraDistance(refined[0], refined[2]), 4.0, 1.0e-12);
    EXPECT_NEAR(points[0].point[0], 10.0, 1.0e-12);
    EXPECT_NEAR(points[0].point[1], 20.0, 1.0e-12);
    EXPECT_NEAR(points[0].point[2], 34.0, 1.0e-12);
}

TEST(SimilarityGaugeNormalizerTest, DegenerateRefinedBaselineDoesNotModifyOutput)
{
    const std::vector<placamera::FramePinholeNumericState> reference{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    std::vector<placamera::FramePinholeNumericState> refined{
        makeCamera(5.0, 6.0, 7.0),
        makeCamera(5.0, 6.0, 7.0),
    };
    const std::vector<placamera::FramePinholeNumericState> before = refined;
    std::vector<plabundle::RefinedPoint> points(1);
    points[0].point = {{1.0, 2.0, 3.0}};

    const xjw::SimilarityGaugeNormalizationResult result =
        xjw::normalizeSimilarityGauge(reference, 0, 1, &refined, &points);

    EXPECT_FALSE(result.applied);
    EXPECT_EQ(result.reason, "degenerate_gauge_baseline");
    EXPECT_EQ(refined[0].pose().center, before[0].pose().center);
    EXPECT_EQ(refined[1].pose().center, before[1].pose().center);
    EXPECT_EQ(points[0].point, (std::array<double, 3>{{1.0, 2.0, 3.0}}));
}
