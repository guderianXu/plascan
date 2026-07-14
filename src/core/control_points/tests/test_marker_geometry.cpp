#include "geometry/MarkerGeometry.h"
#include "geometry/MarkerProjectionPredictor.h"

#include <gtest/gtest.h>

namespace
{

xjw::control_points::MarkerCamera makeCamera(const QString &id, double centerX)
{
    xjw::control_points::MarkerCamera camera;
    camera.imageId = id;
    camera.imagePath = id + QStringLiteral(".png");
    camera.intrinsics = cv::Matx33d(1000.0, 0.0, 320.0,
                                    0.0, 1000.0, 240.0,
                                    0.0, 0.0, 1.0);
    camera.rotation = cv::Matx33d::eye();
    camera.translation = cv::Vec3d(-centerX, 0.0, 0.0);
    camera.imageSize = QSize(640, 480);
    return camera;
}

xjw::control_points::MarkerProjection observation(
    const xjw::control_points::MarkerCamera &camera,
    const cv::Point3d &point)
{
    xjw::control_points::MarkerProjection projection;
    projection.imageId = camera.imageId;
    projection.imagePathSnapshot = camera.imagePath;
    projection.xy = camera.project(point);
    projection.state = xjw::control_points::ProjectionState::ManualPinned;
    projection.sigmaPx = 0.5;
    return projection;
}

} // namespace

TEST(MarkerGeometryTest, TriangulatesPinnedObservationsWithPositiveDepth)
{
    using namespace xjw::control_points;
    const MarkerCamera first = makeCamera(QStringLiteral("image-1"), 0.0);
    const MarkerCamera second = makeCamera(QStringLiteral("image-2"), 1.0);
    const cv::Point3d expected(0.1, -0.2, 5.0);

    Marker marker;
    marker.id = QStringLiteral("marker-1");
    marker.projections = {observation(first, expected), observation(second, expected)};
    const MarkerTriangulation result = triangulateMarker(marker, {first, second});

    ASSERT_TRUE(result.success) << result.error.toStdString();
    EXPECT_NEAR(result.point.x, expected.x, 1.0e-6);
    EXPECT_NEAR(result.point.y, expected.y, 1.0e-6);
    EXPECT_NEAR(result.point.z, expected.z, 1.0e-6);
    EXPECT_LT(result.rmsReprojectionPx, 1.0e-6);
    EXPECT_GT(result.minimumIntersectionAngleDegrees, 1.0);
}

TEST(MarkerGeometryTest, EpipolarBandContainsTrueCorrespondence)
{
    using namespace xjw::control_points;
    const MarkerCamera first = makeCamera(QStringLiteral("image-1"), 0.0);
    const MarkerCamera second = makeCamera(QStringLiteral("image-2"), 1.0);
    const cv::Point3d point(0.1, -0.2, 5.0);
    const QPointF first_pixel = first.project(point);
    const QPointF second_pixel = second.project(point);

    const EpipolarBand band = epipolarSearchBand(first_pixel, first, second, 2.0);
    ASSERT_TRUE(band.valid);
    EXPECT_LT(band.distanceTo(second_pixel), 1.0e-6);
    EXPECT_TRUE(band.contains(second_pixel));
}

TEST(MarkerProjectionPredictorTest, PredictsOnlyPositiveDepthUnmaskedImages)
{
    using namespace xjw::control_points;
    MarkerCamera first = makeCamera(QStringLiteral("image-1"), 0.0);
    MarkerCamera second = makeCamera(QStringLiteral("image-2"), 1.0);
    MarkerCamera accepted = makeCamera(QStringLiteral("image-3"), -0.5);
    MarkerCamera masked = makeCamera(QStringLiteral("image-4"), -1.0);
    masked.acceptsPixel = [](const QPointF &) { return false; };
    MarkerCamera behind = makeCamera(QStringLiteral("image-5"), 0.0);
    behind.rotation = cv::Matx33d(1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, -1.0);

    const cv::Point3d point(0.1, -0.2, 5.0);
    Marker marker;
    marker.id = QStringLiteral("marker-1");
    marker.projections = {observation(first, point), observation(second, point)};

    const MarkerPredictionResult result = MarkerProjectionPredictor::predict(
        marker, {first, second, accepted, masked, behind});
    ASSERT_TRUE(result.triangulation.success);
    ASSERT_EQ(result.predictions.size(), 1);
    EXPECT_EQ(result.predictions.front().imageId, accepted.imageId);
    EXPECT_EQ(result.predictions.front().state, ProjectionState::Predicted);
}
