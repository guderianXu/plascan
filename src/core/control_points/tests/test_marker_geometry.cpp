#include "geometry/MarkerGeometry.h"
#include "geometry/MarkerProjectionPredictor.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace
{

    xjw::control_points::MarkerImageView
    makeView(const QString& id,
             double center_x,
             placamera::BrownConradyDistortion distortion = {},
             placamera::RotationMatrix rotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
             const std::string& frame_name = "local",
             bool depth_flipped = false)
    {
        const std::string identifier = id.toStdString();
        const placamera::FrameId frame(frame_name);
        const placamera::FrameIntrinsics intrinsics{1000.0, 1000.0, 320.0, 240.0, 0.01, 1, 1};
        auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("definition-" + identifier),
                                                      intrinsics,
                                                      distortion,
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame,
                                                      depth_flipped);
        const auto pose = placamera::Pose::create(frame, {center_x, 0.0, 0.0}, rotation);
        auto camera = placamera::FramePinholeModel::create(placamera::CameraInstanceId("instance-" + identifier),
                                                           placamera::ImageId(identifier),
                                                           std::move(definition),
                                                           placamera::ImageSize{640, 480},
                                                           pose);
        xjw::control_points::MarkerImageView view;
        view.imagePath = id + QStringLiteral(".png");
        view.camera = std::make_shared<const placamera::FramePinholeModel>(std::move(camera));
        return view;
    }

    xjw::control_points::MarkerProjection observation(const xjw::control_points::MarkerImageView& view,
                                                      const cv::Point3d& point)
    {
        xjw::control_points::MarkerProjection projection;
        projection.imageId = QString::fromStdString(std::string(view.camera->imageId().value()));
        projection.imagePathSnapshot = view.imagePath;
        const auto projected = view.camera->groundToImage(
            placamera::GroundCoordinate{view.camera->groundFrame(), {point.x, point.y, point.z}});
        if (!projected)
        {
            ADD_FAILURE() << projected.message();
            return projection;
        }
        projection.xy = QPointF(projected.value().image.sample, projected.value().image.line);
        projection.state = xjw::control_points::ProjectionState::ManualPinned;
        projection.sigmaPx = 0.5;
        return projection;
    }

} // namespace

TEST(MarkerGeometryTest, TriangulatesPinnedObservationsWithPositiveDepth)
{
    using namespace xjw::control_points;
    const MarkerImageView first = makeView(QStringLiteral("image-1"), 0.0);
    const MarkerImageView second = makeView(QStringLiteral("image-2"), 1.0);
    const cv::Point3d expected(0.1, -0.2, 5.0);

    Marker marker;
    marker.id = QStringLiteral("marker-1");
    marker.projections = {observation(first, expected), observation(second, expected)};
    const MarkerTriangulation result = triangulateMarker(marker, {first, second});

    ASSERT_TRUE(result.success) << result.error.toStdString();
    ASSERT_TRUE(result.groundFrame.has_value());
    EXPECT_EQ(result.groundFrame->value(), "local");
    EXPECT_NEAR(result.point.x, expected.x, 1.0e-6);
    EXPECT_NEAR(result.point.y, expected.y, 1.0e-6);
    EXPECT_NEAR(result.point.z, expected.z, 1.0e-6);
    EXPECT_LT(result.rmsReprojectionPx, 1.0e-6);
    EXPECT_GT(result.minimumIntersectionAngleDegrees, 1.0);
}

TEST(MarkerGeometryTest, TriangulatesDistortedPlaCameraRays)
{
    using namespace xjw::control_points;
    const placamera::BrownConradyDistortion distortion{0.12, -0.02, 0.001, 0.0005, -0.0004};
    const MarkerImageView first = makeView(QStringLiteral("distorted-1"), 0.0, distortion);
    const MarkerImageView second = makeView(QStringLiteral("distorted-2"), 1.0, distortion);
    const cv::Point3d expected(0.4, -0.3, 4.0);

    Marker marker;
    marker.id = QStringLiteral("distorted-marker");
    marker.projections = {observation(first, expected), observation(second, expected)};
    const MarkerTriangulation result = triangulateMarker(marker, {first, second});

    ASSERT_TRUE(result.success) << result.error.toStdString();
    EXPECT_NEAR(result.point.x, expected.x, 1.0e-5);
    EXPECT_NEAR(result.point.y, expected.y, 1.0e-5);
    EXPECT_NEAR(result.point.z, expected.z, 1.0e-5);
    EXPECT_LT(result.rmsReprojectionPx, 1.0e-5);
}

TEST(MarkerGeometryTest, HonorsPlaCameraFlippedDepthAxisWithoutPoseConversion)
{
    using namespace xjw::control_points;
    const MarkerImageView first =
        makeView(QStringLiteral("flipped-1"), 0.0, {}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, "local", true);
    const MarkerImageView second =
        makeView(QStringLiteral("flipped-2"), 1.0, {}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, "local", true);
    const cv::Point3d expected(0.4, -0.2, -5.0);

    Marker marker;
    marker.projections = {observation(first, expected), observation(second, expected)};
    const MarkerTriangulation result = triangulateMarker(marker, {first, second});

    ASSERT_TRUE(result.success) << result.error.toStdString();
    EXPECT_NEAR(result.point.x, expected.x, 1.0e-6);
    EXPECT_NEAR(result.point.y, expected.y, 1.0e-6);
    EXPECT_NEAR(result.point.z, expected.z, 1.0e-6);
}

TEST(MarkerGeometryTest, RejectsObservationsInDifferentWorldFrames)
{
    using namespace xjw::control_points;
    const MarkerImageView first = makeView(QStringLiteral("frame-1"), 0.0);
    const MarkerImageView second =
        makeView(QStringLiteral("frame-2"), 1.0, {}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, "other");
    const cv::Point3d point(0.1, -0.2, 5.0);
    Marker marker;
    marker.projections = {observation(first, point), observation(second, point)};

    const MarkerTriangulation result = triangulateMarker(marker, {first, second});
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.error.contains(QStringLiteral("世界坐标系")));
}

TEST(MarkerGeometryTest, EpipolarBandContainsTrueCorrespondence)
{
    using namespace xjw::control_points;
    const MarkerImageView first = makeView(QStringLiteral("image-1"), 0.0);
    const MarkerImageView second = makeView(QStringLiteral("image-2"), 1.0);
    const cv::Point3d point(0.1, -0.2, 5.0);
    const QPointF first_pixel = observation(first, point).xy;
    const QPointF second_pixel = observation(second, point).xy;

    const EpipolarBand band = epipolarSearchBand(first_pixel, first, second, 2.0);
    ASSERT_TRUE(band.valid);
    EXPECT_LT(band.distanceTo(second_pixel), 1.0e-6);
    EXPECT_TRUE(band.contains(second_pixel));
}

TEST(MarkerProjectionPredictorTest, PredictsOnlyPositiveDepthUnmaskedImages)
{
    using namespace xjw::control_points;
    const MarkerImageView first = makeView(QStringLiteral("image-1"), 0.0);
    const MarkerImageView second = makeView(QStringLiteral("image-2"), 1.0);
    const MarkerImageView accepted = makeView(QStringLiteral("image-3"), -0.5);
    MarkerImageView masked = makeView(QStringLiteral("image-4"), -1.0);
    masked.acceptsPixel = [](const QPointF&) { return false; };
    const MarkerImageView behind =
        makeView(QStringLiteral("image-5"), 0.0, {}, {-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0});

    const cv::Point3d point(0.1, -0.2, 5.0);
    Marker marker;
    marker.id = QStringLiteral("marker-1");
    marker.projections = {observation(first, point), observation(second, point)};

    const MarkerPredictionResult result =
        MarkerProjectionPredictor::predict(marker, {first, second, accepted, masked, behind});
    ASSERT_TRUE(result.triangulation.success) << result.triangulation.error.toStdString();
    ASSERT_EQ(result.predictions.size(), 1);
    EXPECT_EQ(result.predictions.front().imageId, QStringLiteral("image-3"));
    EXPECT_EQ(result.predictions.front().state, ProjectionState::Predicted);
}
