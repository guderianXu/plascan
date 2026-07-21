#include <gtest/gtest.h>

#include "project/BaInputBuilder.h"
#include "Camera.h"
#include "model/MarkerSet.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>

namespace
{

xjw::Camera makeCamera(double cx)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{cx, 0.0, 0.0}});
    return camera;
}

QJsonObject cameraToJson(const xjw::Camera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    QJsonArray centerArray;
    centerArray.append(center[0]);
    centerArray.append(center[1]);
    centerArray.append(center[2]);

    QJsonArray rotationArray;
    for (double value : rotation)
    {
        rotationArray.append(value);
    }

    QJsonObject object;
    object[QStringLiteral("intrinsics_unit")] = QStringLiteral("px");
    object[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    object[QStringLiteral("pitch")] = camera.pixelPitch();
    object[QStringLiteral("fu")] = intrinsics.focalX;
    object[QStringLiteral("fv")] = intrinsics.focalY;
    object[QStringLiteral("cu")] = intrinsics.principalX;
    object[QStringLiteral("cv")] = intrinsics.principalY;
    object[QStringLiteral("C")] = centerArray;
    object[QStringLiteral("R")] = rotationArray;
    return object;
}

QJsonObject imageEntry(const QString &path, const xjw::Camera &camera)
{
    QJsonObject object;
    object[QStringLiteral("path")] = path;
    object[QStringLiteral("camera")] = cameraToJson(camera);
    return object;
}

QPointF project(const xjw::Camera &camera, const std::array<double, 3> &point)
{
    const double xyz[3] = {point[0], point[1], point[2]};
    double uv[2] = {0.0, 0.0};
    EXPECT_TRUE(camera.projectWorldPoint(xyz, uv));
    return QPointF(uv[0], uv[1]);
}

std::array<double, 3> controlReference(const std::array<double, 3> &local)
{
    return {{100.0 + 3.0 * local[0],
             200.0 + 3.0 * local[1],
             50.0 + 3.0 * local[2]}};
}

} // namespace

TEST(BaInputBuilderSurveyControl, BuildsControlPointTracksFromSurveyObservations)
{
    const QString image0 = QStringLiteral("E:/images/img_001.jpg");
    const QString image1 = QStringLiteral("E:/images/img_002.jpg");

    QJsonObject controlPoint;
    controlPoint[QStringLiteral("id")] = QStringLiteral("GCP001");
    controlPoint[QStringLiteral("enabled")] = true;
    controlPoint[QStringLiteral("x")] = 0.0;
    controlPoint[QStringLiteral("y")] = 0.0;
    controlPoint[QStringLiteral("z")] = 10.0;
    controlPoint[QStringLiteral("sigma_m")] = 0.05;
    controlPoint[QStringLiteral("observations")] = QJsonArray{
        QJsonObject{{QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_path"), image1},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
    };

    QJsonObject surveyControl;
    surveyControl[QStringLiteral("control_points")] = QJsonArray{controlPoint};

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        imageEntry(image0, makeCamera(0.0)),
        imageEntry(image1, makeCamera(1.0)),
    };
    meta[QStringLiteral("survey_control")] = surveyControl;

    xjw::core::project::BaInputBuildResult result;
    const xjw::core::project::BaInputBuildStatus status =
        xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result);

    EXPECT_EQ(status, xjw::core::project::BaInputBuildStatus::Ok);
    EXPECT_EQ(result.surveyControlTrackCount, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().observations.size(), 2u);
    ASSERT_EQ(result.tracks.front().controlPointConstraints.size(), 1u);
    EXPECT_NEAR(result.tracks.front().initialPoint[2], 10.0, 1e-12);
    EXPECT_NEAR(result.tracks.front().controlPointConstraints.front().point[2], 10.0, 1e-12);
    EXPECT_NEAR(result.tracks.front().controlPointConstraints.front().sigmaMeters, 0.05, 1e-12);
}

TEST(BaInputBuilderSurveyControl, BuildsScaleBarConstraintsBetweenSurveyControlTracks)
{
    const QString image0 = QStringLiteral("E:/images/img_001.jpg");
    const QString image1 = QStringLiteral("E:/images/img_002.jpg");

    QJsonObject controlPoint0;
    controlPoint0[QStringLiteral("id")] = QStringLiteral("GCP001");
    controlPoint0[QStringLiteral("enabled")] = true;
    controlPoint0[QStringLiteral("x")] = 0.0;
    controlPoint0[QStringLiteral("y")] = 0.0;
    controlPoint0[QStringLiteral("z")] = 10.0;
    controlPoint0[QStringLiteral("observations")] = QJsonArray{
        QJsonObject{{QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_path"), image1},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
    };

    QJsonObject controlPoint1 = controlPoint0;
    controlPoint1[QStringLiteral("id")] = QStringLiteral("GCP002");
    controlPoint1[QStringLiteral("x")] = 10.0;
    controlPoint1[QStringLiteral("u")] = 0.0;
    controlPoint1[QStringLiteral("observations")] = QJsonArray{
        QJsonObject{{QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 1512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_path"), image1},
                    {QStringLiteral("u"), 1512.0},
                    {QStringLiteral("v"), 384.0}},
    };

    QJsonObject scaleBar;
    scaleBar[QStringLiteral("id")] = QStringLiteral("SB001");
    scaleBar[QStringLiteral("enabled")] = true;
    scaleBar[QStringLiteral("from_id")] = QStringLiteral("GCP001");
    scaleBar[QStringLiteral("to_id")] = QStringLiteral("GCP002");
    scaleBar[QStringLiteral("measured_m")] = 9.5;
    scaleBar[QStringLiteral("sigma_m")] = 0.02;

    QJsonObject surveyControl;
    surveyControl[QStringLiteral("control_points")] = QJsonArray{controlPoint0, controlPoint1};
    surveyControl[QStringLiteral("scale_bars")] = QJsonArray{scaleBar};

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        imageEntry(image0, makeCamera(0.0)),
        imageEntry(image1, makeCamera(1.0)),
    };
    meta[QStringLiteral("survey_control")] = surveyControl;

    xjw::core::project::BaInputBuildResult result;
    const xjw::core::project::BaInputBuildStatus status =
        xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result);

    EXPECT_EQ(status, xjw::core::project::BaInputBuildStatus::Ok);
    EXPECT_EQ(result.surveyControlTrackCount, 2);
    EXPECT_EQ(result.surveyScaleBarConstraintCount, 1);
    ASSERT_EQ(result.scaleBarConstraints.size(), 1u);
    EXPECT_EQ(result.scaleBarConstraints.front().trackIndexA, 0);
    EXPECT_EQ(result.scaleBarConstraints.front().trackIndexB, 1);
    EXPECT_NEAR(result.scaleBarConstraints.front().measuredDistanceMeters, 9.5, 1e-12);
    EXPECT_NEAR(result.scaleBarConstraints.front().sigmaMeters, 0.02, 1e-12);
}

TEST(BaInputBuilderMarkerSet, AppliesAbsoluteOrientationAndExcludesChecksFromConstraints)
{
    const QString image0 = QStringLiteral("E:/images/img_001.jpg");
    const QString image1 = QStringLiteral("E:/images/img_002.jpg");
    const QString image2 = QStringLiteral("E:/images/img_003.jpg");
    const std::vector<xjw::Camera> cameras = {
        makeCamera(-2.0), makeCamera(0.0), makeCamera(2.0)};
    const QStringList image_paths{image0, image1, image2};
    const QStringList image_ids{QStringLiteral("uuid-0"),
                                QStringLiteral("uuid-1"),
                                QStringLiteral("uuid-2")};

    xjw::control_points::MarkerSet marker_set;
    const std::vector<std::array<double, 3>> points = {
        {{-1.0, -0.5, 20.0}},
        {{1.0, -0.5, 20.0}},
        {{-0.5, 1.0, 20.5}},
        {{0.5, 0.5, 22.0}},
        {{0.0, 0.0, 21.0}},
    };
    QVector<xjw::control_points::MarkerId> marker_ids;
    for (int point_index = 0; point_index < static_cast<int>(points.size()); ++point_index)
    {
        const auto role = point_index < 4
            ? xjw::control_points::MarkerRole::ControlPoint
            : xjw::control_points::MarkerRole::CheckPoint;
        const auto marker_id = marker_set.addMarker(
            QStringLiteral("M%1").arg(point_index + 1), role);
        marker_ids.push_back(marker_id);

        xjw::control_points::ReferenceCoordinate reference;
        const auto reference_point = controlReference(points[static_cast<std::size_t>(point_index)]);
        reference.x = reference_point[0];
        reference.y = reference_point[1];
        reference.z = reference_point[2];
        reference.sigmaX = 0.01;
        reference.sigmaY = 0.01;
        reference.sigmaZ = 0.01;
        reference.sourceCrs = QStringLiteral("EPSG:4978");
        reference.verticalDatum = QStringLiteral("ellipsoidal");
        reference.verticalUnit = QStringLiteral("m");
        marker_set.setReferenceCoordinate(marker_id, reference);
        ASSERT_TRUE(marker_set.marker(marker_id).referenceCoordinate->referenceUsable)
            << qPrintable(marker_set.marker(marker_id).referenceCoordinate->referenceError);

        for (int camera_index = 0; camera_index < static_cast<int>(cameras.size()); ++camera_index)
        {
            xjw::control_points::MarkerProjection projection;
            projection.imageId = image_ids[camera_index];
            projection.imagePathSnapshot = image_paths[camera_index];
            projection.xy = project(cameras[static_cast<std::size_t>(camera_index)],
                                    points[static_cast<std::size_t>(point_index)]);
            projection.state = xjw::control_points::ProjectionState::ManualPinned;
            marker_set.upsertProjection(marker_id, projection);
        }
    }
    marker_set.addScaleBar(QStringLiteral("control-scale"),
                           marker_ids[0], marker_ids[1], 6.0, 0.01,
                           xjw::control_points::ScaleBarRole::Control);
    marker_set.addScaleBar(QStringLiteral("check-scale"),
                           marker_ids[2], marker_ids[3], 1.0, 0.01,
                           xjw::control_points::ScaleBarRole::Check);

    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        imageEntry(image0, cameras[0]),
        imageEntry(image1, cameras[1]),
        imageEntry(image2, cameras[2]),
    };
    xjw::core::project::MarkerBaInput marker_input;
    marker_input.markerSet = &marker_set;
    for (int index = 0; index < image_ids.size(); ++index)
    {
        marker_input.imagePathById.insert(image_ids[index], image_paths[index]);
    }

    xjw::core::project::BaInputBuildResult result;
    const auto status = xjw::core::project::buildBaInputFromMeta(
        meta, image_paths, 1, &result, &marker_input);

    ASSERT_EQ(status, xjw::core::project::BaInputBuildStatus::Ok);
    EXPECT_TRUE(result.markerControlNetwork.ok)
        << result.markerControlNetwork.error;
    EXPECT_EQ(result.markerControlTrackCount, 4);
    EXPECT_EQ(result.markerCheckTrackCount, 1);
    EXPECT_EQ(result.markerControlPointConstraintCount, 4);
    EXPECT_EQ(result.markerControlScaleBarConstraintCount, 1);
    EXPECT_EQ(result.markerCheckScaleBarCount, 1);
    ASSERT_EQ(result.tracks.size(), 5u);
    EXPECT_TRUE(result.tracks.back().controlPointConstraints.empty());
    const auto center = result.cameras.front().cameraCenter();
    const auto expected_center = controlReference({{-2.0, 0.0, 0.0}});
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_NEAR(center[axis], expected_center[axis], 1.0e-5);
    }
}
