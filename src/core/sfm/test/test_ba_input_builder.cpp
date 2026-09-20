#include <gtest/gtest.h>

#include "TriangulationService.h"
#include "ProjectCameraIO.h"
#include "coordinate_system/context/CoordinateContext.h"
#include "coordinate_system/gdal/GdalCoordinateTransform.h"
#include "project/BaInputBuilder.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/project/CameraProjectRecords.h"
#include "project/ProjectMetadata.h"
#include "model/MarkerSet.h"
#include "io/MarkerSetJson.h"

#include <QJsonArray>
#include <QMap>
#include <QJsonObject>
#include <QStringList>

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace
{

    xjw::camera_models::frame_pinhole::FramePinholeNumericState makeCamera(double cx)
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
        camera.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}, {{cx, 0.0, 0.0}});
        return camera;
    }

    xjw::coordinate_system::CoordinateContext makeEarthContext()
    {
        using namespace xjw::coordinate_system;
        const auto normalize =
            [](const char* id, const char* frameId, const char* definition, VerticalReference verticalReference)
        {
            GdalSpatialReferenceResult result = normalizeGdalSpatialReference(
                SpatialReferenceId(id), CoordinateFrameId(frameId), definition, verticalReference);
            if (!result.ok())
            {
                throw std::runtime_error(result.error);
            }
            return std::move(*result.reference);
        };
        const auto frame = [](const char* id, CoordinateFrameKind kind, AngleUnit angles)
        {
            return CoordinateFrame::create(
                CoordinateFrameId(id), kind, LinearUnit::Metre, angles, std::nullopt, RigidTransform::identity());
        };
        return CoordinateContext::create(
            CoordinateContextId("coordctx-ba-earth"),
            1,
            {normalize("crs-epsg-4979", "frame-wgs84-geodetic", "EPSG:4979", VerticalReference::Ellipsoidal),
             normalize("crs-epsg-4978", "frame-wgs84-ecef", "EPSG:4978", VerticalReference::NotApplicable)},
            {frame("frame-wgs84-geodetic", CoordinateFrameKind::Geodetic, AngleUnit::Degree),
             frame("frame-wgs84-ecef", CoordinateFrameKind::Ecef, AngleUnit::Radian)},
            SpatialReferenceId("crs-epsg-4978"),
            SolverFrameDefinition::create(
                CoordinateFrameId("frame-wgs84-ecef"), SolverScaleStatus::Metric, "ba-ecef-v1"));
    }

    QJsonObject cameraToJson(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera)
    {
        return xjw::common::project::serializeFramePinholeNumericState(camera);
    }

    QJsonObject makeProjectMeta(
        std::initializer_list<std::pair<QString, xjw::camera_models::frame_pinhole::FramePinholeNumericState>>
            cameraEntries)
    {
        QJsonObject meta;
        QJsonArray images;
        QMap<QString, QJsonObject> metadataByPath;
        int imageIndex = 0;
        for (const auto& [path, camera] : cameraEntries)
        {
            const QString imageId = QStringLiteral("uuid-%1").arg(imageIndex++);
            images.append(QJsonObject{{QStringLiteral("image_uuid"), imageId},
                                      {QStringLiteral("path"), path},
                                      {QStringLiteral("samples"), 1024},
                                      {QStringLiteral("lines"), 768}});
            QJsonObject metadata = cameraToJson(camera);
            metadata.insert(QStringLiteral("image_width"), 1024);
            metadata.insert(QStringLiteral("image_height"), 768);
            metadataByPath.insert(path, metadata);
        }
        meta.insert(QStringLiteral("images"), images);
        meta.insert(QStringLiteral("camera_definitions"), QJsonArray{});
        meta.insert(QStringLiteral("camera_instances"), QJsonArray{});
        const auto update = xjw::camera_project::CameraProjectRecords::upsertByImagePath(&meta, metadataByPath);
        EXPECT_TRUE(update.ok()) << update.errors.join(';').toStdString();
        return meta;
    }

    QJsonObject makeProjectMetaWithFrames(const QString& image0,
                                          const QString& image1,
                                          const QString& frame0,
                                          const QString& frame1)
    {
        QJsonObject meta = makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}});
        QMap<QString, QJsonObject> metadataByPath;
        QJsonObject camera0 = cameraToJson(makeCamera(0.0));
        camera0.insert(QStringLiteral("world_frame"), frame0);
        camera0.insert(QStringLiteral("image_width"), 1024);
        camera0.insert(QStringLiteral("image_height"), 768);
        metadataByPath.insert(image0, camera0);
        QJsonObject camera1 = cameraToJson(makeCamera(1.0));
        camera1.insert(QStringLiteral("world_frame"), frame1);
        camera1.insert(QStringLiteral("image_width"), 1024);
        camera1.insert(QStringLiteral("image_height"), 768);
        metadataByPath.insert(image1, camera1);
        const auto update = xjw::camera_project::CameraProjectRecords::upsertByImagePath(&meta, metadataByPath);
        EXPECT_TRUE(update.ok()) << update.errors.join(';').toStdString();
        return meta;
    }

    QJsonObject makeProjectMetaWithUnselectedFrame(const QString& image0,
                                                   const QString& image1,
                                                   const QString& image2,
                                                   const QString& frame0,
                                                   const QString& frame1,
                                                   const QString& frame2)
    {
        QJsonObject meta =
            makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}, {image2, makeCamera(2.0)}});
        QMap<QString, QJsonObject> metadataByPath;
        const auto addMetadata = [&metadataByPath](const QString& path, double centerX, const QString& frame)
        {
            QJsonObject camera = cameraToJson(makeCamera(centerX));
            camera.insert(QStringLiteral("world_frame"), frame);
            camera.insert(QStringLiteral("image_width"), 1024);
            camera.insert(QStringLiteral("image_height"), 768);
            metadataByPath.insert(path, camera);
        };
        addMetadata(image0, 0.0, frame0);
        addMetadata(image1, 1.0, frame1);
        addMetadata(image2, 2.0, frame2);
        const auto update = xjw::camera_project::CameraProjectRecords::upsertByImagePath(&meta, metadataByPath);
        EXPECT_TRUE(update.ok()) << update.errors.join(';').toStdString();
        return meta;
    }

    QJsonObject rpcMetadata()
    {
        QJsonArray numerator;
        QJsonArray denominator;
        for (int index = 0; index < 20; ++index)
        {
            numerator.append(index == 0 ? 1.0 : 0.0);
            denominator.append(index == 0 ? 1.0 : 0.0);
        }
        return QJsonObject{{QStringLiteral("model"), QStringLiteral("rpc00b")},
                           {QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                           {QStringLiteral("ground_crs"), QStringLiteral("EPSG:4979")},
                           {QStringLiteral("world_frame"), QStringLiteral("EPSG:4978")},
                           {QStringLiteral("height_datum"), QStringLiteral("WGS84_ellipsoidal")},
                           {QStringLiteral("pixel_convention"), QStringLiteral("opencv_zero_based_center")},
                           {QStringLiteral("line_off"), 0.0},
                           {QStringLiteral("samp_off"), 0.0},
                           {QStringLiteral("lat_off"), 0.0},
                           {QStringLiteral("long_off"), 0.0},
                           {QStringLiteral("height_off"), 0.0},
                           {QStringLiteral("line_scale"), 1.0},
                           {QStringLiteral("samp_scale"), 1.0},
                           {QStringLiteral("lat_scale"), 1.0},
                           {QStringLiteral("long_scale"), 1.0},
                           {QStringLiteral("height_scale"), 1.0},
                           {QStringLiteral("image_samples"), 1024},
                           {QStringLiteral("image_lines"), 768},
                           {QStringLiteral("line_num_coeff"), numerator},
                           {QStringLiteral("line_den_coeff"), denominator},
                           {QStringLiteral("samp_num_coeff"), numerator},
                           {QStringLiteral("samp_den_coeff"), denominator}};
    }

    QPointF project(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                    const std::array<double, 3>& point)
    {
        const double xyz[3] = {point[0], point[1], point[2]};
        double uv[2] = {0.0, 0.0};
        EXPECT_TRUE(camera.projectWorldPoint(xyz, uv));
        return QPointF(uv[0], uv[1]);
    }

    std::array<double, 3> controlReference(const std::array<double, 3>& local)
    {
        return {{100.0 + 3.0 * local[0], 200.0 + 3.0 * local[1], 50.0 + 3.0 * local[2]}};
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
        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("uuid-0")},
                    {QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("uuid-1")},
                    {QStringLiteral("image_path"), image1},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
    };

    QJsonObject surveyControl;
    surveyControl[QStringLiteral("control_points")] = QJsonArray{controlPoint};

    QJsonObject meta = makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}});
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
        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("uuid-0")},
                    {QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("uuid-1")},
                    {QStringLiteral("image_path"), image1},
                    {QStringLiteral("u"), 512.0},
                    {QStringLiteral("v"), 384.0}},
    };

    QJsonObject controlPoint1 = controlPoint0;
    controlPoint1[QStringLiteral("id")] = QStringLiteral("GCP002");
    controlPoint1[QStringLiteral("x")] = 10.0;
    controlPoint1[QStringLiteral("u")] = 0.0;
    controlPoint1[QStringLiteral("observations")] = QJsonArray{
        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("uuid-0")},
                    {QStringLiteral("image_path"), image0},
                    {QStringLiteral("u"), 1512.0},
                    {QStringLiteral("v"), 384.0}},
        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("uuid-1")},
                    {QStringLiteral("image_path"), image1},
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

    QJsonObject meta = makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}});
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
    const std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> cameras = {
        makeCamera(-2.0), makeCamera(0.0), makeCamera(2.0)};
    const QStringList image_paths{image0, image1, image2};
    const QStringList image_ids{QStringLiteral("uuid-0"), QStringLiteral("uuid-1"), QStringLiteral("uuid-2")};

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
        const auto role = point_index < 4 ? xjw::control_points::MarkerRole::ControlPoint
                                          : xjw::control_points::MarkerRole::CheckPoint;
        const auto marker_id = marker_set.addMarker(QStringLiteral("M%1").arg(point_index + 1), role);
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
            projection.xy =
                project(cameras[static_cast<std::size_t>(camera_index)], points[static_cast<std::size_t>(point_index)]);
            projection.state = xjw::control_points::ProjectionState::ManualPinned;
            marker_set.upsertProjection(marker_id, projection);
        }
    }
    marker_set.addScaleBar(QStringLiteral("control-scale"),
                           marker_ids[0],
                           marker_ids[1],
                           6.0,
                           0.01,
                           xjw::control_points::ScaleBarRole::Control);
    marker_set.addScaleBar(QStringLiteral("check-scale"),
                           marker_ids[2],
                           marker_ids[3],
                           1.0,
                           0.01,
                           xjw::control_points::ScaleBarRole::Check);

    QJsonObject meta = makeProjectMeta({{image0, cameras[0]}, {image1, cameras[1]}, {image2, cameras[2]}});
    xjw::core::project::MarkerBaInput marker_input;
    marker_input.markerSet = &marker_set;

    xjw::core::project::BaInputBuildResult result;
    const auto status = xjw::core::project::buildBaInputFromMeta(meta, image_paths, 1, &result, &marker_input);

    ASSERT_EQ(status, xjw::core::project::BaInputBuildStatus::Ok);
    EXPECT_TRUE(result.markerControlNetwork.ok) << result.markerControlNetwork.error;
    EXPECT_EQ(result.markerControlTrackCount, 4);
    EXPECT_EQ(result.markerCheckTrackCount, 1);
    EXPECT_EQ(result.markerControlPointConstraintCount, 4);
    EXPECT_EQ(result.markerControlScaleBarConstraintCount, 1);
    EXPECT_EQ(result.markerCheckScaleBarCount, 1);
    ASSERT_EQ(result.imageIdByIndex.size(), image_ids.size());
    for (int index = 0; index < image_ids.size(); ++index)
    {
        EXPECT_EQ(result.imageIdByIndex[static_cast<std::size_t>(index)].value(), image_ids[index].toStdString());
    }
    ASSERT_EQ(result.tracks.size(), 5u);
    EXPECT_TRUE(result.tracks.back().controlPointConstraints.empty());
    const auto center = result.cameras.front().cameraCenter();
    const auto expected_center = controlReference({{-2.0, 0.0, 0.0}});
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_NEAR(center[axis], expected_center[axis], 1.0e-5);
    }
}

TEST(BaInputBuilderMarkerSet, RejectsGeographicReferenceBeforeBundleAdjustment)
{
    const QString image0 = QStringLiteral("E:/images/geographic-gcp-0.jpg");
    const QString image1 = QStringLiteral("E:/images/geographic-gcp-1.jpg");
    const std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> cameras = {makeCamera(-1.0),
                                                                                              makeCamera(1.0)};

    xjw::control_points::MarkerSet markerSet;
    const xjw::control_points::MarkerId markerId =
        markerSet.addMarker(QStringLiteral("geographic-gcp"), xjw::control_points::MarkerRole::ControlPoint);
    xjw::control_points::ReferenceCoordinate reference;
    reference.x = 116.391;
    reference.y = 39.907;
    reference.z = 50.0;
    reference.sigmaX = 0.01;
    reference.sigmaY = 0.01;
    reference.sigmaZ = 0.02;
    reference.sourceCrs = QStringLiteral("EPSG:4979");
    reference.axisOrder = QStringLiteral("longitude_latitude");
    reference.verticalDatum = QStringLiteral("ellipsoidal");
    reference.verticalUnit = QStringLiteral("m");
    markerSet.setReferenceCoordinate(markerId, reference);

    const std::array<double, 3> localPoint{{0.0, 0.0, 20.0}};
    for (int cameraIndex = 0; cameraIndex < static_cast<int>(cameras.size()); ++cameraIndex)
    {
        xjw::control_points::MarkerProjection projection;
        projection.imageId = QStringLiteral("uuid-%1").arg(cameraIndex);
        projection.imagePathSnapshot = cameraIndex == 0 ? image0 : image1;
        projection.xy = project(cameras[static_cast<std::size_t>(cameraIndex)], localPoint);
        projection.state = xjw::control_points::ProjectionState::ManualPinned;
        markerSet.upsertProjection(markerId, projection);
    }

    const QJsonObject meta = makeProjectMeta({{image0, cameras[0]}, {image1, cameras[1]}});
    xjw::core::project::MarkerBaInput markerInput;
    markerInput.markerSet = &markerSet;
    xjw::core::project::BaInputBuildResult result;

    EXPECT_EQ(xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result, &markerInput),
              xjw::core::project::BaInputBuildStatus::InvalidInput);
    EXPECT_EQ(result.markerControlPointConstraintCount, 0);
    EXPECT_TRUE(result.firstControlInputError.contains(QStringLiteral("地理角坐标")));
    EXPECT_TRUE(result.firstControlInputError.contains(QStringLiteral("solver")));
}

TEST(BaInputBuilderMarkerSet, ResolvesGeographicReferenceWithExplicitCoordinateContext)
{
    const QString image0 = QStringLiteral("E:/images/context-gcp-0.jpg");
    const QString image1 = QStringLiteral("E:/images/context-gcp-1.jpg");
    const std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> cameras = {makeCamera(-1.0),
                                                                                              makeCamera(1.0)};

    xjw::control_points::MarkerSet markerSet;
    const xjw::control_points::MarkerId markerId =
        markerSet.addMarker(QStringLiteral("context-gcp"), xjw::control_points::MarkerRole::ControlPoint);
    xjw::control_points::ReferenceCoordinate reference;
    reference.x = 116.391;
    reference.y = 39.907;
    reference.z = 50.0;
    reference.sigmaX = 1.0e-5;
    reference.sigmaY = 1.0e-5;
    reference.sigmaZ = 0.25;
    reference.sourceCrs = QStringLiteral("EPSG:4979");
    reference.axisOrder = QStringLiteral("longitude_latitude");
    reference.verticalDatum = QStringLiteral("ellipsoidal");
    reference.verticalUnit = QStringLiteral("m");
    markerSet.setReferenceCoordinate(markerId, reference);

    const std::array<double, 3> localPoint{{0.0, 0.0, 20.0}};
    for (int cameraIndex = 0; cameraIndex < static_cast<int>(cameras.size()); ++cameraIndex)
    {
        xjw::control_points::MarkerProjection projection;
        projection.imageId = QStringLiteral("uuid-%1").arg(cameraIndex);
        projection.xy = project(cameras[static_cast<std::size_t>(cameraIndex)], localPoint);
        projection.state = xjw::control_points::ProjectionState::ManualPinned;
        markerSet.upsertProjection(markerId, projection);
    }

    const QJsonObject meta = makeProjectMeta({{image0, cameras[0]}, {image1, cameras[1]}});
    const xjw::coordinate_system::CoordinateContext context = makeEarthContext();
    xjw::core::project::MarkerBaInput markerInput;
    markerInput.markerSet = &markerSet;
    markerInput.coordinateContext = &context;
    xjw::core::project::BaInputBuildResult result;

    EXPECT_EQ(xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result, &markerInput),
              xjw::core::project::BaInputBuildStatus::Ok);
    EXPECT_TRUE(result.firstControlInputError.isEmpty()) << qPrintable(result.firstControlInputError);
    ASSERT_EQ(result.markerTrackBindings.size(), 1);
    const auto& resolved = result.markerTrackBindings.front().referencePoint;
    const double radius = std::sqrt(resolved[0] * resolved[0] + resolved[1] * resolved[1] + resolved[2] * resolved[2]);
    EXPECT_GT(radius, 6.3e6);
    EXPECT_LT(radius, 6.4e6);
}

TEST(ProjectMatchInputReaderTest, RejectsSelectedRpcCameraAtStaticPinholeBoundary)
{
    const QString pinholePath = QStringLiteral("E:/images/pinhole.jpg");
    const QString rpcPath = QStringLiteral("E:/images/rpc.tif");
    QJsonObject meta = makeProjectMeta({{pinholePath, makeCamera(0.0)}});
    QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    images.append(QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("rpc-image")},
                              {QStringLiteral("path"), rpcPath},
                              {QStringLiteral("samples"), 1024},
                              {QStringLiteral("lines"), 768}});
    meta.insert(QStringLiteral("images"), images);
    QMap<QString, QJsonObject> rpcByPath;
    rpcByPath.insert(rpcPath, rpcMetadata());
    const auto update = xjw::camera_project::CameraProjectRecords::upsertByImagePath(&meta, rpcByPath);
    ASSERT_TRUE(update.ok()) << update.errors.join(';').toStdString();

    xjw::core::project::ProjectMatchInput pinhole_only_input;
    EXPECT_TRUE(xjw::core::project::readProjectMatchInput(meta, QStringList{pinholePath}, 1, &pinhole_only_input));
    ASSERT_EQ(pinhole_only_input.cameras.size(), 1U);
    EXPECT_TRUE(pinhole_only_input.diagnostics.firstCameraError.isEmpty());

    xjw::core::project::ProjectMatchInput input;
    EXPECT_FALSE(xjw::core::project::readProjectMatchInput(meta, QStringList{pinholePath, rpcPath}, 1, &input));
    EXPECT_TRUE(input.cameras.empty());
    EXPECT_TRUE(input.imagePathByIndex.empty());
    EXPECT_EQ(input.diagnostics.unsupportedCameraCount, 1);
    EXPECT_TRUE(input.diagnostics.firstCameraError.contains(QStringLiteral("rpc-image")));
    EXPECT_TRUE(input.diagnostics.firstCameraError.contains(QStringLiteral("rpc")));
    EXPECT_TRUE(input.diagnostics.firstCameraError.contains(QStringLiteral("static_pose")));

    xjw::core::project::BaInputBuildResult build_result;
    EXPECT_EQ(xjw::core::project::buildBaInputFromMeta(meta, QStringList{pinholePath, rpcPath}, 1, &build_result),
              xjw::core::project::BaInputBuildStatus::NoTracks);
    EXPECT_TRUE(build_result.matchDiagnostics.firstCameraError.contains(QStringLiteral("rpc-image")));
}

TEST(TriangulationServiceTest, PreservesSelectedCameraCapabilityDiagnostic)
{
    const QString pinholePath = QStringLiteral("E:/images/triangulation-pinhole.jpg");
    const QString rpcPath = QStringLiteral("E:/images/triangulation-rpc.tif");
    QJsonObject meta = makeProjectMeta({{pinholePath, makeCamera(0.0)}});
    QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    images.append(QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("triangulation-rpc-image")},
                              {QStringLiteral("path"), rpcPath},
                              {QStringLiteral("samples"), 1024},
                              {QStringLiteral("lines"), 768}});
    meta.insert(QStringLiteral("images"), images);
    QMap<QString, QJsonObject> rpcByPath;
    rpcByPath.insert(rpcPath, rpcMetadata());
    const auto update = xjw::camera_project::CameraProjectRecords::upsertByImagePath(&meta, rpcByPath);
    ASSERT_TRUE(update.ok()) << update.errors.join(';').toStdString();

    xjw::core::project::TriangulationServiceOptions options;
    options.outputDir = QStringLiteral("E:/tmp/triangulation-output");
    const auto result = xjw::core::project::TriangulationService::run(meta, QStringList{pinholePath, rpcPath}, options);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("相机/影像输入检查失败")));
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("triangulation-rpc-image")));
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("static_pose")));
}

TEST(ProjectMatchInputReaderTest, RejectsSelectedImagesWithDifferentWorldFrames)
{
    const QString image0 = QStringLiteral("E:/images/frame-a.jpg");
    const QString image1 = QStringLiteral("E:/images/frame-b.jpg");
    const QJsonObject meta =
        makeProjectMetaWithFrames(image0, image1, QStringLiteral("world-a"), QStringLiteral("world-b"));

    xjw::core::project::ProjectMatchInput input;
    EXPECT_FALSE(xjw::core::project::readProjectMatchInput(meta, QStringList{image0, image1}, 1, &input));
    EXPECT_TRUE(input.diagnostics.firstCameraError.contains(QStringLiteral("uuid-0")));
    EXPECT_TRUE(input.diagnostics.firstCameraError.contains(QStringLiteral("world-a")));
    EXPECT_TRUE(input.diagnostics.firstCameraError.contains(QStringLiteral("uuid-1")));
    EXPECT_TRUE(input.diagnostics.firstCameraError.contains(QStringLiteral("world-b")));
}

TEST(ProjectMatchInputReaderTest, IgnoresUnselectedPinholeFrameConflicts)
{
    const QString image0 = QStringLiteral("E:/images/selected-a.jpg");
    const QString image1 = QStringLiteral("E:/images/selected-b.jpg");
    const QString image2 = QStringLiteral("E:/images/unselected.jpg");
    const QJsonObject meta = makeProjectMetaWithUnselectedFrame(
        image0, image1, image2, QStringLiteral("world-a"), QStringLiteral("world-a"), QStringLiteral("world-b"));

    xjw::core::project::ProjectMatchInput input;
    EXPECT_TRUE(xjw::core::project::readProjectMatchInput(meta, QStringList{image0, image1}, 1, &input));
    EXPECT_EQ(input.cameras.size(), 2U);
    EXPECT_TRUE(input.diagnostics.firstCameraError.isEmpty());
}

TEST(ProjectMatchInputReaderTest, RejectsDuplicateSelectedImagePaths)
{
    const QString image0 = QStringLiteral("E:/images/duplicate-a.jpg");
    const QJsonObject meta =
        makeProjectMeta({{image0, makeCamera(0.0)}, {QStringLiteral("E:/images/duplicate-b.jpg"), makeCamera(1.0)}});

    xjw::core::project::ProjectMatchInput input;
    EXPECT_FALSE(xjw::core::project::readProjectMatchInput(meta, QStringList{image0, image0}, 1, &input));
    EXPECT_TRUE(input.diagnostics.firstInputError.contains(QStringLiteral("same normalized path")));
}

TEST(ProjectMatchInputReaderTest, RejectsDuplicateProjectImagePaths)
{
    const QString image0 = QStringLiteral("E:/images/project-a.jpg");
    const QString image1 = QStringLiteral("E:/images/project-b.jpg");
    QJsonObject meta = makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}});
    QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    QJsonObject duplicate = images.at(1).toObject();
    duplicate.insert(QStringLiteral("path"), image0);
    images.replace(1, duplicate);
    meta.insert(QStringLiteral("images"), images);

    xjw::core::project::ProjectMatchInput input;
    EXPECT_FALSE(xjw::core::project::readProjectMatchInput(meta, QStringList{image0, image1}, 1, &input));
    EXPECT_TRUE(input.diagnostics.firstInputError.contains(QStringLiteral("duplicate normalized paths")));
}

TEST(ProjectMatchInputReaderTest, ReportsMissingMatchShard)
{
    const QString image0 = QStringLiteral("E:/images/missing-shard-a.jpg");
    const QString image1 = QStringLiteral("E:/images/missing-shard-b.jpg");
    QJsonObject meta = makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}});
    meta.insert(
        QStringLiteral("image_match_results"),
        QJsonArray{QJsonObject{{QStringLiteral("output"), QStringLiteral("E:/matches/does-not-exist.pimatch")}}});

    xjw::core::project::BaInputBuildResult result;
    EXPECT_EQ(xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result),
              xjw::core::project::BaInputBuildStatus::NoTracks);
    EXPECT_TRUE(result.matchDiagnostics.firstShardReadError.contains(QStringLiteral("does-not-exist.pimatch")));
}

TEST(BaInputBuilderSurveyControl, UsesCanonicalImageIdBeforePathSnapshot)
{
    const QString image0 = QStringLiteral("E:/images/id-first-0.jpg");
    const QString image1 = QStringLiteral("E:/images/id-first-1.jpg");

    const QJsonObject observation0{{QStringLiteral("image_uuid"), QStringLiteral("uuid-0")},
                                   {QStringLiteral("image_path"), QStringLiteral("stale/zero.jpg")},
                                   {QStringLiteral("u"), 512.0},
                                   {QStringLiteral("v"), 384.0}};
    const QJsonObject observation1{{QStringLiteral("image_uuid"), QStringLiteral("uuid-1")},
                                   {QStringLiteral("image_path"), QStringLiteral("stale/one.jpg")},
                                   {QStringLiteral("u"), 512.0},
                                   {QStringLiteral("v"), 384.0}};
    QJsonObject controlPoint{{QStringLiteral("id"), QStringLiteral("ID-FIRST")},
                             {QStringLiteral("enabled"), true},
                             {QStringLiteral("x"), 0.0},
                             {QStringLiteral("y"), 0.0},
                             {QStringLiteral("z"), 10.0},
                             {QStringLiteral("observations"), QJsonArray{observation0, observation1}}};
    QJsonObject surveyControl{{QStringLiteral("control_points"), QJsonArray{controlPoint}}};
    QJsonObject meta = makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}});
    meta.insert(QStringLiteral("survey_control"), surveyControl);

    xjw::core::project::BaInputBuildResult result;
    EXPECT_EQ(xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result),
              xjw::core::project::BaInputBuildStatus::Ok);
    EXPECT_EQ(result.surveyControlTrackCount, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().observations.size(), 2u);
}

TEST(BaInputBuilderSurveyControl, RejectsPathOnlyObservationAtTheInputBoundary)
{
    const QString image0 = QStringLiteral("E:/images/strict-id-0.jpg");
    const QString image1 = QStringLiteral("E:/images/strict-id-1.jpg");
    const QJsonObject observation0{
        {QStringLiteral("image_path"), image0}, {QStringLiteral("u"), 512.0}, {QStringLiteral("v"), 384.0}};
    const QJsonObject observation1{
        {QStringLiteral("image_path"), image1}, {QStringLiteral("u"), 512.0}, {QStringLiteral("v"), 384.0}};
    const QJsonObject controlPoint{{QStringLiteral("id"), QStringLiteral("STRICT-ID")},
                                   {QStringLiteral("enabled"), true},
                                   {QStringLiteral("x"), 0.0},
                                   {QStringLiteral("y"), 0.0},
                                   {QStringLiteral("z"), 10.0},
                                   {QStringLiteral("observations"), QJsonArray{observation0, observation1}}};
    const QJsonObject surveyControl{{QStringLiteral("control_points"), QJsonArray{controlPoint}}};
    QJsonObject meta = makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}});
    meta.insert(QStringLiteral("survey_control"), surveyControl);

    xjw::core::project::BaInputBuildResult result;
    EXPECT_EQ(xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result),
              xjw::core::project::BaInputBuildStatus::InvalidInput);
    EXPECT_EQ(result.surveyControlTrackCount, 0);
    EXPECT_EQ(result.rejectedSurveyControlPointCount, 1);
    EXPECT_TRUE(result.firstControlInputError.contains(QStringLiteral("image_uuid")));
    EXPECT_TRUE(result.firstControlInputError.contains(QStringLiteral("STRICT-ID")));
}

TEST(BaInputBuilderMarkerSet, RejectsUnknownProjectionIdentityAtTheInputBoundary)
{
    const QString image0 = QStringLiteral("E:/images/strict-marker-0.jpg");
    const QString image1 = QStringLiteral("E:/images/strict-marker-1.jpg");
    QJsonObject markerSetJson{
        {QStringLiteral("schema_version"), 1},
        {QStringLiteral("project_image_revision"), QStringLiteral("revision")},
        {QStringLiteral("created_at"), QStringLiteral("2026-01-01T00:00:00.000Z")},
        {QStringLiteral("updated_at"), QStringLiteral("2026-01-01T00:00:00.000Z")},
        {QStringLiteral("markers"),
         QJsonArray{QJsonObject{{QStringLiteral("id"), QStringLiteral("marker-strict")},
                                {QStringLiteral("label"), QStringLiteral("strict-marker")},
                                {QStringLiteral("role"), QStringLiteral("tie_marker")},
                                {QStringLiteral("enabled"), true},
                                {QStringLiteral("projections"),
                                 QJsonArray{QJsonObject{{QStringLiteral("image_id"), QStringLiteral("stale-image-0")},
                                                        {QStringLiteral("image_path_snapshot"), image0},
                                                        {QStringLiteral("x"), 512.0},
                                                        {QStringLiteral("y"), 384.0},
                                                        {QStringLiteral("state"), QStringLiteral("manual_pinned")},
                                                        {QStringLiteral("sigma_px"), 1.0}},
                                            QJsonObject{{QStringLiteral("image_id"), QStringLiteral("stale-image-1")},
                                                        {QStringLiteral("image_path_snapshot"), image1},
                                                        {QStringLiteral("x"), 512.0},
                                                        {QStringLiteral("y"), 384.0},
                                                        {QStringLiteral("state"), QStringLiteral("manual_pinned")},
                                                        {QStringLiteral("sigma_px"), 1.0}}}}}}},
        {QStringLiteral("scale_bars"), QJsonArray{}}};
    xjw::control_points::MarkerSet markerSet;
    QString decodeError;
    ASSERT_TRUE(xjw::control_points::MarkerSetJson::decode(markerSetJson, &markerSet, &decodeError))
        << decodeError.toStdString();

    const QJsonObject meta = makeProjectMeta({{image0, makeCamera(0.0)}, {image1, makeCamera(1.0)}});
    xjw::core::project::MarkerBaInput markerInput;
    markerInput.markerSet = &markerSet;
    xjw::core::project::BaInputBuildResult result;
    EXPECT_EQ(xjw::core::project::buildBaInputFromMeta(meta, QStringList{image0, image1}, 1, &result, &markerInput),
              xjw::core::project::BaInputBuildStatus::InvalidInput);
    EXPECT_EQ(result.markerControlTrackCount, 0);
    EXPECT_EQ(result.rejectedMarkerTrackCount, 1);
    EXPECT_TRUE(result.firstControlInputError.contains(QStringLiteral("marker-strict")));
    EXPECT_TRUE(result.firstControlInputError.contains(QStringLiteral("stale-image-0")));
}
