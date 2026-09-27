#include "reconstruction/SfmAttemptRunner.h"
#include "reconstruction/MarkerPriorLoader.h"
#include "reconstruction/SfmReconstruction.h"
#include "placamera/reference/CameraReferencePosePrior.h"

#include <placamera/frame_numeric_state.h>
#include <placoordinate/context/CoordinateContext.h>
#include <placoordinate/gdal/GdalCoordinateTransform.h>
#include <placamera/tsai.h>
#include "io/ImageIO.h"
#include "io/MarkerSetStore.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

    placamera::FramePinholeNumericState makeProjectionCamera(const std::string& image_id, double center_x)
    {
        const placamera::FrameId frame("sfm-test-world");
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId("projection-definition-" + image_id),
            {700.0, 700.0, 320.0, 240.0, 1.0, 1, 1},
            {},
            placamera::PixelConvention::PixelCenter,
            frame);
        return placamera::FramePinholeNumericState::fromModel(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("projection-instance-" + image_id),
            placamera::ImageId(image_id),
            definition,
            {640, 480},
            placamera::Pose::create(frame, {center_x, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
    }

    void writeJson(const QString& path, const QJsonObject& object)
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
        file.write(QJsonDocument(object).toJson(QJsonDocument::Compact));
    }

    placoordinate::CoordinateContext makeEarthContext()
    {
        using namespace placoordinate;
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
            CoordinateContextId("coordctx-sfm-earth"),
            1,
            {normalize("crs-epsg-4979", "frame-wgs84-geodetic", "EPSG:4979", VerticalReference::Ellipsoidal),
             normalize("crs-epsg-4978", "frame-wgs84-ecef", "EPSG:4978", VerticalReference::NotApplicable)},
            {frame("frame-wgs84-geodetic", CoordinateFrameKind::Geodetic, AngleUnit::Degree),
             frame("frame-wgs84-ecef", CoordinateFrameKind::Ecef, AngleUnit::Radian)},
            SpatialReferenceId("crs-epsg-4978"),
            SolverFrameDefinition::create(
                CoordinateFrameId("frame-wgs84-ecef"), SolverScaleStatus::Metric, "sfm-ecef-v1"));
    }

    QJsonObject makeKnownPoseTiePoints(const QString& imageA,
                                       const QString& imageB,
                                       const placamera::FramePinholeNumericState& cameraA,
                                       const placamera::FramePinholeNumericState& cameraB)
    {
        QJsonArray tracks;
        int featureIndex = 0;
        for (int row = -2; row <= 2; ++row)
        {
            for (int column = -3; column <= 3; ++column)
            {
                const std::array<double, 3> point{column * 0.18, row * 0.16, 5.0 + 0.08 * ((column + row + 8) % 3)};
                const auto projection_a = cameraA.groundToImage({cameraA.groundFrame(), point});
                const auto projection_b = cameraB.groundToImage({cameraB.groundFrame(), point});
                EXPECT_TRUE(projection_a);
                EXPECT_TRUE(projection_b);
                const double pixelA[2]{projection_a.value().image.sample, projection_a.value().image.line};
                const double pixelB[2]{projection_b.value().image.sample, projection_b.value().image.line};

                tracks.append(QJsonObject{
                    {QStringLiteral("confidence"), 1.0},
                    {QStringLiteral("observations"),
                     QJsonArray{
                         QJsonObject{{QStringLiteral("image_id"), 0},
                                     {QStringLiteral("feature_idx"), featureIndex},
                                     {QStringLiteral("xy"), QJsonArray{pixelA[0], pixelA[1]}}},
                         QJsonObject{{QStringLiteral("image_id"), 1},
                                     {QStringLiteral("feature_idx"), featureIndex},
                                     {QStringLiteral("xy"), QJsonArray{pixelB[0], pixelB[1]}}},
                     }},
                });
                ++featureIndex;
            }
        }

        return QJsonObject{
            {QStringLiteral("format"), QStringLiteral("plascan_tie_points")},
            {QStringLiteral("format_version"), 1},
            {QStringLiteral("images"),
             QJsonArray{
                 QJsonObject{{QStringLiteral("image_id"), 0}, {QStringLiteral("path"), imageA}},
                 QJsonObject{{QStringLiteral("image_id"), 1}, {QStringLiteral("path"), imageB}},
             }},
            {QStringLiteral("tracks"), tracks},
        };
    }

    QJsonObject makeCanonicalPinholeProject(const QString& imageA, const QString& imageB)
    {
        const QJsonObject definition{{QStringLiteral("id"), QStringLiteral("canonical-definition")},
                                     {QStringLiteral("model_type"), QStringLiteral("frame_pinhole")},
                                     {QStringLiteral("schema_version"), 1},
                                     {QStringLiteral("frame"), QStringLiteral("sfm-test-world")},
                                     {QStringLiteral("parameters"),
                                      QJsonObject{{QStringLiteral("intrinsics"),
                                                   QJsonObject{{QStringLiteral("fx_px"), 700.0},
                                                               {QStringLiteral("fy_px"), 700.0},
                                                               {QStringLiteral("cx_px"), 320.0},
                                                               {QStringLiteral("cy_px"), 240.0},
                                                               {QStringLiteral("pixel_pitch_mm"), 0.01},
                                                               {QStringLiteral("u_axis_sign"), 1},
                                                               {QStringLiteral("v_axis_sign"), 1}}},
                                                  {QStringLiteral("distortion"),
                                                   QJsonObject{{QStringLiteral("k1"), 0.0},
                                                               {QStringLiteral("k2"), 0.0},
                                                               {QStringLiteral("k3"), 0.0},
                                                               {QStringLiteral("p1"), 0.0},
                                                               {QStringLiteral("p2"), 0.0}}},
                                                  {QStringLiteral("pixel_convention"), QStringLiteral("center")},
                                                  {QStringLiteral("depth_axis_flipped"), false}}}};
        const auto instance = [](const QString& instanceId, const QString& imageId, const std::array<double, 3>& center)
        {
            return QJsonObject{{QStringLiteral("id"), instanceId},
                               {QStringLiteral("image_uuid"), imageId},
                               {QStringLiteral("definition_id"), QStringLiteral("canonical-definition")},
                               {QStringLiteral("schema_version"), 1},
                               {QStringLiteral("image_size"),
                                QJsonObject{{QStringLiteral("samples"), 640}, {QStringLiteral("lines"), 480}}},
                               {QStringLiteral("pose"),
                                QJsonObject{{QStringLiteral("frame"), QStringLiteral("sfm-test-world")},
                                            {QStringLiteral("center_m"), QJsonArray{center[0], center[1], center[2]}},
                                            {QStringLiteral("camera_to_world_rotation"),
                                             QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}}}};
        };
        return QJsonObject{
            {QStringLiteral("images"),
             QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("image-a-uuid")},
                                    {QStringLiteral("path"), imageA}},
                        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("image-b-uuid")},
                                    {QStringLiteral("path"), imageB}}}},
            {QStringLiteral("camera_definitions"), QJsonArray{definition}},
            {QStringLiteral("camera_instances"),
             QJsonArray{
                 instance(QStringLiteral("canonical-instance-a"), QStringLiteral("image-a-uuid"), {-0.5, 0.0, 0.0}),
                 instance(QStringLiteral("canonical-instance-b"), QStringLiteral("image-b-uuid"), {0.5, 0.0, 0.0})}}};
    }

    QJsonObject makeCanonicalRpcProject(const QString& imageA, const QString& imageB)
    {
        QJsonArray numerator;
        QJsonArray denominator;
        for (int index = 0; index < 20; ++index)
        {
            numerator.append(index == 0 ? 1.0 : 0.0);
            denominator.append(index == 0 ? 1.0 : 0.0);
        }

        const QJsonObject definition{{QStringLiteral("id"), QStringLiteral("canonical-rpc-definition")},
                                     {QStringLiteral("model_type"), QStringLiteral("rpc00b")},
                                     {QStringLiteral("schema_version"), 1},
                                     {QStringLiteral("frame"), QStringLiteral("wgs84-geodetic")},
                                     {QStringLiteral("parameters"),
                                      QJsonObject{{QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                                                  {QStringLiteral("line_offset"), 0.0},
                                                  {QStringLiteral("sample_offset"), 0.0},
                                                  {QStringLiteral("latitude_offset"), 0.0},
                                                  {QStringLiteral("longitude_offset"), 0.0},
                                                  {QStringLiteral("height_offset"), 0.0},
                                                  {QStringLiteral("line_scale"), 1.0},
                                                  {QStringLiteral("sample_scale"), 1.0},
                                                  {QStringLiteral("latitude_scale"), 1.0},
                                                  {QStringLiteral("longitude_scale"), 1.0},
                                                  {QStringLiteral("height_scale"), 1.0},
                                                  {QStringLiteral("line_numerator"), numerator},
                                                  {QStringLiteral("line_denominator"), denominator},
                                                  {QStringLiteral("sample_numerator"), numerator},
                                                  {QStringLiteral("sample_denominator"), denominator}}}};
        const auto instance = [](const QString& instanceId, const QString& imageId)
        {
            return QJsonObject{{QStringLiteral("id"), instanceId},
                               {QStringLiteral("image_uuid"), imageId},
                               {QStringLiteral("definition_id"), QStringLiteral("canonical-rpc-definition")},
                               {QStringLiteral("schema_version"), 1},
                               {QStringLiteral("image_size"),
                                QJsonObject{{QStringLiteral("samples"), 640}, {QStringLiteral("lines"), 480}}}};
        };
        return QJsonObject{
            {QStringLiteral("images"),
             QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("image-a-uuid")},
                                    {QStringLiteral("path"), imageA}},
                        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("image-b-uuid")},
                                    {QStringLiteral("path"), imageB}}}},
            {QStringLiteral("camera_definitions"), QJsonArray{definition}},
            {QStringLiteral("camera_instances"),
             QJsonArray{instance(QStringLiteral("canonical-rpc-instance-a"), QStringLiteral("image-a-uuid")),
                        instance(QStringLiteral("canonical-rpc-instance-b"), QStringLiteral("image-b-uuid"))}}};
    }

} // namespace

TEST(SfmAttemptRunnerTest, ReadsPersistedTiePointTracksIntoCompactObservationGraph)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("latest_tie_points.json"));

    const auto observation = [](int imageId, int featureIndex, double x, double y)
    {
        return QJsonObject{
            {QStringLiteral("image_id"), imageId},
            {QStringLiteral("feature_idx"), featureIndex},
            {QStringLiteral("xy"), QJsonArray{x, y}},
        };
    };
    writeJson(tiePointPath,
              QJsonObject{
                  {QStringLiteral("format"), QStringLiteral("plascan_tie_points")},
                  {QStringLiteral("format_version"), 1},
                  {QStringLiteral("images"),
                   QJsonArray{
                       QJsonObject{{QStringLiteral("image_id"), 0}, {QStringLiteral("path"), imageA}},
                       QJsonObject{{QStringLiteral("image_id"), 1}, {QStringLiteral("path"), imageB}},
                   }},
                  {QStringLiteral("tracks"),
                   QJsonArray{
                       QJsonObject{{QStringLiteral("confidence"), 0.9},
                                   {QStringLiteral("observations"),
                                    QJsonArray{observation(0, 100, 10.0, 20.0), observation(1, 300, 11.0, 20.5)}}},
                       QJsonObject{{QStringLiteral("confidence"), 0.8},
                                   {QStringLiteral("observations"),
                                    QJsonArray{observation(0, 900, 30.0, 40.0), observation(1, 700, 31.0, 40.5)}}},
                   }},
              });

    xjw::aerial_triangulation::PreparedTiePointGraph graph;
    QString errorMessage;
    ASSERT_TRUE(xjw::aerial_triangulation::SfmAttemptRunner::readTiePointGraph(
        tiePointPath, QStringList{imageA, imageB}, &graph, &errorMessage))
        << qPrintable(errorMessage);

    EXPECT_EQ(graph.imagePaths.size(), 2);
    EXPECT_EQ(graph.keypointsByImage.at(0).size(), 2u);
    EXPECT_EQ(graph.keypointsByImage.at(1).size(), 2u);
    EXPECT_FLOAT_EQ(graph.keypointsByImage.at(0).front().scale, 1.0f);
    ASSERT_EQ(graph.matchPairs.size(), 1u);
    EXPECT_EQ(graph.matchPairs.front().matches.size(), 2u);
    EXPECT_EQ(graph.trackCount, 2);
    ASSERT_EQ(graph.tracks.size(), 2u);
    EXPECT_EQ(graph.tracks.front().length(), 2u);
    EXPECT_FALSE(graph.usesRawDirectEdges);
    EXPECT_EQ(graph.directEdgeCount, 0u);
    EXPECT_EQ(graph.synthesizedClosureEdgeCount, 2u);
}

TEST(SfmAttemptRunnerTest, RejectsIncompleteExplicitCameraBindingsBeforeReadingInputs)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};
    input.cameraBindings = {
        {placamera::CameraInstanceId("instance-a"),
         placamera::ImageId("image-a"),
         placoordinate::CoordinateFrameId("world")},
    };

    const auto result = xjw::aerial_triangulation::SfmAttemptRunner().run(input);
    EXPECT_FALSE(result.result.success);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("cameraBindings")), -1);
}

TEST(SfmAttemptRunnerTest, RejectsExternalPoseReferencesWithoutCameraBindingsWithReason)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("a.png")};

    placamera::reference::CameraReferenceObservation observation{placamera::ImageId("image-a"),
                                                                 placamera::reference::ReferenceSourceId("gnss"),
                                                                 placoordinate::CoordinateFrameId("world")};
    placamera::reference::ResolvedCameraReference resolved;
    resolved.status = placamera::reference::ReferenceResolutionStatus::Resolved;
    resolved.targetFrame = placoordinate::CoordinateFrameId("world");
    resolved.pose = placamera::Pose::create(placoordinate::CoordinateFrameId("world"),
                                            {0.0, 0.0, 0.0},
                                            placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}});
    resolved.transformProvenanceHash = "provenance-hash";
    resolved.transformHash = "transform-hash";
    const auto priorResult = placamera::reference::makeResolvedCameraPosePrior(observation, resolved);
    ASSERT_TRUE(priorResult) << priorResult.message();
    input.cameraReferencePosePriors.push_back(priorResult.value());

    const auto result = xjw::aerial_triangulation::SfmAttemptRunner().run(input);
    EXPECT_FALSE(result.result.success);
    EXPECT_FALSE(result.result.errorMessage.isEmpty());
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("cameraBindings")), -1);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("外部相机姿态参考")), -1);
    EXPECT_EQ(result.result.summary, result.result.errorMessage);
}

TEST(SfmAttemptRunnerTest, RejectsUnboundEstimatedCameraBeforeReadingInputs)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("unbound-image.png")};

    const auto result = xjw::aerial_triangulation::SfmAttemptRunner().run(input);
    EXPECT_FALSE(result.result.success);
    EXPECT_TRUE(result.result.errorMessage.contains(QStringLiteral("canonical cameraBindings")))
        << qPrintable(result.result.errorMessage);
}

TEST(SfmAttemptRunnerTest, RejectsDuplicateExplicitCameraBindings)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};
    input.cameraBindings = {
        {placamera::CameraInstanceId("instance-a"),
         placamera::ImageId("image-a"),
         placoordinate::CoordinateFrameId("world")},
        {placamera::CameraInstanceId("instance-a"),
         placamera::ImageId("image-b"),
         placoordinate::CoordinateFrameId("world")},
    };

    const auto result = xjw::aerial_triangulation::SfmAttemptRunner().run(input);
    EXPECT_FALSE(result.result.success);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("重复")), -1);
}

TEST(SfmAttemptRunnerTest, RejectsCameraBindingOrderThatDisagreesWithImageIds)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};
    input.imageIds = {placamera::ImageId("image-a"), placamera::ImageId("image-b")};
    input.cameraBindings = {
        {placamera::CameraInstanceId("instance-a"),
         placamera::ImageId("image-b"),
         placoordinate::CoordinateFrameId("world")},
        {placamera::CameraInstanceId("instance-b"),
         placamera::ImageId("image-a"),
         placoordinate::CoordinateFrameId("world")},
    };

    const auto result = xjw::aerial_triangulation::SfmAttemptRunner().run(input);
    EXPECT_FALSE(result.result.success);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("imageIds")), -1);
}

TEST(SfmAttemptRunnerTest, RejectsExplicitBindingThatDisagreesWithCanonicalProjectIdentity)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};
    input.projectMeta = makeCanonicalPinholeProject(input.images.at(0), input.images.at(1));
    input.cameraBindings = {
        {placamera::CameraInstanceId("wrong-instance-a"),
         placamera::ImageId("image-a-uuid"),
         placoordinate::CoordinateFrameId("sfm-test-world")},
        {placamera::CameraInstanceId("canonical-instance-b"),
         placamera::ImageId("image-b-uuid"),
         placoordinate::CoordinateFrameId("sfm-test-world")},
    };

    const auto result = xjw::aerial_triangulation::SfmAttemptRunner().run(input);
    EXPECT_FALSE(result.result.success);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("canonical")), -1);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("cameraBindings")), -1);
}

TEST(SfmAttemptRunnerTest, RejectsCanonicalRpcBeforeStaticSfMNumericFallback)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};
    input.imageIds = {placamera::ImageId("image-a-uuid"), placamera::ImageId("image-b-uuid")};
    input.projectMeta = makeCanonicalRpcProject(input.images.at(0), input.images.at(1));
    input.preparedTiePointGraph = std::make_shared<const xjw::aerial_triangulation::PreparedTiePointGraph>();

    const auto result = xjw::aerial_triangulation::SfmAttemptRunner().run(input);
    EXPECT_FALSE(result.result.success);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("static_sfm")), -1);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("static_pose")), -1);
}

TEST(SfmAttemptRunnerTest, RejectsAmbiguousCanonicalImagePathBeforeCameraSelection)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("same.png")};
    input.imageIds = {placamera::ImageId("image-b-uuid")};
    input.projectMeta = makeCanonicalPinholeProject(QStringLiteral("same.png"), QStringLiteral("same.png"));
    input.preparedTiePointGraph = std::make_shared<const xjw::aerial_triangulation::PreparedTiePointGraph>();

    const auto result = xjw::aerial_triangulation::SfmAttemptRunner().run(input);
    EXPECT_FALSE(result.result.success);
    EXPECT_NE(result.result.errorMessage.indexOf(QStringLiteral("ambiguous")), -1);
}

TEST(SfmAttemptRunnerTest, PreservesVersion2DirectEdgesWithoutSynthesizingTrackClosure)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    const QString imageC = QDir(tempDir.path()).filePath(QStringLiteral("c.png"));
    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("latest_tie_points.json"));
    const auto observation = [](int imageId, int featureIndex, double x)
    {
        return QJsonObject{
            {QStringLiteral("image_id"), imageId},
            {QStringLiteral("feature_idx"), featureIndex},
            {QStringLiteral("xy"), QJsonArray{x, 20.0}},
            {QStringLiteral("scale"), 2.5},
        };
    };
    writeJson(
        tiePointPath,
        QJsonObject{
            {QStringLiteral("format"), QStringLiteral("plascan_tie_points")},
            {QStringLiteral("format_version"), 2},
            {QStringLiteral("images"),
             QJsonArray{
                 QJsonObject{{QStringLiteral("image_id"), 0}, {QStringLiteral("path"), imageA}},
                 QJsonObject{{QStringLiteral("image_id"), 1}, {QStringLiteral("path"), imageB}},
                 QJsonObject{{QStringLiteral("image_id"), 2}, {QStringLiteral("path"), imageC}},
             }},
            {QStringLiteral("tracks"),
             QJsonArray{
                 QJsonObject{
                     {QStringLiteral("confidence"), 0.9},
                     {QStringLiteral("observations"),
                      QJsonArray{observation(0, 10, 10.0), observation(1, 20, 11.0), observation(2, 30, 12.0)}},
                     {QStringLiteral("direct_edges"), QJsonArray{QJsonArray{0, 1}, QJsonArray{1, 2}}},
                 },
                 QJsonObject{
                     {QStringLiteral("confidence"), 1.0},
                     {QStringLiteral("observations"), QJsonArray{observation(0, 40, 30.0), observation(2, 50, 32.0)}},
                     {QStringLiteral("direct_edges"), QJsonArray{}},
                 },
             }},
        });

    xjw::aerial_triangulation::PreparedTiePointGraph graph;
    QString errorMessage;
    ASSERT_TRUE(xjw::aerial_triangulation::SfmAttemptRunner::readTiePointGraph(
        tiePointPath, QStringList{imageA, imageB, imageC}, &graph, &errorMessage))
        << qPrintable(errorMessage);

    EXPECT_TRUE(graph.usesRawDirectEdges);
    EXPECT_EQ(graph.trackCount, 2);
    ASSERT_EQ(graph.tracks.size(), 2u);
    EXPECT_EQ(graph.tracks.front().length(), 3u);
    EXPECT_EQ(graph.directEdgeCount, 2u);
    EXPECT_EQ(graph.synthesizedClosureEdgeCount, 0u);
    ASSERT_FALSE(graph.keypointsByImage.at(0).empty());
    EXPECT_FLOAT_EQ(graph.keypointsByImage.at(0).front().scale, 2.5f);
    ASSERT_EQ(graph.matchPairs.size(), 2u);
    EXPECT_EQ(graph.matchPairs[0].matches.size(), 1u);
    EXPECT_EQ(graph.matchPairs[1].matches.size(), 1u);
    EXPECT_NE(graph.matchPairs[0].imageA, graph.matchPairs[1].imageA);
}

TEST(SfmAttemptRunnerTest, ReadsVersion3CompactObservations)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("latest_tie_points.json"));
    writeJson(tiePointPath,
              QJsonObject{
                  {QStringLiteral("format"), QStringLiteral("plascan_tie_points")},
                  {QStringLiteral("format_version"), 3},
                  {QStringLiteral("observation_fields"),
                   QJsonArray{QStringLiteral("image_id"),
                              QStringLiteral("feature_idx"),
                              QStringLiteral("x"),
                              QStringLiteral("y"),
                              QStringLiteral("scale")}},
                  {QStringLiteral("images"),
                   QJsonArray{
                       QJsonObject{{QStringLiteral("image_id"), 0}, {QStringLiteral("path"), imageA}},
                       QJsonObject{{QStringLiteral("image_id"), 1}, {QStringLiteral("path"), imageB}},
                   }},
                  {QStringLiteral("tracks"),
                   QJsonArray{QJsonObject{
                       {QStringLiteral("confidence"), 0.75},
                       {QStringLiteral("observations"),
                        QJsonArray{QJsonArray{0, 10, 12.5, 20.0, 1.5}, QJsonArray{1, 20, 13.0, 20.5, 2.5}}},
                       {QStringLiteral("direct_edges"), QJsonArray{QJsonValue(QJsonArray{0, 1})}},
                   }}},
              });

    xjw::aerial_triangulation::PreparedTiePointGraph graph;
    QString errorMessage;
    ASSERT_TRUE(xjw::aerial_triangulation::SfmAttemptRunner::readTiePointGraph(
        tiePointPath, QStringList{imageA, imageB}, &graph, &errorMessage))
        << qPrintable(errorMessage);

    EXPECT_TRUE(graph.usesRawDirectEdges);
    ASSERT_EQ(graph.tracks.size(), 1u);
    EXPECT_EQ(graph.directEdgeCount, 1u);
    ASSERT_EQ(graph.keypointsByImage.at(0).size(), 1u);
    ASSERT_EQ(graph.keypointsByImage.at(1).size(), 1u);
    EXPECT_FLOAT_EQ(graph.keypointsByImage.at(0).front().x, 12.5f);
    EXPECT_FLOAT_EQ(graph.keypointsByImage.at(0).front().scale, 1.5f);
    EXPECT_FLOAT_EQ(graph.keypointsByImage.at(1).front().scale, 2.5f);
}

TEST(SfmAttemptRunnerTest, RejectsTiePointFileFromAnotherImageSet)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());
    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("latest_tie_points.json"));
    writeJson(
        tiePointPath,
        QJsonObject{
            {QStringLiteral("format"), QStringLiteral("plascan_tie_points")},
            {QStringLiteral("format_version"), 1},
            {QStringLiteral("images"),
             QJsonArray{
                 QJsonObject{{QStringLiteral("image_id"), 0}, {QStringLiteral("path"), QStringLiteral("old_a.png")}},
                 QJsonObject{{QStringLiteral("image_id"), 1}, {QStringLiteral("path"), QStringLiteral("old_b.png")}},
             }},
            {QStringLiteral("tracks"), QJsonArray{}}});

    xjw::aerial_triangulation::PreparedTiePointGraph graph;
    QString errorMessage;
    EXPECT_FALSE(xjw::aerial_triangulation::SfmAttemptRunner::readTiePointGraph(
        tiePointPath, QStringList{QStringLiteral("new_a.png"), QStringLiteral("new_b.png")}, &graph, &errorMessage));
    EXPECT_TRUE(errorMessage.contains(QStringLiteral("影像集合")));
}

TEST(SfmAttemptRunnerTest, ResolvesUnicodeTiffSizeWithoutUsingKeypointBounds)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());
    const QString unicodeDir = QDir(tempDir.path()).filePath(QStringLiteral("三维建模"));
    ASSERT_TRUE(QDir().mkpath(unicodeDir));
    const QString imagePath = QDir(unicodeDir).filePath(QStringLiteral("龙宫.tif"));

    const cv::Mat image(1024, 1024, CV_8UC1, cv::Scalar(127));
    const QString temporaryAsciiPath = QDir(tempDir.path()).filePath(QStringLiteral("unicode_source.tif"));
    ASSERT_TRUE(cv::imwrite(temporaryAsciiPath.toStdString(), image));
    ASSERT_TRUE(QFile::rename(temporaryAsciiPath, imagePath));

    EXPECT_EQ(xjw::aerial_triangulation::SfmAttemptRunner::resolveInputImageSize(imagePath), QSize(1024, 1024));
    EXPECT_FALSE(xjw::aerial_triangulation::SfmAttemptRunner::resolveInputImageSize(
                     QDir(unicodeDir).filePath(QStringLiteral("missing.tif")))
                     .isValid());
}

TEST(SfmAttemptRunnerTest, LoadsMarkerTracksAndScaleBarsFromProjectSidecar)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    const QString markerPath = QDir(tempDir.path()).filePath(QStringLiteral("markers.json"));

    xjw::control_points::MarkerSet markerSet;
    const auto addMarker = [&](const QString& label, double xOffset)
    {
        const xjw::control_points::MarkerId markerId =
            markerSet.addMarker(label, xjw::control_points::MarkerRole::ControlPoint);
        xjw::control_points::ReferenceCoordinate reference;
        reference.x = xOffset;
        reference.y = 2.0;
        reference.z = 3.0;
        reference.sigmaX = 0.01;
        reference.sigmaY = 0.02;
        reference.sigmaZ = 0.03;
        reference.sourceCrs = QStringLiteral("EPSG:4978");
        markerSet.setReferenceCoordinate(markerId, reference);

        xjw::control_points::MarkerProjection first;
        first.imageId = QStringLiteral("image-a");
        first.imagePathSnapshot = imageA;
        first.xy = QPointF(100.0 + xOffset, 120.0);
        first.state = xjw::control_points::ProjectionState::ManualPinned;
        markerSet.upsertProjection(markerId, first);

        xjw::control_points::MarkerProjection second;
        second.imageId = QStringLiteral("image-b");
        second.imagePathSnapshot = imageB;
        second.xy = QPointF(101.0 + xOffset, 121.0);
        second.state = xjw::control_points::ProjectionState::AutoDetected;
        second.confidence = 0.8;
        markerSet.upsertProjection(markerId, second);
        return markerId;
    };

    const xjw::control_points::MarkerId firstMarker = addMarker(QStringLiteral("control-a"), 0.0);
    const xjw::control_points::MarkerId secondMarker = addMarker(QStringLiteral("control-b"), 1.0);
    markerSet.addScaleBar(
        QStringLiteral("scale"), firstMarker, secondMarker, 1.0, 0.005, xjw::control_points::ScaleBarRole::Control);
    const xjw::control_points::MarkerSetIoResult saved =
        xjw::control_points::MarkerSetStore(markerPath).save(markerSet);
    ASSERT_TRUE(saved.ok) << qPrintable(saved.error);

    QMap<QString, xjw::ImageId> imageIdByCanonicalId;
    imageIdByCanonicalId.insert(QStringLiteral("image-a"), 5);
    imageIdByCanonicalId.insert(QStringLiteral("image-b"), 8);

    const xjw::aerial_triangulation::MarkerPriorLoadResult loaded =
        xjw::aerial_triangulation::MarkerPriorLoader::load(markerPath, QJsonObject(), imageIdByCanonicalId);

    ASSERT_TRUE(loaded.ok) << qPrintable(loaded.errorMessage);
    ASSERT_EQ(loaded.tracks.size(), 2u);
    ASSERT_EQ(loaded.scaleBars.size(), 1u);
    ASSERT_EQ(loaded.tracks.front().observations.size(), 2u);
    EXPECT_EQ(loaded.tracks.front().observations[0].imageId, 5u);
    EXPECT_EQ(loaded.tracks.front().observations[1].imageId, 8u);
    EXPECT_TRUE(loaded.tracks.front().hasReference);
    EXPECT_DOUBLE_EQ(loaded.tracks.front().referencePoint[1], 2.0);
    EXPECT_DOUBLE_EQ(loaded.tracks.front().referenceSigma[2], 0.03);
    EXPECT_EQ(loaded.scaleBars.front().firstMarkerId, firstMarker.toStdString());
    EXPECT_EQ(loaded.scaleBars.front().secondMarkerId, secondMarker.toStdString());
}

TEST(SfmAttemptRunnerTest, RejectsGeographicMarkerReferenceBeforeSfm)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());

    const QString markerPath = QDir(tempDir.path()).filePath(QStringLiteral("geographic-markers.json"));
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

    for (const QString& imageId : {QStringLiteral("image-a"), QStringLiteral("image-b")})
    {
        xjw::control_points::MarkerProjection projection;
        projection.imageId = imageId;
        projection.xy = QPointF(100.0, 120.0);
        projection.state = xjw::control_points::ProjectionState::ManualPinned;
        markerSet.upsertProjection(markerId, projection);
    }
    const xjw::control_points::MarkerSetIoResult saved =
        xjw::control_points::MarkerSetStore(markerPath).save(markerSet);
    ASSERT_TRUE(saved.ok) << qPrintable(saved.error);

    const QMap<QString, xjw::ImageId> imageIds{{QStringLiteral("image-a"), 5}, {QStringLiteral("image-b"), 8}};
    const xjw::aerial_triangulation::MarkerPriorLoadResult loaded =
        xjw::aerial_triangulation::MarkerPriorLoader::load(markerPath, QJsonObject(), imageIds);

    EXPECT_FALSE(loaded.ok);
    EXPECT_TRUE(loaded.errorMessage.contains(QStringLiteral("geographic-gcp")));
    EXPECT_TRUE(loaded.errorMessage.contains(QStringLiteral("地理角坐标")));
}

TEST(SfmAttemptRunnerTest, ResolvesGeographicMarkerReferenceWithCoordinateContext)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());

    const QString markerPath = QDir(tempDir.path()).filePath(QStringLiteral("context-geographic-markers.json"));
    xjw::control_points::MarkerSet markerSet;
    const xjw::control_points::MarkerId markerId =
        markerSet.addMarker(QStringLiteral("context-geographic-gcp"), xjw::control_points::MarkerRole::ControlPoint);
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

    for (const QString& imageId : {QStringLiteral("image-a"), QStringLiteral("image-b")})
    {
        xjw::control_points::MarkerProjection projection;
        projection.imageId = imageId;
        projection.xy = QPointF(100.0, 120.0);
        projection.state = xjw::control_points::ProjectionState::ManualPinned;
        markerSet.upsertProjection(markerId, projection);
    }
    const xjw::control_points::MarkerSetIoResult saved =
        xjw::control_points::MarkerSetStore(markerPath).save(markerSet);
    ASSERT_TRUE(saved.ok) << qPrintable(saved.error);

    const QMap<QString, xjw::ImageId> imageIds{{QStringLiteral("image-a"), 5}, {QStringLiteral("image-b"), 8}};
    const placoordinate::CoordinateContext context = makeEarthContext();
    const xjw::aerial_triangulation::MarkerPriorLoadResult loaded =
        xjw::aerial_triangulation::MarkerPriorLoader::load(markerPath, QJsonObject(), imageIds, &context);

    ASSERT_TRUE(loaded.ok) << qPrintable(loaded.errorMessage);
    ASSERT_EQ(loaded.tracks.size(), 1U);
    ASSERT_TRUE(loaded.tracks.front().hasReference);
    const auto& point = loaded.tracks.front().referencePoint;
    const double radius = std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
    EXPECT_GT(radius, 6.3e6);
    EXPECT_LT(radius, 6.4e6);
}

TEST(SfmAttemptRunnerTest, RejectsMarkerPathFallbackWhenCanonicalImageIdIsUnknown)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());

    const QString markerPath = QDir(tempDir.path()).filePath(QStringLiteral("markers.json"));
    writeJson(markerPath,
              QJsonObject{{QStringLiteral("schema_version"), 1},
                          {QStringLiteral("project_image_revision"), QStringLiteral("revision")},
                          {QStringLiteral("created_at"), QStringLiteral("2026-01-01T00:00:00.000Z")},
                          {QStringLiteral("updated_at"), QStringLiteral("2026-01-01T00:00:00.000Z")},
                          {QStringLiteral("markers"),
                           QJsonArray{QJsonObject{
                               {QStringLiteral("id"), QStringLiteral("marker-unknown")},
                               {QStringLiteral("label"), QStringLiteral("unknown")},
                               {QStringLiteral("role"), QStringLiteral("tie_marker")},
                               {QStringLiteral("enabled"), true},
                               {QStringLiteral("projections"),
                                QJsonArray{QJsonObject{{QStringLiteral("image_id"), QStringLiteral("missing-image")},
                                                       {QStringLiteral("image_path_snapshot"), QStringLiteral("a.png")},
                                                       {QStringLiteral("x"), 10.0},
                                                       {QStringLiteral("y"), 20.0},
                                                       {QStringLiteral("state"), QStringLiteral("manual_pinned")},
                                                       {QStringLiteral("sigma_px"), 1.0}}}}}}},
                          {QStringLiteral("scale_bars"), QJsonArray{}}});

    const QJsonObject projectMeta{{QStringLiteral("images"),
                                   QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("image-a")},
                                                          {QStringLiteral("path"), QStringLiteral("a.png")}}}}};
    const QMap<QString, xjw::ImageId> imageIds{{QStringLiteral("image-a"), 0}};
    const auto loaded = xjw::aerial_triangulation::MarkerPriorLoader::load(markerPath, projectMeta, imageIds);

    EXPECT_FALSE(loaded.ok);
    EXPECT_TRUE(loaded.errorMessage.contains(QStringLiteral("canonical image_uuid")));
    EXPECT_TRUE(loaded.errorMessage.contains(QStringLiteral("missing-image")));
}

TEST(SfmAttemptRunnerTest, RunsKnownPoseSfmFromPreparedTiePointGraph)
{
    QDir().mkpath(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR));
    QTemporaryDir tempDir(QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR) + QStringLiteral("/run-XXXXXX"));
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    ASSERT_TRUE(xjw::common::io::writeImage(imageA, cv::Mat(480, 640, CV_8UC1, cv::Scalar(127))));
    ASSERT_TRUE(xjw::common::io::writeImage(imageB, cv::Mat(480, 640, CV_8UC1, cv::Scalar(127))));

    const auto cameraA = makeProjectionCamera("image-a-uuid", -0.5);
    const auto cameraB = makeProjectionCamera("image-b-uuid", 0.5);

    const QString cameraPathA = QDir(tempDir.path()).filePath(QStringLiteral("a.tsai"));
    const QString cameraPathB = QDir(tempDir.path()).filePath(QStringLiteral("b.tsai"));
    const placamera::FrameId worldFrame("sfm-test-world");
    const auto definition =
        placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("known-pose-camera"),
                                                  placamera::FrameIntrinsics{700.0, 700.0, 320.0, 240.0, 1.0, 1, 1},
                                                  placamera::BrownConradyDistortion{},
                                                  placamera::PixelConvention::PixelCenter,
                                                  worldFrame,
                                                  false);
    const placamera::RotationMatrix rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const placamera::TsaiFramePinhole nativeCameraA{definition,
                                                    placamera::Pose::create(worldFrame, {-0.5, 0.0, 0.0}, rotation)};
    const placamera::TsaiFramePinhole nativeCameraB{definition,
                                                    placamera::Pose::create(worldFrame, {0.5, 0.0, 0.0}, rotation)};
    const auto savedCameraA = placamera::saveTsaiFramePinhole(nativeCameraA, cameraPathA.toStdString());
    ASSERT_TRUE(savedCameraA) << savedCameraA.message();
    const auto savedCameraB = placamera::saveTsaiFramePinhole(nativeCameraB, cameraPathB.toStdString());
    ASSERT_TRUE(savedCameraB) << savedCameraB.message();

    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("latest_tie_points.json"));
    writeJson(tiePointPath, makeKnownPoseTiePoints(imageA, imageB, cameraA, cameraB));
    auto preparedGraph = std::make_shared<xjw::aerial_triangulation::PreparedTiePointGraph>();
    QString graphError;
    ASSERT_TRUE(xjw::aerial_triangulation::SfmAttemptRunner::readTiePointGraph(
        tiePointPath, {imageA, imageB}, preparedGraph.get(), &graphError))
        << qPrintable(graphError);

    const QString markerPath = QDir(tempDir.path()).filePath(QStringLiteral("markers.json"));
    xjw::control_points::MarkerSet markerSet;
    const xjw::control_points::MarkerId markerId =
        markerSet.addMarker(QStringLiteral("manual-tie"), xjw::control_points::MarkerRole::TieMarker);
    const std::array<double, 3> markerPoint{{0.0, 0.0, 5.0}};
    const auto marker_projection_a = cameraA.groundToImage({cameraA.groundFrame(), markerPoint});
    const auto marker_projection_b = cameraB.groundToImage({cameraB.groundFrame(), markerPoint});
    ASSERT_TRUE(marker_projection_a);
    ASSERT_TRUE(marker_projection_b);
    const double markerPixelA[2]{marker_projection_a.value().image.sample, marker_projection_a.value().image.line};
    const double markerPixelB[2]{marker_projection_b.value().image.sample, marker_projection_b.value().image.line};
    xjw::control_points::MarkerProjection markerProjectionA;
    markerProjectionA.imageId = QStringLiteral("image-a-uuid");
    markerProjectionA.imagePathSnapshot = imageA;
    markerProjectionA.xy = QPointF(markerPixelA[0], markerPixelA[1]);
    markerProjectionA.state = xjw::control_points::ProjectionState::ManualPinned;
    markerSet.upsertProjection(markerId, markerProjectionA);
    xjw::control_points::MarkerProjection markerProjectionB;
    markerProjectionB.imageId = QStringLiteral("image-b-uuid");
    markerProjectionB.imagePathSnapshot = imageB;
    markerProjectionB.xy = QPointF(markerPixelB[0], markerPixelB[1]);
    markerProjectionB.state = xjw::control_points::ProjectionState::ManualPinned;
    markerSet.upsertProjection(markerId, markerProjectionB);
    const xjw::control_points::MarkerSetIoResult markerSaved =
        xjw::control_points::MarkerSetStore(markerPath).save(markerSet);
    ASSERT_TRUE(markerSaved.ok) << qPrintable(markerSaved.error);

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {imageA, imageB};
    input.cameraBindings = {
        {placamera::CameraInstanceId("camera-instance-a"),
         placamera::ImageId("image-a-uuid"),
         placoordinate::CoordinateFrameId("sfm-test-world")},
        {placamera::CameraInstanceId("camera-instance-b"),
         placamera::ImageId("image-b-uuid"),
         placoordinate::CoordinateFrameId("sfm-test-world")},
    };
    input.cameraPaths = {cameraPathA, cameraPathB};
    input.tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("already_prepared.json"));
    input.preparedTiePointGraph = preparedGraph;
    input.markerSetPath = markerPath;
    input.outputDir = tempDir.path();
    input.maxTracksPerImage = 8000;
    input.maxTracksPerGridCell = 125;
    input.trackThinningGridColumns = 8;
    input.trackThinningGridRows = 8;
    QStringList progressStages;
    input.progressFn = [&progressStages](const QString& stage, int) { progressStages.push_back(stage); };

    const xjw::aerial_triangulation::SfmAttemptExecutionResult result =
        xjw::aerial_triangulation::SfmAttemptRunner().run(input);

    ASSERT_TRUE(result.result.success) << qPrintable(result.result.errorMessage);
    ASSERT_NE(result.reconstruction, nullptr);
    EXPECT_EQ(result.graph, preparedGraph);
    ASSERT_TRUE(result.reconstruction->hasCamera(0));
    ASSERT_TRUE(result.reconstruction->hasCamera(1));
    EXPECT_EQ(result.reconstruction->camera(0).imageId(), placamera::ImageId("image-a-uuid"));
    EXPECT_EQ(result.reconstruction->camera(1).imageId(), placamera::ImageId("image-b-uuid"));
    EXPECT_EQ(result.reconstruction->camera(0).groundFrame(), placoordinate::CoordinateFrameId("sfm-test-world"));
    EXPECT_EQ(result.reconstruction->camera(1).groundFrame(), placoordinate::CoordinateFrameId("sfm-test-world"));
    EXPECT_EQ(result.result.numRegisteredImages, 2);
    EXPECT_GE(result.result.numPoints3D, 20);
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("marker_prior_tracks_loaded")).toInt(), 1);
    EXPECT_GE(result.result.sfmDiagnostics.value(QStringLiteral("prior_tracks_accepted")).toInt(), 1);
    EXPECT_TRUE(std::any_of(progressStages.cbegin(),
                            progressStages.cend(),
                            [](const QString& stage) { return stage.contains(QStringLiteral("光束法平差")); }));
    EXPECT_FALSE(result.result.sfmDiagnostics.value(QStringLiteral("ba_requested_backend")).toString().isEmpty());
    EXPECT_FALSE(result.result.sfmDiagnostics.value(QStringLiteral("ba_used_backend")).toString().isEmpty());
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("ba_solve_status")).toString(),
              QStringLiteral("success"));
    EXPECT_TRUE(result.result.sfmDiagnostics.value(QStringLiteral("ba_solution_usable")).toBool());
    EXPECT_TRUE(result.result.sfmDiagnostics.value(QStringLiteral("ba_result_applied")).toBool());
    EXPECT_GT(result.result.sfmDiagnostics.value(QStringLiteral("ba_observations")).toInt(), 0);
    EXPECT_GE(result.result.sfmDiagnostics.value(QStringLiteral("ba_refined_intrinsic_count")).toInt(), 0);
    EXPECT_FALSE(
        result.result.sfmDiagnostics.value(QStringLiteral("ba_adaptive_camera_model_fitting_evaluated")).toBool());
    EXPECT_FALSE(
        result.result.sfmDiagnostics.value(QStringLiteral("ba_adaptive_camera_model_fitting_applied")).toBool());
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("ba_refined_intrinsic_count")).toInt(), 0);
    EXPECT_GT(result.result.sfmDiagnostics.value(QStringLiteral("ba_shared_focal_scale")).toDouble(), 0.0);

    auto estimatedInput = input;
    estimatedInput.cameraPaths.clear();
    estimatedInput.markerSetPath.clear();
    estimatedInput.estimatedFocalScale = 700.0 / 640.0;
    estimatedInput.adaptiveCameraModelFitting = false;
    const auto estimatedResult = xjw::aerial_triangulation::SfmAttemptRunner().run(estimatedInput);
    ASSERT_TRUE(estimatedResult.result.success) << qPrintable(estimatedResult.result.errorMessage);
    ASSERT_NE(estimatedResult.reconstruction, nullptr);
    EXPECT_EQ(estimatedResult.reconstruction->camera(0).imageId(), placamera::ImageId("image-a-uuid"));
    EXPECT_EQ(estimatedResult.reconstruction->camera(1).imageId(), placamera::ImageId("image-b-uuid"));

    auto unboundInput = input;
    unboundInput.cameraBindings.clear();
    const auto unboundResult = xjw::aerial_triangulation::SfmAttemptRunner().run(unboundInput);
    EXPECT_FALSE(unboundResult.result.success);
    EXPECT_TRUE(unboundResult.result.errorMessage.contains(QStringLiteral("canonical cameraBindings")))
        << qPrintable(unboundResult.result.errorMessage);

    // A production project already has canonical camera instances.  The
    // runner must propagate those identities when the caller does not repeat
    // the binding vector, while still using the external camera files for the
    // numerical state.
    auto autoBoundInput = input;
    autoBoundInput.cameraBindings.clear();
    autoBoundInput.projectMeta = makeCanonicalPinholeProject(imageA, imageB);
    const auto autoBoundResult = xjw::aerial_triangulation::SfmAttemptRunner().run(autoBoundInput);
    ASSERT_TRUE(autoBoundResult.result.success) << qPrintable(autoBoundResult.result.errorMessage);
    ASSERT_NE(autoBoundResult.reconstruction, nullptr);
    ASSERT_TRUE(autoBoundResult.reconstruction->hasCamera(0));
    ASSERT_TRUE(autoBoundResult.reconstruction->hasCamera(1));
    EXPECT_EQ(autoBoundResult.reconstruction->camera(0).instanceId(),
              placamera::CameraInstanceId("canonical-instance-a"));
    EXPECT_EQ(autoBoundResult.reconstruction->camera(1).instanceId(),
              placamera::CameraInstanceId("canonical-instance-b"));
    EXPECT_EQ(autoBoundResult.reconstruction->camera(0).imageId(), placamera::ImageId("image-a-uuid"));
    EXPECT_EQ(autoBoundResult.reconstruction->camera(1).imageId(), placamera::ImageId("image-b-uuid"));
    EXPECT_EQ(autoBoundResult.result.sfmDiagnostics.value(QStringLiteral("camera_binding_source")).toString(),
              QStringLiteral("canonical_project_instances"));
    EXPECT_EQ(autoBoundResult.result.sfmDiagnostics.value(QStringLiteral("camera_binding_count")).toInt(), 2);
    EXPECT_EQ(
        result.result.sfmDiagnostics.value(QStringLiteral("ba_intrinsic_parameter_reference_definition")).toString(),
        QStringLiteral("normalized_stable_calibration_group_reference"));
    const QJsonObject intrinsicReference =
        result.result.sfmDiagnostics.value(QStringLiteral("ba_intrinsic_parameter_reference")).toObject();
    const QJsonObject intrinsicFinal =
        result.result.sfmDiagnostics.value(QStringLiteral("ba_intrinsic_parameter_final")).toObject();
    const QJsonObject intrinsicDelta =
        result.result.sfmDiagnostics.value(QStringLiteral("ba_intrinsic_parameter_delta")).toObject();
    EXPECT_DOUBLE_EQ(intrinsicReference.value(QStringLiteral("focal_scale")).toDouble(), 1.0);
    EXPECT_DOUBLE_EQ(intrinsicFinal.value(QStringLiteral("focal_scale")).toDouble(),
                     result.result.sfmDiagnostics.value(QStringLiteral("ba_shared_focal_scale")).toDouble());
    EXPECT_DOUBLE_EQ(intrinsicDelta.value(QStringLiteral("focal_scale")).toDouble(),
                     intrinsicFinal.value(QStringLiteral("focal_scale")).toDouble() - 1.0);
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("final_camera_focal_count")).toInt(), 2);
    EXPECT_GT(result.result.sfmDiagnostics.value(QStringLiteral("final_camera_focal_median_px")).toDouble(), 0.0);
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("input_max_tracks_per_image")).toInt(), 8000);
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("input_max_tracks_per_grid_cell")).toInt(), 125);
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("input_track_thinning_grid_columns")).toInt(), 8);
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("input_track_thinning_grid_rows")).toInt(), 8);
}
