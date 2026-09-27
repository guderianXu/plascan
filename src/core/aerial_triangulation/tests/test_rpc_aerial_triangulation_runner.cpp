#include "file/FileIO.h"
#include "reconstruction/RpcAerialTriangulationRunner.h"
#include "reconstruction/SfmAttemptRunner.h"
#include "workflow/AerialTriangulationPipeline.h"

#include "project/SparseResultQuality.h"
#include "io/ImageIO.h"

#include <placamera/rpc_camera.h>

#include <opencv2/imgcodecs.hpp>

#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QJsonDocument>
#include <QTemporaryDir>

#include <gtest/gtest.h>

#include <memory>
#include <algorithm>

namespace
{

    QString rpcImage(const QString& name)
    {
        return QDir(QString::fromUtf8(TEST_DATA_DIR)).filePath(QStringLiteral("rpc_stereo_pair/Images/") + name);
    }

    TEST(RpcAerialTriangulationRunnerTest, IntersectsRepositoryRpcPairWithoutPinholeDowngrade)
    {
        xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
        input.images = {rpcImage(QStringLiteral("img_01.tif")), rpcImage(QStringLiteral("img_02.tif"))};
        input.imageIds = {placamera::ImageId("rpc-image-01"),
                          placamera::ImageId("rpc-image-02")};
        input.cameraBindings = {
            {placamera::CameraInstanceId("rpc-instance-01"),
             input.imageIds.at(0),
             placoordinate::CoordinateFrameId("EPSG:4978")},
            {placamera::CameraInstanceId("rpc-instance-02"),
             input.imageIds.at(1),
             placoordinate::CoordinateFrameId("EPSG:4978")}};
        input.quality = 2;

        const xjw::aerial_triangulation::RpcCameraInput cameraInput =
            xjw::aerial_triangulation::RpcAerialTriangulationRunner::inspectInput(input);
        ASSERT_EQ(cameraInput.status, xjw::aerial_triangulation::RpcCameraInputStatus::Complete)
            << cameraInput.errorMessage.toStdString();
        ASSERT_EQ(cameraInput.cameras.size(), 2);

        auto graph = std::make_shared<xjw::aerial_triangulation::PreparedTiePointGraph>();
        for (const QString& path : input.images)
        {
            graph->imagePaths.push_back(xjw::common::file::pathFromUtf8(path.toUtf8().toStdString()));
        }
        xjw::aerial_triangulation::PreparedTiePointMatchPair pair;
        pair.imageA = 0;
        pair.imageB = 1;

        const auto& leftCamera = cameraInput.cameras.at(0);
        const auto& rightCamera = cameraInput.cameras.at(1);
        ASSERT_NE(leftCamera, nullptr);
        ASSERT_NE(rightCamera, nullptr);
        for (int row = 0; row < 5; ++row)
        {
            for (int column = 0; column < 5; ++column)
            {
                const placamera::ImageCoordinate leftImage{180.0 + column * 150.0,
                                                           180.0 + row * 150.0};
                const auto ground = leftCamera->imageToGroundAtHeight(leftImage, 2300.0);
                const auto rightImage = ground
                                             ? rightCamera->groundToImageGeodetic(ground.value())
                                             : placamera::EvaluationResult<placamera::Projection>::failure(
                                                   placamera::CameraErrorCode::InvalidArgument, "invalid ground");
                if (!ground || !rightImage ||
                    rightImage.value().image.sample < 0.0 || rightImage.value().image.sample >= 1031.0 ||
                    rightImage.value().image.line < 0.0 || rightImage.value().image.line >= 1102.0)
                {
                    continue;
                }

                const xjw::FeatureIdx featureIndex = static_cast<xjw::FeatureIdx>(graph->keypointsByImage[0].size());
                graph->keypointsByImage[0].push_back(
                    {static_cast<float>(leftImage.sample), static_cast<float>(leftImage.line)});
                graph->keypointsByImage[1].push_back(
                    {static_cast<float>(rightImage.value().image.sample),
                     static_cast<float>(rightImage.value().image.line)});
                pair.matches.push_back({featureIndex, featureIndex, 1.0f});
            }
        }
        ASSERT_GE(pair.matches.size(), 10U);
        graph->trackCount = static_cast<int>(pair.matches.size());
        graph->directEdgeCount = pair.matches.size();
        graph->usesRawDirectEdges = true;
        graph->matchPairs.push_back(std::move(pair));
        input.preparedTiePointGraph = graph;

        const QString root = QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR);
        ASSERT_TRUE(QDir().mkpath(root));
        QTemporaryDir output(root + QStringLiteral("/rpc-XXXXXX"));
        ASSERT_TRUE(output.isValid());
        input.outputDir = output.path();

        const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
            xjw::aerial_triangulation::AerialTriangulationPipeline().run(input);
        ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
        EXPECT_EQ(result.numRegisteredImages, 2);
        EXPECT_GE(result.numPoints3D, 10);
        // The RPC solve follows slightly different floating-point paths across GDAL/compiler builds.
        // Keep this stricter than the 2 px production gate while accepting a stable sub-pixel solution.
        EXPECT_LT(result.meanReprojError, 0.5);
        EXPECT_EQ(result.cameraInstances.size(), 2);
        for (const auto& camera : result.cameraInstances.values())
        {
            if (camera->imageId().value() == "rpc-image-01")
            {
                EXPECT_EQ(camera->instanceId().value(), "rpc-instance-01");
            }
            else if (camera->imageId().value() == "rpc-image-02")
            {
                EXPECT_EQ(camera->instanceId().value(), "rpc-instance-02");
            }
            else
            {
                ADD_FAILURE() << "unexpected RPC ImageId: " << camera->imageId().value();
            }
            EXPECT_EQ(camera->groundFrame().value(), "EPSG:4978");
        }
        EXPECT_EQ(result.resultRecordExtra.value(QStringLiteral("camera_model")).toString(), QStringLiteral("rpc00b"));
        EXPECT_TRUE(result.resultRecordExtra.value(QStringLiteral("absolute_sensor_model")).toBool());
        EXPECT_TRUE(xjw::common::project::isProductionSparseResult(result.resultRecordExtra));
        EXPECT_FALSE(xjw::common::project::isStandardMvsCompatibleSparseResult(result.resultRecordExtra));
        EXPECT_TRUE(
            xjw::common::project::standardMvsBlockingReason(result.resultRecordExtra).contains(QStringLiteral("RPC")));
        EXPECT_TRUE(QFileInfo::exists(result.sparseCloudPath));
        EXPECT_TRUE(QFileInfo::exists(result.resultRecordExtra.value(QStringLiteral("files"))
                                          .toObject()
                                          .value(QStringLiteral("sparse_cloud_points_json"))
                                          .toString()));
        QFile cloud(result.sparseCloudPath);
        ASSERT_TRUE(cloud.open(QIODevice::ReadOnly));
        const QByteArray ply = cloud.readAll();
        const qsizetype payload = ply.indexOf("end_header\n") + 11;
        ASSERT_GT(payload, 10);
        QFile sidecar(result.resultRecordExtra.value(QStringLiteral("files"))
                          .toObject()
                          .value(QStringLiteral("sparse_cloud_points_json"))
                          .toString());
        ASSERT_TRUE(sidecar.open(QIODevice::ReadOnly));
        const QJsonArray points =
            QJsonDocument::fromJson(sidecar.readAll()).object().value(QStringLiteral("points")).toArray();
        ASSERT_EQ(ply.size() - payload, points.size() * 15);
        const cv::Mat left =
            xjw::common::io::readImage(input.images.front(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        ASSERT_EQ(left.type(), CV_8UC3);
        for (qsizetype index = 0; index < points.size(); ++index)
        {
            const QJsonObject observation =
                points.at(index).toObject().value(QStringLiteral("observations")).toArray().first().toObject();
            ASSERT_EQ(observation.value(QStringLiteral("camera_index")).toInt(), 0);
            ASSERT_EQ(observation.value(QStringLiteral("image_id")).toString(), QStringLiteral("rpc-image-01"));
            const auto& keypoint = graph->keypointsByImage[0].at(
                static_cast<std::size_t>(observation.value(QStringLiteral("feature_idx")).toInt()));
            const cv::Vec3b expected = left.at<cv::Vec3b>(std::clamp(qRound(keypoint.y), 0, left.rows - 1),
                                                          std::clamp(qRound(keypoint.x), 0, left.cols - 1));
            for (int channel = 0; channel < 3; ++channel)
            {
                EXPECT_EQ(static_cast<unsigned char>(ply.at(payload + index * 15 + 12 + channel)),
                          expected[2 - channel]);
            }
        }
    }

    TEST(RpcAerialTriangulationRunnerTest, RejectsWritebackWithoutCanonicalCameraBinding)
    {
        xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
        input.images = {rpcImage(QStringLiteral("img_01.tif")), rpcImage(QStringLiteral("img_02.tif"))};
        input.imageIds = {placamera::ImageId("rpc-image-01"), placamera::ImageId("rpc-image-02")};
        const xjw::aerial_triangulation::RpcCameraInput cameraInput =
            xjw::aerial_triangulation::RpcAerialTriangulationRunner::inspectInput(input);
        ASSERT_EQ(cameraInput.status, xjw::aerial_triangulation::RpcCameraInputStatus::Complete);

        const auto result = xjw::aerial_triangulation::RpcAerialTriangulationRunner().run(input, cameraInput.cameras);
        EXPECT_FALSE(result.success);
        EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("canonical cameraBindings")));
        EXPECT_TRUE(result.cameraInstances.empty());
    }

    TEST(RpcAerialTriangulationRunnerTest, RejectsMixedRpcAndNonRpcInput)
    {
        xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
        input.images = {rpcImage(QStringLiteral("img_01.tif")), QStringLiteral("missing-frame-image.png")};

        const xjw::aerial_triangulation::RpcCameraInput cameraInput =
            xjw::aerial_triangulation::RpcAerialTriangulationRunner::inspectInput(input);
        EXPECT_EQ(cameraInput.status, xjw::aerial_triangulation::RpcCameraInputStatus::Mixed);
        EXPECT_TRUE(cameraInput.errorMessage.contains(QStringLiteral("全部影像")));
    }

    TEST(RpcAerialTriangulationRunnerTest, RejectsEmbeddedLegacyCameraMetadata)
    {
        xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
        input.images = {rpcImage(QStringLiteral("img_01.tif")), rpcImage(QStringLiteral("img_02.tif"))};
        QJsonArray imageEntries;
        for (const QString& imagePath : input.images)
        {
            imageEntries.append(QJsonObject{
                {QStringLiteral("path"), imagePath},
                {QStringLiteral("camera"), QJsonObject{{QStringLiteral("model"), QStringLiteral("frame_pinhole")}}}});
        }
        input.projectMeta.insert(QStringLiteral("images"), imageEntries);

        const xjw::aerial_triangulation::RpcCameraInput cameraInput =
            xjw::aerial_triangulation::RpcAerialTriangulationRunner::inspectInput(input);
        EXPECT_EQ(cameraInput.status, xjw::aerial_triangulation::RpcCameraInputStatus::None);
        EXPECT_TRUE(cameraInput.cameras.empty());
        EXPECT_TRUE(cameraInput.errorMessage.contains(QStringLiteral("嵌入式相机")));
    }

    TEST(RpcAerialTriangulationRunnerTest, IgnoresCanonicalPinholeWhenSelectingRpcWorkflow)
    {
        const QJsonObject definition{
            {QStringLiteral("id"), QStringLiteral("pinhole-definition")},
            {QStringLiteral("model_type"), QStringLiteral("frame_pinhole")},
            {QStringLiteral("schema_version"), 1},
            {QStringLiteral("frame"), QStringLiteral("local")},
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
        const auto instance = [](const QString& id, const QString& imageId, double centerX)
        {
            return QJsonObject{{QStringLiteral("id"), id},
                               {QStringLiteral("image_uuid"), imageId},
                               {QStringLiteral("definition_id"), QStringLiteral("pinhole-definition")},
                               {QStringLiteral("schema_version"), 1},
                               {QStringLiteral("image_size"),
                                QJsonObject{{QStringLiteral("samples"), 640}, {QStringLiteral("lines"), 480}}},
                               {QStringLiteral("pose"),
                                QJsonObject{{QStringLiteral("frame"), QStringLiteral("local")},
                                            {QStringLiteral("center_m"), QJsonArray{centerX, 0.0, 0.0}},
                                            {QStringLiteral("camera_to_world_rotation"),
                                             QJsonArray{1.0, 0.0, 0.0,
                                                        0.0, 1.0, 0.0,
                                                        0.0, 0.0, 1.0}}}}};
        };

        xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
        input.images = {QStringLiteral("pinhole-a.png"), QStringLiteral("pinhole-b.png")};
        input.imageIds = {placamera::ImageId("pinhole-image-a"),
                          placamera::ImageId("pinhole-image-b")};
        input.projectMeta = QJsonObject{
            {QStringLiteral("images"),
             QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("pinhole-image-a")},
                                    {QStringLiteral("path"), input.images.at(0)}},
                        QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("pinhole-image-b")},
                                    {QStringLiteral("path"), input.images.at(1)}}}},
            {QStringLiteral("camera_definitions"), QJsonArray{definition}},
            {QStringLiteral("camera_instances"),
             QJsonArray{instance(QStringLiteral("pinhole-instance-a"), QStringLiteral("pinhole-image-a"), -0.5),
                        instance(QStringLiteral("pinhole-instance-b"), QStringLiteral("pinhole-image-b"), 0.5)}}};

        const xjw::aerial_triangulation::RpcCameraInput cameraInput =
            xjw::aerial_triangulation::RpcAerialTriangulationRunner::inspectInput(input);
        EXPECT_EQ(cameraInput.status, xjw::aerial_triangulation::RpcCameraInputStatus::None);
        EXPECT_TRUE(cameraInput.cameras.empty());
        EXPECT_TRUE(cameraInput.errorMessage.isEmpty());
    }

} // namespace
