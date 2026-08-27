#include "reconstruction/RpcAerialTriangulationRunner.h"
#include "reconstruction/SfmAttemptRunner.h"
#include "workflow/AerialTriangulationPipeline.h"

#include "project/SparseResultQuality.h"

#include <QDir>
#include <QFileInfo>
#include <QTemporaryDir>

#include <gtest/gtest.h>

#include <memory>

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
        input.quality = 2;

        const xjw::aerial_triangulation::RpcCameraInput cameraInput =
            xjw::aerial_triangulation::RpcAerialTriangulationRunner::inspectInput(input);
        ASSERT_EQ(cameraInput.status, xjw::aerial_triangulation::RpcCameraInputStatus::Complete)
            << cameraInput.errorMessage.toStdString();
        ASSERT_EQ(cameraInput.cameras.size(), 2);

        auto graph = std::make_shared<xjw::aerial_triangulation::PreparedTiePointGraph>();
        graph->imagePaths = input.images;
        xjw::aerial_triangulation::PreparedTiePointMatchPair pair;
        pair.imageA = 0;
        pair.imageB = 1;

        const xjw::RpcCameraModel& leftCamera = cameraInput.cameras.value(0);
        const xjw::RpcCameraModel& rightCamera = cameraInput.cameras.value(1);
        for (int row = 0; row < 5; ++row)
        {
            for (int column = 0; column < 5; ++column)
            {
                const xjw::CameraImageCoordinate leftImage{180.0 + column * 150.0, 180.0 + row * 150.0};
                xjw::RpcCameraModel::GeodeticCoordinate ground{};
                xjw::CameraImageCoordinate rightImage;
                if (!leftCamera.imageToGroundAtHeight(leftImage, 2300.0, &ground) ||
                    !rightCamera.groundToImageGeodetic(ground, &rightImage) || rightImage.sample < 0.0 ||
                    rightImage.sample >= 1031.0 || rightImage.line < 0.0 || rightImage.line >= 1102.0)
                {
                    continue;
                }

                const xjw::FeatureIdx featureIndex = static_cast<xjw::FeatureIdx>(graph->keypointsByImage[0].size());
                graph->keypointsByImage[0].push_back(
                    {static_cast<float>(leftImage.sample), static_cast<float>(leftImage.line)});
                graph->keypointsByImage[1].push_back(
                    {static_cast<float>(rightImage.sample), static_cast<float>(rightImage.line)});
                pair.matches.push_back({featureIndex, featureIndex, 1.0f});
            }
        }
        ASSERT_GE(pair.matches.size(), 10U);
        graph->trackCount = static_cast<int>(pair.matches.size());
        graph->directEdgeCount = pair.matches.size();
        graph->usesRawDirectEdges = true;
        graph->matchPairs.push_back(std::move(pair));
        input.preparedTiePointGraph = graph;

        QTemporaryDir output;
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
        EXPECT_EQ(result.pendingCamUpdates.size(), 2);
        EXPECT_EQ(result.resultRecordExtra.value(QStringLiteral("camera_model")).toString(), QStringLiteral("rpc"));
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

TEST(RpcAerialTriangulationRunnerTest, EmbeddedRpcOverridesStaleFrameMetadata)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {rpcImage(QStringLiteral("img_01.tif")),
                    rpcImage(QStringLiteral("img_02.tif"))};
    QJsonArray imageEntries;
    for (const QString& imagePath : input.images)
    {
        imageEntries.append(
            QJsonObject{{QStringLiteral("path"), imagePath},
                        {QStringLiteral("camera"),
                         QJsonObject{{QStringLiteral("model"), QStringLiteral("frame_pinhole")}}}});
    }
    input.projectMeta.insert(QStringLiteral("images"), imageEntries);

    const xjw::aerial_triangulation::RpcCameraInput cameraInput =
        xjw::aerial_triangulation::RpcAerialTriangulationRunner::inspectInput(input);
    EXPECT_EQ(cameraInput.status, xjw::aerial_triangulation::RpcCameraInputStatus::Complete);
    EXPECT_EQ(cameraInput.cameras.size(), 2);
}

} // namespace
