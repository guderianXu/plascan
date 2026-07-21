#include "reporting/AerialTriangulationResultWriter.h"

#include "Camera.h"
#include "reconstruction/SfmReconstruction.h"

#include <gtest/gtest.h>

#include <QColor>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonDocument>
#include <QTemporaryDir>

TEST(AerialTriangulationResultWriterTest, WritesSparseCloudSidecarAndQualityMetadata)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageAPath = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageBPath = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    QImage imageA(64, 48, QImage::Format_RGB888);
    imageA.fill(QColor(20, 40, 60));
    ASSERT_TRUE(imageA.save(imageAPath));
    QImage imageB(64, 48, QImage::Format_RGB888);
    imageB.fill(QColor(80, 100, 120));
    ASSERT_TRUE(imageB.save(imageBPath));

    auto reconstruction = std::make_shared<xjw::SfmReconstruction>();
    xjw::ImageData imageAData;
    imageAData.id = 0;
    imageAData.imagePath = imageAPath.toStdString();
    imageAData.keypoints = {{32.0f, 24.0f}};
    imageAData.point3DIds = {xjw::kInvalidPoint3DId};
    reconstruction->addImage(imageAData);
    xjw::ImageData imageBData = imageAData;
    imageBData.id = 1;
    imageBData.imagePath = imageBPath.toStdString();
    reconstruction->addImage(imageBData);

    xjw::Camera cameraA;
    cameraA.setIntrinsics(70.0, 70.0, 32.0, 24.0);
    cameraA.setPose({1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0},
                    {-0.5, 0.0, 0.0});
    xjw::Camera cameraB = cameraA;
    cameraB.setCameraCenter({0.5, 0.0, 0.0});
    reconstruction->registerImage(0, cameraA);
    reconstruction->registerImage(1, cameraB);

    xjw::Track track;
    track.elements = {{0, 0}, {1, 0}};
    const xjw::Point3DId pointId = reconstruction->addPoint3DWithTrack({0.0, 0.0, 5.0}, track);
    reconstruction->point3D(pointId).error = 0.25;

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {imageAPath, imageBPath};
    input.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("sfm"));

    xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
    execution.reconstruction = reconstruction;
    execution.result.success = true;
    execution.result.numRegisteredImages = 2;
    execution.result.numPoints3D = 1;
    execution.result.baTracksTotal = 1;
    execution.result.baTracksOptimized = 1;

    QString errorMessage;
    ASSERT_TRUE(xjw::aerial_triangulation::AerialTriangulationResultWriter().write(
        input, &execution, &errorMessage)) << qPrintable(errorMessage);

    ASSERT_TRUE(QFile::exists(execution.result.sparseCloudPath));
    QFile ply(execution.result.sparseCloudPath);
    ASSERT_TRUE(ply.open(QIODevice::ReadOnly));
    EXPECT_TRUE(ply.read(128).contains("element vertex 1"));

    const QString sidecarPath = execution.result.resultRecordExtra
        .value(QStringLiteral("files")).toObject()
        .value(QStringLiteral("sparse_cloud_points_json")).toString();
    QFile sidecar(sidecarPath);
    ASSERT_TRUE(sidecar.open(QIODevice::ReadOnly));
    const QJsonObject sidecarObject = QJsonDocument::fromJson(sidecar.readAll()).object();
    EXPECT_EQ(sidecarObject.value(QStringLiteral("points")).toArray().size(), 1);
    EXPECT_EQ(execution.result.qualityMetadata.value(QStringLiteral("result_kind")).toString(),
              QStringLiteral("sfm_sparse_reconstruction"));
    EXPECT_EQ(execution.result.perCameraResiduals.size(), 2);
}
