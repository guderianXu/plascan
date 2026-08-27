#include "reporting/AerialTriangulationResultWriter.h"
#include "reporting/QualityReportWriter.h"

#include "FramePinholeCamera.h"
#include "reconstruction/SfmReconstruction.h"

#include <gtest/gtest.h>

#include <QColor>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonDocument>
#include <QTemporaryDir>

#include <array>
#include <cmath>
#include <string>
#include <vector>

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
    imageAData.keypoints = {{32.0f, 24.0f, 2.0f}};
    imageAData.point3DIds = {xjw::kInvalidPoint3DId};
    reconstruction->addImage(imageAData);
    xjw::ImageData imageBData = imageAData;
    imageBData.id = 1;
    imageBData.imagePath = imageBPath.toStdString();
    imageBData.keypoints = {{32.0f, 24.0f, 4.0f}};
    reconstruction->addImage(imageBData);

    xjw::FramePinholeCamera cameraA;
    cameraA.setIntrinsics(70.0, 70.0, 32.0, 24.0);
    cameraA.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {-0.5, 0.0, 0.0});
    xjw::FramePinholeCamera cameraB = cameraA;
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

    const QJsonObject qualitySummary = xjw::aerial_triangulation::QualityReportWriter::buildSparseQualitySummary(
        input, *reconstruction, execution.result);
    EXPECT_EQ(qualitySummary.value(QStringLiteral("point_count")).toInt(), 1);
    EXPECT_TRUE(qualitySummary.contains(QStringLiteral("triangulation_angle")));
    EXPECT_FALSE(qualitySummary.contains(QStringLiteral("points")));
    EXPECT_FALSE(qualitySummary.contains(QStringLiteral("per_camera_residuals")));

    xjw::Track incompleteTrack;
    incompleteTrack.elements = {{0, 0}, {2, 0}};
    const xjw::Point3DId incompletePointId = reconstruction->addPoint3DWithTrack({0.0, 0.0, 6.0}, incompleteTrack);
    reconstruction->point3D(incompletePointId).error = 0.5;
    execution.result.numPoints3D = 2;

    QString errorMessage;
    ASSERT_TRUE(xjw::aerial_triangulation::AerialTriangulationResultWriter().write(input, &execution, &errorMessage))
        << qPrintable(errorMessage);

    ASSERT_TRUE(QFile::exists(execution.result.sparseCloudPath));
    QFile ply(execution.result.sparseCloudPath);
    ASSERT_TRUE(ply.open(QIODevice::ReadOnly));
    EXPECT_TRUE(ply.read(128).contains("element vertex 1"));

    const QString sidecarPath = execution.result.resultRecordExtra.value(QStringLiteral("files"))
                                    .toObject()
                                    .value(QStringLiteral("sparse_cloud_points_json"))
                                    .toString();
    QFile sidecar(sidecarPath);
    ASSERT_TRUE(sidecar.open(QIODevice::ReadOnly));
    const QJsonObject sidecarObject = QJsonDocument::fromJson(sidecar.readAll()).object();
    EXPECT_EQ(sidecarObject.value(QStringLiteral("points")).toArray().size(), 1);
    const QJsonObject sidecarPoint = sidecarObject.value(QStringLiteral("points")).toArray().first().toObject();
    EXPECT_TRUE(sidecarPoint.contains(QStringLiteral("rms_reproj_px")));
    EXPECT_TRUE(sidecarPoint.contains(QStringLiteral("track_len")));
    EXPECT_TRUE(sidecarPoint.contains(QStringLiteral("min_tri_angle_deg")));
    EXPECT_TRUE(sidecarPoint.contains(QStringLiteral("reconstruction_uncertainty")));
    EXPECT_TRUE(sidecarPoint.contains(QStringLiteral("projection_accuracy")));
    EXPECT_TRUE(sidecarPoint.value(QStringLiteral("reconstruction_uncertainty")).toDouble() >= 1.0);
    EXPECT_DOUBLE_EQ(sidecarPoint.value(QStringLiteral("projection_accuracy")).toDouble(), 3.0);
    EXPECT_EQ(execution.result.numPoints3D, 1);
    EXPECT_EQ(execution.result.qualityMetadata.value(QStringLiteral("result_kind")).toString(),
              QStringLiteral("sfm_sparse_reconstruction"));
    EXPECT_EQ(execution.result.perCameraResiduals.size(), 2);
}

TEST(AerialTriangulationResultWriterTest, RemovesWeakAndSpatiallyIsolatedPublishedPoints)
{
    auto reconstruction = std::make_shared<xjw::SfmReconstruction>();
    std::vector<xjw::FramePinholeCamera> cameras(3);
    for (int index = 0; index < 3; ++index)
    {
        cameras[static_cast<std::size_t>(index)].setIntrinsics(500.0, 500.0, 320.0, 240.0);
        cameras[static_cast<std::size_t>(index)].setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                                                         {static_cast<double>(index - 1), 0.0, 0.0});
    }

    std::vector<std::array<double, 3>> worldPoints;
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 5; ++column)
        {
            worldPoints.push_back(
                {-0.4 + 0.2 * static_cast<double>(column), -0.3 + 0.2 * static_cast<double>(row), 5.0});
        }
    }
    const std::size_t isolatedPointIndex = worldPoints.size();
    worldPoints.push_back({10.0, 0.0, 5.0});
    const std::size_t weakTwoViewPointIndex = worldPoints.size();
    worldPoints.push_back({0.0, 0.0, 5.0});
    const std::size_t highErrorPointIndex = worldPoints.size();
    worldPoints.push_back({0.0, 0.0, 6.0});

    for (int imageIndex = 0; imageIndex < 3; ++imageIndex)
    {
        xjw::ImageData image;
        image.id = static_cast<xjw::ImageId>(imageIndex);
        image.imagePath = "synthetic_" + std::to_string(imageIndex) + ".png";
        for (const std::array<double, 3>& worldPoint : worldPoints)
        {
            double projected[2]{};
            ASSERT_TRUE(cameras[static_cast<std::size_t>(imageIndex)].projectWorldPoint(worldPoint.data(), projected));
            image.keypoints.push_back({static_cast<float>(projected[0]), static_cast<float>(projected[1]), 1.0f});
        }
        image.point3DIds.assign(image.keypoints.size(), xjw::kInvalidPoint3DId);
        reconstruction->addImage(image);
        reconstruction->registerImage(image.id, cameras[static_cast<std::size_t>(imageIndex)]);
    }

    for (std::size_t pointIndex = 0; pointIndex < worldPoints.size(); ++pointIndex)
    {
        xjw::Track track;
        const int observationCount = pointIndex == weakTwoViewPointIndex ? 2 : 3;
        for (int imageIndex = 0; imageIndex < observationCount; ++imageIndex)
        {
            track.elements.push_back({static_cast<xjw::ImageId>(imageIndex), static_cast<xjw::FeatureIdx>(pointIndex)});
        }
        const xjw::Point3DId pointId = reconstruction->addPoint3DWithTrack(worldPoints[pointIndex], track);
        reconstruction->point3D(pointId).error =
            pointIndex == weakTwoViewPointIndex ? 1.2 : (pointIndex == highErrorPointIndex ? 2.0 : 0.2);
    }

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {
        QStringLiteral("synthetic_0.png"), QStringLiteral("synthetic_1.png"), QStringLiteral("synthetic_2.png")};
    input.quality = 2;

    xjw::aerial_triangulation::AerialTriangulationReconstructionResult result;
    result.numRegisteredImages = 3;
    result.numPoints3D = static_cast<int>(worldPoints.size());
    const xjw::aerial_triangulation::SparseQualityReport report =
        xjw::aerial_triangulation::QualityReportWriter::build(input, *reconstruction, result);

    EXPECT_EQ(report.points.size(), 20);
    EXPECT_EQ(report.publishedPointIds.size(), 20u);
    for (const QJsonValue& value : report.points)
    {
        const QJsonArray xyz = value.toObject().value(QStringLiteral("point_xyz")).toArray();
        ASSERT_EQ(xyz.size(), 3);
        EXPECT_LT(std::abs(xyz.at(0).toDouble()), 1.0);
    }

    const QJsonObject cleanup = report.diagnostics.value(QStringLiteral("sparse_point_cleanup")).toObject();
    EXPECT_EQ(cleanup.value(QStringLiteral("removed_by_reprojection")).toInt(), 1);
    EXPECT_EQ(cleanup.value(QStringLiteral("removed_by_track_length")).toInt(), 0);
    EXPECT_EQ(cleanup.value(QStringLiteral("removed_weak_two_view")).toInt(), 1);
    EXPECT_EQ(cleanup.value(QStringLiteral("removed_spatial_outliers")).toInt(), 1);
    EXPECT_EQ(cleanup.value(QStringLiteral("published_points")).toInt(), 20);

    (void)isolatedPointIndex;
}

TEST(AerialTriangulationResultWriterTest, MatureHighQualityNetworkPublishesMultiViewPointsOnly)
{
    constexpr int kCameraCount = 8;
    auto reconstruction = std::make_shared<xjw::SfmReconstruction>();
    std::vector<xjw::FramePinholeCamera> cameras(kCameraCount);
    const std::vector<std::array<double, 3>> worldPoints = {{0.0, 0.0, 5.0}, {0.2, 0.1, 5.0}};

    for (int imageIndex = 0; imageIndex < kCameraCount; ++imageIndex)
    {
        xjw::FramePinholeCamera& camera = cameras[static_cast<std::size_t>(imageIndex)];
        camera.setIntrinsics(500.0, 500.0, 320.0, 240.0);
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                       {static_cast<double>(imageIndex) - 3.5, 0.0, 0.0});

        xjw::ImageData image;
        image.id = static_cast<xjw::ImageId>(imageIndex);
        image.imagePath = "mature_network_" + std::to_string(imageIndex) + ".png";
        for (const std::array<double, 3>& worldPoint : worldPoints)
        {
            double projected[2]{};
            ASSERT_TRUE(camera.projectWorldPoint(worldPoint.data(), projected));
            image.keypoints.push_back({static_cast<float>(projected[0]), static_cast<float>(projected[1]), 1.0f});
        }
        image.point3DIds.assign(image.keypoints.size(), xjw::kInvalidPoint3DId);
        reconstruction->addImage(image);
        reconstruction->registerImage(image.id, camera);
    }

    xjw::Track multiViewTrack;
    for (int imageIndex = 0; imageIndex < kCameraCount; ++imageIndex)
    {
        multiViewTrack.elements.push_back({static_cast<xjw::ImageId>(imageIndex), 0});
    }
    const xjw::Point3DId multiViewPoint = reconstruction->addPoint3DWithTrack(worldPoints[0], multiViewTrack);
    reconstruction->point3D(multiViewPoint).error = 0.2;

    xjw::Track twoViewTrack;
    twoViewTrack.elements = {{0, 1}, {7, 1}};
    const xjw::Point3DId twoViewPoint = reconstruction->addPoint3DWithTrack(worldPoints[1], twoViewTrack);
    reconstruction->point3D(twoViewPoint).error = 0.2;

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.quality = 2;
    for (int imageIndex = 0; imageIndex < kCameraCount; ++imageIndex)
    {
        input.images.push_back(QStringLiteral("mature_network_%1.png").arg(imageIndex));
    }

    xjw::aerial_triangulation::AerialTriangulationReconstructionResult result;
    result.numRegisteredImages = kCameraCount;
    result.numPoints3D = 2;
    const xjw::aerial_triangulation::SparseQualityReport report =
        xjw::aerial_triangulation::QualityReportWriter::build(input, *reconstruction, result);

    ASSERT_EQ(report.points.size(), 1);
    ASSERT_EQ(report.publishedPointIds.size(), 1u);
    EXPECT_EQ(report.publishedPointIds.front(), multiViewPoint);
    const QJsonObject cleanup = report.diagnostics.value(QStringLiteral("sparse_point_cleanup")).toObject();
    EXPECT_EQ(cleanup.value(QStringLiteral("removed_by_track_length")).toInt(), 1);
    const QJsonObject policy = cleanup.value(QStringLiteral("policy")).toObject();
    EXPECT_EQ(policy.value(QStringLiteral("min_track_length")).toInt(), 3);
    EXPECT_DOUBLE_EQ(policy.value(QStringLiteral("max_reprojection_error_px")).toDouble(), 1.2);
    EXPECT_DOUBLE_EQ(policy.value(QStringLiteral("min_triangulation_angle_deg")).toDouble(), 7.5);
    EXPECT_DOUBLE_EQ(policy.value(QStringLiteral("max_reconstruction_uncertainty")).toDouble(), 30.0);
    EXPECT_EQ(policy.value(QStringLiteral("spatial_max_track_length")).toInt(), 3);
}
