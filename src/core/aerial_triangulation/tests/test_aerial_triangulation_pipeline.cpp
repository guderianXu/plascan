#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"
#include "reconstruction/SfmAttemptRunner.h"
#include "workflow/AerialTriangulationPipeline.h"

#include "Camera.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>

#include <algorithm>

namespace
{

void writeKnownPoseTiePoints(const QString &path,
                             const QString &imageA,
                             const QString &imageB,
                             const xjw::Camera &cameraA,
                             const xjw::Camera &cameraB)
{
    QJsonArray tracks;
    for (int featureIndex = 0; featureIndex < 30; ++featureIndex)
    {
        const std::array<double, 3> point{
            (featureIndex % 6 - 2.5) * 0.16,
            (featureIndex / 6 - 2.0) * 0.14,
            5.0 + 0.05 * (featureIndex % 3)};
        double pixelA[2]{};
        double pixelB[2]{};
        ASSERT_TRUE(cameraA.projectWorldPoint(point.data(), pixelA));
        ASSERT_TRUE(cameraB.projectWorldPoint(point.data(), pixelB));
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
    }

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(QJsonDocument(QJsonObject{
        {QStringLiteral("format"), QStringLiteral("plascan_tie_points")},
        {QStringLiteral("format_version"), 1},
        {QStringLiteral("images"),
         QJsonArray{
             QJsonObject{{QStringLiteral("image_id"), 0}, {QStringLiteral("path"), imageA}},
             QJsonObject{{QStringLiteral("image_id"), 1}, {QStringLiteral("path"), imageB}},
         }},
        {QStringLiteral("tracks"), tracks},
    }).toJson(QJsonDocument::Compact));
}

} // namespace

TEST(AerialTriangulationPipelineTest, MissingPreparedTiePointFileFailsWithoutFrontendFallback)
{
    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};
    input.tiePointPath = QStringLiteral("missing/latest_tie_points.json");
    input.outputDir = QStringLiteral("output");

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline().run(input);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("连接点")));
}

TEST(AerialTriangulationPipelineTest, ReplaysBestCoarseFocalCandidateWhenBaseCoverageIsIncomplete)
{
    QVector<double> attemptedScales;
    const auto attemptRunner = [&attemptedScales](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.meanReprojError = 0.6;
        execution.result.numRegisteredImages = 9;
        execution.result.numPoints3D = 500;
        if (std::abs(input.estimatedFocalScale - 0.85) < 1.0e-9)
        {
            execution.result.numRegisteredImages = 16;
            execution.result.numPoints3D = 1200;
            execution.result.meanReprojError = 0.4;
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.adaptiveCameraModelFitting = true;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.numRegisteredImages, 16);
    ASSERT_GE(attemptedScales.size(), 3);
    EXPECT_DOUBLE_EQ(attemptedScales.front(), 1.2);
    EXPECT_DOUBLE_EQ(attemptedScales.back(), 0.85);
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(QStringLiteral("adaptive_focal_scale")).toDouble(),
                     0.85);
}

TEST(AerialTriangulationPipelineTest, SearchesNarrowFieldFocalCandidatesEvenWhenBaseRegistersAllImages)
{
    QVector<double> attemptedScales;
    const auto attemptRunner = [&attemptedScales](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 16;
        execution.result.numPoints3D = 1000;
        execution.result.meanReprojError = 0.7;
        if (std::abs(input.estimatedFocalScale - 5.2) < 1.0e-9)
        {
            execution.result.numPoints3D = 2400;
            execution.result.meanReprojError = 0.4;
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.adaptiveCameraModelFitting = true;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_NE(std::find(attemptedScales.cbegin(), attemptedScales.cend(), 5.2),
              attemptedScales.cend());
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(QStringLiteral("adaptive_focal_scale")).toDouble(),
                     5.2);
}

TEST(AerialTriangulationPipelineTest, InitializesUnknownFocalEvenWhenAdaptiveModelFittingIsDisabled)
{
    QVector<double> attemptedScales;
    QVector<bool> attemptedAdaptiveFlags;
    const auto attemptRunner = [&attemptedScales, &attemptedAdaptiveFlags](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        attemptedScales.append(input.estimatedFocalScale);
        attemptedAdaptiveFlags.append(input.adaptiveCameraModelFitting);

        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 16;
        execution.result.numPoints3D = 900;
        execution.result.meanReprojError = 0.8;
        if (std::abs(input.estimatedFocalScale - 5.2) < 1.0e-9)
        {
            execution.result.numPoints3D = 2400;
            execution.result.meanReprojError = 0.4;
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.adaptiveCameraModelFitting = false;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_NE(std::find(attemptedScales.cbegin(), attemptedScales.cend(), 5.2),
              attemptedScales.cend());
    EXPECT_TRUE(std::all_of(attemptedAdaptiveFlags.cbegin(),
                            attemptedAdaptiveFlags.cend(),
                            [](bool enabled) { return !enabled; }));
    EXPECT_TRUE(result.sfmDiagnostics.value(QStringLiteral("focal_initialization_search")).toBool());
    EXPECT_FALSE(result.sfmDiagnostics.value(QStringLiteral("adaptive_camera_model_fitting")).toBool());
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(QStringLiteral("adaptive_focal_scale")).toDouble(),
                     5.2);
}

TEST(AerialTriangulationPipelineTest, DefaultEstimatedFocalScaleUsesLongestImageDimensionRatio)
{
    const xjw::aerial_triangulation::PreparedAerialTriangulationInput input;

    EXPECT_DOUBLE_EQ(input.estimatedFocalScale, 1.2);
}

TEST(AerialTriangulationPipelineTest, PrefersRigidPhotogrammetricNetworkOverMoreWeakPoints)
{
    const auto sparseQuality = [](int pointCount,
                                  int twoViewTrackCount,
                                  double medianAngleDeg,
                                  double gridCoverage)
    {
        return QJsonObject{
            {QStringLiteral("point_count"), pointCount},
            {QStringLiteral("two_view_track_count"), twoViewTrackCount},
            {QStringLiteral("triangulation_angle"),
             QJsonObject{{QStringLiteral("count"), pointCount},
                         {QStringLiteral("p50"), medianAngleDeg}}},
            {QStringLiteral("observation_grid_coverage"),
             QJsonObject{{QStringLiteral("mean"), gridCoverage}}},
        };
    };
    const auto attemptRunner = [sparseQuality](
                                   const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
    {
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.result.success = true;
        execution.result.numRegisteredImages = 8;
        execution.result.numPoints3D = 400;
        execution.result.meanReprojError = 0.8;
        if (std::abs(input.estimatedFocalScale - 1.2) < 1.0e-9)
        {
            execution.result.numRegisteredImages = 16;
            execution.result.numPoints3D = 1800;
            execution.result.meanReprojError = 0.35;
            execution.result.sfmDiagnostics.insert(
                QStringLiteral("sparse_quality"), sparseQuality(1800, 1500, 2.0, 0.05));
        }
        else if (std::abs(input.estimatedFocalScale - 2.4) < 1.0e-9)
        {
            execution.result.numRegisteredImages = 16;
            execution.result.numPoints3D = 1300;
            execution.result.meanReprojError = 0.55;
            execution.result.sfmDiagnostics.insert(
                QStringLiteral("sparse_quality"), sparseQuality(1300, 300, 12.0, 0.22));
        }
        return execution;
    };
    const auto resultWriter = [](
                                  const xjw::aerial_triangulation::PreparedAerialTriangulationInput &,
                                  xjw::aerial_triangulation::SfmAttemptExecutionResult *,
                                  QString *)
    {
        return true;
    };

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    for (int index = 0; index < 16; ++index)
    {
        input.images.append(QStringLiteral("image_%1.png").arg(index));
    }
    input.adaptiveCameraModelFitting = true;

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline(attemptRunner, resultWriter).run(input);

    ASSERT_TRUE(result.success);
    EXPECT_DOUBLE_EQ(result.sfmDiagnostics.value(QStringLiteral("adaptive_focal_scale")).toDouble(),
                     2.4);
}

TEST(AerialTriangulationPipelineTest, RunsSfmAndWritesPreparedReconstruction)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    ASSERT_TRUE(QImage(640, 480, QImage::Format_Grayscale8).save(imageA));
    ASSERT_TRUE(QImage(640, 480, QImage::Format_Grayscale8).save(imageB));

    xjw::Camera cameraA;
    cameraA.setIntrinsics(700.0, 700.0, 320.0, 240.0);
    cameraA.setPose({1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0},
                    {-0.5, 0.0, 0.0});
    xjw::Camera cameraB = cameraA;
    cameraB.setCameraCenter({0.5, 0.0, 0.0});
    const QString cameraAPath = QDir(tempDir.path()).filePath(QStringLiteral("a.tsai"));
    const QString cameraBPath = QDir(tempDir.path()).filePath(QStringLiteral("b.tsai"));
    ASSERT_TRUE(cameraA.saveToFile(cameraAPath.toStdString()));
    ASSERT_TRUE(cameraB.saveToFile(cameraBPath.toStdString()));

    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("tie_points.json"));
    writeKnownPoseTiePoints(tiePointPath, imageA, imageB, cameraA, cameraB);

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {imageA, imageB};
    input.cameraPaths = {cameraAPath, cameraBPath};
    input.tiePointPath = tiePointPath;
    input.outputDir = QDir(tempDir.path()).filePath(QStringLiteral("output"));

    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult result =
        xjw::aerial_triangulation::AerialTriangulationPipeline().run(input);

    ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.numRegisteredImages, 2);
    EXPECT_GE(result.numPoints3D, 20);
    EXPECT_TRUE(QFile::exists(result.sparseCloudPath));
}
