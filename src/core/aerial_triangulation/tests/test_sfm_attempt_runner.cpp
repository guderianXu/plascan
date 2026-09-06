#include "reconstruction/SfmAttemptRunner.h"
#include "reconstruction/MarkerPriorLoader.h"

#include "FramePinholeCamera.h"
#include "io/MarkerSetStore.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QImage>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace
{

    void writeJson(const QString& path, const QJsonObject& object)
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
        file.write(QJsonDocument(object).toJson(QJsonDocument::Compact));
    }

    QJsonObject makeKnownPoseTiePoints(const QString& imageA,
                                       const QString& imageB,
                                       const xjw::FramePinholeCamera& cameraA,
                                       const xjw::FramePinholeCamera& cameraB)
    {
        QJsonArray tracks;
        int featureIndex = 0;
        for (int row = -2; row <= 2; ++row)
        {
            for (int column = -3; column <= 3; ++column)
            {
                const std::array<double, 3> point{column * 0.18, row * 0.16, 5.0 + 0.08 * ((column + row + 8) % 3)};
                double pixelA[2]{};
                double pixelB[2]{};
                EXPECT_TRUE(cameraA.projectWorldPoint(point.data(), pixelA));
                EXPECT_TRUE(cameraB.projectWorldPoint(point.data(), pixelB));

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

} // namespace

TEST(SfmAttemptRunnerTest, ReadsPersistedTiePointTracksIntoCompactObservationGraph)
{
    QTemporaryDir tempDir;
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
    EXPECT_EQ(graph.keypointsByImage.value(0).size(), 2u);
    EXPECT_EQ(graph.keypointsByImage.value(1).size(), 2u);
    EXPECT_FLOAT_EQ(graph.keypointsByImage.value(0).front().scale, 1.0f);
    ASSERT_EQ(graph.matchPairs.size(), 1u);
    EXPECT_EQ(graph.matchPairs.front().matches.size(), 2u);
    EXPECT_EQ(graph.trackCount, 2);
    ASSERT_EQ(graph.tracks.size(), 2u);
    EXPECT_EQ(graph.tracks.front().length(), 2u);
    EXPECT_FALSE(graph.usesRawDirectEdges);
    EXPECT_EQ(graph.directEdgeCount, 0u);
    EXPECT_EQ(graph.synthesizedClosureEdgeCount, 2u);
}

TEST(SfmAttemptRunnerTest, PreservesVersion2DirectEdgesWithoutSynthesizingTrackClosure)
{
    QTemporaryDir tempDir;
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
    ASSERT_FALSE(graph.keypointsByImage.value(0).empty());
    EXPECT_FLOAT_EQ(graph.keypointsByImage.value(0).front().scale, 2.5f);
    ASSERT_EQ(graph.matchPairs.size(), 2u);
    EXPECT_EQ(graph.matchPairs[0].matches.size(), 1u);
    EXPECT_EQ(graph.matchPairs[1].matches.size(), 1u);
    EXPECT_NE(graph.matchPairs[0].imageA, graph.matchPairs[1].imageA);
}

TEST(SfmAttemptRunnerTest, ReadsVersion3CompactObservations)
{
    QTemporaryDir tempDir;
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
                       {QStringLiteral("direct_edges"), QJsonArray{QJsonArray{0, 1}}},
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
    ASSERT_EQ(graph.keypointsByImage.value(0).size(), 1u);
    ASSERT_EQ(graph.keypointsByImage.value(1).size(), 1u);
    EXPECT_FLOAT_EQ(graph.keypointsByImage.value(0).front().x, 12.5f);
    EXPECT_FLOAT_EQ(graph.keypointsByImage.value(0).front().scale, 1.5f);
    EXPECT_FLOAT_EQ(graph.keypointsByImage.value(1).front().scale, 2.5f);
}

TEST(SfmAttemptRunnerTest, RejectsTiePointFileFromAnotherImageSet)
{
    QTemporaryDir tempDir;
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
    QTemporaryDir tempDir;
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
    QTemporaryDir tempDir;
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

    QMap<QString, xjw::ImageId> imageIdByPath;
    imageIdByPath.insert(QDir::cleanPath(QFileInfo(imageA).absoluteFilePath()), 5);
    imageIdByPath.insert(QDir::cleanPath(QFileInfo(imageB).absoluteFilePath()), 8);

    const xjw::aerial_triangulation::MarkerPriorLoadResult loaded =
        xjw::aerial_triangulation::MarkerPriorLoader::load(markerPath, QJsonObject(), imageIdByPath);

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

TEST(SfmAttemptRunnerTest, RunsKnownPoseSfmFromPreparedTiePointGraph)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    ASSERT_TRUE(QImage(640, 480, QImage::Format_Grayscale8).save(imageA));
    ASSERT_TRUE(QImage(640, 480, QImage::Format_Grayscale8).save(imageB));

    xjw::FramePinholeCamera cameraA;
    cameraA.setIntrinsics(700.0, 700.0, 320.0, 240.0);
    cameraA.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {-0.5, 0.0, 0.0});
    xjw::FramePinholeCamera cameraB;
    cameraB.setIntrinsics(700.0, 700.0, 320.0, 240.0);
    cameraB.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {0.5, 0.0, 0.0});

    const QString cameraPathA = QDir(tempDir.path()).filePath(QStringLiteral("a.tsai"));
    const QString cameraPathB = QDir(tempDir.path()).filePath(QStringLiteral("b.tsai"));
    ASSERT_TRUE(cameraA.saveToFile(cameraPathA.toStdString()));
    ASSERT_TRUE(cameraB.saveToFile(cameraPathB.toStdString()));

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
    double markerPixelA[2]{};
    double markerPixelB[2]{};
    ASSERT_TRUE(cameraA.projectWorldPoint(markerPoint.data(), markerPixelA));
    ASSERT_TRUE(cameraB.projectWorldPoint(markerPoint.data(), markerPixelB));
    xjw::control_points::MarkerProjection markerProjectionA;
    markerProjectionA.imageId = QStringLiteral("image-a");
    markerProjectionA.imagePathSnapshot = imageA;
    markerProjectionA.xy = QPointF(markerPixelA[0], markerPixelA[1]);
    markerProjectionA.state = xjw::control_points::ProjectionState::ManualPinned;
    markerSet.upsertProjection(markerId, markerProjectionA);
    xjw::control_points::MarkerProjection markerProjectionB;
    markerProjectionB.imageId = QStringLiteral("image-b");
    markerProjectionB.imagePathSnapshot = imageB;
    markerProjectionB.xy = QPointF(markerPixelB[0], markerPixelB[1]);
    markerProjectionB.state = xjw::control_points::ProjectionState::ManualPinned;
    markerSet.upsertProjection(markerId, markerProjectionB);
    const xjw::control_points::MarkerSetIoResult markerSaved =
        xjw::control_points::MarkerSetStore(markerPath).save(markerSet);
    ASSERT_TRUE(markerSaved.ok) << qPrintable(markerSaved.error);

    xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
    input.images = {imageA, imageB};
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
