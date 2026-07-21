#include "reconstruction/SfmAttemptRunner.h"
#include "reconstruction/MarkerPriorLoader.h"

#include "Camera.h"
#include "io/MarkerSetStore.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QImage>

#include <array>
#include <vector>

namespace
{

void writeJson(const QString &path, const QJsonObject &object)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(QJsonDocument(object).toJson(QJsonDocument::Compact));
}

QJsonObject makeKnownPoseTiePoints(const QString &imageA,
                                   const QString &imageB,
                                   const xjw::Camera &cameraA,
                                   const xjw::Camera &cameraB)
{
    QJsonArray tracks;
    int featureIndex = 0;
    for (int row = -2; row <= 2; ++row)
    {
        for (int column = -3; column <= 3; ++column)
        {
            const std::array<double, 3> point{
                column * 0.18,
                row * 0.16,
                5.0 + 0.08 * ((column + row + 8) % 3)};
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
                                    QJsonArray{observation(0, 100, 10.0, 20.0),
                                               observation(1, 300, 11.0, 20.5)}}},
                       QJsonObject{{QStringLiteral("confidence"), 0.8},
                                   {QStringLiteral("observations"),
                                    QJsonArray{observation(0, 900, 30.0, 40.0),
                                               observation(1, 700, 31.0, 40.5)}}},
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
    ASSERT_EQ(graph.matchPairs.size(), 1u);
    EXPECT_EQ(graph.matchPairs.front().matches.size(), 2u);
    EXPECT_EQ(graph.trackCount, 2);
}

TEST(SfmAttemptRunnerTest, RejectsTiePointFileFromAnotherImageSet)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("latest_tie_points.json"));
    writeJson(tiePointPath,
              QJsonObject{{QStringLiteral("format"), QStringLiteral("plascan_tie_points")},
                          {QStringLiteral("format_version"), 1},
                          {QStringLiteral("images"),
                           QJsonArray{
                               QJsonObject{{QStringLiteral("image_id"), 0},
                                           {QStringLiteral("path"), QStringLiteral("old_a.png")}},
                               QJsonObject{{QStringLiteral("image_id"), 1},
                                           {QStringLiteral("path"), QStringLiteral("old_b.png")}},
                           }},
                          {QStringLiteral("tracks"), QJsonArray{}}});

    xjw::aerial_triangulation::PreparedTiePointGraph graph;
    QString errorMessage;
    EXPECT_FALSE(xjw::aerial_triangulation::SfmAttemptRunner::readTiePointGraph(
        tiePointPath,
        QStringList{QStringLiteral("new_a.png"), QStringLiteral("new_b.png")},
        &graph,
        &errorMessage));
    EXPECT_TRUE(errorMessage.contains(QStringLiteral("影像集合")));
}

TEST(SfmAttemptRunnerTest, LoadsMarkerTracksAndScaleBarsFromProjectSidecar)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    const QString markerPath = QDir(tempDir.path()).filePath(QStringLiteral("markers.json"));

    xjw::control_points::MarkerSet markerSet;
    const auto addMarker = [&](const QString &label, double xOffset)
    {
        const xjw::control_points::MarkerId markerId = markerSet.addMarker(
            label, xjw::control_points::MarkerRole::ControlPoint);
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
    markerSet.addScaleBar(QStringLiteral("scale"),
                          firstMarker,
                          secondMarker,
                          1.0,
                          0.005,
                          xjw::control_points::ScaleBarRole::Control);
    const xjw::control_points::MarkerSetIoResult saved =
        xjw::control_points::MarkerSetStore(markerPath).save(markerSet);
    ASSERT_TRUE(saved.ok) << qPrintable(saved.error);

    QMap<QString, xjw::ImageId> imageIdByPath;
    imageIdByPath.insert(QDir::cleanPath(QFileInfo(imageA).absoluteFilePath()), 5);
    imageIdByPath.insert(QDir::cleanPath(QFileInfo(imageB).absoluteFilePath()), 8);

    const xjw::aerial_triangulation::MarkerPriorLoadResult loaded =
        xjw::aerial_triangulation::MarkerPriorLoader::load(
            markerPath, QJsonObject(), imageIdByPath);

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

    xjw::Camera cameraA;
    cameraA.setIntrinsics(700.0, 700.0, 320.0, 240.0);
    cameraA.setPose({1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0},
                    {-0.5, 0.0, 0.0});
    xjw::Camera cameraB;
    cameraB.setIntrinsics(700.0, 700.0, 320.0, 240.0);
    cameraB.setPose({1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0},
                    {0.5, 0.0, 0.0});

    const QString cameraPathA = QDir(tempDir.path()).filePath(QStringLiteral("a.tsai"));
    const QString cameraPathB = QDir(tempDir.path()).filePath(QStringLiteral("b.tsai"));
    ASSERT_TRUE(cameraA.saveToFile(cameraPathA.toStdString()));
    ASSERT_TRUE(cameraB.saveToFile(cameraPathB.toStdString()));

    const QString tiePointPath = QDir(tempDir.path()).filePath(QStringLiteral("latest_tie_points.json"));
    writeJson(tiePointPath, makeKnownPoseTiePoints(imageA, imageB, cameraA, cameraB));

    const QString markerPath = QDir(tempDir.path()).filePath(QStringLiteral("markers.json"));
    xjw::control_points::MarkerSet markerSet;
    const xjw::control_points::MarkerId markerId = markerSet.addMarker(
        QStringLiteral("manual-tie"), xjw::control_points::MarkerRole::TieMarker);
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
    input.tiePointPath = tiePointPath;
    input.markerSetPath = markerPath;
    input.outputDir = tempDir.path();

    const xjw::aerial_triangulation::SfmAttemptExecutionResult result =
        xjw::aerial_triangulation::SfmAttemptRunner().run(input);

    ASSERT_TRUE(result.result.success) << qPrintable(result.result.errorMessage);
    ASSERT_NE(result.reconstruction, nullptr);
    EXPECT_EQ(result.result.numRegisteredImages, 2);
    EXPECT_GE(result.result.numPoints3D, 20);
    EXPECT_EQ(result.result.sfmDiagnostics.value(QStringLiteral("marker_prior_tracks_loaded")).toInt(), 1);
    EXPECT_GE(result.result.sfmDiagnostics.value(QStringLiteral("prior_tracks_accepted")).toInt(), 1);
}
