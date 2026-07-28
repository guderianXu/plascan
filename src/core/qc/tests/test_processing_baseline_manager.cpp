#include "ProcessingBaselineManager.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

namespace
{

xjw::mesh::TriMesh makeGrid(bool rough)
{
    xjw::mesh::TriMesh mesh;
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            xjw::mesh::MeshVertex vertex;
            vertex.x = static_cast<float>(x) * 0.5f;
            vertex.y = static_cast<float>(y) * 0.5f;
            vertex.z = rough && x == 1 && y == 1 ? 0.75f : 0.0f;
            mesh.vertices.push_back(vertex);
        }
    }
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            const int first = y * 3 + x;
            const int second = first + 1;
            const int third = first + 3;
            const int fourth = third + 1;
            mesh.faces.push_back({{first, second, fourth}});
            mesh.faces.push_back({{first, fourth, third}});
        }
    }
    return mesh;
}

QJsonObject inputSnapshot(const QString &cameraRevision)
{
    QJsonObject camera;
    camera.insert(QStringLiteral("revision"), cameraRevision);
    camera.insert(QStringLiteral("fx"), 3310.4);
    camera.insert(QStringLiteral("fy"), 3325.5);
    camera.insert(QStringLiteral("cx"), 316.73);
    camera.insert(QStringLiteral("cy"), 200.55);

    QJsonArray cameras;
    cameras.push_back(camera);
    QJsonArray accepted_frames;
    accepted_frames.push_back(QStringLiteral("dinoSR0001"));
    accepted_frames.push_back(QStringLiteral("dinoSR0002"));

    QJsonObject settings;
    settings.insert(QStringLiteral("tsdf_truncation_voxels"), 6);
    settings.insert(QStringLiteral("minimum_distinct_cameras"), 3);

    QJsonObject snapshot;
    snapshot.insert(QStringLiteral("cameras"), cameras);
    snapshot.insert(QStringLiteral("accepted_depth_frames"), accepted_frames);
    snapshot.insert(QStringLiteral("processing_settings"), settings);
    return snapshot;
}

TEST(ProcessingBaselineManagerTest, MeasuresSurfaceNormalRoughness)
{
    const xjw::qc::ProcessingBaselineMeshMetrics smooth =
        xjw::qc::ProcessingBaselineManager::analyzeMesh(makeGrid(false));
    const xjw::qc::ProcessingBaselineMeshMetrics rough =
        xjw::qc::ProcessingBaselineManager::analyzeMesh(makeGrid(true));

    EXPECT_EQ(smooth.faceCount, 8);
    EXPECT_EQ(smooth.boundaryEdgeCount, 8);
    EXPECT_NEAR(smooth.adjacentNormalAngleMedianDegrees, 0.0, 1.0e-6);
    EXPECT_NEAR(smooth.adjacentNormalAngleP90Degrees, 0.0, 1.0e-6);
    EXPECT_GT(rough.adjacentNormalAngleMedianDegrees, 20.0);
    EXPECT_GT(rough.adjacentNormalAngleP90Degrees, 30.0);
    EXPECT_GT(rough.normalizedSurfaceArea, smooth.normalizedSurfaceArea);
}

TEST(ProcessingBaselineManagerTest, RejectsRoughCandidateAgainstSmoothBaseline)
{
    xjw::qc::ProcessingBaselineThresholds thresholds;
    thresholds.maximumFaceCountRatio = 2.0;
    thresholds.maximumBoundaryEdgeCountRatio = 2.0;
    const QJsonObject inputs = inputSnapshot(QStringLiteral("middlebury-v1"));
    const xjw::qc::ProcessingBaselineDefinition baseline =
        xjw::qc::ProcessingBaselineManager::create(
            QStringLiteral("dino-metashape"),
            QStringLiteral("dino"),
            inputs,
            makeGrid(false),
            thresholds);

    const xjw::qc::ProcessingBaselineComparison comparison =
        xjw::qc::ProcessingBaselineManager::compare(
            baseline, inputs, makeGrid(true));

    EXPECT_TRUE(comparison.inputMatches);
    EXPECT_FALSE(comparison.passed);
    EXPECT_TRUE(comparison.failures.join(QLatin1Char('|')).contains(
        QStringLiteral("法线")));
    EXPECT_TRUE(comparison.report.contains(QStringLiteral("candidate_mesh")));
}

TEST(ProcessingBaselineManagerTest, RejectsInputDriftBeforeQualityComparison)
{
    const QJsonObject baseline_inputs =
        inputSnapshot(QStringLiteral("middlebury-v1"));
    const xjw::qc::ProcessingBaselineDefinition baseline =
        xjw::qc::ProcessingBaselineManager::create(
            QStringLiteral("dino-metashape"),
            QStringLiteral("dino"),
            baseline_inputs,
            makeGrid(false));

    const xjw::qc::ProcessingBaselineComparison comparison =
        xjw::qc::ProcessingBaselineManager::compare(
            baseline,
            inputSnapshot(QStringLiteral("estimated-camera-v2")),
            makeGrid(false));

    EXPECT_FALSE(comparison.inputMatches);
    EXPECT_FALSE(comparison.passed);
    EXPECT_TRUE(comparison.failures.join(QLatin1Char('|')).contains(
        QStringLiteral("输入快照")));
}

TEST(ProcessingBaselineManagerTest, RejectsExcessFaceCount)
{
    const QJsonObject inputs = inputSnapshot(QStringLiteral("middlebury-v1"));
    const xjw::qc::ProcessingBaselineDefinition baseline =
        xjw::qc::ProcessingBaselineManager::create(
            QStringLiteral("dino-metashape"),
            QStringLiteral("dino"),
            inputs,
            makeGrid(false));
    xjw::mesh::TriMesh inflated = makeGrid(false);
    const std::vector<xjw::mesh::Triangle> original_faces = inflated.faces;
    inflated.faces.insert(inflated.faces.end(),
                          original_faces.begin(),
                          original_faces.end());

    const xjw::qc::ProcessingBaselineComparison comparison =
        xjw::qc::ProcessingBaselineManager::compare(
            baseline, inputs, inflated);

    EXPECT_FALSE(comparison.passed);
    EXPECT_TRUE(comparison.failures.join(QLatin1Char('|')).contains(
        QStringLiteral("面数")));
}

TEST(ProcessingBaselineManagerTest, SavesLoadsAndValidatesFingerprint)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const xjw::qc::ProcessingBaselineDefinition baseline =
        xjw::qc::ProcessingBaselineManager::create(
            QStringLiteral("dino-metashape"),
            QStringLiteral("dino"),
            inputSnapshot(QStringLiteral("middlebury-v1")),
            makeGrid(false));
    const QString path =
        QDir(directory.path()).filePath(QStringLiteral("baseline.json"));

    QString error;
    ASSERT_TRUE(xjw::qc::ProcessingBaselineManager::save(
        path, baseline, &error)) << error.toStdString();
    xjw::qc::ProcessingBaselineDefinition loaded;
    ASSERT_TRUE(xjw::qc::ProcessingBaselineManager::load(
        path, &loaded, &error)) << error.toStdString();

    EXPECT_EQ(loaded.name, baseline.name);
    EXPECT_EQ(loaded.sceneType, baseline.sceneType);
    EXPECT_EQ(loaded.inputFingerprintSha256,
              baseline.inputFingerprintSha256);
    EXPECT_EQ(loaded.referenceMesh.faceCount, 8);

    QJsonObject tampered =
        xjw::qc::ProcessingBaselineManager::toJson(baseline);
    QJsonObject tampered_snapshot =
        tampered.value(QStringLiteral("input_snapshot")).toObject();
    tampered_snapshot.insert(QStringLiteral("untracked_change"), true);
    tampered.insert(QStringLiteral("input_snapshot"), tampered_snapshot);
    EXPECT_FALSE(xjw::qc::ProcessingBaselineManager::fromJson(
        tampered, &loaded, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("指纹")));
}

} // namespace
