#include "SceneGeometryPreparation.h"
#include "PointCloudEditPreparation.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

#include <QSet>

namespace
{

SceneRenderCloud makePointCloud(const std::vector<QVector3D> &points)
{
    SceneRenderCloud cloud(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const auto row = static_cast<plamatrix::Index>(index);
        cloud.points()(row, 0) = points[index].x();
        cloud.points()(row, 1) = points[index].y();
        cloud.points()(row, 2) = points[index].z();
    }
    return cloud;
}

TEST(SceneGeometryPreparationTest, ComputesReusableCloudSummary)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(-2.0f, -1.0f, 3.0f),
        QVector3D(2.0f, 1.0f, 7.0f),
        QVector3D(0.0f, 0.0f, 5.0f)});

    const CloudSpatialSummary summary = prepareCloudSpatialSummary(cloud);

    ASSERT_TRUE(summary.valid);
    EXPECT_EQ(summary.sourcePointCount, 3U);
    EXPECT_EQ(summary.validPointCount, 3U);
    EXPECT_EQ(summary.center, QVector3D(0.0f, 0.0f, 5.0f));
    EXPECT_EQ(summary.aabbMinimum, QVector3D(-2.0f, -1.0f, 3.0f));
    EXPECT_EQ(summary.aabbMaximum, QVector3D(2.0f, 1.0f, 7.0f));
    EXPECT_EQ(summary.boxLineVertices.size(), 24);
    EXPECT_DOUBLE_EQ(summary.elevationRange.minimum, 3.0);
    EXPECT_DOUBLE_EQ(summary.elevationRange.maximum, 7.0);
    EXPECT_GT(summary.p95Radius, 0.0f);
}

TEST(SceneGeometryPreparationTest, SpatialSummaryWeightsOnlyFinitePoints)
{
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(-1.0f, 0.0f, 2.0f),
        QVector3D(infinity, 0.0f, 0.0f),
        QVector3D(nan, 0.0f, 0.0f),
        QVector3D(3.0f, 0.0f, 6.0f)});

    const CloudSpatialSummary summary = prepareCloudSpatialSummary(cloud);

    ASSERT_TRUE(summary.valid);
    EXPECT_EQ(summary.sourcePointCount, 4U);
    EXPECT_EQ(summary.validPointCount, 2U);
    EXPECT_EQ(summary.center, QVector3D(1.0f, 0.0f, 4.0f));
    EXPECT_GT(summary.p95Radius, 2.0f);
}

TEST(SceneGeometryPreparationTest, LargeFiniteCoordinatesKeepFiniteCenter)
{
    const float large = std::numeric_limits<float>::max() * 0.5f;
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(large, large, large),
        QVector3D(large, large, large),
        QVector3D(large, large, large),
        QVector3D(large, large, large)});

    const CloudSpatialSummary summary = prepareCloudSpatialSummary(cloud);

    ASSERT_TRUE(summary.valid);
    EXPECT_TRUE(std::isfinite(summary.center.x()));
    EXPECT_TRUE(std::isfinite(summary.center.y()));
    EXPECT_TRUE(std::isfinite(summary.center.z()));
    EXPECT_EQ(summary.center, QVector3D(large, large, large));
}

TEST(SceneGeometryPreparationTest, KeepsBasePointDataSeparateFromImageCounts)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(0.0f, 0.0f, 1.0f),
        QVector3D(1.0f, 0.0f, 2.0f),
        QVector3D(0.0f, 1.0f, 3.0f)});
    const QVector<int> counts{8, 3, 12};

    const PointRenderPreparation prepared = preparePointRenderData(cloud, counts);

    ASSERT_TRUE(prepared.isValid());
    EXPECT_EQ(prepared.vertexData.size(), 3 * 9 * int(sizeof(float)));
    EXPECT_EQ(prepared.scalarData.size(), 3 * int(sizeof(float)));
    float scalar_values[3]{};
    std::memcpy(scalar_values, prepared.scalarData.constData(), sizeof(scalar_values));
    EXPECT_FLOAT_EQ(scalar_values[0], 8.0f);
    EXPECT_FLOAT_EQ(scalar_values[1], 3.0f);
    EXPECT_FLOAT_EQ(scalar_values[2], 12.0f);
}

TEST(SceneGeometryPreparationTest, BuildsBoundedSpatialChunksWithStableSourceMapping)
{
    std::vector<QVector3D> points;
    points.reserve(512);
    QVector<int> counts;
    counts.reserve(512);
    for (int index = 0; index < 512; ++index)
    {
        points.emplace_back(float(index % 16) / 8.0f - 1.0f,
                            float((index / 16) % 16) / 8.0f - 1.0f,
                            float(index / 256) * 0.25f);
        counts.push_back(index);
    }
    const PointRenderPreparation prepared = preparePointRenderData(
        makePointCloud(points), counts);
    const QVector<PointRenderChunkPreparation> chunks = preparePointRenderChunks(
        prepared.vertexData,
        9 * int(sizeof(float)),
        prepared.scalarData,
        prepared.spatialSummary,
        64,
        128);

    ASSERT_GT(chunks.size(), 1);
    QSet<PointVertexIndex> sourceIndices;
    int totalPointCount = 0;
    for (const PointRenderChunkPreparation &chunk : chunks)
    {
        ASSERT_TRUE(chunk.isValid());
        EXPECT_LE(chunk.pointCount, 64);
        EXPECT_GE(chunk.radius, 0.0f);
        totalPointCount += chunk.pointCount;
        for (const PointVertexIndex sourceIndex : chunk.sourceIndices)
        {
            EXPECT_FALSE(sourceIndices.contains(sourceIndex));
            sourceIndices.insert(sourceIndex);
        }
    }
    EXPECT_EQ(totalPointCount, 512);
    EXPECT_EQ(sourceIndices.size(), 512);
}

TEST(SceneGeometryPreparationTest, CullsOffscreenChunksAndHonorsPointBudget)
{
    auto makeChunk = [](float centerX, int pointCount)
    {
        PointRenderChunkPreparation chunk;
        chunk.pointCount = pointCount;
        chunk.sourceIndices.resize(pointCount);
        chunk.aabbMinimum = QVector3D(centerX - 0.25f, -0.25f, 0.0f);
        chunk.aabbMaximum = QVector3D(centerX + 0.25f, 0.25f, 0.0f);
        chunk.center = (chunk.aabbMinimum + chunk.aabbMaximum) * 0.5f;
        chunk.radius = 0.4f;
        for (int index = 0; index < pointCount; ++index)
        {
            chunk.sourceIndices[index] = static_cast<PointVertexIndex>(index);
        }
        return chunk;
    };
    const QVector<PointRenderChunkPreparation> chunks{
        makeChunk(0.0f, 1'000),
        makeChunk(4.0f, 1'000)};
    QMatrix4x4 clipMatrix;
    clipMatrix.setToIdentity();

    const PointRenderPlan plan = planPointRenderChunks(
        chunks, clipMatrix, QSize(1000, 800), 2.0f, 250);

    ASSERT_EQ(plan.drawCounts.size(), 2);
    EXPECT_GT(plan.drawCounts.at(0), 0);
    EXPECT_EQ(plan.drawCounts.at(1), 0);
    EXPECT_EQ(plan.visibleChunkCount, 1);
    EXPECT_LE(plan.visiblePointCount, 250);
}

TEST(SceneGeometryPreparationTest, SelectsFromImmutablePreparedVertices)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(-0.8f, 0.0f, 0.0f),
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(0.8f, 0.0f, 0.0f)});
    const PointRenderPreparation prepared = preparePointRenderData(cloud);
    QMatrix4x4 identity;
    identity.setToIdentity();

    const std::vector<PointVertexIndex> selected = selectPointVertexIndices(
        prepared.vertexData,
        9 * int(sizeof(float)),
        QRect(40, 40, 20, 20),
        identity,
        QSize(100, 100),
        QPointF());

    ASSERT_EQ(selected.size(), 1U);
    EXPECT_EQ(selected.front(), 1U);
}

TEST(SceneGeometryPreparationTest, CompactsSelectedVertexAndScalarDataInSourceOrder)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(-0.2f, 0.0f, -0.5f),
        QVector3D(0.8f, 0.0f, 0.0f),
        QVector3D(0.2f, 0.0f, 0.5f)});
    const PointRenderPreparation prepared = preparePointRenderData(cloud, {7, 13, 29});
    QMatrix4x4 identity;
    identity.setToIdentity();
    constexpr int stride_bytes = 9 * int(sizeof(float));

    const PointSelectionPreparation selection = preparePointSelection(
        prepared.vertexData,
        stride_bytes,
        prepared.scalarData,
        QRect(35, 40, 31, 20),
        identity,
        QSize(100, 100),
        QPointF());

    ASSERT_TRUE(selection.isValid());
    ASSERT_EQ(selection.indices, (std::vector<PointVertexIndex>{0U, 2U}));
    ASSERT_EQ(selection.pointCount, 2);
    ASSERT_EQ(selection.vertexData.size(), 2 * stride_bytes);
    EXPECT_EQ(std::memcmp(selection.vertexData.constData(),
                          prepared.vertexData.constData(),
                          stride_bytes),
              0);
    EXPECT_EQ(std::memcmp(selection.vertexData.constData() + stride_bytes,
                          prepared.vertexData.constData() + 2 * stride_bytes,
                          stride_bytes),
              0);

    float compact_scalars[2]{};
    std::memcpy(compact_scalars,
                selection.scalarData.constData(),
                sizeof(compact_scalars));
    EXPECT_FLOAT_EQ(compact_scalars[0], 7.0f);
    EXPECT_FLOAT_EQ(compact_scalars[1], 29.0f);
}

TEST(SceneGeometryPreparationTest, KeepsIndicesWithoutDuplicatingOversizedSelection)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(-0.2f, 0.0f, -0.5f),
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(0.2f, 0.0f, 0.5f)});
    const PointRenderPreparation prepared = preparePointRenderData(cloud, {7, 13, 29});
    QMatrix4x4 identity;
    identity.setToIdentity();

    const PointSelectionPreparation selection = preparePointSelection(
        prepared.vertexData,
        9 * int(sizeof(float)),
        prepared.scalarData,
        QRect(0, 0, 100, 100),
        identity,
        QSize(100, 100),
        QPointF(),
        2U);

    ASSERT_EQ(selection.indices.size(), 3U);
    EXPECT_EQ(selection.pointCount, 0);
    EXPECT_TRUE(selection.vertexData.isEmpty());
    EXPECT_TRUE(selection.scalarData.isEmpty());
}

TEST(SceneGeometryPreparationTest, CompactsQualityCandidatesByOriginalPointIndex)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(10.0f, 0.0f, 0.0f),
        QVector3D(20.0f, 0.0f, 0.0f),
        QVector3D(30.0f, 0.0f, 0.0f),
        QVector3D(40.0f, 0.0f, 0.0f)});
    const PointRenderPreparation prepared = preparePointRenderData(
        cloud, {2, 4, 6, 8});
    constexpr int stride_bytes = 9 * int(sizeof(float));

    const PointSelectionPreparation selection = prepareIndexedPointSelection(
        prepared.vertexData,
        stride_bytes,
        prepared.scalarData,
        {1U, 3U});

    ASSERT_TRUE(selection.isValid());
    EXPECT_EQ(selection.indices, (std::vector<PointVertexIndex>{1U, 3U}));
    const float *vertices = reinterpret_cast<const float *>(
        selection.vertexData.constData());
    EXPECT_FLOAT_EQ(vertices[0], 20.0f);
    EXPECT_FLOAT_EQ(vertices[9], 40.0f);
    const float *scalars = reinterpret_cast<const float *>(
        selection.scalarData.constData());
    EXPECT_FLOAT_EQ(scalars[0], 4.0f);
    EXPECT_FLOAT_EQ(scalars[1], 8.0f);
}

TEST(SceneGeometryPreparationTest, RejectsInvalidOrCancelledQualityCandidateIndices)
{
    const PointRenderPreparation prepared = preparePointRenderData(
        makePointCloud({QVector3D(), QVector3D(1.0f, 0.0f, 0.0f)}));
    constexpr int stride_bytes = 9 * int(sizeof(float));
    EXPECT_FALSE(prepareIndexedPointSelection(
        prepared.vertexData,
        stride_bytes,
        prepared.scalarData,
        {1U, 0U}).isValid());
    EXPECT_FALSE(prepareIndexedPointSelection(
        prepared.vertexData,
        stride_bytes,
        prepared.scalarData,
        {2U}).isValid());
    std::atomic_bool cancelled{true};
    EXPECT_FALSE(prepareIndexedPointSelection(
        prepared.vertexData,
        stride_bytes,
        prepared.scalarData,
        {0U},
        &cancelled).isValid());
}

TEST(SceneGeometryPreparationTest, IgnoresPointsOutsideClipDepthOrWithNonFiniteClip)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(0.0f, 0.0f, 2.0f),
        QVector3D(0.0f, 0.0f, -2.0f),
        QVector3D(0.2f, 0.0f, 0.0f)});
    const PointRenderPreparation prepared = preparePointRenderData(cloud);
    QByteArray vertex_data = prepared.vertexData;
    float infinity = std::numeric_limits<float>::infinity();
    std::memcpy(vertex_data.data() + 3 * 9 * int(sizeof(float)),
                &infinity,
                sizeof(float));
    QMatrix4x4 identity;
    identity.setToIdentity();

    const std::vector<PointVertexIndex> selected = selectPointVertexIndices(
        vertex_data,
        9 * int(sizeof(float)),
        QRect(0, 0, 100, 100),
        identity,
        QSize(100, 100),
        QPointF());

    ASSERT_EQ(selected, (std::vector<PointVertexIndex>{0U}));
}

TEST(SceneGeometryPreparationTest, UsesContinuousSelectionRectangleEdges)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(-0.5f, 0.0f, 0.0f),
        QVector3D(0.5f, 0.0f, 0.0f),
        QVector3D(0.0f, 0.5f, 0.0f),
        QVector3D(0.0f, -0.5f, 0.0f),
        QVector3D(0.51f, 0.0f, 0.0f)});
    const PointRenderPreparation prepared = preparePointRenderData(cloud);
    QMatrix4x4 identity;
    identity.setToIdentity();

    const std::vector<PointVertexIndex> selected = selectPointVertexIndices(
        prepared.vertexData,
        9 * int(sizeof(float)),
        QRect(25, 25, 50, 50),
        identity,
        QSize(100, 100),
        QPointF());

    ASSERT_EQ(selected, (std::vector<PointVertexIndex>{0U, 1U, 2U, 3U}));
}

TEST(SceneGeometryPreparationTest, SelectsPointsInsideEllipseInsteadOfItsBoundingCorners)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(0.5f, 0.5f, 0.0f),
        QVector3D(-0.5f, 0.0f, 0.0f)});
    const PointRenderPreparation prepared = preparePointRenderData(cloud);
    QMatrix4x4 identity;
    identity.setToIdentity();
    ScreenSelectionRegion region;
    region.shape = ScreenSelectionShape::Ellipse;
    region.bounds = QRectF(20.0, 20.0, 60.0, 60.0);

    const std::vector<PointVertexIndex> selected = selectPointVertexIndices(
        prepared.vertexData,
        9 * int(sizeof(float)),
        region,
        identity,
        QSize(100, 100),
        QPointF());

    EXPECT_EQ(selected, (std::vector<PointVertexIndex>{0U, 2U}));
}

TEST(SceneGeometryPreparationTest, SelectsPointsInsideFreehandPolygon)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(0.7f, 0.0f, 0.0f),
        QVector3D(0.0f, 0.7f, 0.0f)});
    const PointRenderPreparation prepared = preparePointRenderData(cloud);
    QMatrix4x4 identity;
    identity.setToIdentity();
    ScreenSelectionRegion region;
    region.shape = ScreenSelectionShape::Polygon;
    region.polygon = QPolygonF{
        QPointF(20.0, 80.0), QPointF(50.0, 20.0), QPointF(80.0, 80.0)};
    region.bounds = region.polygon.boundingRect();

    const PointSelectionPreparation selection = preparePointSelection(
        prepared.vertexData,
        9 * int(sizeof(float)),
        prepared.scalarData,
        region,
        identity,
        QSize(100, 100),
        QPointF());

    ASSERT_TRUE(selection.isValid());
    EXPECT_EQ(selection.indices, (std::vector<PointVertexIndex>{0U}));
}

TEST(SceneGeometryPreparationTest, CancelledSelectionNeverReturnsPartialData)
{
    const SceneRenderCloud cloud = makePointCloud({
        QVector3D(-0.5f, 0.0f, 0.0f),
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(0.5f, 0.0f, 0.0f)});
    const PointRenderPreparation prepared = preparePointRenderData(cloud, {2, 4, 6});
    QMatrix4x4 identity;
    identity.setToIdentity();
    std::atomic_bool cancelled{true};

    EXPECT_FALSE(prepareCloudSpatialSummary(cloud, &cancelled).valid);
    EXPECT_FALSE(preparePointRenderData(cloud, {}, &cancelled).isValid());

    const PointSelectionPreparation selection = preparePointSelection(
        prepared.vertexData,
        9 * int(sizeof(float)),
        prepared.scalarData,
        QRect(0, 0, 100, 100),
        identity,
        QSize(100, 100),
        QPointF(),
        std::numeric_limits<std::size_t>::max(),
        &cancelled);

    EXPECT_TRUE(selection.indices.empty());
    EXPECT_TRUE(selection.vertexData.isEmpty());
    EXPECT_TRUE(selection.scalarData.isEmpty());
    EXPECT_EQ(selection.pointCount, 0);
}

TEST(SceneGeometryPreparationTest, PruneUndoStoresOnlyRemovedRowsAndRestoresOrder)
{
    auto cloud = std::make_shared<SceneRenderCloud>(makePointCloud({
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(1.0f, 0.0f, 0.0f),
        QVector3D(2.0f, 0.0f, 0.0f),
        QVector3D(3.0f, 0.0f, 0.0f)}));
    const QVector<int> counts{2, 4, 6, 8};

    PointCloudEditResult filtered = filterPointCloudWithDelta(
        cloud, {1U, 3U}, counts);

    ASSERT_TRUE(filtered.isValid());
    ASSERT_TRUE(filtered.undo.isValid());
    EXPECT_EQ(filtered.cloud->size(), 2U);
    EXPECT_EQ(filtered.undo.removedCloud->size(), 2U);
    EXPECT_FLOAT_EQ(filtered.cloud->points()(1, 0), 2.0f);

    PointCloudEditResult restored = restorePointCloudFromDelta(
        filtered.cloud, std::move(filtered.undo), filtered.imageCounts);

    ASSERT_TRUE(restored.isValid());
    ASSERT_EQ(restored.cloud->size(), 4U);
    ASSERT_EQ(restored.imageCounts, counts);
    for (int index = 0; index < 4; ++index)
    {
        EXPECT_FLOAT_EQ(restored.cloud->points()(index, 0), float(index));
    }
}

TEST(SceneGeometryPreparationTest, PointCloudEditsHonorPreCancelledWork)
{
    auto source = std::make_shared<SceneRenderCloud>(makePointCloud({
        QVector3D(0.0f, 0.0f, 0.0f),
        QVector3D(1.0f, 0.0f, 0.0f)}));
    std::atomic_bool cancelled{true};

    PointCloudEditResult filtered = filterPointCloudWithDelta(
        source, {1U}, {}, &cancelled);
    EXPECT_FALSE(filtered.isValid());

    auto filtered_cloud = std::make_shared<SceneRenderCloud>(makePointCloud({
        QVector3D(0.0f, 0.0f, 0.0f)}));
    PointCloudEditDelta delta;
    delta.originalPointCount = 2;
    delta.removedIndices = {1U};
    delta.removedCloud = std::make_shared<SceneRenderCloud>(makePointCloud({
        QVector3D(1.0f, 0.0f, 0.0f)}));
    PointCloudEditResult restored = restorePointCloudFromDelta(
        filtered_cloud, std::move(delta), {}, &cancelled);
    EXPECT_FALSE(restored.isValid());
}

TEST(SceneGeometryPreparationTest, PruneUndoPreservesPointAlignedAttributes)
{
    auto cloud = std::make_shared<SceneRenderCloud>(makePointCloud({
        QVector3D(0.0f, 0.0f, 1.0f),
        QVector3D(1.0f, 0.0f, 2.0f),
        QVector3D(2.0f, 0.0f, 3.0f)}));
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(3, 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(3, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(3, 1);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> scalars(3, 1);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture(3, 2);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            colors(row, column) = static_cast<std::uint8_t>(row * 10 + column);
            normals(row, column) = float(row * 3 + column);
        }
        intensities(row, 0) = static_cast<std::uint16_t>(100 + row);
        scalars(row, 0) = float(20 + row);
        texture(row, 0) = float(row) * 0.25f;
        texture(row, 1) = float(row) * 0.5f;
    }
    cloud->setColors(std::move(colors));
    cloud->setNormals(std::move(normals));
    cloud->setIntensities(std::move(intensities));
    cloud->setScalarFields({"quality"}, std::move(scalars));
    cloud->setTextureCoords(std::move(texture));
    cloud->setMaterialLibraryFile("material.mtl");
    cloud->setTextureImageFile("texture.png");

    PointCloudEditResult filtered = filterPointCloudWithDelta(cloud, {1U});
    PointCloudEditResult restored = restorePointCloudFromDelta(
        filtered.cloud, std::move(filtered.undo));

    ASSERT_TRUE(restored.isValid());
    ASSERT_TRUE(restored.cloud->hasColors());
    ASSERT_TRUE(restored.cloud->hasNormals());
    ASSERT_TRUE(restored.cloud->hasIntensities());
    ASSERT_TRUE(restored.cloud->hasScalarField("quality"));
    ASSERT_TRUE(restored.cloud->hasPointAlignedTextureCoords());
    EXPECT_EQ(restored.cloud->colors()->getValue(1, 2), 12);
    EXPECT_FLOAT_EQ(restored.cloud->normals()->getValue(1, 1), 4.0f);
    EXPECT_EQ(restored.cloud->intensities()->getValue(1, 0), 101);
    EXPECT_FLOAT_EQ(restored.cloud->scalarFields()->getValue(1, 0), 21.0f);
    EXPECT_FLOAT_EQ(restored.cloud->textureCoords()->getValue(1, 1), 0.5f);
    EXPECT_EQ(restored.cloud->materialLibraryFile(), "material.mtl");
    EXPECT_EQ(restored.cloud->textureImageFile(), "texture.png");
}

} // namespace
