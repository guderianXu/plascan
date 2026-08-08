#include "SceneGeometryPreparation.h"
#include "PointCloudEditPreparation.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

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
