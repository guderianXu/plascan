#pragma once

#include <QByteArray>
#include <QMatrix4x4>
#include <QPointF>
#include <QRect>
#include <QSize>
#include <QVector>
#include <QVector3D>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <plapoint/core/point_cloud.h>

#include "TiePointVisualization.h"

using SceneRenderCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using PointVertexIndex = std::uint32_t;

struct CloudSpatialSummary
{
    QVector3D center;
    QVector3D aabbMinimum;
    QVector3D aabbMaximum;
    QVector<QVector3D> boxLineVertices;
    xjw::gui::tie_points::ScalarRange elevationRange;
    float p95Radius = 1.0f;
    std::size_t sourcePointCount = 0;
    std::size_t validPointCount = 0;
    bool valid = false;
};

struct PointRenderChunkPreparation
{
    QByteArray vertexData;
    QByteArray scalarData;
    QVector<PointVertexIndex> sourceIndices;
    QVector3D aabbMinimum;
    QVector3D aabbMaximum;
    QVector3D center;
    float radius = 0.0f;
    int pointCount = 0;

    bool isValid() const
    {
        return pointCount > 0
            && sourceIndices.size() == pointCount
            && !vertexData.isEmpty()
            && scalarData.size() == pointCount * int(sizeof(float));
    }
};

struct PointRenderPlan
{
    QVector<int> drawCounts;
    qint64 visiblePointCount = 0;
    int visibleChunkCount = 0;
};

struct PointRenderPreparation
{
    QByteArray vertexData;
    QByteArray scalarData;
    QVector<PointRenderChunkPreparation> chunks;
    CloudSpatialSummary spatialSummary;
    int pointCount = 0;

    bool isValid() const
    {
        return pointCount > 0 && !vertexData.isEmpty() && !scalarData.isEmpty();
    }
};

struct PointSelectionPreparation
{
    std::vector<PointVertexIndex> indices;
    QByteArray vertexData;
    QByteArray scalarData;
    int pointCount = 0;

    bool isValid() const
    {
        return pointCount > 0
            && indices.size() == static_cast<std::size_t>(pointCount)
            && !vertexData.isEmpty()
            && scalarData.size() == pointCount * int(sizeof(float));
    }
};

CloudSpatialSummary prepareCloudSpatialSummary(
    const SceneRenderCloud &cloud,
    const std::atomic_bool *cancellationFlag = nullptr);

PointRenderPreparation preparePointRenderData(
    const SceneRenderCloud &cloud,
    const QVector<int> &imageCounts = {},
    const std::atomic_bool *cancellationFlag = nullptr);

QVector<PointRenderChunkPreparation> preparePointRenderChunks(
    const QByteArray &vertexData,
    int strideBytes,
    const QByteArray &scalarData,
    const CloudSpatialSummary &spatialSummary,
    int maximumPointsPerChunk = 262'144,
    int minimumPointCountForChunking = 524'288,
    const std::atomic_bool *cancellationFlag = nullptr);

PointRenderPlan planPointRenderChunks(
    const QVector<PointRenderChunkPreparation> &chunks,
    const QMatrix4x4 &clipMatrix,
    const QSize &viewportSize,
    float pointDiameterPixels,
    qint64 maximumVisiblePoints = 3'000'000);

QByteArray preparePointScalarData(std::size_t pointCount,
                                  const QVector<int> &imageCounts,
                                  xjw::gui::tie_points::ScalarRange *range = nullptr,
                                  const std::atomic_bool *cancellationFlag = nullptr);

std::vector<PointVertexIndex> selectPointVertexIndices(
    const QByteArray &vertexData,
    int strideBytes,
    const QRect &screenRect,
    const QMatrix4x4 &clipMatrix,
    const QSize &viewportSize,
    const QPointF &sceneOffset,
    const std::atomic_bool *cancellationFlag = nullptr);

PointSelectionPreparation preparePointSelection(
    const QByteArray &vertexData,
    int strideBytes,
    const QByteArray &scalarData,
    const QRect &screenRect,
    const QMatrix4x4 &clipMatrix,
    const QSize &viewportSize,
    const QPointF &sceneOffset,
    std::size_t maximumCompactPointCount =
        std::numeric_limits<std::size_t>::max(),
    const std::atomic_bool *cancellationFlag = nullptr);

PointSelectionPreparation prepareIndexedPointSelection(
    const QByteArray &vertexData,
    int strideBytes,
    const QByteArray &scalarData,
    const std::vector<PointVertexIndex> &indices,
    const std::atomic_bool *cancellationFlag = nullptr);
