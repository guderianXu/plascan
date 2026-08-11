#pragma once

#include <QByteArray>
#include <QString>
#include <QVector>

#include <algorithm>
#include <atomic>
#include <functional>

#include <plapoint/core/point_cloud.h>

#include "TiePointVisualization.h"

using ObjRenderCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using ObjPrepareProgressCallback = std::function<void(int, const QString &)>;

struct ObjRenderPreparationLimits
{
    std::size_t maximumFullMeshVertices = 12'000'000;
    std::size_t maximumFullMeshFaces = 24'000'000;
    std::size_t maximumPreviewPoints = 4'000'000;
    std::size_t previewPointsPerChunk = 262'144;
};

struct ObjPointPreviewChunk
{
    QByteArray vertexData;
    int vertexCount = 0;

    bool isValid() const
    {
        return vertexCount > 0 && !vertexData.isEmpty();
    }
};

struct ObjRenderPreparation
{
    // Static, source-vertex-aligned mesh data used by every non-textured view.
    // Layout: position (3), smooth normal (3), source colour (3).
    QByteArray vertexData;
    int vertexCount = 0;
    int strideBytes = 9 * static_cast<int>(sizeof(float));

    // Valid triangle indices and the corresponding unique undirected edges.
    QByteArray triangleIndexData;
    int triangleIndexCount = 0;
    QByteArray wireframeIndexData;
    int wireframeIndexCount = 0;

    // UV seams cannot share source vertices. Keep a separate expanded stream
    // so switching Texture on or off never replaces the static mesh VBO.
    QByteArray texturedVertexData;
    int texturedVertexCount = 0;
    int texturedStrideBytes = 11 * static_cast<int>(sizeof(float));

    bool hasTexture = false;
    bool hasVertexColors = false;
    bool isPointPreview = false;
    std::size_t sourceVertexCount = 0;
    std::size_t sourceFaceCount = 0;
    QVector<ObjPointPreviewChunk> pointPreviewChunks;
    xjw::gui::tie_points::ScalarRange elevationRange;

    bool isValid() const
    {
        if (isPointPreview)
        {
            return vertexCount > 0 && strideBytes > 0
                && !pointPreviewChunks.isEmpty()
                && std::all_of(pointPreviewChunks.cbegin(),
                               pointPreviewChunks.cend(),
                               [](const ObjPointPreviewChunk &chunk)
                               {
                                   return chunk.isValid();
                               });
        }
        return vertexCount > 0 && triangleIndexCount > 0 && strideBytes > 0
            && !vertexData.isEmpty() && !triangleIndexData.isEmpty();
    }

    bool hasTexturedGeometry() const
    {
        return hasTexture && texturedVertexCount > 0
            && texturedStrideBytes > 0 && !texturedVertexData.isEmpty();
    }
};

ObjRenderPreparation prepareObjRenderData(const ObjRenderCloud &cloud,
                                          bool textureImageAvailable,
                                          const std::atomic_bool *cancellationFlag = nullptr,
                                          const ObjPrepareProgressCallback &progress = {},
                                          const ObjRenderPreparationLimits &limits = {});
