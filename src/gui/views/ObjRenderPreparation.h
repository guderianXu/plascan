#pragma once

#include <QByteArray>

#include <atomic>

#include <plapoint/core/point_cloud.h>

#include "TiePointVisualization.h"

using ObjRenderCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

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
    xjw::gui::tie_points::ScalarRange elevationRange;

    bool isValid() const
    {
        return vertexCount > 0 && triangleIndexCount > 0
            && strideBytes > 0 && !vertexData.isEmpty()
            && !triangleIndexData.isEmpty();
    }

    bool hasTexturedGeometry() const
    {
        return hasTexture && texturedVertexCount > 0
            && texturedStrideBytes > 0 && !texturedVertexData.isEmpty();
    }
};

ObjRenderPreparation prepareObjRenderData(const ObjRenderCloud &cloud,
                                          bool textureImageAvailable,
                                          const std::atomic_bool *cancellationFlag = nullptr);
