#pragma once

#include <QByteArray>

#include <plapoint/core/point_cloud.h>

using ObjRenderCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

struct ObjRenderPreparation
{
    QByteArray vertexData;
    int vertexCount = 0;
    int strideBytes = 0;
    bool hasTexture = false;

    bool isValid() const
    {
        return vertexCount > 0 && strideBytes > 0 && !vertexData.isEmpty();
    }
};

ObjRenderPreparation prepareObjRenderData(const ObjRenderCloud &cloud,
                                          bool textureImageAvailable);
