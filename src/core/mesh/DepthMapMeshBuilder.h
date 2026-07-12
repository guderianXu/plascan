#pragma once

#include "MeshTypes.h"
#include "PositiveDepthCameraModel.h"

#include <QString>
#include <QVector>

#include <functional>

namespace xjw::mesh
{

struct DepthFrameArtifact
{
    int refIndex = -1;
    QString refImage;
    QString depthPath;
    QString confidencePath;
    QString previewPath;
    QString validMaskPath;
    int gridWidth = 0;
    int gridHeight = 0;
    PositiveDepthCameraModel cameraModel;
    bool hasCameraModel = false;
};

struct DepthMapVisualHullResult
{
    bool applicable = false;
    bool ok = false;
    int usableViewCount = 0;
    QString message;
    TriMesh mesh;
};

class DepthMapMeshBuilder
{
public:
    static QVector<DepthFrameArtifact> discoverDepthFrames(const QString &sourcePath);
    static DepthMapVisualHullResult buildVisualHull(
        const QString &sourcePath,
        int resolution,
        const std::function<void(const QString &, int)> &progress = {});
    static QString resolveReusableDenseCloud(const QString &sourcePath, QString *errorMessage = nullptr);
};

} // namespace xjw::mesh
