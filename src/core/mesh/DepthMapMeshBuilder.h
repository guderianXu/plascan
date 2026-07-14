#pragma once

#include "Camera.h"
#include "MeshTypes.h"
#include "VisualHullReconstructor.h"

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
    Camera cameraModel;
    bool hasCameraModel = false;
};

struct DepthMapVisualHullResult
{
    bool applicable = false;
    bool ok = false;
    int usableViewCount = 0;
    int depthViewCount = 0;
    bool usedDepthFreeSpaceCarving = false;
    bool retriedWithoutDepthCarving = false;
    bool qualityRejected = false;
    int removedSatelliteComponentCount = 0;
    MeshConnectivityStats connectivity;
    QString actualAlgorithm;
    QString fallbackReason;
    QString message;
    TriMesh mesh;
};

struct DepthMapVisualHullOptions
{
    bool strictVolumetricMasks = false;
    double minimumLargestComponentFaceRatio = 0.85;
    int maximumConnectedComponents = 12;
};

class DepthMapMeshBuilder
{
public:
    static QVector<DepthFrameArtifact> discoverDepthFrames(const QString &sourcePath);
    static DepthMapVisualHullResult buildVisualHull(
        const QString &sourcePath,
        int resolution,
        const std::function<void(const QString &, int)> &progress = {});
    static DepthMapVisualHullResult buildVisualHull(
        const QString &sourcePath,
        int resolution,
        const DepthMapVisualHullOptions &options,
        const std::function<void(const QString &, int)> &progress = {});
    static QString resolveReusableDenseCloud(const QString &sourcePath, QString *errorMessage = nullptr);
};

} // namespace xjw::mesh
