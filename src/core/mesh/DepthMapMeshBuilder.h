#pragma once

#include "Camera.h"
#include "MeshTypes.h"
#include "VisualHullReconstructor.h"

#include <QString>
#include <QStringList>
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
    QString geometrySupportPath;
    QString geometrySourceMaskPath;
    QString adaptiveGeometrySupportWeightPath;
    QString adaptiveGeometryEffectiveViewCountPath;
    QString adaptiveGeometryConflictRatioPath;
    QString adaptiveGeometryConflictWeightPath; ///< Legacy revision-13 absolute conflict mass.
    QString inverseDepthMeanPath;
    QString inverseDepthSpreadPath;
    QString crossViewRepairedMaskPath;
    QString depthProvenancePath;
    QVector<int> sourceIndices;
    QString previewPath;
    QString validMaskPath;
    QString supportMaskPath;
    QString status;
    QString acceptance;
    QString sceneProfile;
    int algorithmRevision = 0;
    bool fusionEligible = true;
    double validCoverage = -1.0;
    double validWithinMaskRatio = -1.0;
    double consistencyRetentionRatio = -1.0;
    double largestComponentRatio = -1.0;
    double meanConfidence = -1.0;
    int sourceViewCount = 0;
    QStringList qualityReasons;
    int gridWidth = 0;
    int gridHeight = 0;
    Camera cameraModel;
    bool hasCameraModel = false;
    bool pyramidFallback = false;
};

struct DepthMapVisualHullResult
{
    bool applicable = false;
    bool ok = false;
    int usableViewCount = 0;
    int depthViewCount = 0;
    int preservedSilhouetteHoleViewCount = 0;
    int topologyClosingIterations = 0;
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
    int topologyClosingIterations = -1;
    bool useContinuousSilhouetteField = false;
    int smoothingIterations = 6;
    float smoothingLambda = 0.18f;
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
