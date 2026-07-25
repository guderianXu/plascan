#pragma once

#include "Camera.h"
#include "PointCloudAlignment.h"

#include <opencv2/core.hpp>

#include <QString>
#include <QJsonObject>
#include <QStringList>
#include <QVector>

#include <limits>

namespace xjw::qc
{

struct ModelRenderResult
{
    bool ok = false;
    cv::Mat color;
    cv::Mat validMask;
    cv::Mat depth;
    int visibleTriangleCount = 0;
    double elapsedMs = 0.0;
    QString error;
};

struct ModelViewQuality
{
    QString viewId;
    QString imagePath;
    bool renderOk = false;
    QString error;
    int width = 0;
    int height = 0;
    double renderElapsedMs = 0.0;
    double referenceCoverage = 0.0;
    double silhouetteIou = 0.0;
    double floatingPixelRate = 1.0;
    double edgeP50Pixels = std::numeric_limits<double>::infinity();
    double edgeP90Pixels = std::numeric_limits<double>::infinity();
    double referenceToRenderEdgeP50Pixels = std::numeric_limits<double>::infinity();
    double referenceToRenderEdgeP90Pixels = std::numeric_limits<double>::infinity();
    double renderToReferenceEdgeP50Pixels = std::numeric_limits<double>::infinity();
    double renderToReferenceEdgeP90Pixels = std::numeric_limits<double>::infinity();
    double silhouetteEdgeP50Pixels = std::numeric_limits<double>::infinity();
    double silhouetteEdgeP90Pixels = std::numeric_limits<double>::infinity();
    double referenceToRenderSilhouetteEdgeP50Pixels =
        std::numeric_limits<double>::infinity();
    double referenceToRenderSilhouetteEdgeP90Pixels =
        std::numeric_limits<double>::infinity();
    double renderToReferenceSilhouetteEdgeP50Pixels =
        std::numeric_limits<double>::infinity();
    double renderToReferenceSilhouetteEdgeP90Pixels =
        std::numeric_limits<double>::infinity();
    double structureEdgeP50Pixels = std::numeric_limits<double>::infinity();
    double structureEdgeP90Pixels = std::numeric_limits<double>::infinity();
    double referenceToRenderStructureEdgeP50Pixels =
        std::numeric_limits<double>::infinity();
    double referenceToRenderStructureEdgeP90Pixels =
        std::numeric_limits<double>::infinity();
    double renderToReferenceStructureEdgeP50Pixels =
        std::numeric_limits<double>::infinity();
    double renderToReferenceStructureEdgeP90Pixels =
        std::numeric_limits<double>::infinity();
    bool appearanceAvailable = false;
    double foregroundSsim = 0.0;
    double foregroundPsnr = 0.0;
    struct EdgeTailDiagnostics
    {
        bool available = false;
        double thresholdPixels = 0.0;
        double referenceThresholdPixels = 0.0;
        double renderedThresholdPixels = 0.0;
        int referenceEdgePixelCount = 0;
        int renderedEdgePixelCount = 0;
        int referenceTailPixelCount = 0;
        int renderedTailPixelCount = 0;
        double tailGeometrySupportMean = 0.0;
        double tailGeometrySourceCountMean = 0.0;
        double tailInverseDepthSpreadMean = 0.0;
        int tailCrossViewRepairedPixelCount = 0;
    } edgeTail;
    struct DepthCoverageAttribution
    {
        bool available = false;
        bool fusionEligible = true;
        QString frameAcceptance;
        int geometrySupportThreshold = 0;
        int foregroundPixelCount = 0;
        int meshMissingPixelCount = 0;
        int outsideSupportMissingPixelCount = 0;
        int depthInvalidMissingPixelCount = 0;
        int geometryUnverifiedMissingPixelCount = 0;
        int verifiedDepthButMeshMissingPixelCount = 0;
        double verifiedDepthForegroundCoverage = 0.0;
        double missingWithoutVerifiedDepthRate = 0.0;
        double verifiedDepthButMeshMissingRate = 0.0;
    } depthAttribution;
};

enum class ModelSceneType
{
    Dino,
    Aerial
};

struct ModelValidationView
{
    QString id;
    QString imagePath;
    xjw::Camera camera;
    int cameraWidth = 0;
    int cameraHeight = 0;
    QString depthPath;
    QString geometrySupportPath;
    QString geometrySourceMaskPath;
    QString inverseDepthMeanPath;
    QString inverseDepthSpreadPath;
    QString crossViewRepairedMaskPath;
    QString validMaskPath;
    QString supportMaskPath;
    QString frameAcceptance;
    bool fusionEligible = true;
    int sourceViewCount = 0;
};

struct ModelImageQualityOptions
{
    QString meshPath;
    QVector<ModelValidationView> validationViews;
    QString referenceCloudPath;
    QString outputDirectory;
    int maximumRenderDimension = 1600;
    bool alignReferenceCloud = false;
    bool hasReferenceTransform = false;
    SimilarityTransform referenceTransform;
    bool cropReferenceToModelBounds = false;
    ModelSceneType sceneType = ModelSceneType::Dino;
};

} // namespace xjw::qc
