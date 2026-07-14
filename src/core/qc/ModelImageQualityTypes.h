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
    bool appearanceAvailable = false;
    double foregroundSsim = 0.0;
    double foregroundPsnr = 0.0;
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
