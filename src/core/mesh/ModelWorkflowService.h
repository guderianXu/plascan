#pragma once

#include "SurfaceReconstructor.h"
#include "TextureMapper.h"

#include <QJsonObject>
#include <QString>

#include <functional>

namespace xjw::mesh
{
struct DepthTsdfOptions;
}

namespace xjw::mesh::workflow
{

struct MeshBuildRequest
{
    QString pointCloudPath;
    QString outputRoot;
    xjw::mesh::ReconstructionConfig reconstruction;
    bool exportObj = false;
    xjw::mesh::TextureMappingConfig texture;
    std::function<void(const QString &, int)> progress;
};

struct DepthMapMeshBuildRequest
{
    QString depthMapSourcePath;
    QString reusableDenseCloudPath;
    QString outputRoot;
    QJsonObject settings;
    xjw::mesh::ReconstructionConfig reconstruction;
    bool exportObj = false;
    xjw::mesh::TextureMappingConfig texture;
    std::function<bool()> isCancelled;
    std::function<void(const QString &, int)> progress;
};

struct TextureBuildRequest
{
    QString meshPath;
    QString outputDir;
    QString depthMapSourcePath;
    xjw::mesh::TextureMappingConfig texture;
    bool allowVertexColorFallback = false;
    std::function<bool()> isCancelled;
    std::function<void(const QString &, int)> progress;
};

struct ModelBuildRequest
{
    QString sourceData = QStringLiteral("point_cloud");
    QString requestedSourcePath;
    QString sourcePointCloudPath;
    QString depthMapSourcePath;
    QString outputRoot;
    QJsonObject settings;
    std::function<bool()> isCancelled;
    std::function<void(const QString &, int)> progress;
};

struct WorkflowResult
{
    bool ok = false;
    QString errorMessage;
    QJsonObject payload;
};

struct PointCloudQualityReport
{
    qint64 pointCount = 0;
    bool hasCount = false;
    bool belowRecommended = false;
};

int meshResolutionFromSettings(const QJsonObject &settings);
QString depthReconstructionModeFromSettings(const QJsonObject &settings);
xjw::mesh::DepthTsdfOptions depthTsdfOptionsFromSettings(const QJsonObject &settings,
                                                         int requestedResolution);
void applyOrbitalDepthTsdfDefaults(const QJsonObject &settings,
                                   xjw::mesh::DepthTsdfOptions *options,
                                   int maximumReliableResolution = 0);
bool visibilityOccupancyDepthRefinementEnabled(const QJsonObject &settings,
                                               bool orbitalWorkspace);
bool shouldUseOrbitalVisualHullCompletion(bool orbitalWorkspace,
                                          bool enabled,
                                          bool observationOnlySurface,
                                          double aggregateProjectionRecall,
                                          int boundaryEdgeCount,
                                          int faceCount);
xjw::mesh::ReconstructionConfig reconstructionConfigFromModelSettings(const QJsonObject &settings);
xjw::mesh::ReconstructionConfig reconstructionConfigForDenseScene(int requestedResolution,
                                                                   bool aerialTerrain,
                                                                   bool preserveDetail);
int holeFillPassesFromArea(double maxHoleArea);
xjw::mesh::TextureMappingConfig defaultTextureConfig();
xjw::mesh::TextureMappingConfig textureConfigFromSettings(const QJsonObject &settings);
PointCloudQualityReport evaluatePointCloudQuality(const QString &pointCloudPath,
                                                  qint64 recommendedMinimum = 200);

WorkflowResult buildMeshAndOptionalTexture(const MeshBuildRequest &request);
WorkflowResult buildMeshFromDepthMaps(const DepthMapMeshBuildRequest &request);
WorkflowResult buildModel(const ModelBuildRequest &request);
WorkflowResult buildTextureOnly(const TextureBuildRequest &request);

} // namespace xjw::mesh::workflow
