#pragma once

#include "SurfaceReconstructor.h"
#include "TextureMapper.h"

#include <QJsonObject>
#include <QString>

#include <functional>

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
    QString outputRoot;
    QJsonObject settings;
    xjw::mesh::ReconstructionConfig reconstruction;
    bool exportObj = false;
    xjw::mesh::TextureMappingConfig texture;
    std::function<void(const QString &, int)> progress;
};

struct TextureBuildRequest
{
    QString meshPath;
    QString outputDir;
    xjw::mesh::TextureMappingConfig texture;
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
xjw::mesh::ReconstructionConfig reconstructionConfigFromModelSettings(const QJsonObject &settings);
int holeFillPassesFromArea(double maxHoleArea);
xjw::mesh::TextureMappingConfig defaultTextureConfig();
xjw::mesh::TextureMappingConfig textureConfigFromSettings(const QJsonObject &settings);
bool exportObjRequested(const QJsonObject &settings);
PointCloudQualityReport evaluatePointCloudQuality(const QString &pointCloudPath,
                                                  qint64 recommendedMinimum = 200);

WorkflowResult buildMeshAndOptionalTexture(const MeshBuildRequest &request);
WorkflowResult buildMeshFromDepthMaps(const DepthMapMeshBuildRequest &request);
WorkflowResult buildTextureOnly(const TextureBuildRequest &request);

} // namespace xjw::mesh::workflow
