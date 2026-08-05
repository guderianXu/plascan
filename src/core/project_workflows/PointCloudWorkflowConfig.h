#pragma once

#include "MvsTypes.h"

#include <plapoint/filters/preprocessing.h>

#include <QJsonObject>
#include <QString>

namespace xjw::core::project {

enum class DepthQualityProfile
{
    Highest,
    High,
    Medium,
    Low,
    Lowest
};

QString depthQualityProfileId(DepthQualityProfile profile);
DepthQualityProfile depthQualityProfileFromId(const QString &profileId);
int depthQualityDownsample(DepthQualityProfile profile);
int depthQualityRank(const QString &profileId);
QString depthQualityProfileForModelQuality(const QString &modelQuality);

struct DepthQualityParameters
{
    double resScale = 0.25;
    int iterations = 8;
    int patchSize = 11;
    int minViews = 6;
    float patchMatchConfidence = 0.60f;
    float fusionMinConfidence = 0.65f;
    int minConsistentViews = 3;
    float fusionRelDepthThreshold = 0.03f;
    float depthConsistency = 1.5f;
    float maxReprojError = 1.5f;
};

DepthQualityParameters depthQualityParameters(DepthQualityProfile profile);

struct DenseGenerationSettings
{
    int atIndex = -1;
    QString outputDir;
    double resScale = 0.5;
    int iterations = 6;
    int threads = 8;
    int gpuFrameWorkers = 0; // 0 表示按线程数和视图数自动选择
    int cpuFrameWorkers = 0; // 0 表示按线程数和视图数自动选择
    int patchSize = 11;
    int minViews = 6;
    float patchMatchConfidence = 0.60f;
    bool useCuda = true;
    float fusionMinConfidence = 0.65f;
    int fusionMaxImageDim = 2048; // 0 表示融合阶段使用深度图原始尺寸
    int minConsistentViews = 3;
    bool geomConsistency = true;
    float fusionRelDepthThreshold = 0.03f;
    float depthConsistency = 1.5f;
    float maxReprojError = 1.5f;
    int speckleMinArea = 16;
    QString qualityProfile = QStringLiteral("medium");
    QString sceneProfile = QStringLiteral("auto");
    QString depthFilterMode = QStringLiteral("auto");
    bool saveIntermediatePyramidLevels = true;
    bool enableTargetedGapRecovery = true;
    int targetedGapRecoverySourceCount = 6;
    int targetedGapRecoveryHypothesisCount = 2;
    float targetedGapRecoveryConfidence = 0.28f;
    float targetedGapRecoveryPriorRelativeDifference = 0.18f;
    float targetedGapRecoveryConsensusInverseDepthSpread = 0.025f;
    float targetedGapRecoveryConsensusPriorRelativeDifference = 0.35f;
    bool enableTargetedGapSurfacePrior = false;
    float targetedGapSurfacePriorMaximumAnchorSpread = 0.12f;
    float targetedGapSurfacePriorMaximumFitResidual = 0.025f;
    int targetedGapRecoveryMaximumPriorDistancePixels = 128;
    bool enableTwoSourceCrossViewGrowth = false;
    int twoSourceGrowthDistancePixels = 3;
    float twoSourceGrowthInverseDepthSpread = 0.01f;
    float twoSourceGrowthNormalAngleDegrees = 15.0f;
    int twoSourceGrowthMaximumComponentArea = 64;
    plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto;
    bool pipelineMode = false;  // 流水线模式：跳过所有交互对话框
};

DenseGenerationSettings denseGenerationSettingsFromJson(const QJsonObject &settings);

xjw::mvs::DepthGenConfig buildDepthGenConfig(const DenseGenerationSettings &settings,
                                             int viewCount);

struct DenseRefineSettings
{
    bool sorEnabled = true;
    int sorK = 30;
    double sorStdDev = 2.0;
    bool voxelEnabled = false;
    double voxelSize = 0.005;
    bool terrainSpikeFilterEnabled = true;
    int terrainSpikeGridResolution = 260;
    int terrainSpikeMinCellPoints = 32;
    double terrainSpikeMinHeightThreshold = 0.25;
    double terrainSpikeMadMultiplier = 3.0;
    bool terrainLocalPlaneFilterEnabled = true;
    int terrainLocalPlaneMinPoints = 12;
    double terrainLocalPlaneMinResidualThreshold = 0.12;
    double terrainLocalPlaneMadMultiplier = 4.0;
    int terrainFilterPasses = 2;
    bool normalsEnabled = true;
    int normalK = 30;
    bool smoothNormals = false;
    int threads = 8;
    plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto;
};

DenseRefineSettings denseRefineSettingsFromJson(const QJsonObject &settings);

} // namespace xjw::core::project
