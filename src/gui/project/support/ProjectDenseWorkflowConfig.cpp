#include "ProjectDenseWorkflowConfig.h"

#include <QtGlobal>

#include <algorithm>
#include <cmath>

namespace xjw::gui::project {

namespace
{

plapoint::ProcessingDevice processingDeviceFromString(const QString &value)
{
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("cpu"))
    {
        return plapoint::ProcessingDevice::CPU;
    }
    if (normalized == QStringLiteral("gpu") || normalized == QStringLiteral("cuda"))
    {
        return plapoint::ProcessingDevice::GPU;
    }
    return plapoint::ProcessingDevice::Auto;
}

int autoGpuFrameWorkers(int threads, int viewCount)
{
    const int maxByViews = std::max(1, viewCount);
    const int maxGpuWorkers = std::min(2, maxByViews);
    const int desired = threads >= 8 ? 2 : 1;
    return std::clamp(desired, 1, maxGpuWorkers);
}

int autoCpuFrameWorkers(int threads, int viewCount)
{
    const int maxByViews = std::max(1, viewCount);
    const int maxCpuWorkers = std::min(4, maxByViews);
    return std::clamp(std::max(1, threads / 4), 1, maxCpuWorkers);
}

bool hasAnyKey(const QJsonObject &settings, std::initializer_list<const char *> keys)
{
    for (const char *key : keys)
    {
        if (settings.contains(QString::fromLatin1(key)))
        {
            return true;
        }
    }
    return false;
}

void applyDenseQualityProfile(DenseGenerationSettings *parsed,
                              const QJsonObject &settings)
{
    if (!parsed)
    {
        return;
    }

    const QString profile = parsed->qualityProfile.trimmed().toLower();
    const bool hasMinViews = hasAnyKey(settings, {"minViews"});
    const bool hasPatchConfidence = hasAnyKey(settings, {"confidence", "min_confidence"});
    const bool hasFusionConfidence = hasAnyKey(settings, {"minConfidence"});
    const bool hasMinConsistentViews = hasAnyKey(settings, {"minConsistentViews"});
    const bool hasFusionRelDepthThreshold = hasAnyKey(settings, {"fusionRelDepthThreshold", "relDepthThreshold"});
    const bool hasDepthConsistency = hasAnyKey(settings, {"depthConsistency"});
    const bool hasMaxReprojError = hasAnyKey(settings, {"maxReprojError"});

    if (profile == QStringLiteral("fast_preview"))
    {
        if (!hasMinViews) parsed->minViews = 3;
        if (!hasPatchConfidence) parsed->patchMatchConfidence = 0.30f;
        if (!hasFusionConfidence) parsed->fusionMinConfidence = 0.30f;
        if (!hasMinConsistentViews) parsed->minConsistentViews = 2;
        if (!hasFusionRelDepthThreshold) parsed->fusionRelDepthThreshold = 0.05f;
        if (!hasDepthConsistency) parsed->depthConsistency = 2.0f;
        if (!hasMaxReprojError) parsed->maxReprojError = 2.0f;
        return;
    }

    if (profile == QStringLiteral("high_quality"))
    {
        parsed->minViews = std::max(parsed->minViews, 7);
        parsed->patchMatchConfidence = std::max(parsed->patchMatchConfidence, 0.70f);
        parsed->fusionMinConfidence = std::max(parsed->fusionMinConfidence, 0.70f);
        parsed->minConsistentViews = std::max(parsed->minConsistentViews, 4);
        if (!hasFusionRelDepthThreshold) parsed->fusionRelDepthThreshold = 0.02f;
        if (!hasDepthConsistency) parsed->depthConsistency = 1.0f;
        if (!hasMaxReprojError) parsed->maxReprojError = 1.0f;
        return;
    }

    parsed->qualityProfile = QStringLiteral("standard");
    parsed->minViews = std::max(parsed->minViews, 6);
    parsed->patchMatchConfidence = std::max(parsed->patchMatchConfidence, 0.60f);
    parsed->fusionMinConfidence = std::max(parsed->fusionMinConfidence, 0.65f);
    parsed->minConsistentViews = std::max(parsed->minConsistentViews, 3);
    if (!hasFusionRelDepthThreshold) parsed->fusionRelDepthThreshold = 0.03f;
    if (!hasDepthConsistency) parsed->depthConsistency = 1.5f;
    if (!hasMaxReprojError) parsed->maxReprojError = 1.5f;
}

} // namespace

DenseGenerationSettings denseGenerationSettingsFromJson(const QJsonObject &settings)
{
    DenseGenerationSettings parsed;
    parsed.atIndex = settings.value(QStringLiteral("at_index")).toInt(-1);
    parsed.outputDir = settings.value(QStringLiteral("output_dir")).toString();
    parsed.resScale = settings.value(QStringLiteral("resScale")).toDouble(0.5);
    parsed.iterations = settings.value(QStringLiteral("iterations")).toInt(6);
    parsed.threads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    parsed.gpuFrameWorkers = std::max(0, settings.value(QStringLiteral("gpu_frame_workers")).toInt(0));
    parsed.cpuFrameWorkers = std::max(0, settings.value(QStringLiteral("cpu_frame_workers")).toInt(0));
    parsed.patchSize = settings.value(QStringLiteral("patchSize")).toInt(11);
    parsed.minViews = settings.value(QStringLiteral("minViews")).toInt(3);
    parsed.patchMatchConfidence = static_cast<float>(
        settings.contains(QStringLiteral("confidence"))
            ? settings.value(QStringLiteral("confidence")).toDouble()
            : settings.value(QStringLiteral("min_confidence")).toDouble(0.20));
    parsed.useCuda = settings.value(QStringLiteral("cuda")).toBool(true);
    parsed.fusionMinConfidence = static_cast<float>(
        settings.value(QStringLiteral("minConfidence")).toDouble(parsed.patchMatchConfidence));
    parsed.fusionMaxImageDim = std::max(0,
        settings.value(QStringLiteral("fusion_max_image_dim")).toInt(
            settings.value(QStringLiteral("fusionMaxImageDim")).toInt(2048)));
    parsed.minConsistentViews = settings.value(QStringLiteral("minConsistentViews")).toInt(2);
    parsed.geomConsistency = settings.value(QStringLiteral("geomConsistency")).toBool(true);
    parsed.fusionRelDepthThreshold = static_cast<float>(
        settings.contains(QStringLiteral("fusionRelDepthThreshold"))
            ? settings.value(QStringLiteral("fusionRelDepthThreshold")).toDouble()
            : settings.value(QStringLiteral("relDepthThreshold")).toDouble(0.05));
    parsed.depthConsistency = static_cast<float>(settings.value(QStringLiteral("depthConsistency")).toDouble(2.0));
    parsed.maxReprojError = static_cast<float>(settings.value(QStringLiteral("maxReprojError")).toDouble(2.0));
    parsed.speckleMinArea = std::max(0, settings.value(QStringLiteral("speckleMinArea")).toInt(16));
    parsed.qualityProfile = settings.value(QStringLiteral("qualityProfile")).toString(QStringLiteral("standard"));
    parsed.processingDevice = processingDeviceFromString(
        settings.value(QStringLiteral("processingDevice")).toString(QStringLiteral("auto")));
    parsed.pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);
    applyDenseQualityProfile(&parsed, settings);
    return parsed;
}

xjw::mvs::DepthGenConfig buildDepthGenConfig(const DenseGenerationSettings &settings,
                                             int viewCount)
{
    xjw::mvs::DepthGenConfig config;
    config.numSourceViews = std::min(settings.minViews, viewCount - 1);
    const int totalThreads = std::max(1, settings.threads);
    const int maxByViews = std::max(1, viewCount);
    const int maxGpuWorkers = std::min(2, maxByViews);
    const int maxCpuWorkers = std::min(4, maxByViews);
    config.gpuFrameWorkerCount = settings.useCuda
        ? std::clamp(settings.gpuFrameWorkers > 0
                         ? settings.gpuFrameWorkers
                         : autoGpuFrameWorkers(totalThreads, viewCount),
                     1,
                     maxGpuWorkers)
        : 0;
    config.cpuFrameWorkerCount = settings.useCuda
        ? std::clamp(settings.cpuFrameWorkers, 0, maxCpuWorkers)
        : std::clamp(settings.cpuFrameWorkers > 0
                         ? settings.cpuFrameWorkers
                         : autoCpuFrameWorkers(totalThreads, viewCount),
                     1,
                     maxCpuWorkers);
    config.cpuWorkerCount = std::max(1, totalThreads / std::max(1, config.cpuFrameWorkerCount));
    config.patchMatch.numIterations = settings.iterations;
    config.patchMatch.patchHalf = (settings.patchSize - 1) / 2;
    config.patchMatch.numSourceViews = config.numSourceViews;
    config.patchMatch.confidenceThresh = settings.patchMatchConfidence;
    config.patchMatch.useCuda = settings.useCuda;
    config.patchMatch.downsampleFactor = std::max(1, static_cast<int>(std::round(1.0 / settings.resScale)));
    config.patchMatch.geomConsistency = settings.geomConsistency;
    config.patchMatch.geomConsistencyMaxErr = settings.maxReprojError;
    config.fusion.confidenceThresh = settings.fusionMinConfidence;
    config.fusion.minConsistentViews = std::max(1, settings.minConsistentViews);
    config.fusion.relDepthThresh = settings.fusionRelDepthThreshold;
    config.fusion.pixelThresh = settings.depthConsistency;
    config.fusion.enableAdaptiveConfidenceFilter = true;
    config.fusion.enableSpeckleFilter = settings.speckleMinArea > 0;
    config.fusion.minSpeckleComponentArea = std::max(0, settings.speckleMinArea);

    if (viewCount <= 2)
    {
        // 两视图立体对：允许融合但不强制全分辨率或超高迭代，避免小样本一键重建被 MVS 拖慢。
        config.patchMatch.confidenceThresh = std::min(config.patchMatch.confidenceThresh, 0.05f);
        config.patchMatch.geomConsistency = false;       // 两视图无法做几何一致性检查
        config.fusion.confidenceThresh = std::min(config.fusion.confidenceThresh, 0.05f);
        config.fusion.minConsistentViews = 1;
        config.fusion.relDepthThresh = std::max(config.fusion.relDepthThresh, 0.20f);
        config.fusion.pixelThresh = std::max(config.fusion.pixelThresh, 5.0f);
    }

    return config;
}

DenseRefineSettings denseRefineSettingsFromJson(const QJsonObject &settings)
{
    DenseRefineSettings parsed;
    parsed.sorEnabled = settings.value(QStringLiteral("sorEnabled")).toBool(true);
    parsed.sorK = settings.value(QStringLiteral("sorK")).toInt(30);
    parsed.sorStdDev = settings.value(QStringLiteral("sorStdDev")).toDouble(2.0);
    parsed.voxelEnabled = settings.value(QStringLiteral("voxelEnabled")).toBool(false);
    parsed.voxelSize = settings.value(QStringLiteral("voxelSize")).toDouble(0.005);
    parsed.terrainSpikeFilterEnabled =
        settings.value(QStringLiteral("terrainSpikeFilterEnabled")).toBool(true);
    parsed.terrainSpikeGridResolution =
        std::clamp(settings.value(QStringLiteral("terrainSpikeGridResolution")).toInt(260), 1, 1024);
    parsed.terrainSpikeMinCellPoints =
        std::max(1, settings.value(QStringLiteral("terrainSpikeMinCellPoints")).toInt(32));
    parsed.terrainSpikeMinHeightThreshold =
        settings.value(QStringLiteral("terrainSpikeMinHeightThreshold")).toDouble(0.25);
    parsed.terrainSpikeMadMultiplier =
        settings.value(QStringLiteral("terrainSpikeMadMultiplier")).toDouble(3.0);
    parsed.terrainLocalPlaneFilterEnabled =
        settings.value(QStringLiteral("terrainLocalPlaneFilterEnabled")).toBool(true);
    parsed.terrainLocalPlaneMinPoints =
        std::max(3, settings.value(QStringLiteral("terrainLocalPlaneMinPoints")).toInt(12));
    parsed.terrainLocalPlaneMinResidualThreshold =
        settings.value(QStringLiteral("terrainLocalPlaneMinResidualThreshold")).toDouble(0.12);
    parsed.terrainLocalPlaneMadMultiplier =
        settings.value(QStringLiteral("terrainLocalPlaneMadMultiplier")).toDouble(4.0);
    parsed.terrainFilterPasses =
        std::clamp(settings.value(QStringLiteral("terrainFilterPasses")).toInt(2), 1, 8);
    parsed.normalsEnabled = settings.value(QStringLiteral("normalsEnabled")).toBool(true);
    parsed.normalK = settings.value(QStringLiteral("normalK")).toInt(30);
    parsed.smoothNormals = settings.value(QStringLiteral("smoothIter")).toInt(0) > 0;
    parsed.threads = qMax(1, settings.value(QStringLiteral("threads")).toInt(8));
    parsed.processingDevice = processingDeviceFromString(
        settings.value(QStringLiteral("processingDevice")).toString(QStringLiteral("auto")));
    return parsed;
}

} // namespace xjw::gui::project
