#include "PointCloudWorkflowConfig.h"

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif
#ifdef PLAPOINT_WITH_OPENCL
#include <plapoint/opencl/opencl_runtime.h>
#endif

#include <QtGlobal>

#include <algorithm>
#include <cmath>

namespace xjw::core::project {

plapoint::ProcessingDevice processingDeviceFromString(const QString &value)
{
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("cpu"))
    {
        return plapoint::ProcessingDevice::CPU;
    }
    if (normalized == QStringLiteral("gpu") || normalized == QStringLiteral("cuda"))
    {
        return plapoint::ProcessingDevice::CUDA;
    }
    if (normalized == QStringLiteral("opencl"))
    {
        return plapoint::ProcessingDevice::OpenCL;
    }
    return plapoint::ProcessingDevice::Auto;
}

QString processingDeviceId(plapoint::ProcessingDevice device)
{
    switch (device)
    {
    case plapoint::ProcessingDevice::CPU:
        return QStringLiteral("cpu");
    case plapoint::ProcessingDevice::CUDA:
        return QStringLiteral("cuda");
    case plapoint::ProcessingDevice::OpenCL:
        return QStringLiteral("opencl");
    case plapoint::ProcessingDevice::Auto:
        return QStringLiteral("auto");
    }
    return QStringLiteral("unknown");
}

QString processingDeviceUnavailableReason(plapoint::ProcessingDevice device)
{
    if (device == plapoint::ProcessingDevice::CUDA)
    {
#ifdef PLAPOINT_WITH_CUDA
        return plapoint::gpu::hasUsableCudaDevice()
            ? QString()
            : QStringLiteral("请求的 CUDA 点云处理后端没有可用设备");
#else
        return QStringLiteral("PlaPoint 构建时未启用 CUDA 点云处理后端");
#endif
    }
    if (device == plapoint::ProcessingDevice::OpenCL)
    {
#ifdef PLAPOINT_WITH_OPENCL
        return plapoint::opencl::hasUsableOpenClDevice()
            ? QString()
            : QStringLiteral("请求的 OpenCL 点云处理后端没有可用 GPU 或在线编译器");
#else
        return QStringLiteral("PlaPoint 构建时未启用 OpenCL 点云处理后端");
#endif
    }
    return {};
}

namespace
{

xjw::mvs::MvsSceneProfile sceneProfileFromString(const QString &value)
{
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("orbital_object"))
    {
        return xjw::mvs::MvsSceneProfile::OrbitalObject;
    }
    if (normalized == QStringLiteral("aerial_terrain"))
    {
        return xjw::mvs::MvsSceneProfile::AerialTerrain;
    }
    return xjw::mvs::MvsSceneProfile::Auto;
}

xjw::mvs::DepthFilterMode depthFilterModeFromString(const QString &value)
{
    const QString normalized = value.trimmed().toLower();
    if (normalized == QStringLiteral("mild"))
    {
        return xjw::mvs::DepthFilterMode::Mild;
    }
    if (normalized == QStringLiteral("aggressive"))
    {
        return xjw::mvs::DepthFilterMode::Aggressive;
    }
    return xjw::mvs::DepthFilterMode::Moderate;
}

int autoGpuFrameWorkers(int threads, int viewCount)
{
    const int maxByViews = std::max(1, viewCount);
    const int maxGpuWorkers = std::min(2, maxByViews);
    // CUDA execution remains serialized inside PatchMatch because its camera
    // constants and reusable workspace are shared. A second host frame slot
    // overlaps CPU preparation/post-processing and asynchronous saving without
    // running two memory-heavy PatchMatch kernels beside the Qt renderer.
    const int desired = threads >= 4 && viewCount >= 2 ? 2 : 1;
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

    const QString raw_profile = parsed->qualityProfile.trimmed().toLower();
    const bool custom_profile = raw_profile == QStringLiteral("custom");
    const DepthQualityProfile profile = depthQualityProfileFromId(raw_profile);
    const DepthQualityParameters defaults = depthQualityParameters(profile);
    parsed->qualityProfile = custom_profile ? QStringLiteral("custom") : depthQualityProfileId(profile);
    const bool hasResScale = hasAnyKey(settings, {"resScale"});
    const bool hasIterations = hasAnyKey(settings, {"iterations"});
    const bool hasPatchSize = hasAnyKey(settings, {"patchSize"});
    const bool hasMinViews = hasAnyKey(settings, {"minViews"});
    const bool hasPatchConfidence = hasAnyKey(settings, {"confidence", "min_confidence"});
    const bool hasFusionConfidence = hasAnyKey(settings, {"minConfidence"});
    const bool hasMinConsistentViews = hasAnyKey(settings, {"minConsistentViews"});
    const bool hasFusionRelDepthThreshold = hasAnyKey(settings, {"fusionRelDepthThreshold", "relDepthThreshold"});
    const bool hasDepthConsistency = hasAnyKey(settings, {"depthConsistency"});
    const bool hasMaxReprojError = hasAnyKey(settings, {"maxReprojError"});

    if (!hasResScale) parsed->resScale = defaults.resScale;
    if (!hasIterations) parsed->iterations = defaults.iterations;
    if (!hasPatchSize) parsed->patchSize = defaults.patchSize;
    if (!hasMinViews) parsed->minViews = defaults.minViews;
    if (!hasPatchConfidence) parsed->patchMatchConfidence = defaults.patchMatchConfidence;
    if (!hasFusionConfidence) parsed->fusionMinConfidence = defaults.fusionMinConfidence;
    if (!hasMinConsistentViews) parsed->minConsistentViews = defaults.minConsistentViews;
    if (!hasFusionRelDepthThreshold)
    {
        parsed->fusionRelDepthThreshold = defaults.fusionRelDepthThreshold;
    }
    if (!hasDepthConsistency) parsed->depthConsistency = defaults.depthConsistency;
    if (!hasMaxReprojError) parsed->maxReprojError = defaults.maxReprojError;
}

} // namespace

QString depthQualityProfileId(DepthQualityProfile profile)
{
    switch (profile)
    {
    case DepthQualityProfile::Highest:
        return QStringLiteral("highest");
    case DepthQualityProfile::High:
        return QStringLiteral("high");
    case DepthQualityProfile::Low:
        return QStringLiteral("low");
    case DepthQualityProfile::Lowest:
        return QStringLiteral("lowest");
    case DepthQualityProfile::Medium:
    default:
        return QStringLiteral("medium");
    }
}

DepthQualityProfile depthQualityProfileFromId(const QString &profileId)
{
    const QString normalized = profileId.trimmed().toLower();
    if (normalized == QStringLiteral("highest"))
    {
        return DepthQualityProfile::Highest;
    }
    if (normalized == QStringLiteral("high") || normalized == QStringLiteral("high_quality"))
    {
        return DepthQualityProfile::High;
    }
    if (normalized == QStringLiteral("low") || normalized == QStringLiteral("fast_preview"))
    {
        return DepthQualityProfile::Low;
    }
    if (normalized == QStringLiteral("lowest"))
    {
        return DepthQualityProfile::Lowest;
    }
    return DepthQualityProfile::Medium;
}

int depthQualityDownsample(DepthQualityProfile profile)
{
    switch (profile)
    {
    case DepthQualityProfile::Highest:
        return 1;
    case DepthQualityProfile::High:
        return 2;
    case DepthQualityProfile::Low:
        return 8;
    case DepthQualityProfile::Lowest:
        return 16;
    case DepthQualityProfile::Medium:
    default:
        return 4;
    }
}

int depthQualityRank(const QString &profileId)
{
    switch (depthQualityProfileFromId(profileId))
    {
    case DepthQualityProfile::Highest:
        return 4;
    case DepthQualityProfile::High:
        return 3;
    case DepthQualityProfile::Medium:
        return 2;
    case DepthQualityProfile::Low:
        return 1;
    case DepthQualityProfile::Lowest:
        return 0;
    }
    return 2;
}

QString depthQualityProfileForModelQuality(const QString &modelQuality)
{
    const QString normalized = modelQuality.trimmed().toLower();
    if (normalized == QStringLiteral("ultra"))
    {
        return QStringLiteral("highest");
    }
    if (normalized == QStringLiteral("high"))
    {
        return QStringLiteral("high");
    }
    if (normalized == QStringLiteral("low"))
    {
        return QStringLiteral("low");
    }
    return QStringLiteral("medium");
}

DepthQualityParameters depthQualityParameters(DepthQualityProfile profile)
{
    switch (profile)
    {
    case DepthQualityProfile::Highest:
        return {1.0, 16, 15, 8, 0.72f, 0.75f, 4, 0.015f, 0.8f, 0.8f};
    case DepthQualityProfile::High:
        return {0.5, 12, 13, 7, 0.68f, 0.70f, 4, 0.020f, 1.0f, 1.0f};
    case DepthQualityProfile::Low:
        return {0.125, 4, 9, 3, 0.30f, 0.30f, 2, 0.050f, 2.0f, 2.0f};
    case DepthQualityProfile::Lowest:
        return {0.0625, 3, 7, 2, 0.22f, 0.25f, 1, 0.070f, 2.5f, 2.5f};
    case DepthQualityProfile::Medium:
    default:
        return {0.25, 8, 11, 6, 0.60f, 0.65f, 3, 0.030f, 1.5f, 1.5f};
    }
}

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
    const bool has_backend_key = settings.contains(QStringLiteral("patchMatchBackend")) ||
        settings.contains(QStringLiteral("patchmatch_backend")) ||
        settings.contains(QStringLiteral("mvsBackend")) ||
        settings.contains(QStringLiteral("mvs_backend"));
    const QString patch_match_backend = settings.value(
        QStringLiteral("patchMatchBackend")).toString(
            settings.value(QStringLiteral("patchmatch_backend")).toString(
                settings.value(QStringLiteral("mvsBackend")).toString(
                    settings.value(QStringLiteral("mvs_backend")).toString(
                        QStringLiteral("auto"))))).trimmed().toLower();
    if (patch_match_backend == QStringLiteral("cpu"))
    {
        parsed.patchMatchBackend = xjw::mvs::PatchMatchBackend::Cpu;
    }
    else if (patch_match_backend == QStringLiteral("cuda"))
    {
        parsed.patchMatchBackend = xjw::mvs::PatchMatchBackend::Cuda;
    }
    else if (patch_match_backend == QStringLiteral("opencl"))
    {
        parsed.patchMatchBackend = xjw::mvs::PatchMatchBackend::OpenCl;
    }
    if (!has_backend_key && !parsed.useCuda)
    {
        // Legacy settings only exposed a CUDA checkbox. Preserve `cuda=false`
        // by migrating it to an explicit CPU backend; an explicit Auto token
        // always means CUDA -> OpenCL -> CPU.
        parsed.patchMatchBackend = xjw::mvs::PatchMatchBackend::Cpu;
    }
    parsed.useCuda = parsed.patchMatchBackend == xjw::mvs::PatchMatchBackend::Auto ||
        parsed.patchMatchBackend == xjw::mvs::PatchMatchBackend::Cuda;
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
    parsed.qualityProfile = settings.value(QStringLiteral("depthQualityProfile")).toString(
        settings.value(QStringLiteral("qualityProfile")).toString(QStringLiteral("medium")));
    parsed.sceneProfile = settings.value(QStringLiteral("sceneProfile")).toString(QStringLiteral("auto"));
    parsed.depthFilterMode = settings.value(QStringLiteral("depthFilterMode")).toString(QStringLiteral("auto"));
    parsed.saveIntermediatePyramidLevels =
        settings.value(QStringLiteral("saveIntermediatePyramidLevels")).toBool(false);
    parsed.enableTargetedGapRecovery = settings.value(
        QStringLiteral("enableTargetedGapRecovery")).toBool(true);
    parsed.targetedGapRecoverySourceCount = std::clamp(settings.value(
        QStringLiteral("targetedGapRecoverySourceCount")).toInt(6), 1, 8);
    parsed.targetedGapRecoveryHypothesisCount = std::clamp(settings.value(
        QStringLiteral("targetedGapRecoveryHypothesisCount")).toInt(2), 1, 3);
    parsed.targetedGapRecoveryConfidence = static_cast<float>(settings.value(
        QStringLiteral("targetedGapRecoveryConfidence")).toDouble(0.28));
    parsed.targetedGapRecoveryPriorRelativeDifference = static_cast<float>(
        settings.value(QStringLiteral(
            "targetedGapRecoveryPriorRelativeDifference")).toDouble(0.18));
    parsed.targetedGapRecoveryConsensusInverseDepthSpread = static_cast<float>(
        settings.value(QStringLiteral(
            "targetedGapRecoveryConsensusInverseDepthSpread")).toDouble(0.025));
    parsed.targetedGapRecoveryConsensusPriorRelativeDifference = static_cast<float>(
        settings.value(QStringLiteral(
            "targetedGapRecoveryConsensusPriorRelativeDifference")).toDouble(0.35));
    parsed.enableTargetedGapSurfacePrior = settings.value(
        QStringLiteral("enableTargetedGapSurfacePrior")).toBool(false);
    parsed.targetedGapSurfacePriorMaximumAnchorSpread = static_cast<float>(
        settings.value(QStringLiteral(
            "targetedGapSurfacePriorMaximumAnchorSpread")).toDouble(0.12));
    parsed.targetedGapSurfacePriorMaximumFitResidual = static_cast<float>(
        settings.value(QStringLiteral(
            "targetedGapSurfacePriorMaximumFitResidual")).toDouble(0.025));
    parsed.targetedGapRecoveryMaximumPriorDistancePixels = std::max(
        1,
        settings.value(QStringLiteral(
            "targetedGapRecoveryMaximumPriorDistancePixels")).toInt(128));
    parsed.enablePostConsistencyResidualReestimation = settings.value(
        QStringLiteral("enablePostConsistencyResidualReestimation")).toBool(true);
    parsed.postConsistencyResidualSourceCount = std::clamp(settings.value(
        QStringLiteral("postConsistencyResidualSourceCount")).toInt(8), 4, 16);
    parsed.postConsistencyResidualConfidence = static_cast<float>(settings.value(
        QStringLiteral("postConsistencyResidualConfidence")).toDouble(0.30));
    parsed.postConsistencyResidualMaximumLayerSpread = static_cast<float>(
        settings.value(QStringLiteral(
            "postConsistencyResidualMaximumLayerSpread")).toDouble(0.025));
    parsed.postConsistencyResidualMaximumPriorRadius = static_cast<float>(
        settings.value(QStringLiteral(
            "postConsistencyResidualMaximumPriorRadius")).toDouble(0.08));
    parsed.enableTwoSourceCrossViewGrowth = settings.value(
        QStringLiteral("enableTwoSourceCrossViewGrowth")).toBool(false);
    parsed.twoSourceGrowthDistancePixels = settings.value(
        QStringLiteral("twoSourceGrowthDistancePixels")).toInt(3);
    parsed.twoSourceGrowthInverseDepthSpread = static_cast<float>(settings.value(
        QStringLiteral("twoSourceGrowthInverseDepthSpread")).toDouble(0.01));
    parsed.twoSourceGrowthNormalAngleDegrees = static_cast<float>(settings.value(
        QStringLiteral("twoSourceGrowthNormalAngleDegrees")).toDouble(15.0));
    parsed.twoSourceGrowthMaximumComponentArea = settings.value(
        QStringLiteral("twoSourceGrowthMaximumComponentArea")).toInt(64);
    parsed.processingDevice = processingDeviceFromString(
        settings.value(QStringLiteral("processingDevice")).toString(
            settings.value(QStringLiteral("pointCloudBackend")).toString(
                settings.value(QStringLiteral("point_cloud_backend")).toString(
                    QStringLiteral("auto")))));
    parsed.pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);
    applyDenseQualityProfile(&parsed, settings);
    return parsed;
}

xjw::mvs::DepthGenConfig buildDepthGenConfig(const DenseGenerationSettings &settings,
                                             int viewCount)
{
    xjw::mvs::DepthGenConfig config;
    config.qualityProfile = settings.qualityProfile.toStdString();
    config.numSourceViews = std::min(settings.minViews, viewCount - 1);
    config.configuredSourceViewCount = config.numSourceViews;
    // This configuration is used by the interactive project workflow. Leave
    // one logical worker out of the historical eight-thread budget so Qt can
    // continue processing input, painting, progress, and cancellation events.
    const int totalThreads = std::clamp(settings.threads, 1, 7);
    const int maxByViews = std::max(1, viewCount);
    const int maxGpuWorkers = std::min(2, maxByViews);
    const int maxCpuWorkers = std::min(4, maxByViews);
    const bool accelerator_backend =
        settings.patchMatchBackend != xjw::mvs::PatchMatchBackend::Cpu;
    config.gpuFrameWorkerCount = accelerator_backend
        ? std::clamp(settings.gpuFrameWorkers > 0
                         ? settings.gpuFrameWorkers
                         : autoGpuFrameWorkers(totalThreads, viewCount),
                     1,
                     maxGpuWorkers)
        : 0;
    config.cpuFrameWorkerCount = std::clamp(
        settings.cpuFrameWorkers > 0
            ? settings.cpuFrameWorkers
            : autoCpuFrameWorkers(totalThreads, viewCount),
        1,
        maxCpuWorkers);
    const int activeFrameWorkers = std::max(
        1, std::max(config.gpuFrameWorkerCount, config.cpuFrameWorkerCount));
    config.totalCpuThreadBudget = totalThreads;
    config.cpuWorkerCount = std::max(1, totalThreads / activeFrameWorkers);
    config.patchMatch.numIterations = settings.iterations;
    config.patchMatch.patchHalf = (settings.patchSize - 1) / 2;
    config.patchMatch.numSourceViews = config.numSourceViews;
    config.patchMatch.confidenceThresh = settings.patchMatchConfidence;
    config.patchMatch.useCuda = settings.patchMatchBackend == xjw::mvs::PatchMatchBackend::Auto ||
        settings.patchMatchBackend == xjw::mvs::PatchMatchBackend::Cuda;
    config.patchMatch.backend = settings.patchMatchBackend;
    config.pointCloudProcessingDevice = settings.processingDevice;
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
    config.sceneProfile = sceneProfileFromString(settings.sceneProfile);
    config.depthFilterMode = depthFilterModeFromString(settings.depthFilterMode);
    config.adaptiveDepthFilterMode = settings.depthFilterMode.trimmed().compare(
        QStringLiteral("auto"), Qt::CaseInsensitive) == 0;
    config.saveIntermediatePyramidLevels = settings.saveIntermediatePyramidLevels;
    config.enableTargetedGapRecovery = settings.enableTargetedGapRecovery;
    config.targetedGapRecoverySourceCount =
        settings.targetedGapRecoverySourceCount;
    config.targetedGapRecoveryHypothesisCount =
        settings.targetedGapRecoveryHypothesisCount;
    config.targetedGapRecoveryConfidence =
        settings.targetedGapRecoveryConfidence;
    config.targetedGapRecoveryPriorRelativeDifference =
        settings.targetedGapRecoveryPriorRelativeDifference;
    config.targetedGapRecoveryConsensusInverseDepthSpread =
        settings.targetedGapRecoveryConsensusInverseDepthSpread;
    config.targetedGapRecoveryConsensusPriorRelativeDifference =
        settings.targetedGapRecoveryConsensusPriorRelativeDifference;
    config.enableTargetedGapSurfacePrior =
        settings.enableTargetedGapSurfacePrior;
    config.targetedGapSurfacePriorMaximumAnchorSpread =
        settings.targetedGapSurfacePriorMaximumAnchorSpread;
    config.targetedGapSurfacePriorMaximumFitResidual =
        settings.targetedGapSurfacePriorMaximumFitResidual;
    config.targetedGapRecoveryMaximumPriorDistancePixels =
        settings.targetedGapRecoveryMaximumPriorDistancePixels;
    config.enablePostConsistencyResidualReestimation =
        settings.enablePostConsistencyResidualReestimation;
    config.postConsistencyResidualSourceCount =
        settings.postConsistencyResidualSourceCount;
    config.postConsistencyResidualConfidence =
        settings.postConsistencyResidualConfidence;
    config.postConsistencyResidualMaximumLayerSpread =
        settings.postConsistencyResidualMaximumLayerSpread;
    config.postConsistencyResidualMaximumPriorRadius =
        settings.postConsistencyResidualMaximumPriorRadius;
    config.enableTwoSourceCrossViewGrowth =
        settings.enableTwoSourceCrossViewGrowth;
    config.twoSourceGrowthDistancePixels =
        settings.twoSourceGrowthDistancePixels;
    config.twoSourceGrowthInverseDepthSpread =
        settings.twoSourceGrowthInverseDepthSpread;
    config.twoSourceGrowthNormalAngleDegrees =
        settings.twoSourceGrowthNormalAngleDegrees;
    config.twoSourceGrowthMaximumComponentArea =
        settings.twoSourceGrowthMaximumComponentArea;

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
        settings.value(QStringLiteral("processingDevice")).toString(
            settings.value(QStringLiteral("pointCloudBackend")).toString(
                settings.value(QStringLiteral("point_cloud_backend")).toString(
                    QStringLiteral("auto")))));
    return parsed;
}

} // namespace xjw::core::project
