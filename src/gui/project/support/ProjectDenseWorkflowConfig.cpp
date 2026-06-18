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
    parsed.minConsistentViews = settings.value(QStringLiteral("minConsistentViews")).toInt(2);
    parsed.depthConsistency = static_cast<float>(settings.value(QStringLiteral("depthConsistency")).toDouble(2.0));
    parsed.maxReprojError = static_cast<float>(settings.value(QStringLiteral("maxReprojError")).toDouble(2.0));
    parsed.processingDevice = processingDeviceFromString(
        settings.value(QStringLiteral("processingDevice")).toString(QStringLiteral("auto")));
    parsed.pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);
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
    config.patchMatch.geomConsistencyMaxErr = settings.maxReprojError;
    config.fusion.confidenceThresh = settings.fusionMinConfidence;
    config.fusion.minConsistentViews = std::max(1, settings.minConsistentViews);
    config.fusion.relDepthThresh = 0.05f;
    config.fusion.pixelThresh = settings.depthConsistency;

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
    parsed.normalsEnabled = settings.value(QStringLiteral("normalsEnabled")).toBool(true);
    parsed.normalK = settings.value(QStringLiteral("normalK")).toInt(30);
    parsed.smoothNormals = settings.value(QStringLiteral("smoothIter")).toInt(0) > 0;
    parsed.threads = qMax(1, settings.value(QStringLiteral("threads")).toInt(8));
    parsed.processingDevice = processingDeviceFromString(
        settings.value(QStringLiteral("processingDevice")).toString(QStringLiteral("auto")));
    return parsed;
}

} // namespace xjw::gui::project
