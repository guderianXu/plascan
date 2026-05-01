#include "ProjectDenseWorkflowConfig.h"

#include <QtGlobal>

#include <algorithm>
#include <cmath>

namespace xjw::gui::project {

DenseGenerationSettings denseGenerationSettingsFromJson(const QJsonObject &settings)
{
    DenseGenerationSettings parsed;
    parsed.atIndex = settings.value(QStringLiteral("at_index")).toInt(-1);
    parsed.outputDir = settings.value(QStringLiteral("output_dir")).toString();
    parsed.resScale = settings.value(QStringLiteral("resScale")).toDouble(0.5);
    parsed.iterations = settings.value(QStringLiteral("iterations")).toInt(6);
    parsed.threads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
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
    parsed.pipelineMode = settings.value(QStringLiteral("pipeline_mode")).toBool(false);
    return parsed;
}

xjw::mvs::DepthGenConfig buildDepthGenConfig(const DenseGenerationSettings &settings,
                                             int viewCount)
{
    xjw::mvs::DepthGenConfig config;
    config.numSourceViews = std::min(settings.minViews, viewCount - 1);
    config.cpuWorkerCount = std::max(1, settings.threads);
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
        // 两视图立体对：全分辨率 + 更多迭代 + 极宽松融合阈值，最大化密集点云覆盖率
        config.patchMatch.downsampleFactor = 1;          // 全分辨率，不降采样
        config.patchMatch.numIterations = std::max(config.patchMatch.numIterations, 32);  // 增加到32次迭代
        config.patchMatch.confidenceThresh = 0.0001f;    // 极度放宽到 0.0001
        config.patchMatch.geomConsistency = false;        // 两视图无法做几何一致性检查
        config.fusion.confidenceThresh = 0.0001f;        // 融合阈值同步放宽
        config.fusion.minConsistentViews = 1;
        config.fusion.relDepthThresh = 0.40f;            // 相对深度阈值放宽到 40%
        config.fusion.pixelThresh = 10.0f;               // 像素阈值放宽到 10 像素
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
    return parsed;
}

} // namespace xjw::gui::project