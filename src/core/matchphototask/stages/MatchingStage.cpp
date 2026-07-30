#include "lightglue/LightGlueFeatureBudget.h"
#include "LightGluePairBatch.h"
#include "LightGlueMatcher.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosRuntime.h"
#include "MatchingStage.h"
#include "io/PathIO.h"

#include <QDir>

#include <algorithm>
#include <exception>
#include <iterator>

namespace xjw
{
namespace matchphotos
{
namespace
{

MatchPhotosStageReport makeMatchingReport(MatchPhotosStageStatus status,
                                          const QString &message,
                                          int itemCount = 0)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("matching");
    report.displayName = QStringLiteral("两两匹配");
    report.status = status;
    report.message = message;
    report.itemCount = itemCount;
    return report;
}

} // namespace

MatchPhotosStageReport MatchingStage::run(const MatchPhotosContext &context,
                                          const MatchPhotosOptions &options,
                                          const MatchPhotosAlgorithmPlan &algorithmPlan,
                                          const PairSelectionResult &pairSelection,
                                          std::vector<MatchPhotosMatchRecord> *matchRecords) const
{
    if (options.planOnly)
    {
        return makeMatchingReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("plan-only 模式，跳过两两匹配"),
                                  static_cast<int>(pairSelection.candidates.size()));
    }

    if (algorithmPlan.matcherAlgorithm.compare(QStringLiteral("lightglue"), Qt::CaseInsensitive) != 0 ||
        algorithmPlan.featureAlgorithm.compare(QStringLiteral("sift"), Qt::CaseInsensitive) != 0)
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("连接点匹配当前只支持 SIFT + LightGlue"),
                                  static_cast<int>(pairSelection.candidates.size()));
    }

    if (pairSelection.candidates.empty())
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("没有可用于匹配的影像对"));
    }

    QDir matchDir(matchPhotosMatchDirectory(context));
    if (!matchDir.exists() && !matchDir.mkpath(QStringLiteral(".")))
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("无法创建匹配目录: %1").arg(matchDir.path()));
    }

    bool useCuda = false;
    QString modelName;
    QString modelPath = resolveLightGlueModelPath(algorithmPlan, options, &useCuda, &modelName);
    if (modelPath.isEmpty())
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("未找到 LightGlue SIFT 模型，请检查 resources/models"));
    }
    if (useCuda && !torch::cuda::is_available())
    {
        MatchPhotosOptions cpuOptions = options;
        cpuOptions.device = ComputeDevice::Cpu;
        modelPath = resolveLightGlueModelPath(algorithmPlan, cpuOptions, &useCuda, &modelName);
        if (modelPath.isEmpty())
        {
            return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                      QStringLiteral("CUDA 不可用，且未找到 CPU 版 LightGlue SIFT 模型"));
        }
    }

    xjw::feature_match::LightGlueConfig lightGlueConfig;
    lightGlueConfig.matcherModelPath = xjw::common::io::toUtf8Path(modelPath);
    lightGlueConfig.useCuda = useCuda;
    lightGlueConfig.cudaDevice = options.cudaDevice;

    const MatchPhotosGpuMemoryInfo gpuMemory =
        useCuda ? queryMatchPhotosGpuMemory(options.cudaDevice)
                : MatchPhotosGpuMemoryInfo{};
    xjw::feature_match::LightGlueGpuMemoryInfo budgetMemory;
    budgetMemory.available = gpuMemory.available;
    budgetMemory.freeBytes = gpuMemory.freeBytes;
    budgetMemory.totalBytes = gpuMemory.totalBytes;
    budgetMemory.deviceIndex = gpuMemory.deviceIndex;
    const int primaryKeypointBudget = xjw::feature_match::resolveLightGlueKeypointBudget(
        algorithmPlan.featureAlgorithm,
        algorithmPlan.matcherAlgorithm,
        useCuda,
        algorithmPlan.maxKeypoints,
        budgetMemory);
    const float effectiveMatchThreshold = xjw::feature_match::resolveLightGlueMatchThreshold(
        algorithmPlan.featureAlgorithm,
        algorithmPlan.matcherAlgorithm,
        useCuda,
        options.matchThreshold,
        primaryKeypointBudget,
        budgetMemory);
    lightGlueConfig.scoreThreshold = effectiveMatchThreshold;

    const int totalPairs = static_cast<int>(pairSelection.candidates.size());
    const LightGlueParallelismDecision parallelism =
        resolveLightGlueParallelism(options.cudaParallelPairs,
                                    totalPairs,
                                    useCuda,
                                    primaryKeypointBudget,
                                    gpuMemory);

    reportMatchPhotosProgress(context,
                              QStringLiteral("matching"),
                              QStringLiteral(
                                  "SIFT + LightGlue 两两匹配：准备处理 %1 对，"
                                  "并发 %2%3")
                                  .arg(totalPairs)
                                  .arg(parallelism.effectiveWorkers)
                                  .arg(parallelism.reason.isEmpty()
                                           ? QString()
                                           : QStringLiteral("（%1）")
                                                 .arg(parallelism.reason)),
                              0,
                              totalPairs);

    LightGluePairBatchConfig batchConfig;
    batchConfig.matcherConfig = lightGlueConfig;
    batchConfig.gpuMemory = gpuMemory;
    batchConfig.primaryKeypointBudget = primaryKeypointBudget;
    batchConfig.requestedWorkers = options.cudaParallelPairs;
    batchConfig.effectiveWorkers = parallelism.effectiveWorkers;
    batchConfig.effectiveMatchThreshold = effectiveMatchThreshold;
    batchConfig.applyTiepointMask = shouldApplyMasksToTiepoints(options);

    LightGluePairBatchResult batch;
    try
    {
        batch = runLightGluePairBatch(
            context, options, algorithmPlan, pairSelection, batchConfig);
    }
    catch (const std::exception &e)
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("LightGlue 匹配失败: %1").arg(QString::fromUtf8(e.what())),
                                  batch.matchedPairs);
    }

    if (batch.cancelled)
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("用户取消连接点匹配"),
                                  batch.matchedPairs);
    }
    if (matchRecords)
    {
        matchRecords->insert(matchRecords->end(),
                             std::make_move_iterator(batch.records.begin()),
                             std::make_move_iterator(batch.records.end()));
    }
    if (batch.matchedPairs == 0)
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  batch.fatalError.isEmpty()
                                      ? QStringLiteral("没有成功写入任何匹配结果，失败影像对 %1")
                                            .arg(batch.failedPairs)
                                      : QStringLiteral("LightGlue 匹配失败: %1")
                                            .arg(batch.fatalError),
                                  0);
    }

    return makeMatchingReport(MatchPhotosStageStatus::Completed,
                              QStringLiteral(
                                  "SIFT + LightGlue 匹配完成：%1 对，匹配点 %2，"
                                  "失败 %3，并发 %4，对应模型 %5%6")
                                  .arg(batch.matchedPairs)
                                  .arg(batch.totalMatches)
                                  .arg(batch.failedPairs)
                                  .arg(parallelism.effectiveWorkers)
                                  .arg(modelName)
                                  .arg(batch.usedSerialRecovery
                                           ? QStringLiteral("，已执行串行恢复（%1）")
                                                 .arg(batch.serialRecoveryReason)
                                           : QString()),
                              batch.matchedPairs);
}

} // namespace matchphotos
} // namespace xjw
