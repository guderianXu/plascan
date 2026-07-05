#include "FeatureData.h"
#include "FeatureFileIO.h"
#include "LightGlueMatcher.h"
#include "MatchFileIO.h"
#include "MatchPhotosRuntime.h"
#include "MatchingStage.h"

#include <QDir>
#include <QFileInfo>

#include <exception>

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

void normalizeMatchResult(xjw::feature_match::MatchResult *matchResult,
                          int keypointCount0,
                          int keypointCount1)
{
    if (!matchResult)
    {
        return;
    }

    if (matchResult->matches0.empty() && !matchResult->cvMatches.empty())
    {
        matchResult->buildIndicesFromCvMatches(keypointCount0, keypointCount1);
    }
    if (matchResult->cvMatches.empty() && !matchResult->matches0.empty())
    {
        matchResult->buildCvMatchesFromIndices();
    }
    matchResult->numMatches = static_cast<int>(matchResult->cvMatches.size());
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
    lightGlueConfig.matcherModelPath = modelPath.toStdString();
    lightGlueConfig.useCuda = useCuda;
    lightGlueConfig.cudaDevice = options.cudaDevice;
    lightGlueConfig.scoreThreshold = options.matchThreshold;

    int matchedPairs = 0;
    int failedPairs = 0;
    int totalMatches = 0;

    try
    {
        xjw::feature_match::LightGlueMatcher matcher(lightGlueConfig);

        for (const PairCandidate &candidate : pairSelection.candidates)
        {
            if (shouldCancelMatchPhotos(context))
            {
                return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                          QStringLiteral("用户取消连接点匹配"),
                                          matchedPairs);
            }

            QString errorMessage;
            ResolvedImagePair pair;
            if (!resolveMatchPhotosPair(context, candidate, &pair, &errorMessage))
            {
                ++failedPairs;
                advanceMatchPhotosProgress(context);
                continue;
            }

            const QString feature0Path = matchPhotosFeaturePath(context, pair.image0Path, algorithmPlan);
            const QString feature1Path = matchPhotosFeaturePath(context, pair.image1Path, algorithmPlan);
            QString image0Name;
            QString image1Name;
            xjw::feature_extractors::FeatureData feature0;
            xjw::feature_extractors::FeatureData feature1;
            if (!FeatureFileIO::readData(feature0Path, image0Name, feature0) ||
                !FeatureFileIO::readData(feature1Path, image1Name, feature1))
            {
                ++failedPairs;
                advanceMatchPhotosProgress(context);
                continue;
            }

            xjw::feature_match::MatchResult matchResult = matcher.match(feature0, feature1);
            normalizeMatchResult(&matchResult, feature0.size(), feature1.size());

            const QString matchPath = matchPhotosMatchPath(context,
                                                           pair.image0Path,
                                                           pair.image1Path,
                                                           algorithmPlan);
            const QString sidecarPath = matchPath + QStringLiteral(".json");
            if (!xjw::feature_match::writeIndexedMatchFile(matchPath,
                                                           QFileInfo(pair.image0Path).completeBaseName(),
                                                           QFileInfo(pair.image1Path).completeBaseName(),
                                                           matchResult) ||
                !writeMatchPhotosSidecar(sidecarPath,
                                         pair,
                                         feature0Path,
                                         feature1Path,
                                         matchPath,
                                         feature0,
                                         feature1,
                                         matchResult,
                                         algorithmPlan,
                                         options))
            {
                ++failedPairs;
                advanceMatchPhotosProgress(context);
                continue;
            }

            if (matchRecords)
            {
                matchRecords->push_back(
                    MatchPhotosMatchRecord{pair.image0Path,
                                           pair.image1Path,
                                           matchPath,
                                           sidecarPath,
                                           matchResult.numMatches,
                                           makeMatchRecordSettings(algorithmPlan,
                                                                   options,
                                                                   pair,
                                                                   feature0Path,
                                                                   feature1Path,
                                                                   matchPath,
                                                                   sidecarPath,
                                                                   matchResult.numMatches)});
            }
            ++matchedPairs;
            totalMatches += matchResult.numMatches;
            advanceMatchPhotosProgress(context);
        }
    }
    catch (const std::exception &e)
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("LightGlue 匹配失败: %1").arg(QString::fromUtf8(e.what())),
                                  matchedPairs);
    }

    if (matchedPairs == 0)
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("没有成功写入任何匹配结果，失败影像对 %1")
                                      .arg(failedPairs),
                                  0);
    }

    return makeMatchingReport(MatchPhotosStageStatus::Completed,
                              QStringLiteral("SIFT + LightGlue 匹配完成：%1 对，匹配点 %2，失败 %3，对应模型 %4")
                                  .arg(matchedPairs)
                                  .arg(totalMatches)
                                  .arg(failedPairs)
                                  .arg(modelName),
                              matchedPairs);
}

} // namespace matchphotos
} // namespace xjw
