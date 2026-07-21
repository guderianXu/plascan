#include "FeatureData.h"
#include "FeatureFileIO.h"
#include "lightglue/LightGlueFeatureBudget.h"
#include "LightGlueMatcher.h"
#include "MatchFileIO.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosRuntime.h"
#include "MatchingStage.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonObject>
#include <QMap>

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
    lightGlueConfig.matcherModelPath = xjw::common::io::toUtf8Path(modelPath);
    lightGlueConfig.useCuda = useCuda;
    lightGlueConfig.cudaDevice = options.cudaDevice;

    const int primaryKeypointBudget = xjw::feature_match::resolveLightGlueKeypointBudget(
        algorithmPlan.featureAlgorithm,
        algorithmPlan.matcherAlgorithm,
        useCuda,
        algorithmPlan.maxKeypoints);
    const float effectiveMatchThreshold = xjw::feature_match::resolveLightGlueMatchThreshold(
        algorithmPlan.featureAlgorithm,
        algorithmPlan.matcherAlgorithm,
        useCuda,
        options.matchThreshold,
        primaryKeypointBudget,
        xjw::feature_match::LightGlueGpuMemoryInfo{});
    lightGlueConfig.scoreThreshold = effectiveMatchThreshold;

    int matchedPairs = 0;
    int failedPairs = 0;
    int totalMatches = 0;
    const bool applyTiepointMask = shouldApplyMasksToTiepoints(options);
    QMap<QString, cv::Mat> maskCache;
    const int totalPairs = static_cast<int>(pairSelection.candidates.size());

    reportMatchPhotosProgress(context,
                              QStringLiteral("matching"),
                              QStringLiteral("SIFT + LightGlue 两两匹配：准备处理 %1 对%2")
                                  .arg(totalPairs)
                                  .arg(applyTiepointMask ? QStringLiteral("，按蒙版过滤连接点")
                                                         : QString()),
                              0,
                              totalPairs);

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
                reportMatchPhotosProgress(context,
                                          QStringLiteral("matching"),
                                          QStringLiteral("SIFT + LightGlue 两两匹配：%1/%2，成功 %3 对，失败 %4 对")
                                              .arg(matchedPairs + failedPairs)
                                              .arg(totalPairs)
                                              .arg(matchedPairs)
                                              .arg(failedPairs),
                                          matchedPairs + failedPairs,
                                          totalPairs);
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
                reportMatchPhotosProgress(context,
                                          QStringLiteral("matching"),
                                          QStringLiteral("SIFT + LightGlue 两两匹配：%1/%2，成功 %3 对，失败 %4 对")
                                              .arg(matchedPairs + failedPairs)
                                              .arg(totalPairs)
                                              .arg(matchedPairs)
                                              .arg(failedPairs),
                                          matchedPairs + failedPairs,
                                          totalPairs);
                continue;
            }

            xjw::feature_match::MatchResult matchResult;
            QJsonObject matchDiagnostics;
            bool matched = false;
            QString matchError;
            for (int keypointBudget : xjw::feature_match::lightGlueRetryKeypointBudgets(primaryKeypointBudget))
            {
                try
                {
                    const xjw::feature_match::BudgetedFeatureData budgetedFeature0 =
                        xjw::feature_match::budgetFeatureDataForLightGlue(feature0, keypointBudget);
                    const xjw::feature_match::BudgetedFeatureData budgetedFeature1 =
                        xjw::feature_match::budgetFeatureDataForLightGlue(feature1, keypointBudget);
                    xjw::feature_match::MatchResult limitedMatch =
                        matcher.match(budgetedFeature0.features, budgetedFeature1.features);
                    normalizeMatchResult(&limitedMatch,
                                         budgetedFeature0.features.size(),
                                         budgetedFeature1.features.size());
                    matchResult = xjw::feature_match::remapLightGlueMatchResultToOriginal(
                        limitedMatch,
                        budgetedFeature0,
                        feature0.size(),
                        budgetedFeature1,
                        feature1.size());
                    normalizeMatchResult(&matchResult, feature0.size(), feature1.size());

                    matchDiagnostics[QStringLiteral("lightglue_keypoint_budget")] = keypointBudget;
                    matchDiagnostics[QStringLiteral("lightglue_used_keypoints0")] = budgetedFeature0.features.size();
                    matchDiagnostics[QStringLiteral("lightglue_used_keypoints1")] = budgetedFeature1.features.size();
                    matchDiagnostics[QStringLiteral("lightglue_limited_keypoints0")] = budgetedFeature0.limited;
                    matchDiagnostics[QStringLiteral("lightglue_limited_keypoints1")] = budgetedFeature1.limited;
                    matchDiagnostics[QStringLiteral("lightglue_effective_match_threshold")] =
                        static_cast<double>(effectiveMatchThreshold);
                    matched = true;
                    break;
                }
                catch (const std::exception &e)
                {
                    matchError = QString::fromUtf8(e.what());
                    if (keypointBudget <= 1024)
                    {
                        break;
                    }
                }
            }

            if (!matched)
            {
                ++failedPairs;
                advanceMatchPhotosProgress(context);
                Q_UNUSED(matchError)
                reportMatchPhotosProgress(context,
                                          QStringLiteral("matching"),
                                          QStringLiteral("SIFT + LightGlue 两两匹配：%1/%2，成功 %3 对，失败 %4 对")
                                              .arg(matchedPairs + failedPairs)
                                              .arg(totalPairs)
                                              .arg(matchedPairs)
                                              .arg(failedPairs),
                                          matchedPairs + failedPairs,
                                          totalPairs);
                continue;
            }
            normalizeMatchResult(&matchResult, feature0.size(), feature1.size());

            if (applyTiepointMask)
            {
                const QString mask0Path = maskPathForImage(context, pair.image0Path);
                const QString mask1Path = maskPathForImage(context, pair.image1Path);
                auto loadCachedMask = [&](const QString &imagePath,
                                          const QString &maskPath,
                                          const xjw::feature_extractors::FeatureData &feature) -> cv::Mat
                {
                    if (maskPath.isEmpty())
                    {
                        return cv::Mat();
                    }
                    if (!maskCache.contains(maskPath))
                    {
                        const cv::Size imageSize(feature.imageWidth, feature.imageHeight);
                        maskCache.insert(maskPath, loadMaskForImage(context, imagePath, imageSize));
                    }
                    return maskCache.value(maskPath);
                };

                const int unmaskedMatchCount = matchResult.numMatches;
                const cv::Mat mask0 = loadCachedMask(pair.image0Path, mask0Path, feature0);
                const cv::Mat mask1 = loadCachedMask(pair.image1Path, mask1Path, feature1);
                if (!mask0.empty() || !mask1.empty())
                {
                    matchResult = filterMatchResultByMasks(matchResult, feature0, feature1, mask0, mask1);
                    normalizeMatchResult(&matchResult, feature0.size(), feature1.size());
                }
                matchDiagnostics[QStringLiteral("mask0_path")] = mask0Path;
                matchDiagnostics[QStringLiteral("mask1_path")] = mask1Path;
                matchDiagnostics[QStringLiteral("mask_unfiltered_matches")] = unmaskedMatchCount;
                matchDiagnostics[QStringLiteral("mask_filtered_matches")] =
                    std::max(0, unmaskedMatchCount - matchResult.numMatches);
            }

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
                                         options,
                                         matchDiagnostics))
            {
                ++failedPairs;
                advanceMatchPhotosProgress(context);
                reportMatchPhotosProgress(context,
                                          QStringLiteral("matching"),
                                          QStringLiteral("SIFT + LightGlue 两两匹配：%1/%2，成功 %3 对，失败 %4 对")
                                              .arg(matchedPairs + failedPairs)
                                              .arg(totalPairs)
                                              .arg(matchedPairs)
                                              .arg(failedPairs),
                                          matchedPairs + failedPairs,
                                          totalPairs);
                continue;
            }

            if (matchRecords)
            {
                MatchPhotosMatchRecord record;
                record.image0Path = pair.image0Path;
                record.image1Path = pair.image1Path;
                record.matchPath = matchPath;
                record.sidecarPath = sidecarPath;
                record.matchCount = matchResult.numMatches;
                record.settings = makeMatchRecordSettings(algorithmPlan,
                                                          options,
                                                          pair,
                                                          feature0Path,
                                                          feature1Path,
                                                          matchPath,
                                                          sidecarPath,
                                                          matchResult.numMatches,
                                                          matchDiagnostics);
                matchRecords->push_back(std::move(record));
            }
            ++matchedPairs;
            totalMatches += matchResult.numMatches;
            advanceMatchPhotosProgress(context);
            reportMatchPhotosProgress(context,
                                      QStringLiteral("matching"),
                                      QStringLiteral("SIFT + LightGlue 两两匹配：%1/%2，成功 %3 对，失败 %4 对，累计匹配 %5")
                                          .arg(matchedPairs + failedPairs)
                                          .arg(totalPairs)
                                          .arg(matchedPairs)
                                          .arg(failedPairs)
                                          .arg(totalMatches),
                                      matchedPairs + failedPairs,
                                      totalPairs);
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
