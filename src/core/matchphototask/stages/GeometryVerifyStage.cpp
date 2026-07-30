#include "GeometryVerifyStage.h"

// Avoid Qt keyword macros rewriting LibTorch's slots() member name.
#ifdef slots
#undef slots
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#endif
#ifdef signals
#undef signals
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#endif
#ifdef emit
#undef emit
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#endif

#include "FeatureData.h"

#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#define slots Q_SLOTS
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#endif
#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#define signals Q_SIGNALS
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#endif
#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#define emit Q_EMIT
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#endif

#include "FeatureFileIO.h"
#include "MatchFileIO.h"
#include "MatchGeometryFilter.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosRuntime.h"
#include "SiftLightGlueRecovery.h"

#include <QElapsedTimer>
#include <QSet>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace xjw
{
namespace matchphotos
{

bool passesGeometryQualityGate(int rawMatchCount,
                               int inlierCount,
                               int minimumInliers)
{
    if (inlierCount < minimumInliers)
    {
        return false;
    }

    // 64 个内点已能稳定约束基础矩阵；低于此数量时，重复结构可能使
    // USAC 找到少量但自洽的伪模型，因此还要求至少 70% 的原始匹配支持它。
    constexpr int strongSupportInliers = 64;
    constexpr double minimumWeakSupportRatio = 0.70;
    if (inlierCount >= strongSupportInliers || rawMatchCount <= 0)
    {
        return true;
    }

    const double inlierRatio = static_cast<double>(inlierCount) /
        static_cast<double>(rawMatchCount);
    return inlierRatio >= minimumWeakSupportRatio;
}

namespace
{

MatchPhotosStageReport makeGeometryReport(MatchPhotosStageStatus status,
                                          const QString &message,
                                          int itemCount = 0)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("geometry_verify");
    report.displayName = QStringLiteral("几何验证");
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

void applyVerifiedMatch(MatchPhotosMatchRecord *record,
                        const xjw::feature_match::MatchResult &filteredMatch,
                        int rawMatchCount,
                        int minimumInliers)
{
    if (!record)
    {
        return;
    }

    record->geometricInlierCount = filteredMatch.numMatches;
    record->passedGeometry = passesGeometryQualityGate(rawMatchCount,
                                                       filteredMatch.numMatches,
                                                       minimumInliers);
    record->inlierIndexPairs.clear();
    if (record->passedGeometry)
    {
        record->inlierIndexPairs.reserve(filteredMatch.cvMatches.size());
        for (const cv::DMatch &match : filteredMatch.cvMatches)
        {
            record->inlierIndexPairs.push_back({match.queryIdx, match.trainIdx});
        }
    }
    record->settings[QStringLiteral("geometry_verified")] = record->passedGeometry;
    record->settings[QStringLiteral("geometry_raw_matches")] = rawMatchCount;
    record->settings[QStringLiteral("geometric_inliers")] = record->geometricInlierCount;
    record->settings[QStringLiteral("geometry_inlier_ratio")] = rawMatchCount > 0
        ? static_cast<double>(record->geometricInlierCount) / static_cast<double>(rawMatchCount)
        : 0.0;
}

xjw::feature_match::MatchResult filterGeometry(
    const xjw::feature_match::MatchResult &rawMatch,
    const xjw::feature_extractors::FeatureData &feature0,
    const xjw::feature_extractors::FeatureData &feature1,
    const xjw::feature_match::OutlierFilterConfig &filterConfig)
{
    int inlierCount = 0;
    xjw::feature_match::MatchResult filtered = xjw::feature_match::MatchGeometryFilter::filter(
        rawMatch, feature0.keypoints, feature1.keypoints, filterConfig, &inlierCount);
    normalizeMatchResult(&filtered, feature0.size(), feature1.size());
    return filtered;
}

struct PrimaryGeometryOutcome
{
    bool processed = false;
    bool success = false;
    xjw::feature_match::MatchResult rawMatch;
    xjw::feature_match::MatchResult filteredMatch;
    qint64 featureReadMs = 0;
    qint64 matchReadMs = 0;
    qint64 filterMs = 0;
    qint64 totalMs = 0;
};

PrimaryGeometryOutcome verifyPrimaryGeometry(
    const MatchPhotosMatchRecord &record,
    const xjw::feature_match::OutlierFilterConfig &filterConfig)
{
    PrimaryGeometryOutcome outcome;
    outcome.processed = true;
    QElapsedTimer totalTimer;
    QElapsedTimer phaseTimer;
    totalTimer.start();

    const QString feature0Path =
        record.settings.value(QStringLiteral("feature0_path")).toString();
    const QString feature1Path =
        record.settings.value(QStringLiteral("feature1_path")).toString();
    QString image0Name;
    QString image1Name;
    xjw::feature_extractors::FeatureData feature0;
    xjw::feature_extractors::FeatureData feature1;

    phaseTimer.start();
    const bool featureRead =
        FeatureFileIO::readGeometryData(feature0Path, image0Name, feature0) &&
        FeatureFileIO::readGeometryData(feature1Path, image1Name, feature1);
    outcome.featureReadMs = phaseTimer.elapsed();

    phaseTimer.restart();
    const bool matchRead = featureRead &&
        xjw::feature_match::readIndexedMatchFile(
            record.matchPath,
            image0Name,
            image1Name,
            outcome.rawMatch);
    outcome.matchReadMs = phaseTimer.elapsed();
    if (!featureRead || !matchRead)
    {
        outcome.totalMs = totalTimer.elapsed();
        return outcome;
    }

    normalizeMatchResult(&outcome.rawMatch, feature0.size(), feature1.size());
    phaseTimer.restart();
    outcome.filteredMatch =
        filterGeometry(outcome.rawMatch, feature0, feature1, filterConfig);
    outcome.filterMs = phaseTimer.elapsed();
    outcome.totalMs = totalTimer.elapsed();
    outcome.success = true;
    return outcome;
}

} // namespace

MatchPhotosStageReport GeometryVerifyStage::run(const MatchPhotosContext &context,
                                                const MatchPhotosOptions &options,
                                                std::vector<MatchPhotosMatchRecord> *matchRecords) const
{
    if (options.planOnly)
    {
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("plan-only 模式，跳过几何验证"));
    }
    if (!options.enableGeometryVerification)
    {
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("几何验证已禁用"));
    }
    if (!matchRecords || matchRecords->empty())
    {
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("没有可用于几何验证的匹配结果"));
    }

    xjw::feature_match::OutlierFilterConfig filterConfig;
    filterConfig.method = xjw::feature_match::OutlierMethod::FundamentalUsacMagsac;
    filterConfig.reprojThreshold = options.geometryReprojThreshold;
    filterConfig.minInliers = options.geometryMinInliers;
    const MatchPhotosAlgorithmPlan plan = MatchPhotosAlgorithmSelector::select(options);

    int passedPairs = 0;
    int failedPairs = 0;
    int totalInliers = 0;
    int denseRecoveredPairs = 0;
    int connectivityRecoveredPairs = 0;
    qint64 totalFeatureReadMs = 0;
    qint64 totalMatchReadMs = 0;
    qint64 totalFilterMs = 0;
    qint64 totalRecoveryMs = 0;
    QElapsedTimer stageTimer;
    stageTimer.start();
    const int totalPairs = static_cast<int>(matchRecords->size());
    const int geometryWorkers = resolveGeometryVerificationWorkers(
        totalPairs,
        std::thread::hardware_concurrency());
    std::vector<PrimaryGeometryOutcome> primaryOutcomes(
        static_cast<std::size_t>(totalPairs));
    std::atomic_int nextIndex{0};
    std::atomic_int activeWorkers{geometryWorkers};
    std::atomic_bool stop{false};
    std::mutex completionMutex;
    std::condition_variable completionCondition;
    std::deque<int> completionQueue;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(geometryWorkers));

    for (int workerIndex = 0; workerIndex < geometryWorkers; ++workerIndex)
    {
        workers.emplace_back([&]()
        {
            while (!stop.load())
            {
                const int index = nextIndex.fetch_add(1);
                if (index >= totalPairs)
                {
                    break;
                }
                if (shouldCancelMatchPhotos(context))
                {
                    stop.store(true);
                    break;
                }

                try
                {
                    primaryOutcomes[static_cast<std::size_t>(index)] =
                        verifyPrimaryGeometry(
                            (*matchRecords)[static_cast<std::size_t>(index)],
                            filterConfig);
                }
                catch (const std::exception &)
                {
                    primaryOutcomes[static_cast<std::size_t>(index)].processed = true;
                }
                {
                    std::lock_guard<std::mutex> lock(completionMutex);
                    completionQueue.push_back(index);
                }
                completionCondition.notify_one();
            }
            activeWorkers.fetch_sub(1);
            completionCondition.notify_one();
        });
    }

    int computedPairs = 0;
    int lastReportedPairs = -1;
    while (true)
    {
        std::unique_lock<std::mutex> lock(completionMutex);
        completionCondition.wait_for(
            lock,
            std::chrono::milliseconds(100),
            [&]()
            {
                return !completionQueue.empty() ||
                    activeWorkers.load() == 0 ||
                    shouldCancelMatchPhotos(context);
            });
        if (shouldCancelMatchPhotos(context))
        {
            stop.store(true);
        }
        while (!completionQueue.empty())
        {
            completionQueue.pop_front();
            ++computedPairs;
        }
        const bool allWorkersFinished =
            activeWorkers.load() == 0 && completionQueue.empty();
        lock.unlock();
        if (computedPairs > 0 && computedPairs != lastReportedPairs)
        {
            reportMatchPhotosProgress(
                context,
                QStringLiteral("geometry"),
                QStringLiteral("几何验证计算：%1/%2，CPU 并发 %3")
                    .arg(computedPairs)
                    .arg(totalPairs)
                    .arg(geometryWorkers),
                computedPairs,
                totalPairs);
            lastReportedPairs = computedPairs;
        }
        if (allWorkersFinished)
        {
            break;
        }
    }

    for (std::thread &worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    if (shouldCancelMatchPhotos(context))
    {
        return makeGeometryReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("用户取消几何验证"),
                                  passedPairs);
    }

    for (int index = 0; index < totalPairs; ++index)
    {
        MatchPhotosMatchRecord &record =
            (*matchRecords)[static_cast<std::size_t>(index)];
        PrimaryGeometryOutcome &outcome =
            primaryOutcomes[static_cast<std::size_t>(index)];
        totalFeatureReadMs += outcome.featureReadMs;
        totalMatchReadMs += outcome.matchReadMs;
        totalFilterMs += outcome.filterMs;
        if (!outcome.success)
        {
            ++failedPairs;
            continue;
        }

        const QString feature0Path =
            record.settings.value(QStringLiteral("feature0_path")).toString();
        const QString feature1Path =
            record.settings.value(QStringLiteral("feature1_path")).toString();
        const int primaryInliers = outcome.filteredMatch.numMatches;
        applyVerifiedMatch(&record,
                           outcome.filteredMatch,
                           outcome.rawMatch.numMatches,
                           options.geometryMinInliers);

        qint64 recoveryMs = 0;
        if (shouldAugmentDenseSiftPair(
                options, record, outcome.rawMatch.numMatches))
        {
            xjw::feature_extractors::FeatureData recoveryFeature0;
            xjw::feature_extractors::FeatureData recoveryFeature1;
            xjw::feature_match::MatchResult recoveredRaw;
            xjw::feature_match::MatchResult recoveredFiltered;
            bool usedCuda = false;
            QString image0Name;
            QString image1Name;
            QElapsedTimer phaseTimer;
            phaseTimer.restart();
            const bool recoveryFeaturesRead =
                FeatureFileIO::readData(feature0Path, image0Name, recoveryFeature0) &&
                FeatureFileIO::readData(feature1Path, image1Name, recoveryFeature1);
            const bool recovered = recoveryFeaturesRead &&
                runFullSiftRecovery(options,
                                    recoveryFeature0,
                                    recoveryFeature1,
                                    filterConfig,
                                    &recoveredRaw,
                                    &recoveredFiltered,
                                    &usedCuda);
            recoveryMs = phaseTimer.elapsed();
            totalRecoveryMs += recoveryMs;
            if (recovered &&
                recoveredFiltered.numMatches > outcome.filteredMatch.numMatches &&
                persistRecoveredMatch(options,
                                      plan,
                                      recoveryFeature0,
                                      recoveryFeature1,
                                      recoveredRaw,
                                      QStringLiteral("dense_overlap_budget_truncation"),
                                      outcome.rawMatch.numMatches,
                                      primaryInliers,
                                      recoveredFiltered.numMatches,
                                      usedCuda,
                                      &record))
            {
                applyVerifiedMatch(&record,
                                   recoveredFiltered,
                                   recoveredRaw.numMatches,
                                   options.geometryMinInliers);
                ++denseRecoveredPairs;
            }
        }

        if (record.passedGeometry)
        {
            ++passedPairs;
            totalInliers += record.geometricInlierCount;
        }
        else
        {
            ++failedPairs;
        }
        record.settings[QStringLiteral("geometry_reproj_threshold")] = options.geometryReprojThreshold;
        record.settings[QStringLiteral("geometry_feature_read_mode")] =
            QStringLiteral("keypoints_only");
        record.settings[QStringLiteral("geometry_feature_read_ms")] =
            static_cast<double>(outcome.featureReadMs);
        record.settings[QStringLiteral("geometry_match_read_ms")] =
            static_cast<double>(outcome.matchReadMs);
        record.settings[QStringLiteral("geometry_filter_ms")] =
            static_cast<double>(outcome.filterMs);
        record.settings[QStringLiteral("geometry_pair_total_ms")] =
            static_cast<double>(outcome.totalMs + recoveryMs);
        record.settings[QStringLiteral("geometry_parallel_pairs_effective")] =
            geometryWorkers;
        record.settings[QStringLiteral("geometry_gpu_used")] = false;
    }

    // 先使用 LightGlue 形成实际匹配图；只对跨连通分量的失败边做全量 SIFT 恢复。
    // 每恢复一条边就重算分量，图已连通后立即停止，避免对所有失败像对运行昂贵全量匹配。
    QSet<int> attemptedRecovery;
    while (true)
    {
        const std::vector<int> candidates =
            disconnectedRecoveryCandidates(*matchRecords, attemptedRecovery);
        if (candidates.empty())
        {
            break;
        }

        bool recoveredOne = false;
        for (const int index : candidates)
        {
            attemptedRecovery.insert(index);
            MatchPhotosMatchRecord &record = (*matchRecords)[static_cast<std::size_t>(index)];
            const QString feature0Path = record.settings.value(QStringLiteral("feature0_path")).toString();
            const QString feature1Path = record.settings.value(QStringLiteral("feature1_path")).toString();
            QString image0Name;
            QString image1Name;
            xjw::feature_extractors::FeatureData feature0;
            xjw::feature_extractors::FeatureData feature1;
            if (!FeatureFileIO::readData(feature0Path, image0Name, feature0) ||
                !FeatureFileIO::readData(feature1Path, image1Name, feature1))
            {
                continue;
            }

            xjw::feature_match::MatchResult recoveredRaw;
            xjw::feature_match::MatchResult recoveredFiltered;
            bool usedCuda = false;
            const int primaryRawCount = record.matchCount;
            const int primaryInlierCount = record.geometricInlierCount;
            if (!runFullSiftRecovery(options,
                                     feature0,
                                     feature1,
                                     filterConfig,
                                     &recoveredRaw,
                                     &recoveredFiltered,
                                     &usedCuda) ||
                recoveredFiltered.numMatches < options.geometryMinInliers ||
                !persistRecoveredMatch(options,
                                       plan,
                                       feature0,
                                       feature1,
                                       recoveredRaw,
                                       QStringLiteral("disconnected_match_graph"),
                                       primaryRawCount,
                                       primaryInlierCount,
                                       recoveredFiltered.numMatches,
                                       usedCuda,
                                       &record))
            {
                continue;
            }

            applyVerifiedMatch(&record,
                               recoveredFiltered,
                               recoveredRaw.numMatches,
                               options.geometryMinInliers);
            record.settings[QStringLiteral("geometry_reproj_threshold")] = options.geometryReprojThreshold;
            ++passedPairs;
            --failedPairs;
            totalInliers += record.geometricInlierCount;
            ++connectivityRecoveredPairs;
            recoveredOne = true;
            break;
        }
        if (!recoveredOne)
        {
            break;
        }
    }

    return makeGeometryReport(MatchPhotosStageStatus::Completed,
                              QStringLiteral("几何验证完成：通过 %1 对，失败 %2 对，内点 %3，"
                                             "强重叠增强 %4 对，断图恢复 %5 对；"
                                             "关键点读取 %6 ms，匹配读取 %7 ms，"
                                             "USAC %8 ms，恢复 %9 ms，CPU 并发 %10，"
                                             "总计 %11 ms")
                                  .arg(passedPairs)
                                  .arg(failedPairs)
                                  .arg(totalInliers)
                                  .arg(denseRecoveredPairs)
                                  .arg(connectivityRecoveredPairs)
                                  .arg(totalFeatureReadMs)
                                  .arg(totalMatchReadMs)
                                  .arg(totalFilterMs)
                                  .arg(totalRecoveryMs)
                                  .arg(geometryWorkers)
                                  .arg(stageTimer.elapsed()),
                              passedPairs);
}

} // namespace matchphotos
} // namespace xjw
