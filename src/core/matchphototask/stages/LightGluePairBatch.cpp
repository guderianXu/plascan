#include "LightGluePairBatch.h"

#include "FeatureData.h"
#include "FeatureFileIO.h"
#include "LightGlueFeatureBudget.h"
#include "MatchFileIO.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosRuntime.h"

#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonObject>
#include <QMap>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#if defined(PLASCAN_TORCH_HAS_CUDA)
#  include <c10/cuda/CUDACachingAllocator.h>
#  include <c10/cuda/CUDAGuard.h>
#  include <c10/cuda/CUDAStream.h>
#endif

namespace xjw
{
namespace matchphotos
{
namespace
{

struct PairOutcome
{
    bool processed = false;
    bool success = false;
    bool cudaOutOfMemory = false;
    QString errorMessage;
    MatchPhotosMatchRecord record;
};

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

void releaseCudaCache()
{
#if defined(PLASCAN_TORCH_HAS_CUDA)
    try
    {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    catch (const std::exception &)
    {
    }
#endif
}

class LightGluePairWorker
{
public:
    LightGluePairWorker(const MatchPhotosContext &context,
                        const MatchPhotosOptions &options,
                        const MatchPhotosAlgorithmPlan &algorithmPlan,
                        const LightGluePairBatchConfig &config,
                        int workerIndex)
        : _context(context),
          _options(options),
          _algorithmPlan(algorithmPlan),
          _config(config),
          _workerIndex(workerIndex)
    {
#if defined(PLASCAN_TORCH_HAS_CUDA)
        if (config.matcherConfig.useCuda)
        {
            _cudaStream = c10::cuda::getStreamFromPool(
                false,
                std::max(0, config.matcherConfig.cudaDevice));
            c10::cuda::CUDAStreamGuard streamGuard(*_cudaStream);
            _matcher = std::make_unique<xjw::feature_match::LightGlueMatcher>(
                config.matcherConfig);
            return;
        }
#endif
        _matcher = std::make_unique<xjw::feature_match::LightGlueMatcher>(
            config.matcherConfig);
    }

    PairOutcome process(const PairCandidate &candidate,
                        bool serialRecovery = false,
                        bool serialOomRetry = false)
    {
        PairOutcome outcome;
        outcome.processed = true;
        QElapsedTimer totalTimer;
        totalTimer.start();

        QString resolveError;
        ResolvedImagePair pair;
        if (!resolveMatchPhotosPair(_context, candidate, &pair, &resolveError))
        {
            outcome.errorMessage = resolveError;
            return outcome;
        }

        const QString feature0Path =
            matchPhotosFeaturePath(_context, pair.image0Path, _algorithmPlan);
        const QString feature1Path =
            matchPhotosFeaturePath(_context, pair.image1Path, _algorithmPlan);
        QString image0Name;
        QString image1Name;
        xjw::feature_extractors::FeatureData feature0;
        xjw::feature_extractors::FeatureData feature1;

        QElapsedTimer phaseTimer;
        phaseTimer.start();
        if (!FeatureFileIO::readData(feature0Path, image0Name, feature0) ||
            !FeatureFileIO::readData(feature1Path, image1Name, feature1))
        {
            outcome.errorMessage = QStringLiteral("无法读取匹配对特征文件");
            return outcome;
        }
        const qint64 featureReadMs = phaseTimer.elapsed();

        xjw::feature_match::MatchResult matchResult;
        QJsonObject diagnostics;
        bool matched = false;
        bool retriedAfterOom = false;
        qint64 budgetAndInferenceMs = 0;
        QString matchError;
        for (int keypointBudget :
             xjw::feature_match::lightGlueRetryKeypointBudgets(_config.primaryKeypointBudget))
        {
            try
            {
                phaseTimer.restart();
                const xjw::feature_match::BudgetedFeatureData budgetedFeature0 =
                    xjw::feature_match::budgetFeatureDataForLightGlue(
                        feature0, keypointBudget);
                const xjw::feature_match::BudgetedFeatureData budgetedFeature1 =
                    xjw::feature_match::budgetFeatureDataForLightGlue(
                        feature1, keypointBudget);
                xjw::feature_match::MatchResult limitedMatch;
#if defined(PLASCAN_TORCH_HAS_CUDA)
                if (_cudaStream)
                {
                    c10::cuda::CUDAStreamGuard streamGuard(*_cudaStream);
                    limitedMatch = _matcher->match(
                        budgetedFeature0.features,
                        budgetedFeature1.features);
                }
                else
#endif
                {
                    limitedMatch = _matcher->match(
                        budgetedFeature0.features,
                        budgetedFeature1.features);
                }
                budgetAndInferenceMs += phaseTimer.elapsed();
                normalizeMatchResult(&limitedMatch,
                                     budgetedFeature0.features.size(),
                                     budgetedFeature1.features.size());
                matchResult =
                    xjw::feature_match::remapLightGlueMatchResultToOriginal(
                        limitedMatch,
                        budgetedFeature0,
                        feature0.size(),
                        budgetedFeature1,
                        feature1.size());
                normalizeMatchResult(&matchResult, feature0.size(), feature1.size());

                diagnostics[QStringLiteral("lightglue_keypoint_budget")] =
                    keypointBudget;
                diagnostics[QStringLiteral("lightglue_used_keypoints0")] =
                    budgetedFeature0.features.size();
                diagnostics[QStringLiteral("lightglue_used_keypoints1")] =
                    budgetedFeature1.features.size();
                diagnostics[QStringLiteral("lightglue_limited_keypoints0")] =
                    budgetedFeature0.limited;
                diagnostics[QStringLiteral("lightglue_limited_keypoints1")] =
                    budgetedFeature1.limited;
                diagnostics[QStringLiteral("lightglue_effective_match_threshold")] =
                    static_cast<double>(_config.effectiveMatchThreshold);
                matched = true;
                break;
            }
            catch (const std::exception &e)
            {
                matchError = QString::fromUtf8(e.what());
                const bool oom = isCudaOutOfMemoryError(matchError);
                outcome.cudaOutOfMemory = outcome.cudaOutOfMemory || oom;
                retriedAfterOom = retriedAfterOom || oom;
                if (keypointBudget <= 1024)
                {
                    break;
                }
            }
        }

        if (!matched)
        {
            outcome.errorMessage = matchError.isEmpty()
                ? QStringLiteral("LightGlue 未生成匹配结果")
                : matchError;
            return outcome;
        }

        qint64 maskFilterMs = 0;
        if (_config.applyTiepointMask)
        {
            phaseTimer.restart();
            const QString mask0Path = maskPathForImage(_context, pair.image0Path);
            const QString mask1Path = maskPathForImage(_context, pair.image1Path);
            const cv::Mat mask0 = loadMask(pair.image0Path, mask0Path, feature0);
            const cv::Mat mask1 = loadMask(pair.image1Path, mask1Path, feature1);
            const int unmaskedMatchCount = matchResult.numMatches;
            if (!mask0.empty() || !mask1.empty())
            {
                matchResult = filterMatchResultByMasks(
                    matchResult, feature0, feature1, mask0, mask1);
                normalizeMatchResult(&matchResult, feature0.size(), feature1.size());
            }
            diagnostics[QStringLiteral("mask0_path")] = mask0Path;
            diagnostics[QStringLiteral("mask1_path")] = mask1Path;
            diagnostics[QStringLiteral("mask_unfiltered_matches")] =
                unmaskedMatchCount;
            diagnostics[QStringLiteral("mask_filtered_matches")] =
                std::max(0, unmaskedMatchCount - matchResult.numMatches);
            maskFilterMs = phaseTimer.elapsed();
        }

        diagnostics[QStringLiteral("cuda_parallel_pairs_requested")] =
            _config.requestedWorkers;
        diagnostics[QStringLiteral("cuda_parallel_pairs_effective")] =
            _config.effectiveWorkers;
        diagnostics[QStringLiteral("cuda_worker_index")] = _workerIndex;
        diagnostics[QStringLiteral("cuda_serial_recovery")] = serialRecovery;
        diagnostics[QStringLiteral("cuda_oom_serial_retry")] = serialOomRetry;
        diagnostics[QStringLiteral("cuda_oom_budget_retry")] = retriedAfterOom;
        diagnostics[QStringLiteral("cuda_free_bytes_at_start")] =
            static_cast<double>(_config.gpuMemory.freeBytes);
        diagnostics[QStringLiteral("cuda_total_bytes")] =
            static_cast<double>(_config.gpuMemory.totalBytes);
        diagnostics[QStringLiteral("feature_read_ms")] =
            static_cast<double>(featureReadMs);
        diagnostics[QStringLiteral("lightglue_inference_ms")] =
            static_cast<double>(budgetAndInferenceMs);
        diagnostics[QStringLiteral("mask_filter_ms")] =
            static_cast<double>(maskFilterMs);

        const QString matchPath = matchPhotosMatchPath(
            _context, pair.image0Path, pair.image1Path, _algorithmPlan);
        const QString sidecarPath = matchPath + QStringLiteral(".json");
        phaseTimer.restart();
        const bool matchWritten = xjw::feature_match::writeIndexedMatchFile(
            matchPath,
            QFileInfo(pair.image0Path).completeBaseName(),
            QFileInfo(pair.image1Path).completeBaseName(),
            matchResult);
        const qint64 writeMs = phaseTimer.elapsed();
        if (!matchWritten)
        {
            outcome.errorMessage = QStringLiteral("无法写入匹配文件");
            return outcome;
        }

        // sidecar 必须包含最终的核心阶段耗时，因此先完成二进制匹配文件，
        // 再固定诊断字段并写元数据。sidecar 自身写盘时间不计入 pair_total_ms。
        diagnostics[QStringLiteral("result_write_ms")] =
            static_cast<double>(writeMs);
        diagnostics[QStringLiteral("pair_total_ms")] =
            static_cast<double>(totalTimer.elapsed());
        if (!writeMatchPhotosSidecar(sidecarPath,
                                     pair,
                                     feature0Path,
                                     feature1Path,
                                     matchPath,
                                     feature0,
                                     feature1,
                                     matchResult,
                                     _algorithmPlan,
                                     _options,
                                     diagnostics))
        {
            QFile::remove(matchPath);
            outcome.errorMessage = QStringLiteral("无法写入匹配 sidecar");
            return outcome;
        }

        MatchPhotosMatchRecord record;
        record.image0Path = pair.image0Path;
        record.image1Path = pair.image1Path;
        record.matchPath = matchPath;
        record.sidecarPath = sidecarPath;
        record.matchCount = matchResult.numMatches;
        record.settings = makeMatchRecordSettings(_algorithmPlan,
                                                  _options,
                                                  pair,
                                                  feature0Path,
                                                  feature1Path,
                                                  matchPath,
                                                  sidecarPath,
                                                  matchResult.numMatches,
                                                  diagnostics);
        outcome.record = std::move(record);
        outcome.success = true;
        return outcome;
    }

private:
    cv::Mat loadMask(
        const QString &imagePath,
        const QString &maskPath,
        const xjw::feature_extractors::FeatureData &feature)
    {
        if (maskPath.isEmpty())
        {
            return cv::Mat();
        }
        if (!_maskCache.contains(maskPath))
        {
            _maskCache.insert(
                maskPath,
                loadMaskForImage(
                    _context,
                    imagePath,
                    cv::Size(feature.imageWidth, feature.imageHeight)));
        }
        return _maskCache.value(maskPath);
    }

    const MatchPhotosContext &_context;
    const MatchPhotosOptions &_options;
    const MatchPhotosAlgorithmPlan &_algorithmPlan;
    const LightGluePairBatchConfig &_config;
    std::unique_ptr<xjw::feature_match::LightGlueMatcher> _matcher;
#if defined(PLASCAN_TORCH_HAS_CUDA)
    std::optional<c10::cuda::CUDAStream> _cudaStream;
#endif
    QMap<QString, cv::Mat> _maskCache;
    int _workerIndex = 0;
};

void reportBatchProgress(const MatchPhotosContext &context,
                         int finalized,
                         int total,
                         int matched,
                         int failed,
                         int totalMatches,
                         int effectiveWorkers)
{
    advanceMatchPhotosProgress(context);
    reportMatchPhotosProgress(
        context,
        QStringLiteral("matching"),
        QStringLiteral(
            "SIFT + LightGlue 两两匹配：%1/%2，成功 %3 对，失败 %4 对，"
            "累计匹配 %5，CUDA 并发 %6")
            .arg(finalized)
            .arg(total)
            .arg(matched)
            .arg(failed)
            .arg(totalMatches)
            .arg(effectiveWorkers),
        finalized,
        total);
}

} // namespace

LightGluePairBatchResult runLightGluePairBatch(
    const MatchPhotosContext &context,
    const MatchPhotosOptions &options,
    const MatchPhotosAlgorithmPlan &algorithmPlan,
    const PairSelectionResult &pairSelection,
    const LightGluePairBatchConfig &config)
{
    LightGluePairBatchResult batch;
    const int total = static_cast<int>(pairSelection.candidates.size());
    if (total <= 0)
    {
        return batch;
    }

    std::vector<PairOutcome> outcomes(static_cast<std::size_t>(total));
    std::vector<bool> finalized(static_cast<std::size_t>(total), false);
    std::atomic_int nextIndex{0};
    std::atomic_int activeWorkers{config.effectiveWorkers};
    std::atomic_bool stop{false};
    std::mutex completionMutex;
    std::condition_variable completionCondition;
    std::deque<int> completionQueue;
    QString workerFailureMessage;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(config.effectiveWorkers));

    for (int workerIndex = 0; workerIndex < config.effectiveWorkers; ++workerIndex)
    {
        workers.emplace_back([&, workerIndex]()
        {
            try
            {
                LightGluePairWorker worker(
                    context, options, algorithmPlan, config, workerIndex);
                while (!stop.load())
                {
                    const int itemIndex = nextIndex.fetch_add(1);
                    if (itemIndex >= total)
                    {
                        break;
                    }
                    if (shouldCancelMatchPhotos(context))
                    {
                        stop.store(true);
                        break;
                    }

                    outcomes[static_cast<std::size_t>(itemIndex)] =
                        worker.process(pairSelection.candidates[
                            static_cast<std::size_t>(itemIndex)]);
                    {
                        std::lock_guard<std::mutex> lock(completionMutex);
                        completionQueue.push_back(itemIndex);
                    }
                    completionCondition.notify_one();
                }
            }
            catch (const std::exception &e)
            {
                // 并发 worker 异常退出时，未完成任务由其他 worker 或后续单
                // worker 回退完成。
                std::lock_guard<std::mutex> lock(completionMutex);
                if (workerFailureMessage.isEmpty())
                {
                    workerFailureMessage = QString::fromUtf8(e.what());
                }
            }
            activeWorkers.fetch_sub(1);
            completionCondition.notify_one();
        });
    }

    int finalizedCount = 0;
    auto finalizeOutcome = [&](int index)
    {
        PairOutcome &outcome = outcomes[static_cast<std::size_t>(index)];
        if (finalized[static_cast<std::size_t>(index)])
        {
            return;
        }
        finalized[static_cast<std::size_t>(index)] = true;
        ++finalizedCount;
        if (outcome.success)
        {
            ++batch.matchedPairs;
            batch.totalMatches += outcome.record.matchCount;
        }
        else
        {
            ++batch.failedPairs;
        }
        reportBatchProgress(context,
                            finalizedCount,
                            total,
                            batch.matchedPairs,
                            batch.failedPairs,
                            batch.totalMatches,
                            config.effectiveWorkers);
    };

    while (activeWorkers.load() > 0)
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
            batch.cancelled = true;
        }
        while (!completionQueue.empty())
        {
            const int index = completionQueue.front();
            completionQueue.pop_front();
            PairOutcome &outcome = outcomes[static_cast<std::size_t>(index)];
            if (!outcome.success && outcome.cudaOutOfMemory)
            {
                continue;
            }
            lock.unlock();
            finalizeOutcome(index);
            lock.lock();
        }
    }

    for (std::thread &worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    if (batch.cancelled || shouldCancelMatchPhotos(context))
    {
        batch.cancelled = true;
        return batch;
    }

    std::vector<int> retryIndices;
    for (int index = 0; index < total; ++index)
    {
        const PairOutcome &outcome = outcomes[static_cast<std::size_t>(index)];
        if (!outcome.processed || (!outcome.success && outcome.cudaOutOfMemory))
        {
            retryIndices.push_back(index);
        }
        else
        {
            finalizeOutcome(index);
        }
    }

    if (!retryIndices.empty())
    {
        batch.usedSerialRecovery = true;
        batch.usedSerialOomRecovery = std::any_of(
            retryIndices.cbegin(),
            retryIndices.cend(),
            [&](int index)
            {
                return outcomes[static_cast<std::size_t>(index)].cudaOutOfMemory;
            });
        if (batch.usedSerialOomRecovery)
        {
            batch.serialRecoveryReason = QStringLiteral("CUDA 显存不足");
        }
        else if (!workerFailureMessage.isEmpty())
        {
            batch.serialRecoveryReason =
                QStringLiteral("并发 worker 异常退出：%1").arg(workerFailureMessage);
        }
        else
        {
            batch.serialRecoveryReason = QStringLiteral("并发 worker 未完成任务");
        }
        releaseCudaCache();
        try
        {
            LightGluePairWorker serialWorker(
                context, options, algorithmPlan, config, 0);
            for (const int index : retryIndices)
            {
                if (shouldCancelMatchPhotos(context))
                {
                    batch.cancelled = true;
                    return batch;
                }
                const bool oomRetry =
                    outcomes[static_cast<std::size_t>(index)].cudaOutOfMemory;
                outcomes[static_cast<std::size_t>(index)] = serialWorker.process(
                    pairSelection.candidates[static_cast<std::size_t>(index)],
                    true,
                    oomRetry);
                finalizeOutcome(index);
            }
        }
        catch (const std::exception &e)
        {
            batch.fatalError = QString::fromUtf8(e.what());
            for (const int index : retryIndices)
            {
                PairOutcome &outcome = outcomes[static_cast<std::size_t>(index)];
                outcome.processed = true;
                outcome.success = false;
                outcome.errorMessage = batch.fatalError;
                finalizeOutcome(index);
            }
        }
    }

    batch.records.reserve(static_cast<std::size_t>(batch.matchedPairs));
    for (PairOutcome &outcome : outcomes)
    {
        if (outcome.success)
        {
            batch.records.push_back(std::move(outcome.record));
        }
    }
    return batch;
}

} // namespace matchphotos
} // namespace xjw
