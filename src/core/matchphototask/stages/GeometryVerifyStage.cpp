#include "GeometryVerifyStage.h"

#include "MatchGeometryVerifier.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosRuntime.h"

#include <QElapsedTimer>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

namespace xjw::matchphotos
{

bool passesGeometryQualityGate(int rawMatchCount,
                               int inlierCount,
                               int minimumInliers)
{
    if (inlierCount < minimumInliers)
    {
        return false;
    }

    // 内点少时，重复纹理可能产生一个局部自洽但错误的基础矩阵。达到 64 个内点
    // 后模型约束通常足够稳定；更小的集合要求至少 70% 原始对应支持该模型。
    constexpr int strongSupportInliers = 64;
    constexpr double minimumWeakSupportRatio = 0.70;
    if (inlierCount >= strongSupportInliers || rawMatchCount <= 0)
    {
        return true;
    }
    return static_cast<double>(inlierCount) / static_cast<double>(rawMatchCount) >=
        minimumWeakSupportRatio;
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

image_matching::MatchRecordFlag withGeometryFlag(
    image_matching::MatchRecordFlag value,
    bool enabled)
{
    const auto raw = static_cast<std::uint32_t>(value);
    const auto bit = static_cast<std::uint32_t>(
        image_matching::MatchRecordFlag::GeometryInlier);
    return static_cast<image_matching::MatchRecordFlag>(enabled ? raw | bit : raw & ~bit);
}

struct VerificationInput
{
    image_matching::FeatureSet features0;
    image_matching::FeatureSet features1;
    image_matching::MatchResult matches;
};

VerificationInput makeVerificationInput(const image_matching::PairMatchData &pair)
{
    VerificationInput input;
    const int count = static_cast<int>(pair.correspondences.size());
    input.features0.imageWidth = static_cast<int>(pair.image0.width);
    input.features0.imageHeight = static_cast<int>(pair.image0.height);
    input.features1.imageWidth = static_cast<int>(pair.image1.width);
    input.features1.imageHeight = static_cast<int>(pair.image1.height);
    input.features0.keypoints.reserve(static_cast<std::size_t>(count));
    input.features1.keypoints.reserve(static_cast<std::size_t>(count));
    input.matches.cvMatches.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index)
    {
        const auto &correspondence = pair.correspondences[static_cast<std::size_t>(index)];
        input.features0.keypoints.emplace_back(
            correspondence.observation0.x,
            correspondence.observation0.y,
            std::max(1.0f, correspondence.observation0.scale),
            correspondence.observation0.orientation,
            correspondence.observation0.response);
        input.features1.keypoints.emplace_back(
            correspondence.observation1.x,
            correspondence.observation1.y,
            std::max(1.0f, correspondence.observation1.scale),
            correspondence.observation1.orientation,
            correspondence.observation1.response);
        input.matches.cvMatches.emplace_back(index,
                                             index,
                                             1.0f - correspondence.confidence);
    }
    input.matches.numMatches = count;
    input.matches.sourceAlgorithm = "sift_lightglue";
    return input;
}

void applyVerification(const image_matching::MatchGeometryResult &geometry,
                       int minimumInliers,
                       MatchPhotosMatchRecord *record)
{
    if (!record || !record->pairData)
    {
        return;
    }
    image_matching::PairMatchData &pair = *record->pairData;
    const int rawCount = static_cast<int>(pair.correspondences.size());
    const bool passed = geometry.modelEstimated &&
        passesGeometryQualityGate(rawCount, geometry.inlierCount, minimumInliers);
    pair.rawMatchCount = static_cast<std::uint32_t>(rawCount);
    pair.geometryInlierCount = static_cast<std::uint32_t>(geometry.inlierCount);
    pair.geometryPassed = passed;
    pair.geometryModel = geometry.modelEstimated ? geometry.model
                                                 : image_matching::GeometryModel::None;
    pair.geometryMatrix = geometry.matrix;

    for (int index = 0; index < rawCount; ++index)
    {
        auto &correspondence = pair.correspondences[static_cast<std::size_t>(index)];
        const bool inlier = index < static_cast<int>(geometry.inlierMask.size()) &&
            geometry.inlierMask[static_cast<std::size_t>(index)];
        correspondence.flags = withGeometryFlag(correspondence.flags, inlier);
        correspondence.residualPixels =
            index < static_cast<int>(geometry.residualPixels.size())
            ? geometry.residualPixels[static_cast<std::size_t>(index)]
            : -1.0f;
    }

    record->matchCount = rawCount;
    record->geometricInlierCount = geometry.inlierCount;
    record->passedGeometry = passed;
    record->settings[QStringLiteral("geometry_verified")] = true;
    record->settings[QStringLiteral("geometry_passed")] = passed;
    record->settings[QStringLiteral("geometry_raw_matches")] = rawCount;
    record->settings[QStringLiteral("geometric_inliers")] = geometry.inlierCount;
    record->settings[QStringLiteral("geometry_inlier_ratio")] = rawCount > 0
        ? static_cast<double>(geometry.inlierCount) / static_cast<double>(rawCount)
        : 0.0;
}

} // namespace

MatchPhotosStageReport GeometryVerifyStage::run(
    const MatchPhotosContext &context,
    const MatchPhotosOptions &options,
    std::vector<MatchPhotosMatchRecord> *matchRecords) const
{
    if (options.planOnly)
    {
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("plan-only 模式，跳过几何验证"));
    }
    if (!matchRecords || matchRecords->empty())
    {
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("没有可用于几何验证的匹配结果"));
    }

    if (!options.enableGeometryVerification)
    {
        int accepted = 0;
        for (MatchPhotosMatchRecord &record : *matchRecords)
        {
            if (!record.pairData)
            {
                continue;
            }
            auto &pair = *record.pairData;
            pair.geometryPassed = true;
            pair.geometryInlierCount = pair.rawMatchCount;
            for (auto &correspondence : pair.correspondences)
            {
                correspondence.flags = withGeometryFlag(correspondence.flags, true);
            }
            record.passedGeometry = true;
            record.geometricInlierCount = record.matchCount;
            ++accepted;
        }
        return makeGeometryReport(MatchPhotosStageStatus::Skipped,
                                  QStringLiteral("几何验证已禁用，保留全部 %1 对匹配").arg(accepted),
                                  accepted);
    }

    image_matching::MatchGeometryOptions geometryOptions;
    geometryOptions.model = image_matching::GeometryModel::Fundamental;
    geometryOptions.reprojectionThresholdPixels = options.geometryReprojThreshold;
    geometryOptions.minimumInliers = options.geometryMinInliers;
    // 最大迭代数属于几何验证配置，而不是原始 LightGlue 匹配缓存键。
    // 用户调整该值时可直接复用原始对应，并重新执行本阶段。
    geometryOptions.maximumIterations = std::max(100, options.geometryMaxIterations);
    geometryOptions.confidence = 0.9999;
    geometryOptions.randomSeed = 0;

    const int totalPairs = static_cast<int>(matchRecords->size());
    const int workerCount = resolveGeometryVerificationWorkers(
        totalPairs, std::thread::hardware_concurrency());
    std::atomic_int nextIndex{0};
    std::atomic_int completed{0};
    std::atomic_int passedPairs{0};
    std::atomic_int totalInliers{0};
    std::mutex callbackMutex;
    QElapsedTimer timer;
    timer.start();

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(workerCount));
    for (int worker = 0; worker < workerCount; ++worker)
    {
        workers.emplace_back([&]()
        {
            while (!shouldCancelMatchPhotos(context))
            {
                const int index = nextIndex.fetch_add(1);
                if (index >= totalPairs)
                {
                    break;
                }
                MatchPhotosMatchRecord &record =
                    (*matchRecords)[static_cast<std::size_t>(index)];
                if (record.pairData)
                {
                    const VerificationInput input = makeVerificationInput(*record.pairData);
                    const image_matching::MatchGeometryResult geometry =
                        image_matching::MatchGeometryVerifier::verify(
                            input.matches, input.features0, input.features1, geometryOptions);
                    applyVerification(geometry, options.geometryMinInliers, &record);
                    if (record.passedGeometry)
                    {
                        ++passedPairs;
                    }
                    totalInliers.fetch_add(record.geometricInlierCount);
                }

                const int done = ++completed;
                std::lock_guard lock(callbackMutex);
                reportMatchPhotosProgress(
                    context,
                    QStringLiteral("geometry"),
                    QStringLiteral("USAC 几何验证：%1/%2，通过 %3，CPU worker %4")
                        .arg(done)
                        .arg(totalPairs)
                        .arg(passedPairs.load())
                        .arg(workerCount),
                    done,
                    totalPairs);
            }
        });
    }
    for (std::thread &worker : workers)
    {
        worker.join();
    }

    if (shouldCancelMatchPhotos(context))
    {
        return makeGeometryReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("用户取消几何验证"),
                                  passedPairs.load());
    }

    const int failedPairs = totalPairs - passedPairs.load();
    return makeGeometryReport(
        MatchPhotosStageStatus::Completed,
        QStringLiteral("几何验证完成：通过 %1 对，失败 %2 对，内点 %3，"
                       "CPU worker %4，总计 %5 ms")
            .arg(passedPairs.load())
            .arg(failedPairs)
            .arg(totalInliers.load())
            .arg(workerCount)
            .arg(timer.elapsed()),
        passedPairs.load());
}

} // namespace xjw::matchphotos
