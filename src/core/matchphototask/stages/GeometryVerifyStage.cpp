#include "GeometryVerifyStage.h"

#include "MatchGeometryVerifier.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosRuntime.h"
#include "concurrency/SafeWorkerGroup.h"

#include <QCryptographicHash>
#include <QElapsedTimer>
#include <QJsonDocument>
#include <QJsonObject>

#include <opencv2/core/version.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace xjw::matchphotos
{

QByteArray geometryVerificationFingerprint(const MatchPhotosOptions &options)
{
    // schema_version 必须在几何实现、内点标志或质量门语义变化时递增。
    // OpenCV 版本也参与键，避免库升级后继续信任不同 USAC 实现生成的内点集。
    QJsonObject object;
    object[QStringLiteral("schema_version")] = 1;
    object[QStringLiteral("enabled")] = options.enableGeometryVerification;
    object[QStringLiteral("model")] = QStringLiteral("fundamental");
    object[QStringLiteral("reprojection_threshold_pixels")] =
        options.geometryReprojThreshold;
    object[QStringLiteral("minimum_inliers")] = options.geometryMinInliers;
    object[QStringLiteral("maximum_iterations")] =
        std::max(100, options.geometryMaxIterations);
    object[QStringLiteral("confidence")] = 0.9999;
    object[QStringLiteral("random_seed")] = 0;
    object[QStringLiteral("quality_gate_version")] = 1;
    object[QStringLiteral("opencv_version")] = QString::fromLatin1(CV_VERSION);
    return QCryptographicHash::hash(
        QJsonDocument(object).toJson(QJsonDocument::Compact),
        QCryptographicHash::Sha256);
}

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

bool hasCompleteReusableGeometry(const MatchPhotosMatchRecord &record,
                                 const MatchPhotosOptions &options,
                                 const QByteArray &expectedFingerprint)
{
    if (!record.pairData ||
        !record.settings.value(QStringLiteral("geometry_cache_reusable")).toBool(false) ||
        record.settings.value(QStringLiteral("geometry_config_fingerprint")).toString() !=
            QString::fromLatin1(expectedFingerprint.toHex()))
    {
        return false;
    }

    const image_matching::PairMatchData &pair = *record.pairData;
    const int rawCount = static_cast<int>(pair.correspondences.size());
    if (pair.rawMatchCount != static_cast<std::uint32_t>(rawCount) ||
        pair.geometryInlierCount > pair.rawMatchCount)
    {
        return false;
    }

    const int flaggedInliers = static_cast<int>(std::count_if(
        pair.correspondences.cbegin(),
        pair.correspondences.cend(),
        [](const image_matching::PairCorrespondence &correspondence)
        {
            return image_matching::hasFlag(
                correspondence.flags,
                image_matching::MatchRecordFlag::GeometryInlier);
        }));
    const bool expectedPassed = pair.geometryModel != image_matching::GeometryModel::None &&
        passesGeometryQualityGate(rawCount,
                                  static_cast<int>(pair.geometryInlierCount),
                                  options.geometryMinInliers);
    return flaggedInliers == static_cast<int>(pair.geometryInlierCount) &&
        pair.geometryPassed == expectedPassed;
}

void updateGeometryRecordMetadata(MatchPhotosMatchRecord *record, bool verified)
{
    if (!record || !record->pairData)
    {
        return;
    }

    const image_matching::PairMatchData &pair = *record->pairData;
    const int rawCount = static_cast<int>(pair.correspondences.size());
    const int inlierCount = static_cast<int>(pair.geometryInlierCount);
    record->matchCount = rawCount;
    record->geometricInlierCount = inlierCount;
    record->passedGeometry = pair.geometryPassed;
    record->settings[QStringLiteral("geometry_verified")] = verified;
    record->settings[QStringLiteral("geometry_passed")] = pair.geometryPassed;
    record->settings[QStringLiteral("geometry_raw_matches")] = rawCount;
    record->settings[QStringLiteral("geometric_inliers")] = inlierCount;
    record->settings[QStringLiteral("geometry_inlier_ratio")] = rawCount > 0
        ? static_cast<double>(inlierCount) / static_cast<double>(rawCount)
        : 0.0;
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

    updateGeometryRecordMetadata(record, true);
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
            pair.rawMatchCount = static_cast<std::uint32_t>(pair.correspondences.size());
            pair.geometryPassed = true;
            pair.geometryInlierCount = pair.rawMatchCount;
            pair.geometryModel = image_matching::GeometryModel::None;
            pair.geometryMatrix = {};
            for (auto &correspondence : pair.correspondences)
            {
                correspondence.flags = withGeometryFlag(correspondence.flags, true);
                correspondence.residualPixels = -1.0f;
            }
            updateGeometryRecordMetadata(&record, false);
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

    const QByteArray expectedFingerprint = geometryVerificationFingerprint(options);
    const int totalPairs = static_cast<int>(matchRecords->size());
    std::vector<int> workIndices;
    workIndices.reserve(matchRecords->size());
    int reusedPairs = 0;
    int reusedPassedPairs = 0;
    int reusedInliers = 0;
    for (int index = 0; index < totalPairs; ++index)
    {
        MatchPhotosMatchRecord &record =
            (*matchRecords)[static_cast<std::size_t>(index)];
        if (!hasCompleteReusableGeometry(record, options, expectedFingerprint))
        {
            workIndices.push_back(index);
            continue;
        }
        ++reusedPairs;
        updateGeometryRecordMetadata(&record, true);
        reusedPassedPairs += record.passedGeometry ? 1 : 0;
        reusedInliers += record.geometricInlierCount;
    }

    if (workIndices.empty())
    {
        return makeGeometryReport(
            MatchPhotosStageStatus::Completed,
            QStringLiteral("几何验证缓存全命中：复用 %1 对，通过 %2 对，内点 %3，未运行 USAC")
                .arg(reusedPairs)
                .arg(reusedPassedPairs)
                .arg(reusedInliers),
            reusedPassedPairs);
    }

    const int workerCount = resolveGeometryVerificationWorkers(
        static_cast<int>(workIndices.size()), std::thread::hardware_concurrency());
    std::atomic_int nextIndex{0};
    std::atomic_int completed{reusedPairs};
    std::atomic_int passedPairs{reusedPassedPairs};
    std::atomic_int totalInliers{reusedInliers};
    std::mutex callbackMutex;
    QElapsedTimer timer;
    timer.start();

    try
    {
        xjw::common::concurrency::runWorkerGroup(
            static_cast<std::size_t>(workerCount),
            [&](std::stop_token stopToken)
        {
            while (!stopToken.stop_requested() && !shouldCancelMatchPhotos(context))
            {
                const int workIndex = nextIndex.fetch_add(1);
                if (workIndex >= static_cast<int>(workIndices.size()))
                {
                    break;
                }
                const int index = workIndices[static_cast<std::size_t>(workIndex)];
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

                if (stopToken.stop_requested())
                {
                    break;
                }
                std::lock_guard lock(callbackMutex);
                const int done = ++completed;
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
    catch (const std::exception &error)
    {
        return makeGeometryReport(
            MatchPhotosStageStatus::Failed,
            QStringLiteral("几何验证 worker 异常：%1")
                .arg(QString::fromUtf8(error.what())),
            passedPairs.load());
    }
    catch (...)
    {
        return makeGeometryReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("几何验证 worker 发生未知异常"),
                                  passedPairs.load());
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
        QStringLiteral("几何验证完成：通过 %1 对，失败 %2 对，内点 %3，复用 %4 对，"
                       "CPU worker %5，总计 %6 ms")
            .arg(passedPairs.load())
            .arg(failedPairs)
            .arg(totalInliers.load())
            .arg(reusedPairs)
            .arg(workerCount)
            .arg(timer.elapsed()),
        passedPairs.load());
}

} // namespace xjw::matchphotos
