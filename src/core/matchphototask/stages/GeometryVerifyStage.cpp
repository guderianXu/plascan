#include "GeometryVerifyStage.h"

#include "MatchGeometryVerifier.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosRuntime.h"
#include "concurrency/SafeWorkerGroup.h"
#include "ImageMatchTypes.h"

#include <QCryptographicHash>
#include <QElapsedTimer>
#include <QDir>
#include <QFileInfo>
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

namespace
{

QString normalizedImagePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

double gridCoverage(const image_matching::PairMatchData &pair,
                    bool firstImage,
                    const MatchPhotosOptions &options)
{
    const int columns = std::clamp(options.geometryGridColumns, 1, 32);
    const int rows = std::clamp(options.geometryGridRows, 1, 32);
    const int width = static_cast<int>(firstImage ? pair.image0.width : pair.image1.width);
    const int height = static_cast<int>(firstImage ? pair.image0.height : pair.image1.height);
    if (width <= 0 || height <= 0)
    {
        return 0.0;
    }
    std::vector<bool> occupied(static_cast<std::size_t>(columns * rows), false);
    for (const auto &correspondence : pair.correspondences)
    {
        if (!image_matching::hasFlag(correspondence.flags,
                                     image_matching::MatchRecordFlag::GeometryInlier))
        {
            continue;
        }
        const auto &observation = firstImage ? correspondence.observation0
                                             : correspondence.observation1;
        const int column = std::clamp(
            static_cast<int>(observation.x * columns / std::max(1, width)), 0, columns - 1);
        const int row = std::clamp(
            static_cast<int>(observation.y * rows / std::max(1, height)), 0, rows - 1);
        occupied[static_cast<std::size_t>(row * columns + column)] = true;
    }
    return static_cast<double>(std::count(occupied.cbegin(), occupied.cend(), true)) /
        static_cast<double>(occupied.size());
}

} // namespace

bool areSequenceAdjacent(const MatchPhotosContext &context,
                         const MatchPhotosOptions &options,
                         const QString &image0Path,
                         const QString &image1Path)
{
    int index0 = -1;
    int index1 = -1;
    const QString normalized0 = normalizedImagePath(image0Path);
    const QString normalized1 = normalizedImagePath(image1Path);
    for (int index = 0; index < context.pairInput.images.size(); ++index)
    {
        const QString normalized = normalizedImagePath(context.pairInput.images.at(index));
        index0 = normalized == normalized0 ? index : index0;
        index1 = normalized == normalized1 ? index : index1;
    }
    if (index0 < 0 || index1 < 0 || index0 == index1)
    {
        return false;
    }
    const int distance = std::abs(index0 - index1);
    return distance == 1 ||
        (options.pairPolicy.closeSequenceLoop &&
         context.pairInput.images.size() > 2 &&
         distance == context.pairInput.images.size() - 1);
}

GeometryQualityMetrics measureGeometryQuality(const image_matching::PairMatchData &pair,
                                              const MatchPhotosOptions &options,
                                              bool adjacentImages)
{
    GeometryQualityMetrics metrics;
    metrics.rawMatchCount = static_cast<int>(pair.correspondences.size());
    metrics.inlierCount = static_cast<int>(std::count_if(
        pair.correspondences.cbegin(), pair.correspondences.cend(), [](const auto &correspondence)
        {
            return image_matching::hasFlag(correspondence.flags,
                                            image_matching::MatchRecordFlag::GeometryInlier);
        }));
    metrics.inlierRatio = metrics.rawMatchCount > 0
        ? static_cast<double>(metrics.inlierCount) / static_cast<double>(metrics.rawMatchCount)
        : 0.0;
    metrics.image0GridCoverage = gridCoverage(pair, true, options);
    metrics.image1GridCoverage = gridCoverage(pair, false, options);
    metrics.adjacentImages = adjacentImages;
    return metrics;
}

GeometryQualityDecision evaluateGeometryQuality(const GeometryQualityMetrics &metrics,
                                                const MatchPhotosOptions &options)
{
    GeometryQualityDecision decision;
    if (metrics.rawMatchCount <= 0 || metrics.inlierCount < options.geometryMinInliers)
    {
        return decision;
    }

    const double minimumRatio = std::clamp(options.geometryMinInlierRatio, 0.01, 0.95);
    const double minimumCoverage = std::clamp(options.geometryMinGridCoverage, 0.01, 1.0);
    const double supportScale = static_cast<double>(std::max(12, options.geometryMinInliers));
    const double support = 1.0 - std::exp(
        -static_cast<double>(metrics.inlierCount - options.geometryMinInliers + 1) / supportScale);

    // 支持度越少，对内点率和空间覆盖要求越高；相邻影像有强先验，但只给予
    // 连续的小幅奖励，不能绕过最低内点数。这里不存在任何特定内点数断点。
    decision.requiredInlierRatio = minimumRatio + (0.52 - minimumRatio) * (1.0 - support);
    decision.requiredGridCoverage = minimumCoverage + (0.30 - minimumCoverage) * (1.0 - support);
    if (metrics.adjacentImages)
    {
        decision.requiredInlierRatio = std::max(minimumRatio * 0.8,
                                                decision.requiredInlierRatio - 0.06);
        decision.requiredGridCoverage = std::max(minimumCoverage * 0.8,
                                                 decision.requiredGridCoverage - 0.04);
    }

    const double coverage = std::min(metrics.image0GridCoverage, metrics.image1GridCoverage);
    const double ratioScore = metrics.inlierRatio / std::max(1e-9, decision.requiredInlierRatio);
    const double coverageScore = coverage / std::max(1e-9, decision.requiredGridCoverage);
    const double countScore = std::min(
        1.0, static_cast<double>(metrics.inlierCount) /
                 static_cast<double>(std::max(1, options.geometryMinInliers * 2)));
    decision.score = 0.45 * std::min(1.5, ratioScore) +
        0.35 * std::min(1.5, coverageScore) + 0.20 * countScore;
    decision.passed = metrics.inlierRatio >= minimumRatio * 0.8 &&
        coverage >= minimumCoverage * 0.8 && decision.score >= 0.92;
    return decision;
}

QByteArray geometryVerificationFingerprint(const MatchPhotosOptions &options)
{
    // schema_version 必须在几何实现、内点标志或质量门语义变化时递增。
    // OpenCV 版本也参与键，避免库升级后继续信任不同 USAC 实现生成的内点集。
    QJsonObject object;
    object[QStringLiteral("schema_version")] = 2;
    object[QStringLiteral("enabled")] = options.enableGeometryVerification;
    object[QStringLiteral("model")] = QStringLiteral("fundamental");
    object[QStringLiteral("reprojection_threshold_pixels")] =
        options.geometryReprojThreshold;
    object[QStringLiteral("minimum_inliers")] = options.geometryMinInliers;
    object[QStringLiteral("minimum_inlier_ratio")] = options.geometryMinInlierRatio;
    object[QStringLiteral("minimum_grid_coverage")] = options.geometryMinGridCoverage;
    object[QStringLiteral("grid_columns")] = options.geometryGridColumns;
    object[QStringLiteral("grid_rows")] = options.geometryGridRows;
    object[QStringLiteral("maximum_iterations")] =
        std::max(100, options.geometryMaxIterations);
    object[QStringLiteral("confidence")] = 0.9999;
    object[QStringLiteral("random_seed")] = 0;
    object[QStringLiteral("quality_gate_version")] = 2;
    object[QStringLiteral("opencv_version")] = QString::fromLatin1(CV_VERSION);
    return QCryptographicHash::hash(
        QJsonDocument(object).toJson(QJsonDocument::Compact),
        QCryptographicHash::Sha256);
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
    input.matches.sourceAlgorithm = pair.algorithmId.toStdString();
    return input;
}

bool hasCompleteReusableGeometry(const MatchPhotosMatchRecord &record,
                                 const MatchPhotosContext &context,
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
    const bool adjacent = areSequenceAdjacent(
        context, options, record.image0Path, record.image1Path);
    const GeometryQualityMetrics metrics = measureGeometryQuality(pair, options, adjacent);
    const bool expectedPassed = pair.geometryModel != image_matching::GeometryModel::None &&
        evaluateGeometryQuality(metrics, options).passed;
    return flaggedInliers == static_cast<int>(pair.geometryInlierCount) &&
        pair.geometryPassed == expectedPassed;
}

void updateGeometryRecordMetadata(MatchPhotosMatchRecord *record,
                                  bool verified,
                                  const GeometryQualityMetrics *quality = nullptr,
                                  const GeometryQualityDecision *decision = nullptr)
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
    if (quality)
    {
        record->settings[QStringLiteral("geometry_grid_coverage_image0")] =
            quality->image0GridCoverage;
        record->settings[QStringLiteral("geometry_grid_coverage_image1")] =
            quality->image1GridCoverage;
        record->settings[QStringLiteral("geometry_adjacent_images")] = quality->adjacentImages;
    }
    if (decision)
    {
        record->settings[QStringLiteral("geometry_quality_score")] = decision->score;
        record->settings[QStringLiteral("geometry_required_inlier_ratio")] =
            decision->requiredInlierRatio;
        record->settings[QStringLiteral("geometry_required_grid_coverage")] =
            decision->requiredGridCoverage;
    }
}

void applyVerification(const image_matching::MatchGeometryResult &geometry,
                       const MatchPhotosContext &context,
                       const MatchPhotosOptions &options,
                       MatchPhotosMatchRecord *record)
{
    if (!record || !record->pairData)
    {
        return;
    }
    image_matching::PairMatchData &pair = *record->pairData;
    const int rawCount = static_cast<int>(pair.correspondences.size());
    pair.rawMatchCount = static_cast<std::uint32_t>(rawCount);
    pair.geometryInlierCount = static_cast<std::uint32_t>(geometry.inlierCount);
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
    const bool adjacent = areSequenceAdjacent(
        context, options, record->image0Path, record->image1Path);
    const GeometryQualityMetrics metrics = measureGeometryQuality(pair, options, adjacent);
    const GeometryQualityDecision decision = evaluateGeometryQuality(metrics, options);
    pair.geometryPassed = geometry.modelEstimated && decision.passed;
    updateGeometryRecordMetadata(record, true, &metrics, &decision);
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
            const bool adjacent = areSequenceAdjacent(
                context, options, record.image0Path, record.image1Path);
            const GeometryQualityMetrics metrics = measureGeometryQuality(pair, options, adjacent);
            const GeometryQualityDecision decision = evaluateGeometryQuality(metrics, options);
            updateGeometryRecordMetadata(&record, false, &metrics, &decision);
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
        if (!hasCompleteReusableGeometry(record, context, options, expectedFingerprint))
        {
            workIndices.push_back(index);
            continue;
        }
        ++reusedPairs;
        const bool adjacent = areSequenceAdjacent(
            context, options, record.image0Path, record.image1Path);
        const GeometryQualityMetrics metrics = measureGeometryQuality(
            *record.pairData, options, adjacent);
        const GeometryQualityDecision decision = evaluateGeometryQuality(metrics, options);
        updateGeometryRecordMetadata(&record, true, &metrics, &decision);
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
                    applyVerification(geometry, context, options, &record);
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
