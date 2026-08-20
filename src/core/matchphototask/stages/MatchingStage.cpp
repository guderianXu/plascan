#include "MatchingStage.h"

#include "ImageMatchFile.h"
#include "ImageMatchRepository.h"
#include "ImageMatchingRegistry.h"
#include "GeometryVerifyStage.h"
#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosRuntime.h"
#include "concurrency/SafeWorkerGroup.h"
#include "io/PathIO.h"
#include "lightglue/LightGlueFeatureBudget.h"
#include "loma_r/LoMaRAlgorithm.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftComputeBackend.h"
#include "sift_lightglue/SiftLightGlueAlgorithm.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xjw::matchphotos
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

QString backendDisplayName(image_matching::SiftComputeBackend backend)
{
    return image_matching::siftBackendDisplayName(backend);
}

QByteArray sha256(const QByteArray &payload)
{
    return QCryptographicHash::hash(payload, QCryptographicHash::Sha256);
}

QByteArray modelFingerprint(const QString &enginePath)
{
    const QFileInfo info(enginePath);
    const QByteArray identity = QDir::cleanPath(info.absoluteFilePath()).toUtf8() + '\n' +
        QByteArray::number(info.size()) + '\n' +
        QByteArray::number(info.lastModified().toMSecsSinceEpoch());
    return sha256(identity);
}

QByteArray modelFingerprint(const QStringList &paths)
{
    QByteArray identity;
    for (const QString &path : paths)
    {
        const QFileInfo info(path);
        identity += QDir::cleanPath(info.absoluteFilePath()).toUtf8() + '\n' +
            QByteArray::number(info.size()) + '\n' +
            QByteArray::number(info.lastModified().toMSecsSinceEpoch()) + '\n';
    }
    return sha256(identity);
}

QString lightGlueModelIdentityPath(const ResolvedLightGlueTensorRtEngine &engine)
{
    return engine.sourceOnnxPath.isEmpty() ? engine.path : engine.sourceOnnxPath;
}

QStringList loMaRModelIdentityPaths(const ResolvedLoMaRTensorRtPackage &package)
{
    QStringList paths{package.manifestPath};
    if (!package.featureOnnxPath.isEmpty() && !package.matcherOnnxPath.isEmpty())
    {
        paths.append(package.featureOnnxPath);
        paths.append(package.matcherOnnxPath);
    }
    else
    {
        paths.append(package.featureEnginePath);
        paths.append(package.matcherEnginePath);
    }
    return paths;
}

QByteArray rawConfigFingerprint(const MatchPhotosOptions &options,
                                const MatchPhotosAlgorithmPlan &plan,
                                int matcherKeypointBudget,
                                float effectiveMatchThreshold,
                                const QByteArray &engineFingerprint)
{
    // JSON 键按固定顺序构造并压缩后计算 SHA-256。几何参数不属于“原始匹配”
    // 缓存键：改变 USAC 阈值可以复用原始对应并重新执行几何验证。
    QJsonObject object;
    object[QStringLiteral("algorithm")] = plan.algorithmId;
    object[QStringLiteral("algorithm_version")] = static_cast<int>(
        plan.algorithmVersion);
    object[QStringLiteral("max_image_dim")] = plan.maxImageDim;
    object[QStringLiteral("feature_keypoint_limit")] = plan.maxKeypoints;
    object[QStringLiteral("feature_schema_version")] = plan.featureSchemaVersion;
    object[QStringLiteral("feature_remove_borders")] = plan.featureRemoveBorders;
    object[QStringLiteral("sift_detection_threshold")] =
        static_cast<double>(plan.siftDetectionThreshold);
    object[QStringLiteral("sift_contrast_threshold")] =
        static_cast<double>(plan.siftContrastThreshold);
    object[QStringLiteral("adaptive_sift")] =
        plan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId);
    object[QStringLiteral("root_sift")] =
        plan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId);
    object[QStringLiteral("guided_matching")] = options.enableGuidedMatching;
    object[QStringLiteral("keypoint_limit_per_mpx")] = options.keypointLimitPerMegapixel;
    object[QStringLiteral("matcher_keypoint_budget")] = matcherKeypointBudget;
    object[QStringLiteral("match_threshold")] = static_cast<double>(effectiveMatchThreshold);
    object[QStringLiteral("mask_apply_mode")] = options.maskApplyMode.trimmed().toLower();
    object[QStringLiteral("engine_fingerprint")] = QString::fromLatin1(engineFingerprint.toHex());
    return sha256(QJsonDocument(object).toJson(QJsonDocument::Compact));
}

QByteArray fullConfigFingerprint(const QByteArray &rawFingerprint,
                                 const QByteArray &geometryFingerprint)
{
    // SHA-256 固定为 32 字节。把原始匹配指纹置于开头，repository 才能在
    // 几何参数变化时按前缀找到坐标对应，并明确要求 GeometryVerifyStage 重算。
    return rawFingerprint + geometryFingerprint;
}

QByteArray maskFileFingerprint(const QString &path)
{
    const QFileInfo info(path);
    if (path.trimmed().isEmpty() || !info.exists() || !info.isFile())
    {
        return QByteArrayLiteral("none");
    }
    return sha256(QDir::cleanPath(info.absoluteFilePath()).toUtf8() + '\n' +
                  QByteArray::number(info.size()) + '\n' +
                  QByteArray::number(info.lastModified().toMSecsSinceEpoch()));
}

QByteArray pairConfigFingerprint(const QByteArray &baseFingerprint,
                                 const MatchPhotosContext &context,
                                 const MatchPhotosOptions &options,
                                 const ResolvedImagePair &pair)
{
    // 蒙版会改变 SIFT 观测集合或最终连接点集合，必须属于像对缓存键。否则用户
    // 修改蒙版后，“复用已有匹配”会静默取回旧区域内的对应关系。
    if (maskApplyModeFromToken(options.maskApplyMode) == MatchPhotosMaskApplyMode::None)
    {
        return baseFingerprint;
    }
    const QByteArray payload = baseFingerprint.toHex() + '\n' +
        maskFileFingerprint(maskPathForImage(context, pair.image0Path)).toHex() + '\n' +
        maskFileFingerprint(maskPathForImage(context, pair.image1Path)).toHex();
    return sha256(payload);
}

image_matching::KeypointObservation observationFor(const cv::KeyPoint &keypoint,
                                                    int featureId)
{
    image_matching::KeypointObservation observation;
    observation.featureId = static_cast<std::uint32_t>(featureId);
    observation.x = keypoint.pt.x;
    observation.y = keypoint.pt.y;
    observation.scale = keypoint.size;
    observation.orientation = keypoint.angle;
    observation.response = keypoint.response;
    return observation;
}

float matchConfidence(const image_matching::MatchResult &result,
                      const cv::DMatch &match)
{
    if (match.queryIdx >= 0 &&
        match.queryIdx < static_cast<int>(result.matchingScores0.size()))
    {
        return std::clamp(result.matchingScores0[static_cast<std::size_t>(match.queryIdx)],
                          0.0f,
                          1.0f);
    }
    return std::isfinite(match.distance)
        ? 1.0f / (1.0f + std::max(0.0f, match.distance))
        : 0.0f;
}

std::shared_ptr<image_matching::PairMatchData> makePairData(
    const ResolvedImagePair &pair,
    const image_matching::FeatureSet &features0,
    const image_matching::FeatureSet &features1,
    const image_matching::MatchResult &raw,
    const MatchPhotosContext &context,
    const MatchPhotosOptions &options,
    const MatchPhotosAlgorithmPlan &plan,
    const QByteArray &configurationFingerprint,
    const QByteArray &engineFingerprint)
{
    auto data = std::make_shared<image_matching::PairMatchData>();
    data->image0 = image_matching::ImageMatchFile::identityForImage(
        pair.image0Path, features0.imageWidth, features0.imageHeight);
    data->image1 = image_matching::ImageMatchFile::identityForImage(
        pair.image1Path, features1.imageWidth, features1.imageHeight);
    data->algorithmId = plan.algorithmId;
    data->algorithmVersion = plan.algorithmVersion;
    data->configFingerprint = configurationFingerprint;
    data->modelFingerprint = engineFingerprint;
    data->createdTimeMs = QDateTime::currentMSecsSinceEpoch();

    const bool applyTiepointMask = shouldApplyMasksToTiepoints(options);
    const cv::Mat mask0 = applyTiepointMask
        ? loadMaskForImage(context,
                           pair.image0Path,
                           cv::Size(features0.imageWidth, features0.imageHeight))
        : cv::Mat();
    const cv::Mat mask1 = applyTiepointMask
        ? loadMaskForImage(context,
                           pair.image1Path,
                           cv::Size(features1.imageWidth, features1.imageHeight))
        : cv::Mat();

    data->correspondences.reserve(raw.cvMatches.size());
    for (const cv::DMatch &match : raw.cvMatches)
    {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= features0.size() || match.trainIdx >= features1.size())
        {
            continue;
        }
        const cv::KeyPoint &keypoint0 =
            features0.keypoints[static_cast<std::size_t>(match.queryIdx)];
        const cv::KeyPoint &keypoint1 =
            features1.keypoints[static_cast<std::size_t>(match.trainIdx)];
        if (applyTiepointMask &&
            (!isPointAllowedByMask(mask0, keypoint0.pt) ||
             !isPointAllowedByMask(mask1, keypoint1.pt)))
        {
            continue;
        }

        image_matching::PairCorrespondence correspondence;
        correspondence.observation0 = observationFor(keypoint0, match.queryIdx);
        correspondence.observation1 = observationFor(keypoint1, match.trainIdx);
        correspondence.confidence = matchConfidence(raw, match);
        correspondence.flags = image_matching::MatchRecordFlag::MaskAccepted;
        data->correspondences.push_back(correspondence);
    }
    data->rawMatchCount = static_cast<std::uint32_t>(data->correspondences.size());
    return data;
}

MatchPhotosMatchRecord makeRecord(
    const image_matching::ImageMatchRepository &repository,
    std::shared_ptr<image_matching::PairMatchData> pair,
    const MatchPhotosAlgorithmPlan &plan,
    const MatchPhotosOptions &options,
    bool reused,
    bool geometryReusable,
    const QByteArray &geometryFingerprint,
    const QString &reuseWarning = QString())
{
    MatchPhotosMatchRecord record;
    record.image0Path = pair->image0.path;
    record.image1Path = pair->image1.path;
    record.image0MatchFilePath = repository.shardPath(record.image0Path);
    record.image1MatchFilePath = repository.shardPath(record.image1Path);
    record.algorithmId = pair->algorithmId;
    record.algorithmVersion = pair->algorithmVersion;
    record.matchCount = static_cast<int>(pair->rawMatchCount);
    record.geometricInlierCount = static_cast<int>(pair->geometryInlierCount);
    record.passedGeometry = pair->geometryPassed;
    record.pairData = std::move(pair);

    record.settings = makeMatchRecordSettings(
        plan,
        options,
        ResolvedImagePair{record.image0Path,
                          record.image1Path,
                          QFileInfo(record.image0Path).completeBaseName() + QStringLiteral(" / ") +
                              QFileInfo(record.image1Path).completeBaseName(),
                          makePairKey(record.image0Path, record.image1Path)},
        record.matchCount);
    record.settings[QStringLiteral("storage_format")] = QStringLiteral("pimatch");
    record.settings[QStringLiteral("storage_format_version")] =
        static_cast<int>(image_matching::kImageMatchFormatVersion);
    record.settings[QStringLiteral("image0_match_file")] = record.image0MatchFilePath;
    record.settings[QStringLiteral("image1_match_file")] = record.image1MatchFilePath;
    record.settings[QStringLiteral("algorithm_id")] = record.algorithmId;
    record.settings[QStringLiteral("algorithm_version")] =
        static_cast<int>(record.algorithmVersion);
    const bool autoSift =
        plan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId);
    record.settings[QStringLiteral("matching_backend")] = autoSift
        ? QString::fromLatin1(image_matching::siftBackendName(plan.executionBackend))
        : plan.algorithmId;
    record.settings[QStringLiteral("backend_fallback")] = plan.backendFallback;
    record.settings[QStringLiteral("match_reused")] = reused;
    record.settings[QStringLiteral("geometry_cache_reusable")] = geometryReusable;
    record.settings[QStringLiteral("geometry_config_fingerprint")] =
        QString::fromLatin1(geometryFingerprint.toHex());
    if (!reuseWarning.isEmpty())
    {
        record.settings[QStringLiteral("match_reuse_warning")] = reuseWarning;
    }
    return record;
}

} // namespace

QByteArray rawMatchConfigurationFingerprint(
    const MatchPhotosOptions &options,
    const MatchPhotosAlgorithmPlan &plan,
    int matcherKeypointBudget,
    float effectiveMatchThreshold,
    const QByteArray &engineFingerprint)
{
    return rawConfigFingerprint(options,
                                plan,
                                matcherKeypointBudget,
                                effectiveMatchThreshold,
                                engineFingerprint);
}

MatchPhotosStageReport MatchingStage::run(
    const MatchPhotosContext &context,
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
    if (!matchRecords || !context.featureCache)
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("内部错误：匹配输出或特征缓存为空"));
    }
    if (pairSelection.candidates.empty())
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("没有可用于匹配的影像对"));
    }
    const MatchPhotosGpuMemoryInfo gpuMemory =
        queryMatchPhotosGpuMemory(options.cudaDevice);
    image_matching::LightGlueGpuMemoryInfo budgetMemory;
    budgetMemory.available = gpuMemory.available;
    budgetMemory.freeBytes = gpuMemory.freeBytes;
    budgetMemory.totalBytes = gpuMemory.totalBytes;
    budgetMemory.deviceIndex = gpuMemory.deviceIndex;
    image_matching::ImageMatchingRuntimeConfig runtime;
    runtime.cudaDevice = options.cudaDevice;
    runtime.maxKeypoints = algorithmPlan.maxKeypoints;
    runtime.siftBackend = algorithmPlan.executionBackend;
    QString modelName;
    QByteArray engineFingerprint;
    int matcherBudget = 0;
    float effectiveThreshold = options.matchThreshold;
    ResolvedLoMaRTensorRtPackage plannedLoMaRPackage;
    ResolvedLightGlueTensorRtEngine plannedLightGlueEngine;
    if (algorithmPlan.algorithmId == QLatin1String(image_matching::kLoMaRAlgorithmId))
    {
        // 先只解析清单与源 ONNX，不能在确认存在冷 pair 前构建本机 engine。
        plannedLoMaRPackage = resolveLoMaRTensorRtPackage(
            options, algorithmPlan.maxKeypoints, false);
        if (!plannedLoMaRPackage.isValid())
        {
            QString message = QStringLiteral("LoMa-R TensorRT 模型包不可用：%1")
                .arg(plannedLoMaRPackage.errorMessage);
            if (!plannedLoMaRPackage.searchedDirectories.isEmpty())
            {
                message += QStringLiteral("\n已搜索：%1")
                    .arg(plannedLoMaRPackage.searchedDirectories.join(QStringLiteral("; ")));
            }
            return makeMatchingReport(MatchPhotosStageStatus::Failed, message);
        }
        runtime.modelInputWidth = plannedLoMaRPackage.inputWidth;
        runtime.modelInputHeight = plannedLoMaRPackage.inputHeight;
        runtime.descriptorDimension = plannedLoMaRPackage.descriptorDimension;
        matcherBudget = plannedLoMaRPackage.keypointCount;
        runtime.maxMatcherKeypoints = matcherBudget;
        runtime.featureKeypointCount = plannedLoMaRPackage.featureKeypointCount;
        // LoMa-R 官方默认门限为 0.1；工作流默认 0.15 仍可作为更严格的用户门限。
        runtime.matchThreshold = effectiveThreshold;
        modelName = QFileInfo(plannedLoMaRPackage.manifestPath).fileName();
        engineFingerprint = modelFingerprint(loMaRModelIdentityPaths(plannedLoMaRPackage));
    }
    else if (algorithmPlan.algorithmId ==
             QLatin1String(image_matching::kSiftLightGlueAlgorithmId))
    {
        const int requestedMatcherBudget =
            image_matching::resolveSiftLightGlueKeypointBudget(
                algorithmPlan.maxKeypoints, budgetMemory);
        plannedLightGlueEngine = resolveLightGlueTensorRtEngine(
            options, requestedMatcherBudget, false);
        if (!plannedLightGlueEngine.isValid())
        {
            QString message = QStringLiteral(
                "未找到可用的 LightGlue ONNX/engine。请在“工作流程 - 设置”中下载 "
                "ONNX 模型；程序会针对本机 TensorRT 和 GPU 自动构建 engine。");
            if (!plannedLightGlueEngine.errorMessage.isEmpty())
            {
                message += QStringLiteral("\n模型解析错误：%1")
                    .arg(plannedLightGlueEngine.errorMessage);
            }
            if (!plannedLightGlueEngine.searchedDirectories.isEmpty())
            {
                message += QStringLiteral("\n已搜索：%1")
                    .arg(plannedLightGlueEngine.searchedDirectories.join(QStringLiteral("; ")));
            }
            return makeMatchingReport(MatchPhotosStageStatus::Failed, message);
        }
        matcherBudget = image_matching::clampLightGlueKeypointBudgetToEngine(
            requestedMatcherBudget, plannedLightGlueEngine.bucketKeypoints);
        effectiveThreshold = image_matching::resolveSiftLightGlueMatchThreshold(
            options.matchThreshold, matcherBudget, budgetMemory);
        runtime.maxMatcherKeypoints = matcherBudget;
        runtime.matchThreshold = effectiveThreshold;
        modelName = plannedLightGlueEngine.name;
        engineFingerprint = modelFingerprint(
            lightGlueModelIdentityPath(plannedLightGlueEngine));
    }
    else if (algorithmPlan.algorithmId ==
             QLatin1String(image_matching::kAutoSiftAlgorithmId))
    {
        // SIFT 不受固定 TensorRT K 桶限制，0 表示匹配本次任务实际提取的
        // 全部特征。算法版本和内置实现身份仍进入缓存键，避免误用旧算法结果。
        matcherBudget = 0;
        runtime.maxMatcherKeypoints = 0;
        runtime.matchThreshold = effectiveThreshold;
        modelName = QStringLiteral("内置 Auto SIFT %1 后端")
                        .arg(backendDisplayName(algorithmPlan.executionBackend));
        engineFingerprint = sha256(
            QByteArrayLiteral("builtin:auto_sift_root:v2:") +
            QByteArray(image_matching::siftBackendName(algorithmPlan.executionBackend)));
    }
    else
    {
        return makeMatchingReport(
            MatchPhotosStageStatus::Failed,
            QStringLiteral("未实现匹配运行时：%1").arg(algorithmPlan.algorithmId));
    }
    const QByteArray configurationFingerprint = rawMatchConfigurationFingerprint(
        options, algorithmPlan, matcherBudget, effectiveThreshold, engineFingerprint);
    const QByteArray geometryFingerprint = geometryVerificationFingerprint(options);
    runtime.configFingerprint = fullConfigFingerprint(configurationFingerprint,
                                                      geometryFingerprint);
    runtime.modelFingerprint = engineFingerprint;

    image_matching::ImageMatchRepository repository(matchPhotosMatchDirectory(context));
    const int totalPairs = static_cast<int>(pairSelection.candidates.size());

    struct WorkItem
    {
        ResolvedImagePair pair;
        QByteArray configurationFingerprint;
        int outputIndex = -1;
        QString cacheWarning;
    };
    std::vector<WorkItem> work;
    std::vector<MatchPhotosMatchRecord> records(static_cast<std::size_t>(totalPairs));
    std::vector<bool> populated(static_cast<std::size_t>(totalPairs), false);
    int reusedPairs = 0;
    for (int index = 0; index < totalPairs; ++index)
    {
        ResolvedImagePair pair;
        QString resolveError;
        if (!resolveMatchPhotosPair(context,
                                    pairSelection.candidates[static_cast<std::size_t>(index)],
                                    &pair,
                                    &resolveError))
        {
            WorkItem failed;
            failed.outputIndex = index;
            failed.cacheWarning = resolveError;
            work.push_back(std::move(failed));
            continue;
        }

        QString cacheError;
        image_matching::PairMatchData cached;
        const QByteArray pairRawFingerprint = pairConfigFingerprint(
            configurationFingerprint, context, options, pair);
        const QByteArray pairFingerprint = fullConfigFingerprint(
            pairRawFingerprint, geometryFingerprint);
        if (options.reuseExistingMatches &&
            repository.loadPair(pair.image0Path,
                                pair.image1Path,
                                algorithmPlan.algorithmId,
                                algorithmPlan.algorithmVersion,
                                pairFingerprint,
                                engineFingerprint,
                                &cached,
                                &cacheError))
        {
            auto data = std::make_shared<image_matching::PairMatchData>(std::move(cached));
            records[static_cast<std::size_t>(index)] =
                makeRecord(repository,
                           std::move(data),
                           algorithmPlan,
                           options,
                           true,
                           true,
                           geometryFingerprint);
            populated[static_cast<std::size_t>(index)] = true;
            ++reusedPairs;
            continue;
        }

        // 旧缓存只有 32 字节原始匹配指纹；新缓存则在其后追加几何指纹。
        // 两者都可以复用坐标对应，但只有上面的完整键命中才允许跳过 USAC。
        image_matching::PairMatchData rawCached;
        QString rawCacheError;
        if (options.reuseExistingMatches &&
            repository.loadPairWithConfigPrefix(pair.image0Path,
                                                pair.image1Path,
                                                algorithmPlan.algorithmId,
                                                algorithmPlan.algorithmVersion,
                                                pairRawFingerprint,
                                                engineFingerprint,
                                                &rawCached,
                                                &rawCacheError))
        {
            rawCached.configFingerprint = pairFingerprint;
            rawCached.createdTimeMs = QDateTime::currentMSecsSinceEpoch();
            auto data = std::make_shared<image_matching::PairMatchData>(
                std::move(rawCached));
            records[static_cast<std::size_t>(index)] =
                makeRecord(repository,
                           std::move(data),
                           algorithmPlan,
                           options,
                           true,
                           false,
                           geometryFingerprint);
            populated[static_cast<std::size_t>(index)] = true;
            ++reusedPairs;
            continue;
        }

        const QString effectiveCacheError = !cacheError.isEmpty() ? cacheError : rawCacheError;
        work.push_back(WorkItem{pair, pairFingerprint, index, effectiveCacheError});
    }

    // 并发度按真正缺失的 pair 计算。warm-cache 仅缺一对时不能仍创建多个
    // TensorRT execution context；全命中则明确为 0，并在下面直接返回。
    int parallelismBudget = matcherBudget;
    if (algorithmPlan.algorithmId == QLatin1String(image_matching::kAutoSiftAlgorithmId))
    {
        // 全量 CUDA SIFT 匹配的计算量随特征数平方增长。使用真正缓存的最大特征
        // 数估算并发，避免 guided/per-megapixel 模式下 plan.maxKeypoints=0 时
        // 被误判成小任务并同时启动四个重型 matcher。
        for (const WorkItem &item : work)
        {
            const auto features0 = context.featureCache->find(item.pair.image0Path);
            const auto features1 = context.featureCache->find(item.pair.image1Path);
            if (features0)
            {
                parallelismBudget = std::max(parallelismBudget, features0->size());
            }
            if (features1)
            {
                parallelismBudget = std::max(parallelismBudget, features1->size());
            }
        }
    }
    const LightGlueParallelismDecision parallelism = resolveLightGlueParallelism(
        options.cudaParallelPairs,
        static_cast<int>(work.size()),
        algorithmPlan.executionBackend == image_matching::SiftComputeBackend::Cuda,
        parallelismBudget,
        gpuMemory);
    const int workerCount = work.empty() ? 0 : std::max(1, parallelism.effectiveWorkers);

    reportMatchPhotosProgress(
        context,
        QStringLiteral("matching"),
        QStringLiteral("%1：%2 对待计算，%3 对复用，%4 worker %5")
            .arg(algorithmPlan.displayName)
            .arg(static_cast<int>(work.size()))
            .arg(reusedPairs)
            .arg(backendDisplayName(algorithmPlan.executionBackend))
            .arg(workerCount),
        reusedPairs,
        totalPairs);

    if (work.empty())
    {
        std::uint64_t totalMatches = 0;
        for (int index = 0; index < totalPairs; ++index)
        {
            if (!populated[static_cast<std::size_t>(index)])
            {
                continue;
            }
            totalMatches += static_cast<std::uint64_t>(
                std::max(0, records[static_cast<std::size_t>(index)].matchCount));
            matchRecords->push_back(std::move(records[static_cast<std::size_t>(index)]));
        }
        return makeMatchingReport(
            MatchPhotosStageStatus::Completed,
            QStringLiteral("%1 匹配缓存全命中：复用 %2 对，原始匹配 %3，未创建 matcher context，模型 %4")
                .arg(algorithmPlan.displayName)
                .arg(reusedPairs)
                .arg(totalMatches)
                .arg(modelName),
            reusedPairs);
    }

    if (options.device != ComputeDevice::Auto && options.device != ComputeDevice::Cuda &&
        algorithmPlan.algorithmId != QLatin1String(image_matching::kAutoSiftAlgorithmId))
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("当前匹配算法 %1 仅支持 TensorRT/CUDA")
                                      .arg(algorithmPlan.algorithmId));
    }

    // 只有确有冷 pair 时才准备 TensorRT engine。源模型身份在缓存扫描前已经
    // 固定；若扫描期间模型发生变化，宁可显式失败并让用户重试，也不能混用键。
    if (algorithmPlan.algorithmId == QLatin1String(image_matching::kLoMaRAlgorithmId))
    {
        const ModelPreparationProgressCallback progress_callback =
            [&context](const QString &message, int current, int maximum)
        {
            reportMatchPhotosProgress(
                context, QStringLiteral("model_prepare"), message, current, maximum);
        };
        const ResolvedLoMaRTensorRtPackage package = resolveLoMaRTensorRtPackage(
            options, algorithmPlan.maxKeypoints, true, progress_callback);
        if (!package.isValid())
        {
            return makeMatchingReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("LoMa-R TensorRT engine 准备失败：%1").arg(package.errorMessage));
        }
        if (modelFingerprint(loMaRModelIdentityPaths(package)) != engineFingerprint ||
            package.keypointCount != matcherBudget)
        {
            return makeMatchingReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("LoMa-R 模型在缓存扫描期间发生变化，请重试"));
        }
        runtime.tensorRtFeatureEnginePath = package.featureEnginePath;
        runtime.tensorRtMatcherEnginePath = package.matcherEnginePath;
    }
    else if (algorithmPlan.algorithmId ==
             QLatin1String(image_matching::kSiftLightGlueAlgorithmId))
    {
        reportMatchPhotosProgress(
            context,
            QStringLiteral("model_prepare"),
            QStringLiteral("正在为冷匹配准备 LightGlue TensorRT engine"),
            0,
            1);
        const ModelPreparationProgressCallback progress_callback =
            [&context](const QString &message, int current, int maximum)
        {
            reportMatchPhotosProgress(
                context, QStringLiteral("model_prepare"), message, current, maximum);
        };
        const ResolvedLightGlueTensorRtEngine engine =
            resolveLightGlueTensorRtEngine(
                options, matcherBudget, true, progress_callback);
        if (!engine.isValid())
        {
            return makeMatchingReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("LightGlue TensorRT engine 准备失败：%1").arg(engine.errorMessage));
        }
        const int preparedBudget = image_matching::clampLightGlueKeypointBudgetToEngine(
            matcherBudget, engine.bucketKeypoints);
        if (modelFingerprint(lightGlueModelIdentityPath(engine)) != engineFingerprint ||
            preparedBudget != matcherBudget)
        {
            return makeMatchingReport(
                MatchPhotosStageStatus::Failed,
                QStringLiteral("LightGlue 模型在缓存扫描期间发生变化，请重试"));
        }
        runtime.tensorRtEnginePath = engine.path;
        reportMatchPhotosProgress(
            context,
            QStringLiteral("model_prepare"),
            QStringLiteral("LightGlue 本机 TensorRT engine 已就绪：%1")
                .arg(engine.environmentSummary),
            1,
            1);
    }
    // SIFT 双向匹配使用已解析的 CUDA、Metal、OpenCL 或 OpenCV CPU 后端，
    // 均不需要准备外部模型或 engine。

    std::atomic_int nextWork{0};
    std::atomic_int completed{reusedPairs};
    std::atomic_int failedPairs{0};
    std::mutex resultMutex;
    QString fatalError;
    try
    {
        xjw::common::concurrency::runWorkerGroup(
            static_cast<std::size_t>(workerCount),
            [&](std::stop_token stopToken)
        {
            QString createError;
            std::unique_ptr<image_matching::IImageMatchingAlgorithm> algorithm =
                image_matching::ImageMatchingRegistry::create(
                    algorithmPlan.algorithmId,
                    runtime,
                    &createError);
            if (!algorithm)
            {
                throw std::runtime_error(createError.toStdString());
            }

            while (!stopToken.stop_requested() &&
                   !shouldCancelMatchPhotos(context))
            {
                const int workIndex = nextWork.fetch_add(1);
                if (workIndex >= static_cast<int>(work.size()))
                {
                    break;
                }
                const WorkItem &item = work[static_cast<std::size_t>(workIndex)];
                if (item.pair.image0Path.isEmpty() || item.pair.image1Path.isEmpty())
                {
                    ++failedPairs;
                    ++completed;
                    continue;
                }

                const auto features0 = context.featureCache->find(item.pair.image0Path);
                const auto features1 = context.featureCache->find(item.pair.image1Path);
                if (!features0 || !features1)
                {
                    ++failedPairs;
                    ++completed;
                    continue;
                }

                try
                {
                    image_matching::MatchResult raw =
                        algorithm->matchFeatures(*features0, *features1);
                    if (raw.cvMatches.empty() && !raw.matches0.empty())
                    {
                        raw.buildCvMatchesFromIndices();
                    }
                    auto pairData = makePairData(item.pair,
                                                 *features0,
                                                 *features1,
                                                 raw,
                                                 context,
                                                 options,
                                                 algorithmPlan,
                                                 item.configurationFingerprint,
                                                 engineFingerprint);
                    MatchPhotosMatchRecord record = makeRecord(repository,
                                                                std::move(pairData),
                                                                algorithmPlan,
                                                                options,
                                                                false,
                                                                false,
                                                                geometryFingerprint,
                                                                item.cacheWarning);
                    {
                        std::lock_guard lock(resultMutex);
                        records[static_cast<std::size_t>(item.outputIndex)] =
                            std::move(record);
                        populated[static_cast<std::size_t>(item.outputIndex)] = true;
                    }
                }
                catch (const std::exception &error)
                {
                    std::lock_guard lock(resultMutex);
                    ++failedPairs;
                    if (fatalError.isEmpty())
                    {
                        fatalError = QStringLiteral("%1 / %2：%3")
                            .arg(item.pair.image0Path,
                                 item.pair.image1Path,
                                 QString::fromUtf8(error.what()));
                    }
                }

                if (stopToken.stop_requested() ||
                    shouldCancelMatchPhotos(context))
                {
                    break;
                }
                std::lock_guard lock(resultMutex);
                const int done = ++completed;
                reportMatchPhotosProgress(
                    context,
                    QStringLiteral("matching"),
                    QStringLiteral("%1 匹配：%2/%3，复用 %4，失败 %5")
                        .arg(algorithmPlan.displayName)
                        .arg(done)
                        .arg(totalPairs)
                        .arg(reusedPairs)
                        .arg(failedPairs.load()),
                    done,
                    totalPairs);
            }
        });
    }
    catch (const std::exception &error)
    {
        return makeMatchingReport(
            MatchPhotosStageStatus::Failed,
            QStringLiteral("%1 匹配 worker 异常：%2")
                .arg(algorithmPlan.displayName, QString::fromUtf8(error.what())),
            completed.load());
    }
    catch (...)
    {
        return makeMatchingReport(
            MatchPhotosStageStatus::Failed,
            QStringLiteral("%1 匹配 worker 发生未知异常")
                .arg(algorithmPlan.displayName),
            completed.load());
    }

    if (shouldCancelMatchPhotos(context))
    {
        return makeMatchingReport(MatchPhotosStageStatus::Failed,
                                  QStringLiteral("用户取消连接点匹配"),
                                  completed.load());
    }

    int matchedPairs = 0;
    std::uint64_t totalMatches = 0;
    for (int index = 0; index < totalPairs; ++index)
    {
        if (!populated[static_cast<std::size_t>(index)])
        {
            continue;
        }
        totalMatches += static_cast<std::uint64_t>(
            std::max(0, records[static_cast<std::size_t>(index)].matchCount));
        matchRecords->push_back(std::move(records[static_cast<std::size_t>(index)]));
        ++matchedPairs;
    }
    if (matchedPairs == 0)
    {
        return makeMatchingReport(
            MatchPhotosStageStatus::Failed,
            fatalError.isEmpty() ? QStringLiteral("没有生成任何有效匹配结果")
                                 : QStringLiteral("%1 匹配失败：%2")
                                       .arg(algorithmPlan.displayName, fatalError));
    }

    return makeMatchingReport(
        MatchPhotosStageStatus::Completed,
        QStringLiteral("%1 匹配完成：%2/%3 对，原始匹配 %4，复用 %5，"
                       "失败 %6，%7 worker %8，模型 %9")
            .arg(algorithmPlan.displayName)
            .arg(matchedPairs)
            .arg(totalPairs)
            .arg(totalMatches)
            .arg(reusedPairs)
            .arg(failedPairs.load())
            .arg(backendDisplayName(algorithmPlan.executionBackend))
            .arg(workerCount)
            .arg(modelName),
        matchedPairs);
}

} // namespace xjw::matchphotos
