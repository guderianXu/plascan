#include "AerialTriangulationWorkflow.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>

#include <algorithm>
#include <cmath>

namespace xjw::gui
{

namespace
{

struct QualityPreset
{
    int sfmQuality = 2;
    int featureMaxImageDim = 0;
    double budgetScale = 0.75;
};

QString normalizedToken(QString value, const QString &fallback)
{
    value = value.trimmed().toLower();
    return value.isEmpty() ? fallback : value;
}

QualityPreset presetForQuality(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest"))
    {
        return {3, -1, 1.0};
    }
    if (token == QStringLiteral("high"))
    {
        return {2, 0, 0.75};
    }
    if (token == QStringLiteral("medium") || token == QStringLiteral("standard"))
    {
        return {1, 4096, 0.50};
    }
    if (token == QStringLiteral("low"))
    {
        return {0, 2048, 0.25};
    }
    if (token == QStringLiteral("lowest") || token == QStringLiteral("fast"))
    {
        return {0, 1536, 0.125};
    }
    return {2, 0, 0.75};
}

QString normalizeMatchAlgorithm(QString value)
{
    value = normalizedToken(value, QStringLiteral("lightglue"));
    value.replace(QStringLiteral("-"), QStringLiteral("_"));
    if (value == QStringLiteral("bf") || value == QStringLiteral("bf_l2") || value == QStringLiteral("sift_bf_l2"))
    {
        return QStringLiteral("bf_l2");
    }
    if (value == QStringLiteral("flann") || value == QStringLiteral("sift_flann"))
    {
        return QStringLiteral("flann");
    }
    return value;
}

void applyMatchPipeline(const QString &pipeline, QString &featureAlgorithm, QString &matchAlgorithm)
{
    const QString token = normalizedToken(pipeline, QString());
    if (token.isEmpty())
    {
        return;
    }

    if (token == QStringLiteral("sift-bf-l2") || token == QStringLiteral("sift_bf_l2"))
    {
        featureAlgorithm = QStringLiteral("sift");
        matchAlgorithm = QStringLiteral("bf_l2");
        return;
    }
    if (token == QStringLiteral("sift-flann") || token == QStringLiteral("sift_flann"))
    {
        featureAlgorithm = QStringLiteral("sift");
        matchAlgorithm = QStringLiteral("flann");
        return;
    }

    const QStringList pieces = token.split(QRegularExpression(QStringLiteral("[-_]")),
                                           Qt::SkipEmptyParts);
    if (pieces.size() >= 2)
    {
        featureAlgorithm = pieces.front();
        matchAlgorithm = pieces.mid(1).join(QStringLiteral("_"));
    }
}

QString effectivePipelineName(const QString &requested,
                              const QString &featureAlgorithm,
                              const QString &matchAlgorithm)
{
    const QString token = normalizedToken(requested, QString());
    if (!token.isEmpty())
    {
        return token;
    }
    QString normalizedMatch = matchAlgorithm;
    normalizedMatch.replace(QStringLiteral("_"), QStringLiteral("-"));
    return QStringLiteral("%1-%2").arg(featureAlgorithm, normalizedMatch);
}

int scaledLimit(int value, double scale)
{
    if (value <= 0)
    {
        return 0;
    }
    return std::max(1, static_cast<int>(std::round(static_cast<double>(value) * scale)));
}

int sequenceWindowForQuality(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest"))
    {
        return 8;
    }
    if (token == QStringLiteral("high"))
    {
        return 6;
    }
    if (token == QStringLiteral("medium") || token == QStringLiteral("standard"))
    {
        return 4;
    }
    if (token == QStringLiteral("low"))
    {
        return 3;
    }
    return 2;
}

matchphotos::MatchPhotosProfile matchPhotosProfileForQuality(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest") || token == QStringLiteral("high"))
    {
        return matchphotos::MatchPhotosProfile::HighAccuracy;
    }
    if (token == QStringLiteral("lowest") || token == QStringLiteral("low") ||
        token == QStringLiteral("fast"))
    {
        return matchphotos::MatchPhotosProfile::Fast;
    }
    return matchphotos::MatchPhotosProfile::Auto;
}

matchphotos::PairSelectionPreset matchPhotosPairPresetForQuality(const QString &quality)
{
    const auto profile = matchPhotosProfileForQuality(quality);
    if (profile == matchphotos::MatchPhotosProfile::HighAccuracy)
    {
        return matchphotos::PairSelectionPreset::HighAccuracy;
    }
    if (profile == matchphotos::MatchPhotosProfile::Fast)
    {
        return matchphotos::PairSelectionPreset::Fast;
    }
    return matchphotos::PairSelectionPreset::Auto;
}

matchphotos::ComputeDevice matchPhotosDevice(const QString &device)
{
    const QString token = normalizedToken(device, QStringLiteral("auto"));
    if (token == QStringLiteral("cpu"))
    {
        return matchphotos::ComputeDevice::Cpu;
    }
    if (token == QStringLiteral("cuda") || token == QStringLiteral("gpu"))
    {
        return matchphotos::ComputeDevice::Cuda;
    }
    return matchphotos::ComputeDevice::Auto;
}

int matchPhotosMaxImageDim(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest")) return 4096;
    if (token == QStringLiteral("high")) return 3072;
    if (token == QStringLiteral("low")) return 1600;
    if (token == QStringLiteral("lowest") || token == QStringLiteral("fast")) return 1200;
    return 2048;
}

int tiePointProgressPercent(const QString &stageId, int current, int maximum)
{
    struct StageRange
    {
        const char *id;
        int start;
        int end;
    };
    static constexpr StageRange ranges[] = {
        {"algorithm_selection", 1, 3}, {"feature", 3, 16},
        {"generic_preselection", 16, 21}, {"reference_preselection", 21, 25},
        {"pair_selection", 25, 28}, {"matching", 28, 34},
        {"geometry", 34, 35}, {"track_build", 35, 35},
        {"guided_matching", 35, 35}};
    StageRange range{"unknown", 1, 34};
    const QString token = normalizedToken(stageId, QStringLiteral("unknown"));
    for (const StageRange &candidate : ranges)
    {
        if (token == QLatin1String(candidate.id))
        {
            range = candidate;
            break;
        }
    }
    const double fraction = std::clamp(
        static_cast<double>(std::max(0, current)) / static_cast<double>(std::max(1, maximum)),
        0.0,
        1.0);
    return std::clamp(range.start +
                          static_cast<int>(std::round((range.end - range.start) * fraction)),
                      0,
                      35);
}

bool clearMatchCacheForReset(const QString &matchDirPath, QString *errorMessage)
{
    if (errorMessage) errorMessage->clear();
    QDir matchDir(matchDirPath);
    if (!matchDir.exists()) return true;

    QStringList failed;
    const QStringList files = matchDir.entryList(
        {QStringLiteral("*.match"), QStringLiteral("*.match.json"),
         QStringLiteral("no_match_pairs.json")},
        QDir::Files);
    for (const QString &fileName : files)
    {
        if (!matchDir.remove(fileName))
        {
            failed.append(matchDir.filePath(fileName));
        }
    }
    if (failed.isEmpty()) return true;
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("无法清理旧匹配缓存：\n%1").arg(failed.join(QLatin1Char('\n')));
    }
    return false;
}

void mergeTiePointOutputs(const matchphotos::MatchPhotosResult &tieResult,
                          AerialTriangulationServiceResult *serviceResult)
{
    if (!serviceResult) return;
    for (const auto &feature : tieResult.features)
    {
        serviceResult->newFeatureFiles.push_back({feature.imagePath, feature.featurePath});
    }
    for (const auto &match : tieResult.matches)
    {
        if (match.matchPath.isEmpty()) continue;
        MatchFileRecord record;
        record.pairName = QFileInfo(match.matchPath).completeBaseName();
        record.matchPath = match.matchPath;
        record.sidecarPath = match.sidecarPath;
        record.settings = match.settings;
        record.settings[QStringLiteral("tie_point_path")] = tieResult.tiePointPath;
        record.settings[QStringLiteral("track_count")] = tieResult.trackCount;
        record.settings[QStringLiteral("track_summary")] = tieResult.trackSummary;
        serviceResult->newMatchFiles.push_back(record);
    }
    QJsonObject extra = serviceResult->resultRecordExtra;
    extra[QStringLiteral("tie_point_path")] = tieResult.tiePointPath;
    extra[QStringLiteral("tie_point_track_count")] = tieResult.trackCount;
    extra[QStringLiteral("tie_point_candidate_pair_count")] =
        static_cast<int>(tieResult.pairSelection.candidates.size());
    extra[QStringLiteral("tie_point_match_file_count")] =
        static_cast<int>(tieResult.matches.size());
    extra[QStringLiteral("tie_point_preparation")] = QStringLiteral("executed");
    serviceResult->resultRecordExtra = extra;
}

void applyPreselection(AerialTriangulationResolvedConfig &resolved,
                       const AerialTriangulationWorkflowOptions &options)
{
    AerialTriangulationServiceOptions &service = resolved.serviceOptions;
    const QString referenceMode = normalizedToken(options.referenceMode, QStringLiteral("source_code"));

    service.autoRestrictKnownCameraPairs = true;
    service.knownCameraPairWindow = 4;
    service.knownCameraSpatialNeighborCount = 8;
    service.knownCameraSequenceLoopClosure = false;
    service.useKnownCameraOverlapPairs = options.referencePreselection;
    QString pairPlanningMode = QStringLiteral("auto");

    if (options.referencePreselection && referenceMode == QStringLiteral("sequence"))
    {
        service.knownCameraPairWindow = sequenceWindowForQuality(options.quality);
        // 用户显式选择“照片序列”时，应让序列候选独立生效，而不是被小数据全配对阈值覆盖。
        service.knownCameraAllPairsMaxImages = 0;
        service.useKnownCameraOverlapPairs = false;
        service.knownCameraSpatialNeighborCount = 0;
        service.knownCameraSequenceLoopClosure = true;
        pairPlanningMode = QStringLiteral("sequence");
    }
    else if (options.referencePreselection && referenceMode == QStringLiteral("estimated"))
    {
        service.useKnownCameraOverlapPairs = false;
        service.knownCameraSpatialNeighborCount = 8;
        pairPlanningMode = QStringLiteral("estimated");
    }
    else if (options.referencePreselection)
    {
        service.useKnownCameraOverlapPairs = true;
        service.knownCameraSpatialNeighborCount = 8;
        pairPlanningMode = QStringLiteral("source_code");
    }
    else if (options.genericPreselection)
    {
        pairPlanningMode = QStringLiteral("generic");
    }

    resolved.resolvedSettings.insert(QStringLiteral("pair_planning_mode"), pairPlanningMode);
    resolved.resolvedSettings.insert(QStringLiteral("sequence_pair_window"), service.knownCameraPairWindow);
    resolved.resolvedSettings.insert(QStringLiteral("sequence_loop_closure"),
                                     service.knownCameraSequenceLoopClosure);
}

} // namespace

AerialTriangulationResolvedConfig AerialTriangulationWorkflow::resolveConfig(
    const AerialTriangulationWorkflowOptions &options)
{
    AerialTriangulationResolvedConfig resolved;
    AerialTriangulationServiceOptions &service = resolved.serviceOptions;

    const QualityPreset preset = presetForQuality(options.quality);
    QString featureAlgorithm = normalizedToken(options.featureAlgorithm, QStringLiteral("disk"));
    QString matchAlgorithm = normalizeMatchAlgorithm(options.matchAlgorithm);
    applyMatchPipeline(options.matchPipeline, featureAlgorithm, matchAlgorithm);
    matchAlgorithm = normalizeMatchAlgorithm(matchAlgorithm);

    service.images = options.images;
    service.cameraPaths = options.cameraPaths;
    service.plascanPath = options.projectPath;
    service.projectMeta = options.projectMeta;
    service.useProjectMetaCameras = !options.resetAlignment;
    service.outputDir = QDir(QDir::cleanPath(options.outputDir)).filePath(QStringLiteral("sfm_sparse"));
    service.quality = preset.sfmQuality;
    service.threads = std::max(1, options.threads);
    service.device = normalizedToken(options.device, QStringLiteral("auto"));
    service.featureAlgorithm = featureAlgorithm;
    service.matchAlgorithm = matchAlgorithm;
    service.featureMaxImageDim = preset.featureMaxImageDim;
    service.featureGrayscaleMin = options.featureGrayscaleMin;
    service.featureGrayscaleMax = options.featureGrayscaleMax;
    service.autoGenerateMissingMatches = options.autoGenerateMissingMatches;
    service.restrictPairs = options.restrictPairs;
    service.allowedPairs = options.allowedPairs;
    service.adaptiveCameraModelFitting = options.adaptiveCameraModelFitting;
    service.enableTwoStageMatching = true;
    service.enableGuidedRematching = options.guidedImageMatching;
    const bool useGuidedKeypointDensity = options.guidedImageMatching;
    service.tiePointFeatureMaxKeypoints = useGuidedKeypointDensity
        ? 0
        : std::max(0, options.keypointLimit);
    service.tiePointKeypointLimitPerMegapixel = useGuidedKeypointDensity
        ? std::max(0, options.keypointLimit)
        : 0;
    service.useTiePointDenseSift = featureAlgorithm == QStringLiteral("sift") &&
        matchAlgorithm == QStringLiteral("lightglue");
    service.skeletonFeatureMaxKeypoints = scaledLimit(options.keypointLimit, preset.budgetScale);
    service.maxTiePointsPerImage = scaledLimit(options.tiepointLimit, preset.budgetScale);
    service.tiePointGridColumns = 8;
    service.tiePointGridRows = 8;
    service.maxTiePointsPerGridCell = service.maxTiePointsPerImage > 0
        ? std::max(1, service.maxTiePointsPerImage / service.tiePointGridColumns)
        : 0;
    service.cancelFlag = options.cancelFlag;
    service.progressFn = options.progressFn;
    service.pairMatchedFn = options.pairMatchedFn;
    service.cudaParallelPairs = service.device == QStringLiteral("cpu")
        ? 1
        : std::clamp(std::max(1, service.threads / 4), 1, 2);

    applyPreselection(resolved, options);

    const QString projectRoot = QFileInfo(options.projectPath).absolutePath();
    const QString assetsDir = options.assetsDir.isEmpty()
        ? QDir(projectRoot).filePath(QStringLiteral("assets"))
        : QDir::cleanPath(options.assetsDir);
    resolved.prepareTiePoints = options.resetAlignment || options.autoGenerateMissingMatches;
    resolved.forceRebuildTiePoints = options.resetAlignment;
    matchphotos::MatchPhotosOptions &tieOptions = resolved.tiePointOptions;
    tieOptions.planOnly = false;
    tieOptions.profile = matchPhotosProfileForQuality(options.quality);
    tieOptions.device = matchPhotosDevice(options.device);
    tieOptions.pairPolicy = matchphotos::makePairSelectionPolicy(
        matchPhotosPairPresetForQuality(options.quality));
    tieOptions.featureAlgorithm = featureAlgorithm;
    tieOptions.matcherAlgorithm = matchAlgorithm;
    tieOptions.maxImageDim = matchPhotosMaxImageDim(options.quality);
    tieOptions.enableGuidedMatching = options.guidedImageMatching;
    tieOptions.useExplicitKeypointLimit = true;
    tieOptions.maxKeypoints = options.guidedImageMatching ? 0 : std::max(0, options.keypointLimit);
    tieOptions.keypointLimitPerMegapixel = options.guidedImageMatching
        ? std::max(0, options.keypointLimit) : 0;
    tieOptions.maxTiePointsPerImage = std::max(0, options.tiepointLimit);
    tieOptions.tiePointGridColumns = 8;
    tieOptions.tiePointGridRows = 8;
    tieOptions.maxTiePointsPerGridCell = options.tiepointLimit > 0
        ? std::max(1, options.tiepointLimit / 8) : 0;
    tieOptions.useGenericPreselection = options.genericPreselection;
    tieOptions.useReferencePreselection = options.referencePreselection;
    tieOptions.excludeStationaryTiePoints = options.excludeFixedTiePoints;
    tieOptions.maskApplyMode = normalizedToken(options.maskApplyMode, QStringLiteral("none"));
    tieOptions.reuseExistingFeatures = !resolved.forceRebuildTiePoints &&
        tieOptions.maskApplyMode != QStringLiteral("keypoints");
    if (options.referencePreselection &&
        normalizedToken(options.referenceMode, QStringLiteral("source_code")) == QStringLiteral("sequence"))
    {
        tieOptions.pairPolicy.mode = matchphotos::PairSelectionMode::Sequence;
        tieOptions.pairPolicy.sequenceWindow = sequenceWindowForQuality(options.quality);
        tieOptions.useReferencePreselection = false;
    }
    if (!options.allowedPairs.isEmpty())
    {
        tieOptions.pairPolicy.mode = matchphotos::PairSelectionMode::ManualOnly;
    }

    matchphotos::MatchPhotosContext &tieContext = resolved.tiePointContext;
    tieContext.projectPath = options.projectPath;
    tieContext.workingDirectory = assetsDir;
    tieContext.featureDirectory = options.featureDir.isEmpty()
        ? QDir(assetsDir).filePath(QStringLiteral("ip")) : QDir::cleanPath(options.featureDir);
    tieContext.matchDirectory = options.matchDir.isEmpty()
        ? QDir(assetsDir).filePath(QStringLiteral("matches")) : QDir::cleanPath(options.matchDir);
    service.assetsDir = tieContext.workingDirectory;
    service.featureDir = tieContext.featureDirectory;
    service.matchDir = tieContext.matchDirectory;
    tieContext.pairInput.images = options.images;
    tieContext.pairInput.manualPairKeys = options.allowedPairs;
    tieContext.referenceCameras = options.referenceCameras;
    tieContext.maskPaths = options.maskPaths;
    tieContext.cancelFlag = options.cancelFlag.get();
    if (options.progressFn)
    {
        tieContext.progressCallback = [progressFn = options.progressFn](const QString &stageId,
                                                                        const QString &message,
                                                                        int current,
                                                                        int maximum)
        {
            progressFn(message.isEmpty() ? stageId : message,
                       tiePointProgressPercent(stageId, current, maximum));
        };
        service.progressFn = [progressFn = options.progressFn](const QString &stage, int percent)
        {
            progressFn(stage, 35 + static_cast<int>(std::round(std::clamp(percent, 0, 100) * 0.65)));
        };
    }
    // 匹配只允许由统一连接点阶段生成；SfM 服务只消费该阶段的落盘结果。
    service.autoGenerateMissingMatches = false;

    resolved.resolvedSettings.insert(QStringLiteral("quality"), normalizedToken(options.quality, QStringLiteral("high")));
    resolved.resolvedSettings.insert(QStringLiteral("match_pipeline"),
                                     effectivePipelineName(options.matchPipeline, featureAlgorithm, matchAlgorithm));
    resolved.resolvedSettings.insert(QStringLiteral("feature_algorithm"), featureAlgorithm);
    resolved.resolvedSettings.insert(QStringLiteral("match_algorithm"), matchAlgorithm);
    resolved.resolvedSettings.insert(QStringLiteral("keypoint_limit"), options.keypointLimit);
    resolved.resolvedSettings.insert(QStringLiteral("resolved_keypoint_budget"),
                                     service.tiePointFeatureMaxKeypoints);
    resolved.resolvedSettings.insert(QStringLiteral("resolved_keypoint_limit_per_megapixel"),
                                     service.tiePointKeypointLimitPerMegapixel);
    resolved.resolvedSettings.insert(QStringLiteral("skeleton_keypoint_budget"),
                                     service.skeletonFeatureMaxKeypoints);
    resolved.resolvedSettings.insert(QStringLiteral("tiepoint_limit"), options.tiepointLimit);
    resolved.resolvedSettings.insert(QStringLiteral("resolved_tiepoint_limit"), service.maxTiePointsPerImage);
    resolved.resolvedSettings.insert(QStringLiteral("mask_apply_mode"),
                                     normalizedToken(options.maskApplyMode, QStringLiteral("none")));
    resolved.resolvedSettings.insert(QStringLiteral("exclude_fixed_tie_points"), options.excludeFixedTiePoints);
    resolved.resolvedSettings.insert(QStringLiteral("reset_current_alignment"), options.resetAlignment);
    resolved.resolvedSettings.insert(QStringLiteral("use_project_camera_metadata"),
                                     service.useProjectMetaCameras);
    resolved.resolvedSettings.insert(QStringLiteral("save_project_after_each_step"), options.saveAfterEachStep);
    resolved.resolvedSettings.insert(QStringLiteral("adaptive_camera_model_fitting"), options.adaptiveCameraModelFitting);
    resolved.resolvedSettings.insert(QStringLiteral("adaptive_known_pose_soft_prior"), true);
    resolved.resolvedSettings.insert(QStringLiteral("restrict_pairs"), service.restrictPairs);
    resolved.resolvedSettings.insert(QStringLiteral("allowed_pair_count"), service.allowedPairs.size());
    resolved.resolvedSettings.insert(
        QStringLiteral("tie_point_preparation"),
        resolved.forceRebuildTiePoints
            ? QStringLiteral("force_rebuild")
            : (resolved.prepareTiePoints ? QStringLiteral("fill_missing")
                                         : QStringLiteral("skipped_reuse_only")));

    return resolved;
}

AerialTriangulationWorkflowResult AerialTriangulationWorkflow::run(
    const AerialTriangulationWorkflowOptions &options,
    const ServiceRunner &runner,
    const TiePointRunner &tiePointRunner)
{
    AerialTriangulationWorkflowResult result;
    result.config = resolveConfig(options);
    if (!runner)
    {
        result.serviceResult.success = false;
        result.serviceResult.errorMessage = QStringLiteral("空中三角测量 workflow 缺少空三服务执行器。");
        return result;
    }
    if (result.config.prepareTiePoints)
    {
        result.tiePointPreparationExecuted = true;
        if (result.config.forceRebuildTiePoints)
        {
            QString cleanupError;
            if (!clearMatchCacheForReset(result.config.tiePointContext.matchDirectory, &cleanupError))
            {
                result.serviceResult.errorMessage = cleanupError;
                result.serviceResult.summary = cleanupError;
                return result;
            }
        }
        const TiePointRunner effectiveRunner = tiePointRunner
            ? tiePointRunner
            : TiePointRunner([](const matchphotos::MatchPhotosOptions &taskOptions,
                                const matchphotos::MatchPhotosContext &context)
            {
                return matchphotos::MatchPhotosTask(taskOptions).run(context);
            });
        result.tiePointResult = effectiveRunner(result.config.tiePointOptions,
                                                result.config.tiePointContext);
        if (!result.tiePointResult.success)
        {
            result.serviceResult.errorMessage = QStringLiteral("连接点准备失败: %1")
                .arg(result.tiePointResult.errorMessage);
            result.serviceResult.summary = result.serviceResult.errorMessage;
            return result;
        }
        if (options.pairMatchedFn)
        {
            for (const matchphotos::MatchPhotosMatchRecord &match : result.tiePointResult.matches)
            {
                options.pairMatchedFn(match.image0Path,
                                      match.image1Path,
                                      match.matchPath,
                                      match.matchCount);
            }
        }
        result.config.serviceOptions.restrictPairs = result.tiePointResult.pairSelection.restrictPairs;
        result.config.serviceOptions.allowedPairs = result.tiePointResult.pairSelection.allowedPairKeys;
    }

    result.serviceResult = runner(result.config.serviceOptions);
    if (result.tiePointPreparationExecuted)
    {
        mergeTiePointOutputs(result.tiePointResult, &result.serviceResult);
    }
    result.serviceResult.resultRecordExtra.insert(QStringLiteral("workflow_kind"),
                                                  QStringLiteral("aerial_triangulation_align_photos"));
    result.serviceResult.resultRecordExtra.insert(QStringLiteral("resolved_settings"),
                                                  result.config.resolvedSettings);
    return result;
}

} // namespace xjw::gui
