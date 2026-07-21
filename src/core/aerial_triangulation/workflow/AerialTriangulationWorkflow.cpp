#include "workflow/AerialTriangulationWorkflow.h"

#include "preparation/MatchResultCatalog.h"
#include "preparation/TiePointPreparation.h"
#include "project/ProjectIO.h"
#include "workflow/AerialTriangulationPipeline.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSet>

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw::aerial_triangulation
{
namespace
{

struct QualityPreset
{
    int sfmQuality = 2;
    int maxImageDimension = 3072;
};

QString normalizedToken(QString value, const QString &fallback = QString())
{
    value = value.trimmed().toLower();
    return value.isEmpty() ? fallback : value;
}

QualityPreset presetForQuality(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest")) return {3, 4096};
    if (token == QStringLiteral("medium") || token == QStringLiteral("standard")) return {1, 2048};
    if (token == QStringLiteral("low")) return {0, 1600};
    if (token == QStringLiteral("lowest") || token == QStringLiteral("fast")) return {0, 1200};
    return {2, 3072};
}

int sequenceWindowForQuality(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest")) return 8;
    if (token == QStringLiteral("high")) return 6;
    if (token == QStringLiteral("medium") || token == QStringLiteral("standard")) return 4;
    if (token == QStringLiteral("low")) return 3;
    return 2;
}

matchphotos::MatchPhotosProfile matchProfile(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest") || token == QStringLiteral("high"))
    {
        return matchphotos::MatchPhotosProfile::HighAccuracy;
    }
    if (token == QStringLiteral("low") || token == QStringLiteral("lowest") ||
        token == QStringLiteral("fast"))
    {
        return matchphotos::MatchPhotosProfile::Fast;
    }
    return matchphotos::MatchPhotosProfile::Auto;
}

matchphotos::PairSelectionPreset pairPreset(const QString &quality)
{
    switch (matchProfile(quality))
    {
    case matchphotos::MatchPhotosProfile::HighAccuracy:
        return matchphotos::PairSelectionPreset::HighAccuracy;
    case matchphotos::MatchPhotosProfile::Fast:
        return matchphotos::PairSelectionPreset::Fast;
    default:
        return matchphotos::PairSelectionPreset::Auto;
    }
}

matchphotos::ComputeDevice computeDevice(const QString &device)
{
    const QString token = normalizedToken(device, QStringLiteral("auto"));
    if (token == QStringLiteral("cpu")) return matchphotos::ComputeDevice::Cpu;
    if (token == QStringLiteral("cuda") || token == QStringLiteral("gpu"))
    {
        return matchphotos::ComputeDevice::Cuda;
    }
    return matchphotos::ComputeDevice::Auto;
}

QString normalizeMatcher(QString matcher)
{
    matcher = normalizedToken(matcher, QStringLiteral("lightglue"));
    matcher.replace(QLatin1Char('-'), QLatin1Char('_'));
    if (matcher == QStringLiteral("bf") || matcher == QStringLiteral("sift_bf_l2"))
    {
        return QStringLiteral("bf_l2");
    }
    if (matcher == QStringLiteral("sift_flann")) return QStringLiteral("flann");
    return matcher;
}

void applyPipelineToken(const QString &pipeline, QString *feature, QString *matcher)
{
    const QString token = normalizedToken(pipeline);
    if (token.isEmpty()) return;
    if (token == QStringLiteral("sift-bf-l2") || token == QStringLiteral("sift_bf_l2"))
    {
        *feature = QStringLiteral("sift");
        *matcher = QStringLiteral("bf_l2");
        return;
    }
    const QStringList pieces = token.split(QRegularExpression(QStringLiteral("[-_]")),
                                           Qt::SkipEmptyParts);
    if (pieces.size() >= 2)
    {
        *feature = pieces.front();
        *matcher = pieces.mid(1).join(QLatin1Char('_'));
    }
}

int tiePointProgress(const QString &stageId, int current, int maximum)
{
    struct Range { const char *id; int first; int last; };
    static constexpr Range ranges[] = {
        {"algorithm_selection", 1, 3}, {"feature", 3, 16},
        {"generic_preselection", 16, 21}, {"reference_preselection", 21, 25},
        {"pair_selection", 25, 28}, {"matching", 28, 33},
        {"geometry", 33, 34}, {"track_build", 34, 35}, {"guided_matching", 34, 35}};
    Range selected{"unknown", 1, 35};
    const QString token = normalizedToken(stageId, QStringLiteral("unknown"));
    for (const Range &range : ranges)
    {
        if (token == QLatin1String(range.id))
        {
            selected = range;
            break;
        }
    }
    const double fraction = std::clamp(static_cast<double>(std::max(0, current)) /
                                           std::max(1, maximum),
                                       0.0,
                                       1.0);
    return selected.first + static_cast<int>(std::round(
        (selected.last - selected.first) * fraction));
}

bool clearTiePointCache(const AerialTriangulationResolvedConfig &config, QString *errorMessage)
{
    QStringList failed;
    QDir matchDirectory(config.tiePointContext.matchDirectory);
    if (matchDirectory.exists())
    {
        MatchResultCatalogConfig catalogConfig;
        catalogConfig.matchDirectory = config.tiePointContext.matchDirectory;
        catalogConfig.targetImagePaths = config.tiePointContext.pairInput.images;
        const MatchResultCatalogSummary catalog = MatchResultCatalog(catalogConfig).scan();
        QSet<QString> removalPaths;
        for (const MatchPairGroup &group : catalog.pairGroups)
        {
            for (const MatchVariant &variant : group.variants)
            {
                if (!variant.matchFilePath.trimmed().isEmpty())
                {
                    removalPaths.insert(QDir::cleanPath(variant.matchFilePath));
                }
                if (!variant.sidecarPath.trimmed().isEmpty())
                {
                    removalPaths.insert(QDir::cleanPath(variant.sidecarPath));
                }
            }
        }

        for (const QString &path : std::as_const(removalPaths))
        {
            if (QFileInfo::exists(path) && !QFile::remove(path))
            {
                failed.append(path);
            }
        }
    }
    const QString tiePointPath = config.pipelineInput.tiePointPath;
    if (QFileInfo::exists(tiePointPath) && !QFile::remove(tiePointPath))
    {
        failed.append(tiePointPath);
    }
    if (failed.isEmpty()) return true;
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("无法清理旧连接点缓存：\n%1")
            .arg(failed.join(QLatin1Char('\n')));
    }
    return false;
}

} // namespace

AerialTriangulationResolvedConfig AerialTriangulationWorkflow::resolveConfig(
    const AerialTriangulationOptions &options)
{
    AerialTriangulationResolvedConfig resolved;
    const QualityPreset quality = presetForQuality(options.quality);
    QString feature = normalizedToken(options.featureAlgorithm, QStringLiteral("sift"));
    QString matcher = normalizeMatcher(options.matchAlgorithm);
    applyPipelineToken(options.matchPipeline, &feature, &matcher);
    matcher = normalizeMatcher(matcher);

    const QString projectRoot = QFileInfo(options.projectPath).absolutePath();
    const QString assetsDirectory = options.assetsDir.isEmpty()
        ? QDir(projectRoot).filePath(QStringLiteral("assets"))
        : QDir::cleanPath(options.assetsDir);
    const QString canonicalTiePointPath = QDir(assetsDirectory)
        .filePath(QStringLiteral("tie_points/latest_tie_points.json"));

    PreparedAerialTriangulationInput &pipeline = resolved.pipelineInput;
    pipeline.images = options.images;
    pipeline.cameraPaths = options.cameraPaths;
    pipeline.projectPath = options.projectPath;
    pipeline.markerSetPath = options.projectPath.isEmpty()
        ? QString()
        : xjw::common::project::ProjectIO::markerSetPath(options.projectPath);
    pipeline.tiePointPath = canonicalTiePointPath;
    pipeline.outputDir = QDir(QDir::cleanPath(options.outputDir))
        .filePath(QStringLiteral("sfm_sparse"));
    pipeline.projectMeta = options.projectMeta;
    pipeline.quality = quality.sfmQuality;
    pipeline.threads = std::max(1, options.threads);
    pipeline.device = normalizedToken(options.device, QStringLiteral("auto"));
    pipeline.useProjectCameraIntrinsics = true;
    pipeline.useProjectCameraPoses = !options.resetAlignment;
    pipeline.adaptiveCameraModelFitting = options.adaptiveCameraModelFitting;
    pipeline.enforceSequencePoseConsistency = options.referencePreselection &&
        normalizedToken(options.referenceMode, QStringLiteral("source_code")) == QStringLiteral("sequence");
    pipeline.sequenceLoopClosure = options.referencePreselection &&
        normalizedToken(options.referenceMode, QStringLiteral("source_code")) == QStringLiteral("sequence");
    pipeline.useInitialPairHint = options.useInitialPairHint;
    pipeline.initialImageId1 = options.initialImageId1;
    pipeline.initialImageId2 = options.initialImageId2;
    pipeline.cancelFlag = options.cancelFlag;
    if (options.progressFn)
    {
        pipeline.progressFn = [progress = options.progressFn](const QString &stage, int percent)
        {
            progress(stage, 35 + std::clamp(percent, 0, 100) * 65 / 100);
        };
    }

    matchphotos::MatchPhotosOptions &tieOptions = resolved.tiePointOptions;
    tieOptions.planOnly = false;
    tieOptions.profile = matchProfile(options.quality);
    tieOptions.device = computeDevice(options.device);
    tieOptions.pairPolicy = matchphotos::makePairSelectionPolicy(pairPreset(options.quality));
    tieOptions.featureAlgorithm = feature;
    tieOptions.matcherAlgorithm = matcher;
    tieOptions.maskApplyMode = normalizedToken(options.maskApplyMode, QStringLiteral("none"));
    tieOptions.maxImageDim = options.featureMaxImageDim == 0
        ? quality.maxImageDimension : options.featureMaxImageDim;
    tieOptions.enableGuidedMatching = options.guidedImageMatching;
    tieOptions.useExplicitKeypointLimit = true;
    tieOptions.maxKeypoints = std::max(0, options.keypointLimit);
    tieOptions.keypointLimitPerMegapixel = 0;
    tieOptions.maxTiePointsPerImage = std::max(0, options.tiepointLimit);
    tieOptions.tiePointGridColumns = 8;
    tieOptions.tiePointGridRows = 8;
    const int tiePointGridCellCount = tieOptions.tiePointGridColumns * tieOptions.tiePointGridRows;
    tieOptions.maxTiePointsPerGridCell = options.tiepointLimit > 0
        ? std::max(1, options.tiepointLimit / std::max(1, tiePointGridCellCount)) : 0;
    tieOptions.useGenericPreselection = options.genericPreselection;
    tieOptions.useReferencePreselection = options.referencePreselection;
    tieOptions.excludeStationaryTiePoints = options.excludeFixedTiePoints;
    tieOptions.reuseExistingFeatures = !options.resetAlignment &&
        tieOptions.maskApplyMode != QStringLiteral("keypoints");

    const QString referenceMode = normalizedToken(options.referenceMode,
                                                   QStringLiteral("source_code"));
    const bool hasReference = !options.referenceCameras.isEmpty() ||
        (!options.cameraPaths.isEmpty() && options.cameraPaths.size() == options.images.size());
    QString pairPlanningMode = options.genericPreselection ? QStringLiteral("generic")
                                                           : QStringLiteral("all_pairs");
    if (options.referencePreselection && referenceMode == QStringLiteral("sequence"))
    {
        tieOptions.pairPolicy.mode = matchphotos::PairSelectionMode::Sequence;
        tieOptions.pairPolicy.sequenceWindow = sequenceWindowForQuality(options.quality);
        tieOptions.useReferencePreselection = false;
        pairPlanningMode = QStringLiteral("sequence");
    }
    else if (options.referencePreselection && hasReference)
    {
        pairPlanningMode = referenceMode;
    }
    else if (options.referencePreselection)
    {
        tieOptions.useReferencePreselection = false;
        pairPlanningMode = QStringLiteral("generic_reference_unavailable");
    }
    if (!options.allowedPairs.isEmpty())
    {
        tieOptions.pairPolicy.mode = matchphotos::PairSelectionMode::ManualOnly;
        pairPlanningMode = QStringLiteral("manual");
    }

    matchphotos::MatchPhotosContext &tieContext = resolved.tiePointContext;
    tieContext.projectPath = options.projectPath;
    tieContext.workingDirectory = assetsDirectory;
    tieContext.featureDirectory = options.featureDir.isEmpty()
        ? QDir(assetsDirectory).filePath(QStringLiteral("ip"))
        : QDir::cleanPath(options.featureDir);
    tieContext.matchDirectory = options.matchDir.isEmpty()
        ? QDir(assetsDirectory).filePath(QStringLiteral("matches"))
        : QDir::cleanPath(options.matchDir);
    tieContext.pairInput.images = options.images;
    tieContext.pairInput.manualPairKeys = options.allowedPairs;
    tieContext.referenceCameras = options.referenceCameras;
    tieContext.maskPaths = options.maskPaths;
    tieContext.cancelFlag = options.cancelFlag.get();
    if (options.progressFn)
    {
        tieContext.progressCallback = [progress = options.progressFn](const QString &stageId,
                                                                      const QString &message,
                                                                      int current,
                                                                      int maximum)
        {
            progress(message.isEmpty() ? stageId : message,
                     tiePointProgress(stageId, current, maximum));
        };
    }

    resolved.prepareTiePoints = options.resetAlignment || options.autoGenerateMissingMatches ||
        !QFileInfo::exists(canonicalTiePointPath);
    resolved.forceRebuildTiePoints = options.resetAlignment;
    QJsonObject &settings = resolved.resolvedSettings;
    settings.insert(QStringLiteral("quality"), normalizedToken(options.quality, QStringLiteral("high")));
    settings.insert(QStringLiteral("feature_algorithm"), feature);
    settings.insert(QStringLiteral("match_algorithm"), matcher);
    settings.insert(QStringLiteral("pair_planning_mode"), pairPlanningMode);
    settings.insert(QStringLiteral("sequence_pair_window"), tieOptions.pairPolicy.sequenceWindow);
    settings.insert(QStringLiteral("sequence_loop_closure"), pipeline.sequenceLoopClosure);
    settings.insert(QStringLiteral("keypoint_limit"), options.keypointLimit);
    settings.insert(QStringLiteral("tiepoint_limit"), options.tiepointLimit);
    settings.insert(QStringLiteral("mask_apply_mode"), tieOptions.maskApplyMode);
    settings.insert(QStringLiteral("reset_current_alignment"), options.resetAlignment);
    settings.insert(QStringLiteral("use_project_camera_intrinsics"),
                    pipeline.useProjectCameraIntrinsics);
    settings.insert(QStringLiteral("use_project_camera_poses"), pipeline.useProjectCameraPoses);
    settings.insert(QStringLiteral("adaptive_camera_model_fitting"),
                    options.adaptiveCameraModelFitting);
    settings.insert(QStringLiteral("cuda_parallel_pairs_requested"), options.cudaParallelPairs);
    settings.insert(QStringLiteral("cuda_parallel_pairs_effective"), 0);
    settings.insert(QStringLiteral("reference_preselection_available"),
                    referenceMode == QStringLiteral("sequence") || hasReference);
    settings.insert(QStringLiteral("tie_point_preparation"),
                    resolved.forceRebuildTiePoints
                        ? QStringLiteral("force_rebuild")
                        : (resolved.prepareTiePoints ? QStringLiteral("fill_missing")
                                                     : QStringLiteral("reuse")));
    return resolved;
}

AerialTriangulationResult AerialTriangulationWorkflow::run(
    const AerialTriangulationOptions &options)
{
    return run(options, [](const PreparedAerialTriangulationInput &input)
    {
        return AerialTriangulationPipeline().run(input);
    });
}

AerialTriangulationResult AerialTriangulationWorkflow::run(
    const AerialTriangulationOptions &options,
    const PipelineRunner &pipelineRunner,
    const TiePointRunner &tiePointRunner)
{
    AerialTriangulationResult result;
    result.config = resolveConfig(options);
    if (!pipelineRunner)
    {
        result.reconstructionResult.errorMessage =
            QStringLiteral("空中三角测量 workflow 缺少 SfM 管线执行器");
        result.reconstructionResult.summary = result.reconstructionResult.errorMessage;
        return result;
    }

    if (result.config.prepareTiePoints)
    {
        result.tiePointPreparationExecuted = true;
        if (result.config.forceRebuildTiePoints)
        {
            QString cleanupError;
            if (!clearTiePointCache(result.config, &cleanupError))
            {
                result.reconstructionResult.errorMessage = cleanupError;
                result.reconstructionResult.summary = cleanupError;
                return result;
            }
        }
        result.tiePointResult = TiePointPreparation::run(result.config.tiePointOptions,
                                                         result.config.tiePointContext,
                                                         tiePointRunner);
        if (!result.tiePointResult.success)
        {
            result.reconstructionResult.errorMessage = QStringLiteral("连接点准备失败: %1")
                .arg(result.tiePointResult.errorMessage);
            result.reconstructionResult.summary = result.reconstructionResult.errorMessage;
            return result;
        }
        if (!result.tiePointResult.tiePointPath.trimmed().isEmpty())
        {
            result.config.pipelineInput.tiePointPath = result.tiePointResult.tiePointPath;
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
    }

    result.reconstructionResult = pipelineRunner(result.config.pipelineInput);
    QJsonObject extra = result.reconstructionResult.resultRecordExtra;
    extra.insert(QStringLiteral("workflow_kind"),
                 QStringLiteral("aerial_triangulation_align_photos"));
    extra.insert(QStringLiteral("resolved_settings"), result.config.resolvedSettings);
    extra.insert(QStringLiteral("tie_point_path"), result.config.pipelineInput.tiePointPath);
    if (result.tiePointPreparationExecuted)
    {
        extra.insert(QStringLiteral("tie_point_track_count"), result.tiePointResult.trackCount);
        extra.insert(QStringLiteral("tie_point_match_file_count"),
                     static_cast<int>(result.tiePointResult.matches.size()));
    }
    result.reconstructionResult.resultRecordExtra = extra;
    return result;
}

} // namespace xjw::aerial_triangulation
