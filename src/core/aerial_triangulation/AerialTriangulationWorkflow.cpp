#include "AerialTriangulationWorkflow.h"

#include <QDir>
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
    bool enableGuidedByDefault = false;
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
        return {3, -1, 1.0, true};
    }
    if (token == QStringLiteral("high"))
    {
        return {2, 0, 0.75, false};
    }
    if (token == QStringLiteral("medium") || token == QStringLiteral("standard"))
    {
        return {1, 4096, 0.50, false};
    }
    if (token == QStringLiteral("low"))
    {
        return {0, 2048, 0.25, false};
    }
    if (token == QStringLiteral("lowest") || token == QStringLiteral("fast"))
    {
        return {0, 1536, 0.125, false};
    }
    return {2, 0, 0.75, false};
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
    service.enableGuidedRematching = options.guidedImageMatching || preset.enableGuidedByDefault;
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

    return resolved;
}

AerialTriangulationWorkflowResult AerialTriangulationWorkflow::run(
    const AerialTriangulationWorkflowOptions &options,
    const ServiceRunner &runner)
{
    AerialTriangulationWorkflowResult result;
    result.config = resolveConfig(options);
    if (!runner)
    {
        result.serviceResult.success = false;
        result.serviceResult.errorMessage = QStringLiteral("空中三角测量 workflow 缺少空三服务执行器。");
        return result;
    }
    result.serviceResult = runner(result.config.serviceOptions);
    result.serviceResult.resultRecordExtra.insert(QStringLiteral("workflow_kind"),
                                                  QStringLiteral("aerial_triangulation_align_photos"));
    result.serviceResult.resultRecordExtra.insert(QStringLiteral("resolved_settings"),
                                                  result.config.resolvedSettings);
    return result;
}

} // namespace xjw::gui
