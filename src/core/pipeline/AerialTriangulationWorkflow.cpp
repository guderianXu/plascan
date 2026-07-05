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

void applyPreselection(AerialTriangulationResolvedConfig &resolved,
                       const AerialTriangulationWorkflowOptions &options)
{
    SFMServiceOptions &sfm = resolved.sfmOptions;
    const QString referenceMode = normalizedToken(options.referenceMode, QStringLiteral("source_code"));

    sfm.autoRestrictKnownCameraPairs = true;
    sfm.knownCameraPairWindow = 4;
    sfm.knownCameraSpatialNeighborCount = 8;
    sfm.useKnownCameraOverlapPairs = options.referencePreselection;
    QString pairPlanningMode = QStringLiteral("auto");

    if (options.referencePreselection && referenceMode == QStringLiteral("sequence"))
    {
        sfm.useKnownCameraOverlapPairs = false;
        sfm.knownCameraSpatialNeighborCount = 0;
        pairPlanningMode = QStringLiteral("sequence");
    }
    else if (options.referencePreselection && referenceMode == QStringLiteral("estimated"))
    {
        sfm.useKnownCameraOverlapPairs = false;
        sfm.knownCameraSpatialNeighborCount = 8;
        pairPlanningMode = QStringLiteral("estimated");
    }
    else if (options.referencePreselection)
    {
        sfm.useKnownCameraOverlapPairs = true;
        sfm.knownCameraSpatialNeighborCount = 8;
        pairPlanningMode = QStringLiteral("source_code");
    }
    else if (options.genericPreselection)
    {
        pairPlanningMode = QStringLiteral("generic");
    }

    resolved.resolvedSettings.insert(QStringLiteral("pair_planning_mode"), pairPlanningMode);
}

} // namespace

AerialTriangulationResolvedConfig AerialTriangulationWorkflow::resolveConfig(
    const AerialTriangulationWorkflowOptions &options)
{
    AerialTriangulationResolvedConfig resolved;
    SFMServiceOptions &sfm = resolved.sfmOptions;

    const QualityPreset preset = presetForQuality(options.quality);
    QString featureAlgorithm = normalizedToken(options.featureAlgorithm, QStringLiteral("disk"));
    QString matchAlgorithm = normalizeMatchAlgorithm(options.matchAlgorithm);
    applyMatchPipeline(options.matchPipeline, featureAlgorithm, matchAlgorithm);
    matchAlgorithm = normalizeMatchAlgorithm(matchAlgorithm);

    sfm.images = options.images;
    sfm.cameraPaths = options.cameraPaths;
    sfm.plascanPath = options.projectPath;
    sfm.projectMeta = options.projectMeta;
    sfm.outputDir = QDir(QDir::cleanPath(options.outputDir)).filePath(QStringLiteral("sfm_sparse"));
    sfm.quality = preset.sfmQuality;
    sfm.threads = std::max(1, options.threads);
    sfm.device = normalizedToken(options.device, QStringLiteral("auto"));
    sfm.featureAlgorithm = featureAlgorithm;
    sfm.matchAlgorithm = matchAlgorithm;
    sfm.featureMaxImageDim = preset.featureMaxImageDim;
    sfm.featureGrayscaleMin = options.featureGrayscaleMin;
    sfm.featureGrayscaleMax = options.featureGrayscaleMax;
    sfm.autoGenerateMissingMatches = options.autoGenerateMissingMatches;
    sfm.restrictPairs = options.restrictPairs;
    sfm.allowedPairs = options.allowedPairs;
    sfm.enableTwoStageMatching = true;
    sfm.enableGuidedRematching = options.guidedImageMatching || preset.enableGuidedByDefault;
    sfm.skeletonFeatureMaxKeypoints = scaledLimit(options.keypointLimit, preset.budgetScale);
    sfm.maxTiePointsPerImage = scaledLimit(options.tiepointLimit, preset.budgetScale);
    sfm.tiePointGridColumns = 8;
    sfm.tiePointGridRows = 8;
    sfm.maxTiePointsPerGridCell = sfm.maxTiePointsPerImage > 0
        ? std::max(1, sfm.maxTiePointsPerImage / sfm.tiePointGridColumns)
        : 0;
    sfm.cancelFlag = options.cancelFlag;
    sfm.progressFn = options.progressFn;
    sfm.pairMatchedFn = options.pairMatchedFn;
    sfm.cudaParallelPairs = sfm.device == QStringLiteral("cpu")
        ? 1
        : std::clamp(std::max(1, sfm.threads / 4), 1, 2);

    applyPreselection(resolved, options);

    resolved.resolvedSettings.insert(QStringLiteral("quality"), normalizedToken(options.quality, QStringLiteral("high")));
    resolved.resolvedSettings.insert(QStringLiteral("match_pipeline"),
                                     effectivePipelineName(options.matchPipeline, featureAlgorithm, matchAlgorithm));
    resolved.resolvedSettings.insert(QStringLiteral("feature_algorithm"), featureAlgorithm);
    resolved.resolvedSettings.insert(QStringLiteral("match_algorithm"), matchAlgorithm);
    resolved.resolvedSettings.insert(QStringLiteral("keypoint_limit"), options.keypointLimit);
    resolved.resolvedSettings.insert(QStringLiteral("resolved_keypoint_budget"), sfm.skeletonFeatureMaxKeypoints);
    resolved.resolvedSettings.insert(QStringLiteral("tiepoint_limit"), options.tiepointLimit);
    resolved.resolvedSettings.insert(QStringLiteral("resolved_tiepoint_limit"), sfm.maxTiePointsPerImage);
    resolved.resolvedSettings.insert(QStringLiteral("mask_apply_mode"), normalizedToken(options.maskApplyMode, QStringLiteral("none")));
    resolved.resolvedSettings.insert(QStringLiteral("exclude_fixed_tie_points"), options.excludeFixedTiePoints);
    resolved.resolvedSettings.insert(QStringLiteral("reset_current_alignment"), options.resetAlignment);
    resolved.resolvedSettings.insert(QStringLiteral("save_project_after_each_step"), options.saveAfterEachStep);
    resolved.resolvedSettings.insert(QStringLiteral("adaptive_camera_model_fitting"), options.adaptiveCameraModelFitting);
    resolved.resolvedSettings.insert(QStringLiteral("adaptive_known_pose_soft_prior"), true);
    resolved.resolvedSettings.insert(QStringLiteral("restrict_pairs"), sfm.restrictPairs);
    resolved.resolvedSettings.insert(QStringLiteral("allowed_pair_count"), sfm.allowedPairs.size());

    return resolved;
}

AerialTriangulationWorkflowResult AerialTriangulationWorkflow::run(
    const AerialTriangulationWorkflowOptions &options,
    const SfmRunner &runner)
{
    AerialTriangulationWorkflowResult result;
    result.config = resolveConfig(options);
    if (!runner)
    {
        result.sfmResult.success = false;
        result.sfmResult.errorMessage = QStringLiteral("空中三角测量 workflow 缺少 SfM 执行器。");
        return result;
    }
    result.sfmResult = runner(result.config.sfmOptions);
    result.sfmResult.resultRecordExtra.insert(QStringLiteral("workflow_kind"),
                                              QStringLiteral("aerial_triangulation_align_photos"));
    result.sfmResult.resultRecordExtra.insert(QStringLiteral("resolved_settings"),
                                              result.config.resolvedSettings);
    return result;
}

} // namespace xjw::gui
