#include "MvsWorkspaceManifest.h"

#include "MvsTypes.h"

#include <QCollator>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

#include <algorithm>

namespace xjw::mvs
{

namespace
{
QString frameSortName(const MvsDepthFrameRecord &record)
{
    if (!record.refImage.isEmpty())
    {
        return QFileInfo(record.refImage).fileName();
    }
    if (!record.depthPng.isEmpty())
    {
        return QFileInfo(record.depthPng).fileName();
    }
    return QString::number(record.refIndex);
}

QStringList jsonArrayToStringList(const QJsonArray &array)
{
    QStringList result;
    result.reserve(array.size());
    for (const QJsonValue &value : array)
    {
        result.push_back(value.toString());
    }
    return result;
}

QJsonArray stringListToJsonArray(const QStringList &strings)
{
    QJsonArray array;
    for (const QString &string : strings)
    {
        array.push_back(string);
    }
    return array;
}
}

QJsonObject MvsDepthFrameRecord::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("ref_index"), refIndex);
    object.insert(QStringLiteral("ref_image"), refImage);
    object.insert(QStringLiteral("source_images"), stringListToJsonArray(sourceImages));
    object.insert(QStringLiteral("source_plan"), sourcePlan);
    object.insert(QStringLiteral("source_view_count"), sourceViewCount);
    object.insert(QStringLiteral("source_quality_mean"), meanSourceQualityScore);
    object.insert(QStringLiteral("source_quality_min"), minSourceQualityScore);
    object.insert(QStringLiteral("depth_confidence_mean"), meanDepthConfidence);
    object.insert(QStringLiteral("valid_pixel_count"), validPixelCount);
    object.insert(QStringLiteral("depth_quality"), depthQuality);
    object.insert(QStringLiteral("quality_decision"), qualityDecision);
    object.insert(QStringLiteral("pyramid_levels"), pyramidLevels);
    object.insert(QStringLiteral("scene_profile"), sceneProfile);
    object.insert(QStringLiteral("filter_mode"), filterMode);
    object.insert(QStringLiteral("acceptance"), acceptance);
    object.insert(QStringLiteral("depth_postprocess"), depthPostprocess);
    object.insert(QStringLiteral("camera_model"), cameraModel);
    object.insert(QStringLiteral("status"), status);
    object.insert(QStringLiteral("device"), device);
    object.insert(QStringLiteral("depth_png"), depthPng);
    object.insert(QStringLiteral("raw_depth_path"), rawDepthPath);
    object.insert(QStringLiteral("raw_confidence_path"), rawConfidencePath);
    object.insert(QStringLiteral("valid_mask_path"), validMaskPath);
    object.insert(QStringLiteral("grid_width"), gridWidth);
    object.insert(QStringLiteral("grid_height"), gridHeight);
    object.insert(QStringLiteral("elapsed_ms"), QString::number(elapsedMs));
    object.insert(QStringLiteral("error"), error);
    object.insert(QStringLiteral("config_hash"), configHash);
    return object;
}

MvsDepthFrameRecord MvsDepthFrameRecord::fromJson(const QJsonObject &object)
{
    MvsDepthFrameRecord record;
    record.refIndex = object.value(QStringLiteral("ref_index")).toInt(-1);
    record.refImage = object.value(QStringLiteral("ref_image")).toString();
    record.sourceImages = jsonArrayToStringList(object.value(QStringLiteral("source_images")).toArray());
    record.sourcePlan = object.value(QStringLiteral("source_plan")).toArray();
    record.sourceViewCount = object.value(QStringLiteral("source_view_count")).toInt(0);
    record.meanSourceQualityScore = object.value(QStringLiteral("source_quality_mean")).toDouble(0.0);
    record.minSourceQualityScore = object.value(QStringLiteral("source_quality_min")).toDouble(0.0);
    record.meanDepthConfidence = object.value(QStringLiteral("depth_confidence_mean")).toDouble(0.0);
    record.validPixelCount = object.value(QStringLiteral("valid_pixel_count")).toInt(0);
    record.depthQuality = object.value(QStringLiteral("depth_quality")).toObject();
    record.qualityDecision = object.value(QStringLiteral("quality_decision")).toObject();
    record.pyramidLevels = object.value(QStringLiteral("pyramid_levels")).toArray();
    record.sceneProfile = object.value(QStringLiteral("scene_profile")).toString();
    record.filterMode = object.value(QStringLiteral("filter_mode")).toString();
    record.acceptance = object.value(QStringLiteral("acceptance")).toString(
        record.qualityDecision.value(QStringLiteral("acceptance")).toString());
    record.depthPostprocess = object.value(QStringLiteral("depth_postprocess")).toObject();
    record.cameraModel = object.value(QStringLiteral("camera_model")).toObject();
    record.status = object.value(QStringLiteral("status")).toString();
    record.device = object.value(QStringLiteral("device")).toString();
    record.depthPng = object.value(QStringLiteral("depth_png")).toString();
    record.rawDepthPath = object.value(QStringLiteral("raw_depth_path")).toString();
    record.rawConfidencePath = object.value(QStringLiteral("raw_confidence_path")).toString();
    record.validMaskPath = object.value(QStringLiteral("valid_mask_path")).toString();
    record.gridWidth = object.value(QStringLiteral("grid_width")).toInt(0);
    record.gridHeight = object.value(QStringLiteral("grid_height")).toInt(0);
    const QJsonValue elapsed = object.value(QStringLiteral("elapsed_ms"));
    record.elapsedMs = elapsed.isString() ? elapsed.toString().toLongLong()
                                          : static_cast<qint64>(elapsed.toDouble(0.0));
    record.error = object.value(QStringLiteral("error")).toString();
    record.configHash = object.value(QStringLiteral("config_hash")).toString();
    if (record.pyramidLevels.isEmpty() && record.gridWidth > 0 && record.gridHeight > 0)
    {
        record.pyramidLevels.append(QJsonObject{
            {QStringLiteral("level"), 1},
            {QStringLiteral("downsample_factor"), 1},
            {QStringLiteral("grid_width"), record.gridWidth},
            {QStringLiteral("grid_height"), record.gridHeight},
            {QStringLiteral("legacy_single_level"), true}
        });
    }
    return record;
}

bool MvsWorkspaceManifest::load(const QString &path, QString *errorMsg)
{
    clear();

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMsg)
        {
            *errorMsg = file.errorString();
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    {
        if (errorMsg)
        {
            *errorMsg = parseError.errorString();
        }
        return false;
    }

    const QJsonObject root = doc.object();
    _configHash = root.value(QStringLiteral("config_hash")).toString();
    const QJsonArray frames = root.value(QStringLiteral("frames")).toArray();
    _frames.reserve(frames.size());
    for (const QJsonValue &value : frames)
    {
        if (value.isObject())
        {
            _frames.push_back(MvsDepthFrameRecord::fromJson(value.toObject()));
        }
    }
    return true;
}

bool MvsWorkspaceManifest::saveAtomic(const QString &path, QString *errorMsg) const
{
    const QFileInfo info(path);
    if (!info.absoluteDir().exists() && !QDir().mkpath(info.absolutePath()))
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("无法创建目录: %1").arg(info.absolutePath());
        }
        return false;
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (errorMsg)
        {
            *errorMsg = file.errorString();
        }
        return false;
    }

    file.write(QJsonDocument(toJson()).toJson(QJsonDocument::Indented));
    if (!file.commit())
    {
        if (errorMsg)
        {
            *errorMsg = file.errorString();
        }
        return false;
    }
    return true;
}

void MvsWorkspaceManifest::clear()
{
    _configHash.clear();
    _frames.clear();
}

QString MvsWorkspaceManifest::configHash() const
{
    return _configHash;
}

void MvsWorkspaceManifest::setConfigHash(const QString &hash)
{
    _configHash = hash;
}

const QVector<MvsDepthFrameRecord> &MvsWorkspaceManifest::frames() const
{
    return _frames;
}

QVector<MvsDepthFrameRecord> MvsWorkspaceManifest::completedFramesSortedByName() const
{
    QVector<MvsDepthFrameRecord> result;
    for (const MvsDepthFrameRecord &record : _frames)
    {
        if (record.status == QStringLiteral("completed"))
        {
            result.push_back(record);
        }
    }

    QCollator collator;
    collator.setNumericMode(true);
    collator.setCaseSensitivity(Qt::CaseInsensitive);
    std::stable_sort(result.begin(), result.end(),
                     [&collator](const MvsDepthFrameRecord &lhs, const MvsDepthFrameRecord &rhs)
                     {
                         return collator.compare(frameSortName(lhs), frameSortName(rhs)) < 0;
                     });
    return result;
}

void MvsWorkspaceManifest::upsertFrame(const MvsDepthFrameRecord &record)
{
    const int index = findFrameIndex(record.refIndex);
    if (index >= 0)
    {
        _frames[index] = record;
    }
    else
    {
        _frames.push_back(record);
    }
}

void MvsWorkspaceManifest::markRunning(int refIndex, const QString &refImage, const QString &configHash)
{
    MvsDepthFrameRecord record;
    const int index = findFrameIndex(refIndex);
    if (index >= 0)
    {
        record = _frames[index];
    }
    record.refIndex = refIndex;
    record.refImage = refImage;
    record.status = QStringLiteral("running");
    record.error.clear();
    record.configHash = configHash;
    upsertFrame(record);
}

void MvsWorkspaceManifest::markCompleted(const MvsDepthFrameRecord &record)
{
    MvsDepthFrameRecord completed = record;
    const int index = findFrameIndex(completed.refIndex);
    if (completed.sourcePlan.isEmpty() && index >= 0)
    {
        completed.sourcePlan = _frames[index].sourcePlan;
    }
    if (completed.sourceViewCount <= 0 && index >= 0)
    {
        completed.sourceViewCount = _frames[index].sourceViewCount;
        completed.meanSourceQualityScore = _frames[index].meanSourceQualityScore;
        completed.minSourceQualityScore = _frames[index].minSourceQualityScore;
    }
    if (completed.validPixelCount <= 0 && index >= 0)
    {
        completed.validPixelCount = _frames[index].validPixelCount;
        completed.meanDepthConfidence = _frames[index].meanDepthConfidence;
    }
    if (completed.depthQuality.isEmpty() && index >= 0)
    {
        completed.depthQuality = _frames[index].depthQuality;
    }
    if (completed.qualityDecision.isEmpty() && index >= 0)
    {
        completed.qualityDecision = _frames[index].qualityDecision;
    }
    if (completed.pyramidLevels.isEmpty() && index >= 0)
    {
        completed.pyramidLevels = _frames[index].pyramidLevels;
    }
    if (completed.sceneProfile.isEmpty() && index >= 0)
    {
        completed.sceneProfile = _frames[index].sceneProfile;
    }
    if (completed.filterMode.isEmpty() && index >= 0)
    {
        completed.filterMode = _frames[index].filterMode;
    }
    if (completed.acceptance.isEmpty() && index >= 0)
    {
        completed.acceptance = _frames[index].acceptance;
    }
    if (completed.depthPostprocess.isEmpty() && index >= 0)
    {
        completed.depthPostprocess = _frames[index].depthPostprocess;
    }
    if (completed.cameraModel.isEmpty() && index >= 0)
    {
        completed.cameraModel = _frames[index].cameraModel;
    }
    completed.status = QStringLiteral("completed");
    completed.error.clear();
    upsertFrame(completed);
}

void MvsWorkspaceManifest::markFailed(int refIndex, const QString &error)
{
    MvsDepthFrameRecord record;
    const int index = findFrameIndex(refIndex);
    if (index >= 0)
    {
        record = _frames[index];
    }
    record.refIndex = refIndex;
    record.status = QStringLiteral("failed");
    record.error = error;
    upsertFrame(record);
}

bool MvsWorkspaceManifest::hasReusableCompletedFrame(int refIndex, const QString &configHash) const
{
    const int index = findFrameIndex(refIndex);
    if (index < 0)
    {
        return false;
    }
    const MvsDepthFrameRecord &record = _frames[index];
    if (record.status != QStringLiteral("completed") || record.configHash != configHash)
    {
        return false;
    }
    return QFileInfo::exists(record.depthPng) &&
           (record.rawDepthPath.isEmpty() || QFileInfo::exists(record.rawDepthPath));
}

QJsonObject MvsWorkspaceManifest::toJson() const
{
    QJsonObject root;
    root.insert(QStringLiteral("schema"), QStringLiteral("plascan.mvs.workspace.v2"));
    root.insert(QStringLiteral("config_hash"), _configHash);

    QJsonArray frames;
    for (const MvsDepthFrameRecord &record : _frames)
    {
        frames.push_back(record.toJson());
    }
    root.insert(QStringLiteral("frames"), frames);
    return root;
}

int MvsWorkspaceManifest::findFrameIndex(int refIndex) const
{
    for (int i = 0; i < _frames.size(); ++i)
    {
        if (_frames[i].refIndex == refIndex)
        {
            return i;
        }
    }
    return -1;
}

QString makeMvsDepthConfigHash(const DepthGenConfig &config, int viewCount)
{
    QJsonObject patch;
    patch.insert(QStringLiteral("num_iterations"), config.patchMatch.numIterations);
    patch.insert(QStringLiteral("patch_half"), config.patchMatch.patchHalf);
    patch.insert(QStringLiteral("num_source_views"), config.patchMatch.numSourceViews);
    patch.insert(QStringLiteral("confidence_thresh"), config.patchMatch.confidenceThresh);
    patch.insert(QStringLiteral("use_cuda"), config.patchMatch.useCuda);
    patch.insert(QStringLiteral("downsample_factor"), config.patchMatch.downsampleFactor);
    patch.insert(QStringLiteral("median_blur"), config.patchMatch.doMedianBlur);
    patch.insert(QStringLiteral("median_kernel"), config.patchMatch.medianKernelSize);
    patch.insert(QStringLiteral("bilateral"), config.patchMatch.doBilateralFilter);
    patch.insert(QStringLiteral("bilateral_d"), config.patchMatch.bilateralD);
    patch.insert(QStringLiteral("bilateral_sigma_color"), config.patchMatch.bilateralSigmaColor);
    patch.insert(QStringLiteral("bilateral_sigma_space"), config.patchMatch.bilateralSigmaSpace);
    patch.insert(QStringLiteral("geom_consistency"), config.patchMatch.geomConsistency);
    patch.insert(QStringLiteral("geom_consistency_max_err"), config.patchMatch.geomConsistencyMaxErr);
    patch.insert(QStringLiteral("epipolar_rectified"), config.patchMatch.epipolarRectified);
    patch.insert(QStringLiteral("cuda_parallel_sweep"), config.patchMatch.cudaUseParallelSweep);
    patch.insert(QStringLiteral("cuda_fallback_to_cpu"), config.patchMatch.cudaFallbackToCpu);

    QJsonObject fusion;
    fusion.insert(QStringLiteral("min_consistent_views"), config.fusion.minConsistentViews);
    fusion.insert(QStringLiteral("rel_depth_thresh"), config.fusion.relDepthThresh);
    fusion.insert(QStringLiteral("pixel_thresh"), config.fusion.pixelThresh);
    fusion.insert(QStringLiteral("confidence_thresh"), config.fusion.confidenceThresh);
    fusion.insert(QStringLiteral("adaptive_confidence"), config.fusion.enableAdaptiveConfidenceFilter);
    fusion.insert(QStringLiteral("adaptive_full_coverage"), config.fusion.adaptiveFullCoverageThreshold);
    fusion.insert(QStringLiteral("adaptive_low_mean_confidence"),
                  config.fusion.adaptiveLowMeanConfidenceThreshold);
    fusion.insert(QStringLiteral("adaptive_strict_confidence"),
                  config.fusion.adaptiveStrictConfidenceThreshold);
    fusion.insert(QStringLiteral("sigma_fusion"), config.fusion.doSigmaFusion);
    fusion.insert(QStringLiteral("sigma_multiplier"), config.fusion.sigmaMultiplier);
    fusion.insert(QStringLiteral("inpaint"), config.fusion.doInpaint);
    fusion.insert(QStringLiteral("inpaint_radius_factor"), config.fusion.inpaintRadiusFactor);
    fusion.insert(QStringLiteral("inpaint_radius"), config.fusion.inpaintRadius);
    fusion.insert(QStringLiteral("local_outlier"), config.fusion.enableLocalDepthOutlierFilter);
    fusion.insert(QStringLiteral("local_outlier_kernel"), config.fusion.localDepthOutlierKernelSize);
    fusion.insert(QStringLiteral("local_outlier_rel_thresh"), config.fusion.localDepthOutlierRelThresh);
    fusion.insert(QStringLiteral("local_outlier_max_removal_ratio"),
                  config.fusion.maxLocalDepthOutlierRemovalRatio);
    fusion.insert(QStringLiteral("speckle_filter"), config.fusion.enableSpeckleFilter);
    fusion.insert(QStringLiteral("speckle_min_area"), config.fusion.minSpeckleComponentArea);
    fusion.insert(QStringLiteral("speckle_max_removal_ratio"), config.fusion.maxSpeckleRemovalRatio);

    QJsonObject root;
    root.insert(QStringLiteral("schema"), QStringLiteral("plascan.mvs.depth.config.v3"));
    root.insert(QStringLiteral("view_count"), viewCount);
    root.insert(QStringLiteral("input_signature"),
                QString::fromStdString(config.inputSignature));
    root.insert(QStringLiteral("num_source_views"), config.numSourceViews);
    root.insert(QStringLiteral("z_near_scale"), config.zNearScale);
    root.insert(QStringLiteral("z_far_scale"), config.zFarScale);
    root.insert(QStringLiteral("scene_profile"), static_cast<int>(config.sceneProfile));
    root.insert(QStringLiteral("depth_filter_mode"), static_cast<int>(config.depthFilterMode));
    root.insert(QStringLiteral("adaptive_depth_filter_mode"), config.adaptiveDepthFilterMode);
    root.insert(QStringLiteral("save_intermediate_pyramid_levels"),
                config.saveIntermediatePyramidLevels);
    root.insert(QStringLiteral("patch_match"), patch);
    root.insert(QStringLiteral("fusion"), fusion);

    const QByteArray canonical = QJsonDocument(root).toJson(QJsonDocument::Compact);
    return QString::fromLatin1(QCryptographicHash::hash(canonical, QCryptographicHash::Sha256).toHex());
}

} // namespace xjw::mvs
