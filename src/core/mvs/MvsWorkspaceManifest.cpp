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
    m_configHash = root.value(QStringLiteral("config_hash")).toString();
    const QJsonArray frames = root.value(QStringLiteral("frames")).toArray();
    m_frames.reserve(frames.size());
    for (const QJsonValue &value : frames)
    {
        if (value.isObject())
        {
            m_frames.push_back(MvsDepthFrameRecord::fromJson(value.toObject()));
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
    m_configHash.clear();
    m_frames.clear();
}

QString MvsWorkspaceManifest::configHash() const
{
    return m_configHash;
}

void MvsWorkspaceManifest::setConfigHash(const QString &hash)
{
    m_configHash = hash;
}

const QVector<MvsDepthFrameRecord> &MvsWorkspaceManifest::frames() const
{
    return m_frames;
}

QVector<MvsDepthFrameRecord> MvsWorkspaceManifest::completedFramesSortedByName() const
{
    QVector<MvsDepthFrameRecord> result;
    for (const MvsDepthFrameRecord &record : m_frames)
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
        m_frames[index] = record;
    }
    else
    {
        m_frames.push_back(record);
    }
}

void MvsWorkspaceManifest::markRunning(int refIndex, const QString &refImage, const QString &configHash)
{
    MvsDepthFrameRecord record;
    const int index = findFrameIndex(refIndex);
    if (index >= 0)
    {
        record = m_frames[index];
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
        completed.sourcePlan = m_frames[index].sourcePlan;
    }
    if (completed.sourceViewCount <= 0 && index >= 0)
    {
        completed.sourceViewCount = m_frames[index].sourceViewCount;
        completed.meanSourceQualityScore = m_frames[index].meanSourceQualityScore;
        completed.minSourceQualityScore = m_frames[index].minSourceQualityScore;
    }
    if (completed.validPixelCount <= 0 && index >= 0)
    {
        completed.validPixelCount = m_frames[index].validPixelCount;
        completed.meanDepthConfidence = m_frames[index].meanDepthConfidence;
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
        record = m_frames[index];
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
    const MvsDepthFrameRecord &record = m_frames[index];
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
    root.insert(QStringLiteral("schema"), QStringLiteral("plascan.mvs.workspace.v1"));
    root.insert(QStringLiteral("config_hash"), m_configHash);

    QJsonArray frames;
    for (const MvsDepthFrameRecord &record : m_frames)
    {
        frames.push_back(record.toJson());
    }
    root.insert(QStringLiteral("frames"), frames);
    return root;
}

int MvsWorkspaceManifest::findFrameIndex(int refIndex) const
{
    for (int i = 0; i < m_frames.size(); ++i)
    {
        if (m_frames[i].refIndex == refIndex)
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
    patch.insert(QStringLiteral("geom_consistency"), config.patchMatch.geomConsistency);
    patch.insert(QStringLiteral("geom_consistency_max_err"), config.patchMatch.geomConsistencyMaxErr);
    patch.insert(QStringLiteral("epipolar_rectified"), config.patchMatch.epipolarRectified);

    QJsonObject fusion;
    fusion.insert(QStringLiteral("min_consistent_views"), config.fusion.minConsistentViews);
    fusion.insert(QStringLiteral("rel_depth_thresh"), config.fusion.relDepthThresh);
    fusion.insert(QStringLiteral("pixel_thresh"), config.fusion.pixelThresh);
    fusion.insert(QStringLiteral("confidence_thresh"), config.fusion.confidenceThresh);
    fusion.insert(QStringLiteral("local_outlier"), config.fusion.enableLocalDepthOutlierFilter);
    fusion.insert(QStringLiteral("local_outlier_kernel"), config.fusion.localDepthOutlierKernelSize);
    fusion.insert(QStringLiteral("local_outlier_rel_thresh"), config.fusion.localDepthOutlierRelThresh);
    fusion.insert(QStringLiteral("speckle_filter"), config.fusion.enableSpeckleFilter);
    fusion.insert(QStringLiteral("speckle_min_area"), config.fusion.minSpeckleComponentArea);
    fusion.insert(QStringLiteral("speckle_max_removal_ratio"), config.fusion.maxSpeckleRemovalRatio);

    QJsonObject root;
    root.insert(QStringLiteral("schema"), QStringLiteral("plascan.mvs.depth.config.v1"));
    root.insert(QStringLiteral("view_count"), viewCount);
    root.insert(QStringLiteral("num_source_views"), config.numSourceViews);
    root.insert(QStringLiteral("z_near_scale"), config.zNearScale);
    root.insert(QStringLiteral("z_far_scale"), config.zFarScale);
    root.insert(QStringLiteral("patch_match"), patch);
    root.insert(QStringLiteral("fusion"), fusion);

    const QByteArray canonical = QJsonDocument(root).toJson(QJsonDocument::Compact);
    return QString::fromLatin1(QCryptographicHash::hash(canonical, QCryptographicHash::Sha256).toHex());
}

} // namespace xjw::mvs
