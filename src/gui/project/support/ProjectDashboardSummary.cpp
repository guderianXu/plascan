#include "ProjectDashboardSummary.h"

#include <QFileInfo>
#include <QStringList>

namespace xjw::gui::project {
namespace {

bool isDashboardResultKey(const QString &key)
{
    return key == QStringLiteral("ipfind_results")
        || key == QStringLiteral("feature_results")
        || key == QStringLiteral("ipmatch_results")
        || key == QStringLiteral("match_results")
        || key == QStringLiteral("aerial_triangulation_results")
        || key == QStringLiteral("sparse_results")
        || key == QStringLiteral("bundle_adjust_results")
        || key == QStringLiteral("depth_map_results")
        || key == QStringLiteral("depth_results")
        || key == QStringLiteral("dense_cloud_results")
        || key == QStringLiteral("dense_results")
        || key == QStringLiteral("model_results")
        || key == QStringLiteral("dem_results")
        || key == QStringLiteral("ortho_results")
        || key == QStringLiteral("report_results")
        || key == QStringLiteral("reference_datasets");
}

QJsonObject normalizeMetadata(const QJsonObject &metadata)
{
    QJsonObject normalized = metadata.value(QStringLiteral("project_files")).toObject();
    if (normalized.isEmpty())
    {
        normalized = metadata;
    }

    for (auto it = metadata.constBegin(); it != metadata.constEnd(); ++it)
    {
        if (isDashboardResultKey(it.key()))
        {
            normalized.insert(it.key(), it.value());
        }
    }

    return normalized;
}

QJsonArray mergedArray(const QJsonObject &metadata, const QStringList &keys)
{
    QJsonArray merged;
    for (const QString &key : keys)
    {
        const QJsonArray values = metadata.value(key).toArray();
        for (const QJsonValue &value : values)
        {
            merged.append(value);
        }
    }
    return merged;
}

bool imageHasCamera(const QJsonObject &image)
{
    if (!image.value(QStringLiteral("camera")).toObject().isEmpty())
    {
        return true;
    }
    return !image.value(QStringLiteral("camera_file")).toString().trimmed().isEmpty();
}

bool isMvsDepthResult(const QJsonObject &record)
{
    const QString kind = record.value(QStringLiteral("result_type")).toString();
    if (kind == QStringLiteral("mvs_depth"))
    {
        return true;
    }
    if (kind == QStringLiteral("legacy_preview"))
    {
        return false;
    }
    return !record.value(QStringLiteral("raw_depth_path")).toString().isEmpty()
        || !record.value(QStringLiteral("ref_image")).toString().isEmpty()
        || !record.value(QStringLiteral("depth_png")).toString().isEmpty();
}

int mvsDepthResultCount(const QJsonArray &records)
{
    int count = 0;
    for (const QJsonValue &value : records)
    {
        if (value.isObject() && isMvsDepthResult(value.toObject()))
        {
            ++count;
        }
    }
    return count;
}

bool isDisplayableModelResult(const QJsonObject &record)
{
    if (record.contains(QStringLiteral("face_count")) &&
        record.value(QStringLiteral("face_count")).toInt(0) <= 0)
    {
        return false;
    }
    return !record.value(QStringLiteral("final_model_path")).toString().isEmpty()
        || !record.value(QStringLiteral("model_obj")).toString().isEmpty()
        || !record.value(QStringLiteral("model_ply")).toString().isEmpty()
        || !record.value(QStringLiteral("mesh_ply")).toString().isEmpty();
}

int displayableModelResultCount(const QJsonArray &records)
{
    int count = 0;
    for (const QJsonValue &value : records)
    {
        if (value.isObject() && isDisplayableModelResult(value.toObject()))
        {
            ++count;
        }
    }
    return count;
}

QString normalizedReferenceType(const QJsonObject &record)
{
    QString type = record.value(QStringLiteral("type")).toString().trimmed().toLower();
    if (!type.isEmpty())
    {
        if (type == QStringLiteral("las") || type == QStringLiteral("laz") || type == QStringLiteral("copc"))
        {
            return QStringLiteral("lidar");
        }
        if (type == QStringLiteral("cloud"))
        {
            return QStringLiteral("point_cloud");
        }
        if (type == QStringLiteral("tiff") || type == QStringLiteral("geotiff") || type == QStringLiteral("reference_dem"))
        {
            return QStringLiteral("dem");
        }
        return type;
    }

    const QFileInfo fileInfo(record.value(QStringLiteral("path")).toString());
    const QString suffix = fileInfo.suffix().toLower();
    if (suffix == QStringLiteral("las") || suffix == QStringLiteral("laz") || suffix == QStringLiteral("copc"))
    {
        return QStringLiteral("lidar");
    }
    if (suffix == QStringLiteral("ply") || suffix == QStringLiteral("xyz") || suffix == QStringLiteral("csv"))
    {
        return QStringLiteral("point_cloud");
    }
    if (suffix == QStringLiteral("tif") || suffix == QStringLiteral("tiff") || suffix == QStringLiteral("vrt"))
    {
        return QStringLiteral("dem");
    }
    return QString();
}

bool isBaPriorRole(QString role)
{
    role = role.trimmed().toLower();
    return role == QStringLiteral("ba_prior")
        || role == QStringLiteral("bundle_adjustment")
        || role == QStringLiteral("reference_prior");
}

bool isValidationRole(QString role)
{
    role = role.trimmed().toLower();
    return role.isEmpty()
        || role == QStringLiteral("validation")
        || role == QStringLiteral("quality_check");
}

bool isQualityReportType(QString type)
{
    type = type.trimmed().toLower();
    return type == QStringLiteral("reconstruction_quality")
        || type == QStringLiteral("reference_quality")
        || type == QStringLiteral("reference_terrain_prior_preflight");
}

void appendStep(ProjectDashboardSummary *summary,
                const QString &id,
                const QString &title,
                ProjectDashboardStepState state,
                const QString &detail)
{
    if (!summary)
    {
        return;
    }

    ProjectDashboardStep step;
    step.id = id;
    step.title = title;
    step.state = state;
    step.detail = detail;
    summary->workflowSteps.append(step);
}

ProjectDashboardStepState readyAfter(bool prerequisiteComplete)
{
    return prerequisiteComplete ? ProjectDashboardStepState::Ready : ProjectDashboardStepState::Missing;
}

} // namespace

QString projectDashboardStepStateName(ProjectDashboardStepState state)
{
    switch (state)
    {
    case ProjectDashboardStepState::Missing:
        return QStringLiteral("missing");
    case ProjectDashboardStepState::Ready:
        return QStringLiteral("ready");
    case ProjectDashboardStepState::Complete:
        return QStringLiteral("complete");
    case ProjectDashboardStepState::Warning:
        return QStringLiteral("warning");
    }
    return QStringLiteral("missing");
}

bool projectDashboardStepById(const ProjectDashboardSummary &summary,
                              const QString &id,
                              ProjectDashboardStep *step)
{
    for (const ProjectDashboardStep &candidate : summary.workflowSteps)
    {
        if (candidate.id == id)
        {
            if (step)
            {
                *step = candidate;
            }
            return true;
        }
    }
    return false;
}

ProjectDashboardSummary buildProjectDashboardSummary(const QJsonObject &metadata)
{
    const QJsonObject normalized = normalizeMetadata(metadata);

    const QJsonArray images = normalized.value(QStringLiteral("images")).toArray();
    const QJsonArray featureResults = mergedArray(normalized, {QStringLiteral("ipfind_results"),
                                                               QStringLiteral("feature_results")});
    const QJsonArray matchResults = mergedArray(normalized, {QStringLiteral("ipmatch_results"),
                                                             QStringLiteral("match_results")});
    const QJsonArray sparseResults = mergedArray(normalized, {QStringLiteral("aerial_triangulation_results"),
                                                              QStringLiteral("sparse_results")});
    const QJsonArray bundleAdjustResults = normalized.value(QStringLiteral("bundle_adjust_results")).toArray();
    const QJsonArray depthResults = mergedArray(normalized, {QStringLiteral("depth_map_results"),
                                                             QStringLiteral("depth_results")});
    const QJsonArray denseResults = mergedArray(normalized, {QStringLiteral("dense_cloud_results"),
                                                             QStringLiteral("dense_results")});
    const QJsonArray modelResults = normalized.value(QStringLiteral("model_results")).toArray();
    const QJsonArray demResults = normalized.value(QStringLiteral("dem_results")).toArray();
    const QJsonArray orthoResults = normalized.value(QStringLiteral("ortho_results")).toArray();
    const QJsonArray reportResults = normalized.value(QStringLiteral("report_results")).toArray();
    const QJsonArray referenceDatasets = normalized.value(QStringLiteral("reference_datasets")).toArray();

    ProjectDashboardSummary summary;
    summary.imageCount = images.size();
    for (const QJsonValue &value : images)
    {
        if (value.isObject() && imageHasCamera(value.toObject()))
        {
            ++summary.cameraCount;
        }
    }

    summary.featureResultCount = featureResults.size();
    summary.matchResultCount = matchResults.size();
    summary.sparseResultCount = sparseResults.size();
    summary.bundleAdjustResultCount = bundleAdjustResults.size();
    summary.depthMapResultCount = mvsDepthResultCount(depthResults);
    summary.denseCloudResultCount = denseResults.size();
    summary.modelResultCount = displayableModelResultCount(modelResults);
    summary.demResultCount = demResults.size();
    summary.orthoResultCount = orthoResults.size();
    summary.reportResultCount = reportResults.size();
    summary.referenceDatasetCount = referenceDatasets.size();

    for (const QJsonValue &value : referenceDatasets)
    {
        if (!value.isObject())
        {
            continue;
        }
        const QJsonObject record = value.toObject();
        const QString type = normalizedReferenceType(record);
        const QString role = record.value(QStringLiteral("role")).toString();
        QJsonObject normalizedRecord = record;
        if (!type.isEmpty())
        {
            normalizedRecord[QStringLiteral("type")] = type;
        }
        summary.referenceDatasets.append(normalizedRecord);
        if (type == QStringLiteral("lidar"))
        {
            ++summary.lidarReferenceCount;
        }
        else if (type == QStringLiteral("point_cloud"))
        {
            ++summary.pointCloudReferenceCount;
        }
        else if (type == QStringLiteral("dem"))
        {
            ++summary.demReferenceCount;
        }
        if (isBaPriorRole(role))
        {
            ++summary.baPriorReferenceCount;
        }
        if (isValidationRole(role))
        {
            ++summary.validationReferenceCount;
        }
    }

    for (const QJsonValue &value : reportResults)
    {
        if (!value.isObject())
        {
            continue;
        }
        const QJsonObject record = value.toObject();
        if (isQualityReportType(record.value(QStringLiteral("type")).toString()))
        {
            summary.qualityReports.append(record);
            ++summary.qualityReportCount;
        }
    }

    const bool hasImages = summary.imageCount > 0;
    const bool allImagesHaveCameras = hasImages && summary.cameraCount == summary.imageCount;
    const bool hasFeatures = summary.featureResultCount > 0;
    const bool hasMatches = summary.matchResultCount > 0;
    const bool hasSparseOrBa = summary.sparseResultCount > 0 || summary.bundleAdjustResultCount > 0;
    const bool hasDenseProducts = summary.depthMapResultCount > 0 || summary.denseCloudResultCount > 0;
    const bool hasTerrainProducts = summary.demResultCount > 0 || summary.orthoResultCount > 0;
    const bool hasReferenceCloud = summary.lidarReferenceCount > 0 || summary.pointCloudReferenceCount > 0;
    const bool hasQualityReports = summary.qualityReportCount > 0;

    appendStep(&summary,
               QStringLiteral("images"),
               QStringLiteral("影像导入"),
               hasImages ? ProjectDashboardStepState::Complete : ProjectDashboardStepState::Missing,
               hasImages ? QStringLiteral("已登记 %1 张影像").arg(summary.imageCount)
                         : QStringLiteral("请先导入影像或打开已有项目"));
    appendStep(&summary,
               QStringLiteral("cameras"),
               QStringLiteral("相机参数"),
               allImagesHaveCameras ? ProjectDashboardStepState::Complete
                                    : (summary.cameraCount > 0 ? ProjectDashboardStepState::Warning
                                                               : readyAfter(hasImages)),
               hasImages ? QStringLiteral("已有相机 %1/%2").arg(summary.cameraCount).arg(summary.imageCount)
                         : QStringLiteral("等待影像导入后检查相机参数"));
    appendStep(&summary,
               QStringLiteral("features"),
               QStringLiteral("特征提取"),
               hasFeatures ? ProjectDashboardStepState::Complete : readyAfter(hasImages),
               hasFeatures ? QStringLiteral("已有 %1 条特征结果").arg(summary.featureResultCount)
                           : QStringLiteral("可在影像导入后运行特征提取"));
    appendStep(&summary,
               QStringLiteral("matches"),
               QStringLiteral("特征匹配"),
               hasMatches ? ProjectDashboardStepState::Complete : readyAfter(hasFeatures),
               hasMatches ? QStringLiteral("已有 %1 条匹配结果").arg(summary.matchResultCount)
                          : QStringLiteral("等待特征结果后进行匹配"));
    appendStep(&summary,
               QStringLiteral("sparse_ba"),
               QStringLiteral("稀疏重建/BA"),
               hasSparseOrBa ? ProjectDashboardStepState::Complete : readyAfter(hasMatches),
               QStringLiteral("稀疏结果 %1，BA 结果 %2")
                   .arg(summary.sparseResultCount)
                   .arg(summary.bundleAdjustResultCount));
    appendStep(&summary,
               QStringLiteral("dense_mvs"),
               QStringLiteral("MVS/稠密点云"),
               hasDenseProducts ? ProjectDashboardStepState::Complete : readyAfter(hasSparseOrBa),
               QStringLiteral("深度图 %1，稠密点云 %2")
                   .arg(summary.depthMapResultCount)
                   .arg(summary.denseCloudResultCount));
    appendStep(&summary,
               QStringLiteral("terrain"),
               QStringLiteral("DEM/DOM"),
               hasTerrainProducts ? ProjectDashboardStepState::Complete : readyAfter(hasDenseProducts),
               QStringLiteral("DEM %1，正射影像 %2")
                   .arg(summary.demResultCount)
                   .arg(summary.orthoResultCount));
    appendStep(&summary,
               QStringLiteral("reference_lidar"),
               QStringLiteral("LiDAR/参考数据"),
               hasReferenceCloud ? ProjectDashboardStepState::Complete : readyAfter(hasImages),
               QStringLiteral("LiDAR %1，点云 %2，DEM %3，BA约束 %4")
                   .arg(summary.lidarReferenceCount)
                   .arg(summary.pointCloudReferenceCount)
                   .arg(summary.demReferenceCount)
                   .arg(summary.baPriorReferenceCount));
    appendStep(&summary,
               QStringLiteral("quality"),
               QStringLiteral("质量检查/报告"),
               hasQualityReports ? ProjectDashboardStepState::Complete
                                 : (hasTerrainProducts || hasReferenceCloud ? ProjectDashboardStepState::Ready
                                                                            : ProjectDashboardStepState::Missing),
               QStringLiteral("质量报告 %1，全部报告 %2")
                   .arg(summary.qualityReportCount)
                   .arg(summary.reportResultCount));

    return summary;
}

} // namespace xjw::gui::project
