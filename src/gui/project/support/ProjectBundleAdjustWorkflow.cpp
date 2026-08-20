#include "ProjectBundleAdjustWorkflow.h"

#include "project/ProjectSessionModel.h"

#include <QDir>
#include <QStringList>

namespace xjw::gui::project {

namespace {

QString formattedMetric(const QJsonObject &object,
                        const QString &key,
                        int precision)
{
    if (!object.contains(key))
    {
        return QStringLiteral("—");
    }
    return QString::number(object.value(key).toDouble(), 'f', precision);
}

void appendOutputPath(QStringList *lines,
                      const QString &label,
                      const QString &path)
{
    if (lines && !path.trimmed().isEmpty())
    {
        lines->append(QStringLiteral("%1: %2").arg(label, path));
    }
}

} // namespace

BundleAdjustPreviewPresentation buildBundleAdjustPreviewPresentation(
    const QJsonObject &ba_result,
    int pending_camera_count)
{
    BundleAdjustPreviewPresentation presentation;

    const int camera_count = pending_camera_count > 0
        ? pending_camera_count
        : ba_result.value(QStringLiteral("refined_camera_count")).toInt();
    const int track_count = ba_result.value(QStringLiteral("track_count")).toInt();
    const int optimized_count = ba_result.value(QStringLiteral("optimized_count")).toInt();
    const bool has_rms = ba_result.contains(QStringLiteral("mean_rms_before"))
        && ba_result.contains(QStringLiteral("mean_rms_after"));
    const double rms_before = ba_result.value(QStringLiteral("mean_rms_before")).toDouble();
    const double rms_after = ba_result.value(QStringLiteral("mean_rms_after")).toDouble();

    presentation.summaryText = QStringLiteral(
        "平差计算已完成，%1 台相机的调整参数尚未写回项目。\n"
        "参与轨迹: %2，成功优化: %3\n"
        "平均重投影 RMS: %4 px → %5 px")
        .arg(camera_count)
        .arg(track_count)
        .arg(optimized_count)
        .arg(has_rms ? QString::number(rms_before, 'f', 6) : QStringLiteral("—"))
        .arg(has_rms ? QString::number(rms_after, 'f', 6) : QStringLiteral("—"));

    presentation.qualityWarning =
        ba_result.value(QStringLiteral("ba_quality_gate_rejected")).toBool(false)
        || (has_rms && rms_before > 0.0 && rms_after > rms_before);

    QStringList details;
    const QString requested_backend =
        ba_result.value(QStringLiteral("ba_requested_backend")).toString();
    const QString used_backend = ba_result.value(QStringLiteral("ba_used_backend")).toString();
    if (!requested_backend.isEmpty() || !used_backend.isEmpty())
    {
        details.append(QStringLiteral("计算后端: %1 → %2")
                           .arg(requested_backend.isEmpty() ? QStringLiteral("—") : requested_backend,
                                used_backend.isEmpty() ? QStringLiteral("—") : used_backend));
    }
    if (ba_result.contains(QStringLiteral("ba_valid_track_ratio")))
    {
        details.append(QStringLiteral("有效轨迹比例: %1%")
                           .arg(ba_result.value(QStringLiteral("ba_valid_track_ratio")).toDouble()
                                    * 100.0,
                                0,
                                'f',
                                2));
    }
    if (ba_result.contains(QStringLiteral("ba_total_seconds")))
    {
        details.append(QStringLiteral("计算耗时: %1 s")
                           .arg(formattedMetric(ba_result,
                                                QStringLiteral("ba_total_seconds"),
                                                3)));
    }

    const QJsonObject terrain_summary =
        ba_result.value(QStringLiteral("reference_terrain_prior_summary")).toObject();
    if (terrain_summary.value(QStringLiteral("enabled")).toBool(false))
    {
        details.append(QStringLiteral("参考 DEM 关联: %1 / %2 条轨迹")
                           .arg(terrain_summary.value(QStringLiteral("associated_tracks")).toInt())
                           .arg(terrain_summary.value(QStringLiteral("input_tracks")).toInt()));
        if (terrain_summary.contains(QStringLiteral("rms_before_m")))
        {
            details.append(QStringLiteral("参考 DEM 初始高程 RMS: %1 m")
                               .arg(formattedMetric(terrain_summary,
                                                    QStringLiteral("rms_before_m"),
                                                    4)));
        }
        appendOutputPath(&details,
                         QStringLiteral("参考 DEM"),
                         terrain_summary.value(QStringLiteral("path")).toString());
    }

    const QJsonObject laser_summary =
        ba_result.value(QStringLiteral("laser_constraints_summary")).toObject();
    if (laser_summary.value(QStringLiteral("enabled")).toBool(false))
    {
        details.append(QStringLiteral("LiDAR 关联: %1 / %2 条轨迹")
                           .arg(laser_summary.value(QStringLiteral("associated_tracks")).toInt())
                           .arg(laser_summary.value(QStringLiteral("total_tracks")).toInt()));
        if (laser_summary.contains(QStringLiteral("laser_rms_before_m"))
            && laser_summary.contains(QStringLiteral("laser_rms_after_m")))
        {
            details.append(QStringLiteral("LiDAR 约束 RMS: %1 m → %2 m")
                               .arg(formattedMetric(laser_summary,
                                                    QStringLiteral("laser_rms_before_m"),
                                                    4),
                                    formattedMetric(laser_summary,
                                                    QStringLiteral("laser_rms_after_m"),
                                                    4)));
        }
        appendOutputPath(&details,
                         QStringLiteral("参考点云"),
                         laser_summary.value(QStringLiteral("cloud_path")).toString());
    }

    const QJsonObject planetary_laser_summary =
        ba_result.value(QStringLiteral("planetary_laser_range_summary")).toObject();
    if (planetary_laser_summary.value(QStringLiteral("enabled")).toBool(false))
    {
        details.append(QStringLiteral("行星激光测距 shot: %1 / %2")
                           .arg(planetary_laser_summary
                                    .value(QStringLiteral("range_constraint_count"))
                                    .toInt())
                           .arg(planetary_laser_summary
                                    .value(QStringLiteral("total_shots"))
                                    .toInt()));
        if (planetary_laser_summary.contains(QStringLiteral("range_rms_before_m")) &&
            planetary_laser_summary.contains(QStringLiteral("range_rms_after_m")))
        {
            const double rangeRmsBefore = planetary_laser_summary
                                              .value(QStringLiteral("range_rms_before_m"))
                                              .toDouble();
            const double rangeRmsAfter = planetary_laser_summary
                                             .value(QStringLiteral("range_rms_after_m"))
                                             .toDouble();
            details.append(QStringLiteral("行星激光 range RMS: %1 m → %2 m")
                               .arg(formattedMetric(
                                        planetary_laser_summary,
                                        QStringLiteral("range_rms_before_m"),
                                        4),
                                    formattedMetric(
                                        planetary_laser_summary,
                                        QStringLiteral("range_rms_after_m"),
                                        4)));
            presentation.qualityWarning = presentation.qualityWarning ||
                (rangeRmsAfter > rangeRmsBefore + 1.0e-12);
        }
        details.append(QStringLiteral("行星激光目标/坐标系: %1 / %2")
                           .arg(planetary_laser_summary
                                    .value(QStringLiteral("target"))
                                    .toString(),
                                planetary_laser_summary
                                    .value(QStringLiteral("body_fixed_frame"))
                                    .toString()));
        appendOutputPath(
            &details,
            QStringLiteral("行星激光数据"),
            planetary_laser_summary.value(QStringLiteral("data_path")).toString());
    }

    const QString quality_message =
        ba_result.value(QStringLiteral("ba_quality_gate_message")).toString().trimmed();
    if (!quality_message.isEmpty())
    {
        details.append(QStringLiteral("质量门控: %1").arg(quality_message));
    }
    const QString backend_reason =
        ba_result.value(QStringLiteral("ba_backend_selection_reason")).toString().trimmed();
    if (!backend_reason.isEmpty())
    {
        details.append(QStringLiteral("后端说明: %1").arg(backend_reason));
    }

    appendOutputPath(&details,
                     QStringLiteral("输出目录"),
                     ba_result.value(QStringLiteral("output_dir")).toString());
    const QJsonObject files = ba_result.value(QStringLiteral("files")).toObject();
    appendOutputPath(&details,
                     QStringLiteral("运行摘要"),
                     files.value(QStringLiteral("run_json")).toString());
    appendOutputPath(&details,
                     QStringLiteral("相机指标"),
                     files.value(QStringLiteral("camera_csv")).toString());
    appendOutputPath(&details,
                     QStringLiteral("连接点指标"),
                     files.value(QStringLiteral("points_csv")).toString());
    presentation.detailedText = details.join(QLatin1Char('\n'));
    return presentation;
}

BundleAdjustCommitResult commitBundleAdjustPreview(ProjectData *projectData,
                                                   const QMap<QString, QJsonObject> &cameraMetaByImage,
                                                   const QJsonObject &baResult)
{
    BundleAdjustCommitResult result;
    if (!projectData)
    {
        result.errorMessage = QStringLiteral("项目未就绪");
        return result;
    }

    if (cameraMetaByImage.isEmpty())
    {
        result.errorMessage = QStringLiteral("没有可应用的平差相机结果");
        return result;
    }

    QString errorMessage;
    if (!projectData->setImageCameras(cameraMetaByImage, &result.updatedCameraCount, &errorMessage))
    {
        result.errorMessage = QStringLiteral("写回相机参数失败: %1").arg(errorMessage);
        return result;
    }

    QJsonObject compactBaResult = baResult;
    compactBaResult.remove(QStringLiteral("point_preview"));
    QString saveWarning;
    if (!projectData->appendBundleAdjustResult(compactBaResult, &saveWarning))
    {
        result.warningMessage = QStringLiteral("保存平差结果失败: %1").arg(saveWarning);
    }

    result.success = true;
    return result;
}

BundleAdjustArtifactsResult finalizeBundleAdjustArtifacts(const QString &assetsDir,
                                                          const QJsonObject &baResult,
                                                          const QStringList &images,
                                                          const QString &reportOutputDir,
                                                          const QString &reportSource,
                                                          const QMap<QString, QJsonObject> &beforeCameras,
                                                          const QMap<QString, QJsonObject> &afterCameras,
                                                          const QString &sparseCloudOutputDir,
                                                          bool useDedicatedFileName)
{
    BundleAdjustArtifactsResult result;

    if (!assetsDir.isEmpty())
    {
        const QJsonObject report = buildBundleAdjustWorkflowReport(baResult,
                                                                   images,
                                                                   reportOutputDir,
                                                                   reportSource,
                                                                   beforeCameras,
                                                                   afterCameras);
        result.reportSaved = writeLatestAndAppendHistoryReport(QDir(assetsDir).filePath(QStringLiteral("reports")),
                                                               QStringLiteral("at_report.json"),
                                                               QStringLiteral("at_report_history.json"),
                                                               report);
        if (!result.reportSaved)
        {
            result.reportWarning = QStringLiteral("保存项目报告失败");
        }
    }

    result.sparseCloudExport = exportBundleAdjustSparseCloud(baResult,
                                                             images,
                                                             sparseCloudOutputDir,
                                                             useDedicatedFileName);
    return result;
}

} // namespace xjw::gui::project
