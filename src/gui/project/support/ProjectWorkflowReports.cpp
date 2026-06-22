#include "ProjectWorkflowReports.h"

#include "ProjectData.h"
#include "ProjectIO.h"
#include "project/SparseResultQuality.h"
#include "ReconstructionQualityReport.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTextStream>

#include <cmath>

namespace xjw::gui::project {

QJsonObject buildBundleAdjustReport(const QJsonObject &baResult,
                                    const QMap<QString, QJsonObject> &beforeCameras,
                                    const QMap<QString, QJsonObject> &afterCameras)
{
    if (baResult.isEmpty())
    {
        return QJsonObject();
    }

    const int baCameraCnt = baResult.value(QStringLiteral("camera_count")).toInt();
    const int baTrackTotal = baResult.value(QStringLiteral("track_count")).toInt();
    const int baOptimized = baResult.value(QStringLiteral("optimized_count")).toInt();
    const double baRmsBefore = baResult.value(QStringLiteral("mean_rms_before")).toDouble();
    const double baRmsAfter = baResult.value(QStringLiteral("mean_rms_after")).toDouble();
    const QJsonArray selectedImages = baResult.value(QStringLiteral("selected_images")).toArray();
    const QJsonArray camPreview = baResult.value(QStringLiteral("camera_preview")).toArray();

    QJsonArray perCamera;
    for (const QJsonValue &cpv : camPreview)
    {
        const QJsonObject cp = cpv.toObject();
        QJsonObject pc;
        pc[QStringLiteral("path")] = cp.value(QStringLiteral("image_path"));
        pc[QStringLiteral("registered")] = true;
        pc[QStringLiteral("residual_px")] = cp.value(QStringLiteral("mean_rms_after"));
        perCamera.append(pc);
    }

    QJsonArray camComp;
    for (auto it = afterCameras.constBegin(); it != afterCameras.constEnd(); ++it)
    {
        const QString &normPath = it.key();
        const QJsonObject &after = it.value();
        const QJsonObject &before = beforeCameras.value(normPath);

        const QJsonArray cA = after.value(QStringLiteral("C")).toArray();
        const QJsonArray cB = before.value(QStringLiteral("C")).toArray();
        double posDelta = -1.0;
        if (cA.size() == 3 && cB.size() == 3)
        {
            const double dx = cA[0].toDouble() - cB[0].toDouble();
            const double dy = cA[1].toDouble() - cB[1].toDouble();
            const double dz = cA[2].toDouble() - cB[2].toDouble();
            posDelta = std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        QJsonObject entry;
        entry[QStringLiteral("name")] = QFileInfo(normPath).fileName();
        entry[QStringLiteral("path")] = normPath;
        entry[QStringLiteral("had_before")] = !before.isEmpty();
        entry[QStringLiteral("fu_before")] = before.value(QStringLiteral("fu")).toDouble();
        entry[QStringLiteral("fu_after")] = after.value(QStringLiteral("fu")).toDouble();
        entry[QStringLiteral("fv_before")] = before.value(QStringLiteral("fv")).toDouble();
        entry[QStringLiteral("fv_after")] = after.value(QStringLiteral("fv")).toDouble();
        entry[QStringLiteral("cu_before")] = before.value(QStringLiteral("cu")).toDouble();
        entry[QStringLiteral("cu_after")] = after.value(QStringLiteral("cu")).toDouble();
        entry[QStringLiteral("cv_before")] = before.value(QStringLiteral("cv")).toDouble();
        entry[QStringLiteral("cv_after")] = after.value(QStringLiteral("cv")).toDouble();
        entry[QStringLiteral("C_before")] = cB;
        entry[QStringLiteral("C_after")] = cA;
        entry[QStringLiteral("yaw_before")] = before.value(QStringLiteral("yaw_deg")).toDouble();
        entry[QStringLiteral("yaw_after")] = after.value(QStringLiteral("yaw_deg")).toDouble();
        entry[QStringLiteral("pitch_before")] = before.value(QStringLiteral("pitch_deg")).toDouble();
        entry[QStringLiteral("pitch_after")] = after.value(QStringLiteral("pitch_deg")).toDouble();
        entry[QStringLiteral("roll_before")] = before.value(QStringLiteral("roll_deg")).toDouble();
        entry[QStringLiteral("roll_after")] = after.value(QStringLiteral("roll_deg")).toDouble();
        entry[QStringLiteral("pos_delta")] = posDelta;
        camComp.append(entry);
    }

    QJsonObject rep;
    rep[QStringLiteral("type")] = QStringLiteral("aerial_triangulation");
    rep[QStringLiteral("mode")] = QStringLiteral("bundle_adjust");
    rep[QStringLiteral("timestamp")] = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"));
    rep[QStringLiteral("num_images")] = selectedImages.size();
    rep[QStringLiteral("num_registered")] = baCameraCnt;
    rep[QStringLiteral("num_points_3d")] = baTrackTotal;
    rep[QStringLiteral("mean_reproj_error_px")] = baRmsAfter;
    rep[QStringLiteral("ba_rms_before")] = baRmsBefore;
    rep[QStringLiteral("ba_rms_after")] = baRmsAfter;
    rep[QStringLiteral("ba_tracks_total")] = baTrackTotal;
    rep[QStringLiteral("ba_tracks_optimized")] = baOptimized;
    rep[QStringLiteral("ba_tracks_filtered")] = qMax(0, baTrackTotal - baOptimized);
    rep[QStringLiteral("duration_s")] = baResult.value(QStringLiteral("duration_s")).toDouble(-1.0);
    rep[QStringLiteral("output_dir")] = baResult.value(QStringLiteral("output_dir")).toString();
    rep[QStringLiteral("per_camera")] = perCamera;
    rep[QStringLiteral("camera_preview")] = camPreview;
    rep[QStringLiteral("point_residuals")] = baResult.value(QStringLiteral("points"));
    rep[QStringLiteral("camera_comparison")] = camComp;

    return rep;
}

QJsonObject buildBundleAdjustWorkflowReport(const QJsonObject &baResult,
                                            const QStringList &images,
                                            const QString &outputDir,
                                            const QString &source,
                                            const QMap<QString, QJsonObject> &beforeCameras,
                                            const QMap<QString, QJsonObject> &afterCameras)
{
    QJsonObject rep = buildBundleAdjustReport(baResult, beforeCameras, afterCameras);
    if (rep.isEmpty())
    {
        return rep;
    }

    rep[QStringLiteral("source")] = source;
    rep[QStringLiteral("num_images")] = images.size();
    rep[QStringLiteral("output_dir")] = outputDir;
    return rep;
}

BundleAdjustSparseCloudExport exportBundleAdjustSparseCloud(const QJsonObject &baResult,
                                                            const QStringList &selectedImages,
                                                            const QString &outputDir,
                                                            bool useDedicatedFileName)
{
    BundleAdjustSparseCloudExport exportResult;
    exportResult.outputDir = outputDir;

    const QJsonArray ptsArr = baResult.value(QStringLiteral("points")).toArray();
    if (ptsArr.isEmpty())
    {
        return exportResult;
    }

    struct Pt3 { double x; double y; double z; };
    QVector<Pt3> validPts;
    QJsonArray pointsForSidecar;
    validPts.reserve(ptsArr.size());

    constexpr double kMaxRmsForExport = 2.5;
    constexpr int kMinTrackLen = 2;
    for (const QJsonValue &pv : ptsArr)
    {
        const QJsonObject point = pv.toObject();
        if (!point.value(QStringLiteral("valid")).toBool(true))
        {
            continue;
        }
        if (!point.value(QStringLiteral("converged")).toBool(true))
        {
            continue;
        }
        const double rmsAfter = point.value(QStringLiteral("rms_after")).toDouble(0.0);
        const int trackLen = point.value(QStringLiteral("track_len")).toInt(2);
        if (rmsAfter > kMaxRmsForExport && rmsAfter > 0.0)
        {
            continue;
        }
        if (trackLen < kMinTrackLen)
        {
            continue;
        }

        const QJsonArray xyz = point.value(QStringLiteral("point_xyz")).toArray();
        if (xyz.size() < 3)
        {
            continue;
        }

        const double x = xyz.at(0).toDouble();
        const double y = xyz.at(1).toDouble();
        const double z = xyz.at(2).toDouble();
        validPts.append({x, y, z});

        QJsonObject sidecarPoint;
        sidecarPoint[QStringLiteral("point_xyz")] = QJsonArray{x, y, z};
        sidecarPoint[QStringLiteral("rms_reproj_px")] = rmsAfter;
        sidecarPoint[QStringLiteral("track_len")] = trackLen;
        sidecarPoint[QStringLiteral("min_tri_angle_deg")] = 0.0;
        pointsForSidecar.append(sidecarPoint);
    }

    if (validPts.isEmpty())
    {
        return exportResult;
    }

    QDir().mkpath(outputDir);
    const QString plyFileName = useDedicatedFileName
        ? QStringLiteral("sparse_cloud_ba_refined.ply")
        : QStringLiteral("sparse_cloud.ply");
    exportResult.sparseCloudPath = QDir(outputDir).filePath(plyFileName);

    QFile plyFile(exportResult.sparseCloudPath);
    if (!plyFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        exportResult.errorMessage = QStringLiteral("无法写入稀疏点云文件: %1").arg(exportResult.sparseCloudPath);
        return exportResult;
    }

    QTextStream out(&plyFile);
    out << "ply\nformat ascii 1.0\n"
        << "element vertex " << validPts.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\nend_header\n";
    for (const Pt3 &pt : validPts)
    {
        out << QStringLiteral("%1 %2 %3\n")
                   .arg(pt.x, 0, 'f', 6)
                   .arg(pt.y, 0, 'f', 6)
                   .arg(pt.z, 0, 'f', 6);
    }
    plyFile.close();

    const QString sidecarPath = QDir(outputDir).filePath(QStringLiteral("sparse_cloud_points.json"));
    const QJsonObject quality = buildSparseQualityMetadata(
        pointsForSidecar,
        baResult.value(QStringLiteral("camera_count")).toInt(selectedImages.size()),
        true,
        kSparseResultKindSparsePostprocess,
        kSparseResultKindSfmSparseReconstruction);
    QJsonObject sidecarRoot = mergeSparseQualityIntoRecord(
        QJsonObject{{QStringLiteral("points"), pointsForSidecar},
                    {QStringLiteral("operation"), QStringLiteral("bundle_adjust")}},
        quality);
    QFile sidecarFile(sidecarPath);
    if (!sidecarFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        exportResult.errorMessage = QStringLiteral("无法写入点级结果文件: %1").arg(sidecarPath);
        return exportResult;
    }
    sidecarFile.write(QJsonDocument(sidecarRoot).toJson(QJsonDocument::Indented));
    sidecarFile.close();

    QJsonObject files;
    files[QStringLiteral("sparse_cloud_points_json")] = sidecarPath;
    exportResult.extraRecord[QStringLiteral("files")] = files;
    exportResult.extraRecord[QStringLiteral("operation")] = QStringLiteral("bundle_adjust");
    exportResult.extraRecord[QStringLiteral("operation_display_name")] = QStringLiteral("BA 精化点云");
    exportResult.extraRecord[QStringLiteral("ba_mean_rms_after")] =
        baResult.value(QStringLiteral("mean_rms_after")).toDouble();
    exportResult.extraRecord[QStringLiteral("selected_images")] = QJsonArray::fromStringList(selectedImages);
    exportResult.extraRecord = mergeSparseQualityIntoRecord(exportResult.extraRecord, quality);
    exportResult.pointCount = validPts.size();
    exportResult.exported = true;
    return exportResult;
}

bool writeBundleAdjustReport(const QString &reportPath,
                             const QJsonObject &baResult,
                             const QMap<QString, QJsonObject> &beforeCameras,
                             const QMap<QString, QJsonObject> &afterCameras)
{
    if (reportPath.isEmpty())
    {
        return false;
    }

    const QJsonObject rep = buildBundleAdjustReport(baResult, beforeCameras, afterCameras);
    if (rep.isEmpty())
    {
        return false;
    }

    QFile rf(reportPath);
    if (!rf.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    rf.write(QJsonDocument(rep).toJson(QJsonDocument::Compact));
    return true;
}

bool writeLatestAndAppendHistoryReport(const QString &reportsDir,
                                       const QString &latestFileName,
                                       const QString &historyFileName,
                                       const QJsonObject &report)
{
    if (reportsDir.isEmpty() || report.isEmpty())
    {
        return false;
    }

    QDir().mkpath(reportsDir);

    const QString latestPath = QDir(reportsDir).filePath(latestFileName);
    QFile latestFile(latestPath);
    if (!latestFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    latestFile.write(QJsonDocument(report).toJson(QJsonDocument::Compact));
    latestFile.close();

    const QString historyPath = QDir(reportsDir).filePath(historyFileName);
    QJsonArray history;
    QFile historyFile(historyPath);
    if (historyFile.open(QIODevice::ReadOnly))
    {
        const QJsonDocument doc = QJsonDocument::fromJson(historyFile.readAll());
        if (doc.isArray())
        {
            history = doc.array();
        }
        historyFile.close();
    }
    history.append(report);

    if (!historyFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    historyFile.write(QJsonDocument(history).toJson(QJsonDocument::Indented));
    historyFile.close();
    return true;
}

ReconstructionQualityProjectReportResult writeReconstructionQualityProjectReport(
    ProjectData *projectData,
    const QString &baseName)
{
    ReconstructionQualityProjectReportResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目未打开，无法生成重建质量报告");
        return result;
    }

    const QString assetsDir = ProjectIO::projectAssetsDir(projectData->currentProjectPath());
    if (assetsDir.isEmpty())
    {
        result.errorMessage = QStringLiteral("无法解析项目 assets 目录");
        return result;
    }

    const QString reportsDir = QDir(assetsDir).filePath(QStringLiteral("reports"));
    const auto writeResult = xjw::qc::ReconstructionQualityReport::writeFromProjectMeta(
        projectData->metadata(),
        reportsDir,
        baseName);
    if (!writeResult.ok)
    {
        result.errorMessage = writeResult.error;
        return result;
    }

    QJsonObject record;
    record[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    record[QStringLiteral("type")] = QStringLiteral("reconstruction_quality");
    record[QStringLiteral("path")] = writeResult.jsonPath;
    record[QStringLiteral("json_path")] = writeResult.jsonPath;
    record[QStringLiteral("csv_path")] = writeResult.csvPath;
    record[QStringLiteral("total_image_count")] = writeResult.report.value(QStringLiteral("total_image_count"));
    record[QStringLiteral("registered_image_count")] =
        writeResult.report.value(QStringLiteral("registered_image_count"));
    record[QStringLiteral("sparse_point_count")] = writeResult.report.value(QStringLiteral("sparse_point_count"));
    record[QStringLiteral("dense_point_count")] = writeResult.report.value(QStringLiteral("dense_point_count"));
    record[QStringLiteral("mvs_valid_coverage")] = writeResult.report.value(QStringLiteral("mvs_valid_coverage"));
    record[QStringLiteral("dem_coverage")] = writeResult.report.value(QStringLiteral("dem_coverage"));
    record[QStringLiteral("control_point_count")] = writeResult.report.value(QStringLiteral("control_point_count"));
    record[QStringLiteral("check_point_count")] = writeResult.report.value(QStringLiteral("check_point_count"));
    record[QStringLiteral("scale_bar_count")] = writeResult.report.value(QStringLiteral("scale_bar_count"));
    record[QStringLiteral("control_point_rmse_m")] = writeResult.report.value(QStringLiteral("control_point_rmse_m"));
    record[QStringLiteral("check_point_rmse_m")] = writeResult.report.value(QStringLiteral("check_point_rmse_m"));
    record[QStringLiteral("scale_bar_rmse_m")] = writeResult.report.value(QStringLiteral("scale_bar_rmse_m"));

    if (!projectData->upsertResultRecordByPath(QStringLiteral("report_results"),
                                               QStringLiteral("path"),
                                               record,
                                               true))
    {
        result.errorMessage = QStringLiteral("质量报告已写出，但写入项目 metadata 失败");
        return result;
    }

    result.saved = true;
    result.jsonPath = writeResult.jsonPath;
    result.csvPath = writeResult.csvPath;
    result.record = record;
    return result;
}

} // namespace xjw::gui::project
