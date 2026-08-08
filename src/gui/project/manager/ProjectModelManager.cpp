#include "ProjectModelManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "ProjectMetadataOperations.h"
#include "ProjectModelWorkflowPolicy.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowUtils.h"
#include "GuiTaskRunner.h"
#include "ProjectOpenGuard.h"
#include "Logger.h"
#include "ModelWorkflowService.h"
#include "project/ProjectCommonUtils.h"

#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QMessageBox>
#include <QMetaObject>
#include <QPointer>
#include <QStringList>
#include <QThread>

#include <algorithm>
#include <memory>
#include <mutex>
#include <utility>

using xjw::gui::project::resolveLatestDenseCloudPath;

#ifndef PLASCAN_VERSION
#define PLASCAN_VERSION "unknown"
#endif

namespace
{

struct ModelTaskResult
{
    QJsonObject result;
    QString errMsg;
    bool ok = false;
};

struct ResolvedModelSource
{
    QString sourcePointCloudPath;
    QString requestedSourcePath;
    QString outputRoot;
};

struct ModelProgressLogState
{
    std::mutex mutex;
    QString lastStage;
    int highestPercent = 0;
};

QString utcNowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
}

void showTaskFailure(QWidget *parentWidget,
                     const QString &title,
                     const QString &prefix,
                     const QString &errorMessage)
{
    QMessageBox::warning(parentWidget,
                         title,
                         QStringLiteral("%1：%2").arg(prefix, errorMessage));
}

void mergeJsonObject(QJsonObject *target, const QJsonObject &source);

auto makeProgressReporter(QPointer<ProjectModelManager> manager,
                          QPointer<ProjectManager> owner,
                          const xjw::gui::project::ProjectSessionContext &session)
{
    const auto log_state = std::make_shared<ModelProgressLogState>();
    return [manager, owner, session, log_state](const QString &stage, int percent)
    {
        if (!manager || !owner)
        {
            return;
        }
        bool new_stage = false;
        int monotonic_percent = 0;
        {
            const std::lock_guard<std::mutex> lock(log_state->mutex);
            log_state->highestPercent = std::max(
                log_state->highestPercent,
                std::clamp(percent, 0, 100));
            monotonic_percent = log_state->highestPercent;
            if (stage != log_state->lastStage)
            {
                log_state->lastStage = stage;
                new_stage = true;
            }
        }
        if (new_stage)
        {
            LOG_INFO(QStringLiteral("[模型生成][%1%] %2")
                         .arg(monotonic_percent)
                         .arg(stage));
        }
        QMetaObject::invokeMethod(
            manager.data(),
            [manager, owner, session, stage, monotonic_percent]()
        {
            if (!manager || !owner || !owner->isCurrentSession(session))
            {
                return;
            }
            emit manager->meshProgressChanged(stage, monotonic_percent);
        }, Qt::QueuedConnection);
    };
}

void logModelWorkflowResult(
    const xjw::mesh::workflow::WorkflowResult &workflowResult)
{
    if (!workflowResult.ok)
    {
        LOG_ERROR(QStringLiteral("[模型生成] 失败：%1")
                      .arg(workflowResult.errorMessage));
        const QJsonObject &payload = workflowResult.payload;
        if (payload.value(QStringLiteral(
                "final_depth_completeness_available")).toBool(false))
        {
            LOG_ERROR(QStringLiteral(
                "[模型生成] 最终完整性：中位=%1，P10=%2，最低=%3，"
                "最差视角=%4")
                .arg(payload.value(QStringLiteral(
                         "final_depth_completeness_median_frame_recall"))
                         .toDouble(),
                     0,
                     'f',
                     4)
                .arg(payload.value(QStringLiteral(
                         "final_depth_completeness_p10_frame_recall"))
                         .toDouble(),
                     0,
                     'f',
                     4)
                .arg(payload.value(QStringLiteral(
                         "final_depth_completeness_minimum_frame_recall"))
                         .toDouble(),
                     0,
                     'f',
                     4)
                .arg(payload.value(QStringLiteral(
                         "final_depth_completeness_worst_frames"))
                         .toString()));
        }
        return;
    }

    const QJsonObject &payload = workflowResult.payload;
    LOG_INFO(QStringLiteral(
        "[模型生成] 完成：算法=%1，顶点=%2，面=%3，输出=%4")
        .arg(payload.value(QStringLiteral("mesh_algorithm"))
                 .toString(QStringLiteral("unknown")))
        .arg(payload.value(QStringLiteral("vertex_count")).toInt())
        .arg(payload.value(QStringLiteral("face_count")).toInt())
        .arg(payload.value(QStringLiteral("model_ply")).toString()));
    if (payload.contains(QStringLiteral("effective_worker_count")))
    {
        const auto metric = [&payload](const char *key)
        {
            return payload.value(QLatin1String(key)).toDouble();
        };
        LOG_INFO(QStringLiteral(
            "[模型生成] CPU阶段统计：线程=%1；TSDF=%2 ms/%3%；"
            "八叉树=%4 ms/%5%；TGV=%6 ms/%7%；占据投影=%8 ms/%9%；"
            "最小割=%10 ms/%11%；清理=%12 ms/%13%"
            "（时间/并行占用率）")
            .arg(metric("effective_worker_count"), 0, 'f', 0)
            .arg(metric("tsdf_integration_elapsed_ms"), 0, 'f', 0)
            .arg(metric("tsdf_integration_cpu_duty") * 100.0, 0, 'f', 1)
            .arg(metric("adaptive_tgv_octree_elapsed_ms"), 0, 'f', 0)
            .arg(metric("adaptive_tgv_octree_cpu_duty") * 100.0, 0, 'f', 1)
            .arg(metric("adaptive_tgv_solver_elapsed_ms"), 0, 'f', 0)
            .arg(metric("adaptive_tgv_solver_cpu_duty") * 100.0, 0, 'f', 1)
            .arg(metric("visibility_occupancy_projection_elapsed_ms"), 0, 'f', 0)
            .arg(metric("visibility_occupancy_projection_cpu_duty") * 100.0, 0, 'f', 1)
            .arg(metric("visibility_occupancy_min_cut_elapsed_ms"), 0, 'f', 0)
            .arg(metric("visibility_occupancy_min_cut_cpu_duty") * 100.0, 0, 'f', 1)
            .arg(metric("visibility_occupancy_cleanup_elapsed_ms"), 0, 'f', 0)
            .arg(metric("visibility_occupancy_cleanup_cpu_duty") * 100.0, 0, 'f', 1));
    }
    if (payload.value(QStringLiteral(
            "final_depth_completeness_available")).toBool(false))
    {
        const bool completeness_passed = payload.value(QStringLiteral(
            "final_depth_completeness_gate_passed")).toBool(false);
        const QString completeness_message = QStringLiteral(
            "[模型生成] 最终完整性%1：中位=%2，P10=%3，最低=%4")
            .arg(completeness_passed
                     ? QStringLiteral("通过")
                     : QStringLiteral("未通过（当前配置未强制质量门）"))
            .arg(payload.value(QStringLiteral(
                     "final_depth_completeness_median_frame_recall"))
                     .toDouble(),
                 0,
                 'f',
                 4)
            .arg(payload.value(QStringLiteral(
                     "final_depth_completeness_p10_frame_recall"))
                     .toDouble(),
                 0,
                 'f',
                 4)
            .arg(payload.value(QStringLiteral(
                     "final_depth_completeness_minimum_frame_recall"))
                     .toDouble(),
                 0,
                 'f',
                 4);
        if (completeness_passed)
        {
            LOG_INFO(completeness_message);
        }
        else
        {
            LOG_WARN(completeness_message);
        }
    }
    if (payload.contains(QStringLiteral("reliably_colored_vertex_count")))
    {
        const int reliable_count = payload.value(QStringLiteral(
            "reliably_colored_vertex_count")).toInt();
        const int fallback_count = payload.value(QStringLiteral(
            "fallback_color_vertex_count")).toInt();
        const QString color_summary = QStringLiteral(
            "[模型生成] 顶点颜色：可靠=%1，最佳视图回退=%2，"
            "邻域传播=%3，固定颜色回退=%4")
            .arg(reliable_count)
            .arg(payload.value(QStringLiteral(
                     "best_view_fallback_color_vertex_count")).toInt())
            .arg(payload.value(QStringLiteral(
                     "propagated_color_vertex_count")).toInt())
            .arg(fallback_count);
        if (reliable_count <= 0)
        {
            LOG_WARN(color_summary + QStringLiteral(
                "；没有影像颜色投影成功，模型将显示为统一回退色"));
        }
        else
        {
            LOG_INFO(color_summary);
        }
    }
}

void applyWorkflowResult(ModelTaskResult *task,
                         const xjw::mesh::workflow::WorkflowResult &workflowResult)
{
    if (!task)
    {
        return;
    }

    task->ok = workflowResult.ok;
    task->errMsg = workflowResult.errorMessage;
    task->result = workflowResult.payload;
}

void mergeJsonObject(QJsonObject *target, const QJsonObject &source)
{
    if (!target)
    {
        return;
    }

    for (auto it = source.begin(); it != source.end(); ++it)
    {
        target->insert(it.key(), it.value());
    }
}

void persistModelResult(ProjectData *projectData,
                        const QJsonObject &modelRecord)
{
    if (!projectData)
    {
        return;
    }

    QJsonObject metadata = projectData->metadata();
    xjw::gui::project::upsertMetaArrayRecordByPath(&metadata,
                                                   QStringLiteral("model_results"),
                                                   QStringLiteral("model_ply"),
                                                   modelRecord);
    xjw::gui::project::persistProjectMeta(projectData, metadata, true);
}

bool pathBelongsToDirectory(const QString &path, const QString &directory)
{
    if (path.isEmpty() || directory.isEmpty())
    {
        return false;
    }
    const QString normalized_path = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    QString normalized_directory = QDir::cleanPath(QFileInfo(directory).absoluteFilePath());
    if (!normalized_directory.endsWith(QDir::separator()))
    {
        normalized_directory += QDir::separator();
    }
    return normalized_path.startsWith(normalized_directory, Qt::CaseInsensitive);
}

QJsonObject depthGenerationSnapshot(const QJsonObject &metadata,
                                    const QString &sourcePath)
{
    QJsonObject snapshot;
    int frame_count = 0;
    int maximum_neighbor_count = -1;
    double processing_elapsed_ms = 0.0;
    qint64 artifact_bytes = 0;
    qint64 support_pixel_count = 0;
    qint64 missing_pixel_count = 0;
    qint64 targeted_recovered_pixel_count = 0;
    qint64 native_patchmatch_pixel_count = 0;
    qint64 targeted_patchmatch_pixel_count = 0;
    qint64 cross_view_measured_pixel_count = 0;
    qint64 anchored_interpolation_pixel_count = 0;
    qint64 residual_patchmatch_pixel_count = 0;
    qint64 unclassified_valid_pixel_count = 0;
    bool has_depth_provenance = false;
    QJsonObject missing_reason_counts;

    for (const QJsonValue &value : metadata.value(
             QStringLiteral("depth_map_results")).toArray())
    {
        const QJsonObject record = value.toObject();
        const QString output_dir = record.value(
            QStringLiteral("mvs_output_dir")).toString();
        const QString depth_path = record.value(
            QStringLiteral("depth_png")).toString();
        if (!sourcePath.isEmpty() &&
            QDir::cleanPath(output_dir).compare(
                QDir::cleanPath(sourcePath), Qt::CaseInsensitive) != 0 &&
            !pathBelongsToDirectory(depth_path, sourcePath))
        {
            continue;
        }

        ++frame_count;
        if (!snapshot.contains(QStringLiteral("quality_profile")))
        {
            snapshot[QStringLiteral("quality_profile")] = record.value(
                QStringLiteral("quality_profile"));
        }
        if (!snapshot.contains(QStringLiteral("filter_mode")))
        {
            snapshot[QStringLiteral("filter_mode")] = record.value(
                QStringLiteral("filter_mode"));
        }
        maximum_neighbor_count = std::max(
            maximum_neighbor_count,
            record.value(QStringLiteral("requested_source_view_count"))
                .toInt(record.value(QStringLiteral("source_view_count")).toInt(-1)));
        processing_elapsed_ms += record.value(
            QStringLiteral("elapsed_ms")).toDouble(0.0);

        const QJsonObject missing_summary = record.value(
            QStringLiteral("missing_reason_summary")).toObject();
        support_pixel_count += static_cast<qint64>(missing_summary.value(
            QStringLiteral("support_pixel_count")).toDouble(0.0));
        missing_pixel_count += static_cast<qint64>(missing_summary.value(
            QStringLiteral("missing_pixel_count")).toDouble(0.0));
        for (const QString &reason_key : {
                 QStringLiteral("patchmatch_unresolved_pixel_count"),
                 QStringLiteral("low_confidence_pixel_count"),
                 QStringLiteral("local_depth_outlier_pixel_count"),
                 QStringLiteral("small_component_pixel_count"),
                 QStringLiteral("geometry_contradiction_pixel_count"),
                 QStringLiteral("insufficient_geometry_support_pixel_count"),
                 QStringLiteral("unclassified_pixel_count")})
        {
            missing_reason_counts[reason_key] =
                missing_reason_counts.value(reason_key).toDouble(0.0) +
                missing_summary.value(reason_key).toDouble(0.0);
        }
        targeted_recovered_pixel_count += static_cast<qint64>(
            record.value(QStringLiteral(
                "targeted_gap_recovery_diagnostics"))
                .toObject()
                .value(QStringLiteral("recovered_pixel_count"))
                .toDouble(0.0));

        const QJsonObject provenance_summary = record.value(
            QStringLiteral("depth_provenance_summary")).toObject();
        if (provenance_summary.value(QStringLiteral("available")).toBool(false))
        {
            has_depth_provenance = true;
            native_patchmatch_pixel_count += static_cast<qint64>(
                provenance_summary.value(QStringLiteral(
                    "native_patchmatch_pixel_count")).toDouble(0.0));
            targeted_patchmatch_pixel_count += static_cast<qint64>(
                provenance_summary.value(QStringLiteral(
                    "targeted_patchmatch_pixel_count")).toDouble(0.0));
            cross_view_measured_pixel_count += static_cast<qint64>(
                provenance_summary.value(QStringLiteral(
                    "cross_view_measured_pixel_count")).toDouble(0.0));
            anchored_interpolation_pixel_count += static_cast<qint64>(
                provenance_summary.value(QStringLiteral(
                    "anchored_interpolation_pixel_count")).toDouble(0.0));
            residual_patchmatch_pixel_count += static_cast<qint64>(
                provenance_summary.value(QStringLiteral(
                    "residual_patchmatch_pixel_count")).toDouble(0.0));
            unclassified_valid_pixel_count += static_cast<qint64>(
                provenance_summary.value(QStringLiteral(
                    "unclassified_valid_pixel_count")).toDouble(0.0));
        }

        for (const QString &path_key : {
                 QStringLiteral("depth_png"),
                 QStringLiteral("raw_depth_path"),
                 QStringLiteral("raw_confidence_path"),
                 QStringLiteral("valid_mask_path"),
                 QStringLiteral("missing_reason_path"),
                 QStringLiteral("missing_reason_preview_path"),
                 QStringLiteral("targeted_gap_recovered_mask_path"),
                 QStringLiteral("residual_reestimated_mask_path"),
                 QStringLiteral("depth_provenance_path")})
        {
            const QFileInfo artifact_info(record.value(path_key).toString());
            if (artifact_info.exists())
            {
                artifact_bytes += artifact_info.size();
            }
        }
    }

    if (frame_count <= 0)
    {
        return {};
    }
    snapshot[QStringLiteral("frame_count")] = frame_count;
    snapshot[QStringLiteral("maximum_neighbor_count")] = maximum_neighbor_count;
    snapshot[QStringLiteral("processing_elapsed_ms")] = processing_elapsed_ms;
    snapshot[QStringLiteral("artifact_bytes")] = static_cast<double>(artifact_bytes);
    if (support_pixel_count > 0)
    {
        snapshot[QStringLiteral("missing_reason_schema_version")] = 1;
        snapshot[QStringLiteral("support_pixel_count")] =
            static_cast<double>(support_pixel_count);
        snapshot[QStringLiteral("missing_pixel_count")] =
            static_cast<double>(missing_pixel_count);
        snapshot[QStringLiteral("missing_within_support_ratio")] =
            static_cast<double>(missing_pixel_count) /
            static_cast<double>(support_pixel_count);
        snapshot[QStringLiteral("missing_reason_counts")] =
            missing_reason_counts;
        snapshot[QStringLiteral("targeted_gap_recovered_pixel_count")] =
            static_cast<double>(targeted_recovered_pixel_count);
    }
    if (has_depth_provenance)
    {
        snapshot[QStringLiteral("depth_provenance_schema_version")] = 2;
        snapshot[QStringLiteral("native_patchmatch_pixel_count")] =
            static_cast<double>(native_patchmatch_pixel_count);
        snapshot[QStringLiteral("targeted_patchmatch_pixel_count")] =
            static_cast<double>(targeted_patchmatch_pixel_count);
        snapshot[QStringLiteral("cross_view_measured_pixel_count")] =
            static_cast<double>(cross_view_measured_pixel_count);
        snapshot[QStringLiteral("anchored_interpolation_pixel_count")] =
            static_cast<double>(anchored_interpolation_pixel_count);
        snapshot[QStringLiteral("residual_patchmatch_pixel_count")] =
            static_cast<double>(residual_patchmatch_pixel_count);
        snapshot[QStringLiteral("unclassified_valid_pixel_count")] =
            static_cast<double>(unclassified_valid_pixel_count);
    }
    return snapshot;
}

QJsonObject buildMeshReconstructionRecord(const QJsonObject &taskResult,
                                          const QString &denseCloudPath,
                                          const QJsonObject &settings,
                                          const QJsonObject &metadata)
{
    const QString sourceData = settings.value(QStringLiteral("source_data")).toString(QStringLiteral("point_cloud"));
    const QString sourcePath = settings.value(QStringLiteral("source_path")).toString(denseCloudPath);
    const QString sourceDenseCloud =
        sourceData == QStringLiteral("point_cloud")
            ? denseCloudPath
            : QString();

    QJsonObject modelRecord = xjw::gui::project::makeModelResultRecord(
        utcNowIso(),
        QStringLiteral("mesh_reconstruction"),
        taskResult.value(QStringLiteral("model_ply")).toString(),
        taskResult.value(QStringLiteral("vertex_count")).toInt(-1),
        taskResult.value(QStringLiteral("face_count")).toInt(-1),
        QString(),
        sourceDenseCloud,
        sourceDenseCloud);

    mergeJsonObject(&modelRecord, taskResult);
    modelRecord[QStringLiteral("source_data")] = sourceData;
    modelRecord[QStringLiteral("source_path")] = sourcePath;
    modelRecord[QStringLiteral("source_label")] = settings.value(QStringLiteral("source_label")).toString();
    modelRecord[QStringLiteral("requested_method")] = settings.value(QStringLiteral("method")).toString();
    modelRecord[QStringLiteral("reconstruction_mode")] =
        taskResult.value(QStringLiteral("reconstruction_mode"))
            .toString(settings.value(QStringLiteral("reconstruction_mode")).toString());
    modelRecord[QStringLiteral("requested_quality_profile")] =
        settings.contains(QStringLiteral("qualityProfile"))
            ? settings.value(QStringLiteral("qualityProfile")).toString()
            : QStringLiteral("balanced");
    modelRecord[QStringLiteral("requested_model_quality_profile")] =
        settings.value(QStringLiteral("modelQualityProfile")).toString(
            modelRecord.value(QStringLiteral("requested_quality_profile")).toString());
    modelRecord[QStringLiteral("requested_depth_quality_profile")] =
        settings.value(QStringLiteral("depthQualityProfile")).toString();
    modelRecord[QStringLiteral("software_version")] = QStringLiteral(PLASCAN_VERSION);
    modelRecord[QStringLiteral("model_property_schema_version")] = 1;

    QJsonObject reconstruction_parameters;
    reconstruction_parameters[QStringLiteral("surface_type")] = settings.value(
        QStringLiteral("surface_type"));
    reconstruction_parameters[QStringLiteral("interpolation")] = settings.value(
        QStringLiteral("interpolation"));
    reconstruction_parameters[QStringLiteral("strict_volumetric_masks")] = settings.value(
        QStringLiteral("strictVolumetricMasks"));
    reconstruction_parameters[QStringLiteral("calculate_vertex_colors")] = settings.value(
        QStringLiteral("calculateVertexColors"));
    reconstruction_parameters[QStringLiteral("quality")] = settings.value(
        QStringLiteral("quality"));
    reconstruction_parameters[QStringLiteral("quality_profile")] = settings.value(
        QStringLiteral("qualityProfile"));
    reconstruction_parameters[QStringLiteral("model_quality_profile")] = settings.value(
        QStringLiteral("modelQualityProfile"));
    reconstruction_parameters[QStringLiteral("depth_quality_profile")] = settings.value(
        QStringLiteral("depthQualityProfile"));
    reconstruction_parameters[QStringLiteral("target_faces")] = settings.value(
        QStringLiteral("simplifyTargetFaces"));
    reconstruction_parameters[QStringLiteral("processing_elapsed_ms")] = taskResult.value(
        QStringLiteral("processing_elapsed_ms"));
    modelRecord[QStringLiteral("reconstruction_parameters")] = reconstruction_parameters;

    if (sourceData == QStringLiteral("depth_maps"))
    {
        modelRecord[QStringLiteral("depth_generation_parameters")] =
            depthGenerationSnapshot(metadata, sourcePath);
    }
    return modelRecord;
}

QJsonObject buildTextureMappingRecord(const QJsonObject &baseRecord,
                                      const QJsonObject &taskResult,
                                      const QString &meshPath)
{
    QJsonObject modelRecord = baseRecord;
    if (modelRecord.isEmpty())
    {
        modelRecord = xjw::gui::project::makeModelResultRecord(utcNowIso(),
                                                                QStringLiteral("texture_mapping"),
                                                                meshPath,
                                                                taskResult.value(QStringLiteral("vertex_count"))
                                                                    .toInt(-1),
                                                                taskResult.value(QStringLiteral("face_count"))
                                                                    .toInt(-1));
    }

    modelRecord[QStringLiteral("created_at")] = utcNowIso();
    mergeJsonObject(&modelRecord, taskResult);
    modelRecord[QStringLiteral("textured")] = true;
    modelRecord[QStringLiteral("final_model_format")] = QStringLiteral("OBJ");
    modelRecord[QStringLiteral("final_model_path")] = taskResult.value(QStringLiteral("model_obj")).toString();
    modelRecord[QStringLiteral("requested_export_format")] = QStringLiteral("OBJ");
    return modelRecord;
}

QString meshReconstructionSuccessMessage(const QJsonObject &taskResult)
{
    QString message = QStringLiteral(
        "模型网格生成完成。\n网格: %1\n顶点: %2  面数: %3\n"
        "如需照片纹理，请继续执行“工作流程 > 生成纹理”。")
        .arg(taskResult.value(QStringLiteral("final_model_path")).toString())
        .arg(taskResult.value(QStringLiteral("vertex_count")).toInt(-1))
        .arg(taskResult.value(QStringLiteral("face_count")).toInt(-1));
    if (!taskResult.contains(QStringLiteral("accepted_frame_count")))
    {
        return message;
    }

    message += QStringLiteral("\n融合深度: %1/%2 帧（辅助 %3，质量剔除 %4）")
                   .arg(taskResult.value(QStringLiteral("accepted_frame_count")).toInt())
                   .arg(taskResult.value(QStringLiteral("input_frame_count")).toInt())
                   .arg(taskResult.value(
                       QStringLiteral("auxiliary_surface_only_frame_count")).toInt())
                   .arg(taskResult.value(
                       QStringLiteral("robust_frame_quality_rejected_frame_count")).toInt());
    const QJsonArray rejected_refs = taskResult.value(
        QStringLiteral("robust_frame_quality_rejected_ref_indices")).toArray();
    if (!rejected_refs.isEmpty())
    {
        QStringList ref_labels;
        ref_labels.reserve(rejected_refs.size());
        for (const QJsonValue &value : rejected_refs)
        {
            ref_labels.push_back(QString::number(value.toInt()));
        }
        message += QStringLiteral("\n被剔除视角: %1").arg(ref_labels.join(
            QStringLiteral(", ")));
    }
    if (taskResult.value(QStringLiteral("orbital_maximum_angular_gap_degrees"))
            .toDouble() > 0.0)
    {
        message += QStringLiteral("\n最大环向视角缺口: %1°（中位间隔的 %2 倍）")
                       .arg(taskResult.value(
                           QStringLiteral("orbital_maximum_angular_gap_degrees"))
                                .toDouble(),
                            0,
                            'f',
                            1)
                       .arg(taskResult.value(
                           QStringLiteral("orbital_maximum_angular_gap_ratio"))
                                .toDouble(),
                            0,
                            'f',
                            2);
    }
    if (taskResult.value(QStringLiteral("depth_completeness_available")).toBool())
    {
        message += QStringLiteral("\n深度完整性: 中位 %1%，P10 %2%（%3）")
                       .arg(100.0 * taskResult.value(
                           QStringLiteral("depth_completeness_median_frame_recall"))
                                        .toDouble(),
                            0,
                            'f',
                            1)
                       .arg(100.0 * taskResult.value(
                           QStringLiteral("depth_completeness_p10_frame_recall"))
                                        .toDouble(),
                            0,
                            'f',
                            1)
                       .arg(taskResult.value(
                           QStringLiteral("depth_completeness_gate_passed")).toBool()
                                ? QStringLiteral("通过")
                                : QStringLiteral("未通过"));
        if (taskResult.value(
                QStringLiteral(
                    "depth_completeness_gap_boundary_available")).toBool())
        {
            message += QStringLiteral("\n最大缺口边界最低召回: %1%（%2）")
                           .arg(100.0 * taskResult.value(
                               QStringLiteral(
                                   "depth_completeness_gap_boundary_minimum_recall"))
                                            .toDouble(),
                                0,
                                'f',
                                1)
                           .arg(taskResult.value(
                               QStringLiteral(
                                   "depth_completeness_gap_boundary_gate_passed"))
                                        .toBool()
                                    ? QStringLiteral("通过")
                                    : QStringLiteral("未通过"));
        }
    }
    return message;
}

QString textureMappingSuccessMessage(const QJsonObject &taskResult)
{
    return QStringLiteral(
        "纹理映射完成。\nOBJ: %1\n纹理: %2\n"
        "映射面: %3  未映射面: %4\n纹理块: %5  使用视角: %6  图集占用率: %7%")
        .arg(taskResult.value(QStringLiteral("model_obj")).toString())
        .arg(taskResult.value(QStringLiteral("texture_png")).toString())
        .arg(taskResult.value(
            QStringLiteral("texture_mapped_face_count")).toInt())
        .arg(taskResult.value(
            QStringLiteral("texture_unmapped_face_count")).toInt())
        .arg(taskResult.value(QStringLiteral("texture_chart_count")).toInt())
        .arg(taskResult.value(
            QStringLiteral("texture_used_view_count")).toInt())
        .arg(taskResult.value(
            QStringLiteral("texture_atlas_occupancy")).toDouble() * 100.0,
            0,
            'f',
            1);
}

QString existingDenseCloudPathFromRecord(const QJsonObject &record)
{
    const QString path = QDir::cleanPath(record.value(QStringLiteral("dense_cloud_xyz")).toString().trimmed());
    if (path.isEmpty() || !QFileInfo::exists(path))
    {
        return QString();
    }
    return path;
}

QString depthSourceRoot(const QString &sourcePath)
{
    const QFileInfo info(sourcePath);
    if (info.isDir())
    {
        return QDir::cleanPath(info.absoluteFilePath());
    }
    return QDir::cleanPath(info.absolutePath());
}

QString findDenseCloudForDepthSource(ProjectData *projectData, const QString &depthSourcePath)
{
    const QString root = depthSourceRoot(depthSourcePath);
    if (!root.isEmpty())
    {
        const QString canonical = QDir(root).filePath(QStringLiteral("dense_cloud.ply"));
        if (QFileInfo::exists(canonical))
        {
            return QDir::cleanPath(canonical);
        }
    }

    const QJsonArray denseResults = projectData
        ? projectData->metadata().value(QStringLiteral("dense_cloud_results")).toArray()
        : QJsonArray();
    for (int index = denseResults.size() - 1; index >= 0; --index)
    {
        const QString densePath = existingDenseCloudPathFromRecord(denseResults.at(index).toObject());
        if (densePath.isEmpty())
        {
            continue;
        }
        if (!root.isEmpty() && QFileInfo(densePath).absolutePath() == root)
        {
            return densePath;
        }
    }

    QString latestDensePath;
    QString ignoredError;
    if (projectData && resolveLatestDenseCloudPath(projectData, &latestDensePath, &ignoredError))
    {
        return latestDensePath;
    }
    return QString();
}

bool resolveModelSourceForMeshing(ProjectData *projectData,
                                  const QJsonObject &settings,
                                  ResolvedModelSource *resolvedSource,
                                  QString *errorMessage)
{
    if (!resolvedSource)
    {
        return false;
    }

    const QString sourceData = settings.value(QStringLiteral("source_data")).toString(QStringLiteral("point_cloud"));
    const QString sourcePath = QDir::cleanPath(
        settings.value(QStringLiteral("source_path"))
            .toString(settings.value(QStringLiteral("denseCloudPath")).toString())
            .trimmed());

    resolvedSource->requestedSourcePath = sourcePath;

    if (sourceData == QStringLiteral("depth_maps"))
    {
        if (sourcePath.isEmpty() || !QFileInfo::exists(sourcePath))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("所选深度图源不存在：\n%1").arg(sourcePath);
            }
            return false;
        }

        resolvedSource->sourcePointCloudPath = findDenseCloudForDepthSource(projectData, sourcePath);
        resolvedSource->outputRoot = depthSourceRoot(sourcePath);
        if (resolvedSource->outputRoot.isEmpty())
        {
            resolvedSource->outputRoot = resolvedSource->sourcePointCloudPath.isEmpty()
                ? QFileInfo(sourcePath).absolutePath()
                : QFileInfo(resolvedSource->sourcePointCloudPath).absolutePath();
        }
        return true;
    }

    if (!sourcePath.isEmpty())
    {
        if (!QFileInfo::exists(sourcePath))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("所选源数据文件不存在：\n%1").arg(sourcePath);
            }
            return false;
        }
        resolvedSource->sourcePointCloudPath = sourcePath;
        resolvedSource->outputRoot = QFileInfo(sourcePath).absolutePath();
        return true;
    }

    QString latestDensePath;
    QString latestError;
    if (!resolveLatestDenseCloudPath(projectData, &latestDensePath, &latestError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("%1\n请先完成密集点云生成。").arg(latestError);
        }
        return false;
    }

    resolvedSource->sourcePointCloudPath = latestDensePath;
    resolvedSource->requestedSourcePath = latestDensePath;
    resolvedSource->outputRoot = QFileInfo(latestDensePath).absolutePath();
    return true;
}

template <typename OnSuccess>
bool handleTaskResult(QWidget *parentWidget,
                      const QString &title,
                      const QString &failurePrefix,
                      const ModelTaskResult &task,
                      OnSuccess &&onSuccess)
{
    if (!task.ok)
    {
        showTaskFailure(parentWidget, title, failurePrefix, task.errMsg);
        return false;
    }

    onSuccess(task.result);
    return true;
}

template <typename Owner, typename Worker, typename OnFinished>
void runModelAsyncTask(Owner *owner,
                       Worker &&worker,
                       OnFinished &&onFinished)
{
    xjw::gui::tasks::runGuardedWithOutcome(
        owner,
        std::forward<Worker>(worker),
        [onFinished = std::forward<OnFinished>(onFinished)](
            Owner *,
            xjw::gui::tasks::TaskOutcome<ModelTaskResult> outcome) mutable
        {
            ModelTaskResult task;
            if (outcome.succeeded())
            {
                task = std::move(*outcome.value);
            }
            else
            {
                task.errMsg = QStringLiteral("模型任务执行异常：%1")
                                  .arg(outcome.errorMessage);
                LOG_ERROR(QStringLiteral("模型异步任务失败：%1")
                              .arg(outcome.errorMessage));
            }
            onFinished(task);
        });
}

} // namespace

ProjectModelManager::ProjectModelManager(ProjectManager *owner,
                                         ProjectData *projectData,
                                         QWidget *parentWidget,
                                         QObject *parent)
    : QObject(parent)
    , _owner(owner)
    , _projectData(projectData)
    , _parentWidget(parentWidget)
{
}

bool ProjectModelManager::startMeshReconstructionAsync(const QJsonObject &settings)
{
    if (!xjw::gui::project::requireOpenProject(_projectData, _parentWidget))
    {
        return false;
    }
    if (_isRunning)
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("生成模型"),
                                 QStringLiteral("已有模型或纹理任务正在运行，请等待其完成。"));
        return false;
    }

    const QString dialogTitle = settings.contains(QStringLiteral("source_data"))
        ? QStringLiteral("生成模型")
        : QStringLiteral("网格重建");

    if (settings.value(QStringLiteral("source_data")).toString() ==
        QStringLiteral("depth_maps"))
    {
        const QString depth_source_path =
            settings.value(QStringLiteral("depthMapSourcePath"))
                .toString(settings.value(QStringLiteral("source_path")).toString());
        const auto batch_compatibility =
            xjw::gui::project::assessStoredDepthBatchCompatibility(
                _projectData->metadata(),
                depth_source_path,
                settings.value(QStringLiteral("at_index")).toInt(-1));
        if (!batch_compatibility.compatible)
        {
            QMessageBox::warning(_parentWidget,
                                 dialogTitle,
                                 QStringLiteral("不能使用当前深度图生成模型：\n%1")
                                     .arg(batch_compatibility.reason));
            return false;
        }
    }

    ResolvedModelSource resolvedSource;
    QString sourceError;
    if (!resolveModelSourceForMeshing(_projectData, settings, &resolvedSource, &sourceError))
    {
        QMessageBox::warning(_parentWidget, dialogTitle, sourceError);
        return false;
    }

    QJsonObject effectiveSettings = settings;
    // 网格生成和纹理映射是两个独立工作流。这里覆盖旧项目可能保留的 OBJ 设置，
    // 防止生成模型阶段再次触发纹理烘焙。
    effectiveSettings[QStringLiteral("export_format")] =
        QStringLiteral("PLY");
    if (!effectiveSettings.contains(QStringLiteral("threads")))
    {
        effectiveSettings[QStringLiteral("threads")] =
            xjw::gui::project::recommendedInteractiveModelWorkerCount(
                QThread::idealThreadCount());
    }
    effectiveSettings[QStringLiteral("source_path")] = resolvedSource.requestedSourcePath;
    effectiveSettings[QStringLiteral("resolved_point_cloud_path")] = resolvedSource.sourcePointCloudPath;

    {
        QJsonObject meta = _projectData->metadata();
        meta[QStringLiteral("mesh_reconstruction_settings")] = effectiveSettings;
        xjw::gui::project::persistProjectMeta(_projectData, meta, false);
    }

    LOG_INFO(QStringLiteral(
        "[模型生成] 启动：来源=%1，模式=%2，质量=%3，目标面数=%4，"
        "CPU工作线程=%5，输入=%6，输出=%7")
        .arg(effectiveSettings.value(QStringLiteral("source_data"))
                 .toString(QStringLiteral("point_cloud")))
        .arg(effectiveSettings.value(QStringLiteral("reconstruction_mode"))
                 .toString(QStringLiteral("auto")))
        .arg(effectiveSettings.value(QStringLiteral("quality"))
                 .toString(QStringLiteral("default")))
        .arg(effectiveSettings.value(QStringLiteral("simplifyTargetFaces"))
                 .toInt())
        .arg(effectiveSettings.value(QStringLiteral("threads")).toInt())
        .arg(resolvedSource.requestedSourcePath,
             resolvedSource.outputRoot));

    _isRunning = true;
    _activeCancelFlag = std::make_shared<std::atomic_bool>(false);
    const std::shared_ptr<std::atomic_bool> cancel_flag = _activeCancelFlag;
    emit meshProgressChanged(tr("正在初始化模型生成..."), 0);
    QPointer<ProjectModelManager> self(this);
    QPointer<ProjectManager> ownerGuard(_owner);
    const auto session = _owner->currentSessionContext();
    runModelAsyncTask(
        this,
        [self,
         ownerGuard,
         resolvedSource,
         effectiveSettings,
         session,
         cancel_flag]() -> ModelTaskResult {
            ModelTaskResult task;
            if (!self)
            {
                task.errMsg = QStringLiteral("模型生成已取消：项目窗口已关闭");
                return task;
            }

            xjw::mesh::workflow::ModelBuildRequest request;
            request.sourceData =
                effectiveSettings.value(QStringLiteral("source_data"))
                    .toString(QStringLiteral("point_cloud"));
            request.requestedSourcePath = resolvedSource.requestedSourcePath;
            request.sourcePointCloudPath = resolvedSource.sourcePointCloudPath;
            request.depthMapSourcePath =
                effectiveSettings.value(QStringLiteral("depthMapSourcePath"))
                    .toString(effectiveSettings.value(QStringLiteral("source_path")).toString());
            request.outputRoot = resolvedSource.outputRoot;
            request.settings = effectiveSettings;
            request.isCancelled = [cancel_flag]()
            {
                return cancel_flag->load(std::memory_order_relaxed);
            };
            const auto progress_reporter =
                makeProgressReporter(self, ownerGuard, session);
            request.progress = [cancel_flag, progress_reporter](
                                   const QString &stage, int percent)
            {
                if (!cancel_flag->load(std::memory_order_relaxed))
                {
                    progress_reporter(stage, percent);
                }
            };

            QElapsedTimer processing_timer;
            processing_timer.start();
            xjw::mesh::workflow::WorkflowResult workflowResult =
                xjw::mesh::workflow::buildModel(request);
            workflowResult.payload[QStringLiteral("processing_elapsed_ms")] =
                static_cast<double>(processing_timer.elapsed());
            logModelWorkflowResult(workflowResult);
            applyWorkflowResult(&task, workflowResult);
            return task;
        },
        [self, ownerGuard, resolvedSource, effectiveSettings, session, dialogTitle](const ModelTaskResult &task) {
            if (!self)
            {
                return;
            }
            self->_isRunning = false;
            self->_activeCancelFlag.reset();
            if (!ownerGuard || !ownerGuard->isCurrentSession(session))
            {
                emit self->meshProgressFinished(false);
                return;
            }
            emit self->meshProgressFinished(task.ok);
            handleTaskResult(self->_parentWidget,
                             dialogTitle,
                             QStringLiteral("模型生成失败"),
                             task,
                             [self, resolvedSource, effectiveSettings, dialogTitle](const QJsonObject &taskResult) {
                if (!self)
                {
                    return;
                }
                const QString sourcePointCloudPath =
                    taskResult.value(QStringLiteral("source_point_cloud_path"))
                        .toString(resolvedSource.sourcePointCloudPath);
                const QJsonObject modelRecord = buildMeshReconstructionRecord(
                    taskResult,
                    sourcePointCloudPath,
                    effectiveSettings,
                    self->_projectData->metadata());
                persistModelResult(self->_projectData, modelRecord);
                if (!effectiveSettings.value(QStringLiteral("pipeline_mode")).toBool(false))
                {
                    QMessageBox::information(self->_parentWidget,
                                             dialogTitle,
                                             meshReconstructionSuccessMessage(taskResult));
                }
            });
        });
    return true;
}

void ProjectModelManager::cancelActiveTask()
{
    if (!_isRunning || !_activeCancelFlag)
    {
        return;
    }
    _activeCancelFlag->store(true, std::memory_order_relaxed);
    emit meshProgressChanged(tr("正在取消模型生成..."), 99);
}

void ProjectModelManager::startTextureMappingAsync(const QJsonObject &settings)
{
    if (!xjw::gui::project::requireOpenProject(_projectData, _parentWidget))
    {
        return;
    }
    if (_isRunning)
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("纹理映射"),
                                 QStringLiteral("已有模型或纹理任务正在运行，请等待其完成。"));
        return;
    }

    if (!_projectData)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("纹理映射"),
                             QStringLiteral("项目未就绪"));
        return;
    }

    const auto lookup = xjw::common::project::resolveLatestModelMeshRecord(_projectData->metadata());
    if (!lookup.ok)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("纹理映射"),
                             lookup.errorMessage);
        return;
    }

    const QString meshPath = lookup.meshPath;
    const QJsonObject baseRecord = lookup.modelRecord;
    const QString recorded_depth_source =
        baseRecord.value(QStringLiteral("depth_map_source_path")).toString();
    const QString depthMapSourcePath = !recorded_depth_source.trimmed().isEmpty()
        ? recorded_depth_source
        : (baseRecord.value(QStringLiteral("source_data")).toString() ==
                   QStringLiteral("depth_maps")
               ? baseRecord.value(QStringLiteral("source_path")).toString()
               : QString());
    bool allow_vertex_color_fallback = false;
    if (depthMapSourcePath.trimmed().isEmpty())
    {
        const auto answer = QMessageBox::question(
            _parentWidget,
            QStringLiteral("纹理映射"),
            QStringLiteral(
                "当前模型没有深度图与相机证据，无法执行多视图纹理映射。\n"
                "是否改用网格顶点色生成平面投影纹理？"));
        if (answer != QMessageBox::Yes)
        {
            return;
        }
        allow_vertex_color_fallback = true;
    }

    {
        QJsonObject meta = _projectData->metadata();
        meta[QStringLiteral("texture_mapping_settings")] = settings;
        xjw::gui::project::persistProjectMeta(_projectData, meta, false);
    }

    const QString productsDir = QFileInfo(meshPath).absolutePath();

    _isRunning = true;
    _activeCancelFlag = std::make_shared<std::atomic_bool>(false);
    const std::shared_ptr<std::atomic_bool> cancel_flag = _activeCancelFlag;
    emit meshProgressChanged(tr("正在初始化纹理映射..."), 0);
    QPointer<ProjectModelManager> self(this);
    QPointer<ProjectManager> ownerGuard(_owner);
    const auto session = _owner->currentSessionContext();
    runModelAsyncTask(
        this,
        [self,
         ownerGuard,
         meshPath,
         productsDir,
         depthMapSourcePath,
         settings,
         session,
         cancel_flag,
         allow_vertex_color_fallback]() -> ModelTaskResult {
            ModelTaskResult task;
            if (!self)
            {
                task.errMsg = QStringLiteral("纹理映射已取消：项目窗口已关闭");
                return task;
            }

            xjw::mesh::workflow::TextureBuildRequest request;
            request.meshPath = meshPath;
            request.outputDir = productsDir;
            request.depthMapSourcePath = depthMapSourcePath;
            request.texture = xjw::mesh::workflow::textureConfigFromSettings(settings);
            request.allowVertexColorFallback = allow_vertex_color_fallback;
            request.isCancelled = [cancel_flag]()
            {
                return cancel_flag->load(std::memory_order_relaxed);
            };
            const auto progress_reporter =
                makeProgressReporter(self, ownerGuard, session);
            request.progress = [cancel_flag, progress_reporter](
                                   const QString &stage, int percent)
            {
                if (!cancel_flag->load(std::memory_order_relaxed))
                {
                    progress_reporter(stage, percent);
                }
            };

            const xjw::mesh::workflow::WorkflowResult workflowResult =
                xjw::mesh::workflow::buildTextureOnly(request);
            applyWorkflowResult(&task, workflowResult);
            return task;
        },
        [self,
         ownerGuard,
         meshPath,
         baseRecord,
         session,
         cancel_flag](const ModelTaskResult &task) {
            if (!self)
            {
                return;
            }
            self->_isRunning = false;
            if (self->_activeCancelFlag == cancel_flag)
            {
                self->_activeCancelFlag.reset();
            }
            if (!ownerGuard || !ownerGuard->isCurrentSession(session))
            {
                emit self->meshProgressFinished(false);
                return;
            }
            emit self->meshProgressFinished(task.ok);
            handleTaskResult(self->_parentWidget,
                             QStringLiteral("纹理映射"),
                             QStringLiteral("纹理映射失败"),
                             task,
                             [self, meshPath, baseRecord](const QJsonObject &taskResult) {
                if (!self)
                {
                    return;
                }
                const QJsonObject modelRecord = buildTextureMappingRecord(baseRecord,
                                                                           taskResult,
                                                                           meshPath);
                persistModelResult(self->_projectData, modelRecord);
                QMessageBox::information(self->_parentWidget,
                                         QStringLiteral("纹理映射"),
                                         textureMappingSuccessMessage(taskResult));
            });
        });
}

bool ProjectModelManager::isRunning() const
{
    return _isRunning;
}
