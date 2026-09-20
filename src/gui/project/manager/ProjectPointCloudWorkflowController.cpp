#include "ProjectPointCloudWorkflowController.h"

#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "MvsSourcePairQualityLoader.h"
#include "PointCloudInputPreparation.h"
#include "PointCloudWorkflowConfig.h"
#include "ProjectMetadataOperations.h"
#include "ProjectModelWorkflowPolicy.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowOperations.h"
#include "project/ProjectIO.h"
#include "project/ProjectMetadata.h"
#include "project/SparseResultQuality.h"
#include "tasks/GuiTaskRunner.h"

#include "DenseCloudBuilder.h"
#include "DepthFrameUtils.h"
#include "../tasks/DepthMapTask.h"
#include "StreamingDepthFusionService.h"
#include "camera/core/capabilities/CameraOperationPlan.h"
#include "camera/models/CameraModelFactories.h"
#include "camera/project/CameraProjectRuntime.h"
#include "io/PathIO.h"
#include "Logger.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QMetaObject>
#include <QPointer>
#include <QSet>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using xjw::common::project::normalizePath;
using xjw::core::project::DenseGenerationSettings;

struct PointCloudWorkflowContext
{
    xjw::gui::project::ProjectTaskContext task;
    xjw::gui::project::ProjectSessionContext session;
    DenseGenerationSettings request;
    QJsonObject settings;
    QString sparseCloudPath;
    QString sparsePointSidecarPath;
    QStringList selectedImages;
    QString outputDir;
    QString projectInputSignature;
    QString reconstructionGenerationId;
    std::vector<xjw::mvs::CameraView> views;
    int atIndex = -1;
    bool reuseDepthMaps = true;
    bool reusedDepthMaps = false;
    bool saveAfterEachStep = false;
    bool calculateColors = true;
    bool replaceDefaultPointCloud = false;
    bool depthMapsOnly = false;
    QString depthError;
};

namespace
{

struct PointCloudTaskResult
{
    bool ok = false;
    bool cancelled = false;
    QString errorMessage;
    QString pointCloudPath;
    int pointCount = 0;
    QJsonObject record;
};

struct DepthEstimationPreparationResult
{
    xjw::core::project::PointCloudInputPreparationResult pointCloudInput;
    xjw::core::project::MvsSourcePairQualityLoadResult sourcePairQuality;
};

QString utcNowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
}

QString workflowDialogTitle(bool depthMapsOnly)
{
    return depthMapsOnly
        ? QStringLiteral("生成模型")
        : QStringLiteral("创建点云");
}

QString patchMatchBackendText(xjw::mvs::PatchMatchBackend backend)
{
    return QString::fromLatin1(xjw::mvs::patchMatchBackendId(backend));
}

QString resolvedSceneProfile(const QJsonObject &settings,
                             QString *resolution_source = nullptr)
{
    const QString configured_profile = settings.value(
        QStringLiteral("sceneProfile")).toString(QStringLiteral("auto"))
                                            .trimmed()
                                            .toLower();
    if (configured_profile == QStringLiteral("aerial_terrain") ||
        configured_profile == QStringLiteral("orbital_object"))
    {
        if (resolution_source)
        {
            *resolution_source = QStringLiteral("explicit");
        }
        return configured_profile;
    }
    if (configured_profile == QStringLiteral("custom") ||
        configured_profile == QStringLiteral("general"))
    {
        if (resolution_source)
        {
            *resolution_source = QStringLiteral("explicit");
        }
        return QStringLiteral("custom");
    }

    if (resolution_source)
    {
        *resolution_source = QStringLiteral("geometry_classifier");
    }
    return QStringLiteral("auto");
}

QString mvsBackendFromStoredFrames(
    const std::vector<xjw::core::project::StoredDepthFrameRecord> &frames)
{
    QStringList devices;
    devices.reserve(static_cast<qsizetype>(frames.size()));
    for (const auto &frame : frames)
    {
        devices.push_back(frame.device);
    }
    return xjw::gui::project::classifyStoredMvsBackendDevices(devices);
}

QString sceneProfileFromStoredFrames(
    const std::vector<xjw::core::project::StoredDepthFrameRecord> &frames)
{
    QString actual_profile;
    for (const auto &frame : frames)
    {
        const QString profile = frame.sceneProfile.trimmed().toLower();
        if (profile.isEmpty())
        {
            continue;
        }
        if (actual_profile.isEmpty())
        {
            actual_profile = profile;
        }
        else if (actual_profile != profile)
        {
            return QStringLiteral("mixed");
        }
    }
    return actual_profile.isEmpty() ? QStringLiteral("unknown") : actual_profile;
}

QString sparseCloudPathFromRecord(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("files"))
                       .toObject()
                       .value(QStringLiteral("sparse_cloud_xyz"))
                       .toString();
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("sparse_cloud_xyz")).toString();
    }
    return QDir::cleanPath(path.trimmed());
}

QString sparsePointSidecarPathFromRecord(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("files"))
                       .toObject()
                       .value(QStringLiteral("sparse_cloud_points_json"))
                       .toString();
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("sparse_cloud_points_json")).toString();
    }
    path = path.trimmed();
    return path.isEmpty() ? QString() : QDir::cleanPath(path);
}

QStringList selectedImagesFromRecord(const QJsonObject &record,
                                     const QJsonObject &metadata,
                                     QString *error_message)
{
    if (error_message)
    {
        error_message->clear();
    }

    QStringList images;
    for (const QJsonValue &value : record.value(QStringLiteral("selected_images")).toArray())
    {
        const QString token = value.toString().trimmed();
        const auto resolved =
            xjw::common::project::resolveProjectImageToken(token, metadata);
        if (resolved.status !=
                xjw::common::project::ImageResolveStatus::Found
            || resolved.path.isEmpty())
        {
            if (error_message)
            {
                *error_message =
                    resolved.status ==
                            xjw::common::project::ImageResolveStatus::Ambiguous
                        ? QStringLiteral(
                              "空三注册影像在当前工程中存在多个同名候选：%1")
                              .arg(token)
                        : QStringLiteral(
                              "无法将空三注册影像映射到当前工程影像：%1")
                              .arg(token);
            }
            return {};
        }
        images.push_back(QDir::cleanPath(resolved.path));
    }
    return images;
}

using PinholeStatesByImageId =
    std::unordered_map<xjw::camera_core::ImageId, xjw::camera_models::frame_pinhole::FramePinholeNumericState>;

bool loadMvsCameras(const QJsonObject &metadata,
                    const std::vector<xjw::camera_core::ImageId> &image_ids,
                    PinholeStatesByImageId *cameras,
                    QString *error_message)
{
    if (!cameras)
    {
        if (error_message)
        {
            *error_message = QStringLiteral("MVS 相机输出对象为空");
        }
        return false;
    }
    cameras->clear();

    const auto runtime = xjw::camera_project::CameraProjectRuntime::load(
        xjw::common::project::projectFilesRootObject(metadata),
        xjw::camera_models::makeBuiltinCameraModelRegistry());
    const auto plan = runtime.planOperationForImages(
        image_ids, xjw::camera_core::CameraOperation::DenseMvs);
    if (!plan.ok())
    {
        if (error_message)
        {
            *error_message = QStringLiteral("MVS 相机能力校验失败：%1")
                                 .arg(QString::fromStdString(plan.failureMessage()));
        }
        return false;
    }

    std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> states;
    std::string state_error;
    if (!runtime.framePinholeStatesForImages(image_ids, &states, &state_error))
    {
        if (error_message)
        {
            *error_message = QStringLiteral("MVS 面阵针孔数值状态解析失败：%1")
                                 .arg(QString::fromStdString(state_error));
        }
        return false;
    }
    if (states.size() != image_ids.size())
    {
        if (error_message)
        {
            *error_message = QStringLiteral("MVS 相机状态与 ImageId 数量不一致");
        }
        return false;
    }
    cameras->reserve(image_ids.size());
    for (std::size_t index = 0; index < image_ids.size(); ++index)
    {
        const auto &state = states[index];
        if (!state.hasBoundIdentity() || state.imageId() != image_ids[index])
        {
            if (error_message)
            {
                *error_message = QStringLiteral("MVS 相机状态的 canonical identity 不匹配");
            }
            cameras->clear();
            return false;
        }
        cameras->emplace(image_ids[index], state);
    }
    return true;
}

bool cameraForImage(const PinholeStatesByImageId& cameras,
                    const xjw::camera_core::ImageId& imageId,
                    xjw::camera_models::frame_pinhole::FramePinholeNumericState* camera)
{
    if (!camera)
    {
        return false;
    }
    const auto it = cameras.find(imageId);
    if (it == cameras.end())
    {
        return false;
    }
    if (!it->second.hasBoundIdentity() || it->second.imageId() != imageId)
    {
        return false;
    }
    *camera = it->second;
    return true;
}

bool buildMvsViews(const QString &projectPath,
                   const QStringList &images,
                   const std::vector<xjw::camera_core::ImageId> &imageIds,
                   const PinholeStatesByImageId &cameras,
                   std::vector<xjw::mvs::CameraView> *views,
                   QString *errorMessage)
{
    if (!views)
    {
        return false;
    }
    if (images.size() != static_cast<qsizetype>(imageIds.size()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("MVS 影像与 canonical ImageId 数量不一致");
        }
        return false;
    }
    std::unordered_set<xjw::camera_core::ImageId> seenImageIds;
    seenImageIds.reserve(imageIds.size());
    views->clear();
    views->reserve(static_cast<std::size_t>(images.size()));
    for (int index = 0; index < images.size(); ++index)
    {
        const QString &image_path = images.at(index);
        const auto &imageId = imageIds[static_cast<std::size_t>(index)];
        if (!seenImageIds.insert(imageId).second)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("MVS 输入包含重复 canonical ImageId");
            }
            return false;
        }
        if (!QFileInfo::exists(image_path))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("空三影像不存在：%1").arg(image_path);
            }
            return false;
        }

        xjw::mvs::CameraView view;
        view.imagePath = xjw::common::io::toUtf8Path(image_path);
        view.validRegionMaskPath = xjw::common::io::toUtf8Path(
            xjw::common::project::ProjectIO::findMaskForImage(projectPath, image_path));
        if (!cameraForImage(cameras, imageId, &view.camera))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("影像缺少有效相机参数：%1").arg(image_path);
            }
            return false;
        }

        // QImageReader 仅读取文件头，不在 GUI 准备阶段解码整幅高分辨率影像。
        const QSize size = QImageReader(image_path).size();
        if (size.isValid())
        {
            view.imageWidth = size.width();
            view.imageHeight = size.height();
        }
        views->push_back(std::move(view));
    }
    return true;
}

QString artifactDirectory(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("raw_depth_path")).toString();
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("depth_png")).toString();
    }
    return path.isEmpty() ? QString() : QFileInfo(path).absolutePath();
}

bool clearDepthWorkspace(xjw::gui::project::ProjectSession* session,
                         const xjw::gui::project::ProjectTaskContext& taskContext,
                         const QString& outputDir,
                         QString* errorMessage)
{
    if (!session || !session->isCurrent(taskContext.session) || !taskContext.cancelFlag ||
        taskContext.cancelFlag->load(std::memory_order_relaxed))
    {
        return false;
    }
    const QString clean_output = QDir::cleanPath(outputDir);
    QDir directory(clean_output);
    const QFileInfoList artifacts = directory.entryInfoList(
        QStringList{QStringLiteral("depth_*"), QStringLiteral("mvs_manifest.json")}, QDir::Files | QDir::Hidden);
    for (const QFileInfo& artifact : artifacts)
    {
        QFile::remove(artifact.absoluteFilePath());
    }
    QDir prepared_directory(directory.filePath(QStringLiteral("prepared_images")));
    const QFileInfoList prepared_artifacts =
        prepared_directory.entryInfoList(QStringList{QStringLiteral("frame_*.png")}, QDir::Files | QDir::Hidden);
    for (const QFileInfo& artifact : prepared_artifacts)
    {
        QFile::remove(artifact.absoluteFilePath());
    }
    directory.rmdir(QStringLiteral("prepared_images"));

    QJsonObject metadata = session->metadata();
    QJsonArray retained;
    for (const QJsonValue& value : metadata.value(QStringLiteral("depth_map_results")).toArray())
    {
        const QJsonObject record = value.toObject();
        if (QDir::cleanPath(artifactDirectory(record)).compare(clean_output, Qt::CaseInsensitive) != 0)
        {
            retained.append(record);
        }
    }
    metadata[QStringLiteral("depth_map_results")] = retained;
    return session->persistMetadata(taskContext.session, metadata, true, errorMessage);
}

std::vector<xjw::camera_core::ImageId>
imageIdsForImages(const QJsonObject& metadata, const QStringList& images, bool* allResolved)
{
    if (allResolved)
    {
        *allResolved = true;
    }
    const QMap<QString, QJsonObject> image_meta_by_path =
        xjw::common::project::projectImageMetaByPath(
            xjw::common::project::projectFilesRootObject(metadata), true);
    std::vector<xjw::camera_core::ImageId> result;
    result.reserve(static_cast<std::size_t>(images.size()));
    QSet<QString> seen_ids;
    for (const QString &image_path : images)
    {
        const QString image_id = image_meta_by_path.value(normalizePath(image_path))
                                     .value(QStringLiteral("image_uuid"))
                                     .toString()
                                     .trimmed();
        if (image_id.isEmpty() || seen_ids.contains(image_id))
        {
            if (allResolved)
            {
                *allResolved = false;
            }
            continue;
        }
        try
        {
            result.emplace_back(image_id.toStdString());
            seen_ids.insert(image_id);
        }
        catch (...)
        {
            if (allResolved)
            {
                *allResolved = false;
            }
        }
    }
    if (allResolved && result.size() != static_cast<std::size_t>(images.size()))
    {
        *allResolved = false;
    }
    return result;
}

QString validMaskPath(const QString &depthPng)
{
    const QFileInfo info(depthPng);
    return QDir(info.absolutePath()).filePath(
        QStringLiteral("%1_mask.png").arg(info.completeBaseName()));
}

QJsonObject depthRecordFromArtifact(const QJsonObject &artifact,
                                    const PointCloudWorkflowContext &context)
{
    const QString depth_png = artifact.value(QStringLiteral("depth_png")).toString();
    if (depth_png.isEmpty())
    {
        return {};
    }

    QJsonObject record = xjw::gui::project::makeDepthResultRecord(
        utcNowIso(),
        depth_png,
        artifact.value(QStringLiteral("grid_width")).toInt(),
        artifact.value(QStringLiteral("grid_height")).toInt(),
        context.sparseCloudPath,
        artifact.value(QStringLiteral("ref_image")).toString());
    for (auto it = artifact.constBegin(); it != artifact.constEnd(); ++it)
    {
        record[it.key()] = it.value();
    }
    // mvs_manifest.json is mutable workspace bookkeeping. Persisting it as a
    // project resource makes background project snapshots read the file while
    // depth workers atomically replace it, which is a sharing violation on
    // Windows. All reusable state is already copied into this depth record.
    record.remove(QStringLiteral("manifest_path"));
    if (record.value(QStringLiteral("raw_depth_path")).toString().isEmpty())
    {
        record[QStringLiteral("raw_depth_path")] =
            xjw::core::project::rawDepthStoragePath(depth_png);
    }
    if (record.value(QStringLiteral("raw_confidence_path")).toString().isEmpty())
    {
        record[QStringLiteral("raw_confidence_path")] =
            xjw::core::project::rawConfidenceStoragePath(depth_png);
    }
    if (record.value(QStringLiteral("valid_mask_path")).toString().isEmpty())
    {
        record[QStringLiteral("valid_mask_path")] = validMaskPath(depth_png);
    }
    record[QStringLiteral("mvs_output_dir")] = context.outputDir;
    record[QStringLiteral("batch_frame_count")] = context.selectedImages.size();
    record[QStringLiteral("project_input_signature")] = context.projectInputSignature;
    record[QStringLiteral("reconstruction_generation_id")] =
        context.reconstructionGenerationId;
    record[QStringLiteral("quality_profile")] = context.request.qualityProfile;
    record[QStringLiteral("mvs_backend_requested")] =
        patchMatchBackendText(context.request.patchMatchBackend);
    record[QStringLiteral("mvs_backend_request_applied")] = true;
    return record;
}

int fusionNeighborCount(const DenseGenerationSettings &request, int frameCount)
{
    if (frameCount <= 1)
    {
        return 0;
    }
    return std::min(frameCount - 1,
                    std::clamp(std::max(8, request.minViews * 2), 8, 16));
}

} // namespace

ProjectPointCloudWorkflowController::ProjectPointCloudWorkflowController(xjw::gui::project::ProjectSession* session,
                                                                         ProjectUiMessageAdapter* messages,
                                                                         QObject* parent)
    : QObject(parent), _session(session), _messages(messages)
{
}

ProjectPointCloudWorkflowController::~ProjectPointCloudWorkflowController()
{
    // runner 可能仍持有共享取消标志；析构前置位可阻止项目关闭后继续读取深度帧。
    cancelActiveTask();
    waitForActiveTask();
}

bool ProjectPointCloudWorkflowController::startCreatePointCloudAsync(
    const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& task_context)
{
    return startWorkflow(settings, false, task_context);
}

bool ProjectPointCloudWorkflowController::startDepthMapsOnlyAsync(
    const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& task_context)
{
    return startWorkflow(settings, true, task_context);
}

bool ProjectPointCloudWorkflowController::startWorkflow(const QJsonObject& settings,
                                                        bool depth_maps_only,
                                                        const xjw::gui::project::ProjectTaskContext& task_context)
{
    const QString dialog_title = workflowDialogTitle(depth_maps_only);
    if (!_session || !_session->hasProject() || !_session->isCurrent(task_context.session) ||
        !task_context.cancelFlag || task_context.cancelFlag->load(std::memory_order_relaxed))
    {
        failTask(QStringLiteral("请先打开项目，并完成正式空中三角测量。"),
                 dialog_title);
        return false;
    }
    if (_isRunning)
    {
        if (_messages)
        {
            _messages->information(nullptr,
                                   dialog_title,
                                   depth_maps_only
                                       ? QStringLiteral("已有深度图或点云任务正在运行，请等待或先取消当前任务。")
                                       : QStringLiteral("已有点云任务正在运行，请等待或先取消当前任务。"));
        }
        return false;
    }

    const QJsonObject metadata = _session->metadata();
    const int at_index = xjw::core::project::findLatestProductionAtResultIndex(metadata);
    const QJsonArray at_results =
        metadata.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (at_index < 0 || at_index >= at_results.size())
    {
        failTask(QStringLiteral("未找到通过质量门控的正式 SfM/BA 稀疏点云结果。"),
                 dialog_title);
        return false;
    }

    const QJsonObject at_record = at_results.at(at_index).toObject();
    if (!xjw::gui::project::isStandardMvsCompatibleSparseResult(at_record))
    {
        failTask(xjw::gui::project::standardMvsBlockingReason(at_record),
                 dialog_title);
        return false;
    }

    QJsonObject effective_settings = settings;
    QString scene_profile_source;
    const QString scene_profile = resolvedSceneProfile(
        settings, &scene_profile_source);
    effective_settings[QStringLiteral("sceneProfile")] = scene_profile;

    auto context = std::make_shared<PointCloudWorkflowContext>();
    context->task = task_context;
    context->session = task_context.session;
    context->settings = effective_settings;
    context->request = xjw::core::project::denseGenerationSettingsFromJson(
        effective_settings);
    context->atIndex = at_index;
    context->sparseCloudPath = sparseCloudPathFromRecord(at_record);
    context->sparsePointSidecarPath = sparsePointSidecarPathFromRecord(at_record);
    QString selected_image_error;
    context->selectedImages =
        selectedImagesFromRecord(at_record, metadata, &selected_image_error);
    if (!selected_image_error.isEmpty())
    {
        failTask(selected_image_error, dialog_title);
        return false;
    }
    context->projectInputSignature =
        xjw::gui::project::projectDepthInputSignature(metadata, at_index);
    context->reconstructionGenerationId =
        at_record.value(QStringLiteral("reconstruction_generation_id")).toString();
    context->reuseDepthMaps = settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
    context->saveAfterEachStep =
        settings.value(QStringLiteral("saveAfterEachStep")).toBool(false);
    context->calculateColors =
        settings.value(QStringLiteral("calculatePointColors")).toBool(true);
    context->replaceDefaultPointCloud =
        settings.value(QStringLiteral("replaceDefaultPointCloud")).toBool(false);
    context->depthMapsOnly = depth_maps_only;
    LOG_INFO(QStringLiteral(
        "[MVS] 场景策略：profile=%1 source=%2")
                 .arg(scene_profile, scene_profile_source));

    if (!QFileInfo::exists(context->sparseCloudPath))
    {
        failTask(QStringLiteral("正式空三稀疏点云不存在：%1")
                     .arg(context->sparseCloudPath),
                 dialog_title);
        return false;
    }
    if (!QFileInfo::exists(context->sparsePointSidecarPath))
    {
        failTask(QStringLiteral("正式空三逐点观测 sidecar 不存在：%1")
                     .arg(context->sparsePointSidecarPath),
                 dialog_title);
        return false;
    }
    if (context->selectedImages.size() < 7)
    {
        failTask(QStringLiteral("recovered 深度要求至少 7 张注册影像，当前只有 %1 张。")
                     .arg(context->selectedImages.size()),
                 dialog_title);
        return false;
    }

    bool all_image_ids = false;
    const std::vector<xjw::camera_core::ImageId> image_ids =
        imageIdsForImages(metadata, context->selectedImages, &all_image_ids);
    PinholeStatesByImageId cameras;
    QString camera_error;
    if (!all_image_ids || !loadMvsCameras(metadata, image_ids, &cameras, &camera_error))
    {
        failTask(camera_error.isEmpty()
                     ? QStringLiteral("注册影像缺少稳定 ImageId，无法准备 MVS。")
                     : camera_error,
                 dialog_title);
        return false;
    }
    QString view_error;
    if (!buildMvsViews(context->session.projectPath,
                       context->selectedImages,
                       image_ids,
                       cameras,
                       &context->views,
                       &view_error))
    {
        failTask(view_error.isEmpty()
                     ? (all_image_ids ? QStringLiteral("部分注册影像缺少有效相机参数。")
                                      : QStringLiteral("注册影像缺少稳定 ImageId，无法准备 MVS。"))
                     : view_error,
                 dialog_title);
        return false;
    }

    const auto compatibility =
        xjw::gui::project::assessStoredDepthBatchCompatibility(
            metadata,
            QString(),
            at_index,
            context->request.sceneProfile);
    const auto stored = xjw::core::project::collectLatestStoredDepthFrames(metadata);
    const QString requested_backend = patchMatchBackendText(
        context->request.patchMatchBackend);
    const QString stored_backend = mvsBackendFromStoredFrames(stored.frames);
    const bool stored_backend_matches_request =
        xjw::gui::project::canReuseStoredMvsBackend(
            requested_backend, stored_backend);
    const bool can_reuse = context->reuseDepthMaps && compatibility.compatible &&
        stored_backend_matches_request;
    context->reusedDepthMaps = can_reuse;
    if (context->reuseDepthMaps && !compatibility.compatible &&
        compatibility.frameCount > 0)
    {
        LOG_INFO(QStringLiteral(
            "[MVS] 已有深度图批次不兼容：%1 本次将重新估计深度图。")
                     .arg(compatibility.reason));
    }
    if (context->reuseDepthMaps && compatibility.compatible &&
        !stored_backend_matches_request)
    {
        LOG_INFO(QStringLiteral(
            "[MVS] 已有深度图后端=%1，与当前请求=%2 不兼容；本次重新估计深度图")
                     .arg(stored_backend, requested_backend));
    }
    const bool runs_point_processing = !context->depthMapsOnly || !can_reuse;
    if (runs_point_processing)
    {
        const QString unavailable_reason =
            xjw::core::project::processingDeviceUnavailableReason(
                context->request.processingDevice);
        if (!unavailable_reason.isEmpty())
        {
            failTask(unavailable_reason, dialog_title);
            return false;
        }
    }
    if (can_reuse)
    {
        context->outputDir = stored.batchDir;
    }
    else
    {
        context->outputDir = xjw::gui::project::resolveProjectOutputDir(
            context->session.projectPath,
            context->request.outputDir,
            QStringLiteral("mvs_output"));
    }
    if (context->outputDir.isEmpty() || !QDir().mkpath(context->outputDir))
    {
        failTask(QStringLiteral("无法创建 MVS 输出目录：%1").arg(context->outputDir),
                 dialog_title);
        return false;
    }

    _isRunning = true;
    _activeContext = task_context;
    emit pointCloudProgressChanged(
        can_reuse ? QStringLiteral("正在复用兼容深度图") : QStringLiteral("正在准备深度图估计"), 0);
    if (settleIfContextNotLive(task_context))
    {
        return false;
    }

    if (can_reuse)
    {
        if (context->depthMapsOnly)
        {
            emit depthMapBatchReady(context->outputDir, static_cast<int>(stored.frames.size()));
            if (settleIfContextNotLive(task_context))
            {
                return false;
            }
            finishTask(true);
        }
        else
        {
            startFusion(context);
        }
    }
    else
    {
        // 未复用时明确清理当前批次，保证“重新计算”不会被 manifest 静默续跑。
        QString cleanup_error;
        const bool workspace_cleared =
            clearDepthWorkspace(_session.get(), task_context, context->outputDir, &cleanup_error);
        if (settleIfContextNotLive(task_context))
        {
            return false;
        }
        if (!workspace_cleared)
        {
            failTask(cleanup_error.isEmpty() ? QStringLiteral("无法更新深度图工作区元数据。") : cleanup_error,
                     dialog_title);
            return false;
        }
        startDepthEstimation(context);
    }
    return true;
}

void ProjectPointCloudWorkflowController::startDepthEstimation(
    const std::shared_ptr<PointCloudWorkflowContext> &context)
{
    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [context]()
        {
            DepthEstimationPreparationResult result;
            result.pointCloudInput = xjw::core::project::preparePointCloudInput(
                context->sparseCloudPath,
                context->views,
                context->request.processingDevice,
                context->sparsePointSidecarPath);
            if (result.pointCloudInput.ok)
            {
                result.sourcePairQuality = xjw::core::project::loadMvsSourcePairQualities(
                    xjw::common::project::ProjectIO::imageMatchOutputDir(context->session.projectPath),
                    context->selectedImages);
            }
            return result;
        },
        [context](ProjectPointCloudWorkflowController* self,
                  xjw::gui::tasks::TaskOutcome<DepthEstimationPreparationResult> outcome)
        {
            if (self->settleIfContextNotLive(context->task))
            {
                return;
            }
            if (!outcome.succeeded())
            {
                self->failTask(outcome.errorMessage.isEmpty() ? QStringLiteral("稀疏点云预处理失败。")
                                                              : outcome.errorMessage,
                               workflowDialogTitle(context->depthMapsOnly));
                return;
            }
            auto preparation = std::move(*outcome.value);
            auto prepared = std::move(preparation.pointCloudInput);
            if (!prepared.ok)
            {
                self->failTask(prepared.errorMessage,
                               workflowDialogTitle(context->depthMapsOnly));
                return;
            }

            auto* generator = new xjw::gui::tasks::DepthMapTask(self);
            xjw::mvs::DepthGenConfig config =
                xjw::core::project::buildDepthGenConfig(
                    context->request,
                    static_cast<int>(context->views.size()));
            config.inputSignature =
                context->projectInputSignature.toUtf8().toStdString();
            config.runFusion = false;
            config.saveIntermediateDepthMaps = true;
            config.intermediateDir =
                xjw::common::io::toUtf8Path(context->outputDir);
            const auto &pair_quality = preparation.sourcePairQuality;
            LOG_INFO(QStringLiteral(
                         "[MVS] 源像对审计：files=%1 pairs=%2 verified=%3 "
                         "failed=%4 missing_stats=%5 incompatible=%6")
                         .arg(pair_quality.matchFileCount)
                         .arg(pair_quality.catalogPairCount)
                         .arg(pair_quality.verifiedPairCount)
                         .arg(pair_quality.failedPairCount)
                         .arg(pair_quality.missingStatisticsPairCount)
                         .arg(pair_quality.incompatibleVariantCount));
            if (pair_quality.matchFileCount <= 0)
            {
                LOG_WARN(QStringLiteral(
                    "[MVS] 项目匹配目录没有可审计的 `.pimatch` 分片，"
                    "源视图将回退到稀疏轨迹几何。"));
            }
            else if (pair_quality.verifiedPairCount <= 0)
            {
                LOG_WARN(QStringLiteral(
                    "[MVS] 当前影像集合没有通过几何验证的 `.pimatch` 像对，"
                    "源视图将回退到稀疏轨迹几何。"));
            }
            xjw::core::project::applyMvsSourcePairQualities(
                &config, std::move(preparation.sourcePairQuality));
            generator->setViews(context->views);
            generator->setSparseCloud(prepared.cloud);
            generator->setConfig(config);
            generator->setOutputDir(
                xjw::common::io::toUtf8Path(context->outputDir));
            self->_activeGenerator = generator;

            connect(generator,
                    &xjw::gui::tasks::DepthMapTask::progressChanged,
                    self,
                    [self, context](const QString& stage, float ratio)
                    {
                        if (self->acceptsContext(context->task))
                        {
                            emit self->pointCloudProgressChanged(stage,
                                                                 std::clamp(static_cast<int>(ratio * 60.0f), 0, 60));
                        }
                    });
            connect(generator,
                    &xjw::gui::tasks::DepthMapTask::errorOccurred,
                    self,
                    [context](const QString& message)
                    {
                        if (context->depthError.isEmpty())
                        {
                            context->depthError = message;
                        }
                    });
            connect(generator,
                    &xjw::gui::tasks::DepthMapTask::depthMapArtifactSaved,
                    self,
                    [self, context](const QJsonObject& artifact)
                    {
                        if (!self->acceptsContext(context->task))
                        {
                            return;
                        }
                        const QJsonObject record = depthRecordFromArtifact(artifact, *context);
                        if (!record.isEmpty())
                        {
                            QString write_error;
                            const bool published =
                                self->_session->upsertResultRecordByPath(context->task.session,
                                                                         QStringLiteral("depth_map_results"),
                                                                         QStringLiteral("depth_png"),
                                                                         record,
                                                                         true,
                                                                         &write_error);
                            if (!self->acceptsContext(context->task))
                            {
                                self->cancelActiveTask();
                            }
                            else if (!published)
                            {
                                context->depthError =
                                    write_error.isEmpty() ? QStringLiteral("深度图成果写入项目失败。") : write_error;
                                self->cancelActiveTask();
                            }
                        }
                    });
            connect(generator,
                    &xjw::gui::tasks::DepthMapTask::finished,
                    self,
                    [self, generator, context](bool success)
                    {
                        if (self->_activeGenerator == generator)
                        {
                            self->_activeGenerator.clear();
                        }
                        generator->deleteLater();
                        if (!self->acceptsContext(context->task, true))
                        {
                            self->finishTask(false, false);
                            return;
                        }
                        if (context->task.cancelFlag->load(std::memory_order_relaxed))
                        {
                            if (context->depthError.isEmpty())
                            {
                                self->finishTask(false);
                            }
                            else
                            {
                                self->failTask(context->depthError, workflowDialogTitle(context->depthMapsOnly));
                            }
                            return;
                        }
                        if (!success)
                        {
                            self->failTask(context->depthError.isEmpty() ? QStringLiteral("深度图估计失败或已取消。")
                                                                         : context->depthError,
                                           workflowDialogTitle(context->depthMapsOnly));
                            return;
                        }
                        if (context->saveAfterEachStep)
                        {
                            self->_session->requestProjectSave(context->task.session);
                            if (self->settleIfContextNotLive(context->task))
                            {
                                return;
                            }
                        }
                        if (context->depthMapsOnly)
                        {
                            const auto stored = xjw::core::project::collectStoredDepthFramesForDirectory(
                                self->_session->metadata(), context->outputDir);
                            if (!stored.status.ok || stored.frames.size() < 2)
                            {
                                self->failTask(stored.status.ok ? QStringLiteral("自动估计后可用深度图不足 2 帧。")
                                                                : stored.status.errorMessage,
                                               QStringLiteral("生成模型"));
                                return;
                            }
                            emit self->depthMapBatchReady(context->outputDir, static_cast<int>(stored.frames.size()));
                            if (self->settleIfContextNotLive(context->task))
                            {
                                return;
                            }
                            self->finishTask(true);
                            return;
                        }
                        self->startFusion(context);
                    });

            emit self->pointCloudProgressChanged(
                QStringLiteral("正在估计多视深度图"), 1);
            generator->start();
        }));
}

void ProjectPointCloudWorkflowController::startFusion(
    const std::shared_ptr<PointCloudWorkflowContext> &context)
{
    if (settleIfContextNotLive(context->task))
    {
        return;
    }

    const auto discovered = xjw::core::project::collectStoredDepthFramesForDirectory(
        _session->metadata(), context->outputDir);
    if (!discovered.status.ok || discovered.frames.size() < 2)
    {
        failTask(discovered.status.ok
                     ? QStringLiteral("可融合深度图不足 2 帧。")
                     : discovered.status.errorMessage);
        return;
    }

    const auto stored =
        xjw::core::project::selectFusionEligibleStoredDepthFrames(discovered);
    if (!stored.status.ok)
    {
        failTask(stored.status.errorMessage);
        return;
    }

    QStringList frame_images;
    frame_images.reserve(static_cast<int>(stored.frames.size()));
    for (const auto &frame : stored.frames)
    {
        frame_images.push_back(frame.refImage);
    }
    bool all_image_ids = false;
    const std::vector<xjw::camera_core::ImageId> frame_image_ids =
        imageIdsForImages(_session->metadata(), frame_images, &all_image_ids);
    PinholeStatesByImageId cameras;
    QString camera_error;
    if (!all_image_ids ||
        !loadMvsCameras(_session->metadata(), frame_image_ids, &cameras, &camera_error))
    {
        failTask(camera_error.isEmpty()
                     ? QStringLiteral("深度图对应影像缺少稳定 ImageId，无法绑定 MVS 相机参数。")
                     : camera_error);
        return;
    }
    if (frame_image_ids.size() != stored.frames.size())
    {
        failTask(all_image_ids
                     ? QStringLiteral("部分深度图对应影像缺少当前空三相机参数。")
                     : QStringLiteral("深度图对应影像缺少稳定 ImageId，无法绑定当前空三相机参数。"));
        return;
    }

    emit pointCloudProgressChanged(
        QStringLiteral("正在加载并融合深度图（点云后端请求：%1）")
            .arg(xjw::core::project::processingDeviceId(
                context->request.processingDevice)),
        65);
    const auto cancel_flag = context->task.cancelFlag;
    QPointer<ProjectPointCloudWorkflowController> self(this);
    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [self, context, stored, frame_image_ids, cameras, cancel_flag]() -> PointCloudTaskResult
        {
            PointCloudTaskResult task;
            if (!cancel_flag || cancel_flag->load(std::memory_order_relaxed))
            {
                task.cancelled = true;
                return task;
            }

            const int frame_count = static_cast<int>(stored.frames.size());
            const xjw::mvs::FusionConfig fusion_config =
                xjw::core::project::buildDepthGenConfig(
                    context->request, frame_count).fusion;
            const xjw::mvs::FusionFrameLoader loader =
                [stored, frame_image_ids, cameras, fusion_config, context](
                    int index, xjw::mvs::FusionFrameInput* frame, std::string* error_message)
            {
                xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
                if (index < 0 || index >= static_cast<int>(frame_image_ids.size()) ||
                    !cameraForImage(cameras, frame_image_ids[static_cast<std::size_t>(index)], &camera))
                {
                    if (error_message)
                    {
                        *error_message = "Missing camera for stored depth frame";
                    }
                    return false;
                }
                auto loaded = xjw::core::project::buildStoredFusionFrame(stored.frames[static_cast<std::size_t>(index)],
                                                                         camera,
                                                                         fusion_config,
                                                                         static_cast<int>(stored.frames.size()),
                                                                         context->request.fusionMaxImageDim);
                if (!loaded.status.ok)
                {
                    if (error_message)
                    {
                        *error_message = xjw::common::io::toUtf8Path(loaded.status.errorMessage);
                    }
                    return false;
                }
                *frame = std::move(loaded.frame);
                frame->viewIndex = index;
                frame->sourceImageIndices = xjw::core::project::storedFusionSourceIndices(stored.frames, index);
                return true;
            };

            xjw::mvs::StreamingDepthFusionConfig config;
            config.minConsistentViews = context->request.minConsistentViews;
            config.depthConsistency = context->request.depthConsistency;
            config.workerCount = std::max(1, context->request.threads);
            config.computeBackend = context->request.patchMatchBackend;
            config.neighborCount = fusionNeighborCount(context->request, frame_count);
            config.cacheFrameLimit = 32;
            config.useColor = context->calculateColors;
            config.cancelFlag = cancel_flag;

            xjw::mvs::StreamingDepthFusionResult fused;
            std::string fusion_error;
            const bool fused_ok = xjw::mvs::fuseDepthMapsStreaming(
                frame_count,
                config,
                loader,
                &fused,
                &fusion_error,
                [self, context](const std::string &stage, int percent)
                {
                    if (!self)
                    {
                        return;
                    }
                    const QString stage_text = QString::fromUtf8(stage.c_str());
                    const int workflow_percent = 65 +
                        std::clamp(percent, 0, 100) * 30 / 100;
                    QMetaObject::invokeMethod(
                        self.data(),
                        [self, context, stage_text, workflow_percent]()
                        {
                            if (self && self->acceptsContext(context->task))
                            {
                                emit self->pointCloudProgressChanged(
                                    stage_text,
                                    workflow_percent);
                            }
                        },
                        Qt::QueuedConnection);
                });
            if (!fused_ok)
            {
                task.cancelled = cancel_flag->load(std::memory_order_relaxed);
                task.errorMessage = fusion_error.empty()
                    ? QStringLiteral("深度图融合没有生成有效点云")
                    : QString::fromUtf8(fusion_error.c_str());
                return task;
            }

            std::vector<xjw::mvs::DensePoint> cloud;
            cloud.reserve(fused.points.size());
            for (const xjw::mvs::FusedPoint &point : fused.points)
            {
                xjw::mvs::DensePoint dense;
                dense.x = point.x;
                dense.y = point.y;
                dense.z = point.z;
                dense.r = context->calculateColors ? point.r : 180;
                dense.g = context->calculateColors ? point.g : 180;
                dense.b = context->calculateColors ? point.b : 180;
                cloud.push_back(dense);
            }

            const std::size_t before_processing = cloud.size();
            plapoint::ProcessingReport processing_report;
            bool processing_skipped = false;
            if (cloud.size() >= 31)
            {
                cloud = xjw::mvs::DenseCloudBuilder::statisticalOutlierRemoval(
                    cloud,
                    30,
                    2.0f,
                    context->request.processingDevice,
                    &processing_report);
                if (cloud.empty() && before_processing > 0)
                {
                    task.errorMessage = QStringLiteral("点云去噪后没有剩余有效点");
                    return task;
                }
            }
            else
            {
                processing_report.requestedDevice = context->request.processingDevice;
                processing_skipped = true;
            }

            task.pointCloudPath = QDir(context->outputDir).filePath(
                QStringLiteral("dense_cloud.ply"));
            std::string save_error;
            if (!xjw::mvs::DenseCloudBuilder::savePLY(
                    xjw::common::io::toUtf8Path(task.pointCloudPath),
                    cloud,
                    &save_error))
            {
                task.errorMessage = QStringLiteral("保存点云失败：%1")
                                        .arg(QString::fromUtf8(save_error.c_str()));
                return task;
            }

            task.pointCount = static_cast<int>(std::min<std::size_t>(
                cloud.size(),
                static_cast<std::size_t>(std::numeric_limits<int>::max())));
            task.record = xjw::gui::project::makeDenseResultRecord(
                utcNowIso(),
                task.pointCloudPath,
                task.pointCount,
                context->sparseCloudPath);
            task.record[QStringLiteral("source_depth_map_dir")] = stored.batchDir;
            task.record[QStringLiteral("source_depth_map_count")] = frame_count;
            task.record[QStringLiteral("source_project_input_signature")] =
                context->projectInputSignature;
            task.record[QStringLiteral("source_reconstruction_generation_id")] =
                context->reconstructionGenerationId;
            task.record[QStringLiteral("fusion_pipeline_version")] =
                xjw::gui::project::kDenseFusionPipelineVersion;
            task.record[QStringLiteral("quality_profile")] = context->request.qualityProfile;
            task.record[QStringLiteral("depth_filter_mode")] =
                context->request.depthFilterMode;
            task.record[QStringLiteral("scene_profile")] = sceneProfileFromStoredFrames(stored.frames);
            task.record[QStringLiteral("mvs_backend_requested")] =
                patchMatchBackendText(context->request.patchMatchBackend);
            task.record[QStringLiteral("mvs_backend_actual")] = mvsBackendFromStoredFrames(stored.frames);
            task.record[QStringLiteral("mvs_backend_selected_in_dialog")] =
                patchMatchBackendText(context->request.patchMatchBackend);
            task.record[QStringLiteral("mvs_backend_request_applied")] = !context->reusedDepthMaps;
            task.record[QStringLiteral("fusion_unprojection_backend_requested")] = QString::fromLatin1(
                xjw::mvs::denseCloudComputeBackendId(fused.unprojectionExecution.requestedBackend));
            task.record[QStringLiteral("fusion_unprojection_backend_actual")] =
                fused.mixedUnprojectionBackends
                    ? QStringLiteral("mixed")
                    : QString::fromLatin1(
                          xjw::mvs::denseCloudComputeBackendId(fused.unprojectionExecution.actualBackend));
            task.record[QStringLiteral("fusion_unprojection_device_index")] =
                fused.unprojectionExecution.deviceIndex;
            task.record[QStringLiteral("fusion_unprojection_device_name")] =
                QString::fromUtf8(fused.unprojectionExecution.deviceName.c_str());
            task.record[QStringLiteral("fusion_unprojection_backend_fallback")] =
                fused.unprojectionExecution.fallbackUsed;
            task.record[QStringLiteral("fusion_unprojection_backend_fallback_reason")] =
                QString::fromUtf8(fused.unprojectionExecution.fallbackReason.c_str());
            task.record[QStringLiteral("fusion_consistency_backend")] = QStringLiteral("cpu");
            task.record[QStringLiteral("depth_maps_reused")] = context->reusedDepthMaps;
            task.record[QStringLiteral("calculate_point_colors")] = context->calculateColors;
            task.record[QStringLiteral("point_confidence_available")] = false;
            task.record[QStringLiteral("point_cloud_processing")] = QJsonObject{
                {QStringLiteral("stage"), QStringLiteral("statistical_outlier_removal")},
                {QStringLiteral("requested"),
                 xjw::core::project::processingDeviceId(processing_report.requestedDevice)},
                {QStringLiteral("actual"),
                 processing_skipped ? QStringLiteral("skipped")
                                    : xjw::core::project::processingDeviceId(processing_report.actualDevice)},
                {QStringLiteral("used_fallback"), processing_report.usedFallback},
                {QStringLiteral("fallback_reason"), QString::fromStdString(processing_report.fallbackReason)},
                {QStringLiteral("input_points"), static_cast<qint64>(before_processing)},
                {QStringLiteral("output_points"), static_cast<qint64>(cloud.size())}};
            if (!stored.frames.empty())
            {
                task.record[QStringLiteral("source_depth_config_hash")] = stored.frames.front().configHash;
            }
            task.ok = true;
            return task;
        },
        [context](ProjectPointCloudWorkflowController* manager,
                  xjw::gui::tasks::TaskOutcome<PointCloudTaskResult> outcome)
        {
            if (manager->settleIfContextNotLive(context->task))
            {
                return;
            }
            if (!outcome.succeeded())
            {
                manager->failTask(outcome.errorMessage.isEmpty()
                                      ? QStringLiteral("点云融合发生未知异常")
                                      : QStringLiteral("点云融合异常：%1").arg(outcome.errorMessage));
                return;
            }

            PointCloudTaskResult result = std::move(*outcome.value);
            if (!result.ok)
            {
                if (result.cancelled)
                {
                    manager->finishTask(false);
                }
                else
                {
                    manager->failTask(result.errorMessage);
                }
                return;
            }

            const QJsonObject processing = result.record.value(QStringLiteral("point_cloud_processing")).toObject();
            emit manager->pointCloudProgressChanged(
                QStringLiteral("点云去噪完成：%1 → %2 点，请求 %3，实际 %4%5")
                    .arg(processing.value(QStringLiteral("input_points")).toInteger())
                    .arg(processing.value(QStringLiteral("output_points")).toInteger())
                    .arg(processing.value(QStringLiteral("requested")).toString())
                    .arg(processing.value(QStringLiteral("actual")).toString())
                    .arg(processing.value(QStringLiteral("used_fallback")).toBool() ? QStringLiteral("（已回退）")
                                                                                    : QString()),
                97);
            if (manager->settleIfContextNotLive(context->task))
            {
                return;
            }

            QString write_error;
            bool published = false;
            if (context->replaceDefaultPointCloud)
            {
                published = manager->_session->replaceResultRecordWithLatest(
                    context->task.session, QStringLiteral("dense_cloud_results"), result.record, true, &write_error);
            }
            else
            {
                published = manager->_session->upsertResultRecordByPath(context->task.session,
                                                                        QStringLiteral("dense_cloud_results"),
                                                                        QStringLiteral("dense_cloud_xyz"),
                                                                        result.record,
                                                                        true,
                                                                        &write_error);
            }
            if (manager->settleIfContextNotLive(context->task))
            {
                return;
            }
            if (!published)
            {
                manager->failTask(write_error.isEmpty() ? QStringLiteral("稠密点云成果写入项目失败。") : write_error);
                return;
            }
            if (context->saveAfterEachStep)
            {
                manager->_session->requestProjectSave(context->task.session);
                if (manager->settleIfContextNotLive(context->task))
                {
                    return;
                }
            }
            emit manager->pointCloudResultReady(result.pointCloudPath, result.pointCount);
            if (manager->settleIfContextNotLive(context->task))
            {
                return;
            }
            if (manager->_messages)
            {
                manager->_messages->information(nullptr,
                                                QStringLiteral("创建点云"),
                                                QStringLiteral("点云已生成。\n点数: %1\n路径: %2")
                                                    .arg(result.pointCount)
                                                    .arg(QDir::toNativeSeparators(result.pointCloudPath)));
            }
            if (manager->settleIfContextNotLive(context->task))
            {
                return;
            }
            manager->finishTask(true);
        }));
}

void ProjectPointCloudWorkflowController::cancelActiveTask()
{
    if (_activeContext.cancelFlag)
    {
        _activeContext.cancelFlag->store(true, std::memory_order_relaxed);
    }
    if (auto* generator = qobject_cast<xjw::gui::tasks::DepthMapTask*>(_activeGenerator.data()))
    {
        generator->requestCancel();
    }
}

void ProjectPointCloudWorkflowController::waitForActiveTask()
{
    cancelActiveTask();
    if (auto* generator = qobject_cast<xjw::gui::tasks::DepthMapTask*>(_activeGenerator.data()))
    {
        _activeGenerator.clear();
        delete generator;
    }
    for (QFuture<void>& future : _futures)
    {
        if (future.isRunning())
        {
            future.waitForFinished();
        }
    }
    _futures.clear();
}

bool ProjectPointCloudWorkflowController::isRunning() const
{
    return _isRunning;
}

bool ProjectPointCloudWorkflowController::hasPendingWork() const noexcept
{
    if (_activeGenerator)
    {
        return true;
    }
    return std::any_of(_futures.cbegin(),
                       _futures.cend(),
                       [](const QFuture<void>& future) { return future.isValid() && !future.isFinished(); });
}

bool ProjectPointCloudWorkflowController::acceptsContext(const xjw::gui::project::ProjectTaskContext& context,
                                                         bool allow_cancelled) const
{
    return _isRunning && _session && _session->isCurrent(context.session) && context.cancelFlag &&
           (allow_cancelled || !context.cancelFlag->load(std::memory_order_relaxed)) &&
           _activeContext.taskId == context.taskId && _activeContext.cancelFlag == context.cancelFlag;
}

bool ProjectPointCloudWorkflowController::settleIfContextNotLive(const xjw::gui::project::ProjectTaskContext& context)
{
    if (!acceptsContext(context, true))
    {
        finishTask(false, false);
        return true;
    }
    if (context.cancelFlag->load(std::memory_order_relaxed))
    {
        finishTask(false);
        return true;
    }
    return false;
}

void ProjectPointCloudWorkflowController::finishTask(bool success, bool emit_terminal)
{
    _activeGenerator.clear();
    _activeContext = {};
    _isRunning = false;
    pruneFinishedFutures();
    if (emit_terminal)
    {
        emit pointCloudProgressFinished(success);
    }
}

void ProjectPointCloudWorkflowController::failTask(const QString &message,
                                                 const QString &title)
{
    const QString effective_message = message.trimmed().isEmpty()
        ? QStringLiteral("点云任务失败，未返回具体错误。")
        : message;
    if (_messages)
    {
        _messages->warning(nullptr, title, effective_message);
    }
    if (_isRunning)
    {
        finishTask(false);
    }
}

void ProjectPointCloudWorkflowController::pruneFinishedFutures()
{
    _futures.erase(std::remove_if(_futures.begin(),
                                  _futures.end(),
                                  [](const QFuture<void> &future) { return future.isFinished(); }),
                   _futures.end());
}

void ProjectPointCloudWorkflowController::trackFuture(QFuture<void> future)
{
    pruneFinishedFutures();
    if (future.isValid())
    {
        _futures.push_back(std::move(future));
    }
}

void ProjectPointCloudWorkflowController::trackFutureForTesting(QFuture<void> future)
{
    trackFuture(std::move(future));
}
