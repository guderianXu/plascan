#include "ProjectPointCloudWorkflowController.h"

#include "ProjectManager.h"
#include "project/ProjectSessionModel.h"
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
#include "DepthMapGenerator.h"
#include "StreamingDepthFusionService.h"
#include "io/PathIO.h"
#include "Logger.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QMessageBox>
#include <QMetaObject>
#include <QPointer>

#include <algorithm>
#include <limits>
#include <utility>

using xjw::common::project::normalizePath;
using xjw::core::project::DenseGenerationSettings;

struct PointCloudWorkflowContext
{
    xjw::gui::project::ProjectSessionContext session;
    DenseGenerationSettings request;
    QJsonObject settings;
    QString sparseCloudPath;
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

QStringList selectedImagesFromRecord(const QJsonObject &record)
{
    QStringList images;
    for (const QJsonValue &value : record.value(QStringLiteral("selected_images")).toArray())
    {
        const QString path = QDir::cleanPath(value.toString().trimmed());
        if (!path.isEmpty())
        {
            images.push_back(path);
        }
    }
    return images;
}

bool cameraForImage(const QMap<QString, xjw::FramePinholeCamera> &cameras,
                    const QString &imagePath,
                    xjw::FramePinholeCamera *camera)
{
    if (!camera)
    {
        return false;
    }
    const auto it = cameras.constFind(normalizePath(imagePath));
    if (it == cameras.constEnd())
    {
        return false;
    }
    *camera = it.value();
    return true;
}

bool buildMvsViews(const QString &projectPath,
                   const QStringList &images,
                   const QMap<QString, xjw::FramePinholeCamera> &cameras,
                   std::vector<xjw::mvs::CameraView> *views,
                   QString *errorMessage)
{
    if (!views)
    {
        return false;
    }
    views->clear();
    views->reserve(static_cast<std::size_t>(images.size()));
    for (const QString &image_path : images)
    {
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
        if (!cameraForImage(cameras, image_path, &view.camera))
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

void clearDepthWorkspace(ProjectData *projectData, const QString &outputDir)
{
    const QString clean_output = QDir::cleanPath(outputDir);
    QDir directory(clean_output);
    const QFileInfoList artifacts = directory.entryInfoList(
        QStringList{QStringLiteral("depth_*"), QStringLiteral("mvs_manifest.json")},
        QDir::Files | QDir::Hidden);
    for (const QFileInfo &artifact : artifacts)
    {
        QFile::remove(artifact.absoluteFilePath());
    }

    if (!projectData)
    {
        return;
    }
    QJsonObject metadata = projectData->metadata();
    QJsonArray retained;
    for (const QJsonValue &value : metadata.value(QStringLiteral("depth_map_results")).toArray())
    {
        const QJsonObject record = value.toObject();
        if (QDir::cleanPath(artifactDirectory(record)).compare(
                clean_output, Qt::CaseInsensitive) != 0)
        {
            retained.append(record);
        }
    }
    metadata[QStringLiteral("depth_map_results")] = retained;
    xjw::gui::project::persistProjectMeta(projectData, metadata, true);
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

ProjectPointCloudWorkflowController::ProjectPointCloudWorkflowController(
    ProjectManager *owner,
    ProjectData *projectData,
    QWidget *parentWidget,
    QObject *parent)
    : QObject(parent)
    , _owner(owner)
    , _projectData(projectData)
    , _parentWidget(parentWidget)
{
}

ProjectPointCloudWorkflowController::~ProjectPointCloudWorkflowController()
{
    // runner 可能仍持有共享取消标志；析构前置位可阻止项目关闭后继续读取深度帧。
    cancelActiveTask();
}

bool ProjectPointCloudWorkflowController::startCreatePointCloudAsync(
    const QJsonObject &settings)
{
    return startWorkflow(settings, false);
}

bool ProjectPointCloudWorkflowController::startDepthMapsOnlyAsync(
    const QJsonObject &settings)
{
    return startWorkflow(settings, true);
}

bool ProjectPointCloudWorkflowController::startWorkflow(
    const QJsonObject &settings,
    bool depth_maps_only)
{
    const QString dialog_title = workflowDialogTitle(depth_maps_only);
    if (!_owner || !_projectData || !_projectData->hasProject())
    {
        failTask(QStringLiteral("请先打开项目，并完成正式空中三角测量。"),
                 dialog_title);
        return false;
    }
    if (_isRunning)
    {
        QMessageBox::information(_parentWidget,
                                 dialog_title,
                                 depth_maps_only
                                     ? QStringLiteral("已有深度图或点云任务正在运行，请等待或先取消当前任务。")
                                     : QStringLiteral("已有点云任务正在运行，请等待或先取消当前任务。"));
        return false;
    }

    const QJsonObject metadata = _projectData->metadata();
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
    if (!xjw::gui::project::isProductionSparseResult(at_record))
    {
        failTask(xjw::gui::project::sparseResultBlockingReason(at_record),
                 dialog_title);
        return false;
    }

    QJsonObject effective_settings = settings;
    QString scene_profile_source;
    const QString scene_profile = resolvedSceneProfile(
        settings, &scene_profile_source);
    effective_settings[QStringLiteral("sceneProfile")] = scene_profile;

    auto context = std::make_shared<PointCloudWorkflowContext>();
    context->session = _owner->currentSessionContext();
    context->settings = effective_settings;
    context->request = xjw::core::project::denseGenerationSettingsFromJson(
        effective_settings);
    context->atIndex = at_index;
    context->sparseCloudPath = sparseCloudPathFromRecord(at_record);
    context->selectedImages = selectedImagesFromRecord(at_record);
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
    if (context->selectedImages.size() < 2)
    {
        failTask(QStringLiteral("正式空三结果中的注册影像不足 2 张。"),
                 dialog_title);
        return false;
    }

    bool all_cameras = false;
    const QMap<QString, xjw::FramePinholeCamera> cameras =
        _owner->getCamerasForImages(context->selectedImages, &all_cameras);
    QString view_error;
    if (!all_cameras || !buildMvsViews(context->session.projectPath,
                                       context->selectedImages,
                                       cameras,
                                       &context->views,
                                       &view_error))
    {
        failTask(view_error.isEmpty()
                     ? QStringLiteral("部分注册影像缺少有效相机参数。")
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
    _cancelFlag = std::make_shared<std::atomic_bool>(false);
    emit pointCloudProgressChanged(
        can_reuse ? QStringLiteral("正在复用兼容深度图")
                  : QStringLiteral("正在准备深度图估计"),
        0);

    if (can_reuse)
    {
        if (context->depthMapsOnly)
        {
            emit depthMapBatchReady(
                context->outputDir,
                static_cast<int>(stored.frames.size()));
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
        clearDepthWorkspace(_projectData, context->outputDir);
        startDepthEstimation(context);
    }
    return true;
}

void ProjectPointCloudWorkflowController::startDepthEstimation(
    const std::shared_ptr<PointCloudWorkflowContext> &context)
{
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [context]()
        {
            DepthEstimationPreparationResult result;
            result.pointCloudInput = xjw::core::project::preparePointCloudInput(
                context->sparseCloudPath,
                context->views,
                context->request.processingDevice);
            if (result.pointCloudInput.ok)
            {
                result.sourcePairQuality =
                    xjw::core::project::loadMvsSourcePairQualities(
                        xjw::common::project::ProjectIO::imageMatchOutputDir(
                            context->session.projectPath),
                        context->selectedImages);
            }
            return result;
        },
        [context](
            ProjectPointCloudWorkflowController *self,
            xjw::gui::tasks::TaskOutcome<
                DepthEstimationPreparationResult> outcome)
        {
            if (!self->_owner ||
                !self->_owner->isCurrentSession(context->session))
            {
                self->finishTask(false);
                return;
            }
            if (!outcome.succeeded())
            {
                self->failTask(outcome.errorMessage.isEmpty()
                                   ? QStringLiteral("稀疏点云预处理失败。")
                                   : outcome.errorMessage,
                               workflowDialogTitle(context->depthMapsOnly));
                return;
            }
            if (self->_cancelFlag &&
                self->_cancelFlag->load(std::memory_order_relaxed))
            {
                self->finishTask(false);
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

            auto *generator = new xjw::mvs::DepthMapGenerator(self);
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
                    &xjw::mvs::DepthMapGenerator::progressChanged,
                    self,
                    [self, context](const QString &stage, float ratio)
                    {
                        if (self->_owner &&
                            self->_owner->isCurrentSession(context->session))
                        {
                            emit self->pointCloudProgressChanged(
                                stage,
                                std::clamp(
                                    static_cast<int>(ratio * 60.0f), 0, 60));
                        }
                    });
            connect(generator,
                    &xjw::mvs::DepthMapGenerator::errorOccurred,
                    self,
                    [context](const QString &message)
                    {
                        context->depthError = message;
                    });
            connect(generator,
                    &xjw::mvs::DepthMapGenerator::depthMapArtifactSaved,
                    self,
                    [self, context](const QJsonObject &artifact)
                    {
                        if (!self->_owner ||
                            !self->_owner->isCurrentSession(context->session))
                        {
                            return;
                        }
                        const QJsonObject record =
                            depthRecordFromArtifact(artifact, *context);
                        if (!record.isEmpty())
                        {
                            xjw::gui::project::upsertProjectRecordByPath(
                                self->_projectData,
                                QStringLiteral("depth_map_results"),
                                QStringLiteral("depth_png"),
                                record);
                        }
                    });
            connect(generator,
                    &xjw::mvs::DepthMapGenerator::finished,
                    self,
                    [self, generator, context](bool success)
                    {
                        if (self->_activeGenerator == generator)
                        {
                            self->_activeGenerator.clear();
                        }
                        generator->deleteLater();
                        if (!self->_owner ||
                            !self->_owner->isCurrentSession(context->session))
                        {
                            self->finishTask(false);
                            return;
                        }
                        if (!success)
                        {
                            self->failTask(
                                context->depthError.isEmpty()
                                    ? QStringLiteral("深度图估计失败或已取消。")
                                    : context->depthError,
                                workflowDialogTitle(context->depthMapsOnly));
                            return;
                        }
                        if (context->saveAfterEachStep && self->_projectData)
                        {
                            self->_projectData->saveProjectAsync();
                        }
                        if (context->depthMapsOnly)
                        {
                            const auto stored =
                                xjw::core::project::collectStoredDepthFramesForDirectory(
                                    self->_projectData->metadata(),
                                    context->outputDir);
                            if (!stored.status.ok || stored.frames.size() < 2)
                            {
                                self->failTask(
                                    stored.status.ok
                                        ? QStringLiteral("自动估计后可用深度图不足 2 帧。")
                                        : stored.status.errorMessage,
                                    QStringLiteral("生成模型"));
                                return;
                            }
                            emit self->depthMapBatchReady(
                                context->outputDir,
                                static_cast<int>(stored.frames.size()));
                            self->finishTask(true);
                            return;
                        }
                        self->startFusion(context);
                    });

            emit self->pointCloudProgressChanged(
                QStringLiteral("正在估计多视深度图"), 1);
            generator->start();
        });
}

void ProjectPointCloudWorkflowController::startFusion(
    const std::shared_ptr<PointCloudWorkflowContext> &context)
{
    if (!_owner || !_owner->isCurrentSession(context->session))
    {
        finishTask(false);
        return;
    }

    const auto discovered = xjw::core::project::collectStoredDepthFramesForDirectory(
        _projectData->metadata(), context->outputDir);
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
    bool all_cameras = false;
    const QMap<QString, xjw::FramePinholeCamera> cameras =
        _owner->getCamerasForImages(frame_images, &all_cameras);
    if (!all_cameras)
    {
        failTask(QStringLiteral("部分深度图对应影像缺少当前空三相机参数。"));
        return;
    }

    emit pointCloudProgressChanged(
        QStringLiteral("正在加载并融合深度图（点云后端请求：%1）")
            .arg(xjw::core::project::processingDeviceId(
                context->request.processingDevice)),
        65);
    const auto cancel_flag = _cancelFlag;
    QPointer<ProjectPointCloudWorkflowController> self(this);
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [self, context, stored, cameras, cancel_flag]() -> PointCloudTaskResult
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
                [stored, cameras, fusion_config, context](
                    int index,
                    xjw::mvs::FusionFrameInput *frame,
                    std::string *error_message)
                {
                    xjw::FramePinholeCamera camera;
                    if (!cameraForImage(cameras,
                                        stored.frames[static_cast<std::size_t>(index)].refImage,
                                        &camera))
                    {
                        if (error_message)
                        {
                            *error_message = "Missing camera for stored depth frame";
                        }
                        return false;
                    }
                    auto loaded = xjw::core::project::buildStoredFusionFrame(
                        stored.frames[static_cast<std::size_t>(index)],
                        camera,
                        fusion_config,
                        static_cast<int>(stored.frames.size()),
                        context->request.fusionMaxImageDim);
                    if (!loaded.status.ok)
                    {
                        if (error_message)
                        {
                            *error_message = xjw::common::io::toUtf8Path(
                                loaded.status.errorMessage);
                        }
                        return false;
                    }
                    *frame = std::move(loaded.frame);
                    frame->viewIndex = index;
                    frame->sourceImageIndices =
                        xjw::core::project::storedFusionSourceIndices(
                            stored.frames, index);
                    return true;
                };

            xjw::mvs::StreamingDepthFusionConfig config;
            config.minConsistentViews = context->request.minConsistentViews;
            config.depthConsistency = context->request.depthConsistency;
            config.workerCount = std::max(1, context->request.threads);
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
                            if (self && self->_owner &&
                                self->_owner->isCurrentSession(context->session))
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
            task.record[QStringLiteral("scene_profile")] =
                sceneProfileFromStoredFrames(stored.frames);
            task.record[QStringLiteral("mvs_backend_requested")] =
                patchMatchBackendText(context->request.patchMatchBackend);
            task.record[QStringLiteral("mvs_backend_actual")] =
                mvsBackendFromStoredFrames(stored.frames);
            task.record[QStringLiteral("mvs_backend_selected_in_dialog")] =
                patchMatchBackendText(context->request.patchMatchBackend);
            task.record[QStringLiteral("mvs_backend_request_applied")] =
                !context->reusedDepthMaps;
            task.record[QStringLiteral("depth_maps_reused")] =
                context->reusedDepthMaps;
            task.record[QStringLiteral("calculate_point_colors")] =
                context->calculateColors;
            task.record[QStringLiteral("point_confidence_available")] = false;
            task.record[QStringLiteral("point_cloud_processing")] = QJsonObject{
                {QStringLiteral("stage"), QStringLiteral("statistical_outlier_removal")},
                {QStringLiteral("requested"),
                 xjw::core::project::processingDeviceId(
                     processing_report.requestedDevice)},
                {QStringLiteral("actual"),
                 processing_skipped
                     ? QStringLiteral("skipped")
                     : xjw::core::project::processingDeviceId(
                           processing_report.actualDevice)},
                {QStringLiteral("used_fallback"), processing_report.usedFallback},
                {QStringLiteral("fallback_reason"),
                 QString::fromStdString(processing_report.fallbackReason)},
                {QStringLiteral("input_points"),
                 static_cast<qint64>(before_processing)},
                {QStringLiteral("output_points"),
                 static_cast<qint64>(cloud.size())}
            };
            if (!stored.frames.empty())
            {
                task.record[QStringLiteral("source_depth_config_hash")] =
                    stored.frames.front().configHash;
            }
            task.ok = true;
            return task;
        },
        [context](
            ProjectPointCloudWorkflowController *manager,
            xjw::gui::tasks::TaskOutcome<PointCloudTaskResult> outcome)
        {
            if (!manager->_owner ||
                !manager->_owner->isCurrentSession(context->session))
            {
                manager->finishTask(false);
                return;
            }
            if (!outcome.succeeded())
            {
                manager->failTask(outcome.errorMessage.isEmpty()
                                      ? QStringLiteral("点云融合发生未知异常")
                                      : QStringLiteral("点云融合异常：%1")
                                            .arg(outcome.errorMessage));
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

            const QJsonObject processing =
                result.record.value(QStringLiteral("point_cloud_processing")).toObject();
            emit manager->pointCloudProgressChanged(
                QStringLiteral("点云去噪完成：%1 → %2 点，请求 %3，实际 %4%5")
                    .arg(processing.value(QStringLiteral("input_points")).toInteger())
                    .arg(processing.value(QStringLiteral("output_points")).toInteger())
                    .arg(processing.value(QStringLiteral("requested")).toString())
                    .arg(processing.value(QStringLiteral("actual")).toString())
                    .arg(processing.value(QStringLiteral("used_fallback")).toBool()
                             ? QStringLiteral("（已回退）")
                             : QString()),
                97);

            if (context->replaceDefaultPointCloud)
            {
                xjw::gui::project::replaceProjectRecordWithLatest(
                    manager->_projectData,
                    QStringLiteral("dense_cloud_results"),
                    result.record);
            }
            else
            {
                xjw::gui::project::upsertProjectRecordByPath(
                    manager->_projectData,
                    QStringLiteral("dense_cloud_results"),
                    QStringLiteral("dense_cloud_xyz"),
                    result.record);
            }
            if (context->saveAfterEachStep)
            {
                manager->_projectData->saveProjectAsync();
            }
            manager->_owner->refreshReconstructionQualityReport();
            emit manager->pointCloudResultReady(
                result.pointCloudPath,
                result.pointCount);
            QMessageBox::information(
                manager->_parentWidget,
                QStringLiteral("创建点云"),
                QStringLiteral("点云已生成。\n点数: %1\n路径: %2")
                    .arg(result.pointCount)
                    .arg(QDir::toNativeSeparators(result.pointCloudPath)));
            manager->finishTask(true);
        });
}

void ProjectPointCloudWorkflowController::cancelActiveTask()
{
    if (_cancelFlag)
    {
        _cancelFlag->store(true, std::memory_order_relaxed);
    }
    if (auto *generator = qobject_cast<xjw::mvs::DepthMapGenerator *>(
            _activeGenerator.data()))
    {
        generator->requestCancel();
    }
}

bool ProjectPointCloudWorkflowController::isRunning() const
{
    return _isRunning;
}

void ProjectPointCloudWorkflowController::finishTask(bool success)
{
    _activeGenerator.clear();
    _cancelFlag.reset();
    _isRunning = false;
    emit pointCloudProgressFinished(success);
}

void ProjectPointCloudWorkflowController::failTask(const QString &message,
                                                 const QString &title)
{
    const QString effective_message = message.trimmed().isEmpty()
        ? QStringLiteral("点云任务失败，未返回具体错误。")
        : message;
    QMessageBox::warning(_parentWidget, title, effective_message);
    if (_isRunning)
    {
        finishTask(false);
    }
}
