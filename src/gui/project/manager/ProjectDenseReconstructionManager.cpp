#include "ProjectDenseReconstructionManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "DenseSparseCloudPreparation.h"
#include "ProjectDenseWorkflowConfig.h"
#include "ProjectMetadataOperations.h"
#include "ProjectModelWorkflowPolicy.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowUtils.h"
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
using xjw::gui::project::DenseGenerationSettings;

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
    bool saveAfterEachStep = false;
    bool calculateColors = true;
    bool replaceDefaultPointCloud = false;
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

QString utcNowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
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

bool cameraForImage(const QMap<QString, xjw::Camera> &cameras,
                    const QString &imagePath,
                    xjw::Camera *camera)
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
                   const QMap<QString, xjw::Camera> &cameras,
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

ProjectDenseReconstructionManager::ProjectDenseReconstructionManager(
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

ProjectDenseReconstructionManager::~ProjectDenseReconstructionManager()
{
    // runner 可能仍持有共享取消标志；析构前置位可阻止项目关闭后继续读取深度帧。
    cancelActiveTask();
}

bool ProjectDenseReconstructionManager::startCreatePointCloudAsync(
    const QJsonObject &settings)
{
    if (!_owner || !_projectData || !_projectData->hasProject())
    {
        failTask(QStringLiteral("请先打开项目，并完成正式空中三角测量。"));
        return false;
    }
    if (_isRunning)
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("创建点云"),
                                 QStringLiteral("已有点云任务正在运行，请等待或先取消当前任务。"));
        return false;
    }

    const QJsonObject metadata = _projectData->metadata();
    const int at_index = xjw::gui::project::findLatestProductionAtResultIndex(metadata);
    const QJsonArray at_results =
        metadata.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (at_index < 0 || at_index >= at_results.size())
    {
        failTask(QStringLiteral("未找到通过质量门控的正式 SfM/BA 稀疏点云结果。"));
        return false;
    }

    const QJsonObject at_record = at_results.at(at_index).toObject();
    if (!xjw::gui::project::isProductionSparseResult(at_record))
    {
        failTask(xjw::gui::project::sparseResultBlockingReason(at_record));
        return false;
    }

    auto context = std::make_shared<PointCloudWorkflowContext>();
    context->session = _owner->currentSessionContext();
    context->settings = settings;
    context->request = xjw::gui::project::denseGenerationSettingsFromJson(settings);
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

    if (!QFileInfo::exists(context->sparseCloudPath))
    {
        failTask(QStringLiteral("正式空三稀疏点云不存在：%1")
                     .arg(context->sparseCloudPath));
        return false;
    }
    if (context->selectedImages.size() < 2)
    {
        failTask(QStringLiteral("正式空三结果中的注册影像不足 2 张。"));
        return false;
    }

    bool all_cameras = false;
    const QMap<QString, xjw::Camera> cameras =
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
                     : view_error);
        return false;
    }

    const auto compatibility =
        xjw::gui::project::assessStoredDepthBatchCompatibility(
            metadata, QString(), at_index);
    const bool can_reuse = context->reuseDepthMaps && compatibility.compatible;
    if (can_reuse)
    {
        const auto stored = xjw::core::project::collectLatestStoredDepthFrames(metadata);
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
        failTask(QStringLiteral("无法创建 MVS 输出目录：%1").arg(context->outputDir));
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
        startFusion(context);
    }
    else
    {
        // 未复用时明确清理当前批次，保证“重新计算”不会被 manifest 静默续跑。
        clearDepthWorkspace(_projectData, context->outputDir);
        startDepthEstimation(context);
    }
    return true;
}

void ProjectDenseReconstructionManager::startDepthEstimation(
    const std::shared_ptr<PointCloudWorkflowContext> &context)
{
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [context]()
        {
            return xjw::gui::project::prepareDenseSparseCloud(
                context->sparseCloudPath,
                context->views);
        },
        [context](
            ProjectDenseReconstructionManager *self,
            xjw::gui::tasks::TaskOutcome<
                xjw::gui::project::DenseSparsePreparationResult> outcome)
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
                                   : outcome.errorMessage);
                return;
            }
            if (self->_cancelFlag &&
                self->_cancelFlag->load(std::memory_order_relaxed))
            {
                self->finishTask(false);
                return;
            }

            auto prepared = std::move(*outcome.value);
            if (!prepared.ok)
            {
                self->failTask(prepared.errorMessage);
                return;
            }

            auto *generator = new xjw::mvs::DepthMapGenerator(self);
            xjw::mvs::DepthGenConfig config =
                xjw::gui::project::buildDepthGenConfig(
                    context->request,
                    static_cast<int>(context->views.size()));
            config.inputSignature =
                context->projectInputSignature.toUtf8().toStdString();
            config.runFusion = false;
            config.saveIntermediateDepthMaps = true;
            config.intermediateDir =
                xjw::common::io::toUtf8Path(context->outputDir);
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
                                    : context->depthError);
                            return;
                        }
                        if (context->saveAfterEachStep && self->_projectData)
                        {
                            self->_projectData->saveProjectAsync();
                        }
                        self->startFusion(context);
                    });

            emit self->pointCloudProgressChanged(
                QStringLiteral("正在估计多视深度图"), 1);
            generator->start();
        });
}

void ProjectDenseReconstructionManager::startFusion(
    const std::shared_ptr<PointCloudWorkflowContext> &context)
{
    if (!_owner || !_owner->isCurrentSession(context->session))
    {
        finishTask(false);
        return;
    }

    const auto stored = xjw::core::project::collectStoredDepthFramesForDirectory(
        _projectData->metadata(), context->outputDir);
    if (!stored.status.ok || stored.frames.size() < 2)
    {
        failTask(stored.status.ok
                     ? QStringLiteral("可融合深度图不足 2 帧。")
                     : stored.status.errorMessage);
        return;
    }

    QStringList frame_images;
    frame_images.reserve(static_cast<int>(stored.frames.size()));
    for (const auto &frame : stored.frames)
    {
        frame_images.push_back(frame.refImage);
    }
    bool all_cameras = false;
    const QMap<QString, xjw::Camera> cameras =
        _owner->getCamerasForImages(frame_images, &all_cameras);
    if (!all_cameras)
    {
        failTask(QStringLiteral("部分深度图对应影像缺少当前空三相机参数。"));
        return;
    }

    emit pointCloudProgressChanged(QStringLiteral("正在加载并融合深度图"), 65);
    const auto cancel_flag = _cancelFlag;
    QPointer<ProjectDenseReconstructionManager> self(this);
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
                xjw::gui::project::buildDepthGenConfig(
                    context->request, frame_count).fusion;
            const xjw::mvs::FusionFrameLoader loader =
                [stored, cameras, fusion_config, context](
                    int index,
                    xjw::mvs::FusionFrameInput *frame,
                    std::string *error_message)
                {
                    xjw::Camera camera;
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
            task.record[QStringLiteral("calculate_point_colors")] =
                context->calculateColors;
            task.record[QStringLiteral("point_confidence_available")] = false;
            if (!stored.frames.empty())
            {
                task.record[QStringLiteral("source_depth_config_hash")] =
                    stored.frames.front().configHash;
            }
            task.ok = true;
            return task;
        },
        [context](
            ProjectDenseReconstructionManager *manager,
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

void ProjectDenseReconstructionManager::cancelActiveTask()
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

bool ProjectDenseReconstructionManager::isRunning() const
{
    return _isRunning;
}

void ProjectDenseReconstructionManager::finishTask(bool success)
{
    _activeGenerator.clear();
    _cancelFlag.reset();
    _isRunning = false;
    emit pointCloudProgressFinished(success);
}

void ProjectDenseReconstructionManager::failTask(const QString &message,
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
