#include "ProjectModelManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "ProjectDenseWorkflowConfig.h"
#include "ProjectMetadataOperations.h"
#include "ProjectModelWorkflowPolicy.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowUtils.h"
#include "Logger.h"
#include "ModelWorkflowService.h"
#include "project/ProjectCommonUtils.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QMessageBox>
#include <QMetaObject>
#include <QPointer>
#include <QStringList>
#include <QThread>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <exception>
#include <utility>

using xjw::gui::project::makeDenseResultRecord;
using xjw::gui::project::makeModelResultRecord;
using xjw::gui::project::replaceMetaArrayWithLatest;
using xjw::gui::project::resolveLatestDenseCloudPath;
using xjw::gui::project::runDemProducts;

namespace
{

struct ModelTaskResult
{
    QJsonObject result;
    QString errMsg;
    bool ok = false;
};

struct GenerateModelTaskInput
{
    QString cloudPath;
    bool sourceIsDense = false;
    QString outputRoot;
    int gridResolution = 512;
    int meshResolution = 128;
    int meshSmoothIterations = 2;
    double meshSmoothLambda = 0.5;
    double meshPadding = 0.05;
    bool exportObj = true;
};

struct ResolvedModelSource
{
    QString sourcePointCloudPath;
    QString requestedSourcePath;
    QString outputRoot;
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
                          const QString &projectPath)
{
    return [manager, owner, projectPath](const QString &stage, int percent)
    {
        if (!manager || !owner)
        {
            return;
        }
        QMetaObject::invokeMethod(manager.data(), [manager, owner, projectPath, stage, percent]()
        {
            if (!manager || !owner || owner->currentProjectPath() != projectPath)
            {
                return;
            }
            emit manager->meshProgressChanged(stage, percent);
        }, Qt::QueuedConnection);
    };
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

ModelTaskResult runGenerateModelTask(QPointer<ProjectModelManager> manager,
                                     QPointer<ProjectManager> owner,
                                     const QString &projectPath,
                                     const GenerateModelTaskInput &input)
{
    ModelTaskResult task;

    if (!input.sourceIsDense)
    {
        const auto runResult = runDemProducts(input.cloudPath,
                                              input.outputRoot,
                                              static_cast<double>(input.gridResolution),
                                              QStringLiteral("float32"),
                                              true);
        if (!runResult.ok)
        {
            task.ok = false;
            task.result = runResult.payload;
            task.errMsg = runResult.error;
            return task;
        }
        task.result = runResult.payload;
    }

    xjw::mesh::ReconstructionConfig reconstruction;
    reconstruction.resolution = input.meshResolution;
    reconstruction.smoothIterations = input.meshSmoothIterations;
    reconstruction.smoothLambda = static_cast<float>(input.meshSmoothLambda);
    reconstruction.padding = static_cast<float>(input.meshPadding);
    reconstruction.fillHoles = true;
    reconstruction.holeFillPasses = 12;
    reconstruction.cleanSmallComponents = true;
    reconstruction.minComponentFaces = std::max(64, input.meshResolution / 2);
    reconstruction.verbose = false;

    xjw::mesh::workflow::MeshBuildRequest request;
    request.pointCloudPath = input.cloudPath;
    request.outputRoot = input.outputRoot;
    request.reconstruction = reconstruction;
    request.exportObj = input.exportObj;
    request.texture = xjw::mesh::workflow::defaultTextureConfig();
    request.progress = makeProgressReporter(manager, owner, projectPath);

    const xjw::mesh::workflow::WorkflowResult workflowResult =
        xjw::mesh::workflow::buildMeshAndOptionalTexture(request);
    if (!workflowResult.ok)
    {
        task.ok = false;
        task.errMsg = workflowResult.errorMessage;
        return task;
    }

    mergeJsonObject(&task.result, workflowResult.payload);
    task.ok = true;
    return task;
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

QJsonObject buildMeshReconstructionRecord(const QJsonObject &taskResult,
                                          const QString &denseCloudPath,
                                          const QJsonObject &settings)
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
        "网格重建完成。\n模型: %1\n纹理模型: %2\n顶点: %3  面数: %4")
        .arg(taskResult.value(QStringLiteral("final_model_path")).toString())
        .arg(taskResult.value(QStringLiteral("model_obj")).toString())
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
    return QStringLiteral("纹理映射完成。\nOBJ: %1\n纹理: %2")
        .arg(taskResult.value(QStringLiteral("model_obj")).toString())
        .arg(taskResult.value(QStringLiteral("texture_png")).toString());
}

QString modelGenerationSuccessMessage(const QJsonObject &terrainResult)
{
    return QStringLiteral("模型流程完成。\n深度图: %1\n稠密点云: %2\n最终模型: %3\n顶点: %4  面数: %5")
        .arg(terrainResult.value(QStringLiteral("depth_png")).toString())
        .arg(terrainResult.value(QStringLiteral("dense_cloud_xyz")).toString())
        .arg(terrainResult.value(QStringLiteral("final_model_path")).toString())
        .arg(terrainResult.value(QStringLiteral("vertex_count")).toInt(-1))
        .arg(terrainResult.value(QStringLiteral("face_count")).toInt(-1));
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

template <typename Worker, typename OnFinished>
void runModelAsyncTask(QObject *owner,
                       Worker &&worker,
                       OnFinished &&onFinished)
{
    auto *watcher = new QFutureWatcher<ModelTaskResult>(owner);
    QObject::connect(watcher, &QFutureWatcher<ModelTaskResult>::finished,
                     watcher, [watcher, onFinished = std::forward<OnFinished>(onFinished)]() mutable {
        ModelTaskResult task;
        try
        {
            task = watcher->result();
        }
        catch (const std::exception &exception)
        {
            task.ok = false;
            task.errMsg = QStringLiteral("异步模型任务异常: %1")
                              .arg(QString::fromLocal8Bit(exception.what()));
            LOG_ERROR("runModelAsyncTask finished callback caught std::exception: %s", exception.what());
        }
        catch (...)
        {
            task.ok = false;
            task.errMsg = QStringLiteral("异步模型任务发生未知异常");
            LOG_ERROR("runModelAsyncTask finished callback caught unknown exception");
        }
        watcher->deleteLater();
        onFinished(task);
    });
    watcher->setFuture(QtConcurrent::run(
        [worker = std::forward<Worker>(worker)]() mutable -> ModelTaskResult {
            try
            {
                return worker();
            }
            catch (const std::exception &exception)
            {
                ModelTaskResult task;
                task.ok = false;
                task.errMsg = QStringLiteral("模型任务执行异常: %1")
                                  .arg(QString::fromLocal8Bit(exception.what()));
                LOG_ERROR("runModelAsyncTask worker caught std::exception: %s", exception.what());
                return task;
            }
            catch (...)
            {
                ModelTaskResult task;
                task.ok = false;
                task.errMsg = QStringLiteral("模型任务执行过程中发生未知异常");
                LOG_ERROR("runModelAsyncTask worker caught unknown exception");
                return task;
            }
        }));
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

bool ProjectModelManager::ensureProjectOpen(const QString &message,
                                            const QString &title) const
{
    if (_projectData && _projectData->hasProject())
    {
        return true;
    }
    QMessageBox::warning(_parentWidget, title, message);
    return false;
}

void ProjectModelManager::startGenerateModelAsync()
{
    QJsonObject settings;
    settings[QStringLiteral("method")] = QStringLiteral("Poisson Surface");
    settings[QStringLiteral("qualityProfile")] = QStringLiteral("balanced");
    settings[QStringLiteral("voxelDensity")] = QStringLiteral("medium");
    settings[QStringLiteral("export_format")] = QStringLiteral("OBJ");
    startMeshReconstructionAsync(settings);
}

bool ProjectModelManager::startMeshReconstructionAsync(const QJsonObject &settings)
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目"), QStringLiteral("提示")))
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

    ResolvedModelSource resolvedSource;
    QString sourceError;
    if (!resolveModelSourceForMeshing(_projectData, settings, &resolvedSource, &sourceError))
    {
        QMessageBox::warning(_parentWidget, dialogTitle, sourceError);
        return false;
    }

    QJsonObject effectiveSettings = settings;
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

    _isRunning = true;
    _activeCancelFlag = std::make_shared<std::atomic_bool>(false);
    const std::shared_ptr<std::atomic_bool> cancel_flag = _activeCancelFlag;
    emit meshProgressChanged(tr("正在初始化模型生成..."), 0);
    QPointer<ProjectModelManager> self(this);
    QPointer<ProjectManager> ownerGuard(_owner);
    const QString projectPath = _owner ? _owner->currentProjectPath() : QString();
    runModelAsyncTask(
        this,
        [self,
         ownerGuard,
         resolvedSource,
         effectiveSettings,
         projectPath,
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
                makeProgressReporter(self, ownerGuard, projectPath);
            request.progress = [cancel_flag, progress_reporter](
                                   const QString &stage, int percent)
            {
                if (!cancel_flag->load(std::memory_order_relaxed))
                {
                    progress_reporter(stage, percent);
                }
            };

            const xjw::mesh::workflow::WorkflowResult workflowResult =
                xjw::mesh::workflow::buildModel(request);
            applyWorkflowResult(&task, workflowResult);
            return task;
        },
        [self, ownerGuard, resolvedSource, effectiveSettings, projectPath, dialogTitle](const ModelTaskResult &task) {
            if (!self)
            {
                return;
            }
            self->_isRunning = false;
            self->_activeCancelFlag.reset();
            if (!ownerGuard || ownerGuard->currentProjectPath() != projectPath)
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
                const QJsonObject modelRecord = buildMeshReconstructionRecord(taskResult,
                                                                               sourcePointCloudPath,
                                                                               effectiveSettings);
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
    if (!ensureProjectOpen(QStringLiteral("请先打开项目"), QStringLiteral("提示")))
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
    const QString depthMapSourcePath =
        baseRecord.value(QStringLiteral("depth_map_source_path"))
            .toString(baseRecord.value(QStringLiteral("source_path")).toString());

    {
        QJsonObject meta = _projectData->metadata();
        meta[QStringLiteral("texture_mapping_settings")] = settings;
        xjw::gui::project::persistProjectMeta(_projectData, meta, false);
    }

    const QString productsDir = QFileInfo(meshPath).absolutePath();

    _isRunning = true;
    emit meshProgressChanged(tr("正在初始化纹理映射..."), 0);
    QPointer<ProjectModelManager> self(this);
    QPointer<ProjectManager> ownerGuard(_owner);
    const QString projectPath = _owner ? _owner->currentProjectPath() : QString();
    runModelAsyncTask(
        this,
        [self, ownerGuard, meshPath, productsDir, depthMapSourcePath, settings, projectPath]() -> ModelTaskResult {
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
            request.progress = makeProgressReporter(self, ownerGuard, projectPath);

            const xjw::mesh::workflow::WorkflowResult workflowResult =
                xjw::mesh::workflow::buildTextureOnly(request);
            applyWorkflowResult(&task, workflowResult);
            return task;
        },
        [self, ownerGuard, meshPath, baseRecord, projectPath](const ModelTaskResult &task) {
            if (!self)
            {
                return;
            }
            self->_isRunning = false;
            if (!ownerGuard || ownerGuard->currentProjectPath() != projectPath)
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

void ProjectModelManager::finalizeModelGenerationSuccess(const QJsonObject &terrainResult,
                                                         const QString &sourceCloudPath,
                                                         bool sourceIsDense)
{
    const QString denseXyz = terrainResult.value(QStringLiteral("dense_cloud_xyz")).toString();
    const int denseCount = terrainResult.value(QStringLiteral("dense_point_count")).toInt(-1);

    const QJsonObject denseResult = makeDenseResultRecord(utcNowIso(),
                                                          denseXyz,
                                                          denseCount,
                                                          sourceCloudPath);
    const QJsonObject modelResult = makeModelResultRecord(
        utcNowIso(),
        sourceIsDense ? QStringLiteral("mvs_dense_cloud_mesh") : QStringLiteral("dense_cloud_grid_mesh"),
        terrainResult.value(QStringLiteral("mesh_ply")).toString(),
        terrainResult.value(QStringLiteral("vertex_count")).toInt(-1),
        terrainResult.value(QStringLiteral("face_count")).toInt(-1),
        QString(),
        sourceCloudPath,
        denseXyz);

    const QJsonObject enrichedModelResult =
        xjw::common::project::enrichModelResultFromTerrain(modelResult, terrainResult);

    QJsonObject updatedMeta = _projectData->metadata();
    replaceMetaArrayWithLatest(&updatedMeta, QStringLiteral("dense_cloud_results"), denseResult);
    replaceMetaArrayWithLatest(&updatedMeta, QStringLiteral("model_results"), enrichedModelResult);
    xjw::gui::project::persistProjectMeta(_projectData, updatedMeta, true);
}
