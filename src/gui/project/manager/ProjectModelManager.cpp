#include "ProjectModelManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "ProjectDenseWorkflowConfig.h"
#include "ProjectMetadataOperations.h"
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

auto makeProgressReporter(QPointer<ProjectModelManager> manager)
{
    return [manager](const QString &stage, int percent) 
    {
        if (!manager)
        {
            return;
        }
        QMetaObject::invokeMethod(manager.data(), [manager, stage, percent]()
        {
            if (!manager)
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
    request.progress = makeProgressReporter(manager);

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
    QJsonObject modelRecord = xjw::gui::project::makeModelResultRecord(
        utcNowIso(),
        QStringLiteral("mesh_reconstruction"),
        taskResult.value(QStringLiteral("model_ply")).toString(),
        taskResult.value(QStringLiteral("vertex_count")).toInt(-1),
        taskResult.value(QStringLiteral("face_count")).toInt(-1),
        QString(),
        denseCloudPath,
        denseCloudPath);

    mergeJsonObject(&modelRecord, taskResult);
    modelRecord[QStringLiteral("requested_method")] = settings.value(QStringLiteral("method")).toString();
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
                                                                taskResult.value(QStringLiteral("vertex_count")).toInt(-1),
                                                                taskResult.value(QStringLiteral("face_count")).toInt(-1));
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
    return QStringLiteral("网格重建完成。\n模型: %1\n纹理模型: %2\n顶点: %3  面数: %4")
        .arg(taskResult.value(QStringLiteral("final_model_path")).toString())
        .arg(taskResult.value(QStringLiteral("model_obj")).toString())
        .arg(taskResult.value(QStringLiteral("vertex_count")).toInt(-1))
        .arg(taskResult.value(QStringLiteral("face_count")).toInt(-1));
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
                     owner, [watcher, onFinished = std::forward<OnFinished>(onFinished)]() mutable {
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
    , m_owner(owner)
    , m_projectData(projectData)
    , m_parentWidget(parentWidget)
{
}

bool ProjectModelManager::ensureProjectOpen(const QString &message,
                                            const QString &title) const
{
    if (m_projectData && m_projectData->hasProject())
    {
        return true;
    }
    QMessageBox::warning(m_parentWidget, title, message);
    return false;
}

void ProjectModelManager::startGenerateModelAsync()
{
    QJsonObject settings;
    settings[QStringLiteral("method")] = QStringLiteral("Poisson Surface");
    settings[QStringLiteral("qualityProfile")] = QStringLiteral("balanced");
    settings[QStringLiteral("voxelDensity")] = QStringLiteral("medium");
    settings[QStringLiteral("export_format")] = QStringLiteral("PLY");
    startMeshReconstructionAsync(settings);
}

void ProjectModelManager::startMeshReconstructionAsync(const QJsonObject &settings)
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目"), QStringLiteral("提示")))
    {
        return;
    }

    QString denseCloudPath = QDir::cleanPath(settings.value(QStringLiteral("denseCloudPath")).toString().trimmed());
    if (!denseCloudPath.isEmpty())
    {
        if (!QFileInfo::exists(denseCloudPath))
        {
            QMessageBox::warning(m_parentWidget,
                                 QStringLiteral("网格重建"),
                                 QStringLiteral("所选密集点云不存在：\n%1").arg(denseCloudPath));
            return;
        }
    }
    else
    {
        QString errorMessage;
        if (!resolveLatestDenseCloudPath(m_projectData, &denseCloudPath, &errorMessage))
        {
            QMessageBox::warning(m_parentWidget,
                                 QStringLiteral("网格重建"),
                                 QStringLiteral("%1\n请先完成密集点云生成。").arg(errorMessage));
            return;
        }
    }

    {
        QJsonObject meta = m_projectData->metadata();
        meta[QStringLiteral("mesh_reconstruction_settings")] = settings;
        xjw::gui::project::persistProjectMeta(m_projectData, meta, false);
    }

    const QString outputRoot = QFileInfo(denseCloudPath).absolutePath();

    emit meshProgressChanged(tr("正在初始化网格重建..."), 0);
    QPointer<ProjectModelManager> self(this);
    runModelAsyncTask(
        this,
        [self, denseCloudPath, outputRoot, settings]() -> ModelTaskResult {
            ModelTaskResult task;
            if (!self)
            {
                task.errMsg = QStringLiteral("网格重建已取消：项目窗口已关闭");
                return task;
            }

            xjw::mesh::ReconstructionConfig cfg;
            cfg.resolution = xjw::mesh::workflow::meshResolutionFromSettings(settings);
            cfg.smoothIterations = qBound(0, settings.value(QStringLiteral("smoothIter")).toInt(3), 50);
            cfg.smoothLambda = 0.5f;
            cfg.padding = 0.05f;
            cfg.forcePoisson = settings.value(QStringLiteral("method"))
                                   .toString(QStringLiteral("Poisson Surface"))
                                   .contains(QStringLiteral("Poisson"), Qt::CaseInsensitive);
            cfg.poissonDepth = qBound(7, settings.value(QStringLiteral("octreeDepth")).toInt(10), 12);
            cfg.poissonThreads = qBound(1, settings.value(QStringLiteral("threads")).toInt(8), 128);
            cfg.poissonPointWeight = std::clamp(static_cast<float>(settings.value(QStringLiteral("poissonPointWeight")).toDouble(cfg.poissonPointWeight)),
                                                0.0f,
                                                8.0f);
            cfg.poissonTrim = std::clamp(static_cast<float>(settings.value(QStringLiteral("poissonTrim")).toDouble(cfg.poissonTrim)),
                                         0.0f,
                                         12.0f);
            cfg.fillHoles = settings.value(QStringLiteral("holeFill")).toBool(true);
            cfg.holeFillPasses =
                xjw::mesh::workflow::holeFillPassesFromArea(settings.value(QStringLiteral("maxHoleSize")).toDouble(100.0));
            cfg.cleanSmallComponents = settings.value(QStringLiteral("cleanSmall")).toBool(true);
            cfg.minComponentFaces = qBound(2, settings.value(QStringLiteral("minFaces")).toInt(100), 100000);

            const QString qualityProfile = settings.contains(QStringLiteral("qualityProfile"))
                ? settings.value(QStringLiteral("qualityProfile")).toString()
                : QStringLiteral("balanced");
            const QString voxelDensity = settings.value(QStringLiteral("voxelDensity")).toString(QStringLiteral("medium"));

            if (qualityProfile == QStringLiteral("detail"))
            {
                cfg.resolution = std::max(cfg.resolution, 320);
                cfg.poissonDepth = std::max(cfg.poissonDepth, 10);
                cfg.poissonPointWeight = std::max(cfg.poissonPointWeight, 4.8f);
                cfg.poissonTrim = std::max(cfg.poissonTrim, 9.0f);
                cfg.simplifyTargetFaces = std::max(cfg.simplifyTargetFaces, 65000);
                cfg.enableDownsample = false;
                cfg.voxelSimplifyFactor = 1.15f;
                cfg.kNormals = std::max(cfg.kNormals, 18);
                cfg.smoothIterations = std::max(0, cfg.smoothIterations - 1);
                cfg.smoothLambda = 0.36f;
            }
            else if (qualityProfile == QStringLiteral("lite"))
            {
                cfg.resolution = std::min(cfg.resolution, 224);
                cfg.poissonDepth = std::min(cfg.poissonDepth, 9);
                cfg.poissonPointWeight = std::min(cfg.poissonPointWeight, 3.2f);
                cfg.poissonTrim = std::min(cfg.poissonTrim, 8.4f);
                cfg.simplifyTargetFaces = 16000;
                cfg.enableDownsample = true;
                cfg.downsampleVoxelScale = 1.0f;
                cfg.voxelSimplifyFactor = 2.5f;
                cfg.smoothIterations = std::min(3, cfg.smoothIterations + 1);
                cfg.smoothLambda = 0.55f;
            }
            else if (voxelDensity == QStringLiteral("coarse"))
            {
                cfg.voxelSimplifyFactor = 2.6f;
                cfg.enableDownsample = true;
                cfg.downsampleVoxelScale = 1.0f;
                cfg.poissonPointWeight = std::min(cfg.poissonPointWeight, 3.6f);
                cfg.simplifyTargetFaces = 14000;
            }
            else if (voxelDensity == QStringLiteral("fine"))
            {
                cfg.voxelSimplifyFactor = 1.25f;
                cfg.enableDownsample = false;
                cfg.denoiseStdMul = 1.8f;
                cfg.kNormals = 18;
                cfg.poissonPointWeight = std::max(cfg.poissonPointWeight, 4.6f);
                cfg.poissonTrim = std::max(8.0f, cfg.poissonTrim);
                cfg.simplifyTargetFaces = 60000;
            }
            else
            {
                cfg.voxelSimplifyFactor = 1.8f;
                cfg.enableDownsample = true;
                cfg.downsampleVoxelScale = 0.8f;
                cfg.simplifyTargetFaces = 28000;
            }

            const bool decimate = settings.value(QStringLiteral("decimate")).toBool(false);
            if (decimate)
            {
                const double decimateRatio = std::clamp(
                    settings.value(QStringLiteral("decimateRatio")).toDouble(0.5),
                    0.05,
                    1.0);
                cfg.simplifyTargetFaces = std::max(1000, static_cast<int>(std::lround(cfg.simplifyTargetFaces * decimateRatio)));
                cfg.enableDownsample = true;
                cfg.voxelSimplifyFactor = std::max(cfg.voxelSimplifyFactor,
                                                   static_cast<float>(1.0 / decimateRatio));
            }
            cfg.verbose = false;

            xjw::mesh::workflow::MeshBuildRequest request;
            request.pointCloudPath = denseCloudPath;
            request.outputRoot = outputRoot;
            request.reconstruction = cfg;
            request.exportObj = xjw::mesh::workflow::exportObjRequested(settings);
            request.texture = xjw::mesh::workflow::defaultTextureConfig();
            request.progress = makeProgressReporter(self);

            const xjw::mesh::workflow::WorkflowResult workflowResult =
                xjw::mesh::workflow::buildMeshAndOptionalTexture(request);
            applyWorkflowResult(&task, workflowResult);
            return task;
        },
        [self, denseCloudPath, settings](const ModelTaskResult &task) {
            if (!self)
            {
                return;
            }
            emit self->meshProgressFinished(task.ok);
            handleTaskResult(self->m_parentWidget,
                             QStringLiteral("网格重建"),
                             QStringLiteral("网格重建失败"),
                             task,
                             [self, denseCloudPath, settings](const QJsonObject &taskResult) {
                if (!self)
                {
                    return;
                }
                const QJsonObject modelRecord = buildMeshReconstructionRecord(taskResult,
                                                                               denseCloudPath,
                                                                               settings);
                persistModelResult(self->m_projectData, modelRecord);
                if (!settings.value(QStringLiteral("pipeline_mode")).toBool(false))
                {
                    QMessageBox::information(self->m_parentWidget,
                                             QStringLiteral("网格重建"),
                                             meshReconstructionSuccessMessage(taskResult));
                }
            });
        });
}

void ProjectModelManager::startTextureMappingAsync(const QJsonObject &settings)
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目"), QStringLiteral("提示")))
    {
        return;
    }

    if (!m_projectData)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("纹理映射"),
                             QStringLiteral("项目未就绪"));
        return;
    }

    const auto lookup = xjw::common::project::resolveLatestModelMeshRecord(m_projectData->metadata());
    if (!lookup.ok)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("纹理映射"),
                             lookup.errorMessage);
        return;
    }

    const QString meshPath = lookup.meshPath;
    const QJsonObject baseRecord = lookup.modelRecord;

    {
        QJsonObject meta = m_projectData->metadata();
        meta[QStringLiteral("texture_mapping_settings")] = settings;
        xjw::gui::project::persistProjectMeta(m_projectData, meta, false);
    }

    const QString productsDir = QFileInfo(meshPath).absolutePath();

    emit meshProgressChanged(tr("正在初始化纹理映射..."), 0);
    QPointer<ProjectModelManager> self(this);
    runModelAsyncTask(
        this,
        [self, meshPath, productsDir, settings]() -> ModelTaskResult {
            ModelTaskResult task;
            if (!self)
            {
                task.errMsg = QStringLiteral("纹理映射已取消：项目窗口已关闭");
                return task;
            }

            xjw::mesh::workflow::TextureBuildRequest request;
            request.meshPath = meshPath;
            request.outputDir = productsDir;
            request.texture = xjw::mesh::workflow::textureConfigFromSettings(settings);
            request.progress = makeProgressReporter(self);

            const xjw::mesh::workflow::WorkflowResult workflowResult =
                xjw::mesh::workflow::buildTextureOnly(request);
            applyWorkflowResult(&task, workflowResult);
            return task;
        },
        [self, meshPath, baseRecord](const ModelTaskResult &task) {
            if (!self)
            {
                return;
            }
            emit self->meshProgressFinished(task.ok);
            handleTaskResult(self->m_parentWidget,
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
                persistModelResult(self->m_projectData, modelRecord);
                QMessageBox::information(self->m_parentWidget,
                                         QStringLiteral("纹理映射"),
                                         textureMappingSuccessMessage(taskResult));
            });
        });
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

    QJsonObject updatedMeta = m_projectData->metadata();
    replaceMetaArrayWithLatest(&updatedMeta, QStringLiteral("dense_cloud_results"), denseResult);
    replaceMetaArrayWithLatest(&updatedMeta, QStringLiteral("model_results"), enrichedModelResult);
    xjw::gui::project::persistProjectMeta(m_projectData, updatedMeta, true);
}
