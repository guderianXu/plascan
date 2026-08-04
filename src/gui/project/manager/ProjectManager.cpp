#include "ProjectManager.h"
#include "ProjectModelManager.h"
#include "ProjectPointCloudWorkflowController.h"
#include "ProjectLifecycleController.h"
#include "ProjectMaskWorkflowController.h"
#include "ProjectSparseReconstructionManager.h"
#include "ProjectTerrainProductsManager.h"
#include "ProjectCameraSetupManager.h"
#include "ProjectUiCommands.h"
#include "ProjectData.h"
#include "project/ProjectAssetImporter.h"
#include "project/ProjectIO.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "ProjectCameraImportService.h"
#include "ProjectBundleAdjustExecution.h"
#include "ProjectBundleAdjustWorkflow.h"
#include "ProjectCameraInitialization.h"
#include "ProjectResourceCleanupService.h"
#include "ProjectTiePointResultService.h"

#include "ProjectMetadataOperations.h"
#include "ProjectSfmWorkflow.h"
#include "ProjectSparseWorkflow.h"
#include "ProjectResultRecords.h"
#include "ProjectReferenceDatasets.h"
#include "ProjectSurveyControl.h"
#include "ProjectWorkflowUtils.h"
#include "ProjectWorkflowReports.h"
#include "camera/SurveyControlDialog.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "filtering/SparsePointCloudProcessor.h"
#include "FileDialogStateManager.h"
#include "Camera.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "SparseCloudPreprocessor.h"
#include "DenseCloudBuilder.h"
#include "Intersection.h"
#include "BundleAdjust.h"
#include "SparseCloudValidator.h"
#include "SurfaceReconstructor.h"


#include <QMessageBox>
#include <QInputDialog>
#include <QLineEdit>
#include <QDir>
#include <QFileDialog>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPointer>
#include <QThread>
#include <cmath>
#include <QFile>
#include <QSet>
#include <QFileInfo>
#include <QTextStream>
#include <QDateTime>
#include <QImage>

#include <algorithm>
#include <atomic>
#include <array>
#include <limits>
#include <memory>

using xjw::common::project::cameraFromJson;
using xjw::common::project::cameraToJson;
using xjw::gui::project::BundleAdjustExecutionResult;
using xjw::gui::project::buildSparsePointWorkflowSuccessMessage;
using xjw::gui::project::commitBundleAdjustPreview;
using xjw::gui::project::existingCameraImages;
using xjw::gui::project::finalizeBundleAdjustArtifacts;
using xjw::gui::project::finalizeInitializedCameraPoses;
using xjw::gui::project::focalPixelsFromExif;
using xjw::gui::project::InitPoseFinalizeResult;
using xjw::gui::project::makeAtResultRecord;
using xjw::gui::project::makeDenseResultRecord;
using xjw::gui::project::makeDepthResultRecord;
using xjw::gui::project::makeInitializedCameraMeta;
using xjw::gui::project::makeModelResultRecord;
using xjw::common::project::normalizePath;
using xjw::common::project::pathTokenMatchesImage;
using xjw::gui::project::persistProjectMeta;
using xjw::gui::project::projectFilesMeta;
using xjw::gui::project::resolveInitTargets;
using xjw::gui::project::resolveLatestDenseCloudPath;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::gui::project::resolveSparsePointContext;
using xjw::gui::project::runBundleAdjustExecution;
using xjw::gui::project::runSparsePointWorkflow;
using xjw::gui::project::replaceMetaArrayWithLatest;
using xjw::gui::project::replaceProjectRecordWithLatest;
using xjw::gui::project::SparsePointContext;
using xjw::gui::project::SparsePointOperationResult;
using xjw::gui::project::SparsePointWorkflowKind;
using xjw::gui::project::SparsePointWorkflowSpec;
using xjw::gui::project::sparseOperationDisplayName;
using xjw::gui::project::sparsePointWorkflowSpec;
using xjw::gui::project::summarizeAtResults;
using xjw::gui::project::findLatestAtResultIndex;
using xjw::gui::project::upsertProjectRecordByPath;
using xjw::gui::project::upsertMetaArrayRecordByIndex;
using xjw::gui::project::upsertMetaArrayRecordByPath;
using xjw::gui::project::withPreparedCameras;
using xjw::gui::project::writeJsonObjectFile;
using xjw::gui::project::writeBundleAdjustReport;

namespace
{

struct ImageFolderScan
{
    bool success = false;
    QString folder;
    QString errorMessage;
    QStringList imagePaths;
};

QStringList imageFolderNameFilters()
{
    return QStringList{
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff"),
        QStringLiteral("*.TIF"),
        QStringLiteral("*.TIFF"),
        QStringLiteral("*.png"),
        QStringLiteral("*.PNG"),
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.JPG"),
        QStringLiteral("*.JPEG")
    };
}

ImageFolderScan scanImageFolder(const QString &folder)
{
    ImageFolderScan scan;
    scan.folder = folder;

    QDir dir(folder);
    if (!dir.exists())
    {
        scan.errorMessage = QStringLiteral("文件夹不存在: %1").arg(folder);
        return scan;
    }

    const QFileInfoList files = dir.entryInfoList(imageFolderNameFilters(),
                                                  QDir::Files | QDir::Readable,
                                                  QDir::Name | QDir::IgnoreCase);
    scan.imagePaths.reserve(files.size());
    for (const QFileInfo &file : files)
    {
        scan.imagePaths.append(file.absoluteFilePath());
    }

    scan.success = true;
    return scan;
}

QString imageLabel(const QString &path)
{
    const QFileInfo fi(path);
    return fi.fileName().isEmpty() ? path : fi.fileName();
}

bool isBaPriorRole(const QString &role)
{
    const QString normalized = role.toLower();
    return normalized == QLatin1String("ba_prior")
        || normalized == QLatin1String("bundle_adjustment")
        || normalized == QLatin1String("reference_prior");
}

QString firstReferenceDemPriorPath(const QJsonObject &meta)
{
    const QJsonArray references = meta.value(QStringLiteral("reference_datasets")).toArray();
    for (const QJsonValue &value : references)
    {
        const QJsonObject reference = value.toObject();
        if (!isBaPriorRole(reference.value(QStringLiteral("role")).toString()))
        {
            continue;
        }

        const QString type = reference.value(QStringLiteral("type")).toString().toLower();
        const QString path = reference.value(QStringLiteral("path")).toString().trimmed();
        if ((type == QLatin1String("dem") || type == QLatin1String("reference_dem")) && !path.isEmpty())
        {
            return path;
        }
    }
    return QString();
}

QString firstReferenceLaserPriorPath(const QJsonObject &meta)
{
    const QJsonArray references = meta.value(QStringLiteral("reference_datasets")).toArray();
    for (const QJsonValue &value : references)
    {
        const QJsonObject reference = value.toObject();
        if (!isBaPriorRole(reference.value(QStringLiteral("role")).toString()))
        {
            continue;
        }

        const QString type = reference.value(QStringLiteral("type")).toString().toLower();
        const QString path = reference.value(QStringLiteral("path")).toString().trimmed();
        if (path.isEmpty())
        {
            continue;
        }

        const QString suffix = QFileInfo(path).suffix().toLower();
        const bool laserType = type == QLatin1String("lidar")
            || type == QLatin1String("reference_lidar")
            || type == QLatin1String("point_cloud");
        if (laserType && suffix == QLatin1String("ply"))
        {
            return path;
        }
    }
    return QString();
}

void appendMetaArrayRecord(QJsonObject *meta,
                           const QString &arrayKey,
                           const QJsonObject &record)
{
    // 通用追加：读取数组 -> append -> 写回，避免各处重复样板。
    if (!meta) return;
    QJsonArray arr = meta->value(arrayKey).toArray();
    arr.append(record);
    (*meta)[arrayKey] = arr;
}

} // namespace

ProjectManager::ProjectManager(ProjectData *projectData, QWidget *parent)
    : QObject(parent)
    , _parent(parent)
    , _projectData(projectData)
    , _sparseReconstructionManager(
          new ProjectSparseReconstructionManager(this, projectData, parent, this))
    , _pointCloudWorkflowController(
          new ProjectPointCloudWorkflowController(this, projectData, parent, this))
    , _modelManager(new ProjectModelManager(this, projectData, parent, this))
    , _terrainProductsManager(new ProjectTerrainProductsManager(this, projectData, parent, this))
    , _cameraSetupManager(new ProjectCameraSetupManager(this, projectData, parent, this))
    , _uiCommands(new ProjectUiCommands(projectData, parent))
    , _lifecycleController(new ProjectLifecycleController(projectData, _uiCommands, parent, this))
    , _maskWorkflowController(new ProjectMaskWorkflowController(projectData, parent, this))
{
    _uiCommands->setDirectoryAccessors(
        [this](const QString &key) { return getLastUsedDir(key); },
        [this](const QString &key, const QString &dir) { saveLastUsedDir(key, dir); });

    // 连接ProjectData信号
    if (_projectData)
    {
        const auto advanceSessionGeneration = [this]()
        {
            ++_projectSessionGeneration;
        };
        connect(_projectData, &ProjectData::projectOpened,
                this, advanceSessionGeneration);
        connect(_projectData, &ProjectData::projectClosed,
                this, advanceSessionGeneration);
        connect(_projectData, &ProjectData::activeChunkChanged,
                this,
                [advanceSessionGeneration](const QString &, const QString &, int)
                {
                    advanceSessionGeneration();
                });
        connect(_projectData, &ProjectData::projectOpened,
                this, &ProjectManager::projectOpened);
        connect(_projectData, &ProjectData::projectSaved,
                this, &ProjectManager::projectSaved);
        connect(_projectData, &ProjectData::projectClosed,
                this, &ProjectManager::projectClosed);
        connect(_projectData, &ProjectData::chunkListChanged,
                this, &ProjectManager::chunkListChanged);
        connect(_projectData, &ProjectData::metadataChanged,
                this,
                [this](const QJsonObject &meta)
                {
                    emit projectMetadataChanged(
                        xjw::gui::project::ProjectTiePointResultService::metadataWithCurrentOnly(
                            meta,
                            currentProjectPath()));
                });
        connect(_projectData, &ProjectData::dirtyStateChanged,
                this, &ProjectManager::metadataDirtyChanged);
    }

    connect(_lifecycleController, &ProjectLifecycleController::projectCreated,
            this, &ProjectManager::projectCreated);
    connect(_lifecycleController, &ProjectLifecycleController::projectOpenStarted,
            this, &ProjectManager::projectOpenStarted);
    connect(_lifecycleController, &ProjectLifecycleController::projectOpenProgressChanged,
            this, &ProjectManager::projectOpenProgressChanged);
    connect(_lifecycleController, &ProjectLifecycleController::projectOpenFinished,
            this, &ProjectManager::projectOpenFinished);
    connect(_lifecycleController, &ProjectLifecycleController::saveStarted,
            this, &ProjectManager::saveStarted);
    connect(_lifecycleController, &ProjectLifecycleController::saveFinished,
            this, &ProjectManager::saveFinished);
    connect(_maskWorkflowController, &ProjectMaskWorkflowController::progressChanged,
            this, &ProjectManager::maskGenerationProgressChanged);
    connect(_maskWorkflowController, &ProjectMaskWorkflowController::finished,
            this, &ProjectManager::maskGenerationFinished);
    connect(_maskWorkflowController, &ProjectMaskWorkflowController::masksGenerated,
            this, &ProjectManager::masksGenerated);
    connect(_maskWorkflowController, &ProjectMaskWorkflowController::projectMetadataUpdated,
            this, &ProjectManager::projectMetadataUpdated);

        connect(_modelManager, &ProjectModelManager::meshProgressChanged,
            this, &ProjectManager::meshProgressChanged);
        connect(_modelManager, &ProjectModelManager::meshProgressFinished,
            this, &ProjectManager::meshProgressFinished);
        connect(_pointCloudWorkflowController,
            &ProjectPointCloudWorkflowController::pointCloudProgressChanged,
            this,
            [this](const QString &stage, int percent)
            {
                if (_automaticModelDepthPreparationActive)
                {
                    emit meshProgressChanged(
                        QStringLiteral("准备深度图：%1").arg(stage),
                        std::clamp(percent * 3 / 5, 0, 59));
                    return;
                }
                emit pointCloudProgressChanged(stage, percent);
            });
        connect(_pointCloudWorkflowController,
            &ProjectPointCloudWorkflowController::pointCloudProgressFinished,
            this,
            [this](bool success)
            {
                if (!_automaticModelDepthPreparationActive)
                {
                    emit pointCloudProgressFinished(success);
                    return;
                }

                _automaticModelDepthPreparationActive = false;
                const bool missing_depth_batch =
                    success && !_pendingAutomaticModelSettings.isEmpty();
                if (!success || missing_depth_batch)
                {
                    _pendingAutomaticModelSettings = QJsonObject();
                    emit meshProgressFinished(false);
                }
            });
        connect(_pointCloudWorkflowController,
            &ProjectPointCloudWorkflowController::pointCloudResultReady,
            this,
            &ProjectManager::pointCloudResultReady);
        connect(_pointCloudWorkflowController,
            &ProjectPointCloudWorkflowController::depthMapBatchReady,
            this,
            [this](const QString &output_directory, int)
            {
                if (_pendingAutomaticModelSettings.isEmpty())
                {
                    return;
                }
                QJsonObject model_settings = _pendingAutomaticModelSettings;
                _pendingAutomaticModelSettings = QJsonObject();
                model_settings[QStringLiteral("source_data")] =
                    QStringLiteral("depth_maps");
                model_settings[QStringLiteral("source_path")] = output_directory;
                model_settings[QStringLiteral("depthMapSourcePath")] =
                    output_directory;
                model_settings[QStringLiteral("reuseDepthMaps")] = true;
                model_settings[QStringLiteral("automatic_depth_maps")] = false;
                model_settings[QStringLiteral("force_depth_recompute")] = false;
                model_settings[QStringLiteral("reconstruction_mode")] =
                    QStringLiteral("depth_tsdf");
                emit meshProgressChanged(
                    QStringLiteral("深度图估计完成，正在生成三维模型..."),
                    60);
                if (!_modelManager->startMeshReconstructionAsync(model_settings))
                {
                    emit meshProgressFinished(false);
                }
            });
        connect(_sparseReconstructionManager,
            &ProjectSparseReconstructionManager::atProgressChanged,
            this, &ProjectManager::atProgressChanged);
        connect(_sparseReconstructionManager,
            &ProjectSparseReconstructionManager::atProgressFinished,
            this, &ProjectManager::atProgressFinished);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::demPipelineProgressChanged,
            this, &ProjectManager::demPipelineProgressChanged);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::demPipelineFinished,
            this, &ProjectManager::demPipelineFinished);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::orthoPipelineStarted,
            this, &ProjectManager::orthoPipelineStarted);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::orthoPipelineProgressChanged,
            this, &ProjectManager::orthoPipelineProgressChanged);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::orthoPipelineFinished,
            this, &ProjectManager::orthoPipelineFinished);

    LOG_INFO(QStringLiteral("ProjectManager 已初始化(精简版)"));
}

ProjectManager::~ProjectManager()
{
}

//==============================================================================
// 项目操作
//==============================================================================

void ProjectManager::createNewProject()
{
    _lifecycleController->createNewProject();
}

void ProjectManager::openProject()
{
    _lifecycleController->openProject();
}

void ProjectManager::openProjectFromPath(const QString &plascanPath)
{
    _lifecycleController->openProjectFromPath(plascanPath);
}

void ProjectManager::saveProject()
{
    _lifecycleController->saveProject();
}

void ProjectManager::closeProject()
{
    _lifecycleController->closeProject();
}

//==============================================================================
// 资源管理
//==============================================================================

void ProjectManager::addPhoto()
{
    if (_uiCommands)
    {
        (void)_uiCommands->addPhoto();
    }
}

void ProjectManager::addFolder()
{
    if (!_uiCommands || !_projectData)
    {
        return;
    }

    QString folder;
    if (!_uiCommands->selectImageFolder(&folder))
    {
        return;
    }

    const auto session = currentSessionContext();
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [folder]() -> ImageFolderScan
        {
            return scanImageFolder(folder);
        },
        [folder, session](ProjectManager *self,
                          xjw::gui::tasks::TaskOutcome<ImageFolderScan> outcome)
        {
            if (!self || !self->_projectData || !self->isCurrentSession(session))
            {
                return;
            }

            if (!outcome.succeeded())
            {
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("添加文件夹失败: %1")
                                          .arg(outcome.errorMessage));
                return;
            }

            ImageFolderScan scan = std::move(*outcome.value);

            if (!scan.success)
            {
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("添加文件夹失败: %1").arg(scan.errorMessage));
                return;
            }

            if (scan.imagePaths.isEmpty())
            {
                QMessageBox::information(self->_parent,
                                         QStringLiteral("提示"),
                                         QStringLiteral("文件夹中没有找到可导入的影像: %1").arg(folder));
                return;
            }

            QString error;
            if (!self->_projectData->addImages(scan.imagePaths, &error))
            {
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("添加文件夹失败: %1").arg(error));
                return;
            }

            LOG_INFO(QStringLiteral("已从文件夹添加 %1 张影像: %2")
                         .arg(scan.imagePaths.size())
                         .arg(folder));
            if (!error.isEmpty())
            {
                QMessageBox::information(self->_parent, QStringLiteral("提示"), error);
            }
        });
}

void ProjectManager::importPointCloud()
{
    importProjectAsset(false);
}

void ProjectManager::importModel()
{
    importProjectAsset(true);
}

void ProjectManager::importProjectAsset(bool modelAsset)
{
    const QString assetName = modelAsset ? QStringLiteral("模型") : QStringLiteral("点云");
    const QString dialogTitle = QStringLiteral("导入 Metashape %1").arg(assetName);
    if (!ensureProjectOpen(QStringLiteral("请先打开或创建项目，再导入%1。").arg(assetName),
                           dialogTitle))
    {
        return;
    }

    const QString dialogKey = modelAsset
        ? QStringLiteral("import_model")
        : QStringLiteral("import_point_cloud");
    const QString filter = modelAsset
        ? QStringLiteral("Metashape/通用模型 (*.obj *.ply);;OBJ 模型 (*.obj);;"
                         "PLY 模型 (*.ply);;所有文件 (*)")
        : QStringLiteral("Metashape/通用点云 (*.obj *.ply *.xyz);;OBJ 点云 (*.obj);;"
                         "PLY 点云 (*.ply);;XYZ 点云 (*.xyz);;所有文件 (*)");
    const QString selectedPath = QFileDialog::getOpenFileName(
        _parent,
        dialogTitle,
        getLastUsedDir(dialogKey),
        filter);
    if (selectedPath.isEmpty())
    {
        return;
    }
    saveLastUsedDir(dialogKey, QFileInfo(selectedPath).absolutePath());

    xjw::common::project::ProjectAssetImportRequest request;
    request.type = modelAsset
        ? xjw::common::project::ProjectAssetType::Model
        : xjw::common::project::ProjectAssetType::PointCloud;
    request.sourcePath = selectedPath;
    request.projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(
        currentProjectPath());
    const auto session = currentSessionContext();

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [request]()
        {
            return xjw::common::project::ProjectAssetImporter::importAsset(request);
        },
        [dialogTitle, modelAsset, session](
            ProjectManager *self,
            xjw::gui::tasks::TaskOutcome<
                xjw::common::project::ProjectAssetImportResult> outcome)
        {
            if (!self || !self->_projectData)
            {
                return;
            }
            if (!self->isCurrentSession(session))
            {
                if (outcome.succeeded() && outcome.value->success)
                {
                    QDir(outcome.value->importDirectory).removeRecursively();
                }
                return;
            }
            if (!outcome.succeeded())
            {
                QMessageBox::critical(self->_parent,
                                      dialogTitle,
                                      QStringLiteral("导入失败: %1")
                                          .arg(outcome.errorMessage));
                return;
            }

            const xjw::common::project::ProjectAssetImportResult result =
                std::move(*outcome.value);
            if (!result.success)
            {
                QMessageBox::critical(self->_parent,
                                      dialogTitle,
                                      result.errorMessage.isEmpty()
                                          ? QStringLiteral("导入失败。")
                                          : result.errorMessage);
                return;
            }

            if (!self->_projectData->upsertResultRecordByPath(
                    result.resultArrayKey,
                    result.resultPathKey,
                    result.projectRecord,
                    true))
            {
                QDir(result.importDirectory).removeRecursively();
                QMessageBox::critical(self->_parent,
                                      dialogTitle,
                                      QStringLiteral("资源已读取，但无法写入项目成果记录。"));
                return;
            }

            QString message = QStringLiteral("已导入 %1\n%2\n顶点/点数: %3")
                                  .arg(modelAsset ? QStringLiteral("模型")
                                                  : QStringLiteral("点云"),
                                       result.importedPath)
                                  .arg(result.vertexCount);
            if (modelAsset)
            {
                message.append(QStringLiteral("\n面数: %1").arg(result.faceCount));
            }
            if (!result.warnings.isEmpty())
            {
                message.append(QStringLiteral("\n\n注意:\n%1")
                                   .arg(result.warnings.join(QLatin1Char('\n'))));
            }
            QMessageBox::information(self->_parent, dialogTitle, message);
            LOG_INFO(QStringLiteral("Metashape %1已导入: %2 -> %3")
                         .arg(modelAsset ? QStringLiteral("模型")
                                         : QStringLiteral("点云"),
                              result.sourcePath,
                              result.importedPath));
        });
}

bool ProjectManager::importCameraForImage(const QString &imagePath)
{
    return _cameraSetupManager && _cameraSetupManager->importCameraForImage(imagePath);
}

void ProjectManager::startTriangulationAsync(const QJsonObject &settings)
{
    if (_sparseReconstructionManager)
    {
        _sparseReconstructionManager->startTriangulationAsync(settings);
    }
}

void ProjectManager::startSparseCloudOutlierRemovalAsync(const QJsonObject &settings)
{
    if (_sparseReconstructionManager)
    {
        _sparseReconstructionManager->startSparseCloudOutlierRemovalAsync(settings);
    }
}

void ProjectManager::startSparseCloudLocalOptimAsync(const QJsonObject &settings)
{
    if (_sparseReconstructionManager)
    {
        _sparseReconstructionManager->startSparseCloudLocalOptimAsync(settings);
    }
}

void ProjectManager::startSparseCloudRefineAsync(const QJsonObject &settings)
{
    if (_sparseReconstructionManager)
    {
        _sparseReconstructionManager->startSparseCloudRefineAsync(settings);
    }
}

bool ProjectManager::importCamerasByFilenameBatch()
{
    return _cameraSetupManager && _cameraSetupManager->importCamerasByFilenameBatch();
}

bool ProjectManager::initializeCamerasFromExifOrDefault(const QJsonObject &settings)
{
    return _cameraSetupManager &&
        _cameraSetupManager->initializeCamerasFromExifOrDefault(settings);
}

bool ProjectManager::initializeCamerasFromIntrinsics(const QJsonObject &settings)
{
    return _cameraSetupManager && _cameraSetupManager->initializeCamerasFromIntrinsics(settings);
}

bool ProjectManager::initializeCameraPosesWithSFM(const QJsonObject &settings)
{
    return _cameraSetupManager && _cameraSetupManager->initializeCameraPosesWithSFM(settings);
}

void ProjectManager::removeResource(const QString &resourcePath)
{
    if (_projectData)
    {
        _projectData->removeResource(resourcePath);
    }
}

void ProjectManager::removeResources(const QStringList &resourcePaths)
{
    if (_projectData)
    {
        _projectData->removeResources(resourcePaths);
    }
}

void ProjectManager::importReferenceDataset()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再导入参考 DEM/LiDAR。")))
    {
        return;
    }

    const QString selected = QFileDialog::getOpenFileName(
        _parent,
        QStringLiteral("导入参考 DEM/LiDAR"),
        getLastUsedDir(QStringLiteral("reference_dataset")),
        QStringLiteral("参考地形/点云 (*.tif *.tiff *.vrt *.las *.laz *.copc *.ply *.xyz *.csv);;"
                       "DEM (*.tif *.tiff *.vrt);;"
                       "LiDAR (*.las *.laz *.copc);;"
                       "点云 (*.ply *.xyz *.csv);;"
                       "所有文件 (*)"));
    if (selected.isEmpty())
    {
        return;
    }

    saveLastUsedDir(QStringLiteral("reference_dataset"), QFileInfo(selected).absolutePath());

    QString error;
    if (!registerReferenceDataset(selected, QString(), QStringLiteral("validation"), &error))
    {
        showWarning(error.isEmpty() ? QStringLiteral("导入参考数据失败。") : error,
                    QStringLiteral("导入参考 DEM/LiDAR"));
        return;
    }

    LOG_INFO(QStringLiteral("参考数据已导入: %1").arg(QFileInfo(selected).fileName()));
}

bool ProjectManager::registerReferenceDataset(const QString &path,
                                              const QString &type,
                                              const QString &role,
                                              QString *errorMsg)
{
    return xjw::gui::project::registerReferenceDataset(_projectData, path, type, role, errorMsg);
}

void ProjectManager::openSurveyControlDialog()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再管理测绘控制点。")))
    {
        return;
    }

    SurveyControlDialog dialog(_parent);
    dialog.setSurveyControlMetadata(_projectData->metadata().value(QStringLiteral("survey_control")).toObject());

    connect(&dialog, &SurveyControlDialog::importCsvRequested, this, [this, &dialog]()
    {
        const QString selected = QFileDialog::getOpenFileName(
            &dialog,
            QStringLiteral("导入测绘控制 CSV"),
            getLastUsedDir(QStringLiteral("survey_control")),
            QStringLiteral("控制点 CSV (*.csv);;所有文件 (*)"));
        if (selected.isEmpty())
        {
            return;
        }

        saveLastUsedDir(QStringLiteral("survey_control"), QFileInfo(selected).absolutePath());
        const auto result = xjw::gui::project::importSurveyControlCsv(_projectData,
                                                                      selected,
                                                                      QString());
        if (!result.imported)
        {
            showWarning(result.errorMessage.isEmpty()
                            ? QStringLiteral("导入测绘控制 CSV 失败。")
                            : result.errorMessage,
                        QStringLiteral("测绘控制"));
            return;
        }

        dialog.setSurveyControlMetadata(_projectData->metadata().value(QStringLiteral("survey_control")).toObject());
        dialog.setStatusMessage(QStringLiteral("已导入：控制点 %1，检查点 %2，比例尺 %3")
                                    .arg(result.controlPointCount)
                                    .arg(result.checkPointCount)
                                    .arg(result.scaleBarCount));
        LOG_INFO(QStringLiteral("测绘控制 CSV 已导入: %1 controls=%2 checks=%3 scale_bars=%4")
                 .arg(QFileInfo(selected).fileName())
                 .arg(result.controlPointCount)
                 .arg(result.checkPointCount)
                 .arg(result.scaleBarCount));
    });

    dialog.exec();
}

void ProjectManager::setActiveImagePath(const QString &imagePath)
{
    _maskWorkflowController->setActiveImagePath(imagePath);
}

void ProjectManager::openGenerateMaskDialog()
{
    _maskWorkflowController->openDialog();
}

void ProjectManager::openGenerateMaskDialogForImages(const QStringList &requestedImages)
{
    _maskWorkflowController->openDialogForImages(requestedImages);
}

void ProjectManager::runReferenceQualityCheck()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再执行点云/DEM 精度检查。")))
    {
        return;
    }

    const auto result = xjw::gui::project::writeReferenceDatasetQualityReport(_projectData);
    if (!result.saved)
    {
        showWarning(result.errorMessage.isEmpty()
                        ? QStringLiteral("生成点云/DEM 精度检查报告失败。")
                        : result.errorMessage,
                    QStringLiteral("点云/DEM 精度检查"));
        return;
    }

    const QString status = result.record.value(QStringLiteral("comparison_available")).toBool()
        ? QStringLiteral("已找到可检查的参考数据与项目成果。")
        : QStringLiteral("报告已生成，但当前缺少参考数据或可比较的项目成果。");
    LOG_INFO(QStringLiteral("参考数据精度检查报告已生成: %1").arg(result.jsonPath));
    QMessageBox::information(_parent,
                             QStringLiteral("点云/DEM 精度检查"),
                             QStringLiteral("%1\nJSON: %2\nCSV: %3")
                                 .arg(status, result.jsonPath, result.csvPath));
}

void ProjectManager::prepareReferenceTerrainBundleAdjust()
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目，再使用参考地形约束重新平差。")))
    {
        return;
    }

    const auto result = xjw::gui::project::writeReferenceTerrainPriorPreflightReport(_projectData);
    if (!result.saved)
    {
        showWarning(result.errorMessage.isEmpty()
                        ? QStringLiteral("生成参考地形平差前置检查报告失败。")
                        : result.errorMessage,
                    QStringLiteral("参考地形约束重新平差"));
        return;
    }

    const bool ready = result.record.value(QStringLiteral("ready")).toBool();
    const QString status = result.record.value(QStringLiteral("status")).toString();
    LOG_INFO(QStringLiteral("参考地形平差前置检查报告已生成: %1 ready=%2")
             .arg(result.jsonPath)
             .arg(ready));
    if (!ready)
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("参考地形约束重新平差"),
                                 QStringLiteral("前置检查未通过：%1。\n请先导入 role=ba_prior 的参考 DEM/LiDAR，并完成正式空三。\nJSON: %2\nCSV: %3")
                                     .arg(status, result.jsonPath, result.csvPath));
        return;
    }

    const QJsonObject meta = _projectData->metadata();
    const QString demPriorPath = firstReferenceDemPriorPath(meta);
    const QString laserPriorPath = firstReferenceLaserPriorPath(meta);
    if (demPriorPath.isEmpty() && laserPriorPath.isEmpty())
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("参考地形约束重新平差"),
                                 QStringLiteral("前置检查通过，但未找到可直接用于 BA 的参考数据。\n"
                                                "请导入 role=ba_prior 的 DEM（GeoTIFF/VRT），或带 LiDAR/点云类型的 PLY 文件。\n"
                                                "JSON: %1\nCSV: %2")
                                     .arg(result.jsonPath, result.csvPath));
        return;
    }

    const QStringList images = getAllImages();
    if (images.size() < 2)
    {
        QMessageBox::warning(_parent,
                             QStringLiteral("参考地形约束重新平差"),
                             QStringLiteral("项目影像少于 2 张，无法执行 BA。"));
        return;
    }

    const QString bundleAdjustDir =
        xjw::common::project::ProjectIO::projectBundleAdjustDir(
            currentProjectPath());
    const QString outputDir = QDir(bundleAdjustDir).filePath(
        QStringLiteral("%1_%2")
            .arg(demPriorPath.isEmpty()
                     ? QStringLiteral("reference_laser")
                     : QStringLiteral("reference_terrain"))
            .arg(QDateTime::currentDateTimeUtc().toString(
                QStringLiteral("yyyyMMdd_HHmmss_zzz"))));

    const QString prompt = demPriorPath.isEmpty()
        ? QStringLiteral("将使用 LiDAR/点云 PLY 作为 BA 点到面 soft prior 重新平差。\n\n"
                         "参考点云: %1\n"
                         "无 normal 字段时: 按水平地形面约束使用\n"
                         "影像数量: %2\n"
                         "输出目录: %3\n\n"
                         "继续执行？")
              .arg(laserPriorPath)
              .arg(images.size())
              .arg(outputDir)
        : QStringLiteral("将使用参考 DEM 作为 BA 高程 soft prior 重新平差。\n\n"
                         "参考 DEM: %1\n"
                         "影像数量: %2\n"
                         "输出目录: %3\n\n"
                         "继续执行？")
              .arg(demPriorPath)
              .arg(images.size())
              .arg(outputDir);

    const QMessageBox::StandardButton choice = QMessageBox::question(
        _parent,
        QStringLiteral("参考地形约束重新平差"),
        prompt,
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::Yes);
    if (choice != QMessageBox::Yes)
    {
        return;
    }

    QJsonObject extra;
    if (demPriorPath.isEmpty())
    {
        extra[QStringLiteral("enable_laser_constraints")] = true;
        extra[QStringLiteral("laser_constraint_cloud_path")] = laserPriorPath;
        extra[QStringLiteral("laser_association_max_distance_m")] = 1.0;
        extra[QStringLiteral("laser_voxel_size_m")] = 0.0;
        extra[QStringLiteral("laser_max_curvature")] = 0.2;
        extra[QStringLiteral("laser_max_samples")] = 500000;
        extra[QStringLiteral("laser_missing_normals_as_height_planes")] = true;
        extra[QStringLiteral("laser_weight")] = 1.0;
        extra[QStringLiteral("laser_huber_delta_m")] =
            result.record.value(QStringLiteral("recommended_huber_delta_m")).toDouble(0.5);
    }
    else
    {
        extra[QStringLiteral("enable_reference_terrain_prior")] = true;
        extra[QStringLiteral("reference_terrain_dem_path")] = demPriorPath;
        extra[QStringLiteral("reference_terrain_sigma_m")] =
            result.record.value(QStringLiteral("recommended_sigma_m")).toDouble(1.0);
        extra[QStringLiteral("reference_terrain_huber_delta_m")] =
            result.record.value(QStringLiteral("recommended_huber_delta_m")).toDouble(0.5);
        extra[QStringLiteral("reference_terrain_max_association_distance_m")] = 2.0;
    }
    extra[QStringLiteral("refine_camera_pose")] = true;
    extra[QStringLiteral("export_run_json")] = true;
    extra[QStringLiteral("export_summary_txt")] = true;
    extra[QStringLiteral("export_camera_csv")] = true;
    extra[QStringLiteral("export_points_csv")] = true;

    if (demPriorPath.isEmpty())
    {
        LOG_INFO(QStringLiteral("LiDAR 点到面 BA soft prior 启动: cloud=%1 images=%2 output=%3")
                 .arg(laserPriorPath)
                 .arg(images.size())
                 .arg(outputDir));
    }
    else
    {
        LOG_INFO(QStringLiteral("参考地形 BA soft prior 启动: dem=%1 images=%2 output=%3")
                 .arg(demPriorPath)
                 .arg(images.size())
                 .arg(outputDir));
    }
    startBundleAdjustAsync(images, outputDir, qMax(1, QThread::idealThreadCount()), false, extra);
}

void ProjectManager::deleteGeneratedData(const QString &section, const QStringList &resourcePaths)
{
    if (!_projectData || resourcePaths.isEmpty())
    {
        return;
    }
    if (section == QStringLiteral("照片"))
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("照片分组不支持删除数据，请使用移除引用。"));
        return;
    }

    const int selectedCount = resourcePaths.size();
    const bool deletingTiePoints =
        section == QStringLiteral("连接点");
    const QString dialogTitle = deletingTiePoints
        ? QStringLiteral("移除连接点")
        : QStringLiteral("删除数据");
    const QMessageBox::StandardButton confirm = QMessageBox::question(
        _parent,
        dialogTitle,
        deletingTiePoints
            ? QStringLiteral("确定移除当前连接点及其关联生成文件吗？此操作不可撤销。")
            : (selectedCount == 1
                   ? QStringLiteral("确定删除所选%1数据及其关联生成文件吗？此操作不可撤销。").arg(section)
                   : QStringLiteral("确定删除所选 %1 项%2数据及其关联生成文件吗？此操作不可撤销。")
                         .arg(selectedCount)
                         .arg(section)),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (confirm != QMessageBox::Yes)
    {
        return;
    }

    if (section == QStringLiteral("连接点"))
    {
        const auto result = xjw::gui::project::ProjectTiePointResultService::deleteAll(_projectData);
        if (!result.success)
        {
            QMessageBox::warning(_parent,
                                 dialogTitle,
                                 QStringLiteral("删除失败：%1").arg(result.errorMessage));
            return;
        }

        refreshReconstructionQualityReport();
        QMessageBox::information(_parent,
                                 dialogTitle,
                                 QStringLiteral("已移除连接点及其关联生成文件。"));
        return;
    }

    const auto cleanupResult = xjw::gui::project::ProjectResourceCleanupService::cleanupGeneratedData(_projectData,
                                                                                                       section,
                                                                                                       resourcePaths);
    if (cleanupResult.unsupportedSection)
    {
        QMessageBox::warning(_parent,
                             QStringLiteral("删除数据"),
                             QStringLiteral("当前分组暂不支持删除数据：%1").arg(section));
        return;
    }

    if (!cleanupResult.success && !cleanupResult.errorMessage.isEmpty())
    {
        QMessageBox::warning(_parent,
                             QStringLiteral("删除数据"),
                             QStringLiteral("删除失败：%1").arg(cleanupResult.errorMessage));
        return;
    }

    if (cleanupResult.noMatchedRecords)
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("未找到可删除的%1数据记录。").arg(section));
        return;
    }

    if (cleanupResult.failedPaths.isEmpty())
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("删除数据"),
                                 QStringLiteral("已删除 %1 项%2数据。").arg(cleanupResult.removedCount).arg(section));
    }
    else
    {
        QMessageBox::warning(_parent,
                             QStringLiteral("删除数据"),
                             QStringLiteral("已移除 %1 项%2数据记录，但以下文件/目录删除失败：\n%3")
                                 .arg(cleanupResult.removedCount)
                                 .arg(section)
                                 .arg(cleanupResult.failedPaths.join(QStringLiteral("\n"))));
    }
}

void ProjectManager::packResource(const QString &resourcePath)
{
    QString err;
    if (_projectData && !_projectData->packResource(resourcePath, &err))
    {
        QMessageBox::warning(_parent, QStringLiteral("提示"), err);
    }
}

// Bundle adjust related APIs removed (core removed).

//==============================================================================
// 设置管理 (委托给ProjectData)
//==============================================================================

QJsonObject ProjectManager::loadUiSettings() const
{
    return _projectData ? _projectData->loadUiSettings() : QJsonObject();
}

void ProjectManager::createChunk()
{
    if (!_projectData || !_projectData->hasProject())
    {
        return;
    }
    bool accepted = false;
    const QString name = QInputDialog::getText(
        _parent,
        QStringLiteral("新建 Chunk"),
        QStringLiteral("名称："),
        QLineEdit::Normal,
        QString(),
        &accepted).trimmed();
    if (!accepted)
    {
        return;
    }
    QString error;
    if (!_projectData->createChunk(name, nullptr, &error))
    {
        QMessageBox::critical(
            _parent,
            QStringLiteral("新建 Chunk 失败"),
            error);
    }
}

void ProjectManager::renameChunk(const QString &chunkId)
{
    if (!_projectData || chunkId.trimmed().isEmpty())
    {
        return;
    }
    QString currentName;
    for (const QJsonValue &value : _projectData->chunks())
    {
        const QJsonObject chunk = value.toObject();
        if (chunk.value(QStringLiteral("id")).toString() == chunkId)
        {
            currentName =
                chunk.value(QStringLiteral("name")).toString();
            break;
        }
    }
    bool accepted = false;
    const QString name = QInputDialog::getText(
        _parent,
        QStringLiteral("重命名 Chunk"),
        QStringLiteral("名称："),
        QLineEdit::Normal,
        currentName,
        &accepted).trimmed();
    if (!accepted)
    {
        return;
    }
    QString error;
    if (!_projectData->renameChunk(chunkId, name, &error))
    {
        QMessageBox::critical(
            _parent,
            QStringLiteral("重命名 Chunk 失败"),
            error);
    }
}

void ProjectManager::removeChunk(const QString &chunkId)
{
    if (!_projectData || chunkId.trimmed().isEmpty())
    {
        return;
    }
    QString chunkName;
    for (const QJsonValue &value : _projectData->chunks())
    {
        const QJsonObject chunk = value.toObject();
        if (chunk.value(QStringLiteral("id")).toString() == chunkId)
        {
            chunkName = chunk.value(QStringLiteral("name")).toString();
            break;
        }
    }
    const QMessageBox::StandardButton answer =
        QMessageBox::warning(
            _parent,
            QStringLiteral("删除 Chunk"),
            QStringLiteral(
                "确定删除“%1”吗？该 Chunk 的影像、处理结果和数字目录都会被删除，此操作不可撤销。")
                .arg(chunkName),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No);
    if (answer != QMessageBox::Yes)
    {
        return;
    }
    QString error;
    if (!_projectData->removeChunk(chunkId, &error))
    {
        QMessageBox::critical(
            _parent,
            QStringLiteral("删除 Chunk 失败"),
            error);
    }
}

void ProjectManager::switchChunk(const QString &chunkId)
{
    if (!_projectData || chunkId.trimmed().isEmpty())
    {
        return;
    }
    QString error;
    if (!_projectData->switchChunk(chunkId, &error))
    {
        QMessageBox::critical(
            _parent,
            QStringLiteral("切换 Chunk 失败"),
            error);
    }
}

void ProjectManager::saveUiSettings(const QJsonObject &settings)
{
    if (_projectData)
    {
        _projectData->saveUiSettings(settings);
    }
}

void ProjectManager::markWorkspaceDirty()
{
    if (_projectData)
    {
        _projectData->markWorkspaceDirty();
    }
}

//==============================================================================
// 查询接口 (委托给ProjectData)
//==============================================================================

bool ProjectManager::isDirty() const
{
    return _projectData
        && _projectData->hasProject()
        && _projectData->isDirty();
}

QString ProjectManager::currentProjectPath() const
{
    return _projectData ? _projectData->currentProjectPath() : QString();
}

xjw::gui::project::ProjectSessionContext ProjectManager::currentSessionContext() const
{
    return {
        currentProjectPath(),
        _projectData ? _projectData->activeChunkId() : QString(),
        _projectSessionGeneration
    };
}

bool ProjectManager::isCurrentSession(
    const xjw::gui::project::ProjectSessionContext &context) const
{
    return context.matches(currentSessionContext());
}

QJsonObject ProjectManager::currentMeta() const
{
    if (!_projectData)
    {
        return {};
    }

    return xjw::gui::project::ProjectTiePointResultService::metadataWithCurrentOnly(
        _projectData->metadata(),
        _projectData->currentProjectPath());
}

QJsonObject ProjectManager::coreProjectMeta() const
{
    return _projectData ? _projectData->coreFilesMeta() : QJsonObject();
}

QStringList ProjectManager::getImagesByCategory(const QString &category) const
{
    return _projectData ? _projectData->getImagesByCategory(category) : QStringList();
}

QStringList ProjectManager::getAllImages() const
{
    return _projectData ? _projectData->getAllImages() : QStringList();
}

QString ProjectManager::findMatchFileForPair(const QString &imgA, const QString &imgB) const
{
    return _projectData ? _projectData->findMatchFile(imgA, imgB) : QString();
}

bool ProjectManager::hasTemporaryMeta() const
{
    return _projectData ? _projectData->hasTemporaryMetadata() : false;
}

void ProjectManager::discardTemporaryMeta()
{
    if (_projectData) {
        _projectData->clearTemporaryMetadata();
    }
}

void ProjectManager::writeMetadataToTempAsync(const QJsonObject &meta, bool markDirty)
{
    if (_projectData) {
        _projectData->updateMetadata(meta, markDirty);
        _projectData->scheduleTemporaryMetadataSave();
    }
}

void ProjectManager::refreshReconstructionQualityReport()
{
    if (!_projectData || !_projectData->hasProject())
    {
        return;
    }

    const auto reportResult =
        xjw::gui::project::writeReconstructionQualityProjectReport(_projectData);
    if (!reportResult.saved && !reportResult.errorMessage.isEmpty())
    {
        LOG_WARN(QStringLiteral("重建质量报告刷新失败: %1").arg(reportResult.errorMessage));
    }
}

void ProjectManager::appendImageMatchResult(const ProjectImageMatchResultRecord &record)
{
    if (!_projectData || record.image.trimmed().isEmpty())
    {
        return;
    }
    _projectData->appendImageMatchResult(record);
    emit imageMatchResultAppended(record.image);
}

void ProjectManager::appendImageMatchResults(
    const QVector<ProjectImageMatchResultRecord> &records)
{
    if (_projectData && !records.isEmpty())
    {
        _projectData->appendImageMatchResults(records);
        for (const ProjectImageMatchResultRecord &record : records)
        {
            emit imageMatchResultAppended(record.image);
        }
    }
}

void ProjectManager::startBundleAdjustAsync(const QStringList &images,
                                            const QString &outputDir,
                                            int threads,
                                            bool dryRun,
                                            const QJsonObject &extraSettings)
{
    // ── 前置检查（仅快速校验，不做任何 IO）───────────────────────────────
    if (!ensureProjectOpen()) return;
    if (images.size() < 2) {
        QMessageBox::warning(_parent, QStringLiteral("提示"), QStringLiteral("至少需要选择两张影像"));
        return;
    }
    if (outputDir.trimmed().isEmpty()) {
        QMessageBox::warning(_parent, QStringLiteral("提示"), QStringLiteral("请指定输出目录"));
        return;
    }

    const QString outDir = QDir::cleanPath(outputDir);

    // ── 只获取核心数据（影像列表/相机，无结果数组，速度极快）──────────────
    // 单影像 `.pimatch` 分片在后台线程直接读取，避免 UI 阻塞。
    const QJsonObject coreData   = _projectData->coreFilesMeta();
    const QString     plascanPath = _projectData->currentProjectPath();

    const int minMatches = qMax(0, extraSettings.value(QStringLiteral("min_matches")).toInt(0));

    // ── 组装 BaServiceOptions（纯参数，不含相机/轨迹——后台填充）──────────
    xjw::gui::BaServiceOptions opts;
    opts.selectedImages   = images;
    opts.outputDir        = outDir;
    opts.dryRun           = dryRun;
    opts.threads          = threads;
    opts.baOpt.maxIterations       = qBound(3,  extraSettings.value(QStringLiteral("max_iterations")).toInt(20),  200);
    opts.baOpt.maxPointIterations  = qBound(1,  extraSettings.value(QStringLiteral("max_point_iterations")).toInt(12), 100);
    opts.baOpt.maxCameraIterations = qBound(1,  extraSettings.value(QStringLiteral("max_camera_iterations")).toInt(10), 100);
    opts.baOpt.refineCameraPose    = extraSettings.value(QStringLiteral("refine_camera_pose")).toBool(true);
    opts.baOpt.huberDelta          = extraSettings.value(QStringLiteral("huber_delta")).toDouble(3.0);
    opts.baOpt.finiteDiffEps       = extraSettings.value(QStringLiteral("finite_diff_eps")).toDouble(1e-6);
    opts.baOpt.damping             = extraSettings.value(QStringLiteral("damping")).toDouble(1e-3);
    opts.baOpt.stepTolerance       = extraSettings.value(QStringLiteral("step_tolerance")).toDouble(1e-8);
    opts.baOpt.numThreads          = threads;
    const QString baBackendName =
        extraSettings.value(QStringLiteral("ba_backend")).toString(QStringLiteral("auto")).trimmed().toLower();
    if (baBackendName == QLatin1String("auto"))
    {
        opts.baOpt.backend = xjw::BABackend::Auto;
    }
    else if (baBackendName == QLatin1String("legacy_cpu"))
    {
        opts.baOpt.backend = xjw::BABackend::LegacyCpu;
    }
    else if (baBackendName == QLatin1String("ceres_cpu"))
    {
        opts.baOpt.backend = xjw::BABackend::CeresCpu;
    }
    else if (baBackendName == QLatin1String("ceres_cuda"))
    {
        opts.baOpt.backend = xjw::BABackend::CeresCuda;
    }
    else if (baBackendName == QLatin1String("native_cuda"))
    {
        opts.baOpt.backend = xjw::BABackend::NativeCuda;
    }
    else
    {
        opts.baOpt.backend = xjw::BABackend::Auto;
    }
    opts.baOpt.ceresCudaDevice = qMax(
        0,
        extraSettings.value(QStringLiteral("ba_cuda_device")).toInt(0));
    opts.baOpt.minCeresCudaCameras = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_cuda_cameras")).toInt(opts.baOpt.minCeresCudaCameras));
    opts.baOpt.minCeresCudaObservations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_cuda_observations")).toInt(
            opts.baOpt.minCeresCudaObservations));
    opts.baOpt.nativeCudaDevice = qMax(
        0,
        extraSettings.value(QStringLiteral("ba_native_cuda_device")).toInt(opts.baOpt.nativeCudaDevice));
    opts.baOpt.nativeCudaMaxPointStepNorm = qMax(
        1e-12,
        extraSettings.value(QStringLiteral("ba_native_cuda_max_point_step")).toDouble(
            opts.baOpt.nativeCudaMaxPointStepNorm));
    opts.baOpt.minCeresCpuObservations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_cpu_observations")).toInt(
            opts.baOpt.minCeresCpuObservations));
    opts.baOpt.maxCeresPointOnlyObservations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_max_ceres_point_only_observations")).toInt(
            opts.baOpt.maxCeresPointOnlyObservations));
    opts.baOpt.maxDenseSchurCameras = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_max_dense_schur_cameras")).toInt(
            opts.baOpt.maxDenseSchurCameras));
    opts.baOpt.maxSparseSchurCameras = qMax(
        opts.baOpt.maxDenseSchurCameras,
        extraSettings.value(QStringLiteral("ba_max_sparse_schur_cameras")).toInt(
            opts.baOpt.maxSparseSchurCameras));
    opts.baOpt.maxCeresCudaMemoryFraction = qBound(
        0.01,
        extraSettings.value(QStringLiteral("ba_max_ceres_cuda_memory_fraction")).toDouble(
            opts.baOpt.maxCeresCudaMemoryFraction),
        1.0);
    opts.baOpt.allowBackendFallback =
        extraSettings.value(QStringLiteral("ba_allow_backend_fallback")).toBool(true);
    opts.baOpt.maxAcceptedConstraintRmsGrowth = qMax(
        1.0,
        extraSettings.value(QStringLiteral("ba_max_accepted_constraint_rms_growth")).toDouble(
            opts.baOpt.maxAcceptedConstraintRmsGrowth));
    opts.baOpt.enableBackendQualityGate =
        extraSettings.value(QStringLiteral("ba_enable_backend_quality_gate")).toBool(true);
    opts.baOpt.maxAcceptedRmsGrowth = qMax(
        0.0,
        extraSettings.value(QStringLiteral("ba_max_accepted_rms_growth")).toDouble(
            opts.baOpt.maxAcceptedRmsGrowth));
    opts.baOpt.minAcceptedValidTrackRatio = qMax(
        0.0,
        extraSettings.value(QStringLiteral("ba_min_accepted_valid_track_ratio")).toDouble(
            opts.baOpt.minAcceptedValidTrackRatio));
    opts.baOpt.compareAutoBackendWithLegacy =
        extraSettings.value(QStringLiteral("ba_compare_auto_backend_with_legacy")).toBool(true);
    opts.baOpt.enablePointFilter    = true;
    opts.baOpt.filterMaxReprojError = extraSettings.value(QStringLiteral("filter_max_reproj_error")).toDouble(2.5);
    opts.baOpt.filterSigmaFactor   = extraSettings.value(QStringLiteral("filter_sigma_factor")).toDouble(3.0);
    opts.exportTsai        = extraSettings.value(QStringLiteral("export_tsai")).toBool(true);
    opts.exportSummaryTxt  = extraSettings.value(QStringLiteral("export_summary_txt")).toBool(true);
    opts.exportPointsCsv   = extraSettings.value(QStringLiteral("export_points_csv")).toBool(true);
    opts.exportCameraCsv   = extraSettings.value(QStringLiteral("export_camera_csv")).toBool(true);
    opts.exportRunJson     = extraSettings.value(QStringLiteral("export_run_json")).toBool(true);
    opts.exportEvalPlot    = extraSettings.value(QStringLiteral("export_eval_plot")).toBool(true);
    opts.enableLaserConstraints = extraSettings.value(QStringLiteral("enable_laser_constraints")).toBool(false);
    opts.laserConstraintCloudPath = extraSettings.value(QStringLiteral("laser_constraint_cloud_path")).toString().trimmed();
    opts.laserAssociationMaxDistanceMeters = qMax(
        0.0,
        extraSettings.value(QStringLiteral("laser_association_max_distance_m")).toDouble(1.0));
    opts.laserVoxelSizeMeters = qMax(
        0.0,
        extraSettings.value(QStringLiteral("laser_voxel_size_m")).toDouble(0.0));
    opts.laserMaxCurvature = qBound(
        0.0,
        extraSettings.value(QStringLiteral("laser_max_curvature")).toDouble(0.2),
        1.0);
    opts.laserMaxSamples = qBound(
        1,
        extraSettings.value(QStringLiteral("laser_max_samples")).toInt(500000),
        10000000);
    opts.laserUseMissingNormalsAsHeightPlanes =
        extraSettings.value(QStringLiteral("laser_missing_normals_as_height_planes")).toBool(false);
    opts.laserWeight = qMax(
        0.0,
        extraSettings.value(QStringLiteral("laser_weight")).toDouble(1.0));
    opts.laserHuberDeltaMeters = qMax(
        1e-9,
        extraSettings.value(QStringLiteral("laser_huber_delta_m")).toDouble(0.2));
    opts.enableReferenceTerrainPrior =
        extraSettings.value(QStringLiteral("enable_reference_terrain_prior")).toBool(false);
    opts.referenceTerrainDemPath =
        extraSettings.value(QStringLiteral("reference_terrain_dem_path")).toString().trimmed();
    opts.referenceTerrainSigmaMeters = qMax(
        1e-9,
        extraSettings.value(QStringLiteral("reference_terrain_sigma_m")).toDouble(1.0));
    opts.referenceTerrainMaxAssociationDistanceMeters = qMax(
        0.0,
        extraSettings.value(QStringLiteral("reference_terrain_max_association_distance_m")).toDouble(2.0));
    opts.referenceTerrainHuberDeltaMeters = qMax(
        1e-9,
        extraSettings.value(QStringLiteral("reference_terrain_huber_delta_m")).toDouble(0.5));

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    setAtCancelFlag(cancelFlag);
    opts.baOpt.cancelFlag = cancelFlag;
    QPointer<ProjectManager> baProgressSelf(this);
    opts.baOpt.progressCallback =
        [baProgressSelf, cancelFlag](int currentIteration, int maxIterations, double avgRms, int validPoints) -> bool
        {
            if (!baProgressSelf ||
                cancelFlag->load(std::memory_order_relaxed))
            {
                return false;
            }

            const int safeMaxIterations = std::max(1, maxIterations);
            const int percent = qBound(
                10,
                10 + static_cast<int>(std::lround(80.0 * currentIteration / safeMaxIterations)),
                90);
            const QString stage = QStringLiteral("光束法平差优化中... %1/%2 RMS=%3 有效点=%4")
                .arg(currentIteration)
                .arg(safeMaxIterations)
                .arg(avgRms, 0, 'f', 4)
                .arg(validPoints);
            QMetaObject::invokeMethod(
                baProgressSelf.data(),
                [baProgressSelf, cancelFlag, stage, percent]()
                {
                    if (!baProgressSelf)
                    {
                        return;
                    }
                    if (baProgressSelf->_atCancelFlag != cancelFlag ||
                        cancelFlag->load(std::memory_order_relaxed))
                    {
                        return;
                    }
                    emit baProgressSelf->atProgressChanged(stage, percent);
                },
                Qt::QueuedConnection);
            return true;
        };

    emit atProgressChanged(QStringLiteral("光束法平差准备中..."), 1);

    LOG_INFO(QStringLiteral("BA: 特征/匹配准备完毕，启动光束法平差"));

    QPointer<ProjectManager> self(this);

xjw::gui::tasks::runGuardedWithOutcome(
    this,
    [self, coreData, plascanPath, images, minMatches,
     cancelFlag, opts = std::move(opts), isDryRun = dryRun]() mutable
    {
        auto finishTask = [self, cancelFlag](bool success)
        {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(
                self.data(),
                [self, cancelFlag, success]()
                {
                    if (!self)
                    {
                        return;
                    }
                    if (self->_atCancelFlag != cancelFlag)
                    {
                        return;
                    }
                    self->_atCancelFlag.reset();
                    emit self->atProgressFinished(success);
                },
                Qt::QueuedConnection);
        };

        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(
            self.data(),
            [self, cancelFlag]()
            {
                if (!self)
                {
                    return;
                }
                if (self->_atCancelFlag != cancelFlag ||
                    cancelFlag->load(std::memory_order_relaxed))
                {
                    return;
                }
                emit self->atProgressChanged(QStringLiteral("光束法平差构建输入..."), 5);
            },
            Qt::QueuedConnection);

        const BundleAdjustExecutionResult executionResult = runBundleAdjustExecution(coreData,
                                                                                     plascanPath,
                                                                                     images,
                                                                                     minMatches,
                                                                                     std::move(opts));

        if (cancelFlag->load(std::memory_order_relaxed))
        {
            LOG_INFO(QStringLiteral("BA: 用户取消了光束法平差"));
            if (self)
            {
                QMetaObject::invokeMethod(self.data(),
                [self, cancelFlag]()
                {
                    if (!self)
                    {
                        return;
                    }
                    if (self->_atCancelFlag == cancelFlag)
                    {
                        emit self->bundleAdjustPreviewReady(QJsonObject());
                    }
                },
                Qt::QueuedConnection);
            }
            finishTask(false);
            return;
        }

        if (executionResult.buildStatus != xjw::core::project::BaInputBuildStatus::Ok) {
            if (!self)
            {
                return;
            }
            QMetaObject::invokeMethod(self.data(),
                [self, cancelFlag, buildStatus = executionResult.buildStatus]() {
                    if (!self)
                    {
                        return;
                    }
                    if (self->_atCancelFlag != cancelFlag)
                    {
                        return;
                    }
                    const QString msg =
                        buildStatus == xjw::core::project::BaInputBuildStatus::NotEnoughCameras
                        ? QStringLiteral("所选影像中可用相机参数不足（至少需要两台相机）")
                        : QStringLiteral("未找到可用于光束法平差的匹配点（请检查选中影像是否已有匹配结果）");
                    QMessageBox::warning(self->_parent, QStringLiteral("提示"), msg);
                    emit self->bundleAdjustPreviewReady(QJsonObject());
                    self->_atCancelFlag.reset();
                    emit self->atProgressFinished(false);
                },
                Qt::QueuedConnection);
            return;
        }

        LOG_INFO(
            QStringLiteral("BA: 启动后台平差计算 (%1 台相机, %2 条轨迹)...")
                .arg(executionResult.serviceResult.pendingCamUpdates.size())
                .arg(executionResult.serviceResult.resultJson.value(QStringLiteral("track_count")).toInt()));

        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(self.data(),
            [self,
             cancelFlag,
             baResult = executionResult.serviceResult,
             beforeCamMeta = executionResult.beforeCamMeta,
             isDryRun]()
            {
                if (!self)
                {
                    return;
                }
                if (self->_atCancelFlag != cancelFlag)
                {
                    return;
                }
                if (cancelFlag->load(std::memory_order_relaxed))
                {
                    emit self->bundleAdjustPreviewReady(QJsonObject());
                    self->_atCancelFlag.reset();
                    emit self->atProgressFinished(false);
                    return;
                }

                emit self->atProgressChanged(QStringLiteral("光束法平差整理结果..."), 95);

                if (!baResult.success && !isDryRun) {
                    QMessageBox::warning(self->_parent,
                        QStringLiteral("平差提示"),
                        QStringLiteral("光束法平差执行出现问题：%1").arg(baResult.errorMessage));
                }
                self->_pendingBaBeforeCameraMeta = beforeCamMeta;
                self->_pendingBaCameraMeta  = baResult.pendingCamUpdates;
                self->_pendingBaResult      = baResult.resultJson;
                self->_hasPendingBaPreview  = !self->_pendingBaCameraMeta.isEmpty();
                emit self->bundleAdjustPreviewReady(baResult.resultJson);
                self->_atCancelFlag.reset();
                emit self->atProgressFinished(baResult.success);
            },
            Qt::QueuedConnection);
    },
    [cancelFlag](ProjectManager *manager,
                 xjw::gui::tasks::TaskOutcome<void> outcome)
    {
        if (outcome.succeeded() || manager->_atCancelFlag != cancelFlag)
        {
            return;
        }

        manager->_atCancelFlag.reset();
        emit manager->atProgressFinished(false);
        QMessageBox::warning(manager->_parent,
                             QStringLiteral("光束法平差"),
                             outcome.errorMessage);
    });
}

// ==============================================================================
// 空中三角测量相关
// ==============================================================================
bool ProjectManager::setImageCameras(const QMap<QString, QJsonObject> &cameras,
                                     int    *updatedCount,
                                     QString *errorMsg)
{
    // 直接委托给数据层——ProjectManager 不持有算法逻辑，仅转发
    if (!_projectData)
    {
        if (errorMsg) *errorMsg = QStringLiteral("ProjectData 未初始化");
        if (updatedCount) *updatedCount = 0;
        return false;
    }
    return _projectData->setImageCameras(cameras, updatedCount, errorMsg);
}

bool ProjectManager::replaceImageCameras(const QStringList &targetImagePaths,
                                         const QMap<QString, QJsonObject> &cameras,
                                         int *updatedCount,
                                         int *clearedCount,
                                         QString *errorMsg)
{
    if (!_projectData)
    {
        if (errorMsg) *errorMsg = QStringLiteral("ProjectData 未初始化");
        if (updatedCount) *updatedCount = 0;
        if (clearedCount) *clearedCount = 0;
        return false;
    }
    return _projectData->replaceImageCameras(targetImagePaths,
                                             cameras,
                                             updatedCount,
                                             clearedCount,
                                             errorMsg);
}

bool ProjectManager::clearImageCameras(const QStringList &imagePaths,
                                       int    *updatedCount,
                                       QString *errorMsg)
{
    if (!_projectData) {
        if (errorMsg) *errorMsg = QStringLiteral("ProjectData 未初始化");
        if (updatedCount) *updatedCount = 0;
        return false;
    }
    return _projectData->clearImageCameras(imagePaths, updatedCount, errorMsg);
}

QMap<QString, xjw::Camera> ProjectManager::getCamerasForImages(
        const QStringList &images,
        bool *hasCamerasForAll) const
{
    if (hasCamerasForAll) *hasCamerasForAll = true;

    QMap<QString, xjw::Camera> result;
    if (!_projectData)
    {
        if (hasCamerasForAll) *hasCamerasForAll = false;
        return result;
    }

    // 从运行时元数据中建立路径 → 影像元数据索引，再按需解析相机。
    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::common::project::projectImageMetaByPath(projectFilesMeta(_projectData), true);

    for (const QString &imgPath : images)
    {
        const QString norm = normalizePath(imgPath);
        const QJsonObject imageMeta = imageMetaByPath.value(norm);
        if (imageMeta.isEmpty())
        {
            if (hasCamerasForAll) *hasCamerasForAll = false;
            continue;
        }

        xjw::Camera cam;
        if (!xjw::common::project::imageCameraFromEntry(imageMeta, &cam))
        {
            if (hasCamerasForAll) *hasCamerasForAll = false;
            continue;
        }

        result.insert(norm, cam);
    }

    // 若有任意请求影像缺少相机，则 hasCamerasForAll 已在循环内置 false
    if (hasCamerasForAll && result.size() != images.size())
        *hasCamerasForAll = false;

    return result;
}

void ProjectManager::startGenerateModelAsync()
{
    if (_modelManager)
    {
        _modelManager->startGenerateModelAsync();
    }
}

void ProjectManager::startGenerateModelAsync(const QJsonObject &settings)
{
    if (!_modelManager)
    {
        return;
    }

    const QString source_data =
        settings.value(QStringLiteral("source_data")).toString();
    const QString depth_source =
        settings.value(QStringLiteral("depthMapSourcePath"))
            .toString(settings.value(QStringLiteral("source_path")).toString())
            .trimmed();
    const bool reuse_depth_maps =
        settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
    const bool force_depth_recompute =
        settings.value(QStringLiteral("force_depth_recompute")).toBool(false);
    const bool prepare_depth_maps =
        source_data == QStringLiteral("depth_maps") &&
        (settings.value(QStringLiteral("automatic_depth_maps")).toBool(false) ||
         force_depth_recompute ||
         !reuse_depth_maps ||
         depth_source.isEmpty());
    if (!prepare_depth_maps)
    {
        _modelManager->startMeshReconstructionAsync(settings);
        return;
    }
    if (!_pointCloudWorkflowController)
    {
        emit meshProgressFinished(false);
        return;
    }
    if (!_pendingAutomaticModelSettings.isEmpty())
    {
        return;
    }

    _pendingAutomaticModelSettings = settings;
    _automaticModelDepthPreparationActive = true;
    QJsonObject depth_settings = settings;
    depth_settings[QStringLiteral("reuseDepthMaps")] = false;
    depth_settings[QStringLiteral("force_depth_recompute")] = true;
    depth_settings[QStringLiteral("depthFilterMode")] =
        settings.value(QStringLiteral("depthFiltering"))
            .toString(QStringLiteral("mild"));
    depth_settings[QStringLiteral("calculatePointColors")] = false;
    depth_settings[QStringLiteral("replaceDefaultPointCloud")] = false;
    if (!_pointCloudWorkflowController->startDepthMapsOnlyAsync(depth_settings))
    {
        _pendingAutomaticModelSettings = QJsonObject();
        _automaticModelDepthPreparationActive = false;
        emit meshProgressFinished(false);
    }
}

void ProjectManager::startCreatePointCloudAsync(const QJsonObject &settings)
{
    if (_pointCloudWorkflowController)
    {
        _pointCloudWorkflowController->startCreatePointCloudAsync(settings);
    }
}

void ProjectManager::startMeshReconstructionAsync(const QJsonObject &settings)
{
    if (_modelManager)
    {
        _modelManager->startMeshReconstructionAsync(settings);
    }
}

void ProjectManager::startTextureMappingAsync(const QJsonObject &settings)
{
    if (_modelManager)
    {
        _modelManager->startTextureMappingAsync(settings);
    }
}

void ProjectManager::startDemFromPointCloudAsync(
    const xjw::gui::project::DemGenerationRequest &request)
{
    if (_terrainProductsManager)
    {
        _terrainProductsManager->startDemFromPointCloudAsync(request);
    }
}

void ProjectManager::startMapProjectAsync(
    const xjw::gui::project::OrthoGenerationRequest &request)
{
    if (_terrainProductsManager)
    {
        _terrainProductsManager->startMapProjectAsync(request);
    }
}

void ProjectManager::cancelMapProject()
{
    if (_terrainProductsManager)
    {
        _terrainProductsManager->cancelMapProject();
    }
}

bool ProjectManager::acceptBundleAdjustPreview(QString *errorMsg)
{
    if (!_hasPendingBaPreview || _pendingBaCameraMeta.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("当前没有可应用的平差预览结果");
        return false;
    }

    const QStringList allImages = _projectData ? _projectData->getAllImages() : QStringList();
    const auto commitResult = commitBundleAdjustPreview(_projectData,
                                                        _pendingBaCameraMeta,
                                                        _pendingBaResult);
    if (!commitResult.success) {
        if (errorMsg) *errorMsg = commitResult.errorMessage;
        return false;
    }
    if (!commitResult.warningMessage.isEmpty()) {
        LOG_WARN(commitResult.warningMessage);
    }

    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(currentProjectPath());
    const QString baOutputDir = assetsDir.isEmpty()
        ? QString()
        : QDir(assetsDir).filePath(
              QStringLiteral("aerial_triangulation/ba_refined_%1")
                  .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss"))));
    const auto artifactsResult = finalizeBundleAdjustArtifacts(
        assetsDir,
        _pendingBaResult,
        allImages,
        _pendingBaResult.value(QStringLiteral("output_dir")).toString(),
        QStringLiteral("reconstruction_bundle_adjust"),
        _pendingBaBeforeCameraMeta,
        _pendingBaCameraMeta,
        baOutputDir,
        true);
    if (!artifactsResult.reportWarning.isEmpty())
    {
        LOG_WARN(QStringLiteral("BA: %1").arg(artifactsResult.reportWarning));
    }
    if (artifactsResult.sparseCloudExport.exported)
    {
        replaceTiePointResult(artifactsResult.sparseCloudExport.sparseCloudPath,
                              artifactsResult.sparseCloudExport.pointCount,
                              allImages,
                              artifactsResult.sparseCloudExport.outputDir,
                              artifactsResult.sparseCloudExport.extraRecord);
        LOG_INFO(QStringLiteral("BA 精化点云已写入 AT 结果：%1 个点，路径=%2")
                     .arg(artifactsResult.sparseCloudExport.pointCount)
                     .arg(artifactsResult.sparseCloudExport.sparseCloudPath));
    }
    else if (!artifactsResult.sparseCloudExport.errorMessage.isEmpty())
    {
        LOG_WARN(QStringLiteral("写入 BA 点云 sidecar 失败: %1")
                     .arg(artifactsResult.sparseCloudExport.errorMessage));
    }

    _pendingBaCameraMeta.clear();
    _pendingBaBeforeCameraMeta.clear();
    _pendingBaResult = QJsonObject();
    _hasPendingBaPreview = false;

    QMessageBox::information(_parent,
                             QStringLiteral("光束法平差"),
                             QStringLiteral("已保留本次平差结果，并更新 %1 台相机参数。")
                                 .arg(commitResult.updatedCameraCount));
    return true;
}

void ProjectManager::discardBundleAdjustPreview()
{
    _pendingBaCameraMeta.clear();
    _pendingBaBeforeCameraMeta.clear();
    _pendingBaResult = QJsonObject();
    _hasPendingBaPreview = false;
}

void ProjectManager::applyBundleAdjustForAt(const QString     &assetsDir,
                                            const QStringList &images,
                                            const QString     &outputDir,
                                            const QMap<QString, QJsonObject> &beforeCameras)
{
    bool success = false;
    if (_hasPendingBaPreview && !_pendingBaCameraMeta.isEmpty())
    {
        const auto commitResult = commitBundleAdjustPreview(_projectData,
                                                            _pendingBaCameraMeta,
                                                            _pendingBaResult);
        if (commitResult.success)
        {
            LOG_INFO(
                QStringLiteral("BA(AT): 已更新 %1 台相机参数").arg(commitResult.updatedCameraCount));
            success = true;
        } 
        else 
        {
            LOG_WARN(
                commitResult.errorMessage);
        }
        if (!commitResult.warningMessage.isEmpty())
        {
            LOG_WARN(QStringLiteral("BA(AT): %1").arg(commitResult.warningMessage));
        }
    } 
    else 
    {
        LOG_WARN(
            QStringLiteral("BA(AT): 没有待应用的平差预览结果，可能是相机或匹配点不足"));
    }

    const auto artifactsResult = finalizeBundleAdjustArtifacts(assetsDir,
                                                               _pendingBaResult,
                                                               images,
                                                               outputDir,
                                                               QStringLiteral("workflow_aerial_triangulation"),
                                                               beforeCameras,
                                                               _pendingBaCameraMeta,
                                                               outputDir,
                                                               false);
    if (!artifactsResult.reportWarning.isEmpty())
    {
        LOG_WARN(QStringLiteral("BA(AT): %1").arg(artifactsResult.reportWarning));
    }

    if (artifactsResult.sparseCloudExport.exported)
    {
        replaceTiePointResult(artifactsResult.sparseCloudExport.sparseCloudPath,
                              artifactsResult.sparseCloudExport.pointCount,
                              images,
                              artifactsResult.sparseCloudExport.outputDir,
                              artifactsResult.sparseCloudExport.extraRecord);
    }
    else if (!artifactsResult.sparseCloudExport.errorMessage.isEmpty())
    {
        LOG_WARN(QStringLiteral("BA(AT): 无法导出稀疏点云文件: %1")
                     .arg(artifactsResult.sparseCloudExport.errorMessage));
    }

    // ── 4. 清理预览缓存 ────────────────────────────────────────────────────
    _pendingBaCameraMeta.clear();
    _pendingBaBeforeCameraMeta.clear();
    _pendingBaResult    = QJsonObject();
    _hasPendingBaPreview = false;

    // ── 5. 发出空三完成信号 ────────────────────────────────────────────────
    emit atProgressFinished(success);
}

bool ProjectManager::appendIntersectionResult(const QJsonObject &result, QString *errorMsg)
{
    return _projectData ? _projectData->appendIntersectionResult(result, errorMsg) : false;
}

QJsonArray ProjectManager::intersectionResults() const
{
    return _projectData ? _projectData->getIntersectionResults() : QJsonArray();
}

//==============================================================================
// FileDialogStateManager
//==============================================================================

void ProjectManager::setFileDialogStateManager(FileDialogStateManager *manager)
{
    _fileDialogState = manager;
}

QString ProjectManager::getLastUsedDir(const QString &key) const
{
    if (!_fileDialogState) return QDir::homePath();
    return _fileDialogState->lastDir(key);
}

void ProjectManager::saveLastUsedDir(const QString &key, const QString &dir)
{
    if (_fileDialogState) {
        _fileDialogState->setLastDir(key, dir);
    }
}

void ProjectManager::showWarning(const QString &message, const QString &title) const
{
    // 统一 warning 出口，后续若切换提示组件只需改这里。
    QMessageBox::warning(_parent, title, message);
}

bool ProjectManager::ensureProjectOpen(const QString &message, const QString &title) const
{
    // 将“项目是否打开”校验统一收口，避免重复 if 与文案分散。
    if (_projectData && _projectData->hasProject()) return true;
    showWarning(message, title);
    return false;
}

bool ProjectManager::replaceTiePointResult(const QString &sparseCloudPath,
                                           int sparsePointCount,
                                           const QStringList &selectedImages,
                                           const QString &outputDir,
                                           const QJsonObject &extraRecord)
{
    const auto result = xjw::gui::project::replaceTiePointResult(_projectData,
                                                                 sparseCloudPath,
                                                                 sparsePointCount,
                                                                 selectedImages,
                                                                 outputDir,
                                                                 extraRecord);
    if (!result.success)
    {
        LOG_ERROR(QStringLiteral("替换当前连接点失败: %1").arg(result.errorMessage));
        showWarning(result.errorMessage, QStringLiteral("连接点写入失败"));
        return false;
    }
    if (!result.cleanupWarnings.isEmpty())
    {
        LOG_WARN(QStringLiteral("当前连接点已更新，但旧文件清理失败: %1")
                     .arg(result.cleanupWarnings.join(QStringLiteral("；"))));
    }
    LOG_INFO(QStringLiteral("空三代次已更新为 %1；旧深度图、稠密点云、模型、DEM 和正射结果已失效")
                 .arg(result.reconstructionGenerationId));
    refreshReconstructionQualityReport();
    return true;
}

void ProjectManager::appendObsNetResult(int nodeCount,
                                        int edgeCount,
                                        const QString &algorithmName,
                                        const QJsonObject &extraInfo)
{
    xjw::gui::project::appendObsNetResult(_projectData,
                                          nodeCount,
                                          edgeCount,
                                          algorithmName,
                                          extraInfo);
}

// ── 追加空三（SFM）结果到 aerial_triangulation_results ─────────────────
QJsonArray ProjectManager::getAvailableAtResults() const
{
    return _sparseReconstructionManager
        ? _sparseReconstructionManager->getAvailableAtResults()
        : QJsonArray();
}

void ProjectManager::cancelModelGeneration()
{
    if (!_pendingAutomaticModelSettings.isEmpty() &&
        _pointCloudWorkflowController)
    {
        _pointCloudWorkflowController->cancelActiveTask();
        return;
    }
    if (_modelManager)
    {
        _modelManager->cancelActiveTask();
    }
}

void ProjectManager::cancelPointCloudGeneration()
{
    if (_pointCloudWorkflowController)
    {
        _pointCloudWorkflowController->cancelActiveTask();
    }
}

// ── 取消正在运行的 AT/SFM 任务 ──────────────────────────────────────────────
void ProjectManager::cancelAt()
{
    if (_atCancelFlag)
    {
        _atCancelFlag->store(true);
        qDebug() << "[AT/BA] 已请求取消";
    }
}

// ── 取消正在运行的照片蒙版生成任务 ───────────────────────────────────────────
void ProjectManager::cancelMaskGeneration()
{
    _maskWorkflowController->cancelActiveTask();
}
