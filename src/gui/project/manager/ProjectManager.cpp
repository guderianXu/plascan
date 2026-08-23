#include "ProjectManager.h"
#include "ProjectModelManager.h"
#include "ProjectPointCloudWorkflowController.h"
#include "ProjectLifecycleController.h"
#include "ProjectMaskWorkflowController.h"
#include "ProjectSparseReconstructionManager.h"
#include "ProjectTerrainProductsManager.h"
#include "ProjectCameraSetupManager.h"
#include "ProjectUiCommands.h"
#include "project/ProjectSessionModel.h"
#include "project/ProjectAssetImporter.h"
#include "project/ProjectIO.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "ProjectCameraImportService.h"
#include "project/ProjectSharedImageStore.h"
#include "ProjectBundleAdjustExecution.h"
#include "ProjectBundleAdjustWorkflow.h"
#include "ProjectCameraInitialization.h"
#include "ProjectResourceCleanup.h"
#include "ProjectTiePointResultService.h"

#include "ProjectMetadataOperations.h"
#include "ProjectOpenGuard.h"
#include "ProjectModelWorkflowPolicy.h"
#include "ProjectSfmWorkflow.h"
#include "ProjectSparseWorkflow.h"
#include "ProjectResultRecords.h"
#include "ReferenceDatasetWorkflow.h"
#include "ProjectSurveyControl.h"
#include "ProjectWorkflowOperations.h"
#include "ProjectWorkflowReports.h"
#include "PointCloudWorkflowConfig.h"
#include "camera/SurveyControlDialog.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "filtering/SparsePointCloudProcessor.h"
#include "FileDialogStateManager.h"
#include "FramePinholeCamera.h"
#include "Intersection.h"
#include "BundleAdjust.h"
#include "LaserConstraintMap.h"
#include "PlanetaryLaserJson.h"
#include "io/PathIO.h"


#include <QMessageBox>
#include <QInputDialog>
#include <QLineEdit>
#include <QDir>
#include <QFileDialog>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPointer>
#include <QPushButton>
#include <QThread>
#include <cmath>
#include <QFile>
#include <QSet>
#include <QFileInfo>
#include <QTextStream>
#include <QDateTime>
#include <QImage>
#include <QThreadPool>
#include <QtConcurrent/QtConcurrentMap>

#include <algorithm>
#include <atomic>
#include <array>
#include <limits>
#include <memory>
#include <mutex>

using xjw::common::project::cameraFromJson;
using xjw::common::project::cameraToJson;
using xjw::gui::project::BundleAdjustExecutionResult;
using xjw::gui::project::buildBundleAdjustPreviewPresentation;
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
using xjw::core::project::resolveSparsePointContext;
using xjw::gui::project::requireOpenProject;
using xjw::gui::project::runBundleAdjustExecution;
using xjw::gui::project::runSparsePointWorkflow;
using xjw::gui::project::replaceProjectRecordWithLatest;
using xjw::core::project::SparsePointContext;
using xjw::core::project::SparsePointOperationResult;
using xjw::gui::project::SparsePointWorkflowKind;
using xjw::gui::project::SparsePointWorkflowSpec;
using xjw::core::project::sparseOperationDisplayName;
using xjw::gui::project::sparsePointWorkflowSpec;
using xjw::core::project::findLatestAtResultIndex;
using xjw::gui::project::upsertProjectRecordByPath;
using xjw::gui::project::upsertMetaArrayRecordByPath;
using xjw::gui::project::withPreparedCameras;
using xjw::core::project::writeJsonObjectFile;

namespace
{

QString normalizedDepthBatchDirectory(const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return QString();
    }
    const QFileInfo info(path);
    return QDir::cleanPath(
        (info.exists() && info.isDir() ? info : QFileInfo(info.absolutePath()))
            .absoluteFilePath());
}

QString storedDepthBatchQualityProfile(const QJsonObject &metadata,
                                       const QString &source_path)
{
    const QString requested_directory = normalizedDepthBatchDirectory(source_path);
    const QJsonArray records = metadata.value(
        QStringLiteral("depth_map_results")).toArray();
    for (int index = records.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = records.at(index).toObject();
        QString record_path = record.value(
            QStringLiteral("mvs_output_dir")).toString();
        if (record_path.isEmpty())
        {
            record_path = record.value(
                QStringLiteral("raw_depth_path")).toString();
        }
        if (!requested_directory.isEmpty() &&
            normalizedDepthBatchDirectory(record_path).compare(
                requested_directory, Qt::CaseInsensitive) != 0)
        {
            continue;
        }
        return record.value(QStringLiteral("quality_profile")).toString();
    }
    return QString();
}

struct ImageFolderScan
{
    bool success = false;
    QString folder;
    QString errorMessage;
    QStringList imagePaths;
};

struct ImageImportBatch
{
    bool success = false;
    QString errorMessage;
    QStringList projectImagePaths;
    int skipped = 0;
};

struct ImageImportItem
{
    bool success = false;
    QString errorMessage;
    QString projectImagePath;
};

ImageImportBatch importImagesToSharedStore(
    const QString &projectPath,
    const QStringList &imagePaths,
    QSet<QString> existingPaths,
    const std::function<void(int, int)> &progress)
{
    ImageImportBatch batch;
    batch.projectImagePaths.reserve(imagePaths.size());
    const int total = imagePaths.size();
    const int progressStep = std::max(1, total / 200);
    const int idealThreads = std::max(1, QThread::idealThreadCount());
    QThreadPool importPool;
    importPool.setMaxThreadCount(std::clamp((idealThreads + 1) / 2, 2, 8));
    int completed = 0;
    std::mutex progressMutex;

    const QList<ImageImportItem> imported = QtConcurrent::blockingMapped(
        &importPool,
        imagePaths,
        [projectPath, total, progressStep, progress, &completed, &progressMutex](
            const QString &imagePath)
        {
            ImageImportItem item;
            QString resourceUri;
            xjw::common::project::ProjectSharedImageStore store(projectPath);
            item.success = store.importImage(imagePath,
                                             &resourceUri,
                                             &item.projectImagePath,
                                             &item.errorMessage);

            std::lock_guard<std::mutex> lock(progressMutex);
            ++completed;
            if (progress && (completed == total || completed % progressStep == 0))
            {
                progress(completed, total);
            }
            return item;
        });

    QStringList successfulReservations;
    successfulReservations.reserve(imported.size());
    QString firstImportError;
    bool importFailed = false;
    for (const ImageImportItem &item : imported)
    {
        if (item.success)
        {
            successfulReservations.append(item.projectImagePath);
        }
        else
        {
            importFailed = true;
            if (firstImportError.isEmpty())
            {
                firstImportError = item.errorMessage;
            }
        }
    }
    if (importFailed)
    {
        batch.errorMessage = firstImportError.isEmpty()
            ? QStringLiteral("共享影像导入失败")
            : firstImportError;
        xjw::common::project::ProjectSharedImageStore(projectPath)
            .releaseReservations(successfulReservations);
        return batch;
    }

    xjw::common::project::ProjectSharedImageStore sharedImageStore(projectPath);
    for (const ImageImportItem &item : imported)
    {
        if (existingPaths.contains(item.projectImagePath))
        {
            ++batch.skipped;
            sharedImageStore.releaseReservations({item.projectImagePath});
        }
        else
        {
            existingPaths.insert(item.projectImagePath);
            batch.projectImagePaths.append(item.projectImagePath);
        }
    }

    batch.success = true;
    return batch;
}

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
        QStringLiteral("*.JPEG"),
        QStringLiteral("*.img"),
        QStringLiteral("*.IMG"),
        QStringLiteral("*.cub"),
        QStringLiteral("*.CUB")
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

bool validateBaPriorImport(const QString &path, QString *errorMessage)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix == QLatin1String("tif")
        || suffix == QLatin1String("tiff")
        || suffix == QLatin1String("vrt"))
    {
        return true;
    }
    if (suffix == QLatin1String("json"))
    {
        xjw::lidar::PlanetaryLaserDataset dataset;
        std::string loadError;
        if (!xjw::lidar::loadPlanetaryLaserJsonFile(
                xjw::common::io::toUtf8Path(path), {}, &dataset, &loadError))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "无法作为行星激光测距数据导入: %1\n"
                    "PlaScan SI JSON 必须显式包含 target/body_fixed_frame/time/units。"
                    "ISIS LidarData JSON 本身缺少这些上下文，请改用 bundle_adjust_cli 的 "
                    "--laser-range-isis-* 参数，或先转换为 PlaScan SI JSON。")
                                    .arg(QString::fromStdString(loadError));
            }
            return false;
        }
        return true;
    }
    if (suffix != QLatin1String("ply"))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "当前 BA 软约束直接支持 DEM、扫描点云 PLY 或行星激光 SI JSON。\n"
                "LAS/LAZ/COPC/XYZ/CSV 请先转换为与影像工程同坐标系的带法向 PLY。");
        }
        return false;
    }
    xjw::lidar::LaserConstraintMapOptions options;
    options.maxSamples = 256;
    options.useMissingNormalsAsHeightPlanes = false;
    options.sampleInputBeforeFiltering = true;
    xjw::lidar::LaserConstraintMap map;
    std::string loadError;
    if (!map.loadPly(xjw::common::io::toUtf8Path(path), options, &loadError))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "该 PLY 没有可用的有限非零表面法向。\n"
                "需要 vertex 的 normal_x/normal_y/normal_z（或 nx/ny/nz）字段；"
                "环绕目标不能按水平面代替法向。\n解析信息: %1")
                                .arg(QString::fromStdString(loadError));
        }
        return false;
    }
    return true;
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

QString firstPlanetaryLaserPriorPath(const QJsonObject &meta)
{
    const QJsonArray references = meta.value(QStringLiteral("reference_datasets")).toArray();
    for (const QJsonValue &value : references)
    {
        const QJsonObject reference = value.toObject();
        if (!isBaPriorRole(reference.value(QStringLiteral("role")).toString()) ||
            reference.value(QStringLiteral("type")).toString().toLower() !=
                QLatin1String("planetary_laser_shots"))
        {
            continue;
        }
        const QString path = reference.value(QStringLiteral("path")).toString().trimmed();
        if (!path.isEmpty())
        {
            return path;
        }
    }
    return {};
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
        xjw::core::project::ProjectResourceCleanupService::
            installAutomaticRecovery(_projectData);
        const auto advanceSessionGeneration = [this]()
        {
            if (_modelManager)
            {
                _modelManager->cancelActiveTask();
            }
            if (_atCancelFlag)
            {
                _atCancelFlag->store(true, std::memory_order_relaxed);
                _atCancelFlag.reset();
            }
            ++_projectSessionGeneration;
            discardBundleAdjustPreview();
            emit projectSessionChanged();
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
            this,
            [this](const QString &projectPath)
            {
                if (_modelManager)
                {
                    _modelManager->cancelActiveTask();
                }
                emit projectOpenStarted(projectPath);
            });
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
        connect(_sparseReconstructionManager,
            &ProjectSparseReconstructionManager::tiePointResultReady,
            this, &ProjectManager::tiePointResultReady);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::demPipelineProgressChanged,
            this, &ProjectManager::demPipelineProgressChanged);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::demPipelineFinished,
            this, &ProjectManager::demPipelineFinished);
        connect(_terrainProductsManager,
            &ProjectTerrainProductsManager::backgroundTaskProgressChanged,
            this,
            &ProjectManager::backgroundTaskProgressChanged);
        connect(_terrainProductsManager, &ProjectTerrainProductsManager::backgroundTaskFinished,
            this, &ProjectManager::backgroundTaskFinished);
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
    waitForResourceCleanup();
}

void ProjectManager::waitForResourceCleanup()
{
    if (!_resourceCleanupRunning)
    {
        return;
    }
    _resourceCleanupFuture.waitForFinished();
    if (_resourceCleanupShutdownFinalize)
    {
        _resourceCleanupShutdownFinalize();
        _resourceCleanupShutdownFinalize = {};
    }
    _resourceCleanupRunning = false;
}

//==============================================================================
// 项目操作
//==============================================================================

void ProjectManager::createNewProject()
{
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("新建项目")))
    {
        return;
    }
    _lifecycleController->createNewProject();
}

void ProjectManager::openProject()
{
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("打开项目")))
    {
        return;
    }
    _lifecycleController->openProject();
}

void ProjectManager::openProjectFromPath(const QString &plascanPath)
{
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("打开项目")))
    {
        return;
    }
    _lifecycleController->openProjectFromPath(plascanPath);
}

void ProjectManager::saveProject()
{
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("保存项目")))
    {
        return;
    }
    _lifecycleController->saveProject();
}

void ProjectManager::closeProject()
{
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("关闭项目")))
    {
        return;
    }
    if (_modelManager)
    {
        _modelManager->cancelActiveTask();
    }
    _lifecycleController->closeProject();
}

//==============================================================================
// 资源管理
//==============================================================================

void ProjectManager::addPhoto()
{
    if (!_uiCommands || !_projectData)
    {
        return;
    }

    QStringList files;
    if (_uiCommands->selectPhotos(&files))
    {
        startImageImport(files, QStringLiteral("所选文件"));
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

    if (_imageImportActive)
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("提示"),
                                 QStringLiteral("已有影像加载任务正在运行，请等待完成后再试。"));
        return;
    }

    _imageImportActive = true;
    emit imageImportProgressChanged(QStringLiteral("正在扫描影像文件夹..."), 0, 0);

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
            if (!self || !self->_projectData)
            {
                return;
            }

            self->_imageImportActive = false;

            if (!self->isCurrentSession(session))
            {
                emit self->imageImportFinished(false, QStringLiteral("项目已切换，影像加载已停止"));
                return;
            }

            if (!outcome.succeeded())
            {
                emit self->imageImportFinished(false, outcome.errorMessage);
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("添加文件夹失败: %1")
                                          .arg(outcome.errorMessage));
                return;
            }

            ImageFolderScan scan = std::move(*outcome.value);

            if (!scan.success)
            {
                emit self->imageImportFinished(false, scan.errorMessage);
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("添加文件夹失败: %1").arg(scan.errorMessage));
                return;
            }

            if (scan.imagePaths.isEmpty())
            {
                emit self->imageImportFinished(true, QStringLiteral("文件夹中没有找到可导入的影像"));
                QMessageBox::information(self->_parent,
                                         QStringLiteral("提示"),
                                         QStringLiteral("文件夹中没有找到可导入的影像: %1").arg(folder));
                return;
            }

            self->startImageImport(scan.imagePaths, folder);
        });
}

void ProjectManager::startImageImport(const QStringList &imagePaths,
                                      const QString &sourceLabel)
{
    if (!_projectData || imagePaths.isEmpty())
    {
        return;
    }
    if (_imageImportActive)
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("提示"),
                                 QStringLiteral("已有影像加载任务正在运行，请等待完成后再试。"));
        return;
    }

    _imageImportActive = true;
    const auto session = currentSessionContext();
    const QString projectPath = currentProjectPath();
    const QStringList currentImages = _projectData->getAllImages();
    const QSet<QString> existingPaths(currentImages.cbegin(), currentImages.cend());
    const QPointer<ProjectManager> owner(this);
    const int total = imagePaths.size();
    emit imageImportProgressChanged(QStringLiteral("正在加载影像..."), 0, total);

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [owner, projectPath, imagePaths, existingPaths, session]()
        {
            return importImagesToSharedStore(
                projectPath,
                imagePaths,
                existingPaths,
                [owner, session](int done, int progressTotal)
                {
                    if (!owner)
                    {
                        return;
                    }
                    xjw::gui::tasks::postGuarded(
                        owner,
                        [session, done, progressTotal](ProjectManager *self)
                        {
                            if (self->_imageImportActive && self->isCurrentSession(session))
                            {
                                emit self->imageImportProgressChanged(
                                    QStringLiteral("正在加载影像..."), done, progressTotal);
                            }
                        });
                });
        },
        [session, sourceLabel, total](
            ProjectManager *self,
            xjw::gui::tasks::TaskOutcome<ImageImportBatch> outcome)
        {
            self->_imageImportActive = false;
            if (!self->isCurrentSession(session))
            {
                emit self->imageImportFinished(false, QStringLiteral("项目已切换，影像加载已停止"));
                return;
            }

            if (!outcome.succeeded())
            {
                emit self->imageImportFinished(false, outcome.errorMessage);
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("加载影像失败: %1").arg(outcome.errorMessage));
                return;
            }

            ImageImportBatch batch = std::move(*outcome.value);
            if (!batch.success)
            {
                emit self->imageImportFinished(false, batch.errorMessage);
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("加载影像失败: %1").arg(batch.errorMessage));
                return;
            }

            QString message;
            if (!self->_projectData->addImagesFromSharedStore(
                    batch.projectImagePaths, batch.skipped, &message))
            {
                emit self->imageImportFinished(false, message);
                QMessageBox::critical(self->_parent,
                                      QStringLiteral("错误"),
                                      QStringLiteral("提交影像元数据失败: %1").arg(message));
                return;
            }

            const int added = batch.projectImagePaths.size();
            const QString finishedMessage = batch.skipped > 0
                ? QStringLiteral("已加载 %1 张影像，跳过 %2 张重复影像")
                      .arg(added)
                      .arg(batch.skipped)
                : QStringLiteral("已加载 %1 张影像").arg(added);
            LOG_INFO(QStringLiteral("%1（来源：%2，共选择 %3 张）")
                         .arg(finishedMessage, sourceLabel)
                         .arg(total));
            emit self->imageImportFinished(true, finishedMessage);
            if (!message.isEmpty())
            {
                QMessageBox::information(self->_parent, QStringLiteral("提示"), message);
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
    if (!requireOpenProject(_projectData,
                            _parent,
                            QStringLiteral("请先打开或创建项目，再导入%1。").arg(assetName),
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

void ProjectManager::removeResources(const QStringList &resourcePaths)
{
    if (_projectData)
    {
        _projectData->removeResources(resourcePaths);
    }
}

void ProjectManager::importReferenceDataset()
{
    if (!requireOpenProject(_projectData,
                            _parent,
                            QStringLiteral("请先打开项目，再导入参考 DEM/LiDAR。")))
    {
        return;
    }

    const QString selected = QFileDialog::getOpenFileName(
        _parent,
        QStringLiteral("导入参考 DEM/LiDAR"),
        getLastUsedDir(QStringLiteral("reference_dataset")),
        QStringLiteral("参考地形/点云/行星激光 (*.tif *.tiff *.vrt *.las *.laz *.copc *.ply *.xyz *.csv *.json);;"
                       "DEM (*.tif *.tiff *.vrt);;"
                       "LiDAR (*.las *.laz *.copc);;"
                       "行星激光测距 shot (*.json);;"
                       "点云 (*.ply *.xyz *.csv);;"
                       "所有文件 (*)"));
    if (selected.isEmpty())
    {
        return;
    }

    saveLastUsedDir(QStringLiteral("reference_dataset"), QFileInfo(selected).absolutePath());

    const QString validationPurpose = QStringLiteral("仅用于精度检查（validation）");
    const QString baPurpose = QStringLiteral("用于光束法平差软约束（ba_prior）");
    bool purposeAccepted = false;
    const QString purpose = QInputDialog::getItem(
        _parent,
        QStringLiteral("选择参考数据用途"),
        QStringLiteral("该参考数据如何参与项目？"),
        QStringList{validationPurpose, baPurpose},
        0,
        false,
        &purposeAccepted);
    if (!purposeAccepted)
    {
        return;
    }
    const QString role = purpose == baPurpose
        ? QStringLiteral("ba_prior")
        : QStringLiteral("validation");

    QString compatibilityError;
    if (role == QLatin1String("ba_prior")
        && !validateBaPriorImport(selected, &compatibilityError))
    {
        showWarning(compatibilityError, QStringLiteral("导入 BA 参考约束"));
        return;
    }

    QString error;
    if (!registerReferenceDataset(selected, QString(), role, &error))
    {
        showWarning(error.isEmpty() ? QStringLiteral("导入参考数据失败。") : error,
                    QStringLiteral("导入参考 DEM/LiDAR"));
        return;
    }

    LOG_INFO(QStringLiteral("参考数据已导入: %1 role=%2")
             .arg(QFileInfo(selected).fileName(), role));
}

bool ProjectManager::registerReferenceDataset(const QString &path,
                                              const QString &type,
                                              const QString &role,
                                              QString *errorMsg)
{
    return xjw::core::project::registerReferenceDataset(_projectData, path, type, role, errorMsg);
}

void ProjectManager::openSurveyControlDialog()
{
    if (!requireOpenProject(_projectData,
                            _parent,
                            QStringLiteral("请先打开项目，再管理测绘控制点。")))
    {
        return;
    }

    SurveyControlDialog dialog(_parent);
    QString metadataError;
    dialog.setSurveyControlMetadata(
        xjw::gui::project::surveyControlDialogMetadata(_projectData, &metadataError));
    if (!metadataError.isEmpty())
    {
        dialog.setStatusMessage(metadataError);
    }

    connect(&dialog, &SurveyControlDialog::importCsvRequested, this, [this, &dialog]()
    {
        const QString selected = QFileDialog::getOpenFileName(
            &dialog,
            QStringLiteral("导入测绘控制 CSV"),
            getLastUsedDir(QStringLiteral("survey_control")),
            QStringLiteral("控制点数据 (*.csv *.txt);;CSV 文件 (*.csv);;所有文件 (*)"));
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

        dialog.setSurveyControlMetadata(
            xjw::gui::project::surveyControlDialogMetadata(_projectData));
        dialog.setStatusMessage(QStringLiteral("已导入：控制点 %1，检查点 %2，比例尺 %3")
                                    .arg(result.controlPointCount)
                                    .arg(result.checkPointCount)
                                    .arg(result.scaleBarCount));
        emit surveyControlChanged();
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

void ProjectManager::clearMasksForImages(const QStringList &requestedImages)
{
    _maskWorkflowController->clearMasksForImages(requestedImages);
}

void ProjectManager::runReferenceQualityCheck()
{
    if (!requireOpenProject(_projectData,
                            _parent,
                            QStringLiteral("请先打开项目，再执行点云/DEM 精度检查。")))
    {
        return;
    }

    const auto result = xjw::core::project::writeReferenceDatasetQualityReport(_projectData);
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
    if (!requireOpenProject(_projectData,
                            _parent,
                            QStringLiteral("请先打开项目，再使用参考地形约束重新平差。")))
    {
        return;
    }

    const auto result = xjw::core::project::writeReferenceTerrainPriorPreflightReport(_projectData);
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
                                 QStringLiteral(
                                     "前置检查未通过：%1。\n"
                                     "请先导入 role=ba_prior 的参考 DEM/LiDAR，并完成正式空三。\n"
                                     "JSON: %2\nCSV: %3")
                                     .arg(status, result.jsonPath, result.csvPath));
        return;
    }

    const QJsonObject meta = _projectData->metadata();
    const QString planetaryLaserPriorPath = firstPlanetaryLaserPriorPath(meta);
    const QString demPriorPath = firstReferenceDemPriorPath(meta);
    const QString laserPriorPath = firstReferenceLaserPriorPath(meta);
    const bool usePlanetaryLaser = !planetaryLaserPriorPath.isEmpty();
    const bool useDem = !usePlanetaryLaser && !demPriorPath.isEmpty();
    const bool useLaserSurface = !usePlanetaryLaser && !useDem && !laserPriorPath.isEmpty();
    if (!usePlanetaryLaser && !useDem && !useLaserSurface)
    {
        QMessageBox::information(_parent,
                                 QStringLiteral("参考地形约束重新平差"),
                                 QStringLiteral("前置检查通过，但未找到可直接用于 BA 的参考数据。\n"
                                                "请导入 role=ba_prior 的 DEM、带法向扫描点云 PLY，"
                                                "或行星激光测距 SI JSON。\n"
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
            .arg(usePlanetaryLaser
                     ? QStringLiteral("planetary_laser_range")
                     : (useLaserSurface
                            ? QStringLiteral("reference_laser_surface")
                            : QStringLiteral("reference_terrain")))
            .arg(QDateTime::currentDateTimeUtc().toString(
                QStringLiteral("yyyyMMdd_HHmmss_zzz"))));

    double laserAssociationDistanceMeters = 0.05;
    double laserSigmaMeters = 0.0025;
    double laserHuberDeltaMeters = 0.05;
    xjw::lidar::PlanetaryLaserDataset planetaryLaserDataset;
    QString planetaryCameraCoordinateFrame;
    QString planetaryCameraSensorFrame;
    bool confirmUnknownSensorIsFrame = false;
    bool confirmUnknownRangeIsOneWay = false;
    if (usePlanetaryLaser)
    {
        std::string laserError;
        if (!xjw::lidar::loadPlanetaryLaserJsonFile(
                xjw::common::io::toUtf8Path(planetaryLaserPriorPath),
                {},
                &planetaryLaserDataset,
                &laserError))
        {
            showWarning(
                QStringLiteral("读取行星激光 SI JSON 失败: %1")
                    .arg(QString::fromStdString(laserError)),
                QStringLiteral("行星激光测距平差"));
            return;
        }
        if (planetaryLaserDataset.sensorModel ==
            xjw::lidar::PlanetaryLaserSensorModel::LineScan)
        {
            QMessageBox::information(
                _parent,
                QStringLiteral("行星激光测距平差"),
                QStringLiteral(
                    "该数据声明为 line_scan。当前 PlaScan 只有每幅影像一个静态位姿，"
                    "尚未实现 ISIS/SPICE 的逐行时变轨迹，因而拒绝按 frame camera 误处理。"));
            return;
        }
        if (planetaryLaserDataset.rangeType ==
            xjw::lidar::PlanetaryLaserRangeType::RoundTrip)
        {
            showWarning(
                QStringLiteral("该数据 range_type=round_trip，请先按产品定义换算为单程几何距离。"),
                QStringLiteral("行星激光测距平差"));
            return;
        }
        if (planetaryLaserDataset.sensorModel ==
            xjw::lidar::PlanetaryLaserSensorModel::Unknown)
        {
            confirmUnknownSensorIsFrame = QMessageBox::question(
                _parent,
                QStringLiteral("确认相机模型"),
                QStringLiteral(
                    "数据没有明确 sensor_model。只有当每个 simultaneous image 可由一个"
                    "静态 frame-camera 位姿代表时才能继续。确认按 frame camera 处理？"),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) == QMessageBox::Yes;
            if (!confirmUnknownSensorIsFrame)
            {
                return;
            }
        }
        if (planetaryLaserDataset.rangeType ==
            xjw::lidar::PlanetaryLaserRangeType::Unknown)
        {
            confirmUnknownRangeIsOneWay = QMessageBox::question(
                _parent,
                QStringLiteral("确认测距语义"),
                QStringLiteral(
                    "数据没有明确 range_type。确认 range_m 已经是激光发射中心到落点的"
                    "单程几何距离，而不是往返光程或未换算时间？"),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) == QMessageBox::Yes;
            if (!confirmUnknownRangeIsOneWay)
            {
                return;
            }
        }

        bool frameAccepted = false;
        const QString datasetFrame = QString::fromStdString(
            planetaryLaserDataset.reference.bodyFixedFrame);
        planetaryCameraCoordinateFrame = QInputDialog::getText(
            _parent,
            QStringLiteral("确认求解坐标系"),
            QStringLiteral(
                "请输入当前 BA 相机中心和普通 tracks 所在坐标系。\n"
                "当前 MVP 不执行坐标转换，必须与激光 body_fixed_frame 完全一致："),
            QLineEdit::Normal,
            datasetFrame,
            &frameAccepted).trimmed();
        if (!frameAccepted || planetaryCameraCoordinateFrame.isEmpty())
        {
            return;
        }
        if (planetaryCameraCoordinateFrame != datasetFrame)
        {
            showWarning(
                QStringLiteral("相机坐标系 %1 与激光坐标系 %2 不一致；请先完成坐标转换。")
                    .arg(planetaryCameraCoordinateFrame, datasetFrame),
                QStringLiteral("行星激光测距平差"));
            return;
        }

        const bool hasNonZeroLeverArm = std::any_of(
            planetaryLaserDataset.shots.begin(),
            planetaryLaserDataset.shots.end(),
            [](const xjw::lidar::PlanetaryLaserShot &shot)
            {
                return std::hypot(
                    std::hypot(shot.leverArmSensorMeters[0], shot.leverArmSensorMeters[1]),
                    shot.leverArmSensorMeters[2]) > 0.0;
            });
        planetaryCameraSensorFrame = QString::fromStdString(
            planetaryLaserDataset.reference.laserFrame);
        if (hasNonZeroLeverArm)
        {
            bool sensorFrameAccepted = false;
            planetaryCameraSensorFrame = QInputDialog::getText(
                _parent,
                QStringLiteral("确认杆臂坐标系"),
                QStringLiteral(
                    "数据包含非零 lever arm。请输入杆臂所用相机/传感器坐标框架；"
                    "当前 MVP 不执行框架旋转："),
                QLineEdit::Normal,
                planetaryCameraSensorFrame,
                &sensorFrameAccepted).trimmed();
            if (!sensorFrameAccepted ||
                planetaryCameraSensorFrame != QString::fromStdString(
                    planetaryLaserDataset.reference.laserFrame))
            {
                showWarning(
                    QStringLiteral("非零杆臂框架不一致，无法安全建立测距方程。"),
                    QStringLiteral("行星激光测距平差"));
                return;
            }
        }
    }
    if (useLaserSurface)
    {
        bool accepted = false;
        laserAssociationDistanceMeters = QInputDialog::getDouble(
            _parent,
            QStringLiteral("LiDAR 平差参数"),
            QStringLiteral("track 到 LiDAR 的最大关联距离（米）:"),
            laserAssociationDistanceMeters,
            0.0001,
            1000.0,
            4,
            &accepted);
        if (!accepted)
        {
            return;
        }
        laserSigmaMeters = QInputDialog::getDouble(
            _parent,
            QStringLiteral("LiDAR 平差参数"),
            QStringLiteral("LiDAR 点到面标准差 sigma（米，权重=1/sigma²）:"),
            laserSigmaMeters,
            0.0001,
            1000.0,
            4,
            &accepted);
        if (!accepted)
        {
            return;
        }
        laserHuberDeltaMeters = QInputDialog::getDouble(
            _parent,
            QStringLiteral("LiDAR 平差参数"),
            QStringLiteral("LiDAR Huber 阈值（米）:"),
            laserHuberDeltaMeters,
            0.0001,
            1000.0,
            4,
            &accepted);
        if (!accepted)
        {
            return;
        }
    }

    QString prompt;
    QString dialogTitle = QStringLiteral("参考地形约束重新平差");
    if (usePlanetaryLaser)
    {
        dialogTitle = QStringLiteral("行星激光测距平差");
        prompt = QStringLiteral(
            "将使用 ISIS 风格的稀疏 laser-range shot 约束 frame-camera BA。\n\n"
            "数据: %1\n"
            "目标/坐标系: %2 / %3\n"
            "shot 数量: %4\n"
            "sensor/range: %5 / %6\n"
            "Huber: 3 sigma\n"
            "影像数量: %7\n"
            "输出目录: %8\n\n"
            "Projected/virtual image measures 不会作为真实像点；普通 track 的 RMS 也不会"
            "混入 shot 统计。继续执行？")
                     .arg(planetaryLaserPriorPath)
                     .arg(QString::fromStdString(planetaryLaserDataset.reference.targetName))
                     .arg(QString::fromStdString(
                         planetaryLaserDataset.reference.bodyFixedFrame))
                     .arg(static_cast<int>(planetaryLaserDataset.shots.size()))
                     .arg(QString::fromLatin1(xjw::lidar::planetaryLaserSensorModelName(
                         planetaryLaserDataset.sensorModel)))
                     .arg(QString::fromLatin1(xjw::lidar::planetaryLaserRangeTypeName(
                         planetaryLaserDataset.rangeType)))
                     .arg(images.size())
                     .arg(outputDir);
    }
    else if (useLaserSurface)
    {
        prompt = QStringLiteral(
            "将使用扫描 LiDAR/点云 PLY 作为 BA 点到面 soft prior 重新平差。\n\n"
            "参考点云: %1\n"
            "法向字段: normal_x/normal_y/normal_z（或 nx/ny/nz）\n"
            "最大关联距离: %2 m\n"
            "标准差 sigma: %3 m（统计权重 %4）\n"
            "Huber 阈值: %5 m\n"
            "影像数量: %6\n"
            "输出目录: %7\n\n"
            "继续执行？")
                     .arg(laserPriorPath)
                     .arg(laserAssociationDistanceMeters, 0, 'g', 8)
                     .arg(laserSigmaMeters, 0, 'g', 8)
                     .arg(1.0 / (laserSigmaMeters * laserSigmaMeters), 0, 'g', 8)
                     .arg(laserHuberDeltaMeters, 0, 'g', 8)
                     .arg(images.size())
                     .arg(outputDir);
    }
    else
    {
        prompt = QStringLiteral("将使用参考 DEM 作为 BA 高程 soft prior 重新平差。\n\n"
                                "参考 DEM: %1\n"
                                "影像数量: %2\n"
                                "输出目录: %3\n\n"
                                "继续执行？")
                     .arg(demPriorPath)
                     .arg(images.size())
                     .arg(outputDir);
    }

    const QMessageBox::StandardButton choice = QMessageBox::question(
        _parent,
        dialogTitle,
        prompt,
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::Yes);
    if (choice != QMessageBox::Yes)
    {
        return;
    }

    QJsonObject extra;
    if (usePlanetaryLaser)
    {
        extra[QStringLiteral("enable_planetary_laser_range_constraints")] = true;
        extra[QStringLiteral("planetary_laser_data_path")] = planetaryLaserPriorPath;
        extra[QStringLiteral("planetary_laser_camera_coordinate_frame")] =
            planetaryCameraCoordinateFrame;
        extra[QStringLiteral("planetary_laser_camera_sensor_frame")] =
            planetaryCameraSensorFrame;
        extra[QStringLiteral("planetary_laser_confirm_unknown_sensor_is_frame")] =
            confirmUnknownSensorIsFrame;
        extra[QStringLiteral("planetary_laser_confirm_unknown_range_is_one_way")] =
            confirmUnknownRangeIsOneWay;
        extra[QStringLiteral("planetary_laser_allow_unmapped_shots")] = false;
        extra[QStringLiteral("planetary_laser_allow_unmapped_measured_images")] = false;
        extra[QStringLiteral("planetary_laser_range_weight")] = 1.0;
        extra[QStringLiteral("planetary_laser_range_huber_delta_sigma")] = 3.0;
    }
    else if (useLaserSurface)
    {
        extra[QStringLiteral("enable_laser_constraints")] = true;
        extra[QStringLiteral("laser_constraint_cloud_path")] = laserPriorPath;
        extra[QStringLiteral("laser_association_max_distance_m")] =
            laserAssociationDistanceMeters;
        extra[QStringLiteral("laser_voxel_size_m")] = 0.0;
        extra[QStringLiteral("laser_max_curvature")] = 0.2;
        extra[QStringLiteral("laser_max_samples")] = 500000;
        extra[QStringLiteral("laser_missing_normals_as_height_planes")] = false;
        extra[QStringLiteral("laser_weight")] = 0.0;
        extra[QStringLiteral("laser_sigma_m")] = laserSigmaMeters;
        extra[QStringLiteral("laser_huber_delta_m")] = laserHuberDeltaMeters;
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

    if (usePlanetaryLaser)
    {
        LOG_INFO(QStringLiteral(
            "行星激光测距 BA 启动: shots=%1 data=%2 frame=%3 images=%4 output=%5")
                     .arg(static_cast<int>(planetaryLaserDataset.shots.size()))
                     .arg(planetaryLaserPriorPath, planetaryCameraCoordinateFrame)
                     .arg(images.size())
                     .arg(outputDir));
    }
    else if (useLaserSurface)
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
    if (_resourceCleanupRunning)
    {
        QMessageBox::information(
            _parent,
            QStringLiteral("删除数据"),
            QStringLiteral("已有资源清理任务正在运行，请等待完成后再试。"));
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

    const auto presentResult = [this, section](
                                   const xjw::core::project::ResourceCleanupResult &cleanupResult)
    {
        if (cleanupResult.unsupportedSection)
        {
            QMessageBox::warning(
                _parent,
                QStringLiteral("删除数据"),
                QStringLiteral("当前分组暂不支持删除数据：%1").arg(section));
            return;
        }

        if (!cleanupResult.success && !cleanupResult.errorMessage.isEmpty())
        {
            QMessageBox::warning(
                _parent,
                QStringLiteral("删除数据"),
                QStringLiteral("删除失败：%1")
                    .arg(cleanupResult.errorMessage));
            return;
        }

        if (cleanupResult.noMatchedRecords)
        {
            QMessageBox::information(
                _parent,
                QStringLiteral("删除数据"),
                QStringLiteral("未找到可删除的%1数据记录。")
                    .arg(section));
            return;
        }

        if (cleanupResult.failedPaths.isEmpty()
            && cleanupResult.errorMessage.isEmpty())
        {
            QMessageBox::information(
                _parent,
                QStringLiteral("删除数据"),
                QStringLiteral("已删除 %1 项%2数据。")
                    .arg(cleanupResult.removedCount)
                    .arg(section));
            return;
        }

        QString detail = cleanupResult.errorMessage;
        if (!cleanupResult.failedPaths.isEmpty())
        {
            if (!detail.isEmpty())
            {
                detail += QLatin1Char('\n');
            }
            detail += cleanupResult.failedPaths.join(QStringLiteral("\n"));
        }
        QMessageBox::warning(
            _parent,
            QStringLiteral("删除数据"),
            QStringLiteral(
                "已移除 %1 项%2数据记录，但部分物理清理将在后续重试：\n%3")
                .arg(cleanupResult.removedCount)
                .arg(section)
                .arg(detail));
    };

    _resourceCleanupRunning = true;
    QPointer<QWidget> requestWidget(qobject_cast<QWidget *>(sender()));
    if (requestWidget)
    {
        requestWidget->setEnabled(false);
    }

    const auto prepared =
        xjw::core::project::ProjectResourceCleanupService::
            prepareGeneratedDataCleanup(_projectData, section, resourcePaths);
    if (!prepared.requiresExecution())
    {
        _resourceCleanupRunning = false;
        if (requestWidget)
        {
            requestWidget->setEnabled(true);
        }
        presentResult(prepared.preparationResult());
        return;
    }

    const auto session = currentSessionContext();
    const QString taskId = QStringLiteral("resource_cleanup");
    auto shutdownResult = std::make_shared<
        xjw::core::project::ResourceCleanupResult>(
            prepared.preparationResult());
    const QPointer<ProjectData> cleanupProjectData(_projectData);
    _resourceCleanupShutdownFinalize =
        [cleanupProjectData,
         prepared,
         requestWidget,
         shutdownResult]()
        {
            if (cleanupProjectData)
            {
                xjw::core::project::ProjectResourceCleanupService::
                    finalizePreparedCleanup(
                        cleanupProjectData.data(),
                        prepared,
                        *shutdownResult);
            }
            if (requestWidget)
            {
                requestWidget->setEnabled(true);
            }
        };
    emit backgroundTaskProgressChanged(taskId, 0, 0);
    _resourceCleanupFuture = xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [prepared, shutdownResult]()
        {
            const auto result =
                xjw::core::project::ProjectResourceCleanupService::
                executePreparedCleanup(prepared);
            *shutdownResult = result;
            return result;
        },
        [prepared, presentResult, requestWidget, session, taskId](
            ProjectManager *self,
            xjw::gui::tasks::TaskOutcome<
                xjw::core::project::ResourceCleanupResult> outcome) mutable
        {
            xjw::core::project::ResourceCleanupResult cleanupResult =
                prepared.preparationResult();
            if (outcome.succeeded())
            {
                cleanupResult = std::move(*outcome.value);
            }
            else
            {
                cleanupResult.success = false;
                cleanupResult.errorMessage = outcome.errorMessage;
            }

            const bool finalized =
                xjw::core::project::ProjectResourceCleanupService::
                    finalizePreparedCleanup(
                        self->_projectData, prepared, cleanupResult);
            self->_resourceCleanupShutdownFinalize = {};
            self->_resourceCleanupRunning = false;
            if (requestWidget)
            {
                requestWidget->setEnabled(true);
            }
            emit self->backgroundTaskFinished(taskId);
            if (!self->isCurrentSession(session) || !finalized)
            {
                LOG_INFO(QStringLiteral(
                    "资源清理完成时项目会话或持久化代次已变化，忽略旧任务 UI 结果"));
                return;
            }
            presentResult(cleanupResult);
        });
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
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("新建 Chunk")))
    {
        return;
    }
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
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("重命名 Chunk")))
    {
        return;
    }
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
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("删除 Chunk")))
    {
        return;
    }
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
    if (rejectLifecycleChangeDuringResourceCleanup(
            QStringLiteral("切换 Chunk")))
    {
        return;
    }
    if (!_projectData || chunkId.trimmed().isEmpty())
    {
        return;
    }
    if (_projectData->activeChunkId() == chunkId.trimmed())
    {
        return;
    }
    if (_modelManager)
    {
        _modelManager->cancelActiveTask();
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

bool ProjectManager::rejectLifecycleChangeDuringResourceCleanup(
    const QString &operation) const
{
    if (!_resourceCleanupRunning)
    {
        return false;
    }
    QMessageBox::information(
        _parent,
        QStringLiteral("资源清理进行中"),
        QStringLiteral("资源清理任务完成前无法%1，请稍候。")
            .arg(operation));
    return true;
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

bool ProjectManager::isModelGenerationRunning() const
{
    return _modelManager && _modelManager->isRunning();
}

QJsonObject ProjectManager::currentMeta() const
{
    if (!_projectData)
    {
        return {};
    }

    return xjw::gui::project::ProjectTiePointResultService::metadataWithCurrentOnly(
        _projectData->metadataIncludingResults(),
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

void ProjectManager::discardTemporaryMeta()
{
    if (_projectData) {
        _projectData->clearTemporaryMetadata();
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
    if (!requireOpenProject(_projectData, _parent)) return;
    if (images.size() < 2) {
        QMessageBox::warning(_parent, QStringLiteral("提示"), QStringLiteral("至少需要选择两张影像"));
        return;
    }
    if (outputDir.trimmed().isEmpty()) {
        QMessageBox::warning(_parent, QStringLiteral("提示"), QStringLiteral("请指定输出目录"));
        return;
    }
    if (_atCancelFlag)
    {
        QMessageBox::information(
            _parent,
            QStringLiteral("光束法平差"),
            QStringLiteral("已有空三或光束法平差任务正在运行，请等待其结束或先取消当前任务。"));
        return;
    }

    // 新一轮运行会替代尚未处理的旧预览，避免失败或取消后误提交旧结果。
    discardBundleAdjustPreview();

    const QString outDir = QDir::cleanPath(outputDir);

    // ── 只获取核心数据（影像列表/相机，无结果数组，速度极快）──────────────
    // 单影像 `.pimatch` 分片在后台线程直接读取，避免 UI 阻塞。
    const QJsonObject coreData   = _projectData->coreFilesMeta();
    const QString     plascanPath = _projectData->currentProjectPath();
    const auto session = currentSessionContext();

    const int minMatches = qMax(0, extraSettings.value(QStringLiteral("min_matches")).toInt(0));

    // ── 组装 BaServiceOptions（纯参数，不含相机/轨迹——后台填充）──────────
    xjw::gui::BaServiceOptions opts;
    opts.selectedImages   = images;
    opts.outputDir        = outDir;
    opts.dryRun           = dryRun;
    opts.threads          = threads;
    opts.baOpt.maxIterations       = qBound(3,  extraSettings.value(QStringLiteral("max_iterations")).toInt(20),  200);
    opts.baOpt.maxPointIterations = qBound(
        1, extraSettings.value(QStringLiteral("max_point_iterations")).toInt(12), 100);
    opts.baOpt.maxCameraIterations = qBound(
        1, extraSettings.value(QStringLiteral("max_camera_iterations")).toInt(10), 100);
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
    else if (baBackendName == QLatin1String("plamatrix_cpu"))
    {
        opts.baOpt.backend = xjw::BABackend::PlaMatrixCpu;
    }
    else if (baBackendName == QLatin1String("plamatrix_cuda"))
    {
        opts.baOpt.backend = xjw::BABackend::PlaMatrixCuda;
    }
    else if (baBackendName == QLatin1String("plamatrix_opencl"))
    {
        opts.baOpt.backend = xjw::BABackend::PlaMatrixOpenCl;
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
    opts.baOpt.plaMatrixDevice = qMax(
        0,
        extraSettings.value(QStringLiteral("ba_plamatrix_device")).toInt(0));
    opts.baOpt.minPlaMatrixGpuCameras = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_cuda_cameras")).toInt(
            opts.baOpt.minPlaMatrixGpuCameras));
    opts.baOpt.minPlaMatrixGpuObservations = qMax(
        1,
        extraSettings.value(QStringLiteral("ba_min_cuda_observations")).toInt(
            opts.baOpt.minPlaMatrixGpuObservations));
    opts.baOpt.minCeresCudaCameras = opts.baOpt.minPlaMatrixGpuCameras;
    opts.baOpt.minCeresCudaObservations = opts.baOpt.minPlaMatrixGpuObservations;
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
    opts.baOpt.maxCeresInitialTrackRms = qMax(
        0.0,
        extraSettings.value(QStringLiteral("ba_max_ceres_initial_track_rms")).toDouble(
            opts.baOpt.maxCeresInitialTrackRms));
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
    opts.exportObservationDetails =
        extraSettings.value(QStringLiteral("export_observation_details")).toBool(true);
    opts.enableLaserConstraints = extraSettings.value(QStringLiteral("enable_laser_constraints")).toBool(false);
    opts.laserConstraintCloudPath =
        extraSettings.value(QStringLiteral("laser_constraint_cloud_path")).toString().trimmed();
    opts.laserAssociationMaxDistanceMeters = qMax(
        0.0,
        extraSettings.value(QStringLiteral("laser_association_max_distance_m")).toDouble(0.05));
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
        extraSettings.value(QStringLiteral("laser_weight")).toDouble(0.0));
    opts.laserSigmaMeters = qMax(
        1e-9,
        extraSettings.value(QStringLiteral("laser_sigma_m")).toDouble(0.0025));
    opts.laserHuberDeltaMeters = qMax(
        1e-9,
        extraSettings.value(QStringLiteral("laser_huber_delta_m")).toDouble(0.05));
    opts.enablePlanetaryLaserRangeConstraints =
        extraSettings.value(
            QStringLiteral("enable_planetary_laser_range_constraints")).toBool(false);
    opts.planetaryLaserDataPath =
        extraSettings.value(QStringLiteral("planetary_laser_data_path")).toString().trimmed();
    opts.planetaryLaserCameraCoordinateFrame =
        extraSettings.value(
            QStringLiteral("planetary_laser_camera_coordinate_frame")).toString().trimmed();
    opts.planetaryLaserCameraSensorFrame =
        extraSettings.value(
            QStringLiteral("planetary_laser_camera_sensor_frame")).toString().trimmed();
    opts.planetaryLaserConfirmUnknownSensorIsFrame =
        extraSettings.value(
            QStringLiteral("planetary_laser_confirm_unknown_sensor_is_frame")).toBool(false);
    opts.planetaryLaserConfirmUnknownRangeIsOneWay =
        extraSettings.value(
            QStringLiteral("planetary_laser_confirm_unknown_range_is_one_way")).toBool(false);
    opts.planetaryLaserAllowUnmappedShots =
        extraSettings.value(
            QStringLiteral("planetary_laser_allow_unmapped_shots")).toBool(false);
    opts.planetaryLaserAllowUnmappedMeasuredImages =
        extraSettings.value(
            QStringLiteral("planetary_laser_allow_unmapped_measured_images")).toBool(false);
    opts.planetaryLaserRangeWeight = qMax(
        1.0e-12,
        extraSettings.value(QStringLiteral("planetary_laser_range_weight")).toDouble(1.0));
    opts.planetaryLaserRangeHuberDeltaSigma = qMax(
        0.0,
        extraSettings.value(
            QStringLiteral("planetary_laser_range_huber_delta_sigma")).toDouble(3.0));
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
        [baProgressSelf, cancelFlag, session](int currentIteration,
                                             int maxIterations,
                                             double avgRms,
                                             int validPoints) -> bool
        {
            if (!baProgressSelf || cancelFlag->load(std::memory_order_relaxed))
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
                [baProgressSelf, cancelFlag, session, stage, percent]()
                {
                    if (!baProgressSelf)
                    {
                        return;
                    }
                    if (baProgressSelf->_atCancelFlag != cancelFlag ||
                        !baProgressSelf->isCurrentSession(session) ||
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
     cancelFlag, opts = std::move(opts), isDryRun = dryRun, session]() mutable
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
            [self, cancelFlag, session]()
            {
                if (!self)
                {
                    return;
                }
                if (self->_atCancelFlag != cancelFlag ||
                    !self->isCurrentSession(session) ||
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
                [self, cancelFlag, session]()
                {
                    if (!self)
                    {
                        return;
                    }
                    if (self->_atCancelFlag == cancelFlag
                        && self->isCurrentSession(session))
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
                [self, cancelFlag, session, buildStatus = executionResult.buildStatus]() {
                    if (!self)
                    {
                        return;
                    }
                    if (self->_atCancelFlag != cancelFlag)
                    {
                        return;
                    }
                    if (!self->isCurrentSession(session))
                    {
                        self->_atCancelFlag.reset();
                        emit self->atProgressFinished(false);
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
             isDryRun,
             session]()
            {
                if (!self)
                {
                    return;
                }
                if (self->_atCancelFlag != cancelFlag)
                {
                    return;
                }
                if (!self->isCurrentSession(session))
                {
                    self->_atCancelFlag.reset();
                    self->discardBundleAdjustPreview();
                    emit self->atProgressFinished(false);
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
                const bool has_applicable_preview =
                    baResult.success && !isDryRun && !baResult.pendingCamUpdates.isEmpty();
                if (has_applicable_preview)
                {
                    self->_pendingBaBeforeCameraMeta = beforeCamMeta;
                    self->_pendingBaCameraMeta = baResult.pendingCamUpdates;
                    self->_pendingBaResult = baResult.resultJson;
                    self->_hasPendingBaPreview = true;
                }
                else
                {
                    self->discardBundleAdjustPreview();
                }
                emit self->bundleAdjustPreviewReady(baResult.resultJson);
                self->_atCancelFlag.reset();
                emit self->atProgressFinished(baResult.success);
                if (has_applicable_preview)
                {
                    self->presentBundleAdjustPreview();
                }
            },
            Qt::QueuedConnection);
    },
    [cancelFlag, session](ProjectManager *manager,
                 xjw::gui::tasks::TaskOutcome<void> outcome)
    {
        if (outcome.succeeded() || manager->_atCancelFlag != cancelFlag)
        {
            return;
        }

        manager->_atCancelFlag.reset();
        emit manager->atProgressFinished(false);
        if (!manager->isCurrentSession(session))
        {
            return;
        }
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

QMap<QString, xjw::FramePinholeCamera> ProjectManager::getCamerasForImages(
        const QStringList &images,
        bool *hasCamerasForAll) const
{
    if (hasCamerasForAll) *hasCamerasForAll = true;

    QMap<QString, xjw::FramePinholeCamera> result;
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

        xjw::FramePinholeCamera cam;
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
    const QString requested_depth_quality = settings.value(
        QStringLiteral("depthQualityProfile")).toString(
            xjw::core::project::depthQualityProfileForModelQuality(
                settings.value(QStringLiteral("quality")).toString(
                    QStringLiteral("high"))));
    const QJsonObject project_metadata =
        _projectData->metadataIncludingResults();
    const QString stored_depth_quality = storedDepthBatchQualityProfile(
        project_metadata, depth_source);
    const bool stored_depth_quality_insufficient =
        !stored_depth_quality.isEmpty() &&
        xjw::core::project::depthQualityRank(stored_depth_quality) <
            xjw::core::project::depthQualityRank(requested_depth_quality);
    const auto sparse_scaffold =
        xjw::gui::project::resolveSparseScaffoldSource(
            project_metadata,
            depth_source);
    const bool allow_sparse_scaffold_fallback =
        settings.value(QStringLiteral(
            "tsdfOrbitalSparseScaffoldCompletion")).toBool(true) &&
        !sparse_scaffold.pointCloudPath.isEmpty() &&
        !sparse_scaffold.pointsJsonPath.isEmpty();
    const auto stored_depth_compatibility =
        source_data == QStringLiteral("depth_maps") && !depth_source.isEmpty()
            ? xjw::gui::project::assessStoredDepthBatchCompatibility(
                  project_metadata,
                  depth_source,
                  settings.value(QStringLiteral("at_index")).toInt(-1),
                  settings.value(QStringLiteral("sceneProfile")).toString(),
                  allow_sparse_scaffold_fallback,
                  xjw::gui::project::depthBatchRequirementsForModelSettings(
                      settings))
            : xjw::gui::project::StoredDepthBatchCompatibility{};
    const bool stored_depth_batch_incompatible =
        !depth_source.isEmpty() && !stored_depth_compatibility.compatible;
    const bool prepare_depth_maps =
        source_data == QStringLiteral("depth_maps") &&
        (settings.value(QStringLiteral("automatic_depth_maps")).toBool(false) ||
         force_depth_recompute ||
         stored_depth_batch_incompatible ||
         stored_depth_quality_insufficient ||
         !reuse_depth_maps ||
         depth_source.isEmpty());
    if (!prepare_depth_maps)
    {
        _modelManager->startMeshReconstructionAsync(settings);
        return;
    }
    if (stored_depth_batch_incompatible)
    {
        LOG_INFO(QStringLiteral(
            "[模型生成] 现有深度图批次不兼容，将自动重新估计：%1")
                     .arg(stored_depth_compatibility.reason));
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
    depth_settings[QStringLiteral("qualityProfile")] = requested_depth_quality;
    depth_settings[QStringLiteral("depthQualityProfile")] = requested_depth_quality;
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

void ProjectManager::cancelDemGeneration()
{
    if (_terrainProductsManager)
    {
        _terrainProductsManager->cancelDemGeneration();
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

void ProjectManager::presentBundleAdjustPreview()
{
    while (_hasPendingBaPreview && !_pendingBaCameraMeta.isEmpty())
    {
        const auto presentation = buildBundleAdjustPreviewPresentation(
            _pendingBaResult,
            _pendingBaCameraMeta.size());

        QMessageBox message_box(_parent);
        message_box.setObjectName(QStringLiteral("bundleAdjustPreviewMessageBox"));
        message_box.setWindowTitle(QStringLiteral("参考地形约束重新平差"));
        message_box.setIcon(presentation.qualityWarning
                               ? QMessageBox::Warning
                               : QMessageBox::Question);
        message_box.setText(presentation.summaryText);
        message_box.setInformativeText(
            presentation.qualityWarning
                ? QStringLiteral("关键质量指标存在警告。请展开详细信息核对后，再决定是否写回项目。")
                : QStringLiteral("请核对指标后选择“保留结果”写回项目，或选择“丢弃结果”保持原相机参数。"));
        if (!presentation.detailedText.isEmpty())
        {
            message_box.setDetailedText(presentation.detailedText);
        }

        QPushButton *keep_button = message_box.addButton(
            QStringLiteral("保留结果"),
            QMessageBox::AcceptRole);
        QPushButton *discard_button = message_box.addButton(
            QStringLiteral("丢弃结果"),
            QMessageBox::DestructiveRole);
        keep_button->setObjectName(QStringLiteral("keepBundleAdjustPreviewButton"));
        discard_button->setObjectName(QStringLiteral("discardBundleAdjustPreviewButton"));
        message_box.setDefaultButton(keep_button);
        message_box.setEscapeButton(discard_button);
        message_box.exec();

        if (message_box.clickedButton() != keep_button)
        {
            discardBundleAdjustPreview();
            LOG_INFO(QStringLiteral("BA: 用户丢弃了待提交的平差结果"));
            return;
        }

        QString error_message;
        if (acceptBundleAdjustPreview(&error_message))
        {
            return;
        }

        QMessageBox::warning(
            _parent,
            QStringLiteral("光束法平差"),
            QStringLiteral("应用平差结果失败：%1\n\n结果仍保留在内存中，可重试或选择丢弃。")
                .arg(error_message));
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
                             QStringLiteral("已保留本次平差结果，并更新 %1 台相机参数。\n"
                                            "详细指标可在“工具 > 查看工作流程报告”中查看。")
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
