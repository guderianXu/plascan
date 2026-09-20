#include "ProjectResourceService.h"

#include "ProjectResourceCleanupCoordinator.h"
#include "ProjectSession.h"
#include "ProjectUiMessageAdapter.h"
#include "ProjectSurveyControl.h"
#include "ProjectWorkflowReports.h"
#include "ReferenceDatasetWorkflow.h"
#include "camera/SurveyControlDialog.h"
#include "Logger.h"
#include "ProjectCameraIO.h"
#include "project/ProjectAssetImporter.h"
#include "project/ProjectIO.h"
#include "project/ProjectSessionModel.h"
#include "project/ProjectWorkspaceStore.h"
#include "LaserConstraintMap.h"
#include "PlanetaryLaserJson.h"
#include "io/PathIO.h"

#include <QDir>
#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QSet>
#include <QThread>

#include <algorithm>
#include <cmath>
#include <string>

namespace xjw::gui::project
{

    namespace
    {

        constexpr QDir::Filters kDialogFilters = QDir::AllEntries | QDir::Hidden | QDir::AllDirs | QDir::NoDotAndDotDot;

        QStringList imageFilters()
        {
            return {QStringLiteral("*.tif"),
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
                    QStringLiteral("*.CUB")};
        }

        bool isBaPriorRole(const QString& role)
        {
            const QString normalized = role.trimmed().toLower();
            return normalized == QLatin1String("ba_prior") || normalized == QLatin1String("bundle_adjustment") ||
                   normalized == QLatin1String("reference_prior");
        }

        bool validateBaPriorImport(const QString& path, QString* errorMessage)
        {
            const QString suffix = QFileInfo(path).suffix().toLower();
            if (suffix == QLatin1String("tif") || suffix == QLatin1String("tiff") || suffix == QLatin1String("vrt"))
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
                        *errorMessage =
                            QStringLiteral("无法作为行星激光测距数据导入: %1\n"
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
                    *errorMessage = QStringLiteral("当前 BA 软约束直接支持 DEM、扫描点云 PLY 或行星激光 SI JSON。\n"
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
                    *errorMessage = QStringLiteral("该 PLY 没有可用的有限非零表面法向。\n"
                                                   "需要 vertex 的 normal_x/normal_y/normal_z（或 nx/ny/nz）字段；"
                                                   "环绕目标不能按水平面代替法向。\n解析信息: %1")
                                        .arg(QString::fromStdString(loadError));
                }
                return false;
            }
            return true;
        }

        QString firstReferenceDemPriorPath(const QJsonObject& metadata)
        {
            for (const QJsonValue& value : metadata.value(QStringLiteral("reference_datasets")).toArray())
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
            return {};
        }

        QString firstReferenceLaserPriorPath(const QJsonObject& metadata)
        {
            for (const QJsonValue& value : metadata.value(QStringLiteral("reference_datasets")).toArray())
            {
                const QJsonObject reference = value.toObject();
                if (!isBaPriorRole(reference.value(QStringLiteral("role")).toString()))
                {
                    continue;
                }
                const QString type = reference.value(QStringLiteral("type")).toString().toLower();
                const QString path = reference.value(QStringLiteral("path")).toString().trimmed();
                const QString suffix = QFileInfo(path).suffix().toLower();
                const bool laser_type = type == QLatin1String("lidar") || type == QLatin1String("reference_lidar") ||
                                        type == QLatin1String("point_cloud");
                if (laser_type && suffix == QLatin1String("ply"))
                {
                    return path;
                }
            }
            return {};
        }

        QString firstPlanetaryLaserPriorPath(const QJsonObject& metadata)
        {
            for (const QJsonValue& value : metadata.value(QStringLiteral("reference_datasets")).toArray())
            {
                const QJsonObject reference = value.toObject();
                if (isBaPriorRole(reference.value(QStringLiteral("role")).toString()) &&
                    reference.value(QStringLiteral("type")).toString().toLower() ==
                        QLatin1String("planetary_laser_shots"))
                {
                    const QString path = reference.value(QStringLiteral("path")).toString().trimmed();
                    if (!path.isEmpty())
                    {
                        return path;
                    }
                }
            }
            return {};
        }

    } // namespace

    ProjectResourceService::ProjectResourceService(ProjectSession* session,
                                                   ProjectUiMessageAdapter* messages,
                                                   ProjectResourceCleanupCoordinator* cleanup,
                                                   QWidget* parentWidget,
                                                   QObject* parent)
        : QObject(parent), _session(session), _messages(messages), _cleanup(cleanup), _parentWidget(parentWidget)
    {
        if (_session)
        {
            connect(_session,
                    &ProjectSession::sessionChanged,
                    this,
                    [this](const ProjectSessionContext&) { _cancellationSource.requestCancellation(); });
        }
    }

    ProjectResourceService::~ProjectResourceService()
    {
        cancelTask();
        if (_taskFuture.isRunning())
        {
            _taskFuture.waitForFinished();
        }
        finishTask();
    }

    void ProjectResourceService::setDirectoryAccessors(
        std::function<QString(const QString& key)> getLastDir,
        std::function<void(const QString& key, const QString& dir)> saveLastDir)
    {
        _getLastDir = std::move(getLastDir);
        _saveLastDir = std::move(saveLastDir);
    }

    void ProjectResourceService::setPortableExportLauncher(
        std::function<bool(const QString& outputPath, QString* errorMessage)> launcher)
    {
        _portableExportLauncher = std::move(launcher);
    }

    void ProjectResourceService::setBundleAdjustLauncher(std::function<bool(const QStringList& images,
                                                                            const QString& outputDir,
                                                                            int threads,
                                                                            bool dryRun,
                                                                            const QJsonObject& extraSettings)> launcher)
    {
        _bundleAdjustLauncher = std::move(launcher);
    }

    OperationResult ProjectResourceService::startImageImportTask(const QStringList& imagePaths,
                                                                 const QString& sourceLabel,
                                                                 bool scanFolder)
    {
        if (!beginTask(QStringLiteral("添加影像")))
        {
            return failed(QStringLiteral("已有项目操作正在运行，请等待完成后再试。"));
        }

        struct ImageImportWork
        {
            QStringList newPaths;
            QMap<QString, QJsonObject> rpcCameras;
            int skipped = 0;
            QString error;
            bool cancelled = false;
            bool emptyFolder = false;
        };

        const ProjectSessionContext sessionContext = _taskContext;
        const auto cancellation = _taskToken;
        const QStringList requestedPaths = imagePaths;
        const QSet<QString> existingPaths = [&]()
        {
            QSet<QString> result;
            const QStringList currentImages = _session->allImages();
            result.reserve(currentImages.size());
            for (const QString& currentImage : currentImages)
            {
                result.insert(QDir::cleanPath(currentImage));
            }
            return result;
        }();
        emit imageImportProgressChanged(QStringLiteral("正在加载影像..."), 0, requestedPaths.size());
        _imageImportActive = true;

        _taskFuture = xjw::gui::tasks::runGuardedWithOutcome(
            this,
            [requestedPaths, sourceLabel, scanFolder, existingPaths, cancellation]()
            {
                ImageImportWork work;
                QStringList candidates = requestedPaths;
                if (scanFolder)
                {
                    const QDir directory(requestedPaths.value(0));
                    if (!directory.exists())
                    {
                        work.error = QStringLiteral("文件夹不存在: %1").arg(requestedPaths.value(0));
                        return work;
                    }
                    const QFileInfoList files = directory.entryInfoList(
                        imageFilters(), QDir::Files | QDir::Readable, QDir::Name | QDir::IgnoreCase);
                    candidates.clear();
                    candidates.reserve(files.size());
                    for (const QFileInfo& file : files)
                    {
                        if (cancellation.isCancellationRequested())
                        {
                            work.cancelled = true;
                            return work;
                        }
                        candidates.append(file.absoluteFilePath());
                    }
                    if (candidates.isEmpty())
                    {
                        work.emptyFolder = true;
                        return work;
                    }
                }

                QSet<QString> seen;
                for (const QString& path : candidates)
                {
                    const QString normalized = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
                    if (normalized.isEmpty() || existingPaths.contains(normalized) || seen.contains(normalized))
                    {
                        ++work.skipped;
                        continue;
                    }
                    seen.insert(normalized);
                    if (!QFileInfo(normalized).isFile())
                    {
                        work.error = QStringLiteral("影像不存在: %1").arg(normalized);
                        return work;
                    }
                    work.newPaths.append(normalized);

                    if ((normalized.endsWith(QStringLiteral(".tif"), Qt::CaseInsensitive) ||
                         normalized.endsWith(QStringLiteral(".tiff"), Qt::CaseInsensitive)) &&
                        !cancellation.isCancellationRequested())
                    {
                        QJsonObject camera;
                        if (xjw::common::project::parseRpcCameraRaster(normalized, &camera, nullptr) &&
                            !camera.isEmpty())
                        {
                            work.rpcCameras.insert(normalized, camera);
                        }
                    }
                    if (cancellation.isCancellationRequested())
                    {
                        work.cancelled = true;
                        return work;
                    }
                }
                return work;
            },
            [sourceLabel, requestedPaths, scanFolder, sessionContext, cancellation](
                ProjectResourceService* self, xjw::gui::tasks::TaskOutcome<ImageImportWork> outcome)
            {
                const auto fail = [self](const QString& message, bool present)
                {
                    emit self->imageImportFinished(false, message);
                    if (present)
                    {
                        self->presentFailure(QStringLiteral("加载影像失败"), message);
                    }
                    self->finishTask();
                };
                if (!outcome.succeeded())
                {
                    fail(outcome.errorMessage.isEmpty() ? QStringLiteral("影像加载任务失败") : outcome.errorMessage,
                         true);
                    return;
                }
                ImageImportWork work = std::move(*outcome.value);
                if (work.cancelled || cancellation.isCancellationRequested() || !self->isCurrent(sessionContext))
                {
                    fail(QStringLiteral("项目已切换，影像加载已停止"), false);
                    return;
                }
                if (!work.error.isEmpty())
                {
                    fail(work.error, true);
                    return;
                }
                if (scanFolder && work.emptyFolder)
                {
                    emit self->imageImportProgressChanged(
                        QStringLiteral("正在加载影像..."), requestedPaths.size(), requestedPaths.size());
                    emit self->imageImportFinished(true, QStringLiteral("文件夹中没有找到可导入的影像"));
                    self->finishTask();
                    return;
                }
                if (work.newPaths.isEmpty())
                {
                    const QString message = QStringLiteral("已跳过 %1 张重复影像").arg(work.skipped);
                    emit self->imageImportProgressChanged(
                        QStringLiteral("正在加载影像..."), requestedPaths.size(), requestedPaths.size());
                    emit self->imageImportFinished(true, message);
                    self->finishTask();
                    return;
                }

                QString error;
                if (!self->_session->data()->addImages(work.newPaths, &error))
                {
                    fail(error.isEmpty() ? QStringLiteral("影像导入失败") : error, true);
                    return;
                }
                int importedRpcCameras = 0;
                if (!work.rpcCameras.isEmpty())
                {
                    QString rpcError;
                    if (!self->_session->setCameraInstances(work.rpcCameras, &importedRpcCameras, &rpcError) &&
                        self->_messages && !rpcError.isEmpty())
                    {
                        self->_messages->warning(
                            self->_parentWidget,
                            QStringLiteral("影像相机"),
                            QStringLiteral("影像已导入，但 RPC 相机写入工程失败：%1").arg(rpcError));
                    }
                }

                emit self->imageImportProgressChanged(
                    QStringLiteral("正在加载影像..."), requestedPaths.size(), requestedPaths.size());
                QString message =
                    QStringLiteral("已加载 %1 张影像%2")
                        .arg(work.newPaths.size())
                        .arg(sourceLabel.isEmpty() ? QString() : QStringLiteral("（来源：%1）").arg(sourceLabel));
                if (work.skipped > 0)
                {
                    message += QStringLiteral("，跳过 %1 张重复影像").arg(work.skipped);
                }
                if (importedRpcCameras > 0)
                {
                    message += QStringLiteral("，自动读取 %1 个 RPC 相机").arg(importedRpcCameras);
                }
                emit self->imageImportFinished(true, message);
                emit self->resourceChanged();
                self->finishTask();
            });

        return success(QStringLiteral("影像加载已启动"));
    }

    OperationResult ProjectResourceService::addPhoto()
    {
        if (!requireProject(QStringLiteral("添加影像")) || !_messages)
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        const UiDialogResult selected = _messages->selectOpenFiles(
            _parentWidget,
            QStringLiteral("添加图片"),
            readLastDir(QStringLiteral("images")),
            QStringLiteral(
                "影像文件 (*.tif *.tiff *.TIF *.TIFF *.png *.PNG *.jpg *.jpeg *.JPG *.JPEG *.img *.IMG *.cub *.CUB)"),
            kDialogFilters);
        if (!selected.accepted || selected.texts.isEmpty())
        {
            return cancelled();
        }
        writeLastDir(QStringLiteral("images"), QFileInfo(selected.texts.constFirst()).absolutePath());
        return addPhotos(selected.texts, QStringLiteral("所选文件"));
    }

    OperationResult ProjectResourceService::addPhotos(const QStringList& imagePaths, const QString& sourceLabel)
    {
        if (!requireProject(QStringLiteral("添加影像")))
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        const QStringList normalizedPaths = normalizedImagePaths(imagePaths);
        if (normalizedPaths.isEmpty())
        {
            return failed(QStringLiteral("没有可导入的影像"));
        }
        return startImageImportTask(normalizedPaths, sourceLabel, false);
    }

    OperationResult ProjectResourceService::addFolder()
    {
        if (!requireProject(QStringLiteral("添加文件夹")) || !_messages)
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        const UiDialogResult selected = _messages->selectDirectory(
            _parentWidget, QStringLiteral("选择文件夹"), readLastDir(QStringLiteral("images")), kDialogFilters);
        if (!selected.accepted || selected.text.trimmed().isEmpty())
        {
            return cancelled();
        }
        writeLastDir(QStringLiteral("images"), selected.text);
        return addFolder(selected.text);
    }

    OperationResult ProjectResourceService::addFolder(const QString& folderPath)
    {
        if (!requireProject(QStringLiteral("添加文件夹")))
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        const QString normalizedFolderPath = normalizedPath(folderPath);
        if (normalizedFolderPath.isEmpty())
        {
            return cancelled();
        }
        return startImageImportTask({normalizedFolderPath}, normalizedFolderPath, true);
    }

    OperationResult ProjectResourceService::importPointCloud()
    {
        if (!requireProject(QStringLiteral("导入点云")) || !_messages)
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        const UiDialogResult selected =
            _messages->selectOpenFile(_parentWidget,
                                      QStringLiteral("导入 Metashape 点云"),
                                      readLastDir(QStringLiteral("import_point_cloud")),
                                      QStringLiteral("Metashape/通用点云 (*.obj *.ply *.xyz);;所有文件 (*)"),
                                      kDialogFilters);
        if (!selected.accepted || selected.text.isEmpty())
        {
            return cancelled();
        }
        writeLastDir(QStringLiteral("import_point_cloud"), QFileInfo(selected.text).absolutePath());
        return importProjectAsset(selected.text, false);
    }

    OperationResult ProjectResourceService::importModel()
    {
        if (!requireProject(QStringLiteral("导入模型")) || !_messages)
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        const UiDialogResult selected =
            _messages->selectOpenFile(_parentWidget,
                                      QStringLiteral("导入 Metashape 模型"),
                                      readLastDir(QStringLiteral("import_model")),
                                      QStringLiteral("Metashape/通用模型 (*.obj *.ply);;所有文件 (*)"),
                                      kDialogFilters);
        if (!selected.accepted || selected.text.isEmpty())
        {
            return cancelled();
        }
        writeLastDir(QStringLiteral("import_model"), QFileInfo(selected.text).absolutePath());
        return importProjectAsset(selected.text, true);
    }

    OperationResult ProjectResourceService::importProjectAsset(const QString& selectedPath, bool modelAsset)
    {
        if (!requireProject(QStringLiteral("导入资源")))
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        if (selectedPath.trimmed().isEmpty())
        {
            return cancelled();
        }

        xjw::common::project::ProjectAssetImportRequest request;
        request.type = modelAsset ? xjw::common::project::ProjectAssetType::Model
                                  : xjw::common::project::ProjectAssetType::PointCloud;
        request.sourcePath = normalizedPath(selectedPath);
        request.projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(_session->projectPath());
        if (!beginTask(modelAsset ? QStringLiteral("导入模型") : QStringLiteral("导入点云")))
        {
            return failed(QStringLiteral("已有项目操作正在运行，请等待完成后再试。"));
        }

        const ProjectSessionContext sessionContext = _taskContext;
        const auto cancellation = _taskToken;
        const auto operationLabel = modelAsset ? QStringLiteral("导入模型") : QStringLiteral("导入点云");
        _taskFuture = xjw::gui::tasks::runGuardedWithOutcome(
            this,
            [request, cancellation]()
            {
                if (cancellation.isCancellationRequested())
                {
                    return xjw::common::project::ProjectAssetImportResult{};
                }
                return xjw::common::project::ProjectAssetImporter::importAsset(request);
            },
            [sessionContext, operationLabel, cancellation](
                ProjectResourceService* self,
                xjw::gui::tasks::TaskOutcome<xjw::common::project::ProjectAssetImportResult> outcome)
            {
                const auto fail = [self, &operationLabel](const QString& message, bool present)
                {
                    if (present)
                    {
                        self->presentFailure(operationLabel, message);
                    }
                    self->finishTask();
                };
                if (!outcome.succeeded())
                {
                    fail(outcome.errorMessage.isEmpty() ? QStringLiteral("资源导入任务失败") : outcome.errorMessage,
                         true);
                    return;
                }
                auto result = std::move(*outcome.value);
                if (cancellation.isCancellationRequested() || !self->isCurrent(sessionContext))
                {
                    if (!result.importDirectory.isEmpty())
                    {
                        QDir(result.importDirectory).removeRecursively();
                    }
                    fail(QStringLiteral("项目已切换，已取消导入。"), false);
                    return;
                }
                if (!result.success)
                {
                    fail(result.errorMessage.isEmpty() ? QStringLiteral("导入失败") : result.errorMessage, true);
                    return;
                }
                if (!self->_session->data()->upsertResultRecordByPath(
                        result.resultArrayKey, result.resultPathKey, result.projectRecord, true))
                {
                    QDir(result.importDirectory).removeRecursively();
                    fail(QStringLiteral("资源已读取，但无法写入项目成果记录。"), true);
                    return;
                }
                const QString message = QStringLiteral("已导入 %1\n%2")
                                            .arg(operationLabel == QStringLiteral("导入模型") ? QStringLiteral("模型")
                                                                                              : QStringLiteral("点云"),
                                                 result.importedPath);
                if (self->_messages)
                {
                    self->_messages->information(self->_parentWidget, operationLabel, message);
                }
                emit self->resourceChanged();
                self->finishTask();
            });
        return success(QStringLiteral("%1已启动").arg(operationLabel));
    }

    OperationResult ProjectResourceService::removeResources(const QStringList& resourcePaths)
    {
        if (!requireProject(QStringLiteral("移除资源")))
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        if (resourcePaths.isEmpty())
        {
            return cancelled();
        }
        QStringList normalizedResources;
        normalizedResources.reserve(resourcePaths.size());
        for (const QString& path : resourcePaths)
        {
            const QString normalized = normalizedPath(path);
            if (!normalized.isEmpty() && !normalizedResources.contains(normalized))
            {
                normalizedResources.append(normalized);
            }
        }
        if (normalizedResources.isEmpty())
        {
            return cancelled();
        }
        if (!beginTask(QStringLiteral("移除资源")))
        {
            return failed(QStringLiteral("已有项目操作正在运行，请等待完成后再试。"));
        }
        const ProjectSessionContext sessionContext = _session->context();
        QString error;
        if (!_session->data()->removeResources(normalizedResources))
        {
            error = QStringLiteral("移除资源失败");
            presentFailure(QStringLiteral("移除资源"), error);
            finishTask();
            return failed(error);
        }
        if (!isCurrent(sessionContext))
        {
            finishTask();
            return failed(QStringLiteral("项目已切换，已取消移除资源。"));
        }
        emit resourceChanged();
        finishTask();
        return success();
    }

    OperationResult ProjectResourceService::deleteGeneratedData(const QString& section,
                                                                const QStringList& resourcePaths,
                                                                QWidget* requestWidget)
    {
        if (!requireProject(QStringLiteral("删除数据")))
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        if (resourcePaths.isEmpty())
        {
            return cancelled();
        }
        if (!_cleanup)
        {
            return failed(QStringLiteral("资源清理服务未初始化"));
        }
        if (_cleanup->rejectLifecycleChange(QStringLiteral("删除数据")))
        {
            return failed(QStringLiteral("资源清理进行中，请稍候。"));
        }
        if (!_cleanup->deleteGeneratedData(section, resourcePaths, requestWidget))
        {
            return cancelled(QStringLiteral("删除数据已取消"));
        }
        return success(QStringLiteral("资源清理已启动"));
    }

    OperationResult ProjectResourceService::importReferenceDataset()
    {
        if (!requireProject(QStringLiteral("导入参考数据")) || !_messages)
        {
            return failed(QStringLiteral("请先打开项目"));
        }
        const UiDialogResult selected = _messages->selectOpenFile(
            _parentWidget,
            QStringLiteral("导入参考 DEM/LiDAR"),
            readLastDir(QStringLiteral("reference_dataset")),
            QStringLiteral("参考地形/点云/行星激光 (*.tif *.tiff *.vrt *.las *.laz *.copc *.ply *.xyz *.csv *.json);;"
                           "所有文件 (*)"),
            kDialogFilters);
        if (!selected.accepted || selected.text.trimmed().isEmpty())
        {
            return cancelled();
        }
        writeLastDir(QStringLiteral("reference_dataset"), QFileInfo(selected.text).absolutePath());

        QString role = QStringLiteral("validation");
        const UiDialogResult purpose =
            _messages->getItem(_parentWidget,
                               QStringLiteral("选择参考数据用途"),
                               QStringLiteral("该参考数据如何参与项目？"),
                               QStringList{QStringLiteral("仅用于精度检查（validation）"),
                                           QStringLiteral("用于光束法平差软约束（ba_prior）")},
                               0);
        if (!purpose.accepted)
        {
            return cancelled();
        }
        if (purpose.text.contains(QStringLiteral("ba_prior")))
        {
            role = QStringLiteral("ba_prior");
        }
        return registerReferenceDataset(selected.text, QString(), role);
    }

    OperationResult
    ProjectResourceService::registerReferenceDataset(const QString& path, const QString& type, const QString& role)
    {
        if (!requireProject(QStringLiteral("导入参考数据")))
        {
            return failed(QStringLiteral("请先打开项目"));
        }
        if (path.trimmed().isEmpty())
        {
            return cancelled();
        }
        if (!beginTask(QStringLiteral("导入参考数据")))
        {
            return failed(QStringLiteral("已有项目操作正在运行，请等待完成后再试。"));
        }
        const QString normalized = normalizedPath(path);
        struct ReferenceValidationWork
        {
            bool valid = true;
            bool cancelled = false;
            QString errorMessage;
        };
        const ProjectSessionContext sessionContext = _taskContext;
        const auto cancellation = _taskToken;
        _taskFuture = xjw::gui::tasks::runGuardedWithOutcome(
            this,
            [normalized, role, cancellation]()
            {
                ReferenceValidationWork work;
                if (cancellation.isCancellationRequested())
                {
                    work.cancelled = true;
                    return work;
                }
                if (isBaPriorRole(role) && !validateBaPriorImport(normalized, &work.errorMessage))
                {
                    work.valid = false;
                }
                return work;
            },
            [sessionContext, normalized, type, role, cancellation](
                ProjectResourceService* self, xjw::gui::tasks::TaskOutcome<ReferenceValidationWork> outcome)
            {
                const auto fail = [self](const QString& message, bool present)
                {
                    if (present)
                    {
                        self->presentFailure(QStringLiteral("导入参考数据"), message);
                    }
                    self->finishTask();
                };
                if (!outcome.succeeded())
                {
                    fail(outcome.errorMessage.isEmpty() ? QStringLiteral("参考数据校验任务失败") : outcome.errorMessage,
                         true);
                    return;
                }
                const ReferenceValidationWork work = std::move(*outcome.value);
                if (work.cancelled || cancellation.isCancellationRequested() || !self->isCurrent(sessionContext))
                {
                    fail(QStringLiteral("项目已切换，已取消导入参考数据。"), false);
                    return;
                }
                if (!work.valid)
                {
                    fail(work.errorMessage.isEmpty() ? QStringLiteral("参考数据不符合 BA prior 要求")
                                                     : work.errorMessage,
                         true);
                    return;
                }

                QString error;
                if (!xjw::core::project::registerReferenceDataset(
                        self->_session->data(), normalized, type, role, &error))
                {
                    fail(error.isEmpty() ? QStringLiteral("导入参考数据失败") : error, true);
                    return;
                }
                emit self->resourceChanged();
                self->finishTask();
            });
        return success(QStringLiteral("参考数据导入已启动"));
    }

    OperationResult ProjectResourceService::importSurveyControlCsv(const QString& csvPath)
    {
        if (!requireProject(QStringLiteral("导入测绘控制")))
        {
            return failed(QStringLiteral("请先打开项目"));
        }
        if (csvPath.trimmed().isEmpty())
        {
            return cancelled();
        }
        const QString normalizedCsvPath = normalizedPath(csvPath);
        QHash<QString, QString> imageIdentityByPath;
        for (const QJsonValue& value : _session->coreMetadata().value(QStringLiteral("images")).toArray())
        {
            const QJsonObject image = value.toObject();
            const QString imagePath = image.value(QStringLiteral("path")).toString().trimmed();
            const QString imageId = image.value(QStringLiteral("image_uuid")).toString().trimmed();
            if (!imagePath.isEmpty() && !imageId.isEmpty())
            {
                imageIdentityByPath.insert(imagePath, imageId);
            }
        }
        if (!beginTask(QStringLiteral("导入测绘控制")))
        {
            return failed(QStringLiteral("已有项目操作正在运行，请等待完成后再试。"));
        }

        struct SurveyWork
        {
            PreparedSurveyControlImport prepared;
        };
        const ProjectSessionContext sessionContext = _taskContext;
        const auto cancellation = _taskToken;
        _taskFuture = xjw::gui::tasks::runGuardedWithOutcome(
            this,
            [imageIdentityByPath, normalizedCsvPath, cancellation]()
            {
                SurveyWork work;
                if (cancellation.isCancellationRequested())
                {
                    return work;
                }
                work.prepared = prepareSurveyControlCsv(imageIdentityByPath, normalizedCsvPath, QString());
                return work;
            },
            [sessionContext, cancellation](ProjectResourceService* self,
                                           xjw::gui::tasks::TaskOutcome<SurveyWork> outcome)
            {
                const auto fail = [self](const QString& message, bool present)
                {
                    if (present)
                    {
                        self->presentFailure(QStringLiteral("测绘控制"), message);
                    }
                    self->finishTask();
                };
                if (!outcome.succeeded())
                {
                    fail(outcome.errorMessage.isEmpty() ? QStringLiteral("导入测绘控制 CSV 失败。")
                                                        : outcome.errorMessage,
                         true);
                    return;
                }
                SurveyWork work = std::move(*outcome.value);
                if (cancellation.isCancellationRequested() || !self->isCurrent(sessionContext))
                {
                    fail(QStringLiteral("项目已切换，已取消导入测绘控制。"), false);
                    return;
                }
                if (!work.prepared.prepared)
                {
                    fail(work.prepared.errorMessage.isEmpty() ? QStringLiteral("导入测绘控制 CSV 失败。")
                                                              : work.prepared.errorMessage,
                         true);
                    return;
                }

                const QString sidecarPath =
                    xjw::common::project::ProjectIO::markerSetPath(self->_session->projectPath());
                SurveyControlSidecarSnapshot previous;
                QString snapshotError;
                if (!captureSurveyControlSidecar(sidecarPath, &previous, &snapshotError))
                {
                    fail(snapshotError.isEmpty() ? QStringLiteral("读取测绘控制 sidecar 失败。") : snapshotError, true);
                    return;
                }

                const auto written = writeSurveyControlMarkerSet(sidecarPath, work.prepared.markerSet);
                if (!written.imported)
                {
                    fail(written.errorMessage.isEmpty() ? QStringLiteral("写入测绘控制 sidecar 失败。")
                                                        : written.errorMessage,
                         true);
                    return;
                }
                const auto restoreSidecar = [&previous]()
                {
                    QString restoreError;
                    return restoreSurveyControlSidecar(previous, &restoreError) ? QString() : restoreError;
                };
                if (cancellation.isCancellationRequested() || !self->isCurrent(sessionContext))
                {
                    const QString restoreError = restoreSidecar();
                    fail(restoreError.isEmpty()
                             ? QStringLiteral("项目已切换，已取消导入测绘控制。")
                             : QStringLiteral("项目已切换，且旧 sidecar 恢复失败: %1").arg(restoreError),
                         false);
                    return;
                }
                const auto result = commitSurveyControlMarkerSet(self->_session->data(), work.prepared.markerSet);
                if (!result.imported)
                {
                    const QString restoreError = restoreSidecar();
                    const QString detail = result.errorMessage.isEmpty()
                                               ? QStringLiteral("写入测绘控制 metadata 失败。")
                                               : result.errorMessage;
                    fail(restoreError.isEmpty()
                             ? detail
                             : QStringLiteral("%1；且旧 sidecar 恢复失败: %2").arg(detail, restoreError),
                         true);
                    return;
                }
                emit self->surveyControlChanged();
                emit self->resourceChanged();
                self->finishTask();
            });
        return success(QStringLiteral("测绘控制导入已启动"));
    }

    OperationResult ProjectResourceService::importSurveyControlCsv()
    {
        if (!requireProject(QStringLiteral("导入测绘控制")) || !_messages)
        {
            return failed(QStringLiteral("请先打开项目"));
        }
        const UiDialogResult selected =
            _messages->selectOpenFile(_parentWidget,
                                      QStringLiteral("导入测绘控制 CSV"),
                                      readLastDir(QStringLiteral("survey_control")),
                                      QStringLiteral("控制点数据 (*.csv *.txt);;CSV 文件 (*.csv);;所有文件 (*)"),
                                      kDialogFilters);
        if (!selected.accepted || selected.text.trimmed().isEmpty())
        {
            return cancelled();
        }
        writeLastDir(QStringLiteral("survey_control"), QFileInfo(selected.text).absolutePath());
        return importSurveyControlCsv(selected.text);
    }

    void ProjectResourceService::openSurveyControlDialog()
    {
        if (!requireProject(QStringLiteral("管理测绘控制点")) || !_session || !_session->data())
        {
            return;
        }

        SurveyControlDialog dialog(_parentWidget);
        QString metadata_error;
        dialog.setSurveyControlMetadata(surveyControlDialogMetadata(_session->data(), &metadata_error));
        if (!metadata_error.isEmpty())
        {
            dialog.setStatusMessage(metadata_error);
        }
        connect(&dialog,
                &SurveyControlDialog::importCsvRequested,
                this,
                [this, &dialog]()
                {
                    const OperationResult result = importSurveyControlCsv();
                    if (!result.succeeded() || !_session || !_session->data())
                    {
                        return;
                    }
                    dialog.setSurveyControlMetadata(surveyControlDialogMetadata(_session->data()));
                    dialog.setStatusMessage(result.message);
                });
        dialog.exec();
    }

    void ProjectResourceService::runReferenceQualityCheck()
    {
        if (!requireProject(QStringLiteral("点云/DEM 精度检查")) || !_session || !_session->data())
        {
            return;
        }
        const auto result = xjw::core::project::writeReferenceDatasetQualityReport(_session->data());
        if (!result.saved)
        {
            if (_messages)
            {
                _messages->warning(_parentWidget,
                                   QStringLiteral("点云/DEM 精度检查"),
                                   result.errorMessage.isEmpty() ? QStringLiteral("生成点云/DEM 精度检查报告失败。")
                                                                 : result.errorMessage);
            }
            return;
        }
        const QString status = result.record.value(QStringLiteral("comparison_available")).toBool()
                                   ? QStringLiteral("已找到可检查的参考数据与项目成果。")
                                   : QStringLiteral("报告已生成，但当前缺少参考数据或可比较的项目成果。");
        LOG_INFO(QStringLiteral("参考数据精度检查报告已生成: %1").arg(result.jsonPath));
        if (_messages)
        {
            _messages->information(
                _parentWidget,
                QStringLiteral("点云/DEM 精度检查"),
                QStringLiteral("%1\nJSON: %2\nCSV: %3").arg(status, result.jsonPath, result.csvPath));
        }
    }

    void ProjectResourceService::prepareReferenceTerrainBundleAdjust()
    {
        if (!_messages)
        {
            return;
        }

        const QString title = QStringLiteral("参考地形约束重新平差");
        const auto warn = [this](const QString& dialog_title, const QString& message)
        {
            if (_messages)
            {
                _messages->warning(_parentWidget, dialog_title, message);
            }
        };
        const auto inform = [this](const QString& dialog_title, const QString& message)
        {
            if (_messages)
            {
                _messages->information(_parentWidget, dialog_title, message);
            }
        };

        if (!_session || !_session->hasProject() || !_session->data())
        {
            warn(title, QStringLiteral("请先打开项目，再使用参考地形约束重新平差。"));
            return;
        }

        const auto result = xjw::core::project::writeReferenceTerrainPriorPreflightReport(_session->data());
        if (!result.saved)
        {
            warn(title,
                 result.errorMessage.isEmpty() ? QStringLiteral("生成参考地形平差前置检查报告失败。")
                                               : result.errorMessage);
            return;
        }

        const bool ready = result.record.value(QStringLiteral("ready")).toBool();
        const QString status = result.record.value(QStringLiteral("status")).toString();
        LOG_INFO(QStringLiteral("参考地形平差前置检查报告已生成: %1 ready=%2").arg(result.jsonPath).arg(ready));
        if (!ready)
        {
            inform(title,
                   QStringLiteral("前置检查未通过：%1。\n"
                                  "请先导入 role=ba_prior 的参考 DEM/LiDAR，并完成正式空三。\n"
                                  "JSON: %2\nCSV: %3")
                       .arg(status, result.jsonPath, result.csvPath));
            return;
        }

        const QJsonObject metadata = _session->metadata();
        const QString planetary_laser_prior_path = firstPlanetaryLaserPriorPath(metadata);
        const QString dem_prior_path = firstReferenceDemPriorPath(metadata);
        const QString laser_prior_path = firstReferenceLaserPriorPath(metadata);
        const bool use_planetary_laser = !planetary_laser_prior_path.isEmpty();
        const bool use_dem = !use_planetary_laser && !dem_prior_path.isEmpty();
        const bool use_laser_surface = !use_planetary_laser && !use_dem && !laser_prior_path.isEmpty();
        if (!use_planetary_laser && !use_dem && !use_laser_surface)
        {
            inform(title,
                   QStringLiteral("前置检查通过，但未找到可直接用于 BA 的参考数据。\n"
                                  "请导入 role=ba_prior 的 DEM、带法向扫描点云 PLY，"
                                  "或行星激光测距 SI JSON。\n"
                                  "JSON: %1\nCSV: %2")
                       .arg(result.jsonPath, result.csvPath));
            return;
        }

        const QStringList images = _session->allImages();
        if (images.size() < 2)
        {
            warn(title, QStringLiteral("项目影像少于 2 张，无法执行 BA。"));
            return;
        }

        const QString bundle_adjust_dir =
            xjw::common::project::ProjectIO::projectBundleAdjustDir(_session->projectPath());
        const QString output_dir =
            QDir(bundle_adjust_dir)
                .filePath(QStringLiteral("%1_%2")
                              .arg(use_planetary_laser ? QStringLiteral("planetary_laser_range")
                                                       : (use_laser_surface ? QStringLiteral("reference_laser_surface")
                                                                            : QStringLiteral("reference_terrain")))
                              .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"))));

        double laser_association_distance_meters = 0.05;
        double laser_sigma_meters = 0.0025;
        double laser_huber_delta_meters = 0.05;
        xjw::lidar::PlanetaryLaserDataset planetary_laser_dataset;
        QString planetary_camera_coordinate_frame;
        QString planetary_camera_sensor_frame;
        bool confirm_unknown_sensor_is_frame = false;
        bool confirm_unknown_range_is_one_way = false;
        QString dialog_title = title;

        if (use_planetary_laser)
        {
            dialog_title = QStringLiteral("行星激光测距平差");
            std::string laser_error;
            if (!xjw::lidar::loadPlanetaryLaserJsonFile(xjw::common::io::toUtf8Path(planetary_laser_prior_path),
                                                        {},
                                                        &planetary_laser_dataset,
                                                        &laser_error))
            {
                warn(dialog_title,
                     QStringLiteral("读取行星激光 SI JSON 失败: %1").arg(QString::fromStdString(laser_error)));
                return;
            }
            if (planetary_laser_dataset.sensorModel == xjw::lidar::PlanetaryLaserSensorModel::LineScan)
            {
                inform(dialog_title,
                       QStringLiteral("该数据声明为 line_scan。当前 PlaScan 只有每幅影像一个静态位姿，"
                                      "尚未实现 ISIS/SPICE 的逐行时变轨迹，因而拒绝按 frame camera 误处理。"));
                return;
            }
            if (planetary_laser_dataset.rangeType == xjw::lidar::PlanetaryLaserRangeType::RoundTrip)
            {
                warn(dialog_title, QStringLiteral("该数据 range_type=round_trip，请先按产品定义换算为单程几何距离。"));
                return;
            }
            if (planetary_laser_dataset.sensorModel == xjw::lidar::PlanetaryLaserSensorModel::Unknown)
            {
                confirm_unknown_sensor_is_frame =
                    _messages && _messages->question(
                                     _parentWidget,
                                     QStringLiteral("确认相机模型"),
                                     QStringLiteral("数据没有明确 sensor_model。只有当每个 simultaneous image 可由一个"
                                                    "静态 frame-camera 位姿代表时才能继续。确认按 frame camera 处理？"),
                                     UiAnswer::No) == UiAnswer::Yes;
                if (!confirm_unknown_sensor_is_frame)
                {
                    return;
                }
            }
            if (planetary_laser_dataset.rangeType == xjw::lidar::PlanetaryLaserRangeType::Unknown)
            {
                confirm_unknown_range_is_one_way =
                    _messages && _messages->question(
                                     _parentWidget,
                                     QStringLiteral("确认测距语义"),
                                     QStringLiteral("数据没有明确 range_type。确认 range_m 已经是激光发射中心到落点的"
                                                    "单程几何距离，而不是往返光程或未换算时间？"),
                                     UiAnswer::No) == UiAnswer::Yes;
                if (!confirm_unknown_range_is_one_way)
                {
                    return;
                }
            }

            const QString dataset_frame = QString::fromStdString(planetary_laser_dataset.reference.bodyFixedFrame);
            const UiDialogResult frame =
                _messages->getText(_parentWidget,
                                   QStringLiteral("确认求解坐标系"),
                                   QStringLiteral("请输入当前 BA 相机中心和普通 tracks 所在坐标系。\n"
                                                  "当前 MVP 不执行坐标转换，必须与激光 body_fixed_frame 完全一致："),
                                   dataset_frame);
            planetary_camera_coordinate_frame = frame.text.trimmed();
            if (!frame.accepted || planetary_camera_coordinate_frame.isEmpty())
            {
                return;
            }
            if (planetary_camera_coordinate_frame != dataset_frame)
            {
                warn(dialog_title,
                     QStringLiteral("相机坐标系 %1 与激光坐标系 %2 不一致；请先完成坐标转换。")
                         .arg(planetary_camera_coordinate_frame, dataset_frame));
                return;
            }

            const bool has_non_zero_lever_arm = std::any_of(
                planetary_laser_dataset.shots.cbegin(),
                planetary_laser_dataset.shots.cend(),
                [](const xjw::lidar::PlanetaryLaserShot& shot)
                {
                    return std::hypot(std::hypot(shot.leverArmSensorMeters[0], shot.leverArmSensorMeters[1]),
                                      shot.leverArmSensorMeters[2]) > 0.0;
                });
            planetary_camera_sensor_frame = QString::fromStdString(planetary_laser_dataset.reference.laserFrame);
            if (has_non_zero_lever_arm)
            {
                const UiDialogResult sensor_frame =
                    _messages->getText(_parentWidget,
                                       QStringLiteral("确认杆臂坐标系"),
                                       QStringLiteral("数据包含非零 lever arm。请输入杆臂所用相机/传感器坐标框架；"
                                                      "当前 MVP 不执行框架旋转："),
                                       planetary_camera_sensor_frame);
                planetary_camera_sensor_frame = sensor_frame.text.trimmed();
                if (!sensor_frame.accepted || planetary_camera_sensor_frame !=
                                                  QString::fromStdString(planetary_laser_dataset.reference.laserFrame))
                {
                    warn(dialog_title, QStringLiteral("非零杆臂框架不一致，无法安全建立测距方程。"));
                    return;
                }
            }
        }

        if (use_laser_surface)
        {
            const UiDialogResult association =
                _messages->getDouble(_parentWidget,
                                     QStringLiteral("LiDAR 平差参数"),
                                     QStringLiteral("track 到 LiDAR 的最大关联距离（米）:"),
                                     laser_association_distance_meters,
                                     0.0001,
                                     1000.0,
                                     4);
            if (!association.accepted)
            {
                return;
            }
            laser_association_distance_meters = association.number;
            const UiDialogResult sigma =
                _messages->getDouble(_parentWidget,
                                     QStringLiteral("LiDAR 平差参数"),
                                     QStringLiteral("LiDAR 点到面标准差 sigma（米，权重=1/sigma²）:"),
                                     laser_sigma_meters,
                                     0.0001,
                                     1000.0,
                                     4);
            if (!sigma.accepted)
            {
                return;
            }
            laser_sigma_meters = sigma.number;
            const UiDialogResult huber = _messages->getDouble(_parentWidget,
                                                              QStringLiteral("LiDAR 平差参数"),
                                                              QStringLiteral("LiDAR Huber 阈值（米）:"),
                                                              laser_huber_delta_meters,
                                                              0.0001,
                                                              1000.0,
                                                              4);
            if (!huber.accepted)
            {
                return;
            }
            laser_huber_delta_meters = huber.number;
        }

        QString prompt;
        if (use_planetary_laser)
        {
            prompt = QStringLiteral("将使用 ISIS 风格的稀疏 laser-range shot 约束 frame-camera BA。\n\n"
                                    "数据: %1\n"
                                    "目标/坐标系: %2 / %3\n"
                                    "shot 数量: %4\n"
                                    "sensor/range: %5 / %6\n"
                                    "Huber: 3 sigma\n"
                                    "影像数量: %7\n"
                                    "输出目录: %8\n\n"
                                    "Projected/virtual image measures 不会作为真实像点；普通 track 的 RMS 也不会"
                                    "混入 shot 统计。继续执行？")
                         .arg(planetary_laser_prior_path)
                         .arg(QString::fromStdString(planetary_laser_dataset.reference.targetName))
                         .arg(QString::fromStdString(planetary_laser_dataset.reference.bodyFixedFrame))
                         .arg(static_cast<int>(planetary_laser_dataset.shots.size()))
                         .arg(QString::fromLatin1(
                             xjw::lidar::planetaryLaserSensorModelName(planetary_laser_dataset.sensorModel)))
                         .arg(QString::fromLatin1(
                             xjw::lidar::planetaryLaserRangeTypeName(planetary_laser_dataset.rangeType)))
                         .arg(images.size())
                         .arg(output_dir);
        }
        else if (use_laser_surface)
        {
            prompt = QStringLiteral("将使用扫描 LiDAR/点云 PLY 作为 BA 点到面 soft prior 重新平差。\n\n"
                                    "参考点云: %1\n"
                                    "法向字段: normal_x/normal_y/normal_z（或 nx/ny/nz）\n"
                                    "最大关联距离: %2 m\n"
                                    "标准差 sigma: %3 m（统计权重 %4）\n"
                                    "Huber 阈值: %5 m\n"
                                    "影像数量: %6\n"
                                    "输出目录: %7\n\n"
                                    "继续执行？")
                         .arg(laser_prior_path)
                         .arg(laser_association_distance_meters, 0, 'g', 8)
                         .arg(laser_sigma_meters, 0, 'g', 8)
                         .arg(1.0 / (laser_sigma_meters * laser_sigma_meters), 0, 'g', 8)
                         .arg(laser_huber_delta_meters, 0, 'g', 8)
                         .arg(images.size())
                         .arg(output_dir);
        }
        else
        {
            prompt = QStringLiteral("将使用参考 DEM 作为 BA 高程 soft prior 重新平差。\n\n"
                                    "参考 DEM: %1\n"
                                    "影像数量: %2\n"
                                    "输出目录: %3\n\n"
                                    "继续执行？")
                         .arg(dem_prior_path)
                         .arg(images.size())
                         .arg(output_dir);
        }
        if (!_messages || _messages->question(_parentWidget, dialog_title, prompt, UiAnswer::Yes) != UiAnswer::Yes)
        {
            return;
        }

        QJsonObject extra;
        if (use_planetary_laser)
        {
            extra[QStringLiteral("enable_planetary_laser_range_constraints")] = true;
            extra[QStringLiteral("planetary_laser_data_path")] = planetary_laser_prior_path;
            extra[QStringLiteral("planetary_laser_camera_coordinate_frame")] = planetary_camera_coordinate_frame;
            extra[QStringLiteral("planetary_laser_camera_sensor_frame")] = planetary_camera_sensor_frame;
            extra[QStringLiteral("planetary_laser_confirm_unknown_sensor_is_frame")] = confirm_unknown_sensor_is_frame;
            extra[QStringLiteral("planetary_laser_confirm_unknown_range_is_one_way")] =
                confirm_unknown_range_is_one_way;
            extra[QStringLiteral("planetary_laser_allow_unmapped_shots")] = false;
            extra[QStringLiteral("planetary_laser_allow_unmapped_measured_images")] = false;
            extra[QStringLiteral("planetary_laser_range_weight")] = 1.0;
            extra[QStringLiteral("planetary_laser_range_huber_delta_sigma")] = 3.0;
        }
        else if (use_laser_surface)
        {
            extra[QStringLiteral("enable_laser_constraints")] = true;
            extra[QStringLiteral("laser_constraint_cloud_path")] = laser_prior_path;
            extra[QStringLiteral("laser_association_max_distance_m")] = laser_association_distance_meters;
            extra[QStringLiteral("laser_voxel_size_m")] = 0.0;
            extra[QStringLiteral("laser_max_curvature")] = 0.2;
            extra[QStringLiteral("laser_max_samples")] = 500000;
            extra[QStringLiteral("laser_missing_normals_as_height_planes")] = false;
            extra[QStringLiteral("laser_weight")] = 0.0;
            extra[QStringLiteral("laser_sigma_m")] = laser_sigma_meters;
            extra[QStringLiteral("laser_huber_delta_m")] = laser_huber_delta_meters;
        }
        else
        {
            extra[QStringLiteral("enable_reference_terrain_prior")] = true;
            extra[QStringLiteral("reference_terrain_dem_path")] = dem_prior_path;
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

        if (use_planetary_laser)
        {
            LOG_INFO(QStringLiteral("行星激光测距 BA 启动: shots=%1 data=%2 frame=%3 images=%4 output=%5")
                         .arg(static_cast<int>(planetary_laser_dataset.shots.size()))
                         .arg(planetary_laser_prior_path, planetary_camera_coordinate_frame)
                         .arg(images.size())
                         .arg(output_dir));
        }
        else if (use_laser_surface)
        {
            LOG_INFO(QStringLiteral("LiDAR 点到面 BA soft prior 启动: cloud=%1 images=%2 output=%3")
                         .arg(laser_prior_path)
                         .arg(images.size())
                         .arg(output_dir));
        }
        else
        {
            LOG_INFO(QStringLiteral("参考地形 BA soft prior 启动: dem=%1 images=%2 output=%3")
                         .arg(dem_prior_path)
                         .arg(images.size())
                         .arg(output_dir));
        }
        if (_bundleAdjustLauncher)
        {
            _bundleAdjustLauncher(images, output_dir, qMax(1, QThread::idealThreadCount()), false, extra);
        }
    }

    void ProjectResourceService::refreshReconstructionQualityReport()
    {
        if (!_session || !_session->hasProject() || !_session->data())
        {
            return;
        }
        const auto result = writeReconstructionQualityProjectReport(_session->data());
        if (!result.saved && !result.errorMessage.isEmpty())
        {
            LOG_WARN(QStringLiteral("重建质量报告刷新失败: %1").arg(result.errorMessage));
        }
    }

    OperationResult ProjectResourceService::packResource(const QString& resourcePath)
    {
        if (!requireProject(QStringLiteral("打包资源")))
        {
            return failed(QStringLiteral("请先打开或创建项目"));
        }
        if (resourcePath.trimmed().isEmpty())
        {
            return cancelled();
        }
        const QString normalized = normalizedPath(resourcePath);
        if (!beginTask(QStringLiteral("打包资源")))
        {
            return failed(QStringLiteral("已有项目操作正在运行，请等待完成后再试。"));
        }
        struct PackWork
        {
            bool success = false;
            QString sourcePath;
            QString stagedPath;
            QString errorMessage;
            bool directory = false;
            bool created = false;
        };
        const ProjectSessionContext sessionContext = _taskContext;
        const auto cancellation = _taskToken;
        const int chunkDirectory = _session->data()->activeChunkDirectory();
        const QString projectPath = _session->projectPath();
        _taskFuture = xjw::gui::tasks::runGuardedWithOutcome(
            this,
            [normalized, projectPath, chunkDirectory, cancellation]()
            {
                PackWork work;
                if (cancellation.isCancellationRequested())
                {
                    return work;
                }
                const QFileInfo resource(normalized);
                work.sourcePath = resource.absoluteFilePath();
                work.directory = resource.isDir();
                if (!resource.isFile() && !resource.isDir())
                {
                    work.errorMessage = QStringLiteral("待打包资源不存在: %1").arg(normalized);
                    return work;
                }
                if (!ProjectWorkspaceStore(projectPath, chunkDirectory)
                         .stagePackedResource(work.sourcePath, &work.stagedPath, &work.errorMessage, &work.created))
                {
                    return work;
                }
                work.success = true;
                return work;
            },
            [sessionContext, cancellation](ProjectResourceService* self, xjw::gui::tasks::TaskOutcome<PackWork> outcome)
            {
                const auto fail = [self](const QString& message, bool present)
                {
                    if (present)
                    {
                        self->presentFailure(QStringLiteral("打包资源"), message);
                    }
                    self->finishTask();
                };
                if (!outcome.succeeded())
                {
                    fail(outcome.errorMessage.isEmpty() ? QStringLiteral("资源打包任务失败") : outcome.errorMessage,
                         true);
                    return;
                }
                const PackWork work = std::move(*outcome.value);
                if (cancellation.isCancellationRequested() || !self->isCurrent(sessionContext))
                {
                    if (work.success && work.created)
                    {
                        const QFileInfo staged(work.stagedPath);
                        if (staged.isDir())
                        {
                            QDir(work.stagedPath).removeRecursively();
                        }
                        else if (staged.exists())
                        {
                            QFile::remove(work.stagedPath);
                        }
                    }
                    fail(QStringLiteral("项目已切换，已取消资源打包。"), false);
                    return;
                }
                if (!work.success)
                {
                    fail(work.errorMessage.isEmpty() ? QStringLiteral("资源打包失败") : work.errorMessage, true);
                    return;
                }

                QJsonObject core = self->_session->coreMetadata();
                QJsonArray packed = core.value(QStringLiteral("packed_resources")).toArray();
                for (const QJsonValue& value : packed)
                {
                    if (QDir::cleanPath(value.toObject().value(QStringLiteral("path")).toString()) ==
                        QDir::cleanPath(work.stagedPath))
                    {
                        self->finishTask();
                        return;
                    }
                }
                packed.append(QJsonObject{{QStringLiteral("name"), QFileInfo(work.sourcePath).fileName()},
                                          {QStringLiteral("path"), work.stagedPath},
                                          {QStringLiteral("resource_type"),
                                           work.directory ? QStringLiteral("directory") : QStringLiteral("file")}});
                core[QStringLiteral("packed_resources")] = packed;
                self->_session->data()->updateMetadata(core, true);
                const QJsonArray committed =
                    self->_session->data()->coreFilesMeta().value(QStringLiteral("packed_resources")).toArray();
                bool committedPath = false;
                for (const QJsonValue& value : committed)
                {
                    if (QDir::cleanPath(value.toObject().value(QStringLiteral("path")).toString()) ==
                        QDir::cleanPath(work.stagedPath))
                    {
                        committedPath = true;
                        break;
                    }
                }
                if (!committedPath)
                {
                    if (work.created)
                    {
                        const QFileInfo staged(work.stagedPath);
                        if (staged.isDir())
                        {
                            QDir(work.stagedPath).removeRecursively();
                        }
                        else if (staged.exists())
                        {
                            QFile::remove(work.stagedPath);
                        }
                    }
                    fail(QStringLiteral("资源打包 metadata 提交失败"), true);
                    return;
                }
                emit self->resourceChanged();
                self->finishTask();
            });
        return success(QStringLiteral("资源打包已启动"));
    }

    OperationResult ProjectResourceService::exportPortableProject(const QString& outputPath)
    {
        if (!requireProject(QStringLiteral("导出项目")))
        {
            return failed(QStringLiteral("请先打开项目"));
        }
        if (outputPath.trimmed().isEmpty())
        {
            return cancelled();
        }
        if (!_portableExportLauncher)
        {
            return failed(QStringLiteral("便携项目导出服务未初始化"));
        }
        QString normalizedOutput = normalizedPath(outputPath);
        if (!normalizedOutput.endsWith(QStringLiteral(".zip"), Qt::CaseInsensitive))
        {
            normalizedOutput += QStringLiteral(".zip");
        }
        QString error;
        if (!_portableExportLauncher(normalizedOutput, &error))
        {
            return failed(error.isEmpty() ? QStringLiteral("无法启动便携项目导出") : error);
        }
        return success(QStringLiteral("便携项目导出已启动"));
    }

    bool ProjectResourceService::isImageImportActive() const
    {
        return _imageImportActive;
    }

    bool ProjectResourceService::isPortableExportInProgress() const
    {
        return _session && _session->busyOperation() == QStringLiteral("导出项目");
    }

    bool ProjectResourceService::isBusy() const
    {
        return _taskActive || (_session && _session->isBusy());
    }

    QString ProjectResourceService::readLastDir(const QString& key) const
    {
        if (_getLastDir)
        {
            const QString value = _getLastDir(key);
            if (!value.isEmpty())
            {
                return value;
            }
        }
        return QDir::homePath();
    }

    QString ProjectResourceService::normalizedPath(const QString& path) const
    {
        if (path.trimmed().isEmpty())
        {
            return {};
        }
        return QDir::cleanPath(QFileInfo(path.trimmed()).absoluteFilePath());
    }

    QStringList ProjectResourceService::normalizedImagePaths(const QStringList& imagePaths) const
    {
        QStringList normalized;
        normalized.reserve(imagePaths.size());
        for (const QString& path : imagePaths)
        {
            const QString cleanPath = normalizedPath(path);
            if (!cleanPath.isEmpty())
            {
                normalized.append(cleanPath);
            }
        }
        return normalized;
    }

    bool ProjectResourceService::isCurrent(const ProjectSessionContext& context) const
    {
        return !_session || _session->isCurrent(context);
    }

    void ProjectResourceService::writeLastDir(const QString& key, const QString& directory) const
    {
        if (_saveLastDir)
        {
            _saveLastDir(key, directory);
        }
    }

    OperationResult ProjectResourceService::failed(const QString& message) const
    {
        return {OperationStatus::Failed, message};
    }

    OperationResult ProjectResourceService::cancelled(const QString& message) const
    {
        return {OperationStatus::Cancelled, message};
    }

    OperationResult ProjectResourceService::success(const QString& message) const
    {
        return {OperationStatus::Success, message};
    }

    bool ProjectResourceService::requireProject(const QString& operation)
    {
        if (_session && _session->hasProject() && _session->data())
        {
            return true;
        }
        presentFailure(operation, QStringLiteral("请先打开或创建项目"));
        return false;
    }

    void ProjectResourceService::presentFailure(const QString& operation, const QString& message) const
    {
        if (_messages && !message.isEmpty())
        {
            _messages->critical(_parentWidget, operation, message);
        }
    }

    bool ProjectResourceService::beginTask(const QString& operation)
    {
        if (_taskActive || !_session || !_session->tryBeginOperation(operation))
        {
            return false;
        }
        _taskActive = true;
        _taskOperation = operation;
        _taskContext = _session->context();
        _taskToken = _cancellationSource.reset();
        return true;
    }

    void ProjectResourceService::finishTask()
    {
        _imageImportActive = false;
        if (_taskActive && _session)
        {
            _session->endOperation(_taskOperation);
        }
        _taskActive = false;
        _taskOperation.clear();
        _taskContext = {};
    }

    void ProjectResourceService::cancelTask()
    {
        _cancellationSource.requestCancellation();
    }

} // namespace xjw::gui::project
