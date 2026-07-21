#include "ProjectTerrainProductsManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "DepthFrameUtils.h"
#include "project/ProjectIO.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "project/ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "ProjectWorkflowUtils.h"
#include "ProjectCameraImportService.h"
#include "MatchPhotosTask.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "Camera.h"
#include "TerrainPipeline.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QMap>
#include <QMessageBox>
#include <QPointer>
#include <QThread>
#include <QtConcurrent>
#include <QFutureWatcher>
#include <QTimer>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <memory>

using xjw::gui::project::makeDemResultRecord;
using xjw::gui::project::makeOrthoResultRecord;
using xjw::core::project::collectLatestStoredDepthFrames;
using xjw::core::project::loadDepthMatStorage;
using xjw::gui::project::persistProjectMeta;
using xjw::gui::project::findLatestProductionAtResultIndex;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::gui::project::resolveLatestDenseCloudPath;
using xjw::gui::project::runDemProducts;
using xjw::gui::project::runOrthoProduct;
using xjw::gui::project::upsertMetaArrayRecordByPath;

namespace
{

struct DirectDepthDemInput
{
    std::vector<cv::Mat> depthMaps;
    std::vector<xjw::Camera> cameras;
    QString batchDir;
    int availableFrameCount = 0;
    int loadedFrameCount = 0;
};

struct DirectDepthDemFrameRequest
{
    xjw::core::project::StoredDepthFrameRecord frame;
    xjw::Camera camera;
};

struct DirectDepthDemRequest
{
    std::vector<DirectDepthDemFrameRequest> frames;
    QString batchDir;
    int availableFrameCount = 0;
};

struct AutomaticDemGenerationTaskResult
{
    bool ok = false;
    QString error;
    QJsonObject terrainResult;
};

struct DemPipelineConnectionState
{
    QMetaObject::Connection metadataConnection;
    QMetaObject::Connection mvsFinishedConnection;
    bool disconnected = false;
};

void disconnectDemPipelineConnections(const std::shared_ptr<DemPipelineConnectionState> &state)
{
    if (!state || state->disconnected)
    {
        return;
    }
    state->disconnected = true;
    QObject::disconnect(state->metadataConnection);
    QObject::disconnect(state->mvsFinishedConnection);
}

QString normalizedAbsolutePath(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
}

bool pathIsInsideDirectory(const QString &path, const QString &directory)
{
    const QString cleanPath = normalizedAbsolutePath(path);
    const QString cleanDirectory = normalizedAbsolutePath(directory);
    if (cleanPath.isEmpty() || cleanDirectory.isEmpty())
    {
        return false;
    }

    const QString relativePath = QDir(cleanDirectory).relativeFilePath(cleanPath);
    return relativePath != QStringLiteral("..") &&
           !relativePath.startsWith(QStringLiteral("../")) &&
           !relativePath.startsWith(QStringLiteral("..\\")) &&
           !QDir::isAbsolutePath(relativePath);
}

bool cameraForTerrainImagePath(const QMap<QString, xjw::Camera> &camMap,
                               const QString &imagePath,
                               xjw::Camera *camera)
{
    if (!camera)
    {
        return false;
    }

    const QString normalizedPath = xjw::common::project::normalizePath(imagePath);
    const auto it = camMap.constFind(normalizedPath);
    if (it == camMap.constEnd())
    {
        return false;
    }

    *camera = it.value();
    return true;
}

bool prepareDirectDepthDemRequest(const QJsonObject &projectMeta,
                                  ProjectManager *owner,
                                  DirectDepthDemRequest *request,
                                  QString *errorMsg)
{
    if (!owner || !request)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("内部错误：DEM 深度图输入参数无效");
        }
        return false;
    }

    const auto storedFramesResult = collectLatestStoredDepthFrames(projectMeta);
    if (!storedFramesResult.status.ok)
    {
        if (errorMsg)
        {
            *errorMsg = storedFramesResult.status.errorMessage;
        }
        return false;
    }

    QStringList imagePaths;
    imagePaths.reserve(static_cast<int>(storedFramesResult.frames.size()));
    for (const auto &frame : storedFramesResult.frames)
    {
        imagePaths.append(frame.refImage);
    }

    bool allCamerasFound = false;
    const QMap<QString, xjw::Camera> camMap = owner->getCamerasForImages(imagePaths, &allCamerasFound);
    Q_UNUSED(allCamerasFound);

    constexpr int kMaxDirectDepthFrames = 32;
    request->batchDir = storedFramesResult.batchDir;
    request->availableFrameCount = static_cast<int>(storedFramesResult.frames.size());

    for (const auto &frame : storedFramesResult.frames)
    {
        if (static_cast<int>(request->frames.size()) >= kMaxDirectDepthFrames)
        {
            break;
        }

        xjw::Camera camera;
        if (!cameraForTerrainImagePath(camMap, frame.refImage, &camera))
        {
            LOG_WARN(QStringLiteral("[DEM流水线] 深度图缺少相机，跳过: %1").arg(frame.refImage));
            continue;
        }

        request->frames.push_back(DirectDepthDemFrameRequest{frame, camera});
    }

    if (request->frames.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("最近深度图批次没有可用于 DEM 的深度图/相机组合");
        }
        return false;
    }

    return true;
}

bool loadDirectDepthDemInput(const DirectDepthDemRequest &request,
                             DirectDepthDemInput *input,
                             QString *errorMsg)
{
    if (!input)
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("内部错误：DEM 深度图输入参数无效");
        }
        return false;
    }

    constexpr int kMaxDirectDepthSide = 2048;
    input->batchDir = request.batchDir;
    input->availableFrameCount = request.availableFrameCount;

    for (const auto &entry : request.frames)
    {
        xjw::Camera camera = entry.camera;
        const auto &frame = entry.frame;
        cv::Mat depthMap;
        const auto loadStatus = loadDepthMatStorage(frame.rawDepthPath, &depthMap);
        if (!loadStatus.ok || depthMap.empty())
        {
            LOG_WARN(QStringLiteral("[DEM流水线] 深度图读取失败，跳过: %1 error=%2")
                         .arg(frame.rawDepthPath, loadStatus.errorMessage));
            continue;
        }

        if (depthMap.type() != CV_32FC1 && depthMap.type() != CV_16UC1)
        {
            LOG_WARN(QStringLiteral("[DEM流水线] 深度图类型不支持，跳过: %1 type=%2")
                         .arg(frame.rawDepthPath)
                         .arg(depthMap.type()));
            continue;
        }

        const int maxSide = std::max(depthMap.cols, depthMap.rows);
        if (maxSide > kMaxDirectDepthSide)
        {
            const double scale = static_cast<double>(kMaxDirectDepthSide) / static_cast<double>(maxSide);
            const cv::Size targetSize(std::max(1, static_cast<int>(std::round(depthMap.cols * scale))),
                                      std::max(1, static_cast<int>(std::round(depthMap.rows * scale))));
            cv::Mat resized;
            cv::resize(depthMap, resized, targetSize, 0.0, 0.0, cv::INTER_NEAREST);
            camera = camera.scaledIntrinsics(
                static_cast<double>(targetSize.width) / static_cast<double>(depthMap.cols),
                static_cast<double>(targetSize.height) / static_cast<double>(depthMap.rows));
            depthMap = std::move(resized);
        }

        input->depthMaps.push_back(std::move(depthMap));
        input->cameras.push_back(camera);
        ++input->loadedFrameCount;
    }

    if (input->depthMaps.empty() || input->cameras.empty())
    {
        if (errorMsg)
        {
            *errorMsg = QStringLiteral("最近深度图批次没有可用于 DEM 的深度图/相机组合");
        }
        return false;
    }

    return true;
}

AutomaticDemGenerationTaskResult runAutomaticDemGenerationTask(const DirectDepthDemRequest &directDepthRequest,
                                                               const QString &directDepthError,
                                                               const QString &plyPath,
                                                               const QString &outDir,
                                                               double demResolution,
                                                               const QString &demType)
{
    AutomaticDemGenerationTaskResult result;
    QJsonObject terrainResult;
    bool demOk = false;
    QString fallbackError;

    if (!directDepthRequest.frames.empty())
    {
        DirectDepthDemInput directDepthInput;
        QString loadError;
        if (loadDirectDepthDemInput(directDepthRequest, &directDepthInput, &loadError))
        {
            LOG_INFO(QStringLiteral("[DEM流水线] 使用深度图直接生成 DEM: loaded=%1/%2 batch=%3")
                         .arg(directDepthInput.loadedFrameCount)
                         .arg(directDepthInput.availableFrameCount)
                         .arg(directDepthInput.batchDir));
            QString demErr;
            demOk = xjw::TerrainPipeline::generateDemFromDepthMaps(
                directDepthInput.depthMaps,
                directDepthInput.cameras,
                outDir,
                &terrainResult,
                &demErr);
            if (!demOk)
            {
                LOG_ERROR(QStringLiteral("[DEM流水线] 深度图直接 DEM 失败: %1，回退到点云方法").arg(demErr));
                fallbackError = demErr;
            }
            else
            {
                terrainResult[QStringLiteral("depth_input_source")] = QStringLiteral("mvs_depth_map_results");
                terrainResult[QStringLiteral("depth_input_batch_dir")] = directDepthInput.batchDir;
                terrainResult[QStringLiteral("depth_input_available_count")] = directDepthInput.availableFrameCount;
                terrainResult[QStringLiteral("depth_input_loaded_count")] = directDepthInput.loadedFrameCount;
            }
        }
        else
        {
            LOG_WARN(QStringLiteral("[DEM流水线] 深度图直接 DEM 输入加载失败: %1，回退到点云方法")
                         .arg(loadError));
            fallbackError = loadError;
        }
    }
    else
    {
        LOG_WARN(QStringLiteral("[DEM流水线] 深度图直接 DEM 输入不可用: %1，回退到点云方法")
                     .arg(directDepthError));
        fallbackError = directDepthError;
    }

    if (!demOk)
    {
        LOG_INFO(QStringLiteral("[DEM流水线] 使用点云方法生成 DEM"));
        const auto terrainRun = runDemProducts(plyPath, outDir, demResolution, demType, false);
        demOk = terrainRun.ok;
        if (terrainRun.ok)
        {
            terrainResult = terrainRun.payload;
        }
        else
        {
            LOG_ERROR(QStringLiteral("[DEM流水线] 点云 DEM 失败: %1").arg(terrainRun.error));
            fallbackError = terrainRun.error;
        }
    }

    if (!demOk)
    {
        result.error = fallbackError.isEmpty() ? QStringLiteral("DEM 生成失败") : fallbackError;
        return result;
    }

    result.ok = true;
    result.terrainResult = terrainResult;
    return result;
}

} // namespace

ProjectTerrainProductsManager::ProjectTerrainProductsManager(ProjectManager *owner,
                                                             ProjectData *projectData,
                                                             QWidget *parentWidget,
                                                             QObject *parent)
    : QObject(parent)
    , _owner(owner)
    , _projectData(projectData)
    , _parentWidget(parentWidget)
{
}

bool ProjectTerrainProductsManager::ensureProjectOpen(const QString &message,
                                                      const QString &title) const
{
    if (_projectData && _projectData->hasProject())
    {
        return true;
    }
    QMessageBox::warning(_parentWidget, title, message);
    return false;
}

void ProjectTerrainProductsManager::startStereoAndPoint2DemAsync(const QStringList &images,
                                                                 const QString &outputDir,
                                                                 int threads,
                                                                 bool genPointCloud,
                                                                 double demResolution,
                                                                 const QString &demType,
                                                                 const QString &tSrs)
{
    Q_UNUSED(threads);
    Q_UNUSED(tSrs);

    if (!ensureProjectOpen())
    {
        return;
    }

    QString denseCloudPath;
    QString denseError;
    if (!resolveLatestDenseCloudPath(_projectData, &denseCloudPath, &denseError))
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("创建相对 DEM"),
                             QStringLiteral("%1\n请先完成[工作流程 → 创建点云]。")
                                 .arg(denseError.isEmpty()
                                          ? QStringLiteral("未找到可用的密集点云结果")
                                          : denseError));
        return;
    }

    const QString projectPath = _owner ? _owner->currentProjectPath() : QString();
    QString outDir = resolveProjectOutputDir(projectPath,
                                             outputDir.trimmed(),
                                             QStringLiteral("assets/dem/relative_dem"));

    emit demPipelineProgressChanged(QStringLiteral("DEM 生成"), 5);

    const QString resolvedDenseCloud = denseCloudPath;
    QPointer<ProjectTerrainProductsManager> self(this);
    auto demWork = [self,
                    resolvedDenseCloud,
                    outDir,
                    demResolution,
                    demType,
                    genPointCloud,
                    images,
                    projectPath]()
    {
        const auto terrainRun = runDemProducts(resolvedDenseCloud,
                                               outDir,
                                               demResolution,
                                               demType,
                                               genPointCloud);
        if (!self)
        {
            return;
        }

        QMetaObject::invokeMethod(self.data(),
            [self,
             terrainRun,
             resolvedDenseCloud,
             outDir,
             demResolution,
             demType,
             images,
             projectPath]()
            {
                if (!self)
                {
                    return;
                }
                if (!self->_owner ||
                    !self->_projectData ||
                    self->_owner->currentProjectPath() != projectPath)
                {
                    return;
                }

                if (!terrainRun.ok)
                {
                    QMessageBox::warning(self->_parentWidget,
                                         QStringLiteral("创建相对 DEM"),
                                         QStringLiteral("处理失败：%1").arg(terrainRun.error));
                    emit self->demPipelineFinished(false, terrainRun.error);
                    return;
                }

                QJsonObject meta = self->_projectData->metadata();
                const QJsonObject terrainResult = terrainRun.payload;
                QJsonObject demResult = makeDemResultRecord(
                    terrainResult.value(QStringLiteral("created_at")).toString(),
                    outDir,
                    QString(),
                    terrainResult.value(QStringLiteral("dem_tif")).toString(),
                    demType,
                    demResolution,
                    QString(),
                    images);
                demResult[QStringLiteral("source_dense_cloud")] = resolvedDenseCloud;
                demResult[QStringLiteral("dem_reference")] = QStringLiteral("relative");
                demResult[QStringLiteral("depth_preview_png")] =
                    terrainResult.value(QStringLiteral("depth_png")).toString();
                demResult[QStringLiteral("relative_z_offset")] =
                    terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
                upsertMetaArrayRecordByPath(&meta,
                                            QStringLiteral("dem_results"),
                                            QStringLiteral("dem_tif"),
                                            demResult);

                persistProjectMeta(self->_projectData, meta, true);
                self->_owner->refreshReconstructionQualityReport();

                emit self->demPipelineProgressChanged(QStringLiteral("完成"), 100);
                emit self->demPipelineFinished(true, QStringLiteral("DEM 生成完成"));
                QMessageBox::information(
                    self->_parentWidget,
                    QStringLiteral("创建相对 DEM"),
                    QStringLiteral("处理完成。\nDEM: %1\n预览图: %2\n参考点云: %3\n高程基准偏移: %4")
                        .arg(terrainResult.value(QStringLiteral("dem_tif")).toString())
                        .arg(terrainResult.value(QStringLiteral("depth_png")).toString())
                        .arg(resolvedDenseCloud)
                        .arg(terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0), 0, 'f', 6));
            },
            Qt::QueuedConnection);
    };

    xjw::gui::tasks::runGuarded(this,
                                std::move(demWork),
                                [](ProjectTerrainProductsManager *) {});
}

void ProjectTerrainProductsManager::startFullDemPipelineAsync(const QStringList &images,
                                                             const QString &outputDir,
                                                             const QJsonObject &pipelineSettings)
{
    using namespace xjw::gui::project;

    if (!ensureProjectOpen())
    {
        return;
    }

    if (images.size() < 2)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("创建相对 DEM"),
                             QStringLiteral("至少需要 2 张影像才能生成立体 DEM。"));
        return;
    }

    // 步骤 1: 注册相机文件（如果提供）
    const QJsonArray camFilesArr = pipelineSettings.value(QStringLiteral("camera_files")).toArray();
    if (!camFilesArr.isEmpty())
    {
        QMap<QString, QJsonObject> cameraMetaByImage;
        QStringList importErrors;

        for (int i = 0; i < qMin(images.size(), camFilesArr.size()); ++i)
        {
            const QString imagePath = images[i];
            const QString tsaiPath = camFilesArr.at(i).toString();

            SingleCameraImportResult importResult;
            const SingleCameraImportStatus status = buildSingleCameraImport(imagePath, tsaiPath, &importResult);

            if (status == SingleCameraImportStatus::Ok)
            {
                cameraMetaByImage[importResult.imageAbsPath] = importResult.cameraMeta;
            }
            else
            {
                importErrors << QStringLiteral("%1: %2").arg(QFileInfo(imagePath).fileName(), importResult.error);
            }
        }

        if (!importErrors.isEmpty())
        {
            QMessageBox::warning(_parentWidget,
                                 QStringLiteral("创建相对 DEM"),
                                 QStringLiteral("部分相机文件导入失败：\n%1").arg(importErrors.join(QStringLiteral("\n"))));
            return;
        }

        if (!cameraMetaByImage.isEmpty())
        {
            int updatedCount = 0;
            QString writeErr;
            if (!_projectData->setImageCameras(cameraMetaByImage, &updatedCount, &writeErr))
            {
                QMessageBox::warning(_parentWidget,
                                     QStringLiteral("创建相对 DEM"),
                                     QStringLiteral("相机参数写入项目失败: %1").arg(writeErr));
                return;
            }
        }
    }

    // 步骤 2-6: 启动异步流水线任务
    DemPipelineContext ctx;
    ctx.projectPath = _owner->currentProjectPath();
    ctx.images = images;
    ctx.referenceCameras = _owner->getCamerasForImages(images);
    ctx.maskPaths = xjw::common::project::ProjectIO::maskPathsForImages(ctx.projectPath, images);
    ctx.outputDir = outputDir;
    ctx.demResolution = pipelineSettings.value(QStringLiteral("dem_resolution")).toDouble(0.0);
    ctx.demType = pipelineSettings.value(QStringLiteral("dem_type")).toString(QStringLiteral("float32"));

    // 启动后台流水线（步骤1-2同步，步骤3-5通过信号链在主线程驱动）
    // demPipelineFinished 由信号链末端（DEM完成或失败）发出，不在此处发出
    LOG_INFO(QStringLiteral("[DEM流水线] 启动后台任务（特征提取+匹配）..."));
    QPointer<ProjectTerrainProductsManager> self(this);
    xjw::gui::tasks::runGuarded(
        this,
        [self, ctx]()
        {
            if (!self)
            {
                return;
            }
            self->runFullDemPipelineInBackground(ctx);
        },
        [](ProjectTerrainProductsManager *) {});
}

void ProjectTerrainProductsManager::startDemFromDenseCloudAsync(const QString &denseCloudPath,
                                                               const QString &outputDir,
                                                               double demResolution,
                                                               const QString &demType)
{
    if (!ensureProjectOpen())
    {
        return;
    }

    QJsonObject meta = _projectData->metadata();
    QString resolvedDenseCloud = denseCloudPath.trimmed();

    // 如果未指定密集点云路径，自动查找最新的
    if (resolvedDenseCloud.isEmpty())
    {
        QString denseError;
        if (!resolveLatestDenseCloudPath(_projectData, &resolvedDenseCloud, &denseError))
        {
            QMessageBox::warning(_parentWidget,
                                 QStringLiteral("创建相对 DEM"),
                                 QStringLiteral("%1\n请先完成[工作流程 → 创建点云]。")
                                     .arg(denseError.isEmpty()
                                              ? QStringLiteral("未找到可用的密集点云结果")
                                              : denseError));
            return;
        }
    }
    else if (!QFileInfo::exists(resolvedDenseCloud))
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("创建相对 DEM"),
                             QStringLiteral("指定的密集点云文件不存在：\n%1").arg(resolvedDenseCloud));
        return;
    }

    QString outDir = resolveProjectOutputDir(_owner->currentProjectPath(),
                                             outputDir.trimmed(),
                                             QStringLiteral("assets/dem/relative_dem"));

    const QString projectPath = _owner ? _owner->currentProjectPath() : QString();
    emit demPipelineProgressChanged(QStringLiteral("DEM 生成"), 5);

    QPointer<ProjectTerrainProductsManager> self(this);
    auto demWork = [self,
                    resolvedDenseCloud,
                    outDir,
                    demResolution,
                    demType,
                    projectPath]()
    {
        const auto terrainRun = runDemProducts(resolvedDenseCloud,
                                               outDir,
                                               demResolution,
                                               demType,
                                               false);
        if (!self)
        {
            return;
        }

        QMetaObject::invokeMethod(self.data(),
            [self,
             terrainRun,
             resolvedDenseCloud,
             outDir,
             demResolution,
             demType,
             projectPath]()
            {
                if (!self)
                {
                    return;
                }
                if (!self->_owner ||
                    !self->_projectData ||
                    self->_owner->currentProjectPath() != projectPath)
                {
                    return;
                }

                if (!terrainRun.ok)
                {
                    QMessageBox::warning(self->_parentWidget,
                                         QStringLiteral("创建相对 DEM"),
                                         QStringLiteral("处理失败：%1").arg(terrainRun.error));
                    emit self->demPipelineFinished(false, terrainRun.error);
                    return;
                }

                QJsonObject meta = self->_projectData->metadata();
                const QJsonObject terrainResult = terrainRun.payload;
                QJsonObject demResult = makeDemResultRecord(
                    terrainResult.value(QStringLiteral("created_at")).toString(),
                    outDir,
                    QString(),
                    terrainResult.value(QStringLiteral("dem_tif")).toString(),
                    demType,
                    demResolution,
                    QString(),
                    QStringList());
                demResult[QStringLiteral("source_dense_cloud")] = resolvedDenseCloud;
                demResult[QStringLiteral("dem_reference")] = QStringLiteral("relative");
                demResult[QStringLiteral("depth_preview_png")] =
                    terrainResult.value(QStringLiteral("depth_png")).toString();
                demResult[QStringLiteral("relative_z_offset")] =
                    terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
                upsertMetaArrayRecordByPath(&meta,
                                            QStringLiteral("dem_results"),
                                            QStringLiteral("dem_tif"),
                                            demResult);

                persistProjectMeta(self->_projectData, meta, true);
                self->_owner->refreshReconstructionQualityReport();

                emit self->demPipelineProgressChanged(QStringLiteral("完成"), 100);
                emit self->demPipelineFinished(true, QStringLiteral("DEM 生成完成"));
                QMessageBox::information(
                    self->_parentWidget,
                    QStringLiteral("创建相对 DEM"),
                    QStringLiteral("处理完成。\nDEM: %1\n预览图: %2\n参考点云: %3\n高程基准偏移: %4")
                        .arg(terrainResult.value(QStringLiteral("dem_tif")).toString())
                        .arg(terrainResult.value(QStringLiteral("depth_png")).toString())
                        .arg(resolvedDenseCloud)
                        .arg(terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0), 0, 'f', 6));
            },
            Qt::QueuedConnection);
    };

    xjw::gui::tasks::runGuarded(this,
                                std::move(demWork),
                                [](ProjectTerrainProductsManager *) {});
}

void ProjectTerrainProductsManager::startMapProjectAsync(const QStringList &images,
                                                         const QString &demPath,
                                                         const QString &outputPath,
                                                         double resolution)
{
    if (!ensureProjectOpen())
    {
        return;
    }

    QJsonObject meta = _projectData->metadata();
    QString resolvedDem = demPath.trimmed();
    QJsonObject matchedDemRecord;
    if (resolvedDem.isEmpty())
    {
        const QJsonArray demArr = meta.value(QStringLiteral("dem_results")).toArray();
        if (!demArr.isEmpty())
        {
            for (int index = demArr.size() - 1; index >= 0; --index)
            {
                const QJsonObject record = demArr.at(index).toObject();
                const QString candidate = record.value(QStringLiteral("dem_tif")).toString();
                if (!candidate.isEmpty() && QFileInfo::exists(candidate))
                {
                    resolvedDem = candidate;
                    matchedDemRecord = record;
                    if (record.value(QStringLiteral("dem_reference")).toString() == QStringLiteral("relative"))
                    {
                        break;
                    }
                }
            }
        }
    }
    if (matchedDemRecord.isEmpty())
    {
        const QJsonArray demArr = meta.value(QStringLiteral("dem_results")).toArray();
        for (int index = demArr.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = demArr.at(index).toObject();
            if (record.value(QStringLiteral("dem_tif")).toString() == resolvedDem)
            {
                matchedDemRecord = record;
                break;
            }
        }
    }
    if (resolvedDem.isEmpty() || !QFileInfo::exists(resolvedDem))
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("生成正射影像"),
                             QStringLiteral("找不到 DEM 文件，请先执行[创建 DEM]。"));
        return;
    }

    QStringList sourceImages = images;
    if (sourceImages.isEmpty())
    {
        sourceImages = _owner->getAllImages();
    }
    if (sourceImages.isEmpty())
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("生成正射影像"),
                             QStringLiteral("项目中没有可用影像。"));
        return;
    }

    QString out = outputPath.trimmed();
    if (out.isEmpty())
    {
        const QString root = xjw::common::project::ProjectIO::projectRootFromPlascan(_owner->currentProjectPath());
        out = QDir(root).filePath(QStringLiteral("assets/ortho/relative_dom.tif"));
    }
    out = QDir::cleanPath(out);

    const double requestedResolution = resolution > 0.0 ? resolution : 0.0;

    const QString projectPath = _owner ? _owner->currentProjectPath() : QString();
    emit demPipelineProgressChanged(QStringLiteral("正射影像生成"), 5);

    QPointer<ProjectTerrainProductsManager> self(this);
    auto orthoWork = [self,
                      sourceImages,
                      resolvedDem,
                      out,
                      requestedResolution,
                      projectMeta = meta,
                      matchedDemRecord,
                      projectPath]()
    {
        const auto orthoRun = runOrthoProduct(sourceImages,
                                              resolvedDem,
                                              out,
                                              requestedResolution,
                                              projectMeta);
        if (!self)
        {
            return;
        }

        QMetaObject::invokeMethod(self.data(),
            [self,
             orthoRun,
             sourceImages,
             resolvedDem,
             requestedResolution,
             matchedDemRecord,
             projectPath]()
            {
                if (!self)
                {
                    return;
                }
                if (!self->_owner ||
                    !self->_projectData ||
                    self->_owner->currentProjectPath() != projectPath)
                {
                    return;
                }

                if (!orthoRun.ok)
                {
                    QMessageBox::warning(self->_parentWidget,
                                         QStringLiteral("生成正射影像"),
                                         QStringLiteral("处理失败：%1").arg(orthoRun.error));
                    emit self->demPipelineFinished(false, orthoRun.error);
                    return;
                }

                QJsonObject meta = self->_projectData->metadata();
                const QJsonObject orthoResult = orthoRun.payload;
                const double effectiveResolution = orthoResult.value(QStringLiteral("output_resolution"))
                                                       .toDouble(requestedResolution > 0.0 ? requestedResolution : 1.0);

                QJsonObject record = makeOrthoResultRecord(
                    orthoResult.value(QStringLiteral("created_at")).toString(),
                    resolvedDem,
                    orthoResult.value(QStringLiteral("output_path")).toString(),
                    orthoResult.value(QStringLiteral("source_image_count")).toInt(),
                    sourceImages,
                    true,
                    effectiveResolution);
                record[QStringLiteral("dom_georeferenced")] =
                    orthoResult.value(QStringLiteral("dom_georeferenced")).toBool(false);
                record[QStringLiteral("projection_wkt_present")] =
                    orthoResult.value(QStringLiteral("projection_wkt_present")).toBool(false);
                if (!matchedDemRecord.isEmpty())
                {
                    record[QStringLiteral("dem_reference")] =
                        matchedDemRecord.value(QStringLiteral("dem_reference")).toString();
                }
                upsertMetaArrayRecordByPath(&meta,
                                            QStringLiteral("ortho_results"),
                                            QStringLiteral("output_path"),
                                            record);
                persistProjectMeta(self->_projectData, meta, true);
                self->_owner->refreshReconstructionQualityReport();

                emit self->demPipelineProgressChanged(QStringLiteral("完成"), 100);
                emit self->demPipelineFinished(true, QStringLiteral("正射影像生成完成"));
                QMessageBox::information(
                    self->_parentWidget,
                    QStringLiteral("生成正射影像"),
                    QStringLiteral("正射影像已生成：\n%1\n分辨率: %2 m/px\n与 DEM 投影一致: %3")
                        .arg(record.value(QStringLiteral("output_path")).toString())
                        .arg(record.value(QStringLiteral("resolution")).toDouble(), 0, 'f', 3)
                        .arg(record.value(QStringLiteral("dom_georeferenced")).toBool(false)
                                 ? QStringLiteral("是")
                                 : QStringLiteral("否")));
            },
            Qt::QueuedConnection);
    };

    xjw::gui::tasks::runGuarded(this,
                                std::move(orthoWork),
                                [](ProjectTerrainProductsManager *) {});
}
void ProjectTerrainProductsManager::runFullDemPipelineInBackground(const DemPipelineContext &ctx)
{
    QPointer<ProjectTerrainProductsManager> self(this);
    QPointer<QWidget> parentWidget(_parentWidget);
    LOG_INFO(QStringLiteral("[DEM流水线] ========== 流水线启动 =========="));
    LOG_INFO(QStringLiteral("[DEM流水线] 影像数量: %1").arg(ctx.images.size()));
    for (int i = 0; i < ctx.images.size(); ++i)
        LOG_INFO(QStringLiteral("[DEM流水线]   [%1] %2").arg(i).arg(ctx.images[i]));
    LOG_INFO(QStringLiteral("[DEM流水线] 连接点算法: SIFT + LightGlue"));
    LOG_INFO(QStringLiteral("[DEM流水线] 输出目录: %1").arg(ctx.outputDir.isEmpty() ? QStringLiteral("(自动)") : ctx.outputDir));

    // 步骤 1-2: 使用统一的连接点任务完成影像对规划、特征提取、匹配、几何验证和轨迹构建。
    LOG_INFO(QStringLiteral("[DEM流水线] ── 步骤 1-2/5: 创建连接点 (SIFT + LightGlue) ──"));
    emit demPipelineProgressChanged(QStringLiteral("创建连接点"), 0);
    std::atomic<bool> cancelFlag(false);
    std::atomic<int> progressCount(0);
    QPointer<ProjectManager> ownerGuard(_owner);

    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = false;
    options.featureAlgorithm = QStringLiteral("sift");
    options.matcherAlgorithm = QStringLiteral("lightglue");
    options.device = xjw::matchphotos::ComputeDevice::Auto;
    options.pairPolicy.exhaustiveMaxImages = 80;
    options.pairPolicy.sequenceWindow = 4;
    options.useReferencePreselection = !ctx.referenceCameras.isEmpty();

    xjw::matchphotos::MatchPhotosContext context;
    context.projectPath = ctx.projectPath;
    context.workingDirectory = xjw::common::project::ProjectIO::projectAssetsDir(ctx.projectPath);
    context.featureDirectory = xjw::common::project::ProjectIO::ipfindOutputDir(ctx.projectPath);
    context.matchDirectory = xjw::common::project::ProjectIO::ipmatchOutputDir(ctx.projectPath);
    context.pairInput.images = ctx.images;
    context.referenceCameras = ctx.referenceCameras;
    context.maskPaths = ctx.maskPaths;
    context.cancelFlag = &cancelFlag;
    context.progressCount = &progressCount;
    context.progressCallback = [self](const QString &, const QString &stage, int current, int total)
    {
        if (!self)
        {
            return;
        }
        const int stagePercent = total > 0 ? qBound(0, current * 40 / total, 40) : 0;
        emit self->demPipelineProgressChanged(stage, stagePercent);
    };

    const xjw::matchphotos::MatchPhotosTask task(options);
    const xjw::matchphotos::MatchPhotosResult matchPhotosResult = task.run(context);
    if (!matchPhotosResult.success)
    {
        const QString error = matchPhotosResult.errorMessage.isEmpty()
            ? QStringLiteral("连接点创建失败")
            : matchPhotosResult.errorMessage;
        LOG_ERROR(QStringLiteral("[DEM流水线] ✗ %1").arg(error));
        if (self)
        {
            emit self->demPipelineFinished(false, error);
        }
        return;
    }

    QVector<ProjectIpfindResultRecord> featureRecords;
    featureRecords.reserve(static_cast<int>(matchPhotosResult.features.size()));
    for (const xjw::matchphotos::MatchPhotosFeatureRecord &feature : matchPhotosResult.features)
    {
        featureRecords.push_back(ProjectIpfindResultRecord{feature.imagePath, feature.featurePath, feature.settings});
    }

    QVector<ProjectIpmatchResultRecord> matchRecords;
    matchRecords.reserve(static_cast<int>(matchPhotosResult.matches.size()));
    for (const xjw::matchphotos::MatchPhotosMatchRecord &match : matchPhotosResult.matches)
    {
        QJsonObject matchSettings = match.settings;
        if (!matchPhotosResult.tiePointPath.isEmpty())
        {
            matchSettings[QStringLiteral("tie_point_path")] = matchPhotosResult.tiePointPath;
            matchSettings[QStringLiteral("track_count")] = matchPhotosResult.trackCount;
            matchSettings[QStringLiteral("track_summary")] = matchPhotosResult.trackSummary;
        }
        matchRecords.push_back(ProjectIpmatchResultRecord{QStringList{match.matchPath}, matchSettings});
    }

    bool writebackOk = false;
    if (ownerGuard)
    {
        const Qt::ConnectionType connectionType = ownerGuard->thread() == QThread::currentThread()
            ? Qt::DirectConnection
            : Qt::BlockingQueuedConnection;
        QMetaObject::invokeMethod(ownerGuard.data(),
                                  [ownerGuard, projectPath = ctx.projectPath, featureRecords, matchRecords, &writebackOk]()
        {
            if (!ownerGuard || ownerGuard->currentProjectPath() != projectPath)
            {
                return;
            }
            ownerGuard->appendIpfindResults(featureRecords);
            ownerGuard->appendIpmatchResults(matchRecords);
            writebackOk = true;
        }, connectionType);
    }
    if (!writebackOk)
    {
        const QString error = QStringLiteral("项目已切换，连接点结果未写回，流水线中止");
        LOG_ERROR(QStringLiteral("[DEM流水线] ✗ %1").arg(error));
        if (self)
        {
            emit self->demPipelineFinished(false, error);
        }
        return;
    }
    LOG_INFO(QStringLiteral("[DEM流水线] ✓ 连接点创建完成并已写回项目"));

    // 步骤 3-5: 正式 SfM/BA 稀疏云 → MVS → DEM 需要在主线程通过信号链驱动
    LOG_INFO(QStringLiteral("[DEM流水线] ── 步骤 3/5: 检查正式 SfM/BA 稀疏点云（切换到主线程信号链）──"));
    emit demPipelineProgressChanged(QStringLiteral("检查正式空三结果"), 40);

    const QString demOutputDir = ctx.outputDir;
    const double demResolution = ctx.demResolution;
    const QString demType = ctx.demType;
    if (!self)
    {
        return;
    }
    QMetaObject::invokeMethod(self.data(), [self, demOutputDir, demResolution, demType]()
    {
        if (!self)
        {
            return;
        }

        const int productionAtIndex = findLatestProductionAtResultIndex(self->_projectData->metadata());
        if (productionAtIndex < 0)
        {
            LOG_ERROR(QStringLiteral("[DEM流水线] ✗ 未找到正式 SfM/BA 稀疏点云，拒绝使用两视预览云继续 DEM 流水线"));
            QMessageBox::warning(self->_parentWidget,
                                 QStringLiteral("创建相对 DEM"),
                                 QStringLiteral("未找到可用的正式 SfM/BA 稀疏点云结果。\n请先运行[工作流程 → 三维重建/空三]，不要使用两视预览三角化作为 DEM 输入。"));
            emit self->demPipelineFinished(false, QStringLiteral("缺少正式 SfM/BA 稀疏点云"));
            return;
        }

        LOG_INFO(QStringLiteral("[DEM流水线] ✓ 使用正式 SfM/BA 稀疏点云结果 index=%1，启动 MVS（CUDA PatchMatch）...")
                     .arg(productionAtIndex));
        emit self->demPipelineProgressChanged(QStringLiteral("密集重建 (MVS)"), 55);

        const QString outDir = resolveProjectOutputDir(self->_owner->currentProjectPath(),
                                                       demOutputDir.trimmed(),
                                                       QStringLiteral("assets/dem/relative_dem"));
        const QString expectedMvsOutputDir = QDir(outDir).filePath(QStringLiteral("mvs"));

        // 监听本次 MVS 输出：由密集重建管理器在写入 dense_cloud_results 后携带明确 PLY 路径发出。
        auto connections = std::make_shared<DemPipelineConnectionState>();
        connections->metadataConnection = connect(self->_owner, &ProjectManager::denseCloudResultReady, self.data(),
            [self, connections, outDir, expectedMvsOutputDir, demResolution, demType](
                const QString &denseCloudPath,
                int pointCount)
            {
                if (!self)
                {
                    return;
                }
                const QString plyPath = denseCloudPath.trimmed();
                LOG_INFO(QStringLiteral("[DEM流水线] denseCloudResultReady: path=%1 points=%2 exists=%3")
                             .arg(plyPath)
                             .arg(pointCount)
                             .arg(QFileInfo::exists(plyPath) ? QStringLiteral("true") : QStringLiteral("false")));
                if (plyPath.isEmpty() || !QFileInfo::exists(plyPath))
                {
                    return;
                }
                if (!pathIsInsideDirectory(plyPath, expectedMvsOutputDir))
                {
                    LOG_WARN(QStringLiteral("[DEM流水线] 忽略非本次 MVS 输出的密集点云: path=%1 expected_dir=%2")
                                 .arg(plyPath, expectedMvsOutputDir));
                    return;
                }

                disconnectDemPipelineConnections(connections);

                LOG_INFO(QStringLiteral("[DEM流水线] ✓ 密集点云就绪（%1），启动 DEM 生成...").arg(plyPath));
                emit self->demPipelineProgressChanged(QStringLiteral("DEM 生成"), 85);

                DirectDepthDemRequest directDepthRequest;
                QString directDepthError;
                const QJsonObject currentMeta = self->_projectData->metadata();
                if (!prepareDirectDepthDemRequest(currentMeta, self->_owner, &directDepthRequest, &directDepthError))
                {
                    LOG_WARN(QStringLiteral("[DEM流水线] 深度图直接 DEM 输入不可用: %1，后台任务将回退到点云方法")
                                 .arg(directDepthError));
                }

                const QString projectPath = self->_owner ? self->_owner->currentProjectPath() : QString();
                xjw::gui::tasks::runGuarded(
                    self.data(),
                    [directDepthRequest, directDepthError, plyPath, outDir, demResolution, demType]()
                    {
                        return runAutomaticDemGenerationTask(
                            directDepthRequest,
                            directDepthError,
                            plyPath,
                            outDir,
                            demResolution,
                            demType);
                    },
                    [projectPath, plyPath, outDir, demResolution, demType](
                        ProjectTerrainProductsManager *manager,
                        const AutomaticDemGenerationTaskResult &taskResult)
                    {
                        if (!manager->_owner ||
                            !manager->_projectData ||
                            manager->_owner->currentProjectPath() != projectPath)
                        {
                            return;
                        }

                        if (!taskResult.ok)
                        {
                            LOG_ERROR(QStringLiteral("[DEM流水线] ✗ DEM 生成失败: %1").arg(taskResult.error));
                            QMessageBox::warning(manager->_parentWidget,
                                                 QStringLiteral("创建相对 DEM"),
                                                 QStringLiteral("处理失败：%1").arg(taskResult.error));
                            emit manager->demPipelineFinished(false, QStringLiteral("DEM 生成失败"));
                            return;
                        }

                        QJsonObject metaUpdated = manager->_projectData->metadata();
                        const QJsonObject terrainResult = taskResult.terrainResult;
                        QJsonObject demResult = makeDemResultRecord(
                            terrainResult.value(QStringLiteral("created_at")).toString(),
                            outDir, QString(),
                            terrainResult.value(QStringLiteral("dem_tif")).toString(),
                            demType, demResolution, QString(), QStringList());
                        demResult[QStringLiteral("source_dense_cloud")] = plyPath;
                        demResult[QStringLiteral("dem_reference")] = QStringLiteral("relative");
                        demResult[QStringLiteral("depth_preview_png")] =
                            terrainResult.value(QStringLiteral("depth_png")).toString();
                        demResult[QStringLiteral("relative_z_offset")] =
                            terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
                        demResult[QStringLiteral("method")] = terrainResult.value(QStringLiteral("method")).toString();
                        demResult[QStringLiteral("depth_input_source")] =
                            terrainResult.value(QStringLiteral("depth_input_source")).toString();
                        demResult[QStringLiteral("depth_input_batch_dir")] =
                            terrainResult.value(QStringLiteral("depth_input_batch_dir")).toString();
                        demResult[QStringLiteral("depth_input_available_count")] =
                            terrainResult.value(QStringLiteral("depth_input_available_count")).toInt(0);
                        demResult[QStringLiteral("depth_input_loaded_count")] =
                            terrainResult.value(QStringLiteral("depth_input_loaded_count")).toInt(0);
                        upsertMetaArrayRecordByPath(&metaUpdated,
                                                    QStringLiteral("dem_results"),
                                                    QStringLiteral("dem_tif"),
                                                    demResult);
                        persistProjectMeta(manager->_projectData, metaUpdated, true);
                        manager->_owner->refreshReconstructionQualityReport();

                        LOG_INFO(QStringLiteral("[DEM流水线] ✓ DEM 生成完成: %1")
                                     .arg(terrainResult.value(QStringLiteral("dem_tif")).toString()));
                        emit manager->demPipelineProgressChanged(QStringLiteral("完成"), 100);
                        emit manager->demPipelineFinished(true, QStringLiteral("DEM 流水线已完成"));
                    });
            });

        // 同时监听 MVS 失败
        connections->mvsFinishedConnection = connect(self->_owner, &ProjectManager::mvsProgressFinished, self.data(),
            [self, connections](bool mvsSuccess)
            {
                if (!self)
                {
                    return;
                }
                LOG_INFO(QStringLiteral("[DEM流水线] mvsProgressFinished: success=%1").arg(mvsSuccess));
                if (mvsSuccess)
                {
                    return; // 成功时由 denseCloudResultReady 处理
                }
                disconnectDemPipelineConnections(connections);
                LOG_ERROR(QStringLiteral("[DEM流水线] ✗ MVS 失败（mvsProgressFinished(false)）"));
                emit self->demPipelineFinished(false, QStringLiteral("MVS 密集重建失败"));
            }, Qt::SingleShotConnection);

        LOG_INFO(QStringLiteral("[DEM流水线] 主线程：调用 startGenerateDenseCloudAsync..."));
        QJsonObject mvsSettings;
        mvsSettings[QStringLiteral("pipeline_mode")] = true;
        mvsSettings[QStringLiteral("at_index")] = productionAtIndex;
        mvsSettings[QStringLiteral("output_dir")] = expectedMvsOutputDir;
        // 两视图立体对：全分辨率，不降采样
        mvsSettings[QStringLiteral("resScale")] = 1.0;
        mvsSettings[QStringLiteral("iterations")] = 16;
        mvsSettings[QStringLiteral("cuda")] = true;
        mvsSettings[QStringLiteral("minConsistentViews")] = 1;
        self->_owner->startGenerateDenseCloudAsync(mvsSettings);
    }, Qt::QueuedConnection);
}
