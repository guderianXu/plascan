#include "ProjectTerrainProductsManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "project/ProjectIO.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowUtils.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "TerrainPipeline.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QMessageBox>
#include <QPointer>

#include <algorithm>
#include <cmath>

using xjw::gui::project::makeDemResultRecord;
using xjw::gui::project::makeOrthoResultRecord;
using xjw::gui::project::persistProjectMeta;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::gui::project::runDemProducts;
using xjw::gui::project::runOrthoProduct;
using xjw::gui::project::TerrainPipelineResult;
using xjw::gui::project::upsertMetaArrayRecordByPath;

namespace
{

QString normalizedAbsolutePath(const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return {};
    }
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
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
    if (_projectData)
    {
        connect(_projectData, &ProjectData::projectClosed, this,
                [this]()
                {
                    cancelMapProject();
                });
        connect(_projectData, &ProjectData::activeChunkChanged, this,
                [this](const QString &chunkId, const QString &, int)
                {
                    if (_orthoCancelFlag && chunkId != _orthoTaskChunkId)
                    {
                        cancelMapProject();
                    }
                });
    }
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

void ProjectTerrainProductsManager::startDemFromPointCloudAsync(
    const xjw::gui::project::DemGenerationRequest &request)
{
    if (!ensureProjectOpen())
    {
        return;
    }

    QString requestError;
    if (!request.validate(&requestError))
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("创建 DEM"), requestError);
        emit demPipelineFinished(false, requestError);
        return;
    }

    const QString pointCloudPath = request.sourcePointCloudPath.trimmed();
    if (!QFileInfo::exists(pointCloudPath))
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("创建 DEM"),
                             QStringLiteral("指定的点云文件不存在：\n%1").arg(pointCloudPath));
        emit demPipelineFinished(false, QStringLiteral("点云文件不存在"));
        return;
    }

    QString outDir = resolveProjectOutputDir(_owner->currentProjectPath(),
                                             request.outputDirectory.trimmed(),
                                             QStringLiteral("assets/dem/relative_dem"));

    const auto session = _owner->currentSessionContext();
    const double demResolution = request.resolution;
    const QString demType = request.dataType;
    emit demPipelineProgressChanged(QStringLiteral("DEM 生成"), 5);

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [pointCloudPath, outDir, demResolution, demType]()
        {
            return runDemProducts(pointCloudPath,
                                  outDir,
                                  demResolution,
                                  demType,
                                  false);
        },
        [pointCloudPath, outDir, demResolution, demType, session](
            ProjectTerrainProductsManager *self,
            xjw::gui::tasks::TaskOutcome<TerrainPipelineResult> outcome)
        {
            if (!self->_owner ||
                !self->_projectData ||
                !self->_owner->isCurrentSession(session))
            {
                return;
            }

            if (!outcome.succeeded())
            {
                const QString error = outcome.errorMessage.isEmpty()
                    ? QStringLiteral("DEM 后台任务失败")
                    : outcome.errorMessage;
                QMessageBox::warning(self->_parentWidget,
                                     QStringLiteral("创建相对 DEM"),
                                     QStringLiteral("处理失败：%1").arg(error));
                emit self->demPipelineFinished(false, error);
                return;
            }

            const TerrainPipelineResult terrainRun = std::move(*outcome.value);
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
            demResult[QStringLiteral("source_point_cloud")] = pointCloudPath;
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
                    .arg(pointCloudPath)
                    .arg(terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0), 0, 'f', 6));
        });
}

void ProjectTerrainProductsManager::startMapProjectAsync(
    const xjw::gui::project::OrthoGenerationRequest &request)
{
    if (!ensureProjectOpen())
    {
        emit orthoPipelineFinished(false, QStringLiteral("请先打开项目"), QJsonObject());
        return;
    }

    if (_orthoCancelFlag)
    {
        emit orthoPipelineFinished(
            false,
            QStringLiteral("已有正射影像生成任务正在运行，请等待其完成或取消"),
            QJsonObject());
        return;
    }

    const auto session = _owner->currentSessionContext();
    const QString projectPath = session.projectPath;
    const QString projectRoot =
        xjw::common::project::ProjectIO::projectRootFromPlascan(projectPath);
    const auto resolveProjectPath = [&projectPath](const QString &path)
    {
        const QString trimmed = path.trimmed();
        return trimmed.isEmpty()
            ? QString()
            : xjw::common::project::ProjectIO::resolveProjectResourcePath(
                  projectPath, trimmed);
    };

    QString requestError;
    if (!request.validate(&requestError))
    {
        emit orthoPipelineFinished(false, requestError, QJsonObject());
        return;
    }

    QJsonObject meta = _projectData->metadata();
    const bool pointCloudMode =
        request.options.surfaceType == xjw::OrthoSurfaceType::PointCloud;
    QString resolvedDem =
        resolveProjectPath(pointCloudMode ? request.pointCloudPath : request.demPath);
    QJsonObject matchedDemRecord;
    if (resolvedDem.isEmpty() && pointCloudMode)
    {
        const QJsonArray denseResults =
            meta.value(QStringLiteral("dense_cloud_results")).toArray();
        for (int index = denseResults.size() - 1; index >= 0; --index)
        {
            const QString candidate = resolveProjectPath(
                denseResults.at(index).toObject()
                    .value(QStringLiteral("dense_cloud_xyz")).toString());
            if (!candidate.isEmpty() && QFileInfo::exists(candidate))
            {
                resolvedDem = candidate;
                break;
            }
        }
    }
    if (resolvedDem.isEmpty() && !pointCloudMode)
    {
        const QJsonArray demArr = meta.value(QStringLiteral("dem_results")).toArray();
        if (!demArr.isEmpty())
        {
            for (int index = demArr.size() - 1; index >= 0; --index)
            {
                const QJsonObject record = demArr.at(index).toObject();
                const QString candidate = resolveProjectPath(
                    record.value(QStringLiteral("dem_tif")).toString());
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
    if (matchedDemRecord.isEmpty() && !pointCloudMode)
    {
        const QJsonArray demArr = meta.value(QStringLiteral("dem_results")).toArray();
        for (int index = demArr.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = demArr.at(index).toObject();
            const QString candidate = resolveProjectPath(
                record.value(QStringLiteral("dem_tif")).toString());
            if (QDir::cleanPath(candidate) == QDir::cleanPath(resolvedDem))
            {
                matchedDemRecord = record;
                break;
            }
        }
    }
    const QFileInfo resolvedDemInfo(resolvedDem);
    if (resolvedDem.isEmpty() || !resolvedDemInfo.exists() || !resolvedDemInfo.isFile())
    {
        const QString message = pointCloudMode
            ? QStringLiteral("找不到彩色点云文件，请先生成或选择包含 RGB 的稠密点云。")
            : QStringLiteral("找不到 DEM 文件，请先执行[创建 DEM]。");
        emit orthoPipelineFinished(false, message, QJsonObject());
        return;
    }

    QStringList sourceImages;
    if (!pointCloudMode)
    {
        sourceImages.reserve(request.sourceImages.size());
        for (const QString &requested_image : request.sourceImages)
        {
            const QString imagePath = resolveProjectPath(requested_image);
            if (!imagePath.isEmpty())
            {
                sourceImages.append(imagePath);
            }
        }
    }
    if (sourceImages.isEmpty() && !pointCloudMode)
    {
        sourceImages = _owner->getAllImages();
        for (QString &imagePath : sourceImages)
        {
            imagePath = resolveProjectPath(imagePath);
        }
        sourceImages.removeAll(QString());
    }
    if (sourceImages.isEmpty() && !pointCloudMode)
    {
        const QString message = QStringLiteral("项目中没有可用影像。");
        emit orthoPipelineFinished(false, message, QJsonObject());
        return;
    }

    QString out = request.outputPath.trimmed();
    if (out.isEmpty())
    {
        out = QDir(projectRoot).filePath(
            pointCloudMode
                ? QStringLiteral("assets/ortho/point_cloud_dom.tif")
                : QStringLiteral("assets/ortho/relative_dom.tif"));
    }
    else if (QFileInfo(out).isRelative() && !projectRoot.isEmpty())
    {
        out = QDir(projectRoot).filePath(out);
    }
    out = QDir::cleanPath(out);
    const QString normalizedOutput = normalizedAbsolutePath(out);
    if (normalizedOutput == normalizedAbsolutePath(resolvedDem))
    {
        emit orthoPipelineFinished(
            false,
            (pointCloudMode
                 ? QStringLiteral("正射输出路径不能覆盖输入点云：%1")
                 : QStringLiteral("正射输出路径不能覆盖输入 DEM：%1")).arg(out),
            QJsonObject());
        return;
    }
    if (normalizedOutput == normalizedAbsolutePath(projectPath))
    {
        emit orthoPipelineFinished(
            false,
            QStringLiteral("正射输出路径不能覆盖当前项目文件：%1").arg(projectPath),
            QJsonObject());
        return;
    }
    for (const QString &imagePath : sourceImages)
    {
        if (normalizedOutput == normalizedAbsolutePath(imagePath))
        {
            emit orthoPipelineFinished(
                false,
                QStringLiteral("正射输出路径不能覆盖源影像：%1").arg(imagePath),
                QJsonObject());
            return;
        }
    }

    QJsonObject resolvedSettings = request.toResolvedSettings();
    resolvedSettings[QStringLiteral("images")] = QJsonArray::fromStringList(sourceImages);
    resolvedSettings[QStringLiteral("dem_path")] = resolvedDem;
    if (pointCloudMode)
    {
        resolvedSettings[QStringLiteral("point_cloud_path")] = resolvedDem;
        resolvedSettings[QStringLiteral("dem_path")] = QString();
    }
    resolvedSettings[QStringLiteral("output_path")] = out;

    QJsonArray runtimeImages;
    const QJsonArray storedImages = meta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : storedImages)
    {
        QJsonObject image = value.toObject();
        const QString imagePath =
            resolveProjectPath(image.value(QStringLiteral("path")).toString());
        image[QStringLiteral("path")] = imagePath;
        QString maskPath =
            resolveProjectPath(image.value(QStringLiteral("mask_path")).toString());
        if (maskPath.isEmpty() || !QFileInfo::exists(maskPath))
        {
            maskPath = xjw::common::project::ProjectIO::findMaskForImage(
                projectPath, imagePath);
        }
        if (!maskPath.isEmpty())
        {
            if (normalizedOutput == normalizedAbsolutePath(maskPath))
            {
                emit orthoPipelineFinished(
                    false,
                    QStringLiteral("正射输出路径不能覆盖项目蒙版：%1")
                        .arg(maskPath),
                    QJsonObject());
                return;
            }
            image[QStringLiteral("mask_path")] = maskPath;
        }
        runtimeImages.append(image);
    }
    QJsonObject runtimeMeta = meta;
    runtimeMeta[QStringLiteral("images")] = runtimeImages;
    QJsonArray runtimeDemResults;
    const QJsonArray storedDemResults =
        meta.value(QStringLiteral("dem_results")).toArray();
    for (const QJsonValue &value : storedDemResults)
    {
        QJsonObject record = value.toObject();
        const QString demPath =
            resolveProjectPath(record.value(QStringLiteral("dem_tif")).toString());
        if (!demPath.isEmpty())
        {
            record[QStringLiteral("dem_tif")] = demPath;
        }
        runtimeDemResults.append(record);
    }
    runtimeMeta[QStringLiteral("dem_results")] = runtimeDemResults;

    const auto cancelFlag = std::make_shared<std::atomic_bool>(false);
    _orthoCancelFlag = cancelFlag;
    _orthoTaskChunkId = session.chunkId;

    emit orthoPipelineStarted();
    emit orthoPipelineProgressChanged(QStringLiteral("准备正射影像生成"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progressCallback =
        [self, cancelFlag, session](
            const QString &stage, int percent)
    {
        if (!self)
        {
            return;
        }

        QMetaObject::invokeMethod(
            self.data(),
            [self, cancelFlag, session, stage, percent]()
            {
                if (!self ||
                    self->_orthoCancelFlag != cancelFlag ||
                    !self->_owner ||
                    !self->_projectData ||
                    !self->_owner->isCurrentSession(session))
                {
                    return;
                }
                emit self->orthoPipelineProgressChanged(stage, std::clamp(percent, 0, 99));
            },
            Qt::QueuedConnection);
    };

    auto orthoWork = [sourceImages,
                      resolvedDem,
                      out,
                      resolvedSettings,
                      projectMeta = runtimeMeta,
                      cancelFlag,
                      progressCallback]() -> xjw::gui::project::TerrainPipelineResult
    {
        return runOrthoProduct(sourceImages,
                               resolvedDem,
                               out,
                               resolvedSettings,
                               projectMeta,
                               cancelFlag.get(),
                               progressCallback);
    };

    auto orthoFinished = [sourceImages,
                          resolvedDem,
                          out,
                          resolvedSettings,
                          matchedDemRecord,
                          pointCloudMode,
                          session,
                          cancelFlag](
                              ProjectTerrainProductsManager *manager,
                              xjw::gui::project::TerrainPipelineResult orthoRun)
    {
        if (manager->_orthoCancelFlag == cancelFlag)
        {
            manager->_orthoCancelFlag.reset();
            manager->_orthoTaskChunkId.clear();
        }

        const bool cancelled =
            !orthoRun.ok
            && orthoRun.error.contains(QStringLiteral("已取消"));
        if (cancelled)
        {
            emit manager->orthoPipelineFinished(
                false,
                QStringLiteral("正射影像生成已取消"),
                orthoRun.payload);
            return;
        }

        if (!manager->_owner ||
            !manager->_projectData ||
            !manager->_owner->isCurrentSession(session))
        {
            emit manager->orthoPipelineFinished(
                false,
                QStringLiteral("项目已切换，正射影像结果未写入当前项目"),
                orthoRun.payload);
            return;
        }

        if (!orthoRun.ok)
        {
            const QString error = orthoRun.error.trimmed().isEmpty()
                ? QStringLiteral("正射影像生成遇到未知错误，请检查日志。")
                : orthoRun.error;
            emit manager->orthoPipelineFinished(false, error, orthoRun.payload);
            return;
        }

        const QJsonObject orthoResult = orthoRun.payload;
        const double effectiveResolution =
            orthoResult.value(QStringLiteral("output_resolution"))
                .toDouble(orthoResult.value(QStringLiteral("pixel_size_x")).toDouble(1.0));

        QJsonObject record = makeOrthoResultRecord(
            orthoResult.value(QStringLiteral("created_at")).toString(),
            pointCloudMode ? QString() : resolvedDem,
            orthoResult.value(QStringLiteral("output_path")).toString(out),
            orthoResult.value(QStringLiteral("source_image_count"))
                .toInt(static_cast<int>(sourceImages.size())),
            sourceImages,
            true,
            effectiveResolution,
            orthoResult);
        if (!record.value(QStringLiteral("resolved_settings")).isObject())
        {
            record[QStringLiteral("resolved_settings")] = resolvedSettings;
        }
        if (pointCloudMode)
        {
            record[QStringLiteral("point_cloud_path")] = resolvedDem;
            record[QStringLiteral("source_surface_type")] = QStringLiteral("point_cloud");
        }
        if (!matchedDemRecord.isEmpty())
        {
            record[QStringLiteral("dem_reference")] =
                matchedDemRecord.value(QStringLiteral("dem_reference")).toString();
        }
        if (!manager->_projectData->upsertResultRecordByPath(
                QStringLiteral("ortho_results"),
                QStringLiteral("output_path"),
                record,
                true))
        {
            emit manager->orthoPipelineFinished(
                false,
                QStringLiteral("正射影像已写出，但项目结果记录保存失败：%1")
                    .arg(record.value(QStringLiteral("output_path")).toString()),
                record);
            return;
        }
        manager->_owner->refreshReconstructionQualityReport();

        emit manager->orthoPipelineProgressChanged(QStringLiteral("完成"), 100);
        const QString completionMessage = pointCloudMode
            ? QStringLiteral("点云正射影像已生成：%1\n覆盖率：%2%；投影点数：%3")
                .arg(record.value(QStringLiteral("output_path")).toString())
                .arg(record.value(QStringLiteral("coverage_ratio")).toDouble()
                         * 100.0, 0, 'f', 1)
                .arg(record.value(QStringLiteral("projected_point_count")).toDouble(),
                     0, 'f', 0)
            : QStringLiteral("正射影像已生成：%1\n直接覆盖率：%2%；贡献相机：%3 张")
                .arg(record.value(QStringLiteral("output_path")).toString())
                .arg(record.value(QStringLiteral("coverage_ratio")).toDouble()
                         * 100.0,
                     0,
                     'f',
                     1)
                .arg(record.value(QStringLiteral("contributing_camera_count")).toInt());
        emit manager->orthoPipelineFinished(
            true,
            completionMessage,
            record);
    };

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        std::move(orthoWork),
        [orthoFinished = std::move(orthoFinished), cancelFlag](
            ProjectTerrainProductsManager *manager,
            xjw::gui::tasks::TaskOutcome<TerrainPipelineResult> outcome) mutable
        {
            if (!outcome.succeeded())
            {
                if (manager->_orthoCancelFlag == cancelFlag)
                {
                    manager->_orthoCancelFlag.reset();
                    manager->_orthoTaskChunkId.clear();
                }
                const QString error = outcome.errorMessage.isEmpty()
                    ? QStringLiteral("正射影像后台任务失败")
                    : outcome.errorMessage;
                emit manager->orthoPipelineFinished(false, error, QJsonObject());
                return;
            }
            orthoFinished(manager, std::move(*outcome.value));
        });
}

void ProjectTerrainProductsManager::cancelMapProject()
{
    if (_orthoCancelFlag)
    {
        _orthoCancelFlag->store(true, std::memory_order_relaxed);
    }
}
