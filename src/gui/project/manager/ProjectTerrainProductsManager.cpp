#include "ProjectTerrainProductsManager.h"

#include "ProjectManager.h"
#include "project/ProjectSessionModel.h"
#include "project/ProjectIO.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowOperations.h"
#include "ProjectOpenGuard.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "DemDomIO.h"
#include "TerrainPipeline.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMessageBox>
#include <QPointer>
#include <QRegularExpression>
#include <QSaveFile>
#include <QUuid>

#include <algorithm>
#include <cmath>

using xjw::gui::project::makeDemResultRecord;
using xjw::gui::project::makeOrthoResultRecord;
using xjw::gui::project::persistProjectMeta;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::core::project::runDemProducts;
using xjw::core::project::runOrthoProduct;
using xjw::core::project::TerrainPipelineResult;
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

bool pathsReferToSameLocation(const QString &left, const QString &right)
{
#if defined(Q_OS_WIN)
    constexpr Qt::CaseSensitivity case_sensitivity = Qt::CaseInsensitive;
#else
    constexpr Qt::CaseSensitivity case_sensitivity = Qt::CaseSensitive;
#endif
    return normalizedAbsolutePath(left).compare(
               normalizedAbsolutePath(right), case_sensitivity) == 0;
}

QString safeStorageComponent(QString value)
{
    value = value.trimmed();
    value.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]")),
                  QStringLiteral("_"));
    return value.isEmpty() ? QStringLiteral("default_chunk") : value;
}

bool isSmallBodyGlobalDemRecord(const QJsonObject &record)
{
    return record.value(QStringLiteral("terrain_mode")).toString()
        == QLatin1String("small_body_global");
}

bool isSmallBodyGlobalDemFile(const QString &path)
{
    xjw::DemGridData metadata;
    QString metadata_error;
    if (!xjw::DemDomIO::readDemMetadata(path, &metadata, &metadata_error))
    {
        return false;
    }

    const auto &items = metadata.projection.metadata;
    const QString vertical_reference =
        items.value(QStringLiteral("VERTICAL_REFERENCE")).trimmed().toLower();
    if (vertical_reference == QLatin1String("radial_distance_from_body_center")
        || vertical_reference == QLatin1String("elevation_above_reference_radius"))
    {
        return true;
    }

    return !items.value(QStringLiteral("BODY_FIXED_FRAME")).trimmed().isEmpty()
        && items.value(QStringLiteral("LATITUDE_TYPE")).compare(
               QLatin1String("planetocentric"), Qt::CaseInsensitive) == 0
        && items.value(QStringLiteral("LONGITUDE_DOMAIN")).compare(
               QLatin1String("0_360"), Qt::CaseInsensitive) == 0;
}

bool pathIsInsideDirectory(const QString &directoryPath, const QString &path)
{
    const QString relative = QDir::fromNativeSeparators(
        QDir(directoryPath).relativeFilePath(path));
    return relative != QLatin1String("..")
        && !relative.startsWith(QLatin1String("../"))
        && !QFileInfo(relative).isAbsolute();
}

QString portableReportPath(const QString &path,
                           const QString &projectRoot,
                           const QString &reportDirectory)
{
    if (path.trimmed().isEmpty() || QFileInfo(path).isRelative())
    {
        return QDir::fromNativeSeparators(path);
    }
    if (!pathIsInsideDirectory(projectRoot, path))
    {
        return QDir::fromNativeSeparators(path);
    }
    return QDir::fromNativeSeparators(
        QDir(reportDirectory).relativeFilePath(path));
}

QString projectStoragePath(const QString &projectRoot, const QString &path)
{
    if (path.trimmed().isEmpty() || QFileInfo(path).isRelative()
        || !pathIsInsideDirectory(projectRoot, path))
    {
        return QDir::fromNativeSeparators(path);
    }
    return QDir::fromNativeSeparators(QDir(projectRoot).relativeFilePath(path));
}

bool copyPortableReportAtomically(const QString &sourcePath,
                                  const QString &destinationPath,
                                  const QString &projectRoot,
                                  QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (sourcePath.trimmed().isEmpty() || destinationPath.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("全球地形报告路径为空。");
        }
        return false;
    }
    if (normalizedAbsolutePath(sourcePath) == normalizedAbsolutePath(destinationPath))
    {
        return QFileInfo::exists(sourcePath);
    }

    QFile source(sourcePath);
    if (!source.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取生成的全球地形报告：%1（%2）")
                                .arg(sourcePath, source.errorString());
        }
        return false;
    }
    const QByteArray contents = source.readAll();
    if (source.error() != QFileDevice::NoError)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("读取生成的全球地形报告失败：%1（%2）")
                                .arg(sourcePath, source.errorString());
        }
        return false;
    }

    QJsonParseError parse_error;
    const QJsonDocument source_document = QJsonDocument::fromJson(contents, &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !source_document.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("生成的全球地形报告不是有效 JSON：%1（%2）")
                                .arg(sourcePath, parse_error.errorString());
        }
        return false;
    }

    const QString destination_dir = QFileInfo(destinationPath).absolutePath();
    if (!QDir().mkpath(destination_dir))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建项目报告目录：%1").arg(destination_dir);
        }
        return false;
    }

    QJsonObject portable_report = source_document.object();
    portable_report[QStringLiteral("source_surface")] = portableReportPath(
        portable_report.value(QStringLiteral("source_surface")).toString(),
        projectRoot,
        destination_dir);
    QJsonObject artifacts = portable_report.value(QStringLiteral("artifacts")).toObject();
    for (auto iterator = artifacts.begin(); iterator != artifacts.end(); ++iterator)
    {
        if (iterator.value().isString())
        {
            iterator.value() = portableReportPath(
                iterator.value().toString(), projectRoot, destination_dir);
        }
    }
    portable_report[QStringLiteral("artifacts")] = artifacts;
    portable_report[QStringLiteral("path_semantics")] =
        QStringLiteral("relative paths are relative to this report JSON");
    const QByteArray portable_contents =
        QJsonDocument(portable_report).toJson(QJsonDocument::Indented);

    QSaveFile destination(destinationPath);
    if (!destination.open(QIODevice::WriteOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入项目全球地形报告：%1（%2）")
                                .arg(destinationPath, destination.errorString());
        }
        return false;
    }
    if (destination.write(portable_contents) != portable_contents.size()
        || !destination.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("原子提交项目全球地形报告失败：%1（%2）")
                                .arg(destinationPath, destination.errorString());
        }
        return false;
    }
    return true;
}

TerrainPipelineResult runSmallBodyGlobalProducts(
    const QString &surfacePath,
    const QString &outputDir,
    const QString &projectReportPath,
    const QString &projectRoot,
    const xjw::SmallBodyGlobalOptions &options,
    const std::atomic_bool *cancelFlag,
    const xjw::TerrainPipeline::OrthoProgressCallback &progressCallback)
{
    TerrainPipelineResult result;
    const auto rollback_run = [&]()
    {
        QFile::remove(projectReportPath);
        QDir(outputDir).removeRecursively();
    };
    result.ok = xjw::TerrainPipeline::generateSmallBodyGlobalProducts(
        surfacePath, outputDir, options, &result.payload, &result.error,
        cancelFlag, progressCallback);
    if (!result.ok)
    {
        rollback_run();
        return result;
    }

    if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
    {
        result.ok = false;
        result.error = QStringLiteral("小天体全球 DEM/DOM 生成已取消。");
        rollback_run();
        return result;
    }

    const QString generated_report =
        result.payload.value(QStringLiteral("report_json")).toString();
    QString copy_error;
    if (!copyPortableReportAtomically(
            generated_report, projectReportPath, projectRoot, &copy_error))
    {
        result.ok = false;
        result.error = QStringLiteral("全球产品已生成，但报告复制到项目目录失败：%1")
                           .arg(copy_error);
        rollback_run();
        return result;
    }
    if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
    {
        result.ok = false;
        result.error = QStringLiteral("小天体全球 DEM/DOM 生成已取消。");
        rollback_run();
        return result;
    }
    result.payload[QStringLiteral("project_report_json")] = projectReportPath;
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
    if (_owner)
    {
        connect(_owner, &ProjectManager::projectSessionChanged, this,
                [this]()
                {
                    cancelDemGeneration();
                    cancelMapProject();
                });
    }
    if (_projectData)
    {
        connect(_projectData, &ProjectData::projectClosed, this,
                [this]()
                {
                    cancelDemGeneration();
                    cancelMapProject();
                });
        connect(_projectData, &ProjectData::activeChunkChanged, this,
                [this](const QString &chunkId, const QString &, int)
                {
                    if (_demCancelFlag && chunkId != _demTaskChunkId)
                    {
                        cancelDemGeneration();
                    }
                    if (_orthoCancelFlag && chunkId != _orthoTaskChunkId)
                    {
                        cancelMapProject();
                    }
                });
    }
}

void ProjectTerrainProductsManager::startDemFromPointCloudAsync(
    const xjw::gui::project::DemGenerationRequest &request)
{
    if (!xjw::gui::project::requireOpenProject(_projectData, _parentWidget))
    {
        emit demPipelineFinished(false, QStringLiteral("请先打开项目。"));
        return;
    }

    if (!_demTaskId.isEmpty())
    {
        const QString message = QStringLiteral(
            "已有 DEM/DOM 任务正在运行，请等待其完成后再启动新任务。");
        QMessageBox::warning(_parentWidget, QStringLiteral("创建 DEM/DOM"), message);
        emit demPipelineFinished(false, message);
        return;
    }

    QString requestError;
    if (!request.validate(&requestError))
    {
        QMessageBox::warning(_parentWidget, QStringLiteral("创建 DEM"), requestError);
        emit demPipelineFinished(false, requestError);
        return;
    }

    if (request.isSmallBodyGlobal())
    {
        startSmallBodyGlobalAsync(request);
        return;
    }
    if (request.isImageStereo())
    {
        startRpcStereoDemAsync(request);
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

    const QString background_task_id = QStringLiteral("dem:%1").arg(
        QUuid::createUuid().toString(QUuid::WithoutBraces));
    QString outDir = resolveProjectOutputDir(_owner->currentProjectPath(),
                                             request.outputDirectory.trimmed(),
                                             QStringLiteral("assets/dem/relative_dem"));

    const auto session = _owner->currentSessionContext();
    const double demResolution = request.resolution;
    const QString demType = request.dataType;
    _demTaskId = background_task_id;
    _demTaskChunkId = session.chunkId;
    emit backgroundTaskProgressChanged(background_task_id, 5, 100);
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
        [pointCloudPath, outDir, demResolution, demType, session, background_task_id](
            ProjectTerrainProductsManager *self,
            xjw::gui::tasks::TaskOutcome<TerrainPipelineResult> outcome)
        {
            emit self->backgroundTaskFinished(background_task_id);
            if (self->_demTaskId != background_task_id)
            {
                return;
            }
            self->_demTaskId.clear();
            self->_demTaskChunkId.clear();
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

void ProjectTerrainProductsManager::startSmallBodyGlobalAsync(
    const xjw::gui::project::DemGenerationRequest &request)
{
    if (!_demTaskId.isEmpty())
    {
        const QString message = QStringLiteral("已有小天体全球 DEM/DOM 任务正在运行，请等待其完成。");
        QMessageBox::warning(_parentWidget, QStringLiteral("创建全球 DEM/DOM"), message);
        emit demPipelineFinished(false, message);
        return;
    }

    const auto session = _owner->currentSessionContext();
    const QString surface_path =
        xjw::common::project::ProjectIO::resolveProjectResourcePath(
            session.projectPath, request.sourceSurfacePath.trimmed());
    const QFileInfo surface_info(surface_path);
    if (surface_path.isEmpty() || !surface_info.exists() || !surface_info.isFile())
    {
        const QString message = QStringLiteral("指定的体固连三角网格不存在：\n%1")
                                    .arg(surface_path.isEmpty()
                                             ? request.sourceSurfacePath
                                             : surface_path);
        QMessageBox::warning(_parentWidget, QStringLiteral("创建全球 DEM/DOM"), message);
        emit demPipelineFinished(false, message);
        return;
    }

    const QString suffix = surface_info.suffix().toLower();
    if (suffix != QLatin1String("ply") && suffix != QLatin1String("obj"))
    {
        const QString message = QStringLiteral("全球 DEM/DOM 输入必须是 PLY 或 OBJ 三角网格：\n%1")
                                    .arg(surface_path);
        QMessageBox::warning(_parentWidget, QStringLiteral("创建全球 DEM/DOM"), message);
        emit demPipelineFinished(false, message);
        return;
    }

    const QString output_root = resolveProjectOutputDir(
        session.projectPath,
        request.outputDirectory.trimmed(),
        QStringLiteral("assets/dem/small_body_global"));
    if (output_root.isEmpty())
    {
        const QString message = QStringLiteral("无法解析全球 DEM/DOM 输出根目录。");
        QMessageBox::warning(_parentWidget, QStringLiteral("创建全球 DEM/DOM"), message);
        emit demPipelineFinished(false, message);
        return;
    }
    const QString chunk_component = safeStorageComponent(session.chunkId);
    const QString run_component = QStringLiteral("%1_%2")
        .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz")),
             QUuid::createUuid().toString(QUuid::WithoutBraces));
    const QString output_dir = QDir(output_root).filePath(
        QStringLiteral("%1/%2").arg(chunk_component, run_component));
    if (!QDir().mkpath(output_dir))
    {
        const QString message = QStringLiteral("无法创建全球 DEM/DOM 输出目录：%1")
                                    .arg(output_dir);
        QMessageBox::warning(_parentWidget, QStringLiteral("创建全球 DEM/DOM"), message);
        emit demPipelineFinished(false, message);
        return;
    }

    const QString assets_dir =
        xjw::common::project::ProjectIO::projectAssetsDir(session.projectPath);
    if (assets_dir.isEmpty())
    {
        const QString message = QStringLiteral("无法解析当前项目的 assets 目录。");
        QMessageBox::warning(_parentWidget, QStringLiteral("创建全球 DEM/DOM"), message);
        emit demPipelineFinished(false, message);
        return;
    }
    const QString project_report_path = QDir(assets_dir).filePath(
        QStringLiteral("reports/small_body_global/%1/%2.json")
            .arg(chunk_component, run_component));
    const QString project_root = QFileInfo(assets_dir).absolutePath();

    const auto cancel_flag = std::make_shared<std::atomic_bool>(false);
    const QString background_task_id = QStringLiteral("dem-global:%1").arg(
        QUuid::createUuid().toString(QUuid::WithoutBraces));
    _demCancelFlag = cancel_flag;
    _demTaskId = background_task_id;
    _demTaskChunkId = session.chunkId;

    emit backgroundTaskProgressChanged(background_task_id, 0, 100);
    emit demPipelineProgressChanged(QStringLiteral("准备小天体全球 DEM/DOM"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progress_callback =
        [self, cancel_flag, session, background_task_id](const QString &stage, int percent)
    {
        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(
            self.data(),
            [self, cancel_flag, session, stage, percent, background_task_id]()
            {
                if (!self
                    || self->_demCancelFlag != cancel_flag
                    || !self->_owner
                    || !self->_projectData
                    || !self->_owner->isCurrentSession(session))
                {
                    return;
                }
                const int bounded_percent = std::clamp(percent, 0, 99);
                emit self->backgroundTaskProgressChanged(
                    background_task_id, bounded_percent, 100);
                emit self->demPipelineProgressChanged(stage, bounded_percent);
            },
            Qt::QueuedConnection);
    };

    xjw::SmallBodyGlobalOptions options;
    options.targetName = request.smallBodyOptions.targetName;
    options.bodyFixedFrame = request.smallBodyOptions.bodyFixedFrame;
    options.surfaceCoordinateUnit = request.smallBodyOptions.surfaceCoordinateUnit;
    options.automaticCenter = request.smallBodyOptions.automaticCenter;
    options.bodyCenter = cv::Vec3d(request.smallBodyOptions.bodyCenterX,
                                   request.smallBodyOptions.bodyCenterY,
                                   request.smallBodyOptions.bodyCenterZ);
    options.referenceRadiusM = request.smallBodyOptions.referenceRadiusM;
    options.angularResolutionDeg = request.smallBodyOptions.angularResolutionDeg;
    options.centralMeridianDeg = request.smallBodyOptions.centralMeridianDeg;
    options.maximumPixelCount = request.smallBodyOptions.maximumPixelCount;
    options.writeReportPreview = request.smallBodyOptions.writeReportPreview;
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [surface_path,
         output_dir,
         project_report_path,
         project_root,
         options,
         cancel_flag,
         progress_callback]()
        {
            return runSmallBodyGlobalProducts(
                surface_path,
                output_dir,
                project_report_path,
                project_root,
                options,
                cancel_flag.get(),
                progress_callback);
        },
        [surface_path,
         output_dir,
         project_report_path,
         project_root,
         options,
         session,
         cancel_flag,
         background_task_id](
            ProjectTerrainProductsManager *manager,
            xjw::gui::tasks::TaskOutcome<TerrainPipelineResult> outcome)
        {
            emit manager->backgroundTaskFinished(background_task_id);
            const auto rollback_generated_run = [&]()
            {
                QFile::remove(project_report_path);
                QDir(output_dir).removeRecursively();
            };
            if (manager->_demTaskId != background_task_id)
            {
                rollback_generated_run();
                return;
            }
            manager->_demTaskId.clear();
            if (manager->_demCancelFlag == cancel_flag)
            {
                manager->_demCancelFlag.reset();
                manager->_demTaskChunkId.clear();
            }

            if (!manager->_owner
                || !manager->_projectData
                || !manager->_owner->isCurrentSession(session))
            {
                rollback_generated_run();
                emit manager->demPipelineFinished(
                    false, QStringLiteral("项目已切换，全球 DEM/DOM 结果未写入当前项目。"));
                return;
            }

            if (!outcome.succeeded())
            {
                rollback_generated_run();
                const QString error = outcome.errorMessage.isEmpty()
                    ? QStringLiteral("小天体全球 DEM/DOM 后台任务失败。")
                    : outcome.errorMessage;
                QMessageBox::warning(
                    manager->_parentWidget,
                    QStringLiteral("创建全球 DEM/DOM"),
                    QStringLiteral("处理失败：%1").arg(error));
                emit manager->demPipelineFinished(false, error);
                return;
            }

            const TerrainPipelineResult terrain_run = std::move(*outcome.value);
            if (cancel_flag->load(std::memory_order_relaxed))
            {
                rollback_generated_run();
                const QString error = QStringLiteral("小天体全球 DEM/DOM 生成已取消。");
                emit manager->demPipelineFinished(false, error);
                return;
            }
            if (!terrain_run.ok)
            {
                const bool cancelled = cancel_flag->load(std::memory_order_relaxed)
                    || terrain_run.error.contains(QStringLiteral("取消"));
                const QString error = cancelled
                    ? QStringLiteral("小天体全球 DEM/DOM 生成已取消。")
                    : terrain_run.error;
                if (!cancelled)
                {
                    QMessageBox::warning(
                        manager->_parentWidget,
                        QStringLiteral("创建全球 DEM/DOM"),
                        QStringLiteral("处理失败：%1").arg(error));
                }
                rollback_generated_run();
                emit manager->demPipelineFinished(false, error);
                return;
            }

            const QJsonObject terrain_result = terrain_run.payload;
            const QString radial_dem_path =
                terrain_result.value(QStringLiteral("radial_dem_tif")).toString();
            const QString elevation_dem_path =
                terrain_result.value(QStringLiteral("elevation_dem_tif")).toString();
            const QString dom_path =
                terrain_result.value(QStringLiteral("dom_tif")).toString();
            const QString generated_report_path =
                terrain_result.value(QStringLiteral("report_json")).toString();
            const QString radial_dem_storage =
                projectStoragePath(project_root, radial_dem_path);
            const QString elevation_dem_storage =
                projectStoragePath(project_root, elevation_dem_path);
            const QString dom_storage = projectStoragePath(project_root, dom_path);
            const QString output_storage = projectStoragePath(project_root, output_dir);
            const QString report_storage =
                projectStoragePath(project_root, project_report_path);
            const QString generated_report_storage =
                projectStoragePath(project_root, generated_report_path);
            const QString preview_path =
                terrain_result.value(QStringLiteral("preview_png")).toString();
            const QString preview_storage =
                projectStoragePath(project_root, preview_path);
            const QString source_surface_storage =
                projectStoragePath(project_root, surface_path);
            const QStringList required_products{
                radial_dem_path,
                elevation_dem_path,
                dom_path,
                project_report_path
            };
            for (const QString &path : required_products)
            {
                if (path.trimmed().isEmpty() || !QFileInfo::exists(path))
                {
                    rollback_generated_run();
                    const QString error = QStringLiteral(
                        "全球 DEM/DOM 管线返回成功，但必要产物不存在：%1").arg(path);
                    QMessageBox::warning(
                        manager->_parentWidget,
                        QStringLiteral("创建全球 DEM/DOM"),
                        error);
                    emit manager->demPipelineFinished(false, error);
                    return;
                }
            }

            QString created_at =
                terrain_result.value(QStringLiteral("created_at")).toString();
            if (created_at.isEmpty())
            {
                created_at = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
            }
            const QJsonObject frame =
                terrain_result.value(QStringLiteral("frame")).toObject();
            const QJsonObject grid =
                terrain_result.value(QStringLiteral("grid")).toObject();
            const QJsonObject metrics =
                terrain_result.value(QStringLiteral("metrics")).toObject();
            const QJsonValue solid_angle_coverage = metrics.value(
                QStringLiteral("solid_angle_weighted_coverage_ratio"));
            const double angular_resolution =
                grid.value(QStringLiteral("angular_resolution_deg"))
                    .toDouble(options.angularResolutionDeg);

            const auto decorate_dem_record =
                [&](QJsonObject *record,
                    const QString &resultType,
                    const QString &verticalReference,
                    const QString &demReference)
            {
                (*record)[QStringLiteral("result_type")] = resultType;
                (*record)[QStringLiteral("terrain_mode")] =
                    QStringLiteral("small_body_global");
                (*record)[QStringLiteral("source_surface")] = source_surface_storage;
                (*record)[QStringLiteral("source_surface_type")] =
                    QStringLiteral("triangle_mesh");
                (*record)[QStringLiteral("target_name")] =
                    frame.value(QStringLiteral("target_name"));
                (*record)[QStringLiteral("body_fixed_frame")] =
                    frame.value(QStringLiteral("body_fixed_frame"));
                (*record)[QStringLiteral("frame_status")] =
                    frame.value(QStringLiteral("frame_status"));
                (*record)[QStringLiteral("body_center_xyz_m")] =
                    frame.value(QStringLiteral("center_xyz_m"));
                (*record)[QStringLiteral("reference_radius_m")] =
                    frame.value(QStringLiteral("reference_radius_m"));
                (*record)[QStringLiteral("central_meridian_deg")] =
                    frame.value(QStringLiteral("central_meridian_deg"));
                (*record)[QStringLiteral("angular_resolution_deg")] =
                    angular_resolution;
                (*record)[QStringLiteral("resolution_unit")] = QStringLiteral("degree");
                (*record)[QStringLiteral("vertical_reference")] = verticalReference;
                (*record)[QStringLiteral("dem_reference")] = demReference;
                (*record)[QStringLiteral("coverage_ratio")] =
                    metrics.value(QStringLiteral("coverage_ratio"));
                (*record)[QStringLiteral("solid_angle_weighted_coverage_ratio")] =
                    solid_angle_coverage;
                (*record)[QStringLiteral("report_json")] = report_storage;
                (*record)[QStringLiteral("generated_report_json")] =
                    generated_report_storage;
                (*record)[QStringLiteral("preview_png")] = preview_storage;
            };

            QJsonObject radial_record = makeDemResultRecord(
                created_at,
                output_storage,
                QString(),
                radial_dem_storage,
                QStringLiteral("float32"),
                angular_resolution,
                QString(),
                QStringList());
            decorate_dem_record(
                &radial_record,
                QStringLiteral("small_body_global_radial_dem"),
                QStringLiteral("radial_distance_from_body_center"),
                QStringLiteral("body_center"));

            QJsonObject elevation_record = makeDemResultRecord(
                created_at,
                output_storage,
                QString(),
                elevation_dem_storage,
                QStringLiteral("float32"),
                angular_resolution,
                QString(),
                QStringList());
            decorate_dem_record(
                &elevation_record,
                QStringLiteral("small_body_global_elevation_dem"),
                QStringLiteral("elevation_above_reference_radius"),
                QStringLiteral("reference_radius"));

            QJsonObject dom_payload;
            dom_payload[QStringLiteral("terrain_mode")] =
                QStringLiteral("small_body_global");
            dom_payload[QStringLiteral("result_type")] =
                QStringLiteral("small_body_global_dom");
            dom_payload[QStringLiteral("source_surface")] = source_surface_storage;
            dom_payload[QStringLiteral("source_surface_type")] =
                QStringLiteral("triangle_mesh");
            dom_payload[QStringLiteral("target_name")] =
                frame.value(QStringLiteral("target_name"));
            dom_payload[QStringLiteral("body_fixed_frame")] =
                frame.value(QStringLiteral("body_fixed_frame"));
            dom_payload[QStringLiteral("frame_status")] =
                frame.value(QStringLiteral("frame_status"));
            dom_payload[QStringLiteral("central_meridian_deg")] =
                frame.value(QStringLiteral("central_meridian_deg"));
            dom_payload[QStringLiteral("angular_resolution_deg")] =
                angular_resolution;
            dom_payload[QStringLiteral("resolution_unit")] = QStringLiteral("degree");
            dom_payload[QStringLiteral("coverage_ratio")] =
                metrics.value(QStringLiteral("coverage_ratio"));
            dom_payload[QStringLiteral("solid_angle_weighted_coverage_ratio")] =
                solid_angle_coverage;
            dom_payload[QStringLiteral("report_json")] = report_storage;
            dom_payload[QStringLiteral("preview_png")] = preview_storage;
            QJsonObject dom_record = makeOrthoResultRecord(
                created_at,
                radial_dem_storage,
                dom_storage,
                0,
                QStringList(),
                true,
                angular_resolution,
                dom_payload);

            QJsonObject report_record = terrain_result;
            report_record[QStringLiteral("result_type")] =
                QStringLiteral("small_body_global_terrain_report");
            report_record[QStringLiteral("path")] = report_storage;
            report_record[QStringLiteral("json_path")] = report_storage;
            report_record[QStringLiteral("source_report_path")] =
                generated_report_storage;
            report_record[QStringLiteral("preview_path")] = preview_storage;
            report_record[QStringLiteral("preview_png")] = preview_storage;
            report_record[QStringLiteral("radial_dem_tif")] = radial_dem_storage;
            report_record[QStringLiteral("elevation_dem_tif")] = elevation_dem_storage;
            report_record[QStringLiteral("dom_tif")] = dom_storage;

            const bool records_saved =
                manager->_projectData->upsertResultRecordByPath(
                    QStringLiteral("dem_results"), QStringLiteral("dem_tif"), radial_record, true)
                && manager->_projectData->upsertResultRecordByPath(
                    QStringLiteral("dem_results"), QStringLiteral("dem_tif"), elevation_record, true)
                && manager->_projectData->upsertResultRecordByPath(
                    QStringLiteral("ortho_results"), QStringLiteral("output_path"), dom_record, true)
                && manager->_projectData->upsertResultRecordByPath(
                    QStringLiteral("report_results"), QStringLiteral("path"), report_record, true);
            if (!records_saved)
            {
                const QString error = QStringLiteral(
                    "全球 DEM/DOM 已生成，但写入当前 Chunk 的项目成果记录失败。");
                QMessageBox::warning(
                    manager->_parentWidget, QStringLiteral("创建全球 DEM/DOM"), error);
                emit manager->demPipelineFinished(false, error);
                return;
            }
            manager->_owner->refreshReconstructionQualityReport();

            emit manager->demPipelineProgressChanged(QStringLiteral("完成"), 100);
            emit manager->demPipelineFinished(
                true, QStringLiteral("小天体全球 DEM/DOM 生成完成"));
            QMessageBox::information(
                manager->_parentWidget,
                QStringLiteral("创建全球 DEM/DOM"),
                QStringLiteral(
                    "处理完成。\n径向 DEM: %1\n高程 DEM: %2\nDOM: %3\n报告: %4\n固体角加权覆盖率: %5%")
                    .arg(radial_dem_path,
                         elevation_dem_path,
                         dom_path,
                         project_report_path)
                    .arg(solid_angle_coverage.toDouble() * 100.0,
                         0,
                         'f',
                         2));
        });
}

void ProjectTerrainProductsManager::startMapProjectAsync(
    const xjw::gui::project::OrthoGenerationRequest &request)
{
    if (!xjw::gui::project::requireOpenProject(_projectData, _parentWidget))
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
    if (request.isRpc())
    {
        startRpcDomAsync(request);
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
                if (isSmallBodyGlobalDemRecord(record))
                {
                    continue;
                }
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
            if (pathsReferToSameLocation(candidate, resolvedDem))
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
    if (!pointCloudMode && isSmallBodyGlobalDemRecord(matchedDemRecord))
    {
        const QString message = QStringLiteral(
            "小天体全球径向 DEM 使用经纬度网格，不能作为局部平面正射反投影的 DEM。"
            "请改用局部平面 DEM，全球 DOM 已由全球地形管线直接生成。");
        emit orthoPipelineFinished(false, message, QJsonObject());
        return;
    }
    if (!pointCloudMode && isSmallBodyGlobalDemFile(resolvedDem))
    {
        const QString message = QStringLiteral(
            "所选 GeoTIFF 的元数据表明它是体固连经纬网径向/高程 DEM，"
            "不能进入局部平面正射反投影。请使用局部平面 DEM；全球 DOM 已由全球地形管线生成。");
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
    if (pathsReferToSameLocation(normalizedOutput, resolvedDem))
    {
        emit orthoPipelineFinished(
            false,
            (pointCloudMode
                 ? QStringLiteral("正射输出路径不能覆盖输入点云：%1")
                 : QStringLiteral("正射输出路径不能覆盖输入 DEM：%1")).arg(out),
            QJsonObject());
        return;
    }
    if (pathsReferToSameLocation(normalizedOutput, projectPath))
    {
        emit orthoPipelineFinished(
            false,
            QStringLiteral("正射输出路径不能覆盖当前项目文件：%1").arg(projectPath),
            QJsonObject());
        return;
    }
    for (const QString &imagePath : sourceImages)
    {
        if (pathsReferToSameLocation(normalizedOutput, imagePath))
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
            if (pathsReferToSameLocation(normalizedOutput, maskPath))
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
        if (isSmallBodyGlobalDemRecord(record))
        {
            continue;
        }
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
    const QString background_task_id = QStringLiteral("ortho:%1").arg(
        QUuid::createUuid().toString(QUuid::WithoutBraces));
    _orthoCancelFlag = cancelFlag;
    _orthoTaskChunkId = session.chunkId;

    emit backgroundTaskProgressChanged(background_task_id, 0, 100);
    emit orthoPipelineStarted();
    emit orthoPipelineProgressChanged(QStringLiteral("准备正射影像生成"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progressCallback =
        [self, cancelFlag, session, background_task_id](
            const QString &stage, int percent)
    {
        if (!self)
        {
            return;
        }

        QMetaObject::invokeMethod(
            self.data(),
            [self, cancelFlag, session, stage, percent, background_task_id]()
            {
                if (!self ||
                    self->_orthoCancelFlag != cancelFlag ||
                    !self->_owner ||
                    !self->_projectData ||
                    !self->_owner->isCurrentSession(session))
                {
                    return;
                }
                emit self->backgroundTaskProgressChanged(
                    background_task_id, std::clamp(percent, 0, 99), 100);
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
                      progressCallback]() -> xjw::core::project::TerrainPipelineResult
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
                          cancelFlag,
                          background_task_id](
                              ProjectTerrainProductsManager *manager,
                              xjw::core::project::TerrainPipelineResult orthoRun)
    {
        emit manager->backgroundTaskFinished(background_task_id);
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
                ? QStringLiteral("正射影像生成遇到未知错误，请检查控制台。")
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
        [orthoFinished = std::move(orthoFinished), cancelFlag, background_task_id](
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
                emit manager->backgroundTaskFinished(background_task_id);
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

void ProjectTerrainProductsManager::cancelDemGeneration()
{
    if (_demCancelFlag)
    {
        _demCancelFlag->store(true, std::memory_order_relaxed);
    }
}
