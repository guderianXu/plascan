#include "ProjectTerrainProductsManager.h"

#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "project/ProjectIO.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowOperations.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "DemDomIO.h"
#include "TerrainPipeline.h"
#include "GlobalTerrainReportRenderer.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPointer>
#include <QRegularExpression>
#include <QSaveFile>
#include <QUuid>

#include <algorithm>
#include <cmath>

using xjw::gui::project::makeDemResultRecord;
using xjw::gui::project::makeOrthoResultRecord;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::core::project::runDemProducts;
using xjw::core::project::runOrthoProduct;
using xjw::core::project::TerrainPipelineResult;

namespace
{

QString smallBodyGlobalStageText(xjw::SmallBodyGlobalStage stage)
{
    switch (stage)
    {
    case xjw::SmallBodyGlobalStage::LoadSurface:
        return QStringLiteral("读取体固连表面模型");
    case xjw::SmallBodyGlobalStage::BuildSpatialIndex:
        return QStringLiteral("建立三角网 BVH");
    case xjw::SmallBodyGlobalStage::RasterizeGlobalProducts:
        return QStringLiteral("生成全球径向 DEM/DOM");
    case xjw::SmallBodyGlobalStage::WriteProducts:
        return QStringLiteral("写出全球 GeoTIFF");
    case xjw::SmallBodyGlobalStage::Completed:
        return QStringLiteral("全球 DEM/DOM 与报告完成");
    }
    return QStringLiteral("处理小天体全球 DEM/DOM");
}

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
    const xjw::SmallBodyProgressCallback &progressCallback)
{
    TerrainPipelineResult result;
    const auto rollback_run = [&]()
    {
        QFile::remove(projectReportPath);
        QDir(outputDir).removeRecursively();
    };
    result.ok = xjw::TerrainPipeline::generateSmallBodyGlobalProducts(surfacePath,
                                                                      outputDir,
                                                                      options,
                                                                      &result.payload,
                                                                      &result.error,
                                                                      cancelFlag,
                                                                      progressCallback,
                                                                      xjw::GlobalTerrainReportRenderer::writePreview);
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


ProjectTerrainProductsManager::ProjectTerrainProductsManager(
    xjw::gui::project::ProjectSession *session,
    ProjectUiMessageAdapter *messages,
    QObject *parent)
    : QObject(parent), _session(session), _messages(messages)
{
}

ProjectTerrainProductsManager::~ProjectTerrainProductsManager()
{
    waitForActiveTask();
}

bool ProjectTerrainProductsManager::demContextMatches(
    const xjw::gui::project::ProjectTaskContext &taskContext,
    bool requireCurrent) const
{
    return taskContext.cancelFlag && _demContext.taskId == taskContext.taskId &&
           _demContext.cancelFlag == taskContext.cancelFlag &&
           (!requireCurrent || (_session && _session->isCurrent(taskContext.session)));
}

bool ProjectTerrainProductsManager::orthoContextMatches(
    const xjw::gui::project::ProjectTaskContext &taskContext,
    bool requireCurrent) const
{
    return taskContext.cancelFlag && _orthoContext.taskId == taskContext.taskId &&
           _orthoContext.cancelFlag == taskContext.cancelFlag &&
           (!requireCurrent || (_session && _session->isCurrent(taskContext.session)));
}

void ProjectTerrainProductsManager::clearDemContextIfMatches(
    const xjw::gui::project::ProjectTaskContext &taskContext)
{
    if (demContextMatches(taskContext, false))
    {
        _demContext = {};
    }
}

void ProjectTerrainProductsManager::clearOrthoContextIfMatches(
    const xjw::gui::project::ProjectTaskContext &taskContext)
{
    if (orthoContextMatches(taskContext, false))
    {
        _orthoContext = {};
    }
}

void ProjectTerrainProductsManager::startDemFromPointCloudAsync(
    const xjw::gui::project::DemGenerationRequest &request,
    const xjw::gui::project::ProjectTaskContext &taskContext)
{
    _demContext = taskContext;
    if (!_session || !_session->hasProject() || !_session->isCurrent(taskContext.session) ||
        !taskContext.cancelFlag || taskContext.cancelFlag->load(std::memory_order_relaxed))
    {
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("请先打开项目"));
        }
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, QStringLiteral("请先打开项目。"));
        return;
    }

    QString requestError;
    if (!request.validate(&requestError))
    {
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("创建 DEM"), requestError);
        }
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, requestError);
        return;
    }

    if (request.isSmallBodyGlobal())
    {
        startSmallBodyGlobalAsync(request, taskContext);
        return;
    }
    if (request.isImageStereo())
    {
        startRpcStereoDemAsync(request, taskContext);
        return;
    }

    const QString pointCloudPath = request.sourcePointCloudPath.trimmed();
    if (!QFileInfo::exists(pointCloudPath))
    {
        if (_messages)
        {
            _messages->warning(nullptr,
                               QStringLiteral("创建 DEM"),
                               QStringLiteral("指定的点云文件不存在：\n%1").arg(pointCloudPath));
        }
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, QStringLiteral("点云文件不存在"));
        return;
    }

    const QString background_task_id = taskContext.taskId;
    QString outDir = resolveProjectOutputDir(taskContext.session.projectPath,
                                             request.outputDirectory.trimmed(),
                                             QStringLiteral("assets/dem/relative_dem"));

    const double demResolution = request.resolution;
    const QString demType = request.dataType;
    emit backgroundTaskProgressChanged(background_task_id, 5, 100);
    emit demPipelineProgressChanged(QStringLiteral("DEM 生成"), 5);

    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [pointCloudPath, outDir, demResolution, demType]()
        {
            return runDemProducts(pointCloudPath,
                                  outDir,
                                  demResolution,
                                  demType,
                                  false);
        },
        [pointCloudPath, outDir, demResolution, demType, taskContext, background_task_id](
            ProjectTerrainProductsManager *self,
            xjw::gui::tasks::TaskOutcome<TerrainPipelineResult> outcome)
        {
            if (!self->demContextMatches(taskContext, false))
            {
                return;
            }
            if (!self->_session || !self->_session->isCurrent(taskContext.session))
            {
                self->clearDemContextIfMatches(taskContext);
                return;
            }
            emit self->backgroundTaskFinished(background_task_id);
            if (taskContext.cancelFlag->load(std::memory_order_relaxed))
            {
                self->clearDemContextIfMatches(taskContext);
                emit self->demPipelineFinished(false, QStringLiteral("DEM 生成已取消"));
                return;
            }

            if (!outcome.succeeded())
            {
                const QString error = outcome.errorMessage.isEmpty()
                    ? QStringLiteral("DEM 后台任务失败")
                    : outcome.errorMessage;
                if (self->_messages)
                {
                    self->_messages->warning(nullptr,
                                             QStringLiteral("创建相对 DEM"),
                                             QStringLiteral("处理失败：%1").arg(error));
                }
                self->clearDemContextIfMatches(taskContext);
                emit self->demPipelineFinished(false, error);
                return;
            }

            const TerrainPipelineResult terrainRun = std::move(*outcome.value);
            if (!terrainRun.ok)
            {
                if (self->_messages)
                {
                    self->_messages->warning(nullptr,
                                             QStringLiteral("创建相对 DEM"),
                                             QStringLiteral("处理失败：%1").arg(terrainRun.error));
                }
                self->clearDemContextIfMatches(taskContext);
                emit self->demPipelineFinished(false, terrainRun.error);
                return;
            }

            const QJsonObject terrainResult = terrainRun.payload;
            QJsonObject demResult = makeDemResultRecord(terrainResult.value(QStringLiteral("created_at")).toString(),
                                                        outDir,
                                                        QString(),
                                                        terrainResult.value(QStringLiteral("dem_path")).toString(),
                                                        demType,
                                                        demResolution,
                                                        QString(),
                                                        QStringList());
            demResult[QStringLiteral("source_point_cloud")] = pointCloudPath;
            demResult[QStringLiteral("dem_reference")] = QStringLiteral("relative");
            demResult[QStringLiteral("preview_path")] = terrainResult.value(QStringLiteral("preview_path")).toString();
            demResult[QStringLiteral("relative_z_offset")] =
                terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
            QString persistence_error;
            if (!self->_session->upsertResultRecordByPath(taskContext.session,
                                                          QStringLiteral("dem_results"),
                                                          QStringLiteral("dem_path"),
                                                          demResult,
                                                          true,
                                                          &persistence_error))
            {
                self->clearDemContextIfMatches(taskContext);
                emit self->demPipelineFinished(false, persistence_error);
                return;
            }
            if (!self->demContextMatches(taskContext) ||
                taskContext.cancelFlag->load(std::memory_order_relaxed))
            {
                self->clearDemContextIfMatches(taskContext);
                return;
            }

            emit self->demPipelineProgressChanged(QStringLiteral("完成"), 100);
            self->clearDemContextIfMatches(taskContext);
            emit self->demPipelineFinished(true, QStringLiteral("DEM 生成完成"));
            if (self->_messages)
            {
                self->_messages->information(
                    nullptr,
                    QStringLiteral("创建相对 DEM"),
                    QStringLiteral("处理完成。\nDEM: %1\n预览图: %2\n参考点云: %3\n高程基准偏移: %4")
                        .arg(terrainResult.value(QStringLiteral("dem_path")).toString())
                        .arg(terrainResult.value(QStringLiteral("preview_path")).toString())
                        .arg(pointCloudPath)
                        .arg(terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0), 0, 'f', 6));
            }
        }));
}

void ProjectTerrainProductsManager::startSmallBodyGlobalAsync(
    const xjw::gui::project::DemGenerationRequest &request,
    const xjw::gui::project::ProjectTaskContext &taskContext)
{
    const QString surface_path =
        xjw::common::project::ProjectIO::resolveProjectResourcePath(
            taskContext.session.projectPath, request.sourceSurfacePath.trimmed());
    const QFileInfo surface_info(surface_path);
    if (surface_path.isEmpty() || !surface_info.exists() || !surface_info.isFile())
    {
        const QString message = QStringLiteral("指定的体固连三角网格不存在：\n%1")
                                    .arg(surface_path.isEmpty()
                                             ? request.sourceSurfacePath
                                             : surface_path);
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("创建全球 DEM/DOM"), message);
        }
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, message);
        return;
    }

    const QString suffix = surface_info.suffix().toLower();
    if (suffix != QLatin1String("ply") && suffix != QLatin1String("obj"))
    {
        const QString message = QStringLiteral("全球 DEM/DOM 输入必须是 PLY 或 OBJ 三角网格：\n%1")
                                    .arg(surface_path);
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("创建全球 DEM/DOM"), message);
        }
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, message);
        return;
    }

    const QString output_root = resolveProjectOutputDir(
        taskContext.session.projectPath,
        request.outputDirectory.trimmed(),
        QStringLiteral("assets/dem/small_body_global"));
    if (output_root.isEmpty())
    {
        const QString message = QStringLiteral("无法解析全球 DEM/DOM 输出根目录。");
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("创建全球 DEM/DOM"), message);
        }
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, message);
        return;
    }
    const QString chunk_component = safeStorageComponent(taskContext.session.chunkId);
    const QString run_component = QStringLiteral("%1_%2")
        .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz")),
             QUuid::createUuid().toString(QUuid::WithoutBraces));
    const QString output_dir = QDir(output_root).filePath(
        QStringLiteral("%1/%2").arg(chunk_component, run_component));
    if (!QDir().mkpath(output_dir))
    {
        const QString message = QStringLiteral("无法创建全球 DEM/DOM 输出目录：%1")
                                    .arg(output_dir);
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("创建全球 DEM/DOM"), message);
        }
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, message);
        return;
    }

    const QString assets_dir =
        xjw::common::project::ProjectIO::projectAssetsDir(taskContext.session.projectPath);
    if (assets_dir.isEmpty())
    {
        const QString message = QStringLiteral("无法解析当前项目的 assets 目录。");
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("创建全球 DEM/DOM"), message);
        }
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, message);
        return;
    }
    const QString project_report_path = QDir(assets_dir).filePath(
        QStringLiteral("reports/small_body_global/%1/%2.json")
            .arg(chunk_component, run_component));
    const QString project_root = QFileInfo(assets_dir).absolutePath();

    const auto cancel_flag = taskContext.cancelFlag;
    const QString background_task_id = taskContext.taskId;

    emit backgroundTaskProgressChanged(background_task_id, 0, 100);
    emit demPipelineProgressChanged(QStringLiteral("准备小天体全球 DEM/DOM"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progress_callback =
        [self, cancel_flag, taskContext, background_task_id](
            const xjw::SmallBodyGlobalProgress &progress)
    {
        if (!self)
        {
            return;
        }
        const QString stage = smallBodyGlobalStageText(progress.stage);
        const int percent = progress.overallPercent;
        QMetaObject::invokeMethod(
            self.data(),
            [self, cancel_flag, taskContext, stage, percent, background_task_id]()
            {
                if (!self || !self->demContextMatches(taskContext) ||
                    cancel_flag->load(std::memory_order_relaxed))
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
    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
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
         taskContext,
         cancel_flag,
         background_task_id](
            ProjectTerrainProductsManager *manager,
            xjw::gui::tasks::TaskOutcome<TerrainPipelineResult> outcome)
        {
            const auto rollback_generated_run = [&]()
            {
                QFile::remove(project_report_path);
                QDir(output_dir).removeRecursively();
            };
            if (!manager->demContextMatches(taskContext, false))
            {
                rollback_generated_run();
                return;
            }
            if (!manager->_session || !manager->_session->isCurrent(taskContext.session))
            {
                manager->clearDemContextIfMatches(taskContext);
                rollback_generated_run();
                return;
            }
            emit manager->backgroundTaskFinished(background_task_id);

            if (!outcome.succeeded())
            {
                rollback_generated_run();
                const QString error = outcome.errorMessage.isEmpty()
                    ? QStringLiteral("小天体全球 DEM/DOM 后台任务失败。")
                    : outcome.errorMessage;
                if (manager->_messages)
                {
                    manager->_messages->warning(nullptr,
                                                QStringLiteral("创建全球 DEM/DOM"),
                                                QStringLiteral("处理失败：%1").arg(error));
                }
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(false, error);
                return;
            }

            const TerrainPipelineResult terrain_run = std::move(*outcome.value);
            if (cancel_flag->load(std::memory_order_relaxed))
            {
                rollback_generated_run();
                const QString error = QStringLiteral("小天体全球 DEM/DOM 生成已取消。");
                manager->clearDemContextIfMatches(taskContext);
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
                    if (manager->_messages)
                    {
                        manager->_messages->warning(nullptr,
                                                    QStringLiteral("创建全球 DEM/DOM"),
                                                    QStringLiteral("处理失败：%1").arg(error));
                    }
                }
                rollback_generated_run();
                manager->clearDemContextIfMatches(taskContext);
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
                    if (manager->_messages)
                    {
                        manager->_messages->warning(nullptr, QStringLiteral("创建全球 DEM/DOM"), error);
                    }
                    manager->clearDemContextIfMatches(taskContext);
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

            if (!manager->demContextMatches(taskContext) ||
                cancel_flag->load(std::memory_order_relaxed))
            {
                rollback_generated_run();
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(
                    false, QStringLiteral("小天体全球 DEM/DOM 生成已取消。"));
                return;
            }
            const QVector<xjw::gui::project::ProjectResultRecordUpsert> records{
                {QStringLiteral("dem_results"), QStringLiteral("dem_path"), radial_record, true},
                {QStringLiteral("dem_results"), QStringLiteral("dem_path"), elevation_record, true},
                {QStringLiteral("ortho_results"), QStringLiteral("output_path"), dom_record, true},
                {QStringLiteral("report_results"), QStringLiteral("path"), report_record, true},
            };
            QString persistence_error;
            const bool records_saved = manager->_session->upsertResultRecordsByPath(
                taskContext.session, records, &persistence_error);
            if (!records_saved)
            {
                const QString error = QStringLiteral(
                    "全球 DEM/DOM 已生成，但写入当前 Chunk 的项目成果记录失败。%1")
                                          .arg(persistence_error.isEmpty()
                                                   ? QString()
                                                   : QStringLiteral("\n%1").arg(persistence_error));
                if (manager->_messages)
                {
                    manager->_messages->warning(nullptr, QStringLiteral("创建全球 DEM/DOM"), error);
                }
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(false, error);
                return;
            }
            if (!manager->demContextMatches(taskContext) ||
                cancel_flag->load(std::memory_order_relaxed))
            {
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(
                    false, QStringLiteral("小天体全球 DEM/DOM 生成已取消。"));
                return;
            }

            emit manager->demPipelineProgressChanged(QStringLiteral("完成"), 100);
            manager->clearDemContextIfMatches(taskContext);
            emit manager->demPipelineFinished(
                true, QStringLiteral("小天体全球 DEM/DOM 生成完成"));
            if (manager->_messages)
            {
                manager->_messages->information(
                    nullptr,
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
            }
        }));
}

void ProjectTerrainProductsManager::startMapProjectAsync(
    const xjw::gui::project::OrthoGenerationRequest &request,
    const xjw::gui::project::ProjectTaskContext &taskContext)
{
    _orthoContext = taskContext;
    if (!_session || !_session->hasProject() || !_session->isCurrent(taskContext.session) ||
        !taskContext.cancelFlag || taskContext.cancelFlag->load(std::memory_order_relaxed))
    {
        if (_messages)
        {
            _messages->warning(nullptr, QStringLiteral("提示"), QStringLiteral("请先打开项目"));
        }
        clearOrthoContextIfMatches(taskContext);
        emit orthoPipelineFinished(false, QStringLiteral("请先打开项目"), QJsonObject());
        return;
    }

    const QString projectPath = taskContext.session.projectPath;
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
        startRpcDomAsync(request, taskContext);
        return;
    }

    QJsonObject meta = _session->metadata();
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
                const QString candidate = resolveProjectPath(record.value(QStringLiteral("dem_path")).toString());
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
            const QString candidate = resolveProjectPath(record.value(QStringLiteral("dem_path")).toString());
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
        sourceImages = _session->allImages();
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
        const QString demPath = resolveProjectPath(record.value(QStringLiteral("dem_path")).toString());
        if (!demPath.isEmpty())
        {
            record[QStringLiteral("dem_path")] = demPath;
        }
        runtimeDemResults.append(record);
    }
    runtimeMeta[QStringLiteral("dem_results")] = runtimeDemResults;

    const auto cancelFlag = taskContext.cancelFlag;
    const QString background_task_id = taskContext.taskId;

    emit backgroundTaskProgressChanged(background_task_id, 0, 100);
    emit orthoPipelineStarted();
    emit orthoPipelineProgressChanged(QStringLiteral("准备正射影像生成"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progressCallback =
        [self, cancelFlag, taskContext, background_task_id](
            const QString &stage, int percent)
    {
        if (!self)
        {
            return;
        }

        QMetaObject::invokeMethod(
            self.data(),
            [self, cancelFlag, taskContext, stage, percent, background_task_id]()
            {
                if (!self || !self->orthoContextMatches(taskContext) ||
                    cancelFlag->load(std::memory_order_relaxed))
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
                          taskContext,
                          cancelFlag,
                          background_task_id](
                              ProjectTerrainProductsManager *manager,
                              xjw::core::project::TerrainPipelineResult orthoRun)
    {
        if (!manager->orthoContextMatches(taskContext, false))
        {
            return;
        }
        if (!manager->_session || !manager->_session->isCurrent(taskContext.session))
        {
            manager->clearOrthoContextIfMatches(taskContext);
            return;
        }
        emit manager->backgroundTaskFinished(background_task_id);

        const bool cancelled =
            cancelFlag->load(std::memory_order_relaxed) ||
            (!orthoRun.ok && orthoRun.error.contains(QStringLiteral("已取消")));
        if (cancelled)
        {
            manager->clearOrthoContextIfMatches(taskContext);
            emit manager->orthoPipelineFinished(
                false,
                QStringLiteral("正射影像生成已取消"),
                orthoRun.payload);
            return;
        }

        if (!orthoRun.ok)
        {
            const QString error = orthoRun.error.trimmed().isEmpty()
                ? QStringLiteral("正射影像生成遇到未知错误，请检查控制台。")
                : orthoRun.error;
            manager->clearOrthoContextIfMatches(taskContext);
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
        QString persistence_error;
        if (!manager->_session->upsertResultRecordByPath(taskContext.session,
                                                         QStringLiteral("ortho_results"),
                                                         QStringLiteral("output_path"),
                                                         record,
                                                         true,
                                                         &persistence_error))
        {
            manager->clearOrthoContextIfMatches(taskContext);
            emit manager->orthoPipelineFinished(
                false,
                QStringLiteral("正射影像已写出，但项目结果记录保存失败：%1%2")
                    .arg(record.value(QStringLiteral("output_path")).toString(),
                         persistence_error.isEmpty() ? QString() : QStringLiteral("\n%1").arg(persistence_error)),
                record);
            return;
        }
        if (!manager->orthoContextMatches(taskContext) ||
            cancelFlag->load(std::memory_order_relaxed))
        {
            manager->clearOrthoContextIfMatches(taskContext);
            emit manager->orthoPipelineFinished(false, QStringLiteral("正射影像生成已取消"), record);
            return;
        }

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
        manager->clearOrthoContextIfMatches(taskContext);
        emit manager->orthoPipelineFinished(
            true,
            completionMessage,
            record);
    };

    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        std::move(orthoWork),
        [orthoFinished = std::move(orthoFinished), taskContext, background_task_id](
            ProjectTerrainProductsManager *manager,
            xjw::gui::tasks::TaskOutcome<TerrainPipelineResult> outcome) mutable
        {
            if (!outcome.succeeded())
            {
                if (!manager->orthoContextMatches(taskContext, false))
                {
                    return;
                }
                if (!manager->_session || !manager->_session->isCurrent(taskContext.session))
                {
                    manager->clearOrthoContextIfMatches(taskContext);
                    return;
                }
                const QString error = outcome.errorMessage.isEmpty()
                    ? QStringLiteral("正射影像后台任务失败")
                    : outcome.errorMessage;
                emit manager->backgroundTaskFinished(background_task_id);
                manager->clearOrthoContextIfMatches(taskContext);
                emit manager->orthoPipelineFinished(false, error, QJsonObject());
                return;
            }
            orthoFinished(manager, std::move(*outcome.value));
        }));
}

void ProjectTerrainProductsManager::cancelMapProject(
    const xjw::gui::project::ProjectTaskContext &taskContext)
{
    if (orthoContextMatches(taskContext, false) && taskContext.cancelFlag)
    {
        taskContext.cancelFlag->store(true, std::memory_order_relaxed);
    }
}

void ProjectTerrainProductsManager::cancelDemGeneration(
    const xjw::gui::project::ProjectTaskContext &taskContext)
{
    if (demContextMatches(taskContext, false) && taskContext.cancelFlag)
    {
        taskContext.cancelFlag->store(true, std::memory_order_relaxed);
    }
}

void ProjectTerrainProductsManager::waitForActiveTask()
{
    if (_demContext.cancelFlag)
    {
        _demContext.cancelFlag->store(true, std::memory_order_relaxed);
    }
    if (_orthoContext.cancelFlag)
    {
        _orthoContext.cancelFlag->store(true, std::memory_order_relaxed);
    }
    for (QFuture<void> &future : _futures)
    {
        if (future.isRunning())
        {
            future.waitForFinished();
        }
    }
    _futures.clear();
    _demContext = {};
    _orthoContext = {};
}

bool ProjectTerrainProductsManager::hasPendingWork() const noexcept
{
    return std::any_of(_futures.cbegin(),
                       _futures.cend(),
                       [](const QFuture<void>& future) { return future.isValid() && !future.isFinished(); });
}

void ProjectTerrainProductsManager::pruneFinishedFutures()
{
    _futures.erase(std::remove_if(_futures.begin(),
                                  _futures.end(),
                                  [](const QFuture<void>& future) { return future.isFinished(); }),
                   _futures.end());
}

void ProjectTerrainProductsManager::trackFuture(QFuture<void> future)
{
    pruneFinishedFutures();
    if (future.isValid())
    {
        _futures.push_back(std::move(future));
    }
}

void ProjectTerrainProductsManager::trackFutureForTesting(QFuture<void> future)
{
    trackFuture(std::move(future));
}
