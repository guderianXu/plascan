#include "ProjectTerrainProductsManager.h"

#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "GuiTaskRunner.h"
#include "project/ProjectIO.h"

#include "RpcDomGenerator.h"
#include "RpcStereoDemGenerator.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QPointer>
#include <QUuid>

#include <algorithm>

namespace
{

    struct RpcProductRun
    {
        bool ok = false;
        QJsonObject payload;
        QString error;
    };

    QString projectStoragePath(const QString& projectRoot, const QString& path)
    {
        if (projectRoot.isEmpty() || path.trimmed().isEmpty() || QFileInfo(path).isRelative())
        {
            return QDir::fromNativeSeparators(path);
        }
        const QString relative = QDir::fromNativeSeparators(QDir(projectRoot).relativeFilePath(path));
        if (relative == QLatin1String("..") || relative.startsWith(QLatin1String("../")))
        {
            return QDir::fromNativeSeparators(path);
        }
        return relative;
    }

    QString uniqueRunDirectory(const QString& root, const QString& chunkId)
    {
        QString chunk = chunkId.trimmed();
        chunk.replace(QLatin1Char('/'), QLatin1Char('_'));
        chunk.replace(QLatin1Char('\\'), QLatin1Char('_'));
        if (chunk.isEmpty())
        {
            chunk = QStringLiteral("default_chunk");
        }
        const QString run =
            QStringLiteral("%1_%2").arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz")),
                                        QUuid::createUuid().toString(QUuid::WithoutBraces));
        return QDir(root).filePath(QStringLiteral("%1/%2").arg(chunk, run));
    }

    bool samePath(const QString& left, const QString& right)
    {
#if defined(Q_OS_WIN)
        constexpr Qt::CaseSensitivity case_sensitivity = Qt::CaseInsensitive;
#else
        constexpr Qt::CaseSensitivity case_sensitivity = Qt::CaseSensitive;
#endif
        return QFileInfo(left).absoluteFilePath().compare(
                   QFileInfo(right).absoluteFilePath(), case_sensitivity) == 0;
    }

} // namespace

void ProjectTerrainProductsManager::startRpcStereoDemAsync(
    const xjw::gui::project::DemGenerationRequest& request,
    const xjw::gui::project::ProjectTaskContext& taskContext)
{
    const QString project_root =
        xjw::common::project::ProjectIO::projectRootFromPlascan(taskContext.session.projectPath);
    const auto resolve_path = [&taskContext](const QString& path)
    {
        return xjw::common::project::ProjectIO::resolveProjectResourcePath(
            taskContext.session.projectPath, path.trimmed());
    };
    const QStringList selected_images = request.imageStereoOptions.sourceImages;
    if (selected_images.size() != 2)
    {
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(
            false,
            QStringLiteral("当前摄影测量内核尚未完成多影像联合平差与 DEM 融合；为避免静默忽略影像，"
                           "本次任务未启动。请暂时只勾选两张影像。"));
        return;
    }
    const QString left_image = resolve_path(selected_images.value(0));
    const QString right_image = resolve_path(selected_images.value(1));
    if (!QFileInfo::exists(left_image) || !QFileInfo::exists(right_image))
    {
        const QString message =
            QStringLiteral("RPC 立体像对不存在或不可访问。\n左：%1\n右：%2").arg(left_image, right_image);
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, message);
        return;
    }

    const QString output_root = xjw::gui::project::resolveProjectOutputDir(
        taskContext.session.projectPath, request.outputDirectory.trimmed(), QStringLiteral("assets/dem/rpc_stereo"));
    const QString output_dir = uniqueRunDirectory(output_root, taskContext.session.chunkId);
    if (!QDir().mkpath(output_dir))
    {
        clearDemContextIfMatches(taskContext);
        emit demPipelineFinished(false, QStringLiteral("无法创建 RPC DEM 输出目录：%1").arg(output_dir));
        return;
    }

    xjw::RpcStereoDemOptions options;
    options.gridResolutionMeters = request.imageStereoOptions.gridResolutionMeters;
    options.maximumFeatures = request.imageStereoOptions.maximumFeatures;
    options.maximumReprojectionErrorPixels = request.imageStereoOptions.maximumReprojectionErrorPixels;

    const auto cancel_flag = taskContext.cancelFlag;
    const QString task_id = taskContext.taskId;
    emit backgroundTaskProgressChanged(task_id, 0, 100);
    emit demPipelineProgressChanged(QStringLiteral("准备 RPC 立体 DEM"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progress_callback = [self, cancel_flag, taskContext, task_id](const QString& stage, int percent)
    {
        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(
            self.data(),
            [self, cancel_flag, taskContext, task_id, stage, percent]()
            {
                if (!self || !self->demContextMatches(taskContext) ||
                    cancel_flag->load(std::memory_order_relaxed))
                {
                    return;
                }
                const int value = std::clamp(percent, 0, 99);
                emit self->backgroundTaskProgressChanged(task_id, value, 100);
                emit self->demPipelineProgressChanged(stage, value);
            },
            Qt::QueuedConnection);
    };

    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [left_image, right_image, output_dir, options, cancel_flag, progress_callback]()
        {
            RpcProductRun run;
            run.ok = xjw::RpcStereoDemGenerator::generate(left_image,
                                                          right_image,
                                                          output_dir,
                                                          options,
                                                          &run.payload,
                                                          &run.error,
                                                          progress_callback,
                                                          cancel_flag.get());
            return run;
        },
        [left_image, right_image, output_dir, project_root, taskContext, cancel_flag, task_id](
            ProjectTerrainProductsManager* manager, xjw::gui::tasks::TaskOutcome<RpcProductRun> outcome)
        {
            if (!manager->demContextMatches(taskContext, false))
            {
                return;
            }
            if (!manager->_session || !manager->_session->isCurrent(taskContext.session))
            {
                manager->clearDemContextIfMatches(taskContext);
                return;
            }
            emit manager->backgroundTaskFinished(task_id);
            if (!outcome.succeeded())
            {
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(false, outcome.errorMessage);
                return;
            }
            const RpcProductRun run = std::move(*outcome.value);
            if (!run.ok)
            {
                const bool cancelled = cancel_flag->load(std::memory_order_relaxed);
                const QString message = cancelled ? QStringLiteral("RPC 立体 DEM 生成已取消。") : run.error;
                if (!cancelled)
                {
                    if (manager->_messages)
                    {
                        manager->_messages->warning(nullptr, QStringLiteral("RPC 立体 DEM"), message);
                    }
                }
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(false, message);
                return;
            }

            const QString dem_path = run.payload.value(QStringLiteral("dem_path")).toString();
            if (!QFileInfo::exists(dem_path))
            {
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(
                    false, QStringLiteral("RPC DEM 管线返回成功，但 DEM 文件不存在：%1").arg(dem_path));
                return;
            }
            const QStringList source_images{projectStoragePath(project_root, left_image),
                                            projectStoragePath(project_root, right_image)};
            QJsonObject record = xjw::gui::project::makeDemResultRecord(
                run.payload.value(QStringLiteral("created_at")).toString(),
                projectStoragePath(project_root, output_dir),
                QString(),
                projectStoragePath(project_root, dem_path),
                QStringLiteral("float32"),
                run.payload.value(QStringLiteral("grid_resolution_m")).toDouble(),
                run.payload.value(QStringLiteral("coordinate_system")).toString(),
                source_images);
            record[QStringLiteral("result_type")] = QStringLiteral("rpc_stereo_dem");
            record[QStringLiteral("terrain_mode")] = QStringLiteral("rpc_stereo");
            record[QStringLiteral("dem_reference")] = QStringLiteral("wgs84_ellipsoidal_height");
            record[QStringLiteral("preview_png")] =
                projectStoragePath(project_root, run.payload.value(QStringLiteral("preview_path")).toString());
            record[QStringLiteral("stereo_point_cloud")] =
                projectStoragePath(project_root, run.payload.value(QStringLiteral("point_cloud_path")).toString());
            record[QStringLiteral("quality_report")] =
                projectStoragePath(project_root, run.payload.value(QStringLiteral("report_path")).toString());
            record[QStringLiteral("direct_coverage_fraction")] =
                run.payload.value(QStringLiteral("direct_coverage_fraction"));
            record[QStringLiteral("median_reprojection_error_px")] =
                run.payload.value(QStringLiteral("median_reprojection_error_px"));
            record[QStringLiteral("rpc_result")] = run.payload;
            QString persistence_error;
            if (!manager->_session->upsertResultRecordByPath(taskContext.session,
                                                             QStringLiteral("dem_results"),
                                                             QStringLiteral("dem_path"),
                                                             record,
                                                             true,
                                                             &persistence_error))
            {
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(
                    false,
                    QStringLiteral("RPC DEM 已生成，但项目成果记录保存失败。%1")
                        .arg(persistence_error.isEmpty() ? QString()
                                                        : QStringLiteral("\n%1").arg(persistence_error)));
                return;
            }
            if (!manager->demContextMatches(taskContext) ||
                cancel_flag->load(std::memory_order_relaxed))
            {
                manager->clearDemContextIfMatches(taskContext);
                emit manager->demPipelineFinished(false, QStringLiteral("RPC 立体 DEM 生成已取消。"));
                return;
            }
            emit manager->demPipelineProgressChanged(QStringLiteral("完成"), 100);
            manager->clearDemContextIfMatches(taskContext);
            emit manager->demPipelineFinished(true, QStringLiteral("RPC 立体 DEM 已生成：%1").arg(dem_path));
        }));
}

void ProjectTerrainProductsManager::startRpcDomAsync(
    const xjw::gui::project::OrthoGenerationRequest& request,
    const xjw::gui::project::ProjectTaskContext& taskContext)
{
    const QString project_root =
        xjw::common::project::ProjectIO::projectRootFromPlascan(taskContext.session.projectPath);
    const auto resolve_path = [&taskContext](const QString& path)
    {
        return xjw::common::project::ProjectIO::resolveProjectResourcePath(
            taskContext.session.projectPath, path.trimmed());
    };
    const QString dem_path = resolve_path(request.demPath);
    if (!QFileInfo::exists(dem_path))
    {
        clearOrthoContextIfMatches(taskContext);
        emit orthoPipelineFinished(false, QStringLiteral("找不到地理正射所需的 DEM：%1").arg(dem_path), QJsonObject());
        return;
    }

    QStringList images;
    const QStringList requested_images = request.sourceImages;
    for (const QString& path : requested_images)
    {
        const QString resolved = resolve_path(path);
        const QString suffix = QFileInfo(resolved).suffix().toLower();
        if (QFileInfo::exists(resolved) && (suffix == QLatin1String("tif") || suffix == QLatin1String("tiff")))
        {
            images.append(resolved);
        }
    }
    if (images.isEmpty())
    {
        clearOrthoContextIfMatches(taskContext);
        emit orthoPipelineFinished(false, QStringLiteral("没有已选择且带有效地理定位模型的 GeoTIFF 影像。"), QJsonObject());
        return;
    }

    QString output_path = request.outputPath.trimmed();
    if (output_path.isEmpty())
    {
        output_path = QDir(project_root).filePath(QStringLiteral("assets/ortho/rpc_dom.tif"));
    }
    else if (QFileInfo(output_path).isRelative())
    {
        output_path = QDir(project_root).filePath(output_path);
    }
    output_path = QDir::cleanPath(output_path);
    if (samePath(output_path, dem_path) || samePath(output_path, taskContext.session.projectPath))
    {
        clearOrthoContextIfMatches(taskContext);
        emit orthoPipelineFinished(
            false, QStringLiteral("正射影像输出路径不能覆盖输入 DEM 或项目文件。"), QJsonObject());
        return;
    }
    for (const QString& image : images)
    {
        if (samePath(output_path, image))
        {
            clearOrthoContextIfMatches(taskContext);
            emit orthoPipelineFinished(
                false, QStringLiteral("正射影像输出路径不能覆盖源影像：%1").arg(image), QJsonObject());
            return;
        }
    }
    if (!QDir().mkpath(QFileInfo(output_path).absolutePath()))
    {
        clearOrthoContextIfMatches(taskContext);
        emit orthoPipelineFinished(false, QStringLiteral("无法创建地理正射输出目录。"), QJsonObject());
        return;
    }

    xjw::RpcDomOptions options;
    options.blendAllImages = request.options.blendMode != xjw::OrthoBlendMode::FirstValid;
    options.writePreview = true;
    const auto cancel_flag = taskContext.cancelFlag;
    const QString task_id = taskContext.taskId;
    emit backgroundTaskProgressChanged(task_id, 0, 100);
    emit orthoPipelineStarted();
    emit orthoPipelineProgressChanged(QStringLiteral("准备地理正射影像"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progress_callback = [self, cancel_flag, taskContext, task_id](const QString& stage, int percent)
    {
        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(
            self.data(),
            [self, cancel_flag, taskContext, task_id, stage, percent]()
            {
                if (!self || !self->orthoContextMatches(taskContext) ||
                    cancel_flag->load(std::memory_order_relaxed))
                {
                    return;
                }
                const int value = std::clamp(percent, 0, 99);
                emit self->backgroundTaskProgressChanged(task_id, value, 100);
                emit self->orthoPipelineProgressChanged(stage, value);
            },
            Qt::QueuedConnection);
    };

    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [images, dem_path, output_path, options, cancel_flag, progress_callback]()
        {
            RpcProductRun run;
            run.ok = xjw::RpcDomGenerator::generate(
                images, dem_path, output_path, options, &run.payload, &run.error, progress_callback, cancel_flag.get());
            return run;
        },
        [images, dem_path, output_path, project_root, taskContext, cancel_flag, task_id](
            ProjectTerrainProductsManager* manager, xjw::gui::tasks::TaskOutcome<RpcProductRun> outcome)
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
            emit manager->backgroundTaskFinished(task_id);
            if (!outcome.succeeded())
            {
                manager->clearOrthoContextIfMatches(taskContext);
                emit manager->orthoPipelineFinished(false, outcome.errorMessage, QJsonObject());
                return;
            }
            RpcProductRun run = std::move(*outcome.value);
            if (!run.ok)
            {
                const QString message =
                    cancel_flag->load(std::memory_order_relaxed) ? QStringLiteral("地理正射影像生成已取消。") : run.error;
                manager->clearOrthoContextIfMatches(taskContext);
                emit manager->orthoPipelineFinished(false, message, run.payload);
                return;
            }
            run.payload[QStringLiteral("output_path")] = output_path;
            run.payload[QStringLiteral("coverage_ratio")] = run.payload.value(QStringLiteral("coverage_fraction"));
            run.payload[QStringLiteral("source_image_count")] = images.size();
            QStringList storage_images;
            for (const QString& image : images)
            {
                storage_images.append(projectStoragePath(project_root, image));
            }
            QJsonObject record =
                xjw::gui::project::makeOrthoResultRecord(run.payload.value(QStringLiteral("created_at")).toString(),
                                                         projectStoragePath(project_root, dem_path),
                                                         projectStoragePath(project_root, output_path),
                                                         images.size(),
                                                         storage_images,
                                                         true,
                                                         run.payload.value(QStringLiteral("pixel_size_x")).toDouble(),
                                                         run.payload);
            record[QStringLiteral("result_type")] = QStringLiteral("rpc_dom");
            record[QStringLiteral("product_mode")] = QStringLiteral("rpc");
            record[QStringLiteral("terrain_mode")] = QStringLiteral("rpc_stereo");
            record[QStringLiteral("preview_png")] =
                projectStoragePath(project_root, run.payload.value(QStringLiteral("preview_path")).toString());
            record[QStringLiteral("quality_report")] =
                projectStoragePath(project_root, run.payload.value(QStringLiteral("report_path")).toString());
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
                    QStringLiteral("地理正射影像已生成，但项目成果记录保存失败。%1")
                        .arg(persistence_error.isEmpty() ? QString()
                                                        : QStringLiteral("\n%1").arg(persistence_error)),
                    record);
                return;
            }
            if (!manager->orthoContextMatches(taskContext) ||
                cancel_flag->load(std::memory_order_relaxed))
            {
                manager->clearOrthoContextIfMatches(taskContext);
                emit manager->orthoPipelineFinished(false, QStringLiteral("地理正射影像生成已取消。"), record);
                return;
            }
            emit manager->orthoPipelineProgressChanged(QStringLiteral("完成"), 100);
            manager->clearOrthoContextIfMatches(taskContext);
            emit manager->orthoPipelineFinished(
                true,
                QStringLiteral("地理正射影像已生成：%1\n覆盖率：%2%")
                    .arg(output_path)
                    .arg(run.payload.value(QStringLiteral("coverage_fraction")).toDouble() * 100.0, 0, 'f', 1),
                record);
        }));
}
