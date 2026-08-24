#include "ProjectTerrainProductsManager.h"

#include "ProjectManager.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "GuiTaskRunner.h"
#include "project/ProjectIO.h"
#include "project/ProjectSessionModel.h"

#include "RpcDomGenerator.h"
#include "RpcStereoDemGenerator.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QMessageBox>
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

void ProjectTerrainProductsManager::startRpcStereoDemAsync(const xjw::gui::project::DemGenerationRequest& request)
{
    const auto session = _owner->currentSessionContext();
    const QString project_root = xjw::common::project::ProjectIO::projectRootFromPlascan(session.projectPath);
    const auto resolve_path = [&session](const QString& path)
    { return xjw::common::project::ProjectIO::resolveProjectResourcePath(session.projectPath, path.trimmed()); };
    const QStringList selected_images = request.imageStereoOptions.sourceImages;
    if (selected_images.size() != 2)
    {
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
        emit demPipelineFinished(false, message);
        return;
    }

    const QString output_root = xjw::gui::project::resolveProjectOutputDir(
        session.projectPath, request.outputDirectory.trimmed(), QStringLiteral("assets/dem/rpc_stereo"));
    const QString output_dir = uniqueRunDirectory(output_root, session.chunkId);
    if (!QDir().mkpath(output_dir))
    {
        emit demPipelineFinished(false, QStringLiteral("无法创建 RPC DEM 输出目录：%1").arg(output_dir));
        return;
    }

    xjw::RpcStereoDemOptions options;
    options.gridResolutionMeters = request.imageStereoOptions.gridResolutionMeters;
    options.maximumFeatures = request.imageStereoOptions.maximumFeatures;
    options.maximumReprojectionErrorPixels = request.imageStereoOptions.maximumReprojectionErrorPixels;

    const auto cancel_flag = std::make_shared<std::atomic_bool>(false);
    const QString task_id = QStringLiteral("dem-rpc:%1").arg(QUuid::createUuid().toString(QUuid::WithoutBraces));
    _demCancelFlag = cancel_flag;
    _demTaskId = task_id;
    _demTaskChunkId = session.chunkId;
    emit backgroundTaskProgressChanged(task_id, 0, 100);
    emit demPipelineProgressChanged(QStringLiteral("准备 RPC 立体 DEM"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progress_callback = [self, cancel_flag, session, task_id](const QString& stage, int percent)
    {
        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(
            self.data(),
            [self, cancel_flag, session, task_id, stage, percent]()
            {
                if (!self || self->_demCancelFlag != cancel_flag || !self->_owner ||
                    !self->_owner->isCurrentSession(session))
                {
                    return;
                }
                const int value = std::clamp(percent, 0, 99);
                emit self->backgroundTaskProgressChanged(task_id, value, 100);
                emit self->demPipelineProgressChanged(stage, value);
            },
            Qt::QueuedConnection);
    };

    xjw::gui::tasks::runGuardedWithOutcome(
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
        [left_image, right_image, output_dir, project_root, session, cancel_flag, task_id](
            ProjectTerrainProductsManager* manager, xjw::gui::tasks::TaskOutcome<RpcProductRun> outcome)
        {
            emit manager->backgroundTaskFinished(task_id);
            if (manager->_demTaskId != task_id)
            {
                return;
            }
            manager->_demTaskId.clear();
            manager->_demTaskChunkId.clear();
            if (manager->_demCancelFlag == cancel_flag)
            {
                manager->_demCancelFlag.reset();
            }
            if (!manager->_owner || !manager->_projectData || !manager->_owner->isCurrentSession(session))
            {
                emit manager->demPipelineFinished(false, QStringLiteral("项目已切换，RPC DEM 结果未写入当前项目。"));
                return;
            }
            if (!outcome.succeeded())
            {
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
                    QMessageBox::warning(manager->_parentWidget, QStringLiteral("RPC 立体 DEM"), message);
                }
                emit manager->demPipelineFinished(false, message);
                return;
            }

            const QString dem_path = run.payload.value(QStringLiteral("dem_path")).toString();
            if (!QFileInfo::exists(dem_path))
            {
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
            if (!manager->_projectData->upsertResultRecordByPath(
                    QStringLiteral("dem_results"), QStringLiteral("dem_tif"), record, true))
            {
                emit manager->demPipelineFinished(false, QStringLiteral("RPC DEM 已生成，但项目成果记录保存失败。"));
                return;
            }
            manager->_owner->refreshReconstructionQualityReport();
            emit manager->demPipelineProgressChanged(QStringLiteral("完成"), 100);
            emit manager->demPipelineFinished(true, QStringLiteral("RPC 立体 DEM 已生成：%1").arg(dem_path));
        });
}

void ProjectTerrainProductsManager::startRpcDomAsync(const xjw::gui::project::OrthoGenerationRequest& request)
{
    const auto session = _owner->currentSessionContext();
    const QString project_root = xjw::common::project::ProjectIO::projectRootFromPlascan(session.projectPath);
    const auto resolve_path = [&session](const QString& path)
    { return xjw::common::project::ProjectIO::resolveProjectResourcePath(session.projectPath, path.trimmed()); };
    const QString dem_path = resolve_path(request.demPath);
    if (!QFileInfo::exists(dem_path))
    {
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
    if (samePath(output_path, dem_path) || samePath(output_path, session.projectPath))
    {
        emit orthoPipelineFinished(
            false, QStringLiteral("正射影像输出路径不能覆盖输入 DEM 或项目文件。"), QJsonObject());
        return;
    }
    for (const QString& image : images)
    {
        if (samePath(output_path, image))
        {
            emit orthoPipelineFinished(
                false, QStringLiteral("正射影像输出路径不能覆盖源影像：%1").arg(image), QJsonObject());
            return;
        }
    }
    if (!QDir().mkpath(QFileInfo(output_path).absolutePath()))
    {
        emit orthoPipelineFinished(false, QStringLiteral("无法创建地理正射输出目录。"), QJsonObject());
        return;
    }

    xjw::RpcDomOptions options;
    options.blendAllImages = request.options.blendMode != xjw::OrthoBlendMode::FirstValid;
    options.writePreview = true;
    const auto cancel_flag = std::make_shared<std::atomic_bool>(false);
    const QString task_id = QStringLiteral("ortho-rpc:%1").arg(QUuid::createUuid().toString(QUuid::WithoutBraces));
    _orthoCancelFlag = cancel_flag;
    _orthoTaskChunkId = session.chunkId;
    emit backgroundTaskProgressChanged(task_id, 0, 100);
    emit orthoPipelineStarted();
    emit orthoPipelineProgressChanged(QStringLiteral("准备地理正射影像"), 0);

    QPointer<ProjectTerrainProductsManager> self(this);
    const auto progress_callback = [self, cancel_flag, session, task_id](const QString& stage, int percent)
    {
        if (!self)
        {
            return;
        }
        QMetaObject::invokeMethod(
            self.data(),
            [self, cancel_flag, session, task_id, stage, percent]()
            {
                if (!self || self->_orthoCancelFlag != cancel_flag || !self->_owner ||
                    !self->_owner->isCurrentSession(session))
                {
                    return;
                }
                const int value = std::clamp(percent, 0, 99);
                emit self->backgroundTaskProgressChanged(task_id, value, 100);
                emit self->orthoPipelineProgressChanged(stage, value);
            },
            Qt::QueuedConnection);
    };

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [images, dem_path, output_path, options, cancel_flag, progress_callback]()
        {
            RpcProductRun run;
            run.ok = xjw::RpcDomGenerator::generate(
                images, dem_path, output_path, options, &run.payload, &run.error, progress_callback, cancel_flag.get());
            return run;
        },
        [images, dem_path, output_path, project_root, session, cancel_flag, task_id](
            ProjectTerrainProductsManager* manager, xjw::gui::tasks::TaskOutcome<RpcProductRun> outcome)
        {
            emit manager->backgroundTaskFinished(task_id);
            if (manager->_orthoCancelFlag == cancel_flag)
            {
                manager->_orthoCancelFlag.reset();
                manager->_orthoTaskChunkId.clear();
            }
            if (!manager->_owner || !manager->_projectData || !manager->_owner->isCurrentSession(session))
            {
                emit manager->orthoPipelineFinished(
                    false, QStringLiteral("项目已切换，正射结果未写入当前项目。"), QJsonObject());
                return;
            }
            if (!outcome.succeeded())
            {
                emit manager->orthoPipelineFinished(false, outcome.errorMessage, QJsonObject());
                return;
            }
            RpcProductRun run = std::move(*outcome.value);
            if (!run.ok)
            {
                const QString message =
                    cancel_flag->load(std::memory_order_relaxed) ? QStringLiteral("地理正射影像生成已取消。") : run.error;
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
            if (!manager->_projectData->upsertResultRecordByPath(
                    QStringLiteral("ortho_results"), QStringLiteral("output_path"), record, true))
            {
                emit manager->orthoPipelineFinished(
                    false, QStringLiteral("地理正射影像已生成，但项目成果记录保存失败。"), record);
                return;
            }
            manager->_owner->refreshReconstructionQualityReport();
            emit manager->orthoPipelineProgressChanged(QStringLiteral("完成"), 100);
            emit manager->orthoPipelineFinished(
                true,
                QStringLiteral("地理正射影像已生成：%1\n覆盖率：%2%")
                    .arg(output_path)
                    .arg(run.payload.value(QStringLiteral("coverage_fraction")).toDouble() * 100.0, 0, 'f', 1),
                record);
        });
}
