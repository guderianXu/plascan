#include "ProjectSparseReconstructionManager.h"

#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "project/ProjectIO.h"
#include "ProjectResultRecords.h"
#include "ProjectSparseWorkflow.h"
#include "ProjectWorkflowOperations.h"
#include "TriangulationService.h"
#include "project/SparseResultQuality.h"
#include "tasks/GuiTaskRunner.h"
#include "Logger.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>

#include <utility>

using xjw::gui::project::buildSparsePointWorkflowSuccessMessage;
using xjw::core::project::findLatestAtResultIndex;
using xjw::gui::project::mergeSparseQualityIntoRecord;
using xjw::core::project::resolveSparsePointContextResult;
using xjw::gui::project::runSparsePointWorkflowResult;
using xjw::core::project::SparsePointContext;
using xjw::core::project::SparsePointOperationResult;
using xjw::gui::project::SparsePointWorkflowResult;
using xjw::gui::project::SparsePointWorkflowKind;
using xjw::gui::project::SparsePointWorkflowSpec;
using xjw::gui::project::sparsePointWorkflowSpec;
using xjw::core::project::writeJsonObjectFile;

namespace
{

bool isCurrentTask(xjw::gui::project::ProjectSession* session,
                   const xjw::gui::project::ProjectTaskContext& taskContext)
{
    return session && taskContext.cancelFlag &&
           !taskContext.cancelFlag->load(std::memory_order_relaxed) &&
           session->isCurrent(taskContext.session);
}

void showWarning(ProjectUiMessageAdapter* messages,
                 QWidget* parent,
                 const QString& title,
                 const QString& text)
{
    if (messages)
    {
        messages->warning(parent, title, text);
    }
}

} // namespace

ProjectSparseReconstructionManager::ProjectSparseReconstructionManager(
    xjw::gui::project::ProjectSession *session,
    ProjectUiMessageAdapter *messages,
    QWidget *parentWidget,
    TiePointResultWriter tiePointResultWriter,
    QObject *parent)
    : QObject(parent)
    , _session(session)
    , _messages(messages)
    , _parentWidget(parentWidget)
    , _tiePointResultWriter(std::move(tiePointResultWriter))
{
}

void ProjectSparseReconstructionManager::setTiePointResultWriter(TiePointResultWriter tiePointResultWriter)
{
    _tiePointResultWriter = std::move(tiePointResultWriter);
}

void ProjectSparseReconstructionManager::waitForActiveTask()
{
    for (QFuture<void>& future : _activeFutures)
    {
        if (future.isValid() && !future.isFinished())
        {
            future.waitForFinished();
        }
    }
    _activeFutures.clear();
    _taskReservation = false;
}

bool ProjectSparseReconstructionManager::reserveTask()
{
    reapFinishedFutures();
    if (_taskReservation || hasRunningTask())
    {
        return false;
    }
    _taskReservation = true;
    return true;
}

void ProjectSparseReconstructionManager::releaseTaskReservation()
{
    _taskReservation = false;
}

bool ProjectSparseReconstructionManager::hasRunningTask() const noexcept
{
    if (_taskReservation)
    {
        return true;
    }
    for (const QFuture<void>& future : _activeFutures)
    {
        if (!future.isFinished())
        {
            return true;
        }
    }
    return false;
}

void ProjectSparseReconstructionManager::trackFutureForTesting(QFuture<void> future)
{
    trackFuture(std::move(future));
}

void ProjectSparseReconstructionManager::reapFinishedFutures()
{
    QVector<QFuture<void>> pending;
    pending.reserve(_activeFutures.size());
    for (const QFuture<void>& future : _activeFutures)
    {
        if (!future.isFinished())
        {
            pending.append(future);
        }
    }
    _activeFutures = std::move(pending);
}

void ProjectSparseReconstructionManager::trackFuture(QFuture<void> future)
{
    reapFinishedFutures();
    _taskReservation = false;
    _activeFutures.append(std::move(future));
}

void ProjectSparseReconstructionManager::startTriangulationAsync(
    const QJsonObject &settings,
    const xjw::gui::project::ProjectTaskContext& taskContext)
{
    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }
    if (!_session->hasProject())
    {
        showWarning(_messages,
                    _parentWidget,
                    QStringLiteral("生成两视预览云"),
                    QStringLiteral("请先打开项目后再执行三角化"));
        if (isCurrentTask(_session, taskContext))
        {
            emit atProgressFinished(false);
        }
        return;
    }

    const QStringList selectedImages = _session ? _session->allImages() : QStringList();
    if (selectedImages.size() < 2)
    {
        showWarning(_messages,
                    _parentWidget,
                    QStringLiteral("生成两视预览云"),
                    QStringLiteral("至少需要两张影像才能执行三角化"));
        if (isCurrentTask(_session, taskContext))
        {
            emit atProgressFinished(false);
        }
        return;
    }

    QJsonObject mergedMeta = _session->coreMetadata();
    if (mergedMeta.value(QStringLiteral("project_files")).isObject())
    {
        mergedMeta = mergedMeta.value(QStringLiteral("project_files")).toObject();
    }
    const QJsonObject runtimeMeta = _session->metadata();
    for (auto it = runtimeMeta.begin(); it != runtimeMeta.end(); ++it)
    {
        if (it.key() != QLatin1String("images") && it.key() != QLatin1String("project_files"))
        {
            mergedMeta.insert(it.key(), it.value());
        }
    }

    const auto session = taskContext.session;
    const QString projectPath = session.projectPath;
    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    const bool overwriteExistingResult = settings.value(QStringLiteral("overwriteExistingResult")).toBool(false);
    int replaceIndex = -1;
    QString outputDir;
    if (overwriteExistingResult)
    {
        replaceIndex = findLatestAtResultIndex(runtimeMeta, QStringLiteral("triangulation"));
        if (replaceIndex < 0)
        {
            replaceIndex = findLatestAtResultIndex(runtimeMeta);
        }
        const QJsonArray existing = runtimeMeta.value(QStringLiteral("aerial_triangulation_results")).toArray();
        if (replaceIndex >= 0 && replaceIndex < existing.size())
        {
            outputDir = existing.at(replaceIndex).toObject().value(QStringLiteral("output_dir")).toString();
        }
    }
    if (outputDir.isEmpty())
    {
        outputDir = QDir(assetsDir).filePath(
            QStringLiteral("aerial_triangulation/triangulation_%1")
                .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss"))));
    }

    xjw::core::project::TriangulationServiceOptions options;
    options.outputDir = outputDir;
    options.minTriAngleDeg = settings.value(QStringLiteral("minAngle")).toDouble(2.0);
    options.maxReprojErrorPx = settings.value(QStringLiteral("reprojThreshold")).toDouble(2.0);
    options.minObservations = settings.value(QStringLiteral("minObservations")).toInt(2);
    options.ignoreTwoViewTracks = settings.value(QStringLiteral("ignoreTwoView")).toBool(false);
    options.minTrackLength = settings.value(QStringLiteral("minTrackLen")).toInt(2);

    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }
    emit atProgressChanged(QStringLiteral("正在构建两视预览云..."), 10);
    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }

    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [mergedMeta, selectedImages, options]()
        {
            return xjw::core::project::TriangulationService::run(mergedMeta, selectedImages, options);
        },
        [selectedImages, options, session, taskContext](
            ProjectSparseReconstructionManager *self,
            xjw::gui::tasks::TaskOutcome<xjw::core::project::TriangulationServiceResult> outcome)
        {
            if (!isCurrentTask(self->_session, taskContext) || !taskContext.session.matches(session))
            {
                return;
            }

            if (!outcome.succeeded())
            {
                showWarning(self->_messages,
                            self->_parentWidget,
                            QStringLiteral("生成两视预览云"),
                            outcome.errorMessage);
                if (isCurrentTask(self->_session, taskContext))
                {
                    emit self->atProgressFinished(false);
                }
                return;
            }

            const auto &result = *outcome.value;

            if (!result.success)
            {
                showWarning(self->_messages,
                            self->_parentWidget,
                            QStringLiteral("生成两视预览云"),
                            result.errorMessage);
                if (isCurrentTask(self->_session, taskContext))
                {
                    emit self->atProgressFinished(false);
                }
                return;
            }
            self->finalizeTriangulationSuccess(result, selectedImages, options, taskContext);
        }));
}

void ProjectSparseReconstructionManager::startSparseCloudOutlierRemovalAsync(
    const QJsonObject &settings,
    const xjw::gui::project::ProjectTaskContext& taskContext)
{
    startSparsePointWorkflow(SparsePointWorkflowKind::OutlierRemoval, settings, taskContext);
}

void ProjectSparseReconstructionManager::startSparseCloudLocalOptimAsync(
    const QJsonObject &settings,
    const xjw::gui::project::ProjectTaskContext& taskContext)
{
    startSparsePointWorkflow(SparsePointWorkflowKind::LocalOptim, settings, taskContext);
}

void ProjectSparseReconstructionManager::startSparseCloudRefineAsync(
    const QJsonObject &settings,
    const xjw::gui::project::ProjectTaskContext& taskContext)
{
    startSparsePointWorkflow(SparsePointWorkflowKind::Refine, settings, taskContext);
}

void ProjectSparseReconstructionManager::finalizeTriangulationSuccess(
    const xjw::core::project::TriangulationServiceResult &result,
    const QStringList &selectedImages,
    const xjw::core::project::TriangulationServiceOptions &options,
    const xjw::gui::project::ProjectTaskContext& taskContext)
{
    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }
    if (!_tiePointResultWriter)
    {
        emit atProgressFinished(false);
        return;
    }

    QJsonObject extraRecord;
    QJsonObject files;
    const QString sidecarPath = QDir(options.outputDir).filePath(QStringLiteral("sparse_cloud_points.json"));
    QString writeError;
    if (writeJsonObjectFile(sidecarPath, result.resultJson, &writeError))
    {
        files[QStringLiteral("sparse_cloud_points_json")] = sidecarPath;
    }
    else
    {
        LOG_WARN(QStringLiteral("写入三角化点级 sidecar 失败: %1").arg(writeError));
    }
    extraRecord[QStringLiteral("files")] = files;
    extraRecord[QStringLiteral("source")] = QStringLiteral("triangulation");
    extraRecord[QStringLiteral("operation")] = QStringLiteral("triangulation");
    extraRecord[QStringLiteral("candidate_track_count")] = result.candidateTrackCount;
    const QJsonObject quality = result.resultJson.value(QStringLiteral("quality")).toObject();
    if (!quality.isEmpty())
    {
        extraRecord = mergeSparseQualityIntoRecord(extraRecord, quality);
    }

    if (!_tiePointResultWriter(taskContext,
                               result.sparseCloudPath,
                               result.exportedPointCount,
                               selectedImages,
                               options.outputDir,
                               extraRecord))
    {
        if (isCurrentTask(_session, taskContext))
        {
            emit atProgressFinished(false);
        }
        return;
    }
    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }

    LOG_INFO(QStringLiteral("三角化完成: 候选轨迹=%1 导出点数=%2 输出=%3")
                 .arg(result.candidateTrackCount)
                 .arg(result.exportedPointCount)
                 .arg(result.sparseCloudPath));

    emit tiePointResultReady(result.sparseCloudPath, sidecarPath);
    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }
    emit atProgressFinished(true);
    if (_messages && isCurrentTask(_session, taskContext))
    {
        _messages->information(
            _parentWidget,
            QStringLiteral("生成两视预览云"),
            QStringLiteral("两视预览云生成完成。\n候选轨迹: %1\n导出点数: %2\n输出文件: %3")
                .arg(result.candidateTrackCount)
                .arg(result.exportedPointCount)
                .arg(result.sparseCloudPath));
    }
}

void ProjectSparseReconstructionManager::startSparsePointWorkflow(
    SparsePointWorkflowKind kind,
    const QJsonObject &settings,
    const xjw::gui::project::ProjectTaskContext& taskContext)
{
    const SparsePointWorkflowSpec spec = sparsePointWorkflowSpec(kind);
    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }
    if (!_session->hasProject())
    {
        showWarning(_messages, _parentWidget, spec.title, spec.projectOpenMessage);
        if (isCurrentTask(_session, taskContext))
        {
            emit atProgressFinished(false);
        }
        return;
    }

    SparsePointContext context;
    if (settings.value(QStringLiteral("sourceKind")).toString() == QLatin1String("external_ply"))
    {
        const QString path = QDir::cleanPath(
            settings.value(QStringLiteral("externalSparseCloudPath")).toString().trimmed());
        if (path.isEmpty() || !QFileInfo::exists(path))
        {
            showWarning(_messages,
                        _parentWidget,
                        spec.title,
                        QStringLiteral("外部 PLY 点云不存在: %1").arg(path));
            if (isCurrentTask(_session, taskContext))
            {
                emit atProgressFinished(false);
            }
            return;
        }
        context.sourceResultIndex = -1;
        context.sparseCloudPath = path;
    }
    else
    {
        const auto contextResult = resolveSparsePointContextResult(
            _session ? _session->metadata() : QJsonObject(),
            settings.value(QStringLiteral("sourceAtIndex")).toInt(-1));
        if (!contextResult.status.ok)
        {
            showWarning(_messages, _parentWidget, spec.title, contextResult.status.errorMessage);
            if (isCurrentTask(_session, taskContext))
            {
                emit atProgressFinished(false);
            }
            return;
        }
        context = contextResult.context;

        const auto normalizedOverride = [&settings](const QString &key)
        {
            const QString value = settings.value(key).toString().trimmed();
            return value.isEmpty() ? QString() : QDir::cleanPath(value);
        };
        const QString sparseCloudOverride = normalizedOverride(
            QStringLiteral("sourceSparseCloudPath"));
        const QString sidecarOverride = normalizedOverride(
            QStringLiteral("sourceSidecarPath"));
        if (!sparseCloudOverride.isEmpty())
        {
            if (!QFileInfo(sparseCloudOverride).isFile())
            {
                showWarning(_messages,
                            _parentWidget,
                            spec.title,
                            QStringLiteral("稀疏点云文件不存在: %1").arg(sparseCloudOverride));
                if (isCurrentTask(_session, taskContext))
                {
                    emit atProgressFinished(false);
                }
                return;
            }
            context.sparseCloudPath = sparseCloudOverride;
        }
        if (!sidecarOverride.isEmpty())
        {
            if (!QFileInfo(sidecarOverride).isFile())
            {
                showWarning(_messages,
                            _parentWidget,
                            spec.title,
                            QStringLiteral("连接点逐点数据文件不存在: %1").arg(sidecarOverride));
                if (isCurrentTask(_session, taskContext))
                {
                    emit atProgressFinished(false);
                }
                return;
            }
            context.sidecarPath = sidecarOverride;
        }
    }

    const auto session = taskContext.session;
    const QString projectPath = session.projectPath;
    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    const QString outputDir = QDir(assetsDir).filePath(
        QStringLiteral("aerial_triangulation/%1_%2")
            .arg(spec.outputDirPrefix,
                 QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss"))));

    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }
    emit atProgressChanged(spec.progressMessage, 20);
    if (!isCurrentTask(_session, taskContext))
    {
        return;
    }

    QJsonObject operationSettings = settings;
    operationSettings.remove(QStringLiteral("sourceSparseCloudPath"));
    operationSettings.remove(QStringLiteral("sourceSidecarPath"));
    trackFuture(xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [kind, context, operationSettings, outputDir]()
        {
            return runSparsePointWorkflowResult(
                kind, context, operationSettings, outputDir);
        },
        [spec, context, session, taskContext](
            ProjectSparseReconstructionManager *self,
            xjw::gui::tasks::TaskOutcome<SparsePointWorkflowResult> outcome)
        {
            if (!isCurrentTask(self->_session, taskContext) || !taskContext.session.matches(session))
            {
                return;
            }

            if (!outcome.succeeded())
            {
                showWarning(self->_messages, self->_parentWidget, spec.title, outcome.errorMessage);
                if (isCurrentTask(self->_session, taskContext))
                {
                    emit self->atProgressFinished(false);
                }
                return;
            }

            const auto &workflowResult = *outcome.value;

            if (!workflowResult.status.ok)
            {
                showWarning(
                    self->_messages, self->_parentWidget, spec.title, workflowResult.status.errorMessage);
                if (isCurrentTask(self->_session, taskContext))
                {
                    emit self->atProgressFinished(false);
                }
                return;
            }

            const SparsePointOperationResult &operationResult = workflowResult.operation;

            if (!self->_tiePointResultWriter ||
                !self->_tiePointResultWriter(taskContext,
                                              operationResult.sparseCloudPath,
                                              operationResult.outputCount,
                                              context.selectedImages,
                                              operationResult.outputDir,
                                              operationResult.extraRecord))
            {
                if (isCurrentTask(self->_session, taskContext))
                {
                    emit self->atProgressFinished(false);
                }
                return;
            }
            if (!isCurrentTask(self->_session, taskContext))
            {
                return;
            }

            emit self->tiePointResultReady(operationResult.sparseCloudPath,
                                           operationResult.sidecarPath);
            if (!isCurrentTask(self->_session, taskContext))
            {
                return;
            }
            emit self->atProgressFinished(true);
            if (self->_messages && isCurrentTask(self->_session, taskContext))
            {
                self->_messages->information(
                    self->_parentWidget,
                    spec.title,
                    buildSparsePointWorkflowSuccessMessage(spec, operationResult));
            }
        }));
}
