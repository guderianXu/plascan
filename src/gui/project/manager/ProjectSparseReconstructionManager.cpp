#include "ProjectSparseReconstructionManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "project/ProjectIO.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "ProjectSparseWorkflow.h"
#include "ProjectWorkflowUtils.h"
#include "ProjectOpenGuard.h"
#include "TriangulationService.h"
#include "project/SparseResultQuality.h"
#include "tasks/GuiTaskRunner.h"
#include "Logger.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QMessageBox>
#include <QPointer>

using xjw::gui::project::buildSparsePointWorkflowSuccessMessage;
using xjw::gui::project::findLatestAtResultIndex;
using xjw::gui::project::mergeSparseQualityIntoRecord;
using xjw::gui::project::projectFilesMeta;
using xjw::gui::project::resolveSparsePointContextResult;
using xjw::gui::project::runSparsePointWorkflowResult;
using xjw::gui::project::SparsePointContext;
using xjw::gui::project::SparsePointOperationResult;
using xjw::gui::project::SparsePointWorkflowResult;
using xjw::gui::project::SparsePointWorkflowKind;
using xjw::gui::project::SparsePointWorkflowSpec;
using xjw::gui::project::sparsePointWorkflowSpec;
using xjw::gui::project::writeJsonObjectFile;

ProjectSparseReconstructionManager::ProjectSparseReconstructionManager(ProjectManager *owner,
                                                                       ProjectData *projectData,
                                                                       QWidget *parentWidget,
                                                                       QObject *parent)
    : QObject(parent)
    , _owner(owner)
    , _projectData(projectData)
    , _parentWidget(parentWidget)
{
}

void ProjectSparseReconstructionManager::startTriangulationAsync(const QJsonObject &settings)
{
    if (!xjw::gui::project::requireOpenProject(
            _projectData,
            _parentWidget,
            QStringLiteral("请先打开项目后再执行三角化"),
            QStringLiteral("生成两视预览云")))
    {
        return;
    }

    const QStringList selectedImages = _owner->getAllImages();
    if (selectedImages.size() < 2)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("生成两视预览云"),
                             QStringLiteral("至少需要两张影像才能执行三角化"));
        return;
    }

    QJsonObject mergedMeta = projectFilesMeta(_projectData);
    const QJsonObject runtimeMeta = _owner->currentMeta();
    for (auto it = runtimeMeta.begin(); it != runtimeMeta.end(); ++it)
    {
        if (it.key() != QLatin1String("images") && it.key() != QLatin1String("project_files"))
        {
            mergedMeta.insert(it.key(), it.value());
        }
    }

    const auto session = _owner->currentSessionContext();
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

    emit atProgressChanged(QStringLiteral("正在构建两视预览云..."), 10);

    QPointer<ProjectManager> ownerGuard(_owner);
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [mergedMeta, selectedImages, options]()
        {
            return xjw::core::project::TriangulationService::run(mergedMeta, selectedImages, options);
        },
        [selectedImages, options, ownerGuard, session](
            ProjectSparseReconstructionManager *self,
            xjw::gui::tasks::TaskOutcome<xjw::core::project::TriangulationServiceResult> outcome)
        {
            if (!ownerGuard || !ownerGuard->isCurrentSession(session))
            {
                return;
            }

            if (!outcome.succeeded())
            {
                emit self->atProgressFinished(false);
                QMessageBox::warning(self->_parentWidget,
                                     QStringLiteral("生成两视预览云"),
                                     outcome.errorMessage);
                return;
            }

            const auto &result = *outcome.value;

            if (!result.success)
            {
                emit self->atProgressFinished(false);
                QMessageBox::warning(self->_parentWidget,
                                     QStringLiteral("生成两视预览云"),
                                     result.errorMessage);
                return;
            }
            self->finalizeTriangulationSuccess(result, selectedImages, options);
        });
}

void ProjectSparseReconstructionManager::startSparseCloudOutlierRemovalAsync(const QJsonObject &settings)
{
    startSparsePointWorkflow(SparsePointWorkflowKind::OutlierRemoval, settings);
}

void ProjectSparseReconstructionManager::startSparseCloudLocalOptimAsync(const QJsonObject &settings)
{
    startSparsePointWorkflow(SparsePointWorkflowKind::LocalOptim, settings);
}

void ProjectSparseReconstructionManager::startSparseCloudRefineAsync(const QJsonObject &settings)
{
    startSparsePointWorkflow(SparsePointWorkflowKind::Refine, settings);
}

void ProjectSparseReconstructionManager::finalizeTriangulationSuccess(
    const xjw::core::project::TriangulationServiceResult &result,
    const QStringList &selectedImages,
    const xjw::core::project::TriangulationServiceOptions &options)
{
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

    if (!_owner->replaceTiePointResult(result.sparseCloudPath,
                                       result.exportedPointCount,
                                       selectedImages,
                                       options.outputDir,
                                       extraRecord))
    {
        emit atProgressFinished(false);
        return;
    }

    LOG_INFO(QStringLiteral("三角化完成: 候选轨迹=%1 导出点数=%2 输出=%3")
                 .arg(result.candidateTrackCount)
                 .arg(result.exportedPointCount)
                 .arg(result.sparseCloudPath));

    emit tiePointResultReady(result.sparseCloudPath, sidecarPath);
    emit atProgressFinished(true);
    QMessageBox::information(_parentWidget,
                             QStringLiteral("生成两视预览云"),
                             QStringLiteral("两视预览云生成完成。\n候选轨迹: %1\n导出点数: %2\n输出文件: %3")
                                 .arg(result.candidateTrackCount)
                                 .arg(result.exportedPointCount)
                                 .arg(result.sparseCloudPath));
}

void ProjectSparseReconstructionManager::startSparsePointWorkflow(SparsePointWorkflowKind kind,
                                                                  const QJsonObject &settings)
{
    const SparsePointWorkflowSpec spec = sparsePointWorkflowSpec(kind);
    if (!xjw::gui::project::requireOpenProject(
            _projectData, _parentWidget, spec.projectOpenMessage, spec.title))
    {
        return;
    }

    SparsePointContext context;
    if (settings.value(QStringLiteral("sourceKind")).toString() == QLatin1String("external_ply"))
    {
        const QString path = QDir::cleanPath(
            settings.value(QStringLiteral("externalSparseCloudPath")).toString().trimmed());
        if (path.isEmpty() || !QFileInfo::exists(path))
        {
            QMessageBox::warning(_parentWidget,
                                 spec.title,
                                 QStringLiteral("外部 PLY 点云不存在: %1").arg(path));
            return;
        }
        context.sourceResultIndex = -1;
        context.sparseCloudPath = path;
    }
    else
    {
        const auto contextResult = resolveSparsePointContextResult(
            _owner->currentMeta(),
            settings.value(QStringLiteral("sourceAtIndex")).toInt(-1));
        if (!contextResult.status.ok)
        {
            QMessageBox::warning(_parentWidget, spec.title, contextResult.status.errorMessage);
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
                QMessageBox::warning(
                    _parentWidget,
                    spec.title,
                    QStringLiteral("稀疏点云文件不存在: %1").arg(sparseCloudOverride));
                return;
            }
            context.sparseCloudPath = sparseCloudOverride;
        }
        if (!sidecarOverride.isEmpty())
        {
            if (!QFileInfo(sidecarOverride).isFile())
            {
                QMessageBox::warning(
                    _parentWidget,
                    spec.title,
                    QStringLiteral("连接点逐点数据文件不存在: %1").arg(sidecarOverride));
                return;
            }
            context.sidecarPath = sidecarOverride;
        }
    }

    const auto session = _owner->currentSessionContext();
    const QString projectPath = session.projectPath;
    const QString assetsDir = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    const QString outputDir = QDir(assetsDir).filePath(
        QStringLiteral("aerial_triangulation/%1_%2")
            .arg(spec.outputDirPrefix,
                 QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss"))));

    emit atProgressChanged(spec.progressMessage, 20);

    QJsonObject operationSettings = settings;
    operationSettings.remove(QStringLiteral("sourceSparseCloudPath"));
    operationSettings.remove(QStringLiteral("sourceSidecarPath"));
    QPointer<ProjectManager> ownerGuard(_owner);
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [kind, context, operationSettings, outputDir]()
        {
            return runSparsePointWorkflowResult(
                kind, context, operationSettings, outputDir);
        },
        [spec, context, ownerGuard, session](
            ProjectSparseReconstructionManager *self,
            xjw::gui::tasks::TaskOutcome<SparsePointWorkflowResult> outcome)
        {
            if (!ownerGuard || !ownerGuard->isCurrentSession(session))
            {
                return;
            }

            if (!outcome.succeeded())
            {
                emit self->atProgressFinished(false);
                QMessageBox::warning(self->_parentWidget,
                                     spec.title,
                                     outcome.errorMessage);
                return;
            }

            const auto &workflowResult = *outcome.value;

            if (!workflowResult.status.ok)
            {
                emit self->atProgressFinished(false);
                QMessageBox::warning(self->_parentWidget,
                                     spec.title,
                                     workflowResult.status.errorMessage);
                return;
            }

            const SparsePointOperationResult &operationResult = workflowResult.operation;

            if (!self->_owner->replaceTiePointResult(operationResult.sparseCloudPath,
                                                      operationResult.outputCount,
                                                      context.selectedImages,
                                                      operationResult.outputDir,
                                                      operationResult.extraRecord))
            {
                emit self->atProgressFinished(false);
                return;
            }

            emit self->tiePointResultReady(operationResult.sparseCloudPath,
                                           operationResult.sidecarPath);
            emit self->atProgressFinished(true);
            QMessageBox::information(self->_parentWidget,
                                     spec.title,
                                     buildSparsePointWorkflowSuccessMessage(spec, operationResult));
        });
}
