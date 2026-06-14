#include "ProjectSparseReconstructionManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "ProjectIO.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "ProjectSparseWorkflow.h"
#include "ProjectTriangulationService.h"
#include "ProjectWorkflowUtils.h"
#include "project/SparseResultQuality.h"
#include "Logger.h"

#include <QDateTime>
#include <QDir>
#include <QJsonArray>
#include <QMessageBox>
#include <QMetaObject>
#include <QPointer>
#include <QtConcurrent/QtConcurrent>

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
using xjw::gui::project::summarizeAtResults;
using xjw::gui::project::writeJsonObjectFile;

ProjectSparseReconstructionManager::ProjectSparseReconstructionManager(ProjectManager *owner,
                                                                       ProjectData *projectData,
                                                                       QWidget *parentWidget,
                                                                       QObject *parent)
    : QObject(parent)
    , m_owner(owner)
    , m_projectData(projectData)
    , m_parentWidget(parentWidget)
{
}

QJsonArray ProjectSparseReconstructionManager::getAvailableAtResults() const
{
    if (!m_projectData || !m_projectData->hasProject())
    {
        return QJsonArray();
    }
    return summarizeAtResults(m_projectData->metadata());
}

bool ProjectSparseReconstructionManager::ensureProjectOpen(const QString &message,
                                                           const QString &title) const
{
    if (m_projectData && m_projectData->hasProject())
    {
        return true;
    }
    QMessageBox::warning(m_parentWidget, title, message);
    return false;
}

void ProjectSparseReconstructionManager::startTriangulationAsync(const QJsonObject &settings)
{
    if (!ensureProjectOpen(QStringLiteral("请先打开项目后再执行三角化"),
                           QStringLiteral("生成两视预览云")))
    {
        return;
    }

    const QStringList selectedImages = m_owner->getAllImages();
    if (selectedImages.size() < 2)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("生成两视预览云"),
                             QStringLiteral("至少需要两张影像才能执行三角化"));
        return;
    }

    QJsonObject mergedMeta = projectFilesMeta(m_projectData);
    const QJsonObject runtimeMeta = m_owner->currentMeta();
    for (auto it = runtimeMeta.begin(); it != runtimeMeta.end(); ++it)
    {
        if (it.key() != QLatin1String("images") && it.key() != QLatin1String("project_files"))
        {
            mergedMeta.insert(it.key(), it.value());
        }
    }

    const QString assetsDir = ProjectIO::projectAssetsDir(m_owner->currentProjectPath());
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

    xjw::gui::project::TriangulationServiceOptions options;
    options.outputDir = outputDir;
    options.minTriAngleDeg = settings.value(QStringLiteral("minAngle")).toDouble(2.0);
    options.maxReprojErrorPx = settings.value(QStringLiteral("reprojThreshold")).toDouble(2.0);
    options.minObservations = settings.value(QStringLiteral("minObservations")).toInt(2);
    options.ignoreTwoViewTracks = settings.value(QStringLiteral("ignoreTwoView")).toBool(false);
    options.minTrackLength = settings.value(QStringLiteral("minTrackLen")).toInt(2);

    emit atProgressChanged(QStringLiteral("正在构建两视预览云..."), 10);

    QPointer<ProjectSparseReconstructionManager> self(this);
    (void)QtConcurrent::run([self, mergedMeta, selectedImages, options, replaceIndex]() {
        if (!self)
        {
            return;
        }

        const xjw::gui::project::TriangulationServiceResult result =
            xjw::gui::project::ProjectTriangulationService::run(mergedMeta, selectedImages, options);

        QMetaObject::invokeMethod(self, [self, result, selectedImages, options, replaceIndex]() {
            if (!self)
            {
                return;
            }

            if (!result.success)
            {
                emit self->atProgressFinished(false);
                QMessageBox::warning(self->m_parentWidget,
                                     QStringLiteral("生成两视预览云"),
                                     result.errorMessage);
                return;
            }
            self->finalizeTriangulationSuccess(result, selectedImages, options, replaceIndex);
        }, Qt::QueuedConnection);
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
    const xjw::gui::project::TriangulationServiceResult &result,
    const QStringList &selectedImages,
    const xjw::gui::project::TriangulationServiceOptions &options,
    int replaceIndex)
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

    m_owner->appendAtResult(result.sparseCloudPath,
                            result.exportedPointCount,
                            selectedImages,
                            options.outputDir,
                            extraRecord,
                            replaceIndex);

    LOG_INFO(QStringLiteral("三角化完成: 候选轨迹=%1 导出点数=%2 输出=%3")
                 .arg(result.candidateTrackCount)
                 .arg(result.exportedPointCount)
                 .arg(result.sparseCloudPath));

    emit atProgressFinished(true);
    QMessageBox::information(m_parentWidget,
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
    if (!ensureProjectOpen(spec.projectOpenMessage, spec.title))
    {
        return;
    }

    const auto contextResult = resolveSparsePointContextResult(
        m_owner->currentMeta(),
        settings.value(QStringLiteral("sourceAtIndex")).toInt(-1));
    if (!contextResult.status.ok)
    {
        QMessageBox::warning(m_parentWidget, spec.title, contextResult.status.errorMessage);
        return;
    }
    const SparsePointContext context = contextResult.context;

    const QString assetsDir = ProjectIO::projectAssetsDir(m_owner->currentProjectPath());
    const QString outputDir = QDir(assetsDir).filePath(
        QStringLiteral("aerial_triangulation/%1_%2")
            .arg(spec.outputDirPrefix,
                 QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss"))));

    emit atProgressChanged(spec.progressMessage, 20);

    QPointer<ProjectSparseReconstructionManager> self(this);
    (void)QtConcurrent::run([self, kind, spec, context, settings, outputDir]() {
        if (!self)
        {
            return;
        }

        const SparsePointWorkflowResult workflowResult = runSparsePointWorkflowResult(kind,
                                                                                       context,
                                                                                       settings,
                                                                                       outputDir);

        QMetaObject::invokeMethod(self, [self, spec, context, workflowResult]() {
            if (!self)
            {
                return;
            }

            if (!workflowResult.status.ok)
            {
                emit self->atProgressFinished(false);
                QMessageBox::warning(self->m_parentWidget,
                                     spec.title,
                                     workflowResult.status.errorMessage);
                return;
            }

            const SparsePointOperationResult &operationResult = workflowResult.operation;

            self->m_owner->appendAtResult(operationResult.sparseCloudPath,
                                          operationResult.outputCount,
                                          context.selectedImages,
                                          operationResult.outputDir,
                                          operationResult.extraRecord);

            emit self->atProgressFinished(true);
            QMessageBox::information(self->m_parentWidget,
                                     spec.title,
                                     buildSparsePointWorkflowSuccessMessage(spec, operationResult));
        }, Qt::QueuedConnection);
    });
}
