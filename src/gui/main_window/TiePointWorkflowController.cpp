#include "TiePointWorkflowController.h"

#include "GuiTaskRunner.h"
#include "Logger.h"
#include "MatchPhotosTask.h"
#include "project/services/ProjectSession.h"
#include "ProjectResultRecords.h"
#include "project/ProjectIO.h"

#include <QDir>
#include <QJsonObject>
#include <QPointer>
#include <QTimer>
#include <QVector>

#include <algorithm>

TiePointWorkflowController::TiePointWorkflowController(xjw::gui::project::ProjectSession *session, QObject *parent)
    : QObject(parent)
    , _session(session)
{
    if (_session)
    {
        connect(_session, &xjw::gui::project::ProjectSession::sessionChanged,
                this, &TiePointWorkflowController::cancel);
    }
}

bool TiePointWorkflowController::isRunning() const
{
    return _cancelFlag != nullptr;
}

void TiePointWorkflowController::cancel()
{
    if (_cancelFlag)
    {
        _cancelFlag->store(true);
    }
}

void TiePointWorkflowController::start(xjw::matchphotos::MatchPhotosOptions options,
                                       const QStringList &manualPairKeys,
                                       const QString &taskTitle)
{
    if (isRunning())
    {
        emit warningRequested(tr("连接点匹配"), tr("已有连接点匹配任务正在运行。"));
        return;
    }

    QPointer<xjw::gui::project::ProjectSession> session_guard = _session;
    if (!session_guard)
    {
        emit warningRequested(tr("连接点匹配"), tr("项目管理器未初始化。"));
        return;
    }

    const auto session = session_guard->context();
    const QString projectPath = session.projectPath;
    const QStringList images = session_guard->allImages();
    if (images.size() < 2)
    {
        emit warningRequested(tr("创建连接点"), tr("当前项目至少需要两张照片才能创建连接点。"));
        return;
    }
    if (options.pairPolicy.mode == xjw::matchphotos::PairSelectionMode::ManualOnly
        && manualPairKeys.isEmpty())
    {
        emit warningRequested(tr("连接点匹配"), tr("没有可用的影像对，请先生成或导入匹配对。"));
        return;
    }

    options.planOnly = false;
    xjw::matchphotos::MatchPhotosContext context;
    context.projectPath = projectPath;
    context.workingDirectory = xjw::common::project::ProjectIO::projectAssetsDir(projectPath);
    context.matchDirectory = xjw::common::project::ProjectIO::imageMatchOutputDir(projectPath);
    context.pairInput.images = images;
    context.pairInput.manualPairKeys = manualPairKeys;
    context.maskPaths = xjw::common::project::ProjectIO::maskPathsForImages(projectPath, images);
    bool allImageIdsResolved = false;
    context.imageIds = session_guard->getImageIdsForImages(images, &allImageIdsResolved);
    if (!allImageIdsResolved)
    {
        emit warningRequested(tr("连接点匹配"), tr("当前影像缺少稳定 ImageId，无法建立参考几何。"));
        return;
    }
    if (options.useReferencePreselection ||
        options.guidedMatchingMode != xjw::matchphotos::GuidedMatchingMode::Disabled)
    {
        bool hasAllReferenceCameras = false;
        context.referenceCameraGeometries = session_guard->getReferenceCameraGeometriesForImages(
            images, &hasAllReferenceCameras);
        if (!hasAllReferenceCameras)
        {
            LOG_INFO(QStringLiteral(
                "参考预选已请求，但项目相机参考不完整；PlaMatch 将使用已有坐标或索引邻域回退"));
        }
    }

    _cancelFlag = std::make_shared<std::atomic_bool>(false);
    _progressCount = std::make_shared<std::atomic_int>(0);
    context.cancelFlag = _cancelFlag.get();
    context.progressCount = _progressCount.get();

    const int imageCount = images.size();
    const int allPairCount = imageCount * (imageCount - 1) / 2;
    const int estimatedPairCount = manualPairKeys.isEmpty()
        ? std::min(allPairCount, imageCount * std::max(1, options.pairPolicy.sequenceWindow))
        : manualPairKeys.size();
    emit progressStarted(std::max(1, imageCount + estimatedPairCount));

    _progressTimer = new QTimer(this);
    _progressTimer->setInterval(100);
    const std::shared_ptr<std::atomic_int> progressCount = _progressCount;
    connect(_progressTimer, &QTimer::timeout, this,
            [this, progressCount, session_guard, session]()
    {
        if (!session_guard || !session_guard->isCurrent(session))
        {
            cancel();
            if (_progressTimer)
            {
                _progressTimer->stop();
            }
            return;
        }
        emit progressUpdated(progressCount->load());
    });
    _progressTimer->start();

    const std::shared_ptr<std::atomic_bool> cancelFlag = _cancelFlag;
    _taskFuture = xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [context, options, cancelFlag = _cancelFlag, progressCount = _progressCount]() mutable
        {
            Q_UNUSED(cancelFlag)
            Q_UNUSED(progressCount)
            const xjw::matchphotos::MatchPhotosTask task(options);
            return task.run(context);
        },
        [this, session_guard, session, taskTitle, cancelFlag](TiePointWorkflowController *,
                                                               xjw::gui::tasks::TaskOutcome<
                                                                   xjw::matchphotos::MatchPhotosResult> outcome)
        {
            if (!outcome.succeeded())
            {
                finishRun(false);
                const QString message = outcome.errorMessage.isEmpty()
                    ? tr("%1失败").arg(taskTitle)
                    : outcome.errorMessage;
                LOG_ERROR("%s", qUtf8Printable(message));
                emit warningRequested(tr("连接点匹配"), message);
                return;
            }

            xjw::matchphotos::MatchPhotosResult result = std::move(*outcome.value);
            const bool cancelled = cancelFlag && cancelFlag->load();
            finishRun(result.success && !cancelled);

            if (!session_guard)
            {
                return;
            }
            if (!session_guard->isCurrent(session))
            {
                emit warningRequested(tr("连接点匹配"), tr("项目已切换，本次连接点匹配结果未写回。"));
                return;
            }

            QString write_error;
            if (!session_guard->appendImageMatchResults(
                    session, xjw::gui::project::makeImageMatchResultRecords(result), &write_error))
            {
                emit warningRequested(tr("连接点匹配"),
                                      write_error.isEmpty() ? tr("匹配结果写回失败。") : write_error);
                return;
            }

            for (const xjw::matchphotos::MatchPhotosMatchRecord &match : result.matches)
            {
                emit matchPairReady(match.image0Path,
                                    match.image1Path,
                                    match.image0MatchFilePath,
                                    match.matchCount);
            }

            if (result.success && !cancelled)
            {
                const QString message = tr("%1完成：%2 个影像分片，%3 对匹配")
                    .arg(taskTitle)
                    .arg(static_cast<int>(result.imageMatchFiles.size()))
                    .arg(static_cast<int>(result.matches.size()));
                LOG_INFO("%s", qUtf8Printable(message));
                emit statusMessageRequested(message, 5000);
            }
            else
            {
                const QString message = result.errorMessage.isEmpty()
                    ? tr("%1失败或被取消").arg(taskTitle)
                    : result.errorMessage;
                LOG_ERROR("%s", qUtf8Printable(message));
                emit warningRequested(tr("连接点匹配"), message);
            }
        });
}

void TiePointWorkflowController::finishRun(bool success)
{
    if (_progressTimer)
    {
        _progressTimer->stop();
        _progressTimer->deleteLater();
        _progressTimer = nullptr;
    }
    _progressCount.reset();
    _cancelFlag.reset();
    _taskFuture = QFuture<void>();
    emit progressFinished(success);
}
