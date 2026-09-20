#pragma once

#include <QObject>
#include <QFuture>
#include <QPointer>
#include <QStringList>

#include <atomic>
#include <memory>

class QTimer;

namespace xjw::gui::project
{
class ProjectSession;
}

namespace xjw::matchphotos
{
struct MatchPhotosOptions;
}

class TiePointWorkflowController : public QObject
{
    Q_OBJECT
public:
    explicit TiePointWorkflowController(xjw::gui::project::ProjectSession *session, QObject *parent = nullptr);

    void start(xjw::matchphotos::MatchPhotosOptions options,
               const QStringList &manualPairKeys,
               const QString &taskTitle);
    void cancel();
    bool isRunning() const;

signals:
    void progressStarted(int total);
    void progressUpdated(int completed);
    void progressFinished(bool success);
    void statusMessageRequested(const QString &message, int timeoutMs);
    void warningRequested(const QString &title, const QString &message);
    void matchPairReady(const QString &image0,
                        const QString &image1,
                        const QString &matchFilePath,
                        int matchCount);

private:
    void finishRun(bool success);

    QPointer<xjw::gui::project::ProjectSession> _session;
    std::shared_ptr<std::atomic_bool> _cancelFlag;
    QPointer<QTimer> _progressTimer;
    std::shared_ptr<std::atomic_int> _progressCount;
    QFuture<void> _taskFuture;
};
