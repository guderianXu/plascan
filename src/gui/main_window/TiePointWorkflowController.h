#pragma once

#include <QObject>
#include <QFuture>
#include <QPointer>
#include <QStringList>

#include <atomic>
#include <memory>

class ProjectManager;
class QTimer;

namespace xjw::matchphotos
{
struct MatchPhotosOptions;
}

class TiePointWorkflowController : public QObject
{
    Q_OBJECT
public:
    explicit TiePointWorkflowController(ProjectManager *projectManager, QObject *parent = nullptr);

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

private:
    void finishRun(bool success);

    QPointer<ProjectManager> _projectManager;
    std::shared_ptr<std::atomic_bool> _cancelFlag;
    QPointer<QTimer> _progressTimer;
    std::shared_ptr<std::atomic_int> _progressCount;
    QFuture<void> _taskFuture;
};
