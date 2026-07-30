#pragma once

#include <QObject>
#include <QJsonObject>
#include <QPointer>
#include <QString>

#include <atomic>
#include <functional>
#include <memory>

class ProjectData;
class ProjectManager;

class ProjectDenseReconstructionManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectDenseReconstructionManager(ProjectManager *owner,
                                               ProjectData *projectData,
                                               QObject *parent = nullptr);

    // 由上层控制器提供 UI 决策；执行器本身不依赖 QWidget 或 QMessageBox。
    void setExistingDepthActionRequester(std::function<int(int, int, const QString &)> requester);

    bool startEstimateDepthMapsAsync(const QJsonObject &settings);
    bool startFuseDepthMapsAsync(const QJsonObject &settings);
    bool startGenerateDenseCloudAsync(const QJsonObject &settings);
    void startDenseCloudRefineAsync(const QJsonObject &settings);
    void cancelMvs();
    bool isMvsRunning() const;

signals:
    void mvsProgressChanged(const QString &stage, int percent);
    void mvsProgressFinished(bool success);
    void depthMapBatchReady(const QString &outputDirectory, int frameCount);
    void denseCloudResultReady(const QString &denseCloudPath, int pointCount);
    void userMessageRequested(bool informational, const QString &title, const QString &message);

private:
    bool ensureProjectOpen(const QString &message,
                           const QString &title);
    void showWarning(const QString &title, const QString &message);
    void showInformation(const QString &title, const QString &message);
    std::shared_ptr<std::atomic_bool> createActiveMvsCancelFlag();
    void clearActiveMvsCancelFlag(const std::shared_ptr<std::atomic_bool> &cancelFlag);

    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    std::function<int(int, int, const QString &)> _existingDepthActionRequester;
    QPointer<QObject> _activeMvsGenerator;
    std::shared_ptr<std::atomic_bool> _activeMvsCancelFlag;
    bool _mvsTransitionPending = false;
};
