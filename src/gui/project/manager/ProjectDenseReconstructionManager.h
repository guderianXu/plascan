#pragma once

#include <QObject>
#include <QJsonObject>
#include <QPointer>

#include <atomic>
#include <memory>

class QWidget;
class ProjectData;
class ProjectManager;

class ProjectDenseReconstructionManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectDenseReconstructionManager(ProjectManager *owner,
                                               ProjectData *projectData,
                                               QWidget *parentWidget,
                                               QObject *parent = nullptr);

    void startEstimateDepthMapsAsync(const QJsonObject &settings);
    void startFuseDepthMapsAsync(const QJsonObject &settings);
    void startGenerateDenseCloudAsync(const QJsonObject &settings);
    void startDenseCloudRefineAsync(const QJsonObject &settings);
    void cancelMvs();

signals:
    void mvsProgressChanged(const QString &stage, int percent);
    void mvsProgressFinished(bool success);

private:
    bool ensureProjectOpen(const QString &message,
                           const QString &title) const;
    std::shared_ptr<std::atomic_bool> createActiveMvsCancelFlag();
    void clearActiveMvsCancelFlag(const std::shared_ptr<std::atomic_bool> &cancelFlag);

    ProjectManager *m_owner = nullptr;
    ProjectData *m_projectData = nullptr;
    QWidget *m_parentWidget = nullptr;
    QPointer<QObject> m_activeMvsGenerator;
    std::shared_ptr<std::atomic_bool> m_activeMvsCancelFlag;
};
