#pragma once

#include "model/AerialTriangulationOptions.h"
#include "model/AerialTriangulationResult.h"
#include "project/tasks/ProjectTaskContext.h"

#include <QFuture>
#include <QJsonObject>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QStringList>
#include <QVector>

#include <functional>

class ProjectTaskOrchestratorCameraTestPeer;
class ProjectUiMessageAdapter;

namespace xjw::gui::project
{
    class ProjectSession;
    class ProjectTaskOrchestrator;
} // namespace xjw::gui::project

class ProjectCameraSetupManager : public QObject
{
    Q_OBJECT

public:
    ~ProjectCameraSetupManager() override;

signals:
    void atProgressChanged(const QString& stage, int percent);
    void atProgressFinished(bool success);
    void matchPairReady(const QString& img0, const QString& img1, const QString& matchFilePath, int numMatches);
    void imageMatchResultAppended(const QString& imagePath);

private:
    friend class xjw::gui::project::ProjectTaskOrchestrator;
    friend class ::ProjectTaskOrchestratorCameraTestPeer;

    using SfmRunner = std::function<xjw::aerial_triangulation::AerialTriangulationResult(
        xjw::aerial_triangulation::AerialTriangulationOptions)>;
    using HeavyWorkEnteredObserver = std::function<void(const QString& operation)>;

    explicit ProjectCameraSetupManager(xjw::gui::project::ProjectSession* session,
                                       ProjectUiMessageAdapter* messages,
                                       QObject* parent = nullptr);

    bool importCameraForImage(const QString& imagePath, const xjw::gui::project::ProjectTaskContext& taskContext);
    bool importCameraProject(const xjw::gui::project::ProjectTaskContext& taskContext);
    bool initializeCamerasFromExifOrDefault(const QJsonObject& settings,
                                            const xjw::gui::project::ProjectTaskContext& taskContext);
    bool initializeCamerasFromIntrinsics(const QJsonObject& settings,
                                         const xjw::gui::project::ProjectTaskContext& taskContext);
    bool initializeCameraPosesWithSFM(const QJsonObject& settings,
                                      const xjw::gui::project::ProjectTaskContext& taskContext);
    void cancelActiveTask(const xjw::gui::project::ProjectTaskContext& taskContext);
    void waitForActiveTask();
    bool isRunning() const noexcept;
    bool hasPendingWork() const noexcept;
    void setDirectoryAccessors(std::function<QString(const QString& key)> getLastDir,
                               std::function<void(const QString& key, const QString& dir)> saveLastDir);
    void setSfmRunnerForTesting(SfmRunner runner);
    void setHeavyWorkEnteredObserverForTesting(HeavyWorkEnteredObserver observer);
    void trackFutureForTesting(QFuture<void> future);

    bool requireProject(const QString& message) const;
    bool contextMatches(const xjw::gui::project::ProjectTaskContext& taskContext,
                        bool requireCurrent = true,
                        bool allowCancelled = false) const;
    void completeTask(const xjw::gui::project::ProjectTaskContext& taskContext, bool success);
    QString readLastDir(const QString& key) const;
    void writeLastDir(const QString& key, const QString& dir) const;
    void pruneFinishedFutures();
    void trackFuture(QFuture<void> future);
    bool startCameraInitialization(const QJsonObject& settings,
                                   const xjw::gui::project::ProjectTaskContext& taskContext,
                                   bool useExif);
    std::function<void(int, int)> makeProgressReporter(const xjw::gui::project::ProjectTaskContext& taskContext,
                                                       const QString& stage,
                                                       int firstPercent,
                                                       int lastPercent) const;

    QPointer<xjw::gui::project::ProjectSession> _session;
    ProjectUiMessageAdapter* _messages = nullptr;
    std::function<QString(const QString& key)> _getLastDir;
    std::function<void(const QString& key, const QString& dir)> _saveLastDir;
    SfmRunner _sfmRunner;
    HeavyWorkEnteredObserver _heavyWorkEnteredObserver;
    QVector<QFuture<void>> _futures;
    xjw::gui::project::ProjectTaskContext _taskContext;
};
