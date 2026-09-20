#pragma once

#include "project/support/ProjectSessionContext.h"

#include <QFuture>
#include <QObject>
#include <QString>

#include <functional>

class ProjectData;
class ProjectUiMessageAdapter;
class QWidget;

namespace xjw::gui::project
{
    class ProjectResourceCleanupCoordinator;
    class ProjectSession;
} // namespace xjw::gui::project

class ProjectLifecycleService final : public QObject
{
    Q_OBJECT

public:
    ProjectLifecycleService(ProjectData* projectData,
                            xjw::gui::project::ProjectSession* session,
                            ProjectUiMessageAdapter* messages,
                            xjw::gui::project::ProjectResourceCleanupCoordinator* cleanup,
                            QWidget* parentWidget,
                            QObject* parent = nullptr);
    ~ProjectLifecycleService() override;

    void setDirectoryAccessors(std::function<QString(const QString& key)> getLastDir,
                               std::function<void(const QString& key, const QString& dir)> saveLastDir);
    void setSessionDrainHandler(std::function<bool(std::function<void()>)> handler);
    void setSessionFutureTracker(std::function<void(QFuture<void>)> tracker);

    bool isOpenInProgress() const;
    bool isPortableExportInProgress() const;
    bool isBusy() const;
    xjw::gui::project::ProjectSession* session() const noexcept
    {
        return _session;
    }
    bool startPortableExport(const QString& outputPath, QString* errorMessage = nullptr);

public slots:
    void createNewProject();
    void openProject();
    void openProjectFromPath(const QString& projectPath);
    void saveProject();
    void exportPortableProject();
    void closeProject();
    void createChunk();
    void renameChunk(const QString& chunkId);
    void removeChunk(const QString& chunkId);
    void switchChunk(const QString& chunkId);

signals:
    void projectCreated(const QString& projectPath);
    void projectOpened(ProjectData* projectData);
    void projectClosed();
    void projectModified();
    void lifecycleError(const QString& operation, const QString& message);

    void projectOpenStarted(const QString& projectPath);
    void projectOpenProgressChanged(const QString& message, int percent);
    void projectOpenFinished(bool success, const QString& message);
    void saveStarted();
    void saveFinished(bool success);
    void portableExportFinished(bool success, const QString& outputPath, const QString& message);

private:
    QString readLastDir(const QString& key) const;
    void writeLastDir(const QString& key, const QString& directory) const;
    bool rejectLifecycleChange(const QString& operation) const;
    void reportError(const QString& operation, const QString& message);
    void loadProjectResultsAsync(const QString& projectPath, const xjw::gui::project::ProjectSessionContext& context);
    void showOpenError(const QString& message);
    void showOpenCancelled(const QString& message);
    bool isCurrent(const xjw::gui::project::ProjectSessionContext& context) const;
    bool beginOperation(const QString& operation);
    void endOperation(const QString& operation);
    bool continueAfterSessionDrain(const QString& operation,
                                   std::function<void(ProjectLifecycleService*)> continuation);

    ProjectData* _projectData = nullptr;
    xjw::gui::project::ProjectSession* _session = nullptr;
    ProjectUiMessageAdapter* _messages = nullptr;
    xjw::gui::project::ProjectResourceCleanupCoordinator* _cleanup = nullptr;
    QWidget* _parentWidget = nullptr;
    std::function<QString(const QString& key)> _getLastDir;
    std::function<void(const QString& key, const QString& dir)> _saveLastDir;
    std::function<bool(std::function<void()>)> _sessionDrainHandler;
    std::function<void(QFuture<void>)> _sessionFutureTracker;
    QString _pendingDrainOperation;
    bool _openInProgress = false;
    bool _saveInProgress = false;
    bool _portableExportInProgress = false;
    xjw::gui::project::ProjectSessionContext _openSessionContext;
    xjw::gui::project::ProjectSessionContext _saveSessionContext;
    xjw::gui::project::ProjectSessionContext _portableExportSessionContext;
};

namespace xjw::gui::project
{
    using ProjectLifecycleService = ::ProjectLifecycleService;
}
