#pragma once

#include <QObject>

class ProjectData;
class ProjectUiCommands;
class QWidget;

class ProjectLifecycleController final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectLifecycleController(ProjectData *projectData,
                                        ProjectUiCommands *uiCommands,
                                        QWidget *parentWidget,
                                        QObject *parent = nullptr);

public slots:
    void createNewProject();
    void openProject();
    void openProjectFromPath(const QString &projectPath);
    void saveProject();
    void closeProject();

signals:
    void projectCreated(const QString &projectPath);
    void projectOpenStarted(const QString &projectPath);
    void projectOpenProgressChanged(const QString &message, int percent);
    void projectOpenFinished(bool success, const QString &message);
    void saveStarted();
    void saveFinished(bool success);

private:
    void loadProjectResultsAsync(const QString &projectPath);
    void showOpenError(const QString &message);

    ProjectData *_projectData = nullptr;
    ProjectUiCommands *_uiCommands = nullptr;
    QWidget *_parentWidget = nullptr;
    bool _openInProgress = false;
};
