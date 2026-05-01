#pragma once

#include <QString>

#include <functional>

class QWidget;
class ProjectData;

class ProjectUiCommands
{
public:
    ProjectUiCommands(ProjectData *projectData, QWidget *parentWidget);

    void setDirectoryAccessors(std::function<QString(const QString &key)> getLastDir,
                               std::function<void(const QString &key, const QString &dir)> saveLastDir);

    bool createNewProject(QString *createdPath = nullptr) const;
    bool openProjectByDialog(QString *openedPath = nullptr) const;
    bool openProjectFromPath(const QString &plascanPath) const;
    bool saveProject() const;
    void closeProject() const;
    bool addPhoto() const;
    bool addFolder() const;

private:
    QString readLastDir(const QString &key) const;
    void writeLastDir(const QString &key, const QString &dir) const;
    bool ensureProjectOpen(const QString &message = QStringLiteral("请先打开或创建项目"),
                           const QString &title = QStringLiteral("提示")) const;

    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    std::function<QString(const QString &key)> _getLastDir;
    std::function<void(const QString &key, const QString &dir)> _saveLastDir;
};
