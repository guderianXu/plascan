#pragma once

#include <QJsonObject>
#include <QString>

class ProjectWorkspaceStore
{
public:
    explicit ProjectWorkspaceStore(const QString &projectPath);

    bool initializeRuntime(QString *runtimeRoot = nullptr,
                           QString *errorMessage = nullptr) const;
    void releaseRuntime() const;

    bool prepareSplitMetadata(QJsonObject *core,
                              QJsonObject *results,
                              QJsonObject *resourceIndex,
                              QString *errorMessage = nullptr) const;
    bool validateProjectLayout(QString *errorMessage = nullptr) const;
    bool materializeMetadata(QJsonObject *metadata,
                             QString *errorMessage = nullptr) const;

    QString runtimeRoot(QString *errorMessage = nullptr) const;

private:
    bool ensureProjectManifest(QString *errorMessage) const;

    QString _projectPath;
};
