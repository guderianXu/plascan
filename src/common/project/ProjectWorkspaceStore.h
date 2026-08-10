#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::common::project
{
class ProjectResourceIndex;
}

class ProjectWorkspaceStore
{
public:
    explicit ProjectWorkspaceStore(const QString &projectPath,
                                   int chunkDirectory = 0);

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
    // Recovery preflight resolves project:// paths without validating the
    // resource index, because staged artifacts may intentionally be absent.
    bool materializeMetadataForRecovery(
        QJsonObject *metadata,
        QString *errorMessage = nullptr) const;

    bool stagePackedResource(const QString &sourcePath,
                             QString *stagedPath,
                             QString *errorMessage = nullptr) const;

    QString runtimeRoot(QString *errorMessage = nullptr) const;

private:
    bool ensureProjectManifest(QString *errorMessage) const;
    bool loadResourceIndex(
        xjw::common::project::ProjectResourceIndex *index,
        QString *errorMessage) const;

    QString _projectPath;
    int _chunkDirectory = 0;
};
