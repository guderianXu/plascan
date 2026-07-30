#pragma once

#include "project/PlascanArchive.h"
#include "project/PortableProjectFormat.h"

#include <QString>

struct ProjectResourceImportOptions
{
    QString kind;
    QString resourceId;
    QString displayName;
    QString mediaType;
    PlascanArchiveCompression compression = PlascanArchiveCompression::Store;
};

class ProjectResourceStore
{
public:
    explicit ProjectResourceStore(const QString &projectPath);

    bool loadIndex(xjw::common::project::ProjectResourceIndex *index,
                   QString *errorMessage = nullptr) const;
    bool saveIndex(const xjw::common::project::ProjectResourceIndex &index,
                   QString *errorMessage = nullptr) const;

    bool importFile(
        const QString &sourcePath,
        const ProjectResourceImportOptions &options,
        xjw::common::project::ProjectResourceRef *resource,
        QString *errorMessage = nullptr);

    QString projectPath() const;

private:
    QString _projectPath;
};

class ProjectResourceResolver
{
public:
    explicit ProjectResourceResolver(const QString &projectPath);

    bool materialize(const xjw::common::project::ProjectResourceRef &resource,
                     QString *materializedPath,
                     QString *errorMessage = nullptr,
                     const QString &cacheRoot = QString()) const;

    bool materialize(const QString &resourceId,
                     QString *materializedPath,
                     QString *errorMessage = nullptr,
                     const QString &cacheRoot = QString()) const;

    QString defaultCacheRoot(QString *errorMessage = nullptr) const;

private:
    QString projectId(QString *errorMessage = nullptr) const;

    QString _projectPath;
};
