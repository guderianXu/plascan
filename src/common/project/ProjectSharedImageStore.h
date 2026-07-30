#pragma once

#include <QString>

namespace xjw::common::project
{

class ProjectSharedImageStore
{
public:
    explicit ProjectSharedImageStore(const QString &projectPath);

    bool importImage(const QString &sourcePath,
                     QString *resourceUri,
                     QString *materializedPath = nullptr,
                     QString *errorMessage = nullptr) const;
    QString materialize(const QString &resourceUri,
                        QString *errorMessage = nullptr) const;
    bool pruneUnreferenced(QString *errorMessage = nullptr) const;

private:
    QString _projectPath;
};

} // namespace xjw::common::project
