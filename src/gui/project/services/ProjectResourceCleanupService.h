#pragma once

#include <QString>
#include <QStringList>

class ProjectData;

namespace xjw::gui::project
{

struct ResourceCleanupResult
{
    bool success = false;
    bool unsupportedSection = false;
    bool noMatchedRecords = false;
    int removedCount = 0;
    QString sectionArrayKey;
    QString errorMessage;
    QStringList failedPaths;
};

class ProjectResourceCleanupService
{
public:
    static ResourceCleanupResult cleanupGeneratedData(ProjectData *projectData,
                                                      const QString &section,
                                                      const QStringList &resourcePaths);
};

} // namespace xjw::gui::project
