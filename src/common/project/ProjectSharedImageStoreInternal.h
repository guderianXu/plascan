#pragma once

#include <QHash>
#include <QLockFile>
#include <QMutex>
#include <QString>

#include <memory>

namespace xjw::common::project::detail
{

struct SharedImageSynchronization
{
    QMutex mutex;
    QHash<QString, int> activeReservations;
    std::unique_ptr<QLockFile> crossProcessLeaseLock;
};

std::shared_ptr<SharedImageSynchronization> synchronizationForProject(
    const QString &projectPath);
QString entryForReference(const QString &projectPath,
                          const QString &reference);
QString sharedImageLockPath(const QString &projectPath);
bool ensureCrossProcessLeaseLock(
    const QString &projectPath,
    const std::shared_ptr<SharedImageSynchronization> &synchronization,
    QString *errorMessage);
void releaseCrossProcessLeaseLockIfUnused(
    const std::shared_ptr<SharedImageSynchronization> &synchronization);
bool pruneSharedImages(
    const QString &projectPath,
    const std::shared_ptr<SharedImageSynchronization> &synchronization,
    QString *errorMessage);

} // namespace xjw::common::project::detail
