#include "project/ProjectLock.h"

#include "project/ProjectPackageLayout.h"

#include <QDir>
#include <QLockFile>

namespace xjw::common::project
{
namespace
{

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

} // namespace

ProjectLock::ProjectLock() = default;

ProjectLock::~ProjectLock()
{
    release();
}

bool ProjectLock::acquire(const QString &projectPath,
                          QString *errorMessage)
{
    release();
    const QString dataDirectory =
        ProjectPackageLayout::dataDirectory(projectPath);
    if (!QDir().mkpath(dataDirectory))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建工程锁目录: %1")
                     .arg(dataDirectory));
        return false;
    }

    auto lock = std::make_unique<QLockFile>(
        QDir(dataDirectory).filePath(QStringLiteral(".plascan.lock")));
    lock->setStaleLockTime(30000);
    if (!lock->tryLock(0))
    {
        qint64 processId = 0;
        QString hostName;
        QString applicationName;
        lock->getLockInfo(&processId, &hostName, &applicationName);
        setError(
            errorMessage,
            QStringLiteral("工程正在被其他进程使用（PID %1，%2，%3）")
                .arg(processId)
                .arg(hostName, applicationName));
        return false;
    }
    _lockFile = std::move(lock);
    return true;
}

void ProjectLock::release()
{
    if (_lockFile)
    {
        _lockFile->unlock();
        _lockFile.reset();
    }
}

bool ProjectLock::isLocked() const
{
    return _lockFile && _lockFile->isLocked();
}

} // namespace xjw::common::project
