#include "GpuDeviceLease.h"

#include <QCryptographicHash>
#include <QDir>
#include <QLockFile>
#include <QSet>
#include <QString>

#include <algorithm>
#include <cctype>

namespace xjw
{
namespace mvs
{
namespace
{

QString lockPathForIdentity(const std::string &identity)
{
    const QByteArray digest = QCryptographicHash::hash(
        QByteArray::fromStdString(identity), QCryptographicHash::Sha256).toHex();
    const QString directory = QDir(QDir::tempPath()).filePath(QStringLiteral("plascan-gpu-locks"));
    QDir().mkpath(directory);
    return QDir(directory).filePath(QString::fromLatin1(digest) + QStringLiteral(".lock"));
}

} // namespace

GpuDeviceLeaseSet::~GpuDeviceLeaseSet() = default;

bool GpuDeviceLeaseSet::acquire(const std::vector<GpuDeviceDescriptor> &devices,
                                QString *errorMessage)
{
    _locks.clear();
    std::vector<GpuDeviceDescriptor> unique_devices;
    QSet<QString> identities;
    for (const GpuDeviceDescriptor &device : devices)
    {
        const QString identity = QString::fromStdString(device.physicalIdentity).trimmed();
        if (identity.isEmpty() || identities.contains(identity))
        {
            continue;
        }
        identities.insert(identity);
        unique_devices.push_back(device);
    }
    std::sort(unique_devices.begin(), unique_devices.end(),
              [](const GpuDeviceDescriptor &left, const GpuDeviceDescriptor &right)
              {
                  return left.physicalIdentity < right.physicalIdentity;
              });

    for (const GpuDeviceDescriptor &device : unique_devices)
    {
        auto lock = std::make_unique<QLockFile>(lockPathForIdentity(device.physicalIdentity));
        if (!lock->tryLock(0))
        {
            qint64 owner_pid = 0;
            QString owner_host;
            QString owner_application;
            lock->getLockInfo(&owner_pid, &owner_host, &owner_application);
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "GPU %1 正被另一进程占用（PID=%2，程序=%3，主机=%4）。"
                    "为避免深度估计竞争，本次任务未启动。")
                                    .arg(QString::fromStdString(device.displayName))
                                    .arg(owner_pid)
                                    .arg(owner_application.isEmpty()
                                             ? QStringLiteral("未知") : owner_application)
                                    .arg(owner_host.isEmpty()
                                             ? QStringLiteral("本机") : owner_host);
            }
            _locks.clear();
            return false;
        }
        _locks.push_back(std::move(lock));
    }
    return true;
}

bool GpuDeviceLeaseSet::empty() const noexcept
{
    return _locks.empty();
}

std::string fallbackGpuPhysicalIdentity(const std::string &vendor,
                                        const std::string &name,
                                        int deviceIndex)
{
    std::string normalized;
    normalized.reserve(vendor.size() + name.size());
    for (const char value : vendor + " " + name)
    {
        const unsigned char character = static_cast<unsigned char>(value);
        if (std::isalnum(character))
        {
            normalized.push_back(static_cast<char>(std::tolower(character)));
        }
    }
    return "name:" + normalized + ":" + std::to_string(std::max(0, deviceIndex));
}

} // namespace mvs
} // namespace xjw
