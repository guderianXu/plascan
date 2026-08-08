#include "PointCloudSnapshotIO.h"

#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryFile>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

#include <plapoint/io/obj_io.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/io/xyz_io.h>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace xjw::gui::point_cloud
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

void clearError(QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
}

QString normalizedFinalPath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

bool replaceFileAtomically(const QString &sourcePath,
                           const QString &targetPath,
                           QString *errorMessage)
{
#ifdef Q_OS_WIN
    const std::wstring source_path = sourcePath.toStdWString();
    const std::wstring target_path = targetPath.toStdWString();
    if (MoveFileExW(source_path.c_str(),
                    target_path.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    {
        clearError(errorMessage);
        return true;
    }

    setError(errorMessage,
             QStringLiteral("无法原子替换点云文件：%1（Windows 错误 %2）")
                 .arg(targetPath)
                 .arg(GetLastError()));
#else
    const QByteArray source_path = QFile::encodeName(sourcePath);
    const QByteArray target_path = QFile::encodeName(targetPath);
    if (std::rename(source_path.constData(), target_path.constData()) == 0)
    {
        clearError(errorMessage);
        return true;
    }

    const int error_number = errno;
    setError(errorMessage,
             QStringLiteral("无法原子替换点云文件：%1（%2）")
                 .arg(targetPath, QString::fromLocal8Bit(std::strerror(error_number))));
#endif
    return false;
}

} // namespace

struct PointCloudSnapshotGuard::State
{
    enum class Status
    {
        Pending,
        Committed,
        Discarded
    };

    QString finalPath;
    QString temporaryPath;
    mutable std::mutex mutex;
    Status status = Status::Pending;
};

PointCloudSnapshotGuard::PointCloudSnapshotGuard(QString finalPath,
                                                 QString temporaryPath)
    : _state(std::make_unique<State>())
{
    _state->finalPath = std::move(finalPath);
    _state->temporaryPath = std::move(temporaryPath);
}

PointCloudSnapshotGuard::~PointCloudSnapshotGuard()
{
    if (!_state)
    {
        return;
    }

    std::lock_guard lock(_state->mutex);
    if (_state->status == State::Status::Pending)
    {
        QFile::remove(_state->temporaryPath);
        _state->status = State::Status::Discarded;
    }
}

QString PointCloudSnapshotGuard::finalPath() const
{
    std::lock_guard lock(_state->mutex);
    return _state->finalPath;
}

QString PointCloudSnapshotGuard::temporaryPath() const
{
    std::lock_guard lock(_state->mutex);
    return _state->temporaryPath;
}

bool PointCloudSnapshotGuard::isPending() const
{
    std::lock_guard lock(_state->mutex);
    return _state->status == State::Status::Pending;
}

bool PointCloudSnapshotGuard::commit(QString *errorMessage)
{
    std::lock_guard lock(_state->mutex);
    if (_state->status == State::Status::Committed)
    {
        clearError(errorMessage);
        return true;
    }
    if (_state->status == State::Status::Discarded)
    {
        setError(errorMessage, QStringLiteral("点云快照已丢弃，无法提交。"));
        return false;
    }

    if (!replaceFileAtomically(_state->temporaryPath,
                               _state->finalPath,
                               errorMessage))
    {
        return false;
    }

    _state->status = State::Status::Committed;
    return true;
}

bool PointCloudSnapshotGuard::discard(QString *errorMessage)
{
    std::lock_guard lock(_state->mutex);
    if (_state->status != State::Status::Pending)
    {
        clearError(errorMessage);
        return true;
    }

    if (QFileInfo::exists(_state->temporaryPath)
        && !QFile::remove(_state->temporaryPath))
    {
        setError(errorMessage,
                 QStringLiteral("无法清理点云临时文件：%1")
                     .arg(_state->temporaryPath));
        return false;
    }

    _state->status = State::Status::Discarded;
    clearError(errorMessage);
    return true;
}

QString PointCloudSnapshotStageResult::temporaryPath() const
{
    return guard ? guard->temporaryPath() : QString();
}

bool PointCloudSnapshotStageResult::commit(QString *errorMessage) const
{
    if (!guard)
    {
        setError(errorMessage, QStringLiteral("没有可提交的点云快照。"));
        return false;
    }
    return guard->commit(errorMessage);
}

bool PointCloudSnapshotStageResult::discard(QString *errorMessage) const
{
    if (!guard)
    {
        clearError(errorMessage);
        return true;
    }
    return guard->discard(errorMessage);
}

PointCloudSnapshotStageResult stagePointCloudSnapshot(
    const QString &finalPath,
    const SnapshotCloud &cloud,
    const std::atomic_bool *cancellationFlag)
{
    PointCloudSnapshotStageResult result;
    result.pointCount = static_cast<int>(cloud.size());
    if (cancellationFlag
        && cancellationFlag->load(std::memory_order_relaxed))
    {
        result.errorMessage = QStringLiteral("点云快照暂存已取消。");
        return result;
    }
    if (finalPath.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("当前点云来源未知，无法暂存快照。");
        return result;
    }

    result.path = normalizedFinalPath(finalPath);
    const QFileInfo final_info(result.path);
    const QString extension = final_info.suffix().toLower();
    if (extension != QLatin1String("ply")
        && extension != QLatin1String("obj")
        && extension != QLatin1String("xyz"))
    {
        result.errorMessage = QStringLiteral("不支持的点云文件扩展名：.%1")
                                  .arg(extension.isEmpty()
                                           ? QStringLiteral("<无>")
                                           : extension);
        return result;
    }

    const QDir parent_dir = final_info.absoluteDir();
    if (!parent_dir.exists())
    {
        result.errorMessage = QStringLiteral("点云输出目录不存在：%1")
                                  .arg(parent_dir.absolutePath());
        return result;
    }

    const QString temporary_template = parent_dir.filePath(
        QStringLiteral(".%1.plascan-stage-XXXXXX")
            .arg(final_info.fileName()));
    QTemporaryFile temporary_file(temporary_template);
    if (!temporary_file.open())
    {
        result.errorMessage = QStringLiteral("无法在点云目录创建临时文件：%1（%2）")
                                  .arg(parent_dir.absolutePath(),
                                       temporary_file.errorString());
        return result;
    }

    const QString temporary_path = temporary_file.fileName();
    temporary_file.setAutoRemove(false);
    temporary_file.close();

    auto guard = std::shared_ptr<PointCloudSnapshotGuard>(
        new PointCloudSnapshotGuard(result.path, temporary_path));
    if (cancellationFlag
        && cancellationFlag->load(std::memory_order_relaxed))
    {
        guard->discard();
        result.errorMessage = QStringLiteral("点云快照暂存已取消。");
        return result;
    }
    const std::string native_path =
        xjw::common::io::toNativeNarrowPath(temporary_path);
    try
    {
        if (extension == QLatin1String("ply"))
        {
            plapoint::io::writePly<float>(
                native_path, cloud, plapoint::io::PlyFormat::BinaryLE);
        }
        else if (extension == QLatin1String("obj"))
        {
            plapoint::io::writeObj<float>(native_path, cloud);
        }
        else
        {
            plapoint::io::writeXyz<float>(native_path, cloud);
        }
    }
    catch (const std::exception &error)
    {
        guard->discard();
        result.errorMessage = QString::fromLocal8Bit(error.what());
        return result;
    }

    if (cancellationFlag
        && cancellationFlag->load(std::memory_order_relaxed))
    {
        guard->discard();
        result.errorMessage = QStringLiteral("点云快照暂存已取消。");
        return result;
    }

    result.success = true;
    result.guard = std::move(guard);
    return result;
}

} // namespace xjw::gui::point_cloud
