#include "MarkerSetStore.h"

#include "MarkerSetJson.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>

namespace xjw::control_points
{

MarkerSetStore::MarkerSetStore(QString path)
    : _path(QDir::cleanPath(std::move(path)))
{
}

MarkerSetIoResult MarkerSetStore::load() const
{
    MarkerSetIoResult result;
    if (_path.trimmed().isEmpty())
    {
        result.error = QStringLiteral("标记 sidecar 路径为空");
        return result;
    }
    if (!QFileInfo::exists(_path))
    {
        result.ok = true;
        return result;
    }

    QFile file(_path);
    if (!file.open(QIODevice::ReadOnly))
    {
        result.error = QStringLiteral("无法读取标记 sidecar: %1").arg(file.errorString());
        return result;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        result.error = QStringLiteral("标记 sidecar JSON 损坏，文件保持只读未覆盖: %1")
                           .arg(parseError.errorString());
        return result;
    }

    if (!MarkerSetJson::decode(document.object(), &result.markerSet, &result.error))
    {
        result.error = QStringLiteral("标记 sidecar 校验失败，文件保持只读未覆盖: %1").arg(result.error);
        return result;
    }
    result.ok = true;
    return result;
}

MarkerSetIoResult MarkerSetStore::save(const MarkerSet &markerSet) const
{
    MarkerSetIoResult result;
    result.markerSet = markerSet;
    if (_path.trimmed().isEmpty())
    {
        result.error = QStringLiteral("标记 sidecar 路径为空");
        return result;
    }

    const QFileInfo fileInfo(_path);
    if (!QDir().mkpath(fileInfo.absolutePath()))
    {
        result.error = QStringLiteral("无法创建标记 sidecar 目录: %1").arg(fileInfo.absolutePath());
        return result;
    }

    QSaveFile file(_path);
    if (!file.open(QIODevice::WriteOnly))
    {
        result.error = QStringLiteral("无法创建标记 sidecar 临时文件: %1").arg(file.errorString());
        return result;
    }

    const QByteArray bytes = QJsonDocument(MarkerSetJson::encode(markerSet)).toJson(QJsonDocument::Indented);
    if (file.write(bytes) != bytes.size())
    {
        file.cancelWriting();
        result.error = QStringLiteral("写入标记 sidecar 失败: %1").arg(file.errorString());
        return result;
    }
    if (!file.commit())
    {
        result.error = QStringLiteral("原子替换标记 sidecar 失败: %1").arg(file.errorString());
        return result;
    }

    result.ok = true;
    return result;
}

QString MarkerSetStore::path() const
{
    return _path;
}

} // namespace xjw::control_points
