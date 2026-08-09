#include "CameraReferenceSetStore.h"

#include "CameraReferenceSetJson.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>

#include <utility>

namespace xjw::camera_reference
{

CameraReferenceSetStore::CameraReferenceSetStore(QString path)
    : _path(path.trimmed().isEmpty() ? QString() : QDir::cleanPath(std::move(path)))
{
}

CameraReferenceSetIoResult CameraReferenceSetStore::load() const
{
    CameraReferenceSetIoResult result;
    if (_path.isEmpty())
    {
        result.error = QStringLiteral("相机参考 sidecar 路径为空");
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
        result.error = QStringLiteral("无法读取相机参考 sidecar: %1").arg(file.errorString());
        return result;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        result.error = QStringLiteral("相机参考 sidecar JSON 损坏，文件保持只读未覆盖: %1")
                           .arg(parseError.errorString());
        return result;
    }

    if (!CameraReferenceSetJson::decode(document.object(), &result.referenceSet, &result.error))
    {
        result.error = QStringLiteral("相机参考 sidecar 校验失败，文件保持只读未覆盖: %1")
                           .arg(result.error);
        return result;
    }
    result.ok = true;
    return result;
}

CameraReferenceSetIoResult CameraReferenceSetStore::save(
    const CameraReferenceSet &referenceSet) const
{
    CameraReferenceSetIoResult result;
    result.referenceSet = referenceSet;
    if (_path.isEmpty())
    {
        result.error = QStringLiteral("相机参考 sidecar 路径为空");
        return result;
    }
    if (!referenceSet.validate(&result.error))
    {
        result.error = QStringLiteral("相机参考集合无效: %1").arg(result.error);
        return result;
    }

    const QFileInfo fileInfo(_path);
    if (!QDir().mkpath(fileInfo.absolutePath()))
    {
        result.error = QStringLiteral("无法创建相机参考 sidecar 目录: %1")
                           .arg(fileInfo.absolutePath());
        return result;
    }

    QSaveFile file(_path);
    if (!file.open(QIODevice::WriteOnly))
    {
        result.error = QStringLiteral("无法创建相机参考 sidecar 临时文件: %1")
                           .arg(file.errorString());
        return result;
    }

    const QByteArray bytes = QJsonDocument(CameraReferenceSetJson::encode(referenceSet))
                                 .toJson(QJsonDocument::Indented);
    if (file.write(bytes) != bytes.size())
    {
        file.cancelWriting();
        result.error = QStringLiteral("写入相机参考 sidecar 失败: %1").arg(file.errorString());
        return result;
    }
    if (!file.commit())
    {
        result.error = QStringLiteral("原子替换相机参考 sidecar 失败: %1").arg(file.errorString());
        return result;
    }

    result.ok = true;
    return result;
}

QString CameraReferenceSetStore::path() const
{
    return _path;
}

} // namespace xjw::camera_reference
