#include "DataTreeResourceUtils.h"

#include <QDir>
#include <QFileInfo>

namespace xjw::gui::widgets::data_tree
{
QString imagePathFromValue(const QJsonValue &value)
{
    if (value.isString())
    {
        return value.toString();
    }
    if (!value.isObject())
    {
        return QString();
    }

    const QJsonObject object = value.toObject();
    QString path = object.value(QStringLiteral("path")).toString();
    if (path.isEmpty()) path = object.value(QStringLiteral("image_path")).toString();
    if (path.isEmpty()) path = object.value(QStringLiteral("file_path")).toString();
    if (path.isEmpty()) path = object.value(QStringLiteral("source_path")).toString();
    return path;
}

QString imagePathKey(QString path)
{
    path = QDir::cleanPath(path.trimmed());
    path.replace(QLatin1Char('\\'), QLatin1Char('/'));
    return path.toCaseFolded();
}

bool jsonArrayHasAtLeast(const QJsonValue &value, int size)
{
    return value.isArray() && value.toArray().size() >= size;
}

bool objectHasTrueFlag(const QJsonObject &object, std::initializer_list<const char *> keys)
{
    for (const char *key : keys)
    {
        const QJsonValue value = object.value(QString::fromLatin1(key));
        if (value.isBool() && value.toBool())
        {
            return true;
        }
    }
    return false;
}

bool objectHasAlignedStatus(const QJsonObject &object)
{
    for (const char *key : {"status", "alignment_status", "orientation_status", "pose_status"})
    {
        const QString status = object.value(QString::fromLatin1(key)).toString().trimmed().toLower();
        if (status == QStringLiteral("aligned") ||
            status == QStringLiteral("registered") ||
            status == QStringLiteral("oriented") ||
            status == QStringLiteral("estimated") ||
            status == QStringLiteral("已对齐") ||
            status == QStringLiteral("已定向"))
        {
            return true;
        }
    }
    return false;
}

bool cameraHasPose(const QJsonObject &camera)
{
    if (camera.isEmpty() || camera.value(QStringLiteral("pose_initialized_as_identity")).toBool(false))
    {
        return false;
    }

    if (objectHasTrueFlag(camera, {"aligned", "is_aligned", "registered", "oriented", "has_pose"}) ||
        objectHasAlignedStatus(camera))
    {
        return true;
    }

    if (jsonArrayHasAtLeast(camera.value(QStringLiteral("C")), 3) &&
        jsonArrayHasAtLeast(camera.value(QStringLiteral("R")), 9))
    {
        return true;
    }

    return camera.value(QStringLiteral("pose")).isObject()
        || camera.value(QStringLiteral("camera_pose")).isObject()
        || camera.value(QStringLiteral("extrinsics")).isObject()
        || camera.value(QStringLiteral("transform")).isObject();
}

bool imageObjectHasAlignedPose(const QJsonObject &image)
{
    if (objectHasTrueFlag(image, {"aligned", "is_aligned", "registered", "oriented", "has_pose"}) ||
        objectHasAlignedStatus(image))
    {
        return true;
    }
    return cameraHasPose(image.value(QStringLiteral("camera")).toObject());
}

bool imageIsAligned(const QJsonValue &image, const QSet<QString> &alignedImageKeys)
{
    const QString key = imagePathKey(imagePathFromValue(image));
    if (!key.isEmpty() && alignedImageKeys.contains(key))
    {
        return true;
    }

    return image.isObject() && imageObjectHasAlignedPose(image.toObject());
}


} // namespace xjw::gui::widgets::data_tree
