#include "DataTreeResourceUtils.h"

#include "placamera_runtime/ProjectCameraStore.h"

#include <placamera/frame_camera.h>

#include <QDir>
#include <QFileInfo>

namespace xjw::gui::widgets::data_tree
{
    QString imagePathFromValue(const QJsonValue& value)
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
        if (path.isEmpty())
            path = object.value(QStringLiteral("image_path")).toString();
        if (path.isEmpty())
            path = object.value(QStringLiteral("file_path")).toString();
        if (path.isEmpty())
            path = object.value(QStringLiteral("source_path")).toString();
        return path;
    }

    QString imagePathKey(QString path)
    {
        path = QDir::cleanPath(path.trimmed());
        path.replace(QLatin1Char('\\'), QLatin1Char('/'));
        return path.toCaseFolded();
    }

    bool objectHasTrueFlag(const QJsonObject& object, std::initializer_list<const char*> keys)
    {
        for (const char* key : keys)
        {
            const QJsonValue value = object.value(QString::fromLatin1(key));
            if (value.isBool() && value.toBool())
            {
                return true;
            }
        }
        return false;
    }

    bool objectHasAlignedStatus(const QJsonObject& object)
    {
        for (const char* key : {"status", "alignment_status", "orientation_status", "pose_status"})
        {
            const QString status = object.value(QString::fromLatin1(key)).toString().trimmed().toLower();
            if (status == QStringLiteral("aligned") || status == QStringLiteral("registered") ||
                status == QStringLiteral("oriented") || status == QStringLiteral("estimated") ||
                status == QStringLiteral("已对齐") || status == QStringLiteral("已定向"))
            {
                return true;
            }
        }
        return false;
    }

    QSet<QString> projectCameraAlignedImageIds(const QJsonObject& projectMetadata)
    {
        QSet<QString> aligned;
        const auto loaded = xjw::placamera_runtime::loadProjectCameras(projectMetadata);
        if (!loaded.ok())
        {
            return aligned;
        }
        for (const QJsonValue& value : projectMetadata.value(QStringLiteral("camera_instances")).toArray())
        {
            const QJsonObject instance = value.toObject();
            const QString image_id = instance.value(QStringLiteral("image_uuid")).toString();
            if (image_id.isEmpty())
            {
                continue;
            }
            const QJsonObject state = instance.value(QStringLiteral("state")).toObject();
            const QJsonObject metadata = state.value(QStringLiteral("metadata")).toObject();
            if (state.value(QStringLiteral("pose_initialized_as_identity")).toBool(false) ||
                metadata.value(QStringLiteral("pose_initialized_as_identity")).toBool(false))
            {
                continue;
            }
            const auto camera = loaded.instances.forImage(placamera::ImageId(image_id.toStdString()));
            if (!camera.ok())
            {
                continue;
            }
            const bool annotated =
                objectHasTrueFlag(state, {"aligned", "is_aligned", "registered", "oriented", "has_pose"}) ||
                objectHasAlignedStatus(state) ||
                objectHasTrueFlag(metadata, {"aligned", "is_aligned", "registered", "oriented", "has_pose"}) ||
                objectHasAlignedStatus(metadata);
            if (annotated || dynamic_cast<const placamera::FramePinholeModel*>(camera.value().get()) != nullptr)
            {
                aligned.insert(image_id);
            }
        }
        return aligned;
    }

    bool imageIsAligned(const QJsonValue& image,
                        const QSet<QString>& alignedImageKeys,
                        const QSet<QString>& cameraAlignedImageIds)
    {
        const QString key = imagePathKey(imagePathFromValue(image));
        if (!key.isEmpty() && alignedImageKeys.contains(key))
        {
            return true;
        }
        if (!image.isObject())
        {
            return false;
        }
        const QJsonObject image_record = image.toObject();
        if (objectHasTrueFlag(image_record, {"aligned", "is_aligned", "registered", "oriented", "has_pose"}) ||
            objectHasAlignedStatus(image_record))
        {
            return true;
        }
        return cameraAlignedImageIds.contains(image_record.value(QStringLiteral("image_uuid")).toString());
    }

} // namespace xjw::gui::widgets::data_tree
