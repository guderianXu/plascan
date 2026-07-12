#include "ImageViewRotationSettings.h"

#include <QDir>

namespace xjw::gui::config
{

int normalizeImageViewRotationDegrees(int degrees)
{
    if (degrees % 90 != 0)
    {
        return 0;
    }

    return ((degrees % 360) + 360) % 360;
}

QString imageViewRotationPathKey(const QString &imagePath)
{
    QString key = QDir::cleanPath(imagePath.trimmed());
    if (key == QStringLiteral("."))
    {
        return QString();
    }
#ifdef Q_OS_WIN
    key = key.toLower();
#endif
    return key;
}

int imageViewRotationForPath(const QJsonObject &rotations, const QString &imagePath)
{
    const QString key = imageViewRotationPathKey(imagePath);
    if (key.isEmpty())
    {
        return 0;
    }

    const QJsonValue value = rotations.value(key);
    if (!value.isDouble())
    {
        return 0;
    }
    return normalizeImageViewRotationDegrees(value.toInt());
}

QJsonObject withImageViewRotation(const QJsonObject &rotations,
                                  const QString &imagePath,
                                  int degrees)
{
    QJsonObject updated = rotations;
    const QString key = imageViewRotationPathKey(imagePath);
    if (key.isEmpty())
    {
        return updated;
    }

    const int normalized = normalizeImageViewRotationDegrees(degrees);
    if (normalized == 0)
    {
        updated.remove(key);
    }
    else
    {
        updated[key] = normalized;
    }
    return updated;
}

} // namespace xjw::gui::config
