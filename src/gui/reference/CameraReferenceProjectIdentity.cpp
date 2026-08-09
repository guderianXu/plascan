#include "CameraReferenceProjectIdentity.h"

#include "project/ProjectMetadata.h"

#include <QCryptographicHash>
#include <QJsonArray>

#include <algorithm>

namespace xjw::gui::reference
{

QString cameraReferenceImageSetFingerprint(const QJsonObject &metadata)
{
    QStringList identities;
    const QJsonObject files = xjw::common::project::projectFilesRootObject(metadata);
    for (const QJsonValue &value : files.value(QStringLiteral("images")).toArray())
    {
        const QJsonObject image = value.toObject();
        const QString imageUuid = image.value(QStringLiteral("image_uuid"))
                                      .toString()
                                      .trimmed();
        if (!imageUuid.isEmpty())
        {
            identities.append(imageUuid);
        }
    }
    std::sort(identities.begin(), identities.end());
    return QString::fromLatin1(
        QCryptographicHash::hash(identities.join(QLatin1Char('\n')).toUtf8(),
                                 QCryptographicHash::Sha256)
            .toHex());
}

} // namespace xjw::gui::reference
