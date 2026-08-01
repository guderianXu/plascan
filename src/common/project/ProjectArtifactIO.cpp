#include "ProjectIO.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>

namespace xjw::common::project
{
namespace
{

QString sanitizedArtifactBase(const QString &imagePath)
{
    QString base = QFileInfo(QDir::fromNativeSeparators(imagePath.trimmed()))
                       .completeBaseName();
    if (base.isEmpty())
    {
        base = QStringLiteral("image");
    }

    for (QChar &character : base)
    {
        if (!character.isLetterOrNumber()
            && character != QLatin1Char('.')
            && character != QLatin1Char('_')
            && character != QLatin1Char('-'))
        {
            character = QLatin1Char('_');
        }
    }
    constexpr qsizetype MaxReadableBaseLength = 80;
    return base.left(MaxReadableBaseLength);
}

} // namespace

QString ProjectIO::imageArtifactKey(const QString &imagePath)
{
    QString identity = QDir::cleanPath(
        QDir::fromNativeSeparators(imagePath.trimmed()));
    if (identity.isEmpty() || identity == QLatin1String("."))
    {
        return {};
    }
#ifdef Q_OS_WIN
    identity = identity.toCaseFolded();
#endif

    const QByteArray digest = QCryptographicHash::hash(
        identity.toUtf8(), QCryptographicHash::Sha256).toHex();
    return QStringLiteral("%1-%2")
        .arg(sanitizedArtifactBase(imagePath), QString::fromLatin1(digest));
}

QString ProjectIO::maskOutputPathForImage(const QString &plascanPath,
                                          const QString &imagePath)
{
    const QString artifact_key = imageArtifactKey(imagePath);
    const QString output_dir = maskOutputDir(plascanPath);
    if (artifact_key.isEmpty() || output_dir.isEmpty())
    {
        return {};
    }
    return QDir(output_dir).filePath(
        artifact_key + QStringLiteral("_mask.png"));
}

QString ProjectIO::findMaskForImage(const QString &plascanPath,
                                    const QString &imagePath)
{
    const QString standard = maskOutputPathForImage(plascanPath, imagePath);
    return !standard.isEmpty() && QFileInfo::exists(standard)
        ? standard
        : QString();
}

QMap<QString, QString> ProjectIO::maskPathsForImages(
    const QString &plascanPath,
    const QStringList &imagePaths)
{
    QMap<QString, QString> masks;
    for (const QString &image_path : imagePaths)
    {
        const QString mask_path = findMaskForImage(plascanPath, image_path);
        if (mask_path.isEmpty())
        {
            continue;
        }
        const QString clean_path = QDir::cleanPath(
            QDir::fromNativeSeparators(image_path));
        masks.insert(image_path, mask_path);
        masks.insert(clean_path, mask_path);
    }
    return masks;
}

} // namespace xjw::common::project
