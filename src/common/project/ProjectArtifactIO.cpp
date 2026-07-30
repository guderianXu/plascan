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

QString ProjectIO::vwipOutputPathForImage(const QString &plascanPath,
                                          const QString &imagePath)
{
    const QString artifact_key = imageArtifactKey(imagePath);
    const QString output_dir = ipfindOutputDir(plascanPath);
    if (artifact_key.isEmpty() || output_dir.isEmpty())
    {
        return {};
    }
    return QDir(output_dir).filePath(artifact_key + QStringLiteral(".sp"));
}

QString ProjectIO::featureOutputPathForImage(const QString &plascanPath,
                                             const QString &imagePath,
                                             const QString &suffix)
{
    const QString artifact_key = imageArtifactKey(imagePath);
    const QString output_dir = ipfindOutputDir(plascanPath);
    if (artifact_key.isEmpty() || output_dir.isEmpty())
    {
        return {};
    }
    return QDir(output_dir).filePath(artifact_key + suffix);
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

QStringList ProjectIO::spCandidates(const QString &plascanPath,
                                    const QString &imagePath)
{
    QStringList candidates;
    const QString artifact_key = imageArtifactKey(imagePath);
    if (artifact_key.isEmpty())
    {
        return candidates;
    }

    const QString file_name = artifact_key + QStringLiteral(".sp");
    const QString temporary_dir = tmpIpfindDir(plascanPath);
    if (!temporary_dir.isEmpty())
    {
        candidates << QDir(temporary_dir).filePath(file_name);
    }
    const QString output_dir = ipfindOutputDir(plascanPath);
    if (!output_dir.isEmpty())
    {
        candidates << QDir(output_dir).filePath(file_name);
    }
    return candidates;
}

QString ProjectIO::findSpForImage(const QString &plascanPath,
                                  const QString &imagePath)
{
    for (const QString &path : spCandidates(plascanPath, imagePath))
    {
        if (QFileInfo::exists(path))
        {
            return path;
        }
    }
    return {};
}

QString ProjectIO::findFeatureForImage(const QString &plascanPath,
                                       const QString &imagePath)
{
    static const QStringList suffixes{
        QStringLiteral(".sp"),
        QStringLiteral(".dsk"),
        QStringLiteral(".alk"),
        QStringLiteral(".sift"),
        QStringLiteral(".orb"),
        QStringLiteral(".akz"),
        QStringLiteral(".dedode")
    };
    for (const QString &sp_path : spCandidates(plascanPath, imagePath))
    {
        QString base = sp_path;
        base.chop(3);
        for (const QString &suffix : suffixes)
        {
            const QString candidate = base + suffix;
            if (QFileInfo::exists(candidate))
            {
                return candidate;
            }
        }
    }
    return {};
}

QString ProjectIO::featureFileForSuffix(const QString &plascanPath,
                                        const QString &imagePath,
                                        const QString &suffix)
{
    for (const QString &sp_path : spCandidates(plascanPath, imagePath))
    {
        QString base = sp_path;
        base.chop(3);
        const QString candidate = base + suffix;
        if (QFileInfo::exists(candidate))
        {
            return candidate;
        }
    }
    return {};
}

QStringList ProjectIO::availableFeatureSuffixes(const QString &plascanPath,
                                                const QString &imagePath)
{
    static const QStringList suffixes{
        QStringLiteral(".sp"),
        QStringLiteral(".dsk"),
        QStringLiteral(".alk"),
        QStringLiteral(".sift"),
        QStringLiteral(".orb"),
        QStringLiteral(".akz"),
        QStringLiteral(".dedode")
    };
    QStringList result;
    for (const QString &sp_path : spCandidates(plascanPath, imagePath))
    {
        QString base = sp_path;
        base.chop(3);
        for (const QString &suffix : suffixes)
        {
            if (QFileInfo::exists(base + suffix) && !result.contains(suffix))
            {
                result.append(suffix);
            }
        }
    }
    return result;
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
