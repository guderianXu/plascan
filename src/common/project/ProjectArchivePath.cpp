#include "PortableProjectFormat.h"

#include <QDir>
#include <QFileInfo>
#include <QSet>

namespace xjw::common::project
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

bool isWindowsReservedName(const QString &part)
{
    const QString base =
        part.section(QLatin1Char('.'), 0, 0).toCaseFolded();
    static const QSet<QString> reserved{
        QStringLiteral("con"),
        QStringLiteral("prn"),
        QStringLiteral("aux"),
        QStringLiteral("nul"),
        QStringLiteral("com1"),
        QStringLiteral("com2"),
        QStringLiteral("com3"),
        QStringLiteral("com4"),
        QStringLiteral("com5"),
        QStringLiteral("com6"),
        QStringLiteral("com7"),
        QStringLiteral("com8"),
        QStringLiteral("com9"),
        QStringLiteral("lpt1"),
        QStringLiteral("lpt2"),
        QStringLiteral("lpt3"),
        QStringLiteral("lpt4"),
        QStringLiteral("lpt5"),
        QStringLiteral("lpt6"),
        QStringLiteral("lpt7"),
        QStringLiteral("lpt8"),
        QStringLiteral("lpt9")
    };
    return reserved.contains(base);
}

bool containsControlCharacter(const QString &part)
{
    for (const QChar character : part)
    {
        const ushort value = character.unicode();
        if (value < 0x20 || value == 0x7f)
        {
            return true;
        }
    }
    return false;
}

QString comparablePath(QString path)
{
    path = QDir::cleanPath(QDir::fromNativeSeparators(path));
#ifdef Q_OS_WIN
    return path.toCaseFolded();
#else
    return path;
#endif
}

bool isPathWithin(const QString &candidate, const QString &root)
{
    const QString comparable_root = comparablePath(root);
    const QString comparable_candidate = comparablePath(candidate);
    return comparable_candidate == comparable_root
        || comparable_candidate.startsWith(
            comparable_root + QLatin1Char('/'));
}

} // namespace

QString PortableProjectFormat::normalizeEntryPath(const QString &entryPath)
{
    QString normalized = entryPath.trimmed();
    normalized.replace(QLatin1Char('\\'), QLatin1Char('/'));
    while (normalized.startsWith(QStringLiteral("./")))
    {
        normalized.remove(0, 2);
    }
    while (normalized.contains(QStringLiteral("//")))
    {
        normalized.replace(QStringLiteral("//"), QStringLiteral("/"));
    }
    return normalized;
}

bool PortableProjectFormat::isSafeEntryPath(const QString &entryPath)
{
    const QString trimmed = entryPath.trimmed();
    if (trimmed != entryPath
        || trimmed.contains(QLatin1Char('\\'))
        || trimmed.contains(QStringLiteral("//")))
    {
        return false;
    }

    const QString normalized = normalizeEntryPath(entryPath);
    if (normalized.isEmpty()
        || normalized.startsWith(QLatin1Char('/'))
        || normalized.endsWith(QLatin1Char('/'))
        || normalized.contains(QChar::Null))
    {
        return false;
    }

    const QStringList parts = normalized.split(QLatin1Char('/'));
    for (const QString &part : parts)
    {
        if (part.isEmpty()
            || part == QStringLiteral(".")
            || part == QStringLiteral("..")
            || part.contains(QLatin1Char(':'))
            || part.endsWith(QLatin1Char('.'))
            || part.endsWith(QLatin1Char(' '))
            || containsControlCharacter(part)
            || isWindowsReservedName(part))
        {
            return false;
        }
    }
    return !parts.isEmpty();
}

QString PortableProjectFormat::resolveEntryPath(const QString &rootPath,
                                                const QString &entryPath,
                                                QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!isSafeEntryPath(entryPath))
    {
        setError(errorMessage,
                 QStringLiteral("归档条目路径不安全: %1").arg(entryPath));
        return {};
    }

    const QFileInfo root_info(rootPath);
    const QString canonical_root = root_info.canonicalFilePath();
    if (!root_info.isDir() || canonical_root.isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("归档目标根目录不存在或无法解析: %1")
                     .arg(rootPath));
        return {};
    }

    const QString candidate = QDir::cleanPath(
        QDir(canonical_root).absoluteFilePath(normalizeEntryPath(entryPath)));
    if (!isPathWithin(candidate, canonical_root))
    {
        setError(errorMessage,
                 QStringLiteral("归档条目越过目标根目录: %1").arg(entryPath));
        return {};
    }

    QFileInfo existing(candidate);
    while (!existing.exists()
           && comparablePath(existing.absoluteFilePath())
                  != comparablePath(canonical_root))
    {
        existing.setFile(existing.absolutePath());
    }
    const QString canonical_existing = existing.canonicalFilePath();
    if (canonical_existing.isEmpty()
        || !isPathWithin(canonical_existing, canonical_root))
    {
        setError(errorMessage,
                 QStringLiteral("归档条目父目录越过目标根目录: %1")
                     .arg(entryPath));
        return {};
    }
    return candidate;
}

} // namespace xjw::common::project
