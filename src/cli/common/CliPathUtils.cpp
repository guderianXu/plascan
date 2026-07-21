#include "CliPathUtils.h"

#include <QFileInfo>

namespace xjw::cli
{

QString fromStdString(const std::string &value)
{
    return QString::fromUtf8(value.data(), static_cast<int>(value.size()));
}

QString cleanAbsolutePath(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
}

QString resolveListToken(const QString &token, const QDir &baseDir)
{
    QString trimmed = token.trimmed();
    if (trimmed.startsWith(QStringLiteral("~/")))
    {
        trimmed = QDir::home().filePath(trimmed.mid(2));
    }

    const QFileInfo info(trimmed);
    if (info.isAbsolute())
    {
        return QDir::cleanPath(info.absoluteFilePath());
    }
    return QDir::cleanPath(QFileInfo(baseDir.filePath(trimmed)).absoluteFilePath());
}

bool ensureDirectory(const QString &directoryPath, QString *errorMessage)
{
    if (directoryPath.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("目录路径为空");
        }
        return false;
    }
    if (!QDir().mkpath(directoryPath))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建目录: %1").arg(directoryPath);
        }
        return false;
    }
    return true;
}

} // namespace xjw::cli
