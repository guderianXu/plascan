#include "CliOutputPolicy.h"

#include <QDir>
#include <QFileInfo>
#include <QFileInfoList>

namespace xjw::cli
{

bool validateOutputDirectory(const QString &outputDir,
                             const OutputDirectoryPolicy &policy,
                             QString *errorMessage)
{
    const QFileInfo outputInfo(outputDir);
    if (outputInfo.exists() && !outputInfo.isDir())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("输出路径已存在但不是目录: %1").arg(outputDir);
        }
        return false;
    }

    if (policy.allowNonEmpty)
    {
        return true;
    }

    if (outputInfo.exists())
    {
        const QFileInfoList entries = QDir(outputDir).entryInfoList(
            QDir::NoDotAndDotDot | QDir::AllEntries);
        if (!entries.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "输出目录非空，拒绝覆盖已有结果: %1；如需复用请添加 --force")
                                    .arg(outputDir);
            }
            return false;
        }
    }

    const QDir outputDirectory(outputDir);
    for (const QString &relativePath : policy.protectedRelativePaths)
    {
        const QString path = outputDirectory.filePath(relativePath);
        if (QFileInfo::exists(path))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "输出目录已有关键输出文件，拒绝覆盖: %1；如需复用请添加 --force")
                                    .arg(path);
            }
            return false;
        }
    }
    return true;
}

bool validateOutputDirectory(const QString &outputDir, bool force, QString *errorMessage)
{
    OutputDirectoryPolicy policy;
    policy.allowNonEmpty = force;
    return validateOutputDirectory(outputDir, policy, errorMessage);
}

} // namespace xjw::cli
