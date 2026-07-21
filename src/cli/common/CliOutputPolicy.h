#pragma once

#include <QString>
#include <QStringList>

namespace xjw::cli
{

struct OutputDirectoryPolicy
{
    bool allowNonEmpty = false;
    QStringList protectedRelativePaths;
};

bool validateOutputDirectory(const QString &outputDir,
                             const OutputDirectoryPolicy &policy,
                             QString *errorMessage);
bool validateOutputDirectory(const QString &outputDir,
                             bool force,
                             QString *errorMessage);

} // namespace xjw::cli
