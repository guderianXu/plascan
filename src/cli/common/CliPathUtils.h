#pragma once

#include <QDir>
#include <QString>

#include <string>

namespace xjw::cli
{

QString fromStdString(const std::string &value);
QString cleanAbsolutePath(const QString &path);
QString resolveListToken(const QString &token, const QDir &baseDir);
bool ensureDirectory(const QString &directoryPath, QString *errorMessage);

} // namespace xjw::cli
