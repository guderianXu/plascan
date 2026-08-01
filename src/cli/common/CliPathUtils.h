#pragma once

#include <QDir>
#include <QString>

#include <string>
#include <string_view>

namespace xjw::cli
{

QString fromStdString(const std::string &value);
QString fromStdString(std::u8string_view value);
QString cleanAbsolutePath(const QString &path);
QString resolveListToken(const QString &token, const QDir &baseDir);
bool ensureDirectory(const QString &directoryPath, QString *errorMessage);

} // namespace xjw::cli
