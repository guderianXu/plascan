#pragma once

#include <QString>

#include <string>

namespace xjw::cli
{

QString normalizedToken(QString value, const QString &fallback = QString());
QString normalizedToken(const std::string &value, const QString &fallback = QString());

} // namespace xjw::cli
