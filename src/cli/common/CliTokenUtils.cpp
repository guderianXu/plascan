#include "CliTokenUtils.h"

#include "CliPathUtils.h"

namespace xjw::cli
{

QString normalizedToken(QString value, const QString &fallback)
{
    value = value.trimmed().toLower();
    value.replace(QStringLiteral("-"), QStringLiteral("_"));
    return value.isEmpty() ? fallback : value;
}

QString normalizedToken(const std::string &value, const QString &fallback)
{
    return normalizedToken(fromStdString(value), fallback);
}

} // namespace xjw::cli
