#pragma once

#include <QProcessEnvironment>
#include <QString>

namespace xjw::common::runtime
{

/**
 * @brief Resolves the Python executable used by PlaScan runtime scripts.
 *
 * The explicit environment setting takes precedence, followed by the
 * repository-local .venv and the legacy generated environment file.
 */
QString resolvePythonExecutable(const QProcessEnvironment &environment, const QString &sourceRoot);

} // namespace xjw::common::runtime
