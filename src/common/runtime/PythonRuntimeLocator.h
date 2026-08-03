#pragma once

#include <QProcessEnvironment>
#include <QString>

namespace xjw::common::runtime
{

/**
 * @brief Resolves the Python executable used by PlaScan runtime scripts.
 *
 * The explicit environment setting takes precedence, followed by the
 * repository-local .venv, the legacy generated environment file, and finally
 * the per-user runtime managed by PlaScan.
 */
QString resolvePythonExecutable(const QProcessEnvironment &environment, const QString &sourceRoot);

/** @brief Returns the per-user runtime directory managed by PlaScan. */
QString defaultUserPythonRuntimeDirectory();

/** @brief Returns the platform-specific Python executable below a runtime directory. */
QString pythonExecutableInRuntime(const QString &runtimeDirectory);

/**
 * @brief Resolves Python while allowing tests and tools to override the managed runtime directory.
 */
QString resolvePythonExecutable(const QProcessEnvironment &environment,
                                const QString &sourceRoot,
                                const QString &userRuntimeDirectory);

} // namespace xjw::common::runtime
