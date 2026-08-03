#include "runtime/PythonRuntimeLocator.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>

namespace xjw::common::runtime
{
namespace
{

QString existingFilePath(const QString &path)
{
    const QString trimmed_path = path.trimmed();
    if (trimmed_path.isEmpty())
    {
        return {};
    }

    const QFileInfo file_info(trimmed_path);
    if (!file_info.exists() || !file_info.isFile())
    {
        return {};
    }

    return file_info.absoluteFilePath();
}

QString runtimePythonRelativePath()
{
#ifdef Q_OS_WIN
    return QStringLiteral(".venv/Scripts/python.exe");
#else
    return QStringLiteral(".venv/bin/python");
#endif
}

QString userRuntimePythonPath(const QString &runtimeDirectory)
{
    if (runtimeDirectory.trimmed().isEmpty())
    {
        return {};
    }

    return existingFilePath(QDir(runtimeDirectory).filePath(
#ifdef Q_OS_WIN
        QStringLiteral("Scripts/python.exe")
#else
        QStringLiteral("bin/python")
#endif
    ));
}

QString repositoryVenvPythonPath(const QString &sourceRoot)
{
    if (sourceRoot.trimmed().isEmpty())
    {
        return {};
    }

    return existingFilePath(QDir(sourceRoot).filePath(runtimePythonRelativePath()));
}

QString generatedEnvironmentPythonPath(const QString &sourceRoot)
{
    if (sourceRoot.trimmed().isEmpty())
    {
        return {};
    }

    QFile file(QDir(sourceRoot).filePath(QStringLiteral("build/env/plascan-env.json")));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return {};
    }

    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject())
    {
        return {};
    }

    return existingFilePath(document.object().value(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE")).toString());
}

QString environmentPythonPath(const QProcessEnvironment &environment)
{
    const QString explicit_path = environment.value(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE")).trimmed();
    if (!explicit_path.isEmpty())
    {
        return explicit_path;
    }

    return environment.value(QStringLiteral("PLASCAN_PYTHON")).trimmed();
}

} // namespace

QString resolvePythonExecutable(const QProcessEnvironment &environment, const QString &sourceRoot)
{
    return resolvePythonExecutable(environment, sourceRoot, defaultUserPythonRuntimeDirectory());
}

QString defaultUserPythonRuntimeDirectory()
{
    return QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
        .filePath(QStringLiteral("python-runtime"));
}

QString pythonExecutableInRuntime(const QString &runtimeDirectory)
{
    return QDir(runtimeDirectory).filePath(
#ifdef Q_OS_WIN
        QStringLiteral("Scripts/python.exe")
#else
        QStringLiteral("bin/python")
#endif
    );
}

QString resolvePythonExecutable(const QProcessEnvironment &environment,
                                const QString &sourceRoot,
                                const QString &userRuntimeDirectory)
{
    const QString configured_path = environmentPythonPath(environment);
    if (!configured_path.isEmpty())
    {
        return configured_path;
    }

    const QString venv_path = repositoryVenvPythonPath(sourceRoot);
    if (!venv_path.isEmpty())
    {
        return venv_path;
    }

    const QString generated_path = generatedEnvironmentPythonPath(sourceRoot);
    if (!generated_path.isEmpty())
    {
        return generated_path;
    }

    return userRuntimePythonPath(userRuntimeDirectory);
}

} // namespace xjw::common::runtime
