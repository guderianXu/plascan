#include "PythonRuntimeBinding.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>

namespace
{
QString existingFilePath(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return {};
    }

    QFileInfo info(trimmed);
    if (!info.exists() || !info.isFile())
    {
        return {};
    }
    return info.absoluteFilePath();
}

QString runtimePythonRelativePath()
{
#ifdef Q_OS_WIN
    return QStringLiteral(".venv/Scripts/python.exe");
#else
    return QStringLiteral(".venv/bin/python");
#endif
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

    const QJsonObject object = document.object();
    return existingFilePath(object.value(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE")).toString());
}

QString environmentPythonPath(const QProcessEnvironment &environment)
{
    const QString explicitPath = environment.value(QStringLiteral("PLASCAN_PYTHON_EXECUTABLE")).trimmed();
    if (!explicitPath.isEmpty())
    {
        return explicitPath;
    }
    return environment.value(QStringLiteral("PLASCAN_PYTHON")).trimmed();
}
} // namespace

QString PythonRuntimeBinding::resolvePythonExecutable(const QProcessEnvironment &environment,
                                                      const QString &sourceRoot,
                                                      const QString &applicationDir)
{
    Q_UNUSED(applicationDir)

    const QString configuredPath = environmentPythonPath(environment);
    if (!configuredPath.isEmpty())
    {
        return configuredPath;
    }

    const QString venvPath = repositoryVenvPythonPath(sourceRoot);
    if (!venvPath.isEmpty())
    {
        return venvPath;
    }

    return generatedEnvironmentPythonPath(sourceRoot);
}

bool PythonRuntimeBinding::bindPythonRuntime(const QString &sourceRoot, const QString &applicationDir)
{
    const QString pythonPath = resolvePythonExecutable(QProcessEnvironment::systemEnvironment(),
                                                       sourceRoot,
                                                       applicationDir);
    if (pythonPath.trimmed().isEmpty())
    {
        return false;
    }

    const QByteArray encodedPath = QFileInfo(pythonPath).absoluteFilePath().toUtf8();
    qputenv("PLASCAN_PYTHON_EXECUTABLE", encodedPath);
    qputenv("PLASCAN_PYTHON", encodedPath);
    return true;
}
