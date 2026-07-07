#pragma once

#include <QProcessEnvironment>
#include <QString>

class PythonRuntimeBinding
{
public:
    static QString resolvePythonExecutable(const QProcessEnvironment &environment,
                                           const QString &sourceRoot,
                                           const QString &applicationDir);
    static bool bindPythonRuntime(const QString &sourceRoot, const QString &applicationDir);
};
