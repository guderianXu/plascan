#include "runtime/PythonRuntimeManager.h"

#include "runtime/PythonRuntimeLocator.h"
#include "settings/GuiSettingsStore.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QTimer>

namespace
{
constexpr auto kSuppressPromptKey = "PythonRuntime/suppressMissingPrompt";

QString firstExistingFile(const QStringList &candidates)
{
    for (const QString &candidate : candidates)
    {
        const QFileInfo info(candidate);
        if (info.exists() && info.isFile())
        {
            return info.absoluteFilePath();
        }
    }
    return {};
}

QString installedShareRoot()
{
    return QDir(QCoreApplication::applicationDirPath())
        .absoluteFilePath(QStringLiteral("../share/plascan"));
}

void bindEnvironmentPath(const char *name, const QString &path)
{
    if (!path.trimmed().isEmpty())
    {
        qputenv(name, QDir::toNativeSeparators(path).toUtf8());
    }
}
} // namespace

PythonRuntimeManager::PythonRuntimeManager(QObject *parent)
    : QObject(parent)
{
    _process.setProcessChannelMode(QProcess::MergedChannels);
    connect(&_process, &QProcess::readyReadStandardOutput, this, [this]()
    {
        emit outputReceived(QString::fromUtf8(_process.readAllStandardOutput()));
    });
    connect(&_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error)
    {
        if (error == QProcess::FailedToStart)
        {
            finish(false, tr("无法启动 Python 环境安装程序：%1").arg(_process.errorString()));
        }
    });
    connect(&_process,
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this,
            [this](int exitCode, QProcess::ExitStatus exitStatus)
    {
        if (_cancelRequested)
        {
            finish(false, tr("Python 环境更新已取消。"));
            return;
        }
        if (exitStatus != QProcess::NormalExit || exitCode != 0)
        {
            finish(false, tr("Python 环境更新失败，退出码：%1。请展开详细信息查看原因。").arg(exitCode));
            return;
        }

        QString errorMessage;
        if (!bindManagedRuntime(&errorMessage))
        {
            finish(false, errorMessage);
            return;
        }
        finish(true, tr("Python 环境已更新并绑定到当前 PlaScan。"));
    });
}

QString PythonRuntimeManager::managedRuntimeDirectory()
{
    return xjw::common::runtime::defaultUserPythonRuntimeDirectory();
}

QString PythonRuntimeManager::managedPythonExecutable()
{
    return xjw::common::runtime::pythonExecutableInRuntime(managedRuntimeDirectory());
}

QString PythonRuntimeManager::setupScriptPath()
{
    return firstExistingFile({
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(
            QStringLiteral("scripts/env/setup_python_runtime.py")),
        QDir(installedShareRoot()).filePath(
            QStringLiteral("scripts/env/setup_python_runtime.py"))
    });
}

QString PythonRuntimeManager::bootstrapScriptPath()
{
    return firstExistingFile({
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(
            QStringLiteral("scripts/env/bootstrap_python_runtime.ps1")),
        QDir(installedShareRoot()).filePath(
            QStringLiteral("scripts/env/bootstrap_python_runtime.ps1"))
    });
}

QString PythonRuntimeManager::currentPythonExecutable()
{
    return xjw::common::runtime::resolvePythonExecutable(
        QProcessEnvironment::systemEnvironment(), QStringLiteral(PLASCAN_SOURCE_DIR));
}

bool PythonRuntimeManager::bindManagedRuntime(QString *errorMessage)
{
    const QString pythonPath = managedPythonExecutable();
    if (!QFileInfo::exists(pythonPath))
    {
        if (errorMessage)
        {
            *errorMessage = tr("安装程序已结束，但未找到 Python 运行时：%1").arg(pythonPath);
        }
        return false;
    }

    bindEnvironmentPath("PLASCAN_PYTHON_EXECUTABLE", pythonPath);
    bindEnvironmentPath("PLASCAN_PYTHON", pythonPath);
    bindEnvironmentPath("PLASCAN_PYTHON_RUNTIME_DIR", managedRuntimeDirectory());

    const QString setupPath = setupScriptPath();
    if (!setupPath.isEmpty())
    {
        QDir scriptRoot = QFileInfo(setupPath).absoluteDir();
        scriptRoot.cdUp();
        bindEnvironmentPath("PLASCAN_SCRIPT_DIR", scriptRoot.absolutePath());
    }
    return true;
}

bool PythonRuntimeManager::startupPromptSuppressed()
{
    QSettings settings = xjw::gui::settings::createGuiSettings();
    return settings.value(QString::fromLatin1(kSuppressPromptKey), false).toBool();
}

void PythonRuntimeManager::setStartupPromptSuppressed(bool suppressed)
{
    QSettings settings = xjw::gui::settings::createGuiSettings();
    settings.setValue(QString::fromLatin1(kSuppressPromptKey), suppressed);
}

bool PythonRuntimeManager::isRunning() const
{
    return _process.state() != QProcess::NotRunning;
}

QString PythonRuntimeManager::sourceRootForSetup(const QString &setupScript) const
{
    const QString sourceSetup = QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(
        QStringLiteral("scripts/env/setup_python_runtime.py"));
    if (QFileInfo(setupScript).absoluteFilePath() == QFileInfo(sourceSetup).absoluteFilePath())
    {
        return QStringLiteral(PLASCAN_SOURCE_DIR);
    }
    return QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral(".."));
}

void PythonRuntimeManager::startInstall()
{
    if (isRunning())
    {
        return;
    }

    _cancelRequested = false;
    _completionEmitted = false;
    const QString setupPath = setupScriptPath();
    if (setupPath.isEmpty())
    {
        finish(false, tr("找不到 setup_python_runtime.py。请重新安装 PlaScan，确保运行时脚本已完整部署。"));
        return;
    }

    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    _process.setProcessEnvironment(environment);

    const QString outputDirectory = QDir(QStandardPaths::writableLocation(
        QStandardPaths::AppLocalDataLocation)).filePath(QStringLiteral("python-env"));

#ifdef Q_OS_WIN
    const QString bootstrapPath = bootstrapScriptPath();
    if (bootstrapPath.isEmpty())
    {
        finish(false, tr("找不到 Windows Python 引导脚本。请重新安装 PlaScan。"));
        return;
    }
    QString powerShell = QStandardPaths::findExecutable(QStringLiteral("powershell.exe"));
    if (powerShell.isEmpty())
    {
        powerShell = QStringLiteral("powershell.exe");
    }
    _process.setProgram(powerShell);
    _process.setArguments({
        QStringLiteral("-NoProfile"),
        QStringLiteral("-ExecutionPolicy"), QStringLiteral("Bypass"),
        QStringLiteral("-File"), bootstrapPath,
        QStringLiteral("-RuntimeDir"), managedRuntimeDirectory(),
        QStringLiteral("-SetupScript"), setupPath,
        QStringLiteral("-SourceDir"), sourceRootForSetup(setupPath),
        QStringLiteral("-OutputDir"), outputDirectory,
        QStringLiteral("-Device"), QStringLiteral("cpu"),
        QStringLiteral("-CudaWheel"), QStringLiteral("cu130")
    });
#else
    QString python = QStandardPaths::findExecutable(QStringLiteral("python3"));
    if (python.isEmpty())
    {
        python = QStandardPaths::findExecutable(QStringLiteral("python"));
    }
    if (python.isEmpty())
    {
        finish(false, tr("系统中未找到 Python 3。请先通过系统软件包管理器安装 Python 3，再重试。"));
        return;
    }
    _process.setProgram(python);
    _process.setArguments({
        setupPath,
        QStringLiteral("--source-dir"), sourceRootForSetup(setupPath),
        QStringLiteral("--runtime-dir"), managedRuntimeDirectory(),
        QStringLiteral("--output-dir"), outputDirectory,
        QStringLiteral("--python"), python,
        QStringLiteral("--device"), QStringLiteral("cpu")
    });
#endif

    emit outputReceived(tr("正在启动 Python 环境更新...\n"));
    _process.start();
}

void PythonRuntimeManager::cancel()
{
    if (!isRunning())
    {
        return;
    }
    _cancelRequested = true;
#ifdef Q_OS_WIN
    QProcess::startDetached(QStringLiteral("taskkill"), {
        QStringLiteral("/PID"), QString::number(_process.processId()),
        QStringLiteral("/T"), QStringLiteral("/F")
    });
    QTimer::singleShot(2000, this, [this]()
    {
        if (isRunning())
        {
            _process.kill();
        }
    });
#else
    _process.kill();
#endif
}

void PythonRuntimeManager::finish(bool success, const QString &message)
{
    if (_completionEmitted)
    {
        return;
    }
    _completionEmitted = true;
    emit finished(success, message);
}
