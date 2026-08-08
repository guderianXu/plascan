#pragma once

#include <QObject>
#include <QProcess>
#include <QString>

class PythonRuntimeManager : public QObject
{
    Q_OBJECT
public:
    explicit PythonRuntimeManager(QObject *parent = nullptr);

    static QString managedRuntimeDirectory();
    static QString managedPythonExecutable();
    static QString setupScriptPath();
    static QString bootstrapScriptPath();
    static QString currentPythonExecutable();
    static bool bindManagedRuntime(QString *errorMessage = nullptr);

    static bool startupPromptSuppressed();
    static void setStartupPromptSuppressed(bool suppressed);

    bool isRunning() const;
    void startInstall();
    void cancel();

signals:
    void outputReceived(const QString &text);
    void finished(bool success, const QString &message);

private:
    QString sourceRootForSetup(const QString &setupScript) const;
    void finish(bool success, const QString &message);

    QProcess _process;
    bool _cancelRequested{};
    bool _completionEmitted{};
};
