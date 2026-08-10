#include "ProjectRunArtifactValidator.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>

namespace xjw::gui::project
{
namespace
{

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

bool isExistingFile(const QString &path)
{
    const QFileInfo info(path);
    return !path.isEmpty() && info.isFile() && info.size() > 0;
}

QString comparablePath(const QString &path)
{
    QString normalized = QDir::cleanPath(QDir::fromNativeSeparators(path));
#ifdef Q_OS_WIN
    normalized = normalized.toCaseFolded();
#endif
    return normalized;
}

QString canonicalPath(const QString &path)
{
    return comparablePath(QFileInfo(path).canonicalFilePath());
}

bool readDiagnostics(const QString &path,
                     QJsonObject *diagnostics,
                     QString *errorMessage)
{
    QFile file(path);
    if (!diagnostics || !file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法读取模型运行诊断：%1").arg(path));
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        setError(errorMessage,
                 QStringLiteral("模型运行诊断不是有效 JSON 对象：%1")
                     .arg(path));
        return false;
    }
    *diagnostics = document.object();
    return true;
}

} // namespace

bool isPhysicalDirectory(const QString &path)
{
    const QFileInfo info(path);
    bool isLink = info.isSymLink();
#ifdef Q_OS_WIN
    isLink = isLink || info.isJunction();
#endif
    return info.isDir() && !isLink && !canonicalPath(path).isEmpty()
        && canonicalPath(path) == comparablePath(info.absoluteFilePath());
}

bool sameExistingPath(const QString &first, const QString &second)
{
    const QString firstCanonical = canonicalPath(first);
    return !firstCanonical.isEmpty()
        && firstCanonical == canonicalPath(second);
}

bool pathBelongsToDirectory(const QString &path, const QString &directory)
{
    const QString candidate = canonicalPath(path);
    QString root = canonicalPath(directory);
    if (candidate.isEmpty() || root.isEmpty())
    {
        return false;
    }
    root += QLatin1Char('/');
    return candidate.startsWith(root);
}

bool validateRunDiagnostics(const QJsonObject &record,
                            const RunDiagnosticsSpec &spec,
                            QString *errorMessage)
{
    const QString runId = record.value(spec.runIdKey).toString().trimmed();
    const QString runDirectory = record.value(
        spec.runDirectoryKey).toString().trimmed();
    const QString diagnosticsPath = record.value(
        spec.diagnosticsPathKey).toString().trimmed();
    if (runId.isEmpty() || !isPhysicalDirectory(runDirectory)
        || comparablePath(QFileInfo(canonicalPath(runDirectory)).fileName())
            != comparablePath(runId))
    {
        setError(errorMessage,
                 QStringLiteral("%1运行身份或实体目录无效：%2 / %3")
                     .arg(spec.label, runId, runDirectory));
        return false;
    }
    if (!isExistingFile(diagnosticsPath)
        || !pathBelongsToDirectory(diagnosticsPath, runDirectory))
    {
        setError(errorMessage,
                 QStringLiteral("%1运行诊断不属于当前 run 目录：%2")
                     .arg(spec.label, diagnosticsPath));
        return false;
    }

    QJsonObject diagnostics;
    if (!readDiagnostics(diagnosticsPath, &diagnostics, errorMessage))
    {
        return false;
    }
    if (!diagnostics.value(QStringLiteral("ok")).isBool()
        || !diagnostics.value(QStringLiteral("ok")).toBool()
        || diagnostics.value(QStringLiteral("diagnostics_type")).toString()
            != spec.diagnosticsType
        || diagnostics.value(spec.runIdKey).toString().trimmed() != runId
        || !sameExistingPath(
            diagnostics.value(spec.runDirectoryKey).toString(),
            runDirectory))
    {
        setError(errorMessage,
                 QStringLiteral("%1运行诊断的类型、状态或 run 身份不匹配：%2")
                     .arg(spec.label, diagnosticsPath));
        return false;
    }

    for (const QString &key : spec.containedArtifactKeys)
    {
        const QString artifactPath = record.value(key).toString().trimmed();
        if (!artifactPath.isEmpty()
            && (!isExistingFile(artifactPath)
                || !pathBelongsToDirectory(artifactPath, runDirectory)))
        {
            setError(errorMessage,
                     QStringLiteral("%1运行产物越出当前 run 目录（%2）：%3")
                         .arg(spec.label, key, artifactPath));
            return false;
        }
    }
    for (const QString &key : spec.matchingPathKeys)
    {
        const QString recordPath = record.value(key).toString().trimmed();
        if (!recordPath.isEmpty()
            && !sameExistingPath(
                recordPath, diagnostics.value(key).toString().trimmed()))
        {
            setError(errorMessage,
                     QStringLiteral("%1运行诊断中的产物路径不匹配（%2）")
                         .arg(spec.label, key));
            return false;
        }
    }
    return true;
}

} // namespace xjw::gui::project
