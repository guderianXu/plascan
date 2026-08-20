#include "ModelOutputPolicy.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QUuid>

namespace xjw::mesh::workflow
{
namespace
{

void setError(QString *errorMessage, const QString &message);

bool isValidRunId(const QString &runId)
{
    static const QRegularExpression expression(
        QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"));
    return expression.match(runId).hasMatch();
}

bool isFilesystemLink(const QFileInfo &info)
{
    bool isLink = info.isSymLink();
#ifdef Q_OS_WIN
    isLink = isLink || info.isJunction();
#endif
    return isLink;
}

QString comparablePath(const QString &path)
{
    QString normalized = QDir::cleanPath(
        QFileInfo(path).absoluteFilePath());
#ifdef Q_OS_WIN
    normalized = normalized.toCaseFolded();
#endif
    return normalized;
}

bool canonicalMatchesAbsolute(const QFileInfo &info)
{
    const QString canonical = comparablePath(info.canonicalFilePath());
    const QString absolute = comparablePath(info.absoluteFilePath());
    if (canonical == absolute)
    {
        return true;
    }
#ifdef Q_OS_MACOS
    // /var is a system-owned alias of /private/var on macOS. QTemporaryDir
    // uses the alias, so allow that root mapping without allowing a link at
    // the run directory itself.
    const QString varAlias = QStringLiteral("/var");
    const QString canonicalVar = comparablePath(
        QFileInfo(varAlias).canonicalFilePath());
    return !canonicalVar.isEmpty()
        && absolute.startsWith(varAlias + QLatin1Char('/'))
        && canonical == canonicalVar + absolute.sliced(varAlias.size());
#else
    return false;
#endif
}

bool directoryContainsLink(const QString &directoryPath)
{
    QStringList pending{QFileInfo(directoryPath).absoluteFilePath()};
    QSet<QString> visited;
    while (!pending.isEmpty())
    {
        const QString current = pending.takeLast();
        const QFileInfo currentInfo(current);
        const QString canonical = currentInfo.canonicalFilePath();
        if (canonical.isEmpty())
        {
            return true;
        }
        const QString identity = comparablePath(canonical);
        if (visited.contains(identity))
        {
            return true;
        }
        visited.insert(identity);

        const QDir directory(current);
        if (!directory.isReadable())
        {
            return true;
        }
        const QFileInfoList entries = directory.entryInfoList(
            QDir::AllEntries | QDir::NoDotAndDotDot
                | QDir::Hidden | QDir::System);
        for (const QFileInfo &entry : entries)
        {
            if (isFilesystemLink(entry))
            {
                return true;
            }
            if (entry.isDir())
            {
                pending.push_back(entry.absoluteFilePath());
            }
        }
    }
    return false;
}

QString ownershipMarkerPath(const QString &runDirectory)
{
    return QDir(runDirectory).filePath(
        QStringLiteral(".plascan_task_run.json"));
}

bool writeOwnershipMarker(const QString &runDirectory,
                          const QString &runCollectionName,
                          const QString &runId,
                          QString *errorMessage)
{
    const QJsonObject marker{
        {QStringLiteral("schema_version"), 1},
        {QStringLiteral("type"), QStringLiteral("plascan_task_run")},
        {QStringLiteral("run_collection"), runCollectionName},
        {QStringLiteral("run_id"), runId}
    };
    const QByteArray bytes = QJsonDocument(marker).toJson(
        QJsonDocument::Compact);
    const QString markerPath = ownershipMarkerPath(runDirectory);
    QSaveFile file(markerPath);
    if (!file.open(QIODevice::WriteOnly)
        || file.write(bytes) != bytes.size()
        || !file.commit())
    {
        setError(errorMessage,
                 QStringLiteral("无法写入运行目录所有权标记：%1")
                     .arg(markerPath));
        return false;
    }
    return true;
}

bool ownershipMarkerMatches(const QString &runDirectory,
                            const QString &runCollectionName,
                            const QString &runId)
{
    const QString markerPath = ownershipMarkerPath(runDirectory);
    const QFileInfo markerInfo(markerPath);
    if (!markerInfo.isFile() || isFilesystemLink(markerInfo)
        || markerInfo.size() <= 0 || markerInfo.size() > 16 * 1024)
    {
        return false;
    }

    QFile file(markerPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        file.readAll(), &parseError);
    const QJsonObject marker = document.object();
    return parseError.error == QJsonParseError::NoError
        && marker.value(QStringLiteral("schema_version")).toInt() == 1
        && marker.value(QStringLiteral("type")).toString()
            == QStringLiteral("plascan_task_run")
        && marker.value(QStringLiteral("run_collection")).toString()
            == runCollectionName
        && marker.value(QStringLiteral("run_id")).toString() == runId;
}

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

bool createRunOutputDirectory(const QString &baseOutputRoot,
                              const QString &runCollectionName,
                              const QString &requestedRunId,
                              QString *runId,
                              QString *runOutputRoot,
                              QString *errorMessage)
{
    if (!runId || !runOutputRoot)
    {
        setError(errorMessage, QStringLiteral("运行目录输出参数为空"));
        return false;
    }
    const QString baseRoot = QDir::cleanPath(baseOutputRoot.trimmed());
    if (baseRoot.isEmpty() || baseRoot == QLatin1String("."))
    {
        setError(errorMessage, QStringLiteral("运行输出根目录为空"));
        return false;
    }

    const QString runsRoot = QDir(baseRoot).filePath(runCollectionName);
    if (!QDir().mkpath(runsRoot))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建运行根目录：%1").arg(runsRoot));
        return false;
    }
    const QFileInfo runsRootInfo(runsRoot);
    if (!runsRootInfo.isDir() || isFilesystemLink(runsRootInfo)
        || !canonicalMatchesAbsolute(runsRootInfo))
    {
        setError(errorMessage,
                 QStringLiteral("运行根目录不是安全的实体目录：%1")
                     .arg(runsRoot));
        return false;
    }

    const bool generatedId = requestedRunId.trimmed().isEmpty();
    for (int attempt = 0; attempt < (generatedId ? 8 : 1); ++attempt)
    {
        const QString candidateId = generatedId
            ? createModelRunId()
            : requestedRunId.trimmed();
        if (!isValidRunId(candidateId))
        {
            setError(errorMessage,
                     QStringLiteral("运行 ID 包含不安全字符：%1")
                         .arg(candidateId));
            return false;
        }
        if (!QDir(runsRoot).mkdir(candidateId))
        {
            if (QFileInfo::exists(QDir(runsRoot).filePath(candidateId))
                && generatedId)
            {
                continue;
            }
            setError(errorMessage,
                     QStringLiteral("运行目录已存在或无法创建：%1")
                         .arg(QDir(runsRoot).filePath(candidateId)));
            return false;
        }

        const QString candidateRoot = QDir(runsRoot).filePath(candidateId);
        if (!writeOwnershipMarker(candidateRoot,
                                  runCollectionName,
                                  candidateId,
                                  errorMessage))
        {
            QDir().rmdir(candidateRoot);
            return false;
        }

        *runId = candidateId;
        *runOutputRoot = candidateRoot;
        return true;
    }

    setError(errorMessage, QStringLiteral("无法生成唯一的运行目录"));
    return false;
}

bool removeUnpublishedRunDirectory(const QString &baseOutputRoot,
                                   const QString &runCollectionName,
                                   const QString &runId,
                                   const QString &runOutputRoot,
                                   QString *errorMessage)
{
    const QString safeRunId = runId.trimmed();
    if (baseOutputRoot.trimmed().isEmpty()
        || runOutputRoot.trimmed().isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("拒绝清理路径为空的运行目录"));
        return false;
    }
    if (!isValidRunId(safeRunId))
    {
        setError(errorMessage,
                 QStringLiteral("拒绝清理包含不安全 ID 的运行目录：%1")
                     .arg(runId));
        return false;
    }

    const QString baseRoot = QDir::cleanPath(
        QFileInfo(baseOutputRoot).absoluteFilePath());
    const QString runsRoot = QDir(baseRoot).filePath(runCollectionName);
    const QString expectedRoot = QDir(runsRoot).filePath(safeRunId);
    if (comparablePath(expectedRoot) != comparablePath(runOutputRoot))
    {
        setError(errorMessage,
                 QStringLiteral("拒绝清理与任务 ID 不匹配的运行目录：%1")
                     .arg(runOutputRoot));
        return false;
    }

    const QFileInfo targetInfo(expectedRoot);
    if (!targetInfo.exists() && !targetInfo.isSymLink())
    {
        return true;
    }
    const QFileInfo runsRootInfo(runsRoot);
    if (!runsRootInfo.isDir() || isFilesystemLink(runsRootInfo)
        || !canonicalMatchesAbsolute(runsRootInfo)
        || !targetInfo.isDir() || isFilesystemLink(targetInfo))
    {
        setError(errorMessage,
                 QStringLiteral("拒绝清理非实体运行目录：%1")
                     .arg(expectedRoot));
        return false;
    }

    const QString canonicalRunsRoot = runsRootInfo.canonicalFilePath();
    const QString canonicalTarget = targetInfo.canonicalFilePath();
    if (canonicalRunsRoot.isEmpty() || canonicalTarget.isEmpty()
        || comparablePath(QFileInfo(canonicalTarget).absolutePath())
            != comparablePath(canonicalRunsRoot)
        || QFileInfo(canonicalTarget).fileName() != safeRunId
        || !ownershipMarkerMatches(expectedRoot,
                                   runCollectionName,
                                   safeRunId)
        || directoryContainsLink(expectedRoot))
    {
        setError(errorMessage,
                 QStringLiteral("运行目录边界校验失败，已拒绝递归清理：%1")
                     .arg(expectedRoot));
        return false;
    }

    if (!QDir(expectedRoot).removeRecursively())
    {
        setError(errorMessage,
                 QStringLiteral("无法清理未发布的运行目录：%1")
                     .arg(expectedRoot));
        return false;
    }
    return true;
}

} // namespace

QString modelOutputPolicyName(ModelOutputPolicy policy)
{
    return policy == ModelOutputPolicy::ReplaceDefault
        ? QStringLiteral("replace_default")
        : QStringLiteral("create_versioned_result");
}

ModelOutputPolicy modelOutputPolicyFromSettings(const QJsonObject &settings)
{
    const QString storedPolicy = settings.value(
        QStringLiteral("model_output_policy")).toString().trimmed().toLower();
    if (storedPolicy == QStringLiteral("replace_default"))
    {
        return ModelOutputPolicy::ReplaceDefault;
    }
    if (storedPolicy == QStringLiteral("create_versioned_result"))
    {
        return ModelOutputPolicy::CreateVersionedResult;
    }
    return settings.value(QStringLiteral("replaceDefaultModel")).toBool(false)
        ? ModelOutputPolicy::ReplaceDefault
        : ModelOutputPolicy::CreateVersionedResult;
}

QString createModelRunId()
{
    return QStringLiteral("%1-%2")
        .arg(QDateTime::currentDateTimeUtc().toString(
                 QStringLiteral("yyyyMMddTHHmmsszzzZ")),
             QUuid::createUuid().toString(QUuid::WithoutBraces));
}

bool createModelRunOutputDirectory(const QString &baseOutputRoot,
                                   const QString &requestedRunId,
                                   QString *runId,
                                   QString *runOutputRoot,
                                   QString *errorMessage)
{
    return createRunOutputDirectory(baseOutputRoot,
                                    QStringLiteral("model_runs"),
                                    requestedRunId,
                                    runId,
                                    runOutputRoot,
                                    errorMessage);
}

bool createTextureRunOutputDirectory(const QString &modelOutputRoot,
                                     const QString &requestedRunId,
                                     QString *runId,
                                     QString *runOutputRoot,
                                     QString *errorMessage)
{
    return createRunOutputDirectory(modelOutputRoot,
                                    QStringLiteral("texture_runs"),
                                    requestedRunId,
                                    runId,
                                    runOutputRoot,
                                    errorMessage);
}

bool removeUnpublishedModelRunDirectory(const QString &baseOutputRoot,
                                        const QString &runId,
                                        const QString &runOutputRoot,
                                        QString *errorMessage)
{
    return removeUnpublishedRunDirectory(baseOutputRoot,
                                         QStringLiteral("model_runs"),
                                         runId,
                                         runOutputRoot,
                                         errorMessage);
}

bool removeUnpublishedTextureRunDirectory(const QString &modelOutputRoot,
                                          const QString &runId,
                                          const QString &runOutputRoot,
                                          QString *errorMessage)
{
    return removeUnpublishedRunDirectory(modelOutputRoot,
                                         QStringLiteral("texture_runs"),
                                         runId,
                                         runOutputRoot,
                                         errorMessage);
}

} // namespace xjw::mesh::workflow
