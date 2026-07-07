#include "model/TorchScriptModelResolver.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSet>

#include <utility>

namespace xjw::common::model
{
namespace
{

QString defaultSourceRoot()
{
#ifdef PLASCAN_SOURCE_DIR
    return QStringLiteral(PLASCAN_SOURCE_DIR);
#else
    return QString();
#endif
}

QString cleanFilePath(const QString &path)
{
    return QDir::cleanPath(path.trimmed());
}

void appendUniquePath(QStringList *paths, QSet<QString> *seen, const QString &path)
{
    if (!paths || !seen)
    {
        return;
    }

    const QString cleanPath = cleanFilePath(path);
    if (cleanPath.isEmpty() || seen->contains(cleanPath))
    {
        return;
    }

    seen->insert(cleanPath);
    paths->append(cleanPath);
}

void appendModelPath(QStringList *paths,
                     QSet<QString> *seen,
                     const QString &dir,
                     const QString &modelName)
{
    const QString cleanDir = cleanFilePath(dir);
    if (cleanDir.isEmpty())
    {
        return;
    }
    appendUniquePath(paths, seen, QDir(cleanDir).filePath(modelName));
}

bool looksLikePath(const QString &modelName)
{
    return QFileInfo(modelName).isAbsolute()
        || modelName.contains(QLatin1Char('/'))
        || modelName.contains(QLatin1Char('\\'));
}

} // namespace

TorchScriptModelResolver::TorchScriptModelResolver(TorchScriptModelSearchOptions options)
    : _options(std::move(options))
{
    if (_options.sourceRoot.trimmed().isEmpty())
    {
        _options.sourceRoot = defaultSourceRoot();
    }
    if (_options.applicationDir.trimmed().isEmpty())
    {
        _options.applicationDir = QCoreApplication::applicationDirPath();
    }
}

QString TorchScriptModelResolver::findModel(const QString &modelName) const
{
    for (const QString &candidate : candidatePaths(modelName))
    {
        const QFileInfo info(candidate);
        if (info.exists() && info.isFile())
        {
            return QDir::cleanPath(info.absoluteFilePath());
        }
    }
    return QString();
}

QString TorchScriptModelResolver::findFirstModel(const QStringList &modelNames,
                                                 QString *pickedModelName) const
{
    for (const QString &modelName : modelNames)
    {
        const QString path = findModel(modelName);
        if (!path.isEmpty())
        {
            if (pickedModelName)
            {
                *pickedModelName = modelName;
            }
            return path;
        }
    }
    return QString();
}

QStringList TorchScriptModelResolver::candidatePaths(const QString &modelName) const
{
    const QString cleanModelName = modelName.trimmed();
    if (cleanModelName.isEmpty())
    {
        return {};
    }

    QStringList candidates;
    QSet<QString> seen;

    if (looksLikePath(cleanModelName))
    {
        appendUniquePath(&candidates, &seen, cleanModelName);
    }

    const QString envName = _options.environmentVariable.trimmed();
    if (!envName.isEmpty())
    {
        appendModelPath(&candidates, &seen, qEnvironmentVariable(envName.toUtf8().constData()), cleanModelName);
    }

    if (!_options.sourceRoot.trimmed().isEmpty())
    {
        appendModelPath(&candidates,
                        &seen,
                        QDir(_options.sourceRoot).filePath(QStringLiteral("resources/models")),
                        cleanModelName);
    }

    if (!_options.applicationDir.trimmed().isEmpty())
    {
        appendModelPath(&candidates,
                        &seen,
                        QDir(_options.applicationDir).filePath(QStringLiteral("resources/models")),
                        cleanModelName);
        appendModelPath(&candidates,
                        &seen,
                        QDir(_options.applicationDir).filePath(QStringLiteral("../resources/models")),
                        cleanModelName);
        appendModelPath(&candidates,
                        &seen,
                        QDir(_options.applicationDir).filePath(QStringLiteral("../../resources/models")),
                        cleanModelName);
        appendModelPath(&candidates,
                        &seen,
                        QDir(_options.applicationDir).filePath(QStringLiteral("../models")),
                        cleanModelName);
    }

    appendModelPath(&candidates, &seen, QDir::current().filePath(QStringLiteral("resources/models")), cleanModelName);

    for (const QString &dir : _options.extraSearchDirs)
    {
        appendModelPath(&candidates, &seen, dir, cleanModelName);
    }

    return candidates;
}

QString TorchScriptModelResolver::defaultModelDir() const
{
    const QString envName = _options.environmentVariable.trimmed();
    if (!envName.isEmpty())
    {
        const QString envDir = qEnvironmentVariable(envName.toUtf8().constData()).trimmed();
        if (!envDir.isEmpty())
        {
            return cleanFilePath(envDir);
        }
    }

    if (!_options.sourceRoot.trimmed().isEmpty())
    {
        return cleanFilePath(QDir(_options.sourceRoot).filePath(QStringLiteral("resources/models")));
    }

    if (!_options.applicationDir.trimmed().isEmpty())
    {
        return cleanFilePath(QDir(_options.applicationDir).filePath(QStringLiteral("resources/models")));
    }

    return cleanFilePath(QDir::current().filePath(QStringLiteral("resources/models")));
}

} // namespace xjw::common::model
