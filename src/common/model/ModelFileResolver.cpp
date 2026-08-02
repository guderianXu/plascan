#include "model/ModelFileResolver.h"

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

    const QString clean_path = cleanFilePath(path);
    if (clean_path.isEmpty() || seen->contains(clean_path))
    {
        return;
    }

    seen->insert(clean_path);
    paths->append(clean_path);
}

void appendModelPath(QStringList *paths,
                     QSet<QString> *seen,
                     const QString &dir,
                     const QString &model_name)
{
    const QString clean_dir = cleanFilePath(dir);
    if (!clean_dir.isEmpty())
    {
        appendUniquePath(paths, seen, QDir(clean_dir).filePath(model_name));
    }
}

bool looksLikePath(const QString &model_name)
{
    return QFileInfo(model_name).isAbsolute()
        || model_name.contains(QLatin1Char('/'))
        || model_name.contains(QLatin1Char('\\'));
}

} // namespace

ModelFileResolver::ModelFileResolver(ModelFileSearchOptions options)
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

QString ModelFileResolver::findModel(const QString &model_name) const
{
    for (const QString &candidate : candidatePaths(model_name))
    {
        const QFileInfo info(candidate);
        if (info.exists() && info.isFile())
        {
            return QDir::cleanPath(info.absoluteFilePath());
        }
    }
    return QString();
}

QString ModelFileResolver::findFirstModel(const QStringList &model_names,
                                          QString *picked_model_name) const
{
    for (const QString &model_name : model_names)
    {
        const QString path = findModel(model_name);
        if (!path.isEmpty())
        {
            if (picked_model_name)
            {
                *picked_model_name = model_name;
            }
            return path;
        }
    }
    return QString();
}

QStringList ModelFileResolver::candidatePaths(const QString &model_name) const
{
    const QString clean_model_name = model_name.trimmed();
    if (clean_model_name.isEmpty())
    {
        return {};
    }

    QStringList candidates;
    QSet<QString> seen;
    if (looksLikePath(clean_model_name))
    {
        appendUniquePath(&candidates, &seen, clean_model_name);
    }

    const QString env_name = _options.environmentVariable.trimmed();
    if (!env_name.isEmpty())
    {
        appendModelPath(&candidates, &seen, qEnvironmentVariable(env_name.toUtf8().constData()), clean_model_name);
    }
    if (!_options.sourceRoot.trimmed().isEmpty())
    {
        appendModelPath(&candidates,
                        &seen,
                        QDir(_options.sourceRoot).filePath(QStringLiteral("resources/models")),
                        clean_model_name);
    }
    if (!_options.applicationDir.trimmed().isEmpty())
    {
        const QDir app_dir(_options.applicationDir);
        appendModelPath(&candidates, &seen, app_dir.filePath(QStringLiteral("resources/models")), clean_model_name);
        appendModelPath(&candidates, &seen, app_dir.filePath(QStringLiteral("../resources/models")), clean_model_name);
        appendModelPath(&candidates, &seen, app_dir.filePath(QStringLiteral("../../resources/models")), clean_model_name);
        appendModelPath(&candidates, &seen, app_dir.filePath(QStringLiteral("../models")), clean_model_name);
    }
    appendModelPath(&candidates,
                    &seen,
                    QDir::current().filePath(QStringLiteral("resources/models")),
                    clean_model_name);
    for (const QString &dir : _options.extraSearchDirs)
    {
        appendModelPath(&candidates, &seen, dir, clean_model_name);
    }
    return candidates;
}

QString ModelFileResolver::defaultModelDir() const
{
    const QString env_name = _options.environmentVariable.trimmed();
    if (!env_name.isEmpty())
    {
        const QString env_dir = qEnvironmentVariable(env_name.toUtf8().constData()).trimmed();
        if (!env_dir.isEmpty())
        {
            return cleanFilePath(env_dir);
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
