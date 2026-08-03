#include "model/ModelFileResolver.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSet>
#include <QStandardPaths>

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

QString absoluteCleanPath(const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

bool pathIsInside(const QString &path, const QString &directory)
{
    const QString clean_path = absoluteCleanPath(path);
    const QString clean_directory = absoluteCleanPath(directory);
    if (clean_path.isEmpty() || clean_directory.isEmpty())
    {
        return false;
    }

    const QString relative = QDir(clean_directory).relativeFilePath(clean_path);
    return relative != QStringLiteral("..")
        && !relative.startsWith(QStringLiteral("../"))
        && !QFileInfo(relative).isAbsolute();
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
    if (_options.userModelDir.trimmed().isEmpty())
    {
        _options.userModelDir = QDir(
            QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
            .filePath(QStringLiteral("models"));
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
    if (isSourceTreeRuntime())
    {
        appendModelPath(&candidates,
                        &seen,
                        QDir(_options.sourceRoot).filePath(QStringLiteral("resources/models")),
                        clean_model_name);
    }
    appendModelPath(&candidates, &seen, _options.userModelDir, clean_model_name);
    if (!_options.applicationDir.trimmed().isEmpty())
    {
        const QDir app_dir(_options.applicationDir);
        appendModelPath(&candidates, &seen, app_dir.filePath(QStringLiteral("resources/models")), clean_model_name);
        appendModelPath(&candidates, &seen, app_dir.filePath(QStringLiteral("../resources/models")), clean_model_name);
        appendModelPath(&candidates, &seen, app_dir.filePath(QStringLiteral("../../resources/models")), clean_model_name);
        appendModelPath(&candidates, &seen, app_dir.filePath(QStringLiteral("../models")), clean_model_name);
    }
    for (const QString &dir : _options.extraSearchDirs)
    {
        appendModelPath(&candidates, &seen, dir, clean_model_name);
    }
    return candidates;
}

QString ModelFileResolver::defaultModelDir() const
{
    return installLocation().directory;
}

ModelInstallLocation ModelFileResolver::installLocation() const
{
    const QString env_name = _options.environmentVariable.trimmed();
    if (!env_name.isEmpty())
    {
        const QString env_dir = qEnvironmentVariable(env_name.toUtf8().constData()).trimmed();
        if (!env_dir.isEmpty())
        {
            return {
                cleanFilePath(env_dir),
                QStringLiteral("PLASCAN_MODEL_DIR"),
                ModelInstallLocationKind::EnvironmentOverride,
            };
        }
    }
    if (isSourceTreeRuntime())
    {
        return {
            cleanFilePath(QDir(_options.sourceRoot).filePath(QStringLiteral("resources/models"))),
            QStringLiteral("源码资源目录"),
            ModelInstallLocationKind::SourceTree,
        };
    }
    if (!_options.userModelDir.trimmed().isEmpty())
    {
        return {
            cleanFilePath(_options.userModelDir),
            QStringLiteral("用户模型目录"),
            ModelInstallLocationKind::UserData,
        };
    }
    return {
        cleanFilePath(QDir::home().filePath(QStringLiteral(".plascan/models"))),
        QStringLiteral("用户模型目录"),
        ModelInstallLocationKind::UserData,
    };
}

bool ModelFileResolver::isSourceTreeRuntime() const
{
    const QString source_root = absoluteCleanPath(_options.sourceRoot);
    const QString application_dir = absoluteCleanPath(_options.applicationDir);
    if (source_root.isEmpty() || application_dir.isEmpty())
    {
        return false;
    }

    const QFileInfo cmake_file(QDir(source_root).filePath(QStringLiteral("CMakeLists.txt")));
    return cmake_file.isFile() && pathIsInside(application_dir, source_root);
}

} // namespace xjw::common::model
