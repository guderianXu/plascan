#include "ProjectIO.h"

#include "ProjectChunkStore.h"
#include "ProjectPackageLayout.h"

#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QReadWriteLock>

namespace xjw::common::project
{

namespace
{

QReadWriteLock runtimeRootsLock;
QHash<QString, QString> runtimeRoots;

QString normalizedProjectKey(const QString &plascanPath)
{
    if (plascanPath.trimmed().isEmpty())
    {
        return {};
    }
    const QString key =
        QDir::cleanPath(QFileInfo(plascanPath).absoluteFilePath());
#ifdef Q_OS_WIN
    return key.toCaseFolded();
#else
    return key;
#endif
}

} // namespace

void ProjectIO::registerRuntimeRoot(const QString &plascanPath,
                                    const QString &runtimeRoot)
{
    const QString key = normalizedProjectKey(plascanPath);
    const QString root = QDir::cleanPath(runtimeRoot.trimmed());
    if (key.isEmpty() || root.isEmpty() || root == QLatin1String("."))
    {
        return;
    }

    QWriteLocker locker(&runtimeRootsLock);
    runtimeRoots.insert(key, root);
}

void ProjectIO::unregisterRuntimeRoot(const QString &plascanPath)
{
    const QString key = normalizedProjectKey(plascanPath);
    if (key.isEmpty())
    {
        return;
    }

    QWriteLocker locker(&runtimeRootsLock);
    runtimeRoots.remove(key);
}

QString ProjectIO::registeredRuntimeRoot(const QString &plascanPath)
{
    const QString key = normalizedProjectKey(plascanPath);
    if (key.isEmpty())
    {
        return {};
    }

    QReadLocker locker(&runtimeRootsLock);
    return runtimeRoots.value(key);
}

QString ProjectIO::projectRootFromPlascan(const QString &plascanPath)
{
    const QString registered = registeredRuntimeRoot(plascanPath);
    if (!registered.isEmpty())
    {
        return registered;
    }
    if (ProjectPackageLayout::isDescriptor(plascanPath))
    {
        QString error;
        const QString chunkRoot =
            ProjectChunkStore(plascanPath).defaultChunkDirectory(&error);
        return chunkRoot.isEmpty()
            ? QString()
            : chunkRoot;
    }
    return physicalProjectRoot(plascanPath);
}

QString ProjectIO::physicalProjectRoot(const QString &plascanPath)
{
    if (plascanPath.trimmed().isEmpty())
    {
        return {};
    }
    return QFileInfo(plascanPath).absolutePath();
}

QString ProjectIO::resolveProjectResourcePath(const QString &plascanPath,
                                              const QString &resourcePath)
{
    const QString cleanPath = QDir::cleanPath(resourcePath.trimmed());
    if (cleanPath.isEmpty() || cleanPath == QLatin1String("."))
    {
        return QString();
    }
    if (QFileInfo(cleanPath).isAbsolute())
    {
        return cleanPath;
    }

    const QString root = projectRootFromPlascan(plascanPath);
    return root.isEmpty()
        ? QString()
        : QDir::cleanPath(QDir(root).absoluteFilePath(cleanPath));
}

QString ProjectIO::projectAssetsDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral("assets"));
}

QString ProjectIO::projectBundleAdjustDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral("bundle_adjust"));
}

QString ProjectIO::projectImagesDir(const QString &plascanPath)
{
    return ProjectPackageLayout::sharedImagesDirectory(plascanPath);
}

QString ProjectIO::projectControlPointsDir(const QString &plascanPath)
{
    const QString assets = projectAssetsDir(plascanPath);
    if (assets.isEmpty()) return QString();
    return QDir(assets).filePath(QStringLiteral("control_points"));
}

QString ProjectIO::markerSetPath(const QString &plascanPath)
{
    const QString directory = projectControlPointsDir(plascanPath);
    if (directory.isEmpty()) return QString();
    return QDir(directory).filePath(QStringLiteral("marker_set.json"));
}

QString ProjectIO::markerDetectionReviewPath(const QString &plascanPath)
{
    const QString directory = projectControlPointsDir(plascanPath);
    if (directory.isEmpty()) return QString();
    return QDir(directory).filePath(QStringLiteral("detection_review.json"));
}

QString ProjectIO::tmpDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral(".plascan_tmp"));
}

QString ProjectIO::tempFilesPath(const QString &plascanPath)
{
    const QString tmp = tmpDir(plascanPath);
    if (tmp.isEmpty()) return QString();
    return QDir(tmp).filePath(QStringLiteral("project_files.json"));
}

QString ProjectIO::tempConfigPath(const QString &plascanPath)
{
    const QString tmp = tmpDir(plascanPath);
    if (tmp.isEmpty()) return QString();
    return QDir(tmp).filePath(QStringLiteral("project_config.json"));
}

QString ProjectIO::tempUiStatePath(const QString &plascanPath)
{
    const QString tmp = tmpDir(plascanPath);
    if (tmp.isEmpty()) return QString();
    return QDir(tmp).filePath(QStringLiteral("project_ui_state.json"));
}

QString ProjectIO::tempResultsPath(const QString &plascanPath)
{
    const QString tmp = tmpDir(plascanPath);
    if (tmp.isEmpty()) return QString();
    return QDir(tmp).filePath(QStringLiteral("project_results.json"));
}

QString ProjectIO::imageMatchOutputDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral("assets/image_matches"));
}

QString ProjectIO::maskOutputDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral("assets/masks"));
}

} // namespace xjw::common::project
