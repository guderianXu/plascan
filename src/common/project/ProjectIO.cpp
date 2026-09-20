#include "ProjectIO.h"

#include "ProjectChunkStore.h"
#include "ProjectPackageLayout.h"
#include "ProjectPathBridge.h"

#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QReadWriteLock>

#include <filesystem>

namespace xjw::common::project
{

namespace
{

QReadWriteLock runtimeRootsLock;
QHash<QString, QString> runtimeRoots;

using ChunkPathMember = std::filesystem::path (xjw::common::plafs::PlaChunkLayout::*)() const;

QString chunkPath(const QString &root, ChunkPathMember member)
{
    const auto layout = path_bridge::chunkLayout(root);
    return layout ? path_bridge::toQtPath((layout.value().*member)()) : QString();
}

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
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::assetsDirectory);
}

QString ProjectIO::projectBundleAdjustDir(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::bundleAdjustDirectory);
}

QString ProjectIO::projectImagesDir(const QString &plascanPath)
{
    return ProjectPackageLayout::sharedImagesDirectory(plascanPath);
}

QString ProjectIO::projectControlPointsDir(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::controlPointsDirectory);
}

QString ProjectIO::markerSetPath(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::markerSetPath);
}

QString ProjectIO::markerDetectionReviewPath(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::markerDetectionReviewPath);
}

QString ProjectIO::projectCameraReferencesDir(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::cameraReferencesDirectory);
}

QString ProjectIO::cameraReferenceSetPath(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::cameraReferenceSetPath);
}

QString ProjectIO::tmpDir(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::temporaryDirectory);
}

QString ProjectIO::tempFilesPath(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::temporaryFilesPath);
}

QString ProjectIO::tempConfigPath(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::temporaryConfigPath);
}

QString ProjectIO::tempUiStatePath(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::temporaryUiStatePath);
}

QString ProjectIO::tempResultsPath(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::temporaryResultsPath);
}

QString ProjectIO::imageMatchOutputDir(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::imageMatchesDirectory);
}

QString ProjectIO::maskOutputDir(const QString &plascanPath)
{
    return chunkPath(projectRootFromPlascan(plascanPath),
                     &xjw::common::plafs::PlaChunkLayout::masksDirectory);
}

} // namespace xjw::common::project
