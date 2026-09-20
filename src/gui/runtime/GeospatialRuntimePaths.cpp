#include "GeospatialRuntimePaths.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStringList>
#include <QTextStream>

namespace
{

    QString existingDataDirectory(const QString& path, const QString& requiredFile)
    {
        if (path.trimmed().isEmpty())
        {
            return {};
        }
        const QString normalized = QDir::cleanPath(path.trimmed());
        if (!QFileInfo(QDir(normalized).filePath(requiredFile)).isFile())
        {
            return {};
        }
        return QFileInfo(normalized).absoluteFilePath();
    }

    void appendVcpkgCandidates(QStringList* candidates, const QString& vcpkgRoot, const QString& dataSubdirectory)
    {
        if (!candidates || vcpkgRoot.trimmed().isEmpty())
        {
            return;
        }

        const QDir root(vcpkgRoot);
        if (!root.exists())
        {
            return;
        }
        for (const QString& triplet : root.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
        {
            if (triplet == QStringLiteral("vcpkg"))
            {
                continue;
            }
            candidates->append(root.filePath(triplet + QStringLiteral("/share/") + dataSubdirectory));
        }
    }

    void appendBuildCacheCandidates(QStringList* projCandidates, QStringList* gdalCandidates, const QDir& buildRoot)
    {
        QFile cache(buildRoot.filePath(QStringLiteral("CMakeCache.txt")));
        if (!cache.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            return;
        }

        QTextStream stream(&cache);
        while (!stream.atEnd())
        {
            const QString line = stream.readLine();
            const qsizetype type_separator = line.indexOf(QLatin1Char(':'));
            const qsizetype value_separator = line.indexOf(QLatin1Char('='), type_separator + 1);
            if (type_separator <= 0 || value_separator <= type_separator)
            {
                continue;
            }
            const QString key = line.left(type_separator);
            const QString value = line.mid(value_separator + 1).trimmed();
            if (value.isEmpty())
            {
                continue;
            }
            if (key == QStringLiteral("VCPKG_INSTALLED_DIR"))
            {
                const QString root = buildRoot.absoluteFilePath(value);
                appendVcpkgCandidates(projCandidates, root, QStringLiteral("proj"));
                appendVcpkgCandidates(gdalCandidates, root, QStringLiteral("gdal"));
            }
            else if (key == QStringLiteral("PLASCAN_SOURCE_DEPENDENCY_PREFIX"))
            {
                const QDir prefix(buildRoot.absoluteFilePath(value));
                projCandidates->append(prefix.filePath(QStringLiteral("share/proj")));
                gdalCandidates->append(prefix.filePath(QStringLiteral("share/gdal")));
            }
        }
    }

    QString
    resolveDataDirectory(const QString& environmentPath, const QStringList& candidates, const QString& requiredFile)
    {
        const QString configured = existingDataDirectory(environmentPath, requiredFile);
        if (!configured.isEmpty())
        {
            return configured;
        }
        for (const QString& candidate : candidates)
        {
            const QString resolved = existingDataDirectory(candidate, requiredFile);
            if (!resolved.isEmpty())
            {
                return resolved;
            }
        }
        return {};
    }

} // namespace

namespace xjw::gui::runtime
{

    GeospatialDataPaths resolveGeospatialDataPaths(const QString& applicationDirectory,
                                                   const QProcessEnvironment& environment)
    {
        const QDir application_dir(QFileInfo(applicationDirectory).absoluteFilePath());
        QDir build_root(application_dir);
        build_root.cdUp();

        QStringList proj_candidates{application_dir.filePath(QStringLiteral("share/proj")),
                                    build_root.filePath(QStringLiteral("share/proj")),
                                    build_root.filePath(QStringLiteral("source-deps/install/share/proj"))};
        appendVcpkgCandidates(
            &proj_candidates, build_root.filePath(QStringLiteral("vcpkg_installed")), QStringLiteral("proj"));
        appendVcpkgCandidates(&proj_candidates,
                              build_root.filePath(QStringLiteral("source-deps/vcpkg_installed")),
                              QStringLiteral("proj"));

        QStringList gdal_candidates{application_dir.filePath(QStringLiteral("share/gdal")),
                                    build_root.filePath(QStringLiteral("share/gdal")),
                                    build_root.filePath(QStringLiteral("source-deps/install/share/gdal"))};
        appendVcpkgCandidates(
            &gdal_candidates, build_root.filePath(QStringLiteral("vcpkg_installed")), QStringLiteral("gdal"));
        appendVcpkgCandidates(&gdal_candidates,
                              build_root.filePath(QStringLiteral("source-deps/vcpkg_installed")),
                              QStringLiteral("gdal"));
        // Source-dependency presets may keep their install tree outside the main build tree.
        // Installed layouts retain precedence and do not require a CMake cache.
        appendBuildCacheCandidates(&proj_candidates, &gdal_candidates, build_root);

        GeospatialDataPaths paths;
        paths.projData = resolveDataDirectory(
            environment.value(QStringLiteral("PROJ_DATA"), environment.value(QStringLiteral("PROJ_LIB"))),
            proj_candidates,
            QStringLiteral("proj.db"));
        paths.gdalData = resolveDataDirectory(
            environment.value(QStringLiteral("GDAL_DATA")), gdal_candidates, QStringLiteral("gdalvrt.xsd"));
        return paths;
    }

    GeospatialDataPaths configureGeospatialDataPaths(const QString& applicationDirectory)
    {
        const GeospatialDataPaths paths = resolveGeospatialDataPaths(applicationDirectory);
        if (!paths.projData.isEmpty())
        {
            const QByteArray encoded_path = QDir::toNativeSeparators(paths.projData).toUtf8();
            qputenv("PROJ_DATA", encoded_path);
            qputenv("PROJ_LIB", encoded_path);
        }
        if (!paths.gdalData.isEmpty())
        {
            qputenv("GDAL_DATA", QDir::toNativeSeparators(paths.gdalData).toUtf8());
        }
        if (!qEnvironmentVariableIsSet("GTIFF_SRS_SOURCE"))
        {
            qputenv("GTIFF_SRS_SOURCE", QByteArrayLiteral("EPSG"));
        }
        return paths;
    }

} // namespace xjw::gui::runtime
