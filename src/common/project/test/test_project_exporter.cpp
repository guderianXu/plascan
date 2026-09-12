#include "project/PlascanArchive.h"
#include "project/ProjectExporter.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectSession.h"

#include <QDir>
#include <QDirIterator>
#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

#include <algorithm>

#include <gtest/gtest.h>

namespace
{

    void writeFile(const QString& path, const QByteArray& bytes)
    {
        ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_EQ(file.write(bytes), bytes.size());
    }

    QByteArray projectTreeDigest(const QString& projectPath)
    {
        QCryptographicHash digest(QCryptographicHash::Sha256);
        const QString dataRoot = xjw::common::project::ProjectPackageLayout::dataDirectory(projectPath);
        QStringList paths{QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath())};
        QDirIterator iterator(dataRoot, QDir::Files, QDirIterator::Subdirectories);
        while (iterator.hasNext())
        {
            paths.append(iterator.next());
        }
        std::sort(paths.begin(), paths.end());
        for (const QString& path : paths)
        {
            QFile file(path);
            EXPECT_TRUE(file.open(QIODevice::ReadOnly)) << qPrintable(path);
            digest.addData(path.toUtf8());
            digest.addData(file.readAll());
        }
        return digest.result();
    }

    TEST(ProjectExporterTest, ExportsExternalImagesAsPortableSharedUris)
    {
        QTemporaryDir temporary;
        ASSERT_TRUE(temporary.isValid());
        const QString projectPath = QDir(temporary.path()).filePath(QStringLiteral("source.plascan"));
        const QString imagePath = QDir(temporary.path()).filePath(QStringLiteral("outside/photo.tif"));
        const QString archivePath = QDir(temporary.path()).filePath(QStringLiteral("portable.zip"));
        writeFile(imagePath, QByteArrayLiteral("portable-image"));

        xjw::common::project::ProjectSession session;
        QString error;
        ASSERT_TRUE(session.create(projectPath, QStringLiteral("source"), &error)) << qPrintable(error);
        ASSERT_TRUE(session.mergeImages(QJsonArray{QJsonObject{{QStringLiteral("path"), imagePath}}}, &error))
            << qPrintable(error);
        ASSERT_TRUE(session.save(&error)) << qPrintable(error);
        const QByteArray sourceDigest = projectTreeDigest(projectPath);
        ASSERT_TRUE(xjw::common::project::ProjectExporter::exportPortableProject(projectPath, archivePath, &error))
            << qPrintable(error);
        EXPECT_EQ(projectTreeDigest(projectPath), sourceDigest);

        PlascanArchive archive(archivePath, PlascanArchivePathType::DirectArchive);
        ASSERT_TRUE(archive.isValid());
        const QVector<QString> entries = archive.listEntries();
        EXPECT_TRUE(entries.contains(QStringLiteral("source.plascan")));
        EXPECT_TRUE(entries.contains(QStringLiteral("source.files/project.zip")));
        EXPECT_TRUE(std::any_of(entries.cbegin(),
                                entries.cend(),
                                [](const QString& entry)
                                { return entry.startsWith(QStringLiteral("source.files/shared/images/")); }));
    }

    TEST(ProjectExporterTest, DoesNotOverwriteExistingOutput)
    {
        QTemporaryDir temporary;
        ASSERT_TRUE(temporary.isValid());
        const QString projectPath = QDir(temporary.path()).filePath(QStringLiteral("source.plascan"));
        const QString archivePath = QDir(temporary.path()).filePath(QStringLiteral("portable.zip"));
        xjw::common::project::ProjectSession session;
        QString error;
        ASSERT_TRUE(session.create(projectPath, QStringLiteral("source"), &error)) << qPrintable(error);
        writeFile(archivePath, QByteArrayLiteral("existing-output"));

        EXPECT_FALSE(xjw::common::project::ProjectExporter::exportPortableProject(projectPath, archivePath, &error));
        EXPECT_TRUE(error.contains(QStringLiteral("已存在")));
        QFile file(archivePath);
        ASSERT_TRUE(file.open(QIODevice::ReadOnly));
        EXPECT_EQ(file.readAll(), QByteArrayLiteral("existing-output"));
    }

} // namespace
