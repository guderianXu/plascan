#include "project/ProjectExporter.h"

#include "project/PlascanArchive.h"
#include "project/ProjectChunkStore.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectWorkspaceStore.h"
#include "project/PortableProjectFormat.h"

#include <QCryptographicHash>
#include <QByteArrayView>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QScopeGuard>
#include <QUuid>

namespace xjw::common::project
{
    namespace
    {

        void setError(QString* errorMessage, const QString& message)
        {
            if (errorMessage)
            {
                *errorMessage = message;
            }
        }

        bool copyFile(const QString& source, const QString& destination, QString* errorMessage)
        {
            QDir().mkpath(QFileInfo(destination).absolutePath());
            QFile::remove(destination);
            if (!QFile::copy(source, destination))
            {
                setError(errorMessage, QStringLiteral("无法复制 %1 到 %2").arg(source, destination));
                return false;
            }
            return true;
        }

        bool shouldSkip(const QString& relative)
        {
            return relative.contains(QStringLiteral(".plascan_tmp/")) ||
                   relative.endsWith(QStringLiteral(".plascan_tmp")) || relative.endsWith(QStringLiteral(".lock")) ||
                   relative.endsWith(QStringLiteral(".recover"));
        }

        bool copyTree(const QString& source, const QString& destination, QString* errorMessage)
        {
            const QFileInfo sourceInfo(source);
            if (!sourceInfo.isDir())
            {
                return copyFile(source, destination, errorMessage);
            }
            QDir sourceRoot(source);
            QDirIterator iterator(source, QDir::Files, QDirIterator::Subdirectories);
            while (iterator.hasNext())
            {
                const QString path = iterator.next();
                const QString relative = QDir::fromNativeSeparators(sourceRoot.relativeFilePath(path));
                if (shouldSkip(relative))
                {
                    continue;
                }
                if (!copyFile(path, QDir(destination).filePath(relative), errorMessage))
                {
                    return false;
                }
            }
            return true;
        }

        QString sha256(const QString& path, QString* errorMessage)
        {
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly))
            {
                setError(errorMessage, QStringLiteral("无法读取外部影像: %1").arg(path));
                return {};
            }
            QCryptographicHash hash(QCryptographicHash::Sha256);
            QByteArray buffer(1024 * 1024, Qt::Uninitialized);
            while (!file.atEnd())
            {
                const qint64 count = file.read(buffer.data(), buffer.size());
                if (count < 0)
                {
                    setError(errorMessage, QStringLiteral("读取外部影像失败: %1").arg(path));
                    return {};
                }
                hash.addData(QByteArrayView(buffer.constData(), count));
            }
            return QString::fromLatin1(hash.result().toHex());
        }

        bool rewriteExternalImages(const QString& projectPath, QString* errorMessage)
        {
            ProjectChunkStore store(projectPath);
            const auto chunks = store.chunks(errorMessage);
            if (chunks.isEmpty())
            {
                return false;
            }
            const QString dataRoot = ProjectPackageLayout::dataDirectory(projectPath);
            for (const ProjectChunkRecord& chunk : chunks)
            {
                QJsonObject document;
                if (!store.readChunkDocument(chunk.directory, &document, errorMessage))
                {
                    return false;
                }
                QJsonObject files =
                    document.value(QString::fromLatin1(PortableProjectFormat::ProjectFilesSection)).toObject();
                QJsonArray images = files.value(QStringLiteral("images")).toArray();
                bool changed = false;
                for (qsizetype index = 0; index < images.size(); ++index)
                {
                    QJsonObject image = images.at(index).toObject();
                    if (image.value(QStringLiteral("type")).toString() != QStringLiteral("external"))
                    {
                        continue;
                    }
                    const QString source =
                        QDir::cleanPath(QFileInfo(image.value(QStringLiteral("path")).toString()).absoluteFilePath());
                    if (!QFileInfo(source).isFile())
                    {
                        setError(errorMessage, QStringLiteral("导出缺少外部影像: %1").arg(source));
                        return false;
                    }
                    const QString digest = sha256(source, errorMessage);
                    if (digest.isEmpty())
                    {
                        return false;
                    }
                    const QString entry =
                        QStringLiteral("shared/images/%1/%2").arg(digest, QFileInfo(source).fileName());
                    const QString destination = QDir(dataRoot).filePath(entry);
                    if (!QFileInfo::exists(destination) && !copyFile(source, destination, errorMessage))
                    {
                        return false;
                    }
                    image[QStringLiteral("path")] = PortableProjectFormat::resourceUriForEntry(entry);
                    image[QStringLiteral("type")] = QStringLiteral("shared");
                    images[index] = image;
                    changed = true;
                }
                if (!changed)
                {
                    continue;
                }
                files[QStringLiteral("images")] = images;
                QJsonObject results =
                    document.value(QString::fromLatin1(PortableProjectFormat::ProjectResultsSection)).toObject();
                QJsonObject resourceIndex;
                ProjectWorkspaceStore workspace(projectPath, chunk.directory);
                if (!workspace.prepareSplitMetadata(&files, &results, &resourceIndex, errorMessage) ||
                    !store.writeChunkSections(
                        chunk.directory,
                        {{QString::fromLatin1(PortableProjectFormat::ProjectFilesSection), files},
                         {QString::fromLatin1(PortableProjectFormat::ProjectResultsSection), results},
                         {QString::fromLatin1(PortableProjectFormat::ResourceIndexSection), resourceIndex}},
                        errorMessage))
                {
                    return false;
                }
            }
            return true;
        }

    } // namespace

    bool ProjectExporter::createStagingProject(const QString& sourceProjectPath,
                                                const QString& outputZipPath,
                                                QString* stagingProjectPath,
                                                QString* errorMessage)
    {
        if (errorMessage)
        {
            errorMessage->clear();
        }
        if (stagingProjectPath)
        {
            stagingProjectPath->clear();
        }
        const QString source = QDir::cleanPath(QFileInfo(sourceProjectPath).absoluteFilePath());
        const QString output = QDir::cleanPath(QFileInfo(outputZipPath).absoluteFilePath());
        if (!QFileInfo(source).isFile())
        {
            setError(errorMessage, QStringLiteral("项目描述文件不存在: %1").arg(source));
            return false;
        }
        const QFileInfo sourceInfo(source);
        const QString base = sourceInfo.completeBaseName();
        const QString staging = QDir(QFileInfo(output).absolutePath())
                                    .filePath(QStringLiteral(".%1.export-%2")
                                                  .arg(base, QUuid::createUuid().toString(QUuid::WithoutBraces)));
        const QString isolatedProject = QDir(staging).filePath(sourceInfo.fileName());
        if (!QDir().mkpath(staging) || !copyFile(source, isolatedProject, errorMessage) ||
            !copyTree(ProjectPackageLayout::dataDirectory(source),
                      QDir(staging).filePath(base + QStringLiteral(".files")),
                      errorMessage))
        {
            QDir(staging).removeRecursively();
            return false;
        }
        if (stagingProjectPath)
        {
            *stagingProjectPath = isolatedProject;
        }
        return true;
    }

    bool ProjectExporter::exportPortableProject(const QString& sourceProjectPath,
                                                const QString& outputZipPath,
                                                QString* errorMessage)
    {
        if (errorMessage)
        {
            errorMessage->clear();
        }
        const QString source = QDir::cleanPath(QFileInfo(sourceProjectPath).absoluteFilePath());
        const QString output = QDir::cleanPath(QFileInfo(outputZipPath).absoluteFilePath());
        if (!QFileInfo(source).isFile())
        {
            setError(errorMessage, QStringLiteral("项目描述文件不存在: %1").arg(source));
            return false;
        }
        if (QFileInfo::exists(output))
        {
            setError(errorMessage, QStringLiteral("导出目标已存在，不会覆盖: %1").arg(output));
            return false;
        }
        const QFileInfo sourceInfo(source);
        QString isolatedProject;
        if (!createStagingProject(source, output, &isolatedProject, errorMessage))
        {
            return false;
        }
        const QString staging = QFileInfo(isolatedProject).absolutePath();
        const auto cleanup = qScopeGuard([&staging]() { QDir(staging).removeRecursively(); });
        if (!rewriteExternalImages(isolatedProject, errorMessage))
        {
            return false;
        }

        const QString temporaryZip =
            output + QStringLiteral(".tmp-") + QUuid::createUuid().toString(QUuid::WithoutBraces);
        const auto cleanupZip = qScopeGuard([&temporaryZip]() { QFile::remove(temporaryZip); });
        if (!PlascanArchive::createArchive(temporaryZip, {{QStringLiteral("placeholder"), QByteArray()}}, errorMessage))
        {
            return false;
        }
        PlascanArchive archive(temporaryZip, PlascanArchivePathType::DirectArchive);
        QVector<QPair<QString, QString>> entries;
        QDir stagingRoot(staging);
        QDirIterator iterator(staging, QDir::Files, QDirIterator::Subdirectories);
        while (iterator.hasNext())
        {
            const QString path = iterator.next();
            entries.append(qMakePair(QDir::fromNativeSeparators(stagingRoot.relativeFilePath(path)), path));
        }
        if (!archive.updateFileEntries(
                entries, {QStringLiteral("placeholder")}, PlascanArchiveCompression::Store, errorMessage) ||
            !QFile::rename(temporaryZip, output))
        {
            if (!errorMessage || errorMessage->isEmpty())
            {
                setError(errorMessage, QStringLiteral("无法原子提交导出 ZIP: %1").arg(output));
            }
            return false;
        }
        return true;
    }

} // namespace xjw::common::project
