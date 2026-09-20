#include "ProjectPackageLayout.h"

#include "ProjectPathBridge.h"

#include <array>
#include <filesystem>
#include <optional>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSaveFile>
#include <QRegularExpression>
#include <QXmlStreamReader>
#include <QXmlStreamWriter>

namespace xjw::common::project
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

QString absoluteProjectPath(const QString &projectPath)
{
    return QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath());
}

QString projectBaseName(const QString &projectPath)
{
    return QFileInfo(projectPath).completeBaseName();
}

QString expectedRelativeArchivePath(const QString &projectPath)
{
    return QStringLiteral("%1.files/project.zip")
        .arg(projectBaseName(projectPath));
}

bool looksLikeXmlDescriptor(const QString &projectPath)
{
    QFile file(projectPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }
    return file.peek(128).trimmed().startsWith('<');
}

QString xmlError(const QString &projectPath, const QXmlStreamReader &xml)
{
    return QStringLiteral("项目描述文件 XML 无效 %1（第 %2 行，第 %3 列）: %4")
        .arg(projectPath)
        .arg(xml.lineNumber())
        .arg(xml.columnNumber())
        .arg(xml.errorString());
}

bool removeDirectoryIfEmpty(const QString &path, QString *errorMessage)
{
    QDir directory(path);
    if (!directory.exists()
        || !directory.entryList(
                QDir::AllEntries | QDir::NoDotAndDotDot).isEmpty())
    {
        return true;
    }
    if (QDir().rmdir(directory.absolutePath()))
    {
        return true;
    }

    setError(errorMessage,
             QStringLiteral("无法清理空的项目目录: %1")
                 .arg(directory.absolutePath()));
    return false;
}

} // namespace

QString ProjectPackageLayout::dataDirectory(const QString &projectPath)
{
    const auto layout = path_bridge::projectLayout(projectPath);
    return layout ? path_bridge::toQtPath(layout->filesDirectory()) : QString();
}

QString ProjectPackageLayout::metadataArchivePath(const QString &projectPath)
{
    const auto layout = path_bridge::projectLayout(projectPath);
    return layout ? path_bridge::toQtPath(layout->metadataArchive()) : QString();
}

QString ProjectPackageLayout::sharedDirectory(const QString &projectPath)
{
    const auto layout = path_bridge::projectLayout(projectPath);
    return layout ? path_bridge::toQtPath(layout->sharedDirectory()) : QString();
}

QString ProjectPackageLayout::sharedImagesDirectory(const QString &projectPath)
{
    const auto layout = path_bridge::projectLayout(projectPath);
    return layout ? path_bridge::toQtPath(layout->sharedImagesDirectory()) : QString();
}

QString ProjectPackageLayout::chunkDirectory(const QString &projectPath,
                                             int directoryNumber)
{
    const auto layout = path_bridge::projectLayout(projectPath);
    return layout ? path_bridge::toQtPath(layout->chunkDirectory(directoryNumber)) : QString();
}

QString ProjectPackageLayout::chunkArchivePath(const QString &projectPath,
                                               int directoryNumber)
{
    const auto layout = path_bridge::projectLayout(projectPath);
    return layout ? path_bridge::toQtPath(layout->chunkArchive(directoryNumber)) : QString();
}

bool ProjectPackageLayout::pruneEmptyOptionalDirectories(
    const QString &projectPath,
    int directoryNumber,
    QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    const QString root = chunkDirectory(projectPath, directoryNumber);
    if (root.isEmpty())
    {
        setError(errorMessage, QStringLiteral("Chunk 目录编号无效"));
        return false;
    }

    const auto layout = path_bridge::projectLayout(projectPath);
    const auto chunk = layout
        ? xjw::common::plafs::PlaChunkLayout::open(layout->chunkDirectory(directoryNumber))
        : std::nullopt;
    if (!chunk)
    {
        setError(errorMessage, QStringLiteral("Chunk 目录编号无效"));
        return false;
    }

    const std::array<std::filesystem::path, 13> optionalDirectories{
        chunk->reconstructionDirectory() / "terrain" / "products",
        chunk->reconstructionDirectory() / "terrain",
        chunk->reconstructionDirectory() / "model",
        chunk->reconstructionDirectory() / "mvs",
        chunk->reconstructionDirectory() / "sparse",
        chunk->reconstructionDirectory(),
        // 统一匹配模块只产生逐影像 `.pimatch`，不再创建特征目录或旧成对匹配目录。
        chunk->imageMatchesDirectory(),
        chunk->tiePointsDirectory(),
        chunk->controlPointsDirectory(),
        chunk->importedDirectory(),
        chunk->assetsDirectory(),
        chunk->bundleAdjustDirectory(),
        chunk->reportsDirectory()
    };
    for (const auto &directory : optionalDirectories)
    {
        if (!removeDirectoryIfEmpty(
                path_bridge::toQtPath(directory), errorMessage))
        {
            return false;
        }
    }

    if (!removeDirectoryIfEmpty(
            path_bridge::toQtPath(layout->sharedImagesDirectory()),
            errorMessage)
        || !removeDirectoryIfEmpty(
            path_bridge::toQtPath(layout->sharedDirectory()),
            errorMessage))
    {
        return false;
    }
    return true;
}

bool ProjectPackageLayout::isChunkDirectoryName(const QString &name)
{
    static const QRegularExpression expression(
        QStringLiteral("^[1-9][0-9]*$"));
    return expression.match(name).hasMatch();
}

QString ProjectPackageLayout::resourcesDirectory(const QString &projectPath)
{
    const auto layout = path_bridge::projectLayout(projectPath);
    return layout ? path_bridge::toQtPath(layout->resourcesDirectory(1)) : QString();
}

bool ProjectPackageLayout::parseDescriptor(const QString &projectPath,
                                           QString *relativeArchivePath,
                                           QString *errorMessage)
{
    QFile file(projectPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法读取项目描述文件 %1: %2")
                     .arg(projectPath, file.errorString()));
        return false;
    }

    QXmlStreamReader xml(&file);
    if (!xml.readNextStartElement())
    {
        setError(errorMessage, xml.hasError()
            ? xmlError(projectPath, xml)
            : QStringLiteral("项目描述文件缺少 document 根节点"));
        return false;
    }
    if (xml.name() != QStringLiteral("document"))
    {
        setError(errorMessage,
                 QStringLiteral("项目描述文件根节点必须是 document，实际为 %1")
                     .arg(xml.name().toString()));
        return false;
    }

    const auto attributes = xml.attributes();
    const QString version =
        attributes.value(QStringLiteral("version")).toString();
    const QString type = attributes.value(QStringLiteral("type")).toString();
    const QString path = QDir::fromNativeSeparators(
        attributes.value(QStringLiteral("path")).toString());
    if (type != QString::fromLatin1(DescriptorType))
    {
        setError(errorMessage,
                 QStringLiteral("项目描述文件类型无效，期望 %1，实际为 %2")
                     .arg(QString::fromLatin1(DescriptorType),
                          type.isEmpty() ? QStringLiteral("<空>") : type));
        return false;
    }
    if (version != QString::fromLatin1(DescriptorVersion))
    {
        setError(errorMessage,
                 QStringLiteral("不支持的项目描述版本，期望 %1，实际为 %2")
                     .arg(QString::fromLatin1(DescriptorVersion),
                          version.isEmpty() ? QStringLiteral("<空>") : version));
        return false;
    }

    const QString expected = expectedRelativeArchivePath(projectPath);
    if (path != QString::fromLatin1(ArchiveRelativePath)
        && path != expected)
    {
        setError(errorMessage,
                 QStringLiteral("项目描述文件包含不安全或不匹配的路径: %1")
                     .arg(path));
        return false;
    }
    if (relativeArchivePath)
    {
        *relativeArchivePath = expected;
    }

    while (!xml.atEnd())
    {
        const QXmlStreamReader::TokenType token = xml.readNext();
        if (token == QXmlStreamReader::StartElement)
        {
            setError(errorMessage,
                     QStringLiteral("项目描述文件不允许包含子节点: %1")
                         .arg(xml.name().toString()));
            return false;
        }
        if (token == QXmlStreamReader::Characters && !xml.isWhitespace())
        {
            setError(errorMessage,
                     QStringLiteral("项目描述文件包含 document 之外的文本"));
            return false;
        }
    }
    if (xml.hasError())
    {
        setError(errorMessage, xmlError(projectPath, xml));
        return false;
    }
    return true;
}

bool ProjectPackageLayout::isDescriptor(const QString &projectPath,
                                        QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!looksLikeXmlDescriptor(projectPath))
    {
        setError(errorMessage,
                 QStringLiteral(
                     "仅支持版本 %1 的分体 PlaScan 工程描述文件: %2")
                     .arg(QString::fromLatin1(DescriptorVersion),
                          absoluteProjectPath(projectPath)));
        return false;
    }
    return parseDescriptor(projectPath, nullptr, errorMessage);
}

bool ProjectPackageLayout::writeDescriptor(const QString &projectPath,
                                           QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    QSaveFile file(absoluteProjectPath(projectPath));
    if (!file.open(QIODevice::WriteOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法写入项目描述文件 %1: %2")
                     .arg(projectPath, file.errorString()));
        return false;
    }

    QXmlStreamWriter xml(&file);
    xml.setAutoFormatting(true);
    xml.writeStartDocument(QStringLiteral("1.0"));
    xml.writeStartElement(QStringLiteral("document"));
    xml.writeAttribute(QStringLiteral("version"),
                       QString::fromLatin1(DescriptorVersion));
    xml.writeAttribute(QStringLiteral("type"),
                        QString::fromLatin1(DescriptorType));
    xml.writeAttribute(QStringLiteral("path"),
                       QString::fromLatin1(ArchiveRelativePath));
    xml.writeEndElement();
    xml.writeEndDocument();
    if (xml.hasError() || !file.commit())
    {
        setError(errorMessage,
                 QStringLiteral("提交项目描述文件失败: %1")
                     .arg(file.errorString()));
        return false;
    }
    return true;
}

bool ProjectPackageLayout::ensureSplitLayout(const QString &projectPath,
                                             QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    const QString absolutePath = absoluteProjectPath(projectPath);
    if (!QFileInfo::exists(absolutePath))
    {
        setError(errorMessage,
                 QStringLiteral("项目文件不存在: %1").arg(absolutePath));
        return false;
    }

    QString descriptorError;
    if (!isDescriptor(absolutePath, &descriptorError))
    {
        setError(errorMessage, descriptorError);
        return false;
    }

    const QString archivePath = metadataArchivePath(absolutePath);
    if (!QFileInfo(archivePath).isFile())
    {
        setError(errorMessage,
                 QStringLiteral("项目元数据归档不存在: %1")
                     .arg(archivePath));
        return false;
    }
    return true;
}

QString ProjectPackageLayout::resolveMetadataArchive(
    const QString &projectPath,
    QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    QString descriptorError;
    if (!isDescriptor(projectPath, &descriptorError))
    {
        setError(errorMessage, descriptorError);
        return {};
    }

    const QString archivePath = metadataArchivePath(projectPath);
    if (!QFileInfo(archivePath).isFile())
    {
        setError(errorMessage,
                 QStringLiteral("项目元数据归档不存在: %1")
                     .arg(archivePath));
        return {};
    }
    return archivePath;
}

} // namespace xjw::common::project
