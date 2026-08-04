#include "ProjectAssetInspection.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSet>
#include <QTextStream>

namespace xjw::common::project::detail
{
namespace
{

QString cleanAbsolutePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

bool isPathInside(const QString &path, const QString &directory)
{
    const QString relativePath = QDir(cleanAbsolutePath(directory)).relativeFilePath(
        cleanAbsolutePath(path));
    return !QFileInfo(relativePath).isAbsolute() &&
        relativePath != QStringLiteral("..") &&
        !relativePath.startsWith(QStringLiteral("../"));
}

QString unquote(QString value)
{
    value = value.trimmed();
    if (value.size() >= 2 && value.startsWith(QLatin1Char('"')) &&
        value.endsWith(QLatin1Char('"')))
    {
        return value.mid(1, value.size() - 2);
    }
    return value;
}

QStringList splitFields(const QString &line)
{
    return line.simplified().split(QLatin1Char(' '), Qt::SkipEmptyParts);
}

bool inspectObj(const QString &path,
                ProjectAssetInspection *inspection,
                QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取 OBJ 文件: %1").arg(path);
        }
        return false;
    }

    QTextStream stream(&file);
    while (!stream.atEnd())
    {
        const QString line = stream.readLine().trimmed();
        if (line.startsWith(QStringLiteral("v ")))
        {
            ++inspection->vertexCount;
            inspection->hasVertexColors = inspection->hasVertexColors ||
                splitFields(line).size() >= 7;
        }
        else if (line.startsWith(QStringLiteral("f ")))
        {
            ++inspection->faceCount;
        }
        else if (line.startsWith(QStringLiteral("mtllib "), Qt::CaseInsensitive))
        {
            const QString library = unquote(line.mid(7));
            if (!library.isEmpty() && !inspection->materialLibraries.contains(library))
            {
                inspection->materialLibraries.append(library);
            }
        }
    }

    if (inspection->vertexCount <= 0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("OBJ 文件不包含顶点: %1").arg(path);
        }
        return false;
    }
    return true;
}

bool inspectPly(const QString &path,
                ProjectAssetInspection *inspection,
                QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取 PLY 文件: %1").arg(path);
        }
        return false;
    }

    if (file.readLine().trimmed() != QByteArrayLiteral("ply"))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("文件不是有效的 PLY: %1").arg(path);
        }
        return false;
    }

    bool inVertexElement = false;
    bool hasRed = false;
    bool hasGreen = false;
    bool hasBlue = false;
    bool headerComplete = false;
    while (!file.atEnd())
    {
        const QString line = QString::fromLatin1(file.readLine().trimmed());
        if (line == QStringLiteral("end_header"))
        {
            headerComplete = true;
            break;
        }

        const QStringList fields = splitFields(line);
        if (fields.size() >= 3 && fields.at(0) == QStringLiteral("element"))
        {
            bool ok = false;
            const qint64 count = fields.at(2).toLongLong(&ok);
            if (!ok || count < 0)
            {
                continue;
            }
            inVertexElement = fields.at(1) == QStringLiteral("vertex");
            if (inVertexElement)
            {
                inspection->vertexCount = count;
            }
            else if (fields.at(1) == QStringLiteral("face"))
            {
                inspection->faceCount = count;
            }
        }
        else if (inVertexElement && fields.size() >= 3 &&
                 fields.at(0) == QStringLiteral("property"))
        {
            const QString propertyName = fields.constLast().toLower();
            hasRed = hasRed || propertyName == QStringLiteral("red") ||
                propertyName == QStringLiteral("r");
            hasGreen = hasGreen || propertyName == QStringLiteral("green") ||
                propertyName == QStringLiteral("g");
            hasBlue = hasBlue || propertyName == QStringLiteral("blue") ||
                propertyName == QStringLiteral("b");
        }
    }

    inspection->hasVertexColors = hasRed && hasGreen && hasBlue;
    if (!headerComplete || inspection->vertexCount <= 0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("PLY 头缺失或不包含顶点: %1").arg(path);
        }
        return false;
    }
    return true;
}

bool inspectXyz(const QString &path,
                ProjectAssetInspection *inspection,
                QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取 XYZ 文件: %1").arg(path);
        }
        return false;
    }

    QTextStream stream(&file);
    while (!stream.atEnd())
    {
        const QString line = stream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
        {
            continue;
        }
        const QStringList fields = splitFields(line);
        if (fields.size() < 3)
        {
            continue;
        }
        bool xOk = false;
        bool yOk = false;
        bool zOk = false;
        fields.at(0).toDouble(&xOk);
        fields.at(1).toDouble(&yOk);
        fields.at(2).toDouble(&zOk);
        if (!xOk || !yOk || !zOk)
        {
            continue;
        }
        ++inspection->vertexCount;
        inspection->hasVertexColors = inspection->hasVertexColors || fields.size() >= 6;
    }

    if (inspection->vertexCount <= 0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("XYZ 文件不包含有效点: %1").arg(path);
        }
        return false;
    }
    return true;
}

QString textureReferenceFromLine(const QString &line)
{
    const QStringList fields = splitFields(line);
    if (fields.size() < 2)
    {
        return QString();
    }
    const int firstValue = line.indexOf(fields.at(1));
    if (firstValue < 0)
    {
        return QString();
    }
    QString value = line.mid(firstValue).trimmed();
    if (value.startsWith(QLatin1Char('-')))
    {
        value = fields.constLast();
    }
    return unquote(value);
}

QStringList textureReferencesFromMtl(const QString &mtlPath, QString *errorMessage)
{
    QFile file(mtlPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法读取材质文件: %1").arg(mtlPath);
        }
        return {};
    }

    static const QSet<QString> textureDirectives = {
        QStringLiteral("map_ka"), QStringLiteral("map_kd"),
        QStringLiteral("map_ks"), QStringLiteral("map_bump"),
        QStringLiteral("bump"), QStringLiteral("disp"),
        QStringLiteral("decal"), QStringLiteral("norm")
    };
    QStringList references;
    QTextStream stream(&file);
    while (!stream.atEnd())
    {
        const QString line = stream.readLine().trimmed();
        const QStringList fields = splitFields(line);
        if (fields.isEmpty() || !textureDirectives.contains(fields.constFirst().toLower()))
        {
            continue;
        }
        const QString reference = textureReferenceFromLine(line);
        if (!reference.isEmpty() && !references.contains(reference))
        {
            references.append(reference);
        }
    }
    return references;
}

void appendDependency(const QString &path, QStringList *dependencies)
{
    const QString cleanPath = cleanAbsolutePath(path);
    if (!dependencies->contains(cleanPath))
    {
        dependencies->append(cleanPath);
    }
}

} // namespace

bool inspectProjectAsset(const QString &path,
                         const QString &format,
                         ProjectAssetInspection *inspection,
                         QString *errorMessage)
{
    if (format == QStringLiteral("obj"))
    {
        return inspectObj(path, inspection, errorMessage);
    }
    if (format == QStringLiteral("ply"))
    {
        return inspectPly(path, inspection, errorMessage);
    }
    if (format == QStringLiteral("xyz"))
    {
        return inspectXyz(path, inspection, errorMessage);
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral(
            "不支持的导入格式 .%1；请从 Metashape 导出 OBJ、PLY 或 XYZ。")
                            .arg(format.isEmpty()
                                     ? QStringLiteral("(无扩展名)")
                                     : format);
    }
    return false;
}

QStringList collectObjDependencies(const QString &objPath,
                                   const ProjectAssetInspection &inspection,
                                   QStringList *warnings,
                                   bool *hasMaterial,
                                   bool *hasTexture)
{
    const QDir sourceDirectory = QFileInfo(objPath).absoluteDir();
    QStringList dependencies;
    for (const QString &libraryReference : inspection.materialLibraries)
    {
        const QString mtlPath = cleanAbsolutePath(sourceDirectory.filePath(libraryReference));
        if (!isPathInside(mtlPath, sourceDirectory.absolutePath()))
        {
            warnings->append(
                QStringLiteral("已跳过工程目录外的材质引用: %1").arg(libraryReference));
            continue;
        }
        if (!QFileInfo::exists(mtlPath))
        {
            warnings->append(QStringLiteral("材质文件不存在: %1").arg(mtlPath));
            continue;
        }

        *hasMaterial = true;
        appendDependency(mtlPath, &dependencies);
        QString mtlError;
        const QStringList textures = textureReferencesFromMtl(mtlPath, &mtlError);
        if (!mtlError.isEmpty())
        {
            warnings->append(mtlError);
        }
        const QDir materialDirectory = QFileInfo(mtlPath).absoluteDir();
        for (const QString &textureReference : textures)
        {
            const QString texturePath =
                cleanAbsolutePath(materialDirectory.filePath(textureReference));
            if (!isPathInside(texturePath, sourceDirectory.absolutePath()))
            {
                warnings->append(
                    QStringLiteral("已跳过工程目录外的纹理引用: %1").arg(textureReference));
                continue;
            }
            if (!QFileInfo::exists(texturePath))
            {
                warnings->append(QStringLiteral("纹理文件不存在: %1").arg(texturePath));
                continue;
            }
            *hasTexture = true;
            appendDependency(texturePath, &dependencies);
        }
    }
    return dependencies;
}

} // namespace xjw::common::project::detail
