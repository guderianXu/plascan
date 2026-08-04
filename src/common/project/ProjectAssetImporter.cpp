#include "ProjectAssetImporter.h"
#include "ProjectAssetInspection.h"

#include <QDir>
#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QRegularExpression>
#include <QUuid>

#include <algorithm>

namespace xjw::common::project
{
namespace
{

QString cleanAbsolutePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

QString sanitizedBaseName(const QString &filePath)
{
    QString name = QFileInfo(filePath).completeBaseName();
    name.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]+")),
                 QStringLiteral("_"));
    name = name.left(48);
    return name.isEmpty() ? QStringLiteral("asset") : name;
}

bool copyFilePreservingRelativePath(const QString &sourcePath,
                                    const QString &sourceRoot,
                                    const QString &destinationRoot,
                                    QString *destinationPath,
                                    QString *errorMessage)
{
    const QString relativePath = QDir(sourceRoot).relativeFilePath(sourcePath);
    if (relativePath.startsWith(QStringLiteral("..")))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("资源位于导入目录之外: %1").arg(sourcePath);
        }
        return false;
    }

    const QString targetPath = QDir(destinationRoot).filePath(relativePath);
    if (!QDir().mkpath(QFileInfo(targetPath).absolutePath()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建导入目录: %1")
                                .arg(QFileInfo(targetPath).absolutePath());
        }
        return false;
    }
    if (!QFile::copy(sourcePath, targetPath))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("复制资源失败: %1 -> %2")
                                .arg(sourcePath, targetPath);
        }
        return false;
    }
    if (destinationPath)
    {
        *destinationPath = QDir::cleanPath(targetPath);
    }
    return true;
}

} // namespace

ProjectAssetImportResult ProjectAssetImporter::importAsset(
    const ProjectAssetImportRequest &request)
{
    ProjectAssetImportResult result;
    result.sourcePath = cleanAbsolutePath(request.sourcePath);
    result.format = QFileInfo(result.sourcePath).suffix().toLower();

    const QFileInfo sourceInfo(result.sourcePath);
    if (!sourceInfo.exists() || !sourceInfo.isFile())
    {
        result.errorMessage = QStringLiteral("导入文件不存在: %1").arg(result.sourcePath);
        return result;
    }
    if (request.projectRoot.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("项目资源目录为空，无法导入。");
        return result;
    }
    if (request.type == ProjectAssetType::Model && result.format == QStringLiteral("xyz"))
    {
        result.errorMessage = QStringLiteral("XYZ 不包含模型面，请导入 OBJ 或带面的 PLY。");
        return result;
    }

    detail::ProjectAssetInspection inspection;
    if (!detail::inspectProjectAsset(result.sourcePath,
                                     result.format,
                                     &inspection,
                                     &result.errorMessage))
    {
        return result;
    }
    if (request.type == ProjectAssetType::Model && inspection.faceCount <= 0)
    {
        result.errorMessage = QStringLiteral("模型不包含面，请从 Metashape 导出 OBJ 或带面的 PLY。");
        return result;
    }

    result.vertexCount = inspection.vertexCount;
    result.faceCount = inspection.faceCount;
    result.hasVertexColors = inspection.hasVertexColors;

    QStringList sourceDependencies;
    if (result.format == QStringLiteral("obj"))
    {
        sourceDependencies = detail::collectObjDependencies(result.sourcePath,
                                                             inspection,
                                                             &result.warnings,
                                                             &result.hasMaterial,
                                                             &result.hasTexture);
    }

    const QString typeDirectory = request.type == ProjectAssetType::PointCloud
        ? QStringLiteral("point_clouds")
        : QStringLiteral("models");
    const QString importedRoot = QDir(request.projectRoot).filePath(
        QStringLiteral("assets/imported/%1").arg(typeDirectory));
    if (!QDir().mkpath(importedRoot))
    {
        result.errorMessage = QStringLiteral("无法创建项目导入目录: %1").arg(importedRoot);
        return result;
    }

    const QString token = QUuid::createUuid().toString(QUuid::WithoutBraces).left(12);
    const QString finalDirectory = QDir(importedRoot).filePath(
        QStringLiteral("%1_%2").arg(sanitizedBaseName(result.sourcePath), token));
    const QString temporaryDirectory = finalDirectory + QStringLiteral(".importing");
    if (!QDir().mkpath(temporaryDirectory))
    {
        result.errorMessage = QStringLiteral("无法创建临时导入目录: %1")
                                  .arg(temporaryDirectory);
        return result;
    }

    const QString sourceRoot = sourceInfo.absolutePath();
    QString temporaryMainPath;
    if (!copyFilePreservingRelativePath(result.sourcePath,
                                        sourceRoot,
                                        temporaryDirectory,
                                        &temporaryMainPath,
                                        &result.errorMessage))
    {
        QDir(temporaryDirectory).removeRecursively();
        return result;
    }

    QStringList temporaryDependencies{temporaryMainPath};
    for (const QString &dependency : sourceDependencies)
    {
        QString copiedPath;
        if (!copyFilePreservingRelativePath(dependency,
                                            sourceRoot,
                                            temporaryDirectory,
                                            &copiedPath,
                                            &result.errorMessage))
        {
            QDir(temporaryDirectory).removeRecursively();
            return result;
        }
        temporaryDependencies.append(copiedPath);
    }

    QDir parentDirectory(importedRoot);
    if (!parentDirectory.rename(QFileInfo(temporaryDirectory).fileName(),
                                QFileInfo(finalDirectory).fileName()))
    {
        QDir(temporaryDirectory).removeRecursively();
        result.errorMessage = QStringLiteral("无法提交导入资源目录: %1")
                                  .arg(finalDirectory);
        return result;
    }

    result.importDirectory = QDir::cleanPath(finalDirectory);
    const auto finalizePath = [&](const QString &temporaryPath)
    {
        const QString relative = QDir(temporaryDirectory).relativeFilePath(temporaryPath);
        return QDir::cleanPath(QDir(finalDirectory).filePath(relative));
    };
    result.importedPath = finalizePath(temporaryMainPath);
    for (const QString &temporaryDependency : temporaryDependencies)
    {
        result.importedDependencies.append(finalizePath(temporaryDependency));
    }

    QJsonObject record;
    record[QStringLiteral("created_at")] =
        QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    record[QStringLiteral("source")] = QStringLiteral("metashape_import");
    record[QStringLiteral("imported")] = true;
    record[QStringLiteral("source_file_name")] = sourceInfo.fileName();
    record[QStringLiteral("import_format")] = result.format;
    record[QStringLiteral("import_directory")] = result.importDirectory;
    record[QStringLiteral("vertex_count")] = static_cast<double>(result.vertexCount);
    record[QStringLiteral("face_count")] = static_cast<double>(result.faceCount);
    record[QStringLiteral("has_vertex_colors")] = result.hasVertexColors;
    record[QStringLiteral("imported_dependencies")] =
        QJsonArray::fromStringList(result.importedDependencies);

    if (request.type == ProjectAssetType::Model)
    {
        record[QStringLiteral("kind")] = QStringLiteral("mesh");
        record[QStringLiteral("result_type")] = QStringLiteral("mesh");
        record[QStringLiteral("final_model_path")] = result.importedPath;
        record[QStringLiteral("final_model_format")] = result.format;
        record[QStringLiteral("has_material")] = result.hasMaterial;
        record[QStringLiteral("textured")] = result.hasTexture;
        record[result.format == QStringLiteral("obj")
                   ? QStringLiteral("model_obj")
                   : QStringLiteral("model_ply")] = result.importedPath;
        for (const QString &dependency : result.importedDependencies)
        {
            const QString suffix = QFileInfo(dependency).suffix().toLower();
            if (suffix == QStringLiteral("mtl") &&
                !record.contains(QStringLiteral("model_mtl")))
            {
                record[QStringLiteral("model_mtl")] = dependency;
            }
            else if ((suffix == QStringLiteral("png") ||
                      suffix == QStringLiteral("jpg") ||
                      suffix == QStringLiteral("jpeg") ||
                      suffix == QStringLiteral("tif") ||
                      suffix == QStringLiteral("tiff")) &&
                     !record.contains(QStringLiteral("texture_image")))
            {
                record[QStringLiteral("texture_image")] = dependency;
            }
        }
        result.resultArrayKey = QStringLiteral("model_results");
        result.resultPathKey = QStringLiteral("final_model_path");
    }
    else
    {
        record[QStringLiteral("kind")] = QStringLiteral("dense_cloud");
        record[QStringLiteral("result_type")] = QStringLiteral("dense_cloud");
        record[QStringLiteral("dense_cloud_xyz")] = result.importedPath;
        record[QStringLiteral("point_count")] = static_cast<double>(result.vertexCount);
        result.resultArrayKey = QStringLiteral("dense_cloud_results");
        result.resultPathKey = QStringLiteral("dense_cloud_xyz");
    }
    result.projectRecord = record;
    result.success = true;
    return result;
}

} // namespace xjw::common::project
