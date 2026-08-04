#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace xjw::common::project
{

enum class ProjectAssetType
{
    PointCloud,
    Model
};

struct ProjectAssetImportRequest
{
    ProjectAssetType type = ProjectAssetType::PointCloud;
    QString sourcePath;
    QString projectRoot;
};

struct ProjectAssetImportResult
{
    bool success = false;
    QString sourcePath;
    QString importedPath;
    QString importDirectory;
    QString format;
    qint64 vertexCount = 0;
    qint64 faceCount = 0;
    bool hasVertexColors = false;
    bool hasMaterial = false;
    bool hasTexture = false;
    QStringList importedDependencies;
    QStringList warnings;
    QString errorMessage;
    QString resultArrayKey;
    QString resultPathKey;
    QJsonObject projectRecord;
};

class ProjectAssetImporter final
{
public:
    static ProjectAssetImportResult importAsset(
        const ProjectAssetImportRequest &request);
};

} // namespace xjw::common::project
