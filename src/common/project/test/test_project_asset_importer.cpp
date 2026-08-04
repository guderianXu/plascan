#include "project/ProjectAssetImporter.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

namespace
{

bool writeFile(const QString &path, const QByteArray &contents)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath()))
    {
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        return false;
    }
    return file.write(contents) == contents.size();
}

} // namespace

TEST(ProjectAssetImporterTest, ImportsMetashapeObjModelWithMaterialAndTexture)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());

    const QString sourceDirectory =
        QDir(temporaryDirectory.path()).filePath(QStringLiteral("Metashape 导出/模型"));
    const QString projectRoot =
        QDir(temporaryDirectory.path()).filePath(QStringLiteral("project.files/Chunk 1"));
    const QString objPath = QDir(sourceDirectory).filePath(QStringLiteral("模型.obj"));
    const QString mtlPath = QDir(sourceDirectory).filePath(QStringLiteral("模型.mtl"));
    const QString texturePath =
        QDir(sourceDirectory).filePath(QStringLiteral("textures/albedo.png"));

    ASSERT_TRUE(writeFile(
        objPath,
        "# Generated with Agisoft Metashape\n"
        "mtllib \xE6\xA8\xA1\xE5\x9E\x8B.mtl\n"
        "v 0 0 0 1 0 0\n"
        "v 1 0 0 0 1 0\n"
        "v 0 1 0 0 0 1\n"
        "f 1 2 3\n"));
    ASSERT_TRUE(writeFile(mtlPath, "newmtl material\nmap_Kd textures/albedo.png\n"));
    ASSERT_TRUE(writeFile(texturePath, QByteArrayLiteral("fake-png")));

    xjw::common::project::ProjectAssetImportRequest request;
    request.type = xjw::common::project::ProjectAssetType::Model;
    request.sourcePath = objPath;
    request.projectRoot = projectRoot;
    const auto result = xjw::common::project::ProjectAssetImporter::importAsset(request);

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.format, QStringLiteral("obj"));
    EXPECT_EQ(result.vertexCount, 3);
    EXPECT_EQ(result.faceCount, 1);
    EXPECT_TRUE(result.hasVertexColors);
    const std::string warningText = result.warnings.join(QLatin1Char('\n')).toStdString();
    EXPECT_TRUE(result.hasMaterial) << warningText;
    EXPECT_TRUE(result.hasTexture) << warningText;
    EXPECT_TRUE(QFileInfo::exists(result.importedPath));
    EXPECT_TRUE(result.importedPath.startsWith(projectRoot));
    EXPECT_EQ(result.resultArrayKey, QStringLiteral("model_results"));
    EXPECT_EQ(result.resultPathKey, QStringLiteral("final_model_path"));
    EXPECT_EQ(result.projectRecord.value(QStringLiteral("model_obj")).toString(),
              result.importedPath);
    EXPECT_TRUE(result.projectRecord.value(QStringLiteral("source_path")).isUndefined());
    ASSERT_EQ(result.importedDependencies.size(), 3) << warningText;
    for (const QString &dependency : result.importedDependencies)
    {
        EXPECT_TRUE(QFileInfo::exists(dependency)) << dependency.toStdString();
        EXPECT_TRUE(dependency.startsWith(result.importDirectory));
    }
}

TEST(ProjectAssetImporterTest, ImportsColoredObjPointCloudWithoutFaces)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());

    const QString objPath =
        QDir(temporaryDirectory.path()).filePath(QStringLiteral("export/point_cloud.obj"));
    ASSERT_TRUE(writeFile(objPath,
                          "v 0 0 0 0.5 0.5 0.5\n"
                          "vn 0 0 1\n"
                          "v 1 2 3 1.0 0.0 0.0\n"
                          "vn 0 0 1\n"));

    xjw::common::project::ProjectAssetImportRequest request;
    request.type = xjw::common::project::ProjectAssetType::PointCloud;
    request.sourcePath = objPath;
    request.projectRoot =
        QDir(temporaryDirectory.path()).filePath(QStringLiteral("project.files/Chunk 1"));
    const auto result = xjw::common::project::ProjectAssetImporter::importAsset(request);

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.vertexCount, 2);
    EXPECT_EQ(result.faceCount, 0);
    EXPECT_TRUE(result.hasVertexColors);
    EXPECT_TRUE(QFileInfo::exists(result.importedPath));
    EXPECT_EQ(result.resultArrayKey, QStringLiteral("dense_cloud_results"));
    EXPECT_EQ(result.projectRecord.value(QStringLiteral("dense_cloud_xyz")).toString(),
              result.importedPath);
    EXPECT_EQ(result.projectRecord.value(QStringLiteral("point_count")).toInt(), 2);
}

TEST(ProjectAssetImporterTest, RejectsPointOnlyObjAsModel)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());

    const QString objPath =
        QDir(temporaryDirectory.path()).filePath(QStringLiteral("point_cloud.obj"));
    ASSERT_TRUE(writeFile(objPath, "v 0 0 0\nv 1 1 1\n"));

    xjw::common::project::ProjectAssetImportRequest request;
    request.type = xjw::common::project::ProjectAssetType::Model;
    request.sourcePath = objPath;
    request.projectRoot = temporaryDirectory.path();
    const auto result = xjw::common::project::ProjectAssetImporter::importAsset(request);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("不包含面")));
}

TEST(ProjectAssetImporterTest, InspectsAsciiPlyAndXyzPointClouds)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());

    const QString plyPath =
        QDir(temporaryDirectory.path()).filePath(QStringLiteral("cloud.ply"));
    ASSERT_TRUE(writeFile(
        plyPath,
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 0\nproperty list uchar int vertex_indices\n"
        "end_header\n0 0 0 255 0 0\n1 1 1 0 255 0\n"));
    const QString xyzPath =
        QDir(temporaryDirectory.path()).filePath(QStringLiteral("cloud.xyz"));
    ASSERT_TRUE(writeFile(xyzPath,
                          "# x y z r g b\n"
                          "0 0 0 255 0 0\n"
                          "1 1 1 0 255 0\n"));

    for (const QString &sourcePath : {plyPath, xyzPath})
    {
        xjw::common::project::ProjectAssetImportRequest request;
        request.type = xjw::common::project::ProjectAssetType::PointCloud;
        request.sourcePath = sourcePath;
        request.projectRoot = temporaryDirectory.path();
        const auto result =
            xjw::common::project::ProjectAssetImporter::importAsset(request);

        ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
        EXPECT_EQ(result.vertexCount, 2);
        EXPECT_TRUE(result.hasVertexColors);
        EXPECT_TRUE(QFileInfo::exists(result.importedPath));
    }
}

TEST(ProjectAssetImporterTest, RejectsMetashapeInternalOc3)
{
    QTemporaryDir temporaryDirectory;
    ASSERT_TRUE(temporaryDirectory.isValid());

    const QString oc3Path =
        QDir(temporaryDirectory.path()).filePath(QStringLiteral("dense_cloud.oc3"));
    ASSERT_TRUE(writeFile(oc3Path, QByteArrayLiteral("proprietary")));

    xjw::common::project::ProjectAssetImportRequest request;
    request.type = xjw::common::project::ProjectAssetType::PointCloud;
    request.sourcePath = oc3Path;
    request.projectRoot = temporaryDirectory.path();
    const auto result = xjw::common::project::ProjectAssetImporter::importAsset(request);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("OBJ、PLY 或 XYZ")));
}

TEST(ProjectAssetImporterTest, ImportsConfiguredMetashapeExportDirectory)
{
    const QString exportDirectory =
        qEnvironmentVariable("PLASCAN_METASHAPE_EXPORT_DIR").trimmed();
    if (exportDirectory.isEmpty())
    {
        GTEST_SKIP() << "PLASCAN_METASHAPE_EXPORT_DIR is not configured";
    }

    const QString pointCloudPath =
        QDir(exportDirectory).filePath(QStringLiteral("点云/hyb.obj"));
    const QString modelPath =
        QDir(exportDirectory).filePath(QStringLiteral("模型/模型.obj"));
    ASSERT_TRUE(QFileInfo::exists(pointCloudPath));
    ASSERT_TRUE(QFileInfo::exists(modelPath));

    QTemporaryDir projectDirectory;
    ASSERT_TRUE(projectDirectory.isValid());

    xjw::common::project::ProjectAssetImportRequest pointCloudRequest;
    pointCloudRequest.type = xjw::common::project::ProjectAssetType::PointCloud;
    pointCloudRequest.sourcePath = pointCloudPath;
    pointCloudRequest.projectRoot = projectDirectory.path();
    const auto pointCloudResult =
        xjw::common::project::ProjectAssetImporter::importAsset(pointCloudRequest);
    ASSERT_TRUE(pointCloudResult.success)
        << pointCloudResult.errorMessage.toStdString();
    EXPECT_GT(pointCloudResult.vertexCount, 0);
    EXPECT_EQ(pointCloudResult.faceCount, 0);
    EXPECT_TRUE(pointCloudResult.hasVertexColors);

    xjw::common::project::ProjectAssetImportRequest modelRequest;
    modelRequest.type = xjw::common::project::ProjectAssetType::Model;
    modelRequest.sourcePath = modelPath;
    modelRequest.projectRoot = projectDirectory.path();
    const auto modelResult =
        xjw::common::project::ProjectAssetImporter::importAsset(modelRequest);
    ASSERT_TRUE(modelResult.success) << modelResult.errorMessage.toStdString();
    EXPECT_GT(modelResult.vertexCount, 0);
    EXPECT_GT(modelResult.faceCount, 0);
    EXPECT_TRUE(modelResult.hasMaterial);
    EXPECT_TRUE(QFileInfo::exists(modelResult.importedPath));
}
