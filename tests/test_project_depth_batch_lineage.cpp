#include "project/support/ProjectDepthBatchLineage.h"

#include "ProjectCameraIO.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/project/CameraProjectRecords.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace
{

    bool writeBytes(const QString& path, const QByteArray& bytes)
    {
        QFile file(path);
        return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
    }

    QJsonObject metadataForSparsePly(const QString& sparsePlyPath)
    {
        const QString imagePath = QStringLiteral("/stable/image_0.tif");
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 10.0});

        QJsonObject metadata{
            {QStringLiteral("images"),
             QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("image-0")},
                                    {QStringLiteral("path"), imagePath},
                                    {QStringLiteral("samples"), 1024},
                                    {QStringLiteral("lines"), 768}}}},
            {QStringLiteral("camera_definitions"), QJsonArray{}},
            {QStringLiteral("camera_instances"), QJsonArray{}},
            {QStringLiteral("aerial_triangulation_results"),
             QJsonArray{QJsonObject{
                 {QStringLiteral("run_id"), QStringLiteral("run-a")},
                 {QStringLiteral("selected_images"), QJsonArray{imagePath}},
                 {QStringLiteral("files"), QJsonObject{{QStringLiteral("sparse_cloud_xyz"), sparsePlyPath}}}}}}};
        QJsonObject cameraMetadata = xjw::common::project::serializeFramePinholeNumericState(camera);
        cameraMetadata.insert(QStringLiteral("image_width"), 1024);
        cameraMetadata.insert(QStringLiteral("image_height"), 768);
        const auto update = xjw::camera_project::CameraProjectRecords::upsertByImagePath(
            &metadata, QMap<QString, QJsonObject>{{imagePath, cameraMetadata}});
        EXPECT_TRUE(update.ok()) << update.errors.join(';').toStdString();
        return metadata;
    }

} // namespace

TEST(ProjectDepthBatchLineageTest, ChangesSignatureWhenSameSizeSparsePlyContentChanges)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString sparsePlyPath = directory.filePath(QStringLiteral("sfm_sparse.ply"));
    ASSERT_TRUE(writeBytes(sparsePlyPath, QByteArrayLiteral("ply-content-a")));
    const qint64 originalSize = QFileInfo(sparsePlyPath).size();

    const QJsonObject metadata = metadataForSparsePly(sparsePlyPath);
    const QString first = xjw::gui::project::canonicalProjectDepthInputSignature(metadata, -1);
    ASSERT_FALSE(first.isEmpty());

    ASSERT_TRUE(writeBytes(sparsePlyPath, QByteArrayLiteral("ply-content-b")));
    ASSERT_EQ(QFileInfo(sparsePlyPath).size(), originalSize);
    const QString second = xjw::gui::project::canonicalProjectDepthInputSignature(metadata, -1);

    EXPECT_NE(first, second);
}

TEST(ProjectDepthBatchLineageTest, RejectsPathOnlyCameraIdentity)
{
    const QString imagePath = QStringLiteral("/stable/image_0.tif");
    const QJsonObject metadata{{QStringLiteral("images"), QJsonArray{QJsonObject{{QStringLiteral("path"), imagePath}}}},
                               {QStringLiteral("aerial_triangulation_results"),
                                QJsonArray{QJsonObject{{QStringLiteral("selected_images"), QJsonArray{imagePath}}}}}};

    EXPECT_TRUE(xjw::gui::project::canonicalProjectDepthInputSignature(metadata, -1).isEmpty());
}

TEST(ProjectDepthBatchLineageTest, RejectsAmbiguousSelectedPath)
{
    const QString imagePath = QStringLiteral("/stable/image_0.tif");
    QJsonObject metadata = metadataForSparsePly(QStringLiteral("/missing/sparse.ply"));
    QJsonArray images = metadata.value(QStringLiteral("images")).toArray();
    QJsonObject duplicate = images.at(0).toObject();
    duplicate[QStringLiteral("image_uuid")] = QStringLiteral("image-1");
    duplicate[QStringLiteral("path")] = imagePath;
    images.append(duplicate);
    metadata[QStringLiteral("images")] = images;
    metadata[QStringLiteral("aerial_triangulation_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("selected_images"), QJsonArray{imagePath}}}};

    EXPECT_TRUE(xjw::gui::project::canonicalProjectDepthInputSignature(metadata, -1).isEmpty());
}
