#include "project/support/ProjectDepthBatchLineage.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
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
        return {{QStringLiteral("images"), QJsonArray{QJsonObject{{QStringLiteral("path"), imagePath}}}},
                {QStringLiteral("aerial_triangulation_results"),
                 QJsonArray{QJsonObject{
                     {QStringLiteral("run_id"), QStringLiteral("run-a")},
                     {QStringLiteral("selected_images"), QJsonArray{imagePath}},
                     {QStringLiteral("files"), QJsonObject{{QStringLiteral("sparse_cloud_xyz"), sparsePlyPath}}}}}}};
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
