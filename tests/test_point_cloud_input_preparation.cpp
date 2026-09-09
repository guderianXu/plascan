#include "PointCloudInputPreparation.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
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

    QJsonObject observation(int imageId, const QString& imagePath)
    {
        return {{QStringLiteral("image_id"), imageId}, {QStringLiteral("image_path"), imagePath}};
    }

} // namespace

TEST(PointCloudInputPreparationTest, LoadsTrackedSidecarByObservationImagePath)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("first.png"));
    const QString second_image = directory.filePath(QStringLiteral("second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writeBytes(first_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writeBytes(second_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writeBytes(sparse_cloud, QByteArrayLiteral("ply")));

    QJsonArray observations;
    observations.append(observation(101, second_image));
    observations.append(observation(205, first_image));
    observations.append(observation(101, second_image));
    QJsonObject point{{QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}},
                      {QStringLiteral("observations"), observations}};
    QJsonObject ignored_point{{QStringLiteral("point_xyz"), QJsonArray{4.0, 5.0, 6.0}},
                              {QStringLiteral("observations"), QJsonArray{observation(101, second_image)}}};
    const QJsonObject reconstruction_region{
        {QStringLiteral("rotation"), QJsonArray{0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}},
        {QStringLiteral("center"), QJsonArray{4.0, 5.0, 6.0}},
        {QStringLiteral("size"), QJsonArray{7.0, 8.0, 9.0}}};
    QJsonObject root{{QStringLiteral("points"), QJsonArray{point, ignored_point}},
                     {QStringLiteral("reconstruction_region"), reconstruction_region}};
    ASSERT_TRUE(writeBytes(sidecar, QJsonDocument(root).toJson(QJsonDocument::Compact)));

    std::vector<xjw::mvs::CameraView> views(2);
    views[0].imagePath = first_image.toStdString();
    views[1].imagePath = second_image.toStdString();
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    ASSERT_EQ(result.cloud.points.size(), 1U);
    ASSERT_EQ(result.cloud.trackIds, std::vector<std::uint32_t>{0U});
    ASSERT_EQ(result.cloud.observingViewIndices, (std::vector<std::vector<int>>{{0, 1}}));
    EXPECT_EQ(result.cloud.minPt, (std::array<float, 3>{1.0F, 2.0F, 3.0F}));
    EXPECT_EQ(result.cloud.maxPt, result.cloud.minPt);
    EXPECT_TRUE(result.cloud.reconstructionRegionSpecified);
    EXPECT_EQ(result.cloud.reconstructionRegionRotation,
              (std::array<double, 9>{0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}));
    EXPECT_EQ(result.cloud.reconstructionRegionCenter, (std::array<double, 3>{4.0, 5.0, 6.0}));
    EXPECT_EQ(result.cloud.reconstructionRegionSize, (std::array<double, 3>{7.0, 8.0, 9.0}));
}

TEST(PointCloudInputPreparationTest, ReportsMissingRequestedSidecar)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    ASSERT_TRUE(writeBytes(sparse_cloud, QByteArrayLiteral("ply")));

    const auto result = xjw::core::project::preparePointCloudInput(
        sparse_cloud, {}, plapoint::ProcessingDevice::CPU, directory.filePath(QStringLiteral("missing.json")));

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("不存在")));
}
