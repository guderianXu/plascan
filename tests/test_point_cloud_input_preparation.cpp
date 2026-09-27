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

    bool writePointCloudPly(const QString& path, const std::vector<std::array<float, 3>>& points)
    {
        QByteArray bytes("ply\nformat ascii 1.0\nelement vertex ");
        bytes.append(QByteArray::number(static_cast<qlonglong>(points.size())));
        bytes.append("\nproperty float x\nproperty float y\nproperty float z\nend_header\n");
        for (const auto& point : points)
        {
            bytes.append(QByteArray::number(point[0]));
            bytes.append(' ');
            bytes.append(QByteArray::number(point[1]));
            bytes.append(' ');
            bytes.append(QByteArray::number(point[2]));
            bytes.append('\n');
        }
        return writeBytes(path, bytes);
    }

    QJsonObject observation(int cameraIndex, const QString& imageId, const QString& imagePath)
    {
        return {{QStringLiteral("camera_index"), cameraIndex},
                {QStringLiteral("image_id"), imageId},
                {QStringLiteral("image_path"), imagePath}};
    }

    QJsonObject imageRow(int cameraIndex, const QString& imageId, const QString& imagePath)
    {
        return {{QStringLiteral("camera_index"), cameraIndex},
                {QStringLiteral("image_id"), imageId},
                {QStringLiteral("image_path"), imagePath},
                {QStringLiteral("image_name"), QFileInfo(imagePath).fileName()}};
    }

    QJsonArray twoImageRows(const QString& firstImage, const QString& secondImage)
    {
        return {imageRow(0, QStringLiteral("canonical-first"), firstImage),
                imageRow(1, QStringLiteral("canonical-second"), secondImage)};
    }

    QJsonObject v3Sidecar(const QJsonArray& images, const QJsonArray& points)
    {
        return {{QStringLiteral("schema"), QStringLiteral("plascan.sfm_sparse_points.v3")},
                {QStringLiteral("images"), images},
                {QStringLiteral("points"), points}};
    }

    xjw::mvs::CameraView boundView(const QString& imagePath,
                                   const char* instanceId,
                                   const char* imageId)
    {
        xjw::mvs::CameraView view;
        view.imagePath = imagePath.toStdString();
        const placamera::FrameId frame("local");
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId("point-cloud-input-definition"),
            placamera::FrameIntrinsics{100.0, 100.0, 32.0, 24.0},
            {}, placamera::PixelConvention::PixelCenter, frame);
        view.camera = std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(instanceId), placamera::ImageId(imageId), definition,
            placamera::ImageSize{64, 48},
            placamera::Pose::create(frame, {0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
        return view;
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
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{1.0F, 2.0F, 3.0F}}));

    QJsonArray observations;
    observations.append(observation(1, QStringLiteral("canonical-second"), second_image));
    observations.append(observation(0, QStringLiteral("canonical-first"), first_image));
    observations.append(observation(1, QStringLiteral("canonical-second"), second_image));
    QJsonObject point{{QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}},
                      {QStringLiteral("observations"), observations}};
    QJsonObject ignored_point{{QStringLiteral("point_xyz"), QJsonArray{4.0, 5.0, 6.0}},
                              {QStringLiteral("observations"),
                               QJsonArray{observation(1, QStringLiteral("canonical-second"), second_image)}}};
    const QJsonObject reconstruction_region{
        {QStringLiteral("rotation"), QJsonArray{0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}},
        {QStringLiteral("center"), QJsonArray{4.0, 5.0, 6.0}},
        {QStringLiteral("size"), QJsonArray{7.0, 8.0, 9.0}}};
    QJsonObject root = v3Sidecar(twoImageRows(first_image, second_image), QJsonArray{point, ignored_point});
    root.insert(QStringLiteral("reconstruction_region"), reconstruction_region);
    ASSERT_TRUE(writeBytes(sidecar, QJsonDocument(root).toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
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

TEST(PointCloudInputPreparationTest, KeepsOnlyTracksWhosePlyVerticesSurviveMiddlePointPrune)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("first.png"));
    const QString second_image = directory.filePath(QStringLiteral("second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writeBytes(first_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writeBytes(second_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{0.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}));

    const QJsonArray observations{observation(0, QStringLiteral("canonical-first"), first_image),
                                  observation(1, QStringLiteral("canonical-second"), second_image)};
    const QJsonObject first{{QStringLiteral("point_xyz"), QJsonArray{0.0, 0.0, 0.0}},
                            {QStringLiteral("observations"), observations}};
    const QJsonObject removed{{QStringLiteral("point_xyz"), QJsonArray{1.0, 0.0, 0.0}},
                              {QStringLiteral("observations"), observations}};
    const QJsonObject last{{QStringLiteral("point_xyz"), QJsonArray{2.0, 0.0, 0.0}},
                           {QStringLiteral("observations"), observations}};
    ASSERT_TRUE(writeBytes(sidecar,
                           QJsonDocument(v3Sidecar(twoImageRows(first_image, second_image),
                                                   QJsonArray{first, removed, last}))
                               .toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    EXPECT_EQ(result.cloud.points, (std::vector<std::array<float, 3>>{{0.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}}));
    EXPECT_EQ(result.cloud.observingViewIndices, (std::vector<std::vector<int>>{{0, 1}, {0, 1}}));
    EXPECT_EQ(result.cloud.minPt, (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
    EXPECT_EQ(result.cloud.maxPt, (std::array<float, 3>{2.0F, 0.0F, 0.0F}));
}

TEST(PointCloudInputPreparationTest, ResolvesCanonicalImageIdsBeforeNumericCameraIndices)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("first.png"));
    const QString second_image = directory.filePath(QStringLiteral("second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writeBytes(first_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writeBytes(second_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{1.0F, 2.0F, 3.0F}}));

    const QJsonArray images{
        QJsonObject{{QStringLiteral("camera_index"), 0},
                    {QStringLiteral("image_id"), QStringLiteral("canonical-first")},
                    {QStringLiteral("image_path"), first_image}},
        QJsonObject{{QStringLiteral("camera_index"), 1},
                    {QStringLiteral("image_id"), QStringLiteral("canonical-second")},
                    {QStringLiteral("image_path"), second_image}}};
    const QJsonArray observations{
        QJsonObject{{QStringLiteral("camera_index"), 1},
                    {QStringLiteral("image_id"), QStringLiteral("canonical-second")},
                    {QStringLiteral("image_path"), second_image}},
        QJsonObject{{QStringLiteral("camera_index"), 0},
                    {QStringLiteral("image_id"), QStringLiteral("canonical-first")},
                    {QStringLiteral("image_path"), first_image}}};
    const QJsonObject point{{QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}},
                            {QStringLiteral("observations"), observations}};
    ASSERT_TRUE(writeBytes(sidecar,
                           QJsonDocument(v3Sidecar(images, QJsonArray{point})).toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    ASSERT_EQ(result.cloud.observingViewIndices.size(), 1U);
    EXPECT_EQ(result.cloud.observingViewIndices.front(), (std::vector<int>{0, 1}));
}

TEST(PointCloudInputPreparationTest, KeepsCanonicalIdentityWhenSidecarPathsAreStale)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("relocated-first.png"));
    const QString second_image = directory.filePath(QStringLiteral("relocated-second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writeBytes(first_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writeBytes(second_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{1.0F, 2.0F, 3.0F}}));

    const QString stale_first = QStringLiteral("/old/project/first.png");
    const QString stale_second = QStringLiteral("/old/project/second.png");
    const QJsonArray observations{
        observation(0, QStringLiteral("canonical-first"), stale_first),
        observation(1, QStringLiteral("canonical-second"), stale_second)};
    const QJsonObject point{{QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}},
                            {QStringLiteral("observations"), observations}};
    ASSERT_TRUE(writeBytes(sidecar,
                           QJsonDocument(v3Sidecar(
                                             QJsonArray{imageRow(0, QStringLiteral("canonical-first"), stale_first),
                                                        imageRow(1, QStringLiteral("canonical-second"), stale_second)},
                                             QJsonArray{point}))
                               .toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    ASSERT_EQ(result.cloud.observingViewIndices.size(), 1U);
    EXPECT_EQ(result.cloud.observingViewIndices.front(), (std::vector<int>{0, 1}));
}

TEST(PointCloudInputPreparationTest, RejectsCanonicalImageIdAndCameraIndexConflict)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("first.png"));
    const QString second_image = directory.filePath(QStringLiteral("second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writeBytes(first_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writeBytes(second_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{1.0F, 2.0F, 3.0F}}));

    const QJsonObject point{
        {QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}},
        {QStringLiteral("observations"),
         QJsonArray{QJsonObject{{QStringLiteral("camera_index"), 0},
                                {QStringLiteral("image_id"), QStringLiteral("canonical-second")},
                                {QStringLiteral("image_path"), second_image}}}}};
    const QJsonArray images{
        QJsonObject{{QStringLiteral("camera_index"), 0},
                    {QStringLiteral("image_id"), QStringLiteral("canonical-first")},
                    {QStringLiteral("image_path"), first_image}},
        QJsonObject{{QStringLiteral("camera_index"), 1},
                    {QStringLiteral("image_id"), QStringLiteral("canonical-second")},
                    {QStringLiteral("image_path"), second_image}}};
    ASSERT_TRUE(writeBytes(sidecar,
                           QJsonDocument(v3Sidecar(images, QJsonArray{point})).toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("camera_index")));
}

TEST(PointCloudInputPreparationTest, RejectsPlyThatCannotBeVerifiedAsAnOrderedSidecarSubset)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("first.png"));
    const QString second_image = directory.filePath(QStringLiteral("second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}}));
    ASSERT_TRUE(writeBytes(
        sidecar,
        QJsonDocument(v3Sidecar(twoImageRows(first_image, second_image),
                                QJsonArray{QJsonObject{{QStringLiteral("point_xyz"), QJsonArray{0.0, 0.0, 0.0}}},
                                           QJsonObject{{QStringLiteral("point_xyz"), QJsonArray{1.0, 0.0, 0.0}}}}))
            .toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("有序子集")));
}

TEST(PointCloudInputPreparationTest, DoesNotRecoverTrackFromADeletedNearbyPoint)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("first.png"));
    const QString second_image = directory.filePath(QStringLiteral("second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writeBytes(first_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writeBytes(second_image, QByteArrayLiteral("image")));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{0.00005F, 0.0F, 0.0F}}));

    const QJsonArray observations{observation(0, QStringLiteral("canonical-first"), first_image),
                                  observation(1, QStringLiteral("canonical-second"), second_image)};
    ASSERT_TRUE(writeBytes(
        sidecar,
        QJsonDocument(v3Sidecar(twoImageRows(first_image, second_image),
                                QJsonArray{QJsonObject{{QStringLiteral("point_xyz"), QJsonArray{0.0, 0.0, 0.0}},
                                                       {QStringLiteral("observations"), observations}},
                                           QJsonObject{{QStringLiteral("point_xyz"), QJsonArray{0.00005, 0.0, 0.0}},
                                                       {QStringLiteral("observations"), observations}}}))
            .toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
    ASSERT_EQ(result.cloud.points.size(), 1U);
    EXPECT_EQ(result.cloud.points.front(), (std::array<float, 3>{0.00005F, 0.0F, 0.0F}));
}

TEST(PointCloudInputPreparationTest, RejectsAmbiguousDuplicateCoordinateMapping)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("first.png"));
    const QString second_image = directory.filePath(QStringLiteral("second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{1.0F, 2.0F, 3.0F}}));
    ASSERT_TRUE(writeBytes(
        sidecar,
        QJsonDocument(v3Sidecar(twoImageRows(first_image, second_image),
                                QJsonArray{QJsonObject{{QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}},
                                                       {QStringLiteral("observations"), QJsonArray{}}},
                                           QJsonObject{{QStringLiteral("point_xyz"), QJsonArray{1.0, 2.0, 3.0}},
                                                       {QStringLiteral("observations"), QJsonArray{}}}}))
            .toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("不唯一")));
}

TEST(PointCloudInputPreparationTest, RejectsSidecarWithFewerPointsThanPly)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString first_image = directory.filePath(QStringLiteral("first.png"));
    const QString second_image = directory.filePath(QStringLiteral("second.png"));
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    const QString sidecar = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}}));
    ASSERT_TRUE(writeBytes(
        sidecar,
        QJsonDocument(v3Sidecar(twoImageRows(first_image, second_image),
                                QJsonArray{QJsonObject{{QStringLiteral("point_xyz"), QJsonArray{0.0, 0.0, 0.0}}}}))
            .toJson(QJsonDocument::Compact)));

    const std::vector<xjw::mvs::CameraView> views{
        boundView(first_image, "instance-first", "canonical-first"),
        boundView(second_image, "instance-second", "canonical-second")};
    const auto result =
        xjw::core::project::preparePointCloudInput(sparse_cloud, views, plapoint::ProcessingDevice::CPU, sidecar);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("少于")));
}

TEST(PointCloudInputPreparationTest, ReportsMissingRequestedSidecar)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString sparse_cloud = directory.filePath(QStringLiteral("sparse.ply"));
    ASSERT_TRUE(writePointCloudPly(sparse_cloud, {{0.0F, 0.0F, 0.0F}}));

    const auto result = xjw::core::project::preparePointCloudInput(
        sparse_cloud, {}, plapoint::ProcessingDevice::CPU, directory.filePath(QStringLiteral("missing.json")));

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("不存在")));
}
