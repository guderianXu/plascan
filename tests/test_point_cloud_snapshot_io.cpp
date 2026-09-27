#include "PointCloudSnapshotIO.h"

#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include <plapoint/io/obj_io.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/io/xyz_io.h>

namespace
{

using xjw::gui::point_cloud::PointCloudSnapshotGuard;
using xjw::gui::point_cloud::PointCloudSnapshotStageResult;
using xjw::gui::point_cloud::SnapshotCloud;
using xjw::gui::point_cloud::stagePointCloudSnapshot;

SnapshotCloud makeCloud()
{
    SnapshotCloud cloud(3);
    cloud.points()(0, 0) = -1.25f;
    cloud.points()(0, 1) = 2.5f;
    cloud.points()(0, 2) = 3.75f;
    cloud.points()(1, 0) = 4.0f;
    cloud.points()(1, 1) = -5.5f;
    cloud.points()(1, 2) = 6.25f;
    cloud.points()(2, 0) = 7.5f;
    cloud.points()(2, 1) = 8.0f;
    cloud.points()(2, 2) = -9.25f;

    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(3, 3);
    for (plamatrix::Index row = 0; row < 3; ++row)
    {
        colors(row, 0) = static_cast<std::uint8_t>(10 + row);
        colors(row, 1) = static_cast<std::uint8_t>(20 + row);
        colors(row, 2) = static_cast<std::uint8_t>(30 + row);
    }
    cloud.setColors(std::move(colors));
    return cloud;
}

void writeOriginalFile(const QString &path)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    ASSERT_EQ(file.write("original point cloud"), 20);
}

void expectCloudCoordinates(const SnapshotCloud &cloud)
{
    ASSERT_EQ(cloud.size(), 3U);
    EXPECT_FLOAT_EQ(cloud.points()(0, 0), -1.25f);
    EXPECT_FLOAT_EQ(cloud.points()(1, 1), -5.5f);
    EXPECT_FLOAT_EQ(cloud.points()(2, 2), -9.25f);
}

void expectStagedBesideFinal(const PointCloudSnapshotStageResult &result,
                             const QString &finalPath)
{
    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    ASSERT_NE(result.guard, nullptr);
    EXPECT_TRUE(result.guard->isPending());
    EXPECT_TRUE(QFileInfo::exists(result.temporaryPath()));
    EXPECT_EQ(QFileInfo(result.temporaryPath()).absolutePath(),
              QFileInfo(finalPath).absolutePath());

    QFile final_file(finalPath);
    ASSERT_TRUE(final_file.open(QIODevice::ReadOnly));
    EXPECT_EQ(final_file.readAll(), QByteArray("original point cloud"));
}

TEST(PointCloudSnapshotIOTest, StagesAndCommitsXyzWithoutTouchingFinalFileEarly)
{
    QTemporaryDir temporary_dir;
    ASSERT_TRUE(temporary_dir.isValid());
    const QString final_path = QDir(temporary_dir.path()).filePath(
        QStringLiteral("edited.xyz"));
    writeOriginalFile(final_path);

    const PointCloudSnapshotStageResult result =
        stagePointCloudSnapshot(final_path, makeCloud());
    expectStagedBesideFinal(result, final_path);

    QString error_message;
    ASSERT_TRUE(result.commit(&error_message)) << error_message.toStdString();
    EXPECT_FALSE(QFileInfo::exists(result.temporaryPath()));
    const auto loaded = plapoint::io::readXyz<float>(
        xjw::common::io::toNativeNarrowPath(final_path));
    ASSERT_NE(loaded, nullptr);
    expectCloudCoordinates(*loaded);
}

TEST(PointCloudSnapshotIOTest, CommitsObjUsingObjSerialization)
{
    QTemporaryDir temporary_dir;
    ASSERT_TRUE(temporary_dir.isValid());
    const QString final_path = QDir(temporary_dir.path()).filePath(
        QStringLiteral("edited.OBJ"));
    writeOriginalFile(final_path);

    const PointCloudSnapshotStageResult result =
        stagePointCloudSnapshot(final_path, makeCloud());
    expectStagedBesideFinal(result, final_path);

    QString error_message;
    ASSERT_TRUE(result.commit(&error_message)) << error_message.toStdString();
    const auto loaded = plapoint::io::readObj<float>(
        xjw::common::io::toNativeNarrowPath(final_path));
    ASSERT_NE(loaded, nullptr);
    expectCloudCoordinates(*loaded);
}

TEST(PointCloudSnapshotIOTest, CommitsBinaryLittleEndianPly)
{
    QTemporaryDir temporary_dir;
    ASSERT_TRUE(temporary_dir.isValid());
    const QString final_path = QDir(temporary_dir.path()).filePath(
        QStringLiteral("edited.ply"));
    writeOriginalFile(final_path);

    auto cloud = makeCloud();
    cloud.setNormals(plamatrix::MatrixXf::Constant(3, 3, 0.5f));
    cloud.setIntensities(plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic>::Constant(3, 1, 513));
    cloud.setScalarFields({"quality"}, plamatrix::MatrixXf::Constant(3, 1, 0.75f));
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces << 2, 1, 0;
    cloud.setFaces(std::move(faces));
    const PointCloudSnapshotStageResult result = stagePointCloudSnapshot(final_path, cloud);
    expectStagedBesideFinal(result, final_path);

    QString error_message;
    ASSERT_TRUE(result.commit(&error_message)) << error_message.toStdString();
    QFile file(final_path);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    const QByteArray header = file.read(512);
    EXPECT_TRUE(header.contains("format binary_little_endian 1.0"));

    const auto loaded = plapoint::io::readPly<float>(
        xjw::common::io::toNativeNarrowPath(final_path));
    ASSERT_NE(loaded, nullptr);
    expectCloudCoordinates(*loaded);
    ASSERT_TRUE(loaded->hasColors());
    ASSERT_TRUE(loaded->hasNormals());
    ASSERT_TRUE(loaded->hasIntensities());
    ASSERT_TRUE(loaded->hasScalarField("quality"));
    ASSERT_TRUE(loaded->hasFaces());
    EXPECT_EQ(loaded->colors()->coeff(2, 1), 22);
    EXPECT_FLOAT_EQ(loaded->normals()->coeff(1, 2), 0.5f);
    EXPECT_EQ(loaded->intensities()->coeff(1, 0), 513);
    EXPECT_FLOAT_EQ(loaded->scalarFields()->coeff(1, 0), 0.75f);
    EXPECT_EQ(loaded->faces()->rows(), 1);
    EXPECT_EQ(loaded->faces()->coeff(0, 0), 2);
    EXPECT_EQ(loaded->faces()->coeff(0, 2), 0);
}

TEST(PointCloudSnapshotIOTest, RejectsUnknownFinalExtension)
{
    QTemporaryDir temporary_dir;
    ASSERT_TRUE(temporary_dir.isValid());
    const QString final_path = QDir(temporary_dir.path()).filePath(
        QStringLiteral("edited.bin"));

    const PointCloudSnapshotStageResult result =
        stagePointCloudSnapshot(final_path, makeCloud());

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.guard, nullptr);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("不支持")));
    EXPECT_FALSE(QFileInfo::exists(final_path));
}

TEST(PointCloudSnapshotIOTest, PreCancelledStageDoesNotCreateFiles)
{
    QTemporaryDir temporary_dir;
    ASSERT_TRUE(temporary_dir.isValid());
    const QString final_path = QDir(temporary_dir.path()).filePath(
        QStringLiteral("edited.ply"));
    std::atomic_bool cancelled{true};

    const PointCloudSnapshotStageResult result =
        stagePointCloudSnapshot(final_path, makeCloud(), &cancelled);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.guard, nullptr);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("取消")));
    EXPECT_FALSE(QFileInfo::exists(final_path));
    EXPECT_TRUE(QDir(temporary_dir.path()).entryList(QDir::Files).isEmpty());
}

TEST(PointCloudSnapshotIOTest, DiscardRemovesStagedFileAndKeepsFinalFile)
{
    QTemporaryDir temporary_dir;
    ASSERT_TRUE(temporary_dir.isValid());
    const QString final_path = QDir(temporary_dir.path()).filePath(
        QStringLiteral("edited.xyz"));
    writeOriginalFile(final_path);

    const PointCloudSnapshotStageResult result =
        stagePointCloudSnapshot(final_path, makeCloud());
    expectStagedBesideFinal(result, final_path);
    const QString temporary_path = result.temporaryPath();

    QString error_message;
    ASSERT_TRUE(result.discard(&error_message)) << error_message.toStdString();
    EXPECT_FALSE(QFileInfo::exists(temporary_path));
    QFile final_file(final_path);
    ASSERT_TRUE(final_file.open(QIODevice::ReadOnly));
    EXPECT_EQ(final_file.readAll(), QByteArray("original point cloud"));
}

TEST(PointCloudSnapshotIOTest, SharedGuardCleansUpWhenLastOwnerDisappears)
{
    QTemporaryDir temporary_dir;
    ASSERT_TRUE(temporary_dir.isValid());
    const QString final_path = QDir(temporary_dir.path()).filePath(
        QStringLiteral("edited.xyz"));
    QString temporary_path;
    std::shared_ptr<PointCloudSnapshotGuard> retained_guard;
    {
        const PointCloudSnapshotStageResult result =
            stagePointCloudSnapshot(final_path, makeCloud());
        ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
        temporary_path = result.temporaryPath();
        retained_guard = result.guard;
    }

    ASSERT_TRUE(QFileInfo::exists(temporary_path));
    retained_guard.reset();
    EXPECT_FALSE(QFileInfo::exists(temporary_path));
    EXPECT_FALSE(QFileInfo::exists(final_path));
}

} // namespace
