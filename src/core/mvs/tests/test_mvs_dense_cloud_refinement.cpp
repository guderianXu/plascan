#include "DenseCloudArtifactValidation.h"
#include "DenseCloudRefinementService.h"
#include "PointCloudArtifactIO.h"
#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QLockFile>
#include <QTemporaryDir>

#include <filesystem>
#include <system_error>

namespace
{

QByteArray asciiCloudBytes()
{
    return QByteArray("ply\n"
                      "format ascii 1.0\n"
                      "element vertex 4\n"
                      "property float x\n"
                      "property float y\n"
                      "property float z\n"
                      "end_header\n"
                      "0 0 0\n"
                      "1 0 0\n"
                      "0 1 0\n"
                      "1 1 0\n");
}

bool writeAsciiCloud(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        return false;
    }
    const QByteArray bytes = asciiCloudBytes();
    if (file.write(bytes) != bytes.size())
    {
        return false;
    }
    file.close();
    return true;
}

bool writeBytes(const QString &path, const QByteArray &bytes)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly)
        && file.write(bytes) == bytes.size()
        && file.flush();
}

TEST(DenseCloudRefinementServiceTest, RejectsMissingPaths)
{
    xjw::mvs::DenseCloudRefinementRequest request;
    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;

    EXPECT_FALSE(xjw::mvs::refineDenseCloud(request, &result, &error));
    EXPECT_FALSE(error.empty());
}

TEST(DenseCloudRefinementServiceTest, RefinesAsciiPlyThroughInMemoryFallback)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString inputPath = QDir(directory.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPath = QDir(directory.path()).filePath(QStringLiteral("output.ply"));
    ASSERT_TRUE(writeAsciiCloud(inputPath));

    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = inputPath.toUtf8().toStdString();
    request.outputPath = outputPath.toUtf8().toStdString();
    request.filterOptions.enabled = false;
    request.filterPasses = 1;
    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;

    ASSERT_TRUE(xjw::mvs::refineDenseCloud(request, &result, &error)) << error;
    EXPECT_EQ(result.mode, "in_memory");
    EXPECT_EQ(result.report.inputPoints, 4U);
    EXPECT_EQ(result.report.outputPoints, 4U);
    EXPECT_TRUE(QFileInfo::exists(outputPath));
}

TEST(DenseCloudRefinementServiceTest, RejectsEquivalentInputOutputWithoutChangingSource)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString inputPath = QDir(directory.path()).filePath(QStringLiteral("input.ply"));
    ASSERT_TRUE(writeAsciiCloud(inputPath));
    QFile input(inputPath);
    ASSERT_TRUE(input.open(QIODevice::ReadOnly));
    const QByteArray originalBytes = input.readAll();
    input.close();

    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = inputPath.toUtf8().toStdString();
    request.outputPath = QDir(directory.path())
                             .filePath(QStringLiteral("./input.ply"))
                             .toUtf8()
                             .toStdString();
    request.filterOptions.enabled = false;
    request.filterPasses = 1;
    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;

    EXPECT_FALSE(xjw::mvs::refineDenseCloud(request, &result, &error));
    EXPECT_NE(error.find("不能相同"), std::string::npos) << error;
    ASSERT_TRUE(input.open(QIODevice::ReadOnly));
    EXPECT_EQ(input.readAll(), originalBytes);
}

TEST(DenseCloudRefinementServiceTest, RejectsHardLinkedInputOutputWithoutChangingSource)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString inputPath = QDir(directory.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPath = QDir(directory.path()).filePath(QStringLiteral("output.ply"));
    ASSERT_TRUE(writeAsciiCloud(inputPath));

    std::error_code linkError;
    std::filesystem::create_hard_link(
        xjw::common::io::toFilesystemPath(inputPath),
        xjw::common::io::toFilesystemPath(outputPath),
        linkError);
    if (linkError)
    {
        GTEST_SKIP() << "hard links are unavailable: " << linkError.message();
    }
    QFile input(inputPath);
    ASSERT_TRUE(input.open(QIODevice::ReadOnly));
    const QByteArray originalBytes = input.readAll();
    input.close();

    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = inputPath.toUtf8().toStdString();
    request.outputPath = outputPath.toUtf8().toStdString();
    request.filterOptions.enabled = false;
    request.filterPasses = 1;
    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;

    EXPECT_FALSE(xjw::mvs::refineDenseCloud(request, &result, &error));
    EXPECT_NE(error.find("不能相同"), std::string::npos) << error;
    ASSERT_TRUE(input.open(QIODevice::ReadOnly));
    EXPECT_EQ(input.readAll(), originalBytes);
}

TEST(DenseCloudRefinementServiceTest, RejectsConcurrentWriterWithoutChangingOldOutput)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString inputPath = QDir(directory.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPath = QDir(directory.path()).filePath(QStringLiteral("output.ply"));
    const QString lockPath = QDir(directory.path()).filePath(
        QStringLiteral(".output.ply.refine.lock"));
    ASSERT_TRUE(writeAsciiCloud(inputPath));
    ASSERT_TRUE(writeBytes(outputPath, QByteArray("old-output")));

    QLockFile competingLock(lockPath);
    ASSERT_TRUE(competingLock.tryLock(0));
    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = inputPath.toUtf8().toStdString();
    request.outputPath = outputPath.toUtf8().toStdString();
    request.filterOptions.enabled = false;
    request.filterPasses = 1;
    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;

    EXPECT_FALSE(xjw::mvs::refineDenseCloud(request, &result, &error));
    EXPECT_NE(error.find("另一个任务"), std::string::npos) << error;
    QFile output(outputPath);
    ASSERT_TRUE(output.open(QIODevice::ReadOnly));
    EXPECT_EQ(output.readAll(), QByteArray("old-output"));
}

TEST(DenseCloudRefinementServiceTest, PublishFailurePreservesExistingOutputDirectory)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString inputPath = QDir(directory.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPath = QDir(directory.path()).filePath(QStringLiteral("existing_output.ply"));
    const QString markerPath = QDir(outputPath).filePath(QStringLiteral("keep.txt"));
    ASSERT_TRUE(writeAsciiCloud(inputPath));
    ASSERT_TRUE(QDir().mkpath(outputPath));
    QFile marker(markerPath);
    ASSERT_TRUE(marker.open(QIODevice::WriteOnly));
    ASSERT_EQ(marker.write("keep"), 4);
    marker.close();

    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = inputPath.toUtf8().toStdString();
    request.outputPath = outputPath.toUtf8().toStdString();
    request.filterOptions.enabled = false;
    request.filterPasses = 1;
    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;

    EXPECT_FALSE(xjw::mvs::refineDenseCloud(request, &result, &error));
    EXPECT_TRUE(QFileInfo::exists(markerPath));
    EXPECT_TRUE(QFileInfo(outputPath).isDir());
}

TEST(DenseCloudRefinementServiceTest, SuccessfulPublishReplacesOldFileAndRemovesTransactionArtifacts)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString inputPath = QDir(directory.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPath = QDir(directory.path()).filePath(QStringLiteral("output.ply"));
    ASSERT_TRUE(writeAsciiCloud(inputPath));
    QFile output(outputPath);
    ASSERT_TRUE(output.open(QIODevice::WriteOnly));
    ASSERT_EQ(output.write("old-output"), 10);
    output.close();

    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = inputPath.toUtf8().toStdString();
    request.outputPath = outputPath.toUtf8().toStdString();
    request.filterOptions.enabled = false;
    request.filterPasses = 1;
    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;

    ASSERT_TRUE(xjw::mvs::refineDenseCloud(request, &result, &error)) << error;
    ASSERT_TRUE(output.open(QIODevice::ReadOnly));
    EXPECT_TRUE(output.readAll().startsWith("ply\n"));
    output.close();
    std::string validationError;
    EXPECT_TRUE(xjw::mvs::detail::validateDenseCloudPlyArtifact(
        xjw::common::io::toFilesystemPath(outputPath),
        result.report.outputPoints,
        &validationError)) << validationError;
    EXPECT_TRUE(result.warnings.empty());
    const QStringList leftovers = QDir(directory.path()).entryList(
        QStringList{QStringLiteral(".*.refine-*"), QStringLiteral(".*.backup-*")},
        QDir::AllEntries | QDir::Hidden | QDir::NoDotAndDotDot);
    EXPECT_TRUE(leftovers.isEmpty()) << qPrintable(leftovers.join(QStringLiteral(", ")));
}

TEST(DenseCloudRefinementServiceTest, CreatesMissingOutputParentBeforeTakingLock)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString inputPath = QDir(directory.path()).filePath(QStringLiteral("input.ply"));
    const QString outputPath = QDir(directory.path()).filePath(
        QStringLiteral("new/nested/output.ply"));
    ASSERT_TRUE(writeAsciiCloud(inputPath));

    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = inputPath.toUtf8().toStdString();
    request.outputPath = outputPath.toUtf8().toStdString();
    request.filterOptions.enabled = false;
    request.filterPasses = 1;
    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;

    ASSERT_TRUE(xjw::mvs::refineDenseCloud(request, &result, &error)) << error;
    EXPECT_TRUE(QFileInfo::exists(outputPath));
}

TEST(DenseCloudArtifactValidationTest, RejectsTruncatedBinaryVertexPayload)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = QDir(directory.path()).filePath(QStringLiteral("truncated.ply"));
    QByteArray bytes("ply\n"
                     "format binary_little_endian 1.0\n"
                     "element vertex 2\n"
                     "property float x\n"
                     "property float y\n"
                     "property float z\n"
                     "end_header\n");
    bytes.append(QByteArray(3 * static_cast<int>(sizeof(float)), '\0'));
    ASSERT_TRUE(writeBytes(path, bytes));

    std::string error;
    EXPECT_FALSE(xjw::mvs::detail::validateDenseCloudPlyArtifact(
        xjw::common::io::toFilesystemPath(path), 2, &error));
    EXPECT_NE(error.find("byte length mismatch"), std::string::npos) << error;
}

TEST(PointCloudArtifactIOTest, WritesBinaryPlyAndCreatesParentDirectory)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString outputPath = QDir(directory.path()).filePath(QStringLiteral("nested/cloud.ply"));
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(1, 3);
    points(0, 0) = 1.0f;
    points(0, 1) = 2.0f;
    points(0, 2) = 3.0f;
    xjw::mvs::DensePointCloud cloud(std::move(points));
    QString error;

    ASSERT_TRUE(xjw::mvs::writeDensePointCloudPly(outputPath, cloud, true, &error))
        << qUtf8Printable(error);
    EXPECT_TRUE(QFileInfo::exists(outputPath));
}

} // namespace
