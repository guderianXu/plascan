#include "DenseCloudRefinementService.h"
#include "PointCloudArtifactIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

namespace
{

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
    QFile input(inputPath);
    ASSERT_TRUE(input.open(QIODevice::WriteOnly | QIODevice::Text));
    input.write("ply\n"
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
    input.close();

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
