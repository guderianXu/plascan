#include "DemDomIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QTemporaryDir>

#include <filesystem>

using xjw::DemGridData;
using xjw::DemQualityArtifacts;
using xjw::DemDomIO;

namespace fs = std::filesystem;

namespace
{

DemGridData makeQualityGrid()
{
    DemGridData grid;
    grid.width = 2;
    grid.height = 2;
    grid.minX = 100.0;
    grid.minY = 200.0;
    grid.stepX = 2.0;
    grid.stepY = 2.0;
    grid.elevation = (cv::Mat_<float>(2, 2) << 10.0f, 11.0f, 12.0f, 13.0f);
    grid.validMask = (cv::Mat_<uchar>(2, 2) << 255, 0, 255, 255);
    grid.triangulationError = (cv::Mat_<float>(2, 2) << 0.1f, 0.0f, 0.3f, 0.4f);
    grid.pointCount = (cv::Mat_<int>(2, 2) << 5, 0, 8, 13);
    grid.confidence = (cv::Mat_<float>(2, 2) << 0.9f, 0.0f, 0.7f, 0.6f);
    grid.coverageMask = (cv::Mat_<uchar>(2, 2) << 255, 0, 255, 255);
    grid.projection.projectionWkt = QStringLiteral("LOCAL_CS[\"PlaScan\"]");
    return grid;
}

} // namespace

TEST(DemQualityRasters, WritesAllAvailableQualityProducts)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    DemQualityArtifacts artifacts;
    QString error;
    ASSERT_TRUE(DemDomIO::writeDemQualityRasters(makeQualityGrid(), tempDir.path(), &artifacts, &error))
        << error.toStdString();

    EXPECT_TRUE(fs::exists(artifacts.errorPath.toStdString()));
    EXPECT_TRUE(fs::exists(artifacts.countPath.toStdString()));
    EXPECT_TRUE(fs::exists(artifacts.confidencePath.toStdString()));
    EXPECT_TRUE(fs::exists(artifacts.coveragePath.toStdString()));
}

TEST(DemQualityRasters, SucceedsAndLeavesMissingPathsEmptyWhenOptionalMatsAreAbsent)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    DemGridData grid = makeQualityGrid();
    grid.triangulationError.release();
    grid.pointCount.release();
    grid.confidence.release();
    grid.coverageMask.release();

    DemQualityArtifacts artifacts;
    QString error;
    ASSERT_TRUE(DemDomIO::writeDemQualityRasters(grid, tempDir.path(), &artifacts, &error))
        << error.toStdString();

    EXPECT_TRUE(artifacts.errorPath.isEmpty());
    EXPECT_TRUE(artifacts.countPath.isEmpty());
    EXPECT_TRUE(artifacts.confidencePath.isEmpty());
    EXPECT_TRUE(artifacts.coveragePath.isEmpty());
}
