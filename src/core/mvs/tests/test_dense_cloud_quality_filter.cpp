#include "DenseCloudQualityFilter.h"
#include "DenseCloudBuilder.h"

#include <gtest/gtest.h>

#include <plamatrix/dense/dense_matrix.h>

#include <cmath>

namespace
{

using DensePointCloud = xjw::mvs::DensePointCloud;

DensePointCloud makeTerrainWithVerticalSpikes()
{
    constexpr int grid = 5;
    constexpr int samplesPerCell = 4;
    constexpr int stablePoints = grid * grid * samplesPerCell;
    constexpr int spikePoints = 2;

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(stablePoints + spikePoints, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(stablePoints + spikePoints, 3);

    int row = 0;
    for (int y = 0; y < grid; ++y)
    {
        for (int x = 0; x < grid; ++x)
        {
            for (int s = 0; s < samplesPerCell; ++s)
            {
                const float ox = (s % 2 == 0) ? 0.20f : 0.70f;
                const float oy = (s < 2) ? 0.20f : 0.70f;
                points(row, 0) = static_cast<float>(x) + ox;
                points(row, 1) = static_cast<float>(y) + oy;
                points(row, 2) = 0.01f * static_cast<float>(x) + 0.02f * static_cast<float>(y);
                colors(row, 0) = static_cast<std::uint8_t>(row + 1);
                colors(row, 1) = static_cast<std::uint8_t>(row + 2);
                colors(row, 2) = static_cast<std::uint8_t>(row + 3);
                ++row;
            }
        }
    }

    points(row, 0) = 2.45f;
    points(row, 1) = 2.45f;
    points(row, 2) = 2.20f;
    colors(row, 0) = 240;
    colors(row, 1) = 16;
    colors(row, 2) = 16;
    ++row;

    points(row, 0) = 2.55f;
    points(row, 1) = 2.55f;
    points(row, 2) = -1.80f;
    colors(row, 0) = 16;
    colors(row, 1) = 16;
    colors(row, 2) = 240;

    DensePointCloud cloud(std::move(points));
    cloud.setColors(std::move(colors));
    return cloud;
}

DensePointCloud makeSlopedTerrainWithPlaneResidualSpike()
{
    constexpr int width = 8;
    constexpr int height = 8;
    constexpr int stablePoints = width * height;
    constexpr int spikePoints = 1;

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(stablePoints + spikePoints, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(stablePoints + spikePoints, 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(stablePoints + spikePoints, 3);

    int row = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const float fx = static_cast<float>(x);
            const float fy = static_cast<float>(y);
            points(row, 0) = fx;
            points(row, 1) = fy;
            points(row, 2) = 0.15f * fx + 0.08f * fy;
            colors(row, 0) = 40;
            colors(row, 1) = static_cast<std::uint8_t>(80 + x);
            colors(row, 2) = static_cast<std::uint8_t>(120 + y);
            normals(row, 0) = -0.15f;
            normals(row, 1) = -0.08f;
            normals(row, 2) = 1.0f;
            ++row;
        }
    }

    points(row, 0) = 3.25f;
    points(row, 1) = 4.25f;
    points(row, 2) = 0.15f * 3.25f + 0.08f * 4.25f + 0.42f;
    colors(row, 0) = 250;
    colors(row, 1) = 10;
    colors(row, 2) = 10;
    normals(row, 0) = -0.15f;
    normals(row, 1) = -0.08f;
    normals(row, 2) = 1.0f;

    DensePointCloud cloud(std::move(points));
    cloud.setColors(std::move(colors));
    cloud.setNormals(std::move(normals));
    return cloud;
}

} // namespace

namespace
{

void expectSkippedProcessingReport(const plapoint::ProcessingReport &report,
                                   plapoint::ProcessingDevice requestedDevice,
                                   const char *reason)
{
    EXPECT_EQ(report.requestedDevice, requestedDevice);
    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_EQ(report.usedDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_EQ(report.neighborBackend, plapoint::ProcessingNeighborBackend::None);
    EXPECT_FALSE(report.usedFallback);
    EXPECT_EQ(report.fallbackReason, reason);
}

} // namespace

TEST(DenseCloudBuilderReportTest, VoxelNoOpInitializesReport)
{
    std::vector<xjw::mvs::DensePoint> cloud(1);
    cloud.front().x = 1.0f;
    plapoint::ProcessingReport report;
    report.requestedDevice = plapoint::ProcessingDevice::CPU;
    report.actualDevice = plapoint::ProcessingDevice::CUDA;
    report.usedFallback = true;
    report.fallbackReason = "stale";

    const auto output = xjw::mvs::DenseCloudBuilder::voxelDownsample(
        cloud, 0.0f, plapoint::ProcessingDevice::OpenCL, &report);

    ASSERT_EQ(output.size(), cloud.size());
    EXPECT_FLOAT_EQ(output.front().x, cloud.front().x);
    expectSkippedProcessingReport(
        report,
        plapoint::ProcessingDevice::OpenCL,
        "skipped: empty cloud or invalid voxel size");
}

TEST(DenseCloudBuilderReportTest, StatisticalNoOpInitializesReport)
{
    std::vector<xjw::mvs::DensePoint> cloud(1);
    plapoint::ProcessingReport report;
    report.requestedDevice = plapoint::ProcessingDevice::OpenCL;
    report.actualDevice = plapoint::ProcessingDevice::OpenCL;
    report.usedFallback = true;
    report.fallbackReason = "stale";

    const auto output = xjw::mvs::DenseCloudBuilder::statisticalOutlierRemoval(
        cloud, 30, 1.2f, plapoint::ProcessingDevice::CUDA, &report);

    EXPECT_EQ(output.size(), cloud.size());
    expectSkippedProcessingReport(
        report,
        plapoint::ProcessingDevice::CUDA,
        "skipped: point count is smaller than k + 1");
}

TEST(DenseCloudBuilderReportTest, RadiusNoOpInitializesReport)
{
    const std::vector<xjw::mvs::DensePoint> cloud;
    plapoint::ProcessingReport report;
    report.requestedDevice = plapoint::ProcessingDevice::CUDA;
    report.actualDevice = plapoint::ProcessingDevice::CUDA;
    report.usedFallback = true;
    report.fallbackReason = "stale";

    const auto output = xjw::mvs::DenseCloudBuilder::radiusOutlierRemoval(
        cloud, 1.0f, 3, plapoint::ProcessingDevice::OpenCL, &report);

    EXPECT_TRUE(output.empty());
    expectSkippedProcessingReport(
        report, plapoint::ProcessingDevice::OpenCL, "skipped: empty cloud");
}

TEST(DenseCloudQualityFilter, TerrainHeightSpikeFilterRemovesVerticalSpikesAndPreservesColors)
{
    DensePointCloud cloud = makeTerrainWithVerticalSpikes();

    xjw::mvs::TerrainHeightSpikeFilterOptions options;
    options.enabled = true;
    options.gridResolution = 5;
    options.minCellPoints = 4;
    options.minHeightThreshold = 0.20f;
    options.madMultiplier = 3.0f;

    xjw::mvs::TerrainHeightSpikeFilterReport report;
    DensePointCloud filtered = xjw::mvs::filterTerrainHeightSpikes(cloud, options, &report);

    ASSERT_EQ(filtered.size(), 100u);
    EXPECT_EQ(report.inputPoints, 102u);
    EXPECT_EQ(report.outputPoints, 100u);
    EXPECT_EQ(report.removedPoints, 2u);
    EXPECT_LE(report.p95CellZRangeAfter, report.p95CellZRangeBefore);
    ASSERT_TRUE(filtered.hasColors());
    ASSERT_NE(filtered.colors(), nullptr);
    EXPECT_EQ(filtered.colors()->getValue(0, 0), 1);
    EXPECT_EQ(filtered.colors()->getValue(0, 1), 2);
    EXPECT_EQ(filtered.colors()->getValue(0, 2), 3);

    for (std::size_t i = 0; i < filtered.size(); ++i)
    {
        EXPECT_LT(std::abs(filtered.points().getValue(static_cast<plamatrix::Index>(i), 2)), 0.20f);
    }
}

TEST(DenseCloudQualityFilter, LocalPlaneFilterRemovesResidualSpikeAndPreservesAttributes)
{
    DensePointCloud cloud = makeSlopedTerrainWithPlaneResidualSpike();

    xjw::mvs::TerrainHeightSpikeFilterOptions options;
    options.enabled = true;
    options.gridResolution = 1;
    options.minCellPoints = 12;
    options.minHeightThreshold = 2.0f;
    options.madMultiplier = 20.0f;
    options.localPlaneFilterEnabled = true;
    options.localPlaneMinPoints = 12;
    options.localPlaneMinResidualThreshold = 0.10f;
    options.localPlaneMadMultiplier = 4.0f;

    xjw::mvs::TerrainHeightSpikeFilterReport report;
    DensePointCloud filtered = xjw::mvs::filterTerrainHeightSpikes(cloud, options, &report);

    ASSERT_EQ(filtered.size(), 64u);
    EXPECT_EQ(report.inputPoints, 65u);
    EXPECT_EQ(report.outputPoints, 64u);
    EXPECT_EQ(report.removedPoints, 1u);
    EXPECT_EQ(report.localPlaneRemovedPoints, 1u);
    ASSERT_TRUE(filtered.hasColors());
    ASSERT_TRUE(filtered.hasNormals());

    for (std::size_t i = 0; i < filtered.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        const float x = filtered.points().getValue(row, 0);
        const float y = filtered.points().getValue(row, 1);
        const float z = filtered.points().getValue(row, 2);
        EXPECT_NEAR(z, 0.15f * x + 0.08f * y, 1.0e-4f);
    }
}
