#include "SurfaceReconstructorHeightGrid.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

TEST(SurfaceReconstructorHeightGridTest,
     RejectsExtremeFiniteSpanBeforeDimensionConversion)
{
    std::vector<xjw::mesh::detail::PointXYZRGB> points(2);
    points[0].x = std::numeric_limits<float>::lowest();
    points[0].y = 0.0f;
    points[0].z = 0.0f;
    points[1].x = std::numeric_limits<float>::max();
    points[1].y = 1.0f;
    points[1].z = 1.0f;

    xjw::mesh::ReconstructionConfig config;
    config.preprocessingDevice = plapoint::ProcessingDevice::CPU;
    config.padding = 0.05f;
    config.resolution = 128;

    try
    {
        (void)xjw::mesh::detail::buildHeightGrid(points, config);
        FAIL() << "Extreme finite bounds must be rejected before an int cast";
    }
    catch (const std::overflow_error &error)
    {
        EXPECT_NE(std::string(error.what()).find("SurfaceReconstructor height-grid"),
                  std::string::npos);
    }
}

TEST(SurfaceReconstructorHeightGridTest,
     AcceptsLargeButRepresentableAnisotropicSpan)
{
    std::vector<xjw::mesh::detail::PointXYZRGB> points(2);
    points[0].x = 0.0f;
    points[0].y = 0.0f;
    points[0].z = 0.0f;
    points[1].x = std::numeric_limits<float>::max() / 4.0f;
    points[1].y = 1.0f;
    points[1].z = 1.0f;

    xjw::mesh::ReconstructionConfig config;
    config.preprocessingDevice = plapoint::ProcessingDevice::CPU;
    config.padding = 0.0f;
    config.resolution = 128;

    const xjw::mesh::detail::HeightGrid grid =
        xjw::mesh::detail::buildHeightGrid(points, config);

    EXPECT_EQ(grid.nx, 128);
    EXPECT_EQ(grid.ny, 8);
    EXPECT_EQ(grid.heights.size(), 128U * 8U);
}
