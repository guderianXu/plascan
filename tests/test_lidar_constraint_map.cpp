#include <gtest/gtest.h>

#include "LaserConstraintMap.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace
{

std::filesystem::path tempPath(const std::string &name)
{
    return std::filesystem::temp_directory_path() / name;
}

void writeTextFile(const std::filesystem::path &path, const std::string &content)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.good());
    out << content;
}

void writeFloat(std::ofstream &out, float value)
{
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeBinaryPoint(std::ofstream &out,
                      float x,
                      float y,
                      float z,
                      float intensity,
                      float nx,
                      float ny,
                      float nz,
                      float curvature)
{
    writeFloat(out, x);
    writeFloat(out, y);
    writeFloat(out, z);
    writeFloat(out, intensity);
    writeFloat(out, nx);
    writeFloat(out, ny);
    writeFloat(out, nz);
    writeFloat(out, curvature);
}

void writeBinaryPly(const std::filesystem::path &path)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.good());
    out << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex 2\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property float intensity\n"
        << "property float normal_x\n"
        << "property float normal_y\n"
        << "property float normal_z\n"
        << "property float curvature\n"
        << "end_header\n";
    writeBinaryPoint(out, 1.0f, 2.0f, 3.0f, 0.5f, 0.0f, 0.0f, 2.0f, 0.02f);
    writeBinaryPoint(out, 5.0f, 2.0f, 3.0f, 0.7f, 0.0f, 1.0f, 0.0f, 0.03f);
}

} // namespace

TEST(LaserConstraintMapTest, LoadsAsciiPlyAndFiltersInvalidPlaneSamples)
{
    const auto path = tempPath("plascan_lidar_ascii_filter_test.ply");
    writeTextFile(path,
                  "ply\n"
                  "format ascii 1.0\n"
                  "element vertex 4\n"
                  "property float x\n"
                  "property float y\n"
                  "property float z\n"
                  "property float normal_x\n"
                  "property float normal_y\n"
                  "property float normal_z\n"
                  "property float curvature\n"
                  "end_header\n"
                  "0 0 0 0 0 1 0.05\n"
                  "0.1 0 0 0 0 0 0.05\n"
                  "1 0 0 0 1 0 0.50\n"
                  "2 0 0 0 0 1 0.10\n");

    xjw::lidar::LaserConstraintMapOptions options;
    options.maxCurvature = 0.2;
    options.voxelSizeMeters = 0.0;

    xjw::lidar::LaserConstraintMap map;
    std::string error;
    ASSERT_TRUE(map.loadPly(path.string(), options, &error)) << error;

    ASSERT_EQ(map.size(), 2u);
    EXPECT_NEAR(map.samples().front().normal[2], 1.0, 1e-12);

    xjw::lidar::LaserPlaneSample nearest;
    double distance = 0.0;
    ASSERT_TRUE(map.nearestPlane({{0.05, 0.0, 0.20}}, &nearest, &distance));
    EXPECT_NEAR(nearest.point[0], 0.0, 1e-12);
    EXPECT_NEAR(distance, std::sqrt(0.05 * 0.05 + 0.20 * 0.20), 1e-12);
}

TEST(LaserConstraintMapTest, LoadsAsciiPlyWithoutNormalsAsHeightPlanesWhenExplicitlyEnabled)
{
    const auto path = tempPath("plascan_lidar_ascii_height_plane_test.ply");
    writeTextFile(path,
                  "ply\n"
                  "format ascii 1.0\n"
                  "element vertex 2\n"
                  "property float x\n"
                  "property float y\n"
                  "property float z\n"
                  "end_header\n"
                  "0 0 12.5\n"
                  "2 0 13.0\n");

    xjw::lidar::LaserConstraintMapOptions options;
    options.voxelSizeMeters = 0.0;
    options.useMissingNormalsAsHeightPlanes = true;

    xjw::lidar::LaserConstraintMap map;
    std::string error;
    ASSERT_TRUE(map.loadPly(path.string(), options, &error)) << error;

    ASSERT_EQ(map.size(), 2u);
    EXPECT_NEAR(map.samples().front().normal[0], 0.0, 1e-12);
    EXPECT_NEAR(map.samples().front().normal[1], 0.0, 1e-12);
    EXPECT_NEAR(map.samples().front().normal[2], 1.0, 1e-12);
    EXPECT_NEAR(map.samples().front().curvature, 0.0, 1e-12);
}

TEST(LaserConstraintMapTest, VoxelDownsampleKeepsOneAveragedPlanePerCell)
{
    const auto path = tempPath("plascan_lidar_voxel_test.ply");
    writeTextFile(path,
                  "ply\n"
                  "format ascii 1.0\n"
                  "element vertex 3\n"
                  "property float x\n"
                  "property float y\n"
                  "property float z\n"
                  "property float normal_x\n"
                  "property float normal_y\n"
                  "property float normal_z\n"
                  "property float curvature\n"
                  "end_header\n"
                  "0.0 0 0 0 0 1 0.02\n"
                  "0.2 0 0 0 0 1 0.04\n"
                  "2.0 0 0 0 1 0 0.03\n");

    xjw::lidar::LaserConstraintMapOptions options;
    options.maxCurvature = 0.2;
    options.voxelSizeMeters = 1.0;

    xjw::lidar::LaserConstraintMap map;
    std::string error;
    ASSERT_TRUE(map.loadPly(path.string(), options, &error)) << error;

    ASSERT_EQ(map.size(), 2u);
    EXPECT_NEAR(map.samples().front().point[0], 0.1, 1e-12);
    EXPECT_NEAR(map.samples().front().normal[2], 1.0, 1e-12);
}

TEST(LaserConstraintMapTest, LoadsBinaryLittleEndianPlyWithNormalAndCurvatureFields)
{
    const auto path = tempPath("plascan_lidar_binary_test.ply");
    writeBinaryPly(path);

    xjw::lidar::LaserConstraintMapOptions options;
    options.maxCurvature = 0.2;
    options.voxelSizeMeters = 0.0;

    xjw::lidar::LaserConstraintMap map;
    std::string error;
    ASSERT_TRUE(map.loadPly(path.string(), options, &error)) << error;

    ASSERT_EQ(map.size(), 2u);
    EXPECT_NEAR(map.samples().front().point[0], 1.0, 1e-6);
    EXPECT_NEAR(map.samples().front().point[1], 2.0, 1e-6);
    EXPECT_NEAR(map.samples().front().point[2], 3.0, 1e-6);
    EXPECT_NEAR(map.samples().front().normal[2], 1.0, 1e-6);
}
