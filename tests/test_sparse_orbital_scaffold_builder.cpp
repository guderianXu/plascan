#include "SparseOrbitalScaffoldBuilder.h"

#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QTemporaryDir>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace
{

using xjw::mesh::SparseOrbitalScaffoldBuilder;
using xjw::mesh::SparseOrbitalScaffoldOptions;

void writeAsciiPly(const std::filesystem::path &path,
                   const std::vector<std::array<float, 3>> &points)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    ASSERT_TRUE(stream.good());
    stream << "ply\n"
           << "format ascii 1.0\n"
           << "element vertex " << points.size() << "\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "end_header\n";
    for (const auto &point : points)
    {
        stream << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    }
}

void writeBinaryPly(const std::filesystem::path &path,
                    const std::vector<std::array<float, 3>> &points)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.good());
    stream << "ply\n"
           << "format binary_little_endian 1.0\n"
           << "element vertex " << points.size() << "\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "end_header\n";
    for (const auto &point : points)
    {
        stream.write(reinterpret_cast<const char *>(point.data()),
                     static_cast<std::streamsize>(sizeof(float) * point.size()));
    }
}

void writePointQualitySidecar(
    const std::filesystem::path &path,
    const std::vector<std::array<float, 3>> &points,
    int track_length = 4,
    float rms_reprojection_pixels = 0.5f,
    float triangulation_angle_degrees = 20.0f)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    ASSERT_TRUE(stream.good());
    stream << "{\"points\":[";
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        if (index > 0)
        {
            stream << ',';
        }
        const auto &point = points[index];
        const float x = std::isfinite(point[0]) ? point[0] : 0.0f;
        const float y = std::isfinite(point[1]) ? point[1] : 0.0f;
        const float z = std::isfinite(point[2]) ? point[2] : 0.0f;
        stream << "{\"min_tri_angle_deg\":" << triangulation_angle_degrees
               << ",\"observations\":[{\"image_name\":\"escaped\\\"name\"}]"
               << ",\"point_xyz\":[" << x << ',' << y << ',' << z << ']'
               << ",\"rms_reproj_px\":" << rms_reprojection_pixels
               << ",\"track_len\":" << track_length
               << ",\"triangulation_angle_deg\":"
               << triangulation_angle_degrees << '}';
    }
    stream << "]}";
}

std::vector<std::array<float, 3>> spherePoints()
{
    constexpr int kLatitudeCount = 7;
    constexpr int kLongitudeCount = 16;
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<std::array<float, 3>> points;
    for (int latitude = 1; latitude < kLatitudeCount; ++latitude)
    {
        const float phi = kPi * static_cast<float>(latitude)
            / static_cast<float>(kLatitudeCount);
        for (int longitude = 0; longitude < kLongitudeCount; ++longitude)
        {
            const float theta = 2.0f * kPi * static_cast<float>(longitude)
                / static_cast<float>(kLongitudeCount);
            points.push_back({std::sin(phi) * std::cos(theta),
                              std::sin(phi) * std::sin(theta),
                              std::cos(phi)});
        }
    }
    points.push_back({0.0f, 0.0f, 1.0f});
    points.push_back({0.0f, 0.0f, -1.0f});
    return points;
}

std::filesystem::path temporaryRoot(QTemporaryDir &directory)
{
    return xjw::common::io::toFilesystemPath(directory.path());
}

TEST(SparseOrbitalScaffoldBuilderTest, ResolvesExplicitThenProjectFallbackPly)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const auto root = temporaryRoot(directory);
    const auto mvs_output = root / "1" / "mvs_output";
    const auto fallback = root / "1" / "assets" / "aerial_triangulation"
        / "sfm_sparse" / "sfm_sparse.ply";
    const auto explicit_path = root / "explicit.ply";
    std::filesystem::create_directories(mvs_output);
    writeAsciiPly(fallback, spherePoints());
    writeAsciiPly(explicit_path, spherePoints());

    std::string error;
    EXPECT_EQ(SparseOrbitalScaffoldBuilder::resolveSparsePly(
                  mvs_output, {}, &error),
              std::filesystem::weakly_canonical(fallback));
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(SparseOrbitalScaffoldBuilder::resolveSparsePly(
                  mvs_output, explicit_path, &error),
              std::filesystem::weakly_canonical(explicit_path));
    EXPECT_TRUE(error.empty());
}

TEST(SparseOrbitalScaffoldBuilderTest, ProducesRobustCenterRadialNormals)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const auto root = temporaryRoot(directory);
    const auto mvs_output = root / "mvs_output";
    const auto sparse_path = root / "sphere.ply";
    const auto sidecar_path = root / "sphere_points.json";
    std::filesystem::create_directories(mvs_output);
    const auto points = spherePoints();
    writeAsciiPly(sparse_path, points);
    writePointQualitySidecar(sidecar_path, points);

    SparseOrbitalScaffoldOptions options;
    options.minimumPointCount = 20;
    options.maximumPointCount = 1000;
    const auto result = SparseOrbitalScaffoldBuilder::build(
        mvs_output, sparse_path, sidecar_path, options);

    ASSERT_TRUE(result.succeeded()) << result.error;
    ASSERT_EQ(result.points.size(), result.normals.size());
    for (std::size_t index = 0; index < result.points.size(); ++index)
    {
        const auto &point = result.points[index];
        const auto &normal = result.normals[index];
        const float radial_x = point[0] - result.robustCenter[0];
        const float radial_y = point[1] - result.robustCenter[1];
        const float radial_z = point[2] - result.robustCenter[2];
        const float dot = radial_x * normal[0]
            + radial_y * normal[1] + radial_z * normal[2];
        EXPECT_GE(dot, -1.0e-6f);
        EXPECT_NEAR(normal[0] * normal[0] + normal[1] * normal[1]
                        + normal[2] * normal[2],
                    1.0f, 1.0e-4f);
    }
}

TEST(SparseOrbitalScaffoldBuilderTest, RejectsNonFiniteAndExtremeRadialOutliers)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const auto root = temporaryRoot(directory);
    const auto mvs_output = root / "mvs_output";
    const auto sparse_path = root / "outliers.ply";
    const auto sidecar_path = root / "outliers_points.json";
    std::filesystem::create_directories(mvs_output);
    auto points = spherePoints();
    points.push_back({100.0f, 100.0f, 100.0f});
    points.push_back({std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f});
    writeBinaryPly(sparse_path, points);
    writePointQualitySidecar(sidecar_path, points);

    SparseOrbitalScaffoldOptions options;
    options.minimumPointCount = 20;
    options.maximumPointCount = 1000;
    const auto result = SparseOrbitalScaffoldBuilder::build(
        mvs_output, sparse_path, sidecar_path, options);

    ASSERT_TRUE(result.succeeded()) << result.error;
    EXPECT_EQ(result.statistics.inputPointCount, points.size());
    EXPECT_EQ(result.statistics.nonFiniteRejectedCount, 1U);
    EXPECT_EQ(result.statistics.radialOutlierRejectedCount, 1U);
    for (const auto &point : result.points)
    {
        const float radius = std::sqrt(
            point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
        EXPECT_LT(radius, 2.0f);
    }
}


TEST(SparseOrbitalScaffoldBuilderTest, RejectsOrbitalAutoScaffoldWithoutQualitySidecar)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const auto root = temporaryRoot(directory);
    const auto mvs_output = root / "1" / "mvs_output";
    const auto sparse_path = root / "1" / "assets" / "aerial_triangulation"
        / "sfm_sparse" / "sfm_sparse.ply";
    std::filesystem::create_directories(mvs_output);
    writeAsciiPly(sparse_path, spherePoints());

    SparseOrbitalScaffoldOptions options;
    options.minimumPointCount = 20;
    const auto result = SparseOrbitalScaffoldBuilder::build(
        mvs_output, {}, {}, options);

    EXPECT_FALSE(result.succeeded());
    EXPECT_NE(result.error.find("sidecar"), std::string::npos);
    EXPECT_TRUE(result.points.empty());
}

TEST(SparseOrbitalScaffoldBuilderTest, RejectsTracksBelowSidecarQualityGate)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const auto root = temporaryRoot(directory);
    const auto mvs_output = root / "mvs_output";
    const auto sparse_path = root / "weak.ply";
    const auto sidecar_path = root / "weak_points.json";
    std::filesystem::create_directories(mvs_output);
    const auto points = spherePoints();
    writeAsciiPly(sparse_path, points);
    writePointQualitySidecar(sidecar_path, points, 2, 0.5f, 20.0f);

    SparseOrbitalScaffoldOptions options;
    options.minimumPointCount = 20;
    const auto result = SparseOrbitalScaffoldBuilder::build(
        mvs_output, sparse_path, sidecar_path, options);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(result.statistics.qualityRejectedCount, points.size());
    EXPECT_TRUE(result.points.empty());
}

} // namespace
