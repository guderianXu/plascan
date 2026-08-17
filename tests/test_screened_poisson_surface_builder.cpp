#include <gtest/gtest.h>

#include "MeshTopologyQuality.h"
#include "ScreenedPoissonSurfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

constexpr float Pi = 3.14159265358979323846f;

void makeSphereSamples(std::size_t sampleCount,
                       const std::array<float, 3> &center,
                       float radius,
                       std::vector<std::array<float, 3>> *points,
                       std::vector<std::array<float, 3>> *normals)
{
    points->reserve(sampleCount);
    normals->reserve(sampleCount);
    const float golden_angle = Pi * (3.0f - std::sqrt(5.0f));
    for (std::size_t index = 0; index < sampleCount; ++index)
    {
        const float y = 1.0f -
            2.0f * (static_cast<float>(index) + 0.5f) /
                static_cast<float>(sampleCount);
        const float radial = std::sqrt(std::max(0.0f, 1.0f - y * y));
        const float angle = golden_angle * static_cast<float>(index);
        const std::array<float, 3> normal{
            radial * std::cos(angle),
            y,
            radial * std::sin(angle)};
        normals->push_back(normal);
        points->push_back({
            center[0] + radius * normal[0],
            center[1] + radius * normal[1],
            center[2] + radius * normal[2]});
    }
}

} // namespace

TEST(ScreenedPoissonSurfaceBuilderTest, RejectsUnscreenedConfiguration)
{
    std::vector<std::array<float, 3>> points(16, {0.0f, 0.0f, 0.0f});
    std::vector<std::array<float, 3>> normals(16, {0.0f, 0.0f, 1.0f});
    xjw::mesh::ScreenedPoissonOptions options;
    options.pointWeight = 0.0f;

    const xjw::mesh::ScreenedPoissonResult result =
        xjw::mesh::ScreenedPoissonSurfaceBuilder::build(points, normals, options);

    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("pointWeight"), std::string::npos);
    EXPECT_TRUE(result.mesh.empty());
}

TEST(ScreenedPoissonSurfaceBuilderTest, ReconstructsClosedSphere)
{
    constexpr std::size_t sample_count = 1800;
    constexpr float radius = 1.35f;
    const std::array<float, 3> center{0.7f, -0.45f, 1.2f};
    std::vector<std::array<float, 3>> points;
    std::vector<std::array<float, 3>> normals;
    makeSphereSamples(sample_count, center, radius, &points, &normals);

    points.push_back({
        std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f});
    normals.push_back({0.0f, 0.0f, 0.0f});

    xjw::mesh::ScreenedPoissonOptions options;
    options.depth = 6;
    options.pointWeight = 4.0f;
    options.solverIterations = 8;
    const xjw::mesh::ScreenedPoissonResult result =
        xjw::mesh::ScreenedPoissonSurfaceBuilder::build(points, normals, options);

    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_FALSE(result.mesh.empty());
    EXPECT_EQ(result.statistics.inputSampleCount, sample_count + 1);
    EXPECT_EQ(result.statistics.acceptedSampleCount, sample_count);
    EXPECT_EQ(result.statistics.rejectedSampleCount, 1);
    EXPECT_EQ(result.statistics.outputVertexCount, result.mesh.vertices.size());
    EXPECT_EQ(result.statistics.outputTriangleCount, result.mesh.faces.size());
    EXPECT_GT(result.mesh.vertices.size(), 500);
    EXPECT_GT(result.mesh.faces.size(), 1000);

    const xjw::mesh::MeshTopologyQualityStatistics topology =
        xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
    EXPECT_TRUE(topology.closedTwoManifold);
    EXPECT_EQ(topology.componentCount, 1);
    EXPECT_EQ(topology.boundaryEdgeCount, 0);
    EXPECT_EQ(topology.nonManifoldEdgeCount, 0);
    EXPECT_EQ(topology.eulerCharacteristic, 2);

    std::vector<double> radius_errors;
    radius_errors.reserve(result.mesh.vertices.size());
    for (const xjw::mesh::MeshVertex &vertex : result.mesh.vertices)
    {
        const double dx = static_cast<double>(vertex.x) - center[0];
        const double dy = static_cast<double>(vertex.y) - center[1];
        const double dz = static_cast<double>(vertex.z) - center[2];
        radius_errors.push_back(std::abs(std::sqrt(dx * dx + dy * dy + dz * dz) - radius));
    }
    std::sort(radius_errors.begin(), radius_errors.end());
    const double median_error = radius_errors[radius_errors.size() / 2];
    const double p95_error = radius_errors[
        static_cast<std::size_t>(0.95 * static_cast<double>(radius_errors.size() - 1))];
    EXPECT_LT(median_error, 0.025);
    EXPECT_LT(p95_error, 0.060);
}
