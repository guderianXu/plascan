#include "MeshTopologyQuality.h"
#include "VisibilityOccupancyBoundaryExtractor.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{

std::size_t index(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) * dimensions[1] + y) *
        dimensions[0] + x;
}

std::vector<std::uint8_t> emptyGrid(
    const std::array<int, 3> &dimensions)
{
    return std::vector<std::uint8_t>(
        static_cast<std::size_t>(dimensions[0]) *
            dimensions[1] * dimensions[2],
        0);
}

double signedVolume(const xjw::mesh::TriMesh &mesh)
{
    double volume = 0.0;
    for (const xjw::mesh::Triangle &face : mesh.faces)
    {
        const xjw::mesh::MeshVertex &a = mesh.vertices[face.v[0]];
        const xjw::mesh::MeshVertex &b = mesh.vertices[face.v[1]];
        const xjw::mesh::MeshVertex &c = mesh.vertices[face.v[2]];
        volume += static_cast<double>(a.x) *
            (static_cast<double>(b.y) * c.z -
             static_cast<double>(b.z) * c.y) -
            static_cast<double>(a.y) *
            (static_cast<double>(b.x) * c.z -
             static_cast<double>(b.z) * c.x) +
            static_cast<double>(a.z) *
            (static_cast<double>(b.x) * c.y -
             static_cast<double>(b.y) * c.x);
    }
    return volume / 6.0;
}

xjw::mesh::VisibilityOccupancyBoundaryResult extract(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied)
{
    return xjw::mesh::VisibilityOccupancyBoundaryExtractor::extract(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        dimensions,
        occupied);
}

} // namespace

TEST(VisibilityOccupancyBoundaryExtractorTest,
     ExtractsOutwardOrientedSingleCellBoundary)
{
    const std::array<int, 3> dimensions{3, 3, 3};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    occupied[index(dimensions, 1, 1, 1)] = 1;

    const auto result = extract(dimensions, occupied);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    EXPECT_EQ(result.statistics.occupiedCellCount, 1U);
    EXPECT_EQ(result.statistics.exposedQuadCount, 6U);
    EXPECT_EQ(result.mesh.vertexCount(), 8);
    EXPECT_EQ(result.mesh.faceCount(), 12);
    const auto quality =
        xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
    EXPECT_EQ(quality.boundaryEdgeCount, 0);
    EXPECT_EQ(quality.nonManifoldEdgeCount, 0);
    EXPECT_EQ(quality.componentCount, 1);
    EXPECT_EQ(quality.eulerCharacteristic, 2);
    EXPECT_TRUE(result.statistics.closedTwoManifold);
    EXPECT_EQ(result.statistics.eulerCharacteristic, 2);
    EXPECT_GT(signedVolume(result.mesh), 0.0);
}

TEST(VisibilityOccupancyBoundaryExtractorTest,
     PreservesOuterAndProtectedCavityComponents)
{
    const std::array<int, 3> dimensions{5, 5, 5};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    for (int z = 1; z <= 3; ++z)
    {
        for (int y = 1; y <= 3; ++y)
        {
            for (int x = 1; x <= 3; ++x)
            {
                occupied[index(dimensions, x, y, z)] =
                    x == 2 && y == 2 && z == 2 ? 0 : 1;
            }
        }
    }

    const auto result = extract(dimensions, occupied);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    const auto quality =
        xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
    EXPECT_EQ(quality.boundaryEdgeCount, 0);
    EXPECT_EQ(quality.nonManifoldEdgeCount, 0);
    EXPECT_EQ(quality.componentCount, 2);
    EXPECT_EQ(quality.eulerCharacteristic, 4);
}

TEST(VisibilityOccupancyBoundaryExtractorTest,
     PreservesStraightTunnelGenus)
{
    const std::array<int, 3> dimensions{5, 5, 5};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    for (int z = 1; z <= 3; ++z)
    {
        for (int y = 1; y <= 3; ++y)
        {
            for (int x = 1; x <= 3; ++x)
            {
                occupied[index(dimensions, x, y, z)] =
                    x == 2 && y == 2 ? 0 : 1;
            }
        }
    }

    const auto result = extract(dimensions, occupied);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    const auto quality =
        xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
    EXPECT_EQ(quality.boundaryEdgeCount, 0);
    EXPECT_EQ(quality.nonManifoldEdgeCount, 0);
    EXPECT_EQ(quality.componentCount, 1);
    EXPECT_EQ(quality.eulerCharacteristic, 0);
    EXPECT_DOUBLE_EQ(quality.closedGenusEstimate, 1.0);
}

TEST(VisibilityOccupancyBoundaryExtractorTest,
     DetectsCellsTouchingOnlyAlongAnEdge)
{
    const std::array<int, 3> dimensions{4, 4, 4};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    occupied[index(dimensions, 1, 1, 1)] = 1;
    occupied[index(dimensions, 1, 2, 2)] = 1;

    const auto result = extract(dimensions, occupied);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    EXPECT_GT(result.statistics.nonManifoldEdgeCount, 0U);
    EXPECT_FALSE(result.statistics.closedTwoManifold);
}

TEST(VisibilityOccupancyBoundaryExtractorTest,
     DetectsCellsTouchingOnlyAtAVertex)
{
    const std::array<int, 3> dimensions{4, 4, 4};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    occupied[index(dimensions, 1, 1, 1)] = 1;
    occupied[index(dimensions, 2, 2, 2)] = 1;

    const auto result = extract(dimensions, occupied);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    EXPECT_EQ(result.statistics.nonManifoldEdgeCount, 0U);
    EXPECT_GT(result.statistics.nonManifoldVertexCount, 0U);
    EXPECT_FALSE(result.statistics.closedTwoManifold);
}
