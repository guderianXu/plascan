#include "VisibilityOccupancyCarrierSubdivider.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <utility>

namespace
{

using EdgeKey = std::pair<int, int>;

xjw::mesh::TriMesh makeClosedTetrahedron()
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices = {
        {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 10, 20, 30},
        {-1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 20, 40, 60},
        {-1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 30, 60, 90},
        {1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 40, 80, 120}};
    mesh.faces = {
        {{0, 2, 1}},
        {{0, 1, 3}},
        {{0, 3, 2}},
        {{1, 2, 3}}};
    mesh.hasVertexColors = true;
    return mesh;
}

std::map<EdgeKey, int> edgeUseCounts(const xjw::mesh::TriMesh &mesh)
{
    std::map<EdgeKey, int> counts;
    for (const xjw::mesh::Triangle &face : mesh.faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const int first = face.v[corner];
            const int second = face.v[(corner + 1) % 3];
            ++counts[{std::min(first, second), std::max(first, second)}];
        }
    }
    return counts;
}

double signedVolume(const xjw::mesh::TriMesh &mesh)
{
    double volume = 0.0;
    for (const xjw::mesh::Triangle &face : mesh.faces)
    {
        const auto &first = mesh.vertices[face.v[0]];
        const auto &second = mesh.vertices[face.v[1]];
        const auto &third = mesh.vertices[face.v[2]];
        volume += static_cast<double>(first.x) *
            (static_cast<double>(second.y) * third.z -
             static_cast<double>(second.z) * third.y) -
            static_cast<double>(first.y) *
            (static_cast<double>(second.x) * third.z -
             static_cast<double>(second.z) * third.x) +
            static_cast<double>(first.z) *
            (static_cast<double>(second.x) * third.y -
             static_cast<double>(second.y) * third.x);
    }
    return volume / 6.0;
}

std::array<double, 3> faceNormal(
    const xjw::mesh::TriMesh &mesh,
    const xjw::mesh::Triangle &face)
{
    const auto &first = mesh.vertices[face.v[0]];
    const auto &second = mesh.vertices[face.v[1]];
    const auto &third = mesh.vertices[face.v[2]];
    const double first_x = static_cast<double>(second.x) - first.x;
    const double first_y = static_cast<double>(second.y) - first.y;
    const double first_z = static_cast<double>(second.z) - first.z;
    const double second_x = static_cast<double>(third.x) - first.x;
    const double second_y = static_cast<double>(third.y) - first.y;
    const double second_z = static_cast<double>(third.z) - first.z;
    return {
        first_y * second_z - first_z * second_y,
        first_z * second_x - first_x * second_z,
        first_x * second_y - first_y * second_x};
}

void expectMeshesEqual(
    const xjw::mesh::TriMesh &first,
    const xjw::mesh::TriMesh &second)
{
    ASSERT_EQ(first.vertices.size(), second.vertices.size());
    ASSERT_EQ(first.faces.size(), second.faces.size());
    EXPECT_EQ(first.hasVertexColors, second.hasVertexColors);
    for (std::size_t index = 0; index < first.vertices.size(); ++index)
    {
        const auto &lhs = first.vertices[index];
        const auto &rhs = second.vertices[index];
        EXPECT_FLOAT_EQ(lhs.x, rhs.x);
        EXPECT_FLOAT_EQ(lhs.y, rhs.y);
        EXPECT_FLOAT_EQ(lhs.z, rhs.z);
        EXPECT_FLOAT_EQ(lhs.nx, rhs.nx);
        EXPECT_FLOAT_EQ(lhs.ny, rhs.ny);
        EXPECT_FLOAT_EQ(lhs.nz, rhs.nz);
        EXPECT_EQ(lhs.r, rhs.r);
        EXPECT_EQ(lhs.g, rhs.g);
        EXPECT_EQ(lhs.b, rhs.b);
    }
    for (std::size_t index = 0; index < first.faces.size(); ++index)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            EXPECT_EQ(
                first.faces[index].v[corner],
                second.faces[index].v[corner]);
        }
    }
}

} // namespace

TEST(VisibilityOccupancyCarrierSubdividerTest,
     PreservesClosedTetrahedronTopologyAndAttributes)
{
    const xjw::mesh::TriMesh source = makeClosedTetrahedron();

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierSubdivider::subdivide(source);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    EXPECT_FALSE(result.cancelled);
    EXPECT_EQ(result.statistics.inputVertexCount, 4U);
    EXPECT_EQ(result.statistics.inputFaceCount, 4U);
    EXPECT_EQ(result.statistics.validatedFaceCount, 4U);
    EXPECT_EQ(result.statistics.uniqueInputEdgeCount, 6U);
    EXPECT_EQ(result.statistics.createdMidpointVertexCount, 6U);
    EXPECT_EQ(result.statistics.subdividedFaceCount, 4U);
    EXPECT_EQ(result.statistics.outputVertexCount, 10U);
    EXPECT_EQ(result.statistics.outputFaceCount, 16U);
    EXPECT_EQ(result.mesh.vertexCount(), 10);
    EXPECT_EQ(result.mesh.faceCount(), 16);

    const std::map<EdgeKey, int> edges = edgeUseCounts(result.mesh);
    const std::size_t boundary_edge_count =
        static_cast<std::size_t>(std::count_if(
            edges.cbegin(),
            edges.cend(),
            [](const auto &entry)
            {
                return entry.second == 1;
            }));
    EXPECT_EQ(edges.size(), 24U);
    EXPECT_EQ(boundary_edge_count, 0U);
    EXPECT_TRUE(std::all_of(
        edges.cbegin(),
        edges.cend(),
        [](const auto &entry)
        {
            return entry.second == 2;
        }));
    const int euler_characteristic =
        result.mesh.vertexCount() -
        static_cast<int>(edges.size()) +
        result.mesh.faceCount();
    EXPECT_EQ(euler_characteristic, 2);
    EXPECT_NEAR(signedVolume(result.mesh), signedVolume(source), 1.0e-6);
    EXPECT_GT(signedVolume(result.mesh), 0.0);
    for (std::size_t face_index = 0;
         face_index < source.faces.size();
         ++face_index)
    {
        const std::array<double, 3> source_normal =
            faceNormal(source, source.faces[face_index]);
        for (std::size_t child = 0; child < 4; ++child)
        {
            const std::array<double, 3> child_normal = faceNormal(
                result.mesh,
                result.mesh.faces[face_index * 4 + child]);
            const double orientation_dot =
                source_normal[0] * child_normal[0] +
                source_normal[1] * child_normal[1] +
                source_normal[2] * child_normal[2];
            EXPECT_GT(orientation_dot, 0.0);
        }
    }

    // Sorted edge order assigns the (0, 1) midpoint the first new index.
    const xjw::mesh::MeshVertex &midpoint = result.mesh.vertices[4];
    EXPECT_FLOAT_EQ(midpoint.x, 0.0f);
    EXPECT_FLOAT_EQ(midpoint.y, 0.0f);
    EXPECT_FLOAT_EQ(midpoint.z, 1.0f);
    const float inverse_sqrt_two = 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR(midpoint.nx, inverse_sqrt_two, 1.0e-6f);
    EXPECT_NEAR(midpoint.ny, inverse_sqrt_two, 1.0e-6f);
    EXPECT_FLOAT_EQ(midpoint.nz, 0.0f);
    EXPECT_EQ(midpoint.r, 15);
    EXPECT_EQ(midpoint.g, 30);
    EXPECT_EQ(midpoint.b, 45);
}

TEST(VisibilityOccupancyCarrierSubdividerTest,
     ProducesIdenticalVertexAndFaceOrdering)
{
    const xjw::mesh::TriMesh source = makeClosedTetrahedron();

    const auto first =
        xjw::mesh::VisibilityOccupancyCarrierSubdivider::subdivide(source);
    const auto second =
        xjw::mesh::VisibilityOccupancyCarrierSubdivider::subdivide(source);

    ASSERT_TRUE(first.ok) << first.errorMessage;
    ASSERT_TRUE(second.ok) << second.errorMessage;
    expectMeshesEqual(first.mesh, second.mesh);
}

TEST(VisibilityOccupancyCarrierSubdividerTest,
     RejectsInvalidAndDegenerateFaces)
{
    xjw::mesh::TriMesh invalid = makeClosedTetrahedron();
    invalid.faces[0].v[2] = 4;
    const auto invalid_result =
        xjw::mesh::VisibilityOccupancyCarrierSubdivider::subdivide(invalid);
    EXPECT_FALSE(invalid_result.ok);
    EXPECT_FALSE(invalid_result.cancelled);
    EXPECT_FALSE(invalid_result.errorMessage.empty());

    xjw::mesh::TriMesh degenerate;
    degenerate.vertices = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f}};
    degenerate.faces = {{{0, 1, 2}}};
    const auto degenerate_result =
        xjw::mesh::VisibilityOccupancyCarrierSubdivider::subdivide(
            degenerate);
    EXPECT_FALSE(degenerate_result.ok);
    EXPECT_FALSE(degenerate_result.cancelled);
    EXPECT_FALSE(degenerate_result.errorMessage.empty());
}

TEST(VisibilityOccupancyCarrierSubdividerTest,
     CancelsWithoutReturningAPartialMesh)
{
    int cancellation_checks = 0;
    xjw::mesh::VisibilityOccupancyCarrierSubdivisionOptions options;
    options.isCancelled = [&cancellation_checks]()
    {
        ++cancellation_checks;
        return cancellation_checks >= 3;
    };

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierSubdivider::subdivide(
            makeClosedTetrahedron(),
            options);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.cancelled);
    EXPECT_FALSE(result.errorMessage.empty());
    EXPECT_TRUE(result.mesh.vertices.empty());
    EXPECT_TRUE(result.mesh.faces.empty());
    EXPECT_EQ(result.statistics.outputVertexCount, 0U);
    EXPECT_EQ(result.statistics.outputFaceCount, 0U);
}
