#include "VisibilityOccupancyCarrierFairer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace
{

using Edge = std::pair<int, int>;

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
    return mesh;
}

xjw::mesh::TriMesh makeClosedCube(bool rough)
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices = {
        {-1.0f, -1.0f, -1.0f},
        {1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, -1.0f},
        {-1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, 1.0f},
        {1.0f, -1.0f, 1.0f},
        {rough ? 1.45f : 1.0f,
         rough ? 1.20f : 1.0f,
         rough ? 1.40f : 1.0f},
        {-1.0f, 1.0f, 1.0f}};
    mesh.faces = {
        {{0, 2, 1}}, {{0, 3, 2}},
        {{4, 5, 6}}, {{4, 6, 7}},
        {{0, 1, 5}}, {{0, 5, 4}},
        {{1, 2, 6}}, {{1, 6, 5}},
        {{2, 3, 7}}, {{2, 7, 6}},
        {{3, 0, 4}}, {{3, 4, 7}}};
    return mesh;
}

std::map<Edge, int> edgeUseCounts(const xjw::mesh::TriMesh &mesh)
{
    std::map<Edge, int> counts;
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

void expectTopologyAndAttributesPreserved(
    const xjw::mesh::TriMesh &source,
    const xjw::mesh::TriMesh &result)
{
    ASSERT_EQ(result.vertices.size(), source.vertices.size());
    ASSERT_EQ(result.faces.size(), source.faces.size());
    EXPECT_EQ(result.hasVertexColors, source.hasVertexColors);
    for (std::size_t index = 0; index < source.faces.size(); ++index)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            EXPECT_EQ(result.faces[index].v[corner],
                      source.faces[index].v[corner]);
        }
    }
    for (std::size_t index = 0; index < source.vertices.size(); ++index)
    {
        EXPECT_FLOAT_EQ(result.vertices[index].nx, source.vertices[index].nx);
        EXPECT_FLOAT_EQ(result.vertices[index].ny, source.vertices[index].ny);
        EXPECT_FLOAT_EQ(result.vertices[index].nz, source.vertices[index].nz);
        EXPECT_EQ(result.vertices[index].r, source.vertices[index].r);
        EXPECT_EQ(result.vertices[index].g, source.vertices[index].g);
        EXPECT_EQ(result.vertices[index].b, source.vertices[index].b);
    }
    const auto source_edges = edgeUseCounts(source);
    const auto result_edges = edgeUseCounts(result);
    EXPECT_EQ(result_edges, source_edges);
    EXPECT_TRUE(std::all_of(
        result_edges.begin(),
        result_edges.end(),
        [](const auto &entry)
        {
            return entry.second == 2;
        }));
    EXPECT_EQ(static_cast<int>(result.vertices.size()) -
                  static_cast<int>(result_edges.size()) +
                  static_cast<int>(result.faces.size()),
              2);
}

double vertexDistance(
    const xjw::mesh::MeshVertex &first,
    const xjw::mesh::MeshVertex &second)
{
    const double dx = static_cast<double>(first.x) - second.x;
    const double dy = static_cast<double>(first.y) - second.y;
    const double dz = static_cast<double>(first.z) - second.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double inverseDistanceRoughness(const xjw::mesh::TriMesh &mesh)
{
    const auto edges = edgeUseCounts(mesh);
    std::vector<std::vector<int>> neighbours(mesh.vertices.size());
    for (const auto &[edge, use_count] : edges)
    {
        (void)use_count;
        neighbours[edge.first].push_back(edge.second);
        neighbours[edge.second].push_back(edge.first);
    }
    double roughness = 0.0;
    for (std::size_t index = 0; index < mesh.vertices.size(); ++index)
    {
        double weighted_x = 0.0;
        double weighted_y = 0.0;
        double weighted_z = 0.0;
        double weight_sum = 0.0;
        for (const int neighbour : neighbours[index])
        {
            const double distance = vertexDistance(
                mesh.vertices[index], mesh.vertices[neighbour]);
            const double weight = 1.0 / distance;
            weighted_x += weight * mesh.vertices[neighbour].x;
            weighted_y += weight * mesh.vertices[neighbour].y;
            weighted_z += weight * mesh.vertices[neighbour].z;
            weight_sum += weight;
        }
        const double dx = weighted_x / weight_sum - mesh.vertices[index].x;
        const double dy = weighted_y / weight_sum - mesh.vertices[index].y;
        const double dz = weighted_z / weight_sum - mesh.vertices[index].z;
        roughness += dx * dx + dy * dy + dz * dz;
    }
    return roughness;
}

void expectMeshesExactlyEqual(
    const xjw::mesh::TriMesh &first,
    const xjw::mesh::TriMesh &second)
{
    ASSERT_EQ(first.vertices.size(), second.vertices.size());
    ASSERT_EQ(first.faces.size(), second.faces.size());
    for (std::size_t index = 0; index < first.vertices.size(); ++index)
    {
        EXPECT_FLOAT_EQ(first.vertices[index].x, second.vertices[index].x);
        EXPECT_FLOAT_EQ(first.vertices[index].y, second.vertices[index].y);
        EXPECT_FLOAT_EQ(first.vertices[index].z, second.vertices[index].z);
    }
    for (std::size_t index = 0; index < first.faces.size(); ++index)
    {
        EXPECT_EQ(first.faces[index].v[0], second.faces[index].v[0]);
        EXPECT_EQ(first.faces[index].v[1], second.faces[index].v[1]);
        EXPECT_EQ(first.faces[index].v[2], second.faces[index].v[2]);
    }
}

xjw::mesh::VisibilityOccupancyCarrierFairingOptions permissiveOptions()
{
    xjw::mesh::VisibilityOccupancyCarrierFairingOptions options;
    options.iterations = 2;
    options.absoluteMaximumDisplacement = 0.30;
    options.minimumNormalDot = -0.25;
    options.minimumFaceAreaRatio = 0.05;
    options.minimumSurfaceAreaRatio = 0.40;
    options.maximumSurfaceAreaRatio = 1.60;
    options.minimumAbsoluteVolumeRatio = 0.35;
    options.maximumAbsoluteVolumeRatio = 1.65;
    return options;
}

} // namespace

TEST(VisibilityOccupancyCarrierFairerTest,
     PreservesClosedCubeAndTetrahedronTopology)
{
    for (const xjw::mesh::TriMesh &source :
         {makeClosedCube(false), makeClosedTetrahedron()})
    {
        const auto result =
            xjw::mesh::VisibilityOccupancyCarrierFairer::fair(
                source, permissiveOptions());
        ASSERT_TRUE(result.ok) << result.errorMessage;
        EXPECT_FALSE(result.cancelled);
        EXPECT_FALSE(result.rolledBack);
        EXPECT_EQ(result.statistics.completedIterationCount, 2);
        EXPECT_EQ(result.statistics.acceptedHalfStepCount, 4);
        expectTopologyAndAttributesPreserved(source, result.mesh);
    }
}

TEST(VisibilityOccupancyCarrierFairerTest,
     ReducesRoughnessAndIsDeterministic)
{
    const xjw::mesh::TriMesh source = makeClosedCube(true);
    auto options = permissiveOptions();
    options.iterations = 3;
    options.absoluteMaximumDisplacement = 0.45;
    options.minimumAbsoluteVolumeRatio = 0.25;

    const auto first =
        xjw::mesh::VisibilityOccupancyCarrierFairer::fair(source, options);
    const auto second =
        xjw::mesh::VisibilityOccupancyCarrierFairer::fair(source, options);

    ASSERT_TRUE(first.ok) << first.errorMessage;
    ASSERT_TRUE(second.ok) << second.errorMessage;
    EXPECT_LT(inverseDistanceRoughness(first.mesh),
              inverseDistanceRoughness(source));
    expectMeshesExactlyEqual(first.mesh, second.mesh);
}

TEST(VisibilityOccupancyCarrierFairerTest,
     HonorsAbsoluteAndMeanEdgeDisplacementLimits)
{
    const xjw::mesh::TriMesh source = makeClosedCube(true);
    auto absolute_options = permissiveOptions();
    absolute_options.iterations = 8;
    absolute_options.absoluteMaximumDisplacement = 0.025;
    absolute_options.maximumDisplacementMeanEdgeRatio = 10.0;
    const auto absolute_result =
        xjw::mesh::VisibilityOccupancyCarrierFairer::fair(
            source, absolute_options);
    ASSERT_TRUE(absolute_result.ok) << absolute_result.errorMessage;
    EXPECT_GT(absolute_result.statistics.displacementClampedVertexCount, 0U);
    EXPECT_NEAR(
        absolute_result.statistics.resolvedMaximumDisplacement,
        0.025,
        1.0e-12);
    for (std::size_t index = 0; index < source.vertices.size(); ++index)
    {
        EXPECT_LE(vertexDistance(
                      source.vertices[index],
                      absolute_result.mesh.vertices[index]),
                  0.025001);
    }

    auto ratio_options = permissiveOptions();
    ratio_options.absoluteMaximumDisplacement = 0.0;
    ratio_options.maximumDisplacementMeanEdgeRatio = 0.01;
    const auto ratio_result =
        xjw::mesh::VisibilityOccupancyCarrierFairer::fair(
            source, ratio_options);
    ASSERT_TRUE(ratio_result.ok) << ratio_result.errorMessage;
    EXPECT_NEAR(
        ratio_result.statistics.resolvedMaximumDisplacement,
        ratio_result.statistics.meanEdgeLength * 0.01,
        1.0e-12);
    EXPECT_LE(ratio_result.statistics.maximumAppliedDisplacement,
              ratio_result.statistics.resolvedMaximumDisplacement + 1.0e-6);
}

TEST(VisibilityOccupancyCarrierFairerTest,
     RollsBackAggressiveProposalRejectedByGlobalGuard)
{
    const xjw::mesh::TriMesh source = makeClosedCube(false);
    auto options = permissiveOptions();
    options.iterations = 1;
    options.lambda = 4.0;
    options.mu = -4.1;
    options.absoluteMaximumDisplacement = 100.0;
    options.minimumNormalDot = -1.0;
    options.minimumFaceAreaRatio = 1.0e-8;
    options.minimumSurfaceAreaRatio = 0.99;
    options.maximumSurfaceAreaRatio = 1.01;
    options.minimumAbsoluteVolumeRatio = 0.99;
    options.maximumAbsoluteVolumeRatio = 1.01;

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierFairer::fair(source, options);

    EXPECT_FALSE(result.ok);
    EXPECT_FALSE(result.cancelled);
    EXPECT_TRUE(result.rolledBack);
    EXPECT_EQ(result.statistics.rollbackCount, 1U);
    EXPECT_FALSE(result.errorMessage.empty());
    expectMeshesExactlyEqual(result.mesh, source);
}

TEST(VisibilityOccupancyCarrierFairerTest,
     CancellationIsAtomic)
{
    const xjw::mesh::TriMesh source = makeClosedCube(true);
    auto options = permissiveOptions();
    options.iterations = 10;
    int cancellation_checks = 0;
    options.isCancelled = [&cancellation_checks]()
    {
        ++cancellation_checks;
        return cancellation_checks >= 30;
    };

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierFairer::fair(source, options);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.cancelled);
    EXPECT_TRUE(result.rolledBack);
    EXPECT_EQ(result.statistics.rollbackCount, 1U);
    EXPECT_FALSE(result.errorMessage.empty());
    expectMeshesExactlyEqual(result.mesh, source);
}
