#include "Mc33IsoSurfaceExtractor.h"
#include "MeshTopologyQuality.h"
#include "MeshVoxelTopologyRepair.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

xjw::mesh::TriMesh makeBox()
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices = {
        {-1.0f, -0.8f, -0.6f}, {1.0f, -0.8f, -0.6f},
        {1.0f, 0.8f, -0.6f}, {-1.0f, 0.8f, -0.6f},
        {-1.0f, -0.8f, 0.6f}, {1.0f, -0.8f, 0.6f},
        {1.0f, 0.8f, 0.6f}, {-1.0f, 0.8f, 0.6f}};
    mesh.faces = {
        {{0, 2, 1}}, {{0, 3, 2}}, {{4, 5, 6}}, {{4, 6, 7}},
        {{0, 1, 5}}, {{0, 5, 4}}, {{1, 2, 6}}, {{1, 6, 5}},
        {{2, 3, 7}}, {{2, 7, 6}}, {{3, 0, 4}}, {{3, 4, 7}}};
    return mesh;
}

xjw::mesh::TriMesh makeTorus(int majorSegments,
                             int minorSegments,
                             float majorRadius,
                             float minorRadius)
{
    constexpr double pi = 3.14159265358979323846;
    xjw::mesh::TriMesh mesh;
    for (int major = 0; major < majorSegments; ++major)
    {
        const double u = 2.0 * pi * major / majorSegments;
        for (int minor = 0; minor < minorSegments; ++minor)
        {
            const double v = 2.0 * pi * minor / minorSegments;
            const double radial = majorRadius + minorRadius * std::cos(v);
            xjw::mesh::MeshVertex vertex;
            vertex.x = static_cast<float>(radial * std::cos(u));
            vertex.y = static_cast<float>(radial * std::sin(u));
            vertex.z = static_cast<float>(minorRadius * std::sin(v));
            mesh.vertices.push_back(vertex);
        }
    }
    const auto index = [minorSegments](int major, int minor)
    {
        return major * minorSegments + minor;
    };
    for (int major = 0; major < majorSegments; ++major)
    {
        const int next_major = (major + 1) % majorSegments;
        for (int minor = 0; minor < minorSegments; ++minor)
        {
            const int next_minor = (minor + 1) % minorSegments;
            const int v00 = index(major, minor);
            const int v10 = index(next_major, minor);
            const int v11 = index(next_major, next_minor);
            const int v01 = index(major, next_minor);
            mesh.faces.push_back({{v00, v10, v11}});
            mesh.faces.push_back({{v00, v11, v01}});
        }
    }
    return mesh;
}

void expectStrictGenusZero(
    const xjw::mesh::MeshVoxelTopologyRepairResult &result)
{
    ASSERT_TRUE(result.ok) << result.errorMessage;
    const auto quality =
        xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
    EXPECT_EQ(quality.componentCount, 1);
    EXPECT_EQ(quality.boundaryEdgeCount, 0);
    EXPECT_EQ(quality.nonManifoldEdgeCount, 0);
    EXPECT_EQ(quality.nonManifoldVertexCount, 0);
    EXPECT_EQ(quality.componentEulerCharacteristics, std::vector<int>({2}));
    EXPECT_EQ(result.statistics.cubicalEulerAfter, 1);
    EXPECT_EQ(result.statistics.outputSurfaceEulerCharacteristic, 2);
}

} // namespace

TEST(MeshVoxelTopologyRepairTest, PreservesClosedGenusZeroBody)
{
    xjw::mesh::MeshVoxelTopologyRepairOptions options;
    options.targetResolution = 40;

    const auto result =
        xjw::mesh::MeshVoxelTopologyRepair::repair(makeBox(), options);

    expectStrictGenusZero(result);
    EXPECT_EQ(result.statistics.selectedClosingRadius, 0);
    EXPECT_FALSE(result.statistics.usedLargestComponentFallback);
    EXPECT_GT(result.statistics.enclosedInteriorCellCountAfter, 0U);
    EXPECT_TRUE(result.statistics.smoothExtractionPreferred);
    if (xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        EXPECT_TRUE(result.statistics.smoothExtractionAvailable);
        EXPECT_TRUE(result.statistics.smoothExtractionAttempted);
        EXPECT_TRUE(result.statistics.smoothExtractionAccepted);
        EXPECT_FALSE(result.statistics.cellBoundaryExtractionUsed);
        EXPECT_EQ(result.statistics.smoothExtractionComponentCount, 1);
        EXPECT_EQ(
            result.statistics.smoothExtractionSurfaceEulerCharacteristic,
            2);
    }
}

TEST(MeshVoxelTopologyRepairTest, ClosesNarrowTorusHandle)
{
    const xjw::mesh::TriMesh torus = makeTorus(48, 24, 0.62f, 0.54f);
    const auto input_quality =
        xjw::mesh::evaluateMeshTopologyQuality(torus);
    ASSERT_EQ(input_quality.componentEulerCharacteristics,
              std::vector<int>({0}));

    xjw::mesh::MeshVoxelTopologyRepairOptions options;
    options.targetResolution = 64;
    options.maximumClosingRadius = 4;
    const auto result =
        xjw::mesh::MeshVoxelTopologyRepair::repair(torus, options);

    expectStrictGenusZero(result);
    EXPECT_EQ(result.statistics.cubicalEulerBefore, 0);
    EXPECT_GT(result.statistics.selectedClosingRadius, 0);
}

TEST(MeshVoxelTopologyRepairTest, DefaultClosingBudgetKeepsVerifiedHeadroom)
{
    const xjw::mesh::MeshVoxelTopologyRepairOptions options;

    EXPECT_EQ(options.maximumClosingRadius, 8);
}

TEST(MeshVoxelTopologyRepairTest, FailsClosedForEmptyInput)
{
    const auto result = xjw::mesh::MeshVoxelTopologyRepair::repair({});

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.mesh.empty());
    EXPECT_FALSE(result.errorMessage.empty());
}

TEST(MeshVoxelTopologyRepairTest, UsesExactBoundaryWhenSmoothIsDisabled)
{
    xjw::mesh::MeshVoxelTopologyRepairOptions options;
    options.targetResolution = 40;
    options.preferSmoothExtraction = false;

    const auto result =
        xjw::mesh::MeshVoxelTopologyRepair::repair(makeBox(), options);

    expectStrictGenusZero(result);
    EXPECT_FALSE(result.statistics.smoothExtractionPreferred);
    EXPECT_FALSE(result.statistics.smoothExtractionAttempted);
    EXPECT_FALSE(result.statistics.smoothExtractionAccepted);
    EXPECT_TRUE(result.statistics.cellBoundaryExtractionUsed);
}

TEST(MeshVoxelTopologyRepairTest,
     FallsBackWhenSmoothingErasesSubVoxelBody)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }
    xjw::mesh::TriMesh input = makeBox();
    input.vertices.push_back({100.0f, 0.0f, 0.0f});
    xjw::mesh::MeshVoxelTopologyRepairOptions options;
    options.targetResolution = 24;
    options.maximumClosingRadius = 0;
    options.requireEnclosedInterior = false;

    const auto result =
        xjw::mesh::MeshVoxelTopologyRepair::repair(input, options);

    expectStrictGenusZero(result);
    EXPECT_TRUE(result.statistics.smoothExtractionAttempted);
    EXPECT_FALSE(result.statistics.smoothExtractionAccepted);
    EXPECT_TRUE(result.statistics.smoothExtractionRejectedByTopology);
    EXPECT_TRUE(result.statistics.cellBoundaryExtractionUsed);
}

TEST(MeshVoxelTopologyRepairTest, RepairsOptInRealMesh)
{
    const char *path = std::getenv("PLASCAN_REAL_MESH_REPAIR_INPUT");
    if (!path || *path == '\0')
    {
        GTEST_SKIP() << "PLASCAN_REAL_MESH_REPAIR_INPUT is not set";
    }
    xjw::mesh::TriMesh input;
    std::string load_error;
    ASSERT_TRUE(xjw::mesh::TriMesh::loadPLY(path, &input, &load_error))
        << load_error;
    xjw::mesh::MeshVoxelTopologyRepairOptions options;
    const char *resolution =
        std::getenv("PLASCAN_REAL_MESH_REPAIR_RESOLUTION");
    options.targetResolution = resolution && *resolution != '\0'
        ? std::max(24, std::atoi(resolution))
        : 192;
    options.maximumClosingRadius = 8;
    const auto result =
        xjw::mesh::MeshVoxelTopologyRepair::repair(input, options);
    EXPECT_TRUE(result.ok)
        << result.errorMessage
        << "; surface=" << result.statistics.surfaceVoxelCount
        << "; interior="
        << result.statistics.enclosedInteriorCellCountBefore
        << "; occupied=" << result.statistics.occupiedCellCountBefore
        << "; components="
        << result.statistics.occupiedComponentCountBefore
        << "; euler=" << result.statistics.cubicalEulerBefore;
    if (result.ok)
    {
        expectStrictGenusZero(result);
        const char *output_path =
            std::getenv("PLASCAN_REAL_MESH_REPAIR_OUTPUT");
        if (output_path && *output_path != '\0')
        {
            std::string save_error;
            ASSERT_TRUE(result.mesh.savePLY(output_path, &save_error))
                << save_error;
        }
        if (xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
        {
            EXPECT_TRUE(result.statistics.smoothExtractionAttempted);
            if (result.statistics.smoothExtractionAccepted)
            {
                EXPECT_TRUE(
                    result.statistics.smoothExtractionStrictGatePassed);
                EXPECT_FALSE(result.statistics.cellBoundaryExtractionUsed);
            }
            else
            {
                EXPECT_TRUE(
                    result.statistics.smoothExtractionRejectedByTopology);
                EXPECT_TRUE(result.statistics.cellBoundaryExtractionUsed);
            }
            const auto final_quality =
                xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
            std::cout
                << "real_mesh_repair smooth_accepted="
                << result.statistics.smoothExtractionAccepted
                << " smooth_vertices="
                << result.statistics.smoothExtractionVertexCount
                << " smooth_faces="
                << result.statistics.smoothExtractionFaceCount
                << " smooth_high_aspect_ratio="
                << result.statistics.smoothExtractionHighAspectFaceRatio
                << " smooth_extreme_aspect_ratio="
                << result.statistics.smoothExtractionExtremeAspectFaceRatio
                << " smooth_optimization_attempted="
                << result.statistics.smoothTriangleOptimizationAttempted
                << " smooth_optimization_accepted="
                << result.statistics.smoothTriangleOptimizationAccepted
                << " smooth_flipped_edges="
                << result.statistics
                       .smoothTriangleOptimizationFlippedEdgeCount
                << " smooth_relaxed_vertices="
                << result.statistics
                       .smoothTriangleOptimizationRelaxedVertexCount
                << " smooth_euler="
                << result.statistics
                       .smoothExtractionSurfaceEulerCharacteristic
                << " final_vertices=" << result.mesh.vertexCount()
                << " final_faces=" << result.mesh.faceCount()
                << " final_boundary=" << final_quality.boundaryEdgeCount
                << " final_non_manifold_edges="
                << final_quality.nonManifoldEdgeCount
                << " final_non_manifold_vertices="
                << final_quality.nonManifoldVertexCount
                << " final_euler=" << final_quality.eulerCharacteristic
                << " final_strict=" << final_quality.strictGatePassed
                << std::endl;
            if (result.statistics.smoothTriangleOptimizationAccepted)
            {
                EXPECT_TRUE(result.statistics.smoothExtractionAccepted);
                EXPECT_LE(
                    result.statistics.smoothExtractionHighAspectFaceRatio,
                    result.statistics
                        .smoothExtractionHighAspectFaceRatioBeforeOptimization);
                EXPECT_LE(
                    result.statistics.smoothExtractionExtremeAspectFaceRatio,
                    result.statistics
                        .smoothExtractionExtremeAspectFaceRatioBeforeOptimization);
            }
            EXPECT_TRUE(final_quality.strictGatePassed)
                << "smooth_high_aspect_ratio="
                << result.statistics.smoothExtractionHighAspectFaceRatio
                << "; smooth_extreme_aspect_ratio="
                << result.statistics.smoothExtractionExtremeAspectFaceRatio;
        }
    }
}
