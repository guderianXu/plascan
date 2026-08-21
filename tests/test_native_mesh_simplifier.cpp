#include "MeshTopologyQuality.h"
#include "NativeMeshSimplifier.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace
{

xjw::mesh::TriMesh makeTorus(int majorSegments, int minorSegments)
{
    xjw::mesh::TriMesh mesh;
    constexpr float pi = 3.14159265358979323846f;
    mesh.vertices.reserve(
        static_cast<std::size_t>(majorSegments * minorSegments));
    mesh.faces.reserve(
        static_cast<std::size_t>(majorSegments * minorSegments * 2));

    for (int major = 0; major < majorSegments; ++major)
    {
        const float u = 2.0f * pi * static_cast<float>(major) /
            static_cast<float>(majorSegments);
        for (int minor = 0; minor < minorSegments; ++minor)
        {
            const float v = 2.0f * pi * static_cast<float>(minor) /
                static_cast<float>(minorSegments);
            const float radial = 2.0f + 0.65f * std::cos(v);
            xjw::mesh::MeshVertex vertex;
            vertex.x = radial * std::cos(u);
            vertex.y = radial * std::sin(u);
            vertex.z = 0.65f * std::sin(v);
            vertex.r = static_cast<std::uint8_t>((major * 17) % 256);
            vertex.g = static_cast<std::uint8_t>((minor * 29) % 256);
            vertex.b = static_cast<std::uint8_t>(((major + minor) * 11) % 256);
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
            const int a = index(major, minor);
            const int b = index(next_major, minor);
            const int c = index(next_major, next_minor);
            const int d = index(major, next_minor);
            mesh.faces.push_back({{a, b, c}});
            mesh.faces.push_back({{a, c, d}});
        }
    }
    return mesh;
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

} // namespace

TEST(NativeMeshSimplifierTest, PreservesClosedManifoldTopologyAndAttributes)
{
    xjw::mesh::TriMesh mesh = makeTorus(40, 16);
    const xjw::mesh::MeshTopologyQualityStatistics topology_before =
        xjw::mesh::evaluateMeshTopologyQuality(mesh);

    xjw::mesh::NativeMeshSimplifyOptions options;
    options.targetFaceCount = 320;
    options.maximumNormalDeviationDegrees = 45.0f;
    const xjw::mesh::NativeMeshSimplifyStatistics statistics =
        xjw::mesh::simplifyMeshTopologySafe(&mesh, options);

    ASSERT_TRUE(statistics.succeeded) << statistics.error;
    EXPECT_TRUE(statistics.reachedTarget);
    EXPECT_LE(mesh.faceCount(), options.targetFaceCount);
    EXPECT_TRUE(mesh.hasVertexColors);
    const xjw::mesh::MeshTopologyQualityStatistics topology_after =
        xjw::mesh::evaluateMeshTopologyQuality(mesh);
    EXPECT_EQ(topology_after.componentCount, topology_before.componentCount);
    EXPECT_EQ(topology_after.boundaryEdgeCount, topology_before.boundaryEdgeCount);
    EXPECT_EQ(topology_after.nonManifoldEdgeCount, topology_before.nonManifoldEdgeCount);
    EXPECT_EQ(topology_after.eulerCharacteristic, topology_before.eulerCharacteristic);
    for (const xjw::mesh::MeshVertex &vertex : mesh.vertices)
    {
        const double normal_length = std::sqrt(
            static_cast<double>(vertex.nx) * vertex.nx +
            static_cast<double>(vertex.ny) * vertex.ny +
            static_cast<double>(vertex.nz) * vertex.nz);
        EXPECT_NEAR(normal_length, 1.0, 1.0e-4);
    }
}

TEST(NativeMeshSimplifierTest, CancellationLeavesInputUnchanged)
{
    xjw::mesh::TriMesh mesh = makeTorus(40, 16);
    const int input_vertices = mesh.vertexCount();
    const int input_faces = mesh.faceCount();
    bool cancel = false;

    xjw::mesh::NativeMeshSimplifyOptions options;
    options.targetFaceCount = 160;
    options.progress = [&cancel](int, int)
    {
        cancel = true;
    };
    options.isCancelled = [&cancel]()
    {
        return cancel;
    };
    const xjw::mesh::NativeMeshSimplifyStatistics statistics =
        xjw::mesh::simplifyMeshTopologySafe(&mesh, options);

    EXPECT_TRUE(statistics.cancelled);
    EXPECT_FALSE(statistics.succeeded);
    EXPECT_EQ(mesh.vertexCount(), input_vertices);
    EXPECT_EQ(mesh.faceCount(), input_faces);
}

TEST(NativeMeshSimplifierTest, SmoothingIsBoundedAndTopologySafe)
{
    xjw::mesh::TriMesh unsmoothed = makeTorus(40, 16);
    xjw::mesh::TriMesh smoothed = unsmoothed;

    xjw::mesh::NativeMeshSimplifyOptions base_options;
    base_options.targetFaceCount = 480;
    base_options.maximumNormalDeviationDegrees = 45.0f;
    const xjw::mesh::NativeMeshSimplifyStatistics unsmoothed_statistics =
        xjw::mesh::simplifyMeshTopologySafe(&unsmoothed, base_options);

    xjw::mesh::NativeMeshSimplifyOptions smooth_options = base_options;
    smooth_options.smoothingIterations = 3;
    smooth_options.smoothingMaximumDisplacement = 0.025f;
    smooth_options.smoothingFeatureAngleDegrees = 120.0f;
    const xjw::mesh::NativeMeshSimplifyStatistics smoothed_statistics =
        xjw::mesh::simplifyMeshTopologySafe(&smoothed, smooth_options);

    ASSERT_TRUE(unsmoothed_statistics.succeeded) << unsmoothed_statistics.error;
    ASSERT_TRUE(smoothed_statistics.succeeded) << smoothed_statistics.error;
    ASSERT_TRUE(smoothed_statistics.smoothingApplied);
    ASSERT_EQ(smoothed.vertexCount(), unsmoothed.vertexCount());
    ASSERT_EQ(smoothed.faceCount(), unsmoothed.faceCount());
    bool any_vertex_moved = false;
    for (std::size_t index = 0; index < smoothed.vertices.size(); ++index)
    {
        const double displacement = vertexDistance(
            smoothed.vertices[index],
            unsmoothed.vertices[index]);
        EXPECT_LE(displacement, smooth_options.smoothingMaximumDisplacement + 1.0e-5);
        any_vertex_moved = any_vertex_moved || displacement > 1.0e-7;
    }
    EXPECT_TRUE(any_vertex_moved);
    const xjw::mesh::MeshTopologyQualityStatistics topology =
        xjw::mesh::evaluateMeshTopologyQuality(smoothed);
    EXPECT_EQ(topology.componentCount, 1);
    EXPECT_EQ(topology.boundaryEdgeCount, 0);
    EXPECT_EQ(topology.nonManifoldEdgeCount, 0);
}
