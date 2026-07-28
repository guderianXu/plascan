#include "MeshTopologyQuality.h"
#include "OpenMeshSimplifier.h"

#include <gtest/gtest.h>

#include <cmath>

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

} // namespace

TEST(OpenMeshSimplifierTest, PreservesClosedManifoldTopology)
{
    if (!xjw::mesh::openMeshSimplifierAvailable())
    {
        GTEST_SKIP() << "OpenMesh is unavailable";
    }

    xjw::mesh::TriMesh mesh = makeTorus(40, 16);
    const xjw::mesh::MeshTopologyQualityStatistics topology_before =
        xjw::mesh::evaluateMeshTopologyQuality(mesh);
    ASSERT_EQ(topology_before.boundaryEdgeCount, 0);
    ASSERT_EQ(topology_before.nonManifoldEdgeCount, 0);
    ASSERT_EQ(topology_before.componentCount, 1);

    xjw::mesh::OpenMeshSimplifyOptions options;
    options.targetFaceCount = 320;
    options.maximumNormalDeviationDegrees = 45.0f;
    const xjw::mesh::OpenMeshSimplifyStatistics statistics =
        xjw::mesh::simplifyMeshWithOpenMesh(&mesh, options);

    ASSERT_TRUE(statistics.succeeded) << statistics.error;
    EXPECT_EQ(statistics.removedContradictoryFaceCount, 0);
    EXPECT_TRUE(statistics.reachedTarget);
    EXPECT_LE(mesh.faceCount(), options.targetFaceCount);
    const xjw::mesh::MeshTopologyQualityStatistics topology_after =
        xjw::mesh::evaluateMeshTopologyQuality(mesh);
    EXPECT_EQ(topology_after.boundaryEdgeCount, 0);
    EXPECT_EQ(topology_after.nonManifoldEdgeCount, 0);
    EXPECT_EQ(topology_after.componentCount, 1);
}

TEST(OpenMeshSimplifierTest, CancellationLeavesInputUnchanged)
{
    if (!xjw::mesh::openMeshSimplifierAvailable())
    {
        GTEST_SKIP() << "OpenMesh is unavailable";
    }

    xjw::mesh::TriMesh mesh = makeTorus(40, 16);
    const int input_vertices = mesh.vertexCount();
    const int input_faces = mesh.faceCount();
    int cancellation_checks = 0;

    xjw::mesh::OpenMeshSimplifyOptions options;
    options.targetFaceCount = 160;
    options.notificationInterval = 1;
    options.isCancelled = [&cancellation_checks]()
    {
        return ++cancellation_checks > 8;
    };
    const xjw::mesh::OpenMeshSimplifyStatistics statistics =
        xjw::mesh::simplifyMeshWithOpenMesh(&mesh, options);

    EXPECT_TRUE(statistics.cancelled);
    EXPECT_FALSE(statistics.succeeded);
    EXPECT_EQ(mesh.vertexCount(), input_vertices);
    EXPECT_EQ(mesh.faceCount(), input_faces);
}
