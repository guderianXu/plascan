#include "VisibilityOccupancyCarrierFieldProjector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace
{

using Edge = std::pair<int, int>;

std::size_t gridIndex(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) * dimensions[1] + y) *
               dimensions[0] +
           x;
}

std::vector<float> makeField(
    const std::array<int, 3> &dimensions,
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::function<double(double, double, double)> &function)
{
    std::vector<float> result(
        static_cast<std::size_t>(dimensions[0]) * dimensions[1] *
        dimensions[2]);
    for (int z = 0; z < dimensions[2]; ++z)
    {
        const double wz = boundsMin[2] +
            (boundsMax[2] - boundsMin[2]) * z /
                static_cast<double>(dimensions[2] - 1);
        for (int y = 0; y < dimensions[1]; ++y)
        {
            const double wy = boundsMin[1] +
                (boundsMax[1] - boundsMin[1]) * y /
                    static_cast<double>(dimensions[1] - 1);
            for (int x = 0; x < dimensions[0]; ++x)
            {
                const double wx = boundsMin[0] +
                    (boundsMax[0] - boundsMin[0]) * x /
                        static_cast<double>(dimensions[0] - 1);
                result[gridIndex(dimensions, x, y, z)] =
                    static_cast<float>(function(wx, wy, wz));
            }
        }
    }
    return result;
}

xjw::mesh::TriMesh makeWall()
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices = {
        {-0.6f, -0.6f, 0.0f},
        {0.6f, -0.6f, 0.0f},
        {0.6f, 0.6f, 0.0f},
        {-0.6f, 0.6f, 0.0f}};
    mesh.faces = {{{0, 1, 2}}, {{0, 2, 3}}};
    return mesh;
}

xjw::mesh::TriMesh makeStairPatch()
{
    xjw::mesh::TriMesh mesh;
    constexpr std::array<float, 3> coordinates{-0.6f, 0.0f, 0.6f};
    constexpr std::array<float, 3> heights{0.20f, -0.18f, 0.16f};
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            mesh.vertices.push_back(
                {coordinates[x], coordinates[y], heights[x]});
        }
    }
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            const int first = y * 3 + x;
            mesh.faces.push_back({{first, first + 1, first + 4}});
            mesh.faces.push_back({{first, first + 4, first + 3}});
        }
    }
    return mesh;
}

xjw::mesh::TriMesh makeClosedCube()
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices = {
        {-0.55f, -0.55f, -0.55f, 1.0f, 0.0f, 0.0f, 10, 20, 30},
        {0.55f, -0.55f, -0.55f, 0.0f, 1.0f, 0.0f, 20, 40, 60},
        {0.55f, 0.55f, -0.55f, 0.0f, 0.0f, 1.0f, 30, 60, 90},
        {-0.55f, 0.55f, -0.55f, -1.0f, 0.0f, 0.0f, 40, 80, 120},
        {-0.55f, -0.55f, 0.55f, 0.0f, -1.0f, 0.0f, 50, 100, 150},
        {0.55f, -0.55f, 0.55f, 0.0f, 0.0f, -1.0f, 60, 120, 180},
        {0.55f, 0.55f, 0.55f, 0.5f, 0.5f, 0.5f, 70, 140, 210},
        {-0.55f, 0.55f, 0.55f, -0.5f, 0.5f, 0.5f, 80, 160, 240}};
    mesh.faces = {
        {{0, 2, 1}}, {{0, 3, 2}},
        {{4, 5, 6}}, {{4, 6, 7}},
        {{0, 1, 5}}, {{0, 5, 4}},
        {{1, 2, 6}}, {{1, 6, 5}},
        {{2, 3, 7}}, {{2, 7, 6}},
        {{3, 0, 4}}, {{3, 4, 7}}};
    return mesh;
}

std::array<double, 3> normalizedFaceNormal(
    const xjw::mesh::TriMesh &mesh,
    const xjw::mesh::Triangle &face)
{
    const auto &a = mesh.vertices[face.v[0]];
    const auto &b = mesh.vertices[face.v[1]];
    const auto &c = mesh.vertices[face.v[2]];
    const double ux = b.x - a.x;
    const double uy = b.y - a.y;
    const double uz = b.z - a.z;
    const double vx = c.x - a.x;
    const double vy = c.y - a.y;
    const double vz = c.z - a.z;
    std::array<double, 3> normal{
        uy * vz - uz * vy,
        uz * vx - ux * vz,
        ux * vy - uy * vx};
    const double norm = std::sqrt(
        normal[0] * normal[0] + normal[1] * normal[1] +
        normal[2] * normal[2]);
    for (double &value : normal)
    {
        value /= norm;
    }
    return normal;
}

double faceNormalRoughness(const xjw::mesh::TriMesh &mesh)
{
    std::map<Edge, std::vector<std::size_t>> adjacent_faces;
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index)
    {
        const auto &face = mesh.faces[face_index];
        for (int corner = 0; corner < 3; ++corner)
        {
            const int first = face.v[corner];
            const int second = face.v[(corner + 1) % 3];
            adjacent_faces[{std::min(first, second), std::max(first, second)}]
                .push_back(face_index);
        }
    }
    double roughness = 0.0;
    for (const auto &[edge, faces] : adjacent_faces)
    {
        static_cast<void>(edge);
        if (faces.size() != 2)
        {
            continue;
        }
        const auto first = normalizedFaceNormal(mesh, mesh.faces[faces[0]]);
        const auto second = normalizedFaceNormal(mesh, mesh.faces[faces[1]]);
        const double normal_dot = std::clamp(
            first[0] * second[0] + first[1] * second[1] +
                first[2] * second[2],
            -1.0,
            1.0);
        roughness += 1.0 - normal_dot;
    }
    return roughness;
}

void expectPositionsEqual(
    const xjw::mesh::TriMesh &first,
    const xjw::mesh::TriMesh &second)
{
    ASSERT_EQ(first.vertices.size(), second.vertices.size());
    for (std::size_t index = 0; index < first.vertices.size(); ++index)
    {
        EXPECT_FLOAT_EQ(first.vertices[index].x, second.vertices[index].x);
        EXPECT_FLOAT_EQ(first.vertices[index].y, second.vertices[index].y);
        EXPECT_FLOAT_EQ(first.vertices[index].z, second.vertices[index].z);
    }
}

xjw::mesh::VisibilityOccupancyCarrierFieldProjectionOptions permissiveOptions()
{
    xjw::mesh::VisibilityOccupancyCarrierFieldProjectionOptions options;
    options.smoothNarrowBand = false;
    options.minimumNormalDot = -1.0;
    options.minimumFaceAreaRatio = 0.01;
    options.minimumSurfaceAreaRatio = 0.40;
    options.maximumSurfaceAreaRatio = 1.60;
    options.minimumAbsoluteVolumeRatio = 0.20;
    options.maximumAbsoluteVolumeRatio = 5.0;
    return options;
}

} // namespace

TEST(VisibilityOccupancyCarrierFieldProjectorTest,
     KeepsStraightWallOnZeroSetUnchanged)
{
    const std::array<int, 3> dimensions{10, 10, 10};
    const std::array<float, 3> bounds_min{-1.0f, -1.0f, -1.0f};
    const std::array<float, 3> bounds_max{1.0f, 1.0f, 1.0f};
    const auto field = makeField(
        dimensions, bounds_min, bounds_max,
        [](double, double, double z) { return z; });
    const xjw::mesh::TriMesh source = makeWall();

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierFieldProjector::project(
            source, dimensions, bounds_min, bounds_max, field,
            permissiveOptions());

    ASSERT_TRUE(result.ok) << result.errorMessage;
    expectPositionsEqual(result.mesh, source);
    EXPECT_NEAR(result.statistics.meanAbsoluteFieldResidualAfter, 0.0, 1.0e-7);
}

TEST(VisibilityOccupancyCarrierFieldProjectorTest,
     ReducesStairResidualAndGeometricNormalRoughness)
{
    const std::array<int, 3> dimensions{10, 10, 10};
    const std::array<float, 3> bounds_min{-1.0f, -1.0f, -1.0f};
    const std::array<float, 3> bounds_max{1.0f, 1.0f, 1.0f};
    const auto field = makeField(
        dimensions, bounds_min, bounds_max,
        [](double, double, double z) { return z; });
    const xjw::mesh::TriMesh source = makeStairPatch();
    auto options = permissiveOptions();
    options.iterations = 3;

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierFieldProjector::project(
            source, dimensions, bounds_min, bounds_max, field, options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    EXPECT_GT(result.statistics.projectedVertexCount, 0U);
    EXPECT_LT(result.statistics.meanAbsoluteFieldResidualAfter,
              result.statistics.meanAbsoluteFieldResidualBefore);
    EXPECT_LE(result.statistics.p90AbsoluteFieldResidualAfter,
              result.statistics.p90AbsoluteFieldResidualBefore);
    EXPECT_LT(faceNormalRoughness(result.mesh), faceNormalRoughness(source));
}

TEST(VisibilityOccupancyCarrierFieldProjectorTest,
     PreservesClosedTopologyFaceBufferAttributesAndFinitePositions)
{
    const std::array<int, 3> dimensions{17, 17, 17};
    const std::array<float, 3> bounds_min{-1.0f, -1.0f, -1.0f};
    const std::array<float, 3> bounds_max{1.0f, 1.0f, 1.0f};
    const auto field = makeField(
        dimensions, bounds_min, bounds_max,
        [](double x, double y, double z)
        {
            return std::sqrt(x * x + y * y + z * z) - 0.80;
        });
    const xjw::mesh::TriMesh source = makeClosedCube();
    auto options = permissiveOptions();
    options.iterations = 2;

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierFieldProjector::project(
            source, dimensions, bounds_min, bounds_max, field, options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    ASSERT_EQ(result.mesh.faces.size(), source.faces.size());
    ASSERT_EQ(result.mesh.vertices.size(), source.vertices.size());
    EXPECT_EQ(result.mesh.hasVertexColors, source.hasVertexColors);
    for (std::size_t index = 0; index < source.faces.size(); ++index)
    {
        EXPECT_EQ(result.mesh.faces[index].v[0], source.faces[index].v[0]);
        EXPECT_EQ(result.mesh.faces[index].v[1], source.faces[index].v[1]);
        EXPECT_EQ(result.mesh.faces[index].v[2], source.faces[index].v[2]);
    }
    for (std::size_t index = 0; index < source.vertices.size(); ++index)
    {
        const auto &vertex = result.mesh.vertices[index];
        EXPECT_TRUE(std::isfinite(vertex.x));
        EXPECT_TRUE(std::isfinite(vertex.y));
        EXPECT_TRUE(std::isfinite(vertex.z));
        EXPECT_FLOAT_EQ(vertex.nx, source.vertices[index].nx);
        EXPECT_FLOAT_EQ(vertex.ny, source.vertices[index].ny);
        EXPECT_FLOAT_EQ(vertex.nz, source.vertices[index].nz);
        EXPECT_EQ(vertex.r, source.vertices[index].r);
        EXPECT_EQ(vertex.g, source.vertices[index].g);
        EXPECT_EQ(vertex.b, source.vertices[index].b);
    }
}

TEST(VisibilityOccupancyCarrierFieldProjectorTest,
     BacktrackingAcceptsGuardedHalfStepWhenFullStepIsTooAggressive)
{
    const std::array<int, 3> dimensions{33, 33, 33};
    const std::array<float, 3> bounds_min{-1.0f, -1.0f, -1.0f};
    const std::array<float, 3> bounds_max{1.0f, 1.0f, 1.0f};
    const auto field = makeField(
        dimensions, bounds_min, bounds_max,
        [](double x, double y, double z)
        {
            return std::sqrt(x * x + y * y + z * z) - 0.20;
        });
    const xjw::mesh::TriMesh source = makeClosedCube();
    auto options = permissiveOptions();
    options.iterations = 1;
    options.maximumBacktrackingSteps = 8;
    options.relaxation = 1.0;
    options.maximumStepSpacingRatio = 20.0;
    options.maximumCumulativeDisplacementSpacingRatio = 20.0;
    options.minimumSurfaceAreaRatio = 0.90;
    options.maximumSurfaceAreaRatio = 1.05;
    options.minimumAbsoluteVolumeRatio = 0.92;
    options.maximumAbsoluteVolumeRatio = 1.05;

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierFieldProjector::project(
            source, dimensions, bounds_min, bounds_max, field, options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    EXPECT_EQ(result.statistics.completedIterationCount, 1);
    EXPECT_EQ(result.statistics.acceptedFullStepCount, 0);
    EXPECT_EQ(result.statistics.acceptedHalfStepCount, 1);
    EXPECT_GT(result.statistics.backtrackingAttemptCount, 0);
    EXPECT_GT(result.statistics.rejectedBlendCount, 0);
    EXPECT_LT(result.statistics.minimumAcceptedBlend, 1.0);
    EXPECT_GE(result.statistics.finalSurfaceAreaRatio,
              options.minimumSurfaceAreaRatio);
    EXPECT_GE(result.statistics.finalAbsoluteVolumeRatio,
              options.minimumAbsoluteVolumeRatio);
    EXPECT_LE(result.statistics.meanAbsoluteFieldResidualAfter,
              result.statistics.meanAbsoluteFieldResidualBefore);
    EXPECT_LE(result.statistics.p90AbsoluteFieldResidualAfter,
              result.statistics.p90AbsoluteFieldResidualBefore);
}

TEST(VisibilityOccupancyCarrierFieldProjectorTest,
     AggressiveProjectionRejectedByGlobalGuardRollsBackAtomically)
{
    const std::array<int, 3> dimensions{33, 33, 33};
    const std::array<float, 3> bounds_min{-1.0f, -1.0f, -1.0f};
    const std::array<float, 3> bounds_max{1.0f, 1.0f, 1.0f};
    const auto field = makeField(
        dimensions, bounds_min, bounds_max,
        [](double x, double y, double z)
        {
            return std::sqrt(x * x + y * y + z * z) - 0.20;
        });
    const xjw::mesh::TriMesh source = makeClosedCube();
    auto options = permissiveOptions();
    options.iterations = 1;
    options.relaxation = 1.0;
    options.maximumStepSpacingRatio = 20.0;
    options.maximumCumulativeDisplacementSpacingRatio = 20.0;
    options.minimumSurfaceAreaRatio = 0.99;
    options.maximumSurfaceAreaRatio = 1.01;
    options.minimumAbsoluteVolumeRatio = 0.99;
    options.maximumAbsoluteVolumeRatio = 1.01;

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierFieldProjector::project(
            source, dimensions, bounds_min, bounds_max, field, options);

    EXPECT_FALSE(result.ok);
    EXPECT_FALSE(result.cancelled);
    EXPECT_TRUE(result.rolledBack);
    EXPECT_EQ(result.statistics.rollbackCount, 1U);
    expectPositionsEqual(result.mesh, source);
}

TEST(VisibilityOccupancyCarrierFieldProjectorTest,
     CancellationDuringFieldSmoothingRollsBackAtomically)
{
    const std::array<int, 3> dimensions{20, 20, 20};
    const std::array<float, 3> bounds_min{-1.0f, -1.0f, -1.0f};
    const std::array<float, 3> bounds_max{1.0f, 1.0f, 1.0f};
    const auto field = makeField(
        dimensions, bounds_min, bounds_max,
        [](double, double, double z) { return z; });
    const xjw::mesh::TriMesh source = makeStairPatch();
    auto options = permissiveOptions();
    options.smoothNarrowBand = true;
    int cancellation_checks = 0;
    options.isCancelled = [&cancellation_checks]()
    {
        ++cancellation_checks;
        return cancellation_checks >= 8;
    };

    const auto result =
        xjw::mesh::VisibilityOccupancyCarrierFieldProjector::project(
            source, dimensions, bounds_min, bounds_max, field, options);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.cancelled);
    EXPECT_TRUE(result.rolledBack);
    EXPECT_EQ(result.statistics.rollbackCount, 1U);
    expectPositionsEqual(result.mesh, source);
}
