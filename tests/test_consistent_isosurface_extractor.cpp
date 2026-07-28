#include <gtest/gtest.h>

#include "ConsistentIsoSurfaceExtractor.h"
#include "IsoSurfaceTopology.h"
#include "MeshTopologyQuality.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace
{

std::size_t sampleIndex(int x, int y, int z)
{
    return static_cast<std::size_t>((z * 4 + y) * 4 + x);
}

std::vector<float> makeEnclosedConfiguration(std::uint32_t configuration)
{
    std::vector<float> field(64u, 1.0f);
    int bit = 0;
    for (int z = 1; z <= 2; ++z)
    {
        for (int y = 1; y <= 2; ++y)
        {
            for (int x = 1; x <= 2; ++x)
            {
                field[sampleIndex(x, y, z)] =
                    (configuration & (1u << bit)) != 0u ? -1.0f : 1.0f;
                ++bit;
            }
        }
    }
    return field;
}

xjw::mesh::ConsistentIsoSurfaceResult extractConfiguration(
    std::uint32_t configuration)
{
    return xjw::mesh::ConsistentIsoSurfaceExtractor::extract(
        {0.0f, 0.0f, 0.0f},
        {3.0f, 3.0f, 3.0f},
        {3, 3, 3},
        makeEnclosedConfiguration(configuration),
        {});
}

std::vector<float> sampleImplicitField(
    int cells,
    const std::function<float(float, float, float)> &function)
{
    const int samples = cells + 1;
    std::vector<float> field(
        static_cast<std::size_t>(samples * samples * samples));
    for (int z = 0; z < samples; ++z)
    {
        for (int y = 0; y < samples; ++y)
        {
            for (int x = 0; x < samples; ++x)
            {
                const float px = static_cast<float>(x) / cells * 2.0f - 1.0f;
                const float py = static_cast<float>(y) / cells * 2.0f - 1.0f;
                const float pz = static_cast<float>(z) / cells * 2.0f - 1.0f;
                field[static_cast<std::size_t>((z * samples + y) * samples + x)] =
                    function(px, py, pz);
            }
        }
    }
    return field;
}

float boxField(float x,
               float y,
               float z,
               float halfX,
               float halfY,
               float halfZ)
{
    return std::max(
        {std::abs(x) - halfX,
         std::abs(y) - halfY,
         std::abs(z) - halfZ});
}

} // namespace

TEST(ConsistentIsoSurfaceTopologyTest, ComplementKeepsAmbiguousFaceConnection)
{
    const xjw::mesh::GridFaceKey key{xjw::mesh::GridAxis::Z, 2, 3, 4};
    const auto decision = xjw::mesh::decideIsoSurfaceFace(
        {-2.0f, 1.0f, -2.0f, 1.0f}, 0.0f, key);
    const auto complement = xjw::mesh::decideIsoSurfaceFace(
        {2.0f, -1.0f, 2.0f, -1.0f}, 0.0f, key);

    ASSERT_TRUE(decision.ambiguous);
    ASSERT_TRUE(complement.ambiguous);
    EXPECT_EQ(decision.connectEdge01And23, complement.connectEdge01And23);
    EXPECT_FALSE(decision.usedTieBreak);
    EXPECT_FALSE(complement.usedTieBreak);
}

TEST(ConsistentIsoSurfaceTopologyTest, DegenerateFaceTieBreakUsesGlobalFaceKey)
{
    const std::array<float, 4> values{-1.0f, 1.0f, -1.0f, 1.0f};
    const xjw::mesh::GridFaceKey first{xjw::mesh::GridAxis::X, 2, 3, 4};
    const xjw::mesh::GridFaceKey same{xjw::mesh::GridAxis::X, 2, 3, 4};
    const xjw::mesh::GridFaceKey adjacent{xjw::mesh::GridAxis::X, 3, 3, 4};

    const auto first_decision =
        xjw::mesh::decideIsoSurfaceFace(values, 0.0f, first);
    const auto same_decision =
        xjw::mesh::decideIsoSurfaceFace(values, 0.0f, same);
    const auto adjacent_decision =
        xjw::mesh::decideIsoSurfaceFace(values, 0.0f, adjacent);

    ASSERT_TRUE(first_decision.usedTieBreak);
    EXPECT_EQ(first_decision.connectEdge01And23,
              same_decision.connectEdge01And23);
    EXPECT_NE(first_decision.connectEdge01And23,
              adjacent_decision.connectEdge01And23);
}

TEST(ConsistentIsoSurfaceExtractorTest, EnclosedBinaryConfigurationsAreWatertight)
{
    for (std::uint32_t configuration = 1; configuration < 256; ++configuration)
    {
        const auto result = extractConfiguration(configuration);
        ASSERT_TRUE(result.ok)
            << "configuration=" << configuration
            << " error=" << result.errorMessage;
        ASSERT_FALSE(result.mesh.empty()) << "configuration=" << configuration;
        const auto quality = xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
        EXPECT_EQ(quality.boundaryEdgeCount, 0)
            << "configuration=" << configuration;
        EXPECT_EQ(quality.nonManifoldEdgeCount, 0)
            << "configuration=" << configuration;
        EXPECT_EQ(result.statistics.unresolvedCellCount, 0u)
            << "configuration=" << configuration;
    }
}

TEST(ConsistentIsoSurfaceExtractorTest, SharesVerticesWithoutCoordinateWelding)
{
    const auto result = extractConfiguration(255u);

    ASSERT_TRUE(result.ok);
    ASSERT_FALSE(result.mesh.empty());
    EXPECT_GT(result.statistics.edgeVertexCacheHitCount, 0u);
    EXPECT_EQ(result.mesh.vertices.size(),
              result.statistics.edgeVertexCacheMissCount +
                  result.statistics.interiorLoopVertexCount);
    EXPECT_LT(result.mesh.vertices.size(), result.mesh.faces.size() * 3u);
    const auto quality = xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
    EXPECT_EQ(quality.boundaryEdgeCount, 0);
    EXPECT_EQ(quality.nonManifoldEdgeCount, 0);
}

TEST(ConsistentIsoSurfaceExtractorTest, ExtractsClosedSphereDeterministically)
{
    constexpr int cells = 16;
    const auto field = sampleImplicitField(
        cells,
        [](float x, float y, float z)
        {
            return std::sqrt(x * x + y * y + z * z) - 0.7f;
        });

    const auto first = xjw::mesh::ConsistentIsoSurfaceExtractor::extract(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        {cells, cells, cells},
        field,
        {});
    const auto second = xjw::mesh::ConsistentIsoSurfaceExtractor::extract(
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        {cells, cells, cells},
        field,
        {});

    ASSERT_TRUE(first.ok);
    ASSERT_TRUE(second.ok);
    EXPECT_EQ(first.mesh.vertices.size(), second.mesh.vertices.size());
    EXPECT_EQ(first.mesh.faces.size(), second.mesh.faces.size());
    EXPECT_EQ(first.statistics.uniqueAmbiguousFaceCount,
              second.statistics.uniqueAmbiguousFaceCount);
    const auto quality = xjw::mesh::evaluateMeshTopologyQuality(first.mesh);
    EXPECT_EQ(quality.boundaryEdgeCount, 0);
    EXPECT_EQ(quality.nonManifoldEdgeCount, 0);
}

TEST(ConsistentIsoSurfaceExtractorTest, PreservesClosedSyntheticTopologies)
{
    constexpr int cells = 24;
    struct Shape
    {
        std::string name;
        std::function<float(float, float, float)> field;
        int expectedComponents = 1;
    };
    const std::vector<Shape> shapes{
        {"cylinder",
         [](float x, float y, float z)
         {
             return std::max(
                 std::sqrt(x * x + y * y) - 0.52f,
                 std::abs(z) - 0.72f);
         }},
        {"thin_plate",
         [](float x, float y, float z)
         {
             return boxField(x, y, z, 0.72f, 0.58f, 0.11f);
         }},
        {"parallel_plates",
         [](float x, float y, float z)
         {
             const float lower = boxField(x, y, z + 0.35f, 0.65f, 0.55f, 0.09f);
             const float upper = boxField(x, y, z - 0.35f, 0.65f, 0.55f, 0.09f);
             return std::min(lower, upper);
         },
         2},
        {"cross",
         [](float x, float y, float z)
         {
             const float horizontal = boxField(x, y, z, 0.70f, 0.18f, 0.18f);
             const float vertical = boxField(x, y, z, 0.18f, 0.18f, 0.70f);
             return std::min(horizontal, vertical);
         }},
        {"arch",
         [](float x, float y, float z)
         {
             const float left =
                 boxField(x + 0.45f, y, z + 0.05f, 0.14f, 0.22f, 0.62f);
             const float right =
                 boxField(x - 0.45f, y, z + 0.05f, 0.14f, 0.22f, 0.62f);
             const float top =
                 boxField(x, y, z - 0.52f, 0.59f, 0.22f, 0.15f);
             return std::min({left, right, top});
         }}};

    for (const Shape &shape : shapes)
    {
        const auto result = xjw::mesh::ConsistentIsoSurfaceExtractor::extract(
            {-1.0f, -1.0f, -1.0f},
            {1.0f, 1.0f, 1.0f},
            {cells, cells, cells},
            sampleImplicitField(cells, shape.field),
            {});
        ASSERT_TRUE(result.ok)
            << "shape=" << shape.name << " error=" << result.errorMessage;
        ASSERT_FALSE(result.mesh.empty()) << "shape=" << shape.name;
        const auto quality = xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
        EXPECT_EQ(quality.boundaryEdgeCount, 0) << "shape=" << shape.name;
        EXPECT_EQ(quality.nonManifoldEdgeCount, 0) << "shape=" << shape.name;
        EXPECT_EQ(quality.componentCount, shape.expectedComponents)
            << "shape=" << shape.name;
    }
}
