#include <gtest/gtest.h>

#include "Mc33IsoSurfaceExtractor.h"
#include "MeshTopologyQuality.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

std::size_t sampleIndex(int x, int y, int z)
{
    return static_cast<std::size_t>((z * 4 + y) * 4 + x);
}

std::size_t sampleIndex(const std::array<int, 3> &sampleDimensions,
                        int x,
                        int y,
                        int z)
{
    return (static_cast<std::size_t>(z) *
                static_cast<std::size_t>(sampleDimensions[1]) +
            static_cast<std::size_t>(y)) *
            static_cast<std::size_t>(sampleDimensions[0]) +
        static_cast<std::size_t>(x);
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

} // namespace

TEST(Mc33IsoSurfaceExtractorTest, EnclosedBinaryConfigurationsAreWatertight)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }

    std::uint64_t total_triangles = 0;
    for (std::uint32_t configuration = 1; configuration < 256; ++configuration)
    {
        const auto result = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
            {0.0f, 0.0f, 0.0f},
            {3.0f, 3.0f, 3.0f},
            {3, 3, 3},
            makeEnclosedConfiguration(configuration));
        ASSERT_TRUE(result.ok)
            << "configuration=" << configuration
            << " error=" << result.errorMessage;
        ASSERT_FALSE(result.mesh.empty()) << "configuration=" << configuration;
        const auto quality = xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
        EXPECT_EQ(quality.boundaryEdgeCount, 0)
            << "configuration=" << configuration;
        EXPECT_EQ(quality.nonManifoldEdgeCount, 0)
            << "configuration=" << configuration;
        total_triangles += result.mesh.faces.size();
    }
    EXPECT_EQ(total_triangles, 8108u);
}

TEST(Mc33IsoSurfaceExtractorTest,
     FullySupportedBinaryConfigurationsMatchUnfilteredExtraction)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }

    const std::vector<std::uint8_t> support(64u, 1u);
    xjw::mesh::Mc33IsoSurfaceOptions options;
    options.requireSupportedSignChange = true;
    for (std::uint32_t configuration = 1; configuration < 256; ++configuration)
    {
        const std::vector<float> field = makeEnclosedConfiguration(configuration);
        const auto baseline = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
            {0.0f, 0.0f, 0.0f},
            {3.0f, 3.0f, 3.0f},
            {3, 3, 3},
            field);
        const auto filtered = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
            {0.0f, 0.0f, 0.0f},
            {3.0f, 3.0f, 3.0f},
            {3, 3, 3},
            field,
            support,
            options);

        ASSERT_TRUE(baseline.ok)
            << "configuration=" << configuration
            << " error=" << baseline.errorMessage;
        ASSERT_TRUE(filtered.ok)
            << "configuration=" << configuration
            << " error=" << filtered.errorMessage;
        EXPECT_EQ(filtered.statistics.rejectedUnsupportedCellFaceCount, 0u)
            << "configuration=" << configuration;
        EXPECT_EQ(filtered.mesh.faces.size(), baseline.mesh.faces.size())
            << "configuration=" << configuration;
        const auto quality = xjw::mesh::evaluateMeshTopologyQuality(filtered.mesh);
        EXPECT_EQ(quality.boundaryEdgeCount, 0u)
            << "configuration=" << configuration;
        EXPECT_EQ(quality.nonManifoldEdgeCount, 0u)
            << "configuration=" << configuration;
    }
}

TEST(Mc33IsoSurfaceExtractorTest, PreservesBoundsAndSupportMask)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }

    constexpr int cells = 16;
    constexpr int samples = cells + 1;
    std::vector<float> field(
        static_cast<std::size_t>(samples * samples * samples));
    std::vector<std::uint8_t> support(field.size(), 1u);
    for (int z = 0; z < samples; ++z)
    {
        for (int y = 0; y < samples; ++y)
        {
            for (int x = 0; x < samples; ++x)
            {
                const float px = static_cast<float>(x) / cells * 4.0f + 2.0f;
                const float py = static_cast<float>(y) / cells * 6.0f - 3.0f;
                const float pz = static_cast<float>(z) / cells * 2.0f + 5.0f;
                const std::size_t index =
                    static_cast<std::size_t>((z * samples + y) * samples + x);
                field[index] = std::sqrt(
                    (px - 4.0f) * (px - 4.0f) +
                    py * py +
                    (pz - 6.0f) * (pz - 6.0f)) - 0.7f;
            }
        }
    }
    support.front() = 0u;

    xjw::mesh::Mc33IsoSurfaceOptions options;
    options.requireSupportedSignChange = true;
    const auto result = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
        {2.0f, -3.0f, 5.0f},
        {6.0f, 3.0f, 7.0f},
        {cells, cells, cells},
        field,
        support,
        options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    ASSERT_FALSE(result.mesh.empty());
    EXPECT_EQ(result.statistics.supportMaskedSampleCount, 1u);
    EXPECT_EQ(result.statistics.rejectedUnsupportedCellFaceCount, 0u);
    const auto quality = xjw::mesh::evaluateMeshTopologyQuality(result.mesh);
    EXPECT_EQ(quality.boundaryEdgeCount, 0);
    EXPECT_EQ(quality.nonManifoldEdgeCount, 0);
    for (const xjw::mesh::MeshVertex &vertex : result.mesh.vertices)
    {
        EXPECT_GE(vertex.x, 2.0f);
        EXPECT_LE(vertex.x, 6.0f);
        EXPECT_GE(vertex.y, -3.0f);
        EXPECT_LE(vertex.y, 3.0f);
        EXPECT_GE(vertex.z, 5.0f);
        EXPECT_LE(vertex.z, 7.0f);
    }
}

TEST(Mc33IsoSurfaceExtractorTest,
     RejectsSurfaceCreatedOnlyByUnsupportedBandBoundary)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }

    std::vector<float> field(64u, 1.0f);
    std::vector<std::uint8_t> support(64u, 1u);
    for (int z = 0; z < 4; ++z)
    {
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                const std::size_t index = sampleIndex(x, y, z);
                field[index] = x <= 1 ? -1.0f : 1.0f;
                if (x == 0)
                {
                    support[index] = 0u;
                }
            }
        }
    }

    xjw::mesh::Mc33IsoSurfaceOptions options;
    options.requireSupportedSignChange = true;
    const auto result = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
        {0.0f, 0.0f, 0.0f},
        {3.0f, 3.0f, 3.0f},
        {3, 3, 3},
        field,
        support,
        options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    ASSERT_FALSE(result.mesh.empty());
    EXPECT_GT(result.statistics.rejectedUnsupportedCellFaceCount, 0u);
    for (const xjw::mesh::MeshVertex &vertex : result.mesh.vertices)
    {
        EXPECT_GT(vertex.x, 1.0f);
    }
}

TEST(Mc33IsoSurfaceExtractorTest,
     MixedSupportCellRetainsOnlySupportedCrossingEdges)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }

    const std::array<int, 3> sample_dimensions{2, 2, 2};
    std::vector<float> field(8u, -1.0f);
    std::vector<std::uint8_t> support(8u, 1u);
    field[sampleIndex(sample_dimensions, 1, 0, 0)] = 1.0f;
    support[sampleIndex(sample_dimensions, 0, 1, 1)] = 0u;

    xjw::mesh::Mc33IsoSurfaceOptions options;
    options.requireSupportedSignChange = true;
    const auto result = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
        {1, 1, 1},
        field,
        support,
        options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    ASSERT_EQ(result.mesh.faces.size(), 1u);
    EXPECT_EQ(result.mesh.vertices.size(), 3u);
    EXPECT_GT(result.statistics.rejectedUnsupportedCellFaceCount, 0u);
    for (const xjw::mesh::MeshVertex &vertex : result.mesh.vertices)
    {
        EXPECT_LT(vertex.y + vertex.z, 0.75f);
    }
}

TEST(Mc33IsoSurfaceExtractorTest,
     TwoCellSeamRejectsUnsupportedForcedOutsideSurface)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }

    const std::array<int, 3> sample_dimensions{3, 2, 2};
    std::vector<float> field(12u, -1.0f);
    std::vector<std::uint8_t> support(12u, 1u);
    for (int z = 0; z < 2; ++z)
    {
        for (int y = 0; y < 2; ++y)
        {
            field[sampleIndex(sample_dimensions, 2, y, z)] = 1.0f;
            support[sampleIndex(sample_dimensions, 0, y, z)] = 0u;
        }
    }

    xjw::mesh::Mc33IsoSurfaceOptions options;
    options.requireSupportedSignChange = true;
    const auto result = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
        {0.0f, 0.0f, 0.0f},
        {2.0f, 1.0f, 1.0f},
        {2, 1, 1},
        field,
        support,
        options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    ASSERT_FALSE(result.mesh.empty());
    EXPECT_GT(result.statistics.rejectedUnsupportedCellFaceCount, 0u);
    for (const xjw::mesh::MeshVertex &vertex : result.mesh.vertices)
    {
        EXPECT_NEAR(vertex.x, 1.5f, 1.0e-5f);
    }
}

TEST(Mc33IsoSurfaceExtractorTest,
     ExactIsoSampleUsesConnectivityPositiveSideConvention)
{
    if (!xjw::mesh::Mc33IsoSurfaceExtractor::isAvailable())
    {
        GTEST_SKIP() << "MC33 dependency is not configured";
    }

    const std::array<int, 3> sample_dimensions{3, 2, 2};
    std::vector<float> field(12u, -1.0f);
    std::vector<std::uint8_t> support(12u, 1u);
    for (int z = 0; z < 2; ++z)
    {
        for (int y = 0; y < 2; ++y)
        {
            field[sampleIndex(sample_dimensions, 1, y, z)] = 0.0f;
            field[sampleIndex(sample_dimensions, 2, y, z)] = 1.0f;
            support[sampleIndex(sample_dimensions, 2, y, z)] = 0u;
        }
    }

    xjw::mesh::Mc33IsoSurfaceOptions options;
    options.requireSupportedSignChange = true;
    const auto result = xjw::mesh::Mc33IsoSurfaceExtractor::extract(
        {0.0f, 0.0f, 0.0f},
        {2.0f, 1.0f, 1.0f},
        {2, 1, 1},
        field,
        support,
        options);

    ASSERT_TRUE(result.ok) << result.errorMessage;
    ASSERT_FALSE(result.mesh.empty());
    for (const xjw::mesh::MeshVertex &vertex : result.mesh.vertices)
    {
        EXPECT_NEAR(vertex.x, 1.0f, 1.0e-5f);
    }
}
