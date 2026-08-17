#include "DepthMeasuredSupportConnectivity.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{

struct Fixture
{
    std::array<int, 3> dimensions{6, 6, 2};
    std::vector<float> tsdf;
    std::vector<float> weight;
    std::vector<float> surfaceWeight;
    std::vector<float> observationWeight;
    std::vector<xjw::mesh::DepthGeometrySourceMask> sourceMask;
    std::vector<std::uint16_t> spread;
    std::vector<std::uint16_t> geometrySupport;
    std::vector<std::uint8_t> supported;

    explicit Fixture(int size_z = 2)
        : dimensions{6, 6, size_z}
    {
        const std::size_t count = static_cast<std::size_t>(6 * 6 * size_z);
        tsdf.resize(count, 0.2f);
        weight.resize(count, 1.0f);
        surfaceWeight.resize(count, 0.8f);
        observationWeight.resize(count, 0.8f);
        sourceMask.resize(count, 0x3);
        spread.resize(count, 500);
        geometrySupport.resize(count, 3);
        supported.resize(count, 1);
        for (int z = 0; z < size_z - 1; ++z)
        {
            for (int y = 0; y < 6; ++y)
            {
                for (int x = 0; x < 6; ++x)
                {
                    tsdf[index(x, y, z)] = -0.2f;
                }
            }
        }
    }

    std::size_t index(int x, int y, int z) const
    {
        return (static_cast<std::size_t>(z) * 6U +
                static_cast<std::size_t>(y)) * 6U +
            static_cast<std::size_t>(x);
    }

    void removeNegativePatch(int z)
    {
        for (int y = 2; y <= 3; ++y)
        {
            for (int x = 2; x <= 3; ++x)
            {
                supported[index(x, y, z)] = 0;
            }
        }
    }

    xjw::mesh::DepthMeasuredSupportConnectivityInput input() const
    {
        return {dimensions,
                &tsdf,
                &weight,
                &surfaceWeight,
                &observationWeight,
                &sourceMask,
                &spread,
                &geometrySupport};
    }
};

xjw::mesh::DepthMeasuredSupportConnectivityOptions testOptions()
{
    xjw::mesh::DepthMeasuredSupportConnectivityOptions options;
    options.minimumComponentCells = 1;
    options.minimumAnchorCells = 1;
    options.maximumSingleVoteAbsoluteTsdf = 1.0f;
    return options;
}

} // namespace

TEST(DepthMeasuredSupportConnectivityTest, PromotesOnlyBoundaryReducingSupport)
{
    Fixture fixture;
    fixture.removeNegativePatch(0);
    const std::vector<float> frozen_tsdf = fixture.tsdf;

    const auto statistics = xjw::mesh::DepthMeasuredSupportConnectivity::recover(
        fixture.input(), testOptions(), &fixture.supported);

    EXPECT_EQ(statistics.recoveredSampleCount, 4U);
    EXPECT_EQ(statistics.unlockedCellCount, 1U);
    EXPECT_EQ(statistics.acceptedComponentCount, 1);
    EXPECT_EQ(fixture.tsdf, frozen_tsdf);
}

TEST(DepthMeasuredSupportConnectivityTest, RejectsSingleSourceAndLargeSpread)
{
    Fixture fixture;
    fixture.removeNegativePatch(0);
    for (int y = 2; y <= 3; ++y)
    {
        for (int x = 2; x <= 3; ++x)
        {
            fixture.sourceMask[fixture.index(x, y, 0)] = 0x1;
        }
    }
    auto statistics = xjw::mesh::DepthMeasuredSupportConnectivity::recover(
        fixture.input(), testOptions(), &fixture.supported);
    EXPECT_EQ(statistics.recoveredSampleCount, 0U);
    EXPECT_EQ(statistics.rejectedSourceCount, 4U);

    fixture = Fixture{};
    fixture.removeNegativePatch(0);
    for (int y = 2; y <= 3; ++y)
    {
        for (int x = 2; x <= 3; ++x)
        {
            fixture.spread[fixture.index(x, y, 0)] = 2500;
        }
    }
    statistics = xjw::mesh::DepthMeasuredSupportConnectivity::recover(
        fixture.input(), testOptions(), &fixture.supported);
    EXPECT_EQ(statistics.recoveredSampleCount, 0U);
    EXPECT_EQ(statistics.rejectedDepthSpreadCount, 4U);
}

TEST(DepthMeasuredSupportConnectivityTest, DoesNotCascadeAcrossSecondLayer)
{
    Fixture fixture(3);
    fixture.removeNegativePatch(1);
    fixture.removeNegativePatch(0);

    const auto statistics = xjw::mesh::DepthMeasuredSupportConnectivity::recover(
        fixture.input(), testOptions(), &fixture.supported);

    EXPECT_EQ(statistics.recoveredSampleCount, 4U);
    for (int y = 2; y <= 3; ++y)
    {
        for (int x = 2; x <= 3; ++x)
        {
            EXPECT_EQ(fixture.supported[fixture.index(x, y, 1)], 1);
            EXPECT_EQ(fixture.supported[fixture.index(x, y, 0)], 0);
        }
    }
}

TEST(DepthMeasuredSupportConnectivityTest, RejectsIsolatedCandidate)
{
    Fixture fixture;
    for (int y = 0; y <= 1; ++y)
    {
        for (int x = 0; x <= 1; ++x)
        {
            fixture.supported[fixture.index(x, y, 0)] = 0;
        }
    }

    const auto statistics = xjw::mesh::DepthMeasuredSupportConnectivity::recover(
        fixture.input(), testOptions(), &fixture.supported);

    EXPECT_EQ(statistics.recoveredSampleCount, 0U);
    EXPECT_GT(statistics.rejectedBoundaryComponentCount, 0);
}
