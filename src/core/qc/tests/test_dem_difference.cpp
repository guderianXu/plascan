#include "DemDifference.h"

#include <gtest/gtest.h>

using xjw::qc::DemDifference;
using xjw::qc::DemGrid;

TEST(DemDifference, ComputesSameGridDifferenceStats)
{
    DemGrid reference;
    reference.width = 3;
    reference.height = 2;
    reference.nodata = -9999.0;
    reference.values = {
        10.0, 11.0, 12.0,
        13.0, -9999.0, 15.0
    };
    reference.projection = QStringLiteral("EPSG:4326");

    DemGrid candidate = reference;
    candidate.values = {
        11.0, 10.0, 14.0,
        12.0, -9999.0, 18.0
    };

    const auto result = DemDifference::compareSameGrid(candidate, reference);

    ASSERT_TRUE(result.success) << result.error.toStdString();
    ASSERT_EQ(result.differences.size(), 6);
    EXPECT_EQ(result.validCount, 5);
    EXPECT_NEAR(result.mean, 0.8, 1e-9);
    EXPECT_NEAR(result.rmse, 1.7888543819998317, 1e-9);
    EXPECT_NEAR(result.median, 1.0, 1e-9);
    EXPECT_NEAR(result.p95, 3.0, 1e-9);
    EXPECT_EQ(result.differences[4], reference.nodata);
}

TEST(DemDifference, RejectsProjectionMismatch)
{
    DemGrid a;
    a.width = 1;
    a.height = 1;
    a.values = {1.0};
    a.projection = QStringLiteral("EPSG:4326");

    DemGrid b = a;
    b.projection = QStringLiteral("EPSG:3857");

    const auto result = DemDifference::compareSameGrid(a, b);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.error.contains(QStringLiteral("投影")));
}
