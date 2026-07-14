#include "reference/CoordinateReference.h"

#include <gtest/gtest.h>

#include <cmath>

namespace cp = xjw::control_points;

TEST(CoordinateReferenceTest, ConvertsTraditionalGisAxisOrderExplicitly)
{
    const cp::CoordinateReference source =
        cp::CoordinateReference::fromEpsg(4326, cp::AxisOrder::LongitudeLatitude);
    const cp::CoordinateReference target = cp::CoordinateReference::fromEpsg(3857);

    const cp::CoordinateTransformResult transformed =
        cp::transformCoordinate({116.391, 39.907, 50.0}, source, target);

    ASSERT_TRUE(transformed.ok) << qPrintable(transformed.error);
    EXPECT_NEAR(transformed.xyz[0], 12956586.0, 100.0);
    EXPECT_NEAR(transformed.xyz[1], 4852436.0, 100.0);
    EXPECT_DOUBLE_EQ(transformed.xyz[2], 50.0);
}

TEST(CoordinateReferenceTest, SupportsExplicitLatitudeLongitudeInput)
{
    const cp::CoordinateReference source =
        cp::CoordinateReference::fromEpsg(4326, cp::AxisOrder::LatitudeLongitude);
    const cp::CoordinateReference target = cp::CoordinateReference::fromEpsg(3857);

    const cp::CoordinateTransformResult transformed =
        cp::transformCoordinate({39.907, 116.391, 50.0}, source, target);

    ASSERT_TRUE(transformed.ok) << qPrintable(transformed.error);
    EXPECT_NEAR(transformed.xyz[0], 12956586.0, 100.0);
    EXPECT_NEAR(transformed.xyz[1], 4852436.0, 100.0);
}

TEST(CoordinateReferenceTest, RoundTripsWktAndRejectsInvalidDefinition)
{
    const cp::CoordinateReference original = cp::CoordinateReference::fromEpsg(4978);
    ASSERT_TRUE(original.isValid()) << qPrintable(original.error());
    ASSERT_FALSE(original.wkt().isEmpty());

    const cp::CoordinateReference restored =
        cp::CoordinateReference::fromWkt(original.wkt(), cp::AxisOrder::TraditionalGis);
    EXPECT_TRUE(restored.isValid()) << qPrintable(restored.error());
    EXPECT_TRUE(restored.isGeocentric());
    EXPECT_EQ(restored.horizontalUnit(), cp::CoordinateUnit::Metre);

    const cp::CoordinateReference invalid =
        cp::CoordinateReference::fromUserInput(QStringLiteral("EPSG:not-a-code"));
    EXPECT_FALSE(invalid.isValid());
    EXPECT_FALSE(invalid.error().isEmpty());
}

TEST(CoordinateReferenceTest, ReportsLinearFootUnitAndConversion)
{
    const cp::CoordinateReference reference = cp::CoordinateReference::fromEpsg(2263);

    ASSERT_TRUE(reference.isValid()) << qPrintable(reference.error());
    EXPECT_EQ(reference.horizontalUnit(), cp::CoordinateUnit::UsSurveyFoot);
    EXPECT_NEAR(reference.horizontalUnitToMetres(), 0.3048006096012192, 1e-12);
}

