#include "reference/CoordinateReference.h"

#include "coordinate_system/context/CoordinateContext.h"
#include "coordinate_system/gdal/GdalCoordinateTransform.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cp = xjw::control_points;
namespace cs = xjw::coordinate_system;

namespace
{
    using namespace xjw::coordinate_system;

    CoordinateFrame referenceFrame(const char* id, CoordinateFrameKind kind, AngleUnit angleUnit)
    {
        return CoordinateFrame::create(
            CoordinateFrameId(id), kind, LinearUnit::Metre, angleUnit, std::nullopt, RigidTransform::identity());
    }

    SpatialReferenceDefinition
    reference(const char* id, const char* frameId, const char* definition, VerticalReference verticalReference)
    {
        GdalSpatialReferenceResult normalized = normalizeGdalSpatialReference(
            SpatialReferenceId(id), CoordinateFrameId(frameId), definition, verticalReference);
        if (!normalized.ok())
        {
            throw std::runtime_error(normalized.error);
        }
        return std::move(*normalized.reference);
    }

    CoordinateContext earthContext(bool includeGeographic = true)
    {
        std::vector<SpatialReferenceDefinition> references;
        std::vector<CoordinateFrame> frames;
        references.push_back(
            reference("crs-epsg-4978", "frame-wgs84-ecef", "EPSG:4978", VerticalReference::NotApplicable));
        frames.push_back(referenceFrame("frame-wgs84-ecef", CoordinateFrameKind::Ecef, AngleUnit::Radian));
        if (includeGeographic)
        {
            references.push_back(
                reference("crs-epsg-4979", "frame-wgs84-geodetic", "EPSG:4979", VerticalReference::Ellipsoidal));
            frames.push_back(referenceFrame("frame-wgs84-geodetic", CoordinateFrameKind::Geodetic, AngleUnit::Degree));
        }
        return CoordinateContext::create(CoordinateContextId("coordctx-earth"),
                                         1,
                                         std::move(references),
                                         std::move(frames),
                                         SpatialReferenceId("crs-epsg-4978"),
                                         SolverFrameDefinition::create(CoordinateFrameId("frame-wgs84-ecef"),
                                                                       SolverScaleStatus::Metric,
                                                                       "ecef-normalization-v1"));
    }

} // namespace

TEST(CoordinateReferenceTest, ConvertsTraditionalGisAxisOrderExplicitly)
{
    const cp::CoordinateReference source = cp::CoordinateReference::fromEpsg(4326, cp::AxisOrder::LongitudeLatitude);
    const cp::CoordinateReference target = cp::CoordinateReference::fromEpsg(3857);

    const cp::CoordinateTransformResult transformed = cp::transformCoordinate({116.391, 39.907, 50.0}, source, target);

    ASSERT_TRUE(transformed.ok) << qPrintable(transformed.error);
    EXPECT_NEAR(transformed.xyz[0], 12956586.0, 100.0);
    EXPECT_NEAR(transformed.xyz[1], 4852436.0, 100.0);
    EXPECT_DOUBLE_EQ(transformed.xyz[2], 50.0);
}

TEST(CoordinateReferenceTest, SupportsExplicitLatitudeLongitudeInput)
{
    const cp::CoordinateReference source = cp::CoordinateReference::fromEpsg(4326, cp::AxisOrder::LatitudeLongitude);
    const cp::CoordinateReference target = cp::CoordinateReference::fromEpsg(3857);

    const cp::CoordinateTransformResult transformed = cp::transformCoordinate({39.907, 116.391, 50.0}, source, target);

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

    const cp::CoordinateReference invalid = cp::CoordinateReference::fromUserInput(QStringLiteral("EPSG:not-a-code"));
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

TEST(CoordinateReferenceTest, RejectsGeographicCoordinatesWithoutSolverContext)
{
    cp::ReferenceCoordinate coordinate;
    coordinate.x = 116.391;
    coordinate.y = 39.907;
    coordinate.z = 50.0;
    coordinate.sigmaX = 0.01;
    coordinate.sigmaY = 0.01;
    coordinate.sigmaZ = 0.02;
    coordinate.sourceCrs = QStringLiteral("EPSG:4979");
    coordinate.axisOrder = QStringLiteral("longitude_latitude");
    coordinate.verticalDatum = QStringLiteral("ellipsoidal");
    coordinate.verticalUnit = QStringLiteral("m");

    const cp::MetricReferenceCoordinateResult resolved = cp::resolveMetricReferenceCoordinate(coordinate);

    EXPECT_FALSE(resolved.ok);
    EXPECT_NE(resolved.error.find("地理角坐标"), std::string::npos);
}

TEST(CoordinateReferenceTest, ConvertsProjectedAndVerticalFeetToMetres)
{
    cp::ReferenceCoordinate coordinate;
    coordinate.x = 1000.0;
    coordinate.y = 2000.0;
    coordinate.z = 10.0;
    coordinate.sigmaX = 0.01;
    coordinate.sigmaY = 0.02;
    coordinate.sigmaZ = 0.03;
    coordinate.sourceCrs = QStringLiteral("EPSG:2263");
    coordinate.verticalDatum = QStringLiteral("NAVD88");
    coordinate.verticalUnit = QStringLiteral("ft");

    const cp::MetricReferenceCoordinateResult resolved = cp::resolveMetricReferenceCoordinate(coordinate);

    ASSERT_TRUE(resolved.ok) << resolved.error;
    EXPECT_NEAR(resolved.pointMetres[0], 304.8006096012192, 1.0e-9);
    EXPECT_NEAR(resolved.pointMetres[1], 609.6012192024384, 1.0e-9);
    EXPECT_NEAR(resolved.pointMetres[2], 3.048, 1.0e-12);
    EXPECT_NEAR(resolved.sigmaMetres[0], 0.003048006096012192, 1.0e-15);
    EXPECT_NEAR(resolved.sigmaMetres[1], 0.006096012192024384, 1.0e-15);
    EXPECT_NEAR(resolved.sigmaMetres[2], 0.009144, 1.0e-15);
    EXPECT_FALSE(resolved.referenceKey.empty());
}

TEST(CoordinateReferenceTest, ConvertsGeographicReferenceThroughExplicitMetricContext)
{
    cp::ReferenceCoordinate coordinate;
    coordinate.x = 116.391;
    coordinate.y = 39.907;
    coordinate.z = 50.0;
    coordinate.sigmaX = 1.0e-5;
    coordinate.sigmaY = 1.0e-5;
    coordinate.sigmaZ = 0.25;
    coordinate.sourceCrs = QStringLiteral("EPSG:4979");
    coordinate.axisOrder = QStringLiteral("longitude_latitude");
    coordinate.verticalDatum = QStringLiteral("WGS84_ellipsoidal");
    coordinate.verticalUnit = QStringLiteral("m");
    const cs::CoordinateContext context = earthContext();

    const cp::MetricReferenceCoordinateResult resolved = cp::resolveMetricReferenceCoordinate(coordinate, &context);

    ASSERT_TRUE(resolved.ok) << resolved.error;
    const double radius = std::sqrt(resolved.pointMetres[0] * resolved.pointMetres[0] +
                                    resolved.pointMetres[1] * resolved.pointMetres[1] +
                                    resolved.pointMetres[2] * resolved.pointMetres[2]);
    EXPECT_GT(radius, 6.3e6);
    EXPECT_LT(radius, 6.4e6);
    for (double sigma : resolved.sigmaMetres)
    {
        EXPECT_GT(sigma, 0.1);
        EXPECT_LT(sigma, 2.0);
    }
    EXPECT_NE(resolved.referenceKey.find(context.contextHash()), std::string::npos);
    EXPECT_EQ(resolved.contextId, context.id().value());
    EXPECT_EQ(resolved.contextHash, context.contextHash());
    EXPECT_EQ(resolved.solverFrameId, "frame-wgs84-ecef");
    EXPECT_EQ(resolved.solverSpatialReferenceId, "crs-epsg-4978");
    EXPECT_EQ(resolved.normalizationHash, "ecef-normalization-v1");
}

TEST(CoordinateReferenceTest, RejectsGeographicReferenceMissingFromContext)
{
    cp::ReferenceCoordinate coordinate;
    coordinate.x = 116.391;
    coordinate.y = 39.907;
    coordinate.z = 50.0;
    coordinate.sigmaX = 1.0e-5;
    coordinate.sigmaY = 1.0e-5;
    coordinate.sigmaZ = 0.25;
    coordinate.sourceCrs = QStringLiteral("EPSG:4979");
    coordinate.axisOrder = QStringLiteral("longitude_latitude");
    coordinate.verticalDatum = QStringLiteral("ellipsoidal");
    coordinate.verticalUnit = QStringLiteral("m");
    const cs::CoordinateContext context = earthContext(false);

    const cp::MetricReferenceCoordinateResult resolved = cp::resolveMetricReferenceCoordinate(coordinate, &context);

    EXPECT_FALSE(resolved.ok);
    EXPECT_NE(resolved.error.find("未注册"), std::string::npos);
}

TEST(CoordinateReferenceTest, RejectsVerticalReferenceMismatchWithContext)
{
    cp::ReferenceCoordinate coordinate;
    coordinate.x = 116.391;
    coordinate.y = 39.907;
    coordinate.z = 50.0;
    coordinate.sigmaX = 1.0e-5;
    coordinate.sigmaY = 1.0e-5;
    coordinate.sigmaZ = 0.25;
    coordinate.sourceCrs = QStringLiteral("EPSG:4979");
    coordinate.axisOrder = QStringLiteral("longitude_latitude");
    coordinate.verticalDatum = QStringLiteral("orthometric");
    coordinate.verticalUnit = QStringLiteral("m");
    const cs::CoordinateContext context = earthContext();

    const cp::MetricReferenceCoordinateResult resolved = cp::resolveMetricReferenceCoordinate(coordinate, &context);

    EXPECT_FALSE(resolved.ok);
    EXPECT_NE(resolved.error.find("垂直基准"), std::string::npos);
}
