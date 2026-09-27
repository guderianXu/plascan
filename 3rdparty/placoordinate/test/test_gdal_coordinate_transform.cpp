#include "placoordinate/gdal/GdalCoordinateTransform.h"

#include <gtest/gtest.h>

#include <array>

namespace
{
    using namespace placoordinate;

    SpatialReferenceDefinition
    requireReference(const char* id, const char* definition, VerticalReference verticalReference)
    {
        GdalSpatialReferenceResult result =
            normalizeGdalSpatialReference(SpatialReferenceId(id), CoordinateFrameId(id), definition, verticalReference);
        if (!result.ok())
        {
            throw std::runtime_error(result.error);
        }
        return std::move(*result.reference);
    }

    TEST(GdalCoordinateTransformTest, NormalizesAuthoritativeThreeDimensionalReferences)
    {
        const SpatialReferenceDefinition geographic =
            requireReference("crs-geographic", "EPSG:4979", VerticalReference::Ellipsoidal);
        const SpatialReferenceDefinition geocentric =
            requireReference("crs-ecef", "EPSG:4978", VerticalReference::NotApplicable);

        EXPECT_EQ(geographic.kind(), SpatialReferenceKind::Geographic3d);
        EXPECT_EQ(geographic.axisMapping(), CoordinateAxisOrder::LongitudeLatitude);
        EXPECT_EQ(geographic.authority(), "EPSG");
        EXPECT_EQ(geographic.authorityCode(), "4979");
        EXPECT_EQ(geocentric.kind(), SpatialReferenceKind::Geocentric3d);
        EXPECT_TRUE(geocentric.isMetricCartesian());
        EXPECT_FALSE(geographic.canonicalDefinition().empty());
    }

    TEST(GdalCoordinateTransformTest, RoundTripsGeographicAndGeocentricCoordinates)
    {
        const SpatialReferenceDefinition geographic =
            requireReference("crs-geographic", "EPSG:4979", VerticalReference::Ellipsoidal);
        const SpatialReferenceDefinition geocentric =
            requireReference("crs-ecef", "EPSG:4978", VerticalReference::NotApplicable);

        const std::array<double, 3> source{116.391, 39.907, 50.0};
        const GdalCoordinateTransformResult ecef =
            transformGdalCoordinate(source, geographic, CoordinateAxisOrder::LongitudeLatitude, geocentric);
        ASSERT_TRUE(ecef.ok) << ecef.error;
        const GdalCoordinateTransformResult restored =
            transformGdalCoordinate(ecef.coordinate, geocentric, CoordinateAxisOrder::CanonicalXyz, geographic);
        ASSERT_TRUE(restored.ok) << restored.error;
        EXPECT_NEAR(restored.coordinate[0], source[0], 1.0e-9);
        EXPECT_NEAR(restored.coordinate[1], source[1], 1.0e-9);
        EXPECT_NEAR(restored.coordinate[2], source[2], 1.0e-5);
    }

    TEST(GdalCoordinateTransformTest, HonorsExplicitLatitudeLongitudeInput)
    {
        const SpatialReferenceDefinition geographic =
            requireReference("crs-geographic", "EPSG:4979", VerticalReference::Ellipsoidal);
        const SpatialReferenceDefinition geocentric =
            requireReference("crs-ecef", "EPSG:4978", VerticalReference::NotApplicable);

        const GdalCoordinateTransformResult longitude_latitude = transformGdalCoordinate(
            {116.391, 39.907, 50.0}, geographic, CoordinateAxisOrder::LongitudeLatitude, geocentric);
        const GdalCoordinateTransformResult latitude_longitude = transformGdalCoordinate(
            {39.907, 116.391, 50.0}, geographic, CoordinateAxisOrder::LatitudeLongitude, geocentric);

        ASSERT_TRUE(longitude_latitude.ok) << longitude_latitude.error;
        ASSERT_TRUE(latitude_longitude.ok) << latitude_longitude.error;
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            EXPECT_NEAR(longitude_latitude.coordinate[axis], latitude_longitude.coordinate[axis], 1.0e-6);
        }
    }

    TEST(GdalCoordinateTransformTest, PropagatesGeographicDiagonalUncertaintyIntoMetres)
    {
        const SpatialReferenceDefinition geographic =
            requireReference("crs-geographic", "EPSG:4979", VerticalReference::Ellipsoidal);
        const SpatialReferenceDefinition geocentric =
            requireReference("crs-ecef", "EPSG:4978", VerticalReference::NotApplicable);

        const GdalCoordinateUncertaintyResult transformed =
            transformGdalCoordinateWithDiagonalUncertainty({116.391, 39.907, 50.0},
                                                           {1.0e-5, 1.0e-5, 0.25},
                                                           geographic,
                                                           CoordinateAxisOrder::LongitudeLatitude,
                                                           geocentric);

        ASSERT_TRUE(transformed.ok) << transformed.error;
        for (double sigma : transformed.standardDeviation)
        {
            EXPECT_GT(sigma, 0.1);
            EXPECT_LT(sigma, 2.0);
        }
    }

    TEST(GdalCoordinateTransformTest, RejectsInvalidOrAmbiguousDefinitions)
    {
        EXPECT_FALSE(normalizeGdalSpatialReference(SpatialReferenceId("invalid"),
                                                   CoordinateFrameId("invalid-frame"),
                                                   "EPSG:not-a-code",
                                                   VerticalReference::Unknown)
                         .ok());
        const GdalSpatialReferenceResult missing_vertical =
            normalizeGdalSpatialReference(SpatialReferenceId("crs-geographic"),
                                          CoordinateFrameId("frame-geographic"),
                                          "EPSG:4979",
                                          VerticalReference::Unknown);
        EXPECT_FALSE(missing_vertical.ok());
        EXPECT_NE(missing_vertical.error.find("vertical reference"), std::string::npos);
    }

} // namespace
