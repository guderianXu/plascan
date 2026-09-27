#include "placoordinate/presets/ProjectionPresets.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
    using namespace placoordinate;

    SpatialReferenceDefinition requireReference(GdalSpatialReferenceResult result)
    {
        if (!result.ok())
        {
            throw std::runtime_error(result.error);
        }
        return std::move(*result.reference);
    }

    SpatialReferenceDefinition
    geographicReference(const char* id, const ReferenceEllipsoid& ellipsoid, VerticalReference verticalReference)
    {
        return requireReference(makeGeographicSpatialReference(
            SpatialReferenceId(id), CoordinateFrameId(std::string(id) + "-frame"), ellipsoid, verticalReference));
    }

    void expectRoundTrip(const SpatialReferenceDefinition& geographic,
                         const SpatialReferenceDefinition& projected,
                         const std::array<double, 3>& coordinate,
                         double tolerance)
    {
        const GdalCoordinateTransformResult forward =
            transformGdalCoordinate(coordinate, geographic, CoordinateAxisOrder::LongitudeLatitude, projected);
        ASSERT_TRUE(forward.ok) << forward.error;
        const GdalCoordinateTransformResult restored =
            transformGdalCoordinate(forward.coordinate, projected, CoordinateAxisOrder::TraditionalGis, geographic);
        ASSERT_TRUE(restored.ok) << restored.error;
        EXPECT_NEAR(restored.coordinate[0], coordinate[0], tolerance);
        EXPECT_NEAR(restored.coordinate[1], coordinate[1], tolerance);
        EXPECT_NEAR(restored.coordinate[2], coordinate[2], tolerance);
    }

    TEST(ProjectionPresetsTest, ProvidesNamedEarthAndMoonReferenceBodies)
    {
        const ReferenceEllipsoid wgs84 = ReferenceEllipsoid::wgs84();
        const ReferenceEllipsoid cgcs2000 = ReferenceEllipsoid::cgcs2000();
        const ReferenceEllipsoid moon = ReferenceEllipsoid::moonMeanSphere();

        EXPECT_FALSE(wgs84.isSphere());
        EXPECT_FALSE(cgcs2000.isSphere());
        EXPECT_TRUE(moon.isSphere());
        EXPECT_DOUBLE_EQ(wgs84.semiMajorAxisMetres, 6378137.0);
        EXPECT_DOUBLE_EQ(cgcs2000.inverseFlattening, 298.257222101);
        EXPECT_DOUBLE_EQ(moon.semiMajorAxisMetres, LunarMeanRadiusMetres);
    }

    TEST(ProjectionPresetsTest, BuildsGaussKrugerFromExplicitCentralMeridian)
    {
        GaussKrugerParameters parameters;
        EXPECT_FALSE(makeGaussKrugerSpatialReference(
                         SpatialReferenceId("gauss-missing"), CoordinateFrameId("gauss-frame"), parameters)
                         .ok());

        parameters.centralMeridianDegrees = 117.0;
        const SpatialReferenceDefinition projected = requireReference(makeGaussKrugerSpatialReference(
            SpatialReferenceId("gauss-117"), CoordinateFrameId("gauss-frame"), parameters));
        const SpatialReferenceDefinition geographic =
            geographicReference("cgcs2000-geographic", parameters.ellipsoid, VerticalReference::Ellipsoidal);

        EXPECT_EQ(projected.kind(), SpatialReferenceKind::Projected2d);
        EXPECT_NE(projected.canonicalDefinition().find("Transverse Mercator"), std::string::npos);
        const GdalCoordinateTransformResult origin =
            transformGdalCoordinate({117.0, 0.0, 0.0}, geographic, CoordinateAxisOrder::LongitudeLatitude, projected);
        ASSERT_TRUE(origin.ok) << origin.error;
        EXPECT_NEAR(origin.coordinate[0], 500000.0, 1.0e-6);
        EXPECT_NEAR(origin.coordinate[1], 0.0, 1.0e-6);
        expectRoundTrip(geographic, projected, {117.5, 30.0, 12.0}, 1.0e-8);
    }

    TEST(ProjectionPresetsTest, BuildsConformalMercatorWithoutConfusingItWithLunarEq)
    {
        const MercatorParameters parameters;
        const SpatialReferenceDefinition projected = requireReference(makeMercatorSpatialReference(
            SpatialReferenceId("mercator"), CoordinateFrameId("mercator-frame"), parameters));
        const SpatialReferenceDefinition geographic =
            geographicReference("wgs84-geographic", parameters.ellipsoid, VerticalReference::Ellipsoidal);

        EXPECT_NE(projected.canonicalDefinition().find("Mercator (variant A)"), std::string::npos);
        EXPECT_EQ(projected.canonicalDefinition().find("Equidistant Cylindrical"), std::string::npos);
        expectRoundTrip(geographic, projected, {116.391, 39.907, 50.0}, 1.0e-8);
    }

    TEST(ProjectionPresetsTest, BuildsLunarEquirectangularWithMeanRadius)
    {
        LunarEquirectangularParameters parameters;
        parameters.verticalReference = VerticalReference::PlanetaryRadius;
        const ReferenceEllipsoid moon = ReferenceEllipsoid::moonMeanSphere(parameters.radiusMetres);
        const SpatialReferenceDefinition geographic =
            geographicReference("moon-geographic", moon, VerticalReference::PlanetaryRadius);
        const SpatialReferenceDefinition projected = requireReference(makeLunarEquirectangularSpatialReference(
            SpatialReferenceId("moon-eq"), CoordinateFrameId("moon-eq-frame"), parameters));

        EXPECT_NE(projected.canonicalDefinition().find("Equidistant Cylindrical"), std::string::npos);
        EXPECT_EQ(projected.canonicalDefinition().find("Mercator"), std::string::npos);
        EXPECT_EQ(projected.verticalReference(), VerticalReference::PlanetaryRadius);
        const GdalCoordinateTransformResult transformed =
            transformGdalCoordinate({1.0, 1.0, 0.0}, geographic, CoordinateAxisOrder::LongitudeLatitude, projected);
        ASSERT_TRUE(transformed.ok) << transformed.error;
        const double one_degree_arc = LunarMeanRadiusMetres * std::numbers::pi_v<double> / 180.0;
        EXPECT_NEAR(transformed.coordinate[0], one_degree_arc, 1.0e-6);
        EXPECT_NEAR(transformed.coordinate[1], one_degree_arc, 1.0e-6);
        expectRoundTrip(geographic, projected, {23.5, -18.25, 100.0}, 1.0e-9);
    }

    TEST(ProjectionPresetsTest, BuildsNorthAndSouthLunarPolarStereographic)
    {
        const ReferenceEllipsoid moon = ReferenceEllipsoid::moonMeanSphere();
        const SpatialReferenceDefinition geographic =
            geographicReference("moon-polar-geographic", moon, VerticalReference::PlanetaryRadius);

        LunarPolarStereographicParameters north_parameters;
        north_parameters.falseEastingMetres = 2000.0;
        north_parameters.falseNorthingMetres = 3000.0;
        north_parameters.verticalReference = VerticalReference::PlanetaryRadius;
        const SpatialReferenceDefinition north = requireReference(makeLunarPolarStereographicSpatialReference(
            SpatialReferenceId("moon-pola-north"), CoordinateFrameId("moon-pola-north-frame"), north_parameters));
        const GdalCoordinateTransformResult north_pole =
            transformGdalCoordinate({0.0, 90.0, 0.0}, geographic, CoordinateAxisOrder::LongitudeLatitude, north);
        ASSERT_TRUE(north_pole.ok) << north_pole.error;
        EXPECT_NEAR(north_pole.coordinate[0], 2000.0, 1.0e-6);
        EXPECT_NEAR(north_pole.coordinate[1], 3000.0, 1.0e-6);
        EXPECT_NE(north.canonicalDefinition().find("Polar Stereographic (variant A)"), std::string::npos);
        expectRoundTrip(geographic, north, {35.0, 80.0, 0.0}, 1.0e-9);

        LunarPolarStereographicParameters south_parameters;
        south_parameters.pole = LunarPole::South;
        south_parameters.verticalReference = VerticalReference::PlanetaryRadius;
        const SpatialReferenceDefinition south = requireReference(makeLunarPolarStereographicSpatialReference(
            SpatialReferenceId("moon-pola-south"), CoordinateFrameId("moon-pola-south-frame"), south_parameters));
        const GdalCoordinateTransformResult south_pole =
            transformGdalCoordinate({0.0, -90.0, 0.0}, geographic, CoordinateAxisOrder::LongitudeLatitude, south);
        ASSERT_TRUE(south_pole.ok) << south_pole.error;
        EXPECT_NEAR(south_pole.coordinate[0], 0.0, 1.0e-6);
        EXPECT_NEAR(south_pole.coordinate[1], 0.0, 1.0e-6);
        expectRoundTrip(geographic, south, {-20.0, -82.0, 0.0}, 1.0e-9);
    }

    TEST(ProjectionPresetsTest, RejectsInvalidProjectionParameters)
    {
        LunarEquirectangularParameters equirectangular;
        equirectangular.radiusMetres = -1.0;
        EXPECT_FALSE(makeLunarEquirectangularSpatialReference(
                         SpatialReferenceId("bad-radius"), CoordinateFrameId("bad-radius-frame"), equirectangular)
                         .ok());

        equirectangular.radiusMetres = LunarMeanRadiusMetres;
        equirectangular.standardParallelDegrees = 90.0;
        EXPECT_FALSE(makeLunarEquirectangularSpatialReference(
                         SpatialReferenceId("bad-parallel"), CoordinateFrameId("bad-parallel-frame"), equirectangular)
                         .ok());

        LunarPolarStereographicParameters polar;
        polar.scaleFactorAtPole = 0.0;
        EXPECT_FALSE(makeLunarPolarStereographicSpatialReference(
                         SpatialReferenceId("bad-scale"), CoordinateFrameId("bad-scale-frame"), polar)
                         .ok());

        MercatorParameters mercator;
        mercator.centralMeridianDegrees = std::numeric_limits<double>::quiet_NaN();
        EXPECT_FALSE(makeMercatorSpatialReference(
                         SpatialReferenceId("bad-meridian"), CoordinateFrameId("bad-meridian-frame"), mercator)
                         .ok());

        mercator.centralMeridianDegrees = 0.0;
        mercator.verticalReference = VerticalReference::Unknown;
        EXPECT_FALSE(makeMercatorSpatialReference(SpatialReferenceId("bad-projected-vertical"),
                                                  CoordinateFrameId("bad-projected-vertical-frame"),
                                                  mercator)
                         .ok());

        EXPECT_FALSE(makeGeographicSpatialReference(SpatialReferenceId("bad-vertical"),
                                                    CoordinateFrameId("bad-vertical-frame"),
                                                    ReferenceEllipsoid::wgs84(),
                                                    VerticalReference::Unknown)
                         .ok());
    }

} // namespace
