#pragma once

#include "placoordinate/gdal/GdalCoordinateTransform.h"

#include <optional>
#include <string>

namespace placoordinate
{

    inline constexpr double LunarMeanRadiusMetres = 1737400.0;

    struct ReferenceEllipsoid
    {
        std::string name;
        double semiMajorAxisMetres = 0.0;
        double inverseFlattening = 0.0;

        static ReferenceEllipsoid wgs84();
        static ReferenceEllipsoid cgcs2000();
        static ReferenceEllipsoid moonMeanSphere(double radiusMetres = LunarMeanRadiusMetres);

        bool isSphere() const noexcept;
    };

    struct GaussKrugerParameters
    {
        ReferenceEllipsoid ellipsoid = ReferenceEllipsoid::cgcs2000();
        std::optional<double> centralMeridianDegrees;
        double latitudeOfOriginDegrees = 0.0;
        double scaleFactor = 1.0;
        double falseEastingMetres = 500000.0;
        double falseNorthingMetres = 0.0;
        VerticalReference verticalReference = VerticalReference::NotApplicable;
    };

    struct MercatorParameters
    {
        ReferenceEllipsoid ellipsoid = ReferenceEllipsoid::wgs84();
        double centralMeridianDegrees = 0.0;
        double latitudeOfOriginDegrees = 0.0;
        double scaleFactor = 1.0;
        double falseEastingMetres = 0.0;
        double falseNorthingMetres = 0.0;
        VerticalReference verticalReference = VerticalReference::NotApplicable;
    };

    struct LunarEquirectangularParameters
    {
        double radiusMetres = LunarMeanRadiusMetres;
        double centralMeridianDegrees = 0.0;
        double latitudeOfOriginDegrees = 0.0;
        double standardParallelDegrees = 0.0;
        double falseEastingMetres = 0.0;
        double falseNorthingMetres = 0.0;
        VerticalReference verticalReference = VerticalReference::NotApplicable;
    };

    enum class LunarPole
    {
        North,
        South,
    };

    struct LunarPolarStereographicParameters
    {
        double radiusMetres = LunarMeanRadiusMetres;
        LunarPole pole = LunarPole::North;
        double centralMeridianDegrees = 0.0;
        double scaleFactorAtPole = 1.0;
        double falseEastingMetres = 0.0;
        double falseNorthingMetres = 0.0;
        VerticalReference verticalReference = VerticalReference::NotApplicable;
    };

    GdalSpatialReferenceResult makeGeographicSpatialReference(SpatialReferenceId id,
                                                              CoordinateFrameId frameId,
                                                              const ReferenceEllipsoid& ellipsoid,
                                                              VerticalReference verticalReference);

    GdalSpatialReferenceResult makeGaussKrugerSpatialReference(SpatialReferenceId id,
                                                               CoordinateFrameId frameId,
                                                               const GaussKrugerParameters& parameters);

    GdalSpatialReferenceResult makeMercatorSpatialReference(SpatialReferenceId id,
                                                            CoordinateFrameId frameId,
                                                            const MercatorParameters& parameters = {});

    GdalSpatialReferenceResult makeLunarEquirectangularSpatialReference(
        SpatialReferenceId id, CoordinateFrameId frameId, const LunarEquirectangularParameters& parameters = {});

    GdalSpatialReferenceResult makeLunarPolarStereographicSpatialReference(
        SpatialReferenceId id, CoordinateFrameId frameId, const LunarPolarStereographicParameters& parameters = {});

} // namespace placoordinate
