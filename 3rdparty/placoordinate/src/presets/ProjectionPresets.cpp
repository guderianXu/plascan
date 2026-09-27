#include "placoordinate/presets/ProjectionPresets.h"

#include <cpl_conv.h>
#include <cpl_error.h>
#include <ogr_spatialref.h>

#include <cmath>
#include <numbers>
#include <optional>
#include <string>
#include <utility>

namespace placoordinate
{
    namespace
    {

        class GdalQuietErrorScope
        {
        public:
            GdalQuietErrorScope()
            {
                CPLPushErrorHandler(CPLQuietErrorHandler);
            }

            ~GdalQuietErrorScope()
            {
                CPLPopErrorHandler();
            }
        };

        GdalSpatialReferenceResult failure(std::string message)
        {
            GdalSpatialReferenceResult result;
            result.error = std::move(message);
            return result;
        }

        std::string lastGdalError(std::string fallback)
        {
            const char* message = CPLGetLastErrorMsg();
            return message && *message ? std::string(message) : std::move(fallback);
        }

        std::optional<std::string> validateEllipsoid(const ReferenceEllipsoid& ellipsoid)
        {
            if (ellipsoid.name.empty())
            {
                return "reference ellipsoid name must not be empty";
            }
            if (!std::isfinite(ellipsoid.semiMajorAxisMetres) || ellipsoid.semiMajorAxisMetres <= 0.0)
            {
                return "reference ellipsoid semi-major axis must be positive and finite";
            }
            if (!std::isfinite(ellipsoid.inverseFlattening) ||
                (ellipsoid.inverseFlattening != 0.0 && ellipsoid.inverseFlattening <= 1.0))
            {
                return "reference ellipsoid inverse flattening must be zero for a sphere or greater than one";
            }
            return std::nullopt;
        }

        std::optional<std::string> validateLongitude(double value, const char* label)
        {
            if (!std::isfinite(value) || value < -180.0 || value > 180.0)
            {
                return std::string(label) + " must be finite and within [-180, 180] degrees";
            }
            return std::nullopt;
        }

        std::optional<std::string> validateNonPolarLatitude(double value, const char* label)
        {
            if (!std::isfinite(value) || value <= -90.0 || value >= 90.0)
            {
                return std::string(label) + " must be finite and strictly within (-90, 90) degrees";
            }
            return std::nullopt;
        }

        std::optional<std::string> validateScale(double value, const char* label)
        {
            if (!std::isfinite(value) || value <= 0.0)
            {
                return std::string(label) + " must be positive and finite";
            }
            return std::nullopt;
        }

        std::optional<std::string> validateOffsets(double falseEastingMetres, double falseNorthingMetres)
        {
            if (!std::isfinite(falseEastingMetres) || !std::isfinite(falseNorthingMetres))
            {
                return "false easting and false northing must be finite";
            }
            return std::nullopt;
        }

        std::optional<std::string> validateVerticalReference(VerticalReference verticalReference)
        {
            if (verticalReference == VerticalReference::Unknown)
            {
                return "projection preset requires explicit vertical semantics";
            }
            return std::nullopt;
        }

        bool setGeographicBase(OGRSpatialReference* reference, const ReferenceEllipsoid& ellipsoid)
        {
            const std::string geographic_name = ellipsoid.name + " geographic";
            const std::string datum_name = ellipsoid.name + " datum";
            return reference && reference->SetGeogCS(geographic_name.c_str(),
                                                     datum_name.c_str(),
                                                     ellipsoid.name.c_str(),
                                                     ellipsoid.semiMajorAxisMetres,
                                                     ellipsoid.inverseFlattening,
                                                     "Reference meridian",
                                                     0.0,
                                                     "degree",
                                                     std::numbers::pi_v<double> / 180.0) == OGRERR_NONE;
        }

        bool initializeProjectedReference(OGRSpatialReference* reference,
                                          const char* projectionName,
                                          const ReferenceEllipsoid& ellipsoid)
        {
            return reference && reference->SetProjCS(projectionName) == OGRERR_NONE &&
                   setGeographicBase(reference, ellipsoid) && reference->SetLinearUnits("metre", 1.0) == OGRERR_NONE;
        }

        GdalSpatialReferenceResult exportReference(SpatialReferenceId id,
                                                   CoordinateFrameId frameId,
                                                   OGRSpatialReference* reference,
                                                   VerticalReference verticalReference)
        {
            if (!reference)
            {
                return failure("GDAL spatial reference is null");
            }
            reference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            const char* const export_options[] = {"FORMAT=WKT2_2019", "MULTILINE=NO", nullptr};
            char* wkt = nullptr;
            if (reference->exportToWkt(&wkt, export_options) != OGRERR_NONE || !wkt)
            {
                CPLFree(wkt);
                return failure(lastGdalError("unable to export projection preset as WKT2"));
            }
            const std::string definition(wkt);
            CPLFree(wkt);
            return normalizeGdalSpatialReference(std::move(id), std::move(frameId), definition, verticalReference);
        }

        GdalSpatialReferenceResult validateCommonProjectedParameters(const ReferenceEllipsoid& ellipsoid,
                                                                     double centralMeridianDegrees,
                                                                     double latitudeOfOriginDegrees,
                                                                     double scaleFactor,
                                                                     double falseEastingMetres,
                                                                     double falseNorthingMetres,
                                                                     VerticalReference verticalReference)
        {
            if (const auto error = validateEllipsoid(ellipsoid))
            {
                return failure(*error);
            }
            if (const auto error = validateLongitude(centralMeridianDegrees, "central meridian"))
            {
                return failure(*error);
            }
            if (const auto error = validateNonPolarLatitude(latitudeOfOriginDegrees, "latitude of origin"))
            {
                return failure(*error);
            }
            if (const auto error = validateScale(scaleFactor, "scale factor"))
            {
                return failure(*error);
            }
            if (const auto error = validateOffsets(falseEastingMetres, falseNorthingMetres))
            {
                return failure(*error);
            }
            if (const auto error = validateVerticalReference(verticalReference))
            {
                return failure(*error);
            }
            GdalSpatialReferenceResult result;
            result.reference = std::nullopt;
            return result;
        }

    } // namespace

    ReferenceEllipsoid ReferenceEllipsoid::wgs84()
    {
        return {"WGS 84", 6378137.0, 298.257223563};
    }

    ReferenceEllipsoid ReferenceEllipsoid::cgcs2000()
    {
        return {"CGCS2000", 6378137.0, 298.257222101};
    }

    ReferenceEllipsoid ReferenceEllipsoid::moonMeanSphere(double radiusMetres)
    {
        return {"Moon 2000", radiusMetres, 0.0};
    }

    bool ReferenceEllipsoid::isSphere() const noexcept
    {
        return inverseFlattening == 0.0;
    }

    GdalSpatialReferenceResult makeGeographicSpatialReference(SpatialReferenceId id,
                                                              CoordinateFrameId frameId,
                                                              const ReferenceEllipsoid& ellipsoid,
                                                              VerticalReference verticalReference)
    {
        if (const auto error = validateEllipsoid(ellipsoid))
        {
            return failure(*error);
        }
        if (verticalReference == VerticalReference::Unknown)
        {
            return failure("geographic projection preset requires explicit vertical semantics");
        }

        const GdalQuietErrorScope quiet_errors;
        CPLErrorReset();
        OGRSpatialReference reference;
        if (!setGeographicBase(&reference, ellipsoid))
        {
            return failure(lastGdalError("unable to create geographic spatial reference preset"));
        }
        return exportReference(std::move(id), std::move(frameId), &reference, verticalReference);
    }

    GdalSpatialReferenceResult makeGaussKrugerSpatialReference(SpatialReferenceId id,
                                                               CoordinateFrameId frameId,
                                                               const GaussKrugerParameters& parameters)
    {
        if (!parameters.centralMeridianDegrees)
        {
            return failure("Gauss-Kruger central meridian must be provided explicitly");
        }
        const GdalSpatialReferenceResult validation =
            validateCommonProjectedParameters(parameters.ellipsoid,
                                              *parameters.centralMeridianDegrees,
                                              parameters.latitudeOfOriginDegrees,
                                              parameters.scaleFactor,
                                              parameters.falseEastingMetres,
                                              parameters.falseNorthingMetres,
                                              parameters.verticalReference);
        if (!validation.error.empty())
        {
            return validation;
        }

        const GdalQuietErrorScope quiet_errors;
        CPLErrorReset();
        OGRSpatialReference reference;
        if (!initializeProjectedReference(&reference, "Gauss-Kruger", parameters.ellipsoid) ||
            reference.SetTM(parameters.latitudeOfOriginDegrees,
                            *parameters.centralMeridianDegrees,
                            parameters.scaleFactor,
                            parameters.falseEastingMetres,
                            parameters.falseNorthingMetres) != OGRERR_NONE)
        {
            return failure(lastGdalError("unable to create Gauss-Kruger projection preset"));
        }
        return exportReference(std::move(id), std::move(frameId), &reference, parameters.verticalReference);
    }

    GdalSpatialReferenceResult
    makeMercatorSpatialReference(SpatialReferenceId id, CoordinateFrameId frameId, const MercatorParameters& parameters)
    {
        const GdalSpatialReferenceResult validation =
            validateCommonProjectedParameters(parameters.ellipsoid,
                                              parameters.centralMeridianDegrees,
                                              parameters.latitudeOfOriginDegrees,
                                              parameters.scaleFactor,
                                              parameters.falseEastingMetres,
                                              parameters.falseNorthingMetres,
                                              parameters.verticalReference);
        if (!validation.error.empty())
        {
            return validation;
        }

        const GdalQuietErrorScope quiet_errors;
        CPLErrorReset();
        OGRSpatialReference reference;
        if (!initializeProjectedReference(&reference, "Mercator", parameters.ellipsoid) ||
            reference.SetMercator(parameters.latitudeOfOriginDegrees,
                                  parameters.centralMeridianDegrees,
                                  parameters.scaleFactor,
                                  parameters.falseEastingMetres,
                                  parameters.falseNorthingMetres) != OGRERR_NONE)
        {
            return failure(lastGdalError("unable to create Mercator projection preset"));
        }
        return exportReference(std::move(id), std::move(frameId), &reference, parameters.verticalReference);
    }

    GdalSpatialReferenceResult makeLunarEquirectangularSpatialReference(
        SpatialReferenceId id, CoordinateFrameId frameId, const LunarEquirectangularParameters& parameters)
    {
        const ReferenceEllipsoid moon = ReferenceEllipsoid::moonMeanSphere(parameters.radiusMetres);
        if (const auto error = validateEllipsoid(moon))
        {
            return failure(*error);
        }
        if (const auto error = validateLongitude(parameters.centralMeridianDegrees, "central meridian"))
        {
            return failure(*error);
        }
        if (const auto error = validateNonPolarLatitude(parameters.latitudeOfOriginDegrees, "latitude of origin"))
        {
            return failure(*error);
        }
        if (const auto error = validateNonPolarLatitude(parameters.standardParallelDegrees, "standard parallel"))
        {
            return failure(*error);
        }
        if (const auto error = validateOffsets(parameters.falseEastingMetres, parameters.falseNorthingMetres))
        {
            return failure(*error);
        }
        if (const auto error = validateVerticalReference(parameters.verticalReference))
        {
            return failure(*error);
        }

        const GdalQuietErrorScope quiet_errors;
        CPLErrorReset();
        OGRSpatialReference reference;
        if (!initializeProjectedReference(&reference, "Moon Equirectangular", moon) ||
            reference.SetEquirectangular2(parameters.latitudeOfOriginDegrees,
                                          parameters.centralMeridianDegrees,
                                          parameters.standardParallelDegrees,
                                          parameters.falseEastingMetres,
                                          parameters.falseNorthingMetres) != OGRERR_NONE)
        {
            return failure(lastGdalError("unable to create lunar Equirectangular projection preset"));
        }
        return exportReference(std::move(id), std::move(frameId), &reference, parameters.verticalReference);
    }

    GdalSpatialReferenceResult makeLunarPolarStereographicSpatialReference(
        SpatialReferenceId id, CoordinateFrameId frameId, const LunarPolarStereographicParameters& parameters)
    {
        const ReferenceEllipsoid moon = ReferenceEllipsoid::moonMeanSphere(parameters.radiusMetres);
        if (const auto error = validateEllipsoid(moon))
        {
            return failure(*error);
        }
        if (const auto error = validateLongitude(parameters.centralMeridianDegrees, "central meridian"))
        {
            return failure(*error);
        }
        if (const auto error = validateScale(parameters.scaleFactorAtPole, "scale factor at pole"))
        {
            return failure(*error);
        }
        if (const auto error = validateOffsets(parameters.falseEastingMetres, parameters.falseNorthingMetres))
        {
            return failure(*error);
        }
        if (const auto error = validateVerticalReference(parameters.verticalReference))
        {
            return failure(*error);
        }

        const double latitude_of_origin = parameters.pole == LunarPole::North ? 90.0 : -90.0;
        const char* projection_name =
            parameters.pole == LunarPole::North ? "Moon North Polar Stereographic" : "Moon South Polar Stereographic";
        const GdalQuietErrorScope quiet_errors;
        CPLErrorReset();
        OGRSpatialReference reference;
        if (!initializeProjectedReference(&reference, projection_name, moon) ||
            reference.SetPS(latitude_of_origin,
                            parameters.centralMeridianDegrees,
                            parameters.scaleFactorAtPole,
                            parameters.falseEastingMetres,
                            parameters.falseNorthingMetres) != OGRERR_NONE)
        {
            return failure(lastGdalError("unable to create lunar Polar Stereographic projection preset"));
        }
        return exportReference(std::move(id), std::move(frameId), &reference, parameters.verticalReference);
    }

} // namespace placoordinate
