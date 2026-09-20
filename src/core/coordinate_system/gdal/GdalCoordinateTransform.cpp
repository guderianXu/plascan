#include "GdalCoordinateTransform.h"

#include <cpl_conv.h>
#include <cpl_error.h>
#include <ogr_spatialref.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace xjw::coordinate_system
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

        struct CoordinateTransformationDeleter
        {
            void operator()(OGRCoordinateTransformation* transformation) const
            {
                if (transformation)
                {
                    OGRCoordinateTransformation::DestroyCT(transformation);
                }
            }
        };

        std::string lastGdalError(std::string fallback)
        {
            const char* message = CPLGetLastErrorMsg();
            return message && *message ? std::string(message) : std::move(fallback);
        }

        void applyAxisOrder(OGRSpatialReference* reference, CoordinateAxisOrder axisOrder)
        {
            if (!reference)
            {
                return;
            }
            reference->SetAxisMappingStrategy(axisOrder == CoordinateAxisOrder::AuthorityCompliant
                                                  ? OAMS_AUTHORITY_COMPLIANT
                                                  : OAMS_TRADITIONAL_GIS_ORDER);
        }

        bool loadSpatialReference(const SpatialReferenceDefinition& definition,
                                  CoordinateAxisOrder axisOrder,
                                  OGRSpatialReference* reference,
                                  std::string* error)
        {
            if (!reference)
            {
                if (error)
                {
                    *error = "GDAL spatial reference output is null";
                }
                return false;
            }
            const GdalQuietErrorScope quiet_errors;
            CPLErrorReset();
            if (reference->SetFromUserInput(definition.canonicalDefinition().c_str()) != OGRERR_NONE)
            {
                if (error)
                {
                    *error = lastGdalError("unable to load canonical spatial reference definition");
                }
                return false;
            }
            applyAxisOrder(reference, axisOrder);
            return true;
        }

        std::optional<SpatialReferenceKind> classifySpatialReference(const OGRSpatialReference& reference)
        {
            const int axes = reference.GetAxesCount();
            if (reference.IsGeocentric())
            {
                return SpatialReferenceKind::Geocentric3d;
            }
            if (reference.IsProjected())
            {
                return axes >= 3 ? SpatialReferenceKind::Projected3d : SpatialReferenceKind::Projected2d;
            }
            if (reference.IsGeographic())
            {
                return axes >= 3 ? SpatialReferenceKind::Geographic3d : SpatialReferenceKind::Geographic2d;
            }
            if (reference.IsLocal())
            {
                return SpatialReferenceKind::Engineering3d;
            }
            return std::nullopt;
        }

        CoordinateAxisOrder canonicalAxisOrder(SpatialReferenceKind kind)
        {
            switch (kind)
            {
            case SpatialReferenceKind::Geographic2d:
            case SpatialReferenceKind::Geographic3d:
                return CoordinateAxisOrder::LongitudeLatitude;
            case SpatialReferenceKind::Geocentric3d:
            case SpatialReferenceKind::Engineering3d:
                return CoordinateAxisOrder::CanonicalXyz;
            case SpatialReferenceKind::Projected2d:
            case SpatialReferenceKind::Projected3d:
                return CoordinateAxisOrder::TraditionalGis;
            }
            return CoordinateAxisOrder::TraditionalGis;
        }

        double derivativeStep(const SpatialReferenceDefinition& source,
                              const std::array<double, 3>& coordinate,
                              std::size_t axis)
        {
            const double relative = std::abs(coordinate[axis]) * 1.0e-8;
            if (source.isGeographic() && axis < 2U)
            {
                return std::max(relative, 1.0e-7);
            }
            return std::max(relative, 1.0e-3);
        }

    } // namespace

    bool GdalSpatialReferenceResult::ok() const noexcept
    {
        return reference.has_value();
    }

    GdalSpatialReferenceResult normalizeGdalSpatialReference(SpatialReferenceId id,
                                                             CoordinateFrameId frameId,
                                                             std::string_view definition,
                                                             VerticalReference verticalReference)
    {
        GdalSpatialReferenceResult result;
        if (definition.empty())
        {
            result.error = "spatial reference definition is empty";
            return result;
        }

        OGRSpatialReference reference;
        const std::string encoded(definition);
        const GdalQuietErrorScope quiet_errors;
        CPLErrorReset();
        if (reference.SetFromUserInput(encoded.c_str()) != OGRERR_NONE)
        {
            result.error = lastGdalError("unable to parse spatial reference definition");
            return result;
        }
        reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        const std::optional<SpatialReferenceKind> kind = classifySpatialReference(reference);
        if (!kind)
        {
            result.error = "unsupported spatial reference kind";
            return result;
        }
        if (*kind == SpatialReferenceKind::Geocentric3d)
        {
            if (verticalReference != VerticalReference::NotApplicable)
            {
                result.error = "geocentric spatial reference requires not-applicable vertical semantics";
                return result;
            }
        }
        else if (*kind == SpatialReferenceKind::Geographic3d && verticalReference == VerticalReference::Unknown)
        {
            result.error = "3D geographic spatial reference requires an explicit vertical reference";
            return result;
        }

        const char* const export_options[] = {"FORMAT=WKT2_2019", "MULTILINE=NO", nullptr};
        char* wkt = nullptr;
        if (reference.exportToWkt(&wkt, export_options) != OGRERR_NONE || !wkt)
        {
            result.error = lastGdalError("unable to export canonical WKT2 spatial reference");
            CPLFree(wkt);
            return result;
        }
        std::string canonical_definition(wkt);
        CPLFree(wkt);

        std::string authority;
        std::string authority_code;
        if (const char* name = reference.GetAuthorityName(nullptr))
        {
            if (const char* code = reference.GetAuthorityCode(nullptr))
            {
                authority = name;
                authority_code = code;
            }
        }
        const double linear_unit_to_metres =
            *kind == SpatialReferenceKind::Geographic2d || *kind == SpatialReferenceKind::Geographic3d
                ? 0.0
                : reference.GetLinearUnits(nullptr);
        try
        {
            result.reference = SpatialReferenceDefinition::create(std::move(id),
                                                                  std::move(frameId),
                                                                  *kind,
                                                                  canonicalAxisOrder(*kind),
                                                                  verticalReference,
                                                                  std::move(canonical_definition),
                                                                  std::move(authority),
                                                                  std::move(authority_code),
                                                                  linear_unit_to_metres);
        }
        catch (const std::exception& exception)
        {
            result.error = exception.what();
        }
        return result;
    }

    GdalCoordinateTransformResult transformGdalCoordinate(const std::array<double, 3>& coordinate,
                                                          const SpatialReferenceDefinition& source,
                                                          CoordinateAxisOrder sourceAxisOrder,
                                                          const SpatialReferenceDefinition& target)
    {
        GdalCoordinateTransformResult result;
        if (!std::all_of(coordinate.begin(), coordinate.end(), [](double value) { return std::isfinite(value); }))
        {
            result.error = "source coordinate contains a non-finite value";
            return result;
        }

        OGRSpatialReference source_reference;
        OGRSpatialReference target_reference;
        if (!loadSpatialReference(source, sourceAxisOrder, &source_reference, &result.error) ||
            !loadSpatialReference(target, target.axisMapping(), &target_reference, &result.error))
        {
            return result;
        }

        OGRCoordinateTransformationOptions options;
        if (!options.SetBallparkAllowed(false) || !options.SetOnlyBest(true))
        {
            result.error = "GDAL rejected strict coordinate transformation options";
            return result;
        }
        const GdalQuietErrorScope quiet_errors;
        CPLErrorReset();
        std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter> transformation(
            OGRCreateCoordinateTransformation(&source_reference, &target_reference, options));
        if (!transformation)
        {
            result.error = lastGdalError("unable to create a non-ballpark coordinate transformation");
            return result;
        }

        result.coordinate = coordinate;
        if (sourceAxisOrder == CoordinateAxisOrder::LatitudeLongitude)
        {
            std::swap(result.coordinate[0], result.coordinate[1]);
        }
        if (!transformation->Transform(1, &result.coordinate[0], &result.coordinate[1], &result.coordinate[2]))
        {
            result.error = lastGdalError("coordinate is outside the valid transformation domain");
            return result;
        }
        if (target.axisMapping() == CoordinateAxisOrder::LatitudeLongitude)
        {
            std::swap(result.coordinate[0], result.coordinate[1]);
        }
        if (!std::all_of(
                result.coordinate.begin(), result.coordinate.end(), [](double value) { return std::isfinite(value); }))
        {
            result.error = "coordinate transformation produced a non-finite value";
            return result;
        }
        result.ok = true;
        return result;
    }

    GdalCoordinateUncertaintyResult
    transformGdalCoordinateWithDiagonalUncertainty(const std::array<double, 3>& coordinate,
                                                   const std::array<double, 3>& sourceStandardDeviation,
                                                   const SpatialReferenceDefinition& source,
                                                   CoordinateAxisOrder sourceAxisOrder,
                                                   const SpatialReferenceDefinition& target)
    {
        GdalCoordinateUncertaintyResult result;
        if (!std::all_of(sourceStandardDeviation.begin(),
                         sourceStandardDeviation.end(),
                         [](double value) { return std::isfinite(value) && value > 0.0; }))
        {
            result.error = "source standard deviation must contain positive finite values";
            return result;
        }
        const GdalCoordinateTransformResult transformed =
            transformGdalCoordinate(coordinate, source, sourceAxisOrder, target);
        if (!transformed.ok)
        {
            result.error = transformed.error;
            return result;
        }
        result.coordinate = transformed.coordinate;

        std::array<std::array<double, 3>, 3> jacobian{};
        for (std::size_t source_axis = 0; source_axis < 3U; ++source_axis)
        {
            const double step = derivativeStep(source, coordinate, source_axis);
            std::array<double, 3> lower = coordinate;
            std::array<double, 3> upper = coordinate;
            lower[source_axis] -= step;
            upper[source_axis] += step;
            const GdalCoordinateTransformResult transformed_lower =
                transformGdalCoordinate(lower, source, sourceAxisOrder, target);
            const GdalCoordinateTransformResult transformed_upper =
                transformGdalCoordinate(upper, source, sourceAxisOrder, target);
            if (!transformed_lower.ok || !transformed_upper.ok)
            {
                result.error = "unable to evaluate coordinate transform Jacobian: " +
                               (transformed_lower.ok ? transformed_upper.error : transformed_lower.error);
                return result;
            }
            for (std::size_t target_axis = 0; target_axis < 3U; ++target_axis)
            {
                jacobian[target_axis][source_axis] =
                    (transformed_upper.coordinate[target_axis] - transformed_lower.coordinate[target_axis]) /
                    (2.0 * step);
            }
        }
        for (std::size_t target_axis = 0; target_axis < 3U; ++target_axis)
        {
            double variance = 0.0;
            for (std::size_t source_axis = 0; source_axis < 3U; ++source_axis)
            {
                const double contribution = jacobian[target_axis][source_axis] * sourceStandardDeviation[source_axis];
                variance += contribution * contribution;
            }
            result.standardDeviation[target_axis] = std::sqrt(variance);
        }
        result.ok = true;
        return result;
    }

} // namespace xjw::coordinate_system
