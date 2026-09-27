#include "placoordinate/serialization/CoordinateContextJsonStrings.h"

namespace placoordinate::detail
{
    std::string_view enumName(SpatialReferenceKind value) noexcept
    {
        switch (value)
        {
        case SpatialReferenceKind::Geographic2d:
            return "geographic_2d";
        case SpatialReferenceKind::Geographic3d:
            return "geographic_3d";
        case SpatialReferenceKind::Projected2d:
            return "projected_2d";
        case SpatialReferenceKind::Projected3d:
            return "projected_3d";
        case SpatialReferenceKind::Geocentric3d:
            return "geocentric_3d";
        case SpatialReferenceKind::Engineering3d:
            return "engineering_3d";
        }
        return {};
    }

    std::string_view enumName(CoordinateAxisOrder value) noexcept
    {
        switch (value)
        {
        case CoordinateAxisOrder::TraditionalGis:
            return "traditional_gis";
        case CoordinateAxisOrder::AuthorityCompliant:
            return "authority_compliant";
        case CoordinateAxisOrder::LongitudeLatitude:
            return "canonical_lon_lat_height";
        case CoordinateAxisOrder::LatitudeLongitude:
            return "latitude_longitude";
        case CoordinateAxisOrder::CanonicalXyz:
            return "canonical_xyz";
        }
        return {};
    }

    std::string_view enumName(VerticalReference value) noexcept
    {
        switch (value)
        {
        case VerticalReference::NotApplicable:
            return "not_applicable";
        case VerticalReference::Ellipsoidal:
            return "ellipsoidal";
        case VerticalReference::Orthometric:
            return "orthometric";
        case VerticalReference::PlanetaryRadius:
            return "planetary_radius";
        case VerticalReference::Relative:
            return "relative";
        case VerticalReference::Unknown:
            return "unknown";
        }
        return {};
    }

    std::string_view enumName(CoordinateFrameKind value) noexcept
    {
        switch (value)
        {
        case CoordinateFrameKind::Ecef:
            return "ecef";
        case CoordinateFrameKind::Geodetic:
            return "geodetic";
        case CoordinateFrameKind::BodyFixed:
            return "body_fixed";
        case CoordinateFrameKind::LocalEnu:
            return "local_enu";
        case CoordinateFrameKind::LocalCartesian:
            return "local_cartesian";
        }
        return {};
    }

    std::string_view enumName(LinearUnit value) noexcept
    {
        switch (value)
        {
        case LinearUnit::Metre:
            return "metre";
        case LinearUnit::Kilometre:
            return "kilometre";
        case LinearUnit::ProjectUnit:
            return "project_unit";
        }
        return {};
    }

    std::string_view enumName(AngleUnit value) noexcept
    {
        switch (value)
        {
        case AngleUnit::Degree:
            return "degree";
        case AngleUnit::Radian:
            return "radian";
        }
        return {};
    }

    std::string_view enumName(SolverScaleStatus value) noexcept
    {
        switch (value)
        {
        case SolverScaleStatus::Metric:
            return "metric";
        case SolverScaleStatus::Unresolved:
            return "unresolved";
        }
        return {};
    }

    std::optional<SpatialReferenceKind> spatialReferenceKindFromName(std::string_view value) noexcept
    {
        if (value == "geographic_2d")
            return SpatialReferenceKind::Geographic2d;
        if (value == "geographic_3d")
            return SpatialReferenceKind::Geographic3d;
        if (value == "projected_2d")
            return SpatialReferenceKind::Projected2d;
        if (value == "projected_3d")
            return SpatialReferenceKind::Projected3d;
        if (value == "geocentric_3d")
            return SpatialReferenceKind::Geocentric3d;
        if (value == "engineering_3d")
            return SpatialReferenceKind::Engineering3d;
        return std::nullopt;
    }

    std::optional<CoordinateAxisOrder> coordinateAxisOrderFromName(std::string_view value) noexcept
    {
        if (value == "traditional_gis")
            return CoordinateAxisOrder::TraditionalGis;
        if (value == "authority_compliant")
            return CoordinateAxisOrder::AuthorityCompliant;
        if (value == "canonical_lon_lat_height")
            return CoordinateAxisOrder::LongitudeLatitude;
        if (value == "latitude_longitude")
            return CoordinateAxisOrder::LatitudeLongitude;
        if (value == "canonical_xyz")
            return CoordinateAxisOrder::CanonicalXyz;
        return std::nullopt;
    }

    std::optional<VerticalReference> verticalReferenceFromName(std::string_view value) noexcept
    {
        if (value == "not_applicable")
            return VerticalReference::NotApplicable;
        if (value == "ellipsoidal")
            return VerticalReference::Ellipsoidal;
        if (value == "orthometric")
            return VerticalReference::Orthometric;
        if (value == "planetary_radius")
            return VerticalReference::PlanetaryRadius;
        if (value == "relative")
            return VerticalReference::Relative;
        if (value == "unknown")
            return VerticalReference::Unknown;
        return std::nullopt;
    }

    std::optional<CoordinateFrameKind> coordinateFrameKindFromName(std::string_view value) noexcept
    {
        if (value == "ecef")
            return CoordinateFrameKind::Ecef;
        if (value == "geodetic")
            return CoordinateFrameKind::Geodetic;
        if (value == "body_fixed")
            return CoordinateFrameKind::BodyFixed;
        if (value == "local_enu")
            return CoordinateFrameKind::LocalEnu;
        if (value == "local_cartesian")
            return CoordinateFrameKind::LocalCartesian;
        return std::nullopt;
    }

    std::optional<LinearUnit> linearUnitFromName(std::string_view value) noexcept
    {
        if (value == "metre")
            return LinearUnit::Metre;
        if (value == "kilometre")
            return LinearUnit::Kilometre;
        if (value == "project_unit")
            return LinearUnit::ProjectUnit;
        return std::nullopt;
    }

    std::optional<AngleUnit> angleUnitFromName(std::string_view value) noexcept
    {
        if (value == "degree")
            return AngleUnit::Degree;
        if (value == "radian")
            return AngleUnit::Radian;
        return std::nullopt;
    }

    std::optional<SolverScaleStatus> solverScaleStatusFromName(std::string_view value) noexcept
    {
        if (value == "metric")
            return SolverScaleStatus::Metric;
        if (value == "unresolved")
            return SolverScaleStatus::Unresolved;
        return std::nullopt;
    }

} // namespace placoordinate::detail
