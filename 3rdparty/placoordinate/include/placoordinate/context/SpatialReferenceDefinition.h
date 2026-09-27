#pragma once

#include "placoordinate/types/CoordinateIds.h"

#include <string>

namespace placoordinate
{

    enum class SpatialReferenceKind
    {
        Geographic2d,
        Geographic3d,
        Projected2d,
        Projected3d,
        Geocentric3d,
        Engineering3d,
    };

    enum class CoordinateAxisOrder
    {
        TraditionalGis,
        AuthorityCompliant,
        LongitudeLatitude,
        LatitudeLongitude,
        CanonicalXyz,
    };

    enum class VerticalReference
    {
        NotApplicable,
        Ellipsoidal,
        Orthometric,
        PlanetaryRadius,
        Relative,
        Unknown,
    };

    class SpatialReferenceDefinition
    {
    public:
        static SpatialReferenceDefinition create(SpatialReferenceId id,
                                                 CoordinateFrameId frameId,
                                                 SpatialReferenceKind kind,
                                                 CoordinateAxisOrder axisMapping,
                                                 VerticalReference verticalReference,
                                                 std::string canonicalDefinition,
                                                 std::string authority,
                                                 std::string authorityCode,
                                                 double linearUnitToMetres);

        const SpatialReferenceId& id() const noexcept;
        const CoordinateFrameId& frameId() const noexcept;
        SpatialReferenceKind kind() const noexcept;
        CoordinateAxisOrder axisMapping() const noexcept;
        VerticalReference verticalReference() const noexcept;
        const std::string& canonicalDefinition() const noexcept;
        const std::string& canonicalDefinitionHash() const noexcept;
        const std::string& authority() const noexcept;
        const std::string& authorityCode() const noexcept;
        double linearUnitToMetres() const noexcept;
        bool isGeographic() const noexcept;
        bool isCartesian() const noexcept;
        bool isMetricCartesian() const noexcept;

    private:
        SpatialReferenceDefinition(SpatialReferenceId id,
                                   CoordinateFrameId frameId,
                                   SpatialReferenceKind kind,
                                   CoordinateAxisOrder axisMapping,
                                   VerticalReference verticalReference,
                                   std::string canonicalDefinition,
                                   std::string canonicalDefinitionHash,
                                   std::string authority,
                                   std::string authorityCode,
                                   double linearUnitToMetres);

        SpatialReferenceId _id;
        CoordinateFrameId _frameId;
        SpatialReferenceKind _kind;
        CoordinateAxisOrder _axisMapping;
        VerticalReference _verticalReference;
        std::string _canonicalDefinition;
        std::string _canonicalDefinitionHash;
        std::string _authority;
        std::string _authorityCode;
        double _linearUnitToMetres;
    };

} // namespace placoordinate
