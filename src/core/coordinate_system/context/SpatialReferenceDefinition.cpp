#include "SpatialReferenceDefinition.h"

#include "CoordinateHash.h"
#include "coordinate_system/types/CoordinateErrors.h"

#include <cmath>
#include <utility>

namespace xjw::coordinate_system
{
    SpatialReferenceDefinition SpatialReferenceDefinition::create(SpatialReferenceId id,
                                                                  CoordinateFrameId frameId,
                                                                  SpatialReferenceKind kind,
                                                                  CoordinateAxisOrder axisMapping,
                                                                  VerticalReference verticalReference,
                                                                  std::string canonicalDefinition,
                                                                  std::string authority,
                                                                  std::string authorityCode,
                                                                  double linearUnitToMetres)
    {
        if (canonicalDefinition.empty())
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidSpatialReference,
                                            "canonical spatial reference definition must not be empty");
        }
        const bool geographic =
            kind == SpatialReferenceKind::Geographic2d || kind == SpatialReferenceKind::Geographic3d;
        if (geographic && linearUnitToMetres != 0.0)
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidSpatialReference,
                                            "geographic spatial reference must not declare a horizontal linear unit");
        }
        if (!geographic && (!std::isfinite(linearUnitToMetres) || linearUnitToMetres <= 0.0))
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidSpatialReference,
                                            "Cartesian spatial reference must declare a positive linear unit");
        }
        if (kind == SpatialReferenceKind::Geographic3d &&
            (verticalReference == VerticalReference::Unknown || verticalReference == VerticalReference::NotApplicable))
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidSpatialReference,
                                            "3D geographic spatial reference requires an explicit vertical reference");
        }
        if (kind == SpatialReferenceKind::Geocentric3d && verticalReference != VerticalReference::NotApplicable)
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidSpatialReference,
                                            "geocentric spatial reference must use not-applicable vertical semantics");
        }
        if (authority.empty() != authorityCode.empty())
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidSpatialReference,
                                            "spatial reference authority and code must be provided together");
        }
        const std::string definition_hash = sha256Hash(canonicalDefinition);
        return SpatialReferenceDefinition(std::move(id),
                                          std::move(frameId),
                                          kind,
                                          axisMapping,
                                          verticalReference,
                                          std::move(canonicalDefinition),
                                          definition_hash,
                                          std::move(authority),
                                          std::move(authorityCode),
                                          linearUnitToMetres);
    }

    SpatialReferenceDefinition::SpatialReferenceDefinition(SpatialReferenceId id,
                                                           CoordinateFrameId frameId,
                                                           SpatialReferenceKind kind,
                                                           CoordinateAxisOrder axisMapping,
                                                           VerticalReference verticalReference,
                                                           std::string canonicalDefinition,
                                                           std::string canonicalDefinitionHash,
                                                           std::string authority,
                                                           std::string authorityCode,
                                                           double linearUnitToMetres)
        : _id(std::move(id)), _frameId(std::move(frameId)), _kind(kind), _axisMapping(axisMapping),
          _verticalReference(verticalReference), _canonicalDefinition(std::move(canonicalDefinition)),
          _canonicalDefinitionHash(std::move(canonicalDefinitionHash)), _authority(std::move(authority)),
          _authorityCode(std::move(authorityCode)), _linearUnitToMetres(linearUnitToMetres)
    {
    }

    const SpatialReferenceId& SpatialReferenceDefinition::id() const noexcept
    {
        return _id;
    }
    const CoordinateFrameId& SpatialReferenceDefinition::frameId() const noexcept
    {
        return _frameId;
    }
    SpatialReferenceKind SpatialReferenceDefinition::kind() const noexcept
    {
        return _kind;
    }
    CoordinateAxisOrder SpatialReferenceDefinition::axisMapping() const noexcept
    {
        return _axisMapping;
    }
    VerticalReference SpatialReferenceDefinition::verticalReference() const noexcept
    {
        return _verticalReference;
    }
    const std::string& SpatialReferenceDefinition::canonicalDefinition() const noexcept
    {
        return _canonicalDefinition;
    }
    const std::string& SpatialReferenceDefinition::canonicalDefinitionHash() const noexcept
    {
        return _canonicalDefinitionHash;
    }
    const std::string& SpatialReferenceDefinition::authority() const noexcept
    {
        return _authority;
    }
    const std::string& SpatialReferenceDefinition::authorityCode() const noexcept
    {
        return _authorityCode;
    }
    double SpatialReferenceDefinition::linearUnitToMetres() const noexcept
    {
        return _linearUnitToMetres;
    }

    bool SpatialReferenceDefinition::isGeographic() const noexcept
    {
        return _kind == SpatialReferenceKind::Geographic2d || _kind == SpatialReferenceKind::Geographic3d;
    }

    bool SpatialReferenceDefinition::isCartesian() const noexcept
    {
        return !isGeographic();
    }

    bool SpatialReferenceDefinition::isMetricCartesian() const noexcept
    {
        const bool three_dimensional = _kind == SpatialReferenceKind::Projected3d ||
                                       _kind == SpatialReferenceKind::Geocentric3d ||
                                       _kind == SpatialReferenceKind::Engineering3d;
        return three_dimensional && std::abs(_linearUnitToMetres - 1.0) <= 1.0e-12;
    }

} // namespace xjw::coordinate_system
