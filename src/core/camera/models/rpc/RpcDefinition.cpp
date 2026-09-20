#include "RpcDefinition.h"

#include "camera/core/types/CameraErrors.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw::camera_models::rpc
{
    namespace
    {

        bool finite(double value)
        {
            return std::isfinite(value);
        }

        bool finiteCoefficients(const RpcDefinition::Coefficients& values)
        {
            return std::all_of(values.begin(), values.end(), finite);
        }

    } // namespace

    std::shared_ptr<const RpcDefinition> RpcDefinition::create(camera_core::CameraDefinitionId definitionId,
                                                               xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                               Parameters parameters)
    {
        validate(parameters);
        return std::shared_ptr<const RpcDefinition>(
            new RpcDefinition(std::move(definitionId), std::move(worldFrame), std::move(parameters)));
    }

    std::unique_ptr<RpcDefinition> RpcDefinition::createUnique(camera_core::CameraDefinitionId definitionId,
                                                               xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                               Parameters parameters)
    {
        validate(parameters);
        return std::unique_ptr<RpcDefinition>(
            new RpcDefinition(std::move(definitionId), std::move(worldFrame), std::move(parameters)));
    }

    const RpcDefinition::Parameters& RpcDefinition::parameters() const noexcept
    {
        return _parameters;
    }

    RpcDefinition::RpcDefinition(camera_core::CameraDefinitionId definitionId,
                                 xjw::coordinate_system::CoordinateFrameId worldFrame,
                                 Parameters parameters)
        : camera_core::CameraDefinition(std::move(definitionId),
                                        "rpc00b",
                                        std::move(worldFrame),
                                        ParameterSchemaVersion,
                                        camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                   camera_core::CapabilityKind::InverseProjection,
                                                                   camera_core::CapabilityKind::Ray}),
          _parameters(std::move(parameters))
    {
    }

    void RpcDefinition::validate(const Parameters& parameters)
    {
        const std::array<double, 10> scalars{parameters.lineOffset,
                                             parameters.sampleOffset,
                                             parameters.latitudeOffset,
                                             parameters.longitudeOffset,
                                             parameters.heightOffset,
                                             parameters.lineScale,
                                             parameters.sampleScale,
                                             parameters.latitudeScale,
                                             parameters.longitudeScale,
                                             parameters.heightScale};
        const bool finiteScalars = std::all_of(scalars.begin(), scalars.end(), finite);
        const bool positiveScales = parameters.lineScale > 0.0 && parameters.sampleScale > 0.0 &&
                                    parameters.latitudeScale > 0.0 && parameters.longitudeScale > 0.0 &&
                                    parameters.heightScale > 0.0;
        const bool coefficientsFinite =
            finiteCoefficients(parameters.lineNumerator) && finiteCoefficients(parameters.lineDenominator) &&
            finiteCoefficients(parameters.sampleNumerator) && finiteCoefficients(parameters.sampleDenominator);
        const bool validErrors = (!parameters.errorBiasMeters ||
                                  (finite(*parameters.errorBiasMeters) && *parameters.errorBiasMeters >= 0.0)) &&
                                 (!parameters.errorRandomMeters ||
                                  (finite(*parameters.errorRandomMeters) && *parameters.errorRandomMeters >= 0.0));
        if (!finiteScalars || !positiveScales || !coefficientsFinite || !validErrors)
        {
            throw camera_core::CameraValidationError(
                camera_core::CameraErrorCode::InvalidFrame,
                "RPC values must be finite, scales positive, and optional errors non-negative");
        }
        constexpr double denominatorEpsilon = 1.0e-14;
        if (std::abs(parameters.lineDenominator[0]) < denominatorEpsilon ||
            std::abs(parameters.sampleDenominator[0]) < denominatorEpsilon)
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidFrame,
                                                     "RPC denominator is singular at its normalization origin");
        }
    }

} // namespace xjw::camera_models::rpc
