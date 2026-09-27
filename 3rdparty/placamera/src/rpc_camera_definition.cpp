#include "placamera/rpc_camera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace placamera
{

    namespace
    {

        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        bool finiteCoefficients(const RpcCoefficients& values) noexcept
        {
            return std::all_of(values.begin(), values.end(), finite);
        }

    } // namespace

    ReferenceEllipsoid ReferenceEllipsoid::wgs84() noexcept
    {
        return {};
    }

    std::shared_ptr<const RpcDefinition> RpcDefinition::create(CameraDefinitionId definitionId,
                                                               FrameId groundFrame,
                                                               RpcParameters parameters,
                                                               ReferenceEllipsoid ellipsoid)
    {
        validate(parameters, ellipsoid);
        return std::shared_ptr<const RpcDefinition>(
            new RpcDefinition(std::move(definitionId), std::move(groundFrame), std::move(parameters), ellipsoid));
    }

    const CameraDefinitionId& RpcDefinition::definitionId() const noexcept
    {
        return _definitionId;
    }

    std::string_view RpcDefinition::modelType() const noexcept
    {
        return "rpc00b";
    }

    int RpcDefinition::parameterSchemaVersion() const noexcept
    {
        return ParameterSchemaVersion;
    }

    const FrameId& RpcDefinition::groundFrame() const noexcept
    {
        return _groundFrame;
    }

    const RpcParameters& RpcDefinition::parameters() const noexcept
    {
        return _parameters;
    }

    const ReferenceEllipsoid& RpcDefinition::ellipsoid() const noexcept
    {
        return _ellipsoid;
    }

    CapabilitySet RpcDefinition::capabilities() const noexcept
    {
        return {CapabilityKind::Projection,
                CapabilityKind::InverseProjection,
                CapabilityKind::ImagingLocus,
                CapabilityKind::ImageCorrection,
                CapabilityKind::Optimization};
    }

    RpcDefinition::RpcDefinition(CameraDefinitionId definitionId,
                                 FrameId groundFrame,
                                 RpcParameters parameters,
                                 ReferenceEllipsoid ellipsoid)
        : _definitionId(std::move(definitionId)), _groundFrame(std::move(groundFrame)),
          _parameters(std::move(parameters)), _ellipsoid(ellipsoid)
    {
    }

    void RpcDefinition::validate(const RpcParameters& parameters, const ReferenceEllipsoid& ellipsoid)
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
        const bool finite_scalars = std::all_of(scalars.begin(), scalars.end(), finite);
        const bool positive_scales = parameters.lineScale > 0.0 && parameters.sampleScale > 0.0 &&
                                     parameters.latitudeScale > 0.0 && parameters.longitudeScale > 0.0 &&
                                     parameters.heightScale > 0.0;
        const bool coefficients_finite =
            finiteCoefficients(parameters.lineNumerator) && finiteCoefficients(parameters.lineDenominator) &&
            finiteCoefficients(parameters.sampleNumerator) && finiteCoefficients(parameters.sampleDenominator);
        const bool valid_errors = (!parameters.errorBiasMeters ||
                                   (finite(*parameters.errorBiasMeters) && *parameters.errorBiasMeters >= 0.0)) &&
                                  (!parameters.errorRandomMeters ||
                                   (finite(*parameters.errorRandomMeters) && *parameters.errorRandomMeters >= 0.0));
        const bool valid_ellipsoid = finite(ellipsoid.semiMajorAxisMeters) && finite(ellipsoid.inverseFlattening) &&
                                     ellipsoid.semiMajorAxisMeters > 0.0 &&
                                     (ellipsoid.inverseFlattening == 0.0 || ellipsoid.inverseFlattening > 1.0);
        if (!finite_scalars || !positive_scales || !coefficients_finite || !valid_errors || !valid_ellipsoid)
        {
            throw CameraValidationError(
                CameraErrorCode::InvalidModelState,
                "RPC values must be finite, scales positive, optional errors non-negative, and ellipsoid valid");
        }

        constexpr double denominator_epsilon = 1.0e-14;
        if (std::abs(parameters.lineDenominator[0]) < denominator_epsilon ||
            std::abs(parameters.sampleDenominator[0]) < denominator_epsilon)
        {
            throw CameraValidationError(CameraErrorCode::InvalidModelState,
                                        "RPC denominator is singular at its normalization origin");
        }
    }

} // namespace placamera
