#pragma once

#include "camera/core/model/CameraDefinition.h"

#include <array>
#include <memory>
#include <optional>

namespace xjw::camera_models::rpc
{

    class RpcDefinition final : public camera_core::CameraDefinition
    {
    public:
        static constexpr int ParameterSchemaVersion = 1;

        using Coefficients = std::array<double, 20>;
        using GeodeticCoordinate = std::array<double, 3>; // longitude deg, latitude deg, height m

        struct Parameters
        {
            double lineOffset = 0.0;
            double sampleOffset = 0.0;
            double latitudeOffset = 0.0;
            double longitudeOffset = 0.0;
            double heightOffset = 0.0;
            double lineScale = 0.0;
            double sampleScale = 0.0;
            double latitudeScale = 0.0;
            double longitudeScale = 0.0;
            double heightScale = 0.0;
            Coefficients lineNumerator{};
            Coefficients lineDenominator{};
            Coefficients sampleNumerator{};
            Coefficients sampleDenominator{};
            std::optional<double> errorBiasMeters;
            std::optional<double> errorRandomMeters;
        };

        static std::shared_ptr<const RpcDefinition> create(camera_core::CameraDefinitionId definitionId,
                                                           xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                           Parameters parameters);
        static std::unique_ptr<RpcDefinition> createUnique(camera_core::CameraDefinitionId definitionId,
                                                           xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                           Parameters parameters);

        const Parameters& parameters() const noexcept;

    private:
        RpcDefinition(camera_core::CameraDefinitionId definitionId,
                      xjw::coordinate_system::CoordinateFrameId worldFrame,
                      Parameters parameters);

        static void validate(const Parameters& parameters);

        Parameters _parameters;
    };

} // namespace xjw::camera_models::rpc
