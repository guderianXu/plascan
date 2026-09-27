#pragma once

#include "placamera/types.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace placamera
{

    enum class OptimizationParameterKind
    {
        RotationVector,
        Translation,
        Intrinsics,
        Distortion,
        ImageCorrection,
        TimeOffset,
    };

    struct OptimizationParameterBlock
    {
        std::string name;
        OptimizationParameterKind kind = OptimizationParameterKind::Translation;
        std::string unit;
        std::size_t offset = 0;
        std::size_t size = 0;
        bool affectsDefinition = false;
    };

    struct OptimizationLayout
    {
        std::vector<OptimizationParameterBlock> blocks;

        std::size_t parameterCount() const noexcept;
        bool isValid() const noexcept;
    };

    struct OptimizationUpdate
    {
        CameraInstanceId instanceId;
        std::optional<CameraDefinitionId> definitionId;
        std::vector<double> delta;
    };

} // namespace placamera
