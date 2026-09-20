#pragma once

#include "LineScanInstance.h"

#include <array>

namespace xjw::camera_models::linescan
{

    struct LineScanTrajectoryBias
    {
        std::array<double, 3> translationMeters{{0.0, 0.0, 0.0}};
        std::array<double, 3> rotationVectorRadians{{0.0, 0.0, 0.0}};
        double timeOffsetSeconds = 0.0;
    };

    class LineScanOptimization
    {
    public:
        static LineScanTrajectoryBias applyUpdate(const LineScanTrajectoryBias& current,
                                                  const std::array<double, 7>& delta);
    };

} // namespace xjw::camera_models::linescan
