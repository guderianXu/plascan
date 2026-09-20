#pragma once

#include "CameraErrors.h"
#include "coordinate_system/types/CoordinateIds.h"

#include <array>
#include <cmath>
#include <utility>

namespace xjw::camera_core
{

    using Rotation = std::array<double, 9>;

    struct Pose
    {
        xjw::coordinate_system::CoordinateFrameId frame;
        std::array<double, 3> center{{0.0, 0.0, 0.0}};
        Rotation cameraToWorldRotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};

        static Pose
        create(xjw::coordinate_system::CoordinateFrameId frameId, std::array<double, 3> cameraCenter, Rotation rotation)
        {
            for (double value : cameraCenter)
            {
                if (!std::isfinite(value))
                {
                    throw CameraValidationError(CameraErrorCode::InvalidRotation,
                                                "pose center must contain finite values");
                }
            }

            const auto dotRow = [&rotation](int first, int second)
            {
                return rotation[static_cast<std::size_t>(first)] * rotation[static_cast<std::size_t>(second)] +
                       rotation[static_cast<std::size_t>(first + 1)] * rotation[static_cast<std::size_t>(second + 1)] +
                       rotation[static_cast<std::size_t>(first + 2)] * rotation[static_cast<std::size_t>(second + 2)];
            };
            for (double value : rotation)
            {
                if (!std::isfinite(value))
                {
                    throw CameraValidationError(CameraErrorCode::InvalidRotation,
                                                "pose rotation must contain finite values");
                }
            }
            const double determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
                                       rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
                                       rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
            constexpr double tolerance = 1.0e-8;
            if (std::abs(dotRow(0, 0) - 1.0) > tolerance || std::abs(dotRow(3, 3) - 1.0) > tolerance ||
                std::abs(dotRow(6, 6) - 1.0) > tolerance || std::abs(dotRow(0, 3)) > tolerance ||
                std::abs(dotRow(0, 6)) > tolerance || std::abs(dotRow(3, 6)) > tolerance ||
                std::abs(determinant - 1.0) > tolerance)
            {
                throw CameraValidationError(CameraErrorCode::InvalidRotation,
                                            "pose rotation must be a proper orthonormal matrix");
            }
            return Pose{std::move(frameId), cameraCenter, rotation};
        }
    };

} // namespace xjw::camera_core
