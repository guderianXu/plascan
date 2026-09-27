#include "placamera/types.h"

#include <cmath>
#include <utility>

namespace placamera
{

    Pose Pose::create(FrameId frame, Vector3 center, RotationMatrix cameraToWorldRotation)
    {
        for (const double value : center)
        {
            if (!std::isfinite(value))
            {
                throw CameraValidationError(CameraErrorCode::InvalidPose, "pose center must contain finite values");
            }
        }
        for (const double value : cameraToWorldRotation)
        {
            if (!std::isfinite(value))
            {
                throw CameraValidationError(CameraErrorCode::InvalidPose, "pose rotation must contain finite values");
            }
        }

        const auto dot_row = [&cameraToWorldRotation](int first, int second)
        {
            return cameraToWorldRotation[static_cast<std::size_t>(first)] *
                       cameraToWorldRotation[static_cast<std::size_t>(second)] +
                   cameraToWorldRotation[static_cast<std::size_t>(first + 1)] *
                       cameraToWorldRotation[static_cast<std::size_t>(second + 1)] +
                   cameraToWorldRotation[static_cast<std::size_t>(first + 2)] *
                       cameraToWorldRotation[static_cast<std::size_t>(second + 2)];
        };
        const double determinant = cameraToWorldRotation[0] * (cameraToWorldRotation[4] * cameraToWorldRotation[8] -
                                                               cameraToWorldRotation[5] * cameraToWorldRotation[7]) -
                                   cameraToWorldRotation[1] * (cameraToWorldRotation[3] * cameraToWorldRotation[8] -
                                                               cameraToWorldRotation[5] * cameraToWorldRotation[6]) +
                                   cameraToWorldRotation[2] * (cameraToWorldRotation[3] * cameraToWorldRotation[7] -
                                                               cameraToWorldRotation[4] * cameraToWorldRotation[6]);

        constexpr double tolerance = 1.0e-8;
        if (std::abs(dot_row(0, 0) - 1.0) > tolerance || std::abs(dot_row(3, 3) - 1.0) > tolerance ||
            std::abs(dot_row(6, 6) - 1.0) > tolerance || std::abs(dot_row(0, 3)) > tolerance ||
            std::abs(dot_row(0, 6)) > tolerance || std::abs(dot_row(3, 6)) > tolerance ||
            std::abs(determinant - 1.0) > tolerance)
        {
            throw CameraValidationError(CameraErrorCode::InvalidPose,
                                        "pose rotation must be a proper orthonormal matrix");
        }

        return Pose{std::move(frame), center, cameraToWorldRotation};
    }

} // namespace placamera
