#include "LineScanOptimization.h"

#include "camera/core/types/CameraErrors.h"

#include <cmath>

namespace xjw::camera_models::linescan
{

    LineScanTrajectoryBias LineScanOptimization::applyUpdate(const LineScanTrajectoryBias& current,
                                                             const std::array<double, 7>& delta)
    {
        for (double value : current.translationMeters)
        {
            if (!std::isfinite(value))
            {
                throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidFrame,
                                                         "line-scan trajectory translation bias must be finite");
            }
        }
        for (double value : current.rotationVectorRadians)
        {
            if (!std::isfinite(value))
            {
                throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidRotation,
                                                         "line-scan trajectory rotation bias must be finite");
            }
        }
        if (!std::isfinite(current.timeOffsetSeconds))
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidTime,
                                                     "line-scan trajectory time bias must be finite");
        }
        for (double value : delta)
        {
            if (!std::isfinite(value))
            {
                throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidFrame,
                                                         "line-scan optimization update must be finite");
            }
        }

        LineScanTrajectoryBias updated = current;
        for (int index = 0; index < 3; ++index)
        {
            updated.translationMeters[static_cast<std::size_t>(index)] += delta[static_cast<std::size_t>(index)];
            updated.rotationVectorRadians[static_cast<std::size_t>(index)] +=
                delta[static_cast<std::size_t>(index + 3)];
        }
        updated.timeOffsetSeconds += delta[6];
        return updated;
    }

} // namespace xjw::camera_models::linescan
