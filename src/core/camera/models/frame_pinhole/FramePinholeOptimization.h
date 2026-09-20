#pragma once

#include "FramePinholeInstance.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace xjw::camera_models::frame_pinhole
{

    struct ParameterBlockSchema
    {
        std::vector<std::string> poseParameters;
        std::vector<std::string> calibrationParameters;
    };

    class FramePinholeOptimization
    {
    public:
        static ParameterBlockSchema schema(bool optimizeCalibration);

        static FramePinholeInstance applyPoseUpdate(const FramePinholeInstance& instance,
                                                    camera_core::CameraInstanceId instanceId,
                                                    const std::array<double, 6>& delta);

        static std::shared_ptr<const FramePinholeDefinition>
        applyCalibrationUpdate(const FramePinholeDefinition& definition,
                               camera_core::CameraDefinitionId definitionId,
                               const std::array<double, 4>& delta);
    };

} // namespace xjw::camera_models::frame_pinhole
