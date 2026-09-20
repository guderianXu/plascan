#pragma once

#include "../types/CameraPose.h"

namespace xjw::camera_core
{

    /** Capability-side interface for models that expose one static camera pose. */
    class StaticPoseProvider
    {
    public:
        virtual ~StaticPoseProvider() = default;

        virtual const Pose& staticPose() const noexcept = 0;
    };

} // namespace xjw::camera_core
